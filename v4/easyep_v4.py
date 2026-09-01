"""EASY-EP (arXiv:2504.06792) ported to DeepSeek-V4-Flash.

Faithful to the paper's score, which is a sum of per-token PRODUCTS
(expert_selection.py:38 in RUCAIBox/EASYEP):

    score[layer, e] += weight_t[e] * simibr_t * norm_t[e]

    weight   gating score for expert e on token t
    norm     || unweighted expert output ||_2
    simibr   max(1 - cos(h, h + y_routed), 0)   "token contribution"

Three V4-specific adaptations, each forced by the architecture:

1. V4 folds the gating weight inside Expert.forward -- expert(x, w) computes
   w2(w * silu(gate)*up). w2 is linear, so the output is exactly w * unweighted.
   The unweighted norm is recovered by dividing, rather than paying a second
   forward pass.

2. V4's MoE returns routed+shared summed. simibr needs the routed part alone,
   so the routed sum is stashed on the module for Block to read.

3. Layers 0..n_hash_layers-1 route by token ID (Gate.tid2eid), not by content.
   Pruning them would disable specific vocabulary items outright, so they are
   protected and keep all experts.

Hyper-connections need no special handling: Block.hc_pre reduces the hc_mult
copies to a single vector before the FFN and hc_post re-expands after, so within
the FFN sub-block the stream is one vector per token, exactly as in R1.

Modes
-----
  profile   calibration forward passes -> per-layer/expert scores
  mask      scores -> keep/prune mask
  eval      generate answers with the mask applied at the gate (or without)
"""
from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoTokenizer

EPS = 1e-6


# ---------------------------------------------------------------- accumulator


class Profiler:
    """Accumulates the EASY-EP score, plus diagnostics, on GPU."""

    def __init__(self, n_layers: int, n_experts: int, device):
        self.enabled = False
        self.n_layers = n_layers
        self.n_experts = n_experts
        # float64: this is a long-running sum of many small products
        self.score = torch.zeros((n_layers, n_experts), dtype=torch.float64, device=device)
        # diagnostics, so the paper's score can be compared against the
        # simpler weight*norm variant without a second calibration run
        self.score_no_simibr = torch.zeros_like(self.score)
        self.counts = torch.zeros((n_layers, n_experts), dtype=torch.int64, device=device)
        self.gate_sums = torch.zeros_like(self.score)
        self.tokens_seen = 0

    def state(self) -> dict[str, Any]:
        return {
            "score": self.score.cpu(),
            "score_no_simibr": self.score_no_simibr.cpu(),
            "counts": self.counts.cpu(),
            "gate_sums": self.gate_sums.cpu(),
            "tokens_seen": self.tokens_seen,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
        }


# ------------------------------------------------------------------ patching


def patch(official: Any, profiler: Profiler) -> None:
    """Patch Gate/MoE/Block forwards. Weights and module structure untouched."""

    Gate, MoE, Block = official.Gate, official.MoE, official.Block
    linear = official.linear
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # ---- Gate: identical to upstream, plus an optional keep-mask -----------
    def gate_forward(self, x, input_ids=None):
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            # token-ID routing: never masked, see module docstring
            indices = self.tid2eid[input_ids]
        else:
            keep = getattr(self, "ep_keep", None)
            if keep is not None:
                scores = scores.masked_fill(~keep.view(1, -1), float("-inf"))
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices

    # ---- MoE: upstream maths, plus per-token unweighted norms --------------
    def moe_forward(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, self.dim)
        n_tok = x.size(0)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)

        prof = profiler.enabled and not self.gate.hash
        if prof:
            norms_local = torch.zeros(
                (self.n_local_experts, n_tok), dtype=torch.float32, device=x.device
            )

        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for e in range(self.experts_start_idx, self.experts_end_idx):
            if counts[e] == 0:
                continue
            rows, slots = torch.where(indices == e)
            w = weights[rows, slots, None]
            contribution = self.experts[e](x[rows], w)
            y[rows] += contribution
            if prof:
                # ||w * out|| / w == ||out||, since w2 is linear in its input
                wn = contribution.float().norm(dim=-1) / w.squeeze(-1).float().clamp_min(EPS)
                norms_local[e - self.experts_start_idx, rows] = wn

        if world_size > 1:
            dist.all_reduce(y)

        if prof:
            # each rank holds norms only for its own experts; gather all
            if world_size > 1:
                bufs = [torch.zeros_like(norms_local) for _ in range(world_size)]
                dist.all_gather(bufs, norms_local)
                norms_global = torch.cat(bufs, dim=0)
            else:
                norms_global = norms_local
            # [n_tok, n_routed] -> per activated slot
            self._ep_norms = norms_global.t().gather(1, indices)   # [n_tok, topk]
            self._ep_weights = weights.float()                     # [n_tok, topk]
            self._ep_indices = indices                             # [n_tok, topk]
            self._ep_y_routed = y.detach()                         # routed only

        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    def moe_accumulate(self, h_flat: torch.Tensor) -> None:
        """Form the per-token product and add it to the running score.

        h_flat is the pre-norm reduced residual entering the FFN sub-block --
        the V4 analogue of EASY-EP's x_before_moe.
        """
        y_r = self._ep_y_routed
        simibr = (1.0 - F.cosine_similarity(h_flat.float(), (h_flat + y_r).float(), dim=-1)).clamp_min(0.0)

        w, idx, nrm = self._ep_weights, self._ep_indices, self._ep_norms
        lid = self.layer_id
        for slot in range(idx.size(1)):
            e = idx[:, slot]
            profiler.score[lid].index_add_(0, e, (w[:, slot] * simibr * nrm[:, slot]).double())
            profiler.score_no_simibr[lid].index_add_(0, e, (w[:, slot] * nrm[:, slot]).double())
            profiler.gate_sums[lid].index_add_(0, e, w[:, slot].double())
            profiler.counts[lid].index_add_(0, e, torch.ones_like(e, dtype=torch.int64))
        profiler.tokens_seen += idx.size(0)

        # release
        self._ep_norms = self._ep_weights = self._ep_indices = self._ep_y_routed = None

    # ---- Block: keep the pre-FFN residual so simibr is computable ---------
    def block_forward(self, x, start_pos, input_ids, *attn_args):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, *attn_args)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = x                                   # <-- EASY-EP's x_before_moe
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        if (profiler.enabled
                and not self.ffn.gate.hash
                and self.layer_id < profiler.n_layers):   # excludes the MTP/DSpark stages
            self.ffn.ep_accumulate(h.view(-1, h.size(-1)))
        x = self.hc_post(x, residual, post, comb)
        return x

    Gate.forward = gate_forward
    MoE.forward = moe_forward
    MoE.ep_accumulate = moe_accumulate
    Block.forward = block_forward


def set_mask(model, mask: dict | None, n_hash_layers: int, device) -> None:
    for lid, layer in enumerate(model.layers):
        if mask is None or lid < n_hash_layers:
            layer.ffn.gate.ep_keep = None
            continue
        kept = mask["layers"][str(lid)]["kept"]
        keep = torch.zeros(layer.ffn.n_routed_experts, dtype=torch.bool, device=device)
        keep[torch.tensor(kept, device=device)] = True
        layer.ffn.gate.ep_keep = keep


# --------------------------------------------------------------------- model


def build(ckpt_path: str, config_path: str, code_dir: str, max_seq_len: int, max_bs: int):
    sys.path.insert(0, code_dir)
    import model as official  # noqa: E402

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config_path) as f:
        cfg = json.load(f)
    args = official.ModelArgs(**cfg)
    args.max_seq_len = max_seq_len
    args.max_batch_size = max_bs

    t0 = time.time()
    with torch.device("cuda"):
        model = official.Transformer(args)
    tok = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    if rank == 0:
        print(f"[easyep] model ready in {time.time()-t0:.1f}s", flush=True)
    return official, model, tok, args, rank, world_size


def load_calibration(path: Path) -> list[str]:
    """Accepts a JSON list of strings, a JSONL with a 'text'/'prompt' field, or a dir of files."""
    if path.is_dir():
        out = []
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix in {".js", ".ts", ".jsx", ".tsx", ".txt", ".md", ".py"}:
                try:
                    out.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
        return out
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        return [r.get("text") or r.get("prompt") or r.get("content") or json.dumps(r) for r in rows]
    data = json.loads(raw)
    if isinstance(data, list):
        return [d if isinstance(d, str) else (d.get("text") or d.get("prompt") or json.dumps(d)) for d in data]
    raise SystemExit(f"unsupported calibration file: {path}")


# --------------------------------------------------------------------- modes


def cmd_profile(a) -> None:
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages

    prof = Profiler(args.n_layers, args.n_routed_experts, torch.device("cuda"))
    patch(official, prof)
    set_mask(model, None, args.n_hash_layers, torch.device("cuda"))

    samples = load_calibration(Path(a.calib))
    if a.limit:
        samples = samples[: a.limit]
    if rank == 0:
        print(f"[easyep] {len(samples)} calibration samples", flush=True)

    prof.enabled = True
    used = 0
    for i, text in enumerate(samples):
        prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        ids = tok.encode(prompt)
        if len(ids) > a.max_seq_len:
            ids = ids[: a.max_seq_len]
        if len(ids) < 8:
            continue
        toks = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            model.forward(toks, 0)
        used += 1
        if rank == 0:
            print(f"[easyep] {i+1}/{len(samples)}  {len(ids)} tok  "
                  f"(cum {prof.tokens_seen} tok-layers)", flush=True)
    prof.enabled = False

    if rank == 0:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        st = prof.state()
        st["samples_used"] = used
        st["source"] = str(a.calib)
        torch.save(st, out)
        print(f"[easyep] wrote {out}  ({used} samples)", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


def cmd_mask(a) -> None:
    st = torch.load(a.scores, map_location="cpu")
    key = "score_no_simibr" if a.no_simibr else "score"
    scores = st[key].float()
    n_layers, n_experts = scores.shape
    keep_n = a.keep
    mask = {
        "keep_per_layer": keep_n,
        "n_experts": n_experts,
        "protected_hash_layers": list(range(a.n_hash_layers)),
        "score_key": key,
        "layers": {},
    }
    for lid in range(n_layers):
        if lid < a.n_hash_layers:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        order = torch.argsort(scores[lid], descending=True)
        kept = sorted(order[:keep_n].tolist())
        mask["layers"][str(lid)] = {
            "kept": kept,
            "pruned": sorted(order[keep_n:].tolist()),
        }
    Path(a.out).write_text(json.dumps(mask, indent=1), encoding="utf-8")

    dead = int((scores[a.n_hash_layers:] == 0).sum())
    print(f"[easyep] mask -> {a.out}")
    print(f"[easyep] keep {keep_n}/{n_experts} per layer on layers "
          f"{a.n_hash_layers}..{n_layers-1}; hash layers kept whole")
    print(f"[easyep] experts never activated during calibration: {dead}")


def cmd_eval(a) -> None:
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    sys.path.insert(0, a.code_dir)
    from generate import generate

    prof = Profiler(args.n_layers, args.n_routed_experts, torch.device("cuda"))
    patch(official, prof)

    mask = json.loads(Path(a.mask).read_text()) if a.mask else None
    set_mask(model, mask, args.n_hash_layers, torch.device("cuda"))
    tag = "pruned" if mask else "full"

    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions", [])
    if a.limit:
        questions = questions[: a.limit]

    out_path = Path(a.out)
    if rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fp = out_path.open("a", encoding="utf-8")

    for i, q in enumerate(questions):
        text = q if isinstance(q, str) else (q.get("question") or q.get("prompt") or json.dumps(q))
        qid = None if isinstance(q, str) else q.get("id")
        prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        ids = tok.encode(prompt)
        t0 = time.time()
        with torch.inference_mode():
            out = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
        completion = tok.decode(out[0])
        if rank == 0:
            fp.write(json.dumps({
                "variant": tag,
                "index": i,
                "id": qid,
                "question": text,
                "completion": completion,
                "message": parse_message_from_completion_text(completion, thinking_mode="chat"),
                "seconds": round(time.time() - t0, 2),
            }) + "\n")
            fp.flush()
            print(f"[easyep:{tag}] {i+1}/{len(questions)}  {time.time()-t0:.1f}s", flush=True)

    if rank == 0:
        fp.close()
        print(f"[easyep] wrote {out_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


# ------------------------------------------------------------------- prompts
# Kept byte-identical to the earlier run_easy_ep.py so calibration and
# evaluation stay comparable with that session's outputs.


def security_review_text(relative_path: str, code: str) -> str:
    return f"""Perform a security review of this source file.

File: {relative_path}

Find concrete vulnerabilities. Trace attacker-controlled inputs to dangerous
operations, identify missing validation or authorization, and explain impact.
Do not assume the code is safe merely because context is incomplete.

<UNTRUSTED_CODE>
{code}
</UNTRUSTED_CODE>
"""


def question_text(q: dict) -> str:
    return f"""Review code snippet {q['id']} for security vulnerabilities.

Language: {q['language']}

<UNTRUSTED_CODE>
{q['snippet']}
</UNTRUSTED_CODE>

Answer in at most 160 words using this structure:
Verdict: VULNERABLE or SAFE
Vulnerability: precise name
CWE: CWE number if known
Reasoning: source, dangerous operation, and why it is unsafe
Impact: realistic consequence
Remediation: concrete fix
"""


def sample_calibration_files(root: Path, n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Stratified sample across CWE directories, so no single class dominates."""
    import random
    by_cwe: dict[str, list[Path]] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in {".js", ".ts", ".jsx", ".tsx"}:
            cwe = next((part for part in f.relative_to(root).parts if part.startswith("CWE-")), "other")
            by_cwe.setdefault(cwe, []).append(f)
    rng = random.Random(seed)
    for v in by_cwe.values():
        rng.shuffle(v)
    picked: list[Path] = []
    cwes = sorted(by_cwe)
    i = 0
    while len(picked) < n and any(by_cwe[c] for c in cwes):
        c = cwes[i % len(cwes)]
        if by_cwe[c]:
            picked.append(by_cwe[c].pop())
        i += 1
    return [(str(f.relative_to(root)), f.read_text(encoding="utf-8", errors="ignore")) for f in picked]


def score_completion(q: dict, completion: str) -> dict:
    """Term-overlap rubric using the fields already present in questions_used.json."""
    low = completion.lower()
    def frac(key):
        terms = [t for t in q.get(key, []) if t]
        if not terms:
            return None
        hit = sum(1 for t in terms if t.lower() in low)
        return hit / len(terms)
    verdict = None
    for line in completion.splitlines():
        if line.strip().lower().startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip().upper()[:10]
            break
    cwe = q.get("cwe", "")
    return {
        "verdict": verdict,
        "verdict_vulnerable": (verdict or "").startswith("VULNERABLE"),
        "cwe_mentioned": bool(cwe) and cwe.replace("CWE-", "") in completion,
        "accepted": frac("accepted_terms"),
        "sink": frac("sink_terms"),
        "source": frac("source_terms"),
        "remediation": frac("remediation_terms"),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ["accepted", "sink", "source", "remediation"]
    out = {"n": len(rows)}
    for k in keys:
        vals = [r["score"][k] for r in rows if r["score"].get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    out["verdict_vulnerable"] = round(
        sum(1 for r in rows if r["score"]["verdict_vulnerable"]) / max(len(rows), 1), 4)
    out["cwe_mentioned"] = round(
        sum(1 for r in rows if r["score"]["cwe_mentioned"]) / max(len(rows), 1), 4)
    out["mean_seconds"] = round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 2)
    return out


def build_mask_from_scores(scores: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    n_layers, n_experts = scores.shape
    mask = {"keep_per_layer": keep_n, "n_experts": n_experts,
            "protected_hash_layers": list(range(n_hash)), "layers": {}}
    for lid in range(n_layers):
        if lid < n_hash:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        order = torch.argsort(scores[lid], descending=True)
        mask["layers"][str(lid)] = {"kept": sorted(order[:keep_n].tolist()),
                                    "pruned": sorted(order[keep_n:].tolist())}
    return mask


def cmd_pipeline(a) -> None:
    """profile -> mask -> eval(full) -> eval(pruned), on ONE model load."""
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    patch(official, prof)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    # ---------------- phase 1: calibration ----------------
    set_mask(model, None, args.n_hash_layers, dev)
    files = sample_calibration_files(Path(a.calib_dir), a.n_calib)
    say(f"phase 1: profiling on {len(files)} calibration files")
    prof.enabled = True
    t0 = time.time()
    for i, (rel, code) in enumerate(files):
        ids = tok.encode(encode_messages(
            [{"role": "user", "content": security_review_text(rel, code)}],
            thinking_mode="chat"))
        if len(ids) > a.max_seq_len:
            ids = ids[: a.max_seq_len]
        if len(ids) < 32:
            continue
        with torch.inference_mode():
            model.forward(torch.tensor([ids], device=dev), 0)
        say(f"  calib {i+1}/{len(files)}  {len(ids)} tok  {rel}")
    prof.enabled = False
    say(f"phase 1 done in {time.time()-t0:.0f}s, {prof.tokens_seen} token-layer records")

    if rank == 0:
        torch.save(prof.state(), out / "expert_scores.pt")

    # ---------------- phase 2: mask ----------------
    scores = prof.score.float().cpu()
    mask = build_mask_from_scores(scores, a.keep, args.n_hash_layers)
    mask_alt = build_mask_from_scores(prof.score_no_simibr.float().cpu(), a.keep, args.n_hash_layers)
    if rank == 0:
        (out / "mask_keep%d.json" % a.keep).write_text(json.dumps(mask, indent=1))
        (out / "mask_keep%d_no_simibr.json" % a.keep).write_text(json.dumps(mask_alt, indent=1))
        # how much does the token-contribution term actually change the choice?
        agree = []
        for lid in range(args.n_hash_layers, args.n_layers):
            k1 = set(mask["layers"][str(lid)]["kept"])
            k2 = set(mask_alt["layers"][str(lid)]["kept"])
            agree.append(len(k1 & k2) / max(len(k1), 1))
        say(f"phase 2: mask keeps {a.keep}/{args.n_routed_experts} per layer; "
            f"overlap with no-simibr variant = {sum(agree)/len(agree):.1%}")
        never = int((scores[args.n_hash_layers:] == 0).sum())
        say(f"         experts never activated during calibration: {never}")

    # ---------------- phase 3+4: evaluate ----------------
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions", [])
    if a.limit:
        questions = questions[: a.limit]

    summary = {}
    for tag, m in (("full", None), ("pruned", mask)):
        set_mask(model, m, args.n_hash_layers, dev)
        say(f"phase 3: evaluating variant={tag} on {len(questions)} questions")
        rows = []
        for i, q in enumerate(questions):
            ids = tok.encode(encode_messages(
                [{"role": "user", "content": question_text(q)}], thinking_mode="chat"))
            t1 = time.time()
            with torch.inference_mode():
                gen = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
            completion = tok.decode(gen[0])
            if rank == 0:
                rows.append({"id": q.get("id"), "cwe": q.get("cwe"),
                             "completion": completion,
                             "seconds": round(time.time() - t1, 2),
                             "score": score_completion(q, completion)})
                say(f"  {tag} {i+1}/{len(questions)}  {time.time()-t1:.1f}s")
        if rank == 0:
            with (out / f"answers_{tag}.jsonl").open("w", encoding="utf-8") as fp:
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
            summary[tag] = summarize(rows)

    if rank == 0:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        say("=" * 60)
        say(f"RESULTS  keep={a.keep}/{args.n_routed_experts} experts "
            f"on layers {args.n_hash_layers}..{args.n_layers-1}")
        for tag in ("full", "pruned"):
            s = summary.get(tag, {})
            say(f"  {tag:7s} accepted={s.get('accepted')} sink={s.get('sink')} "
                f"remediation={s.get('remediation')} "
                f"verdict_vuln={s.get('verdict_vulnerable')} "
                f"mean_s={s.get('mean_seconds')}")
        say("=" * 60)
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    p = ArgumentParser(description="EASY-EP for DeepSeek-V4-Flash")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--ckpt-path", required=True)
        sp.add_argument("--config", required=True)
        sp.add_argument("--code-dir", required=True, help="the model's inference/ directory")
        sp.add_argument("--max-seq-len", type=int, default=8192)
        sp.add_argument("--limit", type=int, default=0)

    sp = sub.add_parser("profile", help="calibration -> expert scores")
    common(sp)
    sp.add_argument("--calib", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("mask", help="scores -> mask")
    sp.add_argument("--scores", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--keep", type=int, default=192)
    sp.add_argument("--n-hash-layers", type=int, default=3)
    sp.add_argument("--no-simibr", action="store_true",
                    help="rank by weight*norm only, i.e. EASY-EP without token contribution")

    sp = sub.add_parser("eval", help="generate with or without a mask")
    common(sp)
    sp.add_argument("--questions", required=True)
    sp.add_argument("--mask", default="")
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=256)

    sp = sub.add_parser("pipeline", help="profile -> mask -> eval(full) -> eval(pruned), one load")
    common(sp)
    sp.add_argument("--calib-dir", required=True)
    sp.add_argument("--questions", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n-calib", type=int, default=25)
    sp.add_argument("--keep", type=int, default=192)
    sp.add_argument("--max-new-tokens", type=int, default=256)

    a = p.parse_args()
    {"profile": cmd_profile, "mask": cmd_mask, "eval": cmd_eval,
     "pipeline": cmd_pipeline}[a.mode](a)


if __name__ == "__main__":
    main()
