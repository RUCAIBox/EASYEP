# EASY-EP for DeepSeek-V4-Flash

Adaptation of EASY-EP ([arXiv:2504.06792](https://arxiv.org/abs/2504.06792)) from
DeepSeek-R1 to **DeepSeek-V4-Flash-0731** (284B MoE, 13B active), running on a
single 4×H100 node.

This is an *adaptation*, not a port: V4-Flash differs from R1 in four ways that
each touch the scoring path. Nothing under `pruning/`, `sglang/` or `configs/` is
modified — the R1 pipeline is left intact and all V4 work is confined to `v4/`.

## Scope of this first version

| | |
|---|---|
| Layers 0–2 | **unchanged**, all 256 experts kept |
| Layers 3–42 | pruned 256 → 128 experts |
| MTP / DSpark layers 43–45 | **excluded** from profiling and masking (see *Open items*) |
| Validation | **masking only** — expert weights are never modified |

## The score

The paper's score is a sum of per-token **products**
(`pruning/expert_selection.py:38` upstream):

```python
score[layer, e] += weight_t[e] * simibr_t * norm_t[e]
```

| term | meaning |
|---|---|
| `weight` | gating score for expert `e` on token `t` |
| `norm`   | ‖unweighted expert output‖₂ |
| `simibr` | `max(1 − cos(h, h + y_routed), 0)` — token contribution |

Implemented in `easyep_v4.py`:

- `moe_forward` — collects per-token `weights`, `indices`, and unweighted `norms`
- `moe_accumulate` — forms the product and adds it to the running score
- `block_forward` — supplies `h`, the pre-FFN residual, so `simibr` is computable

The accumulation must happen per token. A previous attempt accumulated `Σweight`
and `Σnorm` into separate marginal buffers and ranked by the latter; a product-sum
cannot be recovered from marginals, and that variant drops `simibr` entirely.
For comparison, `score_no_simibr` (= `Σ weight·norm`) is recorded in the same
pass, so the effect of the token-contribution term is measurable without a second
calibration run.

## The four V4 divergences

### 1. `sqrtsoftplus` gate

R1 uses `sigmoid`; V4 uses `F.softplus(scores).sqrt()` with a `noaux_tc` bias that
shifts top-k selection but not the returned routing weights. `gate_forward` mirrors
upstream V4 exactly and adds only an optional keep-mask applied to the scores
before `topk`. The returned `weights` are unchanged in meaning, so the score
formula carries over as-is.

### 2. mHC residuals

V4 maintains `hc_mult = 4` copies of the hidden state. This turns out **not** to
need an aggregation choice: `Block.hc_pre` reduces the 4 copies to a single vector
*before* the FFN and `hc_post` re-expands *after*, so inside the FFN sub-block the
stream is one vector per token, exactly as in R1.

```python
residual = x                              # [b,s,hc,d]
x, post, comb = self.hc_pre(x, ...)       # [b,s,d]   <- h, the analogue of x_before_moe
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)                # [b,s,d]
x = self.hc_post(x, residual, post, comb) # [b,s,hc,d]
```

`h` is the pre-norm reduced residual, matching upstream's use of the pre-norm
residual rather than the normed FFN input.

### 3. Hash-routed layers 0–2

`Gate` routes by **token ID** (`tid2eid`) for `layer_id < n_hash_layers`, not by
content. An expert pruned there is unreachable for the specific vocabulary items
mapped to it, so these layers are excluded from both profiling and masking and
keep all 256 experts. (Upstream also skips its first 3 layers, but because they
are dense MLPs — different reason, same range.)

### 4. FP4 expert format

Experts are stored `float4_e2m1fn_x2` — two values per byte, with a per-32
`float8_e8m0fnu` block scale. This does **not** affect masking, which is why this
version validates by masking first. It will matter for actual deletion: removing
an expert means repacking rather than slicing, since expert boundaries need not
land on byte boundaries.

Two consequences already handled:

- V4 folds the gating weight *inside* `Expert.forward`, computing
  `w2(w · silu(gate)·up)`. Since `w2` is linear, the output is exactly
  `w × unweighted`, so the unweighted norm is recovered by dividing by `w`
  rather than paying a second forward pass.
- V4's `MoE.forward` returns routed + shared summed. `simibr` needs the routed
  part alone, so it is stashed on the module for `Block` to read.

## Running it

```bash
sbatch v4/easyep.sbatch      # profile -> mask -> eval(full) -> eval(pruned), one model load
```

The four phases share a single model load; on Rorqual that load is ~15 minutes,
so splitting them into separate jobs would triple the dominant cost.

Outputs, under `--out`:

| file | contents |
|---|---|
| `expert_scores.pt` | per-layer/expert `score`, `score_no_simibr`, `counts`, `gate_sums` |
| `mask_keep128.json` | kept/pruned expert ids per layer |
| `mask_keep128_no_simibr.json` | same, ranked without the token-contribution term |
| `answers_full.jsonl` | 50 questions, unmasked |
| `answers_pruned.jsonl` | 50 questions, masked |
| `summary.json` | term-overlap rubric per variant |

Standalone modes (`profile`, `mask`, `eval`) exist for iterating on one stage.

## Open items

- **MTP layers are excluded, not handled.** `DSparkBlock.forward` delegates to
  `Block.forward` when `start_pos > 0` with `layer_id` 43–45; the accumulator is
  bounds-guarded against that. They have their own experts under the `mtp.*`
  namespace and warrant their own profiling and pruning decision.
- **No weight deletion.** Masking only, by design.
- **Calibration size.** 25 files stratified across CWE directories, matching the
  paper's 25 samples. Not yet swept.
- **`simibr` reference point.** `h` is the pre-`ffn_norm` reduced residual. The
  alternative — comparing in the post-`hc_post` `[b,s,hc,d]` space, with and
  without the routed term — is closer to V4's true residual dynamics but compares
  in a different space. Worth a look if scores appear insensitive to `simibr`.
