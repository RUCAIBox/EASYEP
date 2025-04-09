import re

from symeval import EvaluatorMathBatch


class MATHEvaluator:
    def __init__(self):
        self.evaluator = EvaluatorMathBatch()

    def extract_answer_math(self, s):
        ans = s.split("boxed")
        if len(ans) == 1:
            return s
        ans = ans[-1]
        if len(ans) == 0:
            return ""
        try:
            if ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
        except:
            return ""
        return a

    def score(self, pred_ans, real_ans):
        answers = [self.extract_answer_math(a) for a in real_ans]
        preds = [self.extract_answer_math(a) for a in pred_ans]
        scores = self.evaluator.batch_eq(ref_answers=answers, pred_answers=preds)
        return scores
