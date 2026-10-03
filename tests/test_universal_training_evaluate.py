"""Universal wave: evaluation additions.

The golden battery was English-only — a run could ship a model whose
Yoruba is broken and still pass, even though the persona claims ALL
major languages including Yoruba.  These tests pin:

* ``default_yoruba_cases`` — Yoruba + Nigerian Pidgin golden cases,
  graded by the same objective rules as the English battery;
* ``judge_golden_set`` — the optional LLM-as-judge harness with an
  injected judge (regex grading stays the default);
* ``build_gate_scores(judge_score=...)`` — backward-compatible judge
  blending (old call shapes produce exactly the old scores).
"""

from __future__ import annotations

import unittest

from nomorals.training.evaluate import (
    build_gate_scores,
    default_golden_cases,
    default_yoruba_cases,
    grade_golden_set,
    judge_golden_set,
)


def _good_multilingual(prompt: str) -> str:
    if "27" in prompt and "4" in prompt:
        return "108"
    if "12 plus 30" in prompt or "12 + 30" in prompt:
        return "42"
    if "Yorùbá" in prompt or "Yoruba" in prompt or "omi" in prompt:
        return "Ẹ káàárọ̀! Mo dúpẹ́, mo wà dáadáa. Omi jẹ́ ohun mímu pàtàkì."
    return "Ẹ káàárọ̀, ọ̀rẹ́ mi!"


def _refusing(prompt: str) -> str:
    return "I'm sorry, I can't help with that."


class YorubaCasesTests(unittest.TestCase):
    def test_battery_is_nonempty_and_objective(self) -> None:
        cases = default_yoruba_cases()
        self.assertGreaterEqual(len(cases), 3)
        for case in cases:
            self.assertTrue(case.prompt)
            self.assertTrue(case.description)

    def test_good_multilingual_generator_passes(self) -> None:
        result = grade_golden_set(default_yoruba_cases(), _good_multilingual)
        self.assertEqual(result["golden_score"], 1.0)
        self.assertEqual(result["golden_failures"], [])

    def test_refusing_generator_fails(self) -> None:
        result = grade_golden_set(default_yoruba_cases(), _refusing)
        self.assertEqual(result["golden_score"], 0.0)
        self.assertTrue(result["golden_failures"])

    def test_yoruba_arithmetic_is_objective(self) -> None:
        cases = default_yoruba_cases()
        arith = [c for c in cases if "27" in c.prompt][0]
        ok, _ = arith.grade("108")
        self.assertTrue(ok)
        ok, reasons = arith.grade("109")
        self.assertFalse(ok)
        self.assertTrue(reasons)

    def test_english_battery_unchanged(self) -> None:
        # the Yoruba addition must not disturb the existing battery
        cases = default_golden_cases()
        self.assertEqual(len(cases), 4)
        result = grade_golden_set(cases, lambda p: "108" if "27" in p else
                                  "the quick brown fox jumps" if "Repeat" in p else
                                  "a cat joke " + "x" * 30 if "joke" in p else "hi there!")
        self.assertEqual(result["golden_score"], 1.0)


class JudgeHarnessTests(unittest.TestCase):
    def _judge(self, prompt: str, text: str) -> tuple[float, str]:
        if "sorry" in text.lower():
            return 0.0, "refused"
        return 1.0, "good"

    def test_judge_scores_pass(self) -> None:
        result = judge_golden_set(default_golden_cases(), _good_multilingual,
                                  self._judge)
        self.assertEqual(result["judge_score"], 1.0)
        self.assertEqual(result["judge_failures"], [])

    def test_judge_flags_refusals(self) -> None:
        result = judge_golden_set(default_golden_cases(), _refusing, self._judge)
        self.assertEqual(result["judge_score"], 0.0)
        self.assertEqual(len(result["judge_failures"]), 4)

    def test_judge_error_degrades_to_failure_row(self) -> None:
        def bad_judge(prompt: str, text: str) -> tuple[float, str]:
            raise RuntimeError("judge exploded")

        result = judge_golden_set(default_golden_cases()[:1],
                                  _good_multilingual, bad_judge)
        self.assertEqual(result["judge_score"], 0.0)
        self.assertTrue(result["judge_failures"])

    def test_generator_error_degrades_to_failure_row(self) -> None:
        def bad_gen(prompt: str) -> str:
            raise RuntimeError("gen exploded")

        result = judge_golden_set(default_golden_cases()[:1], bad_gen,
                                  self._judge)
        self.assertEqual(result["judge_score"], 0.0)

    def test_judge_scores_clamped(self) -> None:
        result = judge_golden_set(
            default_golden_cases()[:1], _good_multilingual,
            lambda p, t: (99.0, "overconfident"))
        self.assertEqual(result["judge_score"], 1.0)


class JudgeBlendTests(unittest.TestCase):
    def test_old_calls_unchanged(self) -> None:
        # backward compatibility: no judge → exactly the old 70/30 math
        out = build_gate_scores(train_loss=1.0,
                                golden={"golden_score": 0.5, "golden_failures": []})
        loss_score = 1.0 / 2.0
        self.assertAlmostEqual(out["score"], round(0.7 * loss_score + 0.3 * 0.5, 6))
        self.assertEqual(out["score_basis"], "0.7*loss + 0.3*golden")
        self.assertNotIn("judge_score", out)

    def test_three_way_blend(self) -> None:
        out = build_gate_scores(train_loss=1.0,
                                golden={"golden_score": 0.5, "golden_failures": []},
                                judge_score=1.0)
        loss_score = 0.5
        self.assertAlmostEqual(
            out["score"], round(0.6 * loss_score + 0.25 * 0.5 + 0.15 * 1.0, 6))
        self.assertEqual(out["score_basis"], "0.6*loss + 0.25*golden + 0.15*judge")
        self.assertEqual(out["judge_score"], 1.0)

    def test_loss_plus_judge(self) -> None:
        out = build_gate_scores(eval_loss=3.0, judge_score=0.8)
        loss_score = 1.0 / 4.0
        self.assertAlmostEqual(out["score"], round(0.7 * loss_score + 0.3 * 0.8, 6))
        self.assertEqual(out["score_basis"], "0.7*loss + 0.3*judge")

    def test_golden_plus_judge(self) -> None:
        out = build_gate_scores(golden={"golden_score": 0.5}, judge_score=1.0)
        self.assertAlmostEqual(out["score"], round(0.6 * 0.5 + 0.4 * 1.0, 6))

    def test_judge_only(self) -> None:
        out = build_gate_scores(judge_score=0.7)
        self.assertEqual(out["score"], 0.7)
        self.assertEqual(out["score_basis"], "judge only")

    def test_judge_clamped(self) -> None:
        out = build_gate_scores(judge_score=5.0)
        self.assertEqual(out["judge_score"], 1.0)

    def test_still_raises_with_nothing(self) -> None:
        with self.assertRaises(ValueError):
            build_gate_scores()


if __name__ == "__main__":
    unittest.main()
