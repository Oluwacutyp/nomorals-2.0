"""Multi-model adjudication: Devon's own judge, no external service.

When several models answer the same question (fan-out for research
conflicts, code candidates, plan options), *someone* has to pick the
winner.  Instead of a heuristic ("longest answer wins") or an external
judging API, Devon judges herself: the best judge-capable model on the
broker scores the candidates and the winner is served with its rationale.

:class:`Judge` is the primitive; :meth:`Brain.best_of` is the serving
entry point (fan out → adjudicate → winner).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from .base import LLMResponse, Message, SamplingParams
from .prompts import rubric_prompt, system_prompt_for

__all__ = [
    "Judge",
    "Judgment",
    "PairwiseVerdict",
    "adjudicate",
    "fan_out",
    "format_judgment",
    "pairwise",
]

_log = get_logger(__name__)

#: Candidates longer than this are truncated for the judge — the judge
#: needs the shape of each answer, not every token.
_JUDGE_CANDIDATE_CHARS = 3000
#: Judge call budget.
_JUDGE_MAX_TOKENS = 600


@dataclass
class Judgment:
    """The judge's verdict over a candidate set."""

    #: 0-based index into the candidate list.
    winner: int
    #: All candidate indices, best first.
    ranking: list[int] = field(default_factory=list)
    rationale: str = ""
    #: The judge's raw model/provider, for provenance.
    judge_model: str = ""
    ok: bool = True
    error: str = ""
    #: Per-candidate per-dimension scores, e.g. {0: {"correctness": 4}}.
    scores: dict[int, dict[str, float]] = field(default_factory=dict)
    #: 0..1 — 1.0 when the verdict was consistent across orders/judges.
    confidence: float = 1.0
    #: How the verdict was reached: "single", "pairwise-debiased",
    #: "panel-majority", "fallback".
    method: str = "single"
    #: Character lengths of the judged candidates (length-bias audit).
    candidate_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "ranking": list(self.ranking),
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "ok": self.ok,
            "error": self.error,
            "scores": {str(k): dict(v) for k, v in self.scores.items()},
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "candidate_lengths": list(self.candidate_lengths),
        }


@dataclass
class PairwiseVerdict:
    """Two-candidate comparison with position-bias control.

    ``winner`` is 0/1, or -1 on a tie (the candidates split the two
    orderings — the classic position-bias signature — or the judge could
    not decide).  ``consistent`` is True only when both orderings agreed.
    """

    winner: int = -1
    consistent: bool = False
    rationale: str = ""
    judge_model: str = ""
    confidence: float = 0.0
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "consistent": self.consistent,
            "rationale": self.rationale,
            "judge_model": self.judge_model,
            "confidence": round(self.confidence, 3),
            "ok": self.ok,
            "error": self.error,
        }


class Judge:
    """Ranks candidate answers with a judge-capable model.

    ``ask`` is any ``(messages, params, task_kind) -> LLMResponse``
    callable — in the serving path that is ``Brain.chat`` with
    ``task_kind="judge"`` so the broker picks a judge-capable model.
    """

    def __init__(self, ask: Any) -> None:
        self._ask = ask

    def adjudicate(
        self,
        question: str,
        candidates: Sequence[str],
        *,
        timeout_s: float | None = None,
        reference: str = "",
        length_controlled: bool = False,
    ) -> Judgment:
        """Rank ``candidates`` for ``question``.  Never raises.

        ``reference`` is the gold answer the judge grades closeness to
        (reference-guided judging).  ``length_controlled`` tells the judge
        to ignore verbosity (AlpacaEval-style length control).
        """
        texts = [str(c or "").strip() for c in candidates]
        texts = [t for t in texts if t]
        lengths = [len(t) for t in texts]
        if not texts:
            return Judgment(winner=-1, ok=False, error="no candidates to judge")
        if len(texts) == 1:
            return Judgment(winner=0, ranking=[0],
                            rationale="only one candidate",
                            candidate_lengths=lengths, method="single")
        numbered = "\n\n".join(
            f"--- candidate {i + 1} ---\n{t[:_JUDGE_CANDIDATE_CHARS]}"
            for i, t in enumerate(texts))
        if length_controlled:
            numbered += ("\n\nNOTE: ignore answer length and verbosity — "
                         "grade only correctness, completeness and clarity. "
                         "A shorter correct answer beats a longer padded one.")
        messages = [
            Message.system(rubric_prompt(reference=reference)
                           if reference else system_prompt_for("judge")),
            Message.user(f"question: {question[:2000]}\n\n{numbered}"),
        ]
        # T=0.0: judge consistency is ~1.0 at near-zero temperature and
        # collapses as temperature rises — a judge must not be creative.
        params = SamplingParams(temperature=0.0,
                                max_tokens=_JUDGE_MAX_TOKENS)
        try:
            kw: dict[str, Any] = {"task_kind": "judge"}
            if timeout_s is not None:
                kw["timeout_s"] = timeout_s
            resp: LLMResponse = self._ask(messages, params, **kw)
        except Exception as exc:  # noqa: BLE001 — the judge never breaks the caller
            _log.debug("judge ask failed: %s", exc)
            return Judgment(winner=0, ranking=list(range(len(texts))),
                            ok=False, method="fallback",
                            error=f"judge call failed: {exc}",
                            candidate_lengths=lengths)
        if not getattr(resp, "ok", False) or not (resp.text or "").strip():
            return Judgment(winner=0, ranking=list(range(len(texts))),
                            ok=False, method="fallback",
                            error=getattr(resp, "error", "") or "judge returned nothing",
                            candidate_lengths=lengths)
        judgment = self._parse(resp, len(texts))
        judgment.candidate_lengths = lengths
        return judgment

    def pairwise(
        self,
        question: str,
        a: str,
        b: str,
        *,
        timeout_s: float | None = None,
        reference: str = "",
    ) -> PairwiseVerdict:
        """Two-candidate comparison with position-bias control.

        Asks the judge twice with the candidates swapped; a win counts only
        when BOTH orderings agree (the Wikipedia-recommended mitigation —
        a split verdict is the classic position-bias signature and comes
        back as a tie with low confidence instead of a coin flip).
        Never raises.
        """
        texts = [str(a or "").strip(), str(b or "").strip()]
        if not texts[0] or not texts[1]:
            return PairwiseVerdict(ok=False,
                                   error="pairwise needs two non-empty candidates")
        first = self._pairwise_once(question, texts, False, timeout_s, reference)
        second = self._pairwise_once(question, texts, True, timeout_s, reference)
        if not first["ok"] or not second["ok"]:
            err = first.get("error") or second.get("error") or "judge failed"
            return PairwiseVerdict(ok=False, error=err)
        # Map both verdicts back to candidate indices (0/1 in `texts`).
        w1 = 1 - first["winner"] if first["swapped"] else first["winner"]
        w2 = 1 - second["winner"] if second["swapped"] else second["winner"]
        consistent = w1 == w2
        return PairwiseVerdict(
            winner=w1 if consistent else -1,
            consistent=consistent,
            rationale=(first.get("rationale") or "")[:300],
            judge_model=first.get("judge_model") or "",
            confidence=0.95 if consistent else 0.35,
            ok=True,
        )

    def _pairwise_once(
        self,
        question: str,
        texts: list[str],
        swapped: bool,
        timeout_s: float | None,
        reference: str,
    ) -> dict[str, Any]:
        order = [1, 0] if swapped else [0, 1]
        numbered = "\n\n".join(
            f"--- candidate {i + 1} ---\n{texts[j][:_JUDGE_CANDIDATE_CHARS]}"
            for i, j in enumerate(order))
        messages = [
            Message.system(rubric_prompt(reference=reference)
                           if reference else system_prompt_for("judge")),
            Message.user(
                f"question: {question[:2000]}\n\n{numbered}\n\n"
                "Pick exactly ONE winner (no ties)."),
        ]
        params = SamplingParams(temperature=0.0, max_tokens=_JUDGE_MAX_TOKENS)
        try:
            kw: dict[str, Any] = {"task_kind": "judge"}
            if timeout_s is not None:
                kw["timeout_s"] = timeout_s
            resp: LLMResponse = self._ask(messages, params, **kw)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"judge call failed: {exc}"}
        if not getattr(resp, "ok", False) or not (resp.text or "").strip():
            return {"ok": False,
                    "error": getattr(resp, "error", "") or "judge returned nothing"}
        judgment = self._parse(resp, 2)
        return {
            "ok": True,
            "winner": judgment.winner,
            "swapped": swapped,
            "rationale": judgment.rationale,
            "judge_model": judgment.judge_model,
        }

    def panel(
        self,
        question: str,
        candidates: Sequence[str],
        asks: Sequence[Any],
        *,
        timeout_s: float | None = None,
        reference: str = "",
    ) -> Judgment:
        """Multi-judge majority vote across judge callables.

        Each entry of ``asks`` should hit a *different model family* —
        aggregating verdicts across families offsets any single model's
        bias (and catches self-preference bias).  The winner needs a
        strict majority; without one the verdict is the plurality winner
        with confidence scaled down.  Judge disagreement itself is signal
        and is surfaced in the rationale.  Never raises.
        """
        texts = [str(c or "").strip() for c in candidates]
        texts = [t for t in texts if t]
        lengths = [len(t) for t in texts]
        if not texts:
            return Judgment(winner=-1, ok=False, error="no candidates to judge")
        if not asks:
            return Judgment(winner=-1, ok=False,
                            error="no judges in the panel")
        votes: list[Judgment] = []
        for ask in asks:
            try:
                j = Judge(ask).adjudicate(
                    question, texts, timeout_s=timeout_s, reference=reference)
            except Exception as exc:  # noqa: BLE001
                _log.debug("panel judge failed: %s", exc)
                continue
            if j.ok:
                votes.append(j)
        if not votes:
            return Judgment(winner=0, ranking=list(range(len(texts))),
                            ok=False, method="fallback",
                            error="every panel judge failed",
                            candidate_lengths=lengths)
        tally: dict[int, int] = {}
        for v in votes:
            tally[v.winner] = tally.get(v.winner, 0) + 1
        winner = max(tally, key=lambda k: (tally[k], -k))
        majority = tally[winner] > len(votes) / 2
        confidence = (tally[winner] / len(votes)) if majority \
            else 0.5 * tally[winner] / len(votes)
        # Merge rankings by average position.
        pos: dict[int, float] = {i: 0.0 for i in range(len(texts))}
        for v in votes:
            for rank, idx in enumerate(v.ranking):
                pos[idx] = pos.get(idx, 0.0) + rank
        ranking = sorted(range(len(texts)), key=lambda i: (pos[i], i))
        # Merge per-dimension scores by mean.
        scores: dict[int, dict[str, float]] = {}
        counts: dict[int, dict[str, int]] = {}
        for v in votes:
            for idx, dims in v.scores.items():
                for dim, val in dims.items():
                    scores.setdefault(idx, {}).setdefault(dim, 0.0)
                    scores[idx][dim] += float(val)
                    counts.setdefault(idx, {}).setdefault(dim, 0)
                    counts[idx][dim] += 1
        for idx, dims in scores.items():
            for dim in dims:
                dims[dim] = round(dims[dim] / max(1, counts[idx][dim]), 2)
        rationale = votes[0].rationale
        if len(votes) > 1 and not majority:
            rationale = (f"panel split {tally} across {len(votes)} judges; "
                         f"plurality winner. {rationale}")[:400]
        return Judgment(
            winner=winner, ranking=ranking, rationale=rationale,
            judge_model="+".join(sorted({v.judge_model for v in votes
                                         if v.judge_model})) or "panel",
            ok=True, scores=scores, confidence=round(confidence, 3),
            method="panel-majority" if majority else "panel-plurality",
            candidate_lengths=lengths,
        )

    def _parse(self, resp: LLMResponse, n: int) -> Judgment:
        from ..core.jsonutil import extract_json

        data = extract_json(resp.text or "")
        winner = 0
        ranking: list[int] = list(range(n))
        rationale = ""
        scores: dict[int, dict[str, float]] = {}
        confidence = 0.8
        if isinstance(data, dict):
            try:
                w = int(data.get("winner", 1)) - 1
                if 0 <= w < n:
                    winner = w
            except (TypeError, ValueError):
                pass
            raw_rank = data.get("ranking") or []
            seen: list[int] = []
            for item in raw_rank:
                try:
                    idx = int(item) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n and idx not in seen:
                    seen.append(idx)
            for idx in range(n):
                if idx not in seen:
                    seen.append(idx)
            ranking = seen
            rationale = str(data.get("rationale") or "")[:400]
            raw_scores = data.get("scores") or {}
            if isinstance(raw_scores, dict):
                for key, dims in raw_scores.items():
                    try:
                        idx = int(key) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < n and isinstance(dims, dict):
                        clean: dict[str, float] = {}
                        for dim, val in dims.items():
                            try:
                                clean[str(dim)] = max(0.0, min(5.0, float(val)))
                            except (TypeError, ValueError):
                                continue
                        if clean:
                            scores[idx] = clean
            try:
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
            except (TypeError, ValueError):
                pass
        return Judgment(
            winner=winner, ranking=ranking, rationale=rationale,
            judge_model=getattr(resp, "model", "") or "",
            ok=True, scores=scores, confidence=confidence,
            method="single")


def format_judgment(judgment: Judgment,
                    candidates: Sequence[str] | None = None) -> str:
    """God-tier verdict card: winner, per-dimension scores, confidence."""
    if not judgment.ok:
        return f"⚖️ adjudication failed: {judgment.error}"
    lines = ["⚖️ adjudication — " + judgment.method.replace("-", " ")]
    for rank, idx in enumerate(judgment.ranking):
        marker = "🏆" if rank == 0 else f"#{rank + 1}"
        dims = judgment.scores.get(idx, {})
        dim_str = ("  [" + ", ".join(f"{k} {v:g}" for k, v in dims.items())
                   + "]") if dims else ""
        preview = ""
        if candidates and 0 <= idx < len(candidates):
            preview = " — " + str(candidates[idx])[:80].replace("\n", " ")
        lines.append(f"  {marker} candidate {idx + 1}{dim_str}{preview}")
    lines.append(f"  confidence {judgment.confidence:.0%} · "
                 f"judge {judgment.judge_model or 'unknown'}")
    if judgment.rationale:
        lines.append(f"  “{judgment.rationale}”")
    return "\n".join(lines)


def adjudicate(question: str, candidates: Sequence[str], ask: Any,
               **kw: Any) -> Judgment:
    """One-shot adjudication.  Never raises."""
    return Judge(ask).adjudicate(question, candidates, **kw)


def pairwise(question: str, a: str, b: str, ask: Any,
             **kw: Any) -> PairwiseVerdict:
    """One-shot position-debiased pairwise comparison.  Never raises."""
    return Judge(ask).pairwise(question, a, b, **kw)


def fan_out(ask_one: Any, prompt: str, n: int,
            *,
            task_kind: str = "",
            timeout_s: float | None = None) -> list[str]:
    """Ask ``n`` providers the same prompt in parallel; collect the texts.

    ``ask_one(i) -> LLMResponse`` is the per-provider ask.  Never raises —
    failed providers contribute nothing.
    """
    results: list[str] = []
    lock = threading.Lock()

    def _one(index: int) -> None:
        try:
            resp = ask_one(index)
        except Exception:  # noqa: BLE001
            return
        text = (getattr(resp, "text", "") or "").strip()
        if getattr(resp, "ok", False) and text:
            with lock:
                results.append(text)

    threads = [threading.Thread(target=_one, args=(i,), daemon=True,
                                name=f"fanout-{i}")
               for i in range(max(1, n))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout_s)
    return results
