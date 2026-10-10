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
from .prompts import system_prompt_for

__all__ = ["Judge", "Judgment", "adjudicate"]

_log = get_logger(__name__)

#: Candidates longer than this are truncated for the judge — the judge
#: needs the shape of each answer, not every token.
_JUDGE_CANDIDATE_CHARS = 3000
#: Judge call budget.
_JUDGE_MAX_TOKENS = 400


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "ranking": list(self.ranking),
            "rationale": self.rationale,
            "judge_model": self.judge_model,
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
    ) -> Judgment:
        """Rank ``candidates`` for ``question``.  Never raises."""
        texts = [str(c or "").strip() for c in candidates]
        texts = [t for t in texts if t]
        if not texts:
            return Judgment(winner=-1, ok=False, error="no candidates to judge")
        if len(texts) == 1:
            return Judgment(winner=0, ranking=[0],
                            rationale="only one candidate")
        numbered = "\n\n".join(
            f"--- candidate {i + 1} ---\n{t[:_JUDGE_CANDIDATE_CHARS]}"
            for i, t in enumerate(texts))
        messages = [
            Message.system(system_prompt_for("judge")),
            Message.user(f"question: {question[:2000]}\n\n{numbered}"),
        ]
        params = SamplingParams(temperature=0.1,
                                max_tokens=_JUDGE_MAX_TOKENS)
        try:
            kw: dict[str, Any] = {"task_kind": "judge"}
            if timeout_s is not None:
                kw["timeout_s"] = timeout_s
            resp: LLMResponse = self._ask(messages, params, **kw)
        except Exception as exc:  # noqa: BLE001 — the judge never breaks the caller
            _log.debug("judge ask failed: %s", exc)
            return Judgment(winner=0, ranking=list(range(len(texts))),
                            ok=False,
                            error=f"judge call failed: {exc}")
        if not getattr(resp, "ok", False) or not (resp.text or "").strip():
            return Judgment(winner=0, ranking=list(range(len(texts))),
                            ok=False,
                            error=getattr(resp, "error", "") or "judge returned nothing")
        return self._parse(resp, len(texts))

    def _parse(self, resp: LLMResponse, n: int) -> Judgment:
        from ..core.jsonutil import extract_json

        data = extract_json(resp.text or "")
        winner = 0
        ranking: list[int] = list(range(n))
        rationale = ""
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
        return Judgment(
            winner=winner, ranking=ranking, rationale=rationale,
            judge_model=getattr(resp, "model", "") or "",
            ok=True)


def adjudicate(question: str, candidates: Sequence[str], ask: Any,
               **kw: Any) -> Judgment:
    """One-shot adjudication.  Never raises."""
    return Judge(ask).adjudicate(question, candidates, **kw)


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
