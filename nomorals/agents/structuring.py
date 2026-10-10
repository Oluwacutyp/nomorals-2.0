"""Prompt / mission structuring sub-agent (wave 76).

The system keeps eating raw objectives — a mission goal from the owner,
a goal description, a chat instruction — and turning each one into a
step plan.  Before that happens, the :class:`PromptArchitect` structures
the raw text into a machine-usable brief:

* ``intent``       — one sentence: what this actually is
* ``subgoals``     — the ordered sub-tasks the text implies
* ``inputs``       — files, URLs, and referenced data it must work with
* ``constraints``  — hard rules the text states (must / never / only …)
* ``acceptance``   — the checkable "done" criteria the text states
* ``tools``        — registry tools whose names/descriptions match
* ``risks``        — irreversible or external side effects it implies
* ``brief``        — the whole thing rendered as a tight task brief

The extraction is deterministic (no model needed, works offline, testable
line by line).  When a model is attached and reasoning is enabled, ONE
bounded polish pass may refine the JSON — but the deterministic structure
is always the floor: a broken model reply changes nothing.

Consumers: mission planning injects the brief into the orchestrator's
context, goal creation uses the subgoals as its plan, and the ``/structure``
operator command shows the owner how the system will read their request.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["PromptArchitect", "structure_text", "register"]

# ── deterministic signals ───────────────────────────────────────────────────

#: goal verbs that open a sub-task clause
_GOAL_VERBS = re.compile(
    r"\b(build|create|make|write|fix|implement|add|analyze|analyse|decode|"
    r"crack|run|test|verify|check|deploy|install|find|search|investigate|"
    r"monitor|scrape|fetch|download|send|summarize|summarise|extract|parse|"
    r"convert|compare|review|audit|extend|upgrade|improve|capture|record|"
    r"report|generate|produce|set\s?up|set\s?up|configure|automate|connect|"
    r"integrate|wire|optimize|optimise|harden|secure|rebuild|restore|"
    r"migrate|split|merge|refactor|document|explain|plan|decide)\b", re.I)

#: hard-rule phrasing
_CONSTRAINT_RE = re.compile(
    r"((?:must(?:\s+not)?|never|don'?t|do\s+not|no\s+\w+|only|at\s+most|"
    r"at\s+least|within|before|after|without|unless|instead\s+of|"
    r"keep\s+it\s+|stick\s+to|avoid)\b[^.;:]{0,120})", re.I)

#: done-criteria phrasing
_ACCEPT_RE = re.compile(
    r"((?:verify|verifies|check(?:s|ed)?|ensure[sd]?|confirm(?:s|ed)?|"
    r"prove[sd]?|test(?:s|ed)?|must\s+pass|passes?\s+the|green|"
    r"working|works\s+(?:correctly|end-to-end)|acceptance|"
    r"done\s+(?:when|if))\b[^.;:]{0,140})", re.I)

#: inputs: file paths, URLs, table names, quoted blobs
_INPUT_RE = re.compile(
    r"(https?://[^\s\"'<>]+|/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+"
    r"|\b[A-Za-z0-9_\-]+\.(?:py|md|txt|json|jsonl|csv|yaml|yml|toml|log|"
    r"pdf|zip|gz|tar|ipynb|env)\b)", re.I)

#: side-effect / irreversibility markers
_RISK_RE = re.compile(
    r"\b(send|post|publish|delete|drop|overwrite|push|commit|deploy|"
    r"install|uninstall|restart|kill|format|rm\b|erase|refund|transfer|"
    r"buy|purchase|pay|withdraw)\b", re.I)

#: sentence split that keeps "e.g." and "i.e." intact
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

_STOP = re.compile(
    r"\b(i|a|an|and|or|the|of|to|in|on|for|is|are|was|were|be|been|with|"
    r"as|at|by|it|this|that|these|those|so|then|than|too|very|just|not|"
    r"no|yes|do|does|did|can|could|should|would|will|shall|may|might|"
    r"must|about|into|over|under|again|further|once|here|there|all|any|"
    r"both|each|few|more|most|other|some|such|only|own|same|s|t|don|now)"
    r"\b", re.I)


def _tokens(text: str) -> set[str]:
    return {t for t in _STOP.sub("", (text or "").lower()).split()
            if len(t) >= 3}


class PromptArchitect:
    """Structures raw objectives into machine-usable task briefs."""

    def __init__(self, context: Any) -> None:
        self.context = context

    # ── public API ──────────────────────────────────────────────────────────
    def structure(self, text: str, *, for_: str = "mission",
                  polish: bool = True) -> dict[str, Any]:
        """Structure ``text`` into a task brief.

        ``for_`` is the consumer (``mission`` | ``goal`` | ``chat``) —
        it tunes the brief rendering.  ``polish=False`` skips the
        optional model pass entirely (deterministic only).
        """
        text = (text or "").strip()
        base = self._deterministic(text, for_=for_)
        if not text:
            return base
        if polish:
            refined = self._model_polish(text, base, for_=for_)
            if refined is not None:
                base["polished"] = True
                base = refined
        base["brief"] = self._render(base, for_=for_)
        base["structured_at"] = time.time()
        return base

    # ── deterministic extraction ────────────────────────────────────────────
    def _deterministic(self, text: str, *, for_: str = "mission") -> dict[str, Any]:
        out: dict[str, Any] = {
            "intent": "", "subgoals": [], "inputs": [], "constraints": [],
            "acceptance": [], "tools": [], "risks": [],
        }
        if not text:
            return out
        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        sentences = [re.sub(r"\s+", " ", s) for s in sentences]

        # intent: the first non-imperative sentence, else the longest
        intent = ""
        for s in sentences[:3]:
            if not _GOAL_VERBS.search(s.split(" ", 2)[-1]) or \
                    len(s) < 90:
                intent = s[:220]
                break
        if not intent:
            intent = max(sentences, key=len)[:220]
        out["intent"] = intent

        # subgoals: numbered bullets win; then imperative clauses
        numbered = re.findall(
            r"^\s*(?:\d+[.)]|[-*])\s+(.+)$", text, re.M)
        if len(numbered) >= 2:
            subgoals = [re.sub(r"\s+", " ", n).strip()[:200]
                        for n in numbered[:16] if n.strip()]
        else:
            subgoals: list[str] = []
            for s in sentences:
                clauses = re.split(r"\s+(?:and|then|after that|then next)\s+",
                                   s, flags=re.I)
                for c in clauses:
                    c = c.strip(" .;,")
                    if len(c) >= 8 and _GOAL_VERBS.search(c):
                        c = re.sub(r"\s+", " ", c)[:200]
                        if c not in subgoals:
                            subgoals.append(c)
            if not subgoals:
                subgoals = [intent]
        out["subgoals"] = subgoals[:16]

        # inputs
        seen: set[str] = set()
        for m in _INPUT_RE.finditer(text):
            val = m.group(1)
            key = val.lower()
            if key not in seen:
                seen.add(key)
                out["inputs"].append(val)
        out["inputs"] = out["inputs"][:12]

        # constraints + acceptance (first match per sentence)
        for s in sentences:
            m = _CONSTRAINT_RE.search(s)
            if m:
                c = re.sub(r"\s+", " ", m.group(1)).strip(" .;,")
                if 6 <= len(c) <= 160 and c.lower() not in \
                        {x.lower() for x in out["constraints"]}:
                    out["constraints"].append(c)
            m = _ACCEPT_RE.search(s)
            if m:
                a = re.sub(r"\s+", " ", m.group(1)).strip(" .;,")
                if 6 <= len(a) <= 180 and a.lower() not in \
                        {x.lower() for x in out["acceptance"]}:
                    out["acceptance"].append(a)
        out["constraints"] = out["constraints"][:10]
        out["acceptance"] = out["acceptance"][:10]

        # risks
        risks: list[str] = []
        for m in _RISK_RE.finditer(text):
            w = m.group(1).lower()
            if w not in risks:
                risks.append(w)
        out["risks"] = risks[:10]

        # tools: match the live registry's names + descriptions
        try:
            tools = getattr(self.context, "tools", None)
            catalog = _tool_catalog(tools, limit=400)
            if catalog:
                text_toks = _tokens(text)
                scored: list[tuple[int, str]] = []
                for name, desc in catalog:
                    name_toks = set(name.replace("_", " ").split())
                    hits = len(text_toks & name_toks) + \
                        len(text_toks & _tokens(desc)) // 2
                    if hits:
                        scored.append((hits, name))
                scored.sort(key=lambda x: (-x[0], x[1]))
                out["tools"] = [n for _h, n in scored[:8]]
        except Exception:  # noqa: BLE001 - tools are an enhancement
            pass
        return out

    # ── optional model polish ───────────────────────────────────────────────
    def _model_polish(self, text: str, base: dict[str, Any],
                      *, for_: str = "mission") -> dict[str, Any] | None:
        """One bounded model pass over the deterministic brief.  Returns
        the refined brief or None (keep the deterministic one)."""
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        try:
            from .reasoning import reasoning_enabled
            if not reasoning_enabled(self.context, complex_ok=True):
                return None
        except Exception:  # noqa: BLE001
            return None
        try:
            import json as _json
            from ..llm.base import Message, SamplingParams
            from ..llm.brain import brain_for

            prompt = (
                "You are a task-structuring engine. Given a raw objective "
                "and a draft structured brief (JSON), refine it: sharper "
                "intent, correctly ordered subgoals, any missed "
                "constraints/acceptance criteria, and only genuinely "
                "relevant tool names. Do NOT invent requirements. Respond "
                "with JSON ONLY of the same shape.\n\n"
                f"OBJECTIVE:\n{text[:2500]}\n\nDRAFT BRIEF:\n"
                + _json.dumps(base, default=str)[:4000])
            data, _resp = brain_for(self.context).chat_json(
                [Message.user(prompt)],
                task_kind="plan",
                params=SamplingParams(temperature=0.1, max_tokens=900))
            if not isinstance(data, dict):
                return None
            merged = dict(base)
            for key in ("intent", "subgoals", "inputs", "constraints",
                        "acceptance", "tools", "risks"):
                val = data.get(key)
                if key == "intent":
                    if isinstance(val, str) and val.strip():
                        merged[key] = val.strip()[:220]
                elif isinstance(val, list):
                    merged[key] = [str(v)[:220] for v in val
                                   if str(v).strip()][:16]
            if not merged.get("subgoals"):
                merged["subgoals"] = base["subgoals"]
            return merged
        except Exception as exc:  # noqa: BLE001
            _log.debug("brief polish failed: %s", exc)
            return None

    # ── rendering ──────────────────────────────────────────────────────────
    def _render(self, brief: dict[str, Any], *, for_: str = "mission") -> str:
        lines = [f"Task: {brief.get('intent') or '(no intent detected)'}"]
        if brief.get("subgoals"):
            lines.append("Subgoals:")
            lines.extend(f"  {i}. {s}" for i, s in
                         enumerate(brief["subgoals"], 1))
        if brief.get("inputs"):
            lines.append("Inputs: " + ", ".join(brief["inputs"][:8]))
        if brief.get("constraints"):
            lines.append("Constraints:")
            lines.extend(f"  - {c}" for c in brief["constraints"][:6])
        if brief.get("acceptance"):
            lines.append("Acceptance (done means):")
            lines.extend(f"  - {a}" for a in brief["acceptance"][:6])
        if brief.get("tools") and for_ != "chat":
            lines.append("Suggested tools: " + ", ".join(brief["tools"]))
        if brief.get("risks"):
            lines.append("Side effects to handle carefully: "
                         + ", ".join(brief["risks"][:6]))
        return "\n".join(lines)


def _tool_catalog(registry: Any, limit: int = 400) -> list[tuple[str, str]]:
    """(name, description) pairs from a live registry — never raises."""
    if registry is None:
        return []
    out: list[tuple[str, str]] = []
    try:
        names = registry.names()
        for name in names[:limit]:
            tool = getattr(registry, "get", lambda *_a, **_k: None)(name) \
                if hasattr(registry, "get") else None
            desc = ""
            if tool is not None:
                desc = str(getattr(tool, "description", "") or "")
            out.append((name, desc))
    except Exception:  # noqa: BLE001
        try:
            out = [(n, "") for n in list(registry.names())[:limit]]
        except Exception:  # noqa: BLE001
            out = []
    return out


def structure_text(context: Any, text: str, *, for_: str = "mission",
                   polish: bool = True) -> dict[str, Any]:
    """Module-level convenience for the registry tool."""
    return PromptArchitect(context).structure(text, for_=for_,
                                              polish=polish)


# ── registry ───────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "structure",
        description=(
            "Prompt/mission structuring: turn a raw objective into a "
            "structured brief — intent, ordered subgoals, inputs, "
            "constraints, acceptance criteria, matching tools, and side "
            "effects. The deterministic floor; one optional model polish "
            "pass. action=structure (default) | brief (rendered text)."
        ),
        capability="model.call",
        parameters={
            "action": "str — structure|brief",
            "text": "str — the raw objective/instruction",
            "for": "str — mission|goal|chat (tunes the rendering)",
            "polish": "bool (str) — allow the optional model pass",
        },
    )
    def structure(
        text: str = "", *, action: str = "structure", for_: str = "mission",
        polish: str = "true",
    ) -> dict[str, Any]:
        brief = structure_text(
            context, text, for_=for_.strip().lower(),
            polish=polish.strip().lower() in {"1", "true", "yes", ""})
        if (action or "").strip().lower() == "brief":
            return {"ok": True, "brief": brief.get("brief", "")}
        return {"ok": True, **brief}
