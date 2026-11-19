"""The consent gate: one decision per effect class, journaled.

Port of upstream ``Consent`` + the consent check in ``Engine::call_inner``
(``crates/engine/src/lib.rs``): a tool whose effects need consent refuses
with ``ConsentRequired`` unless the call carries a grant or a standing
rule allows it; the MCP server turns that refusal into ``consent_required``
so the client's own approval flow decides, unless the server was started
with ``--allow-run``.

Devon's shape of the same rule, with the owner's standing rules built
in:

* Consent is per effect, never a blanket review queue. The ``/workflow``
  review gate was REMOVED by the owner (2026-11-16) — nothing here
  blocks on review; a call is allowed, refused for consent, or denied
  by a standing rule, and the decision is recorded with its reason.
* The owner acting directly (``Caller.UI`` / ``Caller.CLI``) is consent
  by definition — his own call is his OK. Programmatic callers
  (``Caller.AGENT`` / ``Caller.MCP``) get ``consent_required`` for
  consent-class effects unless the call carries ``Consent.GRANTED`` or
  a standing rule allows this tool (or this tool + effect).
* Every decision is journaled — granted, consent_required, denied,
  standing rule — with the tool, the effects, the caller and the reason
  named. The journal is the answer to "what did the agent do while I
  wasn't looking" for consent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import effects as _effects
from .caller import Caller

__all__ = ["Consent", "ConsentDecision", "ConsentGate"]


class Consent(StrEnum):
    """Whether the caller has the user's consent for this call (upstream).

    Upstream has exactly these two variants: not asked, or granted for
    this one call. A denial is not a ``Consent`` value — it is a
    standing rule (see :meth:`ConsentGate.deny_tool`) and a journal
    decision.
    """

    NOT_ASKED = "not_asked"
    GRANTED = "granted"


@dataclass(frozen=True)
class ConsentDecision:
    """One gate decision, with the reason named."""

    decision: str  # "allow" | "consent_required" | "deny"
    reason: str
    rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "reason": self.reason,
                "rule": self.rule}


class ConsentGate:
    """Per-effect consent decisions with standing rules and a journal.

    Standing rules are keyed ``(tool, effect-name-or-None)``: an
    effect-specific rule allows/denies one effect class for a tool, a
    ``None`` effect allows/denies the whole tool. ``allow_run`` is the
    upstream ``--allow-run`` switch: every call is treated as consented.
    """

    def __init__(self, *, allow_run: bool = False) -> None:
        self.allow_run = allow_run
        self._allow: set[tuple[str, str | None]] = set()
        self._deny: set[tuple[str, str | None]] = set()
        self._journal: list[dict[str, Any]] = []

    # ── standing rules ────────────────────────────────────────────────
    def allow_tool(self, tool: str, *,
                   effects: int | None = None) -> None:
        """Standing allow: this tool may run these effects without a grant.

        ``effects=None`` allows the whole tool; otherwise one rule per
        effect name in the mask.
        """
        if effects is None:
            self._allow.add((tool, None))
            return
        for name in _effects.effect_names(effects):
            self._allow.add((tool, name))

    def deny_tool(self, tool: str, *,
                  effects: int | None = None) -> None:
        """Standing deny: this tool may never run these effects.

        A standing deny beats a per-call grant — the owner's explicit
        refusal is final for the tool, and the decision says so.
        """
        if effects is None:
            self._deny.add((tool, None))
            return
        for name in _effects.effect_names(effects):
            self._deny.add((tool, name))

    def standing_rules(self) -> dict[str, list[dict[str, Any]]]:
        """The standing rules, for inspection (never a secret)."""
        return {
            "allow": [{"tool": t, "effect": e}
                      for t, e in sorted(self._allow, key=str)],
            "deny": [{"tool": t, "effect": e}
                     for t, e in sorted(self._deny, key=str)],
            "allow_run": self.allow_run,
        }

    # ── decisions ─────────────────────────────────────────────────────
    def decide(self, tool: str, effects: int, caller: str | Caller,
               *, consent: Consent = Consent.NOT_ASKED) -> ConsentDecision:
        """Decide one call. Records the decision in the journal."""
        caller_name = caller.value if isinstance(caller, Caller) else str(caller)
        decision = self._decide(tool, effects, caller_name, consent)
        self._journal.append({
            "ts": time.time(),
            "tool": tool,
            "effects": _effects.effect_names(effects),
            "caller": caller_name,
            "consent": consent.value,
            "decision": decision.decision,
            "reason": decision.reason,
            "rule": decision.rule,
        })
        return decision

    def _decide(self, tool: str, effects: int, caller: str,
                consent: Consent) -> ConsentDecision:
        names = _effects.effect_names(effects)
        if not _effects.needs_consent(effects):
            return ConsentDecision(
                "allow",
                "read-only or non-consent effects "
                f"({', '.join(names) or 'none'}); no consent needed")
        needed = [name for name in names
                  if _effects.needs_consent(
                      _effects.effects_from_names([name]))]
        needed_text = ", ".join(needed)
        # A standing deny beats everything, including a per-call grant.
        for name in needed:
            if (tool, name) in self._deny or (tool, None) in self._deny:
                return ConsentDecision(
                    "deny",
                    f"standing rule denies {name} for {tool}",
                    rule=f"deny:{tool}:{name}")
        if self.allow_run:
            return ConsentDecision(
                "allow",
                "allow-run standing rule: every call is consented",
                rule="allow_run")
        for name in needed:
            if (tool, name) in self._allow or (tool, None) in self._allow:
                return ConsentDecision(
                    "allow",
                    f"standing rule allows {name} for {tool}",
                    rule=f"allow:{tool}:{name}")
        if consent is Consent.GRANTED:
            return ConsentDecision(
                "allow",
                f"granted for this call ({needed_text})")
        if caller in ("ui", "cli"):
            return ConsentDecision(
                "allow",
                "owner acting directly; the call itself is the consent")
        return ConsentDecision(
            "consent_required",
            f"{tool} needs consent for {needed_text}",
            rule="consent_required")

    # ── journal ───────────────────────────────────────────────────────
    @property
    def journal(self) -> list[dict[str, Any]]:
        return list(self._journal)

    def save_journal(self, path: str | Path) -> Path:
        """Append the journal to a JSONL file (one entry per line)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for entry in self._journal:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return target

    def load_journal(self, path: str | Path) -> int:
        """Load journal entries from a JSONL file. Returns the count."""
        target = Path(path)
        count = 0
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self._journal.append(json.loads(line))
                count += 1
        return count
