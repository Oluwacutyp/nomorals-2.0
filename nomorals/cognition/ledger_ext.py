"""Ledger extension: stakes, counterparty memory, prediction.

Wraps :class:`RepresentationLedger
<nomorals.cognition.representation.RepresentationLedger>` — it does
*not* duplicate its logic.  All persistence, outcome vocabulary,
feedback handling, recency decay and scoring live in the wrapped
ledger; this module only adds three layers on top, carried in the
record's ``metadata`` (which the base ledger already stores):

- **Stakes weighting** — every autonomous action records its stakes
  (``low`` / ``medium`` / ``high`` / ``critical`` or a 0..1 number).
  :meth:`ExtendedRepresentationLedger.stakes_weighted_quality_score`
  reuses the base ledger's own ``_score_value`` mapping (approval /
  disapproval handling included) and weights each record by its
  stakes, so a high-stakes failure counts more than a low-stakes one.
- **Counterparty memory** — ``counterparty`` (who the action was
  with/to) is stored per record; :meth:`counterparty_summary` rebuilds
  per-counterparty history from the wrapped ledger, so memory survives
  a restart instead of living only in process RAM.
- **Prediction, not permission** — :meth:`predict_outcome` returns an
  *advisory* predicted quality / risk.  It never grants or denies
  permission: ``permission_required`` is always ``False`` in its
  output because the prediction itself is not an authorization, and
  :meth:`record_autonomous_action` never blocks on a low prediction —
  a prediction the owner hated is still recorded honestly.

Feedback asymmetry is inherited from the base ledger and preserved
here: owner disapproval (-1) hits harder than approval (+1) helps
(the base ``_score_value`` subtracts a flat 0.5 on disapproval while
approval only multiplies by 1.5 capped at 1.0), and ``unknown``
outcomes with no feedback carry zero weight — they never inflate a
score.  Those rules are *reused*, never reimplemented.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .action_registry import (
    ACTION_CLASSES,
    ACTION_CLASS_TO_LEDGER_TYPE,
    normalize_action_class,
)
from .representation import OUTCOMES, RepresentationLedger

_log = logging.getLogger(__name__)

__all__ = [
    "STAKES_WEIGHTS",
    "normalize_stakes",
    "ExtendedRepresentationLedger",
]

#: Stakes label -> weight used in stakes-weighted scoring.
STAKES_WEIGHTS = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


def normalize_stakes(stakes: str | float | int | None) -> tuple[str, float]:
    """Normalize a stakes value to ``(label, weight)``.

    Accepts a label (``low`` / ``medium`` / ``high`` / ``critical``,
    case-insensitive) or a number in [0, 1] (nearest label wins).
    ``None`` means ``"medium"``.  Raises :class:`ValueError` on an
    unknown label or an out-of-range number.
    """
    if stakes is None:
        return "medium", STAKES_WEIGHTS["medium"]
    if isinstance(stakes, (int, float)) and not isinstance(stakes, bool):
        weight = float(stakes)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"stakes {stakes!r} must be in [0, 1]")
        label = min(STAKES_WEIGHTS, key=lambda k: abs(STAKES_WEIGHTS[k] - weight))
        return label, weight
    label = str(stakes).strip().lower()
    if label not in STAKES_WEIGHTS:
        raise ValueError(
            f"unknown stakes {stakes!r}; use one of {sorted(STAKES_WEIGHTS)} or a 0..1 number")
    return label, STAKES_WEIGHTS[label]


class ExtendedRepresentationLedger:
    """Representation ledger + stakes + counterparty + prediction.

    ``ledger`` may be an existing :class:`RepresentationLedger`
    (shared storage) or ``None`` (a fresh in-memory ledger).  Unknown
    attributes (``quality_score``, ``recent``, ``get``, ``distill``,
    ...) are forwarded to the wrapped ledger via ``__getattr__`` so
    this class stays a thin extension, not a fork.
    """

    def __init__(
        self,
        ledger: RepresentationLedger | None = None,
        db: Any | None = None,
    ) -> None:
        if ledger is not None:
            self.ledger = ledger
        else:
            self.ledger = RepresentationLedger(db)

    # ── recording ────────────────────────────────────────────────────
    def record_autonomous_action(
        self,
        action_class: str,
        stakes: str | float | int | None = "medium",
        counterparty: str = "",
        outcome: str = "unknown",
        owner_feedback: int | str | None = None,
        *,
        description: str = "",
        predicted_confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Record one autonomous action with stakes + counterparty.

        Never blocks on prediction: a low predicted quality is stored
        alongside the record, never used to refuse the recording.
        Never raises on storage failure (returns a placeholder record
        with ``id == ""``, mirroring the base ledger's contract).
        Raises :class:`ValueError` for an unknown ``action_class``,
        unknown ``outcome``, or bad ``stakes`` — those are programmer
        errors, not storage failures.
        """
        action_class = normalize_action_class(action_class)
        outcome = outcome or "unknown"
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; use {OUTCOMES}")
        stakes_label, stakes_weight = normalize_stakes(stakes)
        counterparty = (counterparty or "").strip()
        merged: dict[str, Any] = {
            "action_class": action_class,
            "stakes": stakes_label,
            "stakes_weight": stakes_weight,
            "counterparty": counterparty,
            "predicted_confidence": predicted_confidence,
        }
        if metadata:
            # Caller metadata wins on collision, except the identity
            # fields above which must stay truthful.
            for key, value in metadata.items():
                if key not in merged:
                    merged[key] = value
        try:
            rec = self.ledger.log_action(
                ACTION_CLASS_TO_LEDGER_TYPE[action_class],
                description or f"{action_class} action"
                + (f" with {counterparty}" if counterparty else ""),
                outcome=outcome,
                metadata=merged,
            )
            if owner_feedback is not None and getattr(rec, "id", ""):
                self.ledger.record_outcome(
                    rec.id, outcome, owner_feedback=owner_feedback)
                refreshed = self.ledger.get(rec.id)
                if refreshed is not None:
                    return refreshed
            return rec
        except Exception as exc:  # noqa: BLE001 — logging never breaks actions
            _log.warning("extended representation log failed: %s", exc)
            from .representation import ActionRecord

            return ActionRecord(
                id="", action_type=ACTION_CLASS_TO_LEDGER_TYPE[action_class],
                description=description or "",
                outcome=outcome,
                metadata=merged,
            )

    # ── prediction (advisory only — never permission) ─────────────────
    def predict_outcome(
        self,
        action_class: str,
        stakes: str | float | int | None = "medium",
        counterparty: str = "",
    ) -> dict[str, Any]:
        """Advisory prediction for an action *before* it is taken.

        Returns ``{"predicted_quality", "risk", "recommendation",
        "permission_required": False, ...}``.  The prediction is a
        forecast, not an authorization: it neither grants nor denies
        permission, and callers must still obtain whatever approval
        the action itself requires.  ``predicted_quality`` blends the
        base ledger's quality for the mapped type with the
        counterparty's own history when there is enough of it (>= 3
        actions); ``risk`` is stakes × (1 − quality).
        """
        action_class = normalize_action_class(action_class)
        _, stakes_weight = normalize_stakes(stakes)
        counterparty = (counterparty or "").strip()
        base_type = ACTION_CLASS_TO_LEDGER_TYPE[action_class]
        base_quality = self.ledger.quality_score(base_type)
        quality = base_quality
        cp_actions = 0
        if counterparty:
            cp = self.counterparty_summary(counterparty)
            cp_actions = int(cp.get("actions", 0))
            if cp_actions >= 3 and cp.get("quality") is not None:
                quality = 0.5 * base_quality + 0.5 * float(cp["quality"])
        risk = stakes_weight * (1.0 - quality)
        if quality >= 0.7:
            recommendation = (
                "Predicted to represent you well — advisory only;"
                " prediction is not permission.")
        elif quality >= 0.4:
            recommendation = (
                "Mixed track record — proceed with care; prediction is"
                " not permission.")
        else:
            recommendation = (
                "Weak track record — consider a different approach or"
                " asking the owner first; prediction is not permission.")
        return {
            "action_class": action_class,
            "base_type": base_type,
            "counterparty": counterparty,
            "stakes_weight": stakes_weight,
            "predicted_quality": round(quality, 3),
            "base_quality": round(base_quality, 3),
            "counterparty_actions": cp_actions,
            "risk": round(risk, 3),
            "recommendation": recommendation,
            # Prediction is a forecast, never an authorization.
            "permission_required": False,
            "grants_permission": False,
            "denies_permission": False,
        }

    # ── stakes-weighted scoring (reuses base _score_value) ───────────
    def _known_records(
        self,
        *,
        action_class: str | None = None,
        counterparty: str | None = None,
        since_days: float | None = 30.0,
    ) -> list[Any]:
        """Records with a known outcome or feedback.

        Mirrors the base ledger's honesty rule: ``unknown`` outcomes
        with no owner feedback carry zero weight and are excluded —
        they never inflate a score.
        """
        recs = self.ledger.recent(limit=10_000, since_days=since_days)
        out = []
        for rec in recs:
            if rec.outcome == "unknown" and rec.owner_feedback is None:
                continue
            meta = rec.metadata or {}
            if action_class is not None and meta.get("action_class") != action_class:
                continue
            if counterparty is not None and meta.get("counterparty") != counterparty:
                continue
            out.append(rec)
        return out

    def stakes_weighted_quality_score(
        self,
        action_class: str | None = None,
        *,
        counterparty: str | None = None,
        since_days: float | None = 30.0,
    ) -> float:
        """Stakes-weighted quality 0..1 (0.5 prior with no known data).

        Each record's value comes from the base ledger's own
        ``_score_value`` (disapproval −1 hits harder than approval +1
        helps — reused, not reimplemented); each record's weight is
        its stakes weight.  A high-stakes failure therefore counts
        more than a low-stakes one, and unknown outcomes never inflate.
        """
        recs = self._known_records(
            action_class=action_class, counterparty=counterparty,
            since_days=since_days)
        num = 0.0
        den = 0.0
        for rec in recs:
            weight = float((rec.metadata or {}).get("stakes_weight", 0.5))
            value = RepresentationLedger._score_value(rec.outcome, rec.owner_feedback)
            num += weight * value
            den += weight
        return (num / den) if den > 0.0 else 0.5

    # ── counterparty memory ──────────────────────────────────────────
    def counterparty_summary(self, counterparty: str) -> dict[str, Any]:
        """Per-counterparty history rebuilt from the wrapped ledger.

        ``{"counterparty", "actions", "approvals", "disapprovals",
        "feedback_rate", "avg_stakes_weight", "quality"}`` — ``quality``
        is the stakes-weighted score for this counterparty (None when
        there is no known-outcome data).  Rebuilt from stored records
        on every call, so memory survives restarts.
        """
        name = (counterparty or "").strip()
        recs = self.ledger.recent(limit=10_000, since_days=None)
        mine = [r for r in recs if (r.metadata or {}).get("counterparty") == name]
        approvals = sum(1 for r in mine if r.owner_feedback == 1)
        disapprovals = sum(1 for r in mine if r.owner_feedback == -1)
        known = [
            r for r in mine
            if not (r.outcome == "unknown" and r.owner_feedback is None)
        ]
        quality: float | None = None
        if known:
            quality = self.stakes_weighted_quality_score(
                counterparty=name, since_days=None)
        avg_stakes = (
            sum(float((r.metadata or {}).get("stakes_weight", 0.5)) for r in mine)
            / len(mine) if mine else 0.0
        )
        by_class: dict[str, int] = {}
        for r in mine:
            cls = (r.metadata or {}).get("action_class") or r.action_type
            by_class[str(cls)] = by_class.get(str(cls), 0) + 1
        return {
            "counterparty": name,
            "actions": len(mine),
            "approvals": approvals,
            "disapprovals": disapprovals,
            "feedback_rate": (
                (approvals + disapprovals) / len(mine) if mine else 0.0),
            "avg_stakes_weight": round(avg_stakes, 3),
            "quality": (round(quality, 3) if quality is not None else None),
            "by_action_class": by_class,
        }

    def counterparties(self) -> list[str]:
        """Every counterparty with at least one recorded action, sorted."""
        recs = self.ledger.recent(limit=10_000, since_days=None)
        names = {
            (r.metadata or {}).get("counterparty")
            for r in recs
            if (r.metadata or {}).get("counterparty")
        }
        return sorted(str(n) for n in names)

    # ── owner feed ───────────────────────────────────────────────────
    def summary(self, *, limit: int = 10, since_days: float = 30.0) -> str:
        """Owner review: base ledger summary + stakes + counterparties."""
        lines = [self.ledger.summary(limit=limit, since_days=since_days)]
        lines.append("")
        lines.append("⚖️ Stakes-weighted quality (high-stakes counts more):")
        any_stakes = False
        for cls in ACTION_CLASSES:
            recs = self._known_records(action_class=cls, since_days=since_days)
            if not recs:
                continue
            any_stakes = True
            score = self.stakes_weighted_quality_score(
                cls, since_days=since_days)
            lines.append(f"• {cls}: {score:.0%} over {len(recs)} known action(s)")
        if not any_stakes:
            lines.append("No known-outcome actions in this window.")
        cps = self.counterparties()
        if cps:
            lines.append("")
            lines.append("🤝 Counterparty memory:")
            for name in cps:
                cp = self.counterparty_summary(name)
                q = f"{cp['quality']:.0%}" if cp["quality"] is not None else "no data"
                lines.append(
                    f"• {name}: {cp['actions']} action(s),"
                    f" 👍 {cp['approvals']} / 👎 {cp['disapprovals']},"
                    f" quality {q}")
        return "\n".join(lines)

    # ── forwarding: everything else lives in the wrapped ledger ──────
    def __getattr__(self, name: str) -> Any:
        if name in {"ledger"}:
            raise AttributeError(name)
        return getattr(self.__dict__["ledger"], name)


#: Convenience alias — the extension is the ledger call sites use.
__all__ += []
