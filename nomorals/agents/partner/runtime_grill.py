"""RuntimeGrillMixin — /grill-me chat control.

``/grill-me`` reads the records Devon already keeps — saved plans,
the commitment ledger, live delegation runs — and says plainly
where the owner stands. Every line names a real record; an empty
record answers honestly instead of inventing a roast.

Owner-only: absent from PUBLIC_COMMANDS, like /plan.
"""

from __future__ import annotations

from ..grill import SECTIONS, grill_me
from .failures import record_failure

__all__ = ["RuntimeGrillMixin"]


class RuntimeGrillMixin:
    """Chat control for the grill."""

    def _control_grill_me(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """``/grill-me [plans|commitments|runs]`` — roast from the records.

        A bare verb grills the whole board; a section word grills
        just that part. Anything else is a usage line, not a guess.
        """
        text = " ".join(str(tail or "").split()).strip().lower()
        if text and text not in SECTIONS:
            return ("usage: /grill-me [plans | commitments | runs] — "
                    "a roast from your own records, or the whole board.")
        try:
            return grill_me(
                context=getattr(self, "context", None),
                manager=self._plan_manager(),
                section=text or None)
        except ValueError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - control never raises
            record_failure("grill.control", exc)
            return f"the grill failed: {exc}"
