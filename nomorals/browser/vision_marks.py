"""Set-of-Mark vision escalation: perception tiers for the hard 4% of pages.

* Tier 1 — numbered elements (default): the snapshot's element list, each
  with a stable mark id. The model says "click mark 17".
* Tier 2 — runtime structural DOM incl. shadow roots: deeper element
  extraction when Tier 1 has no target.
* Tier 3 — SoM vision: numbered marks overlaid on a screenshot; the model
  says "click mark 17", NEVER pixel coordinates. Pixel coords are
  ungrounded, resolution-dependent, and hallucination-prone.

Escalate when: target not in snapshot, same click failed twice, canvas /
WebGL, snapshot thin vs viewport. After acting in Tier 3, verify in
Tier 1 (read-back on the snapshot) — a Tier 3 act is not verified by
Tier 3.

Vision is injected (``see`` callable) — this module never imports a
vision backend directly, so tests run without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["MarkedElement", "PerceptionEscalator", "SomOverlay", "Tier"]

TIER_NUMBERED = 1
TIER_STRUCTURAL = 2
TIER_SOM_VISION = 3
Tier = int


@dataclass
class MarkedElement:
    """One element with a stable mark id the model can reference."""

    mark: int
    label: str
    kind: str = "link"
    selector: str = ""
    rect: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"mark": self.mark, "label": self.label, "kind": self.kind,
                "selector": self.selector, "rect": dict(self.rect)}


@dataclass
class SomOverlay:
    """Numbered marks over a screenshot — the Tier 3 view."""

    screenshot_path: str
    marks: list[MarkedElement] = field(default_factory=list)
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"screenshot": self.screenshot_path,
                "marks": [m.to_dict() for m in self.marks],
                "prompt": self.prompt}


class PerceptionEscalator:
    """Owns the current tier and the escalation rules."""

    def __init__(self, *,
                 see: Callable[[str, str], str] | None = None,
                 screenshot: Callable[[], str] | None = None,
                 extract_structural: Callable[[], list[dict[str, Any]]] | None = None) -> None:
        self.tier: Tier = TIER_NUMBERED
        self._see = see
        self._screenshot = screenshot
        self._extract_structural = extract_structural
        self._failed_clicks: dict[str, int] = {}
        self.escalations = 0

    def should_escalate(self, *, target_found: bool, last_click_failed: bool,
                        is_canvas: bool, snapshot_thin: bool,
                        click_target: str = "") -> bool:
        """Pure escalation rule — testable without a browser."""
        if click_target and last_click_failed:
            self._failed_clicks[click_target] = \
                self._failed_clicks.get(click_target, 0) + 1
            if self._failed_clicks[click_target] >= 2:
                return True
        return (not target_found) or is_canvas or snapshot_thin

    def escalate(self) -> Tier:
        """Move up one tier (1 -> 2 -> 3, stays at 3)."""
        if self.tier < TIER_SOM_VISION:
            self.tier += 1
            self.escalations += 1
        return self.tier

    def reset(self) -> None:
        self.tier = TIER_NUMBERED
        self._failed_clicks.clear()

    def mark_elements(self, elements: list[dict[str, Any]]) -> list[MarkedElement]:
        """Assign stable 1-based marks to a snapshot element list."""
        out: list[MarkedElement] = []
        for i, el in enumerate(elements, start=1):
            out.append(MarkedElement(
                mark=i, label=str(el.get("label", "")),
                kind=str(el.get("kind", "link")),
                selector=str(el.get("selector", el.get("id", ""))),
                rect=dict(el.get("rect", {}) or {})))
        return out

    def som_view(self, elements: list[dict[str, Any]], *,
                 goal: str = "") -> SomOverlay:
        """Build the Tier 3 view: screenshot + numbered marks + prompt.

        The model must answer with a mark number, never pixel coords.
        """
        if self._screenshot is None:
            raise RuntimeError(
                "SoM vision needs a screenshot callable — none injected")
        marks = self.mark_elements(elements)
        prompt = (
            f"Goal: {goal}\n"
            "You see a screenshot with numbered marks. Answer with a mark "
            "number only, e.g. 'click mark 17'. NEVER answer with pixel "
            "coordinates — they are ungrounded and will be rejected."
        )
        return SomOverlay(screenshot_path=self._screenshot(),
                          marks=marks, prompt=prompt)

    def see_marks(self, overlay: SomOverlay, *, question: str = "") -> str:
        """Ask the injected vision callable about an SoM overlay."""
        if self._see is None:
            raise RuntimeError(
                "SoM vision needs a vision callable — none injected")
        return self._see(overlay.screenshot_path, question or overlay.prompt)

    @staticmethod
    def parse_mark(answer: str, marks: list[MarkedElement]) -> MarkedElement | None:
        """Parse 'click mark 17' -> the marked element. Pixel coords rejected."""
        import re
        m = re.search(r"mark\s+(\d+)", answer or "", re.IGNORECASE)
        if not m:
            return None
        wanted = int(m.group(1))
        for el in marks:
            if el.mark == wanted:
                return el
        return None
