"""Game state capture: turn a page into a structured snapshot.

:class:`GameState` is deliberately game-agnostic: URL, title, readable text,
currency-like numbers, and the interactive elements the decider can act on.
Per-game meaning ("this ₦ figure is my rent") belongs in strategies, not here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ...core.logging_setup import get_logger
from ...tools.web import html_to_text
from .driver import BrowserDriver, InteractiveElement, Page, extract_elements

_log = get_logger(__name__)

__all__ = ["GameState", "capture_state", "extract_numbers", "state_from_page"]

#: ₦1,234 · $5.00 · 1,234 coins · Balance: 500 · Level 12 · 116 XP
_NUMBER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("naira", re.compile(r"₦\s*([\d,]+(?:\.\d+)?)")),
    ("dollars", re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")),
    ("coins", re.compile(r"(?i)([\d,]+)\s*coins?\b")),
    ("balance", re.compile(r"(?i)balance[:\s]+([\d,]+(?:\.\d+)?)")),
    ("level", re.compile(r"(?i)\blevel\s+(\d+)")),
    ("xp", re.compile(r"(?i)([\d,]+)\s*xp\b")),
    ("energy", re.compile(r"(?i)energy[:\s]+(\d+)")),
)


def _to_float(raw: str) -> float:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return 0.0


def extract_numbers(text: str) -> dict[str, float]:
    """Pull currency-like figures out of page text.

    First match per category wins; games repeat balances in headers and
    bodies, and the first occurrence is usually the header summary.
    """
    numbers: dict[str, float] = {}
    for name, pattern in _NUMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            numbers[name] = _to_float(match.group(1))
    return numbers


@dataclass
class GameState:
    """A structured snapshot of where the game stands right now."""

    url: str
    title: str
    text: str
    numbers: dict[str, float] = field(default_factory=dict)
    elements: list[InteractiveElement] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)

    def find_elements(self, *keywords: str) -> list[InteractiveElement]:
        """Elements whose label contains any of the keywords (case-insensitive)."""
        lowered = [k.lower() for k in keywords]
        return [
            el
            for el in self.elements
            if any(k in el.label.lower() for k in lowered)
        ]

    def summary(self, max_chars: int = 2000) -> str:
        """Compact human/LLM-readable snapshot for prompts and logs."""
        lines = [f"url: {self.url}", f"title: {self.title}"]
        if self.numbers:
            lines.append(
                "numbers: "
                + ", ".join(f"{k}={v:g}" for k, v in self.numbers.items())
            )
        lines.append("actions:")
        for el in self.elements[:40]:
            lines.append(f"  [{el.id}] ({el.kind}) {el.label}")
        if len(self.elements) > 40:
            lines.append(f"  … +{len(self.elements) - 40} more")
        text = "\n".join(lines)
        return text[:max_chars]


def _page_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]*>", "", match.group(1))).strip()


def state_from_page(page: Page) -> GameState:
    """Build a :class:`GameState` from an already-fetched page."""
    text = html_to_text(page.html)
    state = GameState(
        url=page.url,
        title=_page_title(page.html),
        text=text,
        numbers=extract_numbers(text),
        elements=extract_elements(page.html, page.url),
    )
    _log.debug(
        "captured state: %s title=%r numbers=%s elements=%d",
        state.url,
        state.title,
        state.numbers,
        len(state.elements),
    )
    return state


def capture_state(driver: BrowserDriver, url: str) -> GameState:
    """Fetch ``url`` and build a :class:`GameState`. Fail-fast on HTTP errors."""
    page: Page = driver.fetch(url)
    return state_from_page(page)
