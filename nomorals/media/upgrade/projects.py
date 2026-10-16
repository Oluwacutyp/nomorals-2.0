"""Non-media extras — project rooms, semantic asset library, highlights,
silence/beat detection, thumbnail A/B, review room, budget ledger.

- :class:`ProjectRoom` holds one project's media state: bible,
  history (op stacks), palette, assets, budget ledger.
- :class:`AssetLibrary` is semantic search over assets: without an
  embedding model it falls back to deterministic token overlap over
  tags + captions (documented); with ``embed`` it uses cosine over
  embeddings.
- :func:`auto_highlights` scores 1s windows from audio energy + face
  presence + motion (all caller-supplied per-second measurements).
- :func:`silence_spans` / :func:`beat_times` are pure signal helpers
  over sample/energy series.
- :func:`thumbnail_ab` picks the thumbnail with the best measured
  critique score.
- :class:`ReviewRoom` holds pending renders + scores + approve/branch.
- :class:`BudgetLedger` tracks render spend per project against a cap
  and refuses to overspend silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .bible import SceneBible
from .critique import Critique, critique_render
from .opstack import OpStack
from .recipes import Recipe


@dataclass
class Asset:
    id: str
    kind: str  # image | video | audio | text
    path: str
    tags: list[str] = field(default_factory=list)
    caption: str = ""

    def text(self) -> str:
        return " ".join([self.caption] + self.tags + [self.kind]).lower()


class AssetLibrary:
    def __init__(self) -> None:
        self._assets: list[Asset] = []

    def add(self, asset: Asset) -> None:
        self._assets.append(asset)

    def search(self, query: str, *, top_n: int = 5,
               embed: Callable[[str], list[float]] | None = None) -> list[Asset]:
        """Semantic search: cosine over embeddings, else token overlap."""
        if embed is not None:
            q = embed(query)
            def _score(a: Asset) -> float:
                v = embed(a.text())
                num = sum(x * y for x, y in zip(q, v))
                den = math.sqrt(sum(x * x for x in q)) * math.sqrt(sum(y * y for y in v))
                return num / den if den else 0.0
            return sorted(self._assets, key=_score, reverse=True)[:top_n]
        qtokens = set(query.lower().split())
        def _overlap(a: Asset) -> float:
            atokens = set(a.text().split())
            return len(qtokens & atokens) / max(1, len(qtokens))
        return sorted(self._assets, key=_overlap, reverse=True)[:top_n]


class BudgetLedger:
    """Render budget per project: spend is recorded, cap is enforced."""

    def __init__(self, cap: float = 0.0) -> None:
        self.cap = float(cap)
        self.spent = 0.0
        self.entries: list[dict[str, Any]] = []

    def charge(self, amount: float, *, what: str) -> bool:
        """Record a charge; returns False (and does NOT charge) if over cap."""
        if amount < 0:
            raise ValueError("charge amount must be >= 0")
        if self.cap > 0 and self.spent + amount > self.cap:
            self.entries.append({"what": what, "amount": amount, "status": "blocked_over_cap"})
            return False
        self.spent += amount
        self.entries.append({"what": what, "amount": amount, "status": "charged"})
        return True

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap - self.spent) if self.cap > 0 else float("inf")


@dataclass
class ProjectRoom:
    name: str
    bible: SceneBible | None = None
    history: list[OpStack] = field(default_factory=list)
    palette: list[str] = field(default_factory=list)
    assets: AssetLibrary = field(default_factory=AssetLibrary)
    ledger: BudgetLedger = field(default_factory=BudgetLedger)
    recipes: list[Recipe] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name,
                "bible": self.bible.to_dict() if self.bible else None,
                "history_stacks": len(self.history),
                "palette": list(self.palette),
                "spent": self.ledger.spent, "cap": self.ledger.cap,
                "recipes": len(self.recipes)}


def auto_highlights(energy: list[float], faces: list[float], motion: list[float],
                    *, top_n: int = 3) -> list[dict[str, Any]]:
    """Top highlight windows: 0.5*energy + 0.3*face + 0.2*motion per second."""
    n = min(len(energy), len(faces), len(motion))
    scored = [{"second": i,
               "score": round(0.5 * energy[i] + 0.3 * faces[i] + 0.2 * motion[i], 4)}
              for i in range(n)]
    return sorted(scored, key=lambda d: d["score"], reverse=True)[:top_n]


def silence_spans(samples: list[float], *, threshold: float = 0.02,
                  min_len: int = 2) -> list[dict[str, int]]:
    """Contiguous runs below ``threshold`` lasting >= ``min_len`` samples."""
    spans, start = [], None
    for i, v in enumerate(samples + [threshold + 1.0]):
        if v < threshold and start is None:
            start = i
        elif v >= threshold and start is not None:
            if i - start >= min_len:
                spans.append({"start": start, "end": i, "length": i - start})
            start = None
    return spans


def beat_times(energy: list[float], *, threshold: float = 0.7) -> list[int]:
    """Local energy peaks above ``threshold`` = beat candidates."""
    beats = []
    for i in range(1, len(energy) - 1):
        if energy[i] >= threshold and energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1]:
            beats.append(i)
    return beats


def thumbnail_ab(thumbnails: list[dict[str, Any]], *, brief: str = "") -> dict[str, Any] | None:
    """A/B: the thumbnail with the best critique overall wins."""
    best, best_score = None, -1.0
    for thumb in thumbnails:
        crit = critique_render(brief, scores=dict(thumb.get("scores", {})),
                               image=thumb.get("image"))
        if crit.overall > best_score:
            best, best_score = thumb, crit.overall
    return best


class ReviewRoom:
    """Pending renders + scores + approve / branch decisions."""

    def __init__(self) -> None:
        self.pending: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, str] = {}  # id -> approved | branched | rejected

    def submit(self, render_id: str, candidate: dict[str, Any], *,
               brief: str = "") -> Critique:
        crit = critique_render(brief, scores=dict(candidate.get("scores", {})),
                               image=candidate.get("image"))
        self.pending[render_id] = {"candidate": candidate, "critique": crit}
        return crit

    def approve(self, render_id: str) -> bool:
        entry = self.pending.get(render_id)
        if entry is None:
            raise KeyError(f"no pending render {render_id!r}")
        if not entry["critique"].ship:
            return False  # floor invariant: cannot approve below-floor work
        self.decisions[render_id] = "approved"
        return True

    def branch(self, render_id: str) -> None:
        if render_id not in self.pending:
            raise KeyError(f"no pending render {render_id!r}")
        self.decisions[render_id] = "branched"

    def reject(self, render_id: str) -> None:
        if render_id not in self.pending:
            raise KeyError(f"no pending render {render_id!r}")
        self.decisions[render_id] = "rejected"
