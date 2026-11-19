"""Picks (Craft-Libs port of ai-core ``picks.rs``).

Given what the connections offer, choose the models worth
showing: the curated pick from the catalog (only when the
connection actually offers that model), the lowest cost (a free
model wins; an unpriced hosted model is never guessed at), and
the newest. No data, no pick.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from .capability import CapabilityId

__all__ = [
    "Catalog",
    "CatalogEntry",
    "MAX_CATALOG_BYTES",
    "Pick",
    "Picked",
    "choose",
]

MAX_CATALOG_BYTES = 256 * 1024

_CLONE_CATALOG = (Path.home() / "workspace" / "echelon-mining" / "repos"
                  / "storytold" / "craft-libs" / "crates" / "ai-core"
                  / "catalog.json")


class Pick(Enum):
    RECOMMENDED = "recommended"
    BEST_RESULT = "best_result"
    FASTEST = "fastest"
    MOST_POPULAR = "most_popular"
    LOWEST_COST = "lowest_cost"
    NEWEST = "newest"


class CatalogEntry:
    """One curated pick: for this capability, this pick is that
    provider's model."""

    def __init__(self, capability, pick, provider: str, model: str
                 ) -> None:
        self.capability = capability if isinstance(capability,
                                                   CapabilityId) \
            else CapabilityId.parse(capability)
        self.pick = pick if isinstance(pick, Pick) else Pick(pick)
        self.provider = provider
        self.model = model

    def to_dict(self) -> dict:
        return {"capability": self.capability.as_str(),
                "pick": self.pick.value, "provider": self.provider,
                "model": self.model}

    @classmethod
    def from_dict(cls, data: dict) -> CatalogEntry:
        return cls(data["capability"], data["pick"], data["provider"],
                   data["model"])

    def __eq__(self, other) -> bool:
        return (isinstance(other, CatalogEntry)
                and self.to_dict() == other.to_dict())


class Catalog:
    """The curated catalog: version 1, a list of entries."""

    def __init__(self, version: int = 1, entries: list | None = None
                 ) -> None:
        self.version = version
        self.entries: list[CatalogEntry] = list(entries or [])

    @classmethod
    def builtin(cls) -> Catalog:
        """The catalog shipped with ai-core. It starts empty: no
        curated pick is claimed until it is checked."""
        if _CLONE_CATALOG.exists():
            try:
                return cls.from_json(_CLONE_CATALOG.read_text())
            except ValueError:
                pass
        return cls()

    @classmethod
    def from_json(cls, text: str) -> Catalog:
        if len(text.encode("utf-8")) > MAX_CATALOG_BYTES:
            raise ValueError(
                f"catalog is over {MAX_CATALOG_BYTES} bytes")
        data = json.loads(text)
        version = int(data.get("version", 1))
        if version != 1:
            raise ValueError(f"catalog version {version} is not read")
        return cls(version, [CatalogEntry.from_dict(e)
                             for e in data.get("entries", [])])

    def to_dict(self) -> dict:
        return {"version": self.version,
                "entries": [e.to_dict() for e in self.entries]}

    def __eq__(self, other) -> bool:
        return (isinstance(other, Catalog)
                and self.to_dict() == other.to_dict())


class Picked:
    """One pick: which pick, which candidate (by index), and why.
    ``by`` is ``("price", micros)``, ``("released", unix_secs)``,
    ``("free", None)`` or ``("catalog", None)``."""

    def __init__(self, pick: Pick, candidate: int, by) -> None:
        self.pick = pick
        self.candidate = candidate
        self.by = by

    def __repr__(self) -> str:
        return (f"Picked({self.pick.value}, candidate={self.candidate}, "
                f"by={self.by!r})")


def _price(model, capability: CapabilityId) -> int | None:
    """What one run costs, in micros. Text models compare input
    plus output per million tokens; everything else compares the
    per-image price. A missing price is never guessed at."""
    pricing = model.pricing
    if pricing is None:
        return None
    if capability.modality().as_str() == "text":
        parts = [p for p in (pricing.input_micros_per_mtok,
                             pricing.output_micros_per_mtok)
                 if p is not None]
        return sum(parts) if parts else None
    return pricing.per_image_micros


def choose(capability: CapabilityId, candidates,
           catalog: Catalog) -> list[Picked]:
    """Pick the models worth showing for a capability.

    ``candidates`` is a list of ``{"provider", "model", "free"}``
    dicts. Curated picks come from the catalog and only when the
    connection offers the model; lowest cost and newest are worked
    out from what is reported."""
    picked: list[Picked] = []
    chosen: set[Pick] = set()
    for entry in catalog.entries:
        if entry.capability != capability or entry.pick in chosen:
            continue
        for index, candidate in enumerate(candidates):
            model = candidate["model"]
            if (candidate.get("provider") == entry.provider
                    and model.id.as_str() == entry.model):
                picked.append(Picked(entry.pick, index, ("catalog", None)))
                chosen.add(entry.pick)
                break
    cheapest = None
    cheapest_price = None
    for index, candidate in enumerate(candidates):
        model = candidate["model"]
        if candidate.get("free"):
            cheapest = (index, ("free", None))
            cheapest_price = -1
            break
        price = _price(model, capability)
        if price is not None and (cheapest_price is None
                                  or price < cheapest_price):
            cheapest = (index, ("price", price))
            cheapest_price = price
    if cheapest is not None and Pick.LOWEST_COST not in chosen:
        picked.append(Picked(Pick.LOWEST_COST, cheapest[0], cheapest[1]))
    newest = None
    newest_released = None
    for index, candidate in enumerate(candidates):
        released = candidate["model"].released
        if released is not None and (newest_released is None
                                     or released > newest_released):
            newest = index
            newest_released = released
    if newest is not None and Pick.NEWEST not in chosen:
        picked.append(Picked(Pick.NEWEST, newest,
                             ("released", newest_released)))
    return picked
