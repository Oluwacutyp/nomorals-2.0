"""Property cross-listing signal + property passport (continuity client).

The same property is often listed on several sources (Jumia House,
PropertyPro, private agents) at slightly different prices and with
overlapping photos. Seeing it on N sources is a real signal: a
property listed in many places is likelier to be genuine and
actively marketed than a one-source listing with no history —
and the same listing duplicated across sources is not N properties.

- Dedupe: two listings are the same property when the normalized
  address matches AND the price is within tolerance AND the photo
  sets overlap (share at least one photo hash) — or when neither
  listing carries photos, address + price alone decide. Clustering
  is union-find, so a chain of pairwise matches merges.
- :func:`cross_listing_score`: the same property on N of M sources
  scores N/M — an honest proportion, not a vibes score.
- :class:`PropertyPassport`: the continuity client for a property —
  one versioned, persisted record (address, history, documents,
  viewings) that accumulates across listings instead of resetting
  every time the property is re-listed.

Photo hashes are perceptual/content hashes supplied by the caller
(this module never fetches images). Prices are compared in the
caller's currency units; tolerance is explicit, default 0 (exact).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Listing",
    "PropertyPassport",
    "PassportEvent",
    "DocRef",
    "Viewing",
    "PropertyPassportStore",
    "normalize_address",
    "same_property",
    "group_listings",
    "cross_listing_score",
]

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_address(address: str) -> str:
    """Canonical address key: lowercase, punctuation → space, collapse
    whitespace. ``"12, Allen Avenue, Ikeja"`` and ``"12 Allen Avenue
    Ikeja"`` normalize to the same key. Never raises."""
    try:
        text = (address or "").lower()
        text = _PUNCT_RE.sub(" ", text)
        return _WS_RE.sub(" ", text).strip()
    except Exception:  # noqa: BLE001
        return ""


@dataclass
class Listing:
    """One listing from one source. ``price`` in the source's currency
    units; ``photo_hashes`` are caller-supplied content/perceptual
    hashes of the listing photos."""

    source: str = ""
    address: str = ""
    price: float = 0.0
    photo_hashes: list[str] = field(default_factory=list)
    url: str = ""
    title: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "address": self.address,
                "price": self.price,
                "photo_hashes": list(self.photo_hashes),
                "url": self.url, "title": self.title,
                "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Listing":
        data = data or {}
        try:
            price = float(data.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        photos = data.get("photo_hashes") or []
        return cls(source=str(data.get("source") or ""),
                   address=str(data.get("address") or ""),
                   price=price,
                   photo_hashes=[str(p) for p in photos
                                 if isinstance(photos, list)],
                   url=str(data.get("url") or ""),
                   title=str(data.get("title") or ""),
                   attrs=dict(data.get("attrs") or {}))


def same_property(a: Listing, b: Listing, *,
                  price_tolerance: float = 0.0) -> bool:
    """Pairwise same-property test: normalized address equal AND price
    within ``price_tolerance`` (absolute, same units as ``price``) AND
    photo sets overlap — or neither listing carries photos, in which
    case address + price alone decide. Never raises."""
    try:
        if normalize_address(a.address) != normalize_address(b.address):
            return False
        if not normalize_address(a.address):
            return False
        if abs(float(a.price) - float(b.price)) > max(0.0, price_tolerance):
            return False
        set_a = {str(p).strip() for p in (a.photo_hashes or []) if str(p).strip()}
        set_b = {str(p).strip() for p in (b.photo_hashes or []) if str(p).strip()}
        if not set_a or not set_b:
            return True  # no photos on at least one side: address+price decide
        return bool(set_a & set_b)
    except Exception:  # noqa: BLE001
        return False


def group_listings(listings: list[Listing] | None, *,
                   price_tolerance: float = 0.0) -> list[list[Listing]]:
    """Cluster listings into same-property groups (union-find, so
    pairwise chains merge). Deterministic order: groups sorted by
    first-seen listing. Never raises."""
    try:
        items = list(listings or [])
        parent = list(range(len(items)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if same_property(items[i], items[j],
                                 price_tolerance=price_tolerance):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
        groups: dict[int, list[Listing]] = {}
        order: list[int] = []
        for idx, item in enumerate(items):
            root = find(idx)
            if root not in groups:
                groups[root] = []
                order.append(root)
            groups[root].append(item)
        return [groups[r] for r in order]
    except Exception:  # noqa: BLE001
        _log.debug("group_listings failed", exc_info=True)
        return [[item] for item in (listings or [])]


def cross_listing_score(group: list[Listing] | None,
                        *, total_sources: int = 0) -> dict[str, Any]:
    """Cross-listing signal for one same-property group.

    ``sources`` = distinct sources the property appears on.
    ``score`` = sources / total_sources when the caller knows the
    total source universe, else sources / sources (= 1.0 for any
    non-empty group — reported honestly as ``basis: "group-only"``).
    A property on 3 of 3 known sources scores 1.0; on 1 of 3, 0.33.
    Never raises.
    """
    try:
        items = list(group or [])
        sources = sorted({str(l.source).strip() for l in items
                          if str(l.source or "").strip()})
        n = len(sources)
        total = int(total_sources or 0)
        if total > 0:
            score = round(n / total, 4)
            basis = "known-universe"
        else:
            score = 1.0 if n else 0.0
            basis = "group-only"
        return {"sources": n, "source_names": sources,
                "listings": len(items), "total_sources": total,
                "score": score, "basis": basis,
                "verdict": ("cross-listed — same property on "
                            f"{n} source(s)" if n > 1 else
                            "single-source listing" if n == 1 else
                            "no listings")}
    except Exception:  # noqa: BLE001
        return {"sources": 0, "source_names": [], "listings": 0,
                "total_sources": 0, "score": 0.0, "basis": "error",
                "verdict": "could not score"}


# ── property passport: the continuity client ─────────────────────

@dataclass
class PassportEvent:
    """One continuity event: listing seen, price changed, relisted…"""

    kind: str = ""  # listed | price_change | relisted | delisted | note
    detail: str = ""
    timestamp: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "timestamp": self.timestamp, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PassportEvent":
        data = data or {}
        try:
            ts = float(data.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        return cls(kind=str(data.get("kind") or ""),
                   detail=str(data.get("detail") or ""),
                   timestamp=ts, source=str(data.get("source") or ""))


@dataclass
class DocRef:
    """A document *reference* (vault/URL id + label) — never the
    document itself."""

    ref_id: str = ""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ref_id": self.ref_id, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocRef":
        data = data or {}
        return cls(ref_id=str(data.get("ref_id") or ""),
                   label=str(data.get("label") or ""))


@dataclass
class Viewing:
    """One viewing appointment / outcome."""

    when: float = 0.0
    outcome: str = ""  # scheduled | done | cancelled | no-show
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"when": self.when, "outcome": self.outcome,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Viewing":
        data = data or {}
        try:
            when = float(data.get("when") or 0.0)
        except (TypeError, ValueError):
            when = 0.0
        return cls(when=when, outcome=str(data.get("outcome") or ""),
                   note=str(data.get("note") or ""))


@dataclass
class PropertyPassport:
    """Continuity record for one property. Versioned: every mutation
    bumps ``version`` and refreshes ``updated_at``, so a consumer can
    tell whether the passport it holds is the latest."""

    passport_id: str = ""
    address: str = ""
    history: list[PassportEvent] = field(default_factory=list)
    documents: list[DocRef] = field(default_factory=list)
    viewings: list[Viewing] = field(default_factory=list)
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"passport_id": self.passport_id, "address": self.address,
                "history": [e.to_dict() for e in self.history],
                "documents": [d.to_dict() for d in self.documents],
                "viewings": [v.to_dict() for v in self.viewings],
                "version": self.version, "created_at": self.created_at,
                "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropertyPassport":
        data = data or {}
        try:
            version = int(data.get("version") or 1)
        except (TypeError, ValueError):
            version = 1
        return cls(
            passport_id=str(data.get("passport_id") or ""),
            address=str(data.get("address") or ""),
            history=[PassportEvent.from_dict(e) for e in
                     (data.get("history") or []) if isinstance(e, dict)],
            documents=[DocRef.from_dict(d) for d in
                       (data.get("documents") or []) if isinstance(d, dict)],
            viewings=[Viewing.from_dict(v) for v in
                      (data.get("viewings") or []) if isinstance(v, dict)],
            version=max(1, version),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0))


class PropertyPassportStore:
    """JSON-persisted, versioned property passports. Never raises."""

    def __init__(self, path: str = "") -> None:
        self._path = path or os.path.join(
            os.path.expanduser("~"), ".nomorals", "property",
            "cross_listing_passports.json")
        self._passports: dict[str, PropertyPassport] = {}
        self._load()

    # ── persistence ──

    def _load(self) -> None:
        try:
            if not os.path.isfile(self._path):
                return
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for pid, raw in data.items():
                    if isinstance(raw, dict):
                        self._passports[str(pid)] = \
                            PropertyPassport.from_dict(raw)
        except (OSError, ValueError):
            _log.debug("cross_listing passport load failed", exc_info=True)

    def _save(self) -> bool:
        try:
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump({pid: p.to_dict()
                           for pid, p in self._passports.items()},
                          fh, sort_keys=True)
            return True
        except OSError:
            _log.debug("cross_listing passport save failed", exc_info=True)
            return False

    # ── API ──

    def create(self, address: str) -> PropertyPassport | None:
        """Create a passport for an address (dedupes on normalized
        address — the same property gets ONE passport)."""
        try:
            address = (address or "").strip()
            if not address:
                return None
            existing = self.by_address(address)
            if existing is not None:
                return existing
            now = time.time()
            passport = PropertyPassport(
                passport_id="ppx_" + uuid.uuid4().hex[:10],
                address=address, version=1,
                created_at=now, updated_at=now)
            self._passports[passport.passport_id] = passport
            self._save()
            return passport
        except Exception:  # noqa: BLE001
            return None

    def get(self, passport_id: str) -> PropertyPassport | None:
        return self._passports.get(str(passport_id or ""))

    def by_address(self, address: str) -> PropertyPassport | None:
        """Find by normalized address — one property, one passport."""
        key = normalize_address(address)
        if not key:
            return None
        for passport in self._passports.values():
            if normalize_address(passport.address) == key:
                return passport
        return None

    def list(self) -> list[PropertyPassport]:
        return sorted(self._passports.values(),
                      key=lambda p: p.updated_at, reverse=True)

    def _bump(self, passport: PropertyPassport) -> None:
        passport.version += 1
        passport.updated_at = time.time()
        self._save()

    def add_event(self, passport_id: str, kind: str, detail: str = "",
                  *, source: str = "") -> bool:
        """Append a history event; bumps the version. Never raises."""
        try:
            passport = self.get(passport_id)
            if passport is None or not (kind or "").strip():
                return False
            passport.history.append(PassportEvent(
                kind=kind.strip()[:60], detail=(detail or "").strip()[:500],
                timestamp=time.time(), source=(source or "").strip()[:120]))
            self._bump(passport)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_document(self, passport_id: str, ref_id: str,
                     label: str = "") -> bool:
        """Add a document *reference*; bumps the version."""
        try:
            passport = self.get(passport_id)
            if passport is None or not (ref_id or "").strip():
                return False
            passport.documents.append(DocRef(
                ref_id=ref_id.strip()[:200], label=(label or "").strip()[:200]))
            self._bump(passport)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_viewing(self, passport_id: str, when: float,
                    outcome: str = "scheduled", note: str = "") -> bool:
        """Add a viewing; bumps the version."""
        try:
            passport = self.get(passport_id)
            if passport is None:
                return False
            passport.viewings.append(Viewing(
                when=float(when or 0.0), outcome=(outcome or "").strip()[:60],
                note=(note or "").strip()[:500]))
            self._bump(passport)
            return True
        except Exception:  # noqa: BLE001
            return False
