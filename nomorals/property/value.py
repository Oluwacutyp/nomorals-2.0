"""DIY automated valuation model (AVM) with confidence intervals + comp-picker.

Build-map #95. A value *range*, never a point estimate.

CRITICAL RULE — estimates inform, humans commit. Zillow Offers lost
$380M trusting its algorithms to drive financial commitments. This
module produces information only. It never triggers a money move;
any payment still goes through the biometric + mandate gates (#8/#69).

Pipeline
--------
1. ``estimate_value(desc)`` — parse area + bedroom count, pull
   comparables (injectable ``comp_source``; default = seeded Lagos
   norms, honestly labeled), weight by similarity, and return
   ``(low, point, high, fsd, comps)`` where ``fsd`` is the
   forecast-standard-deviation fraction (HouseCanary pattern).
2. Thin markets (fewer than 3 comps) get an honestly wide band —
   Homes.com's 16% cautionary: public-records-only is not enough.
3. Comp-picker (Redfin trick): the owner vetoes/swaps weak comps,
   ``pick_comps(estimate_id, keep_ids)`` recomputes the range from
   the kept set only.

Never raises. Fully offline-capable via the ``comp_source`` seam.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid

_log = logging.getLogger("nomorals.property.value")

from .scam import AREA_NORMS  # noqa: E402  (same #93 area-norm data)

__all__ = [
    "Comp",
    "ValueEstimate",
    "ValueStore",
    "estimate_value",
    "pick_comps",
    "control_value",
    "control_valuepick",
    "register",
]

# An estimate is information, not a commitment. Never softened.
DISCLAIMER = (
    "ℹ️ Automated estimate for information only — not a valuation. "
    "Zillow Offers lost $380M letting an algorithm commit money. "
    "Estimates inform; humans commit. Any payment still needs your "
    "approval through the money gates."
)

_MIN_COMPS_FOR_TIGHT = 3
_THIN_MARKET_FSD_FLOOR = 0.28
_CONFIDENCE_MULT = 1.6  # low/high = point ± mult * fsd * point


# ── data types ──────────────────────────────────────────────────────

class Comp:
    """One comparable listing."""

    def __init__(self, comp_id: str = "", price_kobo: int = 0,
                 area: str = "", bedrooms: str = "",
                 title: str = "", source: str = "",
                 similarity: float = 0.5) -> None:
        self.id = comp_id or ("comp_" + uuid.uuid4().hex[:8])
        self.price_kobo = max(0, int(price_kobo or 0))
        self.area = (area or "").strip()
        self.bedrooms = (bedrooms or "").strip()
        self.title = (title or "").strip()[:160]
        self.source = (source or "").strip()[:80]
        try:
            self.similarity = max(0.0, min(1.0, float(similarity)))
        except (TypeError, ValueError):
            self.similarity = 0.5

    def to_dict(self) -> dict:
        return {"id": self.id, "price_kobo": self.price_kobo,
                "area": self.area, "bedrooms": self.bedrooms,
                "title": self.title, "source": self.source,
                "similarity": self.similarity}

    @classmethod
    def from_dict(cls, d: dict) -> "Comp":
        d = d or {}
        return cls(comp_id=str(d.get("id", "")),
                   price_kobo=int(d.get("price_kobo", 0) or 0),
                   area=str(d.get("area", "")),
                   bedrooms=str(d.get("bedrooms", "")),
                   title=str(d.get("title", "")),
                   source=str(d.get("source", "")),
                   similarity=d.get("similarity", 0.5))


class ValueEstimate:
    """A value RANGE with its comps and confidence."""

    def __init__(self, estimate_id: str = "", area: str = "",
                 bedrooms: str = "", point_kobo: int = 0,
                 low_kobo: int = 0, high_kobo: int = 0,
                 fsd: float = 0.0, comps: list[Comp] | None = None,
                 kept_ids: list[str] | None = None,
                 thin_market: bool = False,
                 created_at: float = 0.0) -> None:
        self.id = estimate_id or ("val_" + uuid.uuid4().hex[:8])
        self.area = area
        self.bedrooms = bedrooms
        self.point_kobo = max(0, int(point_kobo or 0))
        self.low_kobo = max(0, int(low_kobo or 0))
        self.high_kobo = max(0, int(high_kobo or 0))
        self.fsd = max(0.0, float(fsd or 0.0))
        self.comps = list(comps or [])
        self.kept_ids = list(kept_ids or [])
        self.thin_market = bool(thin_market)
        self.created_at = created_at or time.time()

    def format(self) -> str:
        """Chat rendering: range + comps + comp-picker nudge + disclaimer."""
        def naira(k: int) -> str:
            return f"₦{k/100:,.0f}" if k else "₦?"
        area_lbl = self.area or "unknown area"
        br = self.bedrooms or "?"
        lines = [
            f"🏠 Value estimate — {br} in {area_lbl}",
            f"   {naira(self.low_kobo)} – {naira(self.high_kobo)}  "
            f"(midpoint {naira(self.point_kobo)}, ±{self.fsd*100:.0f}%)",
        ]
        if self.thin_market:
            lines.append("   ⚠️ thin market — few comparables, band is "
                         "wide on purpose.")
        lines.append(f"   Based on {len(self.comps)} comparable(s):")
        for i, c in enumerate(self.comps[:8], 1):
            kept = " ✓ kept" if c.id in self.kept_ids else ""
            lines.append(f"   {i}. {naira(c.price_kobo)} — {c.title or c.area}"
                         f"{kept}")
        lines.append("   Pick which comps look right: "
                     f"/valuepick {self.id} <1,2,4> "
                     "(numbers above; reruns the range)")
        lines.append(DISCLAIMER)
        return "\n".join(lines)


# ── parsing ─────────────────────────────────────────────────────────

_BEDROOM_RE = re.compile(r"(\d+)\s*(?:br|bedroom|bed)\b", re.I)
_AREA_WORDS = sorted({a for a in AREA_NORMS} |
                     {"lekki phase 1", "lekki phase 2"}, key=len, reverse=True)


def _parse_desc(desc: str) -> tuple[str, str]:
    """(area, bedrooms) from free text. Never raises."""
    try:
        text = (desc or "").lower()
        bedrooms = ""
        m = _BEDROOM_RE.search(text)
        if m:
            bedrooms = f"{m.group(1)}br"
        elif "studio" in text:
            bedrooms = "studio"
        elif re.search(r"\broom\b", text) and "bed" not in text:
            bedrooms = "room"
        area = ""
        for a in _AREA_WORDS:
            if re.search(r"\b" + re.escape(a) + r"\b", text):
                area = a
                break
        if re.search(r"\bvi\b", text):
            area = "victoria island"
        if area == "vi":
            area = "victoria island"
        if area in ("lekki phase 1", "lekki phase 2"):
            area = "lekki"
        return area, bedrooms
    except Exception:  # noqa: BLE001
        return "", ""


# ── comp sources ────────────────────────────────────────────────────

def seed_comp_source(area: str, bedrooms: str) -> list[Comp]:
    """Default comp source: seeded Lagos norms, honestly labeled.

    Returns 6 comps spread ±18% around the area norm so the model has
    something real-shaped to work with when no live source is wired.
    """
    try:
        norm = (AREA_NORMS.get(area, {}) or {}).get(bedrooms or "2br")
        if not norm:
            return []
        spreads = [0.86, 0.92, 0.97, 1.03, 1.09, 1.16]
        comps = []
        for i, s in enumerate(spreads):
            price = int(norm * 100 * s)
            comps.append(Comp(
                price_kobo=price, area=area, bedrooms=bedrooms,
                title=f"{bedrooms} listing #{i+1} — {area.title()}",
                source="seeded Lagos norms (not live listings)",
                similarity=0.9 - 0.05 * abs(i - 2.5),
            ))
        return comps
    except Exception:  # noqa: BLE001
        return []


# ── the model ───────────────────────────────────────────────────────

def _estimate_from_comps(comps: list[Comp]) -> tuple[int, int, int, float, bool]:
    """(point, low, high, fsd) from weighted comps. Pure; never raises."""
    try:
        comps = [c for c in (comps or []) if c.price_kobo > 0]
        if not comps:
            return 0, 0, 0, 0.0, True
        prices = [c.price_kobo for c in comps]
        weights = [max(0.05, c.similarity) for c in comps]
        wsum = sum(weights)
        point = sum(p * w for p, w in zip(prices, weights)) / wsum
        # Weighted standard deviation → fsd fraction.
        var = sum(w * (p - point) ** 2 for p, w in zip(prices, weights)) / wsum
        fsd = math.sqrt(var) / point if point > 0 else 0.0
        thin = len(comps) < _MIN_COMPS_FOR_TIGHT
        if thin:
            fsd = max(fsd, _THIN_MARKET_FSD_FLOOR)
        half = _CONFIDENCE_MULT * fsd * point
        low = max(0, int(round(point - half)))
        high = int(round(point + half))
        return int(round(point)), low, high, fsd, thin
    except Exception:  # noqa: BLE001
        return 0, 0, 0, 0.0, True


def _default_db() -> str:
    try:
        return os.path.join(os.path.expanduser("~"), ".nomorals",
                            "property", "values.db")
    except Exception:  # noqa: BLE001
        return os.path.join(os.path.expanduser("~"), ".nomorals_values.db")


class ValueStore:
    """Persists estimates so comp-picker works later. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS value_estimates (
                       id TEXT PRIMARY KEY, area TEXT, bedrooms TEXT,
                       point_kobo INTEGER, low_kobo INTEGER,
                       high_kobo INTEGER, fsd REAL, comps_json TEXT,
                       kept_ids_json TEXT, thin_market INTEGER,
                       created_at REAL)""")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("value: db unavailable, running memory-only",
                         exc_info=True)
            self._db = None

    def save(self, est: ValueEstimate) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO value_estimates VALUES (?,?,?,?,?,?,"
                "?,?,?,?,?)",
                (est.id, est.area, est.bedrooms, est.point_kobo,
                 est.low_kobo, est.high_kobo, est.fsd,
                 json.dumps([c.to_dict() for c in est.comps]),
                 json.dumps(est.kept_ids), int(est.thin_market),
                 est.created_at))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, estimate_id: str) -> ValueEstimate | None:
        try:
            if self._db is None or not estimate_id:
                return None
            row = self._db.execute(
                "SELECT * FROM value_estimates WHERE id = ?",
                (estimate_id,)).fetchone()
            if not row:
                return None
            comps = [Comp.from_dict(d)
                     for d in json.loads(row["comps_json"] or "[]")]
            return ValueEstimate(
                estimate_id=row["id"], area=row["area"],
                bedrooms=row["bedrooms"], point_kobo=row["point_kobo"],
                low_kobo=row["low_kobo"], high_kobo=row["high_kobo"],
                fsd=row["fsd"], comps=comps,
                kept_ids=json.loads(row["kept_ids_json"] or "[]"),
                thin_market=bool(row["thin_market"]),
                created_at=row["created_at"])
        except Exception:  # noqa: BLE001
            return None


def estimate_value(property_desc: str, *,
                   comp_source=None,
                   store: ValueStore | None = None) -> ValueEstimate:
    """Estimate a value range. Never raises.

    ``comp_source(area, bedrooms) -> [Comp]`` is the injectable seam
    (search + scraped listings). Default: seeded Lagos norms.
    """
    try:
        area, bedrooms = _parse_desc(property_desc)
        source = comp_source or seed_comp_source
        try:
            comps = source(area, bedrooms) or []
        except Exception:  # noqa: BLE001
            _log.debug("value comp source failed", exc_info=True)
            comps = []
        point, low, high, fsd, thin = _estimate_from_comps(comps)
        est = ValueEstimate(area=area, bedrooms=bedrooms,
                            point_kobo=point, low_kobo=low, high_kobo=high,
                            fsd=fsd, comps=comps, thin_market=thin)
        if store is not None:
            try:
                store.save(est)
            except Exception:  # noqa: BLE001
                pass
        return est
    except Exception:  # noqa: BLE001
        _log.warning("estimate_value failed", exc_info=True)
        return ValueEstimate(thin_market=True)


def pick_comps(estimate_id: str, keep_ids: list[str], *,
               store: ValueStore | None = None) -> ValueEstimate | None:
    """Comp-picker: recompute the range from the kept comps only.

    ``keep_ids`` may be comp ids or 1-based numbers as shown in the
    formatted estimate. Returns the refined estimate (saved), or None
    when the estimate or the selection is unusable. Never raises.
    """
    try:
        store = store or ValueStore()
        est = store.get(estimate_id)
        if est is None or not est.comps:
            return None
        wanted = {(k or "").strip().lower() for k in (keep_ids or []) if k}
        if not wanted:
            return None
        kept: list[Comp] = []
        for i, c in enumerate(est.comps, 1):
            if c.id.lower() in wanted or str(i) in wanted:
                kept.append(c)
        if not kept:
            return None
        point, low, high, fsd, thin = _estimate_from_comps(kept)
        refined = ValueEstimate(
            area=est.area, bedrooms=est.bedrooms,
            point_kobo=point, low_kobo=low, high_kobo=high,
            fsd=fsd, comps=kept,
            kept_ids=[c.id for c in kept], thin_market=thin)
        try:
            store.save(refined)
        except Exception:  # noqa: BLE001
            pass
        return refined
    except Exception:  # noqa: BLE001
        _log.warning("pick_comps failed", exc_info=True)
        return None


# ── chat ────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("/value <e.g. '2br flat in Surulere'> — value RANGE with "
            "confidence band + comparables. Then /valuepick <id> <1,2,4> "
            "to refine from the comps you trust. Estimates inform; "
            "humans commit.")


def control_value(tail: str, context=None, chat=None, **kwargs) -> str:
    """Chat entry: /value. Owner + community (not owner-gated)."""
    try:
        tail = (tail or "").strip()
        if not tail:
            return _usage()
        comp_source = (getattr(context, "comp_source", None)
                       if context is not None else None)
        store = ValueStore()
        est = estimate_value(tail, comp_source=comp_source, store=store)
        if not est.comps:
            return ("no comparables for that — tell me the area and "
                    "bedroom count (e.g. /value 2br Surulere). "
                    + DISCLAIMER)
        return est.format()
    except Exception:  # noqa: BLE001
        _log.warning("value failed", exc_info=True)
        return "couldn't estimate that — describe the area and bedrooms."


def control_valuepick(tail: str, context=None, chat=None, **kwargs) -> str:
    """Chat entry: /valuepick <estimate_id> <1,2,4>."""
    try:
        parts = (tail or "").strip().split(None, 1)
        if len(parts) < 2:
            return "/valuepick <estimate id> <comp numbers, e.g. 1,2,4>"
        est_id, nums = parts
        keep = [n.strip() for n in re.split(r"[,\s]+", nums) if n.strip()]
        refined = pick_comps(est_id, keep, store=ValueStore())
        if refined is None:
            return "couldn't refine — check the estimate id and comp numbers."
        return ("🔁 Refined from your picked comps:\n" + refined.format())
    except Exception:  # noqa: BLE001
        _log.warning("valuepick failed", exc_info=True)
        return "couldn't refine — check the estimate id and comp numbers."


def register(registry) -> None:
    """Tool-registry hook."""
    try:
        registry.register("value", control_value,
                          "DIY property valuation — value range with "
                          "confidence band + comp-picker.")
        registry.register("valuepick", control_valuepick,
                          "Refine a value estimate from picked comparables.")
    except Exception:  # noqa: BLE001
        _log.debug("value register failed", exc_info=True)
