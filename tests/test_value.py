"""Offline tests for build-map #95: DIY AVM with confidence interval + comp-picker."""

import os
import tempfile

import pytest

from nomorals.property.value import (
    DISCLAIMER,
    Comp,
    ValueEstimate,
    ValueStore,
    control_value,
    control_valuepick,
    estimate_value,
    pick_comps,
    seed_comp_source,
)


def _tmp_db():
    return os.path.join(tempfile.mkdtemp(), "v.db")


def _comps(prices, sims=None, area="surulere", bedrooms="2br"):
    sims = sims or [0.9] * len(prices)
    return [Comp(price_kobo=p * 100, area=area, bedrooms=bedrooms,
                 title=f"comp {i}", similarity=s)
            for i, (p, s) in enumerate(zip(prices, sims))]


# ── range estimation ──────────────────────────────────────────────

def test_range_not_point():
    est = estimate_value("2br flat in Surulere", store=ValueStore(_tmp_db()))
    assert est.low_kobo < est.point_kobo < est.high_kobo
    assert len(est.comps) == 6


def test_sururere_2br_near_norm():
    est = estimate_value("2br Surulere", store=ValueStore(_tmp_db()))
    # Norm is ₦2.1m annual → kobo midpoint should be in that band.
    assert 1_500_000_00 <= est.point_kobo <= 3_000_000_00


def test_custom_comp_source():
    src = lambda a, b: _comps([2_000_000, 2_100_000, 2_200_000])
    est = estimate_value("2br Surulere", comp_source=src,
                         store=ValueStore(_tmp_db()))
    assert 1_900_000_00 <= est.point_kobo <= 2_300_000_00
    assert est.fsd < 0.10  # tight comps → tight band


def test_fsd_widens_with_spread():
    tight = estimate_value("2br Surulere",
                           comp_source=lambda a, b: _comps([2_000_000] * 4),
                           store=ValueStore(_tmp_db()))
    wide = estimate_value("2br Surulere",
                          comp_source=lambda a, b: _comps([1_000_000, 3_000_000]),
                          store=ValueStore(_tmp_db()))
    assert wide.fsd > tight.fsd


def test_weighted_by_similarity():
    # High-similarity comps dominate the point.
    src = lambda a, b: _comps([1_000_000, 3_000_000], sims=[0.95, 0.05])
    est = estimate_value("2br Surulere", comp_source=src,
                         store=ValueStore(_tmp_db()))
    assert est.point_kobo < 1_500_000_00


# ── thin markets ──────────────────────────────────────────────────

def test_thin_market_wide_band():
    src = lambda a, b: _comps([2_000_000])  # one comp
    est = estimate_value("2br Surulere", comp_source=src,
                         store=ValueStore(_tmp_db()))
    assert est.thin_market
    assert est.fsd >= 0.28
    assert "thin market" in est.format()


def test_no_comps_honest():
    src = lambda a, b: []
    est = estimate_value("2br Surulere", comp_source=src,
                         store=ValueStore(_tmp_db()))
    assert est.point_kobo == 0
    assert est.thin_market


def test_broken_source_never_raises():
    def boom(a, b):
        raise RuntimeError("down")
    est = estimate_value("2br Surulere", comp_source=boom,
                         store=ValueStore(_tmp_db()))
    assert isinstance(est, ValueEstimate)


# ── comp-picker ───────────────────────────────────────────────────

def test_pick_comps_refines():
    db = _tmp_db()
    store = ValueStore(db)
    src = lambda a, b: _comps([1_000_000, 2_000_000, 3_000_000])
    est = estimate_value("2br Surulere", comp_source=src, store=store)
    refined = pick_comps(est.id, ["1"], store=store)
    assert refined is not None
    # Only the ₦1m comp kept → point near ₦1m.
    assert refined.point_kobo < 1_200_000_00
    assert len(refined.kept_ids) == 1


def test_pick_comps_by_number():
    db = _tmp_db()
    store = ValueStore(db)
    est = estimate_value("2br Surulere", store=store)
    refined = pick_comps(est.id, ["1", "2"], store=store)
    assert refined is not None
    assert len(refined.comps) == 2


def test_pick_comps_bad_id_returns_none():
    assert pick_comps("val_nope", ["1"], store=ValueStore(_tmp_db())) is None


def test_pick_comps_empty_selection_returns_none():
    db = _tmp_db()
    store = ValueStore(db)
    est = estimate_value("2br Surulere", store=store)
    assert pick_comps(est.id, [], store=store) is None
    assert pick_comps(est.id, ["99"], store=store) is None


def test_pick_comps_saved_for_later():
    db = _tmp_db()
    store = ValueStore(db)
    est = estimate_value("2br Surulere", store=store)
    refined = pick_comps(est.id, ["1"], store=store)
    assert store.get(refined.id) is not None


# ── parsing ───────────────────────────────────────────────────────

def test_parse_areas():
    for desc, area in [("2br in Lekki", "lekki"),
                       ("3 bedroom Surulere flat", "surulere"),
                       ("studio in VI", "victoria island")]:
        est = estimate_value(desc, store=ValueStore(_tmp_db()))
        assert est.area == area, desc


def test_seed_source_labeled():
    comps = seed_comp_source("yaba", "2br")
    assert len(comps) == 6
    assert all("seeded" in c.source for c in comps)
    assert seed_comp_source("nowhere", "2br") == []


# ── disclaimer ────────────────────────────────────────────────────

def test_disclaimer_on_every_format():
    est = estimate_value("2br Surulere", store=ValueStore(_tmp_db()))
    out = est.format()
    assert "not a valuation" in out
    assert "humans commit" in out


def test_disclaimer_mentions_zillow_lesson():
    assert "380M" in DISCLAIMER


# ── chat ──────────────────────────────────────────────────────────

def test_control_value_empty():
    assert "value" in control_value("").lower()


def test_control_value_renders_range():
    out = control_value("2br Surulere")
    assert "₦" in out
    assert "valuepick" in out


def test_control_valuepick_usage():
    assert "valuepick" in control_valuepick("").lower()


def test_control_valuepick_bad_id():
    out = control_valuepick("val_nope 1")
    assert "couldn't refine" in out


def test_control_never_raises():
    assert control_value(None) is not None
    assert control_valuepick(None) is not None
    assert control_value("??? ###") is not None


def test_full_chat_flow(monkeypatch):
    import nomorals.property.value as vmod
    db = _tmp_db()
    store = ValueStore(db)
    monkeypatch.setattr(vmod, "ValueStore", lambda *a, **k: store)
    est = estimate_value("2br Surulere", store=store)
    out = control_valuepick(f"{est.id} 1,2")
    assert "Refined" in out
    assert "₦" in out
