"""Sweep tests for nomorals/property (2026-10-10 mined upgrade).

Covers the Fredy-weighted scam signals, OpenRent referencing readiness,
Zillow 30-day attestation, LASRERA upfront-rent warning, HouseCanary
confidence score, comp source-trust, and value trend. All offline.
"""

import os
import tempfile
import time

from nomorals.property.scam import (
    ScamFlag,
    ScamStore,
    check_listing,
)
from nomorals.property.passport import (
    NIGERIA_KYC_DOCS,
    CostBreakdown,
    PassportStore,
    affordability,
    can_afford,
    can_guarantee,
    combined_affordability,
    control_passport,
    control_truecost,
    missing_kyc_docs,
    referencing_readiness,
    required_income_band,
    required_monthly_income,
    true_cost,
)
from nomorals.property.value import (
    Comp,
    ValueStore,
    estimate_value,
    seed_comp_source,
)


def _tmpdb(name="t.db"):
    return os.path.join(tempfile.mkdtemp(), name)


# ── scam: no-viewing / abroad (Fredy) ────────────────────────────────

def test_pay_before_viewing_is_danger():
    r = check_listing("2 bedroom in Lekki ₦4,500,000. "
                      "Pay before viewing to secure it.")
    flags = [f for f in r.flags if f.code == "no_viewing"]
    assert flags and flags[0].severity == "danger"
    assert flags[0].weight == 3


def test_keys_by_courier_is_danger():
    r = check_listing("3 bedroom in Ajah ₦3,000,000. "
                      "Keys will be sent by courier.")
    assert any(f.code == "no_viewing" and f.severity == "danger"
               for f in r.flags)


def test_landlord_abroad_is_warn_not_danger():
    r = check_listing("2 bedroom in Yaba ₦2,200,000. "
                      "I am abroad so I can't show the flat myself.")
    flags = [f for f in r.flags if f.code == "no_viewing"]
    assert flags and flags[0].severity == "warn"


# ── scam: irreversible payment routes (FTC) ──────────────────────────

def test_crypto_payment_is_danger():
    r = check_listing("2 bedroom in Lekki ₦4,500,000. "
                      "Payment via USDT or Bitcoin accepted.")
    flags = [f for f in r.flags if f.code == "payment_channel"]
    assert flags and flags[0].severity == "danger"
    assert "irreversible" in flags[0].message


def test_gift_card_plus_urgency_is_hard_stop():
    r = check_listing("Room in Surulere ₦500,000. Pay by gift card now, "
                      "urgent, today only!")
    flags = [f for f in r.flags if f.code == "payment_channel"]
    assert flags and flags[0].severity == "hard_stop"
    assert r.score == 0


def test_western_union_flagged():
    r = check_listing("1 bedroom in Ogba ₦900,000. "
                      "Send money via Western Union.")
    assert any(f.code == "payment_channel" for f in r.flags)


# ── scam: weights + false-positive discipline ────────────────────────

def test_flags_carry_signal_weights():
    f = ScamFlag("illegal_fee", "danger", "x")
    assert f.weight == 3
    f2 = ScamFlag("price_anomaly", "info", "y")
    assert f2.weight == 2


def test_refundable_deposit_is_not_an_illegal_fee():
    r = check_listing("Room in Ogba ₦350,000/yr. Caution deposit ₦50,000 "
                      "(refundable). No 45 Herbert Macaulay Way, Yaba.")
    assert not any(f.code == "illegal_fee" for f in r.flags)


def test_report_has_safety_checklist():
    r = check_listing("2 bedroom flat, 45 Herbert Macaulay Way, Yaba. "
                      "₦2,200,000.")
    out = r.format()
    assert "Before paying anything" in out
    assert "Reverse-image-search" in out


# ── passport: attestation freshness (Zillow 30-day) ──────────────────

def test_new_passport_is_fresh():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze", income_band="500k-1m")
    assert p.attested_at > 0
    assert p.is_fresh()
    assert "fresh" in p.freshness_note()


def test_stale_passport_detected():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze")
    p.attested_at = time.time() - 40 * 86400
    assert not p.is_fresh()
    assert "STALE" in p.freshness_note()


def test_reattest_renews():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze")
    p2 = s.get(p.passport_id)
    p2.attested_at = time.time() - 40 * 86400
    # write the stale stamp through the db, then renew
    s._db.execute("UPDATE passports SET attested_at = ? WHERE id = ?",
                  (p2.attested_at, p.passport_id))
    s._db.commit()
    assert s.reattest(p.passport_id)
    assert s.get(p.passport_id).is_fresh()


def test_export_shows_freshness():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze")
    out = s.export_text(p.passport_id)
    assert "Status:" in out
    assert "attest" in out.lower()


# ── passport: KYC checklist (Kwaba) ──────────────────────────────────

def test_kyc_checklist_matches_labels():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze")
    s.add_doc_ref(p.passport_id, "vault:doc_nin", "NIN slip")
    p2 = s.get(p.passport_id)
    missing = missing_kyc_docs(p2)
    assert "NIN slip / national ID" not in missing
    assert "recent utility bill" in missing
    assert len(NIGERIA_KYC_DOCS) == 5


# ── passport: referencing readiness (OpenRent) ───────────────────────

def test_readiness_empty_passport():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze")
    r = referencing_readiness(s.get(p.passport_id))
    assert r["score"] < 60
    assert "affordability" in r["missing"]
    assert not r["ok"]


def test_readiness_full_passport():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze", income_band="500k-1m",
                   kyc_summary="employed, verified")
    s.add_history(p.passport_id, "12 Allen Ave, Ikeja", "Mr B",
                  "2022–2024")
    s.add_reference(p.passport_id, "Mr B", "former landlord",
                    "contact:ref_1")
    for label in ("NIN slip", "utility bill", "employment letter",
                  "bank statement", "passport photo"):
        s.add_doc_ref(p.passport_id, "vault:" + label.replace(" ", "_"),
                      label)
    r = referencing_readiness(s.get(p.passport_id))
    assert r["score"] == 100
    assert r["ok"]


def test_readiness_never_raises():
    r = referencing_readiness(None)
    assert r["score"] == 0


# ── passport: affordability upgrades ─────────────────────────────────

def test_required_monthly_income():
    # ₦2.2m/yr → ₦183,333/mo; at 33% bar needs ₦555,556/mo
    assert required_monthly_income(220_000_000) == 555_556


def test_required_income_band():
    assert required_income_band(220_000_000) == "500k-1m"
    assert required_income_band(0) == ""


def test_can_guarantee_three_x_rule():
    # 2.2m rent needs 6.6m/yr income. 500k-1m mid 750k/mo = 9m/yr → yes
    assert can_guarantee(220_000_000, "500k-1m")
    # 250k-500k mid 375k/mo = 4.5m/yr → no
    assert not can_guarantee(220_000_000, "250k-500k")
    # the 3× guarantor bar is strictly harder than the 33% tenant bar,
    # so a passing guarantor always passes the tenant bar too
    assert can_guarantee(120_000_000, "250k-500k")
    assert can_afford(120_000_000, "250k-500k")


def test_combined_affordability():
    r = combined_affordability(220_000_000, ["2m-5m", "2m-5m"])
    assert r["ok"] and r["tenants"] == 2
    r2 = combined_affordability(220_000_000, ["under-100k"])
    assert not r2["ok"]
    r3 = combined_affordability(220_000_000, ["nope"])
    assert not r3["ok"]


# ── true cost: upfront warning + monthly ─────────────────────────────

def test_multi_year_upfront_warns():
    bd = true_cost("2br Yaba ₦2,200,000/yr. Two years rent required "
                   "upfront.")
    assert bd.warnings
    assert "LASRERA" in bd.warnings[0]
    assert "⚠️" in bd.format()


def test_single_year_no_upfront_warning():
    bd = true_cost("2br Yaba ₦2,200,000/yr")
    assert bd.warnings == []


def test_monthly_amortized():
    bd = true_cost("2br Yaba ₦2,200,000/yr")  # 2.64m total
    assert bd.total_kobo == 264_000_000
    assert bd.monthly_kobo == 22_000_000
    assert "₦220,000/mo" in bd.format()


# ── chat: new passport commands ──────────────────────────────────────

def _home_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


def test_chat_readiness(monkeypatch, tmp_path):
    _home_tmp(monkeypatch, tmp_path)
    control_passport("generate Adaeze")
    out = control_passport("readiness")
    assert "Referencing readiness" in out
    assert "%" in out


def test_chat_reattest(monkeypatch, tmp_path):
    _home_tmp(monkeypatch, tmp_path)
    control_passport("generate Adaeze")
    assert "30 days" in control_passport("reattest")


def test_chat_truecost_afford_uses_monthly(monkeypatch, tmp_path):
    _home_tmp(monkeypatch, tmp_path)
    control_passport("generate Adaeze")
    control_passport("income 500k-1m")
    out = control_truecost("afford 2br Yaba ₦2,200,000/yr")
    assert "Real move-in" in out
    assert "/mo" in out


# ── value: confidence (HouseCanary) ──────────────────────────────────

def test_confidence_is_one_minus_fsd():
    est = estimate_value("2br Surulere", store=ValueStore(_tmpdb()))
    assert est.confidence == round((1 - est.fsd) * 100)
    assert 0 <= est.confidence <= 100
    assert f"confidence {est.confidence}%" in est.format()


def test_seed_comps_marked_low_trust():
    comps = seed_comp_source("yaba", "2br")
    assert comps and all(c.source_trust == 0.6 for c in comps)
    assert "[seeded]" in estimate_value(
        "2br Yaba", store=ValueStore(_tmpdb())).format()


def test_source_trust_weights_mixed_comps():
    store = ValueStore(_tmpdb())
    comps = [
        Comp(price_kobo=100_000_000, similarity=0.9, source_trust=0.6,
             title="seeded cheap"),
        Comp(price_kobo=300_000_000, similarity=0.9, source_trust=1.0,
             title="live expensive"),
    ]
    est = estimate_value("2br Surulere",
                         comp_source=lambda a, b: comps, store=store)
    # high-trust comp (weight 0.9) beats low-trust (weight 0.54) →
    # point above the simple midpoint of 2.0m
    assert est.point_kobo > 200_000_000


def test_comp_trust_roundtrip():
    c = Comp(price_kobo=1, source_trust=0.6)
    c2 = Comp.from_dict(c.to_dict())
    assert c2.source_trust == 0.6


# ── value: history + trend ───────────────────────────────────────────

def test_history_and_trend():
    store = ValueStore(_tmpdb())
    e1 = estimate_value("2br Surulere", store=store)
    time.sleep(0.02)
    e2 = estimate_value("2br Surulere",
                        comp_source=lambda a, b: [
                            Comp(price_kobo=c.price_kobo * 2,
                                 similarity=c.similarity,
                                 source_trust=c.source_trust)
                            for c in seed_comp_source("surulere", "2br")],
                        store=store)
    hist = store.history("surulere", "2br")
    assert len(hist) >= 2
    t = store.trend("surulere", "2br")
    assert t["ok"] and t["direction"] == "up" and t["pct"] > 0


def test_trend_needs_two_estimates():
    store = ValueStore(_tmpdb())
    estimate_value("2br Surulere", store=store)
    t = store.trend("surulere", "2br")
    assert not t["ok"]


def test_trend_stable():
    store = ValueStore(_tmpdb())
    estimate_value("2br Surulere", store=store)
    time.sleep(0.02)
    estimate_value("2br Surulere", store=store)
    t = store.trend("surulere", "2br")
    assert t["ok"] and t["direction"] == "stable"


# ── never raises ─────────────────────────────────────────────────────

def test_sweep_never_raises():
    assert isinstance(check_listing(None), type(check_listing("x")))
    assert true_cost(None).total_kobo == 0
    assert required_monthly_income(-5) == 0
    assert required_income_band("junk") == ""
    assert not can_guarantee(0, "junk")
    assert combined_affordability(0, [])["ok"] is False
    assert referencing_readiness(None)["score"] == 0
    s = ValueStore(_tmpdb())
    assert s.trend("", "")["ok"] is False
    assert s.history("", "") == []
