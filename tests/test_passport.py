"""Tests for build-map #94: rental passport + true-cost calculator.

All offline. Vault isolation is asserted: the passport DB must never
contain a raw secret.
"""

import os
import re
import sqlite3
import tempfile

from nomorals.property.passport import (
    INCOME_BANDS,
    CostBreakdown,
    PassportStore,
    affordability,
    can_afford,
    control_passport,
    control_truecost,
    true_cost,
)


def _tmpdb():
    return os.path.join(tempfile.mkdtemp(), "passport.db")


# ── passport generation ───────────────────────────────────────────────

def test_generate_basic():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze", income_band="250k-500k",
                   kyc_summary="NIN verified, employed")
    assert p is not None
    assert p.owner_name == "Adaeze"
    assert p.income_band == "250k-500k"
    assert p.kyc_summary == "NIN verified, employed"


def test_generate_from_memory():
    s = PassportStore(db_path=_tmpdb())
    mem = {"owner_name": "Tunde", "income_band": "500k-1m",
           "rental_history": [{"address": "12 Allen Ave, Ikeja",
                               "landlord": "Mr Bello",
                               "years": "2022-2024"}]}
    p = s.generate(memory=mem)
    assert p is not None
    assert p.owner_name == "Tunde"
    assert p.income_band == "500k-1m"
    assert len(p.history) == 1
    assert p.history[0].landlord == "Mr Bello"


def test_bad_income_band_rejected():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("X", income_band="billionaire")
    assert p is not None
    assert p.income_band == ""


def test_set_income_band():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("X")
    assert s.set_income_band(p.passport_id, "1m-2m")
    assert s.get(p.passport_id).income_band == "1m-2m"
    assert not s.set_income_band(p.passport_id, "nope")


def test_history_and_references():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("X")
    assert s.add_history(p.passport_id, "4 Ozumba Mbadiwe, VI",
                         "Mrs Ade", "2020-2023")
    assert s.add_reference(p.passport_id, "Mrs Ade", "former landlord",
                           "contact:mrs_ade_01")
    got = s.get(p.passport_id)
    assert len(got.history) == 1
    assert got.references[0].contact_ref == "contact:mrs_ade_01"


# ── vault isolation ───────────────────────────────────────────────────

def test_doc_ref_stores_reference_only():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("X")
    assert s.add_doc_ref(p.passport_id, "vault:doc_nin_2024", "National ID")
    got = s.get(p.passport_id)
    assert got.doc_refs[0].vault_id == "vault:doc_nin_2024"
    assert got.doc_refs[0].label == "National ID"


def test_no_secrets_in_db():
    path = _tmpdb()
    s = PassportStore(db_path=path)
    p = s.generate("X", income_band="500k-1m", kyc_summary="verified")
    s.add_doc_ref(p.passport_id, "vault:doc_passport_9f2", "Intl passport")
    s.add_reference(p.passport_id, "Emeka", "employer", "contact:emeka_hr")
    raw = open(path, "rb").read().decode("utf-8", "ignore")
    # vault reference IDs are fine; raw secrets are not
    assert "vault:doc_passport_9f2" in raw
    for pattern in (r"sk_live", r"sk-[A-Za-z0-9]{8,}", r"password\s*[:=]",
                    r"\b\d{11}\b"):  # no raw 11-digit phone numbers either
        assert not re.search(pattern, raw, re.IGNORECASE), pattern


def test_export_has_band_not_exact():
    s = PassportStore(db_path=_tmpdb())
    p = s.generate("Adaeze", income_band="500k-1m")
    out = s.export_text(p.passport_id)
    assert "500k-1m" in out or "₦500,000" in out
    assert "attested" in out


# ── true cost ─────────────────────────────────────────────────────────

def test_true_cost_explicit_fees():
    bd = true_cost(
        "3br in Lekki ₦4,500,000/yr. Agency fee: ₦450,000. "
        "Legal: 10%. Service charge ₦300,000.")
    assert bd.headline_kobo == 450_000_000
    assert bd.agency_kobo == 45_000_000
    assert bd.legal_kobo == 45_000_000
    assert bd.service_kobo == 30_000_000
    assert bd.assumptions == []  # all stated, no norms needed


def test_true_cost_lagos_norms():
    bd = true_cost("2br in Yaba ₦2,200,000 per annum. Call 0803...")
    assert bd.headline_kobo == 220_000_000
    assert bd.agency_kobo == 22_000_000   # 10% norm
    assert bd.legal_kobo == 22_000_000    # 10% norm
    assert len(bd.assumptions) == 2
    assert bd.total_kobo == 264_000_000


def test_true_cost_percent_before_keyword():
    bd = true_cost("1br Ikeja ₦1,800,000. 10% agency fee applies.")
    assert bd.agency_kobo == 18_000_000


def test_true_cost_inspection_and_caution():
    bd = true_cost("Room in Ogba ₦350,000/yr. Inspection fee ₦20,000. "
                   "Caution deposit ₦50,000 (refundable).")
    assert bd.inspection_kobo == 2_000_000
    assert bd.caution_kobo == 5_000_000
    assert "refundable" in bd.format()


def test_true_cost_no_rent():
    bd = true_cost("nice flat, call me")
    assert bd.headline_kobo == 0
    assert bd.total_kobo == 0


def test_true_cost_format():
    bd = true_cost("2br Yaba ₦2,200,000/yr")
    out = bd.format()
    assert "₦2,200,000/yr" in out
    assert "Real move-in: ₦2,640,000" in out


# ── affordability ─────────────────────────────────────────────────────

def test_can_afford_true():
    # ₦1.2m/yr = ₦100k/mo; band 500k-1m mid = 750k; 100/750 = 13% ≤ 33%
    assert can_afford(120_000_000, "500k-1m")


def test_can_afford_false():
    # ₦4.5m/yr = ₦375k/mo; band 250k-500k mid = 375k; 100% > 33%
    assert not can_afford(450_000_000, "250k-500k")


def test_affordability_detail():
    a = affordability(120_000_000, "500k-1m")
    assert a["ok"]
    assert a["monthly_rent"] == 100_000
    assert a["max_affordable_annual"] > 0
    assert "33%" in a["verdict"]


def test_affordability_bad_band():
    a = affordability(120_000_000, "nope")
    assert not a["ok"]
    assert not can_afford(120_000_000, "nope")


# ── chat ──────────────────────────────────────────────────────────────

import pytest


@pytest.fixture(autouse=True)
def _home_tmp(monkeypatch, tmp_path):
    """Chat controls use the default DB path — keep them off the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))


def test_chat_passport_generate_show():
    out = control_passport("generate Adaeze")
    assert "passport ready" in out
    out2 = control_passport("show")
    assert "Rental passport" in out2


def test_chat_passport_income():
    control_passport("generate X")
    out = control_passport("income 250k-500k")
    assert "income band set" in out
    out = control_passport("income nope")
    assert "unknown band" in out


def test_chat_passport_subcommands():
    control_passport("generate X")
    assert "rental history added" in control_passport(
        "history add 12 Allen, Ikeja | Mr Bello | 2022-2024")
    assert "reference added" in control_passport(
        "ref add Mr Bello | former landlord | contact:bello_1")
    out = control_passport("doc add vault:doc_nin_1 | National ID")
    assert "vault stays closed" in out
    assert "National ID" in control_passport("export")


def test_chat_truecost():
    out = control_truecost("2br Yaba ₦2,200,000/yr")
    assert "Real move-in" in out


def test_chat_truecost_afford():
    control_passport("generate X")
    control_passport("income 1m-2m")
    out = control_truecost("afford 2br Yaba ₦2,200,000/yr")
    assert "affordability" in out


def test_chat_garbage_never_raises():
    assert isinstance(control_passport(""), str)
    assert isinstance(control_passport("blargh " * 50), str)
    assert isinstance(control_truecost(""), str)
    assert isinstance(true_cost(None), CostBreakdown)
    assert can_afford(None, None) is False
