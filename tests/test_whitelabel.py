"""Offline tests for the white-label travel clients (#73)."""

import tempfile

import pytest

from nomorals.travel.whitelabel import (
    TravelClientStore, client_knowledge, add_knowledge, answer,
    client_sender, client_spend_report, viki_template, NG_AIRLINES,
)


@pytest.fixture()
def store(tmp_path):
    return TravelClientStore(db_path=str(tmp_path / "clients.db"))


def test_create_and_get(store):
    c = store.create("Lagos Tourism", whatsapp_number="+2348000000001")
    assert c.id.startswith("tcl_")
    got = store.get(c.id)
    assert got is not None and got.name == "Lagos Tourism"
    assert got.whatsapp_number == "+2348000000001"


def test_list(store):
    store.create("A")
    store.create("B")
    names = [c.name for c in store.list()]
    assert "A" in names and "B" in names


def test_deactivate(store):
    c = store.create("A")
    assert store.deactivate(c.id)
    assert c.id not in [x.id for x in store.list()]


def test_knowledge_isolation(tmp_path):
    # client A and B get separate index files — no cross-visibility
    import os
    pa = str(tmp_path / "ka.db")
    pb = str(tmp_path / "kb.db")
    # use the real per-client path function via monkeypatched dir
    import nomorals.travel.whitelabel as wl
    orig = wl._KNOWLEDGE_DIR
    wl._KNOWLEDGE_DIR = str(tmp_path)
    try:
        assert add_knowledge("clientA", "Lagos Guide", "Lagos has great beaches and suya spots.")
        assert add_knowledge("clientB", "Abuja Guide", "Abuja has Aso Rock and Millennium Park.")
        res_a = answer("clientA", "beaches")
        res_b = answer("clientB", "beaches")
        assert res_a["grounded"]
        assert not res_b["grounded"], "client B must not see client A's docs"
    finally:
        wl._KNOWLEDGE_DIR = orig


def test_answer_no_corpus(tmp_path):
    import nomorals.travel.whitelabel as wl
    orig = wl._KNOWLEDGE_DIR
    wl._KNOWLEDGE_DIR = str(tmp_path)
    try:
        res = answer("nobody", "flights")
        assert res["hits"] == [] and not res["grounded"]
    finally:
        wl._KNOWLEDGE_DIR = orig


def test_client_sender_scoped(tmp_path):
    sent = []
    aware = client_sender("clientX", sender=lambda p, t: sent.append((p, t)) or True,
                          db_path=str(tmp_path / "cost.db"))
    assert aware("2341", "hello")
    assert aware.client == "clientX"
    assert len(sent) == 1


def test_client_spend_report(tmp_path):
    report = client_spend_report("clientX", db_path=str(tmp_path / "cost.db"))
    assert "clientX" in report or "no spend" in report.lower() or "₦" in report


def test_viki_template_structure():
    tpl = viki_template("ValueJet")
    assert "ValueJet" in tpl["name"]
    flow_ids = [f["id"] for f in tpl["flows"]]
    assert flow_ids == ["search", "book", "checkin", "status"]
    assert tpl["context"]["currency"] == "NGN (₦)"
    assert "Air Peace" in tpl["context"]["airlines"]
    assert "Pidgin" in tpl["context"]["greeting"] or "Wetin" in tpl["context"]["greeting"]


def test_viki_default_airline():
    tpl = viki_template()
    assert "ValueJet" in tpl["name"]


def test_ng_airlines():
    assert len(NG_AIRLINES) >= 5
    assert "Ibom Air" in NG_AIRLINES


def test_budget_key():
    from nomorals.travel.whitelabel import TravelClient
    c = TravelClient(id="tcl_1", name="X")
    assert c.budget_key == "tcl_1"
    c2 = TravelClient(id="tcl_2", name="Y", cost_client="y-corp")
    assert c2.budget_key == "y-corp"


def test_never_raises_on_bad_db(tmp_path):
    s = TravelClientStore(db_path="/nonexistent-dir-xyz/clients.db")
    assert s.list() == []
    assert s.get("x") is None
