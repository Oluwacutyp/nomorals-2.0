"""Tests for build-map #80: regulatory change monitoring (CBN/SEC/NDPA).

All offline: a mock fetcher stands in for the live RSS/web seam.
"""

import tempfile

from nomorals.legal.regulatory import (
    DISCLAIMER,
    REGULATORS,
    OBLIGATION_CONTROLS,
    RegulatoryItem,
    RegulatoryWatch,
    alert_text,
    check_all,
    control_regwatch,
    controls_for,
    ensure_schedule,
    item_from_dict,
)


def _store():
    return RegulatoryWatch(db_path=tempfile.mktemp(suffix=".db"))


def _item(**kw):
    base = dict(regulator="CBN", title="Revised cash-related policies",
                summary="Banks must report cash transactions above limits.",
                source_url="https://www.cbn.gov.ng", published=1.0,
                topics=["cash policy"])
    base.update(kw)
    return RegulatoryItem(id="i1", **{k: v for k, v in base.items()
                                      if k != "id"})


def test_regulators_official_domains():
    assert set(REGULATORS) == {"CBN", "SEC", "NDPA", "FIRS", "CAC", "NCC", "SON"}
    assert REGULATORS["CBN"].site.startswith("https://")
    assert "ndpc" in REGULATORS["NDPA"].site.lower()


def test_watch_creation():
    s = _store()
    w = s.watch("CBN", ["fintech", "AML"])
    assert w is not None and w.regulator == "CBN"
    assert w.topics == ["fintech", "aml"]


def test_watch_bad_regulator_returns_none():
    s = _store()
    assert s.watch("NASA") is None


def test_watch_ndpc_alias():
    s = _store()
    w = s.watch("NDPC", ["data protection"])
    assert w is not None and w.regulator == "NDPA"


def test_unwatch():
    s = _store()
    w = s.watch("SEC")
    assert s.unwatch(w.id) is True
    assert s.unwatch(w.id) is False


def test_new_item_detection_and_dedupe():
    s = _store()
    s.watch("CBN")
    fetcher = lambda r: [{"regulator": "CBN", "id": "x1",
                          "title": "New AML baseline standards",
                          "summary": "Automated AML solution baseline.",
                          "topics": ["AML"]}]
    first = s.check(fetcher=fetcher)
    assert len(first) == 1 and first[0].id == "x1"
    second = s.check(fetcher=fetcher)
    assert second == []  # dedupe: no double alerts


def test_relevance_filter_topic_match():
    s = _store()
    s.watch("CBN", ["fintech"])
    fetcher = lambda r: [
        {"regulator": "CBN", "id": "a", "title": "Fintech licensing update",
         "topics": ["fintech"]},
        {"regulator": "CBN", "id": "b", "title": "BVN enrolment for OFI",
         "topics": ["BVN"]},
    ]
    got = s.check(fetcher=fetcher)
    assert [i.id for i in got] == ["a"]


def test_relevance_profile_rescue():
    s = _store()
    s.watch("CBN", ["fintech"])
    profile = {"business_type": "payment service provider", "keywords": [],
               "sectors": []}
    fetcher = lambda r: [{"regulator": "CBN", "id": "c",
                          "title": "Payment service provider settlement rules",
                          "topics": []}]
    got = s.check(fetcher=fetcher, profile=profile)
    assert [i.id for i in got] == ["c"]


def test_no_fetcher_honest_empty():
    s = _store()
    s.watch("CBN")
    assert s.check() == []  # no silent behavior, no crash


def test_fetcher_failure_never_raises():
    s = _store()
    s.watch("CBN")
    def boom(r):
        raise RuntimeError("down")
    assert s.check(fetcher=boom) == []


def test_urgency_detection():
    urgent = _item(title="Immediate prohibition on unlicensed PSPs",
                   summary="All unlicensed operators must cease immediately.")
    assert urgent.urgency == "urgent"
    normal = _item()
    assert normal.urgency == "normal"


def test_urgent_sorts_first():
    items = [RegulatoryItem(id="n", regulator="FIRS", title="Filing calendar",
                            published=9.0),
             RegulatoryItem(id="u", regulator="CBN",
                            title="Immediate suspension of PSP licence",
                            published=1.0)]
    s = _store()
    s.watch("CBN"); s.watch("FIRS")
    fetcher = lambda r: [{"regulator": i.regulator, "id": i.id,
                          "title": i.title, "published": i.published}
                         for i in items if i.regulator == r]
    got = s.check(fetcher=fetcher)
    assert got[0].id == "u"


def test_alert_format():
    item = _item()
    out = alert_text(item, profile={"business_type": "fintech startup"})
    assert "CBN" in out and "Revised cash-related policies" in out
    assert "https://www.cbn.gov.ng" in out  # official source link
    assert "Verify consequential changes" in out
    assert DISCLAIMER in out
    assert "fintech startup" in out  # relevance line


def test_alert_no_advice_language():
    item = _item()
    from nomorals.legal.contracts import information_only_check
    assert information_only_check(alert_text(item)) == []


def test_obligation_mapping():
    item = _item(regulator="NDPA", title="NDPC audit returns due",
                 summary="Compliance audit return filing deadline March 31",
                 topics=["audit", "data protection"])
    controls = controls_for(item)
    assert controls, "audit item should map to a control"
    assert any("31 March" in c.obligation for c in controls)
    c = controls[0]
    assert c.checklist and c.template


def test_obligation_controls_have_four_areas():
    assert set(OBLIGATION_CONTROLS) >= {
        "data protection audit", "tax filing", "fintech licensing",
        "securities offering"}


def test_item_from_dict_bad_input():
    assert item_from_dict({}) is None
    assert item_from_dict({"regulator": "NASA", "title": "x"}) is None
    assert item_from_dict({"regulator": "CBN"}) is None  # no title


def test_item_source_falls_back_to_homepage():
    item = item_from_dict({"regulator": "CBN", "title": "Test circular"})
    assert item is not None
    assert item.source_url == "https://www.cbn.gov.ng"


def test_chat_usage():
    out = control_regwatch("help")
    assert "/regwatch add" in out and DISCLAIMER in out


def test_chat_add_list_remove():
    class Ctx:
        regulatory_store = _store()
    ctx = Ctx()
    added = control_regwatch("add CBN fintech AML", context=ctx)
    assert "Watching CBN" in added and DISCLAIMER in added
    listed = control_regwatch("list", context=ctx)
    assert "fintech" in listed
    wid = ctx.regulatory_store.list_watches()[0].id
    removed = control_regwatch(f"remove {wid}", context=ctx)
    assert "removed" in removed
    assert control_regwatch("list", context=ctx).startswith(
        "No regulatory watches")


def test_chat_bad_regulator():
    class Ctx:
        regulatory_store = _store()
    out = control_regwatch("add NASA rockets", context=Ctx())
    assert "Unknown regulator" in out


def test_chat_regulators():
    out = control_regwatch("regulators")
    for code in ("CBN", "SEC", "NDPA", "FIRS"):
        assert code in out


def test_chat_check_no_fetcher():
    class Ctx:
        regulatory_store = _store()
    ctx = Ctx()
    ctx.regulatory_store.watch("CBN")
    out = control_regwatch("check", context=ctx)
    assert "isn't configured" in out


def test_chat_check_with_fetcher():
    class Ctx:
        regulatory_store = _store()
        regulatory_fetcher = staticmethod(
            lambda r: [{"regulator": "CBN", "id": "z9",
                        "title": "Urgent: PSP licences suspended",
                        "summary": "Immediate suspension pending review.",
                        "topics": ["fintech"]}])
        business_profile = {"business_type": "fintech startup"}
    ctx = Ctx()
    ctx.regulatory_store.watch("CBN", ["fintech"])
    out = control_regwatch("check", context=ctx)
    assert "URGENT" in out and "PSP licences suspended" in out
    assert DISCLAIMER in out


def test_check_all_sends_alerts():
    sent = []
    store = _store()
    store.watch("FIRS")
    fetcher = lambda r: [{"regulator": "FIRS", "id": "t1",
                          "title": "VAT filing deadline reminder",
                          "topics": ["tax"]}]
    alerts = check_all(store, fetcher=fetcher,
                       sender=lambda t: sent.append(t))
    assert len(alerts) == 1 and len(sent) == 1
    assert "FIRS" in sent[0] and DISCLAIMER in sent[0]


def test_check_all_never_raises():
    assert check_all(None, fetcher=lambda r: 1 / 0) == []


def test_ensure_schedule_never_raises():
    assert ensure_schedule(object()) is False
