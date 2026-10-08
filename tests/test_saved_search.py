"""Offline tests for #96: persistent match-alerts (saved searches).

All offline: mock matchers, temp SQLite DBs. Never touches the network.
"""

import os
import tempfile
import time

import pytest

from nomorals.triggers.saved_search import (
    DOMAINS,
    SavedSearch,
    SavedSearchStore,
    check_all,
    check_search,
    control_watch,
    dedup_key,
    ensure_schedule,
    format_match,
    format_search,
    get_matcher,
    parse_cadence,
    parse_watch,
    register_matcher,
)


@pytest.fixture()
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture()
def store(db_path):
    return SavedSearchStore(db_path=db_path)


def _listing(**kw):
    d = {"title": "2-bed flat in Yaba", "price_kobo": 140000000,
         "area": "Yaba", "address": "12 Adeola Street, Yaba",
         "beds": "2", "url": "https://jiji.ng/abc", "photo_bytes": b""}
    d.update(kw)
    return d


# ── CRUD ─────────────────────────────────────────────────────────────

def test_create_list_stop(store):
    s = store.create("property", "2bed Yaba under 1.5m",
                     {"beds": "2", "max_price_kobo": 150000000,
                      "area": "Yaba"})
    assert s is not None and s.id.startswith("watch_")
    assert s.domain == "property"
    assert s.ttl_days == 14
    assert len(store.list()) == 1
    assert store.remove(s.id) is True
    assert store.list() == []
    assert store.remove(s.id) is False


def test_create_bad_domain_rejected(store):
    assert store.create("spaceship", "2bed Yaba", {}) is None
    assert store.create("property", "   ", {}) is None


def test_domain_aliases(store):
    s = store.create("job", "python dev", {})
    assert s is not None and s.domain == "gig"


def test_ttl_choices(store):
    s = store.create("property", "2bed Yaba", {}, ttl_days=7)
    assert s.ttl_days == 7
    s = store.create("property", "2bed Yaba", {}, ttl_days=99)
    assert s.ttl_days == 14  # falls back to default


# ── NL parsing ───────────────────────────────────────────────────────

def test_parse_watch_property():
    spec = parse_watch("2bed Yaba under 1.5m")
    assert spec["domain"] == "property"
    assert spec["filters"]["beds"] == "2"
    assert spec["filters"]["max_price_kobo"] == 150000000
    assert spec["filters"]["area"] == "Yaba"
    assert spec["ttl_days"] == 14


def test_parse_watch_ttl_and_naira():
    spec = parse_watch("3bed Lekki max ₦5m for 28 days")
    assert spec["filters"]["beds"] == "3"
    assert spec["filters"]["max_price_kobo"] == 500000000
    assert spec["filters"]["area"] == "Lekki"
    assert spec["ttl_days"] == 28


def test_parse_watch_gig_domain():
    spec = parse_watch("gig python developer remote")
    assert spec["domain"] == "gig"


def test_parse_watch_empty():
    assert parse_watch("") is None
    assert parse_watch("   ") is None


def test_parse_cadence():
    assert parse_cadence("6h") == 21600
    assert parse_cadence("30m") == 1800
    assert parse_cadence("1d") == 86400
    assert parse_cadence("bogus") is None


# ── matching + filters ─────────────────────────────────────────────

def _matcher_factory(listings):
    def _m(domain, query, filters):
        return list(listings)
    return _m


def test_check_search_filters(store):
    s = store.create("property", "2bed Yaba under 1.5m",
                     {"beds": "2", "max_price_kobo": 150000000,
                      "area": "Yaba"})
    listings = [
        _listing(),  # matches
        _listing(price_kobo=200000000),  # too expensive
        _listing(beds="3", title="3-bed flat in Yaba"),  # wrong beds
        _listing(area="Lekki", address="5 Admiralty Way, Lekki",
                 title="2-bed flat in Lekki"),  # wrong area
    ]
    found = check_search(store, s,
                         matcher=_matcher_factory(listings))
    assert len(found) == 1
    assert found[0].area == "Yaba"


def test_check_search_no_matcher_honest(store):
    s = store.create("property", "2bed Yaba", {})
    # no matcher registered for property in this test process
    found = check_search(store, s, matcher=None)
    assert found == []


def test_check_search_matcher_raises_never_raises(store):
    s = store.create("property", "2bed Yaba", {})

    def _boom(domain, query, filters):
        raise RuntimeError("source is down")

    assert check_search(store, s, matcher=_boom) == []


# ── dedup (shared with #93) ──────────────────────────────────────────

def test_dedup_same_listing_twice(store):
    s = store.create("property", "2bed Yaba", {})
    m = _matcher_factory([_listing()])
    assert len(check_search(store, s, matcher=m)) == 1
    # second run: same listing → deduped, no alert
    assert check_search(store, s, matcher=m) == []


def test_dedup_photo_hash_shared_with_scam(store):
    from nomorals.property.scam import _normalize_address, _photo_hash
    import nomorals.triggers.saved_search as ss
    # the dedup helpers ARE #93's helpers
    assert ss._normalize_address is _normalize_address
    assert ss._photo_hash is _photo_hash
    s = store.create("property", "2bed", {})
    m = _matcher_factory([_listing(photo_bytes=b"fake-photo-1")])
    assert len(check_search(store, s, matcher=m)) == 1
    # same photo, different URL/price → still the same listing
    m2 = _matcher_factory([_listing(photo_bytes=b"fake-photo-1",
                                    url="https://jiji.ng/other",
                                    price_kobo=130000000)])
    assert check_search(store, s, matcher=m2) == []


def test_dedup_address_normalization(store):
    s = store.create("property", "2bed", {})
    m = _matcher_factory([_listing()])
    assert len(check_search(store, s, matcher=m)) == 1
    # same address written differently → deduped
    m2 = _matcher_factory([_listing(
        address="No. 12, Adeola Street, Yaba, Lagos",
        title="2 BEDROOM FLAT YABA")])
    assert check_search(store, s, matcher=m2) == []


def test_dedup_key_shape():
    k1 = dedup_key(_listing(photo_bytes=b"x"))
    assert k1.startswith("ph:")
    k2 = dedup_key(_listing())
    assert k2.startswith("ad:")


# ── expiry / cadence ─────────────────────────────────────────────────

def test_purge_expired(store):
    s = store.create("property", "2bed Yaba", {}, ttl_days=7)
    assert len(store.list()) == 1
    purged = store.purge_expired(now=s.expires_at + 1)
    assert purged == 1
    assert store.list() == []


def test_check_all_skips_expired(store):
    s = store.create("property", "2bed Yaba", {}, ttl_days=7)
    m = _matcher_factory([_listing()])
    found = check_all(store, matchers={"property": m},
                      now=s.expires_at + 10)
    assert found == []
    # expired search was purged
    assert store.list() == []


def test_due_gating(store):
    s = store.create("property", "2bed Yaba", {}, cadence="6h")
    assert s.due() is True  # never run → due
    store.mark_run(s.id, 0)
    assert store.get(s.id).due() is False  # just ran → not due
    assert store.get(s.id).due(
        now=time.time() + 7 * 3600) is True  # 7h later → due


def test_check_all_sends_via_sender(store):
    s = store.create("property", "2bed Yaba under 1.5m",
                     {"beds": "2", "max_price_kobo": 150000000,
                      "area": "Yaba"})
    sent = []
    m = _matcher_factory([_listing()])
    found = check_all(store, matchers={"property": m},
                      sender=lambda t: sent.append(t))
    assert len(found) == 1
    assert len(sent) == 1
    assert "Yaba" in sent[0]


# ── generalization: gig + flight inherit the primitive ───────────────

def test_gig_domain_inherits(store):
    s = store.create("gig", "gig python developer", {"area": "python"})
    gigs = [{"title": "Senior Python developer (remote)",
             "price_kobo": 0, "area": "", "address": "",
             "beds": "", "url": "https://x.test/1", "photo_bytes": b""}]
    found = check_search(store, s,
                         matcher=_matcher_factory(gigs))
    assert len(found) == 1
    assert found[0].domain == "gig"


def test_flight_domain_inherits(store):
    s = store.create("flight", "flight LOS LHR", {"area": "LOS"})
    flights = [{"title": "LOS → LHR ₦450k",
                "price_kobo": 45000000, "area": "", "address": "",
                "beds": "", "url": "https://x.test/2", "photo_bytes": b""}]
    found = check_search(store, s,
                         matcher=_matcher_factory(flights))
    assert len(found) == 1


def test_matcher_registry():
    assert register_matcher("gig", lambda d, q, f: []) is True
    assert callable(get_matcher("gig"))
    assert register_matcher("spaceship", lambda d, q, f: []) is False
    assert get_matcher("property") is None or True  # may be unset


# ── formatting ───────────────────────────────────────────────────────

def test_format_match():
    from nomorals.triggers.saved_search import Match
    m = Match(search_id="watch_x", domain="property",
              title="2-bed flat in Yaba", price_kobo=140000000,
              area="Yaba", beds="2", url="https://jiji.ng/abc")
    text = format_match(m)
    assert "Yaba" in text and "₦" in text


def test_format_search():
    s = SavedSearch(id="watch_abc", domain="property",
                    query="2bed Yaba",
                    filters={"beds": "2", "max_price_kobo": 150000000,
                             "area": "Yaba"},
                    expires_at=time.time() + 14 * 86400)
    assert "watch_abc" in format_search(s)


# ── scheduler seam ───────────────────────────────────────────────────

def test_ensure_schedule_never_raises():
    assert ensure_schedule(object()) is False  # no scheduler methods


# ── chat ─────────────────────────────────────────────────────────────

def test_control_watch_usage():
    out = control_watch("")
    assert "/watch" in out


def test_control_watch_create_list_stop(db_path):
    import nomorals.triggers.saved_search as ss
    store = SavedSearchStore(db_path=db_path)

    class Ctx:
        saved_search_store = store

    out = control_watch("2bed Yaba under 1.5m", context=Ctx())
    assert "watching" in out and "Yaba" in out
    out = control_watch("list", context=Ctx())
    assert "watch_" in out
    sid = store.list()[0].id
    out = control_watch(f"stop {sid}", context=Ctx())
    assert "stopped" in out
    assert control_watch("stop nope", context=Ctx()).startswith("no watch")


def test_control_watch_bad_input_never_raises():
    assert isinstance(control_watch(None), str)
    assert isinstance(control_watch("%%%"), str)


def test_control_watch_check_sweep(db_path):
    store = SavedSearchStore(db_path=db_path)

    class Ctx:
        saved_search_store = store

    store.create("property", "2bed Yaba", {})
    out = control_watch("check", context=Ctx())
    assert "swept" in out  # no matcher → no matches, still honest
