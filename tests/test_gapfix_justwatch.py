"""JustWatch native build (52-websites #5): the offer relation.

Title -> (service, territory, stream|rent|buy, price, quality),
kept fresh by reconciliation, staleness on every answer, change
tracking (new/leaving) for the watchlist, watchlist alerts on the
daily tick. Fixture catalogues are the proof: a catalogue change
flips the answer on the next reconciliation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from nomorals.media.film_sources import (
    DEFAULT_TERRITORY,
    FRESH_WITHIN_S,
    OFFER_RECONCILE_JOB,
    OFFER_RECONCILE_SPEC,
    ArchiveOrgOfferAdapter,
    Offer,
    OfferAdapter,
    OfferChange,
    OfferStore,
    TMDBOfferAdapter,
    ensure_offer_reconcile_job,
    format_offers_reply,
    normalize_title,
    parse_title_query,
    reconcile_offers,
    run_offer_reconcile_tick_once,
    where_to_watch,
)
from nomorals.tools.watch_offers import watch_offers

NOW = 1795000000.0


# ── fixture adapters ────────────────────────────────────────────────


class FixtureAdapter(OfferAdapter):
    """A catalogue the tests control outright."""

    name = "fixture"

    def __init__(self, catalogue=None):
        # catalogue: {(title_key, territory): [Offer, ...]}
        self.catalogue = catalogue or {}
        self.calls: list[tuple[str, str]] = []
        self.fail_next: str = ""

    def available(self):
        return True, ""

    def fetch_offers(self, title, territory, *, now=None):
        self.calls.append((title, territory))
        if self.fail_next:
            reason = self.fail_next
            self.fail_next = ""
            raise RuntimeError(reason)
        key = (normalize_title(title), territory.upper())
        return [Offer.from_dict(o.to_dict())
                for o in self.catalogue.get(key, [])]


def _offer(title, service, kind, *, territory="NG", price=None,
           currency="", quality="", note="", checked_at=None):
    return Offer(title_ref=title, service=service, territory=territory,
                 kind=kind, price=price, currency=currency,
                 quality=quality, note=note,
                 checked_at=float(checked_at if checked_at is not None
                                  else NOW))


def _store(tmp_path):
    return OfferStore(tmp_path)


class FakeNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title, body, **_kw):
        self.published.append((kind, title, body))


def _context(tmp_path):
    settings = SimpleNamespace(home_path=str(tmp_path), home=None)
    return SimpleNamespace(settings=settings)


# ── titles ──────────────────────────────────────────────────────────


def test_normalize_title_strips_year_and_punctuation():
    assert normalize_title("Dune: Part Two (2024)") == "dune part two"
    assert normalize_title("  The  GODFATHER ") == "the godfather"
    assert normalize_title("Spider-Man: No Way Home") == "spider man no way home"


def test_parse_title_query_extracts_year():
    assert parse_title_query("Dune: Part Two (2024)") == ("Dune: Part Two", 2024)
    assert parse_title_query("Dune 2024") == ("Dune", 2024)
    assert parse_title_query("Dune") == ("Dune", None)


def test_offer_round_trips_and_keys():
    offer = _offer("Dune: Part Two", "Netflix", "stream", quality="4K")
    assert offer.key() == ("dune part two", "Netflix", "NG", "stream")
    assert Offer.from_dict(offer.to_dict()) == offer


def test_offer_change_round_trips():
    change = OfferChange(title="Dune: Part Two", territory="NG",
                         service="Netflix", kind="new", at=NOW)
    assert OfferChange.from_dict(change.to_dict()) == change


# ── store ───────────────────────────────────────────────────────────


def test_store_watchlist_round_trip(tmp_path):
    store = _store(tmp_path)
    assert store.watch("Dune: Part Two", "NG", now=NOW) is True
    assert store.watch("Dune: Part Two", "ng", now=NOW) is False
    assert store.list_watchlist() == [("Dune: Part Two", "NG")]
    assert store.unwatch("Dune: Part Two", "NG") is True
    assert store.unwatch("Dune: Part Two", "NG") is False
    assert store.list_watchlist() == []


def test_store_offers_and_staleness(tmp_path):
    store = _store(tmp_path)
    fresh = _offer("Dune", "Netflix", "stream", checked_at=NOW)
    aging = _offer("Dune", "Showmax", "stream", checked_at=NOW - 2 * 86400)
    stale = _offer("Dune", "Prime Video", "rent", price=3.99,
                   currency="USD", checked_at=NOW - 8 * 86400)
    store.upsert_offers([fresh, aging, stale], now=NOW)
    rows = store.offers_for("Dune", "NG")
    assert [o.service for o in rows] == ["Netflix", "Prime Video", "Showmax"]
    assert store.staleness(fresh, now=NOW) == "fresh"
    assert store.staleness(aging, now=NOW) == "aging"
    assert store.staleness(stale, now=NOW) == "stale"


def test_store_corrupt_file_reads_empty(tmp_path):
    path = tmp_path / "data" / "film_offers.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    store = OfferStore(tmp_path)
    assert store.offers_for("Dune", "NG") == []
    assert store.list_watchlist() == []


def test_store_changes_unseen_then_seen(tmp_path):
    store = _store(tmp_path)
    store.record_change(OfferChange(title="Dune", territory="NG",
                                    service="Netflix", kind="new", at=NOW))
    unseen = store.unseen_changes()
    assert len(unseen) == 1 and unseen[0].service == "Netflix"
    store.mark_changes_seen([unseen[0]])
    assert store.unseen_changes() == []


# ── where + reconcile against fixtures ──────────────────────────────


def _catalogue_v1():
    return {
        ("dune part two", "NG"): [
            _offer("Dune: Part Two", "Netflix", "stream", quality="4K",
                   note="included with subscription"),
            _offer("Dune: Part Two", "Prime Video", "rent", price=3.99,
                   currency="USD", quality="HD"),
        ],
    }


def test_where_answers_from_fixture_catalogue(tmp_path):
    store = _store(tmp_path)
    adapter = FixtureAdapter(_catalogue_v1())
    result = where_to_watch(store, "Dune: Part Two", "NG",
                            adapters=[adapter], now=NOW)
    assert result["ok"] is True
    assert len(result["offers"]) == 2
    services = {o["service"]: o for o in result["offers"]}
    assert services["Netflix"]["kind"] == "stream"
    assert services["Netflix"]["staleness"] == "fresh"
    assert services["Prime Video"]["price"] == 3.99
    for offer in result["offers"]:
        assert offer["checked_at"] == NOW
    assert result["stale"] is False


def test_where_with_no_offers_is_honest(tmp_path):
    store = _store(tmp_path)
    adapter = FixtureAdapter({})
    result = where_to_watch(store, "A Film Nobody Has", "NG",
                            adapters=[adapter], now=NOW)
    assert result["ok"] is False
    assert "no streaming offer" in result["error"].lower()
    assert result["offers"] == []


def test_catalogue_change_flips_answer_on_reconcile(tmp_path):
    store = _store(tmp_path)
    adapter = FixtureAdapter(_catalogue_v1())
    first = reconcile_offers(store, "Dune: Part Two", "NG",
                             adapters=[adapter], now=NOW)
    assert first["ok"] is True
    assert {o["service"] for o in first["offers"]} == {"Netflix", "Prime Video"}
    # v2: the rent offer leaves, a buy offer arrives.
    adapter.catalogue = {
        ("dune part two", "NG"): [
            _offer("Dune: Part Two", "Netflix", "stream", quality="4K",
                   note="included with subscription"),
            _offer("Dune: Part Two", "Apple TV", "buy", price=14.99,
                   currency="USD", quality="4K"),
        ],
    }
    second = reconcile_offers(store, "Dune: Part Two", "NG",
                              adapters=[adapter], now=NOW + 3600)
    services = {o["service"] for o in second["offers"]}
    assert services == {"Netflix", "Apple TV"}
    assert "Prime Video" not in services
    assert second["leaving"] == ["Prime Video"]
    assert second["new"] == ["Apple TV"]


def test_change_events_only_for_watchlisted_titles(tmp_path):
    store = _store(tmp_path)
    store.watch("Dune: Part Two", "NG", now=NOW)
    adapter = FixtureAdapter(_catalogue_v1())
    reconcile_offers(store, "Dune: Part Two", "NG",
                     adapters=[adapter], now=NOW)
    adapter.catalogue = {
        ("dune part two", "NG"): [
            _offer("Dune: Part Two", "Netflix", "stream", quality="4K"),
        ],
    }
    reconcile_offers(store, "Dune: Part Two", "NG",
                     adapters=[adapter], now=NOW + 3600)
    unseen = store.unseen_changes()
    assert [(c.kind, c.service) for c in unseen] == [
        ("new", "Netflix"), ("new", "Prime Video"), ("leaving", "Prime Video")]
    # An unwatched title records no events at all.
    store2 = OfferStore(tmp_path / "other")
    reconcile_offers(store2, "Dune: Part Two", "NG",
                     adapters=[FixtureAdapter(_catalogue_v1())], now=NOW)
    assert store2.unseen_changes() == []


def test_failed_adapter_keeps_old_offers_stale(tmp_path):
    store = _store(tmp_path)
    adapter = FixtureAdapter(_catalogue_v1())
    reconcile_offers(store, "Dune: Part Two", "NG",
                     adapters=[adapter], now=NOW)
    adapter.fail_next = "catalogue fetch down"
    result = reconcile_offers(store, "Dune: Part Two", "NG",
                              adapters=[adapter], now=NOW + 8 * 86400)
    assert result["ok"] is True
    assert len(result["offers"]) == 2
    assert result["adapters"]["fixture"]["ok"] is False
    assert "down" in result["adapters"]["fixture"]["reason"]
    assert all(o["staleness"] == "stale" for o in result["offers"])
    assert result["leaving"] == [] and result["new"] == []


def test_where_from_store_without_reconciling(tmp_path):
    store = _store(tmp_path)
    store.upsert_offers(_catalogue_v1()[("dune part two", "NG")], now=NOW)
    adapter = FixtureAdapter({})
    result = where_to_watch(store, "Dune: Part Two", "NG",
                            adapters=[adapter], now=NOW + 100)
    assert result["ok"] is True
    assert len(result["offers"]) == 2
    assert adapter.calls == []  # stored offers answer without a fetch


def test_where_refreshes_when_store_is_empty(tmp_path):
    store = _store(tmp_path)
    adapter = FixtureAdapter(_catalogue_v1())
    result = where_to_watch(store, "Dune: Part Two", "NG",
                            adapters=[adapter], now=NOW)
    assert result["ok"] is True
    assert len(adapter.calls) == 1


def test_staleness_boundaries():
    store = OfferStore("/nonexistent-home-for-test")
    fresh = _offer("Dune", "Netflix", "stream", checked_at=NOW)
    assert store.staleness(fresh, now=NOW + FRESH_WITHIN_S - 1) == "fresh"
    assert store.staleness(fresh, now=NOW + FRESH_WITHIN_S + 1) == "aging"


def test_bad_territory_refused(tmp_path):
    store = _store(tmp_path)
    result = where_to_watch(store, "Dune", "Nigeria",
                            adapters=[FixtureAdapter({})], now=NOW)
    assert result["ok"] is False
    assert "territory" in result["error"].lower()


# ── adapters ────────────────────────────────────────────────────────


def test_tmdb_adapter_without_key_is_honest():
    adapter = TMDBOfferAdapter(api_key="")
    ok, reason = adapter.available()
    assert ok is False
    assert "NM_TMDB_API_KEY" in reason


def test_tmdb_adapter_maps_providers(tmp_path):
    def fetcher(url):
        if "search/movie" in url:
            return json.dumps({"results": [
                {"id": 693134, "title": "Dune: Part Two",
                 "release_date": "2024-02-27"},
                {"id": 1, "title": "Dune", "release_date": "2021-09-15"},
            ]})
        if "watch/providers" in url:
            return json.dumps({"results": {"NG": {
                "link": "https://www.themoviedb.org/movie/693134/watch",
                "flatrate": [{"provider_name": "Netflix",
                              "display_priority": 0}],
                "rent": [{"provider_name": "Apple TV",
                          "display_priority": 2}],
                "buy": [{"provider_name": "Apple TV",
                         "display_priority": 2}],
            }}})
        raise AssertionError(f"unexpected url {url}")

    adapter = TMDBOfferAdapter(api_key="test-key", fetcher=fetcher)
    assert adapter.available()[0] is True
    offers = adapter.fetch_offers("Dune: Part Two", "NG", now=NOW)
    by_key = {(o.service, o.kind): o for o in offers}
    assert ("Netflix", "stream") in by_key
    assert ("Apple TV", "rent") in by_key
    assert ("Apple TV", "buy") in by_key
    assert by_key[("Netflix", "stream")].price is None
    assert "TMDB" in by_key[("Netflix", "stream")].note
    assert all(o.checked_at == NOW for o in offers)


def test_tmdb_adapter_refuses_inexact_title():
    def fetcher(url):
        if "search/movie" in url:
            return json.dumps({"results": [
                {"id": 1, "title": "Dune", "release_date": "2021-09-15"},
            ]})
        raise AssertionError("providers must not be fetched")

    adapter = TMDBOfferAdapter(api_key="test-key", fetcher=fetcher)
    assert adapter.fetch_offers("Dune: Part Two", "NG", now=NOW) == []


def test_tmdb_adapter_year_mismatch_refuses():
    def fetcher(url):
        if "search/movie" in url:
            return json.dumps({"results": [
                {"id": 693134, "title": "Dune: Part Two",
                 "release_date": "2024-02-27"},
            ]})
        raise AssertionError("providers must not be fetched")

    adapter = TMDBOfferAdapter(api_key="test-key", fetcher=fetcher)
    assert adapter.fetch_offers("Dune: Part Two (2021)", "NG", now=NOW) == []


def test_archiveorg_adapter_free_stream_offer():
    class StubSource:
        def search(self, query, limit=8):
            from nomorals.media.film_sources import FilmCandidate
            return [FilmCandidate(source="archiveorg",
                                  title="Night of the Living Dead",
                                  year="1968",
                                  url="https://archive.org/details/night_of_the_living_dead")]

    adapter = ArchiveOrgOfferAdapter(source=StubSource())
    offers = adapter.fetch_offers("Night of the Living Dead", "NG", now=NOW)
    assert len(offers) == 1
    offer = offers[0]
    assert offer.service == "archive.org"
    assert offer.kind == "stream"
    assert offer.price is None
    assert offer.note == "free"
    assert offer.checked_at == NOW


# ── tick ────────────────────────────────────────────────────────────


def test_tick_quiet_with_empty_watchlist_makes_no_call(tmp_path):
    adapter = FixtureAdapter(_catalogue_v1())
    notifier = FakeNotifier()
    result = run_offer_reconcile_tick_once(
        context=_context(tmp_path), notifier=notifier,
        adapters=[adapter], home=tmp_path, now=NOW)
    assert result["ok"] is True
    assert result["checked"] == 0
    assert result["alerts"] == 0
    assert adapter.calls == []
    assert notifier.published == []


def test_tick_alerts_once_per_unseen_change(tmp_path):
    store = _store(tmp_path)
    store.watch("Dune: Part Two", "NG", now=NOW)
    adapter = FixtureAdapter(_catalogue_v1())
    notifier = FakeNotifier()
    first = run_offer_reconcile_tick_once(
        context=_context(tmp_path), notifier=notifier,
        adapters=[adapter], home=tmp_path, now=NOW)
    assert first["alerts"] == 2
    assert len(notifier.published) == 2
    assert all(kind == "offer_change" for kind, _, _ in notifier.published)
    # Second tick: nothing new, silent.
    notifier.published.clear()
    second = run_offer_reconcile_tick_once(
        context=_context(tmp_path), notifier=notifier,
        adapters=[adapter], home=tmp_path, now=NOW + 3600)
    assert second["alerts"] == 0
    assert notifier.published == []


def test_tick_leaving_alert(tmp_path):
    store = _store(tmp_path)
    store.watch("Dune: Part Two", "NG", now=NOW)
    adapter = FixtureAdapter(_catalogue_v1())
    notifier = FakeNotifier()
    run_offer_reconcile_tick_once(
        context=_context(tmp_path), notifier=notifier,
        adapters=[adapter], home=tmp_path, now=NOW)
    notifier.published.clear()
    adapter.catalogue = {
        ("dune part two", "NG"): [
            _offer("Dune: Part Two", "Netflix", "stream", quality="4K"),
        ],
    }
    result = run_offer_reconcile_tick_once(
        context=_context(tmp_path), notifier=notifier,
        adapters=[adapter], home=tmp_path, now=NOW + 3600)
    assert result["alerts"] == 1
    assert "leaving" in notifier.published[0][1].lower()
    assert "Prime Video" in notifier.published[0][2]


def test_ensure_job_registers_daily_spec():
    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def list_jobs(self):
            return self.jobs

        def add(self, name, spec, kind, payload, routine=False):
            job = {"name": name, "spec": spec, "kind": kind,
                   "payload": payload, "routine": routine}
            self.jobs.append(job)
            return job

        def remove(self, job_id):
            self.jobs = [j for j in self.jobs if j.get("id") != job_id]

    scheduler = FakeScheduler()
    job = ensure_offer_reconcile_job(scheduler)
    assert job["name"] == OFFER_RECONCILE_JOB
    assert job["spec"] == OFFER_RECONCILE_SPEC == "0 6 * * *"
    assert job["payload"]["tool"] == "__offer_reconcile_tick__"
    # Idempotent: a second ensure returns the same job.
    assert ensure_offer_reconcile_job(scheduler) is job


# ── tool ────────────────────────────────────────────────────────────


def test_tool_where_watch_list_reconcile_changes(tmp_path):
    context = _context(tmp_path)
    watched = watch_offers(action="watch", title="Dune: Part Two",
                           territory="NG", context=context)
    assert watched["ok"] is True
    assert "watching" in watched["reply"].lower()
    listed = watch_offers(action="list", context=context)
    assert listed["ok"] is True
    assert "Dune: Part Two" in listed["reply"]
    changes = watch_offers(action="changes", context=context)
    assert changes["ok"] is True
    assert changes["changes"] == []


def test_tool_where_needs_a_title(tmp_path):
    result = watch_offers(action="where", title="", context=_context(tmp_path))
    assert result["ok"] is False
    assert "title" in result["error"].lower()


def test_tool_unknown_action(tmp_path):
    result = watch_offers(action="dance", context=_context(tmp_path))
    assert result["ok"] is False


def test_default_territory_is_nigeria():
    assert DEFAULT_TERRITORY == "NG"


def test_format_offers_reply_names_checked_at():
    reply = format_offers_reply({
        "ok": True, "title": "Dune: Part Two", "territory": "NG",
        "offers": [{
            "service": "Netflix", "kind": "stream", "price": None,
            "currency": "", "quality": "4K", "note": "included",
            "checked_at": NOW, "staleness": "fresh",
        }],
        "stale": False, "error": "",
    })
    assert "Netflix" in reply
    assert "stream" in reply
    assert "4K" in reply
    assert "fresh" in reply
    assert "checked" in reply.lower()
