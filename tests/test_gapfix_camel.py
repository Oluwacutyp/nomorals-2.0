"""CamelCamelCamel (#22) — retail price history: watches, journalled
points, one alert per drop level, back-in-stock, chart from the
journal with the true start date.

No wall-clock reads: every "now" comes from tests.frozen_clock.
"""

from __future__ import annotations

import io

import pytest

from nomorals.commerce.price_history import (
    MARKETPLACES,
    PriceHistoryStore,
    PricePoint,
    _db_path_for,
    ensure_price_history_job,
    format_history,
    format_watches,
    parse_price_request,
    render_price_chart,
    run_price_history_tick_once,
    trend_stats,
)
from tests.frozen_clock import MIDDAY_TS, FrozenClock

DAY = 86400


def _colour_present(png: bytes, hexstr: str, tol: int = 48) -> bool:
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    r = int(hexstr[1:3], 16)
    g = int(hexstr[3:5], 16)
    b = int(hexstr[5:7], 16)
    px = img.load()
    w, h = img.size
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            pr, pg, pb = px[x, y]
            if (abs(pr - r) <= tol and abs(pg - g) <= tol
                    and abs(pb - b) <= tol):
                return True
    return False


@pytest.fixture
def clock():
    return FrozenClock()


@pytest.fixture
def store(tmp_path, clock):
    return PriceHistoryStore(str(tmp_path / "price_history.db"),
                             now=clock.now)


def _seq_fetcher(*observations):
    """Fetcher returning one observation list per call, then []."""
    calls = {"n": 0}

    def fetch(product_ref, marketplace):
        idx = calls["n"]
        calls["n"] += 1
        if idx < len(observations):
            return observations[idx]
        return []

    fetch.calls = calls
    return fetch


def _obs(price_minor, series="amazon", currency="USD", in_stock=True):
    return [{"series": series, "price_minor": price_minor,
             "currency": currency, "in_stock": in_stock}]


# ── parsing ───────────────────────────────────────────────────────────

def test_parse_watch_amazon_target():
    parsed = parse_price_request(
        "watch this iPhone 15 on Amazon under $900")
    assert parsed["kind"] == "watch"
    assert parsed["product_ref"] == "iPhone 15"
    assert parsed["marketplace"] == "amazon"
    assert parsed["series"] == "amazon"
    assert parsed["target_minor"] == 90000
    assert parsed["currency"] == "USD"


def test_parse_watch_jumia_naira():
    parsed = parse_price_request(
        "watch the iPhone 15 on jumia under ₦200k")
    assert parsed["kind"] == "watch"
    assert parsed["product_ref"] == "iPhone 15"
    assert parsed["marketplace"] == "jumia"
    assert parsed["series"] == "jumia"
    assert parsed["target_minor"] == 20000000
    assert parsed["currency"] == "NGN"


def test_parse_watch_series():
    assert parse_price_request(
        "watch the third-party new price of Elden Ring on Amazon "
        "under $40")["series"] == "tp_new"
    assert parse_price_request(
        "track third-party used Elden Ring on amazon under $25"
        )["series"] == "tp_used"


def test_parse_watch_url():
    parsed = parse_price_request(
        "watch https://www.amazon.com/dp/B0C123 under $30")
    assert parsed["product_ref"] == "https://www.amazon.com/dp/B0C123"
    assert parsed["marketplace"] == "amazon"


def test_parse_watch_needs_product():
    parsed = parse_price_request(
        "watch this on Amazon, tell me under $50")
    assert parsed["kind"] == "watch"
    assert parsed["product_ref"] == ""
    assert parsed["marketplace"] == "amazon"


def test_parse_history():
    parsed = parse_price_request("price history of iPhone 15 on Amazon")
    assert parsed["kind"] == "history"
    assert parsed["product_ref"] == "iPhone 15"
    assert parsed["marketplace"] == "amazon"


def test_parse_flight_falls_through():
    assert parse_price_request(
        "track LOS LHR 2026-12-01 under 400k") is None


def test_parse_not_a_price_request():
    assert parse_price_request("hello there") is None
    assert parse_price_request("/track LOS LHR 2026-12-01") is None


# ── marketplaces ──────────────────────────────────────────────────────

def test_marketplaces_seeded_from_shopping_sources():
    # Her shopping sources (integrations/naija_shopping.py SITES),
    # never CamelCamelCamel's fixed 8.
    for slug in ("amazon", "jumia", "konga", "jiji", "kara", "slot",
                 "temu", "aliexpress", "ebay", "banggood"):
        assert slug in MARKETPLACES
    assert MARKETPLACES["amazon"]["currency"] == "USD"
    assert MARKETPLACES["jumia"]["currency"] == "NGN"


# ── CRUD ──────────────────────────────────────────────────────────────

def test_watch_crud(store):
    w = store.watch("iPhone 15", "amazon", target_minor=90000)
    assert w.id.startswith("pwatch_")
    assert w.series == "amazon"
    assert store.get(w.id).product_ref == "iPhone 15"
    assert len(store.list_watches()) == 1
    assert store.unwatch(w.id) is True
    assert store.list_watches() == []
    assert store.list_watches(active_only=False)[0].active is False


def test_watch_validation(store):
    with pytest.raises(ValueError):
        store.watch("", "amazon")
    with pytest.raises(ValueError):
        store.watch("x", "not-a-marketplace")
    with pytest.raises(ValueError):
        store.watch("x", "amazon", series="bogus")


def test_watch_currency_override(store):
    # A ₦ target on Amazon stays ₦ — the marketplace default (USD)
    # must not swallow the currency the owner actually named.
    w = store.watch("iPhone 15", "amazon", target_minor=20000000,
                    currency="NGN")
    assert w.currency == "NGN"
    assert "₦200k" in format_watches([w])
    assert store.watch("iPhone 15", "amazon").currency == "USD"


def test_find_by_product(store):
    store.watch("iPhone 15 Pro", "amazon")
    store.watch("Samsung Galaxy S24", "jumia")
    found = store.find_by_product("iphone 15")
    assert len(found) == 1
    assert found[0].marketplace == "amazon"
    assert store.find_by_product("iphone 15", "jumia") == []


# ── alert discipline ──────────────────────────────────────────────────

def test_first_check_records_no_alert(store, clock):
    w = store.watch("iPhone 15", "amazon", target_minor=90000)
    result = store.check(w, _seq_fetcher(_obs(100000)))
    assert result["ok"] is True
    assert result["alert"] is None
    assert result["points"] == 1
    assert store.get(w.id).last_minor == 100000


def test_drop_below_target_fires_exactly_once(store, clock):
    w = store.watch("iPhone 15", "amazon", target_minor=90000)
    fetch = _seq_fetcher(_obs(100000), _obs(85000), _obs(85000),
                         _obs(75000), _obs(75000))
    assert store.check(w, fetch)["alert"] is None  # first check
    second = store.check(w, fetch)
    alert = second["alert"]
    assert alert is not None
    assert alert.kind == "drop"
    assert alert.previous_minor == 100000
    assert alert.current_minor == 85000
    assert "85" in alert.text or "$850" in alert.text
    # Second tick at the same level stays silent.
    assert store.check(w, fetch)["alert"] is None
    # A fresh low past the threshold fires again — one alert per
    # drop level. A 5.9% new low would stay silent (same idiom as
    # travel/watchers.py: threshold or fresh target crossing).
    fourth = store.check(w, fetch)
    assert fourth["alert"] is not None
    assert fourth["alert"].current_minor == 75000
    assert store.check(w, fetch)["alert"] is None


def test_threshold_drop_without_target(store):
    w = store.watch("iPhone 15", "amazon")  # no target
    fetch = _seq_fetcher(_obs(100000), _obs(88000))
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is not None


def test_small_drop_stays_silent(store):
    w = store.watch("iPhone 15", "amazon")  # no target, 10% default
    fetch = _seq_fetcher(_obs(100000), _obs(95000))
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is None


def test_target_gate_no_early_fire(store):
    # Above target, below the threshold drop: no alert.
    w = store.watch("iPhone 15", "amazon", target_minor=50000)
    fetch = _seq_fetcher(_obs(100000), _obs(95000))
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is None


def test_back_in_stock_fires_once(store):
    w = store.watch("iPhone 15", "amazon")
    fetch = _seq_fetcher(
        _obs(100000, in_stock=False),
        _obs(100000, in_stock=True),
        _obs(100000, in_stock=True))
    assert store.check(w, fetch)["alert"] is None  # first check: out
    second = store.check(w, fetch)
    assert second["alert"] is not None
    assert second["alert"].kind == "stock"
    assert "back in stock" in second["alert"].text
    assert store.check(w, fetch)["alert"] is None  # still in: silent


def test_stock_alert_cooldown(store, clock):
    w = store.watch("iPhone 15", "amazon")
    fetch = _seq_fetcher(
        _obs(100000, in_stock=False),
        _obs(100000, in_stock=True),   # fires
        _obs(100000, in_stock=False),
        _obs(100000, in_stock=True),   # within cooldown: silent
        _obs(100000, in_stock=False),
        _obs(100000, in_stock=True))   # after cooldown: fires
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is not None
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is None
    clock.advance(25 * 3600)
    assert store.check(w, fetch)["alert"] is None
    assert store.check(w, fetch)["alert"] is not None


def test_followed_series_not_observed(store):
    w = store.watch("Elden Ring", "amazon", series="tp_used")
    fetch = _seq_fetcher(_obs(5000, series="amazon"))
    result = store.check(w, fetch)
    assert result["alert"] is None
    assert "tp_used" in result["fetch_error"]
    assert result["points"] == 1  # the amazon point is still journalled


def test_no_fetcher_reports_honestly(store):
    w = store.watch("iPhone 15", "amazon")
    result = store.check(w, None)
    assert result["ok"] is False
    assert "no price source" in result["fetch_error"]


# ── history + chart ───────────────────────────────────────────────────

def _plant(store, watch_id, series_prices, clock, start_days_ago=5):
    """Plant points directly: {series: [(days_ago, minor), ...]}."""
    for series, rows in series_prices.items():
        for days_ago, minor in rows:
            store._record_point(watch_id, PricePoint(
                ts=MIDDAY_TS - days_ago * DAY, series=series,
                price_minor=minor, currency="USD", in_stock=True))


def test_chart_renders_planted_series(store, clock):
    w = store.watch("iPhone 15", "amazon")
    _plant(store, w.id, {
        "amazon": [(5, 100000), (4, 98000), (3, 95000), (0, 90000)],
        "tp_new": [(5, 95000), (3, 92000), (0, 88000)],
        "tp_used": [(5, 70000), (0, 65000)],
    }, clock)
    points = store.history(w.id)
    assert len(points) == 9
    chart = store.chart_for(w.id)
    assert chart is not None
    assert chart.series == ["amazon", "tp_new", "tp_used"]
    assert chart.n_points == 9
    assert chart.started_at == "2026-10-09"
    from PIL import Image
    img = Image.open(io.BytesIO(chart.png))
    assert img.size == (1000, 600)
    assert img.format == "PNG"
    # The planted series are really drawn: each series colour is
    # present in the pixels.
    assert _colour_present(chart.png, "#1f77b4")  # amazon
    assert _colour_present(chart.png, "#ff7f0e")  # tp_new
    assert _colour_present(chart.png, "#2ca02c")  # tp_used


def test_chart_deterministic(store, clock):
    w = store.watch("iPhone 15", "amazon")
    _plant(store, w.id, {
        "amazon": [(3, 100000), (2, 95000), (0, 90000)],
    }, clock)
    first = store.chart_for(w.id)
    second = store.chart_for(w.id)
    assert first.sha256 == second.sha256
    assert first.rung == second.rung == "matplotlib"


def test_chart_pil_fallback(store, clock, monkeypatch):
    w = store.watch("iPhone 15", "amazon")
    _plant(store, w.id, {
        "amazon": [(2, 100000), (0, 90000)],
    }, clock)

    def _no_mpl(*args, **kwargs):
        raise ImportError("no matplotlib here")

    import nomorals.commerce.price_history as ph
    monkeypatch.setattr(ph, "_render_chart_matplotlib", _no_mpl)
    chart = store.chart_for(w.id)
    assert chart is not None
    assert chart.rung == "pil"
    assert chart.series == ["amazon"]
    assert _colour_present(chart.png, "#1f77b4")
    from PIL import Image
    assert Image.open(io.BytesIO(chart.png)).size == (1000, 600)


def test_chart_empty_raises():
    with pytest.raises(ValueError):
        render_price_chart([], title="x")


def test_chart_true_start_date_not_watch_date(store, clock):
    # Watch created at T0; first observation five days later: the
    # chart starts at the first point, never backfilled.
    w = store.watch("iPhone 15", "amazon")
    clock.advance(5 * DAY)
    fetch = _seq_fetcher(_obs(100000))
    assert store.check(w, fetch)["alert"] is None
    chart = store.chart_for(w.id)
    assert chart.started_at == "2026-10-19"
    text = format_history(w, store.history(w.id))
    assert "2026-10-19" in text
    assert "the day watching began" in text


def test_trend_stats():
    points = [PricePoint(ts=1.0, series="amazon", price_minor=m)
              for m in (100000, 90000, 95000)]
    stats = trend_stats(points)
    assert stats["low"] == 90000
    assert stats["high"] == 100000
    assert stats["latest"] == 95000
    assert stats["n"] == 3
    assert trend_stats([]) == {}


def test_format_watches(store):
    assert "No active price watches" in format_watches([])
    w = store.watch("iPhone 15", "amazon", target_minor=90000)
    text = format_watches([w])
    assert "iPhone 15" in text
    assert "$900" in text


# ── tick + scheduler ──────────────────────────────────────────────────

class _FakeNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title, body):
        self.published.append((kind, title, body))
        return True


class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"name": name, "spec": spec, "id": f"job-{len(self.jobs)}"}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


def test_tick_quiet_makes_no_fetch_call(tmp_path, clock):
    calls = {"n": 0}

    def fetch(product_ref, marketplace):
        calls["n"] += 1
        return []

    result = run_price_history_tick_once(
        context=None, notifier=_FakeNotifier(), fetcher=fetch,
        now=MIDDAY_TS, home=str(tmp_path))
    assert result["ok"] is True
    assert result["checked"] == 0
    assert result["alerts"] == 0
    assert calls["n"] == 0


def test_tick_publishes_drop_once(tmp_path, clock):
    home = str(tmp_path)
    store = PriceHistoryStore(_db_path_for(home), now=clock.now)
    w = store.watch("iPhone 15", "amazon", target_minor=90000)
    fetch = _seq_fetcher(_obs(100000), _obs(85000))
    store.check(w, fetch)  # seed the baseline
    notifier = _FakeNotifier()
    result = run_price_history_tick_once(
        context=None, notifier=notifier, fetcher=fetch,
        now=MIDDAY_TS, home=home)
    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["alerts"] == 1
    assert len(notifier.published) == 1
    kind, title, body = notifier.published[0]
    assert kind == "price_history"
    assert "iPhone 15" in title
    assert "$850" in body
    # A second tick at the same level publishes nothing.
    notifier.published.clear()
    result = run_price_history_tick_once(
        context=None, notifier=notifier, fetcher=fetch,
        now=MIDDAY_TS, home=home)
    assert result["alerts"] == 0
    assert notifier.published == []


def test_ensure_job_idempotent():
    sched = _FakeScheduler()
    first = ensure_price_history_job(sched)
    second = ensure_price_history_job(sched)
    assert first == second
    assert len(sched.jobs) == 1
    assert sched.jobs[0]["spec"] == "0 7 * * *"
    # A stale spec is replaced, not duplicated.
    sched.jobs[0]["spec"] = "0 8 * * *"
    ensure_price_history_job(sched)
    assert len(sched.jobs) == 1
    assert sched.jobs[0]["spec"] == "0 7 * * *"


def test_check_all_module_entry(store):
    w = store.watch("iPhone 15", "amazon")
    from nomorals.commerce.price_history import check_all
    results = check_all(store, fetcher=_seq_fetcher(_obs(100000)))
    assert len(results) == 1
    assert results[0]["watch_id"] == w.id
