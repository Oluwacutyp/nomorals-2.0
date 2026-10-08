"""Tests for build-map #71: flight price watchers. All offline (mock Duffel)."""
import tempfile

import pytest

from nomorals.travel.watchers import (
    PriceWatcher,
    PriceAlert,
    parse_watch_request, format_alert,
    PRICE_WATCH_ACTION, check_all,
)


class FakeOffer:
    def __init__(self, amount, currency="NGN", seats=None):
        self.total_amount = str(amount)
        self.total_currency = currency
        self.raw = {"available_services": [{}] * seats} if seats else {}


class FakeDuffel:
    def __init__(self, prices=None):
        # prices: {(origin, dest, date): [(amount, currency, seats), ...]}
        self.prices = prices or {}

    def search_offers(self, origin, destination, departure_date, **kw):
        key = (origin, destination, departure_date)
        return [FakeOffer(a, c, s) for a, c, s in self.prices.get(key, [])]


@pytest.fixture()
def watcher(tmp_path):
    return PriceWatcher(db_path=str(tmp_path / "w.db"))


# ── watch creation ────────────────────────────────────────────────────

def test_watch_creation(watcher):
    w = watcher.watch("LOS", "LHR", "2026-12-01", target_kobo=40_000_000)
    assert w.origin == "LOS" and w.destination == "LHR"
    assert w.target_kobo == 40_000_000
    assert w.route == "LOS→LHR"
    assert len(watcher.list_watches()) == 1


def test_watch_validation(watcher):
    with pytest.raises(ValueError):
        watcher.watch("Lagos", "LHR", "2026-12-01")
    with pytest.raises(ValueError):
        watcher.watch("LOS", "LHR", "next friday")


def test_unwatch(watcher):
    w = watcher.watch("LOS", "LHR", "2026-12-01")
    assert watcher.unwatch(w.id)
    assert watcher.list_watches() == []
    assert not watcher.unwatch("nope")


# ── price drop → alert ────────────────────────────────────────────────

def _w_with_duffel(tmp_path, prices):
    duffel = FakeDuffel(prices)
    w = PriceWatcher(db_path=str(tmp_path / "w2.db"), duffel=duffel)
    return w, duffel


def test_price_drop_alerts(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    watch = w.watch("LOS", "LHR", "2026-12-01")
    assert w.check(watch) is None  # first check sets baseline, no alert
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(380_000, "NGN", 3)]
    alert = w.check(watch)
    assert alert is not None
    assert alert.previous_kobo == 45_000_000
    assert alert.current_kobo == 38_000_000
    assert 15 <= alert.savings_pct <= 16
    assert "3 seats" in alert.scarcity


def test_no_drop_silent(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    watch = w.watch("LOS", "LHR", "2026-12-01")
    w.check(watch)
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(445_000, "NGN", None)]
    assert w.check(watch) is None  # 1% drop < 10% threshold


def test_threshold_respected(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    watch = w.watch("LOS", "LHR", "2026-12-01", threshold_pct=20.0)
    w.check(watch)
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(380_000, "NGN", None)]
    assert w.check(watch) is None  # 15.5% < 20%


def test_target_price_gate(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    watch = w.watch("LOS", "LHR", "2026-12-01", target_kobo=40_000_000)
    w.check(watch)
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(410_000, "NGN", None)]
    assert w.check(watch) is None  # above target → silent
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(390_000, "NGN", None)]
    alert = w.check(watch)
    assert alert is not None  # 13% drop AND under target


def test_no_double_alert(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    watch = w.watch("LOS", "LHR", "2026-12-01")
    w.check(watch)
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(380_000, "NGN", None)]
    assert w.check(watch) is not None
    assert w.check(watch) is None  # same price — no repeat
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(340_000, "NGN", None)]
    assert w.check(watch) is not None  # new low — alerts again


def test_alert_sent_via_sender(tmp_path):
    sent = []
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
    })
    w._sender = lambda text, buttons=None: sent.append((text, buttons)) or True
    watch = w.watch("LOS", "LHR", "2026-12-01")
    w.check(watch)
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(380_000, "NGN", None)]
    w.check(watch)
    assert len(sent) == 1
    text, buttons = sent[0]
    assert "LOS→LHR dropped" in text
    assert buttons and buttons[0][0][0] == "✈️ Book"


def test_check_all(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
        ("ABV", "LOS", "2026-12-05"): [(120_000, "NGN", None)],
    })
    w.watch("LOS", "LHR", "2026-12-01")
    w.watch("ABV", "LOS", "2026-12-05")
    assert check_all(w) == []  # baselines set
    duffel.prices[("LOS", "LHR", "2026-12-01")] = [(380_000, "NGN", None)]
    alerts = check_all(w)
    assert len(alerts) == 1
    assert alerts[0].route == "LOS→LHR"


# ── alert format ──────────────────────────────────────────────────────

def test_format_alert():
    a = PriceAlert(watch_id="w1", route="LOS→LHR",
                   previous_kobo=45_000_000, current_kobo=38_000_000,
                   savings_pct=15.5,
                   scarcity="Only 3 seats left at this fare.")
    text = format_alert(a)
    assert "LOS→LHR dropped" in text
    assert "₦450k → ₦380k" in text
    assert "16% off" in text
    assert "3 seats" in text
    assert a.buttons[0][0][0] == "✈️ Book"
    assert a.buttons[0][1][0] == "Dismiss"


# ── date grid ─────────────────────────────────────────────────────────

def test_price_grid(tmp_path):
    w, duffel = _w_with_duffel(tmp_path, {
        ("LOS", "LHR", "2026-12-01"): [(450_000, "NGN", None)],
        ("LOS", "LHR", "2026-12-02"): [(310_000, "NGN", None)],
        ("LOS", "LHR", "2026-12-03"): [(390_000, "NGN", None)],
    })
    points = w.price_grid("LOS", "LHR",
                          ["2026-12-01", "2026-12-02", "2026-12-03"])
    assert len(points) == 3
    best = [p for p in points if p.cheapest]
    assert len(best) == 1 and best[0].date == "2026-12-02"
    text = w.format_grid("LOS", "LHR", points)
    assert "cheapest" in text
    assert "2026-12-02" in text


def test_price_grid_empty(tmp_path):
    w, _ = _w_with_duffel(tmp_path, {})
    assert w.format_grid("LOS", "LHR", []) == "no fares found for LOS→LHR."


# ── request parsing ───────────────────────────────────────────────────

def test_parse_watch_request():
    p = parse_watch_request("track LOS LHR 2026-12-01 under 400k")
    assert p["origin"] == "LOS" and p["destination"] == "LHR"
    assert p["departure_date"] == "2026-12-01"
    assert p["target_kobo"] == 40_000_000


def test_parse_watch_request_no_target():
    p = parse_watch_request("track this flight LOS to LHR on 2026-12-01")
    assert p["origin"] == "LOS"
    assert p["target_kobo"] == 0


def test_parse_watch_request_no_route():
    assert parse_watch_request("track this flight for me") is None
    assert parse_watch_request("") is None


# ── chat commands ─────────────────────────────────────────────────────

def _runtime(tmp_path):
    from unittest.mock import MagicMock
    import nomorals.travel.watchers as tw
    from nomorals.agents.partner import runtime_memory as rm
    obj = MagicMock()
    obj._control_track = rm.RuntimeMemoryMixin._control_track.__get__(obj)
    obj._control_untrack = rm.RuntimeMemoryMixin._control_untrack.__get__(obj)
    obj._price_watcher = tw.PriceWatcher(db_path=str(tmp_path / "chat.db"))
    return obj


def test_control_track_list_empty(tmp_path):
    rt = _runtime(tmp_path)
    out = rt._control_track("list")
    assert "no active watchers" in out


def test_control_track_usage(tmp_path):
    rt = _runtime(tmp_path)
    out = rt._control_track("")
    assert "usage" in out


def test_control_track_needs_date(tmp_path):
    rt = _runtime(tmp_path)
    out = rt._control_track("LOS LHR")
    assert "route and date" in out


def test_control_untrack_usage(tmp_path):
    rt = _runtime(tmp_path)
    assert "usage" in rt._control_untrack("")


def test_control_untrack_missing(tmp_path):
    rt = _runtime(tmp_path)
    out = rt._control_untrack("watch_nope")
    assert "no watcher" in out


def test_control_track_creates(tmp_path):
    rt = _runtime(tmp_path)
    out = rt._control_track("LOS LHR 2026-12-01 under 400k")
    assert "watching LOS→LHR" in out
    assert "watch_" in out


def test_action_name():
    assert PRICE_WATCH_ACTION == "price_watch_check"
