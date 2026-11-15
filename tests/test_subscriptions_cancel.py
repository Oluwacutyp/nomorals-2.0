"""Failing-first tests: /subscriptions cancel really cancels.

Owner order 2026-11-15 (verbatim): "Asking it to cancel should be able
to cancel the subscription I don't care how.... through Google play web
or app store web or the websites." That reverses the shipped auditor's
NEVER CANCELS rule (395c4413): the draft cancel text becomes the
FALLBACK when a real cancel cannot run, never the default.

The real Play/App Store/merchant flows cannot run on this box (no
Google/Apple account here, and cancelling a real subscription in a
test is impossible), so the browser seam is a duck-typed driver
(``navigate``/``click``/``page_text``/``close`` — the exact methods a
real rendered tab and a plain tab both carry) and every flow test
drives a scripted fake. The routes, the channel evidence, the
verification markers and the failure shapes are the real logic.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.errors import NotFound
from nomorals.finance.commands import control_subscriptions
from nomorals.finance.ledger import Ledger
from nomorals.finance.subscriptions import (
    CancelJournal,
    CancelOutcome,
    ChannelStore,
    Subscription,
    audit,
    resolve_channel,
    route_for,
    run_cancel_flow,
)

NETFLIX = 650_000  # ₦6,500 in kobo


def _ctx(tmp_path: Path, driver_factory=None) -> SimpleNamespace:
    ctx = SimpleNamespace(settings=SimpleNamespace(home_path=tmp_path))
    if driver_factory is not None:
        ctx.driver_factory = driver_factory
    return ctx


def _ledger_with_netflix(tmp_path: Path, note: str = "Netflix") -> Ledger:
    """Three monthly Netflix charges (-85/-55/-25 days) + one one-off."""
    led = Ledger(tmp_path / "finance" / "ledger.jsonl")
    now = time.time()
    for days_ago in (85, 55, 25):
        led.log(NETFLIX, note=note, kind="spend", ts=now - days_ago * 86400)
    led.log(300_000, note="Mama Put", kind="spend", ts=now - 2 * 86400)
    return led


# ── Fake driver ───────────────────────────────────────────────────────────

class FakeDriver:
    """A scripted tab: pages map url -> visible text; clicks move pages.

    ``clicks`` maps (url, target) -> next url. A click with no entry
    raises LookupError, like a real tab whose target is not on the page.
    """

    def __init__(self, pages: dict[str, str],
                 clicks: dict[tuple[str, str], str] | None = None) -> None:
        self.pages = dict(pages)
        self.clicks = dict(clicks or {})
        self.url = ""
        self.clicked: list[str] = []
        self.closed = False

    def navigate(self, url: str) -> dict:
        if url not in self.pages:
            raise LookupError(f"no such page {url}")
        self.url = url
        return {"ok": True, "url": url}

    def click(self, target: str) -> dict:
        self.clicked.append(target)
        nxt = self.clicks.get((self.url, target))
        if nxt is None:
            raise LookupError(f"no {target!r} on {self.url}")
        self.url = nxt
        return {"ok": True, "target": target, "url": self.url}

    def page_text(self) -> str:
        return self.pages.get(self.url, "")

    def close(self) -> None:
        self.closed = True


def _netflix_pages() -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    cancelplan = "https://www.netflix.com/cancelplan"
    account = "https://www.netflix.com/account"
    pages = {
        cancelplan: "Manage your membership — Cancel",
        cancelplan + "#confirm": "Finish Cancellation",
        account: "Your membership will close on Dec 1, 2026. Restart membership",
    }
    clicks = {
        (cancelplan, "Cancel"): cancelplan + "#confirm",
        (cancelplan + "#confirm", "Finish Cancellation"): account,
    }
    return pages, clicks


# ── Channel resolution ────────────────────────────────────────────────────

def test_channel_from_descriptor_evidence() -> None:
    txns = [
        {"merchant": "Spotify", "amount_minor": 500_000,
         "date": date.today(), "narration": "APPLE.COM/BILL Spotify"},
        {"merchant": "Spotify", "amount_minor": 500_000,
         "date": date.today(), "narration": "APPLE.COM/BILL Spotify"},
    ]
    assert resolve_channel("Spotify", txns) == "app_store"
    txns = [
        {"merchant": "Netflix", "amount_minor": NETFLIX,
         "date": date.today(), "narration": "GOOGLE *Netflix"},
        {"merchant": "Netflix", "amount_minor": NETFLIX,
         "date": date.today(), "narration": "GOOGLE *Netflix"},
    ]
    assert resolve_channel("Netflix", txns) == "google_play"
    assert resolve_channel("Netflix", []) == "unknown"
    plain = [
        {"merchant": "Netflix", "amount_minor": NETFLIX,
         "date": date.today(), "narration": "Netflix"},
    ]
    assert resolve_channel("Netflix", plain) == "unknown"


def test_channel_store_override_wins(tmp_path: Path) -> None:
    store = ChannelStore(tmp_path / "channels.json")
    store.set_channel("Netflix", "website",
                      "https://www.netflix.com/cancelplan")
    assert store.channel("netflix") == "website"
    assert store.cancel_url("NETFLIX") == "https://www.netflix.com/cancelplan"
    assert ChannelStore(tmp_path / "channels.json").channel("Netflix") == "website"
    # A stored channel beats descriptor evidence.
    txns = [{"merchant": "Netflix", "amount_minor": NETFLIX,
             "date": date.today(), "narration": "GOOGLE *Netflix"}]
    assert resolve_channel("Netflix", txns, store) == "website"


def test_channel_store_tolerant_load(tmp_path: Path) -> None:
    bad = tmp_path / "channels.json"
    bad.write_text("{not json", encoding="utf-8")
    assert ChannelStore(bad).channel("netflix") is None
    bad.write_text(json.dumps({"netflix": {"channel": "bogus"},
                               "spotify": {"channel": "website",
                                           "cancel_url": "https://x.example/"}}),
                   encoding="utf-8")
    assert ChannelStore(bad).channel("netflix") is None
    assert ChannelStore(bad).channel("spotify") == "website"


def test_route_for_channels(tmp_path: Path) -> None:
    store = ChannelStore(tmp_path / "channels.json")
    subs, _ = audit(_ledger_with_netflix(tmp_path)
                    .transactions(kind="spend"))
    sub = subs[0]
    play = route_for(sub, "google_play", store)
    assert play is not None
    assert play.url == "https://play.google.com/store/account/subscriptions"
    assert play.service == "google"
    apple = route_for(sub, "app_store", store)
    assert apple is not None
    assert apple.url == ("https://account.apple.com/account/manage/section/"
                         "subscriptions")
    assert apple.service == "apple"
    netflix = route_for(sub, "website", store)
    assert netflix is not None
    assert netflix.url == "https://www.netflix.com/cancelplan"
    assert route_for(sub, "unknown", store) is None
    # Website channel, unlisted merchant, no stored URL: no route, no guess.
    other = SimpleNamespace(merchant="Acme VPN")
    assert route_for(other, "website", store) is None
    store.set_channel("Acme VPN", "website", "https://acme.example/cancel")
    generic = route_for(other, "website", store)
    assert generic is not None
    assert generic.url == "https://acme.example/cancel"


# ── The cancel engine (fake driver) ───────────────────────────────────────

def test_netflix_flow_cancelled(tmp_path: Path) -> None:
    pages, clicks = _netflix_pages()
    driver = FakeDriver(pages, clicks)
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    store = ChannelStore(tmp_path / "channels.json")
    route = route_for(sub, "website", store)
    assert route is not None
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "cancelled"
    assert outcome.ends_on == date(2026, 12, 1)
    assert outcome.channel == "website"
    assert "Cancel" in driver.clicked
    assert "Finish Cancellation" in driver.clicked


def test_play_flow_cancelled(tmp_path: Path) -> None:
    play = "https://play.google.com/store/account/subscriptions"
    pages = {
        play: "Subscriptions — Netflix ₦6,500/mo Renews Dec 1, 2026 Manage",
        play + "#netflix": "Netflix — Cancel subscription",
        play + "#reason": "Why are you cancelling? Prefer not to answer Continue",
        play + "#done": "Canceled — Netflix ends on Dec 1, 2026",
    }
    clicks = {
        (play, "Netflix"): play + "#netflix",
        (play + "#netflix", "Cancel subscription"): play + "#reason",
        (play + "#reason", "Prefer not to answer"): play + "#reason",
        (play + "#reason", "Continue"): play + "#done",
    }
    driver = FakeDriver(pages, clicks)
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    route = route_for(sub, "google_play",
                      ChannelStore(tmp_path / "channels.json"))
    assert route is not None
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "cancelled"
    assert outcome.ends_on == date(2026, 12, 1)


def test_already_cancelled_is_not_a_failure(tmp_path: Path) -> None:
    pages, clicks = _netflix_pages()
    pages["https://www.netflix.com/account"] = (
        "Your membership is already canceled. Restart membership")
    driver = FakeDriver(pages, clicks)
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    route = route_for(sub, "website", ChannelStore(tmp_path / "channels.json"))
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "already_cancelled"


def test_unverified_when_page_shows_no_cancellation(tmp_path: Path) -> None:
    pages = {
        "https://www.netflix.com/cancelplan": "Manage your membership — Cancel",
        "https://www.netflix.com/cancelplan#confirm": "Finish Cancellation",
        "https://www.netflix.com/account": "Membership — Renews Dec 1, 2026",
    }
    clicks = {
        ("https://www.netflix.com/cancelplan", "Cancel"):
            "https://www.netflix.com/cancelplan#confirm",
        ("https://www.netflix.com/cancelplan#confirm", "Finish Cancellation"):
            "https://www.netflix.com/account",
    }
    driver = FakeDriver(pages, clicks)
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    route = route_for(sub, "website", ChannelStore(tmp_path / "channels.json"))
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "unverified"
    assert "still subscribed" in outcome.detail.lower()


def test_required_click_missing_fails(tmp_path: Path) -> None:
    pages = {"https://www.netflix.com/cancelplan": "Manage your membership"}
    driver = FakeDriver(pages, {})
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    route = route_for(sub, "website", ChannelStore(tmp_path / "channels.json"))
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "failed"
    assert "Cancel" in outcome.detail


def test_navigate_failure_is_a_failed_outcome(tmp_path: Path) -> None:
    driver = FakeDriver({}, {})
    sub = Subscription(merchant="Netflix", occurrences=3,
                       first_seen=date(2026, 8, 22),
                       last_seen=date(2026, 10, 21),
                       cost_minor=NETFLIX, period_days=30.0)
    route = route_for(sub, "website", ChannelStore(tmp_path / "channels.json"))
    outcome = run_cancel_flow(driver, sub, route)
    assert outcome.status == "failed"


# ── Cancel journal ────────────────────────────────────────────────────────

def test_cancel_journal_round_trip(tmp_path: Path) -> None:
    journal = CancelJournal(tmp_path / "cancels.json")
    journal.record(CancelOutcome(
        merchant="Netflix", channel="website", status="cancelled",
        route_url="https://www.netflix.com/cancelplan",
        ends_on=date(2026, 12, 1), detail="verified on the page"))
    rows = CancelJournal(tmp_path / "cancels.json").all()
    assert len(rows) == 1
    assert rows[0]["merchant"] == "Netflix"
    assert rows[0]["status"] == "cancelled"
    assert rows[0]["ends_on"] == "2026-12-01"
    assert CancelJournal(tmp_path / "cancels.json").last_for("netflix")["status"] == "cancelled"
    assert CancelJournal(tmp_path / "cancels.json").last_for("nobody") is None


def test_cancel_journal_tolerant_load(tmp_path: Path) -> None:
    bad = tmp_path / "cancels.json"
    bad.write_text("{not json", encoding="utf-8")
    assert CancelJournal(bad).all() == []
    bad.write_text(json.dumps([{"merchant": "Netflix", "status": "cancelled"},
                               "junk", 42]), encoding="utf-8")
    assert len(CancelJournal(bad).all()) == 1


# ── Chat handler ──────────────────────────────────────────────────────────

def test_handler_cancel_really_cancels(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ctx = _ctx(tmp_path)
    control_subscriptions(
        "channel Netflix website https://www.netflix.com/cancelplan", ctx)
    pages, clicks = _netflix_pages()
    holder: dict[str, FakeDriver] = {}

    def factory(route):
        driver = FakeDriver(pages, clicks)
        holder["driver"] = driver
        return driver

    ctx.driver_factory = factory
    text = control_subscriptions("cancel Netflix", ctx)
    assert "cancelled" in text.lower()
    assert "Dec 1, 2026" in text or "2026-12-01" in text
    assert "draft" not in text.lower()  # no fallback on success
    assert holder["driver"].closed is True
    journal = CancelJournal(tmp_path / "finance" / "subscription_cancels.json")
    assert journal.last_for("netflix")["status"] == "cancelled"


def test_handler_cancel_login_missing_names_the_account(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ChannelStore(tmp_path / "finance" / "subscription_channels.json"
                 ).set_channel("Netflix", "google_play")

    def factory(route):
        raise NotFound("no account stored for service 'google'")

    text = control_subscriptions("cancel Netflix", _ctx(tmp_path, factory))
    assert "google" in text.lower()
    assert "no account stored" in text.lower()
    assert "draft" in text.lower()  # fallback is labelled, not hidden
    assert "cancel my subscription" in text.lower()
    journal = CancelJournal(tmp_path / "finance" / "subscription_cancels.json")
    assert journal.last_for("netflix")["status"] == "login_missing"


def test_handler_cancel_no_route_names_the_fix(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    # Unknown channel + unlisted merchant page: nothing drivable.
    store = ChannelStore(tmp_path / "finance" / "subscription_channels.json")
    store.set_channel("Netflix", "unknown")
    text = control_subscriptions("cancel Netflix", _ctx(tmp_path))
    assert "channel" in text.lower()
    assert "draft" in text.lower()
    assert "cancel my subscription" in text.lower()
    journal = CancelJournal(tmp_path / "finance" / "subscription_cancels.json")
    assert journal.last_for("netflix")["status"] == "no_route"


def test_handler_cancel_unverified_keeps_the_draft(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ChannelStore(tmp_path / "finance" / "subscription_channels.json"
                 ).set_channel("Netflix", "website",
                               "https://www.netflix.com/cancelplan")
    pages = {
        "https://www.netflix.com/cancelplan": "Manage your membership — Cancel",
        "https://www.netflix.com/cancelplan#confirm": "Finish Cancellation",
        "https://www.netflix.com/account": "Membership — Renews Dec 1, 2026",
    }
    clicks = {
        ("https://www.netflix.com/cancelplan", "Cancel"):
            "https://www.netflix.com/cancelplan#confirm",
        ("https://www.netflix.com/cancelplan#confirm", "Finish Cancellation"):
            "https://www.netflix.com/account",
    }

    def factory(route):
        return FakeDriver(pages, clicks)

    text = control_subscriptions("cancel Netflix", _ctx(tmp_path, factory))
    assert "did not show a cancellation" in text.lower()
    assert "still subscribed" in text.lower()
    assert "draft" in text.lower()
    assert "cancel my subscription" in text.lower()


def test_handler_cancel_no_subscription(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    text = control_subscriptions("cancel Nobody", _ctx(tmp_path))
    assert "no subscription found" in text.lower()


def test_handler_channel_sets_and_shows(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ctx = _ctx(tmp_path)
    text = control_subscriptions(
        "channel Netflix website https://www.netflix.com/cancelplan", ctx)
    assert "recorded" in text.lower()
    assert "website" in text.lower()
    assert "website" in control_subscriptions("channel Netflix", ctx).lower()
    assert "usage:" in control_subscriptions("channel Netflix bogus", ctx).lower()
    assert "usage:" in control_subscriptions("channel", ctx).lower()


def test_handler_list_and_used_unchanged(tmp_path: Path) -> None:
    _ledger_with_netflix(tmp_path)
    ctx = _ctx(tmp_path)
    assert "Netflix" in control_subscriptions("", ctx)
    assert "recorded" in control_subscriptions("used Netflix", ctx).lower()
    assert "suspected unused" not in control_subscriptions("", ctx).lower()


# ── Tool ──────────────────────────────────────────────────────────────────

def test_tool_cancel_really_cancels(tmp_path: Path) -> None:
    from nomorals.finance import tools as tools_mod

    class _CaptureRegistry:
        def __init__(self) -> None:
            self.fns: dict = {}

        def register(self, name, **kwargs):
            def deco(fn):
                self.fns[name] = fn
                return fn
            return deco

    cap = _CaptureRegistry()
    tools_mod.register(cap)
    assert "finance_subscriptions" in cap.fns
    _ledger_with_netflix(tmp_path)
    ChannelStore(tmp_path / "finance" / "subscription_channels.json"
                 ).set_channel("Netflix", "website",
                               "https://www.netflix.com/cancelplan")
    pages, clicks = _netflix_pages()

    def factory(route):
        return FakeDriver(pages, clicks)

    out = cap.fns["finance_subscriptions"](
        _ctx(tmp_path, factory), action="cancel", merchant="Netflix")
    assert out["ok"] is True
    assert out["status"] == "cancelled"
    assert "Netflix" in out["text"]
    assert "draft" not in out["text"].lower()


def test_registry_membership() -> None:
    from nomorals.social.chat.control import CONTROL_COMMANDS, PUBLIC_COMMANDS

    assert "subscriptions" in CONTROL_COMMANDS
    assert "subscriptions" not in PUBLIC_COMMANDS  # owner-only like /spending
