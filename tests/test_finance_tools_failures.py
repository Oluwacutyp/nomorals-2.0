"""Tail-113 pins: nomorals/finance/tools.py census sites (320, 397, 564).

All three are held as exclusions:

320 — ``finance_balances``: attaching the BalanceCache to the
      context fails (frozen/slotted context) -> pass. The cache is
      a local either way and serves this call identically; only
      cross-call reuse is lost. The ok result is unchanged.
397 — ``finance_alert`` (check): a Notifier.publish crash inside
      the ``_notify`` callback is swallowed. The fired message is
      appended to ``fired_msgs`` BEFORE the delivery attempt and is
      returned to the caller, and the evaluation result is intact —
      delivery is a best-effort side channel, the firing itself is
      reported in-band.
564 — ``finance_digest``: the Notifier.publish crash is NOT
      swallowed into success — the return dict reports it honestly
      as ``delivered: False``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.accounts.vault as vault_mod
import nomorals.agents.notifier as notifier_mod
import nomorals.finance.alerts as alerts_mod
import nomorals.finance.goals as goals_mod
import nomorals.finance.insights as insights_mod
import nomorals.finance.overview as overview_mod
from nomorals.finance import digest as digest_mod
from nomorals.finance import tools as tools_mod


class _CaptureRegistry:
    """Minimal ToolRegistry stand-in: register() captures the fn."""

    def __init__(self) -> None:
        self.fns: dict = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.fns[name] = fn
            return fn
        return deco


def _fns() -> dict:
    cap = _CaptureRegistry()
    tools_mod.register(cap)
    return cap.fns


class _RaisingNotifier:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def publish(self, *args, **kwargs):
        raise RuntimeError("bus down")


class _RecordingNotifier:
    sent: list = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def publish(self, *args, **kwargs):
        type(self).sent.append(args)


# ── 320 ─────────────────────────────────────────────────────────────

class _FakeCache:
    def get(self, vault, rails=None):
        return SimpleNamespace(total_ngn=0.0, rails=[])


class _FrozenContext:
    db = None

    def __setattr__(self, key, value):
        raise RuntimeError("frozen context")


def _balances_patches():
    return (
        mock.patch.object(vault_mod, "open_vault", return_value=object()),
        mock.patch.object(overview_mod, "BalanceCache", _FakeCache),
        mock.patch.object(overview_mod, "render_overview", return_value="TEXT"),
        mock.patch.object(overview_mod, "snapshot_net_worth", return_value=None),
    )


def test_balances_cache_attach_failure_is_transparent() -> None:
    p1, p2, p3, p4 = _balances_patches()
    with p1, p2, p3, p4:
        result = _fns()["finance_balances"](_FrozenContext())
    assert result["ok"] is True
    assert result["text"] == "TEXT"


def test_balances_cache_attach_control() -> None:
    p1, p2, p3, p4 = _balances_patches()
    ctx = SimpleNamespace(db=None)
    with p1, p2, p3, p4:
        result = _fns()["finance_balances"](ctx)
    assert result["ok"] is True
    assert isinstance(ctx._finance_balance_cache, _FakeCache)


# ── 397 ─────────────────────────────────────────────────────────────

class _FakeAlertStore:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def list(self):
        return []


def _fake_evaluate(notify=None, **kwargs):
    notify("Title", "Body")
    return {"fired": 1}


def test_alert_notify_publish_crash_keeps_message() -> None:
    with mock.patch.object(alerts_mod, "AlertStore", _FakeAlertStore), \
            mock.patch.object(alerts_mod, "evaluate_alerts", _fake_evaluate), \
            mock.patch.object(notifier_mod, "Notifier", _RaisingNotifier):
        result = _fns()["finance_alert"](
            SimpleNamespace(), action="check", deliver=True)
    assert result["ok"] is True
    assert result["messages"] == ["Title\nBody"]
    assert result["fired"] == 1


def test_alert_notify_publish_control_delivers() -> None:
    _RecordingNotifier.sent = []
    with mock.patch.object(alerts_mod, "AlertStore", _FakeAlertStore), \
            mock.patch.object(alerts_mod, "evaluate_alerts", _fake_evaluate), \
            mock.patch.object(notifier_mod, "Notifier", _RecordingNotifier):
        result = _fns()["finance_alert"](
            SimpleNamespace(), action="check", deliver=True)
    assert result["messages"] == ["Title\nBody"]
    assert len(_RecordingNotifier.sent) == 1


# ── 564 ─────────────────────────────────────────────────────────────

def _digest_patches(notifier):
    # AA-45 conversion (2026-11-05): the digest composition moved
    # from tools.finance_digest to finance.digest.compose_weekly_
    # digest (the tool delegates).  The seams patched below are the
    # composition's new home module; the pinned behavior (site 564:
    # a publish crash reports delivered: False) is unchanged.
    return (
        mock.patch.object(digest_mod, "finance_paths",
                          return_value=(None, None)),
        mock.patch.object(digest_mod, "Ledger",
                          lambda path: object()),
        mock.patch.object(digest_mod, "BudgetStore",
                          lambda path: object()),
        mock.patch.object(digest_mod, "weekly_digest",
                          return_value="WEEK"),
        mock.patch.object(digest_mod, "budget_status", return_value=[]),
        mock.patch.object(digest_mod, "overspend_alerts",
                          return_value=[]),
        mock.patch.object(
            insights_mod, "safe_to_spend",
            return_value={"safe_kobo": 0, "expected_income_kobo": 0,
                          "committed_recurring_kobo": 0,
                          "spent_this_month_kobo": 0}),
        mock.patch.object(insights_mod, "upcoming_bills", return_value=[]),
        mock.patch.object(
            goals_mod, "GoalStore",
            lambda *a, **k: SimpleNamespace(list=lambda: [])),
        mock.patch.object(notifier_mod, "Notifier", notifier),
    )


def test_digest_publish_crash_reports_delivered_false() -> None:
    patches = _digest_patches(_RaisingNotifier)
    for p in patches:
        p.start()
    try:
        result = _fns()["finance_digest"](SimpleNamespace())
    finally:
        for p in patches:
            p.stop()
    assert result["ok"] is True
    assert result["delivered"] is False
    assert "WEEK" in result["text"]


def test_digest_publish_control_reports_delivered_true() -> None:
    patches = _digest_patches(_RecordingNotifier)
    for p in patches:
        p.start()
    try:
        result = _fns()["finance_digest"](SimpleNamespace())
    finally:
        for p in patches:
            p.stop()
    assert result["delivered"] is True
