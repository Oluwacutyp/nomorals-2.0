"""Telegram live-attach: the conversational surface follows the pool.

The WhatsApp half of §15 per-account routing shipped with the
live-attach reconcile (`_sync_whatsapp_surface`): the gateway's
single whatsapp adapter is (re)built against the resolved account's
bridge port.  The Telegram half never existed:

- the gateway's telegram adapter is built ONCE at boot from the
  static `chat.telegram_session` (`data/telegram.session`) and never
  re-pointed;
- `/pair use` on a telegram account promises "surfaces that act as
  you on telegram use it" but syncs nothing;
- a `tgcode` success leaves the session file sitting unused by the
  conversational surface;
- `/pair stop` (telegram) records no platform owner-stopped intent,
  so a reconcile that did exist would resurrect the surface within
  a beat (the WhatsApp stopfix lesson, unapplied).

The fix, pinned here — `_sync_telegram_surface`, the exact mirror
of the WhatsApp reconcile:

1.  Resolve via `resolve_account(registry, "telegram")` (primary →
    first linked → first registered); qualify only `linked`/`down`.
2.  Already serving the resolved session + alive → adopt, no rebuild,
    no probe.
3.  Otherwise PROBE authorization first (`pairer.probe`): the
    adapter's `start` runs `preflight` internally, and preflight
    does an INTERACTIVE first-run login on an unauthorized session
    (prints + reads stdin) — attaching an unverified session from a
    beat or a /pair reply would race the console for the keyboard.
    Probe `down` → no adapter, the record flips to `down`, the
    verdict is a "session down — …" note on the reply AND the
    account record, with ONE failure-ledger entry per episode.
    Probe `linked` → the adapter is (re)built on the resolved
    account's own session file and started in the running process.
4.  Owner-stopped is absolute: while the order stands the reconcile
    launches NEITHER probe success nor adapter — and an owned adapter
    comes down.
5.  No paired Telegram account → a boot adapter on a session outside
    the pool is left alone, exactly like WhatsApp.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.failures import clear_failures, recent_failures
from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.pairing import AccountRecord, PairingRegistry
from nomorals.social.telegram_pairing import TelegramPairer, TwoFactorRequired

ACCT = "telegram:2348012345678"
DIGITS = "2348012345678"
ACCT2 = "telegram:2348098765432"
DIGITS2 = "2348098765432"


# ── fakes ───────────────────────────────────────────────────────────

class FakeTgClient:
    """A Telethon-shaped client the pairer drives (no Telethon, no
    network)."""

    def __init__(self, *, authorized=True, two_factor=False):
        self.authorized = authorized
        self.two_factor = two_factor
        self.connected = False
        self.codes_sent = 0

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def send_code(self, phone):
        self.codes_sent += 1
        return "hash-123"

    async def sign_in_code(self, phone, code, code_hash):
        if self.two_factor:
            raise TwoFactorRequired("account has two-factor auth on")
        self.authorized = True

    async def sign_in_password(self, password):
        self.authorized = True

    async def is_authorized(self):
        return self.authorized

    async def me_id(self):
        return "999001"


class FakeTgAdapter:
    """A TelegramAdapter-shaped object: session path + a live thread
    the reconcile's liveness check can read."""

    name = "telegram"

    def __init__(self, session_path):
        self.session_path = session_path
        self._thread = None
        self._gate = threading.Event()
        self.stopped = False

    def start(self, handler):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._thread = threading.Thread(
            target=self._gate.wait, name="fake-telegram", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.stopped = True
        self._gate.set()

    def health(self):
        return {"name": "telegram",
                "running": self._thread is not None
                and self._thread.is_alive()}


def make_pairer(tmp_path, registry, *, authorized=True,
                two_factor=False):
    state = {"authorized": authorized}
    clients = []

    def factory(stem, api_id, api_hash):
        client = FakeTgClient(authorized=state["authorized"],
                              two_factor=two_factor)
        clients.append(client)
        return client

    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "telegram_sessions",
        api_id="12345", api_hash="hash", client_factory=factory)
    return pairer, clients, state


def make_settings(tmp_path):
    chat = SimpleNamespace(
        whatsapp_enabled=False,
        telegram_enabled=True,
        telegram_api_id="12345", telegram_api_hash="hash",
        telegram_session="",
        telegram_chats="", media_in_groups=False, media_max_mb=20.0,
        threads_enabled=True, telegram_bot_token="",
        telegram_personal_bot_token="")
    return SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)


def make_runtime(tmp_path, registry, pairer):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    settings = make_settings(tmp_path)
    rt.settings = settings
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._pair_supervisor = StubSupervisor(registry)
    rt._pair_personal_svc = {
        "registry": registry, "pairer": pairer,
        "supervisor": rt._pair_supervisor, "personal": None,
        "supervision": None, "last_cycle": 0.0}
    return rt


class StubSupervisor:
    """The pairing supervisor's shape the /pair paths drive
    (registry + remove); the bridge half is never touched here."""

    def __init__(self, registry):
        self.registry = registry

    def remove(self, account_id):
        return self.registry.remove(account_id)


def linked_record(tmp_path, digits=DIGITS, acct=ACCT):
    stem = str(tmp_path / "telegram_sessions" / f"tg-{digits}")
    return AccountRecord(
        id=acct, platform="telegram", identity=digits,
        status="linked", session_path=stem)


def touch_session(tmp_path, digits=DIGITS):
    """The session file the pairer's probe needs to exist before it
    asks MTProto anything."""
    d = tmp_path / "telegram_sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tg-{digits}.session").touch()
    return str(d / f"tg-{digits}")


def boot_runtime(tmp_path, *, authorized=True, records=()):
    """A fresh Devon process over a persisted registry: linked
    Telegram account(s), NO adapter attached yet."""
    registry = PairingRegistry(tmp_path / "paired.json")
    for rec in records:
        registry.upsert(rec)
    pairer, clients, state = make_pairer(tmp_path, registry,
                                         authorized=authorized)
    rt = make_runtime(tmp_path, registry, pairer)
    return rt, registry, pairer, clients, state


@pytest.fixture
def fake_build(monkeypatch):
    """`build_adapter` that mints FakeTgAdapters and counts builds
    (Telethon is not installed here, so the real adapter cannot
    run)."""
    builds = []

    def _build(settings, name, **kw):
        builds.append((name, dict(kw)))
        assert name == "telegram"
        return FakeTgAdapter(kw.get("telegram_session") or "")

    monkeypatch.setattr("nomorals.social.chat.build_adapter", _build)
    return builds


# ── 1. boot: the reconcile attaches the linked account ──────────────

def test_boot_reconcile_attaches_linked_account(tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        note = rt._sync_telegram_surface()
        assert note == ""
        assert len(fake_build) == 1
        assert fake_build[0][1]["telegram_session"] == \
            linked_record(tmp_path).session_path
        adapter = rt.gateway.adapters.get("telegram")
        assert adapter is not None
        assert adapter.session_path == linked_record(tmp_path).session_path
        assert rt._tg_surface_account == ACCT
        # A second reconcile is a no-op: no second build, no probe.
        assert rt._sync_telegram_surface() == ""
        assert len(fake_build) == 1
    finally:
        rt.gateway.stop()


def test_boot_adapter_serving_resolved_session_is_adopted(
        tmp_path, fake_build):
    """The boot-time adapter already serves the resolved account's
    session (same file, live thread): adopt it — no rebuild, and a
    later removal of the account still tears the surface down."""
    stem = touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        adopted = FakeTgAdapter(stem)
        assert rt.gateway.start_one(
            "telegram", lambda m: None, adapter=adopted)["ok"]
        assert rt._sync_telegram_surface() == ""
        assert fake_build == []  # nothing rebuilt
        assert rt.gateway.adapters["telegram"] is adopted
        assert rt._tg_surface_account == ACCT
    finally:
        rt.gateway.stop()


# ── 2. an unauthorized session is never attached ────────────────────

def test_unauthorized_session_is_never_attached(tmp_path, fake_build):
    """Probe says down: no adapter, no "live-attached" claim — a
    session-down note on the reply AND the account record, and ONE
    failure-ledger entry for the whole episode (the next beat must
    not spam it)."""
    clear_failures()
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, authorized=False,
        records=[linked_record(tmp_path)])
    try:
        note = rt._sync_telegram_surface()
        assert "session down" in note
        assert "telegram" not in rt.gateway.adapters
        assert getattr(rt, "_tg_surface_account", "") == ""
        assert fake_build == []
        stored = registry.get(ACCT)
        assert stored.status == "down"
        assert stored.last_error.startswith("telegram surface: ")
        assert "session down" in stored.last_error
        # Same dead session, next beat: same note, still one entry.
        assert rt._sync_telegram_surface() == note
        hits = [f for f in recent_failures(50)
                if f.site == "runtime_provision.tg_session_down"]
        assert len(hits) == 1
        assert ACCT in hits[0].message
    finally:
        rt.gateway.stop()


def test_session_death_tears_surface_down(tmp_path, fake_build):
    """Attached and live, then the session stops authorizing: the
    next reconcile puts the surface down and notes it — never left
    "attached" to a dead session."""
    touch_session(tmp_path)
    rt, registry, pairer, _, state = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        assert rt._sync_telegram_surface() == ""
        assert "telegram" in rt.gateway.adapters
        state["authorized"] = False
        note = rt._sync_telegram_surface()
        assert "session down" in note
        assert "telegram" not in rt.gateway.adapters
        assert rt._tg_surface_account == ""
        assert registry.get(ACCT).status == "down"
    finally:
        rt.gateway.stop()


def test_session_reauthorizes_and_reattaches(tmp_path, fake_build):
    """A down record whose session authorizes again is earned back:
    probe `linked` flips the record and re-attaches."""
    touch_session(tmp_path)
    rt, registry, pairer, _, state = boot_runtime(
        tmp_path, authorized=False,
        records=[linked_record(tmp_path, )])
    try:
        registry.set_status(ACCT, "down")
        state["authorized"] = True
        assert rt._sync_telegram_surface() == ""
        assert "telegram" in rt.gateway.adapters
        assert registry.get(ACCT).status == "linked"
        assert registry.get(ACCT).last_error == ""
    finally:
        rt.gateway.stop()


# ── 3. owner stop is absolute ───────────────────────────────────────

def test_owner_stopped_boot_attaches_nothing(tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        registry.set_owner_stopped("telegram", True)
        assert rt._sync_telegram_surface() == ""
        assert fake_build == []
        assert "telegram" not in rt.gateway.adapters
        assert rt._tg_surface_account == ""
    finally:
        rt.gateway.stop()


def test_owner_stop_tears_owned_surface_down(tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        assert rt._sync_telegram_surface() == ""
        registry.set_owner_stopped("telegram", True)
        assert rt._sync_telegram_surface() == ""
        assert "telegram" not in rt.gateway.adapters
        assert rt._tg_surface_account == ""
        # And the next beat does not resurrect it.
        assert rt._sync_telegram_surface() == ""
        assert len(fake_build) == 1  # attached once, never rebuilt
        assert "telegram" not in rt.gateway.adapters
    finally:
        rt.gateway.stop()


def test_no_paired_account_leaves_boot_adapter_alone(
        tmp_path, fake_build):
    """No Telegram account in the pool: a boot adapter on a session
    outside the pool is left alone — the reconcile only ever stops
    what it owns."""
    rt, registry, pairer, _, _ = boot_runtime(tmp_path)
    try:
        outside = FakeTgAdapter(str(tmp_path / "data" / "telegram.session"))
        assert rt.gateway.start_one(
            "telegram", lambda m: None, adapter=outside)["ok"]
        assert rt._sync_telegram_surface() == ""
        assert fake_build == []
        assert rt.gateway.adapters["telegram"] is outside
        assert getattr(rt, "_tg_surface_account", "") == ""
    finally:
        rt.gateway.stop()


# ── 4. the primary switch re-points the surface ─────────────────────

def test_primary_switch_repoints_surface(tmp_path, fake_build):
    touch_session(tmp_path, DIGITS)
    touch_session(tmp_path, DIGITS2)
    rec_a = linked_record(tmp_path)
    rec_b = linked_record(tmp_path, DIGITS2, ACCT2)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[rec_a, rec_b])
    try:
        assert rt._sync_telegram_surface() == ""
        assert rt.gateway.adapters["telegram"].session_path == \
            rec_a.session_path
        registry.set_primary(ACCT2)
        assert rt._sync_telegram_surface() == ""
        assert len(fake_build) == 2
        assert rt.gateway.adapters["telegram"].session_path == \
            rec_b.session_path
        assert rt._tg_surface_account == ACCT2
    finally:
        rt.gateway.stop()


def test_use_verb_switches_surface(tmp_path, fake_build):
    """`/pair use` on a telegram account: the reply promises the
    surfaces follow — and now they do."""
    touch_session(tmp_path, DIGITS)
    touch_session(tmp_path, DIGITS2)
    rec_a = linked_record(tmp_path)
    rec_b = linked_record(tmp_path, DIGITS2, ACCT2)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[rec_a, rec_b])
    try:
        assert rt._sync_telegram_surface() == ""
        reply = rt.handle_control(f"/pair use {ACCT2}", "local:owner")
        assert "is now the default telegram account" in reply
        assert rt.gateway.adapters["telegram"].session_path == \
            rec_b.session_path
        assert rt._tg_surface_account == ACCT2
    finally:
        rt.gateway.stop()


# ── 5. the pairing flow lands on the surface ────────────────────────

def test_tgcode_success_attaches(tmp_path, fake_build):
    """`/pair telegram` → code → `/pair tgcode`: the link lands and
    the surface attaches in the same turn — the session file no
    longer sits unused."""
    rt, registry, pairer, clients, state = boot_runtime(tmp_path)
    try:
        rec = pairer.start(DIGITS)
        assert rec.status == "pairing"
        touch_session(tmp_path, DIGITS)
        reply = rt.handle_control(
            f"/pair tgcode {rec.id} 12345", "local:owner")
        assert "is linked" in reply
        assert "telegram" in rt.gateway.adapters
        assert rt._tg_surface_account == rec.id
        assert registry.get(rec.id).status == "linked"
    finally:
        rt.gateway.stop()


def test_pair_stop_records_owner_stop_and_tears_down(
        tmp_path, fake_build):
    """/pair stop on a telegram account: the platform stop is
    recorded (sticky) and the owned surface comes down — the next
    beat does not resurrect it."""
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        assert rt._sync_telegram_surface() == ""
        reply = rt.handle_control(f"/pair stop {ACCT}", "local:owner")
        assert "marked stopped" in reply
        assert registry.owner_stopped("telegram")
        assert "telegram" not in rt.gateway.adapters
        assert rt._sync_telegram_surface() == ""
        assert "telegram" not in rt.gateway.adapters
    finally:
        rt.gateway.stop()


def test_pair_remove_tears_surface_down(tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        assert rt._sync_telegram_surface() == ""
        reply = rt.handle_control(f"/pair remove {ACCT}", "local:owner")
        assert "removed from the pool" in reply
        assert "telegram" not in rt.gateway.adapters
        assert rt._tg_surface_account == ""
    finally:
        rt.gateway.stop()


# ── 6. /start telegram + /platforms ─────────────────────────────────

def test_start_platform_telegram_routes_through_reconcile(
        tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        result = rt.start_platform("telegram")
        assert result.get("ok")
        assert "telegram" in rt.gateway.adapters
        assert rt._tg_surface_account == ACCT
    finally:
        rt.gateway.stop()


def test_start_platform_telegram_session_down_is_an_honest_error(
        tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, authorized=False,
        records=[linked_record(tmp_path)])
    try:
        result = rt.start_platform("telegram")
        assert not result.get("ok")
        assert "session down" in result.get("error", "")
        assert "telegram" not in rt.gateway.adapters
    finally:
        rt.gateway.stop()


def test_platforms_renders_session_down_when_never_attached(
        tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, authorized=False,
        records=[linked_record(tmp_path)])
    try:
        registry.set_status(ACCT, "down")
        registry.set_error(ACCT, "telegram surface: session down — "
                           "the saved session no longer authorizes")
        reply = rt.handle_control("/platforms", "local:owner")
        assert "telegram: session down — " in reply
        assert "telegram: running" not in reply
    finally:
        rt.gateway.stop()


def test_platforms_renders_running_when_session_serves(
        tmp_path, fake_build):
    touch_session(tmp_path)
    rt, registry, pairer, _, _ = boot_runtime(
        tmp_path, records=[linked_record(tmp_path)])
    try:
        assert rt._sync_telegram_surface() == ""
        reply = rt.handle_control("/platforms", "local:owner")
        assert "telegram: running" in reply
        assert "session down" not in reply
    finally:
        rt.gateway.stop()
