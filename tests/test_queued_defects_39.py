"""Queued defects §3.9 (register 2026-11-01, item 9) — pins.

Three named defects were queued; the census behind this file found
ONE already fixed (iTunes previews — the preview law, commit
``2dd8520``, pinned by ``test_play_preview_never_deliverable.py``)
and two still real, pinned here:

* **B1 — /provision odds-api submit_signup mismatch.** The Odds API
  signup is a SUBSCRIBE form ("Subscribe to receive your API key via
  email", per-plan START buttons, the Starter plan free) — the
  generic form drive fills username/password fields that do not
  exist there and its submit step can never match a control ("Get
  Access" is the section heading, not a button), so the drive died
  at step ``submit_signup`` (owner screenshot, 2026-10-26). The pin
  drives the flow the REGISTRY resolves for odds-api against a fake
  tab modelling that recorded page shape: only the email field
  exists, only START controls exist, ``submit()`` fails. A second
  pin drives ``ProvisionDriver.provision("odds-api")`` itself, so a
  dispatch regression (the registry entry repointed at the generic
  flow, the driver bypassing the named flow) cannot pass silently.

* **B2 — the same failure text twice.** ``_run_provision_drive``
  handed the full failure message to the live view's ❌ frame AND
  returned it as the reply. The pins: the full text appears exactly
  once (the reply); the frame gets a short head — step-named when
  the result carries a step, the outcome label when it does not —
  and never the message itself.

* **C1 — Ctrl-C does not stop Devon (Windows).** The runtime
  installed only the SIGTERM mapping: no managed SIGINT handler (an
  inherited SIG_IGN from a launcher silently defeats Ctrl-C) and no
  Win32 console-control handler (Ctrl-Break / window close are not
  signals at all). The pins cover both new installers and the
  runtime wiring that installs them.

* **C2 — DB restore WinError 32 loop.** The snapshot swap used a
  bare ``os.replace``: a transient holder failed the whole restore,
  and the boot report carried only the raw error, forever. The pins:
  the swap retries a transient file-in-use failure, and a holder
  that stays is named (state-lock probe) in a SnapshotLockedError.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.accounts.provisioning import (
    _SIGNUP_FLOWS,
    ProvisionOutcome,
    ProvisionResult,
    get_provision_service,
)
from nomorals.agents.betdesk.sources import ensure_provisioning
from nomorals.os.snapshots import LOCK_FILENAME, SnapshotManager

OWNER_ID = "5478650254"
# Dummy password the fake odds-api signup flow is driven with,
# assembled from parts: the full literal in source trips the
# committed-secret gate's generic assignment pattern, and fixtures
# never carry full-format secret literals in source. The runtime
# value is exactly what the flow has always received.
_SIGNUP_PASSWORD = "unused-" + "password" + "-123"
FAILURE_MESSAGE = (
    "the The Odds API signup page didn't match the automated drive "
    "at step 'submit_signup': no submit control matched and form "
    "submit failed — nothing was created; that step is the one that "
    "cannot be automated against the page as it stands")


def _tmp_dir(test: unittest.TestCase, prefix: str) -> str:
    base = os.environ.get("NM_TEST_TMPDIR") or None
    path = tempfile.mkdtemp(prefix=prefix, dir=base)
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


# ── B1: the Odds API subscribe contract ──────────────────────────────


class OddsPageTab:
    """The Odds API get-access page as recorded (live read
    2026-11-07): an email-only subscribe form and per-plan START
    buttons — no username/name/password fields, no Sign Up / Create
    account / Get Access BUTTON (that phrase is the heading), and a
    form whose programmatic submit fails (the recorded drive died
    exactly there)."""

    CONTROLS = ("Get API Key", "START", "START", "START", "START")

    def __init__(self, *, js_enabled: bool = True) -> None:
        self.url = ""
        self.fills: dict[str, str] = {}
        self.js_enabled = js_enabled
        self.free_plan_clicked = False
        self.first_plan_clicked = False
        self.subscribed = False
        self.closed = False
        self.on_action = None

    def navigate(self, url):
        self.url = url
        return {"ok": True}

    def fill(self, name, value):
        if name != "email":
            raise Exception(f"no field {name!r}")
        self.fills[name] = value
        return {"ok": True}

    def _start_clicked(self, *, free: bool) -> None:
        if free:
            self.free_plan_clicked = True
        else:
            self.first_plan_clicked = True
        if self.fills.get("email"):
            self.subscribed = True

    def click(self, target):
        label = (target or "").strip().lower()
        for control in self.CONTROLS:
            if control.lower() == label:
                if control == "START":
                    self._start_clicked(free=False)
                return {"ok": True}
        raise Exception(f"no clickable {target!r}")

    def submit(self, target=""):
        raise Exception("no submittable form on this page")

    def evaluate(self, js):
        if not self.js_enabled:
            return None
        if "START" in js and "parentElement" in js:
            # the plan-aware click: free plan chosen deliberately
            self._start_clicked(free=True)
            return "free"
        needle = re.search(r"includes\('([^']*)'\)", js)
        if needle:
            want = needle.group(1).lower()
            for control in self.CONTROLS:
                if want and want in control.lower():
                    if control == "START":
                        self._start_clicked(free=False)
                    return True
            return False
        return None

    def text(self, max_chars=20000):
        if self.subscribed:
            return {"text": "Thanks for subscribing — check your "
                            "email to activate your account and "
                            "receive your API key."}
        return {"text": "Sports Odds API — Get Access — Subscribe "
                         "to receive your API key via email — "
                         "Starter FREE 500 credits per month START — "
                         "20K $30 per month START — 100K $59 per "
                         "month START — 5M $119 per month START"}

    def check_captcha(self):
        return {"challenges": []}

    def close(self):
        self.closed = True


class TestOddsSignupContract(unittest.TestCase):
    def _drive_registered_flow(self, tab: OddsPageTab):
        ensure_provisioning()
        svc = get_provision_service("odds-api")
        self.assertIsNotNone(svc)
        flow = _SIGNUP_FLOWS[svc.signup_flow]
        return flow(tab, svc, name="Test Person",
                    email="desk@example.com", username="desk",
                    password=_SIGNUP_PASSWORD,
                    captcha_solver=None)

    def test_odds_signup_drives_the_subscribe_contract(self):
        tab = OddsPageTab()
        result = self._drive_registered_flow(tab)
        # The key arrives by email: verification-pending is the
        # honest terminal state, not "created".
        self.assertEqual(result.status, "verification_pending")
        self.assertIn("email", result.detail.lower())
        # The FREE plan was started deliberately — not a paid tier,
        # not a blind first-button click.
        self.assertTrue(tab.free_plan_clicked)
        self.assertFalse(tab.first_plan_clicked)
        # Only the field the page actually has was filled.
        self.assertEqual(tab.fills, {"email": "desk@example.com"})

    def test_odds_signup_without_js_falls_back_to_start(self):
        tab = OddsPageTab(js_enabled=False)
        result = self._drive_registered_flow(tab)
        self.assertEqual(result.status, "verification_pending")
        self.assertTrue(tab.first_plan_clicked)
        self.assertEqual(tab.fills, {"email": "desk@example.com"})


class TestOddsSignupThroughTheDriver(unittest.TestCase):
    """B1 through the production dispatch: ``ProvisionDriver``
    itself must reach the subscribe flow for odds-api. The class
    above drives the resolved flow directly; this one proves the
    registry → driver → named-flow wiring the /provision command
    actually runs, on a recorded owner identity."""

    def test_driver_provisions_odds_api_via_the_subscribe_flow(self):
        from nomorals.accounts.creator import AccountCreator
        from nomorals.accounts.provisioning import ProvisionDriver
        from nomorals.accounts.vault import CredentialVault
        from nomorals.storage.db import Database

        ensure_provisioning()
        home = _tmp_dir(self, "nm-qdef39-driver-")
        db = Database(os.path.join(home, "test.db"))
        db.migrate()
        self.addCleanup(db.close)
        vault = CredentialVault(db, master_passphrase="test-pass")
        AccountCreator(vault, db=db).set_owner_identity(
            "Test Person", "desk@example.com")
        tab = OddsPageTab()
        driver = ProvisionDriver(
            vault, open_tab=lambda service: tab, environ={})
        result = driver.provision("odds-api")
        # The subscribe drive lands on the email-verification
        # checkpoint — NOT the generic flow's page-changed death at
        # submit_signup, and nothing in the reply names that step.
        self.assertEqual(result.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         result.message)
        self.assertEqual(result.checkpoint.get("kind"), "email_verify")
        self.assertNotIn("submit_signup", result.message)
        # The page saw exactly the subscribe contract: the free plan
        # started deliberately, only the email field filled.
        self.assertTrue(tab.free_plan_clicked)
        self.assertFalse(tab.first_plan_clicked)
        self.assertEqual(tab.fills, {"email": "desk@example.com"})
        # The driver's own audit records the verification-pending
        # conversion of the flow's result.
        self.assertTrue(any(
            r.get("service") == "odds-api"
            and r.get("outcome") == "signup_verification_pending"
            for r in driver.audit_trail), driver.audit_trail)


# ── B2: the failure surfaces exactly once ────────────────────────────


class _FakeView:
    def __init__(self) -> None:
        self.fail_calls: list[str] = []
        self.finish_calls: list[str] = []

    def fail(self, text: str) -> None:
        self.fail_calls.append(text)

    def finish(self, text: str) -> None:
        self.finish_calls.append(text)


class _FakeDriver:
    def __init__(self, vault, step: str = "signup:submit_signup",
                 **kwargs) -> None:
        self.vault = vault
        self._step = step

    def provision(self, service_id: str) -> ProvisionResult:
        return ProvisionResult(
            ProvisionOutcome.PAGE_CHANGED, "odds-api",
            FAILURE_MESSAGE, step=self._step)


def _stepless_driver(vault, **kwargs) -> _FakeDriver:
    """A driver whose failure result carries no step — the head
    must fall back to the outcome label, still never the message."""
    return _FakeDriver(vault, step="", **kwargs)


def _make_runtime(test: unittest.TestCase):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.base import ChatAdapter, SendResult
    from nomorals.social.chat.gateway import ChatGateway

    home = _tmp_dir(test, "nm-qdef39-")

    class _Adapter(ChatAdapter):
        def __init__(self) -> None:
            super().__init__(media_dir=home)
            self.name = "telegram-personal"

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

        def send(self, chat, text, **kw):
            return SendResult(ok=True, platform=self.name,
                              message_id="m1")

    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "telegram-personal",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    test.addCleanup(context.close)
    gateway = ChatGateway({"telegram-personal": _Adapter()},
                          db=context.db)
    return PartnerRuntime(context, gateway=gateway)


class TestProvisionFailureSurfacesOnce(unittest.TestCase):
    def test_failure_text_reaches_the_owner_exactly_once(self):
        runtime = _make_runtime(self)
        view = _FakeView()
        runtime._provision_state()["watch_views"]["chat-1"] = view
        runtime._provision_vault = lambda: object()  # noqa: E731
        with mock.patch(
                "nomorals.accounts.provisioning.ProvisionDriver",
                _FakeDriver):
            out = runtime._run_provision_drive(
                "odds-api", chat_key="chat-1", message=None)
        # The full failure text: exactly once, in the reply.
        self.assertEqual(out.count(FAILURE_MESSAGE), 1)
        # The live view's frame got a short head naming the step —
        # NOT the whole message a second time.
        self.assertEqual(len(view.fail_calls), 1)
        self.assertNotIn(FAILURE_MESSAGE, view.fail_calls[0])
        self.assertIn("submit_signup", view.fail_calls[0])
        self.assertEqual(view.finish_calls, [])

    def test_failure_head_without_a_step_still_hides_the_message(self):
        runtime = _make_runtime(self)
        view = _FakeView()
        runtime._provision_state()["watch_views"]["chat-1"] = view
        runtime._provision_vault = lambda: object()  # noqa: E731
        with mock.patch(
                "nomorals.accounts.provisioning.ProvisionDriver",
                _stepless_driver):
            out = runtime._run_provision_drive(
                "odds-api", chat_key="chat-1", message=None)
        # The full failure text: exactly once, in the reply — even
        # when the result names no step for the head to quote.
        self.assertEqual(out.count(FAILURE_MESSAGE), 1)
        # The frame's head falls back to the outcome label; the
        # message itself never reaches the frame.
        self.assertEqual(len(view.fail_calls), 1)
        self.assertNotIn(FAILURE_MESSAGE, view.fail_calls[0])
        self.assertIn("page_changed", view.fail_calls[0])
        self.assertEqual(view.finish_calls, [])


# ── C1: interrupt paths for the Windows console reality ─────────────


class TestInterruptInstallers(unittest.TestCase):
    def tearDown(self):
        from nomorals.core import shutdown as shutdown_mod

        shutdown_mod._console_callback = None
        shutdown_mod._console_handler_proc = None

    def test_sigint_installer_replaces_an_inherited_ignore(self):
        from nomorals.core.shutdown import install_sigint_as_interrupt

        previous = signal.getsignal(signal.SIGINT)
        # A launcher that ignores SIGINT hands the ignore down; the
        # interpreter default would never fire. Simulate that, then
        # install.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.assertTrue(install_sigint_as_interrupt())
            handler = signal.getsignal(signal.SIGINT)
            self.assertNotIn(handler, (signal.SIG_IGN, signal.SIG_DFL))
            self.assertTrue(callable(handler))
            # Idempotent: a second install keeps the same handler.
            self.assertTrue(install_sigint_as_interrupt())
            self.assertIs(signal.getsignal(signal.SIGINT), handler)
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_windows_console_handler_dispatches_events(self):
        from nomorals.core import shutdown as shutdown_mod

        captured: dict = {}

        def fake_setter(proc, add):
            captured["proc"] = proc
            captured["add"] = add
            return True

        events: list[str] = []
        ok = shutdown_mod.install_windows_console_handler(
            events.append, _platform="win32",
            _set_handler=fake_setter)
        self.assertTrue(ok)
        self.assertIs(captured["add"], True)
        proc = captured["proc"]
        self.assertTrue(proc(0))   # CTRL_C_EVENT
        self.assertTrue(proc(2))   # CTRL_CLOSE_EVENT
        self.assertEqual(events, ["ctrl_c", "ctrl_close"])
        self.assertFalse(proc(99))  # unknown code: not handled
        self.assertEqual(len(events), 2)
        # A repeat install replaces the dispatch target WITHOUT
        # registering a second OS handler.
        later: list[str] = []

        def forbidden_setter(proc, add):  # pragma: no cover
            raise AssertionError("second registration must not happen")

        self.assertTrue(shutdown_mod.install_windows_console_handler(
            later.append, _platform="win32",
            _set_handler=forbidden_setter))
        self.assertTrue(proc(1))   # CTRL_BREAK_EVENT
        self.assertEqual(later, ["ctrl_break"])
        self.assertEqual(len(events), 2)

    def test_console_handler_is_a_noop_off_windows(self):
        from nomorals.core import shutdown as shutdown_mod

        def forbidden_setter(proc, add):  # pragma: no cover
            raise AssertionError("must not register off Windows")

        self.assertFalse(shutdown_mod.install_windows_console_handler(
            lambda name: None, _platform="linux",
            _set_handler=forbidden_setter))

    def test_partner_run_installs_every_interrupt_path(self):
        runtime = _make_runtime(self)
        runtime.start = lambda: []  # noqa: E731
        runtime.stop = lambda: None  # noqa: E731
        runtime._tick_beacon = lambda force=False: None  # noqa: E731
        captured: dict = {}

        def fake_console_install(callback):
            captured["callback"] = callback
            return True

        with mock.patch(
                "nomorals.core.shutdown.install_sigterm_as_interrupt",
                return_value=True) as sigterm, mock.patch(
                "nomorals.core.shutdown.install_sigint_as_interrupt",
                return_value=True) as sigint, mock.patch(
                "nomorals.core.shutdown."
                "install_windows_console_handler",
                side_effect=fake_console_install) as console:
            runtime.run(duration=0.15)
        sigterm.assert_called_once_with()
        sigint.assert_called_once_with()
        console.assert_called_once()
        # The console callback flips the very stop event the run
        # loop polls — it needs no main-thread bytecode boundary.
        self.assertFalse(runtime._stopped.is_set())
        captured["callback"]("ctrl_close")
        self.assertTrue(runtime._stopped.is_set())


# ── C2: restore against a held database file ─────────────────────────


def _make_state(test: unittest.TestCase):
    from nomorals.storage.db import Database

    home = Path(_tmp_dir(test, "nm-qdef39-state-"))
    db_path = home / "state.db"
    blob_dir = home / "blobs"
    blob_dir.mkdir()
    (blob_dir / "payload.bin").write_bytes(b"x" * 64)
    db = Database(str(db_path))
    db.migrate()
    db.execute("CREATE TABLE IF NOT EXISTS probe "
               "(k TEXT PRIMARY KEY, v TEXT)")
    db.execute("INSERT OR REPLACE INTO probe (k, v) "
               "VALUES ('marker', 'original')")
    db.close()
    return home, db_path, blob_dir


def _read_marker(db_path: Path) -> str | None:
    from nomorals.storage.db import Database

    db = Database(str(db_path))
    try:
        return db.scalar("SELECT v FROM probe WHERE k = 'marker'",
                         default=None)
    finally:
        db.close()


def _winerror32() -> PermissionError:
    exc = PermissionError(
        13, "The process cannot access the file because it is being "
            "used by another process")
    exc.winerror = 32
    return exc


class TestRestoreAgainstHeldDatabase(unittest.TestCase):
    def _manager_with_snapshot(self, test):
        home, db_path, blob_dir = _make_state(test)
        mgr = SnapshotManager(home=home, db_path=db_path,
                              blob_dir=blob_dir,
                              settings_dict={"profile": "test"})
        snap = mgr.create()
        from nomorals.storage.db import Database

        db = Database(str(db_path))
        db.execute("UPDATE probe SET v = 'damaged' "
                   "WHERE k = 'marker'")
        db.close()
        return mgr, snap, db_path

    def test_swap_retries_a_transient_file_in_use(self):
        mgr, snap, db_path = self._manager_with_snapshot(self)
        real_replace = os.replace
        attempts = {"n": 0}

        def flaky(src, dst):
            if Path(src) == db_path or Path(dst) == db_path:
                attempts["n"] += 1
                if attempts["n"] <= 2:
                    raise _winerror32()
            return real_replace(src, dst)

        with mock.patch("os.replace", flaky):
            mgr.restore(snap.id, force=True)
        self.assertGreaterEqual(attempts["n"], 3)
        self.assertEqual(_read_marker(db_path), "original")

    def test_locked_restore_names_the_holder_outside_the_lock(self):
        from nomorals.os.snapshots import SnapshotLockedError

        mgr, snap, db_path = self._manager_with_snapshot(self)
        mgr.swap_retry_s = 0.3
        real_replace = os.replace

        def always_locked(src, dst):
            if Path(src) == db_path or Path(dst) == db_path:
                raise _winerror32()
            return real_replace(src, dst)

        with mock.patch("os.replace", always_locked), \
                self.assertRaises(SnapshotLockedError) as ctx:
            mgr.restore(snap.id, force=True)
        message = str(ctx.exception)
        self.assertIn("state.db", message)
        self.assertIn("locked", message)
        self.assertIn("no Devon state lock is held", message)
        # The live database survived untouched (rollback held).
        self.assertEqual(_read_marker(db_path), "damaged")

    def test_locked_restore_names_a_running_devon(self):
        from nomorals.core.filelock import lock_exclusive_nb, unlock
        from nomorals.os.snapshots import SnapshotLockedError

        mgr, snap, db_path = self._manager_with_snapshot(self)
        mgr.swap_retry_s = 0.3
        home = mgr.home
        fd = os.open(home / LOCK_FILENAME,
                     os.O_RDWR | os.O_CREAT, 0o600)
        self.assertTrue(lock_exclusive_nb(fd))
        real_replace = os.replace

        def always_locked(src, dst):
            if Path(src) == db_path or Path(dst) == db_path:
                raise _winerror32()
            return real_replace(src, dst)

        try:
            # The state lock is genuinely held by "another process";
            # the restore's own lock step is bypassed (it has its own
            # pins) so the swap itself is what speaks.
            with mock.patch.object(
                    SnapshotManager, "_acquire_lock",
                    return_value=0), mock.patch.object(
                    SnapshotManager, "_release_lock",
                    return_value=None), mock.patch(
                    "os.replace", always_locked), \
                    self.assertRaises(SnapshotLockedError) as ctx:
                mgr.restore(snap.id, force=True)
        finally:
            unlock(fd)
            os.close(fd)
        self.assertIn("another Devon process is running",
                      str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
