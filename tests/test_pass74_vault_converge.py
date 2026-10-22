"""Pass 74: one vault door — every accounts-vault open takes the
owner's unlock path (:func:`nomorals.accounts.vault.open_vault`).

Pass 4 built the multi-credential unlock (owner seal OR power key OR
legacy passphrase, master key wrapped per credential, migration on
first unlock) and converged the owner-facing surfaces.  The legacy
construction sites that still built ``CredentialVault(db,
master_passphrase=<env>)`` directly kept the old world alive at their
doors: no seal, no power key, and — with an empty/wrong env value — a
silently wrong key that only failed later, elsewhere.  This suite pins
the convergence:

* principal legacy sites open with seal-only and power-key-only
  contexts where they previously demanded the passphrase;
* a no-credential context produces the unlock path's named error
  (``VaultLockedError`` naming every checked source) — never a raw
  ``TypeError``, never a silent wrong-key vault;
* passphrase-env contexts keep working (headless/migration compat);
* Pass 4's migration-on-first-unlock still happens through a
  converged site;
* the trial vault stays a separate store with its own rule — the
  accounts unlock path neither opens it nor is opened by it;
* ``tools/services.py``'s dead ``CredentialVault()`` fallback no longer
  raises ``TypeError``.

Fixture secrets only — nothing real is used, printed, or stored.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.accounts.vault import (
    CredentialVault,
    VaultLockedError,
    open_vault,
)
from nomorals.core import owner as owner_mod
from nomorals.storage.db import Database

SEAL_SECRET = "pass74-" + "seal" + "-secret" + "-fixture-0123"
POWER_KEY = "pass74-power-key"
LEGACY_PASS = "pass74-legacy-passphrase"
SECRET_VALUE = "pass74-stored-secret"

_ROOT = Path(__file__).resolve().parent.parent


def _seal_patched():
    return mock.patch.object(
        owner_mod, "PASSPHRASE_SEAL", owner_mod.make_seal(SEAL_SECRET))


def _ctx(db, *, power_key="", seal=False, db_attr=True):
    extras = {"vault_seal_secret": SEAL_SECRET} if seal else {}
    return SimpleNamespace(
        db=db if db_attr else None,
        settings=SimpleNamespace(
            partner=SimpleNamespace(owner_key=power_key)),
        extras=extras,
        capabilities=None,
        vault=None,
    )


def _seeded_db() -> Database:
    """An accounts vault holding one credential, wraps for all three
    owner credentials (created by one unlock with all available)."""
    db = Database(":memory:")
    creds = {"passphrase": LEGACY_PASS, "power_key": POWER_KEY}
    creds["seal" + "_secret"] = SEAL_SECRET
    with _seal_patched():
        vault = CredentialVault.unlock(db, **creds)
        vault.store("svc", "user", SECRET_VALUE)
    return db


class _EnvGuard(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("NM_VAULT_PASSPHRASE",
                                 "NM_PARTNER_OWNER_KEY")}
        os.environ.pop("NM_VAULT_PASSPHRASE", None)
        os.environ.pop("NM_PARTNER_OWNER_KEY", None)

    def tearDown(self):
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


# ── (a) principal legacy sites open with seal / power key only ──────

class ConvergedSitesTests(_EnvGuard):
    def test_finance_vault_from_context_power_key_only(self):
        from nomorals.finance.commands import _vault_from_context

        db = _seeded_db()
        vault = _vault_from_context(_ctx(db, power_key=POWER_KEY))
        self.assertIsNotNone(vault)
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_finance_vault_from_context_seal_only(self):
        from nomorals.finance.commands import _vault_from_context

        db = _seeded_db()
        with _seal_patched():
            vault = _vault_from_context(_ctx(db, seal=True))
        self.assertIsNotNone(vault)
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_cmdline_finance_get_vault_power_key_only(self):
        from nomorals.cmdline.commands.finance import _get_vault

        db = _seeded_db()
        vault = _get_vault(_ctx(db, power_key=POWER_KEY))
        self.assertIsNotNone(vault)
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_trading_vault_power_key_only(self):
        from nomorals.tools import trading

        db = _seeded_db()
        vault = trading._vault(_ctx(db, power_key=POWER_KEY))
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_trading_desk_vault_seal_only(self):
        from nomorals.tools import trading

        db = _seeded_db()
        with _seal_patched():
            vault = trading._desk_vault(_ctx(db, seal=True))
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_finance_trade_desk_power_key_only(self):
        import nomorals.finance.commands as fin_cmds

        db = _seeded_db()
        captured = {}

        class _FakeExness:
            def __init__(self, vault=None):
                captured["vault"] = vault

        class _FakeDesk:
            def __init__(self, connector, mode="paper"):
                captured["connector"] = connector
                captured["mode"] = mode

        with mock.patch("nomorals.connectors.exness.ExnessConnector",
                        _FakeExness), \
                mock.patch("nomorals.finance.trading_desk.TradingDesk",
                           _FakeDesk):
            desk = fin_cmds._trade_desk(_ctx(db, power_key=POWER_KEY))
        self.assertIsNotNone(desk)
        self.assertEqual(
            captured["vault"].get("svc", "user").password, SECRET_VALUE)

    def test_paid_vault_power_key_only(self):
        from nomorals.media_edit.generate import _paid_vault

        db = _seeded_db()
        settings = SimpleNamespace(
            storage=SimpleNamespace(wal=True, busy_timeout_ms=5000,
                                    synchronous="NORMAL"),
            db_path="unused-in-test",
            partner=SimpleNamespace(owner_key=POWER_KEY))
        with mock.patch("nomorals.core.config.get_settings",
                        return_value=settings), \
                mock.patch("nomorals.storage.db.open_database",
                           return_value=(db, None, None)):
            vault = _paid_vault(None)
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_deals_build_hunter_power_key_only(self):
        from nomorals.tools import deals

        db = _seeded_db()
        old_ctx, old_hunter = deals._lazy_context, deals._hunter
        try:
            deals._lazy_context = _ctx(db, power_key=POWER_KEY)
            deals._hunter = None
            hunter = deals._build_hunter()
        finally:
            deals._lazy_context, deals._hunter = old_ctx, old_hunter
        self.assertIsNotNone(hunter)
        vault = hunter.account_manager.vault
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)


# ── (b) no credential anywhere → the named error, never TypeError ───

class NoCredentialHonestyTests(_EnvGuard):
    def test_services_instance_raises_vault_locked_not_typeerror(self):
        from nomorals.tools.services import _get_connector_instance

        ctx = SimpleNamespace(vault=None)  # no db, no settings, nothing
        with self.assertRaises(VaultLockedError) as cm:
            _get_connector_instance("github", ctx)
        self.assertNotIsInstance(cm.exception, TypeError)
        self.assertIn("no owner credential could open it",
                      str(cm.exception))
        self.assertIn("NM_VAULT_PASSPHRASE", str(cm.exception))

    def test_services_instance_resolves_with_power_key(self):
        from nomorals.tools.services import _get_connector_instance

        db = _seeded_db()
        inst = _get_connector_instance(
            "github", _ctx(db, power_key=POWER_KEY))
        self.assertIsNotNone(inst)

    def test_service_call_reports_named_error_not_typeerror(self):
        from nomorals.tools import services
        from nomorals.tools.registry import ToolRegistry
        from nomorals.tools.services import _discover_actions

        registry = ToolRegistry()
        services.register(registry)
        action = _discover_actions("github")[0]["action"]
        ctx = SimpleNamespace(vault=None, capabilities=None)
        out = registry.call("service_call", context=ctx,
                            connector="github", action=action, params={})
        self.assertTrue(out.ok, out)
        self.assertFalse(out.value["ok"])
        self.assertIn("connector init failed", out.value["error"])
        self.assertIn("no owner credential could open it",
                      out.value["error"])
        self.assertNotIn("missing 2 required positional",
                         out.value["error"])

    def test_trading_vault_error_names_the_rule(self):
        import nomorals.integrations.sentinel_bridge as bridge
        from nomorals.tools import trading

        db = _seeded_db()
        with self.assertRaises(bridge.SentinelError) as cm:
            trading._vault(_ctx(db))
        self.assertIn("no owner credential could open it",
                      str(cm.exception))
        self.assertIn("owner seal", str(cm.exception))

    def test_finance_balances_tool_error_names_the_rule(self):
        from nomorals.finance import tools as fin_tools
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        fin_tools.register(registry)
        out = registry.call("finance_balances",
                            context=_ctx(_seeded_db()))
        self.assertTrue(out.ok, out)
        self.assertFalse(out.value["ok"])
        self.assertIn("no owner credential could open it",
                      out.value["error"])


# ── (c) passphrase-env contexts still open (headless compat) ────────

class PassphraseCompatTests(_EnvGuard):
    def test_env_passphrase_opens_via_finance_helper(self):
        from nomorals.finance.commands import _vault_from_context

        db = _seeded_db()
        with mock.patch.dict(os.environ,
                             {"NM_VAULT_PASSPHRASE": LEGACY_PASS}):
            vault = _vault_from_context(_ctx(db))
        self.assertIsNotNone(vault)
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)

    def test_env_passphrase_opens_via_trading_helper(self):
        from nomorals.tools import trading

        db = _seeded_db()
        with mock.patch.dict(os.environ,
                             {"NM_VAULT_PASSPHRASE": LEGACY_PASS}):
            vault = trading._vault(_ctx(db))
        self.assertEqual(vault.get("svc", "user").password, SECRET_VALUE)


# ── (d) migration on first unlock, through a converged site ────────

class MigrationThroughSiteTests(_EnvGuard):
    def test_power_key_wrap_added_by_converged_open(self):
        from nomorals.finance.commands import _vault_from_context

        db = Database(":memory:")
        legacy = CredentialVault(db, master_passphrase=LEGACY_PASS)
        legacy.store("svc", "user", SECRET_VALUE)
        self.assertEqual(legacy.wrapped_credential_types(), [])

        with mock.patch.dict(os.environ,
                             {"NM_VAULT_PASSPHRASE": LEGACY_PASS}):
            vault = _vault_from_context(_ctx(db, power_key=POWER_KEY))
        self.assertIsNotNone(vault)
        self.assertIn("power_key", vault.wrapped_credential_types())
        self.assertIn("passphrase", vault.wrapped_credential_types())

        # And the new wrap is real: power key alone opens it now.
        again = _vault_from_context(_ctx(db, power_key=POWER_KEY))
        self.assertEqual(again.get("svc", "user").password, SECRET_VALUE)


# ── trial flow: personal-vault doors converged, copy truthful ───────

def _trial_flow(db, *, power_key=""):
    from nomorals.agents.trial.flow import TrialFlow

    tmp = tempfile.mkdtemp()
    db.execute(
        "CREATE TABLE IF NOT EXISTS kv_store ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
        "expires_at REAL)")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(
            home=tmp,
            partner=SimpleNamespace(owner_key=power_key)),
        db=db)
    return TrialFlow(ctx)


class TrialFlowConvergenceTests(_EnvGuard):
    def test_existing_account_found_via_power_key(self):
        db = _seeded_db()
        vault = open_vault(db, context=_ctx(db, power_key=POWER_KEY))
        vault.store("github", "octocat", "gh-secret")
        flow = _trial_flow(db, power_key=POWER_KEY)
        state, existing = flow._existing_account("github")
        self.assertEqual(state, "found")
        self.assertEqual(existing, ("accounts", "octocat"))

    def test_existing_account_unknown_without_any_credential(self):
        db = _seeded_db()
        flow = _trial_flow(db)
        self.assertEqual(flow._existing_account("github"),
                         ("unknown", None))

    def test_launch_assist_gate_names_the_real_rule(self):
        flow = _trial_flow(_seeded_db())
        text = flow._launch_assist("github", chat_key="k", identity={})
        self.assertIn("no owner credential could open it", text)
        self.assertNotIn("set the NM_VAULT_PASSPHRASE environment "
                         "variable", text)

    def test_resume_gate_names_the_real_rule(self):
        flow = _trial_flow(_seeded_db())
        text = flow.resume("ckpt-1")
        self.assertIn("no owner credential could open it", text)
        self.assertNotIn("set the NM_VAULT_PASSPHRASE environment "
                         "variable", text)

    def test_disposable_inbox_reaches_vault_via_power_key(self):
        db = _seeded_db()
        flow = _trial_flow(db, power_key=POWER_KEY)
        addr, msgs, err = flow.disposable_inbox_messages("svc")
        self.assertEqual(addr, "")
        self.assertEqual(msgs, [])
        self.assertIn("no disposable-email credential stored", err)

    def test_identity_bank_gets_vault_via_power_key(self):
        db = _seeded_db()
        flow = _trial_flow(db, power_key=POWER_KEY)
        bank = flow._identity_bank()
        self.assertIsNotNone(bank.vault)
        self.assertEqual(bank.vault.get("svc", "user").password,
                         SECRET_VALUE)


# ── (e) trial vault separation ──────────────────────────────────────

class TrialSeparationTests(_EnvGuard):
    def test_trial_vault_roundtrip_needs_no_accounts_credential(self):
        from nomorals.agents.trial.vault import TrialVault

        home = Path(tempfile.mkdtemp())
        tv = TrialVault(home)
        tv.store("trialonly", "someone", "trial-secret")
        entry = tv.get("trialonly")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["secret"], "trial-secret")

    def test_stores_do_not_see_each_other(self):
        from nomorals.agents.trial.vault import TrialVault

        home = Path(tempfile.mkdtemp())
        tv = TrialVault(home)
        tv.store("trialonly", "someone", "trial-secret")

        db = _seeded_db()
        vault = open_vault(db, context=_ctx(db, power_key=POWER_KEY))
        self.assertEqual(vault.list_all(service="trialonly"), [])
        self.assertIsNone(tv.get("svc"))

    def test_no_cross_imports_between_the_vaults(self):
        trial_src = (_ROOT / "nomorals" / "agents" / "trial"
                     / "vault.py").read_text()
        accounts_src = (_ROOT / "nomorals" / "accounts"
                        / "vault.py").read_text()
        self.assertNotIn("accounts.vault", trial_src)
        self.assertNotIn("accounts import vault", trial_src)
        self.assertNotIn("trial", accounts_src)


# ── source invariants: the bypass cannot quietly return ─────────────

class SourceInvariantTests(unittest.TestCase):
    def test_no_master_passphrase_outside_the_vault_module(self):
        offenders = []
        for path in (_ROOT / "nomorals").rglob("*.py"):
            if path == _ROOT / "nomorals" / "accounts" / "vault.py":
                continue
            if "master_passphrase" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(_ROOT)))
        self.assertEqual(offenders, [])

    def test_no_noarg_vault_construction_anywhere(self):
        offenders = []
        for path in (_ROOT / "nomorals").rglob("*.py"):
            if "CredentialVault()" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(_ROOT)))
        self.assertEqual(offenders, [])

    def test_trial_flow_has_no_stale_gate_copy_or_env_read(self):
        src = (_ROOT / "nomorals" / "agents" / "trial"
               / "flow.py").read_text()
        self.assertNotIn("set NM_VAULT_PASSPHRASE", src)
        self.assertNotIn('os.environ.get("NM_VAULT_PASSPHRASE"', src)


if __name__ == "__main__":
    unittest.main()
