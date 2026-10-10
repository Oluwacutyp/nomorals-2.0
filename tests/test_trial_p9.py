"""Phase 9 Slice D: trial system + gig applier god-tier upgrades.

Offline-safe: network, browser, and LLM are mocked or injected.  Real
tests for breakable behavior: crypto roundtrip + tamper detection,
one-account-per-service enforcement, honest submission statuses,
follow-up loop, and secret-free audit trails.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import gig_applier as G
from nomorals.agents import opportunities as O
from nomorals.agents.trial.flow import TrialFlow
from nomorals.agents.trial.vault import TrialVault
from nomorals.core.errors import ToolError
from nomorals.storage.db import Database


def _opp(**kw):
    base = dict(kind="paid_task", title="AI training gig",
                source="outlier.ai", url="https://outlier.ai/apply",
                payout_text="$20/hr", effort="medium",
                regions=["global"], skills=["ai training"])
    base.update(kw)
    return O.Opportunity(**base)


def _flow():
    tmp = tempfile.mkdtemp()
    db = Database(":memory:")
    # full kv_store schema (mirrors migrations: includes expires_at)
    db.execute(
        "CREATE TABLE IF NOT EXISTS kv_store ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
        "expires_at REAL)")
    ctx = SimpleNamespace(settings=SimpleNamespace(home=tmp), db=db)
    return TrialFlow(ctx), db, tmp


def _vault_home(testcase):
    home = Path(tempfile.mkdtemp())
    testcase.addCleanup(shutil.rmtree, home, ignore_errors=True)
    return home


# ── TrialVault ─────────────────────────────────────────────────────────────

class TrialVaultP9Tests(unittest.TestCase):
    def test_store_get_roundtrip(self):
        v = TrialVault(_vault_home(self))
        v.store("github", "octocat", "pw123")
        got = v.get("github")
        self.assertEqual("pw123", got["secret"])
        self.assertEqual("octocat", got["login"])
        self.assertFalse(got.get("unreadable", False))

    def test_store_overwrote_flag(self):
        v = TrialVault(_vault_home(self))
        first = v.store("github", "octocat", "pw1")
        self.assertFalse(first["overwrote"])
        second = v.store("github", "octocat", "pw2")
        self.assertTrue(second["overwrote"])
        self.assertEqual("pw2", v.get("github")["secret"])

    def test_store_requires_all_fields(self):
        v = TrialVault(_vault_home(self))
        with self.assertRaises(ValueError):
            v.store("", "u", "p")
        with self.assertRaises(ValueError):
            v.store("x", "", "p")

    def test_get_missing_none(self):
        v = TrialVault(_vault_home(self))
        self.assertIsNone(v.get("nope"))

    def test_tamper_detected(self):
        home = _vault_home(self)
        v = TrialVault(home)
        v.store("github", "octocat", "pw123")
        p = home / "trial_accounts.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        blob = data["github"]["secret"]
        data["github"]["secret"] = blob[:-1] + ("0" if blob[-1] != "0" else "1")
        p.write_text(json.dumps(data), encoding="utf-8")
        got = v.get("github")
        self.assertTrue(got["unreadable"])
        self.assertIsNone(got["secret"])

    def test_stale_flag(self):
        home = _vault_home(self)
        v = TrialVault(home)
        v.store("old", "u", "p")
        p = home / "trial_accounts.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["old"]["saved_at"] = time.time() - 100 * 86400
        p.write_text(json.dumps(data), encoding="utf-8")
        rows = v.list()
        self.assertTrue(rows[0]["stale"])
        self.assertGreater(rows[0]["age_days"], 90)
        self.assertTrue(v.get("old")["stale"])
        # fresh entries are not stale
        v.store("new", "u", "p")
        fresh = [r for r in v.list() if r["platform"] == "new"][0]
        self.assertFalse(fresh["stale"])

    def test_audit_trail_secret_free(self):
        home = _vault_home(self)
        v = TrialVault(home)
        v.store("github", "octocat", "s3cr3t-pw-999")
        v.get("github")
        v.list()
        v.delete("github")
        rows = v.audit_trail()
        actions = [r["action"] for r in rows]
        for expected in ("store", "get", "list", "delete"):
            self.assertIn(expected, actions)
        raw = (home / "trial_access.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("s3cr3t-pw-999", raw)
        self.assertNotIn("octocat", raw)  # platform names only, no logins
        # platform filter works
        self.assertTrue(all(r["platform"] == "github"
                            for r in v.audit_trail("github")))

    def test_delete(self):
        v = TrialVault(_vault_home(self))
        v.store("github", "u", "p")
        self.assertTrue(v.delete("github"))
        self.assertFalse(v.delete("github"))
        self.assertIsNone(v.get("github"))

    def test_mask(self):
        self.assertEqual("••••", TrialVault.mask("abc"))
        masked = TrialVault.mask("password123")
        self.assertTrue(masked.startswith("p"))
        self.assertIn("(11 chars)", masked)


# ── TrialFlow ──────────────────────────────────────────────────────────────

class TrialFlowP9Tests(unittest.TestCase):
    def _flow(self):
        flow, db, tmp = _flow()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return flow, db, tmp

    def test_save_reports_overwrite(self):
        flow, _, _ = self._flow()
        first = flow.save("github", "octocat", "pw1")
        self.assertFalse(first["overwrote"])
        second = flow.save("github", "octocat", "pw2")
        self.assertTrue(second["overwrote"])

    def test_save_usage_error(self):
        flow, _, _ = self._flow()
        with self.assertRaises(ToolError):
            flow.save("", "u", "p")

    def test_deliver_no_gateway_inline(self):
        flow, _, _ = self._flow()
        flow.save("github", "octocat", "pw123")
        text = flow.deliver("github")
        self.assertIn("no live WhatsApp/Telegram", text)
        self.assertIn("octocat", text)
        self.assertIn("pw123", text)

    def test_deliver_missing_raises(self):
        flow, _, _ = self._flow()
        with self.assertRaises(ToolError):
            flow.deliver("nope")

    def test_assist_refuses_existing_account(self):
        flow, _, _ = self._flow()
        flow.save("github", "octocat", "pw123")
        text = flow.assist("github", auto_yes=True)
        self.assertIn("already have an account", text)
        self.assertIn("one account per service", text)
        self.assertIn("/trial send github", text)

    def test_existing_account_none_when_empty(self):
        flow, _, _ = self._flow()
        self.assertIsNone(flow._existing_account("github"))
        self.assertIsNone(flow._existing_account(""))

    def test_assist_empty_platform_raises(self):
        flow, _, _ = self._flow()
        with self.assertRaises(ToolError):
            flow.assist("")

    def test_launch_assist_vault_locked(self):
        flow, _, _ = self._flow()
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("NM_VAULT_PASSPHRASE", None)
            text = flow._launch_assist("github", chat_key="",
                                       identity={"name": "n", "email": "e"})
        self.assertIn("vault is locked", text)

    def test_launch_assist_refuses_existing_with_passphrase(self):
        flow, _, _ = self._flow()
        flow.save("github", "octocat", "pw123")
        with mock.patch.dict(os.environ, {"NM_VAULT_PASSPHRASE": "t"}):
            text = flow._launch_assist("github", chat_key="",
                                       identity={"name": "n", "email": "e"})
        self.assertIn("already have an account", text)

    def test_temp_number_uses_cascade(self):
        flow, _, _ = self._flow()
        seen = {}

        def fake_cascade(country="us", providers=(), **kw):
            seen["country"] = country
            seen["providers"] = tuple(providers)
            return {"status": "ok", "number": "+15551234567",
                    "masked": "+1555****", "country": "us",
                    "country_name": "United States",
                    "provider": providers[0] if providers else "simcodes",
                    "inbox_id": "x"}

        with mock.patch("nomorals.accounts.temp_sms.grab_number_cascade",
                        fake_cascade):
            text = flow.temp_number("us", provider="7sim")
        self.assertIn("+15551234567", text)
        # preferred provider first, cascade fallbacks kept
        self.assertEqual("7sim", seen["providers"][0])
        self.assertIn("simcodes", seen["providers"])
        self.assertEqual("us", seen["country"])

    def test_temp_number_cascade_failure_honest(self):
        flow, _, _ = self._flow()

        def failed(country="us", providers=(), **kw):
            return {"status": "failed",
                    "notes": "all temp-sms sources exhausted"}

        with mock.patch("nomorals.accounts.temp_sms.grab_number_cascade",
                        failed):
            text = flow.temp_number("us")
        self.assertTrue(text.startswith("❌"))
        self.assertIn("exhausted", text)

    def test_start_stashes_plan_and_url_extraction(self):
        flow, _, _ = self._flow()
        stub = SimpleNamespace(
            run=lambda q, mode="quick", pages=2: {
                "summary": "They ask for email + password. "
                           "Sign up at https://example.com/signup today."})
        with mock.patch.object(TrialFlow, "_search_engine",
                               return_value=stub):
            text = flow.start("example")
        self.assertIn("example", text)
        plan = flow._load_plan("example")
        self.assertIn("signup", plan.get("plan", ""))
        self.assertEqual("https://example.com/signup",
                         flow._plan_signup_url(plan))
        # no signup-looking URL → empty
        self.assertEqual("", flow._plan_signup_url({"plan": "nothing here"}))

    def test_confirm_unknown_token(self):
        flow, _, _ = self._flow()
        text = flow.confirm_signup("nope-not-a-token")
        self.assertIn("unknown or expired", text)
        self.assertIn("usage", flow.confirm_signup(""))

    def test_audit_log_secret_free(self):
        flow, _, tmp = self._flow()
        flow.save("github", "octocat", "s3cr3t-pw-123")
        flow.deliver("github")
        flow.remove("github")
        log = flow.audit_log()
        for action in ("save", "deliver", "delete"):
            self.assertIn(action, log)
        raw = Path(tmp, "trial_audit.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("s3cr3t-pw-123", raw)
        # vault-level audit rows are included
        self.assertIn("vault access", log)

    def test_audit_log_platform_filter(self):
        flow, _, _ = self._flow()
        flow.save("github", "u", "p")
        flow.save("gitlab", "u", "p")
        log = flow.audit_log(platform="github")
        self.assertIn("github", log)
        self.assertNotIn("gitlab", log.split("vault access")[0])

    def test_list_flags_stale(self):
        flow, _, tmp = self._flow()
        flow.save("oldsvc", "u", "p")
        p = Path(tmp) / "trial_accounts.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data["oldsvc"]["saved_at"] = time.time() - 100 * 86400
        p.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("consider rotating", flow.list())

    def test_remove(self):
        flow, _, _ = self._flow()
        flow.save("github", "u", "p")
        self.assertIn("deleted", flow.remove("github"))
        self.assertIn("no stored trial account", flow.remove("github"))


# ── GigApplier ─────────────────────────────────────────────────────────────

class GigApplierP9Tests(unittest.TestCase):
    def setUp(self):
        self._saved_submitters = dict(G.SUBMITTERS)
        self._tmpdirs = []

    def tearDown(self):
        G.SUBMITTERS.clear()
        G.SUBMITTERS.update(self._saved_submitters)
        for tmp in self._tmpdirs:
            shutil.rmtree(tmp, ignore_errors=True)

    def _applier(self, llm=None):
        tmp = tempfile.mkdtemp()
        self._tmpdirs.append(tmp)
        settings = SimpleNamespace(resolve=lambda p: str(Path(tmp) / p))
        return G.GigApplier(
            settings=settings, llm_fn=llm or (lambda p: "DRAFT-TEXT"))

    def test_draft_no_silent_overwrite(self):
        applier = self._applier()
        app = applier.draft(_opp())
        self.assertEqual("DRAFT-TEXT", app.draft_text)
        applier.llm_fn = lambda p: "SECOND-DRAFT"
        again = applier.draft(_opp())
        self.assertEqual("DRAFT-TEXT", again.draft_text)  # untouched
        forced = applier.draft(_opp(), force=True)
        self.assertEqual("SECOND-DRAFT", forced.draft_text)

    def test_revise_second_pass(self):
        applier = self._applier()
        app = applier.draft(_opp())
        applier.llm_fn = lambda p: "REVISED-TEXT"
        app2 = applier.revise(app.gig_id, feedback="make it shorter")
        self.assertEqual("REVISED-TEXT", app2.draft_text)
        self.assertTrue(any(h["kind"] == "revised" for h in app2.history))

    def test_revise_invalid_state(self):
        applier = self._applier()
        G.register_submitter("outlier.ai",
                             lambda a, d: {"ok": True, "evidence": "E"})
        app = applier.draft(_opp())
        applier.submit(app.gig_id, explicit=True)
        with self.assertRaises(ValueError):
            applier.revise(app.gig_id)

    def test_submit_no_submitter_honest(self):
        applier = self._applier()
        app = applier.draft(_opp())
        out = applier.submit(app.gig_id, explicit=True)
        # honest: attempted, NOT "submitted"
        self.assertEqual("submit_attempted", out.status)
        self.assertIsNone(out.submitted_at)
        ev = out.submission_evidence
        self.assertEqual("manual", ev["method"])
        self.assertTrue(any(app.url in step for step in ev["next_steps"]))

    def test_submit_with_board_submitter(self):
        applier = self._applier()
        G.register_submitter(
            "outlier.ai",
            lambda app, draft: {"ok": True, "method": "browser",
                                "evidence": "confirmation id T-1"})
        app = applier.draft(_opp())
        out = applier.submit(app.gig_id, explicit=True)
        self.assertEqual("submitted", out.status)
        self.assertIsNotNone(out.submitted_at)
        self.assertIsNotNone(out.follow_up_at)
        self.assertEqual("confirmation id T-1",
                         out.submission_evidence["note"])

    def test_submit_submitter_needs_human(self):
        applier = self._applier()
        G.register_submitter(
            "outlier.ai",
            lambda app, draft: {"ok": False, "needs_human": True,
                                "detail": "login wall"})
        app = applier.draft(_opp())
        out = applier.submit(app.gig_id, explicit=True)
        self.assertEqual("needs_human", out.status)
        self.assertIn("next_steps", out.submission_evidence)

    def test_submit_submitter_raises_recorded(self):
        applier = self._applier()

        def bad(app, draft):
            raise RuntimeError("board exploded")

        G.register_submitter("outlier.ai", bad)
        app = applier.draft(_opp())
        out = applier.submit(app.gig_id, explicit=True)  # must not raise
        self.assertEqual("submit_attempted", out.status)
        self.assertIn("board exploded", out.submission_evidence["note"])

    def test_submit_already_submitted_no_double(self):
        applier = self._applier()
        calls = []

        def sub(app, draft):
            calls.append(app.gig_id)
            return {"ok": True, "method": "t", "evidence": "E"}

        G.register_submitter("outlier.ai", sub)
        app = applier.draft(_opp())
        applier.submit(app.gig_id, explicit=True)
        applier.submit(app.gig_id, explicit=True)
        self.assertEqual(1, len(calls))

    def test_submit_invalid_state(self):
        applier = self._applier()
        app = applier.draft(_opp())
        applier.withdraw(app.gig_id)
        with self.assertRaises(ValueError):
            applier.submit(app.gig_id, explicit=True)

    def test_submit_missing_raises(self):
        applier = self._applier()
        with self.assertRaises(KeyError):
            applier.submit("nonexistent", explicit=True)

    def test_withdraw_terminal(self):
        applier = self._applier()
        app = applier.draft(_opp())
        out = applier.withdraw(app.gig_id, notes="changed my mind")
        self.assertEqual("withdrawn", out.status)
        self.assertEqual("changed my mind", out.notes)
        with self.assertRaises(ValueError):
            out.transition("drafted")

    def test_invalid_transition_blocked(self):
        applier = self._applier()
        app = applier.draft(_opp())
        with self.assertRaises(ValueError):
            applier.set_status(app.gig_id, "accepted")
        # valid outcome jump works after a real submit
        G.register_submitter("outlier.ai",
                             lambda a, d: {"ok": True, "evidence": "E"})
        applier.submit(app.gig_id, explicit=True)
        applier.set_status(app.gig_id, "interview", notes="call booked")
        got = applier.store.get(app.gig_id)
        self.assertEqual("interview", got.status)

    def test_follow_ups_due_and_snooze(self):
        applier = self._applier()
        G.register_submitter("outlier.ai",
                             lambda a, d: {"ok": True, "evidence": "E"})
        app = applier.draft(_opp())
        applier.submit(app.gig_id, explicit=True)
        self.assertEqual([], applier.follow_ups_due())  # 7d out, not due
        stored = applier.store.get(app.gig_id)
        stored.follow_up_at = time.time() - 10
        applier.store.save(stored)
        due = applier.follow_ups_due()
        self.assertEqual([app.gig_id], [a.gig_id for a in due])
        applier.snooze_follow_up(app.gig_id, days=7)
        self.assertEqual([], applier.follow_ups_due())

    def test_follow_up_draft_template_fallback(self):
        applier = self._applier()  # working LLM for draft + submit
        G.register_submitter("outlier.ai",
                             lambda a, d: {"ok": True, "evidence": "E"})
        app = applier.draft(_opp())
        applier.submit(app.gig_id, explicit=True)
        # ...then the LLM disappears: template fallback must still work.
        applier.llm_fn = lambda prompt: (_ for _ in ()).throw(
            RuntimeError("no LLM here"))
        text = applier.follow_up_draft(app.gig_id)
        self.assertIn("Following up", text)
        self.assertIn(app.title, text)

    def test_follow_up_draft_missing_raises(self):
        applier = self._applier()
        with self.assertRaises(KeyError):
            applier.follow_up_draft("nonexistent")

    def test_register_submitter_validation(self):
        with self.assertRaises(ValueError):
            G.register_submitter("", lambda a, d: {})
        with self.assertRaises(ValueError):
            G.register_submitter("x", "not-callable")

    def test_submitter_matched_by_kind(self):
        applier = self._applier()
        G.register_submitter("paid_task",
                             lambda a, d: {"ok": True, "evidence": "E"})
        app = applier.draft(_opp(source="some-new-board"))
        out = applier.submit(app.gig_id, explicit=True)
        self.assertEqual("submitted", out.status)

    def test_application_from_dict_legacy(self):
        app = G.Application.from_dict({"gig_id": "abc", "title": "t",
                                       "url": "http://x",
                                       "status": "submitted"})
        self.assertEqual("submitted", app.status)
        self.assertEqual({}, app.submission_evidence)
        self.assertEqual([], app.history)
        self.assertIsNone(app.follow_up_at)

    def test_history_trail(self):
        applier = self._applier()
        app = applier.draft(_opp())
        applier.review(app.gig_id)
        got = applier.store.get(app.gig_id)
        kinds = [h["kind"] for h in got.history]
        self.assertIn("drafted", kinds)
        self.assertIn("status", kinds)

    def test_explicit_submit_words(self):
        for text in ("submit", "submit it", "SUBMIT", "  apply now  ",
                     "send it", "go ahead do it"):
            self.assertTrue(G.is_explicit_submit(text), text)
        self.assertFalse(G.is_explicit_submit(""))
        self.assertFalse(G.is_explicit_submit("maybe later"))


if __name__ == "__main__":
    unittest.main()
