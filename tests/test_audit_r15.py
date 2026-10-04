"""R15 audit tests.

- scheduler ``message`` jobs deliver the literal text to the owner's DM
  (no more "sent:" summaries that reach no chat surface);
- spotify ``_force_refresh_access`` fails with a clear error instead of
  hanging on an interactive prompt in headless contexts;
- /trial temp-SMS wiring (grab/stash/poll) and the new chat/CLI verbs;
- BookForge offline title heuristic no longer emits "To X" titles.
"""

from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.scheduler import Scheduler
from nomorals.storage.db import Database


def _ctx():
    return build_context(db=Database(":memory:"), with_tools=True,
                         with_router=False, with_memory=False)


class _FakeSendResult:
    ok = True


class _FakeGateway:
    """Minimal chat gateway: telegram live, records sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def status(self):
        return {"telegram": {"running_in_session": True}}

    def send(self, platform, chat, text):
        self.sent.append((platform, getattr(chat, "chat_id", "?"), text))
        return _FakeSendResult()


class SchedulerMessageTests(unittest.TestCase):
    def _sched(self):
        ctx = _ctx()
        ctx.settings.partner.owner_chats = "telegram:123"
        gw = _FakeGateway()
        return Scheduler(ctx, gateway=gw), gw, ctx

    def test_message_job_delivers_literal_text(self):
        sched, gw, _ctx = self._sched()
        summary = sched._run_message({"text": "take your meds"})
        self.assertIn("take your meds", summary)
        self.assertEqual(len(gw.sent), 1)
        plat, chat_id, text = gw.sent[0]
        self.assertEqual(plat, "telegram")
        self.assertEqual(chat_id, "123")
        # the literal text reaches the DM — not a "job ran" wrapper
        self.assertIn("take your meds", text)

    def test_message_job_no_gateway_stores_pending(self):
        ctx = _ctx()
        sched = Scheduler(ctx)  # no gateway anywhere
        summary = sched._run_message({"text": "hello offline"})
        self.assertIn("hello offline", summary)
        # durable: the notification row is there for redelivery
        rows = sched.notifier.recent(kind="message")
        self.assertTrue(any("hello offline" in (r.get("title") or "")
                            for r in rows))

    def test_message_job_empty_still_raises(self):
        sched, _gw, _ctx = self._sched()
        with self.assertRaises(ValueError):
            sched._run_message({"text": "   "})

    def test_message_job_suppresses_double_alert(self):
        """The meta "job ran" alert must not double-send a message job."""
        sched, gw, _ctx = self._sched()
        job = sched.add("reminder-x", "every 30m", "message",
                        {"text": "drink water"})
        out = sched.run_now(job["id"])
        self.assertTrue(out["ok"])
        # exactly one send: the message itself.  The "✅ scheduled:" meta
        # alert is suppressed for message payloads.
        self.assertEqual(len(gw.sent), 1)
        self.assertIn("drink water", gw.sent[0][2])
        self.assertNotIn("scheduled", gw.sent[0][2].lower())

    def test_tool_job_still_alerts(self):
        """Non-message jobs keep the job-summary notification."""
        sched, gw, _ctx = self._sched()
        job = sched.add("tool-x", "every 30m", "tool",
                        {"tool": "no_such_tool_xyz"})
        out = sched.run_now(job["id"])
        self.assertFalse(out["ok"])  # unknown tool -> failure
        # failure alerts the owner through the notifier
        rows = sched.notifier.recent(kind="schedule")
        self.assertTrue(any("tool-x" in (r.get("title") or "") for r in rows))


class SpotifyRefreshTests(unittest.TestCase):
    def _connected(self):
        import json
        import urllib.parse
        from unittest import mock as _mock

        from nomorals.accounts.vault import CredentialVault
        from nomorals.connectors.spotify import SpotifyConnector

        class FakeResponse:
            def __init__(self, status=200, payload=None):
                self.status = status
                self._payload = payload

            @property
            def ok(self):
                return 200 <= self.status < 300

            @property
            def text(self):
                return json.dumps(self._payload)

            def json(self):
                return self._payload

        class FakeHttp:
            def __init__(self):
                self.routes = []

            def route(self, method, path, resp):
                self.routes.append((method.upper(), path, resp))

            def _dispatch(self, method, url, payload=None, **kw):
                for rm, rp, resp in self.routes:
                    if rm == method.upper() and rp in url:
                        return resp
                return FakeResponse(404, {})

            def get(self, url, **kw):
                return self._dispatch("GET", url)

            def post_form(self, url, form, **kw):
                return self._dispatch("POST", url, form, **kw)

            def post_json(self, url, payload, **kw):
                return self._dispatch("POST", url, payload)

            def put_json(self, url, payload, **kw):
                return self._dispatch("PUT", url, payload)

        http = FakeHttp()
        vault = CredentialVault(Database(":memory:"),
                               master_passphrase="test")
        conn = SpotifyConnector(vault, http=http)
        tokens = {"access_token": "BQD.a", "refresh_token": "AQDrefresh",
                  "expires_in": 3600, "token_type": "Bearer"}
        me = {"id": "user1", "display_name": "Devon",
              "email": "devon@example.com"}
        http.route("POST", "accounts.spotify.com/api/token",
                   FakeResponse(200, tokens))
        http.route("GET", "/v1/me", FakeResponse(200, me))
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_ID": "cid",
                                          "SPOTIFY_CLIENT_SECRET": "csecret"}):
            result = conn.connect(
                redirect_url="http://127.0.0.1:8888/callback?code=authcode")
        assert result.ok
        return conn, http

    def _expire(self, conn):
        cred = conn._load_credential()
        meta = dict(cred.metadata or {})
        meta["access_expires_at"] = time.time() - 10
        conn._store_credential(
            cred.username, cred.password,
            credential_type=cred.credential_type,
            scopes=list(meta.get("scopes", [])),
            metadata=meta,
        )

    def test_refresh_without_secret_fails_clear_not_hangs(self):
        from nomorals.connectors.base import ConnectorError

        conn, _http = self._connected()
        self._expire(conn)
        env = {k: v for k, v in os.environ.items()
               if k != "SPOTIFY_CLIENT_SECRET"}
        with mock.patch.dict(os.environ, env, clear=True):
            # even if stdin looked like a TTY, no prompt may happen —
            # patch it to explode if anything tries.
            with mock.patch(
                    "nomorals.connectors.spotify.prompt_secret",
                    side_effect=AssertionError("must not prompt")):
                with self.assertRaises(ConnectorError) as ctx:
                    conn._force_refresh_access()
        msg = str(ctx.exception)
        self.assertIn("SPOTIFY_CLIENT_SECRET", msg)
        self.assertIn("unattended", msg)

    def test_refresh_with_env_secret_proceeds(self):
        conn, http = self._connected()
        self._expire(conn)
        new_tokens = {"access_token": "BQD.b", "refresh_token": "AQDrefresh2",
                      "expires_in": 3600, "token_type": "Bearer"}

        class FakeResponse:
            status = 200

            @property
            def ok(self):
                return True

            @property
            def text(self):
                return "{}"

            def json(self):
                return new_tokens

        http.routes.clear()  # drop the connect-time token route
        http.route("POST", "accounts.spotify.com/api/token", FakeResponse())
        with mock.patch.dict(os.environ, {"SPOTIFY_CLIENT_SECRET": "csecret"}):
            token = conn._force_refresh_access()
        self.assertEqual(token, "BQD.b")


class TrialTempSmsTests(unittest.TestCase):
    def _flow(self):
        from nomorals.agents.trial.flow import TrialFlow

        return TrialFlow(_ctx())

    def test_temp_number_stash_and_poll(self):
        from nomorals.agents.trial.flow import TEMP_SMS_KV_KEY

        flow = self._flow()
        info = {"status": "ok", "number": "+15550001111",
                "masked": "+1555****111", "country": "us",
                "country_name": "United States", "provider": "simcodes",
                "inbox_id": "abc"}
        with mock.patch("nomorals.accounts.temp_sms.grab_number",
                        return_value=info):
            out = flow.temp_number("us")
        self.assertIn("+15550001111", out)
        self.assertIn("/trial sms code", out)
        # stashed for the code poll
        self.assertEqual(flow._load_temp_number()["number"], "+15550001111")
        row = flow.db.query_one("SELECT value FROM kv_store WHERE key = ?",
                                (TEMP_SMS_KV_KEY,))
        self.assertIsNotNone(row)
        # code poll uses the stash, no JSON pasting needed
        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        return_value="482910") as wc:
            out = flow.temp_sms_code(timeout=5)
        self.assertIn("482910", out)
        self.assertEqual(wc.call_args[0][0]["number"], "+15550001111")

    def test_temp_sms_code_without_stash(self):
        flow = self._flow()
        self.assertIn("/trial sms", flow.temp_sms_code(timeout=1))

    def test_assist_needs_identity(self):
        flow = self._flow()
        out = flow.assist("github")
        self.assertIn("/identity set", out)

    def test_assist_needs_vault_passphrase(self):
        import json as _json

        from nomorals.accounts.creator import AccountCreator
        from nomorals.agents.trial.flow import TrialFlow

        flow = self._flow()
        flow.db.execute(
            "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at)"
            " VALUES (?, ?, 'json', ?)",
            (AccountCreator.IDENTITY_KV_KEY,
             _json.dumps({"name": "Devon", "email": "devon@example.com"}),
             time.time()),
        )
        env = {k: v for k, v in os.environ.items()
               if k != "NM_VAULT_PASSPHRASE"}
        with mock.patch.dict(os.environ, env, clear=True):
            out = flow.assist("github")
        self.assertIn("NM_VAULT_PASSPHRASE", out)

    def test_accounts_exports(self):
        import nomorals.accounts as accounts

        self.assertTrue(callable(accounts.grab_temp_number))
        self.assertTrue(callable(accounts.wait_temp_sms_code))

    def test_trial_chat_verbs(self):
        # /trial sms / sms code / status / inbox route through the flow
        from nomorals.agents.trial.flow import TrialFlow

        seen = {}

        class FakeFlow(TrialFlow):
            def temp_number(self, country="us", provider="simcodes"):
                seen["sms"] = country
                return "number!"

            def temp_sms_code_async(self, chat_key="", timeout=180):
                seen["sms-code"] = chat_key
                return "watching!"

            def assist_status(self):
                seen["status"] = True
                return "status!"

            def disposable_inbox(self, service, limit=10):
                seen["inbox"] = service
                return "inbox!"

        import nomorals.agents.partner.runtime_games as rg
        import nomorals.agents.trial as trial_mod

        mixin = rg.RuntimeGamesMixin.__new__(rg.RuntimeGamesMixin)
        mixin.context = _ctx()
        mixin.gateway = None
        with mock.patch.object(trial_mod, "TrialFlow", FakeFlow):
            self.assertEqual(mixin._control_trial("sms ng", "k"), "number!")
            self.assertEqual(seen["sms"], "ng")
            self.assertEqual(mixin._control_trial("sms code", "k2"),
                             "watching!")
            self.assertEqual(seen["sms-code"], "k2")
            self.assertEqual(mixin._control_trial("status", "k"), "status!")
            self.assertEqual(mixin._control_trial("inbox email_mailtm", "k"),
                             "inbox!")
            self.assertEqual(seen["inbox"], "email_mailtm")


class BookTitleTests(unittest.TestCase):
    def _title(self, topic):
        from nomorals.books.forge import clean_title

        ctx = SimpleNamespace(router=None, settings=None, db=None)
        return clean_title(topic, ctx)

    def test_guide_to_topic(self):
        self.assertEqual(self._title("a guide to urban gardening"),
                         "Urban Gardening")

    def test_no_dangling_to(self):
        title = self._title("write me a guide to sourdough baking")
        self.assertFalse(title.lower().startswith("to "))
        self.assertIn("Sourdough", title)

    def test_normal_titles_untouched(self):
        self.assertEqual(self._title("a novel about a voyage to mars"),
                         "A Voyage to Mars")


if __name__ == "__main__":
    unittest.main()
