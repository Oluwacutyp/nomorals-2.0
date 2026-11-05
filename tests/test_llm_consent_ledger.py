"""Consent ledger (Echelon Wave 4) — core ledger + llm binding.

Plan acceptance (INTEGRATION_PLAN Wave 4): a call that requires
consent and lacks it is denied with ``policy.denied``; a granted call
is allowed and the grant is recorded with who/when and an expiry.
The ledger is data: grants, revocations and checks are inspectable,
and the file-backed form round-trips through JSON.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from nomorals.core.consent import DATA_KINDS, ConsentLedger
from nomorals.core.errors import CapabilityDenied
from nomorals.llm.consent import (
    guarded_chat,
    provider_consent_decision,
    require_provider_consent,
)
from nomorals.llm.base import Message
from nomorals.llm.providers import build_provider


class _Clock:
    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _tmp_path() -> Path:
    base = os.environ.get("NM_TEST_TMPDIR") or None
    return Path(tempfile.mkdtemp(dir=base)) / "consent.json"


class LedgerDecisions(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.ledger = ConsentLedger(clock=self.clock)

    def test_no_grant_denies_with_policy_denied(self):
        decision = self.ledger.check(destination="groq",
                                     data_kind="chat_history")
        self.assertFalse(decision.allowed)
        with self.assertRaises(CapabilityDenied) as ctx:
            self.ledger.require(destination="groq",
                                data_kind="chat_history")
        self.assertEqual(ctx.exception.code, "policy.denied")
        self.assertEqual(ctx.exception.details["destination"], "groq")
        self.assertEqual(ctx.exception.details["data_kind"], "chat_history")
        # The decision object itself carries the same denial.
        with self.assertRaises(CapabilityDenied) as ctx2:
            decision.raise_if_denied()
        self.assertEqual(ctx2.exception.code, "policy.denied")

    def test_grant_allows_and_is_recorded(self):
        record = self.ledger.grant(
            destination="groq", data_kinds=["chat_history"],
            granted_by="owner", note="telegram DM, 2026-11-05")
        self.assertTrue(record.id)
        self.assertEqual(record.granted_by, "owner")
        self.assertEqual(record.granted_at, self.clock.now)
        self.assertIsNone(record.expires_at)
        decision = self.ledger.check(destination="groq",
                                     data_kind="chat_history")
        self.assertTrue(decision.allowed)
        found = self.ledger.require(destination="groq",
                                    data_kind="chat_history")
        self.assertEqual(found.id, record.id)
        self.assertEqual([r.id for r in self.ledger.records()],
                         [record.id])

    def test_matching_is_exact_on_every_key_field(self):
        self.ledger.grant(destination="groq", data_kinds=["chat_history"],
                          granted_by="owner")
        denied = [
            dict(destination="openrouter", data_kind="chat_history"),
            dict(destination="groq", data_kind="documents"),
            dict(destination="groq", data_kind="chat_history",
                 app="other-app"),
            dict(destination="groq", data_kind="chat_history",
                 connection="second"),
            dict(destination="groq", data_kind="chat_history",
                 capability="llm.embed"),
        ]
        for kwargs in denied:
            with self.subTest(**kwargs):
                self.assertFalse(self.ledger.check(**kwargs).allowed)

    def test_wildcard_kind_covers_all_kinds(self):
        self.ledger.grant(destination="groq", data_kinds=["*"],
                          granted_by="owner")
        for kind in DATA_KINDS:
            self.assertTrue(self.ledger.check(
                destination="groq", data_kind=kind).allowed)

    def test_expiry_closes_the_grant(self):
        self.ledger.grant(destination="groq", data_kinds=["prompts"],
                          granted_by="owner", ttl_s=60)
        self.assertTrue(self.ledger.check(
            destination="groq", data_kind="prompts").allowed)
        self.clock.now += 61
        self.assertFalse(self.ledger.check(
            destination="groq", data_kind="prompts").allowed)
        self.assertEqual(self.ledger.records(active_only=True), [])
        self.assertEqual(len(self.ledger.records()), 1)

    def test_revoke_closes_the_grant(self):
        record = self.ledger.grant(destination="groq",
                                   data_kinds=["prompts"],
                                   granted_by="owner")
        self.assertTrue(self.ledger.revoke(record.id, revoked_by="owner"))
        self.assertFalse(self.ledger.check(
            destination="groq", data_kind="prompts").allowed)
        stored = self.ledger.records()[0]
        self.assertEqual(stored.revoked_by, "owner")
        self.assertFalse(self.ledger.revoke("missing-id"))

    def test_grant_validation(self):
        with self.assertRaises(ValueError):
            self.ledger.grant(destination="", data_kinds=["prompts"],
                              granted_by="owner")
        with self.assertRaises(ValueError):
            self.ledger.grant(destination="groq", data_kinds=[],
                              granted_by="owner")
        with self.assertRaises(ValueError):
            self.ledger.grant(destination="groq", data_kinds=["prompts"],
                              granted_by="")


class LedgerPersistence(unittest.TestCase):
    def test_json_round_trip(self):
        path = _tmp_path()
        clock = _Clock()
        first = ConsentLedger(path, clock=clock)
        record = first.grant(destination="openrouter",
                             data_kinds=["chat_history", "images"],
                             granted_by="owner", ttl_s=3600,
                             note="pilot")
        second = ConsentLedger(path, clock=clock)
        self.assertEqual([r.id for r in second.records()], [record.id])
        self.assertTrue(second.check(destination="openrouter",
                                     data_kind="images").allowed)
        loaded = second.records()[0]
        self.assertEqual(loaded.expires_at, record.granted_at + 3600)
        self.assertEqual(loaded.note, "pilot")
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["records"][0]["destination"], "openrouter")

    def test_corrupt_file_is_an_error_not_an_empty_ledger(self):
        path = _tmp_path()
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            ConsentLedger(path)


class ProviderBinding(unittest.TestCase):
    def setUp(self):
        self.ledger = ConsentLedger(clock=_Clock())

    def test_local_providers_need_no_grant(self):
        for kind in ("mock", "ollama", "llama_cpp", "ocr"):
            with self.subTest(provider=kind):
                decision = provider_consent_decision(
                    self.ledger, kind, data_kind="personal_data")
                self.assertTrue(decision.allowed)
                self.assertIn("never leaves", decision.reason)

    def test_hosted_provider_denied_until_granted(self):
        provider = build_provider("groq", api_key="fake")
        decision = provider_consent_decision(
            self.ledger, provider, data_kind="chat_history")
        self.assertFalse(decision.allowed)
        with self.assertRaises(CapabilityDenied) as ctx:
            require_provider_consent(self.ledger, provider,
                                     data_kind="chat_history")
        self.assertEqual(ctx.exception.code, "policy.denied")
        self.ledger.grant(destination="groq", data_kinds=["chat_history"],
                          granted_by="owner")
        self.assertTrue(provider_consent_decision(
            self.ledger, provider, data_kind="chat_history").allowed)

    def test_unknown_provider_is_denied(self):
        decision = provider_consent_decision(
            self.ledger, "not-a-provider", data_kind="prompts")
        self.assertFalse(decision.allowed)

    def test_guarded_chat_local_runs_without_a_grant(self):
        provider = build_provider("mock")
        resp = guarded_chat(self.ledger, provider,
                            [Message.user("hello")])
        self.assertTrue(resp.ok)

    def test_guarded_chat_hosted_denied_before_any_call(self):
        provider = build_provider("groq", api_key="fake")
        calls = []
        provider.http = type("H", (), {  # noqa: SLF001 - test double
            "post_json": lambda self, url, payload, **kw: calls.append(1),
        })()
        with self.assertRaises(CapabilityDenied) as ctx:
            guarded_chat(self.ledger, provider, [Message.user("hello")])
        self.assertEqual(ctx.exception.code, "policy.denied")
        self.assertEqual(calls, [], "denied call reached the provider")
        self.ledger.grant(destination="groq", data_kinds=["chat_history"],
                          granted_by="owner")

        class _Ok:
            def __init__(self):
                self.calls = 0

            def post_json(self, url, payload, **kw):
                self.calls += 1

                class R:
                    @staticmethod
                    def json():
                        return {
                            "model": payload["model"],
                            "choices": [{"message": {
                                "role": "assistant", "content": "hi"},
                                "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1,
                                      "completion_tokens": 1,
                                      "total_tokens": 2},
                        }
                return R()

        ok_http = _Ok()
        provider.http = ok_http
        resp = guarded_chat(self.ledger, provider, [Message.user("hello")])
        self.assertTrue(resp.ok)
        self.assertGreaterEqual(ok_http.calls, 1)


if __name__ == "__main__":
    unittest.main()
