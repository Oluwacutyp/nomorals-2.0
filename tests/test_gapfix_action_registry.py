"""Gapfix R3-61: the autonomous-action record book gets written.

The extended ledger (`ExtendedRepresentationLedger`) is live in
production — `_ledger_for` returns it and `/review` renders its
stakes/counterparty sections — but the one-line hook every production
site uses (`log_representation_action`) recorded through the BASE
`log_action`, so no record ever carried `action_class` (for the
sites that don't set it themselves), `stakes` or `counterparty`.
These tests pin the wiring: the hook routes through
`record_autonomous_action`, the sites that know their counterparty
pass it, and the hook never raises.

All offline — scripted HTTP, tmp ledger DB, patched `_ledger_for`.
"""

from __future__ import annotations

import urllib.parse
from types import SimpleNamespace
from typing import Any
from unittest import mock

import nomorals.cognition.representation as rep
from nomorals.accounts.vault import CredentialVault
from nomorals.cognition.ledger_ext import ExtendedRepresentationLedger
from nomorals.cognition.representation import (
    RepresentationLedger,
    _ledger_for,
    log_representation_action,
)
from nomorals.connectors.gmail import GmailConnector
from nomorals.connectors.stripe import StripeConnector
from nomorals.connectors.twilio import TwilioConnector
from nomorals.storage.db import Database


def _settings(tmp_path):
    return SimpleNamespace(home_path=tmp_path)


def _ext_ledger(tmp_path):
    return ExtendedRepresentationLedger(
        RepresentationLedger(tmp_path / "representation.db"))


# ── seam: the hook records through the extension ─────────────────────


def test_hook_stores_extended_class_for_plain_type(tmp_path) -> None:
    """A site that never sets action_class still lands enriched."""
    settings = _settings(tmp_path)
    action_id = log_representation_action(
        "apply", "gig application submitted: g1",
        settings=settings, outcome="success")
    assert action_id
    rec = _ledger_for(settings).get(action_id)
    assert rec is not None
    assert rec.action_type == "apply"
    assert rec.outcome == "success"
    assert rec.metadata["action_class"] == "apply"
    assert rec.metadata["stakes"] == "medium"
    assert rec.metadata["stakes_weight"] == 0.5
    assert rec.metadata["counterparty"] == ""


def test_hook_preserves_site_action_class(tmp_path) -> None:
    """A site's own extended class survives; base type stays put."""
    settings = _settings(tmp_path)
    action_id = log_representation_action(
        "purchase", "Stripe transfer 100.00 USD",
        settings=settings, outcome="success",
        metadata={"action_class": "transfer", "destination": "acct_dest"})
    assert action_id
    rec = _ledger_for(settings).get(action_id)
    assert rec is not None
    assert rec.action_type == "purchase"
    assert rec.metadata["action_class"] == "transfer"
    assert rec.metadata["destination"] == "acct_dest"
    assert rec.metadata["stakes"] == "medium"


def test_hook_failed_action_recorded_as_failed(tmp_path) -> None:
    """A failed action is recorded failed, never as success."""
    settings = _settings(tmp_path)
    action_id = log_representation_action(
        "post", "published to no platforms: hello",
        settings=settings, outcome="failed")
    assert action_id
    rec = _ledger_for(settings).get(action_id)
    assert rec is not None
    assert rec.outcome == "failed"
    assert rec.description == "published to no platforms: hello"
    assert rec.metadata["action_class"] == "post"


def test_hook_result_is_looked_up_by_action_id(tmp_path) -> None:
    """The id the hook returns resolves to the same record."""
    settings = _settings(tmp_path)
    action_id = log_representation_action(
        "message", "SMS sent to +2348012345678",
        settings=settings, outcome="success",
        metadata={"counterparty": "+2348012345678"})
    assert action_id
    ledger = _ledger_for(settings)
    rec = ledger.get(action_id)
    assert rec is not None
    assert rec.id == action_id
    assert rec.outcome == "success"
    assert rec.metadata["action_class"] == "message"
    assert rec.metadata["counterparty"] == "+2348012345678"


def test_counterparty_memory_populates(tmp_path) -> None:
    """Counterparty memory reads what the hook recorded."""
    settings = _settings(tmp_path)
    log_representation_action(
        "message", "email sent to a@b.c",
        settings=settings, outcome="success",
        metadata={"action_class": "send", "counterparty": "a@b.c"})
    ledger = _ledger_for(settings)
    assert "a@b.c" in ledger.counterparties()
    summary = ledger.counterparty_summary("a@b.c")
    assert summary["actions"] == 1
    assert summary["counterparty"] == "a@b.c"


def test_review_summary_shows_stakes_section(tmp_path) -> None:
    """After a known-outcome hook log, /review's stakes section names it."""
    settings = _settings(tmp_path)
    log_representation_action(
        "apply", "gig application submitted: g1",
        settings=settings, outcome="success")
    text = _ledger_for(settings).summary(limit=5, since_days=30)
    assert "Stakes-weighted quality" in text
    assert "• apply:" in text
    assert "No known-outcome actions" not in text


def test_hook_never_raises(tmp_path) -> None:
    """Broken settings, unknown types and bad stakes never break the hook."""
    class BadSettings:
        @property
        def home_path(self):
            raise RuntimeError("boom")

    assert log_representation_action("post", "x",
                                     settings=BadSettings()) == ""
    # Unknown action type: nothing is recorded, nothing raised.
    assert log_representation_action("teleport", "x",
                                     settings=_settings(tmp_path)) == ""
    # Bad stakes: the base log keeps the record with the value verbatim.
    settings = _settings(tmp_path)
    action_id = log_representation_action(
        "post", "x", settings=settings, outcome="success",
        metadata={"stakes": "cosmic"})
    assert action_id
    rec = _ledger_for(settings).get(action_id)
    assert rec is not None
    assert rec.metadata["stakes"] == "cosmic"


# ── fakes (same shape as test_representation_coverage) ───────────────


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def text(self) -> str:
        return str(self._payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON here")
        return self._payload


class FakeHttp:
    """Scripted stand-in for HttpClient. No network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.routes: list[tuple[str, str, FakeResponse]] = []

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.routes.append((method.upper(), path, response))

    def _dispatch(self, method: str, url: str,
                  payload: Any = None, **kw: Any) -> FakeResponse:
        self.calls.append((method.upper(), url, payload, kw.get("headers")))
        for rm, rp, resp in sorted(self.routes, key=lambda r: -len(r[1])):
            if rm == method.upper() and rp in url:
                return resp
        return FakeResponse(404, {"error": {"message": "not mocked"}})

    def get(self, url: str, **kw: Any) -> FakeResponse:
        params = kw.pop("params", None) or {}
        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urllib.parse.urlencode(params)}"
        return self._dispatch("GET", url, None, **kw)

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, payload, **kw)

    def post_form(self, url: str, form: Any, **kw: Any) -> FakeResponse:
        return self._dispatch("POST", url, form, **kw)

    def request(self, method: str, url: str, **kw: Any) -> FakeResponse:
        return self._dispatch(method, url, None, **kw)


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


TOKENS = {
    "access_token": "ya29.acc",
    "refresh_token": "1//ref",
    "expires_in": 3600,
    "token_type": "Bearer",
}
PROFILE = {"emailAddress": "owner@example.com", "messagesTotal": 42}
TWILIO_ACCOUNT = {"sid": "AC123", "friendly_name": "Devon Twilio",
                  "status": "active"}


def _gmail_connected() -> tuple[GmailConnector, FakeHttp]:
    http = FakeHttp()
    conn = GmailConnector(_vault(), http=http)
    http.route("POST", "oauth2.googleapis.com/token",
               FakeResponse(200, dict(TOKENS)))
    http.route("GET", "/users/me/profile", FakeResponse(200, dict(PROFILE)))
    result = conn.connect(
        client_id="cid123", client_secret="<redacted>", code="auth-code-1"
    )
    assert result.ok
    return conn, http


def _stripe_connected() -> tuple[StripeConnector, FakeHttp]:
    http = FakeHttp()
    conn = StripeConnector(_vault(), http=http)
    http.route("GET", "/v1/account",
               FakeResponse(200, {"id": "acct_1", "email": "o@x.com",
                                  "country": "US"}))
    result = conn.connect(secret_key="sk_test_abc")
    assert result.ok
    return conn, http


def _twilio_connected() -> tuple[TwilioConnector, FakeHttp]:
    http = FakeHttp()
    conn = TwilioConnector(_vault(), http=http)
    http.route("GET", "/2010-04-01/Accounts/AC123.json",
               FakeResponse(200, dict(TWILIO_ACCOUNT)))
    result = conn.connect(account_sid="AC123", auth_token="<redacted>")
    assert result.ok
    return conn, http


# ── sites: the counterparty they already hold lands in the record ────


def test_twilio_sms_records_counterparty(tmp_path) -> None:
    conn, http = _twilio_connected()
    http.route("POST", "/2010-04-01/Accounts/AC123/Messages.json",
               FakeResponse(201, {"sid": "SM1"}))
    ledger = _ext_ledger(tmp_path)
    with mock.patch.object(rep, "_ledger_for", lambda settings=None: ledger):
        out = conn.send_sms("+2348012345678", "hi",
                            from_number="+15550001111", confirmed=True)
    assert out.get("sid") == "SM1"
    rows = ledger.recent(action_type="message", limit=5)
    assert len(rows) == 1
    assert rows[0].metadata["action_class"] == "message"
    assert rows[0].metadata["counterparty"] == "+2348012345678"
    assert rows[0].metadata["stakes"] == "medium"


def test_gmail_send_records_counterparty(tmp_path) -> None:
    conn, http = _gmail_connected()
    http.route("POST", "/users/me/messages/send", FakeResponse(200, {
        "id": "sent1", "threadId": "t9", "labelIds": ["SENT"],
    }))
    ledger = _ext_ledger(tmp_path)
    with mock.patch.object(rep, "_ledger_for", lambda settings=None: ledger):
        result = conn.send_message("a@b.c", "Hello", "body text",
                                   confirmed=True)
    assert result["id"] == "sent1"
    rows = ledger.recent(action_type="message", limit=5)
    assert len(rows) == 1
    assert rows[0].metadata["action_class"] == "send"
    assert rows[0].metadata["counterparty"] == "a@b.c"


def test_stripe_transfer_records_counterparty(tmp_path) -> None:
    conn, http = _stripe_connected()
    http.route("POST", "/v1/transfers", FakeResponse(200, {"id": "tr_1"}))
    ledger = _ext_ledger(tmp_path)
    with mock.patch.object(rep, "_ledger_for", lambda settings=None: ledger):
        out = conn.create_transfer(10000, "usd", "acct_dest",
                                   confirmed=True)
    assert out["id"] == "tr_1"
    rows = ledger.recent(action_type="purchase", limit=5)
    assert len(rows) == 1
    assert rows[0].metadata["action_class"] == "transfer"
    assert rows[0].metadata["counterparty"] == "acct_dest"
    assert rows[0].metadata["destination"] == "acct_dest"
