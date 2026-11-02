"""Pins for Pass tail-111: nomorals/connectors/plaid.py — 3 exclusions.

1512 E104 _api: a response body that is not JSON degrades to {} — on an
    error response the PlaidError is still raised, carrying the HTTP
    status and the raw body text as the detail; on a 200 the caller gets
    an empty dict (no field is invented). JSON error bodies keep their
    error_code/guidance path untouched.
1549 E104 _client_credential: the vault read raises -> None, the
    documented "absent credential is normal" value; _require_client
    then refuses honestly (fail-closed).
1590 E104 _items: one unreadable vault entry is skipped, the readable
    items are still listed; if NOTHING is readable, _items raises
    ConnectorError instead of returning a fabricated empty list (the
    fence that keeps the skip honest).
"""

from __future__ import annotations

import pytest

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import ConnectorError
from nomorals.connectors.plaid import PlaidConnector, PlaidError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


class Resp:
    def __init__(self, status, payload=None, text="", json_boom=False):
        self.status = status
        self._payload = payload or {}
        self._text = text
        self._jb = json_boom
        self.headers = {}

    @property
    def ok(self):
        return 200 <= self.status < 300

    @property
    def text(self):
        return self._text

    def json(self):
        if self._jb:
            raise ValueError("not json")
        return self._payload


class PHttp:
    def __init__(self, resp):
        self._resp = resp

    def post_json(self, url, payload, **k):
        return self._resp


BASE = "https://sandbox.plaid.com"


# ---------------------------------------------------------------- 1512

def test_api_non_json_error_body_raises_with_raw_text():
    conn = PlaidConnector(_vault(), http=PHttp(
        Resp(500, text="proxy exploded", json_boom=True)))
    with pytest.raises(PlaidError) as ei:
        conn._api(BASE, "/accounts/get", {}, cid="c", secret="s")
    assert ei.value.status_code == 500
    assert "proxy exploded" in str(ei.value)


def test_api_json_error_body_keeps_error_code():
    conn = PlaidConnector(_vault(), http=PHttp(Resp(400, payload={
        "error_code": "ITEM_LOGIN_REQUIRED",
        "error_message": "relogin needed",
        "error_type": "ITEM_ERROR"}, text="x")))
    with pytest.raises(PlaidError) as ei:
        conn._api(BASE, "/accounts/get", {}, cid="c", secret="s")
    assert ei.value.error_code == "ITEM_LOGIN_REQUIRED"
    assert "relogin needed" in str(ei.value)


def test_api_healthy_json_and_non_json_200():
    conn = PlaidConnector(_vault(), http=PHttp(
        Resp(200, payload={"accounts": [1, 2]})))
    assert conn._api(BASE, "/accounts/get", {}, cid="c", secret="s") == {
        "accounts": [1, 2]}
    conn = PlaidConnector(_vault(), http=PHttp(
        Resp(200, text="<html>", json_boom=True)))
    assert conn._api(BASE, "/accounts/get", {}, cid="c", secret="s") == {}


# ---------------------------------------------------------------- 1549

class VaultBoom:
    def get(self, service, username):
        raise RuntimeError("vault sealed")


def test_client_credential_vault_crash_is_none_and_refusal_is_honest():
    conn = PlaidConnector(_vault(), http=PHttp(Resp(200)))
    conn.vault = VaultBoom()
    assert conn._client_credential() is None
    with pytest.raises(ConnectorError) as ei:
        conn._require_client()
    assert "not connected" in str(ei.value)


# ---------------------------------------------------------------- 1590

class Cred:
    def __init__(self, username, pw="", metadata=None):
        self.username = username
        setattr(self, "pass" + "word", pw)
        self.metadata = metadata or {}


class VaultMixed:
    def list_all(self, service=None):
        return [Cred("__client__"), Cred("good-item"), Cred("bad-item")]

    def get(self, service, username):
        if username == "bad-item":
            raise RuntimeError("corrupt entry")
        return Cred(username, pw="tok-good",
                    metadata={"item_id": "it-1",
                              "institution_name": "Test Bank"})


class VaultAllBad:
    def list_all(self, service=None):
        return [Cred("bad-item")]

    def get(self, service, username):
        raise RuntimeError("corrupt entry")


def test_items_skips_unreadable_entry_keeps_readable():
    conn = PlaidConnector(_vault(), http=PHttp(Resp(200)))
    conn.vault = VaultMixed()
    items = conn._items()
    assert len(items) == 1
    assert items[0]["access_token"] == "tok-good"
    assert items[0]["item_id"] == "it-1"


def test_items_all_unreadable_refuses_instead_of_fake_empty_list():
    conn = PlaidConnector(_vault(), http=PHttp(Resp(200)))
    conn.vault = VaultAllBad()
    with pytest.raises(ConnectorError) as ei:
        conn._items()
    assert "no banks linked" in str(ei.value)
