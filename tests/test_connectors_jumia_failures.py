"""Pins for Pass tail-111: nomorals/connectors/jumia.py — 3 exclusions.

612 E104 _mint_tokens: a 400 whose body is not JSON -> the error detail
    falls back to the raw response text; the JumiaError (invalid_grant)
    is still raised with the vendor's own words preferred when JSON
    parses. Error-detail extraction, never a swallowed failure.
764 E104 _raw_api: same fallback on the generic error path — a non-JSON
    error body contributes its raw text to the raised JumiaError, with
    the HTTP status preserved on the exception.
781 E103 _account_name: the shops fetch raises -> the documented
    fallback label, the client id ("first shop name, else the client
    id"). Display naming only; connect() decides on the tokens.
"""

from __future__ import annotations

import pytest

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.jumia import JumiaConnector, JumiaError
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


class HttpStub:
    def __init__(self, **routes):
        self.routes = routes

    def post_form(self, url, form, **k):
        return self.routes["post_form"]

    def get(self, url, **k):
        r = self.routes["get"]
        if isinstance(r, Exception):
            raise r
        return r


# ---------------------------------------------------------------- 612

def test_mint_tokens_non_json_400_uses_raw_text_detail():
    http = HttpStub(post_form=Resp(400, text="raw-grant-failure",
                                   json_boom=True))
    conn = JumiaConnector(_vault(), http=http)
    with pytest.raises(JumiaError) as ei:
        conn._mint_tokens("cid", "rtok")
    assert "invalid_grant" in str(ei.value)
    assert "raw-grant-failure" in str(ei.value)


def test_mint_tokens_json_detail_preferred():
    http = HttpStub(post_form=Resp(
        400, payload={"error_description": "grant expired nicely"},
        text="ignored"))
    conn = JumiaConnector(_vault(), http=http)
    with pytest.raises(JumiaError) as ei:
        conn._mint_tokens("cid", "rtok")
    assert "grant expired nicely" in str(ei.value)


# ---------------------------------------------------------------- 764

def test_raw_api_non_json_error_uses_raw_text_detail():
    http = HttpStub(get=Resp(500, text="gateway-html-boom", json_boom=True))
    conn = JumiaConnector(_vault(), http=http)
    with pytest.raises(JumiaError) as ei:
        conn._raw_api("GET", "/shops", None, "tok")
    assert ei.value.status_code == 500
    assert "gateway-html-boom" in str(ei.value)


def test_raw_api_json_message_preferred():
    http = HttpStub(get=Resp(500, payload={"message": "shop exploded"},
                             text="ignored"))
    conn = JumiaConnector(_vault(), http=http)
    with pytest.raises(JumiaError) as ei:
        conn._raw_api("GET", "/shops", None, "tok")
    assert "shop exploded" in str(ei.value)


# ---------------------------------------------------------------- 781

def test_account_name_fetch_crash_falls_back_to_client_id():
    conn = JumiaConnector(_vault(), http=HttpStub(get=RuntimeError("net down")))
    assert conn._account_name("client-9", "tok") == "client-9"


def test_account_name_healthy_uses_shop_name():
    http = HttpStub(get=Resp(200, payload={
        "shops": [{"id": "s1", "name": "Devon Demo Store"}]}))
    conn = JumiaConnector(_vault(), http=http)
    assert conn._account_name("client-9", "tok") == "Devon Demo Store"
