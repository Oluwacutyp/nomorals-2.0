"""Tail-93 pins: nomorals/search/osint.py census sites.

Every site in this module is one rung of the keyless-source
ladder: an adapter helper either gets real data or returns the
module's honest no-data value (None / [] / the preserved HTTP
status). No rung fabricates a positive — a hit exists only when
a real body parsed. All five sites are held as exclusions:

59 — ``_get``, HTTPError body read crash -> body="". ``_get``'s
contract is "Never raises" -> (status, body, final_url); the
STATUS is the decision signal every caller branches on and it
survives intact. The body of an error response is best-effort
detail; an unreadable one is worth exactly "".
73 — ``_get_json`` parse crash -> None. None is the function's
no-data value (also returned for non-200/empty); callers test
for None, never for "parsed but empty".
88 — ``_parse_ts`` float-parse ValueError -> pass. Not a
swallow: the pass IS the cascade — control falls through to
the ISO-8601 strategy, and only if that also fails does the
function return its honest None. The pin proves the fall-
through by feeding a string float() rejects but ISO accepts.
224 — ``_looks_like_ip`` inet_aton OSError -> pass -> None.
A classifier probe: "not an IP" is the answer the function
exists to give; None is a result, not a hidden failure.
385 — XposedOrNotAdapter.search json.loads crash -> []. A
malformed breach-API body yields no hit — the same [] as a
clean no-breach answer, but no positive is ever invented:
a breach is reported only from a body that actually parsed.

NOT-recording is pinned with the snapshot method: osint.py
calls no failures-ledger record function on these paths, so
both the agency and agents ledgers must be unchanged across
each trigger.
"""
from __future__ import annotations

import io
import urllib.error
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.search import osint as O


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


# ── 59: HTTPError body unreadable -> status kept, body "" ─────

class _BadFP:
    def read(self, *args, **kwargs):
        raise RuntimeError("body read boom")

    def close(self):
        pass


def test_59_http_error_body_read_failure_keeps_status():
    before = _snap()
    err = urllib.error.HTTPError(
        "http://x.test/", 503, "unavailable", {}, _BadFP())
    with mock.patch("urllib.request.urlopen", side_effect=err):
        assert O._get("http://x.test/") == (503, "", "http://x.test/")
    # control: a readable error body is returned verbatim
    err2 = urllib.error.HTTPError(
        "http://x.test/", 503, "unavailable", {}, io.BytesIO(b"sorry-body"))
    with mock.patch("urllib.request.urlopen", side_effect=err2):
        assert O._get("http://x.test/") == (503, "sorry-body", "http://x.test/")
    assert _snap() == before


# ── 73: _get_json parse failure -> None ───────────────────────

def test_73_get_json_parse_failure_returns_none():
    before = _snap()
    with mock.patch.object(O, "_get", return_value=(200, "{broken", "u")):
        assert O._get_json("http://x.test/") is None
    # control: a parseable body returns the object
    with mock.patch.object(O, "_get", return_value=(200, '{"x": 2}', "u")):
        assert O._get_json("http://x.test/") == {"x": 2}
    assert _snap() == before


# ── 88: _parse_ts ValueError -> pass -> ISO cascade ──────────

def test_88_parse_ts_float_failure_cascades_to_iso():
    before = _snap()
    # float() raises ValueError on this string; the pass must
    # fall through to the ISO-8601 strategy, not return None
    assert O._parse_ts("2024-06-01T12:00:00Z") == 1717243200.0
    # both strategies failing yields the honest None
    assert O._parse_ts("junk") is None
    # control: the float strategy itself
    assert O._parse_ts("42.5") == 42.5
    assert _snap() == before


# ── 224: _looks_like_ip probe failure -> None ─────────────────

def test_224_looks_like_ip_rejects_non_ip():
    before = _snap()
    assert O._looks_like_ip("not-an-ip") is None
    # control: a real dotted quad is returned as itself
    assert O._looks_like_ip("10.1.2.3") == "10.1.2.3"
    assert _snap() == before


# ── 385: XON adapter parse failure -> [] ──────────────────────

def test_385_xon_parse_failure_returns_no_hits():
    before = _snap()
    adapter = O.XposedOrNotAdapter()
    with mock.patch.object(O, "_get", return_value=(200, "{broken", "u")):
        assert adapter.search("a@b.co", limit=5) == []
    # control: a parseable breach body produces exactly one hit
    body = '{"breaches": [["Adobe", "LinkedIn"]]}'
    with mock.patch.object(O, "_get", return_value=(200, body, "u")):
        hits = adapter.search("a@b.co", limit=5)
    assert len(hits) == 1
    assert "Adobe" in hits[0].snippet or "Adobe" in hits[0].title
    assert _snap() == before
