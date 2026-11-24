"""Signal page (lane 127) — the bet desk's HTML page, redesigned first.

Truths under test: the page renders live calls with their countdown
state (LIVE NOW / opens in Nm), scored calls with the calibration
record in the open, per-game lab verdicts with evidence and what
would change them, and desk health with keys named by presence
only; an empty desk says so honestly in every section; hostile
strings are escaped; the page is served over HTTP on GET /signals
behind the bearer token and the db.read capability (no token 401,
a grant without db.read 403); the page is a snapshot — it stamps
its freshness and never pretends stale data is live.
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from nomorals.agents.betdesk.page import signal_page_for_context, signal_page_html
from nomorals.agents.betdesk.signals import SignalCall
from nomorals.web.shell import page_shell

NOW = 1_764_000_000.0  # a fixed clock: 2025-11-24 16:00:00 UTC


def _call(**over):
    base = {
        "provider": "stake", "game": "crash",
        "target": 2.5,
        "window_start": NOW + 14 * 60,
        "window_end": NOW + 74 * 60,
        "published_at": NOW - 120,
        "flagged_hours": [20, 21],
        "expected_rate": 0.42, "base_rate": 0.31,
        "evidence": "flagged hours beat base on 412 rounds",
        "scored": None, "call_id": "stake-crash-1",
    }
    base.update(over)
    return SignalCall(**base)


def _status(**over):
    base = {
        "rounds_total": 412,
        "games": [{
            "provider": "stake", "game": "crash", "rounds": 412,
            "verdict": "EDGE_CANDIDATE",
            "callable_structure": "time_of_day",
            "evidence": "flagged hours beat base on 412 rounds",
        }],
        "signals": {"scored_now": 0, "hits_now": 0, "calls_total": 0,
                    "calls_scored": 0, "hits_total": 0,
                    "hit_rate": None},
        "min_lead_seconds": 120,
        "sources": {"sources": [], "count": 0},
        "provisioning": [
            {"service": "odds-api", "env_key": "ODDS_API_KEY",
             "key_present": False, "source": "", "validation": "",
             "provisioned_at": 0.0}],
        "library_records": 1,
        "sports_fixtures": 7,
        "session_book": {},
    }
    base.update(over)
    return base


def _library(**over):
    base = {"stake/crash": {
        "verdict": "EDGE_CANDIDATE", "n_rounds": 412,
        "n_timed": 380,
        "evidence": "flagged hours beat base on 412 rounds",
        "what_would_change": "a month of flagged hours at base rate",
        "tests": [], "chain": {}}}
    base.update(over)
    return base


# ── live and upcoming calls ─────────────────────────────────────────

def test_upcoming_call_renders_countdown():
    html = signal_page_html(_status(), [_call()], _library(), now=NOW)
    assert "opens in 14m" in html
    assert "2.50x" in html
    assert "42% vs 31%" in html
    assert "Bet desk signals" in html


def test_live_call_renders_live_now():
    call = _call(window_start=NOW - 600, window_end=NOW + 3000)
    html = signal_page_html(_status(), [call], _library(), now=NOW)
    assert "LIVE NOW" in html
    assert "closes in 50m" in html


def test_calls_sort_soonest_window_first():
    late = _call(window_start=NOW + 120 * 60,
                 call_id="stake-crash-late")
    soon = _call(window_start=NOW + 5 * 60, call_id="stake-crash-soon")
    html = signal_page_html(_status(), [late, soon], _library(), now=NOW)
    assert html.index("opens in 5m") < html.index("opens in 120m")


def test_call_accepts_plain_dicts_too():
    call = {"provider": "stake", "game": "crash", "target": 2.0,
            "window_start": NOW + 600, "window_end": NOW + 4200,
            "published_at": NOW - 120, "expected_rate": 0.4,
            "base_rate": 0.3, "scored": None}
    html = signal_page_html(_status(), [call], _library(), now=NOW)
    assert "opens in 10m" in html
    assert "2.00x" in html


# ── scored calls and calibration ────────────────────────────────────

def test_scored_calls_render_hit_miss_and_rate():
    hit = _call(scored=True, window_start=NOW - 7200,
                window_end=NOW - 3600, call_id="c-hit")
    miss = _call(scored=False, window_start=NOW - 10800,
                 window_end=NOW - 7200, call_id="c-miss")
    status = _status(signals={"calls_total": 2, "calls_scored": 2,
                              "hits_total": 1, "hit_rate": 0.5})
    html = signal_page_html(status, [hit, miss], _library(), now=NOW)
    assert "HIT" in html and "MISS" in html
    assert "1/2" in html and "50%" in html


def test_scored_rate_falls_back_to_the_ledger():
    hit = _call(scored=True, window_start=NOW - 7200,
                window_end=NOW - 3600)
    status = _status(signals={})
    html = signal_page_html(status, [hit], _library(), now=NOW)
    assert "1/1" in html and "100%" in html


# ── honest empty states ─────────────────────────────────────────────

def test_empty_desk_says_so_in_every_section():
    html = signal_page_html(_status(games=[], signals={}), [],
                            {}, now=NOW)
    assert "No live or upcoming calls" in html
    assert "No calls scored yet" in html
    assert "No games studied yet" in html
    assert "0 calls published" in html


def test_lab_verdicts_render_evidence_and_what_would_change():
    html = signal_page_html(_status(), [], _library(), now=NOW)
    assert "EDGE_CANDIDATE" in html
    assert "flagged hours beat base on 412 rounds" in html
    assert "a month of flagged hours at base rate" in html


def test_health_names_missing_key_without_a_value():
    html = signal_page_html(_status(), [], _library(), now=NOW)
    assert "ODDS_API_KEY" in html
    assert "MISSING" in html
    assert "Signal lead floor: 120s" in html
    assert "412" in html  # rounds in the ledger


# ── escaping and the shell ──────────────────────────────────────────

def test_hostile_strings_are_escaped():
    evil = _call(provider="<script>alert(1)</script>",
                 game="crash & burn")
    html = signal_page_html(_status(), [evil], _library(), now=NOW)
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "crash &amp; burn" in html


def test_shell_chrome_and_freshness_stamp():
    html = signal_page_html(_status(), [], {}, now=NOW)
    assert html.startswith("<!doctype html>")
    assert "width=device-width,initial-scale=1" in html
    assert "<title>Devon — bet desk signals</title>" in html
    assert "snapshot" in html  # never pretends stale data is live
    assert "2025-11-24 16:00:00 UTC" in html


def test_page_shell_escapes_its_title():
    html = page_shell(title="<b>nope</b>", heading="H", body="<p>x</p>")
    assert "<b>nope</b>" not in html
    assert "&lt;b&gt;nope&lt;/b&gt;" in html


# ── served over HTTP ────────────────────────────────────────────────

def _api_context(tmp_path):
    from nomorals.storage.db import Database

    return SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir=str(tmp_path), workspace_dir=str(tmp_path),
            web=SimpleNamespace(
                base_url="", bot_username="devonbot",
                wp_page_url="", wp_page_user="",
                wp_page_password="<redacted>",
                wp_rules_user="", wp_rules_password=""),
            chat=SimpleNamespace(telegram_api_id="",
                                 telegram_api_hash="")),
        tools=None, bus=None, memory=None)


def _serve(tmp_path, token="owner-token", principals=None):
    from nomorals.api.server import APIServer, _make_handler

    context = _api_context(tmp_path)
    api = APIServer(context, token=token, principals=principals or {},
                    max_body_bytes=64 * 1024)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, port


def _get(port, path, token=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {}
    if token:
        headers["Authorization"] = "Bearer " + token
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    ctype = resp.getheader("Content-Type") or ""
    conn.close()
    return resp.status, ctype, body


def test_signals_served_over_http_to_the_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(tmp_path / "betdesk"))
    httpd, thread, port = _serve(tmp_path)
    try:
        status, ctype, body = _get(port, "/signals", token="owner-token")
        assert status == 200 and "text/html" in ctype
        assert "Bet desk signals" in body
        assert "No live or upcoming calls" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_signals_requires_a_bearer_token(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(tmp_path / "betdesk"))
    httpd, thread, port = _serve(tmp_path)
    try:
        status, _, _ = _get(port, "/signals")
        assert status == 401
        status, _, _ = _get(port, "/signals", token="wrong-token")
        assert status == 401
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_signals_requires_db_read(tmp_path, monkeypatch):
    from nomorals.api.server import Principal
    from nomorals.core.policy import Capability, CapabilitySet

    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(tmp_path / "betdesk"))
    reader = Principal(name="reader",
                       grant=CapabilitySet.of(Capability.FS_READ))
    httpd, thread, port = _serve(
        tmp_path, principals={"reader-token": reader})
    try:
        status, _, _ = _get(port, "/signals", token="reader-token")
        assert status == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_signal_page_for_context_reads_a_real_desk(tmp_path, monkeypatch):
    desk_dir = tmp_path / "betdesk"
    monkeypatch.setenv("NOMORALS_BETDESK_DIR", str(desk_dir))
    call = _call()
    desk_dir.mkdir(parents=True, exist_ok=True)
    with open(desk_dir / "signal_calls.jsonl", "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps(call.to_dict()) + "\n")
    context = _api_context(tmp_path)
    html = signal_page_for_context(context)
    assert "stake" in html and "crash" in html
    assert "2.50x" in html
