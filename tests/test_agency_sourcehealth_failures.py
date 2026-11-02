"""Pass tail-96 pins: nomorals/agency/sourcehealth.py — 4 sites.

164 — ``default_fetcher``: an HTTPError's body read crashes -> the
FetchResult carries body=b"" with the TRUE status code and headers.
EXCLUSION: best-effort evidence read inside an error handler. The
classification in ``_classify`` is status-dominant (404/410 dead,
451 geo, >=500 unverifiable, 403/429-with-no-items bot-blocked); the
body only refines page-signature matching, so an unreadable error
body degrades evidence, never fabricates a result.

188 — ``_json_items`` parse crash -> 0. EXCLUSION: a counting helper
over untrusted payloads; unparseable JSON has zero usable items by
definition, and ``_classify`` itself names "unparseable" in the
dead-on-200 detail. 0 is the honest count, not a masked success.

227 — ``_count_usable`` JSON_WEATHER branch parse crash -> 0.
EXCLUSION: same parser class as 188.

480 — ``note_result``: ``_save`` crash -> swallowed (pass).
EXCLUSION for this batch, FLAGGED as the batch's top targeted-fix
candidate — a DATA-LOSS shape: the record note_result computes
(failure streak, status transition, reactivation) is read back later
by ``record()`` / ``active_sources()`` / the next ``note_result``'s
prior, so a swallowed save silently diverges the DB from the record
returned to the caller: streaks undercount (stand-down/disable
delayed) and a successful re-check fails to reactivate the source in
every DB-backed view. It stays out of the batch per the data-loss
rule (persona-_archive precedent: targeted fix, not a batch convert).
The pin below proves the divergence is real and observable, and that
the sibling ``verify()`` path does NOT swallow the same failure —
the asymmetry is the targeted fix's starting evidence. The stated
contract "health must never break a run" is also pinned: note_result
itself never raises.
"""

import io
import urllib.error
import urllib.request

import pytest

from nomorals.agency import sourcehealth as sh
from nomorals.agency.sourcehealth import (
    JSON_WEATHER,
    FetchResult,
    SourceHealth,
    SourceSpec,
)
from nomorals.storage.db import Database


def _spec(**kw):
    base = dict(key="news:world:bbc", category="news", name="BBC",
                url="https://feeds.example/rss.xml", kind="rss")
    base.update(kw)
    return SourceSpec(**base)


# ── 164: HTTPError body unreadable -> true status, empty body ────

class _ExplodingFP:
    def read(self, *a, **k):
        raise OSError("connection reset mid-body")


def _http_error(code, fp):
    return urllib.error.HTTPError(
        "https://feeds.example/rss.xml", code, "err", None, fp)


def test_164_http_error_body_read_crash_keeps_status(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise _http_error(403, _ExplodingFP())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = sh.default_fetcher("https://feeds.example/rss.xml")
    assert result.status == 403
    assert result.body == b""
    assert result.error == ""


def test_164_http_error_body_readable_is_returned(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise _http_error(403, io.BytesIO(b"bot wall page"))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = sh.default_fetcher("https://feeds.example/rss.xml")
    assert result.status == 403
    assert result.body == b"bot wall page"


# ── 188 / 227: payload parsers count unparseable as zero ─────────

def test_188_json_items_unparseable_counts_zero():
    assert sh._json_items(b"{not json") == 0
    assert sh._json_items(b'{"results": [1,') == 0
    # controls: parseable payloads count their real items
    assert sh._json_items(b'{"results": [1, 2, 3]}') == 3
    assert sh._json_items(b"[1, 2]") == 2
    assert sh._json_items(b'{"unrelated": 1}') == 0


def test_227_weather_items_unparseable_counts_zero():
    assert sh._count_usable(JSON_WEATHER, b"{bad") == 0
    # controls
    assert sh._count_usable(JSON_WEATHER, b'{"hourly": []}') == 1
    assert sh._count_usable(JSON_WEATHER, b'{"unrelated": 1}') == 0


# ── 480: note_result save swallow — pinned data-loss divergence ──

class _SaveFailingDB:
    """Delegates everything except the health-record INSERT."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, sql, params=()):
        if sql.lstrip().upper().startswith(
                "INSERT INTO AGENCY_SOURCE_HEALTH"):
            raise RuntimeError("disk full (injected)")
        return self._inner.execute(sql, params)

    def query_one(self, sql, params=()):
        return self._inner.query_one(sql, params)

    def query(self, sql, params=()):
        return self._inner.query(sql, params)


def test_480_note_result_save_failure_diverges_db_from_returned_rec():
    spec = _spec()
    inner = Database(":memory:")
    healthy = SourceHealth(inner)
    first = healthy.note_result(spec, ok=False, error="boom", now=1000.0)
    assert first.status == "unverifiable"
    assert healthy.record(spec.key).consecutive_failures == 1

    failing = SourceHealth(_SaveFailingDB(inner))
    # the contract holds: reporting health never breaks the run
    rec = failing.note_result(spec, ok=True, items=5, now=2000.0)
    # the returned record claims reactivation...
    assert rec.status == "verified-live"
    # ...but the persisted record — the one every later read sees —
    # still shows the old failure state. Divergence pinned.
    persisted = failing.record(spec.key)
    assert persisted.status == "unverifiable"
    assert persisted.consecutive_failures == 1


def test_480_sibling_verify_path_does_not_swallow_save_failure():
    spec = _spec()
    failing = SourceHealth(_SaveFailingDB(Database(":memory:")))

    def ok_fetcher(url, headers=None):
        return FetchResult(200, b'{"results": [1]}', {})

    with pytest.raises(RuntimeError) as excinfo:
        failing.verify(spec, fetcher=ok_fetcher, now=1000.0)
    assert isinstance(excinfo.value, RuntimeError)
