"""Dense-2 pins for nomorals/wisdom/ingestor.py — 0 converts,
9 census sites pinned as standing state (8 exclusions + 1 posture
question candidate), plus 2 extra non-census sites found by manual
derivation (1 exclusion, 1 data-loss targeted-fix shelf).

Each test pins one site's fallback behavior plus a control half
proving the except path was really taken (trigger == fallback AND
control != fallback, or the trigger would propagate without the
handler). No production code changes in this pass.

Census sites (scanner rules E103/E104 — broad or pass-body handlers):
* 242 E104 — _check_robots(): a CRASH of the robots check itself
  (any exception from RobotsCache.allowed) is treated as "allowed"
  and the fetch proceeds. POSTURE QUESTION candidate: robots.txt is
  the host's own published access declaration; the fail-open is
  unfenced (nothing downstream re-checks before the bytes are
  fetched and ingested), and the sibling search backends call
  allowed() UNwrapped — the same crash kills a hunt outright there,
  so the codebase has no settled doctrine. Pinned, not converted,
  not quietly excluded.
* 704 E103 — wisdom_registry_for_context(): KVStore bind failure
  leaves the registry unbound (in-memory only). The docstring's
  contract is "persistence is an enhancement; its absence never
  sinks a hunt" — degradation to a documented default.
* 708 E104 — wisdom_registry_for_context(): any resolution failure
  returns None, i.e. the module header's documented pre-#94
  behaviour exactly (no registry in reach). Exclusion.
* 720 E104 — _desc_for_wisdom_backend(): descriptor lookup failure
  returns None (docstring: "or None. Never raises"); consumers
  skip the recording / treat the backend as descriptor-less.
  Bookkeeping only. Exclusion.
* 767 E104 — _wisdom_error_evidence(): a crash inside the error
  classifier itself falls back to ("failure", "search_failed") —
  a value inside the function's own documented vocabulary.
  Telemetry fidelity only. Exclusion.
* 789 E103 — _apply_wisdom_evidence(): a registry record_* crash
  drops the outcome. Printed contract: "Best-effort; never
  raises — bookkeeping never breaks a hunt." Exclusion.
* 824 E103 — _record_wisdom_search(): a crash crediting a hunt
  attempt (e.g. note_candidate) drops the bookkeeping, never the
  hunt. Same printed contract. Exclusion.
* 857 E103 — _record_wisdom_fetch(): same for the fetch path.
  Exclusion.
* 895 E104 — _order_wisdom(): a registry hiccup returns the hunt
  order as-is (docstring's printed contract) — the default
  SEARCH_SOURCES order, a documented default. Exclusion.

Extra sites from manual derivation (typed handlers with default
bodies — outside the scanner's E101–E106 rule set, so not census
sites; pinned because the derivation found them):
* 190 — _load_index(): a corrupt/unreadable blob-cache index.json
  (OSError/ValueError) is treated as an EMPTY index. DATA-LOSS
  SHAPE, targeted-fix shelf: _load_index's result is cached and
  later written back by _save_index (via _store_blob on every
  successful fetch), so one corrupt byte permanently destroys
  every other URL→sha record in the on-disk index (blob files
  survive, orphaned; the corpus itself is separate storage and
  untouched). Pinned standing, NOT converted in this pass.
* 217 — _cached_bytes(): an OSError reading a cached blob returns
  None. The docstring's printed contract is "None on any miss/
  corruption"; the caller refetches and re-stores, and the index
  entry is untouched by the read path. Exclusion.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from unittest import mock

import pytest

from nomorals.media.source_registry import DEFAULT_REGISTRY
from nomorals.wisdom import ingestor as ing_mod
from nomorals.wisdom.ingestor import ArchiveIngestor, IngestError

_LOG_NAME = "nomorals.wisdom.ingestor"
_URL = "https://example.com/texts/seeker.txt"


def _bare(tmp_path: Path) -> ArchiveIngestor:
    """An ArchiveIngestor with only the attributes these methods touch."""
    ing = ArchiveIngestor.__new__(ArchiveIngestor)
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    ing._blobs = blobs
    ing._index_path = blobs / "index.json"
    ing._index = None
    ing._http = None
    return ing


def _module_log_records(caplog) -> list:
    return [r for r in caplog.records if r.name == _LOG_NAME]


# ── 190 (extra): _load_index — corrupt index reads as empty ────────────

def test_190_control_valid_index_loads(tmp_path):
    ing = _bare(tmp_path)
    ing._index_path.write_text(
        json.dumps({_URL: {"sha": "ab12", "at": 1, "bytes": 3}}),
        encoding="utf-8")
    assert ing._load_index() == {_URL: {"sha": "ab12", "at": 1, "bytes": 3}}


def test_190_corrupt_json_degrades_to_empty(tmp_path):
    ing = _bare(tmp_path)
    ing._index_path.write_text("{not json", encoding="utf-8")
    assert ing._load_index() == {}


def test_190_read_oserror_degrades_to_empty(tmp_path):
    ing = _bare(tmp_path)
    # File exists (is_file passes); the READ is what fails, so the
    # (OSError, ValueError) handler is genuinely the path taken.
    ing._index_path.write_text("{}", encoding="utf-8")
    with mock.patch.object(Path, "read_text", side_effect=OSError("gone")):
        assert ing._load_index() == {}


def test_190_corruption_then_store_destroys_prior_records(tmp_path):
    # The data-loss chain, pinned as standing state for the
    # targeted-fix shelf: corrupt index -> treated as empty ->
    # the next _store_blob/_save_index rewrites index.json from
    # the empty in-memory copy, permanently dropping the prior
    # URL->sha records (blob files survive, orphaned).
    ing = _bare(tmp_path)
    ing._index_path.write_text(
        json.dumps({"https://a.test/x": {"sha": "aa" * 32, "at": 1,
                                         "bytes": 3}}),
        encoding="utf-8")
    ing._index_path.write_text("{corrupt", encoding="utf-8")
    assert ing._load_index() == {}
    sha = ing._store_blob("https://b.test/y", b"new-bytes")
    saved = json.loads(ing._index_path.read_text(encoding="utf-8"))
    assert "https://a.test/x" not in saved
    assert saved["https://b.test/y"]["sha"] == sha


def test_190_corrupt_load_is_not_logged(tmp_path, caplog):
    ing = _bare(tmp_path)
    ing._index_path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger=_LOG_NAME):
        assert ing._load_index() == {}
    assert _module_log_records(caplog) == []


# ── 217 (extra): _cached_bytes — blob read OSError is a miss ───────────

def test_217_control_cached_bytes_returned(tmp_path):
    ing = _bare(tmp_path)
    (ing._blobs / "sha1").write_bytes(b"cached-bytes")
    ing._index = {_URL: {"sha": "sha1"}}
    assert ing._cached_bytes(_URL) == b"cached-bytes"


def test_217_blob_read_oserror_is_a_miss_and_index_survives(tmp_path):
    ing = _bare(tmp_path)
    ing._index = {_URL: {"sha": "sha1"}}

    class _BadBlob:
        def is_file(self):
            return True

        def read_bytes(self):
            raise OSError("disk gone")

    ing._blob_path = lambda sha: _BadBlob()
    assert ing._cached_bytes(_URL) is None
    # The read path destroys nothing: the index entry is untouched
    # (contrast with 190's load-then-rewrite chain).
    assert ing._load_index() == {_URL: {"sha": "sha1"}}


# ── 242: _check_robots — robots-check crash fails OPEN (posture Q) ─────

class _Robots:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def allowed(self, url, user_agent, http):
        if isinstance(self._behaviour, BaseException):
            raise self._behaviour
        return self._behaviour


def test_242_control_robots_denial_raises(tmp_path):
    ing = _bare(tmp_path)
    ing._robots = _Robots(False)
    try:
        ing._check_robots(_URL)
    except IngestError as exc:
        assert "robots.txt disallows" in str(exc)
    else:
        raise AssertionError("denial must raise IngestError")


def test_242_control_robots_allowance_passes(tmp_path):
    ing = _bare(tmp_path)
    ing._robots = _Robots(True)
    assert ing._check_robots(_URL) is None


def test_242_robots_crash_fails_open(tmp_path):
    # Standing state, pinned for the posture question: ANY crash of
    # the robots check (network failure fetching robots.txt, parse
    # bug, anything) is treated as "allowed" — no exception, no
    # IngestError, the fetch proceeds.
    ing = _bare(tmp_path)
    ing._robots = _Robots(RuntimeError("robots fetch exploded"))
    assert ing._check_robots(_URL) is None


def test_242_robots_crash_is_not_logged(tmp_path, caplog):
    ing = _bare(tmp_path)
    ing._robots = _Robots(RuntimeError("robots fetch exploded"))
    with caplog.at_level(logging.DEBUG, logger=_LOG_NAME):
        assert ing._check_robots(_URL) is None
    assert _module_log_records(caplog) == []


# ── 704/708: wisdom_registry_for_context ───────────────────────────────

class _DbContext:
    db = object()


def test_708_control_dbless_context_returns_none():
    assert ing_mod.wisdom_registry_for_context(
        types.SimpleNamespace()) is None
    assert ing_mod.wisdom_registry_for_context(None) is None


def test_708_resolution_crash_returns_none():
    class _BadContext:
        def __getattr__(self, name):
            raise RuntimeError("context exploded")

    # getattr's default only suppresses AttributeError, so without
    # the 708 handler this RuntimeError would propagate.
    assert ing_mod.wisdom_registry_for_context(_BadContext()) is None


def test_704_control_bind_attempted_when_healthy():
    bound = []
    with mock.patch("nomorals.storage.kv.KVStore", lambda db: ("kv", db)):
        with mock.patch.object(DEFAULT_REGISTRY, "bind_store",
                               lambda store: bound.append(store)):
            reg = ing_mod.wisdom_registry_for_context(_DbContext())
    assert reg is DEFAULT_REGISTRY
    assert len(bound) == 1


def test_704_bind_crash_still_returns_registry():
    with mock.patch.object(DEFAULT_REGISTRY, "bind_store",
                           side_effect=RuntimeError("kv down")):
        reg = ing_mod.wisdom_registry_for_context(_DbContext())
    assert reg is DEFAULT_REGISTRY


def test_704_kvstore_crash_still_returns_registry():
    with mock.patch("nomorals.storage.kv.KVStore",
                    side_effect=RuntimeError("kv ctor down")):
        reg = ing_mod.wisdom_registry_for_context(_DbContext())
    assert reg is DEFAULT_REGISTRY


# ── 720: _desc_for_wisdom_backend ──────────────────────────────────────

def test_720_control_descriptor_returned():
    sentinel = object()
    reg = types.SimpleNamespace(for_domain=lambda domain: sentinel)
    assert ing_mod._desc_for_wisdom_backend(reg, "gutendex") is sentinel


def test_720_unknown_backend_returns_none_without_registry():
    reg = types.SimpleNamespace(
        for_domain=lambda domain: (_ for _ in ()).throw(
            AssertionError("must not be called")))
    assert ing_mod._desc_for_wisdom_backend(reg, "nope") is None


def test_720_lookup_crash_returns_none():
    def _boom(domain):
        raise RuntimeError("registry down")

    reg = types.SimpleNamespace(for_domain=_boom)
    assert ing_mod._desc_for_wisdom_backend(reg, "gutendex") is None


# ── 767: _wisdom_error_evidence ────────────────────────────────────────

def test_767_control_robots_message_classifies_blocked():
    kind, error_class = ing_mod._wisdom_error_evidence(
        Exception("robots.txt disallows fetching this URL"))
    assert (kind, error_class) == ("blocked", "")


def test_767_control_plain_error_classifies_by_type():
    kind, error_class = ing_mod._wisdom_error_evidence(
        ValueError("bad payload"))
    assert (kind, error_class) == ("failure", "ValueError")


def test_767_classifier_crash_falls_back_to_generic_failure():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    assert ing_mod._wisdom_error_evidence(_BadStr()) == (
        "failure", "search_failed")


# ── 789: _apply_wisdom_evidence ────────────────────────────────────────

class _RecordingReg:
    def __init__(self):
        self.calls = []

    def record_outcome(self, desc, ok, latency_s=None, error_class=None):
        self.calls.append((desc, ok, error_class))

    def record_blocked(self, desc, latency_s=None):
        self.calls.append((desc, "blocked"))

    def record_rate_limited(self, desc, latency_s=None):
        self.calls.append((desc, "rate_limited"))

    def record_no_result(self, desc):
        self.calls.append((desc, "no_result"))


def test_789_control_outcome_recorded():
    reg = _RecordingReg()
    desc = object()
    ing_mod._apply_wisdom_evidence(reg, desc, "served", "", 0.25)
    assert reg.calls == [(desc, True, None)]


def test_789_record_crash_drops_outcome_silently(caplog):
    class _CrashReg(_RecordingReg):
        def record_outcome(self, desc, ok, latency_s=None,
                           error_class=None):
            raise RuntimeError("registry write down")

    with caplog.at_level(logging.DEBUG, logger=_LOG_NAME):
        result = ing_mod._apply_wisdom_evidence(
            _CrashReg(), object(), "served", "", 0.25)
    assert result is None
    assert _module_log_records(caplog) == []


# ── 824: _record_wisdom_search ─────────────────────────────────────────

class _SearchReg(_RecordingReg):
    def __init__(self, candidate_error=None):
        super().__init__()
        self.candidates = []
        self._candidate_error = candidate_error

    def for_domain(self, domain):
        return ("desc", domain)

    def note_candidate(self, url, via=None, kinds=()):
        if self._candidate_error is not None:
            raise self._candidate_error
        self.candidates.append(url)


def test_824_control_serve_records_and_notes_candidates():
    reg = _SearchReg()
    ing_mod._record_wisdom_search(
        reg, "gutendex", [{"url": "https://foreign.test/x"}], 0.1, None)
    assert reg.calls and reg.calls[0][1] is True
    assert reg.candidates == ["https://foreign.test/x"]


def test_824_candidate_crash_drops_bookkeeping_not_the_hunt():
    reg = _SearchReg(candidate_error=RuntimeError("intake down"))
    result = ing_mod._record_wisdom_search(
        reg, "gutendex", [{"url": "https://foreign.test/x"}], 0.1, None)
    assert result is None
    # The serve WAS recorded before the crash; only the candidate
    # intake was dropped — and nothing propagated to the caller.
    assert reg.calls and reg.calls[0][1] is True
    assert reg.candidates == []


# ── 857: _record_wisdom_fetch ──────────────────────────────────────────

class _FetchReg(_RecordingReg):
    def __init__(self, desc=None, url_error=None):
        super().__init__()
        self._desc = desc
        self._url_error = url_error
        self.candidates = []

    def for_url(self, url):
        if self._url_error is not None:
            raise self._url_error
        return self._desc

    def note_candidate(self, url, via=None, kinds=()):
        self.candidates.append(url)


def test_857_control_known_host_serve_recorded():
    desc = object()
    reg = _FetchReg(desc=desc)
    ing_mod._record_wisdom_fetch(reg, _URL, b"bytes", 0.1, None)
    assert reg.calls == [(desc, True, None)]


def test_857_for_url_crash_drops_bookkeeping_silently(caplog):
    reg = _FetchReg(url_error=RuntimeError("registry down"))
    with caplog.at_level(logging.DEBUG, logger=_LOG_NAME):
        result = ing_mod._record_wisdom_fetch(reg, _URL, b"bytes", 0.1, None)
    assert result is None
    assert reg.calls == []
    assert _module_log_records(caplog) == []


# ── 895: _order_wisdom ─────────────────────────────────────────────────

class _OrderReg:
    def __init__(self, stats_error=None):
        self._stats_error = stats_error

    def for_domain(self, domain):
        return ("desc", domain)

    def stats_for(self, desc):
        if self._stats_error is not None:
            raise self._stats_error
        return {"ok": 0, "fail": 0, "no_result": 0}

    def is_active(self, desc):
        return True

    def score(self, desc):
        return 0.0


def test_895_control_no_evidence_returns_hunt_as_is():
    names = ["gutendex", "wikisource", "archive", "web"]
    assert ing_mod._order_wisdom(_OrderReg(), list(names)) == names


def test_895_registry_crash_returns_hunt_as_is():
    names = ["gutendex", "wikisource", "archive", "web"]
    reg = _OrderReg(stats_error=RuntimeError("registry down"))
    assert ing_mod._order_wisdom(reg, list(names)) == names
