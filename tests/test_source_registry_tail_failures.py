"""Pass #131 pins: registry tail-region failures stop vanishing.

The registry's tail regions — deserialization, the status-text
renderer, the Ekiti lexicon, the content-addressed cache — sit behind
the same never-raises contract as the registry itself, and every one
of their fallbacks hid the crash from the caller. A corrupt persisted
row dropped a source with no trace; a malformed status section
dropped the whole render into "unavailable"; a lexicon crash returned
the raw query as if nothing were mappable; a cache-root failure
silently relocated the archive to the temp dir; a cache-read failure
read as "not cached"; a malformed sidecar timestamp aborted a URL
lookup that had a valid match later in the scan. The 8 sites now
record into the shared source-registry ledger (reused from #121)
while returning exactly the same fallback values.

Deliberately excluded: the lookup's inner corrupt-sidecar skip — a
sidecar whose bytes cannot be parsed cannot match any URL, so
skipping it changes no answer; the lookup still returns the good
sidecar's bytes (or None) and the outer handler still records genuine
loop failures. Pinned as behaviour below, not silence.
"""

import ast
import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from nomorals.media import source_registry as sr_mod
from nomorals.media.source_registry import (
    ContentAddressedCache,
    EkitiLexicon,
    LearnedRule,
    _desc_from_dict,
    render_status_text,
)
from nomorals.media.source_registry_failures import (
    clear_failures,
    recent_failures,
)


def _sites():
    return [f.site for f in recent_failures()]


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


# ── deserialization ──────────────────────────────────────────────────


def test_desc_from_dict_malformed_row_records_but_still_returns_none():
    bad = {"domain": "example.com", "quality_map": [["only-one"]]}
    assert _desc_from_dict(bad) is None
    assert _sites() == ["_desc_from_dict"]


def test_desc_from_dict_benign_path_records_nothing():
    desc = _desc_from_dict({"domain": "example.com", "nation": "NG"})
    assert desc is not None and desc.domain == "example.com"
    assert recent_failures() == []


def test_learned_rule_from_dict_malformed_row_records_but_still_none():
    bad = {"stage": "extract", "successes": "not-a-number"}
    assert LearnedRule.from_dict(bad) is None
    assert _sites() == ["LearnedRule.from_dict"]


def test_learned_rule_from_dict_benign_path_records_nothing():
    rule = LearnedRule.from_dict({"stage": "extract", "successes": 3})
    assert rule is not None and rule.successes == 3
    assert recent_failures() == []


# ── status-text renderer ─────────────────────────────────────────────


def test_render_status_malformed_section_records_but_still_unavailable():
    # One malformed source row must not silently drop every section.
    status = {"total": 2, "sources": [{"domain": "example.com"}]}
    assert render_status_text(status) == (
        "📥 download sources: status unavailable")
    assert _sites() == ["render_status_text"]


def test_render_status_benign_path_records_nothing():
    text = render_status_text({"total": 1, "seeds": 1})
    assert "1 sources" in text and "unavailable" not in text
    assert recent_failures() == []


# ── Ekiti lexicon ────────────────────────────────────────────────────


def test_to_standard_crash_records_but_still_returns_word():
    lex = EkitiLexicon({})
    with mock.patch.object(sr_mod, "_strip_tones",
                           side_effect=RuntimeError("tones broken")):
        assert lex.to_standard("xyzabc") == "xyzabc"
    assert _sites() == ["EkitiLexicon.to_standard"]


def test_to_standard_benign_path_records_nothing():
    lex = EkitiLexicon({})
    assert lex.to_standard("xyzabc") == "xyzabc"
    assert recent_failures() == []


def test_normalize_query_crash_records_but_still_returns_fallback():
    lex = EkitiLexicon({})
    with mock.patch.object(EkitiLexicon, "to_standard",
                           side_effect=RuntimeError("lexicon broken")):
        out = lex.normalize_query("ọmọ")
    assert out == {"original": "ọmọ", "search": "ọmọ", "mapped": []}
    assert _sites() == ["EkitiLexicon.normalize_query"]


def test_normalize_query_benign_path_records_nothing():
    lex = EkitiLexicon({})
    out = lex.normalize_query("ọmọ")
    assert out["original"] == "ọmọ" and out["mapped"] == []
    assert recent_failures() == []


# ── content-addressed cache ──────────────────────────────────────────


def test_cache_init_root_failure_records_but_still_falls_back(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    cache = ContentAddressedCache(root=str(blocker / "sub"))
    assert cache.root == Path(tempfile.gettempdir()) / "nomorals-sources-cache"
    assert _sites() == ["ContentAddressedCache.__init__"]


def test_cache_init_benign_path_records_nothing(tmp_path):
    cache = ContentAddressedCache(root=str(tmp_path / "cache"))
    assert cache.root == tmp_path / "cache" and cache.root.is_dir()
    assert recent_failures() == []


def test_cache_load_read_failure_records_but_still_returns_none(tmp_path):
    cache = ContentAddressedCache(root=str(tmp_path / "cache"))
    digest = "ab" + "c" * 62
    blob = cache.root / digest[:2] / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"bytes")
    with mock.patch.object(Path, "read_bytes",
                           side_effect=OSError("disk gone")):
        assert cache.load(digest) is None
    assert _sites() == ["ContentAddressedCache.load"]


def test_cache_load_benign_paths_record_nothing(tmp_path):
    cache = ContentAddressedCache(root=str(tmp_path / "cache"))
    assert cache.load("ab" + "c" * 62) is None  # absent blob: designed None
    digest = "ab" + "d" * 62
    blob = cache.root / digest[:2] / digest
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"bytes")
    assert cache.load(digest) == b"bytes"
    assert recent_failures() == []


def _sidecar(root, sub, name, url, fetched_at):
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({
        "sha256": name[:-5],
        "provenance": {"url": url, "fetched_at": fetched_at},
    }), encoding="utf-8")
    return d / name


def test_cache_lookup_malformed_timestamp_records_and_drops_valid_match(
        tmp_path):
    # The malformed sidecar sorts first and aborts the whole scan, so
    # a valid match later in the scan is never reached — the record is
    # what makes that silent loss visible.
    root = tmp_path / "cache"
    url = "https://example.com/page"
    _sidecar(root, "aa", "aa" + "0" * 62 + ".json", url, "garbage")
    good = _sidecar(root, "bb", "bb" + "1" * 62 + ".json", url, time.time())
    (root / "bb" / ("bb" + "1" * 62)).write_bytes(b"good bytes")
    cache = ContentAddressedCache(root=str(root))
    assert cache.lookup_url(url) is None
    assert _sites() == ["ContentAddressedCache.lookup_url"]
    assert good.exists()  # the valid match was there all along


def test_cache_lookup_benign_path_records_nothing(tmp_path):
    root = tmp_path / "cache"
    url = "https://example.com/page"
    _sidecar(root, "bb", "bb" + "1" * 62 + ".json", url, time.time())
    (root / "bb" / ("bb" + "1" * 62)).write_bytes(b"good bytes")
    cache = ContentAddressedCache(root=str(root))
    assert cache.lookup_url(url) == ("bb" + "1" * 62, b"good bytes")
    assert recent_failures() == []


def test_cache_lookup_corrupt_sidecar_skip_is_behaviour_not_silence(
        tmp_path):
    # A sidecar whose bytes cannot be parsed cannot match any URL, so
    # skipping it changes no answer: the good sidecar still wins, and
    # nothing is recorded for a skip that lost nothing.
    root = tmp_path / "cache"
    url = "https://example.com/page"
    d = root / "00"
    d.mkdir(parents=True, exist_ok=True)
    (d / ("00" + "0" * 62 + ".json")).write_text("{not json", encoding="utf-8")
    _sidecar(root, "bb", "bb" + "1" * 62 + ".json", url, time.time())
    (root / "bb" / ("bb" + "1" * 62)).write_bytes(b"good bytes")
    cache = ContentAddressedCache(root=str(root))
    assert cache.lookup_url(url) == ("bb" + "1" * 62, b"good bytes")
    assert recent_failures() == []


# ── scan pin: file remainder is exactly the deliberate sites ─


def test_scan_tail_remainder_is_exactly_four():
    from nomorals.tools import error_scan as es

    path = sr_mod.__file__
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([path])
    finally:
        import importlib
        importlib.reload(es)
    tree = ast.parse(open(path).read())
    owner_of = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    for t in ast.walk(m):
                        if isinstance(t, ast.Try):
                            for h in t.handlers:
                                owner_of[h.lineno] = (node.name, m.name)
        elif isinstance(node, ast.FunctionDef):
            for t in ast.walk(node):
                if isinstance(t, ast.Try):
                    for h in t.handlers:
                        owner_of.setdefault(h.lineno, (None, node.name))
    # 16 (Pass #130 remainder) minus exactly the 8 Pass #131 conversions.
    # Line-shift arithmetic: each conversion inserts one record line
    # after its except line, shifting every later line by +1 each.
    # Original lines -> shifted: 648 -> 648, 953 -> 955, 1036 -> 1038,
    # 1108 -> 1110, 1713 -> 1715, 1845 -> 1847, 3038 -> 3044,
    # 3059 -> 3066. Converted lines (714, 806, 2540, 2866, 2887, 3012,
    # 3047, 3070) are gone.
    # #140: minus exactly the 4 debug-honest sites the scanner now
    # recognizes via exc_info=True (955, 1038, 1847, 3044).
    expected = {648, 1110, 1715, 3066}
    assert {f.line for f in report.findings} == expected
    owners = {owner_of.get(line) for line in expected}
    assert owners == {
        (None, "_default_health_fetch"),  # #130 deliberate: status 0
        ("SourceRegistry", "_host_of"),  # #121 deliberate: "" is the answer
        ("SourceRegistry", "_verify_one"),  # #121 deliberate: root probe
        ("ContentAddressedCache", "lookup_url"),  # inner skip: behaviour
    }
