"""Pass #124 pins: imagedb failures ledger + all 16 converted sites.

Converted (recorded): jpeg_dimensions, image_exif open/exposure/
f_number/gps, _index_row, _lookup_impl near-dupes, _library_rows,
image_stats dupe_groups, registry bind_store + outer, _image_engine_desc,
ordered_image_engines, _record_image_outcome, _dedupe_urls,
_bot_signalled vocabulary.

Deliberately NOT recorded (pinned as behaviour, not comments):
dhash / _gps_decimal / image_colors — pure best-effort probes where
failure IS the fallback answer (no hash / no coords / no colors for
unprocessable input); _scrape_one parse — a markup surprise IS the
designed malformed_response class, recorded downstream by the caller
into the registry. _scrape_engines' handler logs with exc_info=True
(debug-honest) and stays.
"""
import ast
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from nomorals.tools import imagedb
from nomorals.tools.imagedb_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def sites():
    return [f.site for f in recent_failures(1000)]


# ── ledger basics ─────────────────────────────────────────────────────

def test_ledger_basics_and_format():
    assert recent_failures() == []
    assert failure_count() == 0
    assert format_failures() == "no recorded image failures"
    record_failure("x", ValueError("boom"))
    assert failure_count() == 1
    assert [f.site for f in recent_failures()] == ["x"]
    assert "x: ValueError: boom" in format_failures()


def test_ledger_bound_keeps_newest():
    for i in range(250):
        record_failure(f"s{i}", RuntimeError("r"))
    assert failure_count() == 250
    kept = recent_failures(1000)
    assert len(kept) == 200
    assert kept[0].site == "s249"
    assert kept[-1].site == "s50"


def test_ledger_unprintable_exc_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("bad", BadStr())
    assert failure_count() == 1
    assert "unprintable" in recent_failures()[0].message


def test_direct_import_surface_and_all_unchanged():
    import nomorals.tools.imagedb as m

    assert m.__all__ == ["register", "sniff_format", "image_dimensions", "dhash",
                         "hamming", "image_exif", "image_colors", "find_dupes",
                         "image_stats", "REVERSE_IMAGE_ENGINES",
                         "IMAGE_ENGINE_DOMAINS", "IMAGE_LINK_ONLY_DOMAINS",
                         "image_registry_for_context", "ordered_image_engines"]


# ── helpers ─────────────────────────────────────────────────────────────

class FakeBytes:
    """bytes-like sequence whose slices raise when fail_slice is set."""

    def __init__(self, backing: bytes, fail_slice: bool = False):
        self._backing = backing
        self._fail = fail_slice

    def __len__(self):
        return len(self._backing)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if self._fail:
                raise ValueError("slice boom")
            return self._backing[idx]
        return self._backing[idx]


class FakeImg:
    def __init__(self, raw):
        self._raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getexif(self):
        if isinstance(self._raw, Exception):
            raise self._raw
        return self._raw


class FakeImage:
    def __init__(self, raw):
        self._raw = raw

    def open(self, path):
        return FakeImg(self._raw)


class BadStr:
    def __str__(self):
        raise RuntimeError("str boom")


class BadFloat:
    def __float__(self):
        raise RuntimeError("float boom")


class RawWithFailingIfd(dict):
    def get_ifd(self, tag):
        raise RuntimeError("ifd boom")


class RawWithEmptyIfd(dict):
    def get_ifd(self, tag):
        return {}


@pytest.fixture()
def fake_pil(monkeypatch):
    """Patch imagedb._pil_image to return FakeImage for the given raw."""
    def _set(raw):
        monkeypatch.setattr(imagedb, "_pil_image", lambda: FakeImage(raw))
    return _set


# ── converted sites ─────────────────────────────────────────────────────

def test_jpeg_dimensions_malformed_sof_records_and_aborts():
    # SOI + FF C0 marker; slices raise inside the SOF branch.
    backing = bytes([0xFF, 0xD8, 0xFF, 0xC0, 0, 8, 0, 1, 0, 2] + [0] * 30)
    assert imagedb.jpeg_dimensions(FakeBytes(backing, fail_slice=True)) is None
    assert sites() == ["jpeg_dimensions"]
    clear_failures()
    # height bytes [1, 0] = 256, width bytes [2, 0] = 512
    assert imagedb.jpeg_dimensions(FakeBytes(backing)) == (512, 256)
    assert sites() == []


def test_image_exif_open_failure_records(fake_pil):
    fake_pil(RuntimeError("read boom"))
    assert imagedb.image_exif("whatever.jpg") == {}
    assert sites() == ["image_exif:open"]


def test_image_exif_open_empty_exif_no_record(tmp_path):
    from PIL import Image

    p = tmp_path / "plain.png"
    Image.new("RGB", (4, 4), color="red").save(p)
    assert imagedb.image_exif(p) == {}
    assert sites() == []


def test_image_exif_exposure_and_fnumber_records(fake_pil):
    fake_pil({33434: BadStr()})
    out = imagedb.image_exif("x.jpg")
    assert "exposure_time" not in out
    assert sites() == ["image_exif:exposure"]
    clear_failures()
    fake_pil({33437: BadFloat()})
    out = imagedb.image_exif("x.jpg")
    assert "f_number" not in out
    assert sites() == ["image_exif:f_number"]


def test_image_exif_gps_failure_records_and_empty_ifd_does_not(fake_pil):
    # Raw must be non-empty: an empty getexif result returns {} early,
    # before the GPS block is ever reached.
    fake_pil(RawWithFailingIfd({271: "Canon"}))
    out = imagedb.image_exif("x.jpg")
    assert "gps" not in out
    assert sites() == ["image_exif:gps"]
    clear_failures()
    fake_pil(RawWithEmptyIfd({271: "Canon"}))
    out = imagedb.image_exif("x.jpg")
    assert "gps" not in out
    assert sites() == []


class FakeDb:
    def __init__(self, query_one=None, query=None, execute_raises=False):
        self._query_one = query_one
        self._query = query
        self._execute_raises = execute_raises

    def query_one(self, *a, **k):
        if isinstance(self._query_one, Exception):
            raise self._query_one
        return self._query_one

    def query(self, *a, **k):
        if isinstance(self._query, Exception):
            raise self._query
        return self._query

    def transaction(self):
        return nullcontext()

    def execute(self, *a, **k):
        if self._execute_raises:
            raise RuntimeError("db boom")


def test_index_row_db_failure_records_and_new_row_does_not():
    ctx = SimpleNamespace(db=FakeDb(query_one=None, execute_raises=True))
    assert imagedb._index_row(ctx, "d", __import__("pathlib").Path("p"), 1, "png", "") is None
    assert sites() == ["_index_row"]
    clear_failures()
    ctx = SimpleNamespace(db=FakeDb(query_one=None))
    assert imagedb._index_row(ctx, "d", __import__("pathlib").Path("p"), 1, "png", "") is None
    assert sites() == []
    ctx = SimpleNamespace(db=None)
    assert imagedb._index_row(ctx, "d", __import__("pathlib").Path("p"), 1, "png", "") is None
    assert sites() == []


def test_lookup_impl_near_dupes_db_failure_records(monkeypatch, tmp_path):
    from PIL import Image

    p = tmp_path / "pic.png"
    Image.new("RGB", (8, 8), color="blue").save(p)
    monkeypatch.setattr(imagedb, "dhash", lambda _p: "0123456789abcdef")
    ctx = SimpleNamespace(db=FakeDb(query_one=None, query=RuntimeError("q boom")))
    out = imagedb._lookup_impl(ctx, str(p), "")
    assert out["near_duplicates"] == []
    assert sites() == ["_lookup_impl:near_dupes"]
    clear_failures()
    ctx = SimpleNamespace(db=FakeDb(query_one=None, query=[]))
    out = imagedb._lookup_impl(ctx, str(p), "")
    assert out["near_duplicates"] == []
    assert sites() == []


def test_library_rows_db_failure_records_and_no_db_does_not():
    assert imagedb._library_rows(SimpleNamespace(db=None)) == []
    assert sites() == []
    assert imagedb._library_rows(SimpleNamespace(db=FakeDb(query=RuntimeError("q")))) == []
    assert sites() == ["_library_rows"]


def test_image_stats_dupe_groups_failure_records(monkeypatch):
    monkeypatch.setattr(imagedb, "find_dupes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cluster boom")))
    out = imagedb.image_stats(SimpleNamespace(db=None))
    assert out["dupe_groups"] == 0
    assert sites() == ["image_stats:dupe_groups"]
    clear_failures()
    monkeypatch.undo()
    out = imagedb.image_stats(SimpleNamespace(db=None))
    assert out["dupe_groups"] == 0
    assert sites() == []


def test_registry_bind_store_failure_records_inner_only(monkeypatch):
    class FakeReg:
        def __init__(self, raises):
            self._raises = raises

        def bind_store(self, store):
            if self._raises:
                raise RuntimeError("bind boom")

    monkeypatch.setattr("nomorals.media.source_registry.DEFAULT_REGISTRY", FakeReg(True))
    monkeypatch.setattr("nomorals.storage.kv.KVStore", lambda db: object())
    reg = imagedb.image_registry_for_context(SimpleNamespace(db=object()))
    assert isinstance(reg, FakeReg)
    assert sites() == ["image_registry_for_context:bind_store"]
    clear_failures()
    monkeypatch.setattr("nomorals.media.source_registry.DEFAULT_REGISTRY", FakeReg(False))
    reg = imagedb.image_registry_for_context(SimpleNamespace(db=object()))
    assert isinstance(reg, FakeReg)
    assert sites() == []


def test_registry_outer_failure_records_outer_only():
    class RaisingDb:
        @property
        def db(self):
            raise RuntimeError("ctx boom")

    assert imagedb.image_registry_for_context(RaisingDb()) is None
    assert sites() == ["image_registry_for_context"]
    clear_failures()
    assert imagedb.image_registry_for_context(None) is None
    assert sites() == []


def test_image_engine_desc_failure_records():
    class FakeReg:
        def for_domain(self, domain):
            raise RuntimeError("desc boom")

    assert imagedb._image_engine_desc(FakeReg(), "bing") is None
    assert sites() == ["_image_engine_desc"]
    clear_failures()
    assert imagedb._image_engine_desc(None, "bing") is None
    assert imagedb._image_engine_desc(FakeReg(), "nope") is None
    assert sites() == []


def test_ordered_image_engines_score_failure_records():
    class FakeReg:
        def for_domain(self, domain):
            return object()

        def stats_for(self, desc):
            return {"ok": 1, "fail": 0, "no_result": 0}

        def is_active(self, desc):
            return True

        def score(self, desc):
            raise RuntimeError("score boom")

    pool = ["bing", "yandex"]
    assert imagedb.ordered_image_engines(FakeReg(), pool) == pool
    assert sites() == ["ordered_image_engines"]
    clear_failures()
    assert imagedb.ordered_image_engines(None, pool) == pool
    assert sites() == []


def test_record_image_outcome_failure_records():
    class FakeReg:
        def for_domain(self, domain):
            return object()

        def record_outcome(self, *a, **k):
            raise RuntimeError("book boom")

    imagedb._record_image_outcome(FakeReg(), "bing", "served")
    assert sites() == ["_record_image_outcome"]
    clear_failures()
    imagedb._record_image_outcome(None, "bing", "served")
    assert sites() == []


def test_dedupe_urls_wrong_type_item_records_and_drops():
    assert imagedb._dedupe_urls([b"http://a", "http://b", b"http://a"]) == ["http://a"]
    assert sites() == ["_dedupe_urls"]
    clear_failures()
    assert imagedb._dedupe_urls([b"http://a", b"http://b", b"http://a"]) == ["http://a", "http://b"]
    assert sites() == []


def test_bot_signalled_vocabulary_import_failure_records():
    real = sys.modules["nomorals.media.source_registry"]
    stub = types.ModuleType("stub-source-registry")
    try:
        sys.modules["nomorals.media.source_registry"] = stub
        assert imagedb._bot_signalled(b"please verify you are human") is True
        assert sites() == ["_bot_signalled:vocabulary"]
    finally:
        sys.modules["nomorals.media.source_registry"] = real
    clear_failures()
    assert imagedb._bot_signalled(b"please verify you are human") is True
    assert sites() == []


# ── deliberate exclusions (probe class) ─────────────────────────────────

def test_dhash_probe_failure_is_the_answer():
    assert imagedb.dhash("/nonexistent/nope.png") is None
    assert sites() == []


def test_gps_decimal_probe_failure_is_the_answer():
    assert imagedb._gps_decimal({}) is None
    assert imagedb._gps_decimal({"GPSLatitude": "junk", "GPSLongitude": "junk"}) is None
    assert sites() == []


def test_image_colors_probe_failure_is_the_answer():
    assert imagedb.image_colors("/nonexistent/nope.png") == []
    assert sites() == []


def test_scrape_one_parse_failure_is_malformed_response_class(monkeypatch):
    def _raising_parse(body):
        raise ValueError("markup surprise")

    monkeypatch.setitem(imagedb._ENGINE_SCRAPERS, "bing",
                        (lambda u: u, _raising_parse))
    page = SimpleNamespace(status=200, body=b"<html>")

    class FakeClient:
        def get(self, url):
            return page

    assert imagedb._scrape_one("bing", FakeClient(), "http://img") == ([], "failure", "malformed_response")
    assert sites() == []
    monkeypatch.setitem(imagedb._ENGINE_SCRAPERS, "bing",
                        (lambda u: u, lambda body: []))
    assert imagedb._scrape_one("bing", FakeClient(), "http://img") == ([], "no_result", "")
    assert sites() == []


def test_scrape_engines_handler_is_debug_honest():
    import pathlib

    src = pathlib.Path(imagedb.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_scrape_engines":
            for h in ast.walk(node):
                if isinstance(h, ast.ExceptHandler):
                    assert "exc_info=True" in (ast.get_source_segment(src, h) or "")
                    return
    raise AssertionError("no _scrape_engines handler found")


def test_scan_remainder_is_exactly_exclusions():
    from nomorals.tools import error_scan as es

    path = imagedb.__file__
    orig = es._noqa_suppresses
    try:
        es._noqa_suppresses = lambda line, rule: False
        report = es.scan([path])
    finally:
        es._noqa_suppresses = orig
    funcs = {}
    for f in report.findings:
        # enclosing function by nearest preceding def
        src_lines = open(path).read().split("\n")
        fn = "?"
        for ln in range(f.line - 1, -1, -1):
            line = src_lines[ln]
            if line.startswith("def ") or line.startswith("    def "):
                fn = line.strip().split("(")[0].replace("def ", "")
                break
        funcs[fn] = funcs.get(fn, 0) + 1
    # #140: _scrape_engines is debug-honest (exc_info=True) and is now
    # recognized by the scanner, so it no longer counts.
    assert funcs == {"dhash": 1, "_gps_decimal": 1, "image_colors": 1,
                     "_scrape_one": 1}, funcs
    # Ledger file itself carries no findings.
    try:
        es._noqa_suppresses = lambda line, rule: False
        report2 = es.scan([__import__("nomorals.tools.imagedb_failures", fromlist=["x"]).__file__])
    finally:
        es._noqa_suppresses = orig
    assert report2.findings == []
