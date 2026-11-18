"""Multi-engine verdict organ (52-websites #24, VirusTotal shape,
Devon-local).

Failing-first: before this lane Devon scanned with one engine at a
time (``core.antivirus.scan_file`` / the ``av_scan`` tool) and no
aggregator existed — one artefact in, many independent engines,
a verdict record with per-engine results.  Every test here fails
there (``ImportError`` / missing names).

Proofs the brief demands, each pinned:
* fixture corpus — EICAR-class known-bad → flagged; clean
  fixtures → clean; an engine that errors reports ``unknown``,
  never a silent pass.
* hash-first — a known hash never re-scans content.
* the report names the engine count honestly (five local
  engines, not VirusTotal's 70+).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace

import pytest

# The EICAR standard test string, assembled from parts so this
# file never contains the contiguous signature (same idiom as the
# AV organ itself).
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

#: A real 1x1 PNG (magic bytes + IHDR), for structure-engine tests.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000000020001e221bc330000"
    "000049454e44ae426082")


def eicar_bytes() -> bytes:
    return (_EICAR_A + _EICAR_B).encode("ascii")


def eicar_sha256() -> str:
    return hashlib.sha256(eicar_bytes()).hexdigest()


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _verdict(subject, *, home, **kw):
    from nomorals.security.verdict import verdict_for

    return verdict_for(subject, home=home, **kw)


def _engines(report):
    return {e.name: e for e in report.engines}


# ── the fixture corpus ────────────────────────────────────────

def test_eicar_file_is_flagged_by_two_engines(tmp_path, home):
    target = tmp_path / "eicar.com"
    target.write_bytes(eicar_bytes())
    report = _verdict(str(target), home=home)
    assert report.aggregate == "flagged"
    engines = _engines(report)
    assert engines["hash-reputation"].verdict == "flagged"
    assert "EICAR-Test-File" in engines["hash-reputation"].detail
    assert engines["local-av"].verdict == "flagged"
    assert "EICAR-Test-File" in engines["local-av"].detail
    assert report.sha256 == eicar_sha256()


def test_clean_text_file_is_clean_on_all_engines(tmp_path, home):
    target = tmp_path / "hello.txt"
    target.write_text("hello world, nothing to see here\n")
    report = _verdict(str(target), home=home)
    assert report.aggregate == "clean"
    assert all(e.verdict == "clean" for e in report.engines)
    assert len(report.engines) == 5


def test_clean_png_is_clean_and_structure_matches(tmp_path, home):
    target = tmp_path / "dot.png"
    target.write_bytes(_PNG_1X1)
    report = _verdict(str(target), home=home)
    assert report.aggregate == "clean"
    structure = _engines(report)["structure"]
    assert structure.verdict == "clean"
    assert "png" in structure.detail


def test_double_extension_is_a_flagged_lie(tmp_path, home):
    target = tmp_path / "invoice.pdf.exe"
    target.write_text("totally a pdf, honest\n")
    report = _verdict(str(target), home=home)
    assert report.aggregate == "flagged"
    structure = _engines(report)["structure"]
    assert structure.verdict == "flagged"
    assert "double extension" in structure.detail


def test_magic_mismatch_is_a_flagged_lie(tmp_path, home):
    target = tmp_path / "photo.exe"
    target.write_bytes(_PNG_1X1)
    report = _verdict(str(target), home=home)
    assert report.aggregate == "flagged"
    structure = _engines(report)["structure"]
    assert structure.verdict == "flagged"
    assert "declared .exe" in structure.detail
    assert "png" in structure.detail


def test_secret_in_file_is_flagged_and_redacted(tmp_path, home):
    fake_key = "sk_live_" + "aB3dE5fG7hJ9kL2mN4pQ6" + "xYz12345"
    assert len(fake_key) > 24
    target = tmp_path / "leak.txt"
    target.write_text(f"stripe key: {fake_key}\n")
    report = _verdict(str(target), home=home)
    assert report.aggregate == "flagged"
    battery = _engines(report)["pattern-battery"]
    assert battery.verdict == "flagged"
    assert "stripe_live_key" in battery.detail
    # Redaction law: the full secret never lands in the report.
    blob = json.dumps(report.to_dict())
    assert fake_key not in blob
    assert "sk_l" in battery.detail  # head survives, redacted


def test_engine_that_errors_reports_unknown(tmp_path, home):
    def boom(_inp):
        raise RuntimeError("engine exploded")

    target = tmp_path / "hello.txt"
    target.write_text("hello\n")
    report = _verdict(
        str(target), home=home,
        engine_fns={"pattern-battery": boom})
    battery = _engines(report)["pattern-battery"]
    assert battery.verdict == "unknown"
    assert "engine error" in battery.detail
    assert report.aggregate == "unknown"


def test_missing_file_is_unknown_with_reason(home):
    report = _verdict("/no/such/file.bin", home=home)
    assert report.aggregate == "unknown"
    local_av = _engines(report)["local-av"]
    assert local_av.verdict == "unknown"
    assert "not found" in local_av.detail


# ── hash-first ────────────────────────────────────────────────

def test_second_scan_of_same_hash_uses_cache(tmp_path, home):
    from nomorals.security.verdict import EngineResult

    calls = {"n": 0}

    def counting(inp):
        calls["n"] += 1
        return EngineResult(name="pattern-battery", verdict="clean",
                            detail="counted")

    target = tmp_path / "hello.txt"
    target.write_text("hello\n")
    first = _verdict(str(target), home=home,
                     engine_fns={"pattern-battery": counting})
    assert first.from_cache is False
    assert calls["n"] == 1
    second = _verdict(str(target), home=home,
                      engine_fns={"pattern-battery": counting})
    assert second.from_cache is True
    assert second.aggregate == first.aggregate
    assert calls["n"] == 1  # the engines never re-ran


def test_hash_subject_flags_eicar_without_content(home):
    report = _verdict(eicar_sha256(), home=home)
    assert report.aggregate == "flagged"
    engines = _engines(report)
    assert engines["hash-reputation"].verdict == "flagged"
    # No content: the content engines say unknown, never clean.
    assert engines["local-av"].verdict == "unknown"
    assert engines["pattern-battery"].verdict == "unknown"
    assert engines["structure"].verdict == "unknown"


def test_hash_subject_for_unknown_hash_is_unknown(home):
    unknown = "a" * 64
    assert unknown != eicar_sha256()
    report = _verdict(unknown, home=home)
    assert report.aggregate == "unknown"
    engines = _engines(report)
    assert engines["hash-reputation"].verdict == "clean"
    assert engines["local-av"].verdict == "unknown"


# ── URL subjects ──────────────────────────────────────────────

def test_url_fetch_failure_is_unknown_with_reason(home):
    def failing_fetch(url, *, timeout_s, max_bytes):
        raise ValueError("connection refused")

    report = _verdict("https://example.invalid/x.bin", home=home,
                      fetcher=failing_fetch)
    assert report.aggregate == "unknown"
    assert "could not fetch" in _engines(report)["local-av"].detail \
        or "could not fetch" in _engines(report)["structure"].detail
    blob = json.dumps(report.to_dict())
    assert "connection refused" in blob


def test_url_fetch_success_scans_fetched_bytes(home):
    def eicar_fetch(url, *, timeout_s, max_bytes):
        return eicar_bytes()

    report = _verdict("https://example.invalid/eicar.com", home=home,
                      fetcher=eicar_fetch)
    assert report.aggregate == "flagged"
    assert report.sha256 == eicar_sha256()
    assert _engines(report)["hash-reputation"].verdict == "flagged"


# ── the sandbox rung ──────────────────────────────────────────

def test_non_script_content_needs_no_detonation(tmp_path, home):
    target = tmp_path / "hello.txt"
    target.write_text("hello\n")
    report = _verdict(str(target), home=home)
    sandbox = _engines(report)["sandbox"]
    assert sandbox.verdict == "clean"
    assert "nothing to detonate" in sandbox.detail


def test_script_without_sandbox_wiring_is_unknown(tmp_path, home):
    target = tmp_path / "run.py"
    target.write_text("print('hi')\n")
    report = _verdict(str(target), home=home)
    sandbox = _engines(report)["sandbox"]
    assert sandbox.verdict == "unknown"
    assert "not wired" in sandbox.detail
    assert report.aggregate == "unknown"


def test_script_with_quiet_detonation_is_clean(tmp_path, home):
    def quiet(data, filename):
        return {"outcome": "completed", "returncode": 0,
                "network_attempt": False, "detail": ""}

    target = tmp_path / "run.py"
    target.write_text("print('hi')\n")
    report = _verdict(str(target), home=home, sandbox_fn=quiet)
    sandbox = _engines(report)["sandbox"]
    assert sandbox.verdict == "clean"
    assert "exit 0" in sandbox.detail
    assert report.aggregate == "clean"


def test_script_that_phones_home_is_unknown(tmp_path, home):
    def chatty(data, filename):
        return {"outcome": "completed", "returncode": 0,
                "network_attempt": True, "detail": ""}

    target = tmp_path / "run.py"
    target.write_text("import socket\n")
    report = _verdict(str(target), home=home, sandbox_fn=chatty)
    sandbox = _engines(report)["sandbox"]
    assert sandbox.verdict == "unknown"
    assert "network" in sandbox.detail
    assert report.aggregate == "unknown"


def test_sandbox_fn_that_raises_is_unknown(tmp_path, home):
    def boom(data, filename):
        raise RuntimeError("sandbox down")

    target = tmp_path / "run.py"
    target.write_text("print('hi')\n")
    report = _verdict(str(target), home=home, sandbox_fn=boom)
    sandbox = _engines(report)["sandbox"]
    assert sandbox.verdict == "unknown"
    assert "engine error" in sandbox.detail


# ── honesty + store ───────────────────────────────────────────

def test_reply_names_engine_count_and_locality(tmp_path, home):
    from nomorals.security.verdict import format_verdict_reply

    target = tmp_path / "hello.txt"
    target.write_text("hello\n")
    report = _verdict(str(target), home=home)
    reply = format_verdict_reply(report)
    assert "5 local engines" in reply
    assert "not VirusTotal" in reply
    assert "Nothing left this box" in reply
    for engine in report.engines:
        assert f"{engine.name}: {engine.verdict}" in reply


def test_corrupt_store_starts_empty_and_quarantines(home):
    from nomorals.security.verdict import VerdictStore

    store_path = home / "data" / "verdicts.json"
    store_path.parent.mkdir(parents=True)
    store_path.write_text("{not json")
    store = VerdictStore(home)
    assert store.quarantined is True
    assert store.get("a" * 64) is None
    assert len(list(store_path.parent.glob("verdicts.corrupt-*"))) == 1


def test_subject_parsing():
    from nomorals.security.verdict import ScanSubject

    assert ScanSubject.from_string(
        "https://example.com/x.bin").kind == "url"
    assert ScanSubject.from_string(eicar_sha256()).kind == "hash"
    assert ScanSubject.from_string("/tmp/x.bin").kind == "file"
    assert ScanSubject.from_string(
        eicar_sha256(), kind="file").kind == "file"
    with pytest.raises(ValueError):
        ScanSubject(kind="domain", ref="example.com")
    with pytest.raises(ValueError):
        ScanSubject(kind="hash", ref="not-a-hash")
    with pytest.raises(ValueError):
        ScanSubject.from_string("   ")


# ── the brain tool ────────────────────────────────────────────

def test_tool_end_to_end(tmp_path):
    from nomorals.tools import security as sec_mod
    from nomorals.tools.registry import ToolRegistry

    src = tmp_path / "eicar.com"
    src.write_bytes(eicar_bytes())
    context = SimpleNamespace(
        settings=SimpleNamespace(home=str(tmp_path)))
    reg = ToolRegistry(context=context)
    sec_mod.register(reg)
    tool = reg._tools["file_verdict"]
    result = tool.fn(subject=str(src), sandbox=False)
    assert result["ok"] is True
    assert result["aggregate"] == "flagged"
    assert "reply" in result
    assert "FLAGGED" in result["reply"]
    assert "5 local engines" in result["reply"]
    assert result["engines"][0]["name"] == "hash-reputation"


def test_tool_registered_by_builtins():
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register_builtins()
    assert "file_verdict" in reg._tools
    assert "av_scan" in reg._tools


def test_tool_refuses_empty_subject(tmp_path):
    from nomorals.tools import security as sec_mod
    from nomorals.tools.registry import ToolRegistry

    reg = ToolRegistry(context=SimpleNamespace(
        settings=SimpleNamespace(home=str(tmp_path))))
    sec_mod.register(reg)
    tool = reg._tools["file_verdict"]
    assert tool.fn(subject="  ")["ok"] is False
    assert tool.fn(subject="/tmp/x.bin", kind="domain")["ok"] is False


class TestVerdictStoreRoundTrip(unittest.TestCase):
    def test_put_then_get(self):
        from nomorals.security.verdict import (
            EngineResult,
            ScanSubject,
            VerdictReport,
            VerdictStore,
        )

        with self.subTest("store round trip"), \
                tempfile.TemporaryDirectory() as d:
            store = VerdictStore(d)
            report = VerdictReport(
                subject=ScanSubject(kind="hash",
                                    ref=eicar_sha256()),
                sha256=eicar_sha256(),
                engines=[EngineResult(
                    name="hash-reputation",
                    verdict="flagged", detail="x")],
                aggregate="flagged", scanned_at=1234.0)
            store.put(report)
            got = store.get(eicar_sha256())
            assert got is not None
            assert got.aggregate == "flagged"
            assert got.from_cache is True
            assert got.scanned_at == 1234.0
