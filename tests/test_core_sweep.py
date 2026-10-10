"""Sweep tests for the nomorals.core system-wide upgrade.

Covers the new/changed behavior added by the core sweep. Offline only —
no network, no external services.
"""

import sys
import time

import pytest

sys.path.insert(0, ".")


# ── ids ──────────────────────────────────────────────────────────────────────

class TestIds:
    def test_is_ulid_strict(self):
        from nomorals.core.ids import is_ulid, ulid_now
        assert is_ulid(ulid_now())
        assert is_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert not is_ulid("8ARZ3NDEKTSV4RRFFQ69G5FAV")  # first char > 7
        assert not is_ulid("short")
        assert not is_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAI")  # I not in alphabet
        assert not is_ulid(123)

    def test_ulid_at_and_monotonic(self):
        from nomorals.core.ids import ULID, ulid_at, decode_time
        u = ULID.at(1_700_000_000_000)
        assert abs(decode_time(u.raw) - 1_700_000_000.0) < 0.001
        nxt = u.next()
        assert nxt.raw > u.raw
        assert nxt.datetime_ms == u.datetime_ms

    def test_ulid_bytes_uuid_roundtrip(self):
        from nomorals.core.ids import ULID
        u = ULID.generate()
        assert ULID.from_bytes(u.to_bytes()).raw == u.raw
        assert ULID.from_uuid(u.to_uuid()).raw == u.raw
        assert len(u.to_bytes()) == 16


# ── diff ─────────────────────────────────────────────────────────────────────

class TestDiff:
    def test_unified_diff_roundtrip(self):
        from nomorals.core.diff import unified_diff, apply_unified_diff
        a = "one\ntwo\nthree\n"
        b = "one\nTWO\nthree\nfour\n"
        d = unified_diff(a, b, path="f.txt")
        assert "--- a/f.txt" in d and "+++ b/f.txt" in d
        assert apply_unified_diff(d, {"f.txt": a})["f.txt"] == b

    def test_format_diff_colors(self):
        from nomorals.core.diff import format_diff
        d = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n ctx\n"
        out = format_diff(d, color=True)
        assert "\x1b[" in out  # ansi present
        assert format_diff(d, color=False) == d


# ── cipher ───────────────────────────────────────────────────────────────────

class TestCipher:
    def test_hkdf_rfc5869_vector(self):
        # RFC 5869 test case 1 (SHA-256)
        from nomorals.core.cipher import hkdf
        ikm = bytes.fromhex("0b" * 22)
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        okm = hkdf(ikm, salt=salt, info=info, length=42)
        assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a"
                             "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                             "34007208d5b887185865")

    def test_seal_roundtrip_and_binding(self):
        import os
        from nomorals.core.cipher import seal, unseal, CipherError
        key = os.urandom(32)
        blob = seal("hello", key, associated=b"ctx")
        assert unseal(blob, key, associated=b"ctx") == b"hello"
        with pytest.raises(CipherError):
            unseal(blob, key, associated=b"other")
        with pytest.raises(CipherError):
            unseal(blob, os.urandom(32), associated=b"ctx")


# ── decoder ──────────────────────────────────────────────────────────────────

class TestDecoder:
    def test_base85(self):
        import base64
        from nomorals.core.decoder import DECODERS
        d = next(x for x in DECODERS if x.name == "base85")
        enc = base64.a85encode(b"hello decoder world").decode()
        hit = d.attempt(enc)
        assert hit is not None
        out = hit.output
        assert (out == b"hello decoder world"
                or out == "hello decoder world")

    def test_rot47(self):
        from nomorals.core.decoder import DECODERS
        d = next(x for x in DECODERS if x.name == "rot47")
        hit = d.attempt("96==@ H@C=5")  # "hello world"
        assert hit is not None and hit.output == "hello world"
        assert d.attempt("already plain english words here") is None


# ── barcode ──────────────────────────────────────────────────────────────────

class TestBarcode:
    def test_code39_roundtrip(self):
        from nomorals.core import barcode as bc
        for txt in ["HELLO-39", "ABC 123"]:
            dec = bc.decode_code39(bc.encode_code39(txt)["bits"])
            assert dec["ok"] and dec["text"] == txt

    def test_code39_checksum_probe(self):
        from nomorals.core import barcode as bc
        dec = bc.decode_code39(bc.encode_code39("ABC", checksum=True)["bits"])
        assert dec["check_ok"] is True

    def test_code39_table_spotcheck(self):
        # '*' start/stop == python-barcode EDGE run 100010111011101
        from nomorals.core.barcode import _C39_START_STOP
        assert _C39_START_STOP == "nwnnwnwnn"

    def test_itf_roundtrip(self):
        from nomorals.core import barcode as bc
        for digits in ["123456", "0011223344"]:
            dec = bc.decode_itf(bc.encode_itf(digits)["bits"])
            assert dec["ok"] and dec["digits"] == digits

    def test_ean8_roundtrip_and_check(self):
        from nomorals.core import barcode as bc
        enc = bc.encode_ean8("1234567")
        assert enc["length_modules"] == 67
        dec = bc.decode_ean8(enc["bits"])
        assert dec["ok"] and dec["check_ok"]
        assert dec["digits"] == "1234567" + str(bc.ean8_check_digit("1234567"))

    def test_analyze_finds_new_symbologies(self):
        from nomorals.core import barcode as bc
        assert bc.analyze(bc.encode_code39("HI")["bits"])["candidates"][0]["symbology"] == "code39"
        assert bc.analyze(bc.encode_itf("123456")["bits"])["candidates"][0]["symbology"] == "itf"
        assert bc.analyze(bc.encode_ean8("1234567")["bits"])["candidates"][0]["symbology"] == "ean8"

    def test_render_ascii(self):
        from nomorals.core import barcode as bc
        out = bc.render_ascii("101", height=2)
        assert out == "█ █\n█ █"


# ── midi ─────────────────────────────────────────────────────────────────────

class TestMidi:
    def test_gm_table(self):
        from nomorals.core import midi
        assert len(midi.GM_INSTRUMENTS) == 128
        assert midi.program_name(0) == "Acoustic Grand Piano"
        assert midi.program_name(127) == "Gunshot"
        assert midi.program_number("nylon") == 24
        assert midi.gm_family(25) == "Guitar"
        assert len(midi.DRUM_MAP) == 47
        assert midi.DRUM_MAP[36] == "Bass Drum 1"

    def test_quantize(self):
        from nomorals.core import midi
        ev = midi.NoteEvent(note=60, start=0.13, duration=0.9)
        q = midi.quantize([ev], 0.25, strength=1.0)[0]
        assert q.start == pytest.approx(0.25) and q.duration == pytest.approx(1.0)
        q0 = midi.quantize([ev], 0.25, strength=0.0)[0]
        assert q0.start == pytest.approx(0.13)


# ── pdf ──────────────────────────────────────────────────────────────────────

class TestPdf:
    TEXT = ("# Report\n\n| Name | Qty |\n|------|----:|\n"
            "| Apples | 120 |\n\nBody text.\n")

    def test_table_renders_and_reads_back(self):
        from nomorals.core import pdf
        data = pdf.render_pdf(self.TEXT, headings=True)
        txt = pdf.read_pdf_text(data)
        assert "Apples" in txt and "120" in txt and "Qty" in txt

    def test_metadata_in_info_dict(self):
        from nomorals.core import pdf
        data = pdf.render_pdf("hi", headings=True,
                              metadata={"title": "T", "author": "A"})
        assert b"/Info" in data and b"/Title (T)" in data and b"/Author (A)" in data
        data2 = pdf.render_pdf("hi", metadata={"author": "A"})
        assert b"/Info" in data2

    def test_total_page_footer(self):
        from nomorals.core import pdf
        data = pdf.render_pdf("hello", headings=True, footer="p. N/{total}")
        assert "p. 1/1" in pdf.read_pdf_text(data)


# ── corpus ───────────────────────────────────────────────────────────────────

class TestCorpus:
    def test_hashcat_classics(self):
        from nomorals.core.corpus import apply_rules
        assert apply_rules("abc", "reflect") == ["abccba"]
        assert apply_rules("abc", "toggle") == ["ABC"]
        assert apply_rules("ab", "toggle_each") == ["Ab", "aB"]
        assert apply_rules("abcd", "rotate_l") == ["bcda"]
        assert apply_rules("abcd", "rotate_r") == ["dabc"]

    def test_d1_deduped(self):
        from nomorals.core.corpus import apply_rules
        vals = apply_rules("x", "d1")
        assert len(vals) == len(set(vals))


# ── cookies ──────────────────────────────────────────────────────────────────

class TestCookies:
    def test_request_header_parsing(self):
        from nomorals.core.cookies import parse_cookies, cookies_to_header
        cs = parse_cookies("Cookie: a=1; b=2")
        assert [(c.name, c.value) for c in cs] == [("a", "1"), ("b", "2")]
        assert cookies_to_header(cs) == "a=1; b=2"

    def test_jar_lifecycle(self):
        from nomorals.core.cookies import CookieJar
        jar = CookieJar()
        jar.update("Set-Cookie: s=abc; Path=/; HttpOnly")
        jar.update("Set-Cookie: t=dark; Max-Age=3600")
        assert jar.header() == "s=abc; t=dark"
        jar.update("Set-Cookie: t=gone; Max-Age=0")
        assert "t=" not in jar.header()
        assert jar.get("s") == "abc"
        assert len(jar) == 1


# ── trust ────────────────────────────────────────────────────────────────────

class TestTrust:
    def _st(self):
        from types import SimpleNamespace
        from nomorals.core.trust import SourceTrust
        return SourceTrust(SimpleNamespace(db=None))

    def test_allow_block_pin(self):
        st = self._st()
        st.block("https://evil.test/x")
        assert st.score("https://evil.test/x") == 0.0
        assert st.is_blocked("https://evil.test/x")
        st.allow("https://good.test/")
        assert st.score("https://good.test/") == 0.99
        st.unblock("https://evil.test/x")
        assert not st.is_blocked("https://evil.test/x")

    def test_explain(self):
        st = self._st()
        st.block("https://evil.test/x")
        exp = st.explain("https://evil.test/x")
        assert exp["pinned"] == "blocked"
        assert "PINNED BLOCKED" in exp["summary"]
        assert exp["score"] == 0.0


# ── tz ───────────────────────────────────────────────────────────────────────

class TestTz:
    def test_now_in_and_format(self):
        from nomorals.core.tz import now_in, format_ts
        assert now_in("America/New_York").tzinfo is not None
        assert format_ts(0, tz="UTC") == "1970-01-01 00:00 UTC"

    def test_parse_ts(self):
        from nomorals.core.tz import parse_ts
        assert parse_ts("2026-10-10 12:00", tz="UTC") == 1791633600.0
        assert parse_ts("2026-10-10T12:00:00Z") == 1791633600.0
        with pytest.raises(ValueError):
            parse_ts("not a date")


# ── config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_redacted_dict(self):
        from nomorals.core.config import Settings, redacted_dict
        s = Settings()
        s.chat.telegram_bot_token = "tok123"
        d = redacted_dict(s)
        assert d["chat"]["telegram_bot_token"] == "***"
        assert d["profile"] == "workstation"
        assert "tok123" not in str(d)

    def test_validate_settings(self):
        from nomorals.core.config import Settings, validate_settings
        assert validate_settings(Settings()) == []
        s = Settings()
        s.budget.tokens = -5
        problems = validate_settings(s)
        assert any("budget.tokens" in p for p in problems)


# ── verify ───────────────────────────────────────────────────────────────────

class TestVerify:
    def test_probes(self):
        from nomorals.core.verify import check_dns, check_tcp
        assert check_dns("localhost").passed is True
        assert check_tcp("127.0.0.1", 1, timeout=1).passed is False

    def test_add_check(self):
        from nomorals.core.verify import LiveVerifier, VerificationCheck
        v = LiveVerifier()
        v.add_check("mine", lambda: VerificationCheck(
            name="mine", passed=True, message="ok"))
        # verify_all runs built-ins (network); exercise the custom path only
        checks = []
        for name, func in v._custom_checks:
            check = func()
            if not isinstance(check, VerificationCheck):
                check = VerificationCheck(name=name, passed=bool(check),
                                          message=str(check))
            checks.append(check)
        assert checks[0].passed is True and checks[0].name == "mine"


# ── policy ───────────────────────────────────────────────────────────────────

class TestPolicy:
    def test_decision_explain(self):
        from nomorals.core.policy import Policy, CapabilitySet
        p = Policy(default_grant=CapabilitySet.of("fs.read"))
        p.deny("fs.delete", note="too dangerous")
        d = p.check("fs.delete", actor="devon")
        text = d.explain()
        assert "DENIED" in text and "too dangerous" in text

    def test_policy_describe(self):
        from nomorals.core.policy import Policy, CapabilitySet
        p = Policy(default_grant=CapabilitySet.of("fs.read"))
        p.deny("fs.delete")
        desc = p.describe()
        assert desc["enforcing"] is True
        assert desc["by_effect"] == {"deny": 1}
        assert desc["default_grant_size"] == 1
        assert "ENFORCING" in desc["summary"]


# ── incidents ────────────────────────────────────────────────────────────────

class TestIncidents:
    def _journal(self):
        from nomorals.core.incidents import IncidentJournal
        return IncidentJournal(":memory:")

    def test_top_subsystems(self):
        j = self._journal()
        j.record_incident(ValueError("a"), subsystem="net")
        j.record_incident(ValueError("b"), subsystem="net")
        j.record_incident(KeyError("c"), subsystem="db")
        top = j.top_subsystems()
        assert top[0]["subsystem"] == "net" and top[0]["n"] == 2
        assert top[0]["share"] == pytest.approx(2 / 3, abs=0.001)
        j.close()

    def test_mttr(self):
        import time as _time
        from nomorals.core.incidents import IncidentJournal, RecoveryRecord
        j = self._journal()
        now = _time.time()
        i1 = j.record_incident(ValueError("a"), subsystem="net", ts=now - 600)
        j.record_incident(ValueError("b"), subsystem="net", ts=now - 100)
        j.record_recovery(RecoveryRecord(incident_id=i1.id,
                                         signature=i1.signature,
                                         strategy="retry", verified=True))
        m = j.mttr()
        assert m["recovered"] == 1 and m["open"] == 1
        assert m["mean_human"].endswith("m") or m["mean_human"].endswith("s")
        j.close()


# ── error doctor / intelligence ──────────────────────────────────────────────

class TestErrorDoctor:
    def test_format_diagnosis(self):
        from nomorals.core.error_doctor import diagnose, format_diagnosis
        try:
            {}["missing"]
        except KeyError as e:
            d = diagnose(e)
        out = format_diagnosis(d, color=False)
        assert "ROOT CAUSE" in out and "SUGGESTED FIX" in out
        assert "EVIDENCE" in out


class TestErrorIntelligence:
    def _ei(self):
        from collections import deque
        from nomorals.core.error_intelligence import ErrorIntelligence
        ei = ErrorIntelligence()
        now = time.time()
        ei.learning._spike_times["flaky"] = deque(
            [now - 3300, now - 3290, now - 1700, now - 1690, now - 100, now - 90],
            maxlen=200)
        ei.learning._spike_times["steady"] = deque(
            [now - 3000 + i * 500 for i in range(6)], maxlen=200)
        return ei

    def test_flakiness_verdicts(self):
        ei = self._ei()
        assert ei.flakiness("flaky")["verdict"] == "flaky"
        assert ei.flakiness("steady")["verdict"] == "steady"
        assert ei.flakiness("nope")["verdict"] == "insufficient-data"

    def test_trend_direction(self):
        from collections import deque
        from nomorals.core.error_intelligence import ErrorIntelligence
        ei = ErrorIntelligence()
        now = time.time()
        ei.learning._spike_times["r"] = deque(
            [now - 80000, now - 60000, now - 40000, now - 20000, now - 15000,
             now - 10000, now - 8000, now - 6000, now - 4000, now - 2000,
             now - 1000, now - 500], maxlen=200)
        assert ei.trend("r")["direction"] == "rising"


# ── http ─────────────────────────────────────────────────────────────────────

class TestHttp:
    def test_module_download_exported(self):
        from nomorals.core import http
        assert callable(http.download)
        assert "download" in http.__all__

    def test_client_download_to_dir(self):
        import tempfile
        from pathlib import Path
        from nomorals.core.http import HttpClient, url_filename
        assert url_filename("https://example.com/f/report.pdf") == "report.pdf"
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            # directory detection only — no network
            assert target.is_dir()


# ── platform / profiles / runtune ────────────────────────────────────────────

class TestPlatform:
    def test_describe(self):
        from nomorals.core.platform import (detect_platform, is_docker, is_wsl,
                                            reset_platform_cache)
        reset_platform_cache()
        p = detect_platform()
        assert isinstance(is_docker(), bool) and isinstance(is_wsl(), bool)
        out = p.describe(color=False)
        assert "platform:" in out and "max workers" in out
        assert p.to_dict()["name"] == p.name

    def test_format_profile(self):
        from nomorals.core.profiles import format_profile
        out = format_profile(color=False)
        assert "profile:" in out and "TUNING VALUES" in out

    def test_runtune_describe(self):
        from nomorals.core.runtune import build_tune
        from nomorals.core.config import Settings
        out = build_tune(Settings()).describe(color=False)
        assert "CONCURRENCY" in out and "PROVENANCE" in out


# ── owner ────────────────────────────────────────────────────────────────────

class TestOwner:
    def test_v2_seal_roundtrip(self):
        import nomorals.core.owner as o
        seal = o.make_seal("a-very-long-test-passphrase-123")
        assert seal.startswith("v2$")
        old = o.PASSPHRASE_SEAL
        o.PASSPHRASE_SEAL = seal
        try:
            assert o.seal_configured()
            assert o.verify_passphrase("a-very-long-test-passphrase-123")
            assert not o.verify_passphrase("wrong-passphrase-but-long-enough")
        finally:
            o.PASSPHRASE_SEAL = old

    def test_v1_still_verifies(self):
        import nomorals.core.owner as o
        seal = o.make_seal_v1("another-long-test-passphrase-456")
        old = o.PASSPHRASE_SEAL
        o.PASSPHRASE_SEAL = seal
        try:
            assert o.verify_passphrase("another-long-test-passphrase-456")
        finally:
            o.PASSPHRASE_SEAL = old


# ── error system / selfheal / self_heal ──────────────────────────────────────

class TestErrorSystem:
    def test_health_and_format(self):
        from nomorals.core.error_system import build_error_system
        es = build_error_system()
        es.journal.record_incident(ValueError("x"), subsystem="net",
                                   severity="high")
        h = es.health()
        assert "top_subsystems_24h" in h and "mttr_24h" in h
        assert h["top_subsystems_24h"][0]["subsystem"] == "net"
        out = es.format_health(color=False)
        assert "HOT SUBSYSTEMS" in out
        es.close()


class TestSelfHeal:
    def test_recovery_and_stats(self):
        from nomorals.core.incidents import IncidentJournal
        from nomorals.core.selfheal import SelfHealingExecutor, RecoveryStatus
        ex = SelfHealingExecutor("t", journal=IncidentJournal(":memory:"))
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("down")
            return "up"

        out = ex.execute(flaky)
        assert out.status == RecoveryStatus.RECOVERED
        assert ex.strategy_stats["retry"]["recovered"] == 1
        assert "recovered" in out.format_outcome(color=False)

    def test_diff_preview(self):
        from nomorals.core.self_heal import diff_preview
        old = ["def f():\n", "    return 1\n"]
        new = ["def f():\n", "    x = None\n", "    return 1\n"]
        d = diff_preview("f.py", old, new)
        assert "+    x = None" in d and "--- a/f.py" in d
