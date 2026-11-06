"""Targeted-fix Pass 4: small true defects, batched by organ.

Each test is labelled TEETH (fails on the pristine tree for the
defect itself) or PIN (passes on both trees — the preserved
behaviour around the fix). Rows whose proving test is a converted
pin live in their own files and are not duplicated here: collab
SCORE parse (test_books_collab_failures.py test_429), weakness
vestigial commits (test_autonomy_weakness_failures.py *_commit
pins), music _collab save honesty (test_tools_music_failures.py
test_347 — its converse mock now returns a real Dialogue shape).
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import tempfile
import time
import types
from array import array
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.accounts.temp_sms as ts
import nomorals.api.mcp_server as mcp_mod
import nomorals.builders_proxy as proxy_mod
import nomorals.characters as characters_mod
import nomorals.characters.dialogue as dialogue_mod
import nomorals.core.decoder as decoder_mod
import nomorals.games.achievements as ach_mod
import nomorals.health.form as form_mod
import nomorals.health.patterns as patterns_mod
import nomorals.llm.cost_display as cd
import nomorals.media.song_draft as song_draft_mod
import nomorals.media_edit.videos as videos_mod
import nomorals.tools.hashcrack as hashcrack_mod
import nomorals.tools.music as music_mod
from nomorals.autonomy import failures as wf
from nomorals.autonomy import weakness as wk
from nomorals.games.gear import GearStore
from nomorals.health.drift import DriftMonitor
from nomorals.integrations.payment_integration import (
    PaymentError,
    PaymentIntegration,
)
from nomorals.tools.hashcrack import CrackResult

# nomorals.audio.fingerprint the MODULE — a plain attribute import
# binds the package's fingerprint() function instead (the quirk
# documented in test_audio_fingerprint_failures.py)
fp_mod = importlib.import_module("nomorals.audio.fingerprint")


# ── 1.1 music _collab: converse called with its real signature ───

def test_music_collab_calls_converse_with_real_signature():
    """TEETH: pristine passes context=ctx to a converse() that has
    no context parameter (and char in the brain_persona slot), so
    every collab died with TypeError -> 'collab failed: ...'."""
    key = "pass4-collab"
    draft = SimpleNamespace(topic="rain", style="pop",
                            performance_notes="")
    music_mod._ACTIVE_DRAFTS[key] = draft
    char = SimpleNamespace(name="Zara")
    store = SimpleNamespace(get=lambda name: char)
    calls = []

    def fake_converse(brain_persona, c, opener, suggest, rounds=3,
                      memories="", graph=None, learn_store=None,
                      candidates=1):
        calls.append((brain_persona, c, opener, suggest, rounds))
        return SimpleNamespace(turns=[
            SimpleNamespace(speaker="Devon", text=opener),
            SimpleNamespace(speaker=c.name,
                            text="sixteen bars of rain")])

    try:
        with mock.patch.object(characters_mod, "CharacterStore",
                               lambda: store), \
             mock.patch.object(dialogue_mod, "converse", fake_converse), \
             mock.patch.object(song_draft_mod, "save_draft",
                               lambda d: None):
            out = music_mod._collab("Zara verse", key,
                                    SimpleNamespace(suggest=None))
        assert "sixteen bars of rain" in out
        assert "Zara wrote the verse" in out
        assert "collab failed" not in out
        assert "[Zara — verse]" in draft.performance_notes
        assert "sixteen bars of rain" in draft.performance_notes
        persona, seen_char, _opener, _suggest, _rounds = calls[0]
        assert isinstance(persona, str) and seen_char is char
    finally:
        music_mod._ACTIVE_DRAFTS.pop(key, None)


# ── 1.2 leads _log: the module logger exists ─────────────────────

def test_leads_set_public_reply_without_name_or_dm_pointer(tmp_path):
    """TEETH: a template with neither {name} nor a DM pointer took
    the _log.debug branch — _log was never defined, so the call
    raised NameError instead of storing the template."""
    from nomorals.social.leads import LeadStore

    store = LeadStore(db_path=str(tmp_path / "leads.db"))
    trig = store.add_trigger("instagram", "post1", "VAULT")
    assert store.set_public_reply(
        trig.trigger_id, "Sent — tell everyone!") is True


def test_leads_set_public_reply_with_name_still_works(tmp_path):
    """PIN: the {name} template path behaves exactly as before."""
    from nomorals.social.leads import LeadStore

    store = LeadStore(db_path=str(tmp_path / "leads.db"))
    trig = store.add_trigger("instagram", "post1", "VAULT")
    assert store.set_public_reply(
        trig.trigger_id, "Sent it, {name} — check your DMs") is True


# ── 1.3 achievements _log: same class, games package ─────────────

def test_achievements_update_game_stats_db_failure_is_quiet():
    """TEETH: when the stats write failed, the except handler's
    _log.debug raised NameError into the game-end path."""
    class BadDB:
        def execute(self, *a, **k):
            raise RuntimeError("db down")

    # must return quietly — debug-logged, never raised
    assert ach_mod.update_game_stats(
        BadDB(), "p1", "chess", True, 10) is None


def test_achievements_update_game_stats_happy_path():
    """PIN: a working store records the game exactly as before."""
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    ach_mod.update_game_stats(db, "p1", "chess", True, 42)
    stats = ach_mod.get_game_stats(db, "p1")
    assert stats and stats[0]["game"] == "chess"
    assert stats[0]["played"] == 1 and stats[0]["won"] == 1


# ── 1.10 hashcrack: learned reflects learn_hash's verdict ───────

class _FakeEngine:
    def __init__(self, targets, **kw):
        self._targets = list(targets)

    def run(self):
        return CrackResult(
            targets=list(self._targets),
            found={self._targets[0]: "opensesame"},
            remaining=[], algo="md5", mode="hybrid", tested=1,
            elapsed=0.1, capped=False, backend="python")


def _crack_with_learn(monkeypatch, learn_ok):
    monkeypatch.setattr(hashcrack_mod, "Engine", _FakeEngine)
    monkeypatch.setattr(decoder_mod, "known_hash_lookup_chained",
                        lambda db, t: None)
    monkeypatch.setattr(decoder_mod, "learn_hash",
                        lambda *a, **k: learn_ok)
    return hashcrack_mod.crack_hash(
        "d41d8cd98f00b204e9800998ecf8427e", db=object())


def test_hashcrack_learned_false_when_learn_fails(monkeypatch):
    """TEETH: pristine set res.learned = True unconditionally,
    even when every learn_hash call returned False."""
    res = _crack_with_learn(monkeypatch, False)
    assert res.found and res.learned is False


def test_hashcrack_learned_true_when_learn_succeeds(monkeypatch):
    """PIN: a fully learned crack still reports learned=True."""
    res = _crack_with_learn(monkeypatch, True)
    assert res.learned is True


# ── 1.9 form: analyze_gait all-unusable reads are unavailable ───

def _gait(monkeypatch, vision_text):
    monkeypatch.setattr(form_mod, "_frames_for",
                        lambda p: (["f1", "f2"], ""))
    return form_mod.analyze_gait(
        "run.mp4", vision=lambda frame, prompt: vision_text)


def test_gait_unusable_vision_reads_are_unavailable(monkeypatch):
    """TEETH: pristine returned available=True with no issues for
    frames the vision model said nothing usable about — and
    format() rendered that as 'no risk flags spotted'."""
    got = _gait(monkeypatch, "the runner wears a red top, blue shoes")
    assert got.available is False
    assert "didn't return usable gait reads" in got.note
    assert "no risk flags spotted" not in got.format()


def test_gait_all_pass_reads_available_and_clean(monkeypatch):
    """PIN: real per-risk PASS reads are a legitimate clean read."""
    text = "\n".join(f"{rid}: PASS — steady"
                     for rid in form_mod.GAIT_RISKS)
    got = _gait(monkeypatch, text)
    assert got.available is True and got.issues == []


def test_gait_flag_read_surfaces_issue(monkeypatch):
    """PIN: a FLAG read still surfaces its issue."""
    rids = list(form_mod.GAIT_RISKS)
    lines = [f"{rid}: PASS — steady" for rid in rids]
    lines[0] = f"{rids[0]}: FLAG — foot lands way ahead of the hips"
    got = _gait(monkeypatch, "\n".join(lines))
    assert got.available is True and len(got.issues) == 1
    assert got.issues[0].risk == rids[0]


# ── 2.9 builders_proxy: the cryptography rung lives again ───────

def test_crypto_rung_generates_cert_without_openssl(tmp_path):
    """TEETH: on cryptography >= 50 the BasicConstraints(pathlen=)
    kwarg raises TypeError, so the preferred rung was dead and the
    openssl fallback silently carried every generation. With
    openssl 'missing', pristine raises RuntimeError; the fixed
    rung generates through cryptography itself."""
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    with mock.patch.object(proxy_mod.subprocess, "run",
                           side_effect=FileNotFoundError("no openssl")):
        got = proxy_mod.ensure_self_signed_cert(
            str(cert), str(key), days=30)
    assert got == (str(cert), str(key))
    from cryptography import x509

    loaded = x509.load_pem_x509_certificate(cert.read_bytes())
    bc = loaded.extensions.get_extension_for_class(
        x509.BasicConstraints).value
    assert bc.ca is False


# ── 2.17 weakness: no vestigial commit() on the real Database ───

@pytest.fixture()
def _nm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    wf.clear_failures()
    yield
    wf.clear_failures()


def test_weakness_success_records_no_commit_failures(tmp_path, _nm_home):
    """TEETH (production-shaped): the real Database is autocommit
    and has no commit() — pristine weakness code called it after
    every successful write and recorded a spurious *_commit
    failure each time."""
    from nomorals.storage.db import Database

    db = Database(tmp_path / "wk.db")
    wk.ensure_schema(db)
    wid = wk.report_weakness(db, "tool_failure", "toolx",
                             {"error": "boom"})
    assert wid
    assert wk.record_fix(db, wid, "do the thing") is True
    out = wk.record_degradation_sample(db, "subj", "metric", 1.0)
    assert out["subject"] == "subj"
    sites = [f.site for f in wf.recent_failures()]
    assert not any(s.startswith("weakness.") for s in sites)


# ── 2.11 drift: pings_today guarded like the rest of the module ─

def test_drift_pings_today_db_failure_returns_zero(tmp_path):
    """TEETH: pristine pings_today raised unguarded on DB failure,
    and maybe_notify calls it outside its own try block."""
    mon = DriftMonitor(None, db_path=tmp_path / "drift.db")

    class BadDB:
        def query(self, *a, **k):
            raise RuntimeError("db down")

    mon._db = BadDB()
    assert mon.pings_today() == 0


def test_drift_pings_today_counts_rows(tmp_path):
    """PIN: the healthy count is unchanged."""
    import nomorals.health.drift as drift_mod

    mon = DriftMonitor(None, db_path=tmp_path / "drift.db")
    now = time.time()
    mon._db.execute(
        "INSERT INTO drift_pings (day, severity, ts) VALUES (?,?,?)",
        (drift_mod._day(now), "act", now))
    assert mon.pings_today() == 1


# ── 2.12 patterns: _factor_days honours 'never raises' ──────────

def test_patterns_factor_days_survives_nonstring_text():
    """TEETH: a non-string event text made the loop-head
    rx.search raise TypeError out of _factor_days — and out of
    detect_lag_patterns, whose docstring promises 'Never raises'."""
    ev = SimpleNamespace(text=b"slept 5h", ts=time.time())
    assert patterns_mod._factor_days([ev]) == {}


def test_patterns_detect_lag_patterns_never_raises_on_bad_text():
    """TEETH: the same defect through the public caller."""
    ev = SimpleNamespace(text=b"deadline looming", ts=time.time(),
                         event_type="note")
    tl = SimpleNamespace(timeline=lambda since, limit: [ev])
    assert patterns_mod.detect_lag_patterns(tl) == []


def test_patterns_factor_days_normal_text_pin():
    """PIN: ordinary text still factors exactly as before."""
    ev = SimpleNamespace(text="brutal deadline at work today",
                         ts=time.time())
    days = patterns_mod._factor_days([ev])
    assert "stress" in days


# ── 2.13 payment: ETH construction failure is a PaymentError ────

def test_payment_eth_construction_failure_is_payment_error(monkeypatch):
    """TEETH: the BTC branch wraps construction + fetch in a
    PaymentError guard; the ETH branch built Web3(...) outside
    any guard, so a construction error escaped raw."""
    fake = types.ModuleType("web3")

    class BoomWeb3:
        def __init__(self, *a, **k):
            raise RuntimeError("provider ctor boom")

        @staticmethod
        def HTTPProvider(*a, **k):
            return object()

    fake.Web3 = BoomWeb3
    monkeypatch.setitem(sys.modules, "web3", fake)
    gw = PaymentIntegration(mock.Mock(), mock.Mock())
    with pytest.raises(PaymentError):
        asyncio.run(gw.track_tx("0xabc", "ETH"))


# ── 2.18 temp_sms: only an href-shaped group(1) is an inbox ─────

def test_temp_sms_pattern2_number_has_no_inbox_id():
    """TEETH: pattern 2's group(1) is the number text, not a link,
    but it was stored as inbox_id — listed and grabbable, yet
    get_messages could never poll it."""
    prov = ts.SevenSimProvider()
    page = "<p>Call me</p> +442071234567</a>"
    with mock.patch.object(ts, "_fetch", return_value=page):
        numbers = prov.list_numbers()
    assert len(numbers) == 1
    assert numbers[0].number == "+442071234567"
    assert numbers[0].inbox_id == ""


def test_temp_sms_pattern1_number_keeps_inbox_url():
    """PIN: an href-shaped group(1) still becomes the inbox URL."""
    prov = ts.SevenSimProvider()
    page = '<a href="/number/9">+1 202 555 0134</a>'
    with mock.patch.object(ts, "_fetch", return_value=page):
        numbers = prov.list_numbers()
    assert numbers[0].inbox_id == "https://7sim.net/number/9"


# ── 2.19 gear: migration pops only what was granted ─────────────

def _gear_db():
    from nomorals.games.players import PlayerStore
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    players = PlayerStore(db)
    prof = players.get("p1")
    prof.items["gear_sword"] = 3
    players._write(prof)
    return db


def test_gear_migration_partial_grant_keeps_remainder():
    """TEETH: pristine broke out of the grant loop on the first
    failure, then popped the FULL legacy count anyway — the
    ungranted remainder vanished."""
    from nomorals.games.players import PlayerStore

    db = _gear_db()
    store = GearStore(db)
    calls = []

    def flaky_grant(player_key, slug):
        calls.append(slug)
        if len(calls) > 1:
            raise RuntimeError("grant boom")
        return None

    store.grant = flaky_grant
    moved = store._migrate_legacy("p1")
    assert moved == 1
    after = PlayerStore(db).get("p1")
    assert after.items.get("gear_sword") == 2


def test_gear_migration_full_grant_pops():
    """PIN: when every grant lands, the legacy key is popped."""
    from nomorals.games.players import PlayerStore

    db = _gear_db()
    store = GearStore(db)
    store.grant = lambda player_key, slug: None
    moved = store._migrate_legacy("p1")
    assert moved == 3
    after = PlayerStore(db).get("p1")
    assert "gear_sword" not in after.items


# ── 2.8 debate: the fallback's verdict is always bound ──────────

def test_render_debate_hostile_verdict_still_renders_fallback():
    """TEETH: verdict was bound inside the try, so a failure on
    the try's first line made the fallback f-string itself raise
    NameError — against render_debate's 'Never raises' contract."""
    import nomorals.agents.debate as debate_mod

    class Hostile:
        @property
        def verdict(self):
            raise RuntimeError("sick verdict")

    out = debate_mod.render_debate(Hostile())
    assert out == "debate result (Hostile): ?"


# ── 2.15 mcp: a sick DB is not RESOURCE_NOT_FOUND ───────────────

def _mcp_server(db):
    srv = object.__new__(mcp_mod.MCPServer)
    # two_tier is a read-only property backed by _memory
    srv._memory = SimpleNamespace(db=db)
    return srv


def test_mcp_read_fact_db_failure_is_internal_error():
    """TEETH: pristine returned None for a DB exception exactly
    as for a missing fact, so an existing fact on a sick DB was
    reported RESOURCE_NOT_FOUND."""
    class BadDB:
        def query(self, *a, **k):
            raise RuntimeError("db down")

    with pytest.raises(mcp_mod.MCPError) as ei:
        _mcp_server(BadDB())._read_fact("f1")
    assert ei.value.code == mcp_mod.INTERNAL_ERROR


def test_mcp_read_fact_missing_returns_none():
    """PIN: a genuinely missing fact still returns None."""
    class EmptyDB:
        def query(self, *a, **k):
            return []

    assert _mcp_server(EmptyDB())._read_fact("nope") is None


def test_mcp_read_fact_found_returns_dict():
    """PIN: a found fact renders exactly as before."""
    row = {"id": "f1", "text": "sky is blue", "confidence": 0.9,
           "valid_from": 100.0, "valid_to": None}

    class RowDB:
        def query(self, *a, **k):
            return [row]

    got = _mcp_server(RowDB())._read_fact("f1")
    assert got["id"] == "f1" and got["text"] == "sky is blue"
    assert got["valid_to"] is None


# ── 1.8 fingerprint: rmdir survives a failed remove ─────────────

def test_fingerprint_tmp_dir_removed_when_file_remove_fails(
        tmp_path, monkeypatch):
    """TEETH: the finally ran remove + rmdir in one try — when
    remove raised (here: the decoded file is absent, so it raises
    FileNotFoundError), rmdir was skipped and the empty tmp dir
    leaked. Pristine leaves a devon-fp-* dir behind."""
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(fp_mod, "_ffmpeg_path", lambda: "/bin/true")
    monkeypatch.setattr(videos_mod, "run_ffmpeg",
                        lambda *a, **k: None)
    monkeypatch.setattr(fp_mod, "_read_wav",
                        lambda p: (array("f", [0.0] * 64), 22050))
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"not really audio - decode is stubbed")
    samples, sr = fp_mod.read_mono(str(clip),
                                   target_sr=22050, max_seconds=0)
    assert sr == 22050 and len(samples) == 64
    assert list(tmp_path.glob("devon-fp-*")) == []


# ── by-catch: /cost budget — the advertised form parses ─────────

def test_cost_advertised_budget_form_sets_budget():
    """TEETH: _usage() advertises '/cost budget $0.50', but
    control_cost stripped the 'budget' keyword before calling
    parse_budget_nl — whose regexes require that keyword — so
    the advertised form returned the usage text back."""
    ctx = mock.Mock()
    out = cd.control_cost("budget $0.50", context=ctx)
    assert "research budget set" in out
    ctx.set_research_budget.assert_called_once_with(0.5)


def test_cost_budget_spend_form_still_works():
    """PIN: the longer NL form keeps working."""
    ctx = mock.Mock()
    out = cd.control_cost("budget spend $0.50", context=ctx)
    assert "research budget set" in out
    ctx.set_research_budget.assert_called_once_with(0.5)
