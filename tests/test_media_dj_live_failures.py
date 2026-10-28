"""Pass #214 pins: `nomorals/media/dj_live.py` error-handling sites.

0 converts / 12 exclusions (see nm214-verify/SURVEY.md). Each site gets
a behavior pin (the documented fallback happens) and a NOT pin (the
failure is not recorded or claimed as something it is not).

No exception handling in this file at all: failures are injected with
mock side_effect / module blocks, never caught here — so the pins file
itself contributes zero scanner sites.
"""

import sys
import time
from types import SimpleNamespace
from unittest import mock

from nomorals.media import dj_live as live


def _mk_set(tmp_path):
    heard = []
    said = []
    s = live.LiveSet(
        chat_key="pins:1",
        deliver_audio=lambda p, c: heard.append((p, c)),
        deliver_text=lambda t: said.append(t),
        context=None, workdir=str(tmp_path / "work"))
    return s, heard, said


# ── 80/85/92: _dj_character chain ─────────────────────────────────────

def test_character_import_failure_returns_none():
    with mock.patch.dict(sys.modules, {"nomorals.characters": None,
                                       "nomorals.characters.seeds": None}):
        assert live._dj_character() is None


def test_none_character_does_not_claim_llm():
    p = live.DJPersona()
    p.character = None
    p._llm_fn = None
    line = p.talk(situation="hype", energy=0.9)
    assert line  # assembled, never empty
    assert p.character is None  # NOT: no fake character claimed


def test_character_get_failure_falls_through_to_seed():
    import nomorals.characters as chars_mod
    import nomorals.characters.seeds as seeds_mod
    calls = {"n": 0}

    class FakeStore:
        def get_by_name(self, name):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("db down")
            return SimpleNamespace(name="Vrede")

    with mock.patch.object(chars_mod, "CharacterStore", FakeStore), \
            mock.patch.object(seeds_mod, "seed_bank", lambda store: None):
        got = live._dj_character()
    assert got is not None and got.name == "Vrede"
    assert calls["n"] == 2  # NOT: the failure did not end the lookup


def test_character_seed_failure_returns_none():
    import nomorals.characters as chars_mod
    import nomorals.characters.seeds as seeds_mod

    def _raise(store):
        raise RuntimeError("seed down")

    with mock.patch.object(
            chars_mod, "CharacterStore",
            lambda: SimpleNamespace(get_by_name=lambda n: None)), \
            mock.patch.object(seeds_mod, "seed_bank", _raise):
        assert live._dj_character() is None


# ── 306: _talk_via_llm ────────────────────────────────────────────────

def test_llm_failure_falls_back_to_assembled_line():
    p = live.DJPersona()
    p.character = SimpleNamespace(
        speak=lambda q, fn: (_ for _ in ()).throw(RuntimeError("llm down")))
    p._llm_fn = lambda s: s
    ctx = {"situation": "hype", "track": "", "style": "", "energy": 0.9,
           "requester": "", "extra": ""}
    assert p._talk_via_llm(ctx) == ""
    assert p.talk(situation="hype", energy=0.9)  # assembled, never empty


def test_empty_talk_line_is_never_delivered(tmp_path):
    s, heard, said = _mk_set(tmp_path)
    s._dj_says("")
    assert heard == [] and said == []  # NOT: "" is never sent as a line


# ── 480: notify ────────────────────────────────────────────────────────

def test_notify_sensor_failure_does_not_raise(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    with mock.patch.object(s.sensor, "note_message",
                           side_effect=RuntimeError("sensor down")):
        s.notify("hello", sender="ada")  # must not raise


def test_notify_failure_records_no_event(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    with mock.patch.object(s.sensor, "note_message",
                           side_effect=RuntimeError("sensor down")):
        s.notify("hello", sender="ada")
    assert s.sensor._events == []  # NOT: the dropped message is not recorded


# ── 564: _analyze + fit-skip ───────────────────────────────────────────

def test_analyze_engine_failure_returns_none(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    with mock.patch.dict(sys.modules, {"nomorals.media.dj_engine": None}):
        assert s._analyze("/tmp/nope.wav", SimpleNamespace(
            title="T", tempo=None, key="", mode="")) is None


class _FakeSong:
    def __init__(self, path):
        self.audio_path = path
        self.title = "Req Song"
        self.tempo = None
        self.key = ""
        self.mode = ""


class _Creator:
    def __init__(self, context):
        self.context = context

    def compose(self, prompt, **kw):
        return _FakeSong(kw.get("_path", ""))


def test_request_honored_without_analysis_but_analysis_stays_none(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    wav = tmp_path / "req.wav"
    wav.write_bytes(b"")

    class _LocalCreator(_Creator):
        def compose(self, prompt, **kw):
            return _FakeSong(str(wav))

    cur = live._CrateTrack(path="/tmp/cur.wav", title="Cur",
                           style="afrobeats", analysis=None)
    req = live.TrackRequest(query="play amapiano", requester="ada")
    with mock.patch.dict(sys.modules, {"nomorals.media.music":
                                       SimpleNamespace(MusicCreator=_LocalCreator)}):
        with mock.patch.object(live.LiveSet, "_analyze", return_value=None):
            verdict = s._judge_request(req, cur)
    assert verdict is not None and verdict.is_request
    assert verdict.analysis is None  # NOT: no fabricated analysis


# ── 579: _order_crate ──────────────────────────────────────────────────

def test_order_crate_arc_failure_returns_original_order(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    t1 = live._CrateTrack(path="/tmp/a.wav", title="A", style="afrobeats",
                          analysis=SimpleNamespace(energy=0.9))
    t2 = live._CrateTrack(path="/tmp/b.wav", title="B", style="house",
                          analysis=SimpleNamespace(energy=0.2))
    s._crate = [t1, t2]
    fake_engine = SimpleNamespace(
        plan_energy_arc=lambda xs: (_ for _ in ()).throw(
            RuntimeError("arc down")))
    with mock.patch.dict(sys.modules, {"nomorals.media.dj_engine": fake_engine}):
        order = s._order_crate()
    assert order == [t1, t2]  # NOT: same objects, no fabricated arc order


# ── 612: _wait_track Empty slice ───────────────────────────────────────

def test_wait_track_empty_slice_returns_without_raise(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    t0 = time.time()
    s._wait_track(0.05)  # empty cmd queue: Empty slices only
    assert time.time() - t0 >= 0.05
    assert s._cmd.empty()  # NOT: no command consumed
    assert not s._stop.is_set()  # NOT: the wait did not stop the set


# ── 719: _pivot_crate ──────────────────────────────────────────────────

class _EvilAnalysis:
    @property
    def energy(self):
        raise RuntimeError("analysis corrupt")


def test_pivot_sort_failure_leaves_order_unchanged(tmp_path):
    s, _, _ = _mk_set(tmp_path)
    e1 = live._CrateTrack(path="/tmp/e1.wav", title="E1", style="house",
                          analysis=_EvilAnalysis())
    e2 = live._CrateTrack(path="/tmp/e2.wav", title="E2", style="house",
                          analysis=SimpleNamespace(energy=0.1))
    s._crate = [e1, e2]
    s._played = []
    s._pivot_crate(None)  # must not raise
    assert [t.title for t in s._crate] == ["E1", "E2"]
    assert s._played == []  # NOT: the failed pivot marks nothing played


# ── 740/746/750: _dj_says chain ────────────────────────────────────────

def test_dj_says_render_failure_falls_back_to_text(tmp_path):
    s, heard, said = _mk_set(tmp_path)
    with mock.patch.object(live.DJPersona, "render_voice",
                           side_effect=RuntimeError("tts down")):
        s._dj_says("hello room")
    assert said == ["🎙️ DJ Vrede: hello room"]
    assert heard == []  # NOT: no voice attempt recorded as delivered


def test_dj_says_audio_failure_falls_back_to_text_once(tmp_path):
    s, heard, said = _mk_set(tmp_path)
    with mock.patch.object(live.DJPersona, "render_voice",
                           return_value="/tmp/talk.wav"):
        with mock.patch.object(s, "_deliver_audio",
                               side_effect=RuntimeError("gateway down")):
            s._dj_says("hello again")
    assert said == ["🎙️ DJ Vrede: hello again"]  # exactly once, no double
    assert heard == []


def test_dj_says_both_deliverers_failing_raises_nothing(tmp_path):
    s, heard, said = _mk_set(tmp_path)
    with mock.patch.object(live.DJPersona, "render_voice",
                           side_effect=RuntimeError("tts down")):
        with mock.patch.object(s, "_deliver_text",
                               side_effect=RuntimeError("gateway down")):
            s._dj_says("lost line")  # must not raise
    assert heard == [] and said == []  # NOT: nothing claimed as delivered
