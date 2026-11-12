"""Tests for nomorals.media.dj_desk — the DJ DESK you play.

The desk is the performance half of the DJ: beat grid + downbeat, hot
cues, sample pads, loops, two decks, crossfader, tempo sync — and a
live-set runner that plays a show THROUGH the desk on real rendered
audio. Ground truth is synthesized click-track WAVs with KNOWN BPM and
beat times: the grid must match them, cue jumps must land on beats, pad
triggers must quantize, loop lengths must be exact beats.
"""

import math
import struct
import wave
from array import array
from pathlib import Path

import pytest

from nomorals.media.dj_desk import (
    BeatGrid,
    CueBank,
    Deck,
    DeskStore,
    DJDesk,
    FXChain,
    Library,
    PadBank,
    attach_desk_plan,
    build_cue_training_rows,
    compute_waveform,
    control_dj_desk,
    detect_beat_grid,
    detect_downbeat,
    fit_cue_suggester,
    madmom_available,
    madmom_beat_track,
    run_live_set,
    suggest_cues,
)
from nomorals.media.dj_engine import TrackAnalysis, _goertzel_mag

SR = 22050


# ── synthesis helpers (known ground truth) ───────────────────────────────────

def _click_track(bpm: float, seconds: float = 12.0, *,
                 accent_every: int = 0, freq: float = 180.0,
                 marker_beat: int = -1) -> array:
    """Clicks exactly on the beat grid; every `accent_every`-th beat louder.

    `marker_beat` is one beat struck extra-loud (amp 1.5) so a loop
    replay or a cue jump can be told apart from the plain grid.
    """
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    beat_n = int(SR * 60.0 / bpm)
    for k, start in enumerate(range(0, n, beat_n)):
        amp = 1.0 if (accent_every and k % accent_every == 0) else 0.6
        if k == marker_beat:
            amp = 1.5
        for i in range(min(2205, n - start)):
            t = i / SR
            out[start + i] += amp * math.exp(-t * 60.0) * math.sin(
                2 * math.pi * freq * t)
    return out


def _structured_track(bars: int, bpm: float, bar_energy: list[float],
                      *, freq: float = 180.0) -> tuple[array, set[int]]:
    """One bar = 4 beats of clicks + a sine bed at the bar's energy.

    Cue bars are the ground truth: bar 0 (set start) plus every bar
    whose energy jumps >= 0.4 over the previous bar (a drop).
    """
    beat_s = 60.0 / bpm
    bar_s = beat_s * 4
    n = int(SR * bars * bar_s)
    out = array("d", [0.0]) * n
    cues: set[int] = {0}
    for b in range(bars):
        e = bar_energy[b]
        if b > 0 and e - bar_energy[b - 1] >= 0.4:
            cues.add(b)
        start = int(b * bar_s * SR)
        for beat in range(4):
            cs = start + int(beat * beat_s * SR)
            for i in range(min(2205, n - cs)):
                t = i / SR
                out[cs + i] += e * math.exp(-t * 60.0) * math.sin(
                    2 * math.pi * freq * t)
        # sine bed at the bar's energy
        for i in range(start, min(n, start + int(bar_s * SR))):
            t = i / SR
            out[i] += 0.25 * e * math.sin(2 * math.pi * 110.0 * t)
    return out, cues


def _write_wav(path, samples: array) -> str:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack(
            f"<{len(samples)}h",
            *[max(-32768, min(32767, int(s * 32767))) for s in samples]))
    return str(path)


def _click_times(samples: array, sr: int = SR, thresh: float = 0.25,
                 min_gap_s: float = 0.25) -> list[float]:
    """Onset times of loud clicks: threshold crossings with a refractory gap."""
    times: list[float] = []
    last = -min_gap_s
    for i, v in enumerate(samples):
        t = i / sr
        if abs(v) >= thresh and t - last >= min_gap_s:
            times.append(round(t, 3))
            last = t
    return times


def _grid(bpm: float = 120.0, seconds: float = 12.0) -> BeatGrid:
    return detect_beat_grid(_click_track(bpm, seconds), SR, bpm)


# ── beat grid ────────────────────────────────────────────────────────────────

def test_grid_matches_click_track():
    grid = _grid(120.0, 12.0)
    assert grid.bpm == 120.0
    assert len(grid.beat_times) >= 20
    truth = [round(i * 0.5, 3) for i in range(24)]
    matched = sum(
        1 for t in truth
        if any(abs(b - t) <= 0.04 for b in grid.beat_times))
    assert matched >= 21, f"matched {matched}/24, grid={grid.beat_times[:6]}"
    assert grid.confidence >= 0.8


def test_grid_matches_98_bpm():
    grid = _grid(98.0, 12.0)
    truth = [round(i * 60.0 / 98.0, 3) for i in range(19)]
    matched = sum(
        1 for t in truth
        if any(abs(b - t) <= 0.04 for b in grid.beat_times))
    assert matched >= 17, f"matched {matched}/19"
    assert grid.confidence >= 0.8


def test_grid_unknown_on_silence():
    grid = detect_beat_grid(array("d", [0.0]) * SR * 8, SR, 120.0)
    assert grid.beat_times == []
    assert grid.confidence == 0.0
    assert not grid.known


def test_grid_unknown_without_bpm():
    grid = detect_beat_grid(_click_track(120.0), SR, None)
    assert grid.beat_times == []
    assert grid.confidence == 0.0
    assert not grid.known


def test_grid_nearest_beat():
    grid = _grid(120.0, 12.0)
    assert abs(grid.nearest_beat(1.3) - 1.5) <= 0.01
    assert abs(grid.nearest_beat(1.2) - 1.0) <= 0.01
    assert abs(grid.next_beat(1.3) - 1.5) <= 0.01
    assert abs(grid.next_beat(1.0) - 1.5) <= 0.01


# ── downbeat ─────────────────────────────────────────────────────────────────

def test_downbeat_accented():
    samples = _click_track(120.0, 12.0, accent_every=4)
    grid = detect_beat_grid(samples, SR, 120.0)
    idx, conf = detect_downbeat(samples, SR, grid)
    assert idx == 0, f"downbeat {idx} conf {conf}"
    # accents are 1.0 vs 0.6, so the true margin is 0.4
    assert conf >= 0.3


def test_downbeat_uniform_is_unknown():
    samples = _click_track(120.0, 12.0)
    grid = detect_beat_grid(samples, SR, 120.0)
    idx, conf = detect_downbeat(samples, SR, grid)
    assert idx is None, f"uniform clicks must not fake a downbeat (got {idx})"


# ── hot cues ─────────────────────────────────────────────────────────────────

def test_cue_set_and_jump_lands_on_beat():
    grid = _grid(120.0, 12.0)
    bank = CueBank()
    bank.set_cue("drop", 8, grid)
    assert bank.jump("drop") == grid.beat_times[8]
    assert abs(bank.jump("drop") - 4.0) <= 0.001  # beat 8 at 120 BPM
    name, beat, t = bank.list_cues()[0]
    assert (name, beat) == ("drop", 8) and abs(t - 4.0) <= 0.001


def test_cue_rejects_out_of_range_beat():
    grid = _grid(120.0, 12.0)
    bank = CueBank()
    with pytest.raises(ValueError):
        bank.set_cue("too-far", 999, grid)


def test_cue_needs_a_grid():
    bank = CueBank()
    unknown = BeatGrid(bpm=120.0)
    with pytest.raises(ValueError):
        bank.set_cue("drop", 4, unknown)


def test_cue_persists_in_store(tmp_path):
    store = DeskStore(str(tmp_path / "desk.json"))
    grid = _grid(120.0, 12.0)
    bank = CueBank()
    bank.set_cue("drop", 8, grid)
    bank.set_cue("break", 16, grid)
    store.put_track("track-a", grid, bank)
    store.save()
    reloaded = DeskStore(str(tmp_path / "desk.json"))
    reloaded.load()
    grid2, bank2 = reloaded.get_track("track-a")
    assert grid2 is not None and len(grid2.beat_times) == len(grid.beat_times)
    assert bank2 is not None
    assert abs(bank2.jump("drop") - 4.0) <= 0.001
    assert abs(bank2.jump("break") - 8.0) <= 0.001


def test_store_corrupt_loads_empty(tmp_path):
    path = tmp_path / "desk.json"
    path.write_text("{not json,,,", encoding="utf-8")
    store = DeskStore(str(path))
    store.load()  # never raises
    assert store.get_track("anything") == (None, None)


# ── pads ─────────────────────────────────────────────────────────────────────

def test_pad_bank_slots():
    bank = PadBank()
    assert bank.slots == ["kick", "snare", "clap", "hat", "airhorn",
                          "custom1", "custom2", "custom3"]
    for slot in ("kick", "snare", "clap", "hat", "airhorn"):
        s = bank.sample(slot, SR)
        assert len(s) > 1000, f"{slot} sample too short"


def test_pad_trigger_quantizes_to_grid():
    grid = _grid(120.0, 12.0)
    bank = PadBank()
    ev = bank.trigger("kick", 1.3, grid)
    assert abs(ev.scheduled_s - 1.5) <= 0.01  # next beat at 120 BPM
    ev2 = bank.trigger("hat", 1.0, grid)
    assert abs(ev2.scheduled_s - 1.5) <= 0.01  # on-beat waits for NEXT
    ev3 = bank.trigger("clap", 0.2, None)
    assert ev3.scheduled_s == 0.2  # no grid: unquantized, honest


def test_pad_trigger_unknown_raises():
    bank = PadBank()
    with pytest.raises(ValueError):
        bank.trigger("cowbell", 1.0, None)


def test_pad_custom_sample_loads(tmp_path):
    wav = _write_wav(tmp_path / "one.wav", _click_track(120.0, 1.0))
    bank = PadBank()
    bank.load_custom("custom1", wav)
    s = bank.sample("custom1", SR)
    assert len(s) > SR // 2


# ── loops ────────────────────────────────────────────────────────────────────

def test_loop_repeats_region_exactly():
    samples = _click_track(120.0, 12.0)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0), samples)
    deck.play()
    deck.set_loop(4, 4)  # beats 4..8, 4 beats long
    out = deck.render(6.0)  # 12 beats out: 4 live + loop twice
    clicks = _click_times(out)
    # loop window is beats 4..12 of output: 8 clicks, not 4
    in_window = [t for t in clicks if 2.0 <= t < 6.0]
    assert len(in_window) == 8, f"loop did not repeat: {in_window}"


def test_loop_roll_resumes_at_ghost():
    samples = _click_track(120.0, 12.0)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0), samples)
    deck.play()
    deck.set_loop(4, 4, roll=True)
    deck.render(6.0)  # 12 beats: 4 live + 4 rolled + 4 rolled (ghost runs on)
    deck.release_loop()
    # ghost playhead kept moving: 4 beats in + 8 beats rendered = beat 12
    assert abs(deck.position_beats() - 12.0) < 0.01


def test_loop_release_plain_continues_at_loop_end():
    samples = _click_track(120.0, 12.0)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0), samples)
    deck.play()
    deck.set_loop(4, 4)
    deck.render(6.0)
    deck.release_loop()
    assert abs(deck.position_beats() - 8.0) < 0.01


# ── decks: tempo, sync, pause, crossfader ────────────────────────────────────

def test_deck_tempo_fader():
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0),
              _click_track(120.0, 12.0))
    deck.set_rate(1.06)
    assert abs(deck.effective_bpm - 127.2) < 0.01


def test_deck_sync_matches_other_deck():
    a = Deck("A")
    a.load(TrackAnalysis(bpm=120.0, duration_s=12.0),
           _click_track(120.0, 12.0))
    b = Deck("B")
    b.load(TrackAnalysis(bpm=126.0, duration_s=12.0),
           _click_track(126.0, 12.0))
    assert b.sync_to(a) is True
    assert abs(b.effective_bpm - 120.0) < 0.6


def test_deck_sync_out_of_range_is_honest():
    a = Deck("A")
    a.load(TrackAnalysis(bpm=120.0, duration_s=12.0),
           _click_track(120.0, 12.0))
    b = Deck("B")
    b.load(TrackAnalysis(bpm=200.0, duration_s=12.0),
           _click_track(200.0, 12.0))
    assert b.sync_to(a) is False
    assert b.effective_bpm == 200.0  # untouched, not faked


def test_deck_pause_holds_position():
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0),
              _click_track(120.0, 12.0))
    deck.play()
    deck.render(1.0)
    deck.pause()
    pos = deck.position_s
    out = deck.render(1.0)
    assert deck.position_s == pos
    assert max((abs(v) for v in out), default=0.0) == 0.0


def test_crossfader_ends():
    desk = DJDesk()
    desk.load_deck("A", TrackAnalysis(bpm=120.0, duration_s=12.0),
                   _click_track(120.0, 12.0, freq=180.0))
    desk.load_deck("B", TrackAnalysis(bpm=120.0, duration_s=12.0),
                   _click_track(120.0, 12.0, freq=320.0))
    desk.deck_a.play()
    desk.deck_b.play()
    desk.crossfader(0.0)
    out_a = desk.render(2.0)
    ga = _goertzel_mag(out_a, 0, len(out_a), 180.0, SR)
    gb = _goertzel_mag(out_a, 0, len(out_a), 320.0, SR)
    assert ga > 3 * gb, "xf=0 must be deck A only"
    desk.crossfader(1.0)
    out_b = desk.render(2.0)
    ga2 = _goertzel_mag(out_b, 0, len(out_b), 180.0, SR)
    gb2 = _goertzel_mag(out_b, 0, len(out_b), 320.0, SR)
    assert gb2 > 3 * ga2, "xf=1 must be deck B only"
    desk.crossfader(0.5)
    out_m = desk.render(2.0)
    ga3 = _goertzel_mag(out_m, 0, len(out_m), 180.0, SR)
    gb3 = _goertzel_mag(out_m, 0, len(out_m), 320.0, SR)
    assert ga3 > 0 and gb3 > 0, "xf=0.5 must carry both decks"


# ── live set through the desk ────────────────────────────────────────────────

def test_live_set_through_desk(tmp_path):
    # A: marker at beat 6 proves the cue jump (jump to beat 4 at set 0
    # puts the marker at set 1.0s; without the jump it would be 3.0s).
    # B: marker at beat 2 proves the loop replay (loop from beat 0 at
    # set 4.0 replays beat 2 at set 7.0s; without the loop, set 7.0s is
    # beat 6, a plain click).
    a_samples = _click_track(120.0, 8.0, accent_every=4, marker_beat=6)
    b_samples = _click_track(120.0, 8.0, accent_every=4, freq=320.0,
                             marker_beat=2)
    out_path = str(tmp_path / "set.wav")
    script = [
        {"at_s": 0.0, "deck": "A", "kind": "cue", "name": "drop",
         "beat": 4},
        {"at_s": 3.5, "deck": "A", "kind": "pad", "pad": "airhorn"},
        {"at_s": 4.0, "deck": "A", "kind": "xfade", "to": 1.0, "beats": 4},
        {"at_s": 4.0, "deck": "B", "kind": "loop", "in_beat": 0,
         "length_beats": 4},
    ]
    res = run_live_set(
        [("track-a", TrackAnalysis(bpm=120.0, duration_s=8.0), a_samples),
         ("track-b", TrackAnalysis(bpm=120.0, duration_s=8.0), b_samples)],
        script, out_path, duration_s=8.0)
    assert res["ok"] is True
    assert Path(out_path).is_file()
    import wave as _w
    with _w.open(out_path, "rb") as f:
        n = f.getnframes()
        sr = f.getframerate()
        raw = f.readframes(n)
    assert 7.5 <= n / sr <= 8.5, f"set duration {n / sr}s"
    import struct as _st
    vals = _st.unpack(f"<{len(raw) // 2}h", raw)
    samples = array("d", [v / 32768.0 for v in vals])

    def _peak(t0: float, t1: float) -> float:
        seg = samples[int(t0 * sr):int(t1 * sr)]
        return max((abs(v) for v in seg), default=0.0)

    # cue jump: A's marker (beat 6) lands at set 1.0s, not 3.0s
    assert _peak(0.9, 1.1) >= 0.8, "cue jump did not move the marker"
    assert _peak(2.9, 3.1) < 0.8, "marker still at its un-jumped place"
    # pad signature: airhorn (440 Hz) present just after its quantized
    # fire (triggered at set 3.5s, next beat is set 4.0s)
    win = samples[int(4.0 * sr):int(4.7 * sr)]
    ctl = samples[int(0.2 * sr):int(0.9 * sr)]
    m_pad = _goertzel_mag(win, 0, len(win), 440.0, sr)
    m_ctl = _goertzel_mag(ctl, 0, len(ctl), 440.0, sr)
    assert m_pad > 3 * max(m_ctl, 1e-9), "airhorn pad not in the render"
    # loop replay: B's marker (beat 2) sounds again at set 7.0s
    assert _peak(6.9, 7.1) >= 0.8, "loop did not replay the marker beat"


# ── cue suggester (learned) ──────────────────────────────────────────────────

def test_cue_suggester_beats_code_rule():
    rows = build_cue_training_rows(n_tracks=12, seed=7)
    outcome = fit_cue_suggester(rows)
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.score > outcome.report.code_score, (
        f"model {outcome.report.score} did not beat code "
        f"{outcome.report.code_score}")
    # the learned model finds the known drop bar on a fresh track
    samples, cues = _structured_track(
        8, 120.0, [0.3, 0.3, 0.3, 0.9, 0.9, 0.4, 0.9, 0.9])
    got = suggest_cues(samples, SR, 120.0, model=outcome.model)
    top = [bar for bar, _s in got[:3]]
    assert 3 in top or 6 in top, f"drop bars missed: {got}"


def test_cue_suggester_below_floor_is_honest():
    rows = build_cue_training_rows(n_tracks=1, seed=7)
    outcome = fit_cue_suggester(rows)
    assert outcome.status == "insufficient"
    assert outcome.model is None


def test_cue_rule_suggests_without_model():
    samples, cues = _structured_track(
        8, 120.0, [0.3, 0.3, 0.3, 0.9, 0.9, 0.4, 0.9, 0.9])
    got = suggest_cues(samples, SR, 120.0, model=None)
    assert 3 in [bar for bar, _s in got], f"code rule missed the drop: {got}"


# ── beat-tracking model rung (madmom) ────────────────────────────────────────

def test_madmom_rung_real_beats(tmp_path):
    ok, _reason = madmom_available()
    if not ok:
        pytest.skip("madmom venv not provisioned on this box")
    wav = _write_wav(tmp_path / "click.wav", _click_track(120.0, 12.0))
    beats = madmom_beat_track(wav)
    assert beats, "madmom returned no beats"
    truth = [round(i * 0.5, 3) for i in range(24)]
    matched = sum(1 for t in truth
                  if any(abs(b - t) <= 0.05 for b in beats))
    assert matched >= 21, f"madmom matched {matched}/24: {beats[:8]}"


def test_madmom_rung_honest_unavailable():
    ok, reason = madmom_available(venv_python="/nonexistent/python")
    assert ok is False
    assert "madmom" in reason.lower()


# ── ladder registration ──────────────────────────────────────────────────────

def test_ladder_jobs_registered():
    from nomorals.llm.ladder import list_jobs, resolve
    assert "dj_beat_grid" in list_jobs()
    assert "dj_cue_suggest" in list_jobs()
    r = resolve("dj_beat_grid", specialist_available=False)
    assert r.rung == "code"
    r2 = resolve("dj_cue_suggest", specialist_available=True)
    assert r2.rung == "model"


# ── desk surface + show-builder wiring ───────────────────────────────────────

def test_desk_status_text():
    text = control_dj_desk("")
    assert "desk" in text.lower()
    assert "beat grid" in text.lower() or "grid" in text.lower()


def test_attach_desk_plan():
    analyses = [
        TrackAnalysis(path="a.wav", title="a", bpm=120.0, duration_s=12.0),
        TrackAnalysis(path="b.wav", title="b", bpm=122.0, duration_s=12.0),
    ]
    plans = attach_desk_plan(
        analyses,
        samples={ "a.wav": _click_track(120.0, 12.0),
                  "b.wav": _click_track(122.0, 12.0)})
    assert len(plans) == 2
    for plan in plans:
        assert plan["grid"] is not None
        assert len(plan["grid"].beat_times) >= 20
        assert isinstance(plan["cues"], list)


# ── real songs through the music chain ───────────────────────────────────────

def _fake_fetch_ok(monkeypatch, tmp_path):
    """The chain's seam, doubled: fetch_track_audio returns a local WAV."""
    import nomorals.media.dj_desk as desk
    from nomorals.media.dj_mixdrop import MixFetch
    wav = _write_wav(tmp_path / "real.wav", _click_track(120.0, 12.0))

    def fake_fetch(query, wd, ctx=None):
        return MixFetch(query=query, path=wav, title="Real One",
                        artist="Test Artist", source="archiveorg-search")
    monkeypatch.setattr(desk, "fetch_track_audio", fake_fetch)
    return wav


def test_load_real_track_detects_bpm(monkeypatch, tmp_path):
    _fake_fetch_ok(monkeypatch, tmp_path)
    from nomorals.media.dj_desk import load_real_track
    got = load_real_track("Test Artist Real One", str(tmp_path / "wd"))
    assert got["ok"] is True
    assert got["source"] == "archiveorg-search"
    # BPM/key are DETECTED from the real audio, never ground-truth
    assert got["analysis"].bpm_source == "detected"
    assert abs(got["analysis"].bpm - 120.0) <= 2.0
    assert got["grid"] is not None and got["grid"].known
    assert len(got["grid"].beat_times) >= 20


def test_load_real_track_failure_is_honest(monkeypatch, tmp_path):
    import nomorals.media.dj_desk as desk
    from nomorals.media.dj_mixdrop import MixFetch

    def fake_fetch(query, wd, ctx=None):
        return MixFetch(query=query, error="couldn't find 'x': no results")
    monkeypatch.setattr(desk, "fetch_track_audio", fake_fetch)
    from nomorals.media.dj_desk import load_real_track
    got = load_real_track("no such song", str(tmp_path / "wd"))
    assert got["ok"] is False
    assert "no results" in got["error"]
    assert got.get("analysis") is None  # no silent composed substitute


def test_dig_real_crate_partial(monkeypatch, tmp_path):
    import nomorals.media.dj_desk as desk
    from nomorals.media.dj_mixdrop import MixFetch
    wav = _write_wav(tmp_path / "real.wav", _click_track(120.0, 12.0))

    def fake_fetch(query, wd, ctx=None):
        if "good" in query:
            return MixFetch(query=query, path=wav, title="Good",
                            artist="A", source="archiveorg-search")
        return MixFetch(query=query, error="not found")
    monkeypatch.setattr(desk, "fetch_track_audio", fake_fetch)
    from nomorals.media.dj_desk import dig_real_crate
    tracks, failures = dig_real_crate(["good song", "bad song"],
                                      str(tmp_path / "wd"))
    assert len(tracks) == 1 and tracks[0]["ok"] is True
    assert len(failures) == 1 and "bad song" in failures[0]["query"]


def test_resolve_show_track_real_then_fallback(tmp_path):
    import nomorals.media.dj_desk as desk
    from nomorals.media.dj_mixdrop import MixFetch
    wav = _write_wav(tmp_path / "real.wav", _click_track(120.0, 12.0))
    composed = _write_wav(tmp_path / "composed.wav",
                          _click_track(120.0, 4.0))
    pick = {"title": "Real One", "artist": "Test Artist"}
    calls: list[str] = []

    def compose_fn(p, wd):
        calls.append("compose")
        return composed

    # real fetch wins — compose never runs
    got = desk.resolve_show_track(
        pick, str(tmp_path), fetch_fn=lambda q, wd, ctx=None: MixFetch(
            query=q, path=wav, title="Real One", artist="Test Artist",
            source="archiveorg-search"),
        compose_fn=compose_fn)
    assert got["source"] == "real" and got["path"] == wav
    assert calls == []
    # real fetch fails — composed fallback, NAMED as such
    got2 = desk.resolve_show_track(
        pick, str(tmp_path),
        fetch_fn=lambda q, wd, ctx=None: MixFetch(query=q, error="offline"),
        compose_fn=compose_fn)
    assert got2["source"] == "composed" and got2["path"] == composed
    assert "real fetch failed" in got2["reason"]
    assert calls == ["compose"]


def test_control_dj_desk_live_renders_real_set(monkeypatch, tmp_path):
    import nomorals.media.dj_desk as desk
    delivered: list[str] = []

    def fake_crate(queries, wd, ctx=None):
        out = []
        for qi, q in enumerate(queries):
            out.append({
                "ok": True, "track_id": f"t{qi}",
                "analysis": TrackAnalysis(
                    path=q, title=q, bpm=120.0, duration_s=8.0),
                "samples": _click_track(120.0, 8.0, freq=180.0 + 140 * qi,
                                        marker_beat=2 if qi else 6),
                "grid": _grid(120.0, 8.0), "cues": [],
                "source": "archiveorg-search", "title": q, "artist": "A"})
        return out, []
    monkeypatch.setattr(desk, "dig_real_crate", fake_crate)
    text = control_dj_desk(
        "live song a, song b", deliver_audio=lambda p, c: delivered.append(p),
        workdir=str(tmp_path))
    assert delivered, "live set did not deliver audio"
    assert Path(delivered[0]).is_file()
    assert "real" in text.lower()


# ── the full professional desk: jog, bend, scratch, key lock ────────────────

def _sine_track(freq: float = 440.0, seconds: float = 4.0,
                bpm: float = 120.0) -> array:
    """A sine bed at `freq` plus clicks on the beat grid (so the grid
    is real and the pitch is measurable with Goertzel)."""
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    for i in range(n):
        out[i] = 0.5 * math.sin(2 * math.pi * freq * i / SR)
    beat_n = int(SR * 60.0 / bpm)
    for start in range(0, n, beat_n):
        for i in range(min(2205, n - start)):
            t = i / SR
            out[start + i] += 0.6 * math.exp(-t * 60.0) * math.sin(
                2 * math.pi * 180.0 * t)
    return out


def _load_deck(freq: float = 440.0, seconds: float = 12.0,
               bpm: float = 120.0, name: str = "A") -> Deck:
    deck = Deck(name)
    deck.load(TrackAnalysis(bpm=bpm, duration_s=seconds),
              _sine_track(freq, seconds, bpm) if freq else
              _click_track(bpm, seconds))
    return deck


def test_deck_jog_moves_playhead():
    deck = _load_deck(freq=0.0)
    deck.play()
    deck.render(1.0)  # playhead at beat 2
    assert abs(deck.position_beats() - 2.0) < 0.01
    deck.jog(2.0)
    assert abs(deck.position_beats() - 4.0) < 0.01
    deck.jog(-1.0)
    assert abs(deck.position_beats() - 3.0) < 0.01
    deck.jog(-100.0)
    assert deck.position_s == 0.0  # clamped, never negative
    deck.jog(1000.0)
    assert abs(deck.position_s - deck.duration_s) <= 0.01


def test_deck_beat_jump_snaps_to_grid():
    deck = _load_deck(freq=0.0)
    deck.play()
    deck.render(1.3)  # between beats 2 and 3
    deck.beat_jump(4)
    # nearest beat is 1.5s (beat 3); +4 beats = beat 7 = 3.5s
    assert abs(deck.position_s - 3.5) < 0.01
    deck.beat_jump(-2)
    assert abs(deck.position_s - 2.5) < 0.01


def test_deck_bend_speeds_playhead_then_releases():
    deck = _load_deck(freq=0.0)
    deck.play()
    deck.bend(1.10, 2.0)  # +10% for 2 beats
    deck.render(1.0)  # 2 beats of set time at 1.10x = 2.2 beats of audio
    assert abs(deck.position_beats() - 2.2) < 0.05
    deck.render(1.0)  # bend spent: back to 1.0x
    assert abs(deck.position_beats() - 4.2) < 0.05


def test_deck_scratch_burst_is_reversed():
    samples = _click_track(120.0, 12.0)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=12.0), samples)
    deck.play()
    deck.render(2.0)  # playhead 2.0s (beat 4)
    pos = deck.scratch(-0.5)  # back half a second
    assert abs(pos - 1.5) < 0.01
    queued = deck.queued_pads()
    assert queued, "scratch queued no burst"
    pad, when, burst = queued[-1]
    assert pad == "__scratch__"
    assert abs(when - deck.position_s) < 0.01
    src = samples[int(1.5 * SR) - len(burst):int(1.5 * SR)]
    assert len(burst) == len(src)
    for got, want in zip(burst, reversed(src), strict=True):
        assert abs(got - want) < 1e-9


def test_key_lock_preserves_pitch():
    samples = _sine_track(440.0, 4.0)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=4.0), samples)
    deck.set_rate(1.08)
    deck.key_lock = True
    deck.play()
    out = deck.render(2.0)
    m440 = _goertzel_mag(out, 0, len(out), 440.0, SR)
    m475 = _goertzel_mag(out, 0, len(out), 440.0 * 1.08, SR)
    assert m440 > 2 * m475, f"key lock lost the pitch: 440={m440} 475={m475}"
    # without key lock the pitch rides the fader
    deck2 = Deck("A")
    deck2.load(TrackAnalysis(bpm=120.0, duration_s=4.0), samples)
    deck2.set_rate(1.08)
    deck2.play()
    out2 = deck2.render(2.0)
    m440b = _goertzel_mag(out2, 0, len(out2), 440.0, SR)
    m475b = _goertzel_mag(out2, 0, len(out2), 440.0 * 1.08, SR)
    assert m475b > 2 * m440b, "no-key-lock render kept the pitch"


# ── mixer: 3-band EQ + channel faders ────────────────────────────────────────

def _eq_mag(freq: float, band: str, value: float) -> tuple[float, float]:
    """Goertzel magnitude at `freq` with the EQ flat vs cut."""
    dry_deck = _load_deck(freq=freq, seconds=4.0)
    dry_deck.play()
    dry = dry_deck.render(1.0)
    cut_deck = _load_deck(freq=freq, seconds=4.0)
    cut_deck.set_eq(band, value)
    cut_deck.play()
    cut = cut_deck.render(1.0)
    return (_goertzel_mag(cut, 0, len(cut), freq, SR),
            _goertzel_mag(dry, 0, len(dry), freq, SR))


def test_eq_low_kills_bass():
    cut, dry = _eq_mag(100.0, "low", 0.0)
    assert cut < 0.1 * dry, f"low EQ 0 left {cut/dry:.2f} of the bass"


def test_eq_high_kills_treble():
    cut, dry = _eq_mag(8000.0, "high", 0.0)
    assert cut < 0.1 * dry, f"high EQ 0 left {cut/dry:.2f} of the treble"


def test_eq_mid_cuts_the_middle():
    cut, dry = _eq_mag(1000.0, "mid", 0.0)
    assert cut < 0.7 * dry, f"mid EQ 0 left {cut/dry:.2f} of the middle"


def test_eq_rejects_bad_band_and_range():
    deck = Deck("A")
    with pytest.raises(ValueError):
        deck.set_eq("sub", 1.0)
    with pytest.raises(ValueError):
        deck.set_eq("low", 3.0)
    with pytest.raises(ValueError):
        deck.set_eq("low", -0.5)


def test_channel_fader_silences_deck():
    desk = DJDesk()
    desk.load_deck("A", TrackAnalysis(bpm=120.0, duration_s=12.0),
                   _click_track(120.0, 12.0, freq=180.0))
    desk.load_deck("B", TrackAnalysis(bpm=120.0, duration_s=12.0),
                   _click_track(120.0, 12.0, freq=320.0))
    desk.deck_a.play()
    desk.deck_b.play()
    desk.crossfader(0.5)
    desk.deck_b.set_fader(0.0)
    out = desk.render(2.0)
    m180 = _goertzel_mag(out, 0, len(out), 180.0, SR)
    m320 = _goertzel_mag(out, 0, len(out), 320.0, SR)
    assert m320 < 0.05 * m180, "fader 0 left deck B audible"
    assert m180 > 0, "fader 0 on B silenced A too"


# ── effects: filter, echo, reverb, flanger ───────────────────────────────────

def _impulse_track(seconds: float = 4.0) -> array:
    out = array("d", [0.0]) * int(SR * seconds)
    out[0] = 1.0
    return out


def test_filter_lowpass_attenuates():
    dry_deck = _load_deck(freq=8000.0, seconds=4.0)
    dry_deck.play()
    dry = dry_deck.render(1.0)
    fx_deck = _load_deck(freq=8000.0, seconds=4.0)
    fx_deck.set_fx("filter", mode="lowpass", cutoff=500.0)
    fx_deck.play()
    wet = fx_deck.render(1.0)
    m_dry = _goertzel_mag(dry, 0, len(dry), 8000.0, SR)
    m_wet = _goertzel_mag(wet, 0, len(wet), 8000.0, SR)
    assert m_wet < 0.1 * m_dry, f"lowpass left {m_wet/m_dry:.2f} of 8kHz"


def test_filter_highpass_attenuates():
    dry_deck = _load_deck(freq=100.0, seconds=4.0)
    dry_deck.play()
    dry = dry_deck.render(1.0)
    fx_deck = _load_deck(freq=100.0, seconds=4.0)
    fx_deck.set_fx("filter", mode="highpass", cutoff=5000.0)
    fx_deck.play()
    wet = fx_deck.render(1.0)
    m_dry = _goertzel_mag(dry, 0, len(dry), 100.0, SR)
    m_wet = _goertzel_mag(wet, 0, len(wet), 100.0, SR)
    assert m_wet < 0.1 * m_dry, f"highpass left {m_wet/m_dry:.2f} of 100Hz"


def test_echo_repeats_impulse():
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=4.0),
              _impulse_track(4.0))
    deck.set_fx("echo", on=True, beats=1.0, feedback=0.5, mix=0.5)
    deck.play()
    out = deck.render(2.0)

    def _energy(t0: float, t1: float) -> float:
        seg = out[int(t0 * SR):int(t1 * SR)]
        return sum(v * v for v in seg)

    assert _energy(0.45, 0.55) > 0.05 * _energy(0.0, 0.1), \
        "no echo repeat at 1 beat (0.5s)"
    assert _energy(0.95, 1.05) > 0.01 * _energy(0.0, 0.1), \
        "no second echo repeat at 2 beats (1.0s)"
    # control: no echo, no repeats
    ctl = Deck("A")
    ctl.load(TrackAnalysis(bpm=120.0, duration_s=4.0),
             _impulse_track(4.0))
    ctl.play()
    out_ctl = ctl.render(2.0)
    seg = out_ctl[int(0.45 * SR):int(0.55 * SR)]
    assert max((abs(v) for v in seg), default=0.0) < 1e-9


def test_reverb_tail_outlives_input():
    # a 0.2s tone burst, not a single-sample impulse: reverb is a tail
    # AFTER a sound ends, and an impulse puts all its energy in one
    # sample — the honest probe is a real sound with a real tail.
    # Measured 0.19 on this design (Freeverb feedback 0.84); the bar
    # is a quarter of that — a tail, not a rounding error.
    burst = array("d", [0.0]) * int(SR * 4.0)
    for i in range(int(SR * 0.2)):
        burst[i] = 0.5 * math.sin(2 * math.pi * 440.0 * i / SR)
    deck = Deck("A")
    deck.load(TrackAnalysis(bpm=120.0, duration_s=4.0), burst)
    deck.set_fx("reverb", on=True, mix=0.8)
    deck.play()
    out = deck.render(2.0)

    def _energy(t0: float, t1: float) -> float:
        seg = out[int(t0 * SR):int(t1 * SR)]
        return sum(v * v for v in seg)

    assert _energy(0.3, 1.0) > 0.05 * _energy(0.0, 0.2), \
        "reverb left no tail after the burst"
    ctl = Deck("A")
    ctl.load(TrackAnalysis(bpm=120.0, duration_s=4.0), burst)
    ctl.play()
    out_ctl = ctl.render(2.0)
    seg = out_ctl[int(0.3 * SR):int(1.0 * SR)]
    assert max((abs(v) for v in seg), default=0.0) < 1e-9


def test_flanger_changes_signal():
    dry_deck = _load_deck(freq=440.0, seconds=4.0)
    dry_deck.play()
    dry = dry_deck.render(1.0)
    fx_deck = _load_deck(freq=440.0, seconds=4.0)
    fx_deck.set_fx("flanger", on=True, mix=0.5, rate_hz=0.5)
    fx_deck.play()
    wet = fx_deck.render(1.0)
    diff = max(abs(a - b) for a, b in zip(dry, wet, strict=True))
    assert diff > 0.02, "flanger left the signal untouched"
    rms = math.sqrt(sum(v * v for v in wet) / len(wet))
    assert rms > 0.05, "flanger killed the signal"


def test_fx_chain_defaults_off():
    chain = FXChain()
    assert chain.filter_mode == "off"
    assert chain.echo_on is False
    assert chain.reverb_on is False
    assert chain.flanger_on is False


# ── library: crates, search, track loading ───────────────────────────────────

def test_library_crates_search_load(tmp_path):
    store = DeskStore(str(tmp_path / "desk.json"))
    store.load()
    lib = Library(store)
    wav_a = _write_wav(tmp_path / "a.wav", _click_track(120.0, 4.0))
    wav_b = _write_wav(tmp_path / "b.wav", _click_track(126.0, 4.0,
                                                       freq=320.0))
    lib.add_track("a", title="Neon Skyline", artist="Nova Rae",
                  bpm=120.0, key="Am", mode="minor", source="real",
                  path=wav_a, duration_s=4.0)
    lib.add_track("b", title="Midnight Fuel", artist="DJ Carbon",
                  bpm=126.0, key="C", mode="major", source="composed",
                  path=wav_b, duration_s=4.0)
    assert lib.create_crate("peak-time") is True
    assert lib.create_crate("peak-time") is False  # already exists
    assert lib.add_to_crate("peak-time", "a") is True
    assert lib.add_to_crate("peak-time", "b") is True
    assert lib.add_to_crate("nope", "a") is False
    assert lib.add_to_crate("peak-time", "ghost") is False
    assert [t["track_id"] for t in lib.search("neon")] == ["a"]
    assert [t["track_id"] for t in lib.search(bpm_min=125)] == ["b"]
    assert [t["track_id"] for t in lib.search(key="Am")] == ["a"]
    assert [t["track_id"] for t in lib.search("dj")] == ["b"]
    assert lib.crate_tracks("peak-time") == ["a", "b"]
    assert lib.crate_tracks("nope") == []
    assert lib.crates() == ["peak-time"]
    store.save()
    store2 = DeskStore(str(tmp_path / "desk.json"))
    store2.load()
    lib2 = Library(store2)
    assert lib2.crate_tracks("peak-time") == ["a", "b"]
    assert lib2.get_track("a")["title"] == "Neon Skyline"
    desk = DJDesk()
    lib2.load_deck(desk, "A", "a")
    assert desk.deck_a.bpm == 120.0
    assert abs(desk.deck_a.duration_s - 4.0) < 0.01
    with pytest.raises(ValueError):
        lib2.load_deck(desk, "B", "ghost")


# ── waveforms ────────────────────────────────────────────────────────────────

def test_waveform_peaks_follow_clicks():
    samples = _click_track(120.0, 4.0)
    wf = compute_waveform(samples, SR, columns=40)
    assert len(wf) == 40
    assert max(wf) > 0.5
    # clicks every 0.5s = every 5 columns at 40 cols / 4s
    assert wf[0] > 0.5 and wf[5] > 0.5 and wf[10] > 0.5
    assert wf[2] < 0.1 and wf[7] < 0.1
    silence = compute_waveform(array("d", [0.0]) * SR * 4, SR)
    assert silence == [0.0] * 200


def test_deck_carries_waveform():
    deck = _load_deck(freq=0.0, seconds=4.0)
    assert len(deck.waveform) == 200
    assert max(deck.waveform) > 0.5


# ── recording ────────────────────────────────────────────────────────────────

def test_desk_recording_captures_mix(tmp_path):
    desk = DJDesk()
    desk.load_deck("A", TrackAnalysis(bpm=120.0, duration_s=12.0),
                   _click_track(120.0, 12.0))
    desk.deck_a.play()
    assert desk.record_start() is True
    assert desk.record_start() is False  # already recording
    desk.render(2.0)
    path = str(tmp_path / "set.wav")
    rep = desk.record_stop(path)
    assert rep["ok"] is True
    assert Path(path).is_file()
    import wave as _w
    with _w.open(path, "rb") as f:
        assert abs(f.getnframes() / f.getframerate() - 2.0) < 0.1
        raw = f.readframes(f.getnframes())
    assert max(struct.unpack(f"<{len(raw) // 2}h", raw)) > 1000
    assert desk.record_stop(str(tmp_path / "x.wav"))["ok"] is False


# ── show builder places cues/pads ────────────────────────────────────────────

def test_build_show_places_desk_plan(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    from nomorals.media.dj import DJ

    monkeypatch.setenv("NM_DJ_DESK_PATH", str(tmp_path / "desk.json"))
    workdir = tmp_path / "dj"
    workdir.mkdir()
    wav = _write_wav(workdir / "t1.wav", _click_track(120.0, 4.0))
    dj = DJ(context=None)
    fake_tracks = [{"title": "Neon Skyline", "artist": "Nova Rae",
                    "rank": 1, "genre_hint": "pop",
                    "source": "billboard"}]

    def _mock_song(*a, **k):
        m = MagicMock()
        m.audio_path = wav
        m.tempo = 120.0
        m.key = "A"
        m.mode = "minor"
        return m

    with patch.object(DJ, "fetch_trending",
                      return_value={"ok": True, "tracks": fake_tracks,
                                    "source": "billboard",
                                    "cached": False}), \
            patch("nomorals.media.music.MusicCreator") as MC:
        MC.return_value.compose.side_effect = _mock_song
        show = dj.build_show(genre="", n_tracks=1,
                             workdir=str(workdir))
    assert show["ok"] is True, show.get("reason")
    plan = show.get("desk_plan")
    assert plan and len(plan) == 1, f"no desk plan in show: {show.keys()}"
    assert plan[0]["grid_beats"] >= 4
    assert isinstance(plan[0]["cues"], list)
    store = DeskStore(str(tmp_path / "desk.json"))
    store.load()
    grid, bank = store.get_track(wav)
    assert grid is not None and grid.known
    assert bank is not None


# ── /dj desk surface: mixer, FX, crate, record, wave ────────────────────────

def _fresh_live_desk(monkeypatch):
    import nomorals.media.dj_desk as desk_mod
    monkeypatch.setattr(desk_mod, "_LIVE_DESK", DJDesk())
    desk_mod._LIVE_DESK.load_deck(
        "A", TrackAnalysis(bpm=120.0, duration_s=12.0),
        _click_track(120.0, 12.0))
    return desk_mod


def test_control_desk_mixer_verbs(monkeypatch):
    desk_mod = _fresh_live_desk(monkeypatch)
    live = desk_mod._LIVE_DESK
    assert "low" in desk_mod.control_dj_desk("eq A low 0").lower()
    assert live.deck_a.eq_low == 0.0
    assert "filter" in desk_mod.control_dj_desk(
        "fx A filter lowpass 800").lower()
    assert live.deck_a.fx.filter_mode == "lowpass"
    assert "fader" in desk_mod.control_dj_desk("fader B 0.5").lower()
    assert live.deck_b.fader == 0.5
    assert "key lock" in desk_mod.control_dj_desk("keylock A on").lower()
    assert live.deck_a.key_lock is True
    text = desk_mod.control_dj_desk("jog A 2")
    assert "jog" in text.lower()
    assert abs(live.deck_a.position_beats() - 2.0) < 0.01
    bad = desk_mod.control_dj_desk("eq A low 99")
    assert "0" in bad and "2" in bad  # honest range refusal, never raises
    assert "usage" in desk_mod.control_dj_desk("eq").lower()


def test_control_desk_crate_and_record(monkeypatch, tmp_path):
    desk_mod = _fresh_live_desk(monkeypatch)
    monkeypatch.setenv("NM_DJ_DESK_PATH", str(tmp_path / "desk.json"))
    assert "empty" in desk_mod.control_dj_desk("crate list").lower()
    assert "empty" in desk_mod.control_dj_desk(
        "crate search neon").lower()
    live = desk_mod._LIVE_DESK
    live.deck_a.play()
    assert "record" in desk_mod.control_dj_desk("record start").lower()
    live.render(1.0)
    delivered: list[str] = []
    text = desk_mod.control_dj_desk(
        "record stop",
        deliver_audio=lambda p, c: delivered.append(p),
        workdir=str(tmp_path))
    assert delivered and Path(delivered[0]).is_file()
    assert "record" in text.lower()


def test_control_desk_wave(monkeypatch):
    desk_mod = _fresh_live_desk(monkeypatch)
    text = desk_mod.control_dj_desk("wave A")
    assert "waveform" in text.lower() and "200" in text
    assert "no track" in desk_mod.control_dj_desk("wave B").lower()


def test_live_set_script_uses_mixer_actions(tmp_path):
    a = _click_track(120.0, 8.0, freq=180.0)
    script = [
        {"at_s": 0.0, "deck": "A", "kind": "eq", "band": "low",
         "value": 0.0},
        {"at_s": 0.0, "deck": "A", "kind": "jog", "beats": 2},
        {"at_s": 0.5, "deck": "A", "kind": "bend", "factor": 1.05,
         "beats": 2},
        {"at_s": 1.0, "deck": "A", "kind": "fader", "value": 0.5},
        {"at_s": 1.0, "deck": "A", "kind": "fx", "fx": "filter",
         "mode": "lowpass", "cutoff": 800.0},
        {"at_s": 1.0, "deck": "A", "kind": "keylock", "on": True},
    ]
    out_path = str(tmp_path / "set.wav")
    res = run_live_set(
        [("t", TrackAnalysis(bpm=120.0, duration_s=8.0), a)],
        script, out_path, duration_s=4.0)
    assert res["ok"] is True
    assert res["actions_applied"] == len(script), res
    # the EQ action is audible: low band cut on 180 Hz clicks quiets it
    plain = run_live_set(
        [("t", TrackAnalysis(bpm=120.0, duration_s=8.0), a)],
        [], str(tmp_path / "plain.wav"), duration_s=4.0)
    assert plain["ok"] is True
    import wave as _w
    def _rms(path: str) -> float:
        with _w.open(path, "rb") as f:
            raw = f.readframes(f.getnframes())
        vals = struct.unpack(f"<{len(raw) // 2}h", raw)
        return math.sqrt(sum(v * v for v in vals) / max(len(vals), 1))
    assert _rms(out_path) < 0.5 * _rms(str(tmp_path / "plain.wav"))
