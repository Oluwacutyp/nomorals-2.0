"""Sweep tests: audio module upgrades (2026-10-10).

Covers the new behavior added in the audio sweep — new DSP effects and
presets, fingerprint/analysis upgrades, transcript-editing features,
audiobook store rules + compliance, pipeline diarization/minutes,
overview interactivity, and character lorebook/state.
"""

import math
import os
import sqlite3
import wave
from array import array
from pathlib import Path

import importlib

import pytest

fpmod = importlib.import_module("nomorals.audio.fingerprint")
from nomorals.audio import dsp
from nomorals.audio import edit as ae
from nomorals.audio import audiobook as ab
from nomorals.audio import pipeline as vp
from nomorals.audio import overview as ov
from nomorals.audio import characters as ch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

SR = 22050


def _sine(freq: float, seconds: float, sr: int = SR, amp: float = 0.5,
          when: tuple[float, float] | None = None) -> array:
    """Sine burst; ``when=(start, end)`` places it in a silent buffer."""
    total = int(seconds * sr)
    out = array("d", [0.0]) * total
    s0 = int((when[0] if when else 0.0) * sr)
    s1 = int((when[1] if when else seconds) * sr)
    for i in range(max(0, s0), min(total, s1)):
        t = (i - s0) / sr
        out[i] = amp * math.sin(2.0 * math.pi * freq * t)
    return out


def _write_wav(path: Path, samples: array, sr: int = SR) -> str:
    pcm = array("h", (max(-32768, min(32767, int(s * 32767)))
                      for s in samples))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture()
def wav_440(tmp_path: Path) -> str:
    return _write_wav(tmp_path / "a440.wav", _sine(440.0, 3.0))


@pytest.fixture()
def wav_445(tmp_path: Path) -> str:
    # ~19.6 cents sharp of A440
    return _write_wav(tmp_path / "a445.wav", _sine(445.0, 3.0))


# ---------------------------------------------------------------------------
# dsp — new effects
# ---------------------------------------------------------------------------

def test_biquad_highpass_kills_rumble():
    low = _sine(40.0, 1.0)
    out = dsp.highpass(low, SR, 160.0)  # two octaves above → ~24 dB down
    in_e = sum(s * s for s in low) / len(low)
    out_e = sum(s * s for s in out) / len(out)
    assert out_e < in_e * 0.01


def test_biquad_lowpass_kills_hiss():
    hi = _sine(9000.0, 1.0)
    out = dsp.lowpass(hi, SR, 4000.0)  # an octave above cutoff
    in_e = sum(s * s for s in hi) / len(hi)
    out_e = sum(s * s for s in out) / len(out)
    assert out_e < in_e * 0.05


def test_biquad_notch_delegates():
    # a Q=30 notch at 50 Hz rings for ~1 s — measure the settled tail
    hum = _sine(50.0, 4.0)
    out = dsp.notch(hum, SR, 50.0)
    tail_in = hum[3 * SR:]
    tail_out = out[3 * SR:]
    in_e = sum(s * s for s in tail_in) / len(tail_in)
    out_e = sum(s * s for s in tail_out) / len(tail_out)
    assert out_e < in_e * 1e-6


def test_biquad_bad_kind_passthrough():
    s = _sine(440.0, 0.5)
    out = dsp.biquad(s, SR, "nonsense", 440.0)
    assert list(out) == list(s)


def test_telephone_passes_voice_band():
    mid = _sine(1000.0, 1.0)
    out = dsp.telephone(mid, SR)
    in_e = sum(s * s for s in mid) / len(mid)
    out_e = sum(s * s for s in out) / len(out)
    assert out_e > in_e * 0.3  # voice band survives
    low = _sine(60.0, 1.0)
    out2 = dsp.telephone(low, SR)
    e2 = sum(s * s for s in out2) / len(out2)
    in2 = sum(s * s for s in low) / len(low)
    assert e2 < in2 * 0.1  # rumble gone


def test_bitcrush_quantizes():
    s = _sine(440.0, 0.5)
    out = dsp.bitcrush(s, SR, bits=2)
    assert len(out) == len(s)
    assert len(set(out)) < len(set(s))  # coarser quantization


def test_distortion_adds_harmonics_and_limits():
    s = _sine(440.0, 1.0, amp=0.3)
    out = dsp.distortion(s, SR, drive=12.0)
    assert len(out) == len(s)
    assert max(abs(v) for v in out) <= 0.96


@pytest.mark.parametrize("fn,kw", [
    (dsp.chorus, {}),
    (dsp.tremolo, {}),
    (dsp.vibrato, {}),
    (dsp.phaser, {}),
    (dsp.delay, {}),
    (dsp.gain_db, {"db": 6.0}),
])
def test_modulation_effects_change_signal(fn, kw):
    s = _sine(440.0, 1.0)
    out = fn(s, SR, **kw)
    assert len(out) == len(s)
    assert any(abs(a - b) > 1e-6 for a, b in zip(s, out))


def test_autolevel_evens_out_drifter():
    # quiet first half, loud second half
    q = _sine(440.0, 1.0, amp=0.05)
    l = _sine(440.0, 1.0, amp=0.8)
    s = array("d", list(q) + list(l))
    out = dsp.autolevel(s, SR, target_rms_db=-20.0)
    half = len(out) // 2
    rms_q = math.sqrt(sum(v * v for v in out[:half]) / half)
    rms_l = math.sqrt(sum(v * v for v in out[half:]) / (len(out) - half))
    assert rms_q > 0.02  # quiet part lifted
    assert rms_l < 0.5   # loud part tamed
    assert max(abs(v) for v in out) <= 1.0


def test_duck_under_ducks_bed_where_voice_is():
    bed = _sine(220.0, 2.0, amp=0.5)
    voice = _sine(440.0, 2.0, amp=0.5, when=(0.5, 1.5))
    out = dsp.duck_under(bed, voice, SR, duck_db=-12.0)
    assert len(out) == len(bed)
    e_silent = sum(v * v for v in out[:int(0.4 * SR)])
    e_voiced = sum(v * v for v in out[int(0.7 * SR):int(1.3 * SR)])
    assert e_voiced < e_silent * 0.5


def test_effect_chain_from_preset():
    chain = dsp.EffectChain.from_preset("lofi")
    s = _sine(440.0, 1.0)
    out = chain.run(s, SR)
    assert len(out) == len(s)
    assert "bitcrush" in chain.describe()
    with pytest.raises(ValueError):
        dsp.EffectChain.from_preset("nope")


def test_enhance_new_profiles():
    for prof in ("podcast", "audiobook"):
        assert prof in dsp.ENHANCE_PROFILES
        res = dsp.enhance(_sine(440.0, 1.0), SR, profile=prof)
        assert res["ok"] and len(res["samples"]) == SR


def test_list_presets_catalogue():
    txt = dsp.list_presets()
    assert "lofi" in txt and "telephone" in txt


# ---------------------------------------------------------------------------
# fingerprint — hashing + analysis upgrades
# ---------------------------------------------------------------------------

def test_fingerprint_hashes_and_localmax_mode(wav_440):
    h1 = fpmod.fingerprint(wav_440)
    assert len(h1) > 10
    h2 = fpmod.fingerprint(wav_440, peak_mode="localmax")
    assert len(h2) > 0
    assert all(isinstance(h, int) and t >= 0 for h, t in h1)


def test_fingerprint_fanout_upgrade():
    assert fpmod._FANOUT == 15


def test_db_roundtrip_with_real_duration(wav_440, tmp_path):
    db = fpmod.FingerprintDB(str(tmp_path / "fpmod.db"))
    res = db.add_track(wav_440, title="A440", artist="test")
    assert res["ok"] and res["hashes"] > 0
    assert res["duration_s"] > 2.5  # the duration bug wrote 0.0
    tracks = db.list_tracks()
    assert tracks and tracks[0]["duration_s"] > 2.5
    m = db.match(wav_440)
    assert m["ok"] and m["title"] == "A440"
    assert db.stats()["tracks"] == 1
    db.close()


def test_identify_and_batch(wav_440, wav_445, tmp_path):
    db = fpmod.FingerprintDB(str(tmp_path / "fp2.db"))
    db.add_track(wav_440, title="A440", artist="t")
    r = fpmod.identify(wav_440, db=db)
    assert r["ok"] and r["title"] == "A440"
    out = fpmod.match_batch([wav_440, "/nope/missing.wav"], db=db)
    assert len(out) == 2
    assert out[0]["ok"] and not out[1]["ok"]
    db.close()


def test_index_directory(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    _write_wav(d / "one.wav", _sine(440.0, 2.0))
    _write_wav(d / "two.wav", _sine(660.0, 2.0))
    db = fpmod.FingerprintDB(str(tmp_path / "fp3.db"))
    res = db.index_directory(d)
    assert res["ok"] and res["indexed"] == 2 and res["scanned"] == 2
    db.close()


def test_export_constellation(wav_440):
    res = fpmod.export_constellation(wav_440)
    assert res["ok"] and res["count"] > 0
    f, t = res["peaks"][0]
    assert f > 0 and t >= 0


def test_analysis_tuning_sharp(wav_445):
    a = fpmod.analyze(wav_445)
    assert a.ok
    # 445 Hz is ~19.6 cents sharp of A440
    assert 10.0 < a.tuning_cents < 30.0
    assert set(a.quality) >= {"snr_db", "hum", "saturated", "dc_offset"}


def test_analysis_tuning_in_tune(wav_440):
    a = fpmod.analyze(wav_440)
    assert a.ok
    assert abs(a.tuning_cents) < 8.0


def test_analysis_chord_and_quality_keys(wav_440):
    a = fpmod.analyze(wav_440)
    d = a.to_dict()
    assert "chord" in d and "tuning_cents" in d and "quality" in d
    assert isinstance(d["quality"], dict)


def test_describe_audio_mentions_tuning(wav_445):
    txt = fpmod.describe_audio(wav_445)
    assert "sharp" in txt and "¢" in txt


def test_quality_flags_hum_detection(tmp_path):
    hum = _sine(50.0, 3.0, amp=0.4)
    p = _write_wav(tmp_path / "hum.wav", hum)
    a = fpmod.analyze(p)
    assert a.ok and a.quality["hum"] is True


# ---------------------------------------------------------------------------
# edit — transcript editing features
# ---------------------------------------------------------------------------

def test_fillers_new_languages():
    assert "euh" in ae.fillers_for("fr")
    assert "este" in ae.fillers_for("es")
    assert "ähm" in ae.fillers_for("de")
    assert "i mean" in ae.fillers_for("en")


def test_nl_intent_routes_widely():
    assert ae.nl_audio_intent(
        "remove all the filler words from note.wav")["command"] == "fillers"
    r = ae.nl_audio_intent("tighten the silences in note.wav")
    assert r["command"] == "silences" and r["file"] == "note.wav"
    r = ae.nl_audio_intent("clean up this audio file note.wav")
    assert r["command"] == "enhance"
    r = ae.nl_audio_intent("add reverb to note.wav")
    assert r["command"] == "fx" and r["effect"] == "reverb"
    r = ae.nl_audio_intent("what's in this audio note.wav")
    assert r["command"] == "analyze"
    assert ae.nl_audio_intent("hello world") is None


def test_list_fx_catalogue_style():
    txt = ae.list_fx()
    assert "telephone" in txt and "bitcrush" in txt
    assert "preset=lofi" in txt and "style presets" in txt


def _words():
    from nomorals.media_edit.captions import Word
    return [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4)
            for i, t in enumerate(["hello", "um", "world"])]


def test_preview_edits_dry_run():
    t = ae.EditableTranscript(audio_path="x.wav", words=_words())
    res = ae.preview_edits(t, [ae.Edit.delete(0.5, 0.9),
                               ae.Edit.move(1.0, 1.4, dest=0.0),
                               ae.Edit.silence(0.0, 0.4)])
    assert res["ok"] and len(res["preview"]) == 3
    kinds = [p["kind"] for p in res["preview"]]
    assert kinds == ["delete", "move", "silence"]
    assert res["preview"][1]["dest"] == 0.0
    assert "um" in res["preview"][0]["words"]


def test_correct_text_mode():
    t = ae.EditableTranscript(audio_path="x.wav", words=_words())
    n = t.correct_text("um", "uh")
    assert n == 1 and t.words[1].text == "uh"


def test_find_silences_native(tmp_path):
    # 1s tone, 2s silence, 1s tone
    s = array("d", list(_sine(440.0, 1.0)) + [0.0] * SR * 2
              + list(_sine(440.0, 1.0)))
    p = _write_wav(tmp_path / "gaps.wav", s)
    regions = ae.find_silences(p, min_silence_s=0.7)
    assert len(regions) == 1
    st, en = regions[0]
    assert 0.8 < st < 1.4 and 2.6 < en < 3.2


def test_remove_silences_end_to_end(tmp_path):
    s = array("d", list(_sine(440.0, 1.0)) + [0.0] * SR * 2
              + list(_sine(440.0, 1.0)))
    p = _write_wav(tmp_path / "gaps.wav", s)
    res = ae.remove_silences(p, min_silence_s=0.7, out_dir=str(tmp_path))
    assert res["ok"], res.get("reason")
    assert res["cut"] == 1 and res["saved_s"] > 1.0
    assert Path(res["output"]).exists()


def test_apply_edits_move_and_silence(tmp_path):
    s = array("d", list(_sine(440.0, 1.0)) + list(_sine(660.0, 1.0))
              + list(_sine(880.0, 1.0)))
    p = _write_wav(tmp_path / "three.wav", s)
    t = ae.EditableTranscript(audio_path=p, words=[])
    # move the middle second to the front; mute the last half-second
    res = ae.apply_edits(t, [ae.Edit.move(1.0, 2.0, dest=0.0),
                             ae.Edit.silence(2.5, 3.0)],
                         out_dir=str(tmp_path))
    assert res["ok"], res.get("reason")
    assert res["moves"] == 1 and res["silences"] == 1
    assert Path(res["output"]).exists()


def test_apply_edits_rejects_bad_move(tmp_path):
    p = _write_wav(tmp_path / "s.wav", _sine(440.0, 2.0))
    t = ae.EditableTranscript(audio_path=p, words=[])
    res = ae.apply_edits(t, [ae.Edit.move(0.5, 1.0, dest=0.7)],
                         out_dir=str(tmp_path))
    assert not res["ok"] and "inside its own" in res["reason"]
    res = ae.apply_edits(t, [ae.Edit.move(0.5, 1.0, dest=0.2),
                             ae.Edit.delete(0.0, 0.4)],
                         out_dir=str(tmp_path))
    assert not res["ok"] and "inside another" in res["reason"]


def test_duck_audio_end_to_end(tmp_path):
    voice = _write_wav(tmp_path / "v.wav",
                        _sine(440.0, 2.0, amp=0.5, when=(0.5, 1.5)))
    bed = _write_wav(tmp_path / "b.wav", _sine(220.0, 2.0, amp=0.4))
    res = ae.duck_audio(voice, bed, out_dir=str(tmp_path))
    assert res["ok"], res.get("reason")
    assert Path(res["output"]).exists()


# ---------------------------------------------------------------------------
# audiobook — store rules + compliance
# ---------------------------------------------------------------------------

def test_store_rules_corrected():
    acx = ab.store_rules("acx")
    assert acx["ok"] and acx["allowed"] is False
    assert "Virtual Voice" in acx["block_reason"]
    ar = ab.store_rules("authors_republic")
    assert ar["ok"] and ar["allowed"] is False
    sp = ab.store_rules("spotify")
    assert sp["ok"] and sp["allowed"] is True
    assert "digital voice narration" in sp["disclosure_text"]
    assert ab.DISCLOSURE_RULES["version"] == "2026-10b"


def test_check_store_allowed_splits():
    allowed, blocked = ab.check_store_allowed(["acx", "spotify", "nope"])
    assert allowed == ["spotify"]
    assert {b["store"] for b in blocked} == {"acx", "nope"}


def test_platform_targets_sane():
    assert ab.PLATFORM_TARGETS["spotify"]["lufs"] == -16.0
    assert ab.PLATFORM_TARGETS["acx"]["lufs"] == -20.0
    assert ab.PLATFORM_TARGETS["acx"]["bitrate"] == "192k"


def test_master_and_compliance(wav_440, tmp_path):
    mastered = ab.master_lufs(wav_440, out_dir=str(tmp_path), store="spotify")
    assert mastered and Path(mastered).exists()
    m = ab.measure_loudness(mastered)
    assert m["ok"] and m["lufs"] is not None
    assert abs(m["lufs"] - (-16.0)) < 2.5  # linear mode may land under target
    c = ab.check_compliance(mastered, "spotify")
    assert c["ok"] and c["verdict"] in ("PASS", "WARN")
    assert any(chk["metric"] == "integrated LUFS" for chk in c["checks"])


def test_chapter_pacing(tmp_path):
    c1 = ab.BookChapter(index=1, title="Ch 1",
                        audio_path=_write_wav(tmp_path / "c1.wav",
                                              _sine(440.0, 2.0, amp=0.5)))
    c2 = ab.BookChapter(index=2, title="Ch 2",
                        audio_path=_write_wav(tmp_path / "c2.wav",
                                              _sine(440.0, 2.0, amp=0.05)))
    res = ab.chapter_pacing([c1, c2])
    assert res["ok"] and res["max_drift_db"] > 6.0
    assert res["verdict"] == "FAIL"


def test_export_mp3_192(wav_440, tmp_path):
    out = ab.export_mp3(wav_440, bitrate="192k", out_dir=str(tmp_path))
    assert out and Path(out).exists() and out.endswith(".mp3")


def test_epub_refuses_blocked_stores(tmp_path):
    fake_epub = tmp_path / "book.epub"
    fake_epub.write_bytes(b"PK fake")
    res = ab.epub_to_audiobook(str(fake_epub), {"narrator": "x"},
                               ["acx", "authors_republic"])
    assert not res["ok"] and "no shippable store" in res["reason"]


def test_audiobook_chat_stores_and_check(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_HOME", str(tmp_path))
    out = ab.control_audiobook("stores")
    assert "acx" in out and "⛔" in out and "2026-10b" in out
    assert ab.control_audiobook("check") .startswith("🎧 check which?")


# ---------------------------------------------------------------------------
# pipeline — diarization, minutes, subtitles
# ---------------------------------------------------------------------------

def test_diarize_real_honest_without_token(monkeypatch):
    monkeypatch.delenv("WHISPERX_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    res = vp.diarize_real("whatever.wav")
    assert not res["ok"] and "token" in res["reason"].lower()


def test_group_turns_splits_speakers():
    from nomorals.media_edit.captions import Word
    words = [Word(text="hi", start=0.0, end=0.4),
             Word(text="yo", start=0.5, end=0.9),
             Word(text="hey", start=1.0, end=1.4)]
    turns = vp.group_turns(words, ["A", "B", "B"])
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A" and turns[1]["text"] == "yo hey"


def test_srt_vtt_minutes_formats():
    turns = [{"speaker": "A", "start": 1.0, "end": 2.5, "text": "hello"},
             {"speaker": "B", "start": 3.0, "end": 4.0, "text": "hi"}]
    srt = vp.transcript_to_srt(turns)
    assert "00:00:01,000 --> 00:00:02,500" in srt and "[A] hello" in srt
    vtt = vp.transcript_to_vtt(turns)
    assert vtt.startswith("WEBVTT") and "<v A>hello</v>" in vtt
    md = vp.meeting_minutes(turns, title="Sync")
    assert md.startswith("# Sync") and "**[B]**" in md


def test_pipeline_config_speaker_knobs():
    cfg = vp.PipelineConfig.from_dict({"min_speakers": 2, "max_speakers": 3})
    assert cfg.min_speakers == 2 and cfg.max_speakers == 3


def test_pipeline_missing_file_honest():
    res = vp.run_pipeline("/nope/missing.wav", {"transcribe": False})
    assert not res["ok"] and res["stage"] == "input"


def test_control_voice_engines_and_diarize_usage():
    out = vp.control_voice("engines")
    assert "STT order" in out and "TTS (public)" in out
    assert vp.control_voice("") .startswith("🎛️ /voice pipeline")


# ---------------------------------------------------------------------------
# overview — interactivity upgrades
# ---------------------------------------------------------------------------

def _script_fn(prompt: str) -> str:
    return ("CHAPTER: Money\n"
            "A: Rents rose 20% this year [S1].\n"
            "B: That tracks the market data [S1].\n"
            "A: But wages did not keep up [S2].\n"
            "B: Right, a real squeeze [S2].\n")


def _sources():
    return [{"title": "Rents", "text": "Rents rose 20% this year."},
            {"title": "Wages", "text": "Wages did not keep up."}]


def test_make_overview_grounded_and_interview():
    for fmt in ("deep-dive", "interview"):
        res = ov.make_overview(_sources(), fmt, script_fn=_script_fn,
                               render_audio=False)
        assert res["ok"], res.get("reason")
        assert res["overview"].script.cited_share == 1.0
    res = ov.make_overview(_sources(), "opera", script_fn=_script_fn,
                           render_audio=False)
    assert not res["ok"] and "unknown format" in res["reason"]


def test_make_overview_refuses_ungrounded():
    res = ov.make_overview(_sources(), "deep-dive",
                           script_fn=lambda p: "A: hello there.\nB: hi!\n",
                           render_audio=False)
    assert not res["ok"] and "citations" in res["reason"]


def test_suggest_questions_fallback():
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False)
    qs = ov.suggest_questions(res["overview"], n=3)
    assert len(qs) == 3 and all(q.endswith("?") for q in qs)


def test_suggest_questions_with_llm():
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False)
    qs = ov.suggest_questions(
        res["overview"], llm_fn=lambda p: "Why did rents rise?\nWho was hit?",
        n=2)
    assert qs == ["Why did rents rise?", "Who was hit?"]


def test_export_markdown():
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False, title="Rents pod")
    md = ov.export_markdown(res["overview"])
    assert "# Rents pod" in md and "**Adaeze:**" in md and "### Sources" in md


def test_interactive_ask_refuses_without_llm():
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False)
    sess = ov.InteractiveSession(res["overview"])
    ans = sess.ask("anything?")
    assert not ans["ok"] and "refusing to guess" in ans["reason"]


def test_interactive_ask_steers_with_history():
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False)
    seen: list[str] = []

    class FakeSession:
        def ask(self, q, llm_fn=None):
            seen.append(q)
            return type("A", (), {"refused": False,
                                  "text": "Rents rose 20%.",
                                  "sources": []})()

    sess = ov.InteractiveSession(res["overview"], llm_fn=lambda s, q: q)
    sess._session = FakeSession()
    sess.ask("why did rents rise?")
    sess.ask("and wages?")
    assert "Conversation so far" in seen[1]
    assert "why did rents rise?" in seen[1]


def test_overview_chat_questions_and_export(tmp_path):
    store = ov.OverviewStore(str(tmp_path / "ov.db"))
    res = ov.make_overview(_sources(), "deep-dive", script_fn=_script_fn,
                           render_audio=False, store=store)
    assert res["ok"]
    out = ov.control_overview("questions", store=store)
    assert "try asking the hosts" in out
    oid = res["overview"].overview_id
    md = ov.control_overview(f"export {oid}", store=store)
    assert "### Sources" in md
    assert "no overview" in ov.control_overview("export nope", store=store)


# ---------------------------------------------------------------------------
# characters — lorebook, state, history, cards
# ---------------------------------------------------------------------------

def test_character_lorebook_and_prompt():
    c = ch.Character(name="Vex", book_title="Arena", knowledge_cutoff=3)
    c.remember(2, "lost the duel")
    c.remember(9, "wins the war")  # past cutoff — must not leak
    assert c.pin_fact("owes Jax a life-debt")
    assert c.pin_fact("owes Jax a life-debt") is False  # dedupe
    c.set_relationship("Jax", "grudging respect")
    c.set_mood("reckless")
    prompt = c.to_system_prompt()
    assert "lost the duel" in prompt
    assert "wins the war" not in prompt
    assert "owes Jax a life-debt" in prompt
    assert "Jax: grudging respect" in prompt
    assert "reckless" in prompt
    assert "re-entering your own mind" in prompt  # inner alignment
    assert "Never format your replies like an AI" in prompt


def test_character_store_migration_and_roundtrip(tmp_path):
    # simulate an OLD database without the new columns
    old = tmp_path / "old.db"
    conn = sqlite3.connect(str(old))
    conn.execute(
        "CREATE TABLE characters (id TEXT PRIMARY KEY, name TEXT NOT NULL,"
        " book_id TEXT NOT NULL DEFAULT '', book_title TEXT NOT NULL DEFAULT '',"
        " voice TEXT NOT NULL DEFAULT '', knowledge_cutoff INTEGER NOT NULL DEFAULT 1,"
        " goals TEXT NOT NULL DEFAULT '[]', speech_patterns TEXT NOT NULL DEFAULT '[]',"
        " personality TEXT NOT NULL DEFAULT '{}', chapter_memories TEXT NOT NULL DEFAULT '{}',"
        " created_at REAL NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO characters (id, name) VALUES ('c1', 'Vex')")
    conn.commit()
    conn.close()
    store = ch.CharacterStore(str(old))  # migrates on open
    got = store.get("Vex")
    assert got is not None and got.facts == [] and got.mood == ""
    got.pin_fact("fears deep water")
    got.set_relationship("Mara", "protective")
    assert store.save(got)
    again = store.get("c1")
    assert "fears deep water" in again.facts
    assert again.relationships["Mara"] == "protective"
    store.close()


def test_talk_to_history_and_voice():
    c = ch.Character(name="Vex", book_title="Arena", knowledge_cutoff=2)
    seen: dict[str, str] = {}

    def fake_llm(system, question):
        seen["system"] = system
        seen["question"] = question
        return "I fight because I must."

    res = ch.talk_to(c, "why do you fight?",
                     llm_fn=fake_llm,
                     history=[{"q": "who are you?", "a": "I am Vex."}],
                     voice_fn=lambda name, text: "/tmp/vex.wav")
    assert res.ok and res.text == "I fight because I must."
    assert res.audio == "/tmp/vex.wav"
    assert "who are you?" in seen["question"]  # history injected


def test_talk_to_refuses_without_llm():
    c = ch.Character(name="Vex")
    res = ch.talk_to(c, "hi")
    assert not res.ok and "no dialogue engine" in res.reason


def test_export_card():
    c = ch.Character(name="Vex", book_title="Arena", knowledge_cutoff=2,
                     mood="reckless")
    c.pin_fact("owes Jax a life-debt")
    c.set_relationship("Jax", "grudging respect")
    card = ch.export_card(c)
    assert "# 🎭 Vex" in card and "reckless" in card
    assert "owes Jax a life-debt" in card and "grudging respect" in card


def test_character_chat_verbs(tmp_path):
    store = ch.CharacterStore(str(tmp_path / "chars.db"))
    assert "talkable now" in ch.control_character("add Vex Arena", store=store)
    out = ch.control_character("pin Vex fears deep water", store=store)
    assert "pinned" in out
    out = ch.control_character("relate Vex Jax grudging respect", store=store)
    assert "grudging respect" in out
    out = ch.control_character("mood Vex reckless", store=store)
    assert "reckless" in out
    out = ch.control_character("remember Vex 1 lost the duel", store=store)
    assert "chapter 1" in out
    card = ch.control_character("card Vex", store=store)
    assert "fears deep water" in card and "lost the duel" in card
    talk = ch.control_character(
        "talk Vex why do you fight",
        store=store, llm_fn=lambda s, q: "Because I must.")
    assert "Because I must." in talk
    store.close()


def test_audio_package_imports_star():
    import nomorals.audio as a
    for name in ("autolevel", "duck_under", "telephone", "bitcrush",
                 "identify", "match_batch", "export_constellation",
                 "remove_silences", "preview_edits", "duck_audio",
                 "check_compliance", "store_rules", "chapter_pacing",
                 "diarize_real", "group_turns", "meeting_minutes",
                 "transcript_to_srt", "suggest_questions", "export_markdown",
                 "export_card", "list_presets", "CHAIN_PRESETS"):
        assert name in a.__all__, name
