"""Tests for MUSIC GENERATION tiers — all offline, no models, no network.

Covers: tonal_g2p (Tier 1 moat + global), global_rhythm (Tier 2),
sonic_bible, music_critique, deliverables, model_license (Tier 3),
taste_model, lora_global, musicgen_chain (Tier 0, fakes labelled).
"""

import json
import os
import wave

import pytest

from nomorals.media.tonal_g2p import (
    ToneUnit, check_units, check_vowel_harmony, detect_languages,
    fix_units, g2p, profile_for, quality_of, sandhi, syllabify,
    tone_check, tone_fix_melody, tone_of, tones_of, units_of,
)
from nomorals.media.global_rhythm import (
    apply_groove, extract_groove_template, generate_genre_drums,
    genre_family, genre_rhythm,
)
from nomorals.media.sonic_bible import (
    SonicBible, TrackDescriptor, album_consistency, check_consistency)
from nomorals.media.music_critique import (
    critique_loop, critique_song, singability)
from nomorals.media.deliverables import (
    PLATFORM_LOUDNESS, cut_wav, deliverable_plan)
from nomorals.media.model_license import (
    can_ship, licence_audit, license_report, probe_matchering, release_gate)
from nomorals.media.taste_model import TasteModel, candidate_features
from nomorals.media.lora_global import (
    corpus_licence_check, genre_corpus_spec, kaggle_runbook)
from nomorals.media.musicgen_chain import (
    ChainRequest, draft_melody, fake_bed, fake_vocal, run_chain,
    write_midi, write_sine_wav)


# ── Tier 1: Yoruba G2P (the moat) ──────────────────────────────────────────


def test_yoruba_syllabify_cv_and_syllabic_nasal():
    assert syllabify("owó", "yo") == ["o", "wó"]
    assert syllabify("ọmọ", "yo") == ["ọ", "mọ"]
    # syllabic nasal: "n" before a consonant is its own syllable
    assert syllabify("ńlọ", "yo")[0] == "ń"


def test_yoruba_g2p_digraphs_kept_whole():
    ph = g2p("gbogbo", "yo")
    assert "gb" in ph  # gb is ONE phoneme, never split
    assert g2p("ṣe", "yo")[0] == "ṣ"
    assert g2p("kó", "yo") == ["k", "o"]  # tone mark is not a phoneme


def test_yoruba_nasal_vowel():
    ph = g2p("ọmọ", "yo")
    assert all("n" not in p for p in ph)  # no stray nasal consonant
    assert "ọ" in ph


def test_yoruba_tones_acute_grave_unmarked():
    assert tone_of("wó", "yo") == 4
    assert tone_of("wò", "yo") == 0
    assert tone_of("wo", "yo") == 2  # unmarked = MID
    assert tones_of("owó", "yo") == [2, 4]


def test_vowel_harmony_warning():
    assert check_vowel_harmony("orẹ")["ok"] is False  # e/o + ẹ/ọ mix
    assert check_vowel_harmony("ọmọ")["ok"] is True


def test_sandhi_mid_raises_before_high():
    s = sandhi("owo owó", "yo")
    # syllables: o, wo, o, wó — the 3rd (MID) sits before HIGH, surfaces raised
    assert s[2]["surface"] > s[2]["lexical"]


def test_igbo_unmarked_defaults_low():
    assert tone_of("a", "ig") == 0


# ── Tier 1 (correction): tone languages are the moat; engine is language-agnostic ──


def test_punjabi_three_tones():
    assert tone_of("ka1", "pa") == (2, 4)  # rising
    assert tone_of("ka2", "pa") == (2, 2)  # level
    assert tone_of("ka3", "pa") == (4, 0)  # falling
    assert profile_for("pa").tonal is True


def test_zulu_shona_somali_are_tonal():
    for code in ("zu", "sn", "so"):
        assert profile_for(code).tonal is True, code
    assert profile_for("zu").tone_system == "level-3+downstep"


def test_lao_five_tones_thai_lao_family():
    assert profile_for("lo").tone_system == "contour-5"
    assert profile_for("th").tone_system == "contour-5"


def test_hausa_downstep():
    assert "downstep" in profile_for("ha").tone_system


def test_vietnamese_voice_quality_slot():
    assert quality_of("mã", "vi") == "creaky"       # ngã
    assert quality_of("mạ", "vi") == "glottalized"  # nặng
    assert quality_of("mà", "vi") == "breathy"      # huyền
    assert quality_of("mả", "vi") == "creaky"       # hỏi
    assert quality_of("ma", "vi") == "modal"        # ngang
    assert quality_of("nei5", "yue") == "modal"
    assert quality_of("sik1", "yue") == "short"     # checked entering tone


def test_units_interface_carries_quality():
    units = units_of("mã", "vi")
    assert len(units) == 1
    assert units[0].quality == "creaky"
    assert units[0].phoneme  # G2P present on the unit


def test_engine_is_language_agnostic():
    # Hand-built units, NO language string anywhere: the engine must
    # still catch a rising-tone vs falling-melody clash and fix it.
    units = [ToneUnit(phoneme="ma", tone=(2, 4), quality="modal", syllable="ma"),
             ToneUnit(phoneme="ma", tone=(2, 4), quality="modal", syllable="ma")]
    mel = [(60, 0.0, 0.5), (55, 0.5, 0.5)]
    assert not check_units(units, mel)["ok"]
    fixed = fix_units(units, mel)
    assert check_units(units, fixed)["ok"]
    # Quality never changes the pitch verdict: creaky rising tone,
    # same clash, same fix.
    units_q = [ToneUnit(phoneme="ma", tone=(2, 4), quality="creaky"),
               ToneUnit(phoneme="ma", tone=(2, 4), quality="creaky")]
    assert check_units(units_q, mel)["mismatches"] == \
        check_units(units, mel)["mismatches"]


def test_second_g2p_proves_generality():
    # Yoruba (G2P #1) and Mandarin (G2P #2) both flow through the SAME
    # engine: tone_check == check_units(units_of(...)) for both.
    for lyrics, lang, mel in (
            ("owó", "yo", [(64, 0.0, 0.5), (60, 0.5, 0.5)]),
            ("ma2 ma2", "zh", [(60, 0.0, 0.5), (55, 0.5, 0.5)]),
            ("mã mã", "vi", [(60, 0.0, 0.5), (55, 0.5, 0.5)])):
        via_wrapper = tone_check(lyrics, mel, lang)
        via_engine = check_units(units_of(lyrics, lang), mel)
        assert via_wrapper["mismatches"] == via_engine["mismatches"]
        assert via_wrapper["ok"] == via_engine["ok"]
        assert tone_fix_melody(lyrics, mel, lang) == fix_units(
            units_of(lyrics, lang), mel)


# ── Tier 1: global tones ───────────────────────────────────────────────────


def test_mandarin_pinyin_tones():
    assert tone_of("ma1", "zh") == (4, 4)
    assert tone_of("ma2", "zh") == (2, 4)
    assert tone_of("ma3", "zh") == (2, 0)
    assert tone_of("ma4", "zh") == (4, 0)
    assert tone_of("mā", "zh") == (4, 4)
    assert tone_of("mǎ", "zh") == (2, 0)
    assert g2p("ma1", "zh") == ["m", "a"]


def test_cantonese_jyutping():
    assert tone_of("nei5", "yue") == (1, 2)
    assert tone_of("nei1", "yue") == (4, 4)


def test_vietnamese_six_tones():
    assert tone_of("ma", "vi") == (2, 2)      # ngang unmarked
    assert tone_of("mà", "vi") == (1, 0)      # huyền
    assert tone_of("má", "vi") == (3, 4)      # sắc
    assert tone_of("mả", "vi") == (2, 0)      # hỏi
    assert tone_of("mã", "vi") == (2, 4)      # ngã
    assert tone_of("mạ", "vi") == (0, 0)      # nặng


def test_hausa_and_ewe_profiles_exist():
    assert profile_for("ha").tonal is True
    assert profile_for("ee").tonal is True
    assert profile_for("tw").tonal is True
    assert profile_for("fon").tonal is True
    assert profile_for("en").tonal is False


def test_tone_check_global_mandarin():
    # rising tone (ma2) against a falling melody = mismatch
    r = tone_check("ma2 ma2", [(60, 0.0, 0.5), (55, 0.5, 0.5)], "zh")
    assert r["mismatches"], "rising tone vs falling melody must mismatch"
    fixed = tone_fix_melody("ma2 ma2", [(60, 0.0, 0.5), (55, 0.5, 0.5)], "zh")
    r2 = tone_check("ma2 ma2", fixed, "zh")
    assert r2["ok"]


def test_tone_check_nontonal_always_passes():
    r = tone_check("hello world", [(60, 0.0, 0.5), (40, 0.5, 0.5)], "en")
    assert r["ok"] and r["score"] == 1.0


def test_code_switch_detection():
    lines = detect_languages("Mo dé, ayọ dé\nNa wa, this love na wahala")
    assert lines[0]["languages"][0]["language"] == "yo"
    assert any(l["language"] == "pcm" for l in lines[1]["languages"])


# ── Tier 2: global rhythm ───────────────────────────────────────────────────


def test_genre_rhythm_descriptors():
    assert genre_rhythm("afrobeats").kick_rule == "tresillo"
    assert genre_rhythm("reggaeton").kick_rule == "dembow"
    assert genre_rhythm("trap").snare_rule == "half_time"
    assert genre_rhythm("salsa").snare_rule == "clave"
    assert genre_rhythm("house").kick_rule == "four_floor"
    assert genre_rhythm("unknown-genre-xyz").genre == "pop"  # fallback


def test_genre_family():
    assert genre_family("amapiano") == "african"
    assert genre_family("k-pop") == "pop"


def test_generate_genre_drums_deterministic():
    a = generate_genre_drums("afrobeats", bars=2, energy=0.7, seed=42)
    b = generate_genre_drums("afrobeats", bars=2, energy=0.7, seed=42)
    assert a == b
    assert any(e["voice"] == "kick" for e in a)
    assert any(e["voice"] == "snare" for e in a)
    # energy scales density: high energy has strictly more events
    low = generate_genre_drums("afrobeats", bars=2, energy=0.2, seed=1)
    high = generate_genre_drums("afrobeats", bars=2, energy=0.9, seed=1)
    assert len(high) >= len(low)


def test_dembow_kick_shape():
    events = generate_genre_drums("reggaeton", bars=1, energy=0.3, seed=0)
    kicks = sorted(e["time_beats"] for e in events if e["voice"] == "kick")
    assert kicks == [0.0, 1.5, 2.0, 3.5]  # dembow: 1, +and, 3, +and


def test_groove_template_extraction_and_apply():
    # onsets pushed ~10ms late on offbeat slots
    bpm = 120.0
    step_s = 0.5 / 4  # 16th at 120bpm = 0.125s
    onsets = [i * step_s + (0.012 if i % 2 else 0.0) for i in range(64)]
    t = extract_groove_template(onsets, bpm)
    assert t.confidence > 0.0
    assert t.source_onsets == 64
    events = [{"step": 1, "time_beats": 0.25, "voice": "hat"}]
    shifted = apply_groove(events, t)
    assert shifted[0]["time_beats"] != 0.25  # NOT straight-quantized


# ── Tier 2: sonic bible ─────────────────────────────────────────────────────


def test_sonic_bible_consistency():
    bible = SonicBible(artist="Test", album="One", keys=("C", "G"),
                       bpm_range=(95, 110), genres=("afrobeats",),
                       palette=("log drum", "shaker"), languages=("yo", "en"))
    good = TrackDescriptor(title="t1", key="C", bpm=100,
                           genre="afrobeats", palette=("log drum", "shaker"),
                           language="yo")
    assert check_consistency(bible, good)["consistent"]
    bad = TrackDescriptor(title="t2", key="Eb", bpm=160, genre="metal",
                          palette=("blast beat",), language="ru")
    r = check_consistency(bible, bad)
    assert not r["consistent"] and "bpm" in r["drift"]


def test_album_consistency():
    bible = SonicBible(artist="A", album="B", keys=("C",),
                       bpm_range=(90, 120), genres=("pop",))
    tracks = [TrackDescriptor(title="1", key="C", bpm=100, genre="pop"),
              TrackDescriptor(title="2", key="C", bpm=110, genre="pop")]
    assert album_consistency(bible, tracks)["consistent"]


# ── Tier 2: critique ────────────────────────────────────────────────────────


def test_singability_catches_range_and_tone():
    # "owó o": tones MID→HIGH→MID; melody falls then leaps to 95 (out of range)
    r = singability("owó o", [(60, 0.0, 0.5), (55, 0.5, 0.5), (95, 1.0, 0.5)], "yo")
    assert r.axes["tone"] < 1.0
    assert r.axes["range"] < 1.0
    assert not r.ship


def test_singability_clean_passes():
    r = singability("owó", [(60, 0.0, 0.5), (64, 0.5, 0.5)], "yo")
    assert r.ship


def test_critique_loop_fixes_tone():
    sections = [{"name": "verse",
                 "lyrics": ["owó"],
                 "melody": [(64, 0.0, 0.5), (60, 0.5, 0.5)]},
                {"name": "chorus",
                 "lyrics": ["owó owó"],
                 "melody": [(60, 0.0, 0.5), (64, 0.5, 0.5)]}]
    r = critique_loop("t", sections, "yo")
    assert r.fixes_applied  # loop actually rewrote something
    assert r.score >= critique_song("t", sections, "yo").score


def test_critique_needs_chorus():
    r = critique_song("t", [{"name": "verse", "lyrics": ["la la"],
                              "melody": [(60, 0.0, 0.5), (62, 0.5, 0.5)]}],
                      "en")
    assert r.axes["structure"] == 0.0


# ── Tier 2: deliverables ────────────────────────────────────────────────────


def test_deliverable_plan_missing_is_honest(tmp_path):
    plan = deliverable_plan("Song", 180.0, [],
                            {}, "", "", "")
    assert not plan["complete"]
    assert "master" in plan["missing"] and "acapella" in plan["missing"]
    assert any(d["kind"] == "cut_15s" for d in plan["deliverables"])
    assert any(d["kind"] == "cut_30s" for d in plan["deliverables"])
    plats = {p["platform"]: p["target_lufs"] for p in plan["platform_loudness"]}
    assert plats["spotify"] == -14.0 and plats["apple_music"] == -16.0


def test_cut_wav_real_audio(tmp_path):
    src = str(tmp_path / "src.wav")
    write_sine_wav(src, duration_s=2.0)
    dst = str(tmp_path / "cut.wav")
    r = cut_wav(src, dst, 0.5, 1.5)
    assert r["ok"] and abs(r["duration_s"] - 1.0) < 0.01
    with wave.open(dst, "rb") as w:
        assert w.getnframes() > 0


def test_cut_wav_missing_source_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        cut_wav(str(tmp_path / "nope.wav"), str(tmp_path / "o.wav"), 0, 1)


# ── Tier 3: licenses / taste / lora ────────────────────────────────────────


def test_license_registry():
    # Licence is a LABEL, not a gate: commercial_use flag per model,
    # personal_use True for every registered model.
    assert can_ship("acestep", "commercial")["commercial_use"] is True
    assert can_ship("musicgen", "commercial")["commercial_use"] is False
    assert can_ship("musicgen", "commercial")["personal_use"] is True
    assert can_ship("xtts", "commercial")["commercial_use"] is False
    assert can_ship("xtts", "private")["personal_use"] is True
    assert can_ship("matchering", "commercial")["commercial_use"] is True
    assert can_ship("totally-unknown-model", "commercial")["commercial_use"] is False


def test_licence_audit_is_query_not_gate():
    audit = licence_audit(["acestep", "diffsinger", "rvc"])
    assert audit["any_noncommercial"] is False
    audit = licence_audit(["acestep", "musicgen"])
    assert audit["any_noncommercial"] is True
    assert audit["noncommercial"] == ["musicgen"]
    assert any("GPL" in w or "copyleft" in w.lower()
               for w in licence_audit(["matchering"])["warnings"])
    # release_gate is a back-compat alias returning the same audit
    assert release_gate(["musicgen"]) == licence_audit(["musicgen"])


def test_license_report_covers_pipeline():
    models = {r["model"] for r in license_report()}
    assert {"acestep", "diffsinger", "rvc", "demucs", "matchering",
            "musicgen", "xtts"} <= models
    assert all("commercial_use" in r for r in license_report())


def test_probe_matchering_never_raises():
    p = probe_matchering()
    assert isinstance(p["ok"], bool)


def test_taste_model_cold_start_and_learning(tmp_path):
    m = TasteModel(path=str(tmp_path / "t.json"))
    spec = {"genre": "afrobeats", "bpm": 100, "energy": 0.7,
            "language": "yo"}
    assert m.score(spec)["score"] == 0.5
    assert m.score(spec)["confidence"] == 0.0
    for _ in range(5):
        m.feedback(spec, True)
    assert m.score(spec)["score"] > 0.5
    other = {"genre": "metal", "bpm": 180, "energy": 0.9, "language": "en"}
    m.feedback(other, False)
    assert m.score(other)["score"] < m.score(spec)["score"]
    m.save()
    assert TasteModel.load(str(tmp_path / "t.json")).accepts == 5


def test_genre_corpus_spec_licence_rule():
    spec = genre_corpus_spec("k-pop")
    assert "NEVER" in spec["licence"]["rule"]
    assert spec["activation_tag"] == "kpop-style"


def test_corpus_licence_check_gates_uncleared(tmp_path):
    (tmp_path / "track.wav").write_bytes(b"fake")
    r = corpus_licence_check(str(tmp_path))
    assert r["ok"] is False  # missing .licence sidecar blocks training
    (tmp_path / "track.licence").write_text("cc0")
    r2 = corpus_licence_check(str(tmp_path))
    assert r2["ok"] is True and r2["licensed"] == 1


def test_kaggle_runbook():
    rb = kaggle_runbook("afrobeats")
    assert "30h" in rb["budget_note"]
    assert any("resume" in s.lower() for s in rb["steps"])


# ── Tier 0: chain ───────────────────────────────────────────────────────────


def test_draft_melody_tone_compliant():
    notes = draft_melody("owó owó", "yo", seed=1)
    assert len(notes) == 4  # one note per syllable
    r = tone_check("o wó o wó", notes, "yo")
    assert r["ok"]


def test_write_midi_is_real_midi(tmp_path):
    p = write_midi([(60, 0.0, 0.5), (64, 0.5, 0.5)],
                   str(tmp_path / "m.mid"))
    data = open(p, "rb").read()
    assert data[:4] == b"MThd" and b"MTrk" in data


def test_chain_with_fakes_full_pipeline(tmp_path):
    req = ChainRequest(title="hello", lyrics="Mo dé, ayọ dé",
                       language="yo", workdir=str(tmp_path),
                       rvc_model="fake-voice")

    def fake_rvc(vocal_path: str, model: str, workdir: str) -> str:
        import shutil
        dst = os.path.join(workdir, "vocal-rvc.wav")
        shutil.copyfile(vocal_path, dst)
        return dst

    r = run_chain(req, bed_fn=fake_bed, vocal_fn=fake_vocal,
                  rvc_fn=fake_rvc)
    assert r.ok
    assert r.stage_status("bed") == "fake"
    assert r.stage_status("vocal") == "fake"
    assert r.stage_status("rvc") == "fake"
    assert os.path.exists(r.master_path)
    assert os.path.exists(r.midi_path)
    manifest = json.loads(open(r.manifest_path).read())
    assert manifest["stages"][1]["status"] == "fake"
    assert "FAKE" in manifest["provenance"]


def test_chain_without_models_never_fakes_a_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_SVS_DIR", str(tmp_path / "empty-svs"))
    req = ChainRequest(title="hello", lyrics="Mo dé",
                       workdir=str(tmp_path / "out"))
    r = run_chain(req)
    assert r.ok is False
    assert r.error  # explains exactly what is missing
    assert r.stage_status("bed") in ("skipped", "real")
    assert r.stage_status("vocal") in ("skipped", "real")


def test_chain_commercial_audience_labels_licence_not_gate(tmp_path):
    # Licence is a label: commercial audience does NOT block the chain.
    # The manifest's licence_audit answers "was any non-commercial?".
    req = ChainRequest(title="hello", lyrics="Mo dé",
                       workdir=str(tmp_path / "out"),
                       audience="commercial")
    r = run_chain(req, bed_fn=fake_bed, vocal_fn=fake_vocal)
    assert r.ok  # licence never gates
    manifest = json.loads(open(r.manifest_path).read())
    assert manifest["licence_audit"]["any_noncommercial"] is False
    assert manifest["licence_audit"]["commercial_use"] == [
        "acestep", "diffsinger"]
    # And the audit still flags a non-commercial model if one were used:
    assert licence_audit(["acestep", "musicgen"])["any_noncommercial"] is True
