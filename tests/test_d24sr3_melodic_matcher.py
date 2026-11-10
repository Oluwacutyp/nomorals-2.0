"""Doc 24 SR3 — the melodic matcher (class-3 recognition).

Pins the SR3 doctrine end to end over synthesized fixtures:

* a re-performance of an enrolled work (different timbre, tempo
  and key) returns the source work as the TOP ranked hypothesis;
* a hummed-style query (single noisy line) ranks the right work
  first among distractors;
* an unrelated melody gets NO hypothesis (no forced match);
* transposition is handled by the representation and the receipt
  names the estimated key relationship;
* hypotheses are labelled hypothesis-class everywhere — receipt,
  renderer, /sham and /audio surfaces — and NEVER render as
  identifications (no "matched" status, no /play line);
* enrollment derives works from enrolled recordings (the shipped
  idiom); matching never enrolls anything.

Fixture scope: every melody here is synthesized. These tests pin
behaviour, not real-world cover/hum accuracy — per doc 24 §9 that
stays unproven until the owner's-library blind set runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nomorals.audio.fingerprint import (
    MATCH_CLASS_EXACT_RECORDING,
    FingerprintDB,
)
from nomorals.audio.melodic import (
    MATCH_CLASS_MELODIC_HYPOTHESIS,
    MelodyCatalog,
    enroll_work,
    extract_melody,
    match_melody_db,
    melody_intervals,
    render_hypothesis_lines,
)
from nomorals.audio.songrecon_bench import (
    excerpt,
    synth_sibling,
    synth_track,
    write_wav,
)

SR = 11025

MELODY_A = [(64, 1), (67, 1), (69, 1), (67, 1), (64, 1), (62, 1),
            (60, 2), (62, 1), (64, 1), (69, 1), (67, 1), (65, 1),
            (64, 2)]
MELODY_B = [(62, 1), (65, 1), (69, 1), (67, 1), (65, 1), (64, 1),
            (62, 2), (64, 1), (65, 1), (67, 1), (65, 2)]
MELODY_C = [(72, 2), (71, 1), (69, 1), (67, 1), (69, 2), (72, 1),
            (74, 1), (76, 2)]
#: A leap-grammar melody sharing almost no interval vocabulary
#: with A/B/C — the unrelated negative.
MELODY_STRANGER = [(48, 1), (60, 1), (55, 1), (67, 1), (53, 1),
                   (65, 1), (50, 1), (62, 1), (57, 1), (69, 1),
                   (52, 1), (64, 2)]


def render_melody(notes, *, tempo=1.0, transpose=0,
                  harmonics=(1.0, 0.35, 0.15), seed=1, noise=0.0,
                  accompaniment=True, vibrato=0.0, sr=SR):
    """Render a (midi, beats) melody as a sine-stack performance."""
    rng = np.random.default_rng(seed)
    beat = 0.42 / tempo
    total = sum(d for _, d in notes) * beat + 0.4
    out = np.zeros(int(total * sr) + 2048)
    t = 0.15
    for midi, dur_beats in notes:
        dur = dur_beats * beat
        f0 = 440.0 * 2 ** ((midi + transpose - 69) / 12)
        n = int(dur * sr)
        tt = np.arange(n) / sr
        env = np.minimum(1.0, tt / 0.02) * np.exp(-tt / (dur * 1.4))
        warp = tt + vibrato * np.sin(2 * np.pi * 5.0 * tt)
        wave = sum(a * np.sin(2 * np.pi * f0 * k * warp)
                   for k, a in enumerate(harmonics, 1) if a > 0)
        s = int(t * sr)
        out[s:s + n] += wave * env * 0.5
        if accompaniment:
            f0b = 440.0 * 2 ** ((midi - 24 + transpose - 69) / 12)
            nb = int(min(dur, beat) * sr)
            tb = np.arange(nb) / sr
            envb = np.minimum(1.0, tb / 0.02) \
                * np.exp(-tb / (beat * 1.2))
            out[s:s + nb] += 0.3 * np.sin(2 * np.pi * f0b * tb) * envb
        t += dur
    if noise:
        out = out + rng.normal(0, noise, len(out))
    return out


@pytest.fixture()
def catalog(tmp_path):
    cat = MelodyCatalog(str(tmp_path / "melodies.db"))
    yield cat
    cat.close()


@pytest.fixture()
def wav(tmp_path):
    def _write(name, samples):
        return write_wav(tmp_path / name, samples)
    return _write


def _enroll_abc(catalog, wav):
    for title, melody, seed in (("Work A", MELODY_A, 101),
                                ("Work B", MELODY_B, 102),
                                ("Work C", MELODY_C, 103)):
        path = wav(f"{title}.wav", render_melody(melody, seed=seed))
        res = catalog.add_work(path, title=title, artist="Fixture",
                               prefer_stem=False)
        assert res["ok"], res


# ---------------------------------------------------------------------------
# the matcher itself
# ---------------------------------------------------------------------------

def test_reperformance_ranks_source_work_first(catalog, wav):
    _enroll_abc(catalog, wav)
    query = wav("b-cover.wav", render_melody(
        MELODY_B, tempo=1.12, transpose=2, harmonics=(1.0, 0.2, 0.55),
        seed=9))
    res = catalog.match(query, prefer_stem=False)
    assert res["ok"], res
    top = res["hypotheses"][0]
    assert top["title"] == "Work B"
    receipt = res["receipt"]
    assert receipt["status"] == "hypotheses"
    assert receipt["match_class"] == MATCH_CLASS_MELODIC_HYPOTHESIS
    strength = receipt["match_strength"]
    assert strength["name"] == "interval_alignment_fraction"
    assert strength["value"] == top["match_strength"]["value"]
    assert "transposition-invariant" in strength["basis"]
    assert "not a probability" in strength["calibration"]


def test_hummed_query_ranks_right_work_first(catalog, wav):
    _enroll_abc(catalog, wav)
    hum = wav("b-hum.wav", render_melody(
        MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
        noise=0.01, accompaniment=False, vibrato=0.0016))
    res = catalog.match(hum, prefer_stem=False)
    assert res["ok"], res
    assert res["hypotheses"][0]["title"] == "Work B"
    assert res["receipt"]["status"] == "hypotheses"


def test_unrelated_melody_gets_no_hypothesis(catalog, wav):
    _enroll_abc(catalog, wav)
    query = wav("stranger.wav", render_melody(MELODY_STRANGER,
                                              seed=31))
    res = catalog.match(query, prefer_stem=False)
    assert not res["ok"]
    assert res["hypotheses"] == []
    assert res["receipt"]["status"] == "no_match"
    assert res["receipt"]["match_class"] == "no_reliable_match"
    assert res["receipt"]["match_class"] != MATCH_CLASS_EXACT_RECORDING


def test_transposition_is_invariant_and_named(catalog, wav):
    _enroll_abc(catalog, wav)
    query = wav("b-up7.wav", render_melody(MELODY_B, transpose=7,
                                           seed=11))
    res = catalog.match(query, prefer_stem=False)
    assert res["ok"], res
    top = res["hypotheses"][0]
    assert top["title"] == "Work B"
    assert top["transposition_semitones"] == 7
    alignment = res["receipt"]["alignment"]
    assert alignment["transposition_invariant"] is True


def test_self_match_is_near_perfect(catalog, wav):
    path = wav("a.wav", render_melody(MELODY_A, seed=101))
    res = catalog.add_work(path, title="Work A", artist="Fixture",
                           prefer_stem=False)
    assert res["ok"]
    got = catalog.match(path, prefer_stem=False)
    assert got["ok"]
    assert got["hypotheses"][0]["title"] == "Work A"
    assert got["hypotheses"][0]["match_strength"]["value"] >= 0.9


def test_too_little_melody_is_an_honest_no_match(catalog, wav):
    _enroll_abc(catalog, wav)
    tiny = wav("tiny.wav", render_melody(
        [(62, 1), (65, 1), (64, 1), (62, 2)], seed=41))
    res = catalog.match(tiny, prefer_stem=False)
    assert not res["ok"]
    assert "too little melody" in res["reason"]
    assert res["receipt"]["status"] == "no_match"


def test_matching_never_enrolls(catalog, wav):
    _enroll_abc(catalog, wav)
    before = catalog.catalog_snapshot()
    hum = wav("b-hum.wav", render_melody(
        MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
        noise=0.01, accompaniment=False, vibrato=0.0016))
    catalog.match(hum, prefer_stem=False)
    stranger = wav("stranger.wav", render_melody(MELODY_STRANGER,
                                                 seed=31))
    catalog.match(stranger, prefer_stem=False)
    assert catalog.catalog_snapshot() == before


def test_extraction_recovers_interval_shape():
    melody = extract_melody(render_melody(MELODY_B, seed=102), SR)
    truth = [b[0] - a[0]
             for a, b in zip(MELODY_B, MELODY_B[1:], strict=False)]
    got = list(melody_intervals(melody))
    # One merged or split note is tolerable; the shape must be
    # recognizably the same line (the catalog self-match above pins
    # the end-to-end consequence).
    assert len(got) >= len(truth) - 2
    matches = sum(1 for g, t in zip(got, truth, strict=False)
                  if g == t)
    assert matches >= len(truth) - 3


# ---------------------------------------------------------------------------
# enrollment idiom + class separation
# ---------------------------------------------------------------------------

def test_work_derives_from_enrolled_recording(tmp_path, wav):
    fdb = FingerprintDB(str(tmp_path / "fp.db"))
    cat = MelodyCatalog(str(tmp_path / "melodies.db"))
    try:
        track = wav("track00.wav", synth_track(20000, 18.0))
        added = fdb.add_track(track, title="Bench Track 00",
                              artist="SongRecon Bench")
        assert added["ok"]
        work = cat.enroll_fingerprint_track(
            fdb, int(added["track_id"]), track)
        assert work["ok"], work
        assert work["title"] == "Bench Track 00"
        query = wav("sibling.wav",
                    excerpt(synth_sibling(20000, 18.0), 0.0, 10.0))
        res = cat.match(query, prefer_stem=False)
        assert res["ok"], res
        assert res["hypotheses"][0]["title"] == "Bench Track 00"
        # The same excerpt through the recording matcher is a
        # class-1 identification — the classes stay distinct.
        fp = fdb.match(query)
        assert fp["ok"]
        assert fp["receipt"]["match_class"] == \
            MATCH_CLASS_EXACT_RECORDING
        assert fp["receipt"]["status"] == "matched"
        assert res["receipt"]["match_class"] != \
            fp["receipt"]["match_class"]
    finally:
        cat.close()
        fdb.close()


def test_hypothesis_rendering_never_identifies(catalog, wav):
    _enroll_abc(catalog, wav)
    hum = wav("b-hum.wav", render_melody(
        MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
        noise=0.01, accompaniment=False, vibrato=0.0016))
    res = catalog.match(hum, prefer_stem=False)
    text = "\n".join(render_hypothesis_lines(res["hypotheses"]))
    assert "interval alignment fraction" in text
    assert "not an identification" in text
    assert "confidence" not in text.lower()
    assert "/play" not in text
    assert "I know this one" not in text


# ---------------------------------------------------------------------------
# surfaces (HOME redirected into tmp so the real catalogs are untouched)
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_audio_fingerprint_also_enrolls_the_melody(fake_home, wav):
    from nomorals.audio.edit import control_audio

    path = wav("song.wav", render_melody(MELODY_B, seed=102))
    reply = control_audio(f"fingerprint {path} My Song")
    assert "fingerprinted 'My Song'" in reply
    assert "melody enrolled for cover/hum matching" in reply


def test_audio_melody_verbs(fake_home, wav):
    from nomorals.audio.edit import control_audio

    path = wav("song.wav", render_melody(MELODY_B, seed=102))
    reply = control_audio(f"melody enroll {path} Work B")
    assert "melody enrolled 'Work B'" in reply
    hum = wav("hum.wav", render_melody(
        MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
        noise=0.01, accompaniment=False, vibrato=0.0016))
    reply = control_audio(f"melody match {hum}")
    assert "melodic hypotheses" in reply
    assert "Work B" in reply
    assert "not an identification" in reply
    assert "/play" not in reply
    stranger = wav("stranger.wav", render_melody(MELODY_STRANGER,
                                                seed=31))
    reply = control_audio(f"melody match {stranger}")
    assert "no melodic hypothesis" in reply


def test_sham_melodic_step_reply_shape(fake_home, wav):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    from nomorals.audio.melodic import default_melody_catalog

    path = wav("song.wav", render_melody(MELODY_B, seed=102))
    res = enroll_work(path, title="Work B", artist="Fixture",
                      catalog=default_melody_catalog())
    assert res["ok"], res
    mixin = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    hum = wav("hum.wav", render_melody(
        MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
        noise=0.01, accompaniment=False, vibrato=0.0016))
    reply = mixin._sham_melodic_hypotheses(hum)
    assert reply is not None
    assert "no recording match" in reply
    assert "Work B" in reply
    assert "not an identification" in reply
    assert "AudD external catalogue was not searched" in reply
    assert "/play" not in reply
    stranger = wav("stranger.wav", render_melody(MELODY_STRANGER,
                                                seed=31))
    assert mixin._sham_melodic_hypotheses(stranger) is None


def test_sham_command_details_stay_honest():
    from nomorals.social.chat.control import COMMAND_DETAILS

    what = COMMAND_DETAILS["sham"]["what"]
    assert "coherent-offset fraction" in what
    assert "not a probability" in what
    assert "AudD is the labelled external fallback" in what
    assert "melodic hypotheses" in what
    assert "confidence" not in what.lower()


def test_match_melody_db_module_api(tmp_path, wav):
    cat = MelodyCatalog(str(tmp_path / "melodies.db"))
    try:
        path = wav("song.wav", render_melody(MELODY_B, seed=102))
        res = enroll_work(path, title="Work B", artist="Fixture",
                          catalog=cat)
        assert res["ok"]
        hum = wav("hum.wav", render_melody(
            MELODY_B, tempo=0.9, harmonics=(1.0, 0.08, 0.0), seed=10,
            noise=0.01, accompaniment=False, vibrato=0.0016))
        got = match_melody_db(hum, catalog=cat)
        assert got["ok"]
        assert got["hypotheses"][0]["title"] == "Work B"
        assert os.path.exists(cat.db_path)
    finally:
        cat.close()


def test_melody_catalog_lives_beside_the_fingerprint_db(fake_home):
    # The default melody catalog is a sibling file of the class-1
    # catalog — the two stores never share a table.
    from nomorals.audio.melodic import _default_catalog_path

    path = _default_catalog_path()
    assert path.endswith("melodies.db")
    assert str(fake_home) in path
