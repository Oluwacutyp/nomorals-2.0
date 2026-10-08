"""Tests for nomorals.media.afrobeats — all offline, no models, no network."""

import json
import unicodedata

import pytest

from nomorals.media.afrobeats import (
    HIGH, MID, LOW,
    Alignment,
    AfrobeatsLoRA,
    LoRAUnavailable,
    LoRATrainingJob,
    NaijaSungVocals,
    corpus_spec,
    parse_naija_song_request,
    parse_tones,
    tone_aware_melody,
    tone_to_melody,
)


# ── corpus spec ─────────────────────────────────────────────────────────────


def test_corpus_spec_format():
    spec = corpus_spec()
    assert spec["bpm_ranges"]["afrobeats"] == (95, 110)
    assert spec["bpm_ranges"]["amapiano"] == (110, 115)
    assert "licensing" in spec
    assert "scrape" in spec["licensing"].lower() or "cleared" in spec["licensing"].lower()
    assert len(spec["pipeline"]) == 3  # prepare → preprocess → train
    assert "acestep-prepare" in spec["pipeline"][0]


# ── tone parsing ────────────────────────────────────────────────────────────


def test_parse_tones_yoruba():
    syls = parse_tones("owó", "yoruba")
    assert [s.text for s in syls] == ["o", "wó"] or len(syls) == 2
    tones = [s.tone for s in syls]
    assert tones == [MID, HIGH]


def test_parse_tones_low():
    syls = parse_tones("ìgba", "yoruba")
    assert [s.tone for s in syls] == [LOW, MID]


def test_parse_tones_precomposed_equals_decomposed():
    pre = parse_tones("owó", "yoruba")  # precomposed ó
    decomp = parse_tones(unicodedata.normalize("NFD", "owó"), "yoruba")
    assert [s.tone for s in pre] == [s.tone for s in decomp]


def test_parse_tones_igbo_unmarked_defaults_low():
    syls = parse_tones("akwa", "igbo")
    assert all(s.tone in (HIGH, LOW) for s in syls)
    # unmarked → LOW per Igbo marking convention
    assert syls[0].tone == LOW


def test_parse_tones_multiword():
    syls = parse_tones("mo dúpẹ́", "yoruba")
    assert [s.tone for s in syls] == [MID, HIGH, HIGH]


# ── tone ↔ melody alignment ─────────────────────────────────────────────────


def _mel(rising=True):
    # two notes: rising or falling
    return [(60, 0.0, 1.0), (64 if rising else 55, 1.0, 1.0)]


def test_alignment_matching_contours_ok():
    # "owó": MID → HIGH (rises); melody rises → OK
    al = tone_to_melody("owó", _mel(rising=True), "yoruba")
    assert isinstance(al, Alignment)
    assert al.score == 1.0
    assert al.mismatches == []


def test_alignment_opposite_contours_flagged():
    # tone rises, melody falls → the hard error
    al = tone_to_melody("owó", _mel(rising=False), "yoruba")
    assert al.score == 0.0
    assert len(al.mismatches) == 1
    assert "fights the word's tone" in al.mismatches[0].note_text


def test_alignment_level_tone_is_flexible():
    # "baba" all MID — any melody direction is singable
    al = tone_to_melody("baba", _mel(rising=False), "yoruba")
    assert al.score == 1.0


def test_alignment_pidgin_skips():
    al = tone_to_melody("how far na", _mel(rising=False), "pidgin")
    assert al.score == 1.0
    assert al.verdicts == []


def test_tone_aware_melody_fixes_clash():
    fixed = tone_aware_melody("owó", _mel(rising=False), "yoruba")
    # second note must no longer fall below the first on a rising tone
    assert fixed[1][0] >= fixed[0][0]
    # and the fix is now clean
    al = tone_to_melody("owó", fixed, "yoruba")
    assert al.score == 1.0


def test_tone_aware_melody_capped_nudge():
    fixed = tone_aware_melody("owó", _mel(rising=False), "yoruba",
                              max_nudge=3)
    # never pushed more than max_nudge past the compliant floor (first+1),
    # and the result is compliant
    assert fixed[1][0] <= fixed[0][0] + 1 + 3
    al = tone_to_melody("owó", fixed, "yoruba")
    assert al.score == 1.0


def test_tone_aware_melody_falling_tone():
    # "é à": HIGH → LOW (falls); melody rises → mismatch → fixed downward
    mel = [(60, 0.0, 1.0), (64, 1.0, 1.0)]
    al = tone_to_melody("é à", mel, "yoruba")
    assert len(al.mismatches) == 1
    fixed = tone_aware_melody("é à", mel, "yoruba")
    assert fixed[1][0] <= fixed[0][0]
    assert tone_to_melody("é à", fixed, "yoruba").score == 1.0


def test_tone_aware_melody_leaves_good_melody_alone():
    mel = _mel(rising=True)
    assert tone_aware_melody("owó", mel, "yoruba") == mel


# ── LoRA training pipeline ──────────────────────────────────────────────────


def test_training_job_validates_missing_corpus(tmp_path):
    job = LoRATrainingJob(corpus_dir=str(tmp_path / "nope"),
                          output_dir=str(tmp_path / "out"))
    problems = job.validate()
    assert any("corpus" in p for p in problems)


def test_training_job_flags_thin_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.wav").write_bytes(b"RIFF....")
    job = LoRATrainingJob(corpus_dir=str(corpus),
                          output_dir=str(tmp_path / "out"),
                          preset="medium")
    problems = job.validate()
    assert any("tracks" in p for p in problems)


def test_training_job_commands(tmp_path):
    job = LoRATrainingJob(corpus_dir="/c", output_dir="/o", preset="medium")
    assert "--r 16" in job.train_command()
    assert "--alpha 32" in job.train_command()
    assert "acestep-prepare" in job.prepare_command()
    assert "naija-afrobeats-style" in job.prepare_command()
    recipe = job.recipe()
    assert "acestep-prepare" in recipe and "acestep-train" in recipe


def test_training_job_run_fails_closed(tmp_path):
    job = LoRATrainingJob(corpus_dir=str(tmp_path / "nope"),
                          output_dir=str(tmp_path / "out"))
    with pytest.raises(LoRAUnavailable):
        job.run()


def test_train_lora_dry_run_does_not_execute(tmp_path):
    lora = AfrobeatsLoRA()
    job = lora.train_lora(str(tmp_path / "corpus"), dry_run=True)
    assert isinstance(job, LoRATrainingJob)
    # dry run never raises, never shells out


def test_load_lora_rejects_empty_dir(tmp_path):
    with pytest.raises(LoRAUnavailable):
        AfrobeatsLoRA.load_lora(str(tmp_path))


def test_load_lora_accepts_weights(tmp_path):
    d = tmp_path / "lora-afrobeats"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (d / "adapter_config.json").write_text(json.dumps({"r": 16}))
    w = AfrobeatsLoRA.load_lora(str(d))
    assert w.name == "lora-afrobeats"
    assert w.path == str(d)


def test_apply_lora_attaches_to_backend(tmp_path):
    from nomorals.media.ace_step import ACEStepBackend
    d = tmp_path / "lora-x"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"x")
    backend = ACEStepBackend()
    AfrobeatsLoRA().apply_lora(backend, str(d), scale=0.8)
    assert backend.lora_path == str(d)
    assert backend.lora_scale == 0.8


def test_lora_registry_roundtrip(tmp_path):
    d = tmp_path / "lora-ab"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"x")
    lora = AfrobeatsLoRA(registry_path=str(tmp_path / "loras.json"))
    lora.register_lora("afrobeats", str(d))
    assert lora.for_style("afrobeats") == str(d)
    assert lora.for_style("highlife") == ""
    # persists
    lora2 = AfrobeatsLoRA(registry_path=str(tmp_path / "loras.json"))
    assert lora2.for_style("afrobeats") == str(d)


# ── chat parsing ────────────────────────────────────────────────────────────


def test_parse_naija_natural():
    from nomorals.media.music import STYLE_ALIASES
    req = parse_naija_song_request(
        "make me an afrobeats song in Yoruba about Lagos",
        styles=STYLE_ALIASES)
    assert req is not None
    assert req.topic == "Lagos"
    assert req.style == "afrobeats"
    assert req.language == "yoruba"


def test_parse_naija_igbo():
    from nomorals.media.music import STYLE_ALIASES
    req = parse_naija_song_request(
        "generate a highlife song in Igbo about home",
        styles=STYLE_ALIASES)
    assert req is not None
    assert req.language == "igbo"
    assert req.style == "highlife"


def test_parse_naija_slash():
    from nomorals.media.music import STYLES
    req = parse_naija_song_request(
        "/music naija lagos nights afrobeats yoruba", styles=STYLES)
    assert req is not None
    assert req.topic == "lagos nights"
    assert req.style == "afrobeats"
    assert req.language == "yoruba"


def test_parse_naija_rejects_non_song():
    assert parse_naija_song_request("what's the weather") is None
    assert parse_naija_song_request("") is None


# ── singing (mocked chain) ──────────────────────────────────────────────────


class _FakeChain:
    def __init__(self):
        self.render_melody = None

    def render(self, lyrics, melody, workdir="vocals", title="vocal"):
        from nomorals.media.vocals import VocalResult
        self.render_melody = list(melody)
        return VocalResult(ok=True, audio_path="/tmp/dry.wav",
                           stage="render")

    def convert(self, vocal_wav, voice_id, audience="private",
                workdir="vocals"):
        from nomorals.media.vocals import VocalResult
        return VocalResult(ok=True, audio_path="/tmp/conv.wav",
                           stage="convert", voice_id=voice_id,
                           audience=audience)


def test_sing_fixes_melody_before_render():
    fake = _FakeChain()
    singer = NaijaSungVocals(chain=fake)
    res = singer.sing("owó", _mel(rising=False), "test-voice",
                      language="yoruba")
    assert res.ok
    assert res.fixed_melody is True
    # the chain received the FIXED melody, not the clashing one
    assert fake.render_melody[1][0] >= fake.render_melody[0][0]
    assert res.alignment.score == 0.0  # pre-fix alignment reported honestly


def test_sing_rejects_bad_language():
    singer = NaijaSungVocals(chain=_FakeChain())
    from nomorals.media.vocals import VocalModelUnavailable
    with pytest.raises(VocalModelUnavailable):
        singer.sing("hello", _mel(), "v", language="french")


def test_sing_rejects_empty():
    singer = NaijaSungVocals(chain=_FakeChain())
    from nomorals.media.vocals import VocalModelUnavailable
    with pytest.raises(VocalModelUnavailable):
        singer.sing("", _mel(), "v")
