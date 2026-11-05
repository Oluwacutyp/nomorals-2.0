"""§1.4 acceptance corpus: caption timing, measured on real speech.

Until this corpus existed, the only proof that measured caption
timings were *accurate* was a fake transcriber whose word timings
matched the reference by construction (MAE 0 — it proved the
plumbing, never the measurement). This module is the measurement:

* **Fixtures are scripts**, synthesized through Devon's real TTS
  ladder (:class:`nomorals.voice.tts.UniversalTTS`). Two
  constructions:

  - ``word_by_word`` — each token is synthesized on its own and
    the clips are concatenated with fixed silence gaps. The
    reference word timings are *exact by construction*: they are
    computed from the sample counts of the clips actually
    concatenated, read back from the written files.
  - ``fluent`` — the whole script in one synthesis, the way real
    content is made. The reference transcript is known; per-word
    reference timings are not (no TTS rung here aligns fluent
    speech), so timing is recorded as **not scored** for these
    fixtures — WER still is.

* **Transcription runs the real §2.4 path** —
  :func:`edit_engine.text.captions_from_audio` over
  :class:`edit_engine.text.STTTranscriber` — and the scores are
  computed from whatever it actually returns.

* **Scoring is pure** (WER + word-timing MAE over a Levenshtein
  alignment) and unit-tested without any rung installed.

``tests/golden/caption_corpus.json`` records the floors and the
values measured when the manifest was written, in the mask-corpus
idiom: a rung that cannot run on a machine is a *named gap* in the
manifest, never an invented number.
"""

from __future__ import annotations

import re
import wave
from array import array
from pathlib import Path
from typing import Any

#: silence before the first word / between words, seconds. The gap
#: sits above faster-whisper's default VAD min-silence (500 ms is
#: the *segment* floor; 0.45 s of true silence between isolated
#: words keeps word boundaries unambiguous without letting VAD
#: swallow short words).
LEAD_S = 0.30
GAP_S = 0.45

FIXTURES: list[dict[str, Any]] = [
    {"name": "fox_pangram",
     "text": "the quick brown fox jumps over the lazy dog",
     "construction": "word_by_word",
     "notes": "The classic pangram; short function words (the, "
              "over) are the timing stress — easy for a transcriber "
              "to merge or drop."},
    {"name": "market_morning",
     "text": "she sells fresh bread at the market every morning",
     "construction": "word_by_word",
     "notes": "Sibilant-heavy opening (she sells fresh) — a WER "
              "trap for small STT models."},
    {"name": "small_tools",
     "text": "we build small tools that solve real problems",
     "construction": "word_by_word",
     "notes": "Plain declarative sentence, mid-length words."},
    {"name": "tin_roof",
     "text": "the rain fell hard on the tin roof last night",
     "construction": "word_by_word",
     "notes": "Two 'the'-class function words plus a hard/rain "
              "consonant cluster."},
    {"name": "water_plants",
     "text": "please remember to water the plants before you leave",
     "construction": "word_by_word",
     "notes": "Polite-imperative phrasing; 'please' and 'before' "
              "are the words small models most often soften."},
    {"name": "fluent_caption_timing",
     "text": "caption timing matters because viewers read along "
             "with the speaker",
     "construction": "fluent",
     "notes": "Fluent synthesis: WER scored, per-word timing not "
              "scored (no fluent reference timings exist)."},
    {"name": "fluent_good_edit",
     "text": "a good edit keeps the words and the pictures in step",
     "construction": "fluent",
     "notes": "Fluent synthesis: WER scored, per-word timing not "
              "scored (no fluent reference timings exist)."},
]


def fixture_names() -> list[str]:
    return [f["name"] for f in FIXTURES]


def get_fixture(name: str) -> dict[str, Any]:
    for f in FIXTURES:
        if f["name"] == name:
            return f
    raise KeyError(f"unknown caption corpus fixture {name!r}; "
                   f"use one of {fixture_names()}")


# ---------------------------------------------------------------------------
# scoring — pure functions, no rungs involved
# ---------------------------------------------------------------------------

_TOKEN_EDGE = re.compile(r"^[^a-z0-9']+|[^a-z0-9']+$")


def normalize_token(tok: Any) -> str:
    """Lowercase + strip edge punctuation (WER's token contract)."""
    return _TOKEN_EDGE.sub("", str(tok or "").strip().lower())


def _edit_alignment(ref: list[str], hyp: list[str]
                    ) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment: (ref_i, hyp_i) pairs; None = gap.

    Substitution pairs both indices (the hypothesis word occupies
    the reference word's time slot — it counts for timing MAE);
    deletions pair (ref_i, None), insertions (None, hyp_i).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,
                           dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
            continue
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def word_error_rate(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """(S + D + I) / N over normalized tokens. Empty ref → 0/1."""
    ref = [t for t in (normalize_token(t) for t in ref_tokens) if t]
    hyp = [t for t in (normalize_token(t) for t in hyp_tokens) if t]
    if not ref:
        return 0.0 if not hyp else 1.0
    pairs = _edit_alignment(ref, hyp)
    errors = sum(1 for a, b in pairs if a is None or b is None
                 or ref[a] != hyp[b])
    return errors / len(ref)


def score_transcription(reference_text: str,
                        reference_words: list[dict[str, Any]] | None,
                        got_words: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one transcription against its reference.

    ``reference_words`` (exact timings) is required for the timing
    scores; pass ``None`` for fluent fixtures — the result then
    carries ``timing: "not_scored"`` with the reason, and only WER
    is a number. Got words are dicts with text/word + start/end,
    the :func:`captions_from_audio` shape.
    """
    ref_tokens = [t for t in
                  (normalize_token(t) for t in reference_text.split())
                  if t]
    hyp_words = [w for w in got_words
                 if normalize_token(w.get("text") or w.get("word"))]
    hyp_tokens = [normalize_token(w.get("text") or w.get("word"))
                  for w in hyp_words]
    wer = word_error_rate(ref_tokens, hyp_tokens)
    out: dict[str, Any] = {
        "wer": round(wer, 4),
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
    }
    if reference_words is None:
        out["timing"] = "not_scored"
        out["timing_note"] = (
            "fluent construction: reference transcript known, "
            "per-word reference timings do not exist")
        return out
    ref_norm = [normalize_token(w["text"]) for w in reference_words]
    pairs = _edit_alignment(ref_norm, hyp_tokens)
    start_errs: list[float] = []
    end_errs: list[float] = []
    exact = 0
    for a, b in pairs:
        if a is None or b is None:
            continue
        if ref_norm[a] == hyp_tokens[b]:
            exact += 1
        start_errs.append(abs(float(hyp_words[b]["start"])
                              - float(reference_words[a]["start"])))
        end_errs.append(abs(float(hyp_words[b]["end"])
                            - float(reference_words[a]["end"])))
    if not start_errs:
        out["timing"] = "not_scored"
        out["timing_note"] = "no aligned word pairs to time"
        return out
    out["timing"] = "scored"
    out["mae_start_s"] = round(sum(start_errs) / len(start_errs), 4)
    out["mae_end_s"] = round(sum(end_errs) / len(end_errs), 4)
    out["timed_pairs"] = len(start_errs)
    out["exact_pairs"] = exact
    return out


# ---------------------------------------------------------------------------
# fixture synthesis — through the real TTS ladder
# ---------------------------------------------------------------------------

def _read_wav_samples(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(
                f"{path}: expected mono 16-bit WAV, got "
                f"{w.getnchannels()}ch/{w.getsampwidth() * 8}bit")
        frames = w.readframes(w.getnframes())
        return array("h", frames), w.getframerate()


def _write_wav_samples(path: Path, samples: array, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def build_fixture_audio(fixture: dict[str, Any], *, tts: Any,
                        out_dir: str | Path) -> dict[str, Any]:
    """Synthesize one fixture through a :class:`UniversalTTS`.

    Returns ``{path, sample_rate, backend, construction,
    reference_text, reference_words}`` — ``reference_words`` is the
    exact-by-construction timing list for ``word_by_word``
    fixtures, ``None`` for fluent ones.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = fixture["text"]
    tokens = text.split()
    if fixture["construction"] == "fluent":
        res = tts.speak(text, out_path=str(out / f"{fixture['name']}.wav"))
        return {"path": res["path"], "sample_rate": res["sample_rate"],
                "backend": res["backend"],
                "construction": "fluent",
                "reference_text": text, "reference_words": None}
    # word_by_word: one synthesis per token; the reference timeline
    # is built from the sample counts of the files actually written.
    clips: list[tuple[str, array, int]] = []
    sample_rate: int | None = None
    backend = ""
    for i, tok in enumerate(tokens):
        res = tts.speak(tok, out_path=str(
            out / f"{fixture['name']}.w{i:02d}.wav"))
        samples, sr = _read_wav_samples(Path(res["path"]))
        if sample_rate is None:
            sample_rate, backend = sr, res["backend"]
        elif sr != sample_rate:
            raise ValueError(
                f"{fixture['name']}: TTS sample rate changed "
                f"mid-fixture ({sample_rate} → {sr}); cannot build "
                "an exact reference timeline")
        if len(samples) == 0:
            raise ValueError(
                f"{fixture['name']}: TTS produced no audio for "
                f"token {tok!r}")
        clips.append((tok, samples, sr))
    assert sample_rate is not None
    merged: array = array("h")
    reference: list[dict[str, Any]] = []
    cursor = 0.0

    def _silence(seconds: float) -> array:
        return array("h", bytes(2 * int(round(seconds * sample_rate))))

    merged.extend(_silence(LEAD_S))
    cursor += LEAD_S
    for i, (tok, samples, _sr) in enumerate(clips):
        dur = len(samples) / sample_rate
        reference.append({"text": tok, "start": round(cursor, 4),
                          "end": round(cursor + dur, 4)})
        merged.extend(samples)
        cursor += dur
        if i < len(clips) - 1:
            merged.extend(_silence(GAP_S))
            cursor += GAP_S
    final = out / f"{fixture['name']}.wav"
    _write_wav_samples(final, merged, sample_rate)
    return {"path": str(final), "sample_rate": sample_rate,
            "backend": backend, "construction": "word_by_word",
            "reference_text": text, "reference_words": reference}


# ---------------------------------------------------------------------------
# the harness — generation → transcription → scoring
# ---------------------------------------------------------------------------

def run_corpus(out_dir: str | Path, *, tts: Any = None,
               transcriber: Any = None) -> dict[str, Any]:
    """Run every fixture end to end and return per-fixture results.

    A fixture whose rungs cannot run is reported with
    ``status: "gap"`` and the rung's own reason — the harness never
    substitutes a number it did not measure.
    """
    from .text import captions_from_audio

    if tts is None:
        from ...voice.tts import UniversalTTS
        tts = UniversalTTS(backend="piper")
    results: list[dict[str, Any]] = []
    for fixture in FIXTURES:
        entry: dict[str, Any] = {"name": fixture["name"],
                                 "construction": fixture["construction"]}
        try:
            built = build_fixture_audio(fixture, tts=tts,
                                        out_dir=Path(out_dir)
                                        / fixture["name"])
        except Exception as exc:
            entry.update(status="gap",
                         gap=f"TTS rung produced no fixture audio: "
                             f"{type(exc).__name__}: {exc}")
            results.append(entry)
            continue
        entry["tts_backend"] = built["backend"]
        try:
            got = captions_from_audio(built["path"],
                                      text=fixture["text"],
                                      transcriber=transcriber)
        except Exception as exc:
            entry.update(status="gap",
                         gap=f"transcription path failed: "
                             f"{type(exc).__name__}: {exc}")
            results.append(entry)
            continue
        if got["timing_source"] != "measured":
            entry.update(
                status="gap",
                gap="no STT rung produced word timings "
                    f"(captions_from_audio fell back to "
                    f"{got['timing_source']!r})")
            results.append(entry)
            continue
        scores = score_transcription(built["reference_text"],
                                     built["reference_words"],
                                     got["words"])
        entry.update(status="measured", transcriber=got["transcriber"],
                     **scores)
        results.append(entry)
    return {"fixtures": results}
