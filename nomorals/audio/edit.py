"""Transcript-as-timeline audio editing (Descript pattern, audio edition).

Editing becomes writing: transcribe anything (voice note, meeting, podcast
draft) with word timings, edit the TEXT, and the edits land back on the
audio. The audio twin of ``media_edit.transcript_edit`` (which is video).

Pipeline::

    t = transcript_edit("voice_note.m4a")      # enhance → transcribe → words
    remove_fillers("voice_note.m4a")           # "cut the umms" for audio
    apply_edits(t, [Edit.delete(1.2, 3.4),     # deletion = cut
                    Edit.rewrite(5.0, 6.5,     # rewrite = re-synthesize in
                                 "the new words")])  # the speaker's cloned voice

Cleanup-as-preprocessor (the Adobe Enhance pattern): every transcription
first passes through an ffmpeg denoiser (``afftdn``, with a highpass/lowpass
fallback). If no denoiser is available the original is used, honestly.

Rewrite re-synthesis uses the PRIVATE voice stack (XTTS v2 zero-shot
cloning, non-commercial license — owner's own audio only). The speaker's
voice profile is built from the source audio itself, so the inserted
words sound like the same speaker. No TTS backend → rewrite edits are
refused with a clear reason; never fake audio.

Filler removal is multi-language. ``FILLERS_BY_LANG`` ships starter sets
for English, Yoruba, Ekiti/Ilawe Ekiti, Nigerian Pidgin, Hausa and Igbo —
the user can extend them per call. Ekiti inherits the Yoruba base (it is
a Yoruba dialect); matching is Unicode-aware so "ẹẹm" works.

Natural language (routed via ``nl_audio_intent``)::

    "remove all the filler words from this voice note"  →  /audio fillers

All public functions are total: they return result dicts (or None) and
never raise.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from ..media_edit.captions import Word

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# language-scoped filler sets (starter sets — extend per call as needed)
# ---------------------------------------------------------------------------

#: Whole-word filler markers per language. Checked case-insensitively;
#: multi-word entries ("you know") match consecutive words. Ekiti/Ilawe
#: Ekiti inherits the Yoruba base — it is a Yoruba dialect.
FILLERS_BY_LANG: dict[str, tuple[str, ...]] = {
    "en": ("um", "uh", "umm", "uhm", "er", "ah", "like", "you know"),
    "yo": ("ẹẹm", "ehm", "hmm"),
    "yo-ekiti": ("ẹẹm", "ehm", "hmm"),
    "pcm": ("ehn", "eh", "you know"),
    "ha": ("eh", "toh", "to"),
    "ig": ("eh", "ehn"),
}

_LANG_ALIASES: dict[str, str] = {
    "english": "en",
    "yoruba": "yo",
    "ekiti": "yo-ekiti",
    "ilawe": "yo-ekiti",
    "ilawe-ekiti": "yo-ekiti",
    "pidgin": "pcm",
    "naija": "pcm",
    "nigerian-pidgin": "pcm",
    "hausa": "ha",
    "igbo": "ig",
}


def fillers_for(lang: str = "en") -> tuple[str, ...]:
    """Filler set for ``lang`` (aliases resolved). Unknown → English base."""
    key = (lang or "en").strip().lower()
    key = _LANG_ALIASES.get(key, key)
    return FILLERS_BY_LANG.get(key, FILLERS_BY_LANG["en"])


# Unicode-aware word token: any run of letters (handles ẹẹm, ehn, ...),
# with an optional internal apostrophe ("don't").
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def find_fillers(words: Sequence[Word | dict[str, Any]],
                 lang: str = "en",
                 fillers: Sequence[str] | None = None) -> list[tuple[float, float]]:
    """Locate filler regions [(start, end), ...] in word-timed words.

    Whole-word, case-insensitive, Unicode-aware — "umbrella" never matches
    "um", and "ẹẹm" matches in Yoruba/Ekiti. Multi-word fillers match
    consecutive words.
    """
    try:
        ws = [w if isinstance(w, Word) else Word.from_dict(w)
              for w in (words or [])]
        filler_sets = [f.lower().split() for f in (fillers if fillers is not None
                                                   else fillers_for(lang))
                       if f and f.strip()]
        if not filler_sets or not ws:
            return []
        toks = [_TOKEN_RE.findall(w.text.lower()) for w in ws]
        hits: list[tuple[float, float]] = []
        for f in filler_sets:
            n = len(f)
            for i in range(len(ws) - n + 1):
                window = [t for j in range(n) for t in toks[i + j]]
                if window == f:
                    hits.append((ws[i].start, ws[i + n - 1].end))
        return _merge_ranges(hits)
    except Exception:  # noqa: BLE001 - never raises
        return []


# ---------------------------------------------------------------------------
# range math (timeline primitives)
# ---------------------------------------------------------------------------

def _merge_ranges(ranges: list[tuple[float, float]],
                  *, gap: float = 0.05) -> list[tuple[float, float]]:
    """Sort + merge overlapping or near-adjacent ranges."""
    try:
        rs = sorted((float(s), float(e)) for s, e in ranges if float(e) > float(s))
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[float, float]] = []
    for s, e in rs:
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _complement(cuts: list[tuple[float, float]],
                start: float, end: float) -> list[tuple[float, float]]:
    """Ranges to KEEP = [start, end) minus the cut ranges."""
    cuts = _merge_ranges(cuts)
    keeps: list[tuple[float, float]] = []
    cur = start
    for s, e in cuts:
        if s > cur:
            keeps.append((cur, min(s, end)))
        cur = max(cur, e)
        if cur >= end:
            break
    if cur < end:
        keeps.append((cur, end))
    return [(s, e) for s, e in keeps if e - s > 1e-3]


def _pad(ranges: list[tuple[float, float]], pad: float,
         start: float, end: float) -> list[tuple[float, float]]:
    """Expand each range by ``pad`` seconds, clamped to [start, end]."""
    return _merge_ranges([(max(start, s - pad), min(end, e + pad))
                          for s, e in ranges])


# ---------------------------------------------------------------------------
# ffmpeg plumbing (audio only — no video assumptions)
# ---------------------------------------------------------------------------

def _ffmpeg() -> str | None:
    try:
        from ..media_edit.videos import ffmpeg_path
        return ffmpeg_path()
    except Exception:  # noqa: BLE001
        return None


def _audio_duration(path: str | os.PathLike[str]) -> float:
    from ..media_edit.videos import video_probe
    info = video_probe(path)  # ffprobe reads audio files fine
    dur = float(info.get("duration") or 0)
    return dur if dur > 0 else 0.0


def enhance_audio(audio: str | os.PathLike[str], *,
                  out_dir: str | os.PathLike[str] | None = None,
                  engine: str = "auto",
                  profile: str = "voice") -> dict[str, Any]:
    """Denoise ``audio`` (Adobe Enhance pattern) before transcription.

    Strategy chain (native-first):

    1. ``ffmpeg`` — the ``afftdn`` FFT denoiser, then a highpass/lowpass
       safety net (``engine="ffmpeg"`` forces this).
    2. ``devon`` — Devon's OWN DSP chain (:mod:`nomorals.audio.dsp`:
       de-hum → spectral-gate denoise → trim → normalize → compress →
       limit → fade). No ffmpeg needed, pure Python (+numpy when
       present). ``engine="devon"`` forces this.
    3. Honest passthrough — the original path with a note; never a fake
       "enhancement".

    ``engine="auto"`` (default) tries 1 then 2. ``profile`` selects the
    native chain tuning: "voice" (voice notes/speech) or "music".
    """
    p = Path(audio)
    if not p.exists():
        return {"ok": False, "reason": f"no such audio file: {audio}"}
    eng = (engine or "auto").lower()
    out = (Path(out_dir) if out_dir else p.parent) / f"{p.stem}-enhanced.wav"
    ff = _ffmpeg()
    if eng in ("auto", "ffmpeg") and ff is not None:
        from ..media_edit.videos import run_ffmpeg
        for label, af in (("afftdn", "afftdn"),
                          ("highpass+lowpass", "highpass=f=60,lowpass=f=15000")):
            try:
                run_ffmpeg(["-i", str(p), "-af", af, "-ar", "16000", "-ac", "1",
                            str(out)], timeout=300.0)
                if out.exists() and out.stat().st_size > 0:
                    return {"ok": True, "output": str(out), "filter": label,
                            "engine": "ffmpeg"}
            except Exception as exc:  # noqa: BLE001
                _log.debug("enhance filter %s failed: %s", label, exc)
                continue
    if eng in ("auto", "devon"):
        res = _enhance_native(p, out, profile=profile)
        if res.get("ok"):
            return res
        if eng == "devon":
            return res  # forced engine: report the honest failure
    note = ("ffmpeg not installed and native DSP could not decode — "
            "using the original audio" if ff is None else
            "denoise filters unavailable — using the original audio")
    return {"ok": True, "output": str(p), "filter": "none", "engine": "none",
            "note": note}


def _enhance_native(p: Path, out: Path, profile: str = "voice") -> dict[str, Any]:
    """Devon's own enhancement: decode → DSP chain → write. Never raises."""
    try:
        from .dsp import enhance, write_mono_wav
        from .fingerprint import read_mono, AudioReadError
        try:
            samples, sr = read_mono(p, target_sr=22050, max_seconds=1800.0)
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc), "engine": "devon"}
        res = enhance(samples, sr, profile=profile)
        if not res.get("ok"):
            return {"ok": False,
                    "reason": str(res.get("reason", "native DSP failed")),
                    "engine": "devon"}
        written = write_mono_wav(out, res["samples"], sr)
        if not written:
            return {"ok": False, "reason": "could not write enhanced wav",
                    "engine": "devon"}
        return {"ok": True, "output": written, "filter": "devon-dsp",
                "engine": "devon", "chain": res.get("chain", ""),
                "profile": profile,
                "note": "enhanced by Devon's own DSP (no ffmpeg)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"native enhance failed: {exc}",
                "engine": "devon"}


def _splice(audio: str | os.PathLike[str],
            keeps: list[tuple[float, float]], *,
            out_dir: str | os.PathLike[str] | None = None,
            suffix: str = "edited",
            sample_rate: int = 0) -> dict[str, Any]:
    """Assemble kept ranges into one audio file (trim + concat only).

    ``sample_rate`` > 0 forces a re-encode to that rate (mono) so pieces
    from different sources (e.g. TTS output) concat cleanly.
    """
    from ..media_edit.videos import concat, run_ffmpeg, trim
    if not keeps:
        return {"ok": False, "reason": "nothing to keep — every range was cut"}
    ff = _ffmpeg()
    if ff is None:
        return {"ok": False, "reason": "ffmpeg is required to splice audio"}
    parts: list[str] = []
    try:
        for i, (s, e) in enumerate(keeps):
            res = trim(audio, s, e, out_dir=out_dir,
                       suffix=f"{suffix}-p{i}", ext=".wav")
            part = res["output"]
            if sample_rate > 0:
                norm = str(Path(part).with_name(
                    f"{Path(part).stem}-norm.wav"))
                run_ffmpeg(["-i", part, "-ar", str(sample_rate), "-ac", "1",
                            norm], timeout=300.0)
                Path(part).unlink(missing_ok=True)
                part = norm
            parts.append(part)
        if len(parts) == 1:
            return {"ok": True, "output": parts[0], "segments": 1,
                    "mode": "trim-only"}
        res = concat(parts, out_dir=out_dir, suffix=suffix, ext=".wav")
        for pt in parts:
            try:
                Path(pt).unlink(missing_ok=True)
            except OSError:
                pass
        return {"ok": True, "output": res["output"], "segments": len(parts),
                "mode": "trim+concat"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"splice failed: {exc}"}


# ---------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------

@dataclass
class EditableTranscript:
    """A word-timed transcript you edit like text."""

    audio_path: str = ""
    enhanced_path: str = ""
    language: str = "en"
    words: list[Word] = field(default_factory=list)
    filler_hits: list[tuple[float, float]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {"audio_path": self.audio_path,
                "enhanced_path": self.enhanced_path,
                "language": self.language,
                "words": [w.to_dict() for w in self.words],
                "filler_hits": [list(h) for h in self.filler_hits]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditableTranscript":
        return cls(
            audio_path=str(data.get("audio_path", "")),
            enhanced_path=str(data.get("enhanced_path", "")),
            language=str(data.get("language", "en")),
            words=[Word.from_dict(w) for w in (data.get("words") or [])],
            filler_hits=[tuple(h) for h in (data.get("filler_hits") or [])])


def _default_transcriber(audio_path: str,
                         language: str) -> list[Word] | None:
    """faster-whisper with word timestamps (the WhisperX pattern)."""
    try:
        from ..media_edit.captions import transcribe_words
        return transcribe_words(audio_path, language=language or "en")
    except Exception as exc:  # noqa: BLE001
        _log.warning("transcription failed: %s", exc)
        return None


def transcript_edit(audio: str | os.PathLike[str], *,
                    language: str = "en",
                    enhance: bool = True,
                    transcriber: Callable[[str, str], list[Word] | None] | None = None,
                    ) -> EditableTranscript | None:
    """Transcribe ``audio`` → editable word-timed transcript.

    Cleanup-as-preprocessor: the audio is enhanced before transcription.
    Returns None (never raises) when the file is missing or no STT
    backend is available — never fake words.
    """
    try:
        p = Path(audio)
        if not p.exists():
            return None
        src = str(p)
        if enhance:
            enh = enhance_audio(p)
            if enh.get("ok"):
                src = enh["output"]
        fn = transcriber or _default_transcriber
        words = fn(src, language or "en")
        if not words:
            return None
        ws = [w if isinstance(w, Word) else Word.from_dict(w) for w in words]
        return EditableTranscript(
            audio_path=str(p), enhanced_path=src,
            language=language or "en", words=ws,
            filler_hits=find_fillers(ws, language or "en"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# filler removal — "remove all the filler words from this voice note"
# ---------------------------------------------------------------------------

def remove_fillers(audio: str | os.PathLike[str], *,
                   lang: str = "en",
                   fillers: Sequence[str] | None = None,
                   pad: float = 0.2,
                   words: Sequence[Word | dict[str, Any]] | None = None,
                   transcriber: Callable[[str, str], list[Word] | None] | None = None,
                   out_dir: str | os.PathLike[str] | None = None,
                   ) -> dict[str, Any]:
    """Cut filler words ("um", "uh", "you know", "ẹẹm"...) from audio.

    Returns ``{"ok", "output", "cut", "note"}``. No fillers found → the
    original path is returned with a note; never a fake edit.
    """
    try:
        p = Path(audio)
        if not p.exists():
            return {"ok": False, "reason": f"no such audio file: {audio}"}
        ws = [w if isinstance(w, Word) else Word.from_dict(w)
              for w in (words or [])]
        if not ws:
            t = transcript_edit(p, language=lang, transcriber=transcriber)
            if t is None or not t.words:
                return {"ok": False,
                        "reason": "could not transcribe — install faster-whisper "
                                  "(pip install faster-whisper) or pass words="}
            ws = t.words
        dur = _audio_duration(p)
        if dur <= 0:
            return {"ok": False, "reason": "could not determine audio duration"}
        cuts = _pad(find_fillers(ws, lang, fillers), pad, 0.0, dur)
        if not cuts:
            return {"ok": True, "input": str(p), "output": str(p), "cut": 0,
                    "note": "no filler words found — returned the original"}
        res = _splice(p, _complement(cuts, 0.0, dur),
                      out_dir=out_dir, suffix="nofillers")
        if not res.get("ok"):
            return res
        res["cut"] = len(cuts)
        res["note"] = f"cut {len(cuts)} filler region(s)"
        return res
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"filler removal failed: {exc}"}


# ---------------------------------------------------------------------------
# apply_edits — deletion = cut, rewrite = re-synthesize in cloned voice
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """One timeline edit. ``delete`` cuts the range; ``rewrite`` replaces
    [start, end) with ``text`` spoken in the speaker's cloned voice."""

    kind: str = "delete"          # "delete" | "rewrite"
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    @classmethod
    def delete(cls, start: float, end: float) -> "Edit":
        return cls(kind="delete", start=float(start), end=float(end))

    @classmethod
    def rewrite(cls, start: float, end: float, text: str) -> "Edit":
        return cls(kind="rewrite", start=float(start), end=float(end),
                   text=str(text or ""))


def _default_synthesizer(text: str, ref_wav: str, *,
                         lang: str = "en") -> str | None:
    """Re-speak ``text`` in the voice cloned from ``ref_wav``.

    PRIVATE stack only: UniversalTTS with audience="private" (XTTS v2
    zero-shot cloning when installed). Returns the wav path or None —
    never fake audio.
    """
    try:
        from ..voice.tts import UniversalTTS
        tmp = tempfile.mkdtemp(prefix="audioedit-voice-")
        tts = UniversalTTS(backend="auto", voices_dir=tmp, audience="private")
        tts.voices.upload_voice("audioedit-src", ref_wav, language=lang or "en")
        out = tts.speak(text, voice_name="audioedit-src", audience="private")
        path = (out or {}).get("path", "")
        return path if path and Path(path).exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("re-synthesis failed: %s", exc)
        return None


def _normalize_wav(path: str | os.PathLike[str], sample_rate: int,
                   out_dir: str | os.PathLike[str] | None) -> str | None:
    """Re-encode to mono wav at ``sample_rate`` (concat-safe)."""
    try:
        from ..media_edit.videos import run_ffmpeg
        p = Path(path)
        out = (Path(out_dir) if out_dir else p.parent) / f"{p.stem}-norm.wav"
        run_ffmpeg(["-i", str(p), "-ar", str(sample_rate), "-ac", "1",
                    str(out)], timeout=300.0)
        return str(out) if out.exists() else None
    except Exception:  # noqa: BLE001
        return None


def apply_edits(transcript: EditableTranscript | dict[str, Any],
                edits: Sequence[Edit | dict[str, Any]], *,
                voice: str = "source",
                lang: str = "en",
                pad: float = 0.15,
                synthesizer: Callable[..., str | None] | None = None,
                out_dir: str | os.PathLike[str] | None = None,
                ) -> dict[str, Any]:
    """Apply text edits back onto the audio timeline.

    Deletions cut the range (splice). Rewrites replace the range with the
    text re-spoken in the speaker's cloned voice (private XTTS stack).
    ``voice="source"`` clones from the source audio itself.

    Returns ``{"ok", "output", "applied", "note"}`` — or ``{"ok": False,
    "reason"}``. Rewrite edits are refused (not faked) when no TTS
    backend is available.
    """
    try:
        t = (transcript if isinstance(transcript, EditableTranscript)
             else EditableTranscript.from_dict(transcript or {}))
        p = Path(t.audio_path or t.enhanced_path)
        if not p.exists():
            return {"ok": False,
                    "reason": f"no such audio file: {t.audio_path}"}
        dur = _audio_duration(p)
        if dur <= 0:
            return {"ok": False, "reason": "could not determine audio duration"}

        norm_edits: list[Edit] = []
        for e in (edits or []):
            ed = e if isinstance(e, Edit) else Edit(
                kind=str(e.get("kind", "delete")),
                start=float(e.get("start", 0)), end=float(e.get("end", 0)),
                text=str(e.get("text", "")))
            if ed.kind not in ("delete", "rewrite"):
                return {"ok": False,
                        "reason": f"unknown edit kind: {ed.kind!r}"}
            if not (0 <= ed.start < ed.end <= dur + 1e-3):
                return {"ok": False,
                        "reason": f"edit range [{ed.start}, {ed.end}] outside "
                                  f"audio duration {dur:.1f}s"}
            if ed.kind == "rewrite" and not ed.text.strip():
                return {"ok": False, "reason": "rewrite edit needs text"}
            norm_edits.append(ed)
        if not norm_edits:
            return {"ok": False, "reason": "no edits given"}
        norm_edits.sort(key=lambda e: (e.start, e.end))
        # overlapping edits are ambiguous — refuse rather than corrupt
        for a, b in zip(norm_edits, norm_edits[1:]):
            if b.start < a.end - 1e-3:
                return {"ok": False,
                        "reason": "overlapping edits are ambiguous — "
                                  "split them into non-overlapping ranges"}

        synth = synthesizer or _default_synthesizer
        rewrites = [e for e in norm_edits if e.kind == "rewrite"]
        if rewrites and _ffmpeg() is None:
            return {"ok": False, "reason": "ffmpeg is required to splice audio"}

        tmpdir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(
            prefix="audioedit-"))
        tmpdir.mkdir(parents=True, exist_ok=True)
        pieces: list[str] = []          # ordered wav pieces for concat
        synth_paths: list[str] = []

        if rewrites:
            # normalize the source once so TTS pieces concat cleanly
            sr = 24000
            norm_src = _normalize_wav(t.enhanced_path or str(p), sr, tmpdir)
            if norm_src is None:
                return {"ok": False,
                        "reason": "could not normalize source audio"}
            src_for_trims: str = norm_src
            ref_wav = t.enhanced_path or str(p)
        else:
            sr = 0
            src_for_trims = str(p)
            ref_wav = ""

        cursor = 0.0
        from ..media_edit.videos import trim as _trim
        for ed in norm_edits:
            s = max(0.0, ed.start - pad)
            e = min(dur, ed.end + pad)
            if s > cursor + 1e-3:
                keep = _trim(src_for_trims, cursor, s, out_dir=tmpdir,
                             suffix="keep", ext=".wav")["output"]
                pieces.append(keep)
            if ed.kind == "rewrite":
                wav = synth(ed.text, ref_wav, lang=lang or t.language)
                if not wav:
                    return {"ok": False,
                            "reason": "re-synthesis unavailable — install a "
                                      "TTS backend (e.g. pip install TTS for "
                                      "XTTS v2 cloning); rewrite not applied"}
                normed = _normalize_wav(wav, sr, tmpdir)
                if normed is None:
                    return {"ok": False,
                            "reason": "could not normalize synthesized audio"}
                synth_paths.append(normed)
                pieces.append(normed)
            cursor = max(cursor, e)
        if cursor < dur - 1e-3:
            keep = _trim(src_for_trims, cursor, dur, out_dir=tmpdir,
                         suffix="keep", ext=".wav")["output"]
            pieces.append(keep)

        if not pieces:
            return {"ok": False,
                    "reason": "nothing to keep — every range was cut"}
        if len(pieces) == 1:
            out = pieces[0]
        else:
            from ..media_edit.videos import concat as _concat
            res = _concat(pieces, out_dir=out_dir, suffix="edited", ext=".wav")
            out = res["output"]
            for pt in pieces:
                try:
                    Path(pt).unlink(missing_ok=True)
                except OSError:
                    pass
        return {"ok": True, "output": out, "applied": len(norm_edits),
                "deletes": sum(1 for e in norm_edits if e.kind == "delete"),
                "rewrites": len(rewrites),
                "note": f"applied {len(norm_edits)} edit(s)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"apply_edits failed: {exc}"}


# ---------------------------------------------------------------------------
# fx — Devon-owned effect chains
# ---------------------------------------------------------------------------

def apply_fx(audio: str | os.PathLike[str], chain_spec: str, *,
             out_dir: str | os.PathLike[str] | None = None,
             suffix: str = "fx") -> dict[str, Any]:
    """Run one of Devon's own effect chains over ``audio``.

    ``chain_spec``: ``"reverb wet=0.3, eq low=3 high=-2, normalize"`` —
    parsed by :meth:`nomorals.audio.dsp.EffectChain.parse_lenient`.
    Known effects: denoise, dehum, normalize, compress, limit, trim,
    fade, reverb, echo, eq, pitch.

    Returns ``{"ok", "output", "chain", "skipped"}`` — or ``{"ok":
    False, "reason"}``. Never raises, never fake audio.
    """
    try:
        from .dsp import EffectChain, write_mono_wav
        from .fingerprint import read_mono, AudioReadError
        p = Path(audio)
        if not p.exists():
            return {"ok": False, "reason": f"no such audio file: {audio}"}
        chain, unknown = EffectChain.parse_lenient(chain_spec or "")
        if not chain.steps:
            known = ", ".join(sorted(EFFECTS_NAMES))
            return {"ok": False,
                    "reason": "no known effects in that chain — known: "
                              f"{known}" + (f" (unknown: {', '.join(unknown)})"
                                             if unknown else "")}
        try:
            samples, sr = read_mono(p, target_sr=22050, max_seconds=1800.0)
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc)}
        out_samples = chain.run(samples, sr)
        dest = ((Path(out_dir) if out_dir else p.parent)
                / f"{p.stem}-{suffix}.wav")
        written = write_mono_wav(dest, out_samples, sr)
        if not written:
            return {"ok": False, "reason": "could not write fx output"}
        note = f"chain: {chain.describe()}"
        if unknown:
            note += f" (unknown effects skipped: {', '.join(unknown)})"
        if chain.skipped:
            note += f" (failed mid-chain: {', '.join(chain.skipped)})"
        return {"ok": True, "output": written, "chain": chain.describe(),
                "skipped": chain.skipped, "unknown": unknown, "note": note}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"fx failed: {exc}"}


EFFECTS_NAMES = (
    "denoise", "dehum", "normalize", "compress", "limit", "trim",
    "fade", "reverb", "echo", "eq", "pitch",
)


def list_fx() -> str:
    """Human-readable effect catalogue. Never raises."""
    try:
        from .dsp import EFFECTS
        lines = ["Devon-owned effects (all native, no plugins):"]
        for name in EFFECTS_NAMES:
            _, desc = EFFECTS[name]
            lines.append(f"  {name} — {desc}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return "effects unavailable"


# ---------------------------------------------------------------------------
# natural-language intent ("remove all the filler words from this voice note")
# ---------------------------------------------------------------------------

_FILLER_NL = re.compile(
    r"\b(?:remove|cut|delete|strip|take out)\b.{0,40}?\b"
    r"(?:filler words?|um+?s?(?:\s+and\s+uh+?s?)?|uh+?s?|um+s)\b",
    re.IGNORECASE)
_FILE_RE = re.compile(r"['\"]?([\w\-./\\]+\.(?:m4a|mp3|wav|ogg|opus|flac|aac|wma))['\"]?",
                      re.IGNORECASE)


def nl_audio_intent(text: str) -> dict[str, Any] | None:
    """Detect transcript-editing intent in natural language.

    "remove all the filler words from this voice note" →
    ``{"command": "fillers", "file": "...", "lang": "en"}``.
    Returns None when the text carries no audio-edit intent.
    """
    try:
        t = (text or "").strip()
        if not t or not _FILLER_NL.search(t):
            return None
        m = _FILE_RE.search(t)
        lang = "en"
        for alias, key in _LANG_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", t, re.IGNORECASE):
                lang = key
                break
        return {"command": "fillers", "file": m.group(1) if m else "",
                "lang": lang}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

_USAGE = (
    "🎙️ /audio — transcript-as-timeline audio editing (owner only)\n"
    "  /audio edit <file> [lang] — transcribe with word timings\n"
    "  /audio fillers <file> [lang] — cut filler words (um, uh, you know, ẹẹm…)\n"
    "  /audio enhance <file> [voice|music|light] — denoise (ffmpeg → Devon's own DSP)\n"
    "  /audio fx <file> <effects…> — Devon-owned effect chain, e.g.\n"
    "      /audio fx note.wav reverb wet=0.3, eq low=3 high=-2, normalize\n"
    "  /audio fx-list — the native effect catalogue\n"
    "  /audio analyze <file> — what Devon hears (tempo, key, loudness…)\n"
    "  /audio fingerprint <file> [title] — index into local recognition memory\n"
    "  /audio match <file> — recognize against Devon's local library\n"
    "Natural: \"remove all the filler words from this voice note\""
)


def _fmt_ts(s: float) -> str:
    m, sec = divmod(max(0.0, float(s)), 60)
    return f"{int(m):02d}:{sec:05.2f}"


def control_audio(tail: str, *, context: Any = None,
                  chat: Any = None, **kwargs: Any) -> str:
    """Chat entry: /audio edit|fillers|enhance. Owner-only at dispatch."""
    try:
        parts = (tail or "").split()
        if not parts:
            return _USAGE
        verb = parts[0].lower()
        if verb == "edit" and len(parts) >= 2:
            t = transcript_edit(parts[1],
                                language=parts[2] if len(parts) > 2 else "en")
            if t is None:
                return ("🎙️ couldn't transcribe that — needs faster-whisper "
                        "(pip install faster-whisper) and an audio file.")
            lines = [f"🎙️ transcript — {len(t.words)} words "
                     f"({t.language})"]
            for w in t.words[:40]:
                lines.append(f"  {_fmt_ts(w.start)}  {w.text}")
            if len(t.words) > 40:
                lines.append(f"  … +{len(t.words) - 40} more words")
            if t.filler_hits:
                lines.append(f"🧹 {len(t.filler_hits)} filler region(s) — "
                             f"run /audio fillers {parts[1]}")
            return "\n".join(lines)
        if verb == "fillers" and len(parts) >= 2:
            res = remove_fillers(parts[1],
                                 lang=parts[2] if len(parts) > 2 else "en")
            if not res.get("ok"):
                return f"🎙️ {res.get('reason', 'failed')}"
            return (f"🧹 {res.get('note', 'done')}\n"
                    f"📁 {res.get('output')}")
        if verb == "enhance" and len(parts) >= 2:
            prof = parts[2].lower() if len(parts) > 2 else "voice"
            res = enhance_audio(parts[1], profile=prof)
            if not res.get("ok"):
                return f"🎙️ {res.get('reason', 'failed')}"
            note = res.get("note", f"filter: {res.get('filter')}")
            eng = res.get("engine", "?")
            return (f"✨ enhanced [{eng}] via {res.get('filter')}\n"
                    f"📁 {res.get('output')}\n{note}")
        if verb == "fx-list":
            return "🎛️ " + list_fx().replace("\n", "\n")
        if verb == "fx" and len(parts) >= 3:
            spec = " ".join(parts[2:])
            res = apply_fx(parts[1], spec)
            if not res.get("ok"):
                return f"🎛️ {res.get('reason', 'failed')}"
            return f"🎛️ {res.get('note')}\n📁 {res.get('output')}"
        if verb == "analyze" and len(parts) >= 2:
            from .fingerprint import describe_audio
            return f"👂 {describe_audio(parts[1])}"
        if verb == "fingerprint" and len(parts) >= 2:
            from .fingerprint import default_fingerprint_db
            title = " ".join(parts[2:]) if len(parts) > 2 else ""
            res = default_fingerprint_db().add_track(
                parts[1], title=title or Path(parts[1]).stem,
                artist="local", source="library")
            if not res.get("ok"):
                return f"🎛️ {res.get('reason', 'failed')}"
            return (f"🎛️ fingerprinted '{res.get('title')}' — "
                    f"{res.get('hashes')} hashes in local memory")
        if verb == "match" and len(parts) >= 2:
            from .fingerprint import match_local_db
            res = match_local_db(parts[1])
            if not res.get("ok"):
                return f"🎛️ {res.get('reason')}"
            return (f"🎵 I know this one — {res.get('artist')} — "
                    f"{res.get('title')} (local match, "
                    f"confidence {res.get('score')})")
        return _USAGE
    except Exception as exc:  # noqa: BLE001
        return f"🎙️ audio edit failed: {exc}"
