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
for English, Yoruba, Ekiti/Ilawe Ekiti, Nigerian Pidgin, Hausa, Igbo,
French, Spanish and German — the user can extend them per call. Ekiti
inherits the Yoruba base (it is a Yoruba dialect); matching is
Unicode-aware so "ẹẹm" works.

Silence tightening (``remove_silences``) cuts dead air in bulk while
keeping a breath of natural pause — Descript's second-biggest win.

Edit kinds: delete (cut), rewrite (re-synthesize in the cloned voice),
silence (mute, timing preserved — the strikethrough), move (cut a range
and paste it elsewhere). ``preview_edits`` dry-runs any edit list.

Natural language (routed via ``nl_audio_intent``)::

    "remove all the filler words from this voice note"  →  /audio fillers
    "tighten the silences"                              →  /audio silences
    "add reverb to this"                                →  /audio fx

All public functions are total: they return result dicts (or None) and
never raise.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
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
    "en": ("um", "uh", "umm", "uhm", "er", "ah", "like", "you know",
           "i mean", "sort of", "kind of"),
    "yo": ("ẹẹm", "ehm", "hmm"),
    "yo-ekiti": ("ẹẹm", "ehm", "hmm"),
    "pcm": ("ehn", "eh", "you know", "sha"),
    "ha": ("eh", "toh", "to"),
    "ig": ("eh", "ehn"),
    "fr": ("euh", "ben", "hein", "en fait", "tu vois"),
    "es": ("este", "bueno", "eh", "o sea", "pues"),
    "de": ("äh", "ähm", "also", "halt"),
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
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "german": "de",
    "deutsch": "de",
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

    def correct_text(self, old: str, new: str) -> int:
        """Correct-text mode (the Descript pattern): fix transcript text
        WITHOUT touching the media — the word timings stay, only the
        spelling changes. Returns the number of words corrected."""
        try:
            n = 0
            for w in self.words:
                if w.text == old:
                    w.text = new
                    n += 1
            return n
        except Exception:  # noqa: BLE001
            return 0

    def words_in(self, start: float, end: float) -> list[Word]:
        """Words overlapping [start, end) — for edit previews."""
        try:
            return [w for w in self.words
                    if w.end > start and w.start < end]
        except Exception:  # noqa: BLE001
            return []

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


def find_silences(audio: str | os.PathLike[str], *,
                  threshold_db: float = -40.0,
                  min_silence_s: float = 0.7,
                  max_keep_s: float = 0.35) -> list[tuple[float, float]]:
    """Locate long silences [(start, end), ...] in ``audio``.

    Energy-based VAD over 30 ms windows: windows below ``threshold_db``
    are silence; runs longer than ``min_silence_s`` are reported with
    ``max_keep_s`` of natural pause preserved at each edge (Descript
    keeps the pacing human). Never raises — [] on failure.
    """
    try:
        from .fingerprint import read_mono
        samples, sr = read_mono(audio, target_sr=16000, max_seconds=3600.0)
        n = len(samples)
        if n < sr:
            return []
        thr = 10.0 ** (threshold_db / 20.0)
        win = max(1, int(0.03 * sr))
        silent = [False] * ((n + win - 1) // win)
        for w in range(len(silent)):
            seg = samples[w * win:(w + 1) * win]
            rms = (sum(s * s for s in seg) / max(1, len(seg))) ** 0.5
            silent[w] = rms < thr
        regions: list[tuple[float, float]] = []
        i = 0
        while i < len(silent):
            if silent[i]:
                j = i
                while j < len(silent) and silent[j]:
                    j += 1
                dur = (j - i) * win / sr
                if dur >= min_silence_s:
                    s = i * win / sr + max_keep_s / 2
                    e = j * win / sr - max_keep_s / 2
                    if e > s:
                        regions.append((s, e))
                i = j
            else:
                i += 1
        return regions
    except Exception:  # noqa: BLE001
        return []


def remove_silences(audio: str | os.PathLike[str], *,
                    threshold_db: float = -40.0,
                    min_silence_s: float = 0.7,
                    max_keep_s: float = 0.35,
                    out_dir: str | os.PathLike[str] | None = None,
                    ) -> dict[str, Any]:
    """Tighten long silences (Descript's bulk gap removal): dead air
    longer than ``min_silence_s`` is cut, keeping ``max_keep_s`` of
    natural pause so pacing stays human.

    Returns ``{"ok", "output", "cut", "saved_s", "note"}``. Never raises.
    """
    try:
        p = Path(audio)
        if not p.exists():
            return {"ok": False, "reason": f"no such audio file: {audio}"}
        dur = _audio_duration(p)
        if dur <= 0:
            return {"ok": False, "reason": "could not determine audio duration"}
        cuts = _merge_ranges(find_silences(
            p, threshold_db=threshold_db, min_silence_s=min_silence_s,
            max_keep_s=max_keep_s))
        if not cuts:
            return {"ok": True, "input": str(p), "output": str(p), "cut": 0,
                    "saved_s": 0.0,
                    "note": "no long silences found — returned the original"}
        saved = sum(e - s for s, e in cuts)
        res = _splice(p, _complement(cuts, 0.0, dur),
                      out_dir=out_dir, suffix="tight")
        if not res.get("ok"):
            return res
        res["cut"] = len(cuts)
        res["saved_s"] = round(saved, 1)
        res["note"] = (f"tightened {len(cuts)} silence region(s), "
                       f"saved {saved:.1f}s")
        return res
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"silence removal failed: {exc}"}


# ---------------------------------------------------------------------------
# apply_edits — deletion = cut, rewrite = re-synthesize in cloned voice
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """One timeline edit.

    - ``delete`` cuts the range.
    - ``rewrite`` replaces [start, end) with ``text`` spoken in the
      speaker's cloned voice.
    - ``silence`` mutes [start, end) (Descript's strikethrough: the
      timing stays, the sound goes).
    - ``move`` cuts [start, end) and re-inserts it at ``dest``
      (Descript's "move a paragraph, the audio moves with it").
    """

    kind: str = "delete"          # "delete" | "rewrite" | "silence" | "move"
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    dest: float = 0.0             # move only: insertion point (original timeline)

    @classmethod
    def delete(cls, start: float, end: float) -> "Edit":
        return cls(kind="delete", start=float(start), end=float(end))

    @classmethod
    def rewrite(cls, start: float, end: float, text: str) -> "Edit":
        return cls(kind="rewrite", start=float(start), end=float(end),
                   text=str(text or ""))

    @classmethod
    def silence(cls, start: float, end: float) -> "Edit":
        """Mute the range — timing preserved, sound removed."""
        return cls(kind="silence", start=float(start), end=float(end))

    @classmethod
    def move(cls, start: float, end: float, dest: float) -> "Edit":
        """Cut [start, end) and paste it at ``dest`` (original-timeline
        seconds)."""
        return cls(kind="move", start=float(start), end=float(end),
                   dest=float(dest))


def _audio_sample_rate(path: str | os.PathLike[str]) -> int:
    """Source sample rate via probe. 0 when unknown. Never raises."""
    try:
        from ..media_edit.videos import video_probe
        info = video_probe(path) or {}
        for st in info.get("streams", []) or []:
            if st.get("codec_type") == "audio" and st.get("sample_rate"):
                return int(st["sample_rate"])
        return int(info.get("sample_rate") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _silence_wav(duration_s: float, sample_rate: int,
                 out_dir: str | os.PathLike[str]) -> str | None:
    """Render ``duration_s`` of digital silence as a mono wav. None on
    failure (never raises)."""
    try:
        from ..media_edit.videos import run_ffmpeg
        sr = int(sample_rate) or 24000
        out = Path(out_dir) / f"silence-{uuid.uuid4().hex[:8]}.wav"
        run_ffmpeg(["-f", "lavfi", "-i",
                    f"anullsrc=r={sr}:cl=mono",
                    "-t", f"{max(0.01, duration_s):.3f}",
                    "-ar", str(sr), "-ac", "1", str(out)],
                   timeout=120.0)
        return str(out) if out.exists() else None
    except Exception:  # noqa: BLE001
        return None


def preview_edits(transcript: EditableTranscript | dict[str, Any],
                  edits: Sequence[Edit | dict[str, Any]],
                  ) -> dict[str, Any]:
    """Dry-run: describe what ``edits`` would do WITHOUT rendering.

    Returns ``{"ok", "preview": [...], "cut_s", "note"}`` where each
    preview row shows the kind, range, and the words caught in the
    range — Descript's "read first" pattern. Never raises.
    """
    try:
        t = (transcript if isinstance(transcript, EditableTranscript)
             else EditableTranscript.from_dict(transcript or {}))
        rows: list[dict[str, Any]] = []
        cut_s = 0.0
        for e in (edits or []):
            ed = e if isinstance(e, Edit) else Edit(
                kind=str(e.get("kind", "delete")),
                start=float(e.get("start", 0)), end=float(e.get("end", 0)),
                text=str(e.get("text", "")),
                dest=float(e.get("dest", 0)))
            words = " ".join(w.text for w in
                             t.words_in(ed.start, ed.end))[:120]
            note = {"kind": ed.kind, "start": round(ed.start, 2),
                    "end": round(ed.end, 2), "words": words}
            if ed.kind == "delete":
                cut_s += ed.end - ed.start
                note["effect"] = f"cut {ed.end - ed.start:.1f}s"
            elif ed.kind == "rewrite":
                note["effect"] = f"re-speak as: {ed.text[:80]!r}"
                note["new_text"] = ed.text
            elif ed.kind == "silence":
                note["effect"] = f"mute {ed.end - ed.start:.1f}s (timing kept)"
            elif ed.kind == "move":
                note["dest"] = round(ed.dest, 2)
                note["effect"] = (f"move {ed.end - ed.start:.1f}s → "
                                  f"{ed.dest:.1f}s")
            else:
                note["effect"] = f"unknown kind {ed.kind!r}"
            rows.append(note)
        return {"ok": True, "preview": rows,
                "cut_s": round(cut_s, 1), "edits": len(rows),
                "note": f"{len(rows)} edit(s) previewed, "
                        f"{cut_s:.1f}s would be cut"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"preview failed: {exc}"}


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
                text=str(e.get("text", "")),
                dest=float(e.get("dest", 0)))
            if ed.kind not in ("delete", "rewrite", "silence", "move"):
                return {"ok": False,
                        "reason": f"unknown edit kind: {ed.kind!r}"}
            if not (0 <= ed.start < ed.end <= dur + 1e-3):
                return {"ok": False,
                        "reason": f"edit range [{ed.start}, {ed.end}] outside "
                                  f"audio duration {dur:.1f}s"}
            if ed.kind == "rewrite" and not ed.text.strip():
                return {"ok": False, "reason": "rewrite edit needs text"}
            if ed.kind == "move":
                if not (0 <= ed.dest <= dur):
                    return {"ok": False,
                            "reason": f"move destination {ed.dest} outside "
                                      f"audio duration {dur:.1f}s"}
                if ed.start - 1e-3 <= ed.dest <= ed.end + 1e-3:
                    return {"ok": False,
                            "reason": "move destination lands inside its own "
                                      "range — pick a point outside it"}
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
        # move destinations must not land inside another edit's range
        for m in norm_edits:
            if m.kind != "move":
                continue
            for o in norm_edits:
                if o is not m and o.start - 1e-3 <= m.dest <= o.end + 1e-3:
                    return {"ok": False,
                            "reason": "move destination lands inside another "
                                      "edit's range — ambiguous"}

        synth = synthesizer or _default_synthesizer
        rewrites = [e for e in norm_edits if e.kind == "rewrite"]
        moves = [e for e in norm_edits if e.kind == "move"]
        silences = [e for e in norm_edits if e.kind == "silence"]
        if (rewrites or silences or moves) and _ffmpeg() is None:
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

        # extract move pieces up front (original-timeline trims)
        move_pieces: dict[int, str] = {}
        from ..media_edit.videos import trim as _trim
        for idx, m in enumerate(moves):
            piece = _trim(src_for_trims, m.start, m.end, out_dir=tmpdir,
                          suffix=f"move{idx}", ext=".wav")["output"]
            if sr:
                normed = _normalize_wav(piece, sr, tmpdir)
                if normed:
                    piece = normed
            move_pieces[idx] = piece
        # insertions sorted by destination (original-timeline seconds)
        insertions = sorted(
            ((m.dest, move_pieces[idx]) for idx, m in enumerate(moves)),
            key=lambda x: x[0])
        ins_i = 0

        def _flush_insertions(upto: float, cursor: float,
                              ) -> tuple[list[str], float]:
            """Emit moved pieces whose dest is in [cursor, upto)."""
            nonlocal ins_i
            out_p: list[str] = []
            while ins_i < len(insertions) and \
                    cursor - 1e-3 <= insertions[ins_i][0] < upto - 1e-3:
                dest, piece = insertions[ins_i]
                if dest > cursor + 1e-3:
                    keep = _trim(src_for_trims, cursor, dest, out_dir=tmpdir,
                                 suffix="keep", ext=".wav")["output"]
                    out_p.append(keep)
                    cursor = dest
                out_p.append(piece)
                ins_i += 1
            return out_p, cursor

        cursor = 0.0
        for ed in norm_edits:
            s = max(0.0, ed.start - pad)
            e = min(dur, ed.end + pad)
            if s > cursor + 1e-3:
                flushed, cursor = _flush_insertions(s, cursor)
                pieces.extend(flushed)
                if s > cursor + 1e-3:
                    keep = _trim(src_for_trims, cursor, s, out_dir=tmpdir,
                                 suffix="keep", ext=".wav")["output"]
                    pieces.append(keep)
            else:
                flushed, cursor = _flush_insertions(s, cursor)
                pieces.extend(flushed)
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
            elif ed.kind == "silence":
                mute = _silence_wav(e - s, sr or _audio_sample_rate(p) or 24000,
                                    tmpdir)
                if mute is None:
                    return {"ok": False,
                            "reason": "could not render silence for mute edit"}
                if sr:
                    normed = _normalize_wav(mute, sr, tmpdir)
                    if normed:
                        mute = normed
                synth_paths.append(mute)
                pieces.append(mute)
            # delete/move: nothing emitted for the range itself
            cursor = max(cursor, e)
        if cursor < dur - 1e-3:
            flushed, cursor = _flush_insertions(dur, cursor)
            pieces.extend(flushed)
            if cursor < dur - 1e-3:
                keep = _trim(src_for_trims, cursor, dur, out_dir=tmpdir,
                             suffix="keep", ext=".wav")["output"]
                pieces.append(keep)
        else:
            flushed, _ = _flush_insertions(dur, cursor)
            pieces.extend(flushed)

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
                "silences": sum(1 for e in norm_edits if e.kind == "silence"),
                "moves": len(moves),
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
    parsed by :meth:`nomorals.audio.dsp.EffectChain.parse_lenient` — or a
    style preset: ``"preset=lofi"``. Known effects: denoise, dehum,
    highpass, lowpass, telephone, bitcrush, gain, normalize, autolevel,
    compress, limit, trim, fade, reverb, echo, delay, chorus, phaser,
    tremolo, vibrato, distortion, eq, pitch.

    Returns ``{"ok", "output", "chain", "skipped"}`` — or ``{"ok":
    False, "reason"}``. Never raises, never fake audio.
    """
    try:
        from .dsp import EffectChain, write_mono_wav, CHAIN_PRESETS
        from .fingerprint import read_mono, AudioReadError
        p = Path(audio)
        if not p.exists():
            return {"ok": False, "reason": f"no such audio file: {audio}"}
        spec = (chain_spec or "").strip()
        preset_m = re.match(r"(?i)^preset\s*=\s*([a-z0-9_-]+)$", spec)
        if preset_m:
            try:
                chain = EffectChain.from_preset(preset_m.group(1))
                unknown: list[str] = []
            except ValueError as exc:
                return {"ok": False, "reason": str(exc)}
        else:
            chain, unknown = EffectChain.parse_lenient(spec)
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
    "denoise", "dehum", "highpass", "lowpass", "telephone", "bitcrush",
    "gain", "normalize", "autolevel", "compress", "limit", "trim",
    "fade", "reverb", "echo", "delay", "chorus", "phaser", "tremolo",
    "vibrato", "distortion", "eq", "pitch",
)


def duck_audio(voice: str | os.PathLike[str], bed: str | os.PathLike[str], *,
               duck_db: float = -9.0,
               out_dir: str | os.PathLike[str] | None = None,
               ) -> dict[str, Any]:
    """Podcast mix: duck the music ``bed`` under ``voice`` and mix them.

    The Auphonic multitrack pattern — the bed drops whenever the voice
    is present, then the voice sits on top. Returns ``{"ok", "output",
    "note"}``. Never raises.
    """
    try:
        from .dsp import duck_under, mix_under, write_mono_wav
        from .fingerprint import read_mono, AudioReadError
        pv, pb = Path(voice), Path(bed)
        if not pv.exists():
            return {"ok": False, "reason": f"no such voice file: {voice}"}
        if not pb.exists():
            return {"ok": False, "reason": f"no such bed file: {bed}"}
        try:
            vs, sr = read_mono(pv, target_sr=24000, max_seconds=3600.0)
            bs, _ = read_mono(pb, target_sr=24000, max_seconds=3600.0)
        except AudioReadError as exc:
            return {"ok": False, "reason": str(exc)}
        ducked_bed = duck_under(bs, vs, sr, duck_db=duck_db)
        mixed = mix_under(ducked_bed, vs, gain=1.0)
        dest = ((Path(out_dir) if out_dir else pv.parent)
                / f"{pv.stem}-ducked.wav")
        written = write_mono_wav(dest, mixed, sr)
        if not written:
            return {"ok": False, "reason": "could not write ducked mix"}
        return {"ok": True, "output": written,
                "note": f"bed ducked {duck_db}dB under voice, mixed"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"duck failed: {exc}"}


def list_fx() -> str:
    """Human-readable effect catalogue, grouped by family. Never raises."""
    try:
        from .dsp import EFFECTS, CHAIN_PRESETS
        families = [
            ("🧹 cleanup", ["denoise", "dehum", "highpass", "lowpass",
                            "trim", "fade", "autolevel"]),
            ("🎚️ dynamics", ["normalize", "gain", "compress", "limit"]),
            ("🌌 space", ["reverb", "echo", "delay", "chorus", "phaser"]),
            ("🎛️ tone", ["eq", "pitch", "telephone", "bitcrush",
                         "distortion", "tremolo", "vibrato"]),
        ]
        lines = ["🎛️ Devon-owned effects — all native, no plugins, no ffmpeg:"]
        for title, names in families:
            lines.append(f"\n{title}")
            for name in names:
                if name in EFFECTS:
                    _, desc = EFFECTS[name]
                    lines.append(f"  {name:10s} — {desc}")
        lines.append("\n🎨 style presets (one word, whole chain):")
        for name in sorted(CHAIN_PRESETS):
            lines.append(f"  {name}")
        lines.append("\nusage: /audio fx note.wav reverb wet=0.3, eq low=3 high=-2")
        lines.append("       /audio fx note.wav preset=lofi")
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
_SILENCE_NL = re.compile(
    r"\b(?:remove|cut|delete|strip|tighten|trim)\b.{0,40}?\b"
    r"(?:silences?|dead air|long pauses?|gaps?)\b",
    re.IGNORECASE)
_ENHANCE_NL = re.compile(
    r"\b(?:enhance|clean ?up|denoise|fix|improve)\b.{0,30}?\b"
    r"(?:audio|sound|voice ?note|recording)\b",
    re.IGNORECASE)
_FX_NL = re.compile(
    r"\b(?:add|apply|put)\b.{0,30}?\b"
    r"(reverb|echo|delay|chorus|distortion|eq|equali[sz]er|pitch|lo-?fi|telephone)\b",
    re.IGNORECASE)
_ANALYZE_NL = re.compile(
    r"\b(?:what(?:'s| is)|analy[sz]e|describe|identify|recognize|shazam)\b"
    r".{0,40}?\b(?:in this|this|that)\b.{0,20}?\b"
    r"(?:audio|song|track|recording|voice ?note)\b",
    re.IGNORECASE)
_FILE_RE = re.compile(r"['\"]?([\w\-./\\]+\.(?:m4a|mp3|wav|ogg|opus|flac|aac|wma))['\"]?",
                      re.IGNORECASE)


def _detect_lang(text: str) -> str:
    for alias, key in _LANG_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
            return key
    return "en"


def nl_audio_intent(text: str) -> dict[str, Any] | None:
    """Detect audio-editing intent in natural language.

    "remove all the filler words from this voice note" →
    ``{"command": "fillers", "file": "...", "lang": "en"}``.
    Also routes: silences → ``silences``, enhance/cleanup → ``enhance``,
    add reverb/echo/… → ``fx``, "what's in this audio" → ``analyze``.
    Returns None when the text carries no audio intent.
    """
    try:
        t = (text or "").strip()
        if not t:
            return None
        m = _FILE_RE.search(t)
        file = m.group(1) if m else ""
        lang = _detect_lang(t)
        if _FILLER_NL.search(t):
            return {"command": "fillers", "file": file, "lang": lang}
        if _SILENCE_NL.search(t):
            return {"command": "silences", "file": file}
        if _ENHANCE_NL.search(t):
            return {"command": "enhance", "file": file}
        fx = _FX_NL.search(t)
        if fx:
            return {"command": "fx", "file": file,
                    "effect": fx.group(1).lower()}
        if _ANALYZE_NL.search(t):
            return {"command": "analyze", "file": file}
        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

_USAGE = (
    "🎙️ /audio — transcript-as-timeline audio editing (owner only)\n"
    "  /audio edit <file> [lang] — transcribe with word timings\n"
    "  /audio fillers <file> [lang] — cut filler words (um, uh, you know, ẹẹm…)\n"
    "  /audio silences <file> — tighten long silences, keep pacing human\n"
    "  /audio enhance <file> [voice|music|light|podcast|audiobook] — denoise\n"
    "  /audio fx <file> <effects…> — Devon-owned chain, e.g.\n"
    "      /audio fx note.wav reverb wet=0.3, eq low=3 high=-2, normalize\n"
    "      /audio fx note.wav preset=lofi\n"
    "  /audio fx-list — the native effect catalogue (+ style presets)\n"
    "  /audio presets — style preset chains at a glance\n"
    "  /audio duck <voice> <bed> — duck music under voice, podcast mix\n"
    "  /audio analyze <file> — what Devon hears (tempo, key, chord, tuning…)\n"
    "  /audio fingerprint <file> [title] — index into local recognition memory\n"
    "  /audio match <file> — recognize against Devon's local library\n"
    "Natural: \"remove all the filler words from this voice note\" · "
    "\"tighten the silences\" · \"add reverb to this\""
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
        if verb == "presets":
            from .dsp import list_presets
            return list_presets()
        if verb == "silences" and len(parts) >= 2:
            res = remove_silences(parts[1])
            if not res.get("ok"):
                return f"🎙️ {res.get('reason', 'failed')}"
            return (f"✂️ {res.get('note', 'done')}\n"
                    f"📁 {res.get('output')}")
        if verb == "duck" and len(parts) >= 3:
            res = duck_audio(parts[1], parts[2])
            if not res.get("ok"):
                return f"🎙️ {res.get('reason', 'failed')}"
            return (f"🎚️ {res.get('note')}\n📁 {res.get('output')}")
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
