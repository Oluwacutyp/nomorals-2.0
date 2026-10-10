"""Unified voice pipeline — capture → cleanup → transcribe → edit → multi-speaker TTS → master.

The architecture under all of Phase 24. One composed pipeline with a
pluggable stage registry and an engine registry that picks backends by
``resource_profile`` (VoiceStudio pattern):

- termux:      ONNX / whisper-cpp STT (CPU), piper/chatterbox TTS
- laptop:      faster-whisper STT (CUDA when available), xtts (private) / chatterbox TTS
- workstation: everything

Keyterm prompting: a per-user vocabulary (names, Ekiti dialect words,
jargon) is injected into STT as an ``initial_prompt`` so transcription
is accurate on the user's actual speech.

Audience routing (#30) is enforced structurally at the synthesize stage:
private surfaces → XTTS allowed; public/community surfaces → XTTS
removed from the candidate list. Never the wrong way round.

Every function never raises; failures are honest dicts, never fake audio.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_log = logging.getLogger("nomorals.audio.pipeline")

# ---------------------------------------------------------------------------
# stage registry
# ---------------------------------------------------------------------------

#: Canonical stage order. Stages are individually skippable via config.
STAGES: tuple[str, ...] = (
    "denoise",      # cleanup-as-preprocessor (#103 enhance_audio)
    "transcribe",   # word-timed STT (#25, WhisperX pattern)
    "diarize",      # speaker assignment (pluggable; honest single-speaker default)
    "edit",         # filler removal + text edits (#103 apply_edits)
    "synthesize",   # multi-speaker TTS (#30 UniversalTTS, audience-routed)
    "master",       # LUFS normalize (#106 audiobook mastering)
)

StageFn = Callable[..., dict[str, Any]]


# ---------------------------------------------------------------------------
# profile detection (mirrors synth_backend / agent_loop conventions)
# ---------------------------------------------------------------------------

def detect_profile(context: Any = None) -> str:
    """Environment profile kind: termux | laptop | workstation | pc."""
    try:
        settings = getattr(context, "settings", None)
        prof = (getattr(settings, "profile", "") or "").strip().lower()
        if prof:
            return prof
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        if os.environ.get("PREFIX", "").startswith("/data/data/com.termux"):
            return "termux"
    except Exception:  # noqa: BLE001
        pass
    try:
        if os.environ.get("NM_PROFILE", "").strip().lower() in (
                "termux", "laptop", "workstation", "pc"):
            return os.environ["NM_PROFILE"].strip().lower()
    except Exception:  # noqa: BLE001
        pass
    return "pc"


# ---------------------------------------------------------------------------
# engine registry — VoiceStudio pattern, profile-gated backends
# ---------------------------------------------------------------------------

#: XTTS v2 is non-commercial — private (owner's own) use only.
_NONCOMMERCIAL = frozenset({"xtts"})

# Preference order per (profile, task). First available backend wins.
_STT_ORDER: dict[str, tuple[str, ...]] = {
    "termux": ("parakeet-onnx", "whisper-cpp", "faster-whisper"),
    "laptop": ("faster-whisper", "parakeet-onnx", "whisper-cpp"),
    "workstation": ("faster-whisper", "parakeet-onnx", "whisper-cpp"),
    "pc": ("faster-whisper", "parakeet-onnx", "whisper-cpp"),
}

_TTS_ORDER: dict[str, tuple[str, ...]] = {
    "termux": ("chatterbox", "piper", "kokoro"),
    "laptop": ("xtts", "chatterbox", "kokoro", "piper"),
    "workstation": ("xtts", "chatterbox", "kokoro", "piper", "f5tts"),
    "pc": ("chatterbox", "kokoro", "piper"),
}


class EngineRegistry:
    """Picks the best STT/TTS backend for the current profile and audience.

    The audience rule is structural: for ``audience="public"`` the
    non-commercial backends are removed from the candidate list before
    any selection happens — they cannot be chosen, not merely discouraged.
    """

    def __init__(self, profile: str = "", context: Any = None) -> None:
        self.profile = (profile or detect_profile(context) or "pc").lower()
        if self.profile not in _STT_ORDER:
            self.profile = "pc"

    # -- STT -----------------------------------------------------------------
    def stt_candidates(self) -> list[str]:
        return list(_STT_ORDER[self.profile])

    def pick_stt(self, available: Sequence[str] | None = None) -> str | None:
        """Best STT backend for this profile, or None when nothing is available."""
        try:
            avail = {a.lower() for a in (available or [])}
            for cand in _STT_ORDER[self.profile]:
                if cand in avail:
                    return cand
            return None
        except Exception:  # noqa: BLE001
            return None

    # -- TTS -----------------------------------------------------------------
    def tts_candidates(self, audience: str = "private") -> list[str]:
        """TTS backends for this profile, audience-filtered.

        audience="public" structurally removes non-commercial backends
        (XTTS). Never the wrong way round — this is enforced here, not
        merely advised.
        """
        try:
            cands = list(_TTS_ORDER[self.profile])
            if (audience or "private").lower() == "public":
                cands = [c for c in cands if c not in _NONCOMMERCIAL]
            return cands
        except Exception:  # noqa: BLE001
            return []

    def pick_tts(self, audience: str = "private",
                 available: Sequence[str] | None = None) -> str | None:
        try:
            avail = {a.lower() for a in (available or [])}
            for cand in self.tts_candidates(audience):
                if cand in avail:
                    return cand
            return None
        except Exception:  # noqa: BLE001
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "stt_order": self.stt_candidates(),
            "tts_private": self.tts_candidates("private"),
            "tts_public": self.tts_candidates("public"),
            "noncommercial_excluded_public": sorted(_NONCOMMERCIAL),
        }


# ---------------------------------------------------------------------------
# keyterm prompting — per-user vocabulary injected into STT
# ---------------------------------------------------------------------------

_KEYTERM_KINDS = ("name", "dialect", "jargon", "place")


def _default_keyterm_db() -> str:
    d = Path.home() / ".nomorals" / "audio"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return str(d / "keyterms.db")


class KeytermStore:
    """Per-user vocabulary for STT prompting: names, Ekiti dialect words, jargon.

    The terms become faster-whisper's ``initial_prompt`` so transcription is
    accurate on the user's actual speech. Never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        try:
            self._path = db_path or _default_keyterm_db()
            self._db = sqlite3.connect(self._path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS keyterms ("
                "term TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'jargon',"
                "lang TEXT NOT NULL DEFAULT '', added REAL NOT NULL)")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.warning("keyterm store unavailable")
            self._db = None

    # -- CRUD ----------------------------------------------------------------
    def add(self, term: str, kind: str = "jargon", lang: str = "") -> bool:
        try:
            term = (term or "").strip()
            if not term or self._db is None:
                return False
            kind = (kind or "jargon").lower()
            if kind not in _KEYTERM_KINDS:
                kind = "jargon"
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO keyterms VALUES (?,?,?,?)",
                    (term, kind, (lang or "").lower(), time.time()))
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def remove(self, term: str) -> bool:
        try:
            term = (term or "").strip()
            if not term or self._db is None:
                return False
            with self._lock:
                cur = self._db.execute(
                    "DELETE FROM keyterms WHERE lower(term)=lower(?)", (term,))
                self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def list(self, kind: str = "", lang: str = "") -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            q = "SELECT term, kind, lang FROM keyterms"
            clauses, params = [], []
            if kind:
                clauses.append("kind=?"); params.append(kind.lower())
            if lang:
                clauses.append("(lang=? OR lang='')"); params.append(lang.lower())
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY added"
            rows = self._db.execute(q, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    # -- prompting ------------------------------------------------------------
    def initial_prompt(self, lang: str = "") -> str:
        """Build the faster-whisper ``initial_prompt`` from the vocabulary.

        Terms are grouped so the model sees names, dialect words and jargon
        as a hint list. Empty string when there is nothing to prompt.
        """
        try:
            terms = self.list(lang=lang or "")
            if not terms:
                return ""
            grouped: dict[str, list[str]] = {}
            for t in terms:
                grouped.setdefault(t.get("kind") or "jargon", []).append(
                    t.get("term") or "")
            bits = []
            labels = {"name": "Names", "dialect": "Dialect words",
                      "jargon": "Terms", "place": "Places"}
            for kind, words in grouped.items():
                words = [w for w in words if w]
                if words:
                    bits.append(f"{labels.get(kind, kind)}: {', '.join(words)}")
            return ". ".join(bits) + "." if bits else ""
        except Exception:  # noqa: BLE001
            return ""


# ---------------------------------------------------------------------------
# default stage implementations — compose the existing pieces
# ---------------------------------------------------------------------------

def _stage_denoise(audio: str, **kw: Any) -> dict[str, Any]:
    """Cleanup-as-preprocessor: enhance before transcribing (#103)."""
    try:
        from .edit import enhance_audio
        res = enhance_audio(audio)
        if res.get("ok"):
            return {"ok": True, "audio": res["output"],
                    "filter": res.get("filter", "")}
        return {"ok": True, "audio": audio, "filter": "none",
                "note": res.get("reason", "enhance skipped")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": True, "audio": audio, "filter": "none",
                "note": f"enhance failed, using original: {exc}"}


def _stage_transcribe(audio: str, **kw: Any) -> dict[str, Any]:
    """Word-timed transcription with keyterm prompting."""
    try:
        from .edit import transcript_edit, Word
        lang = (kw.get("lang") or "en")
        keyterms: KeytermStore | None = kw.get("keyterms")

        def _prompted(src: str, language: str) -> list | None:
            # Use faster-whisper directly so we can inject initial_prompt.
            try:
                from ..media_edit.captions import _extract_wav  # type: ignore
            except Exception:
                _extract_wav = None  # noqa: F841
            prompt = keyterms.initial_prompt(language) if keyterms else ""
            try:
                from ..voice.stt import FasterWhisperBackend
                backend = FasterWhisperBackend()
                wav = src
                if _extract_wav is not None:
                    try:
                        wav = str(_extract_wav(Path(src)))
                    except Exception:  # noqa: BLE001
                        wav = src
                kwargs: dict[str, Any] = dict(
                    language=language or None, vad_filter=True,
                    condition_on_previous_text=False, word_timestamps=True)
                if prompt:
                    kwargs["initial_prompt"] = prompt
                segments, _info = backend.model.transcribe(str(wav), **kwargs)
                words: list[Word] = []
                for seg in segments:
                    for w in (seg.words or []):
                        words.append(Word(text=w.word, start=w.start,
                                          end=w.end))
                return words
            except Exception as exc:  # noqa: BLE001
                _log.warning("keyterm-prompted transcription failed: %s", exc)
                return None

        fn = _prompted if (keyterms and keyterms.initial_prompt(lang)) else None
        t = transcript_edit(audio, language=lang,
                            enhance=False,  # denoise is its own stage
                            transcriber=fn)
        if t is None:
            return {"ok": False, "stage": "transcribe",
                    "reason": "no STT backend available — "
                              "pip install faster-whisper"}
        return {"ok": True, "audio": audio, "text": t.text,
                "language": t.language,
                "words": [w.to_dict() for w in t.words],
                "keyterm_prompt_used": bool(fn)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "transcribe", "reason": str(exc)}


def _stage_diarize(audio: str, **kw: Any) -> dict[str, Any]:
    """Speaker assignment. Tries real diarization first (WhisperX /
    pyannote when installed and authorized), degrades honestly to the
    single-speaker default — never invents speaker turns."""
    try:
        override = kw.get("diarizer")
        if callable(override):
            return override(audio, **kw)
        words = kw.get("words") or []
        real = diarize_real(
            audio, words,
            min_speakers=kw.get("min_speakers"),
            max_speakers=kw.get("max_speakers"))
        if real.get("ok"):
            return real
        # Honest default: one speaker. Never invent speaker turns.
        return {"ok": True, "speakers": ["A"],
                "segments": [{"speaker": "A", "words": words}],
                "method": "single-speaker-default",
                "note": real.get("reason", "")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "diarize", "reason": str(exc)}


def _hf_token() -> str:
    import os as _os
    for k in ("WHISPERX_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        v = (_os.environ.get(k) or "").strip()
        if v:
            return v
    return ""


def diarize_real(audio: str, words: Sequence[Any] | None = None, *,
                 min_speakers: int | None = None,
                 max_speakers: int | None = None) -> dict[str, Any]:
    """Real speaker diarization (the WhisperX pattern).

    Tries pyannote.audio's ``speaker-diarization-3.1`` (needs a Hugging
    Face token with the model EULA accepted — ``WHISPERX_HF_TOKEN`` /
    ``HF_TOKEN``), then assigns each word to a speaker by timestamp
    overlap (WhisperX's ``assign_word_speakers`` idea) so labels respect
    word boundaries.

    Returns ``{"ok", "speakers", "segments", "method"}`` or
    ``{"ok": False, "reason"}`` — the honest "can't" when pyannote or
    the token is missing. Never raises.
    """
    try:
        token = _hf_token()
        if not token:
            return {"ok": False,
                    "reason": "no diarization token — set WHISPERX_HF_TOKEN "
                              "(accept the pyannote/speaker-diarization-3.1 "
                              "EULA on Hugging Face first)"}
        try:
            from pyannote.audio import Pipeline
        except Exception:
            return {"ok": False,
                    "reason": "pyannote.audio not installed "
                              "(pip install pyannote.audio)"}
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token)
        if min_speakers or max_speakers:
            try:
                import inspect as _inspect
                sig = _inspect.signature(pipeline.__call__)
                call_kw: dict[str, Any] = {}
                if "min_speakers" in sig.parameters and min_speakers:
                    call_kw["min_speakers"] = int(min_speakers)
                if "max_speakers" in sig.parameters and max_speakers:
                    call_kw["max_speakers"] = int(max_speakers)
                diar = pipeline(audio, **call_kw)
            except Exception:
                diar = pipeline(audio)
        else:
            diar = pipeline(audio)
        # diarization segments: (segment, track, speaker)
        raw: list[tuple[float, float, str]] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            raw.append((float(turn.start), float(turn.end), str(speaker)))
        if not raw:
            return {"ok": False, "reason": "diarizer found no speech"}
        speakers = sorted({s for _, _, s in raw})
        label = {s: chr(ord("A") + i) for i, s in enumerate(speakers)}
        ws = list(words or [])

        def _word_speaker(w: Any) -> str:
            s = float(getattr(w, "start", 0.0) or 0.0)
            e = float(getattr(w, "end", s) or s)
            best, best_ov = "A", 0.0
            for rs, re_, sp in raw:
                ov = max(0.0, min(e, re_) - max(s, rs))
                if ov > best_ov:
                    best_ov, best = ov, label[sp]
            return best

        word_speakers = [_word_speaker(w) for w in ws] if ws else []
        # group consecutive same-speaker words into turns
        segments: list[dict[str, Any]] = []
        cur_sp, cur_words = None, []
        for w, sp in zip(ws, word_speakers):
            if sp != cur_sp and cur_words:
                segments.append({"speaker": cur_sp, "words": cur_words})
                cur_words = []
            cur_sp = sp
            cur_words.append(w)
        if cur_words:
            segments.append({"speaker": cur_sp, "words": cur_words})
        if not segments and not ws:
            segments = [{"speaker": label[s], "start": rs, "end": re_}
                        for rs, re_, s in raw]
        return {"ok": True, "speakers": [label[s] for s in speakers],
                "segments": segments, "method": "pyannote-3.1",
                "word_speakers": word_speakers}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"real diarization failed: {exc}"}


def group_turns(words: Sequence[Any],
                speakers: Sequence[str] | None = None
                ) -> list[dict[str, Any]]:
    """Group word-timed words into speaker turns.

    ``speakers`` parallels ``words`` (as from ``diarize_real``); without
    it, everything is one turn. Each turn: ``{"speaker", "start", "end",
    "text"}``. Never raises.
    """
    try:
        ws = list(words or [])
        sp = list(speakers or [])
        turns: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for i, w in enumerate(ws):
            s = sp[i] if i < len(sp) else "A"
            text = getattr(w, "text", "") or ""
            st = float(getattr(w, "start", 0.0) or 0.0)
            en = float(getattr(w, "end", st) or st)
            if cur is None or cur["speaker"] != s:
                if cur is not None:
                    turns.append(cur)
                cur = {"speaker": s, "start": st, "end": en, "text": text}
            else:
                cur["end"] = en
                cur["text"] = (cur["text"] + " " + text).strip()
        if cur is not None:
            turns.append(cur)
        return turns
    except Exception:  # noqa: BLE001
        return []


def _fmt_ts_srt(s: float) -> str:
    s = max(0.0, float(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")


def _fmt_ts_vtt(s: float) -> str:
    s = max(0.0, float(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def transcript_to_srt(turns: Sequence[dict[str, Any]]) -> str:
    """Speaker turns → SubRip subtitles. Never raises."""
    try:
        out: list[str] = []
        for i, t in enumerate(turns or [], 1):
            out.append(str(i))
            out.append(f"{_fmt_ts_srt(t.get('start', 0.0))} --> "
                       f"{_fmt_ts_srt(t.get('end', 0.0))}")
            out.append(f"[{t.get('speaker', 'A')}] {t.get('text', '')}".strip())
            out.append("")
        return "\n".join(out).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


def transcript_to_vtt(turns: Sequence[dict[str, Any]]) -> str:
    """Speaker turns → WebVTT subtitles. Never raises."""
    try:
        out = ["WEBVTT", ""]
        for t in turns or []:
            out.append(f"{_fmt_ts_vtt(t.get('start', 0.0))} --> "
                       f"{_fmt_ts_vtt(t.get('end', 0.0))}")
            out.append(f"<v {t.get('speaker', 'A')}>{t.get('text', '')}"
                       "</v>".strip())
            out.append("")
        return "\n".join(out).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


def meeting_minutes(turns: Sequence[dict[str, Any]], *,
                    title: str = "") -> str:
    """Speaker-grouped turns → Markdown meeting minutes (the
    whisperx-transcriber output pattern). Never raises."""
    try:
        lines = [f"# {title}" if title else "# Meeting minutes", ""]
        for t in turns or []:
            st = t.get("start", 0.0)
            m, sec = divmod(max(0.0, float(st)), 60)
            lines.append(f"**[{t.get('speaker', 'A')}]** "
                         f"`{int(m):02d}:{sec:04.1f}` — {t.get('text', '')}")
        return "\n".join(lines).strip() + "\n"
    except Exception:  # noqa: BLE001
        return ""


def _stage_edit(audio: str, **kw: Any) -> dict[str, Any]:
    """Filler removal + text edits (#103)."""
    try:
        from .edit import remove_fillers, apply_edits, transcript_edit, Edit
        lang = (kw.get("lang") or "en")
        cur = audio
        notes: list[str] = []
        if kw.get("remove_fillers"):
            res = remove_fillers(cur, lang=lang)
            if res.get("ok"):
                cur = res["output"]
                notes.append(res.get("note", "fillers removed"))
            else:
                notes.append(f"filler removal skipped: {res.get('reason')}")
        edits = kw.get("edits") or []
        if edits:
            t = transcript_edit(cur, language=lang, enhance=False)
            if t is None:
                return {"ok": False, "stage": "edit",
                        "reason": "could not re-transcribe for edits"}
            parsed: list[Edit] = []
            for e in edits:
                if isinstance(e, Edit):
                    parsed.append(e)
                elif isinstance(e, dict):
                    kind = (e.get("kind") or "").lower()
                    if kind == "delete":
                        parsed.append(Edit.delete(e.get("start", 0.0),
                                                 e.get("end", 0.0)))
                    elif kind == "rewrite":
                        parsed.append(Edit.rewrite(e.get("start", 0.0),
                                                  e.get("end", 0.0),
                                                  e.get("text", "")))
            res = apply_edits(t, parsed, ref_wav=cur,
                              synthesizer=kw.get("synthesizer"))
            if res.get("ok"):
                cur = res["output"]
                notes.append(f"{len(parsed)} edit(s) applied")
            else:
                return {"ok": False, "stage": "edit",
                        "reason": res.get("reason", "edits failed")}
        return {"ok": True, "audio": cur, "notes": notes}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "edit", "reason": str(exc)}


def _stage_synthesize(audio: str, **kw: Any) -> dict[str, Any]:
    """Multi-speaker TTS with audience routing enforced structurally.

    ``audience="public"`` removes XTTS from candidates before selection —
    it cannot be chosen, not merely discouraged.
    """
    try:
        from ..voice.tts import UniversalTTS
        audience = (kw.get("audience") or "private").lower()
        if audience not in ("private", "public"):
            audience = "private"
        segments = kw.get("segments") or []
        speakers = kw.get("speakers") or {}   # speaker -> voice ref path
        registry: EngineRegistry | None = kw.get("engine_registry")
        requested = (kw.get("tts_backend") or "auto").lower()

        # Structural audience guard — runs before any synthesis.
        if audience == "public" and requested in _NONCOMMERCIAL:
            return {"ok": False, "stage": "synthesize",
                    "reason": "XTTS is non-commercial and cannot be used for "
                              "public/community audio — pick a MIT-safe voice "
                              "(chatterbox/kokoro/piper)"}

        if not segments:
            return {"ok": False, "stage": "synthesize",
                    "reason": "nothing to speak — no script segments"}
        tts_kwargs: dict[str, Any] = {"audience": audience}
        if registry is not None:
            avail = kw.get("available_tts") or []
            pick = registry.pick_tts(audience, avail) if avail else None
            if pick:
                tts_kwargs["backend"] = pick
        elif requested != "auto":
            tts_kwargs["backend"] = requested
        engine = UniversalTTS(**tts_kwargs)

        pieces: list[str] = []
        for seg in segments:
            speaker = seg.get("speaker", "A")
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            voice_ref = speakers.get(speaker)
            out = kw.get("workdir", "/tmp")
            res = engine.speak(text, voice_ref=voice_ref, out_dir=out) \
                if hasattr(engine, "speak") else None
            if res is None:
                # Fall back to the backend-agnostic synthesize path.
                res = engine.synthesize(text, None) \
                    if hasattr(engine, "synthesize") else None
            if not res or not res.get("path"):
                return {"ok": False, "stage": "synthesize",
                        "reason": f"TTS failed for speaker {speaker!r} — "
                                  "no backend produced audio"}
            pieces.append(res["path"])
        if not pieces:
            return {"ok": False, "stage": "synthesize",
                    "reason": "no speakable text"}
        # Concat pieces into one wav.
        from ..media_edit.videos import concat
        if len(pieces) == 1:
            return {"ok": True, "audio": pieces[0],
                    "backend": getattr(engine, "_backend_name", "auto"),
                    "audience": audience, "pieces": 1}
        res = concat(pieces, out_ext=".wav")
        out_path = res.get("output") if isinstance(res, dict) else res
        if not out_path:
            return {"ok": False, "stage": "synthesize",
                    "reason": "could not join voice pieces"}
        return {"ok": True, "audio": out_path,
                "backend": getattr(engine, "_backend_name", "auto"),
                "audience": audience, "pieces": len(pieces)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "synthesize", "reason": str(exc)}


def _stage_master(audio: str, **kw: Any) -> dict[str, Any]:
    """LUFS normalize (Auphonic pattern, same as #106 audiobook mastering)."""
    try:
        target = float(kw.get("target_lufs", -16.0))
        if shutil.which("ffmpeg") is None:
            return {"ok": True, "audio": audio, "lufs": None,
                    "note": "ffmpeg missing — unmastered"}
        out = str(Path(kw.get("workdir", "/tmp"))
                  / f"mastered_{uuid.uuid4().hex[:8]}.wav")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", audio,
               "-filter:a", f"loudnorm=I={target}:TP=-1.5:LRA=11",
               "-ar", "24000", "-ac", "1", out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not Path(out).exists():
            return {"ok": True, "audio": audio, "lufs": None,
                    "note": "mastering failed — using unmastered audio"}
        return {"ok": True, "audio": out, "lufs": target}
    except Exception as exc:  # noqa: BLE001
        return {"ok": True, "audio": audio, "lufs": None,
                "note": f"mastering skipped: {exc}"}


_DEFAULT_STAGES: dict[str, StageFn] = {
    "denoise": _stage_denoise,
    "transcribe": _stage_transcribe,
    "diarize": _stage_diarize,
    "edit": _stage_edit,
    "synthesize": _stage_synthesize,
    "master": _stage_master,
}


# ---------------------------------------------------------------------------
# VoicePipeline — the composed pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """What the run should do. Stages not requested are skipped."""
    lang: str = "en"
    audience: str = "private"          # private | public — routed structurally
    denoise: bool = True
    transcribe: bool = True
    diarize: bool = False
    remove_fillers: bool = False
    edits: list[dict[str, Any]] = field(default_factory=list)
    synthesize: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)
    speakers: dict[str, str] = field(default_factory=dict)  # speaker -> voice ref
    min_speakers: int = 0          # diarization hint (0 = auto)
    max_speakers: int = 0          # diarization hint (0 = auto)
    master: bool = True
    target_lufs: float = -16.0
    tts_backend: str = "auto"
    keyterms: bool = True              # inject user vocabulary into STT
    workdir: str = "/tmp"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PipelineConfig":
        d = dict(d or {})
        try:
            return cls(
                lang=str(d.get("lang") or "en"),
                audience=str(d.get("audience") or "private"),
                denoise=bool(d.get("denoise", True)),
                transcribe=bool(d.get("transcribe", True)),
                diarize=bool(d.get("diarize", False)),
                remove_fillers=bool(d.get("remove_fillers", False)),
                edits=list(d.get("edits") or []),
                synthesize=bool(d.get("synthesize", False)),
                segments=list(d.get("segments") or []),
                speakers=dict(d.get("speakers") or {}),
                min_speakers=int(d.get("min_speakers") or 0),
                max_speakers=int(d.get("max_speakers") or 0),
                master=bool(d.get("master", True)),
                target_lufs=float(d.get("target_lufs", -16.0)),
                tts_backend=str(d.get("tts_backend") or "auto"),
                keyterms=bool(d.get("keyterms", True)),
                workdir=str(d.get("workdir") or "/tmp"),
            )
        except Exception:  # noqa: BLE001
            return cls()


class VoicePipeline:
    """capture → cleanup → transcribe → edit → multi-speaker TTS → master.

    Stages are pluggable via ``register_stage``; defaults compose the
    existing pieces (#25 STT, #30 TTS, #103 editing, #106 mastering).
    Never raises — failures are honest per-stage dicts.
    """

    def __init__(self, *, profile: str = "",
                 engine_registry: EngineRegistry | None = None,
                 keyterm_store: KeytermStore | None = None) -> None:
        self.registry = engine_registry or EngineRegistry(profile)
        self.keyterms = keyterm_store or KeytermStore()
        self._stages: dict[str, StageFn] = dict(_DEFAULT_STAGES)

    def register_stage(self, name: str, fn: StageFn) -> bool:
        """Plug in (or replace) a stage. Returns False on bad input."""
        try:
            if (name or "") not in STAGES or not callable(fn):
                return False
            self._stages[name] = fn
            return True
        except Exception:  # noqa: BLE001
            return False

    def stage_names(self) -> list[str]:
        return [s for s in STAGES if s in self._stages]

    # -- run ------------------------------------------------------------------
    def run(self, audio: str | os.PathLike[str],
            config: PipelineConfig | dict[str, Any] | None = None,
            ) -> dict[str, Any]:
        """Run the pipeline. Returns an honest result dict; never raises."""
        try:
            cfg = config if isinstance(config, PipelineConfig) \
                else PipelineConfig.from_dict(config)
            p = Path(audio)
            if not p.exists():
                return {"ok": False, "stage": "input",
                        "reason": f"no such audio: {audio}"}
            workdir = Path(cfg.workdir or "/tmp")
            try:
                workdir.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001
                pass

            cur = str(p)
            stage_results: dict[str, Any] = {}
            transcript_text = ""
            words: list[dict[str, Any]] = []

            def _fail(stage: str, reason: str) -> dict[str, Any]:
                return {"ok": False, "stage": stage, "reason": reason,
                        "stages": stage_results, "audio": cur}

            # 1. denoise
            if cfg.denoise:
                r = self._stages["denoise"](cur)
                stage_results["denoise"] = r
                if r.get("audio"):
                    cur = r["audio"]

            # 2. transcribe
            if cfg.transcribe:
                r = self._stages["transcribe"](
                    cur, lang=cfg.lang,
                    keyterms=self.keyterms if cfg.keyterms else None)
                stage_results["transcribe"] = r
                if not r.get("ok"):
                    return _fail("transcribe", r.get("reason", "failed"))
                transcript_text = r.get("text", "")
                words = r.get("words") or []

            # 3. diarize
            if cfg.diarize:
                r = self._stages["diarize"](
                    cur, words=words,
                    min_speakers=cfg.min_speakers or None,
                    max_speakers=cfg.max_speakers or None)
                stage_results["diarize"] = r
                if not r.get("ok"):
                    return _fail("diarize", r.get("reason", "failed"))

            # 4. edit
            if cfg.remove_fillers or cfg.edits:
                r = self._stages["edit"](
                    cur, lang=cfg.lang, remove_fillers=cfg.remove_fillers,
                    edits=cfg.edits,
                    synthesizer=__import__("functools").partial(
                        _pipeline_synthesizer, self, cfg))
                stage_results["edit"] = r
                if not r.get("ok"):
                    return _fail("edit", r.get("reason", "failed"))
                cur = r.get("audio", cur)

            # 5. synthesize (multi-speaker, audience-routed)
            if cfg.synthesize:
                segments = cfg.segments
                if not segments and transcript_text:
                    # Default: speak the transcript as one voice.
                    segments = [{"speaker": "A", "text": transcript_text}]
                r = self._stages["synthesize"](
                    cur, segments=segments, speakers=cfg.speakers,
                    audience=cfg.audience, tts_backend=cfg.tts_backend,
                    engine_registry=self.registry,
                    workdir=str(workdir))
                stage_results["synthesize"] = r
                if not r.get("ok"):
                    return _fail("synthesize", r.get("reason", "failed"))
                cur = r.get("audio", cur)

            # 6. master
            if cfg.master:
                r = self._stages["master"](cur, target_lufs=cfg.target_lufs,
                                          workdir=str(workdir))
                stage_results["master"] = r
                cur = r.get("audio", cur)

            return {"ok": True, "audio": cur,
                    "transcript": transcript_text,
                    "stages": stage_results,
                    "profile": self.registry.profile,
                    "audience": cfg.audience}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "stage": "pipeline", "reason": str(exc)}


def _pipeline_synthesizer(pipeline: VoicePipeline, cfg: PipelineConfig,
                          text: str, ref_wav: str, **kw: Any) -> str | None:
    """Adapter so #103's apply_edits can re-speak via the pipeline's routing."""
    try:
        r = _stage_synthesize(
            ref_wav,
            segments=[{"speaker": "A", "text": text}],
            speakers={"A": ref_wav},
            audience=cfg.audience, tts_backend=cfg.tts_backend,
            engine_registry=pipeline.registry,
            workdir=cfg.workdir or "/tmp")
        return r.get("audio") if r.get("ok") else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# convenience entry points
# ---------------------------------------------------------------------------

def run_pipeline(audio: str | os.PathLike[str],
                 config: PipelineConfig | dict[str, Any] | None = None,
                 **kw: Any) -> dict[str, Any]:
    """One-shot: ``run_pipeline("note.m4a", {"remove_fillers": True})``."""
    try:
        pipeline = VoicePipeline(**{k: v for k, v in kw.items()
                                    if k in ("profile", "engine_registry",
                                             "keyterm_store")})
        return pipeline.run(audio, config)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "pipeline", "reason": str(exc)}


def transcribe_clean(audio: str | os.PathLike[str], *,
                      lang: str = "en", keyterms: bool = True,
                      **kw: Any) -> dict[str, Any]:
    """Denoise → transcribe (with keyterm prompting). The common read path."""
    try:
        return run_pipeline(audio, {
            "lang": lang, "keyterms": keyterms,
            "denoise": True, "transcribe": True,
            "diarize": False, "synthesize": False, "master": False,
            **kw})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "pipeline", "reason": str(exc)}


# ---------------------------------------------------------------------------
# chat — /voice pipeline | engines | keyterms
# ---------------------------------------------------------------------------

_USAGE = (
    "🎛️ /voice pipeline <audio> [lang] [private|public] — run the full voice pipeline\n"
    "🎛️ /voice engines — show profile-gated backend picks\n"
    "🎛️ /voice diarize <audio> [min] [max] — who spoke when (pyannote when available)\n"
    "🎛️ /voice minutes <audio> [lang] — speaker-grouped meeting minutes\n"
    "🎛️ /voice srt <audio> [lang] — SubRip subtitles with speaker labels\n"
    "🎛️ /voice keyterms add <term> [kind] [lang] — teach STT your vocabulary\n"
    "🎛️ /voice keyterms list — show the vocabulary\n"
    "🎛️ /voice keyterms remove <term>"
)


def _turns_for(audio: str, lang: str = "en") -> list[dict[str, Any]] | None:
    """Transcribe + diarize → speaker turns. None on failure."""
    try:
        res = run_pipeline(audio, {
            "lang": lang, "denoise": True, "transcribe": True,
            "diarize": True, "synthesize": False, "master": False})
        if not res.get("ok"):
            return None
        st = (res.get("stages") or {}).get("transcribe") or {}
        words = st.get("words") or []
        dz = (res.get("stages") or {}).get("diarize") or {}
        speakers = dz.get("word_speakers") or []
        from ..media_edit.captions import Word
        ws = [Word.from_dict(w) if isinstance(w, dict) else w for w in words]
        return group_turns(ws, speakers)
    except Exception:  # noqa: BLE001
        return None


def control_voice(tail: str, *, context: Any = None,
                  chat: Any = None, **kwargs: Any) -> str:
    """Chat entry: /voice pipeline|engines|keyterms. Owner-only at dispatch."""
    try:
        parts = (tail or "").split()
        if not parts:
            return _USAGE
        verb = parts[0].lower()

        if verb == "engines":
            reg = EngineRegistry(context=context)
            d = reg.describe()
            lines = [f"🎛️ voice engines — profile: {d['profile']}"]
            lines.append("  STT order: " + ", ".join(d["stt_order"]))
            lines.append("  TTS (private): " + ", ".join(d["tts_private"]))
            lines.append("  TTS (public): " + ", ".join(d["tts_public"])
                      + "  ← XTTS excluded (non-commercial)")
            return "\n".join(lines)

        if verb == "keyterms":
            store = KeytermStore()
            if len(parts) >= 2 and parts[1].lower() == "add" and len(parts) >= 3:
                kind = parts[3] if len(parts) >= 4 else "jargon"
                lang = parts[4] if len(parts) >= 5 else ""
                ok = store.add(parts[2], kind=kind, lang=lang)
                return (f"📝 keyterm added: {parts[2]} ({kind})"
                        if ok else "📝 couldn't add that keyterm")
            if len(parts) >= 2 and parts[1].lower() == "remove" \
                    and len(parts) >= 3:
                ok = store.remove(parts[2])
                return (f"📝 keyterm removed: {parts[2]}"
                        if ok else "📝 no such keyterm")
            terms = store.list()
            if not terms:
                return ("📝 no keyterms yet — /voice keyterms add <term> "
                        "[name|dialect|jargon|place] [lang]")
            lines = [f"📝 keyterms ({len(terms)}):"]
            for t in terms[:30]:
                lines.append(f"  · {t['term']} [{t['kind']}]"
                             + (f" ({t['lang']})" if t.get("lang") else ""))
            return "\n".join(lines)

        if verb == "pipeline" and len(parts) >= 2:
            audio = parts[1]
            lang = parts[2] if len(parts) >= 3 else "en"
            audience = parts[3].lower() if len(parts) >= 4 else "private"
            if audience not in ("private", "public"):
                audience = "private"
            res = run_pipeline(audio, {
                "lang": lang, "audience": audience,
                "denoise": True, "transcribe": True,
                "remove_fillers": True, "master": True})
            if not res.get("ok"):
                return (f"🎛️ pipeline stopped at {res.get('stage')}: "
                        f"{res.get('reason')}")
            lines = [f"🎛️ pipeline done — profile {res.get('profile')}, "
                     f"audience {res.get('audience')}"]
            tx = (res.get("transcript") or "")[:200]
            if tx:
                lines.append(f"📝 {tx}{'…' if len(res['transcript']) > 200 else ''}")
            lines.append(f"📁 {res.get('audio')}")
            return "\n".join(lines)

        if verb == "diarize" and len(parts) >= 2:
            turns = _turns_for(parts[1])
            if turns is None:
                return ("🎛️ couldn't diarize that — needs faster-whisper "
                        "(pip install faster-whisper) and an audio file. "
                        "Real speaker labels need pyannote.audio + "
                        "WHISPERX_HF_TOKEN.")
            speakers = sorted({t["speaker"] for t in turns})
            lines = [f"🎛️ {len(speakers)} speaker(s): "
                     f"{', '.join(speakers)} — {len(turns)} turn(s)"]
            for t in turns[:20]:
                m, sec = divmod(max(0.0, t["start"]), 60)
                lines.append(f"  [{t['speaker']}] {int(m):02d}:{sec:04.1f} — "
                             f"{t['text'][:80]}")
            if len(turns) > 20:
                lines.append(f"  … +{len(turns) - 20} more turns")
            return "\n".join(lines)

        if verb == "minutes" and len(parts) >= 2:
            lang = parts[2] if len(parts) >= 3 else "en"
            turns = _turns_for(parts[1], lang)
            if turns is None:
                return "🎛️ couldn't transcribe that — see /voice diarize."
            return "🎛️ " + meeting_minutes(turns).replace("\n", "\n")

        if verb == "srt" and len(parts) >= 2:
            lang = parts[2] if len(parts) >= 3 else "en"
            turns = _turns_for(parts[1], lang)
            if turns is None:
                return "🎛️ couldn't transcribe that — see /voice diarize."
            return "🎛️ subtitles:\n```\n" + transcript_to_srt(turns) + "```"

        return _USAGE
    except Exception as exc:  # noqa: BLE001
        return f"🎛️ voice pipeline failed: {exc}"
