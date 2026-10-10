"""EPUB → audiobook, one click, with store-compliant AI disclosure.

``epub_to_audiobook(epub, voice_cast, target_stores)`` → chapter split
(from the EPUB TOC or heading structure) → per-chapter TTS (private
XTTS stack for the user's own books; multi-voice casting optional) →
per-store LUFS master (Auphonic pattern via ffmpeg ``loudnorm``:
Spotify/Kobo −16 LUFS, ACX-shaped −20 LUFS RMS-gated) → packaged
audiobook with AI-disclosure metadata attached per target store.

Disclosure rules are a versioned config (``DISCLOSURE_RULES``); they
are re-checked before every publish run because store rules shift.
Stores that ban AI narration (ACX third-party voices, Author's
Republic) are REFUSED with the honest reason — never shipped and
hoped. ``check_compliance`` PASS/WARN/FAILs the master against the
store's targets; ``chapter_pacing`` catches the RMS drift between
chapters that triggers real rejections.

Every method never raises and refuses rather than producing fake
audio or wrong metadata.

Pairs with #44 (creator voice): pass the author's cloned voice ref
as the narrator and the audiobook speaks in their voice.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("nomorals.audio.audiobook")

# ── disclosure rules (versioned — checked before every publish) ────────────

DISCLOSURE_RULES: dict[str, Any] = {
    "version": "2026-10b",
    "checked": "2026-10-10",
    # ACX/Audible: third-party AI voices are REJECTED under the ToS —
    # Audible's own "Virtual Voice" beta program is the only AI path,
    # and Audible marks those titles itself. Devon's cloned-voice
    # pipeline cannot ship to ACX: refuse with the honest reason.
    "acx": {
        "allowed": False,
        "requires_disclosure": True,
        "disclosure_text": "AI narration is only accepted via Audible's "
                           "own Virtual Voice program; third-party "
                           "synthetic voices are rejected.",
        "field": "ai_narration_disclosure",
        "note": "ACX rejects third-party AI narration — use Audible's "
                "Virtual Voice program or ship human-narrated.",
        "block_reason": "ACX terms reject third-party AI-generated voices "
                        "(Audible Virtual Voice program only).",
    },
    # Spotify for Authors (Findaway Voices): digital narration accepted
    # when the uploader ticks "This audiobook uses digital voice
    # narration".
    "spotify": {
        "allowed": True,
        "requires_disclosure": True,
        "disclosure_text": "This audiobook uses digital voice narration.",
        "field": "ai_generated",
        "note": "Spotify for Authors: tick 'This audiobook uses digital "
                "voice narration' at upload.",
    },
    # Kobo Writing Life: list the narrator as "Synthesised voice".
    "kobo": {
        "allowed": True,
        "requires_disclosure": True,
        "disclosure_text": "Narrator: Synthesised voice.",
        "field": "ai_narration",
        "note": "Kobo Writing Life: set the narrator field to "
                "'Synthesised voice'.",
    },
    # Author's Republic: NO AI-narrated components at all — the book can
    # be removed and royalties withheld if discovered.
    "authors_republic": {
        "allowed": False,
        "requires_disclosure": False,
        "disclosure_text": "",
        "field": "",
        "note": "Author's Republic does not allow any AI-narrated "
                "components.",
        "block_reason": "Author's Republic bans AI-narrated audiobooks "
                        "outright (removal + withheld royalties).",
    },
}

KNOWN_STORES = ("acx", "spotify", "kobo")

#: Every store with a versioned rule card (KNOWN_STORES plus stores that
#: ban AI narration outright — targeting them is refused with the reason).
RULED_STORES = tuple(s for s in DISCLOSURE_RULES if s not in ("version", "checked"))

#: Mastering targets per store. ACX measures RMS (−18…−23 dB) and noise
#: floor (< −60 dB); podcast/music stores use integrated LUFS.
PLATFORM_TARGETS: dict[str, dict[str, Any]] = {
    "acx": {"lufs": -20.0, "tp_db": -3.0, "lra": 11.0,
            "rms_db": (-23.0, -18.0), "noise_floor_db": -60.0,
            "codec": "mp3", "bitrate": "192k", "sr": 44100,
            "note": "ACX: RMS −18…−23 dB, noise floor < −60 dB, MP3 192k"},
    "spotify": {"lufs": -16.0, "tp_db": -1.0, "lra": 11.0,
                "codec": "wav", "sr": 44100,
                "note": "Spotify/Findaway: −16 LUFS, −1 dBTP"},
    "kobo": {"lufs": -16.0, "tp_db": -1.0, "lra": 11.0,
             "codec": "wav", "sr": 44100,
             "note": "Kobo: −16 LUFS, −1 dBTP"},
    "podcast": {"lufs": -16.0, "tp_db": -1.0, "lra": 11.0,
                "codec": "mp3", "bitrate": "128k", "sr": 44100,
                "note": "Podcast universal: −16 LUFS, −1 dBTP"},
    "youtube": {"lufs": -14.0, "tp_db": -1.0, "lra": 11.0,
                "codec": "wav", "sr": 44100,
                "note": "YouTube: −14 LUFS, −1 dBTP"},
}


def store_rules(store: str) -> dict[str, Any]:
    """The disclosure/mastering rule card for one store. Never raises."""
    try:
        s = (store or "").strip().lower()
        rules = DISCLOSURE_RULES.get(s, {})
        target = PLATFORM_TARGETS.get(s, {})
        if not rules:
            return {"ok": False, "reason": f"unknown store {store!r}",
                    "known": list(RULED_STORES)}
        return {"ok": True, "store": s, "allowed": rules.get("allowed", True),
                "requires_disclosure": rules.get("requires_disclosure", False),
                "disclosure_text": rules.get("disclosure_text", ""),
                "note": rules.get("note", ""),
                "block_reason": rules.get("block_reason", ""),
                "target": target,
                "rules_version": DISCLOSURE_RULES["version"]}
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "rule lookup failed"}


def check_store_allowed(stores: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Split ``stores`` into (allowed, blocked-with-reason). Never raises."""
    allowed: list[str] = []
    blocked: list[dict[str, str]] = []
    for s in (stores or []):
        r = store_rules(s)
        if not r.get("ok"):
            blocked.append({"store": s, "reason": r.get("reason", "?")})
        elif not r.get("allowed", True):
            blocked.append({"store": s,
                            "reason": r.get("block_reason") or
                            "AI narration not allowed on this store"})
        else:
            allowed.append(s)
    return allowed, blocked

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9'’]+")


# ── model ──────────────────────────────────────────────────────────────────


@dataclass
class BookChapter:
    """One chapter of the audiobook."""

    index: int = 0
    title: str = ""
    text: str = ""
    audio_path: str = ""
    duration_s: float = 0.0
    voice: str = "narrator"


@dataclass
class Audiobook:
    """A produced audiobook package."""

    book_id: str = ""
    title: str = ""
    author: str = ""
    chapters: list[BookChapter] = field(default_factory=list)
    master_path: str = ""
    duration_s: float = 0.0
    lufs: float = -16.0
    disclosure: dict[str, dict[str, str]] = field(default_factory=dict)
    target_stores: list[str] = field(default_factory=list)
    created_at: float = 0.0


# ── chapter split ──────────────────────────────────────────────────────────


def split_chapters(epub_path: str, *,
                   parse_fn: Callable[[str], Any] | None = None
                   ) -> list[BookChapter]:
    """EPUB → chapters (TOC/heading structure, else length-based split).

    Never raises; returns [] when the book cannot be parsed.
    """
    try:
        path = Path(epub_path or "")
        if not path.exists():
            return []
        if parse_fn is not None:
            doc = parse_fn(str(path))
        else:
            from ..documents import parse_path
            doc = parse_path(str(path))
        sections = getattr(doc, "sections", None) or []
        chapters: list[BookChapter] = []
        idx = 0
        for sec in sections:
            heading = (getattr(sec, "heading", "") or "").strip()
            text = (getattr(sec, "text", "") or "").strip()
            if not text:
                continue
            level = int(getattr(sec, "level", 2) or 2)
            if level <= 1 or len(chapters) == 0 and not heading:
                # new chapter at every h1; first section starts chapter 0
                if heading or not chapters:
                    idx += 1
                    chapters.append(BookChapter(
                        index=idx, title=heading or f"Chapter {idx}",
                        text=text))
                elif chapters:
                    chapters[-1].text += "\n\n" + text
            else:
                chapters[-1].text += ("\n\n" + heading + "\n" if heading else "\n\n") + text
        if not chapters:
            # fallback: length-based split of full text
            full = ""
            try:
                from ..documents.model import full_text
                full = full_text(doc) or ""
            except Exception:  # noqa: BLE001
                pass
            if not full.strip():
                return []
            words = full.split()
            size = 2500
            for i in range(0, len(words), size):
                idx += 1
                chunk = " ".join(words[i:i + size])
                chapters.append(BookChapter(index=idx, title=f"Part {idx}",
                                            text=chunk))
        return [c for c in chapters if c.text.strip()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("split_chapters failed: %s", exc)
        return []


def _book_title_author(epub_path: str, *,
                       parse_fn: Callable[[str], Any] | None = None
                       ) -> tuple[str, str]:
    try:
        if parse_fn is not None:
            doc = parse_fn(epub_path)
        else:
            from ..documents import parse_path
            doc = parse_path(epub_path)
        meta = getattr(doc, "metadata", None) or {}
        title = str(meta.get("title", "") or "").strip() or \
            Path(epub_path).stem.replace("_", " ")
        author = str(meta.get("author", "") or meta.get("creator", "") or
                     "").strip()
        return title, author
    except Exception:  # noqa: BLE001
        return Path(epub_path).stem.replace("_", " "), ""


# ── TTS ────────────────────────────────────────────────────────────────────


def _default_tts_fn(text: str, voice_ref: str, *,
                    lang: str = "en") -> str | None:
    """Speak ``text`` in the cloned voice at ``voice_ref`` (private XTTS).

    Returns the WAV path, or None (never fake audio) when anything is
    missing — the caller refuses the chapter rather than faking it.
    """
    try:
        if not voice_ref or not Path(voice_ref).exists():
            return None
        from ..voice.tts import UniversalTTS
        tmp = tempfile.mkdtemp(prefix="audiobook-tts-")
        tts = UniversalTTS(backend="auto", voices_dir=tmp, audience="private")
        voice_name = "audiobook-narrator"
        tts.voices.upload_voice(voice_name, voice_ref, language=lang or "en")
        # chunk long text so no single TTS call explodes
        chunks: list[str] = []
        words = text.split()
        size = 400
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i + size]))
        pieces: list[str] = []
        for chunk in chunks:
            out = tts.speak(chunk, voice_name=voice_name, audience="private")
            p = (out or {}).get("path", "")
            if not p or not Path(p).exists():
                return None
            pieces.append(p)
        if len(pieces) == 1:
            return pieces[0]
        from ..media_edit.videos import concat
        res = concat(pieces, out_dir=tmp, suffix="chapter", ext=".wav")
        out = (res or {}).get("output", "") if isinstance(res, dict) else ""
        return out if out and Path(out).exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("audiobook TTS failed: %s", exc)
        return None


def _audio_duration(p: Path) -> float:
    """Duration in seconds via ffprobe. Never raises."""
    try:
        if shutil.which("ffprobe") is None:
            return 0.0
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=30)
        return float((out.stdout or "").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def master_lufs(wav_path: str, *, target_lufs: float = -16.0,
                out_dir: str | None = None,
                store: str = "") -> str | None:
    """LUFS-normalize a WAV (Auphonic pattern) via ffmpeg loudnorm.

    ``store`` selects that store's mastering target from
    ``PLATFORM_TARGETS`` (ACX → −20 LUFS MP3-ready chain, etc.);
    ``target_lufs`` overrides when no store is given. Returns the
    mastered path, or None when ffmpeg is unavailable. Never raises.
    """
    try:
        p = Path(wav_path or "")
        if not p.exists():
            return None
        if shutil.which("ffmpeg") is None:
            return None
        tgt = PLATFORM_TARGETS.get((store or "").strip().lower(), {})
        lufs = float(tgt.get("lufs", target_lufs))
        tp = float(tgt.get("tp_db", -1.5))
        lra = float(tgt.get("lra", 11.0))
        sr = int(tgt.get("sr", 44100))
        tmp = Path(out_dir or tempfile.mkdtemp(prefix="audiobook-master-"))
        out = tmp / (p.stem + f"-lufs{int(lufs)}.wav")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(p),
               "-af", f"loudnorm=I={lufs}:TP={tp}:LRA={lra}",
               "-ar", str(sr), "-ac", "2", str(out)]
        subprocess.run(cmd, capture_output=True, timeout=600)
        return str(out) if out.exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("master_lufs failed: %s", exc)
        return None


def measure_loudness(wav_path: str) -> dict[str, Any]:
    """Measure integrated LUFS, true peak, RMS and noise floor via
    ffmpeg (ebur128 + astats). The podcast_leveler measurement pattern.

    Returns ``{"ok", "lufs", "true_peak_db", "rms_db", "noise_floor_db"}``
    — or ``{"ok": False, "reason"}``. Never raises.
    """
    try:
        p = Path(wav_path or "")
        if not p.exists():
            return {"ok": False, "reason": f"no such file: {wav_path}"}
        if shutil.which("ffmpeg") is None:
            return {"ok": False, "reason": "ffmpeg unavailable"}
        out = {"ok": True, "lufs": None, "true_peak_db": None,
               "rms_db": None, "noise_floor_db": None}
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(p), "-map", "0:a",
             "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300)
        txt = (r.stderr or "") + (r.stdout or "")
        m = re.search(r"Integrated loudness:\s*\n?\s*I:\s*([-\d.]+)\s*LUFS",
                      txt)
        if m:
            out["lufs"] = float(m.group(1))
        m = re.search(r"Peak:\s*\n?\s*Peak:\s*([-\d.]+)\s*dBFS", txt)
        if m:
            out["true_peak_db"] = float(m.group(1))
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(p), "-map", "0:a",
             "-af", "astats=metadata=1:reset=1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300)
        txt = (r.stderr or "") + (r.stdout or "")
        m = re.search(r"RMS level dB:\s*([-\d.]+)", txt)
        if m:
            out["rms_db"] = float(m.group(1))
        m = re.search(r"Noise floor dB:\s*([-\d.]+)", txt)
        if m:
            out["noise_floor_db"] = float(m.group(1))
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"loudness measurement failed: {exc}"}


def check_compliance(wav_path: str, store: str = "spotify") -> dict[str, Any]:
    """PASS / WARN / FAIL the mastered file against a store's targets
    (the podcast_leveler compliance-card pattern).

    Never raises; ``{"ok", "store", "verdict", "checks": [...]}``.
    """
    try:
        tgt = PLATFORM_TARGETS.get((store or "").strip().lower())
        if tgt is None:
            return {"ok": False,
                    "reason": f"unknown store {store!r} — known: "
                              f"{', '.join(sorted(PLATFORM_TARGETS))}"}
        m = measure_loudness(wav_path)
        if not m.get("ok"):
            return {"ok": False, "reason": m.get("reason", "?")}
        checks: list[dict[str, Any]] = []
        lufs, tp = m.get("lufs"), m.get("true_peak_db")
        if lufs is not None:
            off = abs(lufs - float(tgt["lufs"]))
            checks.append({"metric": "integrated LUFS",
                           "value": round(lufs, 1),
                           "target": tgt["lufs"],
                           "status": "PASS" if off <= 1.0
                           else "WARN" if off <= 2.0 else "FAIL"})
        if tp is not None:
            checks.append({"metric": "true peak dBTP",
                           "value": round(tp, 1),
                           "target": f"≤ {tgt['tp_db']}",
                           "status": "PASS" if tp <= float(tgt["tp_db"]) + 0.1
                           else "FAIL"})
        rms = m.get("rms_db")
        rms_tgt = tgt.get("rms_db")
        if rms is not None and rms_tgt:
            lo, hi = rms_tgt
            checks.append({"metric": "RMS dB",
                           "value": round(rms, 1),
                           "target": f"{lo}…{hi}",
                           "status": "PASS" if lo - 1.0 <= rms <= hi + 1.0
                           else "WARN"})
        nf = m.get("noise_floor_db")
        nf_tgt = tgt.get("noise_floor_db")
        if nf is not None and nf_tgt:
            checks.append({"metric": "noise floor dB",
                           "value": round(nf, 1),
                           "target": f"< {nf_tgt}",
                           "status": "PASS" if nf < float(nf_tgt)
                           else "WARN"})
        statuses = [c["status"] for c in checks]
        verdict = ("FAIL" if "FAIL" in statuses
                   else "WARN" if "WARN" in statuses else "PASS")
        return {"ok": True, "store": store, "verdict": verdict,
                "checks": checks,
                "measured": {k: m.get(k) for k in
                             ("lufs", "true_peak_db", "rms_db",
                              "noise_floor_db")}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"compliance check failed: {exc}"}


def export_mp3(wav_path: str, *, bitrate: str = "192k",
               out_dir: str | None = None) -> str | None:
    """Encode an MP3 at ``bitrate`` (the ACX delivery preset is 192k).

    Returns the MP3 path or None. Never raises.
    """
    try:
        p = Path(wav_path or "")
        if not p.exists() or shutil.which("ffmpeg") is None:
            return None
        tmp = Path(out_dir or p.parent)
        out = tmp / (p.stem + f"-{bitrate.replace('k', 'kbps')}.mp3")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(p),
             "-codec:a", "libmp3lame", "-b:a", bitrate, str(out)],
            capture_output=True, timeout=600)
        return str(out) if r.returncode == 0 and out.exists() else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("export_mp3 failed: %s", exc)
        return None


def chapter_pacing(chapters: list[BookChapter]) -> dict[str, Any]:
    """Check narration consistency across chapters (a known store
    rejection trigger): RMS drift between chapters.

    Returns ``{"ok", "max_drift_db", "verdict", "note"}``. Never raises.
    """
    try:
        rms_vals: list[float] = []
        for ch in (chapters or []):
            if ch.audio_path and Path(ch.audio_path).exists():
                m = measure_loudness(ch.audio_path)
                if m.get("ok") and m.get("rms_db") is not None:
                    rms_vals.append(m["rms_db"])
        if len(rms_vals) < 2:
            return {"ok": True, "max_drift_db": 0.0, "verdict": "PASS",
                    "note": "not enough chapters measured"}
        drift = max(rms_vals) - min(rms_vals)
        verdict = "PASS" if drift <= 3.0 else "WARN" if drift <= 6.0 else "FAIL"
        return {"ok": True, "max_drift_db": round(drift, 1),
                "verdict": verdict,
                "note": f"chapter RMS drift {drift:.1f} dB — {verdict}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"pacing check failed: {exc}"}


# ── the one-click pipeline ─────────────────────────────────────────────────


def epub_to_audiobook(epub_path: str, voice_cast: dict[str, str] | None = None,
                      target_stores: list[str] | None = None, *,
                      lang: str = "en",
                      tts_fn: Callable[..., str | None] | None = None,
                      parse_fn: Callable[[str], Any] | None = None,
                      out_dir: str | None = None,
                      store: "AudiobookStore | None" = None,
                      ) -> dict[str, Any]:
    """EPUB in → chaptered, mastered audiobook out.

    ``voice_cast`` maps ``{"narrator": <voice ref path>}`` (the #44
    creator-voice pairing: the author's own cloned voice); optional
    per-chapter voices via ``{"chapter:<n>": ref}``.

    Returns ``{"ok", "audiobook", "note"}`` or ``{"ok": False,
    "reason"}``. Never raises; refuses rather than faking audio.
    """
    try:
        path = Path(epub_path or "")
        if not path.exists():
            return {"ok": False, "reason": "no EPUB found — nothing to read"}
        # fail fast on store rules before the expensive chapter work
        stores = [s.strip().lower() for s in (target_stores or [])
                  if s and s.strip().lower() in RULED_STORES]
        if not stores:
            stores = ["spotify"]  # sensible default target
        allowed, blocked = check_store_allowed(stores)
        if not allowed:
            why = "; ".join(f"{b['store']}: {b['reason']}" for b in blocked)
            return {"ok": False,
                    "reason": f"no shippable store — {why}"}
        stores = allowed
        chapters = split_chapters(str(path), parse_fn=parse_fn)
        if not chapters:
            return {"ok": False,
                    "reason": "could not read any chapters from the EPUB"}
        cast = dict(voice_cast or {})
        narrator_ref = cast.get("narrator", "")
        if not narrator_ref:
            return {"ok": False,
                    "reason": "no narrator voice — pass voice_cast={'narrator': <voice reference audio>}"}
        title, author = _book_title_author(str(path), parse_fn=parse_fn)
        work = Path(out_dir or tempfile.mkdtemp(prefix="audiobook-"))
        rendered: list[BookChapter] = []
        for ch in chapters:
            ref = cast.get(f"chapter:{ch.index}", narrator_ref)
            wav = None
            if tts_fn is not None:
                try:
                    wav = tts_fn(ch.text, ref, lang=lang)
                except TypeError:
                    wav = tts_fn(ch.text, ref)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("tts_fn failed for chapter %d: %s",
                                 ch.index, exc)
                    wav = None
            else:
                wav = _default_tts_fn(ch.text, ref, lang=lang)
            if not wav or not Path(wav).exists():
                return {"ok": False,
                        "reason": f"no TTS audio for chapter {ch.index} "
                                  f"({ch.title}) — stopping rather than "
                                  "shipping a broken audiobook"}
            from .edit import _normalize_wav
            norm = _normalize_wav(wav, 24000, work)
            ch.audio_path = norm or wav
            ch.duration_s = _audio_duration(Path(ch.audio_path))
            rendered.append(ch)
        # concat chapters → master
        from ..media_edit.videos import concat
        if len(rendered) == 1:
            combined = rendered[0].audio_path
        else:
            res = concat([c.audio_path for c in rendered],
                         out_dir=str(work), suffix="audiobook",
                         ext=".wav")
            combined = (res or {}).get("output", "") \
                if isinstance(res, dict) else ""
        if not combined or not Path(combined).exists():
            return {"ok": False, "reason": "chapter concat failed"}
        mastered = master_lufs(combined, out_dir=str(work),
                              store=stores[0])
        if not mastered:
            return {"ok": False,
                    "reason": "LUFS mastering failed (ffmpeg unavailable?)"}
        tgt = PLATFORM_TARGETS.get(stores[0], {})
        compliance = check_compliance(mastered, stores[0])
        pacing = chapter_pacing(rendered)
        # ACX-shaped stores want an MP3-192 delivery file too
        mp3_path = ""
        if tgt.get("codec") == "mp3":
            mp3_path = export_mp3(mastered,
                                  bitrate=str(tgt.get("bitrate", "192k")),
                                  out_dir=str(work)) or ""
        disclosure = {
            s: {"text": DISCLOSURE_RULES[s]["disclosure_text"],
                "field": DISCLOSURE_RULES[s]["field"],
                "note": DISCLOSURE_RULES[s]["note"],
                "rules_version": DISCLOSURE_RULES["version"]}
            for s in stores
        }
        book = Audiobook(
            book_id="ab_" + uuid.uuid4().hex[:8],
            title=title, author=author, chapters=rendered,
            master_path=mastered,
            duration_s=_audio_duration(Path(mastered)),
            disclosure=disclosure, target_stores=stores,
            created_at=time.time())
        if store is not None:
            try:
                store.save(book)
            except Exception:  # noqa: BLE001
                pass
        lufs_tgt = tgt.get("lufs", -16.0)
        note = (f"{len(rendered)} chapters, {book.duration_s / 3600:.1f}h, "
                f"mastered to {lufs_tgt} LUFS ({stores[0]} target), "
                f"disclosure attached for {', '.join(stores)} "
                f"(rules v{DISCLOSURE_RULES['version']})")
        if compliance.get("ok"):
            note += f" — compliance {stores[0]}: {compliance['verdict']}"
        if pacing.get("ok") and pacing.get("verdict") != "PASS":
            note += f" — pacing {pacing['verdict']}: {pacing['note']}"
        if mp3_path:
            note += f" — MP3 delivery: {mp3_path}"
        if blocked:
            note += (" — skipped: " + "; ".join(
                f"{b['store']} ({b['reason'][:60]}…)" for b in blocked))
        return {"ok": True, "audiobook": book, "note": note,
                "compliance": compliance if compliance.get("ok") else None,
                "pacing": pacing if pacing.get("ok") else None,
                "mp3_path": mp3_path, "blocked_stores": blocked}
    except Exception as exc:  # noqa: BLE001
        _log.warning("epub_to_audiobook failed: %s", exc)
        return {"ok": False, "reason": f"audiobook build failed: {exc}"}


# ── store ──────────────────────────────────────────────────────────────────


def _default_db() -> str:
    base = Path(os.environ.get("NOMORALS_HOME", Path.home() / ".nomorals"))
    base = base / "audio" / "audiobooks"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return str(base / "audiobooks.db")


class AudiobookStore:
    """SQLite persistence for audiobook runs. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        try:
            p = Path(db_path or _default_db())
            self._db = sqlite3.connect(str(p), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS audiobooks(
                book_id TEXT PRIMARY KEY, title TEXT, author TEXT,
                master_path TEXT, duration_s REAL, target_stores TEXT,
                disclosure TEXT, created_at REAL)""")
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            _log.warning("AudiobookStore init failed: %s", exc)
            self._db = None

    def save(self, book: Audiobook) -> bool:
        try:
            if self._db is None or not book or not book.book_id:
                return False
            import json
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO audiobooks VALUES (?,?,?,?,?,?,?,?)",
                    (book.book_id, book.title, book.author, book.master_path,
                     book.duration_s, ",".join(book.target_stores),
                     json.dumps(book.disclosure), book.created_at))
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT book_id, title, author, master_path, duration_s,"
                " target_stores, created_at FROM audiobooks"
                " ORDER BY created_at DESC LIMIT ?", (max(1, limit),))
            return [dict(r) for r in rows.fetchall()]
        except Exception:  # noqa: BLE001
            return []

    def get(self, book_id: str) -> dict[str, Any] | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM audiobooks WHERE book_id = ?",
                ((book_id or "").strip(),)).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            return None


# ── chat ───────────────────────────────────────────────────────────────────


def _get_store() -> AudiobookStore:
    return AudiobookStore()


def _usage() -> str:
    return ("🎧 /audiobook make <epub> [voices...] [stores...] — EPUB in, "
            "chaptered mastered audiobook out.\n"
            "🎧 /audiobook status — list produced audiobooks.\n"
            "🎧 /audiobook stores — disclosure + mastering rules per store.\n"
            "🎧 /audiobook check <book_id> [store] — PASS/WARN/FAIL the master.\n"
            "Voices: narrator=<ref audio> [chapter:2=<ref>] · stores: acx, "
            "spotify, kobo, authors_republic. AI disclosure is attached automatically "
            f"(rules v{DISCLOSURE_RULES['version']}, checked {DISCLOSURE_RULES['checked']}).")


def _fmt_rules_card() -> str:
    lines = [f"🎧 store rules (v{DISCLOSURE_RULES['version']}, "
             f"checked {DISCLOSURE_RULES['checked']}):"]
    for s in RULED_STORES:
        r = store_rules(s)
        mark = "✅" if r.get("allowed") else "⛔"
        tgt = r.get("target") or {}
        tstr = (f" — {tgt['lufs']} LUFS" if tgt.get("lufs") else "")
        lines.append(f"{mark} {s}{tstr}")
        if r.get("allowed"):
            lines.append(f"   disclosure: {r.get('disclosure_text')}")
        else:
            lines.append(f"   blocked: {r.get('block_reason')}")
        if r.get("note"):
            lines.append(f"   ({r['note']})")
    return "\n".join(lines)


def control_audiobook(tail: str, context=None, chat=None, **kwargs) -> str:
    """Chat entry point. Owner-only (wired at dispatch). Never raises."""
    try:
        tail = (tail or "").strip()
        store = _get_store()
        if not tail or tail.lower() in ("help", "?"):
            return _usage()
        low = tail.lower()
        if low.startswith("status"):
            books = store.list()
            if not books:
                return "🎧 no audiobooks produced yet."
            lines = ["🎧 audiobooks:"]
            for b in books:
                hrs = (b.get("duration_s") or 0) / 3600
                lines.append(f"• {b.get('title','?')} — {hrs:.1f}h "
                             f"→ {b.get('target_stores','')} "
                             f"({b.get('book_id','')})")
            return "\n".join(lines)
        if low.startswith("stores"):
            return _fmt_rules_card()
        if low.startswith("check"):
            parts = tail.split()
            if len(parts) < 2:
                return "🎧 check which? /audiobook check <book_id> [store]"
            book = store.get(parts[1])
            if not book:
                return f"🎧 no audiobook {parts[1]!r} — /audiobook status."
            tgt_store = (parts[2] if len(parts) > 2 else
                         (book.get("target_stores") or "spotify").split(",")[0])
            res = check_compliance(book.get("master_path", ""), tgt_store)
            if not res.get("ok"):
                return f"🎧 couldn't check it — {res.get('reason')}"
            lines = [f"🎧 compliance [{res['store']}]: {res['verdict']}"]
            for c in res["checks"]:
                mark = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(
                    c["status"], "•")
                lines.append(f"  {mark} {c['metric']}: {c['value']} "
                             f"(target {c['target']})")
            return "\n".join(lines)
        if low.startswith("make "):
            rest = tail[5:].strip()
            parts = rest.split()
            epub = parts[0] if parts else ""
            cast: dict[str, str] = {}
            stores: list[str] = []
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    if k in ("narrator",) or k.startswith("chapter:"):
                        cast[k] = v.strip()
                elif p.strip().lower() in RULED_STORES:
                    stores.append(p.strip().lower())
            res = epub_to_audiobook(epub, cast or None, stores or None,
                                    store=store)
            if res.get("ok"):
                book: Audiobook = res["audiobook"]
                return (f"🎧 audiobook ready: {book.title}\n"
                        f"📖 {len(book.chapters)} chapters · "
                        f"⏱️ {book.duration_s / 3600:.1f}h · mastered "
                        f"-16 LUFS\n📦 {res['note']}")
            return f"🎧 couldn't make the audiobook — {res.get('reason', '?')}"
        return _usage()
    except Exception as exc:  # noqa: BLE001
        return f"🎧 audiobook hiccup: {exc}"
