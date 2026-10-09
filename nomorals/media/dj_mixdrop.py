"""DJ mix-drop mode — full beatmatched mixes rendered as one file.

The user's answer to the live-streaming constraint: "she can also just
drop mixes at once then mix the next and drop." So mix-drop renders a
complete continuous mix (real fetched tracks, real BPM/key detection,
real club-style beatmatched crossfades across the whole file) and drops
it in the chat. Then she renders the next one.

Unlike the live performer (dj_live.py — voice-note delivery, radio-style
talk-overs), mix-drop is a mixtape DJ: one seamless audio file.

Pipeline: fetch real tracks -> download -> analyze (dj_engine) ->
energy-arc order -> plan transitions -> render beatmatched mix ->
drop WAV in chat.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_SR = 22050
PLAY_S = 120.0      # seconds of each track played in the mix
CUE_SKIP_S = 8.0    # skip past cold intros
MAX_TRACKS = 8
MAX_MIX_MINUTES = 25


@dataclass
class MixFetch:
    query: str
    path: str = ""          # local WAV ready for analysis
    title: str = ""
    artist: str = ""
    source: str = ""        # how it was fetched
    error: str = ""         # set when the fetch failed

    @property
    def ok(self) -> bool:
        return bool(self.path) and not self.error


# ── fetching real tracks ─────────────────────────────────────────────

def _have_ytdlp() -> tuple[bool, bool]:
    """(has_python_module, has_cli)."""
    try:
        import yt_dlp  # noqa: F401
        mod = True
    except ImportError:
        mod = False
    return mod, shutil.which("yt-dlp") is not None


def fetch_track_audio(query: str, workdir: str | Path,
                      context: Any = None) -> MixFetch:
    """Fetch one real track and return a local WAV. Never raises.

    Uses SourceResolver for search, then downloads via yt-dlp.
    Spotify/DRM -> honest error, never a substitute.
    """
    from .resolver import SourceResolver
    out = MixFetch(query=query)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    res = SourceResolver(context).resolve(query)
    if not res.ok:
        out.error = f"couldn't find '{query}': " + \
            "; ".join(res.errors[:2] or ["no results"])
        return out
    out.title = res.title or query
    out.artist = res.artist or ""
    out.source = res.source_name or res.kind

    if res.kind == "file":
        wav = _to_wav(res.path_or_url, wd)
        if wav:
            out.path = wav
        else:
            out.error = f"couldn't read local file {res.path_or_url}"
        return out
    if not res.downloadable:
        out.error = (f"'{out.title}' is {res.kind} — DRM, can't download "
                     "for mixing")
        return out
    # youtube / soundcloud / url -> download with yt-dlp
    url = res.path_or_url
    dl_path = _ytdlp_download(url, wd)
    if not dl_path:
        out.error = f"download failed for '{out.title}' ({url})"
        return out
    wav = _to_wav(dl_path, wd)
    if wav:
        out.path = wav
    else:
        out.error = f"couldn't convert '{out.title}' to WAV (need ffmpeg)"
    return out


def _ytdlp_download(url: str, wd: Path) -> str:
    """Download best audio to wd. Returns path or ''."""
    mod, cli = _have_ytdlp()
    if not mod and not cli:
        _log.warning("mix-drop: no yt-dlp (module or CLI)")
        return ""
    tmpl = str(wd / "dl_%(id)s.%(ext)s")
    if mod:
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True,
                    "format": "bestaudio/best",
                    "outtmpl": tmpl,
                    "noplaylist": True}
            try:
                from .cookies import ytdlp_cookie_opts as _co
                opts.update(_co())
            except Exception:  # noqa: BLE001
                pass
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            if info:
                fid = info.get("id", "")
                for f in wd.glob(f"dl_{fid}.*"):
                    return str(f)
        except Exception as exc:  # noqa: BLE001
            _log.warning("mix-drop yt-dlp module failed: %s", exc)
            return ""
    # CLI fallback
    try:
        proc = subprocess.run(
            ["yt-dlp", "-f", "bestaudio/best", "--no-playlist",
             "-o", tmpl, url],
            capture_output=True, timeout=300)
        if proc.returncode == 0:
            files = sorted(wd.glob("dl_*.*"),
                           key=lambda p: p.stat().st_mtime)
            if files:
                return str(files[-1])
    except Exception as exc:  # noqa: BLE001
        _log.warning("mix-drop yt-dlp CLI failed: %s", exc)
    return ""


def _to_wav(src: str, wd: Path) -> str:
    """Convert any audio to 22050Hz mono WAV. Returns path or ''."""
    if not src or not os.path.exists(src):
        return ""
    if src.lower().endswith(".wav"):
        # normalize sample rate/channels anyway
        pass
    out = str(wd / f"mix_{abs(hash(src)) % 10**8}.wav")
    ff = shutil.which("ffmpeg")
    if not ff:
        # already a usable wav? check quickly
        try:
            import wave
            with wave.open(src, "rb") as f:
                if f.getframerate() == _SR and f.getnchannels() == 1:
                    return src
        except Exception:  # noqa: BLE001
            pass
        return ""
    try:
        proc = subprocess.run(
            [ff, "-y", "-v", "error", "-i", src,
             "-ar", str(_SR), "-ac", "1", "-sample_fmt", "s16", out],
            capture_output=True, timeout=180)
        if proc.returncode == 0 and os.path.exists(out):
            return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("mix-drop ffmpeg convert failed: %s", exc)
    return ""


# ── rendering the mix ────────────────────────────────────────────────

def _read_segment(path: str, skip_s: float, play_s: float) -> array:
    """Read [skip_s, skip_s+play_s) as mono float samples."""
    try:
        import wave
        import struct
        with wave.open(path, "rb") as f:
            sr = f.getframerate()
            ch = f.getnchannels()
            sw = f.getsampwidth()
            n = f.getnframes()
            start = int(sr * skip_s)
            want = int(sr * play_s)
            if start >= n:
                start = 0
            f.setpos(start)
            raw = f.readframes(min(want, n - start))
        if sw == 2:
            vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
            scale = 32768.0
        elif sw == 1:
            return array("d", [(v - 128) / 128.0
                               for v in bytes(raw)][::ch])
        else:
            return array("d")
        if ch > 1:
            vals = vals[::ch]
        return array("d", [v / scale for v in vals])
    except Exception:  # noqa: BLE001
        return array("d")


def _write_wav(path: str, samples: array, sr: int = _SR) -> str:
    try:
        import wave
        import struct
        clipped = array("d", (max(-1.0, min(1.0, s)) for s in samples))
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(struct.pack("<%dh" % len(clipped),
                                     *(int(s * 32767)
                                       for s in clipped)))
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("mix-drop write failed: %s", exc)
        return ""


def _render_echo_drop(mix: array, out_samples: array, in_samples: array,
                      out_trk: Any, in_trk: Any) -> array:
    """Echo-out tail of outgoing, then drop incoming on the one."""
    out = array("d", mix)
    # simple echo tail: last 2s of outgoing decayed x3
    sr = _SR
    tail_n = min(len(out_samples), sr * 2)
    tail = array("d", out_samples[-tail_n:])
    for rep in range(1, 4):
        g = 0.5 ** rep
        at = len(out) - tail_n + rep * (tail_n // 3)
        for i, s in enumerate(tail[::2]):
            idx = at + i * 2
            if idx < len(out):
                out[idx] += s * g * 0.5
    out.extend(in_samples)
    return out


def render_mixdrop(fetches: list[MixFetch], workdir: str | Path,
                   play_s: float = PLAY_S,
                   mix_name: str = "") -> dict[str, Any]:
    """Render a full continuous mix from fetched tracks. Never raises.

    Returns {"ok", "path", "tracklist", "duration_s", "transitions",
    "skipped"}.
    """
    from . import dj_engine as de
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    good = [f for f in fetches if f.ok]
    skipped = [{"query": f.query, "error": f.error}
               for f in fetches if not f.ok]
    if not good:
        return {"ok": False, "reason": "no tracks fetched",
                "skipped": skipped}
    good = good[:MAX_TRACKS]

    # analyze each real track (DSP detection on fetched audio)
    analyses = []
    for f in good:
        a = de.analyze_track(f.path, title=f.title or f.query)
        a.duration_s = play_s  # we play a segment; plan around that
        analyses.append((f, a))
    # drop tracks with no usable audio
    analyses = [(f, a) for f, a in analyses if a.duration_s > 0]
    if not analyses:
        return {"ok": False, "reason": "no analyzable audio",
                "skipped": skipped}

    ordered = de.plan_energy_arc([a for _, a in analyses])
    # map back to fetches
    by_id = {id(a): f for f, a in analyses}
    pairs = [(by_id[id(a)], a) for a in ordered]

    mix = array("d")
    tracklist: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    prev_samples: array | None = None
    prev_a = None

    for i, (f, a) in enumerate(pairs):
        seg = _read_segment(f.path, CUE_SKIP_S, play_s)
        if not seg:
            skipped.append({"query": f.query,
                            "error": "unreadable audio segment"})
            continue
        if i == 0:
            mix.extend(seg)
        else:
            plan = de.plan_transition(prev_a, a)
            if plan.kind == "blend":
                mix = de.render_beatmatched_transition(
                    mix, prev_samples, seg, plan, prev_a, a)
            elif plan.kind == "echo-drop":
                mix = _render_echo_drop(mix, prev_samples, seg,
                                        prev_a, a)
            else:  # break — 1s gap, honest separation
                mix.extend(array("d", [0.0] * _SR))
                mix.extend(seg)
            transitions.append({"from": prev_a.title, "to": a.title,
                                **plan.to_dict()})
        prev_samples, prev_a = seg, a
        tracklist.append({
            "title": f.title or f.query, "artist": f.artist,
            "bpm": a.bpm, "bpm_source": a.bpm_source,
            "key": f"{a.key} {a.mode}".strip(),
            "camelot": a.camelot,
            "energy": round(a.energy, 2),
            "source": f.source,
        })

    if not mix:
        return {"ok": False, "reason": "render produced no audio",
                "skipped": skipped}
    # hard cap on mix length
    max_n = int(_SR * 60 * MAX_MIX_MINUTES)
    if len(mix) > max_n:
        mix = mix[:max_n]

    name = re.sub(r"[^\w\-]+", "_", mix_name or
                  f"mixdrop_{int(time.time())}")[:40]
    out_path = str(wd / f"{name}.wav")
    if not _write_wav(out_path, mix):
        return {"ok": False, "reason": "couldn't write mix file",
                "skipped": skipped}
    return {"ok": True, "path": out_path,
            "duration_s": round(len(mix) / _SR, 1),
            "tracklist": tracklist, "transitions": transitions,
            "skipped": skipped}


# ── chat control ─────────────────────────────────────────────────────

def _trending_queries(genre: str, n: int, context: Any = None) -> list[str]:
    """Track queries from trending charts."""
    from .dj import fetch_trending
    res = fetch_trending(genre or "")
    tracks = res.get("tracks", []) if res.get("ok") else []
    out = []
    for t in tracks[:n]:
        title, artist = t.get("title", ""), t.get("artist", "")
        q = f"{artist} {title}".strip() or title
        if q:
            out.append(q)
    return out


def control_dj_mix(tail: str, *, chat_key: str = "",
                   deliver_audio: Any = None,
                   deliver_text: Any = None,
                   context: Any = None) -> str:
    """`/dj mix ...` — render a full beatmatched mix and drop it.

    * ``/dj mix <genre>`` — trending tracks in that lane, fetched + mixed
    * ``/dj mix <song1>, <song2>, ...`` — mix exactly these (comma-separated)
    * ``/dj mix`` — trending overall
    """
    queries: list[str] = []
    genre = ""
    if tail:
        parts = [p.strip() for p in tail.split(",")]
        if len(parts) > 1:
            queries = [p for p in parts if p]
        else:
            genre = tail.strip()
            queries = _trending_queries(genre, 5, context)
            if not queries:
                # treat as a single song query
                queries = [genre]
                genre = ""
    else:
        queries = _trending_queries("", 5, context)
    if not queries:
        return ("couldn't find any tracks to mix — charts are empty and "
                "no songs were named")

    wd = Path("dj_mixes") / f"mix_{int(time.time())}"
    if deliver_text:
        try:
            deliver_text(f"🎧 digging {len(queries)} tracks for the mix…")
        except Exception:  # noqa: BLE001
            pass

    fetches = [fetch_track_audio(q, wd, context) for q in queries]
    got = sum(1 for f in fetches if f.ok)
    if got == 0:
        errs = "; ".join(f"{f.query}: {f.error}"
                         for f in fetches[:3])
        return f"couldn't fetch any tracks — {errs}"

    if deliver_text:
        try:
            deliver_text(f"🎧 got {got}/{len(queries)} — analyzing + "
                         "beatmatching…")
        except Exception:  # noqa: BLE001
            pass

    res = render_mixdrop(fetches, wd,
                         mix_name=f"devon_mix_{genre or 'session'}")
    if not res.get("ok"):
        return f"mix failed: {res.get('reason', '?')}"

    lines = [f"🎧 **Devon mixdrop** — {res['duration_s']}s, "
             f"{len(res['tracklist'])} tracks, one continuous blend"]
    for i, t in enumerate(res["tracklist"], 1):
        bpm = f"{t['bpm']} BPM" if t.get("bpm") else "BPM?"
        key = t.get("camelot") or "?"
        lines.append(f"  {i}. {t['title']}"
                     + (f" — {t['artist']}" if t.get("artist") else "")
                     + f" [{bpm}, {key}]")
    for tr in res.get("transitions", []):
        lines.append(f"     ⇄ {tr['from']} → {tr['to']}: "
                     f"{tr['kind']} ({tr.get('reason', '')})")
    for s in res.get("skipped", []):
        lines.append(f"  ⚠ skipped {s['query']}: {s['error']}")
    text = "\n".join(lines)

    if deliver_audio and res.get("path"):
        try:
            deliver_audio(res["path"],
                          f"🎧 Devon mixdrop — {len(res['tracklist'])} "
                          "tracks, beatmatched")
        except Exception as exc:  # noqa: BLE001
            text += f"\n(file drop failed: {exc} — {res['path']})"
    elif res.get("path"):
        text += f"\nmix file: {res['path']}"
    return text
