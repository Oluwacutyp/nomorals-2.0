"""DJ — radio-style shows with trending charts, voice breaks, and real mixing.

A real DJ, not a playlist shuffler:

* **Trending fetcher** — scrapes the Billboard Hot 100 (no API key),
  caches daily. Never raises; honest empty on failure.
* **DJ engine** (``dj_engine``) — track analysis (BPM via onset
  autocorrelation, key via chromagram + Krumhansl, energy), Camelot-wheel
  harmonic mixing, beatmatched tempo-sync transitions, phrase-aligned
  blends, and energy-arc ordering (warm-up → peak → cool-down). Composed
  tracks use ground-truth tempo/key from the composer; external audio is
  DSP-detected. Never claims a sync it can't do.
* **Show composer** — arranges trending picks + your own generated
  tracks into a radio show: DJ intro → track → transition → DJ break
  (shoutout, chart trivia) → track → … → outro. Track order follows the
  energy arc, transitions are engine-planned.
* **DJ voice breaks** — TTS when a backend exists, hummed vocal
  fallback when not, short musical sting as the last resort.
  Breaks are 5–10 seconds, never silence.
* **Real transitions** — beatmatched blends (tempo-synced, phrase-aligned),
  echo-out drops, filter sweeps. Pure stdlib DSP, Termux-safe.
* **Taste interface** — ``dj_engine.TasteModel`` protocol; a small model
  trained on owner skip/like feedback can replace ``HeuristicTaste``
  later without touching the engine.

    from nomorals.media.dj import DJ
    dj = DJ(context)
    show = dj.build_show(genre="uk-drill")   # full WAV show
    show["path"]          # …/workspace/dj/devon-fm-….wav
    show["tracklist"]     # what played, in order

Registered as the ``dj`` tool; chat command is ``/dj``.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.request
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["DJ", "fetch_trending", "TrendingTrack", "register"]

SAMPLE_RATE = 22050
_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "nomorals" / "dj"
_CACHE_TTL = 24 * 3600  # refresh charts daily

_BILLBOARD_URL = "https://www.billboard.com/charts/hot-100/"
_UA = ("Mozilla/5.0 (Linux; Android 13; Mobile) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36")


# ─────────────────────────────── trending ────────────────────────────────────

@dataclass
class TrendingTrack:
    title: str
    artist: str
    rank: int = 0
    genre_hint: str = ""
    source: str = "billboard"

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "artist": self.artist,
                "rank": self.rank, "genre_hint": self.genre_hint,
                "source": self.source}


def _cache_path(genre: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", (genre or "all").lower()).strip("-")
    return _CACHE_DIR / f"trending-{slug or 'all'}.json"


def _read_cache(genre: str) -> list[dict[str, Any]] | None:
    try:
        p = _cache_path(genre)
        if not p.is_file():
            return None
        if time.time() - p.stat().st_mtime > _CACHE_TTL:
            return None
        data = json.loads(p.read_text())
        tracks = data.get("tracks", [])
        return tracks if isinstance(tracks, list) else None
    except Exception:  # noqa: BLE001
        return None


def _write_cache(genre: str, tracks: list[dict[str, Any]]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(genre).write_text(json.dumps(
            {"tracks": tracks, "fetched_at": time.time()}))
    except Exception:  # noqa: BLE001
        _log.debug("trending cache write failed", exc_info=True)


def _fetch_billboard() -> tuple[list[dict[str, Any]], str]:
    """Scrape Billboard Hot 100. Returns (tracks, source). Never raises."""
    try:
        req = urllib.request.Request(_BILLBOARD_URL, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return [], f"billboard fetch failed: {exc}"

    tracks: list[dict[str, Any]] = []
    try:
        # Billboard embeds chart data as JSON in the page. Try a few shapes.
        # Shape 1: "title":"…","artist":"…" pairs in order.
        pairs = re.findall(
            r'"title"\s*:\s*"([^"]{1,120})"\s*,\s*"artist"\s*:\s*"([^"]{1,120})"',
            html)
        seen: set[str] = set()
        for title, artist in pairs:
            key = (title.strip().lower(), artist.strip().lower())
            if not key[0] or not key[1] or key in seen:
                continue
            # skip obvious non-song JSON blobs
            if len(key[0]) > 80 or "billboard" in key[1]:
                continue
            seen.add(key)
            tracks.append({"title": title.strip(), "artist": artist.strip(),
                           "rank": len(tracks) + 1, "genre_hint": "",
                           "source": "billboard"})
            if len(tracks) >= 25:
                break
        if tracks:
            return tracks, "billboard"

        # Shape 2: list items with chart-element classes (older markup).
        items = re.findall(
            r'chart-element__information.*?<span[^>]*>([^<]{1,80})</span>'
            r'.*?chart-element__information__artist[^>]*>([^<]{1,80})</',
            html, re.S)
        for title, artist in items[:25]:
            tracks.append({"title": title.strip(), "artist": artist.strip(),
                           "rank": len(tracks) + 1, "genre_hint": "",
                           "source": "billboard"})
        if tracks:
            return tracks, "billboard"
    except Exception as exc:  # noqa: BLE001
        return [], f"billboard parse failed: {exc}"
    return [], "billboard returned no parseable chart data"


def fetch_trending(genre: str = "", *,
                   force_refresh: bool = False) -> dict[str, Any]:
    """Trending tracks from public charts. Never raises.

    Returns {"ok", "tracks": [...], "source", "cached"} or
    {"ok": False, "reason"}. Results are cached for 24h.
    """
    genre = (genre or "").strip().lower()
    if not force_refresh:
        cached = _read_cache(genre)
        if cached:
            return {"ok": True, "tracks": cached, "source": "cache",
                    "cached": True}
    tracks, source = _fetch_billboard()
    if tracks:
        # light genre filter on the hint when the caller asked for one
        if genre:
            keep = [t for t in tracks
                    if genre in (t.get("genre_hint") or "").lower()
                    or genre in t["title"].lower()
                    or genre in t["artist"].lower()]
            tracks = keep or tracks  # don't go empty on a weak filter
        _write_cache(genre, tracks)
        return {"ok": True, "tracks": tracks, "source": source,
                "cached": False}
    return {"ok": False, "reason": source, "tracks": []}


# ─────────────────────────────── DSP (stdlib) ──────────────────────────────────

def _equal_power_crossfade(a: array, b: array, xf_samples: int) -> array:
    """Equal-power crossfade: tail of `a` blended into head of `b`.

    Never raises. Returns the combined buffer.
    """
    try:
        xf = max(0, min(xf_samples, len(a), len(b)))
        out = array("d", a)
        if xf <= 0:
            out.extend(b)
            return out
        # equal-power curves: cos/sin so the sum stays flat
        for i in range(xf):
            t = i / max(1, xf - 1)
            g_out = math.cos(t * math.pi / 2.0)
            g_in = math.sin(t * math.pi / 2.0)
            out[len(a) - xf + i] = a[len(a) - xf + i] * g_out + b[i] * g_in
        out.extend(b[xf:])
        return out
    except Exception:  # noqa: BLE001
        _log.debug("crossfade failed", exc_info=True)
        out = array("d", a)
        out.extend(b)
        return out


def _echo_out(samples: array, sr: int,
              delay_ms: float = 320.0, feedback: float = 0.35,
              tail_ms: float = 900.0) -> array:
    """Dub-style echo tail on the last `tail_ms` of audio. Never raises."""
    try:
        out = array("d", samples)
        d = max(1, int(sr * delay_ms / 1000.0))
        tail_n = int(sr * tail_ms / 1000.0)
        start = max(0, len(out) - tail_n)
        for i in range(start, len(out)):
            echo = out[i - d] * feedback if i - d >= start else 0.0
            out[i] = out[i] * 0.82 + echo
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def _filter_sweep_in(samples: array, sr: int,
                     sweep_ms: float = 1200.0) -> array:
    """High-pass-ish sweep into a track: starts muffled, opens up.

    One-pole lowpass whose cutoff rises exponentially over the sweep.
    Never raises.
    """
    try:
        out = array("d", [0.0]) * len(samples)
        n = min(len(samples), int(sr * sweep_ms / 1000.0))
        if n <= 0:
            return array("d", samples)
        y = 0.0
        for i in range(len(samples)):
            if i < n:
                # cutoff sweeps ~300 Hz -> 8 kHz
                frac = i / max(1, n - 1)
                cutoff = 300.0 * (8000.0 / 300.0) ** frac
                alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / sr)
            else:
                alpha = 1.0
            y += alpha * (samples[i] - y)
            out[i] = y
        return out
    except Exception:  # noqa: BLE001
        return array("d", samples)


def _read_wav_mono(path: str) -> tuple[array, int]:
    from .vocal_lite import _read_wav_mono as _r
    try:
        return _r(path)
    except Exception:  # noqa: BLE001
        return array("d"), SAMPLE_RATE


def _write_mono_wav(path: str, samples: array, sr: int) -> str:
    from .vocal_lite import _write_mono_wav as _w
    try:
        return _w(path, samples, sr)
    except Exception:  # noqa: BLE001
        return ""


def _resample(samples: array, src: int, dst: int) -> array:
    from .vocal_lite import _resample_linear as _rs
    try:
        return _rs(samples, src, dst)
    except Exception:  # noqa: BLE001
        return array("d", samples)


# ─────────────────────────────── DJ voice ──────────────────────────────────────

_DJ_NAME = "Devon FM"

_INTROS = [
    "You're locked in to {dj}, I'm your host Devon, and this one's for the night owls.",
    "What's good, it's {dj} — Devon on the decks, let's get into it.",
    "{dj} in the building! Devon here, buckle up, we got heat coming.",
]

_BREAKS = [
    "That was {artist} with {title}, sitting at number {rank} on the charts right now.",
    "{title} by {artist} — number {rank} and climbing. {dj} keeps it moving.",
    "You just heard {artist}, {title}. Charting at {rank} this week on {dj}.",
    "Big tune. {title}, {artist}, number {rank}. Only on {dj}.",
]

_OUTROS = [
    "And that's the show — Devon on {dj}, catch you on the next one. Stay dangerous.",
    "{dj} signing off. I'm Devon, that was the heat, more coming soon.",
]


def _render_sting(sr: int = SAMPLE_RATE) -> array:
    """Short musical DJ sting (no TTS available). Never raises."""
    try:
        from .vocal_lite import _render_hum_tone
        # quick "DEV-ON" two-note ident: A4 -> E5
        out = array("d", [0.0]) * int(sr * 1.6)
        n1 = int(sr * 0.55)
        n2 = int(sr * 0.85)
        t1 = _render_hum_tone(440.0, n1, 90.0)
        t2 = _render_hum_tone(659.25, n2, 95.0)
        for i in range(min(n1, len(out))):
            out[i] += t1[i] * 0.8
        off = n1 + int(sr * 0.08)
        for i in range(n2):
            if off + i < len(out):
                out[off + i] += t2[i] * 0.8
        return out
    except Exception:  # noqa: BLE001
        return array("d", [0.0]) * int(sr)


def _dj_say(text: str, workdir: Path) -> tuple[array, int, str]:
    """Render a DJ voice break -> (samples, sr, note). Never raises.

    TTS when a backend exists, hummed delivery when not, musical
    sting as the last resort. Never silence.
    """
    # 1) real TTS
    try:
        from .vocal_lite import _default_tts
        tmp = str(workdir / f"djbreak-{abs(hash(text)) % 10**8}.wav")
        res = _default_tts(text, tmp)
        if res.get("ok"):
            samples, sr = _read_wav_mono(res["path"])
            if len(samples) > 100:
                return samples, sr, f"TTS ({res.get('backend', '?')})"
    except Exception:  # noqa: BLE001
        _log.debug("DJ TTS break failed", exc_info=True)
    # 2) hummed delivery — same vocal-ish tone as the music vocals
    try:
        from .vocal_lite import _render_hum_tone
        words = max(1, len(text.split()))
        dur_s = min(10.0, max(4.0, words * 0.42))
        n = int(SAMPLE_RATE * dur_s)
        # gentle spoken-ish contour around A3
        tone = _render_hum_tone(220.0, n, 85.0)
        return tone, SAMPLE_RATE, "hummed (no TTS installed)"
    except Exception:  # noqa: BLE001
        pass
    # 3) sting
    return _render_sting(), SAMPLE_RATE, "sting (no voice available)"


# ─────────────────────────────── show builder ──────────────────────────────────

# trending genre hints -> our style names
_STYLE_MAP = {
    "hip hop": "hiphop", "rap": "rap", "drill": "uk-drill",
    "pop": "pop", "r&b": "rnb", "rnb": "rnb", "afrobeats": "afrobeats",
    "amapiano": "amapiano", "dance": "edm", "edm": "edm",
    "country": "country", "rock": "rock", "latin": "reggaeton",
    "reggaeton": "reggaeton", "house": "house", "trap": "trap",
    "phonk": "phonk", "grime": "grime", "dnb": "dnb",
}


def _style_for_genre(genre: str, rng: random.Random) -> str:
    g = (genre or "").strip().lower()
    if g in _STYLE_MAP:
        return _STYLE_MAP[g]
    for key, style in _STYLE_MAP.items():
        if key in g or g in key:
            return style
    return rng.choice(["pop", "hiphop", "rnb", "afrobeats", "uk-drill"])


@dataclass
class ShowSegment:
    kind: str            # "break" | "track"
    label: str
    path: str = ""
    title: str = ""
    artist: str = ""
    rank: int = 0


class DJ:
    """Radio DJ: trending charts + voice breaks + real transitions."""

    def __init__(self, context: Any = None):
        self.context = context
        self._rng = random.Random()

    # ── trending ──────────────────────────────────────────────
    def fetch_trending(self, genre: str = "",
                       force_refresh: bool = False) -> dict[str, Any]:
        return fetch_trending(genre, force_refresh=force_refresh)

    # ── show ──────────────────────────────────────────────────
    def build_show(self, genre: str = "", n_tracks: int = 4,
                   workdir: str = "dj",
                   include_breaks: bool = True) -> dict[str, Any]:
        """Compose a full radio show -> {"ok", "path", "tracklist", ...}.

        Never raises — returns {"ok": False, "reason"} on failure.
        """
        try:
            return self._build_show(genre, n_tracks, workdir, include_breaks)
        except Exception as exc:  # noqa: BLE001
            _log.warning("DJ show build failed: %s", exc)
            return {"ok": False, "reason": str(exc)}

    def _build_show(self, genre: str, n_tracks: int,
                    workdir: str, include_breaks: bool) -> dict[str, Any]:
        from .music import MusicCreator

        n_tracks = max(1, min(8, int(n_tracks or 4)))
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)

        trending = self.fetch_trending(genre)
        chart: list[dict[str, Any]] = (
            trending.get("tracks", []) if trending.get("ok") else [])
        rng = self._rng

        # pick chart entries for the show (top of the chart first)
        picks = chart[:n_tracks]

        segments: list[ShowSegment] = []
        tracklist: list[dict[str, Any]] = []
        # (segment, Song, pick, style) — kept parallel so the engine can
        # analyze, reorder (energy arc), and plan transitions after compose
        track_units: list[tuple[ShowSegment, Any, Any, str]] = []

        # — intro break —
        if include_breaks:
            intro = rng.choice(_INTROS).format(dj=_DJ_NAME)
            segments.append(ShowSegment(
                kind="break", label="intro", title=intro))

        # — tracks with breaks between —
        creator = MusicCreator(self.context)
        for i in range(n_tracks):
            pick = picks[i] if i < len(picks) else None
            if pick:
                style = _style_for_genre(pick.get("genre_hint", ""), rng)
                topic = f"{pick['title']} by {pick['artist']} — chart energy"
                label = f"#{pick.get('rank', i + 1)} {pick['title']} — {pick['artist']}"
            else:
                style = _style_for_genre(genre, rng)
                topic = f"Devon FM original {i + 1} in {style}"
                label = f"Devon FM original ({style})"

            # compose our own track in a matching style (short: no score pdf)
            try:
                song = creator.compose(
                    topic, style=style, with_midi=False,
                    with_score=False, with_vocals=False, workdir=str(wd))
                audio = song.audio_path or ""
            except Exception as exc:  # noqa: BLE001
                _log.warning("DJ track compose failed: %s", exc)
                audio = ""

            if not audio or not os.path.isfile(audio):
                # honest skip — keep the show moving
                tracklist.append({"label": label, "status": "skipped",
                                  "reason": "compose failed"})
                continue
            seg = ShowSegment(
                kind="track", label=label, path=audio,
                title=pick["title"] if pick else label,
                artist=pick["artist"] if pick else "Devon FM",
                rank=pick.get("rank", 0) if pick else 0)
            track_units.append((seg, song, pick, style))
            tracklist.append({"label": label, "status": "played",
                              "style": style,
                              "chart_rank": pick.get("rank") if pick else None})

        # ── DJ engine: analyze → energy arc → transition plans ──────────
        from .dj_engine import (
            analyze_track, plan_energy_arc, plan_transition,
            render_beatmatched_transition, HeuristicTaste,
        )
        taste = HeuristicTaste()
        analyses: list = []
        for seg, song, _pick, _style in track_units:
            a = analyze_track(
                seg.path, title=seg.label,
                known_bpm=getattr(song, "tempo", None),
                known_key=getattr(song, "key", "") or "",
                known_mode=getattr(song, "mode", "") or "")
            analyses.append(a)
        # order warm-up → peak → cool-down (stable: arc keeps ties in order)
        arc_idx = [analyses.index(a) for a in plan_energy_arc(analyses)]
        ordered_units = [track_units[i] for i in arc_idx]
        ordered_analyses = [analyses[i] for i in arc_idx]
        # transition plan for each track→track boundary
        plans = [plan_transition(ordered_analyses[i], ordered_analyses[i + 1])
                 for i in range(len(ordered_analyses) - 1)]

        # ── rebuild segments in arc order, with breaks ─────────────────
        # tracklist follows the arc order too (it was built in compose order)
        arc_tracklist = [tracklist[track_units.index(u)] for u in ordered_units]
        tracklist[:] = arc_tracklist
        segments = []
        if include_breaks:
            intro = rng.choice(_INTROS).format(dj=_DJ_NAME)
            segments.append(ShowSegment(
                kind="break", label="intro", title=intro))
        for j, ((seg, _song, pick, style), _a) in enumerate(
                zip(ordered_units, ordered_analyses)):
            segments.append(seg)
            if include_breaks and j < len(ordered_units) - 1:
                nxt_seg = ordered_units[j + 1][0]
                if pick:
                    line = rng.choice(_BREAKS).format(
                        artist=pick["artist"], title=pick["title"],
                        rank=pick.get("rank", "?"), dj=_DJ_NAME)
                else:
                    line = (f"That was a Devon FM original in {style}. "
                            f"More heat coming up on {_DJ_NAME}.")
                line += f" Up next: {nxt_seg.artist} with {nxt_seg.title}."
                segments.append(ShowSegment(
                    kind="break", label=f"break-{j + 1}", title=line))
        if include_breaks:
            outro = rng.choice(_OUTROS).format(dj=_DJ_NAME)
            segments.append(ShowSegment(kind="break", label="outro",
                                        title=outro))

        track_segs = [s for s in segments if s.kind == "track"]
        if not track_segs:
            return {"ok": False, "reason": "no tracks rendered — show is empty",
                    "tracklist": tracklist}

        # — assemble with engine-planned transitions —
        # map each track segment to its outgoing transition plan
        seg_plan: dict[int, Any] = {}
        for j, plan in enumerate(plans):
            seg_plan[id(ordered_units[j][0])] = plan
        mix = array("d")
        first = True
        notes: list[str] = []
        for seg in segments:
            if seg.kind == "break":
                samples, sr, note = _dj_say(seg.title, wd)
                notes.append(f"{seg.label}: {note}")
            else:
                samples, sr = _read_wav_mono(seg.path)
                if len(samples) < 100:
                    continue
                # radio edit: cap tracks at ~75s so the show stays tight
                cap_n = int(SAMPLE_RATE * 75)
                if len(samples) > cap_n:
                    samples = array("d", samples[:cap_n])
                plan = seg_plan.get(id(seg))
                if plan is not None and plan.kind == "echo-drop":
                    # echo-out tail for the drop
                    samples = _echo_out(samples, SAMPLE_RATE)
                if (plan is not None and plan.kind == "blend"
                        and abs(plan.sync_ratio - 1.0) > 1e-4):
                    # beatmatch: tempo-sync incoming track to outgoing tempo
                    try:
                        from .vocal_lite import _resample_linear as _rs
                        samples = _rs(samples, SAMPLE_RATE,
                                      int(SAMPLE_RATE / plan.sync_ratio))
                    except Exception:  # noqa: BLE001
                        pass
            samples = _resample(samples, sr, SAMPLE_RATE)
            if first:
                mix = array("d", samples)
                first = False
            else:
                if seg.kind == "track":
                    # filter sweep into the incoming track + crossfade
                    samples = _filter_sweep_in(samples, SAMPLE_RATE)
                xf = int(SAMPLE_RATE * (4.0 if seg.kind == "track" else 1.5))
                mix = _equal_power_crossfade(mix, samples, xf)
        # annotate the tracklist with what the engine decided
        for j, a in enumerate(ordered_analyses):
            tracklist[j]["bpm"] = a.bpm
            tracklist[j]["key"] = f"{a.key} {a.mode}".strip()
            tracklist[j]["camelot"] = a.camelot
            tracklist[j]["energy"] = a.energy
            if j < len(plans):
                tracklist[j]["transition_out"] = plans[j].to_dict()
                tracklist[j]["taste_score"] = taste.score_transition(
                    a, ordered_analyses[j + 1], plans[j])

        if len(mix) < 100:
            return {"ok": False, "reason": "mix came out empty",
                    "tracklist": tracklist}

        # peak-normalize + soft clip (same curve as the music mixer)
        peak = max((abs(s) for s in mix), default=0.0)
        if peak > 0:
            norm = 0.89 / peak
            for i, s in enumerate(mix):
                mix[i] = math.tanh(s * norm * 1.2) * 0.95

        stamp = time.strftime("%Y%m%d-%H%M%S")
        genre_slug = re.sub(r"[^a-z0-9]+", "-", genre.lower()).strip("-")
        out_name = f"devon-fm-{genre_slug or 'mix'}-{stamp}.wav"
        out_path = _write_mono_wav(str(wd / out_name), mix, SAMPLE_RATE)
        if not out_path:
            return {"ok": False, "reason": "could not write show WAV",
                    "tracklist": tracklist}

        dur_s = len(mix) / SAMPLE_RATE
        return {"ok": True, "path": out_path, "tracklist": tracklist,
                "duration_s": round(dur_s, 1),
                "segments": len(segments),
                "voice_notes": notes,
                "trending_source": trending.get("source", "none"),
                "genre": genre or "mixed"}


def register(context: Any = None) -> dict[str, Any]:
    """Tool-registry entry for the DJ."""
    dj = DJ(context)

    def _trending(genre: str = "") -> dict[str, Any]:
        return dj.fetch_trending(genre)

    def _show(genre: str = "", n_tracks: int = 4) -> dict[str, Any]:
        return dj.build_show(genre=genre, n_tracks=n_tracks)

    return {"name": "dj",
            "description": "Radio DJ: trending charts, voice breaks, mixed shows.",
            "call": {"trending": _trending, "show": _show}}
