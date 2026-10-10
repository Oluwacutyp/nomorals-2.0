"""Scene intelligence + generative editing spine tools.

The brain reaches movie analysis, character reels, film downloads, and
generative shots from plain language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCENE_DIR = "scene_intel"


def _out_dir(context: Any, kind: str) -> Path:
    base = Path(getattr(context, "workspace", None) or ".")
    d = base / SCENE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(registry: Any) -> None:
    """Attach scene-intelligence tools to a registry."""
    context = registry.context
    from ..core.policy import Capability

    # ── film sourcing ─────────────────────────────────────────────

    @registry.register(
        "film_search",
        description=("Search movie download sources (NetNaija movies, Nkiri, "
                     "FzMovies) for a film. ('find John Wick 4 to download', "
                     "'get me that Nollywood movie'). Returns candidates."),
        capability=Capability.NET_OUT,
    )
    def film_search(query: str, *, limit: int = 8) -> dict[str, Any]:
        from ..media.film_sources import search_films
        cands = search_films(query, limit=limit)
        return {"ok": True, "candidates": [
            {"source": c.source, "title": c.title, "year": c.year,
             "quality": c.quality, "url": c.url} for c in cands]}

    @registry.register(
        "film_download",
        description=("Download a film from a search candidate (pass the "
                     "candidate dict from film_search). Returns the local "
                     "file path. ('download that movie')."),
        capability=Capability.NET_DOWNLOAD,
    )
    def film_download(candidate: dict[str, Any]) -> dict[str, Any]:
        from ..media.film_sources import (
            FilmCandidate, resolve_film_url)
        from ..media.downloader import MediaDownloader
        cand = FilmCandidate(**{k: v for k, v in candidate.items()
                                if k in FilmCandidate.__dataclass_fields__})
        url = resolve_film_url(cand)
        dl = MediaDownloader()
        out = _out_dir(context, "films")
        path = dl.download(url, out_dir=str(out))
        return {"ok": True, "path": str(path), "url": url}

    # ── scene intelligence ────────────────────────────────────────

    @registry.register(
        "movie_analyze",
        description=("Analyze a movie file: segment into scenes, score every "
                     "scene for coolness (action/emotional/dialogue), track "
                     "characters across the film when vision models are "
                     "installed. ('find the cool scenes in this movie', "
                     "'analyze this film'). Returns top scenes + characters."),
        capability=Capability.FS_READ,
    )
    def movie_analyze(path: str, *, top_n: int = 20,
                      with_tracking: bool = True) -> dict[str, Any]:
        from ..media.scene_intel import analyze_movie, ModelUnavailable
        try:
            a = analyze_movie(path, with_tracking=with_tracking, top_n=top_n)
        except ModelUnavailable as exc:
            # honest: scenes scored, tracking unavailable
            from ..media.scene_intel import segment_movie
            from ..media.scene_intel.score import score_segments
            seg = segment_movie(path)
            bounds = [(s.start, s.end) for s in seg.segments
                      if s.kind == "scene"]
            scored = score_segments(path, bounds)[:top_n]
            return {"ok": True, "tracking": False,
                    "tracking_reason": str(exc),
                    "top_scenes": [
                        {"start": s.start, "end": s.end,
                         "score": s.score, "label": s.label}
                        for s in scored]}
        return {"ok": True, "tracking": True,
                "backend": a.segmentation.backend,
                "duration_s": a.segmentation.duration,
                "top_scenes": [
                    {"start": s.start, "end": s.end, "score": s.score,
                     "label": s.label} for s in a.scored[:top_n]],
                "characters": [
                    {"id": c.char_id,
                     "screen_s": c.total_screen_s,
                     "scenes": len(a.profiles[c.char_id].best_scenes)}
                    for c in a.characters]}

    @registry.register(
        "character_reel",
        description=("Pull a character's best scenes from an analyzed movie "
                     "as an edit list. Needs the movie analyzed first "
                     "(movie_analyze with tracking). ('give me all of "
                     "char_0's best moments', 'make a reel of the villain'). "
                     "Returns scene list ready for the editor."),
        capability=Capability.FS_READ,
    )
    def character_reel(path: str, char_id: str, *,
                       top_n: int = 10) -> dict[str, Any]:
        from ..media.scene_intel import (
            analyze_movie, character_reel as _reel)
        a = analyze_movie(path, with_tracking=True, top_n=50)
        scenes = _reel(a.profiles, char_id, top_n=top_n)
        if not scenes and char_id not in a.profiles:
            return {"ok": False,
                    "reason": f"unknown character {char_id!r} — "
                              f"known: {sorted(a.profiles)}"}
        return {"ok": True, "char_id": char_id, "scenes": [
            {"start": s.start, "end": s.end, "score": s.score,
             "label": s.label} for s in scenes]}

    @registry.register(
        "reel_to_edit",
        description=("Cut a character reel (or top-scenes list) into a real "
                     "edit: extracts the clips and concatenates them with "
                     "transitions into one video. ('turn those scenes into "
                     "an edit', 'make the highlight reel')."),
        capability=Capability.FS_WRITE,
    )
    def reel_to_edit(path: str, scenes: list[dict[str, Any]], *,
                     transition_s: float = 0.5,
                     out_name: str = "reel.mp4") -> dict[str, Any]:
        import subprocess
        from shutil import which
        ff = which("ffmpeg")
        if not ff:
            return {"ok": False, "reason": "ffmpeg not found"}
        out_d = _out_dir(context, "reels")
        # extract each scene to a temp clip
        clips = []
        for i, s in enumerate(scenes):
            cp = out_d / f"clip_{i:03d}.mp4"
            subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(s["start"]), "-to", str(s["end"]),
                 "-i", path, "-c", "copy", str(cp)],
                check=True, capture_output=True, timeout=300)
            clips.append(cp)
        # concat with xfade chain (or plain concat if 1 clip)
        out = out_d / out_name
        if len(clips) == 1:
            import shutil
            shutil.copy(str(clips[0]), str(out))
        else:
            # build filtergraph: xfade chain
            inputs = []
            for c in clips:
                inputs += ["-i", str(c)]
            # probe durations
            import json as _json
            durs = []
            for c in clips:
                p = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of", "csv=p=0", str(c)],
                    capture_output=True, text=True, timeout=30)
                try:
                    durs.append(float(p.stdout.strip()))
                except ValueError:
                    durs.append(5.0)
            filt = ""
            last = "0:v"
            offset = durs[0] - transition_s
            for i in range(1, len(clips)):
                nxt = f"x{i}"
                filt += (f"[{last}][{i}:v]xfade=transition=fade:"
                         f"duration={transition_s}:offset={offset}[{nxt}];")
                last = nxt
                if i + 1 < len(clips):
                    offset += durs[i] - transition_s
            filt = filt.rstrip(";")
            afilt = "".join(
                f"[{i}:a]aresample=48000[a{i}];" for i in range(len(clips)))
            amix = "".join(f"[a{i}]" for i in range(len(clips)))
            amix += f"amix=inputs={len(clips)}:normalize=0[aout]"
            subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-y", *inputs,
                 "-filter_complex", filt + ";" + afilt + ";" + amix,
                 "-map", f"[{last}]", "-map", "[aout]",
                 "-pix_fmt", "yuv420p", str(out)],
                check=True, capture_output=True, timeout=900)
        return {"ok": True, "path": str(out),
                "clips": len(clips)}

    # ── generative editing ops ────────────────────────────────────

    @registry.register(
        "gen_shot",
        description=("Generate a video shot as an EDITING operation: "
                     "text-to-video ('generate a shot of a Lagos street at "
                     "night'), image-to-video ('animate this image'), or "
                     "video-to-video restyle ('restyle this clip as anime'). "
                     "Returns a clip file ready to cut into a timeline."),
        capability=Capability.GEN,
    )
    def gen_shot(mode: str, prompt: str, *, image: str = "",
                 video: str = "", duration_s: float = 5.0,
                 strength: float = 0.6) -> dict[str, Any]:
        from ..media import genedit
        out_d = _out_dir(context, "gen_shots")
        mode = mode.lower()
        if mode in ("t2v", "text"):
            shot = genedit.text_to_shot(prompt, duration_s=duration_s,
                                        workdir=str(out_d))
        elif mode in ("i2v", "image"):
            if not image:
                return {"ok": False, "reason": "i2v needs image="}
            shot = genedit.image_to_shot(image, prompt,
                                         duration_s=duration_s,
                                         workdir=str(out_d))
        elif mode in ("v2v", "video"):
            if not video:
                return {"ok": False, "reason": "v2v needs video="}
            shot = genedit.video_to_shot(video, prompt, strength=strength,
                                         workdir=str(out_d))
        elif mode == "extend":
            if not video:
                return {"ok": False, "reason": "extend needs video="}
            shot = genedit.extend_shot(video, prompt, seconds=duration_s,
                                       workdir=str(out_d))
        else:
            return {"ok": False,
                    "reason": f"unknown mode {mode!r} — t2v|i2v|v2v|extend"}
        return {"ok": True, "path": shot.path, "mode": shot.mode,
                "backend": shot.backend, "duration_s": shot.duration_s}

    @registry.register(
        "gen_shot_to_timeline",
        description=("Drop a generated shot (from gen_shot) into an edit "
                     "timeline file at a position. The shot is then graded, "
                     "cut, and transitioned like any footage."),
        capability=Capability.FS_WRITE,
    )
    def gen_shot_to_timeline(shot_path: str, timeline_path: str, *,
                             at: int = -1) -> dict[str, Any]:
        from ..media.genedit import GenShot, shot_to_timeline
        shot = GenShot(path=shot_path, mode="placed")
        out = shot_to_timeline(shot, timeline_path, at=at)
        return {"ok": True, "timeline": out}
