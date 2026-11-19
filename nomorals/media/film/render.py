"""Film render — the timeline becomes a finished video.

The render path is the chain, not a new island:

* per-frame compositing: video files via an ffmpeg raw pipe, stills
  with a Ken Burns move, solids, and :class:`Composition` clips
  rendered through the effects compositor; per-clip colour grade +
  raster effects; transitions (cross dissolve / wipe / dip to
  black) blended in the cut window; titles drawn with the one font
  door; captions burned in through ``videogen.captions``;
* segments: the timeline is split at shot boundaries and rendered
  through the edit-engine :class:`RenderQueue` (resumable, one
  part MP4 per segment), then concatenated — a 4-second project
  and a 40-minute project render the same way, so no clip-length
  cap exists anywhere in this path;
* audio: audio-track clips + an optional narration wav + a music
  bed, mixed with ducking under the narration and loudness
  measured + normalised, then muxed onto the picture;
* honesty: a missing source, a bad project, or a failed encode
  returns ``{"ok": False, "error": ...}`` naming the cause — never
  a silent short file. The profile gate
  (``resolve_render_profile``) never silently downgrades: a small
  profile renders the proxy height and the report says so.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from ..edit_engine.render_queue import (
    RenderQueue,
    resolve_render_profile,
    segments_for_ranges,
)
from ..effects import effects as fx_effects
from ..effects.composition import render_composition
from ..effects.store import CompositionStore, effects_path
from ..motion_studio._core import encode_frames, ffmpeg_path, find_font
from ..videogen.captions import captions_srt, draw_captions
from .audio import mix_audio, normalize_loudness, write_wav
from .project import Clip, FilmProject

__all__ = ["export_srt", "render_project"]

_VIDEO_KINDS = ("video", "still", "solid", "composition")


def _ffprobe(path: str | Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr[-300:]}")
    return json.loads(proc.stdout)


class _VideoReader:
    """Sequential ffmpeg raw-pipe reader for one video file.

    Frames are read in order; a backward seek restarts the pipe.
    Past the end of the file the last frame holds — a transition
    window that runs a few frames past a clip's end still gets a
    picture."""

    def __init__(self, path: str, width: int, height: int,
                 fps: float) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self._proc: subprocess.Popen | None = None
        self._index = -1
        self._last: np.ndarray | None = None

    def _open(self) -> None:
        self._close()
        cmd = [ffmpeg_path(), "-v", "error", "-i", self.path,
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.width}x{self.height}",
               "-r", f"{self.fps:.3f}", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)
        self._index = -1

    def _close(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.kill()
            self._proc = None

    def frame_at(self, t: float) -> np.ndarray:
        target = max(0, int(t * self.fps))
        if self._proc is None or target <= self._index:
            self._open()
        assert self._proc is not None and self._proc.stdout is not None
        frame_bytes = self.width * self.height * 3
        while self._index < target:
            raw = self._proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            self._index += 1
            self._last = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)).copy()
        if self._last is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self._last

    def close(self) -> None:
        self._close()


def _parse_color(source: str) -> tuple[int, int, int]:
    text = source.strip()
    if text.startswith("#") and len(text) == 7:
        return (int(text[1:3], 16), int(text[3:5], 16),
                int(text[5:7], 16))
    raise ValueError(f"solid clip source must be a #rrggbb color, "
                     f"got {source!r}")


def _apply_grade(arr: np.ndarray, grade: dict[str, float]) -> np.ndarray:
    lift = float(grade.get("lift", 0.0))
    gamma = float(grade.get("gamma", 1.0))
    gain = float(grade.get("gain", 1.0))
    saturation = float(grade.get("saturation", 1.0))
    img = arr.astype(np.float64) / 255.0
    img = np.clip(img * gain + lift, 0.0, 1.0)
    if gamma != 1.0:
        img = np.clip(img, 0.0, 1.0) ** gamma
    out = Image.fromarray((img * 255.0).astype(np.uint8))
    if saturation != 1.0:
        out = ImageEnhance.Color(out).enhance(saturation)
    return np.asarray(out)


def _apply_clip_effects(arr: np.ndarray, clip: Clip) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8))
    for effect in clip.effects:
        img = fx_effects.apply_effect(effect["name"], img,
                                      effect.get("params") or {})
        if img.mode != "RGB":
            img = img.convert("RGB")
    return np.asarray(img)


class _Sampler:
    """One project's frame function: clip -> picture at time t."""

    def __init__(self, project: FilmProject, width: int, height: int,
                 comp_store: CompositionStore) -> None:
        self.project = project
        self.width = width
        self.height = height
        self.fps = project.fps
        self.comp_store = comp_store
        self._readers: dict[str, _VideoReader] = {}
        self._stills: dict[str, Image.Image] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    # -- clip pixels ------------------------------------------------------
    def _clip_frame(self, clip: Clip, t: float) -> np.ndarray:
        media_t = clip.start + (t - clip.at) * clip.speed
        if clip.kind == "video":
            reader = self._readers.get(clip.id)
            if reader is None:
                reader = _VideoReader(clip.source, self.width,
                                      self.height, self.fps)
                self._readers[clip.id] = reader
            frame = reader.frame_at(media_t)
        elif clip.kind == "still":
            frame = self._still_frame(clip, t)
        elif clip.kind == "solid":
            color = _parse_color(clip.source)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = color
        elif clip.kind == "composition":
            comp = self.comp_store.load(clip.source)
            frame = np.asarray(render_composition(comp, media_t))
            if frame.shape[:2] != (self.height, self.width):
                frame = np.asarray(
                    Image.fromarray(frame).resize(
                        (self.width, self.height), Image.BICUBIC))
        else:
            raise RuntimeError(
                f"clip {clip.id} kind {clip.kind!r} has no picture")
        frame = _apply_grade(frame, clip.grade)
        if clip.effects:
            frame = _apply_clip_effects(frame, clip)
        return frame

    def _still_frame(self, clip: Clip, t: float) -> np.ndarray:
        if clip.id not in self._stills:
            src = Image.open(clip.source).convert("RGB")
            self._stills[clip.id] = src
        src = self._stills[clip.id]
        kb = clip.kenburns
        zoom_from = float(kb.get("zoom_from", 1.0))
        zoom_to = float(kb.get("zoom_to", 1.08))
        u = 0.0
        if clip.duration > 0:
            u = min(1.0, max(0.0, (t - clip.at) / clip.duration))
        zoom = zoom_from + (zoom_to - zoom_from) * u
        pan_x = float(kb.get("pan_x", 0.0))
        pan_y = float(kb.get("pan_y", 0.0))
        # cover-fit the canvas, then crop the zoomed window
        scale = max(self.width / src.size[0], self.height / src.size[1])
        fitted = src.resize((max(1, int(src.size[0] * scale)),
                             max(1, int(src.size[1] * scale))),
                            Image.BICUBIC)
        cw, ch = int(self.width / zoom), int(self.height / zoom)
        cx = (fitted.size[0] - cw) // 2 + int(pan_x * fitted.size[0] / 4)
        cy = (fitted.size[1] - ch) // 2 + int(pan_y * fitted.size[1] / 4)
        cx = max(0, min(fitted.size[0] - cw, cx))
        cy = max(0, min(fitted.size[1] - ch, cy))
        return np.asarray(
            fitted.crop((cx, cy, cx + cw, cy + ch)).resize(
                (self.width, self.height), Image.BICUBIC))

    # -- timeline -----------------------------------------------------------
    def _video_clip_at(self, track_kind: str, t: float,
                       track_index: int | None = None) -> Clip | None:
        tracks = [tr for tr in self.project.tracks
                  if tr.kind == track_kind]
        if track_index is not None:
            tracks = tracks[track_index:track_index + 1]
        for track in reversed(tracks):  # later tracks sit on top
            for clip in track.items:
                if clip.at <= t < clip.end:
                    return clip
        return None

    def _transition_at(self, t: float) -> tuple[Any, float] | None:
        for tr in self.project.transitions:
            _ta, clip_a = self.project.find_clip(tr.from_id)
            _tb, clip_b = self.project.find_clip(tr.to_id)
            cut = clip_a.end
            if tr.align == "center":
                start, end = cut - tr.duration / 2, cut + tr.duration / 2
            elif tr.align == "start":
                start, end = cut, cut + tr.duration
            else:
                start, end = cut - tr.duration, cut
            if start <= t < end:
                return tr, (t - start) / tr.duration
        return None

    def frame(self, t: float) -> np.ndarray:
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        clip = self._video_clip_at("video", t)
        if clip is None:
            clip = self._video_clip_at("composition", t)
        base = black if clip is None else self._clip_frame(clip, t)
        transition = self._transition_at(t)
        if transition is not None:
            tr, u = transition
            _ta, clip_a = self.project.find_clip(tr.from_id)
            _tb, clip_b = self.project.find_clip(tr.to_id)
            frame_a = self._clip_frame(clip_a, t)
            frame_b = self._clip_frame(clip_b, t)
            u = min(1.0, max(0.0, u))
            eased = u * u * (3.0 - 2.0 * u)
            if tr.kind == "cross_dissolve":
                base = (frame_a.astype(np.float64) * (1 - eased)
                        + frame_b.astype(np.float64) * eased)
            elif tr.kind == "wipe":
                cut_x = int(eased * self.width)
                base = frame_a.copy()
                base[:, cut_x:] = frame_b[:, cut_x:]
            elif tr.kind == "dip_to_black":
                if u < 0.5:
                    k = 1.0 - (u / 0.5)
                    base = frame_a.astype(np.float64) * k
                else:
                    k = (u - 0.5) / 0.5
                    base = frame_b.astype(np.float64) * k
            base = np.clip(base, 0, 255).astype(np.uint8)
        # titles on top — the font shrinks until the line fits the
        # frame (a title that runs off the edges is a defect, not a
        # style); the size recorded on the clip is the *starting* size
        img = Image.fromarray(base.astype(np.uint8))
        for track in self.project.tracks:
            if track.kind != "title":
                continue
            for title in track.items:
                if title.at <= t < title.end and title.text:
                    kb = title.kenburns
                    size = int(kb.get("font_size", 48))
                    color = (int(kb.get("r", 255)), int(kb.get("g", 255)),
                             int(kb.get("b", 255)))
                    draw = ImageDraw.Draw(img, "RGBA")
                    font = find_font(size)
                    max_w = int(self.width * 0.92)
                    while size > 8:
                        box = draw.textbbox((0, 0), title.text, font=font)
                        if box[2] - box[0] <= max_w:
                            break
                        size = max(8, int(size * 0.9))
                        font = find_font(size)
                    draw.text((self.width / 2, self.height / 2),
                              title.text, font=font, fill=(*color, 255),
                              anchor="mm", stroke_width=2,
                              stroke_fill=(0, 0, 0, 180))
        # captions on top (videogen/captions draws in place)
        words: list[dict[str, Any]] = []
        for track in self.project.tracks:
            if track.kind != "caption":
                continue
            for cap in track.items:
                words.extend(cap.words)
        if words:
            draw_captions(img, words, t)
        return np.asarray(img)


# ---------------------------------------------------------------------------
# the render
# ---------------------------------------------------------------------------

def export_srt(project: FilmProject, out_path: str | Path) -> Path:
    """All caption words of the project as one SRT document."""
    words: list[dict[str, Any]] = []
    for track in project.tracks:
        if track.kind != "caption":
            continue
        for cap in track.items:
            words.extend(cap.words)
    words.sort(key=lambda w: (float(w["start"]), float(w["end"])))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(captions_srt(words), encoding="utf-8")
    return out


def render_project(project: FilmProject, out_path: str | Path, *,
                   workdir: str | Path,
                   narration_wav: str | Path | None = None,
                   bed_wav: str | Path | None = None,
                   narration_segments: list[tuple[float, float]] | None = None,
                   profile: str = "workstation",
                   allow_full_res: bool = False,
                   comp_store: CompositionStore | None = None) -> dict[str, Any]:
    """Render the project to ``out_path`` (MP4). Returns a report
    dict — ``{"ok": True, ...}`` on success, ``{"ok": False,
    "error": ...}`` naming the cause on any failure."""
    out = Path(out_path)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    try:
        return _render_project(project, out, work,
                               narration_wav=narration_wav,
                               bed_wav=bed_wav,
                               narration_segments=narration_segments,
                               profile=profile,
                               allow_full_res=allow_full_res,
                               comp_store=comp_store)
    except Exception as exc:  # noqa: BLE001 - the report names the cause
        return {"ok": False, "error": str(exc), "path": str(out),
                "project": project.name}


def _render_project(project: FilmProject, out: Path, work: Path, *,
                    narration_wav: str | Path | None,
                    bed_wav: str | Path | None,
                    narration_segments: list[tuple[float, float]] | None,
                    profile: str, allow_full_res: bool,
                    comp_store: CompositionStore | None) -> dict[str, Any]:
    problems = project.validate()
    if problems:
        return {"ok": False, "error": "; ".join(problems),
                "path": str(out), "project": project.name}
    # every media source must exist — named before any render work
    store = comp_store or CompositionStore(effects_path(None))
    for track in project.tracks:
        for clip in track.items:
            if clip.kind in ("video", "still", "audio"):
                if not clip.source or not Path(clip.source).exists():
                    return {"ok": False,
                            "error": f"clip {clip.id} source not found: "
                                     f"{clip.source}",
                            "path": str(out), "project": project.name}
            elif clip.kind == "composition":
                try:
                    store.load(clip.source)
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False,
                            "error": f"composition clip {clip.id}: {exc}",
                            "path": str(out), "project": project.name}
    gate = resolve_render_profile(profile=profile, height=project.height,
                                  allow_full_res=allow_full_res)
    height = int(gate["height"])
    width = max(2, int(round(project.width * height / project.height)))
    width += width % 2
    height += height % 2
    duration = project.duration
    if duration <= 0:
        return {"ok": False, "error": "project is empty — no clips",
                "path": str(out), "project": project.name}
    sampler = _Sampler(project, width, height, store)
    try:
        # segments: split at shot boundaries of the first video track
        video_tracks = [tr for tr in project.tracks
                        if tr.kind in _VIDEO_KINDS]
        bounds = {0.0, duration}
        if video_tracks:
            for clip in video_tracks[0].items:
                bounds.add(clip.at)
                bounds.add(clip.end)
        points = sorted(b for b in bounds if 0.0 <= b <= duration)
        ranges = [(a, b) for a, b in zip(points, points[1:])  # noqa: B905 - sliding window: the second iterable is one shorter by construction
                  if b > a]
        if not ranges:
            ranges = [(0.0, duration)]
        segments = segments_for_ranges(ranges)
        parts_dir = work / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        queue = RenderQueue(segments,
                            state_path=work / "render_queue.json")

        def _render_segment(seg: Any) -> str:
            part = parts_dir / f"{seg.id}.mp4"
            n = max(1, int(round((seg.end - seg.start) * project.fps)))

            def _gen() -> Any:
                for i in range(n):
                    yield sampler.frame(seg.start + i / project.fps)

            encode_frames(_gen(), part, width, height, project.fps)
            return str(part)

        summary = queue.run_all(_render_segment, profile=profile)
        if summary["failed"]:
            failed = queue.failed
            return {"ok": False,
                    "error": f"segment {failed[0].id} failed: "
                             f"{failed[0].error}",
                    "path": str(out), "project": project.name,
                    "segments": summary}
        # concatenate the parts
        concat_list = work / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{s.output}'\n" for s in queue.done),
            encoding="utf-8")
        video_only = work / "video_only.mp4"
        proc = subprocess.run(
            [ffmpeg_path(), "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(video_only)],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"concat failed: {proc.stderr[-500:]}",
                    "path": str(out), "project": project.name}
        # audio: track clips + narration + bed, ducked + normalised
        loudness: dict[str, Any] | None = None
        audio_sources: list[dict[str, Any]] = []
        for track in project.tracks:
            if track.kind != "audio" or track.muted:
                continue
            for clip in track.items:
                audio_sources.append({"path": clip.source, "at": clip.at,
                                      "gain": clip.gain})
        bed_index: int | None = None
        speech: list[tuple[float, float]] = []
        if narration_wav is not None:
            audio_sources.append({"path": str(narration_wav), "at": 0.0,
                                  "gain": 1.0})
            speech = (list(narration_segments)
                      if narration_segments is not None
                      else [(0.0, duration)])
        if bed_wav is not None:
            bed_index = len(audio_sources)
            audio_sources.append({"path": str(bed_wav), "at": 0.0,
                                  "gain": 0.5})
        final = video_only
        if audio_sources:
            audio_dir = work / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            ducking = None
            if bed_index is not None and speech:
                ducking = {"bed_index": bed_index, "speech": speech,
                           "duck_gain": float(
                               project.ducking.get("duck_gain", 0.25))}
            mixed = mix_audio(audio_sources, duration_s=duration,
                              workdir=audio_dir, ducking=ducking)
            mixed_wav = write_wav(audio_dir / "mixed.wav", mixed)
            norm = normalize_loudness(mixed_wav,
                                      audio_dir / "mixed_norm.wav")
            loudness = norm["after"]
            out.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [ffmpeg_path(), "-y", "-i", str(video_only),
                 "-i", str(norm["path"]), "-c:v", "copy", "-c:a", "aac",
                 "-b:a", "160k", "-shortest", str(out)],
                capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                return {"ok": False,
                        "error": f"audio mux failed: {proc.stderr[-500:]}",
                        "path": str(out), "project": project.name}
            final = out
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            final = out
            video_only.replace(out)
        srt_path = export_srt(project, out.with_suffix(".srt"))
        report: dict[str, Any] = {
            "ok": True,
            "path": str(final),
            "project": project.name,
            "duration": duration,
            "fps": project.fps,
            "width": width,
            "height": height,
            "segments": [{"id": s.id, "start": s.start, "end": s.end,
                          "output": s.output} for s in queue.done],
            "srt": str(srt_path),
            "loudness": loudness,
            "profile": gate,
            "rungs": {"video": "film", "audio": "mix+ducking",
                      "captions": "videogen/captions",
                      "titles": "film/find_font"},
            "engine": "film",
        }
        (out.with_suffix(".json")).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        sampler.close()
