"""TrendOps — trend-driven video editing, end to end.

The class the owner named on 2026-10-21: Devon catches a trend and
the clip herself and edits in that format, instead of the owner
downloading clips by hand.

The three legs, in order:

1. **Intake** (:meth:`TrendOps.capture_trend`) — a trending clip URL
   (TikTok/Shorts/Reels) goes through the real download chain
   (:class:`~nomorals.media.downloader.MediaDownloader`, the same
   chain ``/download`` uses), is probed for duration/fps/beats, and
   lands as a :class:`~nomorals.media.trendops.formats.TrendFormat`
   in the library with an expiry date. What is measured is recorded
   as measured; what is not measurable takes the stated roadster
   defaults, never an invented measurement.
2. **Format library** — trends are data with expiry dates, built on
   the style-agnostic edit engine (never a hardcoded look).
3. **Apply** (:meth:`TrendOps.apply_format`) — footage + a live
   format → the format's grammar (ramp curve, beat grid, caption
   system, transitions, structure) → engine render → MP4, with a
   post-render review (probe assertions on the output file).

Autonomy: :meth:`TrendOps.propose` ranks the live formats by
expiry urgency (least-used first within an urgency band) and returns
ready apply commands — "this format is peaking; here's your clip cut
to it". The unprompted proposal hook (a scheduler job calling
``propose()``) is the next step after this class lands, not part of
this commit.

Honest limits, stated where they bite: the post-render review
watches the finished video over time with the director's WatchVideo
organ (:mod:`nomorals.media.director.watchvideo`) — cuts on the
beat grid, segment pacing, motion where an effect should be moving
the picture. Footage that carries no measurable frame variation at
all (a solid-colour clip) is reported as ``qa_unavailable`` with
the reason, never a fake pass. There is no video Museum artifact
organ in the repo, so the rendered MP4 + its JSON manifest in the
workspace is the artifact; nothing is written to a Museum that does
not exist.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError, video_probe
from ..contentops.beats import detect_beats_full
from ..edit_engine import render_report as render_spec_report
from .formats import TrendFormat
from .library import FormatLibrary

__all__ = ["TrendOps", "draft_for_render"]

_log = get_logger(__name__)

#: Where TrendOps keeps formats + renders (the shorts convention:
#: ``~/workspace/devon/.shorts_runs``).
DEFAULT_ROOT = Path.home() / "workspace" / "devon" / ".trendops"


class TrendOps:
    """Trend intake → format library → rendered trend edit."""

    def __init__(self, *, context: Any = None,
                 library: FormatLibrary | None = None,
                 root: str | Path | None = None,
                 render_fn: Callable[..., dict[str, Any]] | None = None,
                 beats_fn: Callable[..., Any] | None = None,
                 downloader: Any = None) -> None:
        self.context = context
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.library = library if library is not None \
            else FormatLibrary(self.root)
        self.renders_dir = self.root / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self._render_fn = render_fn or render_spec_report
        self._beats_fn = beats_fn or detect_beats_full
        self._downloader = downloader

    # -- formats ---------------------------------------------------------

    def list_formats(self, *, live_only: bool = True) -> list[dict[str, Any]]:
        """Live formats as plain records (slug, name, niche, days_left,
        uses, description)."""
        now = time.time()
        out: list[dict[str, Any]] = []
        for fmt in self.library.all(live_only=live_only, now=now):
            out.append({
                "slug": fmt.slug, "name": fmt.name, "niche": fmt.niche,
                "source": fmt.source,
                "days_left": fmt.days_left(now),
                "uses": fmt.uses, "description": fmt.description,
            })
        return out

    def get_format(self, slug: str) -> TrendFormat:
        return self.library.get(slug)

    def add_format(self, fmt: TrendFormat) -> TrendFormat:
        return self.library.save(fmt)

    def remove_format(self, slug: str) -> bool:
        return self.library.remove(slug)

    def propose(self, limit: int = 5) -> list[dict[str, Any]]:
        """Ranked applications: most urgent expiry first, least-used
        first inside an urgency band. Each proposal carries the ready
        apply command."""
        now = time.time()
        live = self.library.all(live_only=True, now=now)

        def _key(fmt: TrendFormat) -> tuple[float, int]:
            days = fmt.days_left(now)
            return (days if days is not None else 365.0, fmt.uses)

        out: list[dict[str, Any]] = []
        for fmt in sorted(live, key=_key)[:max(1, int(limit))]:
            days = fmt.days_left(now)
            out.append({
                "slug": fmt.slug, "name": fmt.name, "niche": fmt.niche,
                "days_left": round(days, 1) if days is not None else None,
                "uses": fmt.uses,
                "apply": f"/trendops apply {fmt.slug} <footage>",
                "why": (f"{fmt.name} peaks now — "
                        f"{days:.0f}d left, used {fmt.uses}x"
                        if days is not None else
                        f"{fmt.name} is live — used {fmt.uses}x"),
            })
        return out

    # -- intake ----------------------------------------------------------

    def capture_trend(self, source: str, *, name: str | None = None,
                      niche: str = "", expires_days: float = 14.0,
                      ramp_curve: Any = None, caption_style: str | None = None,
                      caption_text: str | None = None) -> dict[str, Any]:
        """Catch a trending clip and register its format.

        Downloads ``source`` through the real chain, probes it, and
        saves a format derived from what was measured. A clip with no
        detectable beat structure gets ``cut_rule`` interval (stated in
        the record) — the format never claims beats it did not measure.
        """
        url = str(source or "").strip()
        if not url:
            raise MediaEditError("capture_trend needs a clip URL")
        downloader = self._downloader
        if downloader is None:
            if self.context is None:
                raise MediaEditError(
                    "capture_trend needs the runtime context for the "
                    "download chain — call it from chat or the trendops "
                    "tool, not a bare TrendOps()")
            from ..downloader import MediaDownloader

            downloader = MediaDownloader(self.context)
        slug = (str(name or "").strip().lower().replace(" ", "_")
                or f"trend_{int(time.time())}")
        dest = self.root / "captured"
        dest.mkdir(parents=True, exist_ok=True)
        report = downloader.download(url, dest_dir=dest)
        if not getattr(report, "ok", False):
            raise MediaEditError(
                f"could not catch the trend clip: {report.summary()}")
        clip_path = str(getattr(report, "path", "") or "")
        if not clip_path or not Path(clip_path).exists():
            raise MediaEditError(
                "the download reported success but left no file — "
                "refusing to register a format from nothing")
        probe = video_probe(clip_path)
        duration = float(probe.get("duration") or 0.0)
        beats: list[float] = []
        try:
            info = self._beats_fn(clip_path)
            beats = [float(b) for b in (info.beats or [])]
        except Exception as exc:  # noqa: BLE001 - no beats is data, not a crash
            _log.debug("trendops: beat detection failed on capture: %s", exc)
        measured: dict[str, Any] = {
            "duration": duration,
            "fps": probe.get("fps"),
            "width": probe.get("width"), "height": probe.get("height"),
            "beats": beats, "beat_count": len(beats),
            "clip_path": clip_path,
            "title": getattr(report, "title", "") or "",
        }
        overrides: dict[str, Any] = {
            "measured": measured,
            "description": (
                f"Captured from {url} — {duration:.1f}s, "
                f"{len(beats)} beats measured."),
        }
        if ramp_curve is not None:
            overrides["ramp_curve"] = ramp_curve
        if caption_style is not None:
            cap = {"style": caption_style, "text": "", "position": "center"}
            if caption_text is not None:
                cap["text"] = caption_text
            overrides["caption"] = cap
        elif caption_text is not None:
            cap = {"style": "hormozi", "text": caption_text,
                   "position": "center"}
            overrides["caption"] = cap
        if not beats:
            overrides["cut_rule"] = "interval"
            overrides["interval_s"] = 1.0
            overrides["description"] += (
                " No beat structure measured — cuts fall back to a 1s grid.")
        fmt = self.library.new_format(
            slug, source="capture", niche=niche, reference_url=url,
            expires_days=expires_days, **overrides)
        return {"ok": True, "slug": fmt.slug, "name": fmt.name,
                "clip": clip_path, "measured": measured,
                "expires_in_days": expires_days,
                "format": fmt.to_dict()}

    # -- apply -----------------------------------------------------------

    def apply_format(self, footage: str | list[str], format_name: str, *,
                     music: str | None = None, caption: str | None = None,
                     beats: list[float] | None = None,
                     out: str | None = None, seed: int = 0,
                     crf: int | None = None) -> dict[str, Any]:
        """Cut ``footage`` to a live format → rendered MP4 + review.

        Returns the run report (never a bare path): output, format,
        duration, beats used, review. Raises MediaEditError naming
        the problem on: unknown/expired format, missing footage, a
        beats cut rule with no beat source, or a render failure.
        """
        fmt = self.library.get(format_name)
        paths = [footage] if isinstance(footage, (str, Path)) else list(footage or [])
        paths = [str(p) for p in paths]
        if not paths:
            raise MediaEditError("apply_format needs a footage file")
        for p in paths:
            if not Path(p).exists():
                raise MediaEditError(f"no such footage: {p}")
        grid = self._resolve_beats(fmt, beats, music, paths)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = out or str(
            self.renders_dir / f"{fmt.slug}-{stamp}.mp4")
        spec = fmt.build_spec(
            paths, beats=grid, music=music, caption=caption,
            out=out_path, seed=seed, crf=crf)
        report = self._render_fn(
            spec, workdir=str(self.renders_dir / f".tmp-{stamp}"))
        output = str(report.get("output") or out_path)
        if not Path(output).exists():
            raise MediaEditError(
                f"the render reported success but wrote no file "
                f"(expected {output})")
        review = self._review(output, fmt, spec)
        self.library.bump_uses(fmt.slug)
        return {
            "ok": True, "output": output, "format": fmt.slug,
            "format_name": fmt.name, "duration": review.get("duration"),
            "width": review.get("width"), "height": review.get("height"),
            "beats_used": len(grid or []),
            "segments": len(spec.segments or []),
            "review": review,
        }

    def _resolve_beats(self, fmt: TrendFormat,
                       beats: list[float] | None, music: str | None,
                       paths: list[str]) -> list[float] | None:
        if fmt.cut_rule != "beats":
            return list(beats) if beats else None
        if beats:
            return [float(b) for b in beats]
        for candidate in ([music] if music else []) + paths[:1]:
            if not candidate:
                continue
            try:
                info = self._beats_fn(str(candidate))
            except Exception as exc:  # noqa: BLE001 - try the next source
                _log.debug("trendops: beat detection failed on %s: %s",
                           candidate, exc)
                continue
            found = [float(b) for b in (info.beats or [])]
            if found:
                _log.info("trendops: %d beats from %s (%s engine)",
                          len(found), candidate, info.backend)
                return found
        raise MediaEditError(
            "cut_rule is beats but no beat grid could be found — pass "
            "beats= explicitly or a music file with a clear beat")

    def _review(self, output: str, fmt: TrendFormat,
                spec: Any = None) -> dict[str, Any]:
        """Post-render review: probe assertions on the finished file,
        then the WatchVideo organ watches it over time.

        The probe assertions are the render's own health (duration,
        canvas, non-empty file) — a file that fails them is a failed
        render, reported as such, and ``review["ok"]`` stays theirs
        alone. The ``qa`` leg is the temporal verdict: cuts on the
        beat grid, segment pacing, motion where an effect should be
        moving the picture. It never blocks the render and never
        fakes a pass — footage with no measurable frame variation,
        or a watch that cannot run, reports ``qa_unavailable`` with
        the reason.
        """
        info = video_probe(output)
        duration = float(info.get("duration") or 0.0)
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        problems: list[str] = []
        if duration <= 0:
            problems.append("output has no measurable duration")
        if width != int(fmt.canvas["width"]) \
                or height != int(fmt.canvas["height"]):
            problems.append(
                f"output is {width}x{height}, format wants "
                f"{fmt.canvas['width']}x{fmt.canvas['height']}")
        if Path(output).stat().st_size <= 0:
            problems.append("output file is empty")
        return {
            "ok": not problems,
            "problems": problems,
            "duration": duration, "width": width, "height": height,
            "bytes": Path(output).stat().st_size,
            "qa": self._temporal_qa(output, spec),
        }

    def _temporal_qa(self, output: str, spec: Any) -> dict[str, Any]:
        """Watch the finished render over time against the spec's
        cut grid + segment plan. Any failure to watch — unreadable
        file, missing ffmpeg, sampling error — is ``qa_unavailable``
        with the reason, never a crash and never a pass."""
        try:
            from ..director.watchvideo import watch_video

            cut_points = list(getattr(spec, "cut_points", None) or [])
            segments = [float(d) for _, d in
                        (getattr(spec, "segments", None) or [])]
            return watch_video(output, cut_points=cut_points,
                               segments=segments or None)
        except Exception as exc:  # noqa: BLE001 - QA reports, never crashes
            return {"status": "qa_unavailable",
                    "notes": f"WatchVideo could not run: {exc}"}


def draft_for_render(report: dict[str, Any], *, footage: list[str],
                     music: str | None = None,
                     caption: str | None = None) -> tuple[str, dict[str, Any]]:
    """Build a post draft's (content, metadata) for a finished render.

    Pure data shaping — no queue, no publish. The caller (the chat
    layer) creates the draft through the drafts organ and proposes
    it, so the owner is asked "post directly or send for review?"
    through the existing drafts mechanism (the #43 rule), not
    handed a bare file path.

    ``report`` is the :meth:`TrendOps.apply_format` run report.
    ``footage`` is the footage the cut was made from — the ready
    re-cut command is built from the real args, not a placeholder.
    The content is the post text: the owner's caption when one was
    given (the meme caption is the natural post text), else a
    description of the cut. The metadata carries the MP4 path, the
    format, the render manifest (duration, canvas, beats, segments,
    review) and the ready re-cut command — and the media entry the
    drafts organ's publish path resolves (``{"path": ...}`` for a
    file on disk, the shape
    :func:`...social.drafts.execute_post` materializes before
    posting).
    """
    import shlex

    output = str(report.get("output") or "")
    fmt_slug = str(report.get("format") or "")
    fmt_name = str(report.get("format_name") or fmt_slug)
    duration = float(report.get("duration") or 0.0)
    width = int(report.get("width") or 0)
    height = int(report.get("height") or 0)
    beats_used = int(report.get("beats_used") or 0)
    segments = int(report.get("segments") or 0)
    cmd_parts = ["trendops", "apply", fmt_slug] + [str(p) for p in footage]
    if music:
        cmd_parts += ["--music", str(music)]
    if caption:
        cmd_parts += ["--caption", str(caption)]
    apply_cmd = "/" + " ".join(shlex.quote(p) for p in cmd_parts)
    content = (str(caption).strip() if caption else
               f"🎬 {fmt_name} cut — {duration:.1f}s · {width}x{height} · "
               f"{segments} segments · {beats_used} beats")
    metadata: dict[str, Any] = {
        "origin": "trendops",
        "format": fmt_slug,
        "format_name": fmt_name,
        "output": output,
        "apply": apply_cmd,
        "duration": duration,
        "width": width, "height": height,
        "beats_used": beats_used,
        "segments": segments,
        "review": report.get("review") or {},
        "media": [{"path": output,
                   "name": Path(output).name or "trend-cut.mp4",
                   "mime": "video/mp4"}],
    }
    return content, metadata
