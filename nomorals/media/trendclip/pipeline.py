"""The trendclip pipeline — ingest, analyze, render, validate, package,
queue. One class, injectable collaborators, honest receipts.

Stages (mirroring the source's job machine)::

    queued -> ingesting -> transcribing -> analyzing -> ready
    ready -> rendering -> validating -> complete | failed

The deterministic rung always runs. The brain rung runs when the caller
supplies a ``pick_fn`` (the /clip handler's ``brain_for`` consult); a
tool call from the brain itself runs the deterministic rung only — the
brain is already driving, it reads the candidates and picks. Both rungs
are recorded in the job receipt either way.
"""

from __future__ import annotations

import logging
import math
import struct
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import render as _render
from . import scoring as _scoring
from . import validate as _validate
from .brain import brain_pick_moments, brain_platform_copy
from .discovery import discover_youtube_trends, inspect_youtube_url
from .export_package import export_package
from .jobs import JobStore, TrendclipJob
from .rights import validate_rights
from .topics import TopicConfig
from .transcribe import transcribe_audio

__all__ = [
    "TrendclipPipeline",
    "prerendered_clip_for_job",
    "trendclip_post_copy",
    "trendclip_root",
]

_log = logging.getLogger(__name__)


def trendclip_root(settings: Any = None) -> Path:
    """The trendclip home: ``settings.resolve("trendclip")`` when the
    settings object offers it, else ``~/trendclip``."""
    resolve = getattr(settings, "resolve", None)
    if callable(resolve):
        try:
            return Path(str(resolve("trendclip")))
        except Exception:  # noqa: BLE001 - fall through to the home default
            pass
    home = Path(getattr(settings, "home", "") or "~").expanduser()
    return home / "trendclip"


def prerendered_clip_for_job(root: str | Path | None, job_id: str) -> str:
    """The validated rendered clip for a trendclip job, or "".

    This is the shorts publisher sweep's hook: a planned post whose
    ``job_id`` names a trendclip job uses this clip instead of
    re-rendering a scripted short (see
    ``nomorals/agents/shorts_publisher.py``). Only a clip whose validation
    passed counts — anything else returns "" and the sweep falls through
    to its normal render path. ``root`` defaults to the trendclip home
    (``~/trendclip``) when None.
    """
    try:
        store = JobStore(Path(root) if root is not None else
                         Path.home() / "trendclip")
        job = store.get(job_id)
    except (KeyError, OSError, ValueError):
        return ""
    if job.stage != "complete":
        return ""
    receipt = job.receipt or {}
    clip = str(receipt.get("render", {}).get("clip_path") or "")
    validation = receipt.get("validation") or {}
    if not clip or not validation.get("passed"):
        return ""
    if not Path(clip).is_file():
        return ""
    return clip


def trendclip_post_copy(root: str | Path | None, job_id: str
                        ) -> dict[str, Any] | None:
    """Title/description/hashtags + transcript basis for a trendclip
    post, from its export package and job receipt. None when the job is
    not a completed trendclip job. The publisher sweep uses this as the
    post copy (the skeptic checks its numbers against the transcript,
    the same way it checks a scripted short's copy against its script).
    """
    try:
        store = JobStore(Path(root) if root is not None else
                         Path.home() / "trendclip")
        job = store.get(job_id)
    except (KeyError, OSError, ValueError):
        return None
    if job.stage != "complete":
        return None
    receipt = job.receipt or {}
    render_info = receipt.get("render") or {}
    package = render_info.get("package") or {}
    platforms = package.get("platforms") or {}
    title = ""
    description = ""
    hashtags: list[str] = []
    for _dest, meta in platforms.items():
        title = str(meta.get("title") or "")
        description = str(meta.get("description") or "")
        hashtags = list(meta.get("hashtags") or [])
        if title:
            break
    if not title:
        title = str((receipt.get("source") or {}).get("title")
                    or job.topic)
    return {
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "script": str(receipt.get("transcript_lines") or ""),
    }


class TrendclipPipeline:
    """Ingest → transcribe → score (both rungs) → render → validate →
    package → queue. The collaborators are injectable so tests run offline
    with fakes and production wires the real network, STT and FFmpeg."""

    def __init__(self, root: str | Path, *,
                 ytdlp: str = "yt-dlp",
                 runner: Callable[..., Any] | None = None,
                 transcriber: Callable[..., dict[str, Any]] | None = None,
                 pick_fn: Callable[[str], Any] | None = None,
                 copy_fn: Callable[[str], Any] | None = None,
                 shorts_runs_root: str | Path | None = None):
        self.root = Path(root)
        self.cache_dir = self.root / "cache"
        self.clips_dir = self.root / "clips"
        self.exports_dir = self.root / "exports"
        for d in (self.cache_dir, self.clips_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.jobs = JobStore(self.root)
        self.ytdlp = ytdlp
        self.runner = runner
        self.transcriber = transcriber
        self.pick_fn = pick_fn
        self.copy_fn = copy_fn
        self.shorts_runs_root = (
            Path(shorts_runs_root) if shorts_runs_root else
            Path.home() / "workspace" / "devon" / ".shorts_runs")

    # -- discovery -----------------------------------------------------
    def discover(self, topic: TopicConfig, *, limit: int = 10
                 ) -> list[dict[str, Any]]:
        """Trending videos for a topic, ranked by views-per-hour."""
        from ...media.resolver import ensure_ytdlp
        ensure_ytdlp()
        return discover_youtube_trends(topic, limit=limit,
                                       runner=self.runner,
                                       ytdlp=self.ytdlp)

    # -- ingest --------------------------------------------------------
    def _run(self, cmd: list[str], timeout: int = 600) -> Any:
        run = self.runner or subprocess.run
        return run(cmd, capture_output=True, text=True, timeout=timeout)

    def _download(self, url: str, job_dir: Path) -> dict[str, Any]:
        """Audio + low-res proxy for one source via yt-dlp."""
        audio_out = job_dir / "audio.%(ext)s"
        proxy_out = job_dir / "proxy.%(ext)s"
        audio = self._run([self.ytdlp, "-f", "bestaudio/best",
                           "--no-warnings", "-o", str(audio_out), url])
        if getattr(audio, "returncode", 1) != 0:
            return {"ok": False,
                    "error": "audio download failed: "
                    + str(getattr(audio, "stderr", "") or "")[:300]}
        proxy = self._run([self.ytdlp, "-f",
                           "best[height<=360]/best",
                           "--no-warnings", "-o", str(proxy_out), url])
        proxy_fallback = False
        if getattr(proxy, "returncode", 1) != 0:
            # Single-format sources (a direct mp4 link) have no
            # height-filtered variant — the only format IS the proxy.
            proxy = self._run([self.ytdlp, "-f", "best",
                               "--no-warnings", "-o", str(proxy_out), url])
            proxy_fallback = True
        if getattr(proxy, "returncode", 1) != 0:
            return {"ok": False,
                    "error": "proxy download failed: "
                    + str(getattr(proxy, "stderr", "") or "")[:300]}
        audios = sorted(job_dir.glob("audio.*"))
        proxies = sorted(job_dir.glob("proxy.*"))
        if not audios or not proxies:
            return {"ok": False,
                    "error": "yt-dlp finished but left no files"}
        return {"ok": True, "audio": str(audios[0]),
                "proxy": str(proxies[0]),
                "proxy_fallback": proxy_fallback}

    @staticmethod
    def _extract_wav(audio_path: str, wav_path: Path) -> bool:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-vn",
             "-ac", "1", "-ar", "16000", str(wav_path)],
            capture_output=True, text=True, timeout=300)
        return proc.returncode == 0 and wav_path.is_file()

    @staticmethod
    def loudness_series(wav_path: str | Path) -> list[float]:
        """dB RMS per second from a 16k mono wav — pure stdlib."""
        series: list[float] = []
        with wave.open(str(wav_path), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            frames = wav.getnframes()
            per_second = rate
            for start in range(0, frames, per_second):
                chunk = wav.readframes(min(per_second, frames - start))
                if width == 2:
                    samples = struct.unpack(
                        f"<{len(chunk) // 2}h", chunk)
                elif width == 1:
                    samples = tuple((b - 128) * 256 for b in chunk)
                else:
                    samples = ()
                if channels > 1 and samples:
                    samples = samples[::channels]
                if not samples:
                    series.append(-60.0)
                    continue
                mean_sq = sum(float(s) * float(s) for s in samples) \
                    / len(samples)
                rms = math.sqrt(mean_sq) / 32768.0
                series.append(round(20 * math.log10(rms), 1)
                              if rms > 0 else -60.0)
        return series

    @staticmethod
    def _transcript_lines(words: list[dict[str, Any]]) -> str:
        """Words grouped into timestamped lines for the brain prompt."""
        lines: list[str] = []
        current: list[str] = []
        current_start = 0.0
        for word in words:
            start = float(word.get("s", 0.0))
            if current and start - current_start > 8.0:
                lines.append(f"{current_start:.0f}s: "
                             + " ".join(current))
                current = []
                current_start = start
            if not current:
                current_start = start
            current.append(str(word.get("w", "")))
        if current:
            lines.append(f"{current_start:.0f}s: " + " ".join(current))
        return "\n".join(lines)

    # -- analyze -------------------------------------------------------
    def analyze(self, source_url: str, topic: TopicConfig, *,
                strategy: str | None = None,
                model: str = "small",
                rights_category: str = "third_party_review_draft",
                brain: bool = True) -> TrendclipJob:
        """Ingest one source, transcribe it, score its moments with both
        rungs, and leave the job ``ready`` with its drafts."""
        job = self.jobs.create(topic=topic.slug, source_url=source_url)
        job_dir = self.cache_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._analyze(job, job_dir, source_url, topic,
                                 strategy=strategy or topic.strategy,
                                 model=model,
                                 rights_category=rights_category,
                                 brain=brain)
        except Exception as exc:  # noqa: BLE001 - a failed job is a receipt, not a crash
            self.jobs.set_stage(job, "failed", error=str(exc))
            job.receipt["error"] = str(exc)
            self.jobs.save(job)
            return job

    def _analyze(self, job: TrendclipJob, job_dir: Path, source_url: str,
                 topic: TopicConfig, *, strategy: str, model: str,
                 rights_category: str, brain: bool) -> TrendclipJob:
        rights = validate_rights({"category": rights_category},
                                 source_url=source_url)
        self.jobs.set_stage(job, "ingesting")
        info = inspect_youtube_url(source_url, runner=self.runner,
                                   ytdlp=self.ytdlp)
        downloaded = self._download(source_url, job_dir)
        if not downloaded.get("ok"):
            raise RuntimeError(str(downloaded.get("error")
                                   or "download failed"))
        wav_path = job_dir / "audio16k.wav"
        if not self._extract_wav(str(downloaded["audio"]), wav_path):
            raise RuntimeError("audio extraction failed (ffmpeg)")
        self.jobs.set_stage(job, "transcribing")
        if self.transcriber is not None:
            transcribed = self.transcriber(str(wav_path), language="en",
                                           model=model)
        else:
            from .provision import ensure_faster_whisper
            if not ensure_faster_whisper():
                raise RuntimeError(
                    "Faster Whisper is not installed and provisioning "
                    "failed — transcription needs it")
            transcribed = transcribe_audio(str(wav_path), language="en",
                                           model=model)
        words = transcribed.get("words") or []
        duration = float(info.get("duration") or 0.0) or (
            float(words[-1].get("e", 0.0)) if words else 0.0)
        self.jobs.set_stage(job, "analyzing")
        loudness = self.loudness_series(wav_path)
        heat = _render.visual_signals(str(downloaded["proxy"]),
                                      len(loudness))
        moments = _scoring.select_moments(
            loudness=loudness, heat=heat, words=words, strategy=strategy,
            target_seconds=topic.target_clip_seconds,
            min_seconds=topic.min_clip_seconds,
            max_seconds=topic.max_clip_seconds)
        drafts = _scoring.assemble_drafts(moments)
        transcript = self._transcript_lines(words)
        brain_result: dict[str, Any] = {"picks": [], "error": None,
                                        "ran": False}
        if brain and self.pick_fn is not None:
            brain_result = brain_pick_moments(
                self.pick_fn, topic=topic.topic_name,
                transcript=transcript, candidates=drafts)
            brain_result["ran"] = True
            for pick in brain_result.get("picks") or []:
                cand = pick.get("candidate") or {}
                cand["brain_pick"] = True
                cand["brain_reason"] = pick.get("reason")
        elif brain and self.pick_fn is None:
            brain_result = {
                "picks": [], "ran": False,
                "error": ("no brain consult supplied — the brain rung did "
                          "not run; deterministic picks only")}
        job.receipt = {
            "source": {
                "url": source_url,
                "title": str(info.get("title") or ""),
                "uploader": str(info.get("uploader") or ""),
                "duration": duration,
                "view_count": int(info.get("view_count") or 0),
            },
            "rights": rights,
            "files": {
                "audio": str(downloaded["audio"]),
                "proxy": str(downloaded["proxy"]),
                "wav": str(wav_path),
                "proxy_fallback": bool(downloaded.get("proxy_fallback",
                                                      False)),
            },
            "transcript": {
                "words": len(words),
                "text_chars": len(transcribed.get("text") or ""),
                "backend": str(transcribed.get("backend") or ""),
                "model": str(transcribed.get("model") or model),
            },
            "signals": {
                "seconds": len(loudness),
                "heat_available": any(v > 0 for v in heat),
                "heat_note": ("visual motion measured"
                              if any(v > 0 for v in heat) else
                              "no visual signal — OpenCV unavailable or "
                              "no motion detected"),
            },
            "strategy": strategy,
            "moments": drafts,
            "editorial_summary": _scoring.editorial_summary(drafts),
            "editorial_issues": _scoring.editorial_issues(drafts),
            "brain": brain_result,
            "transcript_lines": transcript,
        }
        self.jobs.set_stage(job, "ready")
        return job

    # -- render --------------------------------------------------------
    def render(self, job_id: str, pick_n: int, *,
               layout: str = "balanced",
               destinations: list[str] | None = None,
               rights_category: str | None = None) -> TrendclipJob:
        """Render one picked moment into a validated clip + package."""
        job = self.jobs.get(job_id)
        receipt = job.receipt or {}
        if job.stage not in ("ready", "complete"):
            raise RuntimeError(
                f"job {job_id} is {job.stage} — analyze it first")
        moments = receipt.get("moments") or []
        moment = next((m for m in moments if int(m.get("n", -1)) == pick_n),
                      None)
        if moment is None:
            raise RuntimeError(f"job {job_id} has no moment #{pick_n}")
        rights = receipt.get("rights") or validate_rights(
            {"category": rights_category or "third_party_review_draft"},
            source_url=job.source_url)
        if not rights.get("ok"):
            raise RuntimeError(
                "render blocked: "
                + "; ".join(rights.get("reasons") or ["no rights"]))
        self.jobs.set_stage(job, "rendering")
        try:
            return self._render(job, moment, rights, layout=layout,
                                destinations=destinations)
        except Exception as exc:  # noqa: BLE001
            self.jobs.set_stage(job, "failed", error=str(exc))
            job.receipt["render_error"] = str(exc)
            self.jobs.save(job)
            return job

    def _render(self, job: TrendclipJob, moment: dict[str, Any],
                rights: dict[str, Any], *, layout: str,
                destinations: list[str] | None) -> TrendclipJob:
        receipt = job.receipt or {}
        files = receipt.get("files") or {}
        job_dir = self.cache_dir / job.job_id
        start = float(moment.get("start", 0.0))
        end = float(moment.get("end", 0.0))
        section_audio = job_dir / "section.wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start),
             "-i", str(files.get("audio") or ""), "-vn",
             "-ac", "2", "-ar", "48000", str(section_audio)],
            capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not section_audio.is_file():
            raise RuntimeError("section audio extraction failed (ffmpeg)")
        clip_path = self.clips_dir / (job.job_id + ".mp4")
        hook_text = str(moment.get("brain_reason") or "")[:80] or (
            receipt.get("source", {}).get("title") or "")
        words = moment.get("words") or []
        rendered = _render.render_vertical_clip(
            video_path=str(files.get("proxy") or ""),
            audio_path=str(section_audio),
            words=words, start=start, end=end,
            output_path=clip_path, layout=layout, hook_text=hook_text)
        if not rendered.get("ok"):
            raise RuntimeError(str(rendered.get("error") or "render failed"))
        self.jobs.set_stage(job, "validating")
        validation = _validate.validate_clip(clip_path)
        thumb_path = job_dir / "thumbnail.jpg"
        thumb = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start + 1.0),
             "-i", str(files.get("proxy") or ""), "-vframes", "1",
             str(thumb_path)],
            capture_output=True, text=True, timeout=120)
        copy: dict[str, Any] = {}
        if self.copy_fn is not None:
            copy = brain_platform_copy(
                self.copy_fn, topic=receipt.get("source", {}).get("title")
                or job.topic,
                title=receipt.get("source", {}).get("title") or job.topic,
                transcript=receipt.get("transcript_lines") or "")
        package = export_package(
            clip_path=clip_path, words=words, job_id=job.job_id,
            topic=job.topic, source_url=job.source_url,
            rights_label=str(rights.get("label") or ""),
            moment=moment, destinations=destinations, copy=copy,
            thumbnail_path=str(thumb_path) if thumb.returncode == 0
            and thumb_path.is_file() else None,
            out_dir=self.exports_dir / job.job_id)
        job.receipt["render"] = {
            "clip_path": str(clip_path),
            "duration": rendered.get("duration"),
            "frames": rendered.get("frames"),
            "layout": rendered.get("layout"),
            "moment_n": moment.get("n"),
            "validation": validation,
            "package": package,
        }
        job.receipt["validation"] = validation
        if not validation.get("passed"):
            self.jobs.set_stage(
                job, "failed",
                error="validation failed: "
                + "; ".join(validation.get("problems") or ["unknown"]))
            return job
        self.jobs.set_stage(job, "complete")
        return job

    # -- queue + repurpose ---------------------------------------------
    def queue(self, job_id: str, *, platforms: list[str] | None = None,
              scheduled_for: str = "", style: str = "") -> dict[str, Any]:
        """Queue a rendered clip into the shorts calendar for the
        publisher sweep. The sweep uses the pre-rendered clip (see
        ``prerendered_clip_for_job``) — it does not re-render it. Manual
        publish discipline stands: the post only fires because the owner
        queued it, and the sweep's ledger records the delivery."""
        job = self.jobs.get(job_id)
        if job.stage != "complete":
            raise RuntimeError(
                f"job {job_id} is {job.stage} — render it first")
        from ...media.contentops.pipeline import ContentCalendar
        self.shorts_runs_root.mkdir(parents=True, exist_ok=True)
        calendar = ContentCalendar(self.shorts_runs_root)
        topic = (job.receipt.get("source") or {}).get("title") or job.topic
        post = calendar.add("trendclip", topic,
                            platforms=list(platforms or []),
                            scheduled_for=scheduled_for, style=style)
        # The post stays "queued" — that is the status the sweep's due()
        # returns — with the trendclip job linked, so the sweep uses the
        # pre-rendered clip instead of rendering a scripted short.
        calendar.mark(post.id, "queued", job_id=job.job_id)
        job.receipt.setdefault("queue", []).append({
            "post_id": post.id,
            "platforms": list(platforms or []),
            "scheduled_for": scheduled_for,
        })
        self.jobs.save(job)
        return {"post_id": post.id, "job_id": job.job_id,
                "platforms": list(platforms or []),
                "scheduled_for": scheduled_for}

    def repurpose_source(self, job_id: str) -> dict[str, Any]:
        """The rendered clip as a repurpose input: a real video + its
        transcript + its title, for the repurpose organ to adapt into a
        text post, a thread, or a longer piece."""
        job = self.jobs.get(job_id)
        receipt = job.receipt or {}
        render_info = receipt.get("render") or {}
        return {
            "source_kind": "trendclip-clip",
            "source_handle": job.job_id,
            "title": (receipt.get("source") or {}).get("title")
            or job.topic,
            "clip_path": str(render_info.get("clip_path") or ""),
            "transcript": str(receipt.get("transcript_lines") or ""),
            "source_url": job.source_url,
            "rights": (receipt.get("rights") or {}).get("label") or "",
        }

    # -- status --------------------------------------------------------
    def status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        receipt = job.receipt or {}
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "topic": job.topic,
            "source_url": job.source_url,
            "error": job.error,
            "moments": len(receipt.get("moments") or []),
            "brain_ran": bool((receipt.get("brain") or {}).get("ran")),
            "brain_picks": len((receipt.get("brain") or {}).get("picks")
                               or []),
            "clip": (receipt.get("render") or {}).get("clip_path") or "",
            "validation_passed": bool(
                (receipt.get("validation") or {}).get("passed")),
        }

    def list_jobs(self, *, stage: str = "") -> list[dict[str, Any]]:
        return [self.status(j.job_id) for j in self.jobs.list(stage=stage)]
