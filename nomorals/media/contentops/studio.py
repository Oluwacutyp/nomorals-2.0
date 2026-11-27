"""Animato Studio — one persistent project per series, and the spark → build → share flow.

A :class:`StudioProject` holds everything a series needs to keep making
episodes the same way: the cast (characters + reference images + voices),
the scene plan's home (niche/style), voice and music choices, the channel
it posts to, and the schedule it runs on. A project is a manifest + a
directory, so it is versioned and re-runnable — an episode is just another
run of the same project.

The runner (:class:`StudioRunner`) composes organs that already exist:
the shorts factory's :class:`ShortPipeline` renders, the YouTube
publisher posts, and the file itself is the export. This module renders
nothing itself; the composition is the build. Where a step cannot run —
no brain, no voice, no ffmpeg, no YouTube credentials — the run fails
closed with a plain message naming what is missing. Nothing is faked:
a failed publish is reported, never claimed, and the finished video is
still handed back as the export.

Seams for the later port steps (kept clean, not built here): the
``schedule`` field is what the auto-post mission (step 2) will read,
``topic_queue`` is what the series templates (step 3) will fill, and
``face_ref`` on the cast is what character-consistency hardening
(step 4) will lock on. The GitHub-studio export (step 5) will read the
manifest this module writes.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger

_log = get_logger(__name__)

#: Default studio root — a sibling of the shorts factory's runs root,
#: so studio jobs and shorts jobs share one ledger (``nm shorts status``).
DEFAULT_STUDIO_ROOT = Path.home() / "workspace" / "devon" / ".studio"


def _utcnow() -> str:
    # NOTE: timezone.utc, not datetime.UTC — this file imports the
    # datetime CLASS, and the UTC alias lives on the datetime MODULE.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def slugify(name: str) -> str:
    """Project name → filesystem-safe slug (``My Cooking Show`` → ``my-cooking-show``)."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "studio"


def new_episode_id() -> str:
    """Unique episode id."""
    return "ep_" + uuid.uuid4().hex[:10]


@dataclass
class CastMember:
    """One character in a project's cast.

    ``voice`` is the voice the render speaks as (``"edge"`` = free Edge
    TTS first, catalogue on failure; anything else = a catalogue voice
    name). ``face_ref`` is the reference image for this character —
    recorded now, locked on by character-consistency hardening (port
    step 4); it does not change the step-1 render.
    """

    name: str
    voice: str = "edge"
    face_ref: str = ""
    character_id: str = ""  # "" = ad-hoc cast member, not a Character

    def to_dict(self) -> dict:
        """JSON-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CastMember:
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Episode:
    """One finished (or failed) episode of a project."""

    id: str
    topic: str
    job_id: str = ""
    run_id: str = ""
    video_path: str = ""
    status: str = "queued"  # queued|rendering|ready|posted|failed
    published: dict = field(default_factory=dict)  # platform → video id/url
    error: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """JSON-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class StudioProject:
    """One series: everything an episode run needs, persisted as JSON.

    ``niche`` is the shorts-factory niche the episodes render in
    (``did_you_know`` by default); ``style``/``voice``/``footage`` are
    the render knobs the factory already understands. ``music`` is the
    music-bed preset the series wants — recorded now, wired when the
    series templates land (port step 3). ``channel`` names the YouTube
    channel the series posts to (the OAuth account decides the real
    destination today). ``schedule`` is the cadence string the auto-post
    mission (port step 2) will read (``"daily"``, ``"weekly"``, or an
    ISO time). ``topic_queue`` holds episode ideas in order; a spark
    with no idea takes the next queued topic.
    """

    slug: str
    name: str
    niche: str = "did_you_know"
    style: str = ""
    voice: str = "edge"
    footage: str = "auto"
    music: str = ""
    channel: str = ""
    schedule: str = ""
    cast: list = field(default_factory=list)  # list[CastMember]
    topic_queue: list = field(default_factory=list)  # list[str]
    episodes: list = field(default_factory=list)  # list[Episode]
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        """JSON-safe."""
        d = asdict(self)
        d["cast"] = [c.to_dict() if isinstance(c, CastMember) else c
                     for c in self.cast]
        d["episodes"] = [e.to_dict() if isinstance(e, Episode) else e
                         for e in self.episodes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StudioProject:
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in known}
        data["cast"] = [CastMember.from_dict(c) if isinstance(c, dict) else c
                        for c in data.get("cast") or []]
        data["episodes"] = [Episode.from_dict(e) if isinstance(e, dict) else e
                            for e in data.get("episodes") or []]
        return cls(**data)

    def episode(self, episode_id: str) -> Episode:
        """Find an episode by id (or id prefix); raise KeyError."""
        for ep in self.episodes:
            if ep.id == episode_id or ep.id.startswith(episode_id):
                return ep
        raise KeyError(f"unknown episode {episode_id!r} in project {self.slug!r}")


class StudioStore:
    """JSON-per-project persistence under ``<root>/projects/<slug>/``."""

    def __init__(self, root: str | os.PathLike = "") -> None:
        self.root = Path(root) if root else DEFAULT_STUDIO_ROOT
        self.dir = self.root / "projects"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, slug: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", slug)
        return self.dir / safe / "manifest.json"

    def save(self, project: StudioProject) -> Path:
        """Write (atomically) and return the manifest path."""
        project.updated_at = _utcnow()
        p = self._path(project.slug)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    def load(self, slug: str) -> StudioProject:
        """Raise FileNotFoundError for unknown slugs."""
        p = self._path(slug)
        if not p.exists():
            raise FileNotFoundError(f"unknown studio project {slug!r}")
        return StudioProject.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self) -> list[StudioProject]:
        """All projects, newest first (skips corrupt manifests)."""
        projects = []
        for p in sorted(self.dir.glob("*/manifest.json"), reverse=True):
            try:
                projects.append(StudioProject.from_dict(
                    json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        projects.sort(key=lambda pr: pr.created_at or "", reverse=True)
        return projects

    def create(self, name: str, **fields: Any) -> StudioProject:
        """Create a project; raise ValueError when the slug is taken."""
        slug = slugify(name)
        if self._path(slug).exists():
            raise ValueError(f"studio project {slug!r} already exists")
        project = StudioProject(slug=slug, name=name,
                                created_at=_utcnow(), **fields)
        self.save(project)
        _log.info("studio: created project %s (%s)", slug, name)
        return project

    def queue_topic(self, slug: str, topic: str) -> StudioProject:
        """Append an episode idea to a project's queue; returns it."""
        project = self.load(slug)
        project.topic_queue.append(topic)
        self.save(project)
        return project


def _load_post_copy(pipeline: Any, job: Any) -> dict:
    """Title/description/hashtags the pipeline wrote for this job."""
    runs_root = getattr(pipeline, "runs_root", None)
    run_id = getattr(job, "run_id", "") or ""
    if runs_root and run_id:
        pc = Path(runs_root) / run_id / "post_copy.json"
        if pc.exists():
            try:
                return json.loads(pc.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a bad copy file is not fatal
                pass
    return {"title": getattr(job, "topic", ""), "description": "",
            "hashtags": []}


class StudioRunner:
    """Spark → build → share, composing the organs that already exist.

    ``pipeline`` / ``publisher_factory`` are constructor-injectable;
    when omitted the real subsystems are wired lazily (the shorts
    factory's :class:`ShortPipeline` and the YouTube publisher).
    ``spark()`` never raises for a failed render or publish — it
    returns the honest rung dict instead.
    """

    def __init__(self, root: str | os.PathLike = "",
                 *, pipeline: Any = None,
                 publisher_factory: Callable[[], Any] | None = None) -> None:
        self.store = StudioStore(root)
        self._pipeline = pipeline
        self._publisher_factory = publisher_factory

    @property
    def pipeline(self) -> Any:
        """The real shorts pipeline, built on first use."""
        if self._pipeline is None:
            from .pipeline import ShortPipeline

            self._pipeline = ShortPipeline()
        return self._pipeline

    def _publisher(self) -> Any:
        if self._publisher_factory is not None:
            return self._publisher_factory()
        from .publish.youtube import YouTubePublisher

        return YouTubePublisher()

    # -- the loop ----------------------------------------------------
    def spark(self, project: StudioProject | str, topic: str = "", *,
              publish: str = "", private: bool = False) -> dict[str, Any]:
        """Run one episode end to end: spark in, finished video out.

        ``project`` is a slug or a :class:`StudioProject`. ``topic`` is
        the spark; with no topic the next queued episode idea is used.
        ``publish`` is ``""`` (export only — the file path is the
        result) or ``"youtube"`` (post straight to the channel).
        Returns ``{"ok", "project", "episode_id", "job_id", "run_id",
        "video_path", "published", "rungs", "notes"}`` — every rung
        named honestly.
        """
        rungs: dict[str, str] = {}
        notes: list[str] = []
        if isinstance(project, str):
            try:
                project = self.store.load(project)
            except FileNotFoundError:
                return {"ok": False, "project": project, "episode_id": "",
                        "job_id": "", "run_id": "", "video_path": "",
                        "published": {}, "rungs": {"project": "unknown project"},
                        "notes": [f"no studio project {project!r} — "
                                  "create it first (`animato new`)"]}
        else:
            # Re-load: the caller's copy may predate a queue_topic or an
            # episode another run recorded. Unsaved projects keep the
            # passed object (a fresh project has nothing to reload).
            with contextlib.suppress(FileNotFoundError):
                project = self.store.load(project.slug)
        topic = (topic or "").strip()
        if not topic and project.topic_queue:
            topic = project.topic_queue[0]
        if not topic:
            return {"ok": False, "project": project.slug, "episode_id": "",
                    "job_id": "", "run_id": "", "video_path": "",
                    "published": {}, "rungs": {"spark": "no idea"},
                    "notes": ["no spark — give an idea or queue a topic "
                              f"(`animato queue {project.slug} \"<topic>\")"]}
        episode = Episode(id=new_episode_id(), topic=topic,
                          created_at=_utcnow())
        project.episodes.append(episode)
        voice = (project.cast[0].voice if project.cast
                 else project.voice) or "edge"
        try:
            job = self.pipeline.plan(project.niche, topic, style=project.style,
                                     footage=project.footage, voice=voice)
        except Exception as exc:  # noqa: BLE001 — planning must not raise
            episode.status = "failed"
            episode.error = str(exc)
            self.store.save(project)
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id, "job_id": "", "run_id": "",
                    "video_path": "", "published": {},
                    "rungs": {"plan": f"failed: {exc}"},
                    "notes": [f"could not plan the episode: {exc}"]}
        episode.job_id = job.id
        rungs["plan"] = f"job {job.id}"
        try:
            preflight = self.pipeline.preflight(job)
        except Exception as exc:  # noqa: BLE001
            preflight = {"ok": False, "checks": [
                {"name": "preflight", "ok": False, "detail": str(exc)}]}
        if not preflight.get("ok"):
            failed = [c for c in preflight.get("checks", []) if not c.get("ok")]
            detail = "; ".join(
                f"{c.get('name')}: {c.get('detail')}" for c in failed) or "unknown"
            episode.status = "failed"
            episode.error = f"preflight failed: {detail}"
            self.store.save(project)
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id, "job_id": job.id, "run_id": "",
                    "video_path": "", "published": {},
                    "rungs": {"preflight": f"failed: {detail}"},
                    "notes": [f"preflight failed — fix this first: {detail}"]}
        rungs["preflight"] = "passed"
        episode.status = "rendering"
        self.store.save(project)
        try:
            result = self.pipeline.run(job)
        except Exception as exc:  # noqa: BLE001 — run() promises not to raise
            result = None
            episode.status = "failed"
            episode.error = str(exc)
            self.store.save(project)
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id, "job_id": job.id, "run_id": "",
                    "video_path": "", "published": {},
                    "rungs": {"render": f"failed: {exc}"},
                    "notes": [f"render crashed: {exc}"]}
        episode.run_id = result.run_id
        if not result.ok:
            err = result.error.to_dict() if result.error else {}
            episode.status = "failed"
            episode.error = str(err.get("message") or err or "render failed")
            self.store.save(project)
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id, "job_id": job.id,
                    "run_id": result.run_id, "video_path": "",
                    "published": {},
                    "rungs": {"render": f"failed: {episode.error}"},
                    "notes": [f"render failed: {episode.error}"]}
        episode.video_path = result.final_path or ""
        episode.status = "ready"
        rungs["render"] = f"→ {episode.video_path}"
        if project.topic_queue and project.topic_queue[0] == topic:
            project.topic_queue.pop(0)
        published: dict[str, Any] = {}
        if publish == "youtube":
            pub = self.publish_video(episode.video_path, job,
                                     private=private)
            if pub.get("ok"):
                episode.status = "posted"
                episode.published["youtube"] = pub.get("video_id") or pub.get("url") or ""
                published["youtube"] = episode.published["youtube"]
                rungs["publish"] = "posted to youtube"
                notes.append(f"posted to youtube: {published['youtube']}")
            else:
                rungs["publish"] = f"not posted: {pub.get('reason', '')}"
                notes.append(f"publish skipped: {pub.get('reason', '')} — "
                             f"the video is still at {episode.video_path}")
        else:
            rungs["publish"] = "skipped (export only)"
            notes.append(f"exported: {episode.video_path}")
        self.store.save(project)
        return {"ok": True, "project": project.slug,
                "episode_id": episode.id, "job_id": job.id,
                "run_id": result.run_id, "video_path": episode.video_path,
                "published": published, "rungs": rungs, "notes": notes}

    # -- share -------------------------------------------------------
    def publish_video(self, video_path: str, job: Any, *,
                      private: bool = False) -> dict[str, Any]:
        """Post a rendered video to YouTube (honest failure, never faked)."""
        if not video_path or not Path(video_path).is_file():
            return {"ok": False, "reason": f"no rendered video at {video_path!r}"}
        copy = _load_post_copy(self.pipeline, job)
        try:
            publisher = self._publisher()
            result = publisher.publish(
                video_path, copy.get("title") or job.topic,
                description=copy.get("description", ""),
                tags=copy.get("hashtags", []) or [],
                privacy="private" if private else "public",
                confirmed=True)
            return {"ok": True, "video_id": result.get("video_id", ""),
                    "url": result.get("url", ""), "result": result}
        except Exception as exc:  # noqa: BLE001 — publish must not raise
            return {"ok": False, "reason": str(exc)}

    def publish_episode(self, project: StudioProject | str,
                        episode_id: str, *, private: bool = False
                        ) -> dict[str, Any]:
        """Publish an already-rendered episode to YouTube."""
        if isinstance(project, str):
            project = self.store.load(project)
        episode = project.episode(episode_id)
        if episode.status not in ("ready", "posted"):
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id,
                    "reason": f"episode is '{episode.status}', not ready — "
                              "render it first"}
        if not episode.video_path or not Path(episode.video_path).is_file():
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id,
                    "reason": "no rendered video file for this episode"}
        fake_job = _FakeJobShim(episode)
        pub = self.publish_video(episode.video_path, fake_job,
                                 private=private)
        if pub.get("ok"):
            episode.status = "posted"
            episode.published["youtube"] = pub.get("video_id") or pub.get("url") or ""
            self.store.save(project)
            return {"ok": True, "project": project.slug,
                    "episode_id": episode.id,
                    "video_id": episode.published["youtube"]}
        return {"ok": False, "project": project.slug,
                "episode_id": episode.id, "reason": pub.get("reason", "")}

    def export_episode(self, project: StudioProject | str,
                       episode_id: str) -> dict[str, Any]:
        """Hand back an already-rendered episode's file path."""
        if isinstance(project, str):
            project = self.store.load(project)
        episode = project.episode(episode_id)
        if not episode.video_path or not Path(episode.video_path).is_file():
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id,
                    "reason": "no rendered video file for this episode"}
        return {"ok": True, "project": project.slug,
                "episode_id": episode.id, "path": episode.video_path}


class _FakeJobShim:
    """Minimal job-shaped view of an episode for post-copy lookup."""

    def __init__(self, episode: Episode) -> None:
        self.topic = episode.topic
        self.run_id = episode.run_id
