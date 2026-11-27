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
``schedule`` field is what the auto-post mission (step 2) reads,
``topic_queue`` is what the series templates (step 3) fill, and
``face_ref`` on the cast is what character-consistency hardening
(step 4) locks on — resolved by :func:`resolve_cast_lock` and
carried into the render by :meth:`StudioRunner.spark`. The
GitHub-studio export (step 5) will read the manifest this module
writes.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
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

#: Default YouTube publisher factory, installed by the composition
#: root (:func:`nomorals.agents.context.build_context`) — the L4
#: studio may not import the L5 publisher itself, so the factory
#: lives here as a slot and the layer above fills it. ``None`` =
#: unwired; publishing then fails closed naming the missing wire.
_DEFAULT_PUBLISHER_FACTORY: Callable[[], Any] | None = None


def set_publisher_factory(factory: Callable[[], Any] | None) -> None:
    """Install (or clear, with None) the default publisher factory.

    Called by the composition root with the real ``YouTubePublisher``
    class; tests pass a fake (or clear it). A runner built with an
    explicit ``publisher_factory`` never consults this slot.
    """
    global _DEFAULT_PUBLISHER_FACTORY
    _DEFAULT_PUBLISHER_FACTORY = factory


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


#: image suffixes a face reference may carry
_FACE_REF_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _member_field(member: Any, key: str) -> str:
    """One cast field off a CastMember or a plain dict."""
    if isinstance(member, dict):
        return str(member.get(key) or "")
    return str(getattr(member, key, "") or "")


def resolve_cast_lock(cast: list) -> tuple[list[dict], str]:
    """Validate a project's cast face references into a render lock.

    Returns ``(lock, error)`` — ``lock`` is a list of ``{"name",
    "voice", "character_id", "face_ref"}`` dicts with absolute
    reference paths, in cast order; ``error`` is ``""`` on success
    and a plain named reason otherwise. A member with no ``face_ref``
    is skipped (a voice-only member needs no lock). A member whose
    reference is missing, not a file, empty, wrong-suffixed, or
    unreadable as an image fails the WHOLE lock — the run must not
    render that character as someone else, or as nobody. Without
    Pillow only existence/size/suffix are checked; with it the file
    must also decode as an image.
    """
    lock: list[dict] = []
    for member in cast or []:
        name = _member_field(member, "name")
        face_ref = _member_field(member, "face_ref")
        if not face_ref.strip():
            continue
        p = Path(face_ref).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            return [], (f"face reference for {name!r} not found: "
                        f"{face_ref!r}")
        if not p.is_file():
            return [], (f"face reference for {name!r} is not a file: "
                        f"{face_ref!r}")
        if p.stat().st_size == 0:
            return [], (f"face reference for {name!r} is empty: "
                        f"{face_ref!r}")
        if p.suffix.lower() not in _FACE_REF_EXTS:
            return [], (f"face reference for {name!r} is not an image "
                        f"file: {face_ref!r}")
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[assignment]
        if Image is not None:
            try:
                with Image.open(p) as img:
                    img.verify()
            except Exception as exc:
                return [], (f"face reference for {name!r} is not a "
                            f"readable image: {face_ref!r} ({exc})")
        lock.append({"name": name,
                     "voice": _member_field(member, "voice") or "edge",
                     "character_id": _member_field(member, "character_id"),
                     "face_ref": str(p)})
    return lock, ""


@dataclass
class CastMember:
    """One character in a project's cast.

    ``voice`` is the voice the render speaks as (``"edge"`` = free Edge
    TTS first, catalogue on failure; anything else = a catalogue voice
    name). ``face_ref`` is the reference image for this character —
    resolved and locked into every scene's render by
    character-consistency hardening (port step 4): a missing or
    unreadable reference fails the spark closed before any render
    spend, it is never silently dropped or replaced.
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

    def add_cast(self, slug: str, member: CastMember) -> StudioProject:
        """Add a cast member (same name replaces); returns the project.

        The face reference is stored as an absolute path at write
        time, so the lock is unambiguous when the episode renders
        later. Existence/readability is checked at spark time
        (:func:`resolve_cast_lock`), not here — a member may be
        cast before its reference image lands.
        """
        project = self.load(slug)
        if member.face_ref:
            member.face_ref = str(
                Path(member.face_ref).expanduser().resolve())
        for i, existing in enumerate(project.cast):
            if isinstance(existing, CastMember) \
                    and existing.name == member.name:
                project.cast[i] = member
                break
        else:
            project.cast.append(member)
        self.save(project)
        _log.info("studio: cast %s → %s (%s)", slug, member.name,
                  member.face_ref or "no reference")
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


def _cast_local_name(member: dict) -> str:
    """Repo-local copy name for a cast member's reference image."""
    return slugify(member["name"]) + Path(member["face_ref"]).suffix.lower()


def _export_brief(project: StudioProject, lock: list[dict]) -> str:
    """The project's brief: everything an episode run needs, in words."""
    lines = [f"# {project.name}", "",
             f"Studio project `{project.slug}` — exported from Devon "
             "(Animato Studio). The manifest beside this brief "
             "(`manifest.json`) is the source of truth; this brief is "
             "the human-readable copy of it.", "",
             "## Series",
             f"- niche: {project.niche}",
             f"- style: {project.style or '(default)'}",
             f"- voice: {project.voice}",
             f"- footage: {project.footage}",
             f"- music: {project.music or '(none)'}",
             f"- channel: {project.channel or '(not set)'}",
             f"- schedule: {project.schedule or '(not set)'}",
             ""]
    if lock:
        lines.append("## Cast")
        for member in lock:
            extra = (f", character {member['character_id']}"
                     if member["character_id"] else "")
            lines.append(f"- {member['name']} — voice "
                         f"{member['voice']}{extra}, reference "
                         f"cast/{_cast_local_name(member)}")
        lines.append("")
    lines.append("## Topic queue")
    for i, topic in enumerate(project.topic_queue, 1):
        lines.append(f"{i}. {topic}")
    lines.append("")
    if project.episodes:
        lines.append("## Episodes (history — re-rendered from the queue)")
        for ep in project.episodes:
            lines.append(f"- {ep.id} — {ep.status} — {ep.topic}")
        lines.append("")
    lines.append("## Rebuild")
    lines.append("Run `./render.sh` — it renders every queued topic, in "
                 "order, through the same pipeline stages (plan → "
                 "preflight → run). A topic renders with the cast lock, "
                 "so every scene uses the cast's reference images in "
                 "`cast/`.")
    lines.append("")
    return "\n".join(lines)


def _export_readme(project: StudioProject) -> str:
    """Short operational readme for the exported repo."""
    return (f"# {project.name} — Animato studio export\n\n"
            f"A self-contained studio repo for the `{project.slug}` "
            "series: the brief (`BRIEF.md`), the manifest "
            "(`manifest.json`), and the render scripts (`render.py`, "
            "`render.sh`).\n\n"
            "## Rebuild the videos\n\n"
            "From a checkout of the nomorals-2.0 repo (or with "
            "`nomorals` on PYTHONPATH):\n\n"
            "```sh\n./render.sh\n```\n\n"
            "That renders every topic in the queue, in order, through "
            "the same pipeline stages Devon uses. One topic only:\n\n"
            "```sh\n./render.sh --topic \"a lighthouse keeper\"\n```\n\n"
            "Finished videos land under the pipeline's runs root; a "
            "failed preflight or render is printed and the script "
            "exits non-zero.\n")


_EXPORT_RENDER_SH = """#!/bin/sh
# Render every queued episode of this studio export.
# Needs `nomorals` importable — run from a checkout of nomorals-2.0
# (or set PYTHONPATH to one). Pass --topic "..." for one episode only.
set -e
cd "$(dirname "$0")"
exec python3 render.py "$@"
"""

#: The render script template. ``__SNAPSHOT_REPR__`` is replaced at
#: export time with a ``repr()`` of the project's manifest JSON — the
#: script then carries the project's real topics and cast as data,
#: not template filler. At run time the manifest beside the script
#: is the source of truth; the snapshot is the fallback when the
#: manifest cannot be read.
_EXPORT_RENDER_PY = '''#!/usr/bin/env python3
"""Render the queued episodes of this studio export.

Generated from the studio manifest (manifest.json beside this script).
It drives the same ShortPipeline stages Devon uses — plan, preflight,
run — one episode per queued topic, with the project's cast lock
applied so every scene renders from the cast's reference images.

Run it from a checkout of the nomorals-2.0 repo (or with `nomorals`
on PYTHONPATH):

    ./render.sh                  # every queued topic, in order
    ./render.sh --topic "TOPIC"  # one topic only

It never fakes a render: a failed preflight or render is printed and
the script exits non-zero; finished videos land under the pipeline's
runs root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from nomorals.media.contentops.pipeline import ShortPipeline
    from nomorals.media.contentops.studio import (
        resolve_cast_lock,
        slugify,
    )
except ImportError:  # pragma: no cover — the import is the environment
    ShortPipeline = None  # type: ignore[assignment]
    resolve_cast_lock = None  # type: ignore[assignment]
    slugify = None  # type: ignore[assignment]

# The project as exported — the manifest beside this script is the
# source of truth at run time; this snapshot is the fallback when the
# manifest cannot be read, and the record of what the export held.
SNAPSHOT = json.loads(__SNAPSHOT_REPR__)


def _load_project() -> dict:
    """The manifest, or the embedded snapshot when it cannot be read."""
    here = Path(__file__).resolve().parent
    manifest = here / "manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"render: manifest.json unreadable ({exc}) — using the "
              "snapshot baked into this script", file=sys.stderr)
        return dict(SNAPSHOT)


def _local_cast(project: dict) -> list[dict]:
    """The cast lock, from the repo-local reference copies when present.

    A member whose reference was copied into ``cast/`` at export
    time renders from that copy — the repo is self-contained — and
    only falls back to the manifest's (absolute) path when no copy
    is present. A reference that is missing or unreadable either
    way fails the render closed before any spend.
    """
    here = Path(__file__).resolve().parent
    cast = []
    for member in project.get("cast") or []:
        face_ref = str(member.get("face_ref") or "")
        if not face_ref:
            continue
        local = here / "cast" / (slugify(str(member.get("name") or ""))
                                 + Path(face_ref).suffix.lower())
        if local.is_file():
            face_ref = str(local)
        cast.append({"name": str(member.get("name") or ""),
                     "voice": str(member.get("voice") or "edge"),
                     "character_id": str(member.get("character_id") or ""),
                     "face_ref": face_ref})
    lock, error = resolve_cast_lock(cast)
    if error:
        print(f"render: cast reference failed — {error}", file=sys.stderr)
        sys.exit(2)
    return lock


def main() -> int:
    """Render the queue, one episode per topic; 0 only if all render."""
    parser = argparse.ArgumentParser(
        description="Render this studio export's queued episodes.")
    parser.add_argument("--topic", default="",
                        help="render one topic instead of the whole queue")
    args = parser.parse_args()
    if ShortPipeline is None:
        print("render: the nomorals package is not importable — run "
              "from a checkout of nomorals-2.0 or set PYTHONPATH",
              file=sys.stderr)
        return 2
    project = _load_project()
    topics = [args.topic] if args.topic else list(
        project.get("topic_queue") or [])
    if not topics:
        print("render: no topics queued — nothing to render",
              file=sys.stderr)
        return 2
    lock = _local_cast(project)
    pipeline = ShortPipeline()
    voice = (lock[0]["voice"] if lock
             else project.get("voice") or "edge")
    failed = 0
    for n, topic in enumerate(topics, 1):
        try:
            job = pipeline.plan(
                project.get("niche") or "did_you_know", topic,
                style=project.get("style") or "",
                footage=project.get("footage") or "auto",
                voice=voice, music=project.get("music") or "",
                cast=lock)
        except Exception as exc:
            print(f"[{n}/{len(topics)}] {topic}: plan failed — {exc}",
                  file=sys.stderr)
            failed += 1
            continue
        preflight = pipeline.preflight(job)
        if not preflight.get("ok"):
            detail = "; ".join(
                f"{c.get('name')}: {c.get('detail')}"
                for c in preflight.get("checks", [])
                if not c.get("ok"))
            print(f"[{n}/{len(topics)}] {topic}: preflight failed — "
                  f"{detail or 'unknown'}", file=sys.stderr)
            failed += 1
            continue
        result = pipeline.run(job)
        if not result.ok:
            error = result.error.to_dict() if result.error else {}
            print(f"[{n}/{len(topics)}] {topic}: render failed — "
                  f"{error.get('message') or error or 'render failed'}",
                  file=sys.stderr)
            failed += 1
            continue
        print(f"[{n}/{len(topics)}] {topic} → {result.final_path}")
    if failed:
        print(f"render: {failed} of {len(topics)} episode(s) failed",
              file=sys.stderr)
        return 1
    print(f"render: {len(topics)} episode(s) done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_export(root: Path, project: StudioProject,
                  lock: list[dict]) -> list[str]:
    """Write every export file under ``root``; return relative paths."""
    (root / "manifest.json").write_text(
        json.dumps(project.to_dict(), indent=2), encoding="utf-8")
    (root / "BRIEF.md").write_text(
        _export_brief(project, lock), encoding="utf-8")
    (root / "README.md").write_text(
        _export_readme(project), encoding="utf-8")
    render = _EXPORT_RENDER_PY.replace(
        "__SNAPSHOT_REPR__",
        repr(json.dumps(project.to_dict(), indent=2)))
    (root / "render.py").write_text(render, encoding="utf-8")
    render_sh = root / "render.sh"
    render_sh.write_text(_EXPORT_RENDER_SH, encoding="utf-8")
    render_sh.chmod(0o755)
    files = ["BRIEF.md", "README.md", "manifest.json", "render.py",
             "render.sh"]
    for member in lock:
        src = Path(member["face_ref"])
        cast_dir = root / "cast"
        cast_dir.mkdir(exist_ok=True)
        shutil.copyfile(src, cast_dir / _cast_local_name(member))
        files.append(f"cast/{_cast_local_name(member)}")
    return sorted(files)


class StudioRunner:
    """Spark → build → share, composing the organs that already exist.

    ``pipeline`` / ``publisher_factory`` are constructor-injectable;
    when omitted the shorts pipeline is wired lazily and the
    publisher comes from the module default installed by the
    composition root (:func:`set_publisher_factory`) — with neither,
    publishing fails closed naming the missing wire instead of
    importing the publisher itself.
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
        # The publisher is injected, never imported here: this module
        # is L4 and the publisher organ is L5. The constructor factory
        # wins; else the module default the composition root installed;
        # else fail closed naming the missing wire.
        factory = self._publisher_factory or _DEFAULT_PUBLISHER_FACTORY
        if factory is None:
            raise RuntimeError(
                "no YouTube publisher configured — pass "
                "publisher_factory=... to StudioRunner (or wire it via "
                "set_publisher_factory at the composition root)")
        return factory()

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
        # The cast lock resolves BEFORE plan() — a broken reference
        # fails the episode before any render spend, and a locked
        # cast is carried into the job so every scene renders from
        # its reference. A member with no face_ref needs no lock.
        lock, lock_error = resolve_cast_lock(project.cast)
        if lock_error:
            episode.status = "failed"
            episode.error = lock_error
            self.store.save(project)
            return {"ok": False, "project": project.slug,
                    "episode_id": episode.id, "job_id": "", "run_id": "",
                    "video_path": "", "published": {},
                    "rungs": {"cast": f"failed: {lock_error}"},
                    "notes": [f"cast reference failed — fix this first: "
                              f"{lock_error}"]}
        voice = (project.cast[0].voice if project.cast
                 else project.voice) or "edge"
        try:
            job = self.pipeline.plan(project.niche, topic, style=project.style,
                                     footage=project.footage, voice=voice,
                                     music=project.music, cast=lock)
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
        if lock:
            rungs["cast"] = ("locked: " +
                             ", ".join(m["name"] for m in lock))
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

    def export_project(self, project: StudioProject | str,
                       dest: str | os.PathLike | None = None
                       ) -> dict[str, Any]:
        """Export a project as a self-contained studio repo.

        The repo rebuilds the project's videos without Devon
        running: ``BRIEF.md`` (niche, style, voice, cast, schedule,
        topic queue), ``manifest.json`` (the stored manifest,
        verbatim), ``render.py`` + ``render.sh`` (the pipeline
        stages, runnable — plan → preflight → run, one episode per
        queued topic), and ``cast/`` (the cast's reference images,
        so the repo is self-contained). ``dest`` defaults to
        ``<studio root>/exports/<slug>``.
        Fails closed with a named reason — missing project, no
        topics, an unreadable face reference, or a non-empty
        destination — and writes nothing partial: the repo is built
        in a staging directory and moved into place only when every
        file is written.
        Returns ``{"ok", "project", "path", "files", "topics",
        "cast"}`` or ``{"ok": False, "project", "reason"}``.
        """
        slug = project if isinstance(project, str) else project.slug
        if isinstance(project, str):
            try:
                project = self.store.load(project)
            except FileNotFoundError:
                return {"ok": False, "project": slug,
                        "reason": f"no studio project {slug!r} — "
                                  "create it first (`animato new`)"}
        else:
            with contextlib.suppress(FileNotFoundError):
                project = self.store.load(project.slug)
        if not project.topic_queue:
            return {"ok": False, "project": project.slug,
                    "reason": "no topics queued — queue a topic first "
                              f"(`animato queue {project.slug} "
                              "\"<topic>\")"}
        lock, lock_error = resolve_cast_lock(project.cast)
        if lock_error:
            return {"ok": False, "project": project.slug,
                    "reason": lock_error}
        target = Path(dest).expanduser() if dest else (
            self.store.root / "exports" / project.slug)
        if target.is_file():
            return {"ok": False, "project": project.slug,
                    "reason": f"export path {target} is a file, not a "
                              "directory — pass --dest <dir>"}
        if target.exists() and any(target.iterdir()):
            return {"ok": False, "project": project.slug,
                    "reason": f"export directory {target} already exists "
                              "and is not empty — remove it or pass "
                              "--dest <dir>"}
        staging = target.with_name(
            target.name + ".tmp-" + uuid.uuid4().hex[:8])
        try:
            staging.mkdir(parents=True, exist_ok=False)
            files = _write_export(staging, project, lock)
        except Exception as exc:  # noqa: BLE001 — export must not raise
            shutil.rmtree(staging, ignore_errors=True)
            return {"ok": False, "project": project.slug,
                    "reason": f"could not write the export: {exc}"}
        if target.exists():
            target.rmdir()
        staging.replace(target)
        return {"ok": True, "project": project.slug,
                "path": str(target), "files": files,
                "topics": len(project.topic_queue),
                "cast": [m["name"] for m in lock]}


class _FakeJobShim:
    """Minimal job-shaped view of an episode for post-copy lookup."""

    def __init__(self, episode: Episode) -> None:
        self.topic = episode.topic
        self.run_id = episode.run_id
