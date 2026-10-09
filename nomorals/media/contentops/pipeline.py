"""Short-form content pipeline orchestrator.

End-to-end: niche + topic → script (brain) → voiceover (TTS) → visuals
(imggen per scene + ken-burns) → beat-synced edit (EditSpec) → captions →
music mix → final MP4.

Design points:

* Every stage logs structured start/finish with timings and writes its
  artifacts into a run directory
  (``<runs_root>/<run_id>/``).  A ``manifest.json`` records per-stage
  status/timings/artifacts, so ``resume(run_id)`` replays from the last
  good artifact instead of redoing the run.
* Failures surface as :class:`StageError` — a clean object with
  ``stage``/``code``/``message``/``hint``/``retriable``.  Tracebacks go to
  the log only, never into the error object shown to chat.
* The edit contract (``EditSpec``/``render``/``detect_beats``) and the
  niche contract (``get_niche``) are imported defensively: the sibling
  ``edit.py`` / ``niches/`` streams are tried first, with the marked
  fallbacks in ``_shims`` used only until they land.  Constructor injection
  (``edit=``, ``get_niche=``, ``brain=``, ``tts=``, ``studio=``) lets tests
  and operators swap any stage dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ...core.logging_setup import get_logger
from . import _shims

_log = get_logger(__name__)

#: Canonical stage order.  Do not reorder without updating resume logic.
STAGES: tuple[str, ...] = (
    "script",
    "voiceover",
    "visuals",
    "edit",
    "captions",
    "music",
    "final",
)

_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


# ── edit / niche contracts (sibling streams, defensive import) ─────────


def _load_edit_contract() -> SimpleNamespace:
    """Real ``edit.py`` first, marked shim fallback until it lands."""
    try:
        from .edit import (  # type: ignore[import-not-found]
            EditSpec as _ES, detect_beats as _db, render as _rn,
        )
        from . import edit as _mod  # type: ignore[import-not-found]
        ns = SimpleNamespace(
            EditSpec=_ES, render=_rn, detect_beats=_db,
            build_captions=getattr(_mod, "build_captions", None),
            make_music_bed=getattr(_mod, "make_music_bed", None),
            source="nomorals.media.contentops.edit",
        )
        _log.info("pipeline: using sibling edit contract (%s)", ns.source)
        return ns
    except ImportError:
        ns = SimpleNamespace(
            EditSpec=_shims.EditSpec, render=_shims.render,
            detect_beats=_shims.detect_beats,
            build_captions=_shims.build_captions,
            make_music_bed=_shims.make_music_bed,
            source="nomorals.media.contentops._shims (fallback)",
        )
        _log.warning("pipeline: sibling edit.py not importable — "
                     "using marked shim fallback")
        return ns


def _load_niche_contract() -> tuple[Callable[[str], Any], str]:
    try:
        from .niches import (  # type: ignore[import-not-found]
            get_niche as _gn,
        )
        _log.info("pipeline: using sibling niches contract")
        return _gn, "nomorals.media.contentops.niches"
    except ImportError:
        _log.warning("pipeline: sibling niches/ not importable — "
                     "using marked shim fallback")
        return _shims.get_niche, "nomorals.media.contentops._shims (fallback)"


# ── errors & results ───────────────────────────────────────────────────


@dataclass
class StageError(Exception):
    """A clean, chat-safe pipeline failure.  Never carries a traceback."""

    stage: str
    code: str
    message: str
    hint: str = ""
    retriable: bool = True

    def __str__(self) -> str:  # noqa: D105
        base = f"[{self.stage}] {self.code}: {self.message}"
        return f"{base} — hint: {self.hint}" if self.hint else base

    def to_dict(self) -> dict:
        """JSON-safe."""
        return asdict(self)


@dataclass
class RunResult:
    """Outcome of ``run()`` / ``resume()``."""

    ok: bool
    run_id: str
    job_id: str
    stages: dict = field(default_factory=dict)   # stage → status/ms
    artifacts: dict = field(default_factory=dict)  # dotted name → path
    final_path: str | None = None
    error: StageError | None = None

    def to_dict(self) -> dict:
        """JSON-safe."""
        d = asdict(self)
        d["error"] = self.error.to_dict() if self.error else None
        return d


# ── Job model & store ──────────────────────────────────────────────────


@dataclass
class Job:
    """One short to produce.  Persisted as JSON — no DB dependency."""

    id: str
    niche: str
    topic: str
    status: str = "queued"  # queued|rendering|ready|posted|failed
    attempts: int = 0
    platforms: list = field(default_factory=list)
    run_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    posted_at: str = ""
    error: dict | None = None

    def to_dict(self) -> dict:
        """JSON-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class JobStore:
    """JSON-per-job persistence under ``<runs_root>/jobs/``."""

    def __init__(self, runs_root: str | os.PathLike) -> None:
        self.dir = Path(runs_root) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)
        return self.dir / f"{safe}.json"

    def save(self, job: Job) -> Path:
        """Write (atomically) and return the path."""
        job.updated_at = _utcnow()
        p = self._path(job.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    def load(self, job_id: str) -> Job:
        """Raise FileNotFoundError for unknown ids."""
        p = self._path(job_id)
        if not p.exists():
            raise FileNotFoundError(f"unknown job {job_id!r}")
        return Job.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self, status: str = "") -> list[Job]:
        """All jobs, optionally filtered by status, newest first."""
        jobs = []
        for p in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                j = Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            if not status or j.status == status:
                jobs.append(j)
        jobs.sort(key=lambda j: j.created_at or "", reverse=True)
        return jobs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_job_id() -> str:
    """Unique job id."""
    return "job_" + uuid.uuid4().hex[:10]


def new_run_id() -> str:
    """Unique run id, roughly sortable by wall time."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


# ── Content calendar ───────────────────────────────────────────────────


@dataclass
class PlannedPost:
    """One scheduled short on the calendar."""

    id: str
    niche: str
    topic: str
    platforms: list = field(default_factory=list)
    scheduled_for: str = ""  # ISO UTC; "" = ASAP
    status: str = "queued"   # queued|rendering|ready|posted|failed
    job_id: str = ""
    posted_at: str = ""

    def to_dict(self) -> dict:
        """JSON-safe."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlannedPost":
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def due_at(self) -> datetime:
        """Scheduled time as aware datetime; empty means 'now'."""
        if not self.scheduled_for:
            return datetime.now(timezone.utc)
        dt = datetime.fromisoformat(self.scheduled_for)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class ContentCalendar:
    """Queue of planned posts with per-niche cadence enforcement.

    Cadence comes from the niche plugin's ``.cadence`` mapping
    (``posts_per_day``).  :meth:`due` only lists what the cadence allows
    right now — the rest stays queued for later.
    """

    def __init__(self, runs_root: str | os.PathLike,
                 get_niche: Callable[[str], Any] | None = None) -> None:
        self.path = Path(runs_root) / "calendar.json"
        if get_niche is None:
            get_niche, _src = _load_niche_contract()
        self._get_niche = get_niche
        self._posts: list[PlannedPost] = []
        self._load()

    # -- persistence -------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._posts = [PlannedPost.from_dict(d) for d in raw]
                return
            except (OSError, ValueError) as exc:
                _log.warning("calendar: corrupt %s (%s) — starting empty",
                             self.path, exc)
        self._posts = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([p.to_dict() for p in self._posts],
                                  indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- queue -------------------------------------------------------
    def add(self, niche: str, topic: str, *,
            platforms: list | None = None,
            scheduled_for: str = "") -> PlannedPost:
        """Schedule a post.  Returns the entry."""
        post = PlannedPost(
            id="post_" + uuid.uuid4().hex[:10],
            niche=niche, topic=topic,
            platforms=list(platforms or []),
            scheduled_for=scheduled_for,
        )
        self._posts.append(post)
        self._save()
        _log.info("calendar: queued %s (%s — %s)", post.id, niche, topic[:60])
        return post

    def get(self, post_id: str) -> PlannedPost:
        """Raise KeyError for unknown ids."""
        for p in self._posts:
            if p.id == post_id:
                return p
        raise KeyError(f"unknown planned post {post_id!r}")

    def mark(self, post_id: str, status: str, job_id: str = "") -> PlannedPost:
        """Update status (and job link); persists."""
        post = self.get(post_id)
        post.status = status
        if job_id:
            post.job_id = job_id
        if status == "posted":
            post.posted_at = _utcnow()
        self._save()
        return post

    def _cadence_for(self, niche: str) -> dict:
        try:
            plugin = self._get_niche(niche)
            if hasattr(plugin, "cadence_spec"):
                return dict(plugin.cadence_spec())
            cad = getattr(plugin, "cadence", 1)
            if isinstance(cad, dict):
                return dict(cad)
            return {"posts_per_day": float(cad or 1)}
        except Exception:  # noqa: BLE001 — a broken plugin must not
            _log.warning("calendar: get_niche(%r) failed; cadence=1/day",
                         niche, exc_info=True)
            return {"posts_per_day": 1}

    def due(self, now: datetime | None = None) -> list[PlannedPost]:
        """Posts ready to render/post now, cadence-enforced.

        A post is due when it is ``queued`` and its scheduled time has
        passed.  Per niche, at most ``posts_per_day`` (from the niche
        plugin's cadence) may be returned per UTC day, counting posts
        already marked ``posted`` today.
        """
        now = now or datetime.now(timezone.utc)
        today = now.date().isoformat()
        posted_today: dict[str, int] = {}
        for p in self._posts:
            if p.status == "posted" and (p.posted_at or "")[:10] == today:
                posted_today[p.niche] = posted_today.get(p.niche, 0) + 1
        allowance: dict[str, int] = {}
        ready: list[PlannedPost] = []
        for p in sorted(self._posts, key=lambda p: p.due_at()):
            if p.status != "queued" or p.due_at() > now:
                continue
            if p.niche not in allowance:
                per_day = int(self._cadence_for(p.niche)
                              .get("posts_per_day", 1) or 1)
                allowance[p.niche] = max(0, per_day
                                         - posted_today.get(p.niche, 0))
            if allowance[p.niche] > 0:
                allowance[p.niche] -= 1
                ready.append(p)
        return ready

    def list(self, status: str = "") -> list[PlannedPost]:
        """All entries, optionally filtered, soonest first."""
        posts = [p for p in self._posts if not status or p.status == status]
        return sorted(posts, key=lambda p: p.due_at())


# ── script parsing ───────────────────────────────────────────────────


def parse_script(text: str) -> dict:
    """Parse brain output into ``{hook, scenes:[{say, show}], title, desc}``.

    Understands the HOOK:/[SCENE n]/SAY:/SHOW:/TITLE:/DESCRIPTION: format
    the niche prompt requests, and degrades gracefully: with no markers
    at all, blank-line paragraphs become scenes (paragraph = SAY, and the
    SHOW falls back to the paragraph text in the visuals stage).
    """
    hook = _field(text, "HOOK")
    title = _field(text, "TITLE")
    description = _field(text, "DESCRIPTION")
    scenes: list[dict] = []
    blocks = re.split(r"(?m)^\[SCENE\s*\d+\]\s*$", text)
    if len(blocks) > 1:
        for block in blocks[1:]:
            say = _field(block, "SAY")
            show = _field(block, "SHOW")
            if say or show:
                scenes.append({"say": say, "show": show})
    if not scenes:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        # drop the meta lines from the fallback paragraphs
        paras = [p for p in paras
                 if not re.match(r"(?i)^(HOOK|TITLE|DESCRIPTION)\s*:", p)]
        for p in paras[:8]:
            scenes.append({"say": p, "show": ""})
    return {"hook": hook, "scenes": scenes or [{"say": text.strip(),
                                                "show": ""}],
            "title": title, "description": description}


def _field(text: str, name: str) -> str:
    m = re.search(rf"(?m)^{name}:\s*(.+?)(?=^(?:HOOK|TITLE|DESCRIPTION|SAY|"
                  rf"SHOW|\[SCENE)\s*:|\Z)", text,
                  re.S | re.I)
    if not m:
        m = re.search(rf"(?m)^{name}:\s*(.+)$", text, re.I)
    return " ".join(m.group(1).split()) if m else ""


def _visual_plan_raw(plugin: Any, script_text: str) -> Any:
    """Call ``plugin.visual_strategy``; plugin bugs degrade, never crash."""
    try:
        return plugin.visual_strategy(script_text)
    except Exception:  # noqa: BLE001
        _log.warning("visual_strategy failed; falling back to scene text",
                     exc_info=True)
        return None


def normalise_visual_prompts(plugin: Any, scenes: list[dict]) -> list[str]:
    """``plugin.visual_strategy`` → one image prompt per scene.

    Accepts a ``VisualPlan`` (``.render_prompts()``), a list of prompts, a
    mapping with a ``"prompts"`` key, or a single string.  Falls back to
    the scene's SHOW/SAY text so a quirky plugin can never leave a scene
    imageless.
    """
    full = "\n".join(s.get("say", "") for s in scenes)
    strat = _visual_plan_raw(plugin, full)
    prompts: list[str] = []
    if strat is not None and hasattr(strat, "render_prompts"):
        try:
            prompts = [str(p) for p in strat.render_prompts()
                       if str(p).strip()]
        except Exception:  # noqa: BLE001
            _log.warning("render_prompts() failed; falling back",
                         exc_info=True)
    if not prompts:
        if isinstance(strat, dict):
            strat = strat.get("prompts") or strat.get("scenes")
        if isinstance(strat, str):
            strat = [strat]
        if isinstance(strat, (list, tuple)):
            prompts = [str(p) for p in strat if str(p).strip()]
    for i, scene in enumerate(scenes):
        if i >= len(prompts) or not prompts[i]:
            fb = scene.get("show") or scene.get("say") or full[:160]
            if i >= len(prompts):
                prompts.append(fb)
            else:
                prompts[i] = fb
    return prompts[:len(scenes)] or [s.get("say", "") for s in scenes]


def extract_scene_motions(plugin: Any, script_text: str,
                          n: int) -> list[tuple[str, str]]:
    """Per-scene ``(effect, zoom_direction)`` from a VisualPlan's motions.

    ``zoom-in-*`` → ken-burns in, ``zoom-out-*`` → ken-burns out,
    ``static``/unknown → static.  Anything that is not a VisualPlan
    yields ken-burns with alternating direction (the classic shorts look).
    """
    motions: list[str] = []
    strat = _visual_plan_raw(plugin, script_text)
    scenes = getattr(strat, "scenes", None)
    if isinstance(scenes, (list, tuple)):
        motions = [str(getattr(s, "motion", "") or "") for s in scenes]
    out: list[tuple[str, str]] = []
    for i in range(n):
        m = motions[i].lower() if i < len(motions) else ""
        if m.startswith("zoom-out"):
            out.append(("kenburns", "out"))
        elif m.startswith("zoom-in"):
            out.append(("kenburns", "in"))
        elif "static" in m:
            out.append(("static", "in"))
        else:
            out.append(("kenburns", "in" if i % 2 == 0 else "out"))
    return out


# ── audio helpers ────────────────────────────────────────────────────


def wav_duration(path: str) -> float:
    """Duration of a WAV in seconds; 0.0 when unreadable."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except (OSError, wave.Error):
        return 0.0


def _ffmpeg(*args: str, timeout: int = 600) -> None:
    try:
        subprocess.run([_FFMPEG, "-y", "-v", "error", *args],
                       check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-600:].decode("utf-8", "replace")
        raise RuntimeError(f"ffmpeg failed: {tail}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed "
                           "(apt install ffmpeg / pkg install ffmpeg)")


def estimate_word_timings(text: str, total_s: float) -> list[tuple[str, float, float]]:
    """Plausible word timings when the TTS backend gives none.

    Distributes ``total_s`` across words proportionally to word length
    (longer words take longer to say — roughly true), with a 90 ms floor
    per word.  Clearly an estimate: the captions stage labels it as such
    in the SRT-adjacent metadata.
    """
    words = text.split()
    if not words or total_s <= 0:
        return []
    weights = [max(len(w), 1) for w in words]
    floor = 0.09
    floors = floor * len(words)
    rest = max(total_s - floors, 0.0)
    wsum = sum(weights)
    out: list[tuple[str, float, float]] = []
    t = 0.0
    for w, wt in zip(words, weights):
        d = floor + rest * wt / wsum
        out.append((w, round(t, 3), round(t + d, 3)))
        t += d
    return out


def phrase_cues(word_timings: list[tuple[str, float, float]],
                max_words: int = 4, max_s: float = 1.8
                ) -> list[tuple[str, float, float]]:
    """Group word timings into short caption phrases (shorts style)."""
    cues: list[tuple[str, float, float]] = []
    cur: list[tuple[str, float, float]] = []
    for wt in word_timings:
        cur.append(wt)
        if len(cur) >= max_words or wt[2] - cur[0][1] >= max_s:
            cues.append((" ".join(w for w, _, _ in cur), cur[0][1], cur[-1][2]))
            cur = []
    if cur:
        cues.append((" ".join(w for w, _, _ in cur), cur[0][1], cur[-1][2]))
    return cues


def snap_to_beats(boundaries: list[float],
                  beats: list[float], tol: float = 0.45) -> list[float]:
    """Nudge scene boundaries onto nearby beat times (±``tol`` s)."""
    if not beats:
        return boundaries
    out = []
    for b in boundaries:
        near = min(beats, key=lambda x: abs(x - b))
        out.append(round(near, 3) if abs(near - b) <= tol else b)
    return out


# ── the pipeline ─────────────────────────────────────────────────────


class ShortPipeline:
    """Niche + topic → rendered vertical MP4, stage by stage.

    ``run(job)`` executes every stage in :data:`STAGES`, logging timings
    and persisting artifacts under
    ``<runs_root>/<run_id>/``::

        script.txt  script.json  post_copy.json
        voiceover.wav  voiceover_meta.json
        frames/scene_000.png …  frames.json
        beats.json  edit_spec.json  draft.mp4
        captions.srt  captions_meta.json  captioned.mp4
        music_bed.wav  mixed_audio.wav
        final.mp4
        manifest.json  error.json (on failure)

    ``resume(run_id)`` replays from the first stage whose artifacts are
    missing or incomplete.  Heavy dependencies (brain/tts/studio/edit/
    niche plugin) are constructor-injectable; when omitted the real
    subsystems are wired lazily.
    """

    def __init__(self, runs_root: str | os.PathLike = "",
                 *, brain: Any = None, tts: Any = None, studio: Any = None,
                 edit: Any = None,
                 get_niche: Callable[[str], Any] | None = None) -> None:
        self.runs_root = Path(runs_root) if runs_root else (
            Path.home() / "workspace" / "devon" / ".shorts_runs")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.jobs = JobStore(self.runs_root)
        self._brain = brain
        self._tts = tts
        self._studio = studio
        if edit is None:
            self._edit = _load_edit_contract()
        else:
            self._edit = SimpleNamespace(
                EditSpec=getattr(edit, "EditSpec"),
                render=getattr(edit, "render"),
                detect_beats=getattr(edit, "detect_beats"),
                build_captions=getattr(edit, "build_captions",
                                       _shims.build_captions),
                make_music_bed=getattr(edit, "make_music_bed",
                                       _shims.make_music_bed),
                source=f"injected {type(edit).__name__}",
            )
        if get_niche is None:
            get_niche, _src = _load_niche_contract()
        self._get_niche = get_niche
        self.calendar = ContentCalendar(self.runs_root,
                                        get_niche=self._get_niche)

    # -- lazy real dependencies --------------------------------------
    @property
    def brain(self) -> Any:
        """The real brain, built on first use (never raises on call)."""
        if self._brain is None:
            from ...llm.brain import get_brain
            self._brain = get_brain()
        return self._brain

    @property
    def tts(self) -> Any:
        """The real voice catalogue, built on first use."""
        if self._tts is None:
            from ...voice.catalogue import default_catalogue
            self._tts = default_catalogue()
        return self._tts

    @property
    def studio(self) -> Any:
        """The real image studio, built on first use."""
        if self._studio is None:
            from ..imggen.studio import Studio
            self._studio = Studio()
        return self._studio

    # -- planning ----------------------------------------------------
    def plan(self, niche: str, topic: str, *,
             platforms: list | None = None) -> Job:
        """Create a queued :class:`Job` (no rendering yet)."""
        job = Job(id=new_job_id(), niche=niche, topic=topic,
                  platforms=list(platforms or []),
                  created_at=_utcnow())
        self.jobs.save(job)
        _log.info("pipeline: planned %s (%s — %s)", job.id, niche,
                  topic[:60])
        return job

    # -- run / resume ------------------------------------------------
    def run(self, job: Job) -> RunResult:
        """Render a job end-to-end.  Returns a RunResult (never raises)."""
        job.status = "rendering"
        job.attempts += 1
        job.run_id = job.run_id or new_run_id()
        self.jobs.save(job)
        run_dir = self.runs_root / job.run_id
        manifest = self._new_manifest(job, run_dir)
        ctx = self._ctx(job, run_dir, manifest)
        try:
            self._run_stages(ctx, from_stage=None)
        except StageError as err:
            return self._fail(ctx, job, err)
        return self._succeed(ctx, job)

    def resume(self, run_id: str) -> RunResult:
        """Replay a run from the first incomplete stage.

        Stages recorded ``done`` in ``manifest.json`` whose artifacts
        still exist on disk are skipped; everything from the first gap
        onward re-runs.  Raises FileNotFoundError for unknown run ids.
        """
        run_dir = self.runs_root / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"unknown run {run_id!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        job = self.jobs.load(manifest["job_id"])
        job.status = "rendering"
        job.attempts += 1
        self.jobs.save(job)
        ctx = self._ctx(job, run_dir, manifest)
        _log.info("pipeline: resuming %s", run_id)
        try:
            self._run_stages(ctx, from_stage=None)
        except StageError as err:
            return self._fail(ctx, job, err)
        return self._succeed(ctx, job)

    # -- internals ---------------------------------------------------
    def _new_manifest(self, job: Job, run_dir: Path) -> dict:
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": job.run_id, "job_id": job.id,
            "niche": job.niche, "topic": job.topic,
            "platforms": list(job.platforms),
            "created_at": _utcnow(),
            "stages": {s: {"status": "pending", "ms": 0, "attempts": 0,
                           "artifacts": {}} for s in STAGES},
            "error": None,
        }
        self._write_manifest(run_dir, manifest)
        return manifest

    @staticmethod
    def _write_manifest(run_dir: Path, manifest: dict) -> None:
        tmp = run_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(run_dir / "manifest.json")

    def _ctx(self, job: Job, run_dir: Path, manifest: dict) -> dict:
        return {"job": job, "run_dir": run_dir, "manifest": manifest,
                "niche": self._get_niche(job.niche),
                "artifacts": {}, "stage_ms": {}}

    def _run_stages(self, ctx: dict, from_stage: str | None) -> None:
        """Execute stages in order; skips completed ones with artifacts.

        Raises StageError on the first failure.  Tracebacks are logged,
        never embedded in the raised error.
        """
        manifest, run_dir = ctx["manifest"], ctx["run_dir"]
        started = from_stage is None
        for stage in STAGES:
            if from_stage is not None:
                if stage == from_stage:
                    started = True
                if not started:
                    continue
            state = manifest["stages"][stage]
            if state["status"] == "done" and self._artifacts_ok(
                    run_dir, stage, state["artifacts"]):
                _log.info("pipeline %s: stage %-9s skipped (done, %dms)",
                          manifest["run_id"], stage, state["ms"])
                self._load_stage_artifacts(ctx, stage, state["artifacts"])
                continue
            state["status"] = "running"
            state["attempts"] += 1
            self._write_manifest(run_dir, manifest)
            t0 = time.monotonic()
            _log.info("pipeline %s: stage %-9s start",
                      manifest["run_id"], stage)
            try:
                artifacts = getattr(self, f"_stage_{stage}")(ctx)
            except StageError:
                raise
            except Exception as exc:  # noqa: BLE001 — clean error object
                _log.exception("pipeline %s: stage %s crashed",
                               manifest["run_id"], stage)
                raise StageError(stage, "STAGE_CRASH",
                                 f"{stage} stage failed unexpectedly: "
                                 f"{type(exc).__name__}: {exc}",
                                 hint="see the service log for the "
                                      "traceback; fix and resume()",
                                 retriable=True) from None
            ms = int((time.monotonic() - t0) * 1000)
            state.update(status="done", ms=ms, artifacts=artifacts)
            ctx["artifacts"][stage] = artifacts
            ctx["stage_ms"][stage] = ms
            self._write_manifest(run_dir, manifest)
            _log.info("pipeline %s: stage %-9s done in %dms",
                      manifest["run_id"], stage, ms)

    _REQUIRED_ARTIFACTS = {
        "script": ("script.txt", "script.json"),
        "voiceover": ("voiceover.wav",),
        "visuals": ("frames.json",),
        "edit": ("draft.mp4", "edit_spec.json"),
        "captions": ("captions.srt", "captioned.mp4"),
        "music": ("mixed_audio.wav",),
        "final": ("final.mp4",),
    }

    def _artifacts_ok(self, run_dir: Path, stage: str,
                      artifacts: dict) -> bool:
        """True when every required artifact file exists on disk."""
        for name in self._REQUIRED_ARTIFACTS.get(stage, ()):
            rel = artifacts.get(name) or name
            if not (run_dir / rel).exists() and not Path(rel).exists():
                return False
        return True

    def _load_stage_artifacts(self, ctx: dict, stage: str,
                              artifacts: dict) -> None:
        ctx["artifacts"][stage] = dict(artifacts)

    def _fail(self, ctx: dict, job: Job, err: StageError) -> RunResult:
        manifest, run_dir = ctx["manifest"], ctx["run_dir"]
        manifest["error"] = err.to_dict()
        manifest["stages"][err.stage]["status"] = "failed"
        self._write_manifest(run_dir, manifest)
        (run_dir / "error.json").write_text(
            json.dumps(err.to_dict(), indent=2), encoding="utf-8")
        job.status = "failed"
        job.error = err.to_dict()
        self.jobs.save(job)
        _log.warning("pipeline %s: FAILED at %s — %s",
                     manifest["run_id"], err.stage, err)
        return RunResult(ok=False, run_id=manifest["run_id"], job_id=job.id,
                         stages={s: v["status"]
                                 for s, v in manifest["stages"].items()},
                         artifacts=self._flat_artifacts(ctx),
                         error=err)

    def _succeed(self, ctx: dict, job: Job) -> RunResult:
        manifest, run_dir = ctx["manifest"], ctx["run_dir"]
        manifest["error"] = None
        self._write_manifest(run_dir, manifest)
        job.status = "ready"
        job.error = None
        self.jobs.save(job)
        final_rel = ctx["artifacts"].get("final", {}).get("final.mp4")
        final = str(run_dir / final_rel) if final_rel else None
        _log.info("pipeline %s: READY → %s", manifest["run_id"], final)
        return RunResult(ok=True, run_id=manifest["run_id"], job_id=job.id,
                         stages={s: v["status"]
                                 for s, v in manifest["stages"].items()},
                         artifacts=self._flat_artifacts(ctx),
                         final_path=final)

    @staticmethod
    def _flat_artifacts(ctx: dict) -> dict:
        flat: dict = {}
        for stage, arts in ctx["artifacts"].items():
            for name, rel in arts.items():
                flat[f"{stage}/{name}"] = str(ctx["run_dir"] / rel)
        return flat

    # ── stage: script ──────────────────────────────────────────────
    def _stage_script(self, ctx: dict) -> dict:
        job, run_dir, niche = ctx["job"], ctx["run_dir"], ctx["niche"]
        text, brain_detail = self._generate_script_text(niche, job.topic)
        if not text.strip():
            raise StageError("script", "BRAIN_UNAVAILABLE",
                             f"the brain could not write the script: "
                             f"{brain_detail}",
                             hint="check provider keys / local model server, "
                                  "then resume()",
                             retriable=True)
        parsed = parse_script(text)
        if not parsed["scenes"]:
            raise StageError("script", "SCRIPT_UNPARSEABLE",
                             "brain returned a script with no usable scenes",
                             hint="retry — the model may comply on a second "
                                  "attempt; then resume()",
                             retriable=True)
        (run_dir / "script.txt").write_text(text, encoding="utf-8")
        (run_dir / "script.json").write_text(
            json.dumps(parsed, indent=2), encoding="utf-8")
        copy = self._post_copy(niche, job, parsed, text)
        (run_dir / "post_copy.json").write_text(
            json.dumps(copy, indent=2), encoding="utf-8")
        ctx["script"] = parsed
        n_words = len(text.split())
        _log.info("pipeline %s: script %d scenes, ~%d words",
                  ctx["manifest"]["run_id"], len(parsed["scenes"]), n_words)
        return {"script.txt": "script.txt", "script.json": "script.json",
                "post_copy.json": "post_copy.json"}

    def _generate_script_text(self, niche: Any,
                              topic: str) -> tuple[str, str]:
        """Script text + failure detail ("" when ok).

        Prefers the niche's own ``generate_script(brain, topic)`` helper
        (fence-stripped, word-counted); on an empty result falls back to a
        direct ``brain.complete`` so the StageError can name the providers
        that failed instead of a bare "empty".
        """
        gen = getattr(niche, "generate_script", None)
        if callable(gen):
            try:
                result = gen(self.brain, topic)
            except Exception:  # noqa: BLE001 — helper must never raise
                _log.warning("niche.generate_script raised; using direct "
                             "brain call", exc_info=True)
                result = None
            text = (getattr(result, "text", "") or "").strip()
            if text:
                return text, ""
        prompt = niche.script_prompt(topic)
        resp = self.brain.complete(prompt, task_kind="creative")
        if getattr(resp, "ok", False) and (resp.text or "").strip():
            return resp.text, ""
        detail = (getattr(resp, "error", "") or "empty response").strip()
        tried = getattr(resp, "failed_providers", None) or []
        if tried:
            detail += f" (tried: {', '.join(map(str, tried))})"
        return "", detail

    def _post_copy(self, niche: Any, job: Job, parsed: dict,
                   script_text: str) -> dict:
        """Title/description/hashtags via the niche's render helpers."""
        topic = job.topic
        platform = (job.platforms or ["tiktok"])[0]
        if hasattr(niche, "title_for"):
            title = niche.title_for(topic)
        else:
            title = self._fmt_tpl(getattr(niche, "title_template",
                                          "{topic}"),
                                  niche, topic, parsed, script_text)
        if hasattr(niche, "description_for"):
            try:
                description = niche.description_for(topic, script_text)
            except TypeError:
                description = niche.description_for(topic)
        else:
            description = self._fmt_tpl(
                getattr(niche, "description_template", "{topic}"),
                niche, topic, parsed, script_text)
        tags = self._niche_tags(niche, platform)
        return {"title": title, "description": description,
                "hashtags": tags, "platform": platform}

    @staticmethod
    def _fmt_tpl(tpl: str, niche: Any, topic: str, parsed: dict,
                 script_text: str) -> str:
        try:
            return tpl.format(topic=topic,
                              niche=getattr(niche, "name", ""),
                              hook=parsed.get("hook", ""),
                              title=parsed.get("title", ""),
                              script=script_text,
                              hashtags=" ".join(
                                  ShortPipeline._niche_tags(niche, "tiktok")))
        except (KeyError, IndexError, ValueError):
            return tpl

    @staticmethod
    def _niche_tags(niche: Any, platform: str) -> list[str]:
        if hasattr(niche, "tags_for"):
            try:
                return [str(t) for t in niche.tags_for(platform)]
            except Exception:  # noqa: BLE001
                _log.warning("tags_for(%r) failed", platform, exc_info=True)
        tags = getattr(niche, "hashtags", []) or []
        if isinstance(tags, dict):
            tags = tags.get(platform) or tags.get("default") or []
        return [str(t) for t in tags]

    # ── stage: voiceover ───────────────────────────────────────────
    def _stage_voiceover(self, ctx: dict) -> dict:
        run_dir, niche = ctx["run_dir"], ctx["niche"]
        parsed = ctx.get("script") or self._read_script(run_dir)
        ctx["script"] = parsed
        text = " ".join([parsed.get("hook", "")] +
                        [s.get("say", "") for s in parsed["scenes"]]).strip()
        if not text:
            raise StageError("voiceover", "EMPTY_SCRIPT",
                             "nothing to speak — the script has no narration",
                             hint="re-run the script stage", retriable=False)
        out_path = str(run_dir / "voiceover.wav")
        result, voice_name = self._speak_with_fallbacks(text, niche,
                                                        out_path)
        audio_path = (result or {}).get("path") or out_path
        if not os.path.exists(audio_path):
            raise StageError("voiceover", "TTS_NO_AUDIO",
                             "TTS returned without writing an audio file",
                             hint="check the TTS backend, then resume()",
                             retriable=True)
        meta = {"voice": voice_name, "path": audio_path,
                "duration_s": round(wav_duration(audio_path), 2),
                "word_timings": ((result or {}).get("word_timings")
                                 or (result or {}).get("words")),
                "timings_source": ("tts" if ((result or {}).get("word_timings")
                                             or (result or {}).get("words"))
                                   else "estimated"),
                "text": text}
        (run_dir / "voiceover_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        ctx["voiceover"] = meta
        _log.info("pipeline %s: voiceover %.1fs (%s timings)",
                  ctx["manifest"]["run_id"], meta["duration_s"],
                  meta["timings_source"])
        return {"voiceover.wav": os.path.relpath(audio_path, run_dir)
                if os.path.dirname(audio_path) == str(run_dir)
                else audio_path,
                "voiceover_meta.json": "voiceover_meta.json"}

    def _voice_candidates(self, niche: Any) -> list[str]:
        """Voice names to try, in order.  Never empty."""
        cands: list[str] = []
        spec = getattr(niche, "voice_spec", None)
        if spec is not None:
            try:
                resolved = spec.resolve(self.tts)
            except Exception:  # noqa: BLE001 — resolve never raises, but
                resolved = ""  # a foreign spec might
            if resolved:
                return [resolved]
            fn = getattr(spec, "candidates", None)
            if callable(fn):
                try:
                    cands.extend(str(c) for c in fn() if c)
                except Exception:  # noqa: BLE001
                    pass
            for attr in ("preferred_voice", "fallback_voice"):
                v = getattr(spec, attr, "") or ""
                if v and v not in cands:
                    cands.append(v)
        legacy = getattr(niche, "voice_style", "") or ""
        if legacy and legacy not in cands:
            cands.append(legacy)
        cands.append("narrator")
        return cands

    def _speak_with_fallbacks(self, text: str, niche: Any,
                              out_path: str) -> tuple[dict, str]:
        """speak_as over the voice candidates; clean errors on failure."""
        tried: list[str] = []
        for voice_name in self._voice_candidates(niche):
            tried.append(voice_name)
            try:
                result = self.tts.speak_as(text, voice_name,
                                           out_path=out_path)
            except KeyError:
                _log.info("voiceover: %r not in catalogue, trying next",
                          voice_name)
                continue
            except Exception as exc:  # noqa: BLE001 — engine errors → clean
                _log.exception("voiceover engine failed")
                raise StageError(
                    "voiceover", "TTS_FAILED",
                    f"TTS engine failed on voice {voice_name!r}: "
                    f"{type(exc).__name__}: {exc}",
                    hint="check the TTS backend for this voice, then "
                         "resume()",
                    retriable=True) from None
            return result or {}, voice_name
        raise StageError(
            "voiceover", "TTS_UNKNOWN_VOICE",
            f"none of the niche's voices are in the catalogue "
            f"(tried: {', '.join(tried)})",
            hint="clone/add one of these voices, or set the niche's "
                 "voice_spec.preferred_voice to a catalogue voice name",
            retriable=False)

    @staticmethod
    def _read_script(run_dir: Path) -> dict:
        p = run_dir / "script.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        txt = run_dir / "script.txt"
        return parse_script(txt.read_text(encoding="utf-8")) if txt.exists() \
            else {"hook": "", "scenes": [], "title": "", "description": ""}

    # ── stage: visuals ─────────────────────────────────────────────
    def _stage_visuals(self, ctx: dict) -> dict:
        run_dir, niche = ctx["run_dir"], ctx["niche"]
        parsed = ctx.get("script") or self._read_script(run_dir)
        ctx["script"] = parsed
        scenes = parsed["scenes"]
        prompts = normalise_visual_prompts(niche, scenes)
        script_text = "\n".join(s.get("say", "") for s in scenes)
        motions = extract_scene_motions(niche, script_text, len(scenes))
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        records: list[dict] = []
        for i, (scene, prompt) in enumerate(zip(scenes, prompts)):
            dest = str(frames_dir / f"scene_{i:03d}.png")
            try:
                saved = self.studio.generate(
                    prompt, save_to=dest,
                    width=768, height=1344)  # 9:16-ish; studio filters
            except Exception as exc:  # noqa: BLE001
                _log.exception("visuals: scene %d failed", i)
                raise StageError(
                    "visuals", "IMGGEN_FAILED",
                    f"image {i + 1}/{len(scenes)} failed: "
                    f"{type(exc).__name__}: {exc}",
                    hint="the native checkpoint may be missing or the "
                         "profile too small for imggen — check `nm imggen`, "
                         "then resume() (finished scenes are kept)",
                    retriable=True) from None
            path = saved[0] if isinstance(saved, (list, tuple)) else saved
            path = str(path or dest)
            if not os.path.exists(path):
                raise StageError("visuals", "IMGGEN_NO_FILE",
                                 f"image {i + 1} produced no file",
                                 hint="check the imggen pipeline, then "
                                      "resume()",
                                 retriable=True)
            records.append({"scene": i, "prompt": prompt, "path": path,
                            "say": scene.get("say", ""),
                            "effect": motions[i][0],
                            "zoom_direction": motions[i][1]})
            _log.info("pipeline %s: visual %d/%d → %s",
                      ctx["manifest"]["run_id"], i + 1, len(scenes), path)
        (run_dir / "frames.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8")
        ctx["frames"] = records
        return {"frames.json": "frames.json",
                **{f"frames/scene_{i:03d}.png":
                   os.path.relpath(r["path"], run_dir)
                   for i, r in enumerate(records)}}

    # ── stage: edit (beat-synced) ──────────────────────────────────
    def _stage_edit(self, ctx: dict) -> dict:
        run_dir = ctx["run_dir"]
        vo = ctx.get("voiceover") or self._read_vo_meta(run_dir)
        ctx["voiceover"] = vo
        frames = ctx.get("frames") or self._read_frames(run_dir)
        ctx["frames"] = frames
        audio_path = vo["path"]
        total = vo.get("duration_s") or wav_duration(audio_path) or 8.0

        beats = self._edit.detect_beats(audio_path)
        (run_dir / "beats.json").write_text(
            json.dumps({"beats": beats, "count": len(beats)}, indent=2),
            encoding="utf-8")

        # scene durations ∝ narration length, boundaries snapped to beats
        says = [f.get("say", "") for f in frames] or [""] * len(frames)
        weights = [max(len(s.split()), 1) for s in says]
        wsum = sum(weights)
        bounds = [0.0]
        acc = 0.0
        for w in weights:
            acc += total * w / wsum
            bounds.append(round(acc, 3))
        bounds[-1] = round(total, 3)
        if len(bounds) > 2:
            bounds[1:-1] = snap_to_beats(bounds[1:-1], beats)
        scenes = []
        for i, f in enumerate(frames):
            dur = max(bounds[i + 1] - bounds[i], 0.5)
            scenes.append({"image": f["path"], "duration": round(dur, 3),
                           "effect": f.get("effect", "kenburns"),
                           "zoom_direction": f.get("zoom_direction", "in")})

        payload = {"scenes": scenes, "audio": audio_path, "captions": "",
                   "music": "", "beat_times": beats,
                   "output": str(run_dir / "draft.mp4"),
                   "width": 1080, "height": 1920, "fps": 30}
        (run_dir / "edit_spec.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        spec = self._construct_spec(payload)
        try:
            out = self._edit.render(spec)
        except Exception as exc:  # noqa: BLE001
            _log.exception("edit render failed")
            raise StageError(
                "edit", "RENDER_FAILED",
                f"video render failed: {type(exc).__name__}: {exc}",
                hint="if this names a contract mismatch, the sibling edit.py "
                     "landed with a different EditSpec/render shape — align "
                     "the contract, then resume()",
                retriable=True) from None
        out = str(out or payload["output"])
        if not os.path.exists(out):
            raise StageError("edit", "RENDER_NO_FILE",
                             "render returned without writing a video file",
                             hint="check the edit backend, then resume()",
                             retriable=True)
        _log.info("pipeline %s: draft %.1fs, %d scenes, %d beats",
                  ctx["manifest"]["run_id"], total, len(scenes), len(beats))
        return {"edit_spec.json": "edit_spec.json", "beats.json": "beats.json",
                "draft.mp4": os.path.relpath(out, run_dir)}

    def _construct_spec(self, payload: dict) -> Any:
        """Build an EditSpec defensively against sibling shape drift."""
        ES = self._edit.EditSpec
        try:
            return ES(**payload)
        except TypeError:
            _log.warning("EditSpec(**payload) rejected — trying EditSpec(payload)")
        try:
            return ES(payload)
        except Exception:
            _log.warning("EditSpec(payload) failed — passing raw mapping "
                         "to render()")
            return payload

    @staticmethod
    def _read_vo_meta(run_dir: Path) -> dict:
        p = run_dir / "voiceover_meta.json"
        meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        meta.setdefault("path", str(run_dir / "voiceover.wav"))
        meta.setdefault("duration_s", wav_duration(meta["path"]))
        return meta

    @staticmethod
    def _read_frames(run_dir: Path) -> list[dict]:
        p = run_dir / "frames.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return [{"scene": i, "path": str(f), "say": ""}
                for i, f in enumerate(sorted((run_dir / "frames").glob("*.png")))]

    # ── stage: captions ────────────────────────────────────────────
    def _stage_captions(self, ctx: dict) -> dict:
        run_dir = ctx["run_dir"]
        vo = ctx.get("voiceover") or self._read_vo_meta(run_dir)
        draft_rel = ctx["artifacts"].get("edit", {}).get("draft.mp4",
                                                         "draft.mp4")
        draft = run_dir / draft_rel
        if not draft.exists():
            raise StageError("captions", "NO_DRAFT",
                             "draft.mp4 missing — the edit stage did not "
                             "complete",
                             hint="resume() will re-run edit first",
                             retriable=True)
        timings = vo.get("word_timings")
        if timings and isinstance(timings, list) and timings \
                and isinstance(timings[0], dict):
            wt = [(d.get("word", ""), float(d.get("start", 0)),
                   float(d.get("end", 0))) for d in timings]
            source = "tts"
        elif timings and isinstance(timings, list) and timings \
                and isinstance(timings[0], (list, tuple)):
            wt = [(str(w), float(a), float(b)) for w, a, b in timings]
            source = "tts"
        else:
            wt = estimate_word_timings(vo.get("text", ""),
                                       vo.get("duration_s")
                                       or wav_duration(vo["path"]) or 8.0)
            source = "estimated"
        cues = phrase_cues(wt)
        if not cues:
            raise StageError("captions", "NO_CUES",
                             "could not build caption cues from the voiceover",
                             hint="check voiceover_meta.json, then resume()",
                             retriable=False)
        srt_path = str(run_dir / "captions.srt")
        self._edit.build_captions(cues, srt_path)
        (run_dir / "captions_meta.json").write_text(json.dumps(
            {"cues": len(cues), "timings_source": source}, indent=2),
            encoding="utf-8")
        captioned = str(run_dir / "captioned.mp4")
        srt_esc = srt_path.replace("'", r"'\''")
        try:
            _ffmpeg("-i", str(draft), "-vf",
                    "subtitles='%s':force_style="
                    "'FontSize=24,PrimaryColour=&HFFFFFF&,"
                    "OutlineColour=&H80000000&,BorderStyle=1,Outline=2,"
                    "MarginV=130,Alignment=2'" % srt_esc,
                    "-c:a", "copy", captioned)
        except RuntimeError as exc:
            raise StageError("captions", "BURNIN_FAILED",
                             f"caption burn-in failed: {exc}",
                             hint="ffmpeg needs libass for the subtitles "
                                  "filter; install a full ffmpeg build, then "
                                  "resume()",
                             retriable=True) from None
        _log.info("pipeline %s: %d caption cues (%s timings)",
                  ctx["manifest"]["run_id"], len(cues), source)
        return {"captions.srt": "captions.srt",
                "captions_meta.json": "captions_meta.json",
                "captioned.mp4": "captioned.mp4"}

    # ── stage: music ───────────────────────────────────────────────
    def _stage_music(self, ctx: dict) -> dict:
        run_dir = ctx["run_dir"]
        vo = ctx.get("voiceover") or self._read_vo_meta(run_dir)
        duration = vo.get("duration_s") or wav_duration(vo["path"]) or 8.0
        duration = round(duration + 1.0, 2)  # tail for the end card
        bed_path = str(run_dir / "music_bed.wav")
        self._edit.make_music_bed(
            duration, bed_path,
            seed=abs(hash(ctx["manifest"]["run_id"])) % (2 ** 31))
        mixed = str(run_dir / "mixed_audio.wav")
        try:
            _ffmpeg("-i", vo["path"], "-i", bed_path,
                    "-filter_complex",
                    "[1:a]volume=0.16,apad[bed];"
                    "[0:a][bed]amix=inputs=2:duration=first:"
                    "dropout_transition=0[a]",
                    "-map", "[a]", "-c:a", "pcm_s16le", mixed)
        except RuntimeError as exc:
            raise StageError("music", "MIX_FAILED",
                             f"music mix failed: {exc}",
                             hint="check ffmpeg, then resume()",
                             retriable=True) from None
        (run_dir / "music_meta.json").write_text(json.dumps(
            {"bed_s": duration, "bed_db": "-16dB under voiceover",
             "source": self._edit.source}, indent=2), encoding="utf-8")
        _log.info("pipeline %s: music bed %.1fs mixed",
                  ctx["manifest"]["run_id"], duration)
        return {"music_bed.wav": "music_bed.wav",
                "mixed_audio.wav": "mixed_audio.wav",
                "music_meta.json": "music_meta.json"}

    # ── stage: final ───────────────────────────────────────────────
    def _stage_final(self, ctx: dict) -> dict:
        run_dir = ctx["run_dir"]
        cap_rel = ctx["artifacts"].get("captions", {}).get("captioned.mp4",
                                                           "captioned.mp4")
        mus_rel = ctx["artifacts"].get("music", {}).get("mixed_audio.wav",
                                                        "mixed_audio.wav")
        captioned, mixed = run_dir / cap_rel, run_dir / mus_rel
        if not captioned.exists():
            raise StageError("final", "NO_CAPTIONED",
                             "captioned.mp4 missing", retriable=True,
                             hint="resume() will re-run captions first")
        if not mixed.exists():
            raise StageError("final", "NO_MIX",
                             "mixed_audio.wav missing", retriable=True,
                             hint="resume() will re-run music first")
        final = str(run_dir / "final.mp4")
        try:
            _ffmpeg("-i", str(captioned), "-i", str(mixed),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", final)
        except RuntimeError as exc:
            raise StageError("final", "MUX_FAILED",
                             f"final mux failed: {exc}",
                             hint="check ffmpeg, then resume()",
                             retriable=True) from None
        size = os.path.getsize(final)
        _log.info("pipeline %s: FINAL %s (%.1f MB)",
                  ctx["manifest"]["run_id"], final, size / 1e6)
        return {"final.mp4": "final.mp4"}

    # ── cost / time estimator ──────────────────────────────────────
    def estimate(self, niche: str, topic: str,
                 scenes: int = 5) -> dict:
        """Rough render-time + cost estimate BEFORE a run starts.

        Stage timings come from measured medians of past runs in this
        runs root when any exist (labelled ``measured``); otherwise from
        documented first-run guesses (labelled ``guess`` — wide ranges,
        because imggen speed depends on the checkpoint and the profile).
        Costs are honest: the local stack (brain failover → free tiers,
        local TTS, native imggen, synthesized bed) is $0 API spend; a
        paid provider in the brain chain is the only thing that can bill.
        """
        measured = self._measured_stage_ms()
        guesses = {  # seconds, (low, high) — first-run guesses
            "script": (8, 40),
            "voiceover": (20, 90),
            "visuals": (60, 300),   # per scene, native checkpoint, laptop
            "edit": (30, 120),
            "captions": (5, 20),
            "music": (10, 30),
            "final": (10, 30),
        }
        per_scene = {"visuals"}
        rows, lo_total, hi_total = [], 0.0, 0.0
        for stage in STAGES:
            if stage in measured:
                lo = hi = round(measured[stage] / 1000, 1)
                basis = f"measured median of {measured['_n']} past run(s)"
            else:
                lo, hi = guesses[stage]
                if stage in per_scene:
                    lo, hi = lo * scenes, hi * scenes
                basis = ("guess — no past runs yet; imggen especially "
                         "depends on checkpoint + profile")
            rows.append({"stage": stage, "time_low_s": lo, "time_high_s": hi,
                         "cost_usd_low": 0.0, "cost_usd_high": 0.0,
                         "basis": basis})
            lo_total += lo
            hi_total += hi
        notes = [
            "Local stack = $0 API spend: brain (free-tier failover chain), "
            "local TTS, native imggen, synthesized music bed.",
            "The brain can bill ONLY if its chain routes to a paid provider "
            "— check the active chain before a long batch.",
            "Wall-clock scales with profile: workstation < laptop < termux. "
            "imggen dominates the total.",
            f"Script target ~120-150 words ≈ 55-70s of voiceover; "
            f"{scenes} scenes requested.",
        ]
        return {"niche": niche, "topic": topic, "scenes": scenes,
                "stages": rows,
                "total_time_low_s": round(lo_total, 1),
                "total_time_high_s": round(hi_total, 1),
                "total_cost_usd_low": 0.0, "total_cost_usd_high": 0.0,
                "notes": notes}

    def _measured_stage_ms(self) -> dict:
        """Median per-stage ms across past manifests ({} when none)."""
        samples: dict[str, list[float]] = {}
        manifests = sorted(self.runs_root.glob("run_*/manifest.json"))[-20:]
        for mp in manifests:
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for stage, st in (data.get("stages") or {}).items():
                if st.get("status") == "done" and st.get("ms"):
                    samples.setdefault(stage, []).append(st["ms"])
        if not samples:
            return {}
        out = {"_n": len(manifests)}
        for stage, vals in samples.items():
            vals.sort()
            out[stage] = vals[len(vals) // 2]
        return out


# keep the module importable even where the sibling contracts are absent
__all__ = [
    "STAGES",
    "ContentCalendar",
    "Job",
    "JobStore",
    "PlannedPost",
    "RunResult",
    "ShortPipeline",
    "StageError",
    "estimate_word_timings",
    "extract_scene_motions",
    "new_job_id",
    "new_run_id",
    "normalise_visual_prompts",
    "parse_script",
    "phrase_cues",
    "snap_to_beats",
    "wav_duration",
]
