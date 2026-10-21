"""Builtin tools for the Director — moviemaking from plain language.

Wires ``nomorals.media.director`` into the spine so the brain can
direct, not just generate: "give me coverage on this script", "break
it down and design the shots", "shoot production X", "regenerate
shot 2.3". Stages are separate tools on purpose — coverage, plan,
and shoot are owner-gated steps, matching the chat ``/director``
surface.

Scripts arrive as text (``script_text``) or as a file
(``script_path``). Productions persist under
``<workspace>/generated/director/<id>/``; shooting goes through the
videogen registry machinery with its capability gate, never a
parallel generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)


def _director(context: Any):
    from ..media.director import Director
    root = None
    ws = getattr(context, "workspace", None)
    if ws:
        root = Path(ws) / "generated" / "director"
    return Director(context=context, root=root)


def _script_text(script_text: str, script_path: str) -> str:
    if (script_text or "").strip():
        return script_text
    if script_path:
        p = Path(script_path)
        if not p.is_file():
            raise ValueError(f"script file not found: {script_path}")
        return p.read_text(encoding="utf-8", errors="replace")
    raise ValueError("give me the script: script_text or script_path")


def register(registry: Any) -> None:
    """Attach the director tools to a registry."""
    context = registry.context

    @registry.register(
        "script_coverage",
        description=("Professional script coverage: logline, synopsis, "
                     "structure read, graded assessments (premise / "
                     "character / dialogue / visual / pacing) and a "
                     "verdict — pass, consider, or recommend — with "
                     "reasons. Give script_text or script_path."),
        capability=Capability.FS_WRITE,
    )
    def script_coverage(script_text: str = "", script_path: str = "",
                        title: str = "") -> dict[str, Any]:
        from ..media.director import DirectorError
        try:
            cov = _director(context).cover(
                _script_text(script_text, script_path), title=title)
        except DirectorError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "coverage": cov.to_dict(),
                "report": cov.report()}

    @registry.register(
        "movie_plan",
        description=("Break a script down into scenes and shots and "
                     "design every shot (lighting + camera composed "
                     "from primitives; style_pack biases the design "
                     "— neutral, naturalist, noir, neon, golden). "
                     "Creates a persisted production the owner can "
                     "review before anything is generated. Give "
                     "script_text or script_path."),
        capability=Capability.FS_WRITE,
    )
    def movie_plan(script_text: str = "", script_path: str = "",
                   title: str = "", style_pack: str = "neutral",
                   refine: bool = False) -> dict[str, Any]:
        from ..media.director import DirectorError
        try:
            prod = _director(context).plan(
                _script_text(script_text, script_path), title=title,
                style_pack=style_pack, refine=refine)
        except (DirectorError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "production_id": prod.id,
                "title": prod.title, "style_pack": prod.style_pack,
                "dir": prod.dir,
                "scenes": len(prod.screenplay.scenes),
                "shots": [{"id": s.id, "kind": s.kind,
                           "description": s.description,
                           "duration_s": s.duration_s}
                          for s in prod.shots]}

    @registry.register(
        "movie_shoot",
        description=("Generate a planned production's shots through "
                     "the videogen registry (backend auto|ltx|wan|"
                     "motion, registry model/LoRA selection), then "
                     "run the frame/Seer QA loop on each clip. Omit "
                     "shot_ids to shoot everything planned."),
        capability=Capability.FS_WRITE,
    )
    def movie_shoot(production_id: str, shot_ids: str = "",
                    backend: str = "auto", model: str = "",
                    qa: bool = True) -> dict[str, Any]:
        from ..media.director import DirectorError
        ids = [s.strip() for s in shot_ids.split(",") if s.strip()]
        try:
            prod = _director(context).shoot(
                production_id, shot_ids=ids or None, backend=backend,
                model=model or None, qa=qa)
        except DirectorError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "production_id": prod.id,
                "shots": [{"id": s.id, "status": s.status,
                           "clip": s.clip_path, "engine": s.engine,
                           "qa": s.qa} for s in prod.shots]}

    @registry.register(
        "movie_regen",
        description=("Regenerate ONE failed shot of a production "
                     "(fresh seed, prompt emphasis from the QA "
                     "notes), then re-run QA. Each shot gets exactly "
                     "one regeneration."),
        capability=Capability.FS_WRITE,
    )
    def movie_regen(production_id: str, shot_id: str,
                    backend: str = "auto", model: str = ""
                    ) -> dict[str, Any]:
        from ..media.director import DirectorError
        try:
            prod = _director(context).regen(
                production_id, shot_id, backend=backend,
                model=model or None)
        except DirectorError as exc:
            return {"ok": False, "error": str(exc)}
        shot = prod.shot(shot_id)
        return {"ok": True, "production_id": prod.id,
                "shot": {"id": shot.id, "status": shot.status,
                         "clip": shot.clip_path, "qa": shot.qa}}

    @registry.register(
        "movie_show",
        description=("Show a production's state: shot statuses, QA "
                     "verdicts, and where the plan/report live. Omit "
                     "production_id for the newest production."),
        capability=Capability.FS_READ,
    )
    def movie_show(production_id: str = "") -> dict[str, Any]:
        from ..media.director import DirectorError
        d = _director(context)
        try:
            prod = d.load(production_id)
        except DirectorError as exc:
            return {"ok": False, "error": str(exc),
                    "productions": d.list_productions()}
        return {"ok": True, "production_id": prod.id,
                "title": prod.title, "style_pack": prod.style_pack,
                "dir": prod.dir, "summary": d.summary(production_id),
                "shots": [{"id": s.id, "kind": s.kind,
                           "status": s.status, "clip": s.clip_path}
                          for s in prod.shots]}
