"""The Director — coverage → plan → shoot → QA, as staged artifacts.

A production is a directory on disk::

    <root>/<production-id>/
        plan.json       screenplay + shots + designs + live status
        coverage.json   the reader's report (when covered)
        coverage.md
        report.md       human summary, refreshed at every stage
        clips/          generated shot clips (+ QA frames beside them)

Stages are separate calls on purpose — the conversational gate is
the product: the owner reads coverage, approves the plan, then
shoots. Every stage reloads ``plan.json`` from disk, so stages can
happen in different conversations and survive restarts.

Generation goes through the shipped videogen front door
(:func:`nomorals.media.videogen.generate`) — registry model/LoRA
selection and the profile/VRAM capability gate included; this layer
adds directing, not a second generator.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ...core.logging_setup import get_logger
from .breakdown import Shot, break_down, break_down_prose
from .coverage import Coverage, cover_script
from .design import design_shot, get_style_pack, refine_design
from .errors import DirectorError
from .prompts import assemble_prompt
from .qa import qa_shot
from .screenplay import Screenplay, parse_screenplay

_log = get_logger(__name__)

__all__ = ["Director", "Production"]


def _slug(text: str, fallback: str = "untitled") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or fallback


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class Production:
    id: str
    title: str
    style_pack: str
    screenplay: Screenplay
    shots: list[Shot] = field(default_factory=list)
    coverage: Coverage | None = None
    dir: str = ""

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise DirectorError(
            f"production {self.id} has no shot {shot_id!r} — "
            f"shots: {', '.join(s.id for s in self.shots)}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title,
                "style_pack": self.style_pack,
                "screenplay": self.screenplay.to_dict(),
                "shots": [s.to_dict() for s in self.shots],
                "coverage": self.coverage.to_dict()
                if self.coverage else None}

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, dir: str = "") -> "Production":
        cov = d.get("coverage")
        return cls(
            id=str(d.get("id", "")), title=str(d.get("title", "")),
            style_pack=str(d.get("style_pack", "neutral")),
            screenplay=Screenplay.from_dict(d.get("screenplay") or {}),
            shots=[Shot.from_dict(s) for s in d.get("shots", [])],
            coverage=Coverage.from_dict(cov)
            if isinstance(cov, dict) else None,
            dir=dir)


class Director:
    """The director layer over the videogen registry.

    Seams (all default to the real organs): ``brain`` (coverage /
    prose segmentation / design refinement / QA judging), ``seer``
    (frame description), ``generate_fn`` (videogen front door),
    ``extract_frames_fn`` (cv_video). ``root`` is where productions
    live; default ``<workspace>/generated/director``.
    """

    def __init__(self, context: Any = None, *, brain: Any = None,
                 seer: Any = None,
                 generate_fn: Callable[..., Any] | None = None,
                 extract_frames_fn: Callable[[str], list[str]] | None
                 = None,
                 root: str | Path | None = None) -> None:
        self.context = context
        self._brain = brain
        self._seer = seer
        self._generate_fn = generate_fn
        self._extract_frames_fn = extract_frames_fn
        if root is not None:
            self.root = Path(root)
        else:
            base = Path(getattr(context, "workspace", None) or ".")
            self.root = base / "generated" / "director"

    # -- seams ------------------------------------------------------------
    @property
    def brain(self) -> Any:
        if self._brain is None:
            from ...llm.brain import brain_for
            self._brain = brain_for(self.context)
        return self._brain

    @property
    def seer(self) -> Any:
        if self._seer is None:
            from ...vision.seer import Seer
            self._seer = Seer(context=self.context)
        return self._seer

    def _generate(self, prompt: str, **kw: Any) -> Any:
        if self._generate_fn is not None:
            return self._generate_fn(prompt, **kw)
        from ..videogen import generate
        return generate(prompt, **kw)

    # -- stage 1: coverage --------------------------------------------------
    def cover(self, text: str, *, title: str = "") -> Coverage:
        """Coverage for a script text. Brain-required, honest failure."""
        play = parse_screenplay(text, title=title)
        return cover_script(text, brain=self.brain, play=play,
                            title=title)

    def save_coverage(self, cov: Coverage, *, title: str = "") -> str:
        """Persist a standalone coverage report (the pre-plan stage
        gate) under ``<root>/coverage/<slug>-<stamp>/``; returns the
        report path. A production's own coverage is saved by
        :meth:`save` when one is attached."""
        d = self.root / "coverage" / f"{_slug(title or cov.title)}-{_stamp()}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "coverage.json").write_text(
            json.dumps(cov.to_dict(), indent=1), encoding="utf-8")
        path = d / "coverage.md"
        path.write_text(cov.report(), encoding="utf-8")
        return str(path)

    # -- stage 2: plan (breakdown + design) ---------------------------------
    def plan(self, text: str, *, title: str = "",
             style_pack: str = "neutral",
             refine: bool = False,
             coverage: Coverage | None = None) -> Production:
        """Break a script down and design every shot. Persists the
        production and returns it. ``coverage`` (from :meth:`cover`)
        is attached to the production when the caller ran the stages
        back to back."""
        pack = get_style_pack(style_pack)  # raises on unknown pack
        play = parse_screenplay(text, title=title)
        if play.is_standard:
            shots = break_down(play)
        else:
            play, shots = break_down_prose(text, brain=self.brain,
                                           title=title)
        scenes = {sc.number: sc for sc in play.scenes}
        for shot in shots:
            scene = scenes.get(shot.scene_number)
            shot.design = design_shot(
                shot.kind,
                time_of_day=scene.time_of_day if scene else "",
                interior=scene.interior if scene else "",
                style_pack=pack["name"])
            if refine:
                shot.design = refine_design(
                    shot.design,
                    f"{shot.description} (scene: "
                    f"{scene.heading if scene else ''})",
                    brain=self.brain)
        prod_title = title or play.title or "untitled"
        prod = Production(
            id=f"{_slug(prod_title)}-{_stamp()}", title=prod_title,
            style_pack=pack["name"], screenplay=play, shots=shots,
            coverage=coverage)
        prod.dir = str(self.root / prod.id)
        self.save(prod)
        return prod

    # -- persistence ----------------------------------------------------------
    def save(self, prod: Production) -> None:
        d = Path(prod.dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text(
            json.dumps(prod.to_dict(), indent=1), encoding="utf-8")
        if prod.coverage is not None:
            (d / "coverage.json").write_text(
                json.dumps(prod.coverage.to_dict(), indent=1),
                encoding="utf-8")
            (d / "coverage.md").write_text(prod.coverage.report(),
                                          encoding="utf-8")
        (d / "report.md").write_text(self._report(prod), encoding="utf-8")

    def load(self, production_id: str = "") -> Production:
        """Load a production by id (or the newest when id is empty)."""
        if production_id:
            d = self.root / production_id
        else:
            if not self.root.is_dir():
                raise DirectorError(
                    f"no productions yet — {self.root} does not exist")
            dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
            if not dirs:
                raise DirectorError("no productions yet")
            d = dirs[-1]
        plan_file = d / "plan.json"
        if not plan_file.is_file():
            raise DirectorError(
                f"no production at {d} (missing plan.json)")
        data = json.loads(plan_file.read_text(encoding="utf-8"))
        return Production.from_dict(data, dir=str(d))

    def list_productions(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "plan.json").is_file())

    # -- stage 3: shoot -------------------------------------------------------
    def shoot(self, production_id: str = "", *,
              shot_ids: list[str] | None = None, backend: str = "auto",
              model: str | None = None, lora: Any = None,
              qa: bool = True, seed: int = 0) -> Production:
        """Generate the planned shots (all, or ``shot_ids``) through
        the videogen registry machinery, then run the QA loop on each
        generated shot. Shot-level failures are recorded on the shot
        and in the report — the shoot continues and reports honestly.
        """
        from ..motion_studio._core import record_ledger
        from ..videogen import VideogenError

        prod = self.load(production_id)
        targets = [prod.shot(sid) for sid in shot_ids] if shot_ids \
            else list(prod.shots)
        clips_dir = Path(prod.dir) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for n, shot in enumerate(targets):
            prompt = assemble_prompt(shot)
            try:
                result = self._generate(
                    prompt, backend=backend, duration_s=shot.duration_s,
                    seed=(seed or 41) + n * 7, model=model, lora=lora,
                    out=str(clips_dir /
                            f"shot-{shot.id.replace('.', '-')}.mp4"))
            except VideogenError as exc:
                shot.status, shot.qa = "failed", {"notes": str(exc)}
                continue
            except Exception as exc:  # noqa: BLE001 - recorded per shot
                shot.status = "failed"
                shot.qa = {"notes": f"generation failed: {exc}"}
                continue
            src = Path(result.path)
            dest = clips_dir / f"shot-{shot.id.replace('.', '-')}.mp4"
            if src.is_file() and src.resolve() != dest.resolve():
                shutil.copyfile(src, dest)
                shot.clip_path = str(dest)
            else:
                shot.clip_path = str(src)
            shot.engine = getattr(result, "engine", "")
            shot.status = "generated"
            shot.qa = {"generation_note": getattr(result, "note", "")}
            record_ledger({"kind": "director.shot", "path": shot.clip_path,
                           "production": prod.id, "shot": shot.id,
                           "engine": shot.engine,
                           "routed": getattr(result, "routed", "")})
            if qa:
                self._run_qa(shot)
        self.save(prod)
        record_ledger({"kind": "director.production", "path": prod.dir,
                       "production": prod.id,
                       "shots": len(prod.shots)})
        return prod

    def _run_qa(self, shot: Shot) -> None:
        record = qa_shot(shot, shot.clip_path, seer=self._seer_or_none(),
                         brain=self.brain,
                         extract_frames_fn=self._extract_frames_fn)
        shot.qa.update(record)
        if record["status"] == "pass":
            shot.status = "qa_pass"
        elif record["status"] == "fail":
            shot.status = "qa_fail"
        else:
            shot.status = "qa_unavailable"

    def _seer_or_none(self) -> Any:
        """The Seer for QA; None (→ honest qa_unavailable) if it can't
        even be constructed. A constructed Seer that can't see still
        raises inside qa_shot and is reported there."""
        try:
            return self.seer
        except Exception as exc:  # noqa: BLE001
            _log.debug("director: seer construction failed: %s", exc)
            return None

    # -- stage 4: regenerate a failed shot (once) ------------------------------
    def regen(self, production_id: str, shot_id: str, *,
              backend: str = "auto", model: str | None = None,
              lora: Any = None) -> Production:
        """Regenerate one shot — once. The regen leans on the QA notes
        (the prompt gains an explicit subject emphasis) and a fresh
        seed, then re-runs QA. A shot that already used its regen, or
        was never generated, refuses honestly."""
        from ..videogen import VideogenError

        prod = self.load(production_id)
        shot = prod.shot(shot_id)
        if shot.regen_count >= 1:
            raise DirectorError(
                f"shot {shot_id} already used its one regeneration — "
                f"its status is {shot.status}; redesign the shot or "
                "accept it, a second blind retry is not directing")
        if shot.status not in ("qa_fail", "generated", "qa_unavailable",
                               "failed"):
            raise DirectorError(
                f"shot {shot_id} is {shot.status} — nothing to "
                "regenerate (shoot it first)")
        notes = str(shot.qa.get("notes") or "")
        emphasis = f"ensure the subject is clearly visible: {shot.subject}"
        prompt = assemble_prompt(shot) + f", {emphasis}"
        clips_dir = Path(prod.dir) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        dest = clips_dir / f"shot-{shot.id.replace('.', '-')}-regen.mp4"
        try:
            result = self._generate(
                prompt, backend=backend, duration_s=shot.duration_s,
                seed=1041 + shot.scene_number * 100 + shot.index,
                model=model, lora=lora, out=str(dest))
        except VideogenError as exc:
            shot.qa = {**shot.qa, "regen_error": str(exc)}
            self.save(prod)
            raise DirectorError(
                f"regeneration of shot {shot_id} failed: {exc}") from exc
        src = Path(result.path)
        if src.is_file() and src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
            shot.clip_path = str(dest)
        else:
            shot.clip_path = str(src)
        shot.engine = getattr(result, "engine", "")
        shot.regen_count += 1
        shot.status = "generated"
        shot.qa = {"regen_of_notes": notes,
                   "generation_note": getattr(result, "note", "")}
        self._run_qa(shot)
        if shot.status == "qa_pass":
            shot.status = "regenerated"  # passed after regen — say so
            shot.qa["post_regen_verdict"] = "pass"
        self.save(prod)
        return prod

    # -- reporting ---------------------------------------------------------------
    def _report(self, prod: Production) -> str:
        lines = [f"# Production: {prod.title} ({prod.id})",
                 f"style pack: {prod.style_pack} · "
                 f"{len(prod.screenplay.scenes)} scenes · "
                 f"{len(prod.shots)} shots", ""]
        if prod.coverage is not None:
            lines += ["## Coverage",
                      f"verdict: **{prod.coverage.verdict.upper()}** — "
                      + "; ".join(prod.coverage.verdict_reasons), ""]
        lines.append("## Shots")
        for s in prod.shots:
            qa_note = ""
            if s.qa.get("status"):
                qa_note = f" · QA {s.qa['status']}"
                if s.qa.get("notes"):
                    qa_note += f" ({s.qa['notes']})"
            lines.append(
                f"- **{s.id}** [{s.kind}] {s.status}{qa_note}\n"
                f"  {s.description}"
                + (f"\n  clip: {s.clip_path}" if s.clip_path else ""))
        return "\n".join(lines) + "\n"

    def summary(self, production_id: str = "") -> str:
        """Chat-sized status: counts by status + the QA picture."""
        prod = self.load(production_id)
        counts: dict[str, int] = {}
        for s in prod.shots:
            counts[s.status] = counts.get(s.status, 0) + 1
        tally = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines = [f"🎬 {prod.title} ({prod.id}) — {tally}"]
        for s in prod.shots:
            if s.status in ("qa_fail", "failed", "qa_unavailable"):
                lines.append(f"  ⚠ {s.id} {s.status}: "
                             f"{s.qa.get('notes', '')}")
        lines.append(f"plan: {prod.dir}/report.md")
        return "\n".join(lines)
