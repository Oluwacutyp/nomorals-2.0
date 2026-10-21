"""``nm director`` — Devon directs, on the command line.

    nm director cover <script-file>
    nm director plan <script-file> [--style neutral|naturalist|noir|neon|golden] [--refine]
    nm director shoot <production-id> [--backend auto|ltx|wan|motion]
                    [--model NAME] [--lora NAME[:W]] [--shots 1.1,2.3] [--no-qa]
    nm director regen <production-id> <shot-id> [--backend X] [--model NAME]
    nm director show [production-id]
    nm director styles

The pipeline: coverage (brain-read, industry verdict) → breakdown +
lighting/camera design (primitives, style packs are data) →
generation through the videogen registry with its capability gate →
frame/Seer QA with one flagged regeneration per failed shot.
Nothing here raises a traceback — failures print cleanly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _err(msg: str) -> int:
    print(f"director: {msg}", file=sys.stderr)
    return 1


def _ok(msg: str) -> int:
    print(msg)
    return 0


def _director(context):
    from ...media.director import Director
    root = None
    ws = getattr(context, "workspace", None)
    if ws:
        root = Path(ws) / "generated" / "director"
    return Director(context=context, root=root)


def _read(args) -> str:
    p = Path(args.script)
    if not p.is_file():
        raise ValueError(f"script file not found: {args.script}")
    return p.read_text(encoding="utf-8", errors="replace")


def _cmd_cover(args: argparse.Namespace, context) -> int:
    from ...media.director import DirectorError
    try:
        d = _director(context)
        cov = d.cover(_read(args))
        path = d.save_coverage(cov)
    except (DirectorError, ValueError) as exc:
        return _err(str(exc))
    return _ok(cov.report() + f"\n\nsaved: {path}")


def _cmd_plan(args: argparse.Namespace, context) -> int:
    from ...media.director import DirectorError
    try:
        prod = _director(context).plan(
            _read(args), style_pack=args.style, refine=args.refine)
    except (DirectorError, ValueError) as exc:
        return _err(str(exc))
    lines = [f"🎬 production {prod.id} — {prod.title}",
             f"style pack: {prod.style_pack} · "
             f"{len(prod.screenplay.scenes)} scenes · "
             f"{len(prod.shots)} shots"]
    for s in prod.shots:
        lines.append(f"  {s.id} [{s.kind}] {s.description[:80]}")
    lines.append(f"plan: {prod.dir}/report.md")
    lines.append(f"shoot: nm director shoot {prod.id}")
    return _ok("\n".join(lines))


def _cmd_shoot(args: argparse.Namespace, context) -> int:
    from ...media.director import DirectorError
    d = _director(context)
    shot_ids = [s for s in (args.shots or "").split(",") if s]
    try:
        prod = d.shoot(args.production, shot_ids=shot_ids or None,
                       backend=args.backend, model=args.model or None,
                       lora=args.lora or None, qa=not args.no_qa)
    except DirectorError as exc:
        return _err(str(exc))
    return _ok(d.summary(prod.id))


def _cmd_regen(args: argparse.Namespace, context) -> int:
    from ...media.director import DirectorError
    d = _director(context)
    try:
        prod = d.regen(args.production, args.shot,
                       backend=args.backend, model=args.model or None)
    except DirectorError as exc:
        return _err(str(exc))
    shot = prod.shot(args.shot)
    return _ok(f"🎬 shot {shot.id} → {shot.status}\n"
               f"clip: {shot.clip_path}\nQA: {shot.qa.get('notes', '')}")


def _cmd_show(args: argparse.Namespace, context) -> int:
    from ...media.director import DirectorError
    d = _director(context)
    try:
        return _ok(d.summary(args.production or ""))
    except DirectorError as exc:
        prods = d.list_productions()
        if prods:
            return _err(f"{exc}\nproductions: {', '.join(prods)}")
        return _err(str(exc))


def _cmd_styles(args: argparse.Namespace, context) -> int:
    from ...media.director import STYLE_PACKS
    lines = ["style packs (data over the design primitives):"]
    for name, pack in sorted(STYLE_PACKS.items()):
        lines.append(f"  {name:12} {pack['label']}")
    return _ok("\n".join(lines))


def cmd_director(args: argparse.Namespace, context) -> int:
    action = getattr(args, "director_action", "")
    fn = {"cover": _cmd_cover, "plan": _cmd_plan, "shoot": _cmd_shoot,
          "regen": _cmd_regen, "show": _cmd_show,
          "styles": _cmd_styles}.get(action)
    if fn is None:
        return _err("usage: nm director cover|plan|shoot|regen|show|"
                    "styles ...")
    return fn(args, context)
