"""Tier 1 — 7-stage pipeline: BRIEF -> DRAFT -> CULL -> REFINE -> FIX
-> FINISH -> QA, with a quality ladder (preview/standard/max) and a
contact sheet.

The pipeline is renderer-agnostic: the caller supplies ``generate``
(a brief + count + seed -> list of candidate dicts, each carrying at
least a ``scores`` dict) and ``refine`` / ``fix`` / ``finish`` hooks
(defaults are identity + re-critique). Every stage is checkable: CULL
drops anything below the floor, REFINE keeps the top 3, FIX applies
regional inpaint descriptors, FINISH applies the deterministic finish
ops (upscale/grade/text), QA re-critiques and refuses to ship below
the floor. The contact sheet is a real PIL grid of candidate images
when candidates carry ``image`` entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .critique import Critique, critique_render, passes_floor

QUALITY_LADDER = {
    # rung: (draft_count, refine_keep, finish_ops, description)
    "preview": (8, 2, ("grade",), "cheap, fast, for direction checks"),
    "standard": (20, 3, ("grade", "upscale"), "default delivery rung"),
    "max": (50, 3, ("grade", "upscale", "text"), "full quality, slowest"),
}

STAGES = ("brief", "draft", "cull", "refine", "fix", "finish", "qa")


@dataclass
class PipelineResult:
    brief: str
    rung: str
    drafts: list[dict[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    refined: list[dict[str, Any]] = field(default_factory=list)
    fixed: list[dict[str, Any]] = field(default_factory=list)
    finished: list[dict[str, Any]] = field(default_factory=list)
    qa: list[Critique] = field(default_factory=list)
    shipped: list[dict[str, Any]] = field(default_factory=list)
    stage_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"brief": self.brief, "rung": self.rung,
                "drafts": len(self.drafts), "kept": len(self.kept),
                "refined": len(self.refined), "fixed": len(self.fixed),
                "finished": len(self.finished),
                "shipped": len(self.shipped),
                "qa": [c.to_dict() for c in self.qa],
                "stage_log": list(self.stage_log)}


def run_pipeline(brief: str, *, rung: str = "standard",
                 generate: Callable[..., list[dict[str, Any]]] | None = None,
                 refine: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 fix: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 finish: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
                 seed: int = 0) -> PipelineResult:
    """Run all seven stages. ``generate`` is the only model call.

    Default hooks are pure pass-throughs that re-score; a caller that
    wires a real model passes it in. CULL keeps only floor-passing
    drafts; if none pass, the result ships nothing (never garbage).
    """
    if rung not in QUALITY_LADDER:
        raise ValueError(f"unknown rung {rung!r}; use {sorted(QUALITY_LADDER)}")
    draft_count, keep_n, finish_ops, _desc = QUALITY_LADDER[rung]
    result = PipelineResult(brief=brief, rung=rung)
    gen = generate or (lambda brief, count, seed: [
        {"id": f"draft-{i}", "brief": brief, "seed": seed + i,
         "scores": {"prompt_adherence": 0.7, "technical": 0.7,
                    "composition": 0.6, "consistency": 0.6, "aesthetic": 0.6}}
        for i in range(count)])
    refine_fn = refine or (lambda c: dict(c))
    fix_fn = fix or (lambda c: dict(c))
    finish_fn = finish or (lambda c: dict(c, finish_ops=list(finish_ops)))

    result.drafts = gen(brief, draft_count, seed)
    result.stage_log.append(f"draft: {len(result.drafts)} candidates")
    for cand in result.drafts:
        crit = critique_render(brief, scores=dict(cand.get("scores", {})),
                               image=cand.get("image"))
        cand["_critique"] = crit
        if passes_floor(crit):
            result.kept.append(cand)
    result.stage_log.append(f"cull: {len(result.kept)} above floor")
    ranked = sorted(result.kept, key=lambda c: c["_critique"].overall, reverse=True)
    result.refined = [refine_fn(dict(c)) for c in ranked[:keep_n]]
    result.stage_log.append(f"refine: top {len(result.refined)}")
    result.fixed = [fix_fn(dict(c)) for c in result.refined]
    result.stage_log.append(f"fix: {len(result.fixed)} regional fixes applied")
    result.finished = [finish_fn(dict(c)) for c in result.fixed]
    result.stage_log.append(f"finish: {finish_ops}")
    for cand in result.finished:
        crit = critique_render(brief, scores=dict(cand.get("scores", {})),
                               image=cand.get("image"))
        result.qa.append(crit)
        if passes_floor(crit):
            result.shipped.append(cand)
    result.stage_log.append(f"qa: {len(result.shipped)} shipped")
    return result


def contact_sheet(candidates: list[dict[str, Any]], *,
                  cols: int = 4, thumb: int = 256) -> Any:
    """Real PIL contact sheet grid of candidates carrying ``image``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("contact_sheet needs Pillow") from exc
    images = [c["image"] for c in candidates if isinstance(c.get("image"), Image.Image)]
    if not images:
        raise ValueError("contact_sheet needs candidates with PIL images")
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (16, 16, 16))
    for i, img in enumerate(images):
        sheet.paste(img.convert("RGB").resize((thumb, thumb)),
                    ((i % cols) * thumb, (i // cols) * thumb))
    return sheet
