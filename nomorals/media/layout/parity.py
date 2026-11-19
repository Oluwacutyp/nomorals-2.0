"""Machine-checked parity for the layout organ.

A parity TOML lists real design fixtures + operations + expected
outcomes; :func:`run_parity` runs every case and fails on any
mismatch. A missing fixture FAILS its case — never a silent skip
(the same habit as the pdfcraft/wordcraft/vectorcraft organs).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .codec import parse_document
from .compose import compose_story
from .edit import LayoutEditor
from .errors import LayoutError
from .flex import layout_all
from .render import render_document

__all__ = ["PARITY_TOML", "run_parity"]

PARITY_TOML = Path(__file__).resolve().parent.parent \
    / "layout-parity.toml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_fixture(name: str, toml_path: Path) -> Path | None:
    path = Path(name)
    candidates = [path] if path.is_absolute() else [
        toml_path.parent / name,
        _repo_root() / "tests" / "fixtures" / "layout" / name,
        _repo_root() / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _facts_for(op: str, raw: bytes) -> dict:
    """Run one op on a fresh parse of the fixture and measure the
    facts it produces."""
    doc = parse_document(raw)
    editor = LayoutEditor(doc)
    if op == "inspect":
        return {"items": doc.item_count(), "pages": doc.page_count(),
                "stories": doc.story_count(),
                "layers": len(doc.layers)}
    if op == "render":
        pngs = render_document(doc)
        return {"pngs": len(pngs),
                "png_ok": all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)}
    if op == "roundtrip":
        return {"byte_identical": doc.to_bytes() == raw}
    if op == "move":
        item = doc.all_items()[0]
        editor.move(item.id, 50.0, 50.0)
        return {"first_x": item.xf.e, "first_y": item.xf.f}
    if op == "flex":
        container = next((i for i in doc.all_items()
                          if i.flex is not None), None)
        if container is None:
            raise LayoutError("no flex container in fixture",
                              kind="not_found")
        layout_all(doc.all_items())
        kids = container.content_items()
        return {"first_child_x": kids[0].xf.e,
                "last_child_x": kids[-1].xf.e}
    if op == "compose":
        text_items = [i for i in doc.all_items()
                      if i.content_type() == "text"]
        if not text_items:
            raise LayoutError("no text frame in fixture",
                              kind="not_found")
        item = max(text_items,
                   key=lambda i: (i.local_bounds()[2] - i.local_bounds()[0])
                   * (i.local_bounds()[3] - i.local_bounds()[1]))
        story = doc.stories.get(item.story_id or "")
        if story is None:
            raise LayoutError("text frame has no story",
                              kind="not_found")
        x0, y0, x1, y1 = item.local_bounds()
        size = 12.0
        for para in story.paras:
            for run in para.runs:
                if run.font_size is not None:
                    size = float(run.font_size)
        composed = compose_story(story, (x1 - x0, y1 - y0),
                                 font_size=size)
        return {"overset": composed.overset,
                "lines": len(composed.lines)}
    if op == "thread":
        items = [i for i in doc.all_items()
                 if i.content_type() == "text"]
        by_id = {i.id: i for i in items}
        best: list = []
        for item in items:
            chain = [item]
            seen = {item.id}
            while True:
                nxt = chain[-1].text_options().next_frame_id
                if nxt is None or nxt not in by_id or nxt in seen:
                    break
                chain.append(by_id[nxt])
                seen.add(nxt)
            if len(chain) > len(best):
                best = chain
        if len(best) < 2:
            raise LayoutError("no threaded chain in fixture",
                              kind="not_found")
        story = doc.stories.get(best[0].story_id or "")
        if story is None:
            raise LayoutError("threaded chain has no story",
                              kind="not_found")
        frames = []
        for member in best:
            x0, y0, x1, y1 = member.local_bounds()
            frames.append((x1 - x0, y1 - y0))
        size = 12.0
        for para in story.paras:
            for run in para.runs:
                if run.font_size is not None:
                    size = float(run.font_size)
        composed = compose_story(story, frames, font_size=size)
        return {"total_lines": len(composed.lines),
                "first_frame_lines": len(composed.frames[0].lines)
                if composed.frames else 0,
                "overset": composed.overset}
    if op == "group":
        items = doc.all_items()[:2]
        group = editor.group([i.id for i in items])
        return {"group_children": len(group.content_items())}
    if op == "snap":
        page = doc.spreads[0].pages[0]
        editor.add_guide(page.id, "vertical", 100.0)
        item = doc.all_items()[0]
        editor.move(item.id, 103.0, item.xf.f)
        snapped = editor.snap(item.id, tolerance=5.0)
        return {"snapped": snapped, "snapped_x": item.xf.e}
    if op == "liquid":
        page = doc.spreads[0].pages[0]
        item = doc.all_items()[0]
        _, _, x1, y1 = doc.item_bounds(item)
        right_before = page.width - x1
        bottom_before = page.height - y1
        editor.set_liquid(item.id, pin_right=True, pin_bottom=True)
        editor.resize_page(page.id, page.width + 100.0,
                           page.height + 100.0)
        _, _, x1, y1 = doc.item_bounds(item)
        right_after = page.width - x1
        bottom_after = page.height - y1
        return {"kept_right": abs(right_after - right_before) < 0.001,
                "kept_bottom": abs(bottom_after - bottom_before) < 0.001}
    if op == "columns":
        page = doc.spreads[0].pages[0]
        margin_width = (page.width - page.margins.inside
                        - page.margins.outside)
        return {"edges": page.columns.edges(margin_width)}
    raise LayoutError(f"unknown parity op {op!r}",
                      kind="invalid_argument")


def _check_expect(facts: dict, expect: dict) -> list[str]:
    mismatches: list[str] = []
    for key, wanted in expect.items():
        got = facts.get(key, "<missing>")
        if isinstance(wanted, float):
            if not isinstance(got, (int, float)) or abs(got - wanted) > 0.001:
                mismatches.append(f"{key}: got {got!r}, want {wanted!r}")
        elif got != wanted:
            mismatches.append(f"{key}: got {got!r}, want {wanted!r}")
    return mismatches


def run_parity(toml_path: str | Path | None = None) -> dict:
    """Run the parity TOML (default: the shipped
    ``layout-parity.toml``). Returns ``{"passed": n, "failed":
    [ids], "cases": [...]}``."""
    path = Path(toml_path) if toml_path is not None else PARITY_TOML
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {"passed": 0, "failed": ["<toml>"],
                "cases": [{"id": "<toml>", "status": "fail",
                           "detail": f"cannot read parity TOML: {exc}"}]}
    cases = data.get("cases", [])
    passed = 0
    failed: list[str] = []
    results: list[dict] = []
    for case in cases:
        case_id = str(case.get("id", "<unnamed>"))
        fixture = _resolve_fixture(str(case.get("fixture", "")),
                                   path)
        if fixture is None:
            failed.append(case_id)
            results.append({"id": case_id, "status": "fail",
                            "detail": f"missing fixture "
                            f"{case.get('fixture')!r}"})
            continue
        try:
            raw = fixture.read_bytes()
            facts: dict = {}
            for op in case.get("ops", ["inspect"]):
                facts.update(_facts_for(str(op), raw))
            mismatches = _check_expect(facts, dict(case.get("expect", {})))
        except LayoutError as exc:
            mismatches = [f"{exc.kind}: {exc}"]
        if mismatches:
            failed.append(case_id)
            results.append({"id": case_id, "status": "fail",
                            "detail": "; ".join(mismatches)})
        else:
            passed += 1
            results.append({"id": case_id, "status": "pass",
                            "detail": ""})
    return {"passed": passed, "failed": failed, "cases": results}
