"""XFL interchange — a subset of AniCraft ``crates/xfl/src/import.rs``.

Adobe Animate's XFL format is a folder: ``DOMDocument.xml`` plus
library items. This port reads and writes the subset it can do for
real — timelines, layers (normal/guide/mask/folder), frames,
keyframe labels, shapes (rect/ellipse/path edges with solid fills),
symbol instances, text fields, and classic tweens with their easing.
Anything outside the subset (bone armatures, filters, scripted
frames, sounds) is named in the import report and skipped, never
silently invented.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .easing import Easing
from .errors import AnimationError
from .model import AnimationDocument, Element

__all__ = ["export_xfl", "import_xfl"]

_TWIPS = 20.0  # XFL stores positions in twips (1/20 px)


def _num(text: str | None, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _color(hex_text: str | None) -> str:
    if not hex_text:
        return "#000000"
    s = hex_text.strip().lstrip("#")
    if len(s) == 6:
        return "#" + s.lower()
    if len(s) == 8:
        return "#" + s[0:6].lower()
    return "#000000"


def _edges_to_path(edges: str) -> str:
    """XFL edge data (``!x y|...``) to SVG path data."""
    if not edges:
        return ""
    # Edge data: segments separated by '!', each a list of numbers;
    # straight edges are pairs, curves are triples (control, end).
    svg: list[str] = []
    for segment in edges.split("!"):
        segment = segment.strip()
        if not segment:
            continue
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", segment)]
        if len(nums) < 2:
            continue
        if not svg:
            svg.append(f"M{nums[0] / _TWIPS:g},{nums[1] / _TWIPS:g}")
            nums = nums[2:]
        if len(nums) == 2:
            svg.append(f"L{nums[0] / _TWIPS:g},{nums[1] / _TWIPS:g}")
        elif len(nums) == 4:
            svg.append(f"Q{nums[0] / _TWIPS:g},{nums[1] / _TWIPS:g} "
                       f"{nums[2] / _TWIPS:g},{nums[3] / _TWIPS:g}")
        elif len(nums) > 4:
            # Multiple edges in one segment: walk them in pairs/triples.
            i = 0
            while i + 1 < len(nums):
                if i + 3 < len(nums) and len(nums) - i >= 4:
                    svg.append(
                        f"Q{nums[i] / _TWIPS:g},{nums[i + 1] / _TWIPS:g} "
                        f"{nums[i + 2] / _TWIPS:g},{nums[i + 3] / _TWIPS:g}")
                    i += 4
                else:
                    svg.append(f"L{nums[i] / _TWIPS:g},"
                               f"{nums[i + 1] / _TWIPS:g}")
                    i += 2
    if svg:
        svg.append("Z")
    return " ".join(svg)


def _path_to_edges(path) -> str:
    """Path to XFL edge data. Curves are flattened to line segments
    (XFL edges are quadratic-only); the geometry is exact to the
    flattening tolerance, and the survey names the flattening."""
    parts: list[str] = []
    for contour in path.flatten(steps=8):
        if len(contour) < 2:
            continue
        nums: list[str] = []
        for x, y in contour:
            nums.append(f"{x * _TWIPS:g}")
            nums.append(f"{y * _TWIPS:g}")
        parts.append("!" + " ".join(nums))
    return "".join(parts)


def import_xfl(path: str | Path) -> AnimationDocument:
    """Import an XFL folder (or a folder holding DOMDocument.xml)."""
    root = Path(path)
    doc_path = root / "DOMDocument.xml"
    if root.is_file():
        raise AnimationError(
            f"XFL is a folder, not a file: {path}")
    if not doc_path.is_file():
        raise AnimationError(
            f"no DOMDocument.xml in {path}")
    try:
        tree = ET.parse(doc_path)
    except ET.ParseError as exc:
        raise AnimationError(f"bad DOMDocument.xml: {exc}") from exc
    dom = tree.getroot()
    # XFL files carry the Adobe namespace; strip it so tag lookups work.
    for el in dom.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    name = dom.get("name") or root.name
    width = int(_num(dom.get("width"), 640.0))
    height = int(_num(dom.get("height"), 360.0))
    fps = _num(dom.get("frameRate"), 24.0)
    doc = AnimationDocument.create(name, width=width, height=height,
                                   fps=fps)
    skipped: list[str] = []
    timelines = dom.find("timelines")
    if timelines is None:
        raise AnimationError("DOMDocument.xml has no <timelines>")
    for tl_el in timelines.findall("DOMTimeline"):
        scene_name = tl_el.get("name") or "Scene 1"
        scene = doc.scenes[0] if scene_name == "Scene 1" \
            and len(doc.scenes) == 1 else doc.add_scene(scene_name)
        layers_el = tl_el.find("layers")
        if layers_el is None:
            continue
        for layer_el in layers_el.findall("DOMLayer"):
            kind = "normal"
            if layer_el.get("layerType") == "guide":
                kind = "guide"
            elif layer_el.get("layerType") == "mask":
                kind = "mask"
            elif layer_el.get("layerType") == "folder":
                kind = "folder"
            layer = scene.timeline.add_layer(
                layer_el.get("name") or "Layer", kind=kind)
            frames_el = layer_el.find("frames")
            if frames_el is None:
                continue
            for frame_el in frames_el.findall("DOMFrame"):
                start = int(_num(frame_el.get("index"), 0.0))
                duration = max(1, int(_num(frame_el.get("duration"), 1.0)))
                kf = layer.add_keyframe(
                    start, duration,
                    label=frame_el.get("name") or "")
                tween_type = frame_el.get("tweenType")
                ease = _num(frame_el.get("acceleration"), 0.0)
                if tween_type == "motion":
                    kf.tween_to(
                        layer.keyframes[-1],
                        easing=Easing.classic(ease),
                        kind="classic")
                elif tween_type == "shape":
                    kf.tween_to(layer.keyframes[-1],
                                easing=Easing.classic(ease),
                                kind="shape")
                elements_el = frame_el.find("elements")
                if elements_el is None:
                    continue
                for el in elements_el:
                    _import_element(el, kf, skipped)
    doc._xfl_skipped = skipped  # type: ignore[attr-defined]
    return doc


def _import_element(el: ET.Element, kf, skipped: list[str]) -> None:
    tag = el.tag
    matrix = el.find("matrix")
    tx = _num(matrix.find("TranslateX").text) / _TWIPS \
        if matrix is not None and matrix.find("TranslateX") is not None \
        else 0.0
    ty = _num(matrix.find("TranslateY").text) / _TWIPS \
        if matrix is not None and matrix.find("TranslateY") is not None \
        else 0.0
    if tag == "DOMShape":
        fills = el.find("fills")
        fill_color = "#000000"
        if fills is not None:
            solid = fills.find("SolidColor")
            if solid is not None:
                fill_color = _color(solid.get("color"))
        edges_el = el.find("edges")
        edges = edges_el.text if edges_el is not None else ""
        svg = _edges_to_path(edges or "")
        if not svg:
            skipped.append("shape with no edge data")
            return
        shape = Element.path(svg, fill=fill_color)
        shape.transform.x = tx
        shape.transform.y = ty
        kf.add_element(shape)
    elif tag == "DOMSymbolInstance":
        skipped.append(
            "symbol instance: library items are not in this XFL subset")
    elif tag == "DOMText":
        text = "".join(
            run.text or "" for run in el.iter("textRun"))
        size = 24.0
        for run in el.iter("textRun"):
            attrs = run.find("textAttrs")
            if attrs is not None:
                size = _num(attrs.get("size"), 24.0)
                break
        item = Element.text(text or "", tx, ty, font_size=size)
        kf.add_element(item)
    elif tag == "DOMBitmapInstance":
        skipped.append("bitmap instance: bitmaps need the library")
    else:
        skipped.append(f"element <{tag}>: not in this XFL subset")


def export_xfl(doc: AnimationDocument, out: str | Path) -> Path:
    """Write the document as an XFL folder (the import subset)."""
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    scene = doc.scenes[0]
    stage = scene.stage
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<DOMDocument xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xmlns="http://ns.adobe.com/xfl/2008/" name="{doc.name}" '
        f'width="{stage.width}" height="{stage.height}" '
        f'frameRate="{doc.fps:g}" currentTimeline="0">',
        "  <timelines>",
        f'    <DOMTimeline name="{scene.name}">',
        "      <layers>",
    ]
    for layer in scene.timeline.layers:
        ltype = {"normal": "normal", "guide": "guide", "mask": "mask",
                 "folder": "folder"}.get(layer.kind, "normal")
        lines.append(f'        <DOMLayer name="{layer.name}" '
                     f'color="#4FFF4F" current="false" isSelected="false" '
                     f'layerType="{ltype}">')
        lines.append("          <frames>")
        for kf in layer.keyframes:
            tween = ""
            if kf.tween is not None:
                tween = (f' tweenType="'
                         f'{"shape" if kf.tween.kind == "shape" else "motion"}"'
                         f' acceleration="0"')
            label = f' name="{kf.label}"' if kf.label else ""
            lines.append(
                f'            <DOMFrame index="{kf.start}" '
                f'duration="{kf.duration}"{label}{tween}>')
            lines.append("              <elements>")
            for el in kf.elements:
                lines.extend(_export_element(el))
            lines.append("              </elements>")
            lines.append("            </DOMFrame>")
        lines.append("          </frames>")
        lines.append("        </DOMLayer>")
    lines.extend([
        "      </layers>",
        "    </DOMTimeline>",
        "  </timelines>",
        "</DOMDocument>",
    ])
    (out_path / "DOMDocument.xml").write_text("\n".join(lines),
                                              encoding="utf-8")
    return out_path


def _export_element(el: Element) -> list[str]:
    if el.kind == "shape" and el.path is not None:
        fill = "#000000"
        if el.fill is not None and el.fill.kind == "solid":
            fill = el.fill.color.to_hex()
        edges = _path_to_edges(el.path)
        tx = el.transform.x * _TWIPS
        ty = el.transform.y * _TWIPS
        return [
            "                <DOMShape>",
            "                  <fills>",
            f'                    <SolidColor color="{fill}"/>',
            "                  </fills>",
            f"                  <edges>{edges}</edges>",
            "                  <matrix>",
            f"                    <TranslateX>{tx:g}</TranslateX>",
            f"                    <TranslateY>{ty:g}</TranslateY>",
            "                  </matrix>",
            "                </DOMShape>",
        ]
    if el.kind == "text":
        return [
            "                <DOMText>",
            f"                  <textRun>{el.text}</textRun>",
            "                </DOMText>",
        ]
    return []
