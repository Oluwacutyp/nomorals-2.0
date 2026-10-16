"""Vision -> critique bridge: wire vision outputs into the critique engine.

Vision is a sensor, not a judge: :mod:`nomorals.vision.native` already
produces deterministic perceptual hashes and pixel diffs. This module
is the bridge that turns those sensor readings into critique-engine
checklist items — so a visual artifact (a rendered page, a screenshot,
a generated image) can be critiqued with the same checklist /
floor / verdict machinery as prose, instead of living in a separate
unverifiable "looks fine" channel.

Two honesty rules, enforced by construction:

- :func:`compare_images` is pure sensor output — deterministic for the
  same byte inputs, no model involved.
- :func:`describe_screen` only ever reports the ``elements`` the caller
  supplies. It never invents buttons, inputs, or text that were not in
  the input. An empty ``elements`` list means "no elements supplied —
  none inferred", not a hallucinated screen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "compare_images",
    "critique_from_vision",
    "visual_dimension",
    "describe_screen",
    "SCREEN_ELEMENT_KINDS",
]

#: Element kinds :func:`describe_screen` recognises. Anything else is
#: dropped (with a count) rather than guessed into a known kind.
SCREEN_ELEMENT_KINDS = ("button", "input", "text", "link", "image",
                        "checkbox", "select", "heading")

#: Diff thresholds (changed fraction of pixels, threshold 25/255 —
#: same threshold as :func:`nomorals.vision.native.compare_native`).
IDENTICAL_MAX_FRACTION = 0.0
NEAR_DUPLICATE_MAX_FRACTION = 0.01
SIMILAR_MAX_FRACTION = 0.10


def _read_bytes(data: bytes | str | Path) -> bytes | None:
    """Accept raw bytes or a path to an image file. None when unreadable."""
    try:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        path = Path(str(data))
        if path.is_file():
            return path.read_bytes()
        return None
    except (OSError, TypeError, ValueError):
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_images(a: bytes | str | Path,
                   b: bytes | str | Path) -> dict[str, Any]:
    """Deterministic sensor comparison of two images.

    Returns ``hash_a`` / ``hash_b`` (perceptual dhash, 16 hex chars),
    ``hash_distance`` (Hamming, 0-64), ``diff_score`` (0..1 — the
    fraction of pixels that changed, so 0.0 means pixel-identical),
    ``mean_abs_diff`` (0-255), ``identical`` and ``regions`` (a list
    of changed bboxes in 0-1000 coords; empty when identical).
    Same byte inputs always produce the same output — no model, no
    randomness. Returns ``{"ok": False, "reason": ...}`` when either
    input cannot be decoded (never raises).
    """
    try:
        raw_a = _read_bytes(a)
        raw_b = _read_bytes(b)
        if raw_a is None or raw_b is None:
            return {"ok": False, "reason": "could not read one or both images",
                    "hash_a": "", "hash_b": "", "hash_distance": -1,
                    "diff_score": -1.0, "mean_abs_diff": -1.0,
                    "identical": False, "regions": []}
        from . import native
        hash_a = native.phash(raw_a)
        hash_b = native.phash(raw_b)
        distance = native.hash_distance(hash_a, hash_b)
        diff = native.compare_native(raw_a, raw_b)
        changed = float(diff.get("changed_fraction") or 0.0)
        bbox = diff.get("changed_bbox_1000")
        regions = [bbox] if isinstance(bbox, dict) else []
        return {
            "ok": True,
            "hash_a": hash_a,
            "hash_b": hash_b,
            "hash_distance": distance,
            "diff_score": round(changed, 4),
            "mean_abs_diff": diff.get("mean_abs_diff"),
            "identical": bool(diff.get("identical")),
            "regions": regions,
            "byte_hash_a": _sha256(raw_a),
            "byte_hash_b": _sha256(raw_b),
            "width": diff.get("width"),
            "height": diff.get("height"),
            "method": "dhash+pixel-diff",
        }
    except Exception:  # noqa: BLE001 — sensor failure is a result, not a crash
        _log.debug("critique_bridge compare_images failed", exc_info=True)
        return {"ok": False, "reason": "comparison failed",
                "hash_a": "", "hash_b": "", "hash_distance": -1,
                "diff_score": -1.0, "mean_abs_diff": -1.0,
                "identical": False, "regions": []}


def _checklist_for(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Checklist items derived from one :func:`compare_images` result."""
    items: list[dict[str, Any]] = []
    if not comparison.get("ok"):
        return [{"check": "images comparable",
                 "status": "fail",
                 "evidence": comparison.get("reason")
                 or "comparison failed"}]
    identical = bool(comparison.get("identical"))
    diff = float(comparison.get("diff_score") or 0.0)
    distance = int(comparison.get("hash_distance") or 0)
    if identical:
        items.append({"check": "pixel-identical",
                      "status": "pass",
                      "evidence": "mean_abs_diff 0.0 — no pixel changed"})
    elif diff <= NEAR_DUPLICATE_MAX_FRACTION:
        items.append({"check": "pixel-identical",
                      "status": "fail",
                      "evidence": f"{diff:.4f} of pixels changed "
                                  "(near-duplicate)"})
    else:
        items.append({"check": "pixel-identical",
                      "status": "fail",
                      "evidence": f"{diff:.4f} of pixels changed"})
    if distance <= 5:
        items.append({"check": "perceptual near-duplicate",
                      "status": "pass",
                      "evidence": f"perceptual hash distance {distance} "
                                  "(0-5 ≈ near-duplicate)"})
    elif distance <= 12:
        items.append({"check": "perceptual near-duplicate",
                      "status": "fail",
                      "evidence": f"perceptual hash distance {distance} "
                                  "(6-12 ≈ similar, not duplicate)"})
    else:
        items.append({"check": "perceptual near-duplicate",
                      "status": "fail",
                      "evidence": f"perceptual hash distance {distance} "
                                  "(>12 ≈ different image)"})
    if comparison.get("regions"):
        items.append({"check": "changed regions located",
                      "status": "info",
                      "evidence": f"{len(comparison['regions'])} changed "
                                  f"region(s): {comparison['regions']}"})
    else:
        items.append({"check": "changed regions located",
                      "status": "info",
                      "evidence": "no changed region — images identical"})
    if diff > SIMILAR_MAX_FRACTION:
        items.append({"check": "visual drift within tolerance",
                      "status": "fail",
                      "evidence": f"changed fraction {diff:.4f} exceeds "
                                  f"{SIMILAR_MAX_FRACTION} tolerance"})
    else:
        items.append({"check": "visual drift within tolerance",
                      "status": "pass",
                      "evidence": f"changed fraction {diff:.4f} within "
                                  f"{SIMILAR_MAX_FRACTION} tolerance"})
    return items


def critique_from_vision(
    a: bytes | str | Path | dict[str, Any] | None = None,
    b: bytes | str | Path | None = None,
    *,
    comparison: dict[str, Any] | None = None,
    floor: float = 0.6,
) -> dict[str, Any]:
    """Critique-engine checklist from vision output.

    Either pass two images (``a``, ``b``) or a ready-made
    :func:`compare_images` result as ``comparison``. Returns
    ``{"checklist", "score", "verdict", "comparison"}`` where
    ``verdict`` follows the critique engine's floor rule: every
    fail item costs, ``score >= floor`` → ``ship``,
    ``score >= floor - 0.15`` → ``revise``, else ``rewrite``.
    The ``score`` is 1.0 minus the fail share — a deterministic
    function of the sensor reading, never a model judgement.
    Never raises.
    """
    try:
        comp = comparison
        if comp is None:
            if a is None or b is None:
                return {"checklist": [{"check": "images comparable",
                                       "status": "fail",
                                       "evidence": "need two images or a "
                                                   "comparison result"}],
                        "score": 0.0, "verdict": "rewrite",
                        "comparison": {"ok": False,
                                       "reason": "missing input"}}
            comp = compare_images(a, b)
        checklist = _checklist_for(comp)
        judged = [i for i in checklist if i["status"] in ("pass", "fail")]
        fails = sum(1 for i in judged if i["status"] == "fail")
        score = round(1.0 - (fails / len(judged)), 4) if judged else 0.0
        if score >= floor:
            verdict = "ship"
        elif score >= floor - 0.15:
            verdict = "revise"
        else:
            verdict = "rewrite"
        return {"checklist": checklist, "score": score,
                "verdict": verdict, "floor": floor, "comparison": comp}
    except Exception:  # noqa: BLE001
        _log.debug("critique_from_vision failed", exc_info=True)
        return {"checklist": [{"check": "images comparable",
                               "status": "fail",
                               "evidence": "critique failed"}],
                "score": 0.0, "verdict": "rewrite",
                "comparison": {"ok": False, "reason": "critique failed"}}


def visual_dimension(comparison: dict[str, Any] | None = None, *,
                     a: bytes | str | Path | None = None,
                     b: bytes | str | Path | None = None) -> Any:
    """The vision reading as a critique-engine dimension.

    Returns a plain dict ``{dimension, score, evidence}`` — no import of
    the L5 critique engine (layering). Callers that own a
    ``DimensionScore`` can convert this dict with one line.
    """
    comp = comparison
    if comp is None and a is not None and b is not None:
        comp = compare_images(a, b)
    comp = comp or {}
    diff = float(comp.get("diff_score") or 0.0)
    if diff < 0:
        diff = 1.0
    evidence: list[str] = []
    if not comp.get("ok"):
        evidence.append(comp.get("reason") or "comparison failed")
        return {"dimension": "visual", "score": 0.0, "evidence": evidence}
    evidence.append(f"changed fraction {diff:.4f}, "
                    f"hash distance {comp.get('hash_distance')}")
    if comp.get("regions"):
        evidence.append(f"changed region(s): {comp['regions']}")
    return {"dimension": "visual",
            "score": round(max(0.0, 1.0 - diff), 4),
            "evidence": evidence}


def describe_screen(screenshot_hash: str,
                    elements: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Screen understanding for the browser: structure supplied elements.

    ``screenshot_hash`` identifies the screenshot (a perceptual or byte
    hash — this function never sees pixels, so it cannot invent what is
    on screen). ``elements`` is the caller-supplied element list, e.g.
    from accessibility / DOM extraction — each item needs ``kind``
    (button/input/text/…) and ``label``/``text``. The return echoes
    back ONLY those elements, normalized (kind, label, bbox when the
    caller supplied one, state when supplied). Elements with an
    unknown kind or no label/text are dropped and counted in
    ``dropped`` — never guessed into shape. An empty input yields an
    empty element list with ``note`` stating nothing was supplied.
    Never hallucinates elements. Never raises.
    """
    try:
        supplied = elements or []
        normalized: list[dict[str, Any]] = []
        dropped = 0
        for el in supplied:
            if not isinstance(el, dict):
                dropped += 1
                continue
            kind = str(el.get("kind") or "").strip().lower()
            label = str(el.get("label") or el.get("text") or "").strip()
            if kind not in SCREEN_ELEMENT_KINDS or not label:
                dropped += 1
                continue
            item: dict[str, Any] = {"kind": kind, "label": label[:500]}
            bbox = el.get("bbox") or el.get("rect")
            if isinstance(bbox, dict) and all(
                    k in bbox for k in ("x", "y", "w", "h")):
                try:
                    item["bbox"] = {k: float(bbox[k])
                                    for k in ("x", "y", "w", "h")}
                except (TypeError, ValueError):
                    pass
            for key in ("state", "role", "name", "href"):
                if el.get(key) is not None:
                    item[key] = str(el.get(key))[:300]
            normalized.append(item)
        counts: dict[str, int] = {}
        for item in normalized:
            counts[item["kind"]] = counts.get(item["kind"], 0) + 1
        # Deterministic order: buttons, inputs, text first (interaction
        # order), then the rest — same input always yields same output.
        rank = {"button": 0, "input": 1, "text": 2}
        normalized.sort(key=lambda e: (rank.get(e["kind"], 3), e["label"]))
        return {
            "screenshot_hash": str(screenshot_hash or ""),
            "elements": normalized,
            "counts": counts,
            "supplied": len(supplied),
            "dropped": dropped,
            "source": "supplied-elements-only",
            "note": ("no elements supplied — none inferred; run DOM / "
                     "accessibility extraction and pass the elements in."
                     if not normalized else
                     "elements below are exactly the supplied ones, "
                     "normalized — nothing was hallucinated."),
        }
    except Exception:  # noqa: BLE001
        _log.debug("describe_screen failed", exc_info=True)
        return {"screenshot_hash": str(screenshot_hash or ""),
                "elements": [], "counts": {}, "supplied": 0,
                "dropped": 0, "source": "supplied-elements-only",
                "note": "screen description failed — no elements reported."}
