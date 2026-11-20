"""Landmarks to blendshapes — the geometric mapping.

MediaPipe FaceMesh gives 468 landmarks (478 with irises).
This module turns them into the rig's 52 ARKit blendshape
weights plus the head bone rotation — from landmark GEOMETRY,
not from MediaPipe's own blendshape output. MediaPipe's
blendshapes are the cross-check (the tests compare the two on
25 real faces), never the source: the mapping must work from
landmarks alone, and every weight's formula is stated here and
pinned in tests.

LEFT / RIGHT — subject-relative, matching the rig's ARKit
names. The subject faces the camera, so the subject's left is
on the IMAGE right. Verified empirically on asymmetric smiles
in the 25-face corpus (men-8, women-1, u-man2): the corner
that lifts is the subject's corner MediaPipe names — landmark
291 is the subject's LEFT mouth corner, 61 the RIGHT; 263 is
the subject's LEFT eye, 33 the RIGHT. The annotated pictures
are in ~/workspace/facemesh-verify/ (annotated-eyes.png,
annotated-mouth2.png).

INDEX MAP (verified by eye on real portraits):
  right eye (subject's): outer 33, inner 133, upper 159/158,
      lower 145/153, iris 468
  left eye (subject's): outer 263, inner 362, upper 386/385,
      lower 374/380, iris 473
  mouth: right corner 61, left corner 291, upper inner 13,
      lower inner 14, upper outer 0, lower outer 17,
      upper lip right 81 / left 308, lower lip right 84 / left 314
      (308 is the inner upper lip's left side — 178 sits on the
      inner LOWER lip and reads low when the mouth opens; the
      first version used 178 and the left upper lip read 0 on
      every smile. The FACEMESH_LIPS loop settles it: after 14
      it runs 87, 178, 88, 95 back toward 78, i.e. the lower
      lip's right side, while the upper lip's left side runs
      312, 311, 310, 415, 308.)
  nose: tip 1, bridge 6, wing right 98 / left 327,
      nostril right 123 / left 220
  brows: right inner 55 / mid 52 / outer 46,
      left inner 285 / mid 282 / outer 276
  chin 152, jaw right 234 / left 454, cheek right 205 / left 425,
      dimple right 330 / left 410

SCALE — D is the eye-outer distance dist(33, 263): every
vertical or lateral measure is divided by D (or by the mouth
width where the mouth itself is the ruler), so the weights are
scale-invariant. All frames are derotated by the eye-line
angle first — head roll corrupts every vertical measure
(the corpus has rolls to 26 degrees).

REST — several weights need a per-face rest baseline (an eye's
open EAR, a brow's rest gap, the mouth's rest width). A video
estimates it from its own frames (``estimate_rest``: max EAR,
min brow gap, max mouth width, ...). A single image uses the
population defaults below, taken from the 25-face corpus
percentiles and named here. Three weights are REST-GATED and
read 0 on a single image — cheekPuff, mouthDimple, noseSneer:
their population variance (face shape, nose shape) is larger
than the signal, so a population baseline would call wide
faces puffed and high-nosed faces sneering. A weight that has
no rest signal at all reads 0 rather than a guess.

KNOWN CONFOUNDS (measured on the 25-face corpus, named
rather than hidden):
- A wide smile parts the lips without dropping the jaw, so
  jawOpen (lip-opening ratio) reads hot on big smiles
  (u-woman1: 0.52 against MediaPipe's 0.22). The video rest
  does not fix this — it is a 2D limit, and the direction
  (open jaw > closed jaw) holds.
- The brow gap varies by face shape (0.067-0.241 in the
  corpus), so single-image brow weights use a dead band and
  can still false-positive on high/low-brow faces; the video
  rest is the accurate path.
- The upper lip corner lifts when the jaw drops, so
  mouthUpperUp reads hot on wide-open mouths; the video rest
  separates the two.
- The lid-narrow signal splits into blink vs squint by the
  cheek: the cheek point (205/425) moves DOWN-OUT as the
  cheek engages, so its distance to the lower lid GROWS —
  the first version had the sign backwards and squint read 0
  everywhere. blink + squint = the lid narrowing, always.

TONGUE — tongueOut is 0.0 always. There is no tongue landmark
in the 468; a tongue is visible only when out, and no landmark
tracks it. Reporting 0 with the reason named beats faking a
signal from lip geometry.

HEAD — the head bone comes from MediaPipe's facial
transformation matrix (a landmark-geometry PnP solve): its
3x3 rotation becomes Euler degrees (z, x, y), the rig's BVH
channel order, the same convention as the facemcap mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..rig.rig import ARKIT_BLENDSHAPES
from .errors import FaceMeshError

__all__ = [
    "FaceRest",
    "POPULATION_REST",
    "derotate",
    "estimate_rest",
    "landmarks_to_blendshapes",
    "landmarks_to_head_bones",
    "matrix_to_euler_zxy",
]

# -- landmark indices (subject-relative; see module docstring) --
EYE_R_OUTER = 33
EYE_R_INNER = 133
EYE_R_UP = 159
EYE_R_UP2 = 158
EYE_R_LOW = 145
EYE_R_LOW2 = 153
EYE_R_IRIS = 468
EYE_L_OUTER = 263
EYE_L_INNER = 362
EYE_L_UP = 386
EYE_L_UP2 = 385
EYE_L_LOW = 374
EYE_L_LOW2 = 380
EYE_L_IRIS = 473
MOUTH_R = 61
MOUTH_L = 291
LIP_UP_IN = 13
LIP_LOW_IN = 14
LIP_UP_OUT = 0
LIP_LOW_OUT = 17
LIP_UP_R = 81
LIP_UP_L = 308
LIP_LOW_R = 84
LIP_LOW_L = 314
NOSE_TIP = 1
NOSE_BRIDGE = 6
NOSE_WING_R = 98
NOSE_WING_L = 327
BROW_R_IN = 55
BROW_R_MID = 52
BROW_R_OUT = 46
BROW_L_IN = 285
BROW_L_MID = 282
BROW_L_OUT = 276
CHIN = 152
JAW_R = 234
JAW_L = 454
CHEEK_R = 205
CHEEK_L = 425
DIMPLE_R = 330
DIMPLE_L = 410

#: Landmarks the mapping reads (468 base; 478 with irises).
NEED_LANDMARKS = 468


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class FaceRest:
    """A face's rest baseline — the values each weight is
    measured against. Population defaults are the 25-face
    corpus medians/percentiles (see module docstring); a video
    replaces them with its own frames' extremes, and sets
    ``has_rest`` — the rest-gated weights (cheekPuff,
    mouthDimple, noseSneer) read 0 until a video supplies one."""

    has_rest: bool = False
    ear_open_L: float = 0.45
    ear_open_R: float = 0.45
    brow_in_L: float = 0.20
    brow_in_R: float = 0.20
    brow_out_L: float = 0.17
    brow_out_R: float = 0.17
    cheek_eye_L: float = 0.45
    cheek_eye_R: float = 0.45
    cheek_width: float = 0.95
    mouth_width: float = 0.62
    lip_up_h: float = 0.06
    lip_low_h: float = 0.10
    dimple_L: float = 0.64
    dimple_R: float = 0.50
    wing_up_L: float = -0.03
    wing_up_R: float = -0.03
    nose_chin_z: float = -0.107


#: The population rest baseline (corpus medians/percentiles).
POPULATION_REST = FaceRest()


def derotate(landmarks: list[tuple[float, float, float]]
             ) -> list[tuple[float, float, float]]:
    """Rotate the landmarks so the eye line is horizontal —
    head roll corrupts every vertical measure. The rotation is
    about the eye midpoint, in the image plane."""
    if len(landmarks) < NEED_LANDMARKS:
        raise FaceMeshError(
            f"need at least {NEED_LANDMARKS} landmarks, "
            f"got {len(landmarks)}",
            kind="bad_landmarks")
    cx = (landmarks[EYE_R_OUTER][0] + landmarks[EYE_L_OUTER][0]) / 2
    cy = (landmarks[EYE_R_OUTER][1] + landmarks[EYE_L_OUTER][1]) / 2
    ang = math.atan2(
        landmarks[EYE_L_OUTER][1] - landmarks[EYE_R_OUTER][1],
        landmarks[EYE_L_OUTER][0] - landmarks[EYE_R_OUTER][0])
    ca, sa = math.cos(-ang), math.sin(-ang)
    out = []
    for x, y, z in landmarks:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * ca - dy * sa,
                    cy + dx * sa + dy * ca, z))
    return out


def _ear(lm: list[tuple[float, float, float]], *,
         outer: int, inner: int, up: int, up2: int,
         low: int, low2: int) -> float:
    """Eye aspect ratio: lid height over eye width."""
    def p(i: int) -> tuple[float, float]:
        return (lm[i][0], lm[i][1])

    return (_dist(p(up), p(low)) + _dist(p(up2), p(low2))) / (
        2 * _dist(p(outer), p(inner)))


def _signals(lm: list[tuple[float, float, float]]
             ) -> dict[str, float]:
    """The raw geometric signals, before rest normalisation.
    ``lm`` must already be derotated."""
    def p(i: int) -> tuple[float, float]:
        return (lm[i][0], lm[i][1])

    d = _dist(p(EYE_R_OUTER), p(EYE_L_OUTER))
    mw = _dist(p(MOUTH_R), p(MOUTH_L))
    cy = (p(LIP_UP_IN)[1] + p(LIP_LOW_IN)[1]) / 2
    mid_x = (p(EYE_R_OUTER)[0] + p(EYE_L_OUTER)[0]) / 2
    mouth_cx = (p(MOUTH_R)[0] + p(MOUTH_L)[0]) / 2
    sig: dict[str, float] = {"D": d, "mw": mw, "cy": cy,
                             "mid_x": mid_x, "mouth_cx": mouth_cx}
    sig["open_ratio"] = _dist(p(LIP_UP_IN), p(LIP_LOW_IN)) / mw
    sig["lift_L"] = (cy - p(MOUTH_L)[1]) / mw
    sig["lift_R"] = (cy - p(MOUTH_R)[1]) / mw
    sig["ear_L"] = _ear(lm, outer=EYE_L_OUTER, inner=EYE_L_INNER,
                        up=EYE_L_UP, up2=EYE_L_UP2,
                        low=EYE_L_LOW, low2=EYE_L_LOW2)
    sig["ear_R"] = _ear(lm, outer=EYE_R_OUTER, inner=EYE_R_INNER,
                        up=EYE_R_UP, up2=EYE_R_UP2,
                        low=EYE_R_LOW, low2=EYE_R_LOW2)
    sig["cheek_eye_L"] = _dist(p(CHEEK_L), p(EYE_L_LOW)) / d
    sig["cheek_eye_R"] = _dist(p(CHEEK_R), p(EYE_R_LOW)) / d
    sig["gap_in_L"] = (p(EYE_L_INNER)[1] - p(BROW_L_IN)[1]) / d
    sig["gap_in_R"] = (p(EYE_R_INNER)[1] - p(BROW_R_IN)[1]) / d
    sig["gap_out_L"] = (p(EYE_L_OUTER)[1] - p(BROW_L_OUT)[1]) / d
    sig["gap_out_R"] = (p(EYE_R_OUTER)[1] - p(BROW_R_OUT)[1]) / d
    sig["cheek_width"] = _dist(p(CHEEK_R), p(CHEEK_L)) / d
    sig["mouth_width_D"] = mw / d
    sig["lip_up_h"] = _dist(p(LIP_UP_OUT), p(LIP_UP_IN)) / d
    sig["lip_low_h"] = _dist(p(LIP_LOW_OUT), p(LIP_LOW_IN)) / d
    sig["dimple_L"] = _dist(p(DIMPLE_L), p(MOUTH_L)) / d
    sig["dimple_R"] = _dist(p(DIMPLE_R), p(MOUTH_R)) / d
    sig["wing_up_L"] = (p(NOSE_TIP)[1] - p(NOSE_WING_L)[1]) / d
    sig["wing_up_R"] = (p(NOSE_TIP)[1] - p(NOSE_WING_R)[1]) / d
    sig["up_lip_L"] = (cy - p(LIP_UP_L)[1]) / d
    sig["up_lip_R"] = (cy - p(LIP_UP_R)[1]) / d
    sig["low_lip_L"] = (p(LIP_LOW_L)[1] - cy) / d
    sig["low_lip_R"] = (p(LIP_LOW_R)[1] - cy) / d
    sig["chin_off"] = (p(CHIN)[0] - mid_x) / d
    sig["mouth_off"] = (mouth_cx - mid_x) / d
    sig["nose_chin_z"] = lm[NOSE_TIP][2] - lm[CHIN][2]
    # Iris offsets, as fractions of the eye's half-width/height.
    for side, outer, inner, up, low, iris in (
            ("L", EYE_L_OUTER, EYE_L_INNER, EYE_L_UP, EYE_L_LOW,
             EYE_L_IRIS),
            ("R", EYE_R_OUTER, EYE_R_INNER, EYE_R_UP, EYE_R_LOW,
             EYE_R_IRIS)):
        ecx = (p(outer)[0] + p(inner)[0]) / 2
        ecy = (p(up)[1] + p(low)[1]) / 2
        sig[f"iris_{side}_dx"] = (
            (p(iris)[0] - ecx) / (_dist(p(outer), p(inner)) / 2))
        sig[f"iris_{side}_dy"] = (
            (p(iris)[1] - ecy) / (_dist(p(up), p(low)) / 2))
    return sig


def estimate_rest(
        frames: list[list[tuple[float, float, float]]]) -> FaceRest:
    """A video's own rest baseline: the extremes over its
    frames — the widest-open eye is the open eye, the lowest
    brow gap is the resting brow, the widest mouth is the
    resting mouth, and so on. A face that never opens its eyes
    in the video gets a rest from the population defaults
    (the max over frames is still the best estimate)."""
    if not frames:
        raise FaceMeshError("estimate_rest needs at least one "
                            "frame of landmarks",
                            kind="bad_landmarks")
    sigs = [_signals(derotate(f)) for f in frames]
    rest = FaceRest(
        has_rest=True,
        ear_open_L=max(s["ear_L"] for s in sigs),
        ear_open_R=max(s["ear_R"] for s in sigs),
        brow_in_L=min(s["gap_in_L"] for s in sigs),
        brow_in_R=min(s["gap_in_R"] for s in sigs),
        brow_out_L=min(s["gap_out_L"] for s in sigs),
        brow_out_R=min(s["gap_out_R"] for s in sigs),
        cheek_eye_L=max(s["cheek_eye_L"] for s in sigs),
        cheek_eye_R=max(s["cheek_eye_R"] for s in sigs),
        cheek_width=min(s["cheek_width"] for s in sigs),
        mouth_width=max(s["mouth_width_D"] for s in sigs),
        lip_up_h=max(s["lip_up_h"] for s in sigs),
        lip_low_h=max(s["lip_low_h"] for s in sigs),
        dimple_L=max(s["dimple_L"] for s in sigs),
        dimple_R=max(s["dimple_R"] for s in sigs),
        wing_up_L=min(s["wing_up_L"] for s in sigs),
        wing_up_R=min(s["wing_up_R"] for s in sigs),
        nose_chin_z=sorted(s["nose_chin_z"] for s in sigs)[
            len(sigs) // 2],
    )
    # Guards: a degenerate video (one frame, or a face that
    # never rests) must not produce a rest that inverts a
    # weight — clamp each rest into its sane population band.
    rest.ear_open_L = max(0.25, min(0.60, rest.ear_open_L))
    rest.ear_open_R = max(0.25, min(0.60, rest.ear_open_R))
    rest.brow_in_L = max(0.05, min(0.25, rest.brow_in_L))
    rest.brow_in_R = max(0.05, min(0.25, rest.brow_in_R))
    rest.brow_out_L = max(0.05, min(0.25, rest.brow_out_L))
    rest.brow_out_R = max(0.05, min(0.25, rest.brow_out_R))
    rest.cheek_width = max(0.75, min(1.05, rest.cheek_width))
    rest.mouth_width = max(0.45, min(0.90, rest.mouth_width))
    return rest


def landmarks_to_blendshapes(
        landmarks: list[tuple[float, float, float]], *,
        rest: FaceRest | None = None) -> dict[str, float]:
    """468 (or 478) landmarks to the 52 ARKit blendshape
    weights, in 0..1. Every weight is a stated geometric ratio
    against the rest baseline; the scales are the corpus
    calibration (see module docstring)."""
    lm = derotate(landmarks)
    r = rest or POPULATION_REST
    s = _signals(lm)

    def p(i: int) -> tuple[float, float]:
        return (lm[i][0], lm[i][1])

    w: dict[str, float] = {}
    # Jaw + mouth.
    w["jawOpen"] = _clamp(s["open_ratio"] / 0.85)
    w["jawForward"] = _clamp(
        (s["nose_chin_z"] - r.nose_chin_z) / 0.08)
    w["jawLeft"] = _clamp(-s["chin_off"] / 0.06)
    w["jawRight"] = _clamp(s["chin_off"] / 0.06)
    w["mouthLeft"] = _clamp(-s["mouth_off"] / 0.05)
    w["mouthRight"] = _clamp(s["mouth_off"] / 0.05)
    w["mouthSmileLeft"] = _clamp(s["lift_L"] / 0.25)
    w["mouthSmileRight"] = _clamp(s["lift_R"] / 0.25)
    w["mouthFrownLeft"] = _clamp(-s["lift_L"] / 0.15)
    w["mouthFrownRight"] = _clamp(-s["lift_R"] / 0.15)
    w["mouthPucker"] = _clamp(
        (r.mouth_width - s["mouth_width_D"]) / 0.12)
    # A pucker bunches the lips and lifts the corners, so a
    # puckered mouth reads as smiling on the raw lift — the
    # smile is net of the pucker.
    w["mouthSmileLeft"] *= 1 - w["mouthPucker"]
    w["mouthSmileRight"] *= 1 - w["mouthPucker"]
    # Rest-gated: the dimple point's distance to its corner
    # varies more by face shape (0.40-0.90 in the corpus)
    # than any dimple moves it, so a single frame reads 0.
    w["mouthDimpleLeft"] = (
        _clamp((r.dimple_L - s["dimple_L"]) / 0.05)
        if r.has_rest else 0.0)
    w["mouthDimpleRight"] = (
        _clamp((r.dimple_R - s["dimple_R"]) / 0.05)
        if r.has_rest else 0.0)
    # Stretch — the corner pulled outward from the face
    # midline past its rest half-width, net of the smile:
    # a smile pulls the corners out too, and the stretch is
    # the lateral pull the smile does not explain.
    w["mouthStretchLeft"] = _clamp(
        ((p(MOUTH_L)[0] - s["mid_x"]) / s["D"]
         - r.mouth_width / 2) / 0.05) * (
        1 - w["mouthSmileLeft"])
    w["mouthStretchRight"] = _clamp(
        ((s["mid_x"] - p(MOUTH_R)[0]) / s["D"]
         - r.mouth_width / 2) / 0.05) * (
        1 - w["mouthSmileRight"])
    w["mouthFunnel"] = _clamp(s["open_ratio"] / 0.35) * (
        1 - max(w["mouthSmileLeft"], w["mouthSmileRight"]))
    closed = 1.0 if s["open_ratio"] < 0.03 else 0.0
    # Press — the lips compressed together. A pucker thins
    # the visible lip the same way, so a puckered mouth does
    # not read as pressed: the press is gated off as the
    # pucker rises.
    lip_compression = _clamp(
        (r.lip_up_h + r.lip_low_h - s["lip_up_h"]
         - s["lip_low_h"]) / (r.lip_up_h + r.lip_low_h))
    press_gate = 1 - _clamp(w["mouthPucker"] / 0.3)
    w["mouthPressLeft"] = closed * press_gate * lip_compression
    w["mouthPressRight"] = w["mouthPressLeft"]
    w["mouthRollUpper"] = closed * _clamp(
        (r.lip_up_h - s["lip_up_h"]) / r.lip_up_h)
    w["mouthRollLower"] = closed * _clamp(
        (r.lip_low_h - s["lip_low_h"]) / r.lip_low_h)
    # Shrug — one lip pushed toward the other past its rest
    # line (the upper inner lip rests just above the mouth
    # centre, the lower just below it).
    w["mouthShrugUpper"] = _clamp(
        (lm[LIP_UP_IN][1] - (s["cy"] - 0.02 * s["D"]))
        / s["D"] / 0.05)
    w["mouthShrugLower"] = _clamp(
        ((s["cy"] + 0.02 * s["D"]) - lm[LIP_LOW_IN][1])
        / s["D"] / 0.05)
    w["mouthUpperUpLeft"] = _clamp(s["up_lip_L"] / 0.20)
    w["mouthUpperUpRight"] = _clamp(s["up_lip_R"] / 0.20)
    w["mouthLowerDownLeft"] = _clamp(
        (s["low_lip_L"] - 0.15) / 0.20)
    w["mouthLowerDownRight"] = _clamp(
        (s["low_lip_R"] - 0.15) / 0.20)
    w["mouthClose"] = closed * (1 - w["jawOpen"])
    # Brows — dead-banded around the population rest: the
    # gap varies by face shape, so only a clear departure
    # reads on a single frame; a video's own rest is exact.
    w["browInnerUp"] = _clamp(
        ((s["gap_in_L"] + s["gap_in_R"]) / 2
         - (r.brow_in_L + r.brow_in_R) / 2) / 0.12)
    w["browOuterUpLeft"] = _clamp(
        (s["gap_out_L"] - r.brow_out_L) / 0.12)
    w["browOuterUpRight"] = _clamp(
        (s["gap_out_R"] - r.brow_out_R) / 0.12)
    w["browDownLeft"] = _clamp((0.09 - s["gap_in_L"]) / 0.08)
    w["browDownRight"] = _clamp((0.09 - s["gap_in_R"]) / 0.08)
    # Eyes — the lid-narrow signal splits into blink vs squint
    # by the context that narrowed it: the cheek engaged, or
    # the mouth smiling. The cheek point moves DOWN-OUT as the
    # cheek engages, so its distance to the lower lid GROWS —
    # measured, not assumed. The cheek signal itself is
    # rest-gated (face shape dominates it in a single frame);
    # the smile is the single-frame disambiguator. The split
    # conserves the lid signal: blink + squint = lid narrow.
    for side, word in (("L", "Left"), ("R", "Right")):
        ear = s[f"ear_{side}"]
        ear_open = getattr(r, f"ear_open_{side}")
        lid = _clamp(1 - ear / ear_open)
        cheek_eye = s[f"cheek_eye_{side}"]
        cheek_rest = getattr(r, f"cheek_eye_{side}")
        rise = (_clamp((cheek_eye - cheek_rest) / 0.10)
                if r.has_rest else 0.0)
        squint_factor = max(
            rise, w["mouthSmileLeft"], w["mouthSmileRight"])
        w[f"eyeBlink{word}"] = lid * (1 - squint_factor)
        w[f"eyeSquint{word}"] = lid * squint_factor
        w[f"cheekSquint{word}"] = rise
        w[f"eyeWide{word}"] = _clamp((ear - ear_open) / 0.15)
        dx = s[f"iris_{side}_dx"]
        dy = s[f"iris_{side}_dy"]
        if side == "L":
            # Subject's left eye: the nose is toward -x.
            w["eyeLookInLeft"] = _clamp(-dx / 0.5)
            w["eyeLookOutLeft"] = _clamp(dx / 0.5)
        else:
            w["eyeLookInRight"] = _clamp(dx / 0.5)
            w["eyeLookOutRight"] = _clamp(-dx / 0.5)
        w[f"eyeLookUp{word}"] = _clamp(-dy / 0.5)
        w[f"eyeLookDown{word}"] = _clamp(dy / 0.5)
    # Cheeks + nose. Puff and sneer are rest-gated like the
    # dimple: cheek width varies by face shape and by smile,
    # and the nose wing's height varies by nose shape.
    w["cheekPuff"] = (
        _clamp((s["cheek_width"] - r.cheek_width) / 0.08)
        if r.has_rest else 0.0)
    w["noseSneerLeft"] = (
        _clamp((s["wing_up_L"] - r.wing_up_L) / 0.06)
        if r.has_rest else 0.0)
    w["noseSneerRight"] = (
        _clamp((s["wing_up_R"] - r.wing_up_R) / 0.06)
        if r.has_rest else 0.0)
    # No tongue landmark exists — 0, named, never faked.
    w["tongueOut"] = 0.0
    missing = [name for name in ARKIT_BLENDSHAPES
               if name not in w]
    if missing:
        raise FaceMeshError(
            f"mapping left blendshapes unfilled: {missing}",
            kind="bad_landmarks")
    return {name: w[name] for name in ARKIT_BLENDSHAPES}


def matrix_to_euler_zxy(matrix: object
                        ) -> tuple[float, float, float]:
    """A 3x3 rotation (or a 4x4 transformation matrix) as Euler
    degrees ``(z, x, y)`` — intrinsic rotations applied in
    order Z, then X, then Y (``R = Rz(z) Rx(x) Ry(y)``), the
    rig's BVH channel order, the same convention as the
    facemcap mapping. The pole (|x| = 90 deg) gives Z the
    whole angle and Y reads 0 — named, not hidden."""
    rows = [[float(v) for v in row] for row in matrix]  # type: ignore[union-attr]
    if len(rows) < 3 or any(len(row) < 3 for row in rows):
        raise FaceMeshError(
            "head matrix needs at least 3x3 entries",
            kind="bad_landmarks")
    r00, r01, r02 = rows[0][:3]
    r10, r11, r12 = rows[1][:3]
    r20, r21, r22 = rows[2][:3]
    x_angle = math.asin(max(-1.0, min(1.0, r21)))
    cos_x = math.cos(x_angle)
    if abs(cos_x) < 1e-9:
        z_angle = math.atan2(r10, r00)
        y_angle = 0.0
    else:
        z_angle = math.atan2(-r01, r11)
        y_angle = math.atan2(-r20, r22)
    return (math.degrees(z_angle), math.degrees(x_angle),
            math.degrees(y_angle))


def landmarks_to_head_bones(matrix: object | None
                            ) -> dict[str, tuple[float, float, float]]:
    """The head bone rotation for one frame: the facial
    transformation matrix's rotation as (z, x, y) degrees on
    the rig's ``head`` bone. No matrix (IMAGE mode with no
    pose output) means the head holds at rest — an empty
    bones dict, never a guessed rotation."""
    if matrix is None:
        return {}
    return {"head": matrix_to_euler_zxy(matrix)}
