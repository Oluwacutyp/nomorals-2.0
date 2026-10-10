"""Parametric pose rig — the DIRECTION layer.

An action description ("raise two fingers") becomes a keypoint trajectory:
COCO-18 body + 21-pt hands (MediaPipe order), DWPose-compatible layout,
normalized [0,1] coordinates. The trajectory renders as a skeleton pose
video — the exact input MimicMotion / AnimateAnyone consume — and also
drives the CPU mesh-warp animator.

No model needed here: the ACTION is authored parametrically, the neural
model (when present) only renders it photoreal. Deterministic, editable,
offline.

The ACTIONS below are SEED EXAMPLES + instant offline presets — never the
ceiling. Any describable human action goes through
motion_score.generate_track(): preset fast path when matched, otherwise
the brain choreographs a motion score (JSON phases) compiled onto this
rig. The only limit is physical plausibility, enforced by the compiler.

Rest-pose assumption (honest): without a pose detector the rig places a
frontal rest pose scaled to the frame, person centered. When DWPose is
available the rest pose is extracted from the image and actions retarget
onto it (see retarget()).
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

# ── keypoint layout ──────────────────────────────────────────────────
# Body: COCO-18. Hands: MediaPipe 21-pt order each.
BODY_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]
N_BODY = len(BODY_NAMES)
HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
N_HAND = len(HAND_NAMES)

# total keypoints: body + right hand + left hand
N_KP = N_BODY + 2 * N_HAND
R_HAND = slice(N_BODY, N_BODY + N_HAND)
L_HAND = slice(N_BODY + N_HAND, N_KP)

BODY_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (1, 11), (8, 9), (9, 10), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17),
]
HAND_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def rest_pose() -> np.ndarray:
    """Default frontal rest pose, normalized [0,1]. Person centered."""
    kp = np.zeros((N_KP, 2), dtype=np.float64)
    body = {
        "nose": (0.50, 0.16), "neck": (0.50, 0.23),
        "r_shoulder": (0.39, 0.25), "r_elbow": (0.35, 0.41),
        "r_wrist": (0.33, 0.57),
        "l_shoulder": (0.61, 0.25), "l_elbow": (0.65, 0.41),
        "l_wrist": (0.67, 0.57),
        "r_hip": (0.43, 0.60), "r_knee": (0.42, 0.79),
        "r_ankle": (0.42, 0.96),
        "l_hip": (0.57, 0.60), "l_knee": (0.58, 0.79),
        "l_ankle": (0.58, 0.96),
        "r_eye": (0.47, 0.145), "l_eye": (0.53, 0.145),
        "r_ear": (0.44, 0.16), "l_ear": (0.56, 0.16),
    }
    for i, name in enumerate(BODY_NAMES):
        kp[i] = body[name]
    # hands: fists resting at the wrists
    for sl, wrist_idx in ((R_HAND, 4), (L_HAND, 7)):  # r_wrist / l_wrist
        kp[sl] = _fist(kp[wrist_idx], size=0.035)
    return kp


# ── parametric fingers ─────────────────────────────────────────────
# Each finger is a serial chain: MCP (flexion + abduction) -> PIP -> DIP.
# Flexion 0.0 = fully extended, 1.0 = fully curled. DIP follows PIP via
# tendon coupling (~0.65) unless given explicitly. Angles in radians.
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_KP = {  # MediaPipe-order keypoint indices per finger
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_FINGER_BASE_ANG = {  # fan-out from wrist, 0 = up (-y)
    "thumb": 0.85, "index": 0.30, "middle": 0.02,
    "ring": -0.26, "pinky": -0.52,
}
_FINGER_SEG = {  # (proximal, middle, distal) segment lens, fraction of hand size
    "thumb": (0.30, 0.26, 0.22),
    "index": (0.32, 0.24, 0.18),
    "middle": (0.34, 0.26, 0.20),
    "ring": (0.32, 0.24, 0.18),
    "pinky": (0.26, 0.20, 0.16),
}
_FLEX_MAX = {  # max radians per joint row (mcp, pip, dip)
    "thumb": (1.1, 1.3, 0.0),
    "index": (1.6, 1.8, 1.2),
    "middle": (1.6, 1.8, 1.2),
    "ring": (1.6, 1.8, 1.2),
    "pinky": (1.6, 1.8, 1.2),
}
_KNUCKLE_DIST = 0.34   # wrist -> knuckle, fraction of hand size
_DIP_COUPLING = 0.65   # DIP flexion follows PIP when not specified

# Finger state: finger -> (mcp_flex, pip_flex, dip_flex, abduct), all 0..1
# (abduct -1..1 spreads the finger away from the middle finger axis)


def finger_state_from_pose(pose: str) -> dict[str, tuple[float, float, float, float]]:
    """Legacy unit pose -> per-finger flexion state."""
    curled = {f: (1.0, 1.0, _DIP_COUPLING, 0.0) for f in FINGERS}
    if pose == "fist":
        return curled
    if pose == "open":
        return {f: (0.0, 0.0, 0.0, 0.0) for f in FINGERS}
    if pose == "peace":
        s = dict(curled)
        s["index"] = (0.0, 0.0, 0.0, 0.25)
        s["middle"] = (0.0, 0.0, 0.0, -0.25)
        return s
    if pose == "point":
        s = dict(curled)
        s["index"] = (0.05, 0.05, 0.03, 0.0)
        return s
    if pose == "thumbs":
        s = dict(curled)
        s["thumb"] = (0.15, 0.10, 0.0, 0.45)
        return s
    return curled


def expand_flex(flex: float | list | tuple) -> tuple[float, float, float]:
    """Uniform flex or explicit (mcp, pip, dip) triple -> triple.

    Uniform flex applies DIP tendon coupling automatically.
    """
    if isinstance(flex, (list, tuple)):
        vals = [max(0.0, min(1.0, float(v))) for v in list(flex)[:3]]
        while len(vals) < 3:
            vals.append(vals[-1] if vals else 0.0)
        return (vals[0], vals[1], vals[2])
    f = max(0.0, min(1.0, float(flex)))
    return (f, f, f * _DIP_COUPLING)


def render_fingers(wrist: np.ndarray, size: float,
                   state: dict[str, tuple[float, float, float, float]],
                   spread: float = 1.0) -> np.ndarray:
    """Parametric finger forward kinematics -> (21, 2) MediaPipe-order keypoints.

    state: finger -> (mcp_flex, pip_flex, dip_flex, abduct). Deterministic —
    no randomness, so frames don't jitter during blends.
    """
    pts = np.zeros((N_HAND, 2))
    pts[0] = wrist
    for f in FINGERS:
        mcp_f, pip_f, dip_f, abduct = state.get(f, (1.0, 1.0, 0.65, 0.0))
        maxes = _FLEX_MAX[f]
        segs = _FINGER_SEG[f]
        base = -math.pi / 2 + (_FINGER_BASE_ANG[f] + abduct * 0.35) * spread
        kx = wrist[0] + math.cos(base) * size * _KNUCKLE_DIST
        ky = wrist[1] + math.sin(base) * size * _KNUCKLE_DIST
        j0, j1, j2, j3 = _FINGER_KP[f]
        pts[j0] = (kx, ky)
        curl = 0.0
        p = np.array([kx, ky])
        for s, (fl, mx) in enumerate(zip((mcp_f, pip_f, dip_f), maxes)):
            curl += fl * mx
            ang = base + curl
            p = p + np.array([math.cos(ang), math.sin(ang)]) * size * segs[s]
            pts[(j1, j2, j3)[s]] = p
    return pts


def _fist(wrist: np.ndarray, size: float = 0.035) -> np.ndarray:
    """Curled fist keypoints around a wrist position."""
    return render_fingers(wrist, size, finger_state_from_pose("fist"))


def _open_hand(wrist: np.ndarray, size: float = 0.045,
               spread: float = 1.0) -> np.ndarray:
    """Open palm, fingers extended upward from wrist."""
    return render_fingers(wrist, size, finger_state_from_pose("open"),
                          spread=spread)


def _peace_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    """Two fingers up (index + middle extended, rest curled)."""
    return render_fingers(wrist, size, finger_state_from_pose("peace"))


def _point_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    return render_fingers(wrist, size, finger_state_from_pose("point"))


def _thumbs_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    return render_fingers(wrist, size, finger_state_from_pose("thumbs"))


# ── finger IK ────────────────────────────────────────────────────────
# Two-bone analytic IK per finger: bone A = proximal segment, bone B =
# middle + distal as one link (the DIP joint then follows PIP via the
# existing tendon coupling, refined by two chord-correction iterations).
# Forward kinematics (render_fingers) is the ground truth; the solver is
# checked against it. Unreachable targets clamp to the reach circle —
# the finger points at the target but never teleports past its anatomy.

_REACH_EPS = 1e-6


@dataclass
class IKSolution:
    """Result of solving a finger onto a target."""
    flex: tuple[float, float, float]   # (mcp, pip, dip) flexion 0..1
    abduct: float                      # kept from the request
    reached: bool                      # False when the target was clamped
    tip: tuple[float, float]           # actual fingertip after solving
    residual: float                    # |tip - clamped target| (normalized)


def finger_reach(finger: str, size: float = 0.045) -> tuple[float, float]:
    """(min, max) reach of a fingertip from the knuckle, normalized units."""
    s1, s2, s3 = _FINGER_SEG[finger]
    return (abs(s1 - (s2 + s3)) * size + _REACH_EPS,
            (s1 + s2 + s3) * size - _REACH_EPS)


def _knuckle(wrist: np.ndarray, finger: str, size: float,
             spread: float, abduct: float) -> tuple[np.ndarray, float]:
    """Knuckle (MCP) position + base direction angle, matching render_fingers."""
    base = -math.pi / 2 + (_FINGER_BASE_ANG[finger] + abduct * 0.35) * spread
    k = np.array([wrist[0] + math.cos(base) * size * _KNUCKLE_DIST,
                  wrist[1] + math.sin(base) * size * _KNUCKLE_DIST])
    return k, base


def _tip_fk(wrist: np.ndarray, finger: str, size: float,
            flex: tuple[float, float, float],
            spread: float, abduct: float) -> np.ndarray:
    """Single-finger forward kinematics — ground truth for the solver."""
    mcp_f, pip_f, dip_f = flex
    maxes = _FLEX_MAX[finger]
    segs = _FINGER_SEG[finger]
    k, base = _knuckle(wrist, finger, size, spread, abduct)
    curl = 0.0
    p = k.copy()
    for fl, mx, s in zip((mcp_f, pip_f, dip_f), maxes, segs):
        curl += fl * mx
        ang = base + curl
        p = p + np.array([math.cos(ang), math.sin(ang)]) * size * s
    return p


def finger_ik(wrist: np.ndarray, finger: str,
              target: tuple[float, float] | np.ndarray,
              size: float = 0.045, spread: float = 1.0,
              abduct: float = 0.0) -> IKSolution:
    """Two-bone IK: bend ``finger`` so its tip reaches ``target``.

    All coordinates normalized [0,1] (same space as render_fingers).
    Tendon coupling: DIP flexion = PIP flexion * _DIP_COUPLING, refined
    with two chord-correction iterations so the solved tip matches the
    target to <1% of hand size. Targets outside the reach annulus are
    clamped to it (``reached=False``) — the finger stretches toward the
    target as far as anatomy allows.
    """
    if finger not in FINGERS:
        raise ValueError(f"unknown finger {finger!r}; pick from {FINGERS}")
    wrist = np.asarray(wrist, dtype=float)
    target = np.asarray(target, dtype=float)
    k, base = _knuckle(wrist, finger, size, spread, abduct)
    s1, s2, s3 = (s * size for s in _FINGER_SEG[finger])
    mcp_max, pip_max, dip_max = _FLEX_MAX[finger]

    d = target - k
    dist = float(np.linalg.norm(d))
    rmin, rmax = finger_reach(finger, size)
    reached = rmin <= dist <= rmax
    dc = min(max(dist, rmin), rmax)
    u = d / (dist + 1e-12)
    aim = k + u * dc  # clamped target

    # Analytic two-bone seed: bone A = proximal, bone B = middle+distal
    # chord. Fast, gets us into the right basin.
    bone_a = s1
    bone_b = s2 + s3
    cos_a = (bone_a ** 2 + dc ** 2 - bone_b ** 2) / (2 * bone_a * dc + 1e-12)
    ang_a = math.acos(max(-1.0, min(1.0, cos_a)))
    cands: list[tuple[np.ndarray, float]] = []
    for sgn in (1.0, -1.0):
        rot = np.array([[math.cos(sgn * ang_a), -math.sin(sgn * ang_a)],
                        [math.sin(sgn * ang_a), math.cos(sgn * ang_a)]])
        elbow = k + rot @ (u * bone_a)
        v1 = (elbow - k) / (bone_a + 1e-12)
        v2 = (aim - elbow)
        v2 = v2 / (float(np.linalg.norm(v2)) + 1e-12)
        pip_ang = math.atan2(v1[0] * v2[1] - v1[1] * v2[0],
                             float(np.dot(v1, v2)))
        cands.append((elbow, pip_ang))
    pos = [c for c in cands if c[1] >= 0.0]
    elbow, pip_ang = (max(pos, key=lambda c: c[1]) if pos
                      else max(cands, key=lambda c: c[1]))
    v0 = np.array([math.cos(base), math.sin(base)])
    v1 = (elbow - k) / (bone_a + 1e-12)
    mcp_ang = math.atan2(v0[0] * v1[1] - v0[1] * v1[0],
                         float(np.dot(v0, v1)))
    q = np.array([max(0.0, min(1.0, mcp_ang / (mcp_max + 1e-12))),
                  max(0.0, min(1.0, pip_ang / (pip_max + 1e-12)))])

    # Damped-least-squares polish on the TRUE forward kinematics
    # (MCP + PIP free, DIP follows tendon coupling). Analytic Jacobian.
    def _fk_tip(qv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t0 = base + qv[0] * mcp_max
        t1 = t0 + qv[1] * pip_max
        t2 = t1 + min(1.0, qv[1] * _DIP_COUPLING) * dip_max
        tip = (k + s1 * np.array([math.cos(t0), math.sin(t0)])
               + s2 * np.array([math.cos(t1), math.sin(t1)])
               + s3 * np.array([math.cos(t2), math.sin(t2)]))
        d0 = np.array([-math.sin(t0), math.cos(t0)])
        d1 = np.array([-math.sin(t1), math.cos(t1)])
        d2 = np.array([-math.sin(t2), math.cos(t2)])
        j0 = (s1 * mcp_max) * d0 + (s2 * mcp_max) * d1 + (s3 * mcp_max) * d2
        j1 = (s2 * pip_max) * d1 + (s3 * (pip_max + _DIP_COUPLING * dip_max)) * d2
        return tip, np.column_stack([j0, j1])

    lam2 = (0.25 * size) ** 2
    for _ in range(24):
        tip, jac = _fk_tip(q)
        err = aim - tip
        if float(np.linalg.norm(err)) < 1e-6:
            break
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(2), err)
        q = np.clip(q + dq, 0.0, 1.0)

    mcp_flex, pip_flex = float(q[0]), float(q[1])
    dip_flex = min(1.0, pip_flex * _DIP_COUPLING)
    flex = (mcp_flex, pip_flex, dip_flex)
    tip = _tip_fk(wrist, finger, size, flex, spread, abduct)
    residual = float(np.linalg.norm(tip - (k + u * dc)))
    if residual > 0.02 * size:
        reached = False
    return IKSolution(flex=flex, abduct=abduct, reached=bool(reached),
                      tip=(float(tip[0]), float(tip[1])), residual=residual)


def _ease(t: float) -> float:
    """Smoothstep easing."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _phase(t: float, a: float, b: float) -> float:
    """Eased progress of t within [a, b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return _ease((t - a) / (b - a))


@dataclass
class PoseTrack:
    """A full keypoint trajectory: frames × keypoints × 2."""
    frames: np.ndarray          # (F, N_KP, 2) normalized
    fps: float = 10.0
    action: str = ""
    width: int = 512
    height: int = 512
    root: np.ndarray | None = None  # (F, 3): root dx, dy, scale per frame.
    # Root = pelvis translation channel. Foot-planted frames have ankles
    # pinned in world space (see motion_score.compile_score); the channel
    # lets consumers distinguish travel from in-place motion.

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


# ── action programs: t in [0,1] → keypoints ─────────────────────────
ActionFn = Callable[[np.ndarray, float], np.ndarray]


def _act_raise_two_fingers(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    p1 = _phase(t, 0.0, 0.45)    # arm rises
    p2 = _phase(t, 0.30, 0.70)   # fingers extend
    hold = _phase(t, 0.70, 1.0)
    # right arm arcs up-out
    kp[3] = base[3] + np.array([0.03, -0.10]) * p1          # r_elbow
    wrist = base[4] + np.array([0.10, -0.30]) * p1          # r_wrist
    wrist = wrist + np.array([0.0, -0.012 * math.sin(hold * math.pi * 2)]) * hold
    kp[4] = wrist
    fist = _fist(wrist)
    peace = _peace_hand(wrist)
    kp[R_HAND] = fist * (1 - p2) + peace * p2
    return kp


def _act_wave(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    p1 = _phase(t, 0.0, 0.35)
    wrist = base[4] + np.array([0.06, -0.34]) * p1
    sway = math.sin(t * math.pi * 6) * 0.035 * p1 * _phase(t, 0.35, 1.0)
    wrist = wrist + np.array([sway, 0.0])
    kp[3] = base[3] + np.array([0.02, -0.12]) * p1
    kp[4] = wrist
    kp[R_HAND] = _open_hand(wrist, spread=1.1)
    return kp


def _act_thumbs_up(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    p1 = _phase(t, 0.0, 0.45)
    p2 = _phase(t, 0.35, 0.70)
    wrist = base[4] + np.array([0.02, -0.22]) * p1
    kp[3] = base[3] + np.array([0.01, -0.08]) * p1
    kp[4] = wrist
    kp[R_HAND] = _fist(wrist) * (1 - p2) + _thumbs_hand(wrist) * p2
    return kp


def _act_point(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    p1 = _phase(t, 0.0, 0.45)
    p2 = _phase(t, 0.35, 0.70)
    wrist = base[4] + np.array([0.10, -0.26]) * p1
    kp[2] = base[2] + np.array([0.02, -0.04]) * p1
    kp[3] = base[3] + np.array([0.06, -0.16]) * p1
    kp[4] = wrist
    kp[R_HAND] = _fist(wrist) * (1 - p2) + _point_hand(wrist) * p2
    return kp


def _act_nod(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    amp = 0.018 * math.sin(t * math.pi * 4) * _phase(t, 0.1, 0.9)
    for i in (0, 14, 15, 16, 17):
        kp[i] = base[i] + np.array([0.0, amp])
    return kp


def _act_shake_head(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    amp = 0.022 * math.sin(t * math.pi * 4) * _phase(t, 0.1, 0.9)
    for i in (0, 14, 15, 16, 17):
        kp[i] = base[i] + np.array([amp, 0.0])
    return kp


def _act_raise_hand(base: np.ndarray, t: float) -> np.ndarray:
    kp = base.copy()
    p1 = _phase(t, 0.0, 0.5)
    wrist = base[4] + np.array([0.04, -0.36]) * p1
    kp[3] = base[3] + np.array([0.02, -0.14]) * p1
    kp[4] = wrist
    kp[R_HAND] = _open_hand(wrist)
    return kp


ACTIONS: dict[str, ActionFn] = {
    "raise_two_fingers": _act_raise_two_fingers,
    "peace_sign": _act_raise_two_fingers,
    "two_fingers_up": _act_raise_two_fingers,
    "wave": _act_wave,
    "wave_hand": _act_wave,
    "thumbs_up": _act_thumbs_up,
    "point": _act_point,
    "point_up": _act_point,
    "nod": _act_nod,
    "nod_yes": _act_nod,
    "shake_head": _act_shake_head,
    "shake_head_no": _act_shake_head,
    "raise_hand": _act_raise_hand,
}


def list_actions() -> list[str]:
    """Canonical action names (aliases included)."""
    return sorted(ACTIONS)


def resolve_action(text: str) -> str | None:
    """Match free text to an action. Returns canonical key or None."""
    t = (text or "").lower()
    # direct hit first
    for key in ACTIONS:
        if key.replace("_", " ") in t or key in t:
            return key
    keywords = {
        "raise_two_fingers": ["two finger", "peace", "v sign", "victory"],
        "wave": ["wave", "waving", "hello"],
        "thumbs_up": ["thumbs up", "thumb up", "like this"],
        "point": ["point"],
        "nod": ["nod"],
        "shake_head": ["shake head", "no "],
        "raise_hand": ["raise hand", "hand up"],
    }
    for key, words in keywords.items():
        if any(w in t for w in words):
            return key
    return None


def build_track(action: str, *, n_frames: int = 40, fps: float = 10.0,
                base: np.ndarray | None = None,
                width: int = 512, height: int = 512) -> PoseTrack:
    """Build a keypoint trajectory for an action."""
    fn = ACTIONS.get(action)
    if fn is None:
        raise ValueError(f"unknown action {action!r}; see list_actions()")
    b = rest_pose() if base is None else base
    frames = np.stack([fn(b, i / max(1, n_frames - 1))
                       for i in range(n_frames)])
    return PoseTrack(frames=frames, fps=fps, action=action,
                     width=width, height=height)


def retarget(track: PoseTrack, detected: np.ndarray) -> PoseTrack:
    """Retarget a parametric track onto a detected rest pose.

    detected: (N_KP, 2) keypoints from DWPose on the reference image.
    Offsets from the parametric rest pose are transferred per frame.
    """
    rest = rest_pose()
    delta = detected - rest
    # scale-aware: use shoulder width ratio
    def _w(kp):
        return abs(kp[2, 0] - kp[5, 0]) + 1e-6
    s = _w(detected) / _w(rest)
    new_frames = []
    for f in track.frames:
        moved = rest + (f - rest) * s + delta
        new_frames.append(moved)
    return PoseTrack(frames=np.stack(new_frames), fps=track.fps,
                     action=track.action, width=track.width,
                     height=track.height)


# ── pose video rendering (MimicMotion input) ─────────────────────────
_LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]


def render_pose_video(track: PoseTrack, out_path: str) -> str:
    """Render the skeleton pose video (black bg, colored limbs)."""
    W, H = track.width, track.height
    tmp = Path(tempfile.mkdtemp(prefix="posevid_"))
    for i, kp in enumerate(track.frames):
        img = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        pts = (kp * np.array([W, H])).astype(int)
        for li, (a, b) in enumerate(BODY_LIMBS):
            d.line([tuple(pts[a]), tuple(pts[b])],
                   fill=_LIMB_COLORS[li % len(_LIMB_COLORS)], width=4)
        for sl, off in ((R_HAND, N_BODY), (L_HAND, N_BODY + N_HAND)):
            hp = pts[sl]
            for a, b in HAND_LIMBS:
                d.line([tuple(hp[a]), tuple(hp[b])],
                       fill=(0, 255, 255), width=2)
        for (x, y) in pts:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
        img.save(tmp / f"p_{i:04d}.png")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(track.fps), "-i", str(tmp / "p_%04d.png"),
         "-pix_fmt", "yuv420p", out_path],
        check=True, capture_output=True, timeout=300)
    return out_path
