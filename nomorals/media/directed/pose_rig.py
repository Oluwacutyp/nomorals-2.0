"""Parametric pose rig — the DIRECTION layer.

An action description ("raise two fingers") becomes a keypoint trajectory:
COCO-18 body + 21-pt hands (MediaPipe order), DWPose-compatible layout,
normalized [0,1] coordinates. The trajectory renders as a skeleton pose
video — the exact input MimicMotion / AnimateAnyone consume — and also
drives the CPU mesh-warp animator.

No model needed here: the ACTION is authored parametrically, the neural
model (when present) only renders it photoreal. Deterministic, editable,
offline.

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


def _fist(wrist: np.ndarray, size: float = 0.035) -> np.ndarray:
    """Curled fist keypoints around a wrist position."""
    pts = np.zeros((N_HAND, 2))
    pts[0] = wrist
    rng = np.random.RandomState(7)
    for f in range(5):
        base_ang = -math.pi / 2 + (f - 2) * 0.28
        for j, frac in enumerate((0.35, 0.55, 0.62, 0.60)):
            idx = 1 + f * 4 + j
            ang = base_ang + rng.randn() * 0.05
            pts[idx] = wrist + np.array(
                [math.cos(ang), math.sin(ang)]) * size * frac
    return pts


def _open_hand(wrist: np.ndarray, size: float = 0.045,
               spread: float = 1.0) -> np.ndarray:
    """Open palm, fingers extended upward from wrist."""
    pts = np.zeros((N_HAND, 2))
    pts[0] = wrist
    for f in range(5):
        ang = -math.pi / 2 + (f - 2) * 0.30 * spread
        lens = (0.45, 0.75, 0.95, 1.0) if f else (0.4, 0.6, 0.75, 0.8)
        for j, frac in enumerate(lens):
            idx = 1 + f * 4 + j
            pts[idx] = wrist + np.array(
                [math.cos(ang), math.sin(ang)]) * size * frac
    return pts


def _peace_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    """Two fingers up (index + middle extended, rest curled)."""
    pts = _fist(wrist, size)
    for f, idx0 in ((1, 5), (2, 9)):  # index, middle
        for j, frac in enumerate((0.5, 0.8, 1.0, 1.05)):
            ang = -math.pi / 2 + (f - 1.5) * 0.22
            pts[idx0 + j] = wrist + np.array(
                [math.cos(ang), math.sin(ang)]) * size * frac
    return pts


def _point_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    pts = _fist(wrist, size)
    for j, frac in enumerate((0.5, 0.8, 1.0, 1.05)):
        ang = -math.pi / 2
        pts[5 + j] = wrist + np.array(
            [math.cos(ang), math.sin(ang)]) * size * frac
    return pts


def _thumbs_hand(wrist: np.ndarray, size: float = 0.045) -> np.ndarray:
    pts = _fist(wrist, size)
    for j, frac in enumerate((0.45, 0.7, 0.9, 1.0)):
        ang = -math.pi / 2 + 0.9  # thumb out to the side-up
        pts[1 + j] = wrist + np.array(
            [math.cos(ang), math.sin(ang)]) * size * frac
    return pts


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
