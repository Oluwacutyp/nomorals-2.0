"""Open-ended motion generation — natural language -> pose track.

RESEARCH (mined before building):
- T2M-GPT (CVPR 2023): text -> discrete motion tokens -> SMPL sequence.
  Discrete-token motion language; the gold is the token framing.
- MDM: diffusion over motion; the gold is iterative refinement.
- MotionGPT/MotionGPT-2: LLM vocabulary extended with motion tokens —
  the gold is "motion as a foreign language", instruction-tuned.
- Kimodo (NVIDIA): kinematic motion diffusion, text -> 3D joints with
  optional kinematic constraints (pose keyframes, end-effector paths).
  ~17GB VRAM. Wired as the heavy path (kimodo_status()).
- llm-animation-lab: the UNIVERSAL path gold — LLM receives skeleton
  structure + animation principles + JSON keyframe format, generates
  timed keyframes with easing. No training, works on any LLM.

This module implements the universal path: any action description ->
LLM motion score (JSON) -> compiled PoseTrack on the existing rig.
The 12 parametric presets are seed examples + instant offline path,
never the ceiling. The only limit is physical plausibility.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .pose_rig import (
    BODY_NAMES, N_KP, R_HAND, L_HAND, PoseTrack, rest_pose,
    FINGERS, finger_state_from_pose, expand_flex, render_fingers,
    _fist, _open_hand, _peace_hand, _point_hand, _thumbs_hand,
    _ease, _phase, resolve_action, build_track,
)

SuggestFn = Callable[[str], str]  # (prompt) -> model text

# ── hand pose vocabulary the LLM may reference ──────────────────────
# Unit poses are shorthands; fine finger control uses "fingers"/"finger".
HAND_POSES = ("fist", "open", "peace", "point", "thumbs")

_HAND_FN = {
    "fist": _fist,
    "open": _open_hand,
    "peace": _peace_hand,
    "point": _point_hand,
    "thumbs": _thumbs_hand,
}

# ── anatomical plausibility: max travel per phase, normalized units ──
# A joint target farther than this from its phase-start position is
# clamped. Values are generous — the limit is teleportation, not style.
_MAX_TRAVEL = {
    "nose": 0.10, "neck": 0.10,
    "r_shoulder": 0.06, "l_shoulder": 0.06,
    "r_elbow": 0.35, "l_elbow": 0.35,
    "r_wrist": 0.55, "l_wrist": 0.55,
    "r_hip": 0.08, "l_hip": 0.08,
    "r_knee": 0.30, "l_knee": 0.30,
    "r_ankle": 0.45, "l_ankle": 0.45,
    "r_eye": 0.10, "l_eye": 0.10, "r_ear": 0.10, "l_ear": 0.10,
}
# limb segment stretch tolerance: compiled segments may not exceed
# rest length by more than this (catches detached-limb nonsense)
_MAX_STRETCH = 1.35

_BODY_IDX = {name: i for i, name in enumerate(BODY_NAMES)}

# ── joint table for the LLM system prompt ───────────────────────────
def _joint_table() -> str:
    rest = rest_pose()
    lines = []
    for i, name in enumerate(BODY_NAMES):
        x, y = rest[i]
        limit = _MAX_TRAVEL.get(name, 0.3)
        lines.append(f"  {name}: rest=({x:.2f},{y:.2f}) max_travel={limit:.2f}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You are a motion director. Convert an action description into a \
motion score: timed phases moving body joints from a rest pose.

COORDINATES: normalized [0,1]. x: 0=left edge, 1=right edge. y: 0=top, \
1=bottom. The person faces the camera, centered, standing.

JOINTS (rest position, max travel per phase):
{joint_table}

HANDS: "hand": "right"|"left". Three ways to pose, from coarse to fine:
- Unit pose (shorthand): "pose": fist|open|peace|point|thumbs.
- Per-finger curl: "fingers": {{"index": 0.0, "middle": 0.0, "ring": 1.0,
  "pinky": 1.0, "thumb": 0.9}} — each finger 0.0=extended to 1.0=fully
  curled. Example — count to three: index/middle/ring at 0.0, pinky at 1.0.
  Piano playing: alternate fingers between 0.0 and 1.0 across phases.
- Single-finger detail: "finger": "index", "flex": [mcp, pip, dip]
  (each 0..1, explicit joints) or "flex": 0.5 (uniform; DIP follows PIP
  via tendon coupling), plus "abduct": 0..1 to spread it sideways.

ROOT MOTION: "root": {{"to": [dx, dy]}} — pelvis translation, RELATIVE
offset in normalized units (e.g. [0.08, 0] steps right, [0, -0.05] hops
up). Optional "scale": 1.1 to dolly in. Keep |dx|,|dy| <= 0.3 per phase.
FOOT PLANTING: phase-level "plant": "left"|"right"|"both" — the planted
foot stays glued in world space while the root moves (no ice-skating);
knees solve automatically. Walking: alternate "plant": "left"/"right"
per step. Jumping: no plant (both feet leave the ground).

RULES:
- Output ONLY valid JSON, no prose, no markdown fences.
- "phases": ordered list. Each phase: "name", "t": [start,end] in [0,1],
  "moves": list of joint/hand moves, "easing": ease_in|ease_out|ease_in_out.
- Joint targets are ABSOLUTE normalized positions, not offsets.
- Keep targets within max_travel of where the joint was at phase start.
- Limbs cannot stretch: don't place wrist farther from elbow than ~1.3x rest.
- The head (nose/eyes/ears) moves as a unit — give all five the same offset.
- "repeat": N on a phase replays its motion N times within its time window.
- If the action is physically impossible for a standing human (flying,
  teleporting, detaching limbs), set "implausible": true and explain in "notes".

EXAMPLE — "raise two fingers":
{{"action":"raise two fingers","implausible":false,"phases":[
 {{"name":"arm rises","t":[0.0,0.45],"easing":"ease_out","moves":[
   {{"joint":"r_elbow","to":[0.38,0.31]}},
   {{"joint":"r_wrist","to":[0.43,0.27]}},
   {{"hand":"right","pose":"fist"}}]}},
 {{"name":"fingers extend","t":[0.30,0.70],"easing":"ease_in_out","moves":[
   {{"hand":"right","pose":"peace"}}]}},
 {{"name":"hold","t":[0.70,1.0],"easing":"ease_in_out","moves":[]}}
]}}

EXAMPLE — "nod yes":
{{"action":"nod yes","implausible":false,"phases":[
 {{"name":"nod","t":[0.1,0.9],"easing":"ease_in_out","repeat":2,"moves":[
   {{"joint":"nose","to":[0.50,0.19]}},
   {{"joint":"r_eye","to":[0.47,0.175]}},{{"joint":"l_eye","to":[0.53,0.175]}},
   {{"joint":"r_ear","to":[0.44,0.19]}},{{"joint":"l_ear","to":[0.56,0.19]}}]}}
]}}

Now direct this action: {description}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model text."""
    text = text.strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("unbalanced JSON in model output")
    return json.loads(text[start:end])


def generate_motion_score(description: str, suggest: SuggestFn) -> dict[str, Any]:
    """Ask the LLM to direct the action -> motion score dict.

    Raises ValueError on unparseable output, RuntimeError when the
    model flags the action implausible.
    """
    prompt = _SYSTEM_PROMPT.format(
        joint_table=_joint_table(), description=description.strip())
    raw = suggest(prompt)
    if not raw or not raw.strip():
        raise ValueError("model returned nothing for motion direction")
    score = _extract_json(raw)
    if score.get("implausible"):
        raise RuntimeError(
            "physically implausible action: "
            + str(score.get("notes", "no reason given")))
    if not isinstance(score.get("phases"), list) or not score["phases"]:
        raise ValueError("motion score has no phases")
    return score


@dataclass
class _CompiledMove:
    joint: str | None = None       # body joint name
    hand: str | None = None        # "right" | "left"
    to: tuple[float, float] | None = None
    hand_pose: str | None = None   # legacy unit pose shorthand
    fingers: dict | None = None    # {finger: (mcp,pip,dip,abduct)}
    spread: float | None = None    # whole-hand abduction multiplier


@dataclass
class _CompiledRoot:
    to: tuple[float, float]        # relative (dx, dy)
    scale: float = 1.0             # relative scale multiplier


@dataclass
class _CompiledPhase:
    name: str
    t0: float
    t1: float
    easing: str
    moves: list[_CompiledMove] = field(default_factory=list)
    roots: list[_CompiledRoot] = field(default_factory=list)
    plant: str | None = None       # "left" | "right" | "both"
    repeat: int = 1


def _parse_finger_state(m: dict[str, Any]) -> tuple[dict | None, float | None]:
    """Parse fine-finger choreography -> (fingers dict, spread).

    fingers dict: finger -> (mcp_flex, pip_flex, dip_flex, abduct).
    """
    fingers: dict[str, tuple[float, float, float, float]] = {}
    spread: float | None = None
    if "spread" in m:
        try:
            spread = max(0.2, min(2.5, float(m["spread"])))
        except (TypeError, ValueError):
            pass
    fm = m.get("fingers")
    if isinstance(fm, dict):
        for fname, flex in fm.items():
            if fname in FINGERS:
                fingers[fname] = (*expand_flex(flex), 0.0)
    if m.get("finger") in FINGERS:
        fname = m["finger"]
        flex = expand_flex(m.get("flex", 0.5))
        try:
            abduct = max(-1.0, min(1.0, float(m.get("abduct", 0.0))))
        except (TypeError, ValueError):
            abduct = 0.0
        fingers[fname] = (*flex, abduct)
    return (fingers or None), spread


def _parse_phase(raw: dict[str, Any]) -> _CompiledPhase:
    t = raw.get("t", [0.0, 1.0])
    moves: list[_CompiledMove] = []
    roots: list[_CompiledRoot] = []
    for m in raw.get("moves", []) or []:
        if not isinstance(m, dict):
            continue
        if "root" in m and isinstance(m["root"], dict):
            rm = m["root"]
            to = rm.get("to", [0.0, 0.0])
            if (isinstance(to, (list, tuple)) and len(to) == 2
                    and all(isinstance(v, (int, float)) for v in to)):
                try:
                    sc = float(rm.get("scale", 1.0))
                except (TypeError, ValueError):
                    sc = 1.0
                roots.append(_CompiledRoot(
                    to=(float(to[0]), float(to[1])),
                    scale=max(0.5, min(2.0, sc))))
            continue
        if "joint" in m and m["joint"] in _BODY_IDX:
            to = m.get("to")
            if (isinstance(to, (list, tuple)) and len(to) == 2
                    and all(isinstance(v, (int, float)) for v in to)):
                moves.append(_CompiledMove(
                    joint=m["joint"], to=(float(to[0]), float(to[1]))))
        elif m.get("hand") in ("right", "left"):
            fingers, spread = _parse_finger_state(m)
            pose = str(m.get("pose", "")).lower() or None
            if pose not in HAND_POSES:
                pose = None
            if fingers or spread is not None or pose:
                moves.append(_CompiledMove(hand=m["hand"], hand_pose=pose,
                                           fingers=fingers, spread=spread))
    plant = raw.get("plant")
    if plant not in ("left", "right", "both"):
        plant = None
    return _CompiledPhase(
        name=str(raw.get("name", "phase")),
        t0=float(t[0]), t1=float(t[1]),
        easing=str(raw.get("easing", "ease_in_out")),
        moves=moves,
        roots=roots,
        plant=plant,
        repeat=max(1, int(raw.get("repeat", 1) or 1)),
    )


def _eased(u: float, kind: str) -> float:
    u = max(0.0, min(1.0, u))
    if kind == "ease_in":
        return u * u * u
    if kind == "ease_out":
        return 1 - (1 - u) ** 3
    if kind == "linear":
        return u
    return _ease(u)  # ease_in_out default


def _two_bone_ik(hip: np.ndarray, target: np.ndarray,
                 l1: float, l2: float, pole: np.ndarray) -> np.ndarray:
    """Solve knee position for a hip->knee->ankle chain reaching target.

    Law of cosines in 2D; the knee bends toward `pole` (unit-ish vector
    derived from the rest pose). Returns the knee position.
    """
    d = target - hip
    dist = float(np.linalg.norm(d))
    dc = min(max(dist, abs(l1 - l2) * 1.001 + 1e-9), (l1 + l2) * 0.999)
    u = d / (dist + 1e-9)
    a = (l1 * l1 - l2 * l2 + dc * dc) / (2 * dc + 1e-9)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    perp = np.array([-u[1], u[0]])
    if float(np.dot(perp, pole)) < 0:
        perp = -perp
    return hip + u * a + perp * h


def _knee_pole(base: np.ndarray, hip: str, knee: str,
               ankle: str) -> np.ndarray:
    """Knee-bend direction from the rest pose (which side of hip->ankle)."""
    h, k, an = (base[_BODY_IDX[hip]], base[_BODY_IDX[knee]],
                base[_BODY_IDX[ankle]])
    d = an - h
    dist = float(np.linalg.norm(d)) + 1e-9
    u = d / dist
    proj = h + u * float(np.dot(k - h, u))
    pole = k - proj
    n = float(np.linalg.norm(pole))
    if n < 1e-6:
        return np.array([0.0, 1.0])  # screen-down fallback
    return pole / n


def _blend_finger_states(cur: dict, nxt: dict, e: float) -> dict:
    """Interpolate per-finger (mcp,pip,dip,abduct) tuples."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for f in FINGERS:
        a = cur.get(f, (1.0, 1.0, 0.65, 0.0))
        b = nxt.get(f, a)
        out[f] = tuple(ai + (bi - ai) * e for ai, bi in zip(a, b))
    return out


def compile_score(score: dict[str, Any], *, n_frames: int = 40,
                  fps: float = 10.0,
                  base: np.ndarray | None = None) -> tuple[PoseTrack, list[str]]:
    """Compile a motion score dict -> PoseTrack.

    Returns (track, notes). Notes record every clamp applied — honest
    about where the model asked for the impossible and we reined it in.
    The track carries the root channel (dx, dy, scale per frame).

    Running state (joint_pos, hand_state, root_pos) is kept in PRE-ROOT
    space so phase handoffs stay continuous even through foot planting:
    while a leg is planted, its knee/ankle running state is fed from the
    IK solution converted back to pre-root space every frame.
    """
    notes: list[str] = []
    b = rest_pose() if base is None else base.copy()
    phases = [_parse_phase(p) for p in score.get("phases", [])]
    if phases and phases[-1].t1 < 1.0:
        phases[-1].t1 = 1.0

    # running pose state, all PRE-ROOT space (starts at rest)
    joint_pos: dict[str, np.ndarray] = {
        name: b[_BODY_IDX[name]].copy() for name in BODY_NAMES}
    hand_state = {
        "right": {"fingers": finger_state_from_pose("fist"), "spread": 1.0},
        "left": {"fingers": finger_state_from_pose("fist"), "spread": 1.0},
    }
    root_pos = np.zeros(2)
    root_scale = 1.0
    # foot planting: side -> planted ankle world position (or None)
    plant_state: dict[str, np.ndarray | None] = {"left": None, "right": None}
    prev_plant_spec: dict[str, bool] = {"left": False, "right": False}

    def _seg(a: str, c: str) -> float:
        return float(np.linalg.norm(b[_BODY_IDX[c]] - b[_BODY_IDX[a]]))

    leg_len = {
        "left": (_seg("l_hip", "l_knee"), _seg("l_knee", "l_ankle")),
        "right": (_seg("r_hip", "r_knee"), _seg("r_knee", "r_ankle")),
    }
    knee_pole = {
        "left": _knee_pole(b, "l_hip", "l_knee", "l_ankle"),
        "right": _knee_pole(b, "r_hip", "r_knee", "r_ankle"),
    }
    _SIDE_J = {
        "left": ("l_hip", "l_knee", "l_ankle"),
        "right": ("r_hip", "r_knee", "r_ankle"),
    }

    def _active(t: float) -> list[_CompiledPhase]:
        return [ph for ph in phases
                if ph.t0 <= t <= ph.t1 or (t >= ph.t1 and ph is phases[-1])]

    def _plant_spec(active: list[_CompiledPhase]) -> dict[str, bool]:
        spec = {"left": False, "right": False}
        for ph in active:
            if ph.plant in ("left", "both"):
                spec["left"] = True
            if ph.plant in ("right", "both"):
                spec["right"] = True
        return spec

    def _eased_u(ph: _CompiledPhase, t: float) -> float:
        span = max(1e-6, ph.t1 - ph.t0)
        u = (t - ph.t0) / span
        if ph.repeat > 1:
            u = (u * ph.repeat) % 1.0
        return _eased(u, ph.easing), u

    frames: list[np.ndarray] = []
    root_chan: list[list[float]] = []
    prev_kp = b.copy()
    committed: set[int] = set()  # phase indices whose effects are committed
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        kp = b.copy()
        for name, pos in joint_pos.items():
            kp[_BODY_IDX[name]] = pos

        active = _active(t)
        # foot-plant transitions: capture ankle world pos when planting starts
        spec = _plant_spec(active)
        for side in ("left", "right"):
            if spec[side] and not prev_plant_spec[side]:
                ankle = _BODY_IDX[_SIDE_J[side][2]]
                plant_state[side] = prev_kp[ankle].copy()
            elif not spec[side]:
                plant_state[side] = None
        prev_plant_spec = spec

        # root motion from active phases (eased relative offsets)
        off = np.zeros(2)
        sc = 1.0
        for ph in active:
            e, _ = _eased_u(ph, t)
            for rt in ph.roots:
                mag = math.hypot(*rt.to)
                rto = rt.to
                if mag > 0.35:
                    rto = (rt.to[0] / mag * 0.35, rt.to[1] / mag * 0.35)
                    notes.append(
                        f"clamped root travel {mag:.2f}->0.35 in '{ph.name}'")
                off += np.array(rto) * e
                sc *= 1.0 + (rt.scale - 1.0) * e
        frame_root = root_pos + off
        frame_scale = root_scale * sc
        pelvis = (kp[_BODY_IDX["r_hip"]] + kp[_BODY_IDX["l_hip"]]) / 2

        def _to_world(pre: np.ndarray) -> np.ndarray:
            return pelvis + (pre - pelvis) * frame_scale + frame_root

        def _to_pre(world: np.ndarray) -> np.ndarray:
            return (world - frame_root - pelvis) / frame_scale + pelvis

        kp[:len(BODY_NAMES)] = _to_world(kp[:len(BODY_NAMES)])
        root_chan.append([float(frame_root[0]), float(frame_root[1]),
                          float(frame_scale)])

        # foot planting: pin stance ankles, solve knees with two-bone IK.
        # Planted joints are IK-owned: joint moves targeting them are
        # skipped, and running state feeds from the IK solution so the
        # plant->release handoff stays continuous.
        ik_owned: set[str] = set()
        for side in ("left", "right"):
            if plant_state[side] is not None:
                hip_n, knee_n, ankle_n = _SIDE_J[side]
                l1, l2 = leg_len[side]
                knee_p = _two_bone_ik(kp[_BODY_IDX[hip_n]],
                                      plant_state[side], l1, l2,
                                      knee_pole[side])
                kp[_BODY_IDX[knee_n]] = knee_p
                kp[_BODY_IDX[ankle_n]] = plant_state[side]
                ik_owned.add(knee_n)
                ik_owned.add(ankle_n)
                joint_pos[knee_n] = _to_pre(knee_p)
                joint_pos[ankle_n] = _to_pre(plant_state[side])

        # joint + hand moves. Running state (joint_pos / hand_state /
        # root_pos) is committed only when a phase ELAPSES (t >= t1),
        # never mid-phase — so phases that end between frames still hand
        # off cleanly. Repeat phases don't commit (they oscillate).
        for ph in active:
            e, _ = _eased_u(ph, t)
            for mv in ph.moves:
                if mv.joint:
                    if mv.joint in ik_owned:
                        continue  # IK owns this joint this frame
                    start = joint_pos[mv.joint]
                    target = np.array(mv.to)
                    limit = _MAX_TRAVEL.get(mv.joint, 0.3)
                    d = target - start
                    dist = float(np.linalg.norm(d))
                    if dist > limit:
                        target = start + d / dist * limit
                        notes.append(
                            f"clamped {mv.joint} travel "
                            f"{dist:.2f}->{limit:.2f} in '{ph.name}'")
                    moved = start + (target - start) * e
                    kp[_BODY_IDX[mv.joint]] = _to_world(moved)
                elif mv.hand:
                    sl = R_HAND if mv.hand == "right" else L_HAND
                    wname = "r_wrist" if mv.hand == "right" else "l_wrist"
                    wrist = kp[_BODY_IDX[wname]]
                    cur = hand_state[mv.hand]["fingers"]
                    tgt = dict(cur)
                    if mv.hand_pose:
                        tgt = finger_state_from_pose(mv.hand_pose)
                    if mv.fingers:
                        tgt.update(mv.fingers)
                    spread = (mv.spread if mv.spread is not None
                              else hand_state[mv.hand]["spread"])
                    kp[sl] = render_fingers(wrist, 0.045,
                                            _blend_finger_states(cur, tgt, e),
                                            spread)

        # limb-stretch guard
        kp, stretched = _guard_stretch(kp, b)
        notes.extend(stretched)

        frames.append(kp.copy())
        prev_kp = kp.copy()
        # commit phase effects once their time has elapsed
        for pi, ph in enumerate(phases):
            if pi in committed or t < ph.t1 - 1e-9 or ph.repeat != 1:
                continue
            committed.add(pi)
            for mv in ph.moves:
                if mv.joint and mv.joint not in ik_owned:
                    start = joint_pos[mv.joint]
                    target = np.array(mv.to)
                    limit = _MAX_TRAVEL.get(mv.joint, 0.3)
                    d = target - start
                    dist = float(np.linalg.norm(d))
                    if dist > limit:
                        target = start + d / dist * limit
                    joint_pos[mv.joint] = target
                elif mv.hand:
                    hs = hand_state[mv.hand]
                    if mv.hand_pose:
                        hs["fingers"] = finger_state_from_pose(mv.hand_pose)
                    if mv.fingers:
                        hs["fingers"].update(mv.fingers)
                    if mv.spread is not None:
                        hs["spread"] = mv.spread
            for rt in ph.roots:
                mag = math.hypot(*rt.to)
                rto = rt.to
                if mag > 0.35:
                    rto = (rt.to[0] / mag * 0.35, rt.to[1] / mag * 0.35)
                root_pos = root_pos + np.array(rto)
                root_scale = root_scale * rt.scale

    track = PoseTrack(
        frames=np.stack(frames), fps=fps,
        action=str(score.get("action", "")),
        root=np.array(root_chan))
    seen: set[str] = set()
    uniq = [n for n in notes if not (n in seen or seen.add(n))]
    return track, uniq


_BODY_SEGMENTS = [
    ("neck", "r_shoulder"), ("neck", "l_shoulder"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("neck", "r_hip"), ("neck", "l_hip"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
]


def _guard_stretch(kp: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Pull overstretched limb segments back toward rest length."""
    notes: list[str] = []
    for a, b in _BODY_SEGMENTS:
        ia, ib = _BODY_IDX[a], _BODY_IDX[b]
        rest_len = float(np.linalg.norm(rest[ib] - rest[ia])) + 1e-9
        cur = kp[ib] - kp[ia]
        cur_len = float(np.linalg.norm(cur)) + 1e-9
        if cur_len > rest_len * _MAX_STRETCH:
            kp[ib] = kp[ia] + cur / cur_len * rest_len * _MAX_STRETCH
            notes.append(f"stretch-guarded {a}->{b}")
    return kp, notes


def generate_track(description: str, *, suggest: SuggestFn | None = None,
                   n_frames: int = 40, fps: float = 10.0,
                   base: np.ndarray | None = None,
                   width: int = 512, height: int = 512
                   ) -> tuple[PoseTrack, dict[str, Any]]:
    """Natural language -> PoseTrack. The open-ended generator.

    1. Preset fast path: known actions use the parametric rig, no model.
    2. LLM path: any other description -> motion score -> compiled track.
    3. Without a model and without a preset: honest failure, never a guess.

    Returns (track, meta) where meta has source/notes.
    """
    desc = (description or "").strip()
    if not desc:
        raise ValueError("empty action description")
    # 1. preset fast path — instant, offline, deterministic
    key = resolve_action(desc)
    if key is not None:
        track = build_track(key, n_frames=n_frames, fps=fps, base=base,
                            width=width, height=height)
        return track, {"source": "preset", "action": key, "notes": [],
                       "plausibility": 100.0}
    # 2. open generation needs a model
    if suggest is None:
        raise RuntimeError(
            f"no preset matches {desc!r} and no model is connected to "
            "direct it — connect a model or use a preset action")
    score = generate_motion_score(desc, suggest)
    from .validate import validate_score, validate_track
    rep = validate_score(score)
    notes: list[str] = []
    if rep.warnings:
        notes.extend(f"validator: {w}" for w in rep.warnings)
    if not rep.ok:
        raise ValueError(
            "motion score failed structural validation "
            f"(plausibility {rep.score:.0f}/100): "
            + "; ".join(rep.errors))
    track, cnotes = compile_score(score, n_frames=n_frames, fps=fps, base=base)
    trep = validate_track(track, score)
    if not trep.ok:
        raise ValueError(
            "compiled track failed validation: " + "; ".join(trep.errors))
    notes.extend(f"validator: {w}" for w in trep.warnings)
    track, notes = track, notes + cnotes
    track.width, track.height = width, height
    return track, {"source": "generated",
                   "action": score.get("action", desc), "notes": notes,
                   "plausibility": rep.score}


# ── neural text-to-motion (heavy path) ───────────────────────────────
KIMODO_REPO = "https://github.com/nv-tlabs/kimodo"
KIMODO_MODELS_DIR = None  # resolved lazily (pathlib.Path.home())


def kimodo_status() -> dict[str, Any]:
    """Check NVIDIA Kimodo text-to-motion availability.

    Kinematic motion diffusion: text -> 3D joints with optional kinematic
    constraints. ~17GB VRAM. The gold from research: constraint-aware
    generation (pose keyframes, end-effector paths as JSON).
    """
    from pathlib import Path
    models_dir = Path.home() / ".devon-models" / "kimodo"
    try:
        import torch  # noqa: F401
        has_torch = True
        cuda = torch.cuda.is_available()
        vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if cuda else 0.0)
    except Exception:
        has_torch, cuda, vram = False, False, 0.0
    ok = bool(has_torch and cuda and vram >= 15.0
              and (models_dir / "Kimodo-SOMA-RP-v1").exists())
    return {
        "available": ok,
        "torch": has_torch, "cuda": cuda,
        "vram_gb": round(vram, 1),
        "reason": (
            "Kimodo needs torch+CUDA with ~17GB VRAM and the "
            "Kimodo-SOMA-RP-v1 weights from HuggingFace in "
            f"{models_dir}. Install: pip install kimodo && weights "
            f"auto-download on first use. Repo: {KIMODO_REPO}. "
            "Until then, the LLM motion-score path above handles "
            "any describable action on CPU."
        ),
    }
