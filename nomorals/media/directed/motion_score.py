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
    _fist, _open_hand, _peace_hand, _point_hand, _thumbs_hand,
    _ease, _phase, resolve_action, build_track,
)

SuggestFn = Callable[[str], str]  # (prompt) -> model text

# ── hand pose vocabulary the LLM may reference ──────────────────────
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

HANDS: "hand": "right"|"left", "pose": one of fist|open|peace|point|thumbs.
Fingers are posed as a unit; fine finger choreography is not supported.

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
- Whole-body translation (walking, jumping, falling): move hips/knees/ankles
  together; the rig has no root-motion channel, so keep it subtle.

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
    hand_pose: str = "fist"


@dataclass
class _CompiledPhase:
    name: str
    t0: float
    t1: float
    easing: str
    moves: list[_CompiledMove] = field(default_factory=list)
    repeat: int = 1


def _parse_phase(raw: dict[str, Any]) -> _CompiledPhase:
    t = raw.get("t", [0.0, 1.0])
    moves: list[_CompiledMove] = []
    for m in raw.get("moves", []) or []:
        if not isinstance(m, dict):
            continue
        if "joint" in m and m["joint"] in _BODY_IDX:
            to = m.get("to")
            if (isinstance(to, (list, tuple)) and len(to) == 2
                    and all(isinstance(v, (int, float)) for v in to)):
                moves.append(_CompiledMove(
                    joint=m["joint"], to=(float(to[0]), float(to[1]))))
        elif m.get("hand") in ("right", "left"):
            pose = str(m.get("pose", "fist")).lower()
            if pose not in HAND_POSES:
                pose = "fist"
            moves.append(_CompiledMove(hand=m["hand"], hand_pose=pose))
    return _CompiledPhase(
        name=str(raw.get("name", "phase")),
        t0=float(t[0]), t1=float(t[1]),
        easing=str(raw.get("easing", "ease_in_out")),
        moves=moves,
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


def compile_score(score: dict[str, Any], *, n_frames: int = 40,
                  fps: float = 10.0,
                  base: np.ndarray | None = None) -> tuple[PoseTrack, list[str]]:
    """Compile a motion score dict -> PoseTrack.

    Returns (track, notes). Notes record every clamp applied — honest
    about where the model asked for the impossible and we reined it in.
    """
    notes: list[str] = []
    b = rest_pose() if base is None else base.copy()
    phases = [_parse_phase(p) for p in score.get("phases", [])]
    # sanity: phases must cover [0,1]; extend last phase if short
    if phases and phases[-1].t1 < 1.0:
        phases[-1].t1 = 1.0

    # current pose state per joint (starts at rest)
    joint_pos: dict[str, np.ndarray] = {
        name: b[_BODY_IDX[name]].copy() for name in BODY_NAMES}
    hand_state = {"right": "fist", "left": "fist"}

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        kp = b.copy()
        # start from the running pose state
        for name, pos in joint_pos.items():
            kp[_BODY_IDX[name]] = pos
        for side, sl in (("right", R_HAND), ("left", L_HAND)):
            wrist = kp[_BODY_IDX["r_wrist" if side == "right" else "l_wrist"]]
            kp[sl] = _HAND_FN[hand_state[side]](wrist)

        # apply active phase(s)
        for ph in phases:
            if not (ph.t0 <= t <= ph.t1 or
                    (t >= ph.t1 and ph is phases[-1])):
                continue
            span = max(1e-6, ph.t1 - ph.t0)
            u = (t - ph.t0) / span
            if ph.repeat > 1:
                u = (u * ph.repeat) % 1.0
            e = _eased(u, ph.easing)
            for mv in ph.moves:
                if mv.joint:
                    start = joint_pos[mv.joint]
                    target = np.array(mv.to)
                    # plausibility clamp: max travel from phase start
                    limit = _MAX_TRAVEL.get(mv.joint, 0.3)
                    d = target - start
                    dist = float(np.linalg.norm(d))
                    if dist > limit:
                        target = start + d / dist * limit
                        notes.append(
                            f"clamped {mv.joint} travel "
                            f"{dist:.2f}->{limit:.2f} in '{ph.name}'")
                    kp[_BODY_IDX[mv.joint]] = start + (target - start) * e
                elif mv.hand:
                    # blend hand pose shapes
                    sl = R_HAND if mv.hand == "right" else L_HAND
                    wname = "r_wrist" if mv.hand == "right" else "l_wrist"
                    wrist = kp[_BODY_IDX[wname]]
                    cur = _HAND_FN[hand_state[mv.hand]](wrist)
                    nxt = _HAND_FN[mv.hand_pose](wrist)
                    kp[sl] = cur * (1 - e) + nxt * e

        # limb-stretch guard: rescale overstretched segments toward rest
        kp, stretched = _guard_stretch(kp, b)
        notes.extend(stretched)

        frames.append(kp.copy())
        # commit end-of-frame pose as the running state for the next phase
        # (only for joints targeted by a phase ending at/after t)
        for ph in phases:
            if ph.t0 <= t <= ph.t1:
                span = max(1e-6, ph.t1 - ph.t0)
                u = (t - ph.t0) / span
                if ph.repeat > 1:
                    u = (u * ph.repeat) % 1.0
                if u >= 0.999:
                    for mv in ph.moves:
                        if mv.joint:
                            joint_pos[mv.joint] = kp[_BODY_IDX[mv.joint]].copy()
                        elif mv.hand:
                            hand_state[mv.hand] = mv.hand_pose

    track = PoseTrack(
        frames=np.stack(frames), fps=fps,
        action=str(score.get("action", "")))
    # dedupe notes, keep order
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
        return track, {"source": "preset", "action": key, "notes": []}
    # 2. open generation needs a model
    if suggest is None:
        raise RuntimeError(
            f"no preset matches {desc!r} and no model is connected to "
            "direct it — connect a model or use a preset action")
    score = generate_motion_score(desc, suggest)
    track, notes = compile_score(score, n_frames=n_frames, fps=fps, base=base)
    track.width, track.height = width, height
    return track, {"source": "generated",
                   "action": score.get("action", desc), "notes": notes}


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
