"""Structural choreography validator — offline, no model needed.

Every generated motion score is validated BEFORE compilation:
- JSON schema compliance (phases, windows, easings, move shapes)
- Joint/finger/hand/root vocabulary
- Velocity plausibility (no teleport-speed joints)
- Phase coverage, overlaps, discontinuities
- Physical plausibility score 0-100

`validate_track` checks the COMPILED track: planted feet must not skate,
root channel must match the score's intent.

This cures "motion quality unproven against live LLM" for everything
testable offline. What remains genuinely unprovable offline is aesthetic
quality — whether the motion LOOKS like the described action — which
needs eyes (human or VLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .motion_score import _BODY_IDX, _MAX_TRAVEL, _parse_phase, HAND_POSES
from .pose_rig import BODY_NAMES, FINGERS, PoseTrack, rest_pose

_KNOWN_EASINGS = ("ease_in", "ease_out", "ease_in_out", "linear")
# max plausible joint speed: normalized units per unit of normalized time.
# A wrist crossing half the frame in a 0.1-duration phase = 5.0 -> error.
_V_MAX_WARN = 3.0
_V_MAX_ERROR = 6.0
# max plausible root speed (pelvis shouldn't fly)
_ROOT_V_MAX = 2.5


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 100.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def deduct(self, pts: float, msg: str, *, error: bool = False) -> None:
        self.score = max(0.0, self.score - pts)
        (self.errors if error else self.warnings).append(msg)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_score(score: dict) -> ValidationReport:
    """Validate a motion-score dict structurally. No model needed."""
    rep = ValidationReport()
    if not isinstance(score, dict):
        rep.deduct(100, "score is not a dict", error=True)
        return rep
    phases = score.get("phases")
    if not isinstance(phases, list) or not phases:
        rep.deduct(40, "score has no phases", error=True)
        return rep

    # simulated running joint state for velocity checks (pre-root space).
    # Seed from the rest pose so even FIRST moves get velocity-checked.
    _rest = rest_pose()
    joint_at: dict[str, tuple[float, float]] = {
        name: (float(_rest[i, 0]), float(_rest[i, 1]))
        for i, name in enumerate(BODY_NAMES)}
    prev_t1: float | None = None
    seen_windows: list[tuple[float, float, str]] = []

    for pi, praw in enumerate(phases):
        tag = f"phase[{pi}]"
        if not isinstance(praw, dict):
            rep.deduct(15, f"{tag}: not a dict", error=True)
            continue
        name = str(praw.get("name", f"phase{pi}"))
        t = praw.get("t", [0.0, 1.0])
        if (not isinstance(t, (list, tuple)) or len(t) != 2
                or not all(_is_num(v) for v in t)):
            rep.deduct(12, f"'{name}': bad time window {t!r}", error=True)
            continue
        t0, t1 = float(t[0]), float(t[1])
        if not (0.0 <= t0 < t1 <= 1.0):
            rep.deduct(12, f"'{name}': window [{t0},{t1}] outside [0,1] "
                           f"or inverted", error=True)
            continue
        dur = t1 - t0
        if prev_t1 is not None and t0 - prev_t1 > 0.02:
            rep.deduct(2, f"'{name}': gap of {t0 - prev_t1:.2f} before it "
                          f"(holds rest pose — usually fine)")
        for (a, b, oname) in seen_windows:
            if t0 < b and t1 > a:
                rep.deduct(3, f"'{name}': overlaps '{oname}' — same-joint "
                              f"moves in overlapping phases fight")
        seen_windows.append((t0, t1, name))
        prev_t1 = t1

        easing = str(praw.get("easing", "ease_in_out"))
        if easing not in _KNOWN_EASINGS:
            rep.deduct(4, f"'{name}': unknown easing {easing!r}", error=True)
        rep_raw = praw.get("repeat", 1)
        if not isinstance(rep_raw, int) or rep_raw < 1:
            rep.deduct(4, f"'{name}': bad repeat {rep_raw!r}", error=True)
        moves = praw.get("moves", []) or []
        if not isinstance(moves, list):
            rep.deduct(6, f"'{name}': moves not a list", error=True)
            continue
        if rep_raw != 1 and not moves:
            rep.deduct(2, f"'{name}': repeat with no moves does nothing")
        plant = praw.get("plant")
        if plant is not None and plant not in ("left", "right", "both"):
            rep.deduct(5, f"'{name}': bad plant {plant!r} "
                          f"(left|right|both)", error=True)

        for mi, m in enumerate(moves):
            mtag = f"'{name}'.move[{mi}]"
            if not isinstance(m, dict):
                rep.deduct(5, f"{mtag}: not a dict", error=True)
                continue
            if "root" in m:
                _check_root_move(m["root"], mtag, dur, rep)
                continue
            if "joint" in m:
                jn = m["joint"]
                if jn not in _BODY_IDX:
                    rep.deduct(8, f"{mtag}: unknown joint {jn!r}", error=True)
                    continue
                to = m.get("to")
                if (not isinstance(to, (list, tuple)) or len(to) != 2
                        or not all(_is_num(v) for v in to)):
                    rep.deduct(8, f"{mtag}: bad target {to!r}", error=True)
                    continue
                if not (0.0 <= to[0] <= 1.0 and 0.0 <= to[1] <= 1.0):
                    rep.deduct(8, f"{mtag}: target {to!r} outside [0,1]",
                               error=True)
                    continue
                start = joint_at.get(jn)
                if start is not None:
                    travel = float(np.hypot(to[0] - start[0],
                                            to[1] - start[1]))
                    vel = travel / max(dur, 1e-6)
                    if vel > _V_MAX_ERROR:
                        rep.deduct(10, f"{mtag}: {jn} moves {travel:.2f} in "
                                       f"{dur:.2f} time ({vel:.1f} u/t) — "
                                       f"teleport speed", error=True)
                    elif vel > _V_MAX_WARN:
                        rep.deduct(3, f"{mtag}: {jn} moves fast "
                                      f"({vel:.1f} u/t)")
                joint_at[jn] = (float(to[0]), float(to[1]))
            elif m.get("hand") in ("right", "left"):
                _check_hand_move(m, mtag, rep)
            else:
                rep.deduct(6, f"{mtag}: not a joint, hand, or root move",
                           error=True)

    # coverage: last phase should reach 1.0 (compiler extends it, but flag)
    if prev_t1 is not None and prev_t1 < 0.999:
        rep.deduct(2, f"phases end at t={prev_t1:.2f}; tail holds last pose")
    return rep


def _check_root_move(rm, mtag: str, dur: float, rep: ValidationReport) -> None:
    if not isinstance(rm, dict):
        rep.deduct(6, f"{mtag}: root move not a dict", error=True)
        return
    to = rm.get("to", [0.0, 0.0])
    if (not isinstance(to, (list, tuple)) or len(to) != 2
            or not all(_is_num(v) for v in to)):
        rep.deduct(6, f"{mtag}: bad root target {to!r}", error=True)
        return
    mag = float(np.hypot(to[0], to[1]))
    if mag > 0.35:
        rep.deduct(8, f"{mtag}: root travel {mag:.2f} > 0.35/phase "
                      f"(teleport)", error=True)
    elif mag / max(dur, 1e-6) > _ROOT_V_MAX:
        rep.deduct(4, f"{mtag}: root moves fast "
                      f"({mag / max(dur, 1e-6):.1f} u/t)")
    try:
        sc = float(rm.get("scale", 1.0))
    except (TypeError, ValueError):
        rep.deduct(6, f"{mtag}: bad root scale", error=True)
        return
    if not 0.5 <= sc <= 2.0:
        rep.deduct(6, f"{mtag}: root scale {sc} outside [0.5, 2.0]",
                   error=True)


def _check_hand_move(m: dict, mtag: str, rep: ValidationReport) -> None:
    pose = m.get("pose")
    if pose is not None and str(pose).lower() not in HAND_POSES:
        rep.deduct(5, f"{mtag}: unknown hand pose {pose!r}", error=True)
    fingers = m.get("fingers")
    if isinstance(fingers, dict):
        for fname, flex in fingers.items():
            if fname not in FINGERS:
                rep.deduct(5, f"{mtag}: unknown finger {fname!r}", error=True)
                continue
            if isinstance(flex, (list, tuple)):
                if (len(flex) != 3 or not all(_is_num(v) for v in flex)
                        or not all(0.0 <= float(v) <= 1.0 for v in flex)):
                    rep.deduct(5, f"{mtag}: bad flex triple for {fname}",
                               error=True)
            elif not (_is_num(flex) and 0.0 <= float(flex) <= 1.0):
                rep.deduct(5, f"{mtag}: bad flex for {fname}: {flex!r}",
                           error=True)
    fname = m.get("finger")
    if fname is not None:
        if fname not in FINGERS:
            rep.deduct(5, f"{mtag}: unknown finger {fname!r}", error=True)
        flex = m.get("flex", 0.5)
        if isinstance(flex, (list, tuple)):
            ok = (len(flex) == 3 and all(_is_num(v) for v in flex)
                  and all(0.0 <= float(v) <= 1.0 for v in flex))
        else:
            ok = _is_num(flex) and 0.0 <= float(flex) <= 1.0
        if not ok:
            rep.deduct(5, f"{mtag}: bad flex value {flex!r}", error=True)
        ab = m.get("abduct", 0.0)
        if not (_is_num(ab) and -1.0 <= float(ab) <= 1.0):
            rep.deduct(4, f"{mtag}: bad abduct {ab!r}", error=True)
    if "spread" in m:
        sp = m["spread"]
        if not (_is_num(sp) and 0.2 <= float(sp) <= 2.5):
            rep.deduct(4, f"{mtag}: bad spread {sp!r}", error=True)


def validate_track(track: PoseTrack, score: dict) -> ValidationReport:
    """Validate a COMPILED track against its score.

    Black-box checks on the final keypoints:
    - planted feet must not skate (the anti-ice-skating proof)
    - root channel present and finite when the score uses root motion
    - no NaN/inf anywhere, keypoints finite
    """
    rep = ValidationReport()
    fr = track.frames
    if not np.all(np.isfinite(fr)):
        rep.deduct(30, "track contains NaN/inf keypoints", error=True)
        return rep
    n = fr.shape[0]
    if n < 2:
        rep.deduct(20, "track has fewer than 2 frames", error=True)
        return rep

    phases = []
    try:
        phases = [_parse_phase(p) for p in score.get("phases", [])]
    except Exception as exc:  # noqa: BLE001
        rep.deduct(10, f"could not parse phases for track check: {exc}")
        return rep

    ankle = {"left": BODY_NAMES.index("l_ankle"),
             "right": BODY_NAMES.index("r_ankle")}
    for ph in phases:
        if not ph.plant:
            continue
        i0 = max(0, int(ph.t0 * (n - 1)))
        i1 = min(n - 1, int(ph.t1 * (n - 1)))
        if i1 <= i0:
            continue
        sides = ("left", "right") if ph.plant == "both" else (ph.plant,)
        for side in sides:
            seg = fr[i0:i1 + 1, ankle[side]]
            drift = float(np.max(np.linalg.norm(seg - seg[0], axis=1)))
            if drift > 0.02:
                rep.deduct(15, f"foot skate: {side} ankle drifts "
                               f"{drift:.3f} during planted phase "
                               f"'{ph.name}' (limit 0.02)", error=True)

    uses_root = any(getattr(ph, "roots", None) for ph in phases)
    if uses_root:
        if track.root is None:
            rep.deduct(10, "score uses root motion but track has no "
                           "root channel", error=True)
        elif not np.all(np.isfinite(track.root)):
            rep.deduct(10, "root channel has NaN/inf", error=True)

    # global smoothness: no frame-to-frame joint teleport
    step = np.linalg.norm(fr[1:] - fr[:-1], axis=2).max(axis=1)
    worst = float(step.max())
    if worst > 0.25:
        rep.deduct(8, f"frame snap: max joint step {worst:.2f} between "
                      f"frames (limit 0.25)")
    return rep
