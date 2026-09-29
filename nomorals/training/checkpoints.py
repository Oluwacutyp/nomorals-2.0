"""Checkpoint validation + selection for the free-tier training runs.

Colab/Kaggle free sessions get KILLED mid-run (12 h cap, wifi, quota) —
sometimes in the MIDDLE of saving a checkpoint.  A ``checkpoint-1250/``
directory can therefore be a corpse: no ``trainer_state.json``, a half
written optimizer file, or a state whose step does not match its folder.
Resuming from a corpse makes HF Trainer crash on a confusing error —
the user's whole next 12 h is gone to debugging.

This module is the single source of truth for "which checkpoint is
actually safe to resume from":

* :func:`validate_checkpoint` — one directory: does it hold a complete,
  parseable, step-consistent HF checkpoint?
* :func:`pick_checkpoint` — a run directory: the NEWEST *valid*
  checkpoint, skipping corrupted tails (what the generated Colab script
  inlines).
* :func:`report` — the human/agent status of a run directory: how many
  checkpoints, which are valid/corrupt, where the resume would start,
  and how far it is from the session bound.

Pure stdlib, fully hermetic (fake checkpoint trees in tests), and the
same selection logic is inlined into the generated Colab script so the
GPU box — no `nomorals` installed — behaves identically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "REQUIRED_FILES",
    "checkpoint_step",
    "validate_checkpoint",
    "pick_checkpoint",
    "report",
    "EMBEDDED_PICKER",
]

#: the files a usable HF Trainer checkpoint must contain.  The optimizer
#: + scheduler state are what "resume" actually restores (the adapter
#: weights alone would silently start the optimizer from scratch).
REQUIRED_FILES: tuple[str, ...] = (
    "trainer_state.json",
    "optimizer.bin",
    "scheduler.pt",
)

#: adapter/model payload — at least one must exist (naming varies by
#: HF/PEFT version: safetensors first, then classic bin)
_PAYLOAD_FILES: tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_model.bin",
    "model.safetensors",
)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def checkpoint_step(path: str | Path) -> int:
    """The step encoded in a ``checkpoint-<n>`` directory name (0 if the
    name does not match — such directories are never resumed)."""
    m = _CHECKPOINT_RE.search(Path(path).name)
    return int(m.group(1)) if m else 0


def validate_checkpoint(path: str | Path) -> tuple[bool, str]:
    """Is this one directory a complete, consistent HF checkpoint?

    Checks (in order, first failure reported):
      * the directory exists and is named ``checkpoint-<n>``
      * every :data:`REQUIRED_FILES` entry exists and is non-empty
      * at least one adapter/model payload exists and is non-empty
      * ``trainer_state.json`` parses as JSON with a ``global_step``
      * (consistency) ``global_step`` equals the folder's ``<n>`` —
        a mismatch means the save was interrupted between the folder
        rename and the state write.

    Returns ``(ok, reason)`` — the reason doubles as the diagnostic
    shown in :func:`report` and printed by the Colab script.
    """
    p = Path(path)
    step = checkpoint_step(p)
    if not p.is_dir():
        return False, "not a directory"
    if step <= 0:
        return False, f"directory name {p.name!r} is not checkpoint-<n>"
    for name in REQUIRED_FILES:
        f = p / name
        if not f.is_file():
            return False, f"missing {name}"
        if f.stat().st_size == 0:
            return False, f"{name} is empty (interrupted save?)"
    payload = next((n for n in _PAYLOAD_FILES if (p / n).is_file()
                    and (p / n).stat().st_size > 0), "")
    if not payload:
        return False, "no adapter/model payload (safetensors or bin)"
    state_file = p / "trainer_state.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"trainer_state.json unreadable ({exc})"
    gstep = state.get("global_step")
    if not isinstance(gstep, int):
        return False, "trainer_state.json has no integer global_step"
    if gstep != step:
        return False, (f"step mismatch: folder says {step}, state says "
                       f"{gstep} (interrupted rename?)")
    return True, "ok"


def pick_checkpoint(out_dir: str | Path,
                    max_step: int | None = None) -> dict[str, Any]:
    """The newest VALID checkpoint in a run directory — or nothing.

    Corrupted tails are skipped (a session killed mid-save leaves a
    half checkpoint AFTER the last good one), and the skip is reported
    so the Colab log says *why* it resumed from an older folder.

    Returns ``{"path": str|None, "step": int, "skipped":
    [{path, reason}...]}``.  ``max_step`` (the run's ``MAX_STEPS``) is
    only used to flag completion — a valid checkpoint at/above it means
    the run is done, and resuming would restart it.
    """
    root = Path(out_dir)
    if not root.is_dir():
        return {"path": None, "step": 0, "skipped": []}
    entries: list[tuple[int, Path]] = []
    for p in root.iterdir():
        step = checkpoint_step(p)
        if step > 0 and p.is_dir():
            entries.append((step, p))
    entries.sort(key=lambda t: t[0])
    skipped: list[dict[str, str]] = []
    best: tuple[int, str] | None = None
    for step, p in entries:
        ok, reason = validate_checkpoint(p)
        if ok:
            best = (step, str(p))
        else:
            skipped.append({"path": p.name, "step": step,
                            "reason": reason})
    out: dict[str, Any] = {
        "path": best[1] if best else None,
        "step": best[0] if best else 0,
        "skipped": skipped,
        "complete": False,
    }
    if max_step and best and best[0] >= max_step:
        out["complete"] = True
        out["note"] = (f"checkpoint-{best[0]} already reached MAX_STEPS="
                       f"{max_step} — the run is DONE, do not resume")
    return out


def report(out_dir: str | Path,
           max_step: int | None = None) -> dict[str, Any]:
    """The full status of a run directory for a human/agent.

    ``nm data checkpoints <out_dir>`` and the agent both consume this:
    every checkpoint with its validity + reason, the resume target,
    and (with ``max_step``) how far the next session has to go.
    """
    root = Path(out_dir)
    rows: list[dict[str, Any]] = []
    total = 0
    if root.is_dir():
        for p in sorted(root.iterdir(),
                        key=lambda x: checkpoint_step(x)):
            step = checkpoint_step(p)
            if step <= 0 or not p.is_dir():
                continue
            total += 1
            ok, reason = validate_checkpoint(p)
            rows.append({"dir": p.name, "step": step, "ok": ok,
                         "reason": reason})
    picked = pick_checkpoint(root, max_step=max_step)
    resume_step = picked["step"]
    remaining = (max(0, int(max_step) - resume_step) if max_step
                 else None)
    return {
        "out_dir": str(root),
        "checkpoints": total,
        "valid": sum(1 for r in rows if r["ok"]),
        "corrupt": sum(1 for r in rows if not r["ok"]),
        "details": rows,
        "resume_from": picked["path"],
        "resume_step": resume_step,
        "skipped": picked["skipped"],
        "max_step": max_step,
        "steps_remaining": remaining,
        "complete": bool(picked.get("complete")),
    }


# ── the picker, inlined for the GPU box ────────────────────────────────────
# The generated Colab script runs on a machine WITHOUT nomorals, so the
# same selection logic is embedded as a self-contained snippet (stdlib
# only, no f-strings in the outer string — % formatting keeps it safe to
# interpolate MAX_STEPS).  Keep in sync with pick_checkpoint above.
EMBEDDED_PICKER = '''
def _pick_checkpoint(out, max_steps):
    """Newest VALID checkpoint in out/ (skips corrupted mid-save corpses).

    A free-tier kill can land mid-save: the newest checkpoint-* dir may
    be missing trainer_state.json / optimizer / have a step mismatch.
    Resume from the newest dir that is actually complete, and print
    exactly what was skipped and why."""
    import json as _json, os as _os, re as _re
    cands = []
    if _os.path.isdir(out):
        for n in _os.listdir(out):
            m = _re.search(r"checkpoint-(\\d+)$", n)
            if m and _os.path.isdir(_os.path.join(out, n)):
                cands.append((int(m.group(1)), n))
    cands.sort()
    skipped, best = [], None
    for step, n in cands:
        p = _os.path.join(out, n)
        ok, why = True, "ok"
        for need in ("trainer_state.json", "optimizer.bin",
                     "scheduler.pt"):
            f = _os.path.join(p, need)
            if not _os.path.isfile(f):
                ok, why = False, "missing " + need
                break
            if _os.path.getsize(f) == 0:
                ok, why = False, need + " is empty (interrupted save?)"
                break
        payload_ok = any(
            _os.path.isfile(_os.path.join(p, f))
            and _os.path.getsize(_os.path.join(p, f)) > 0
            for f in ("adapter_model.safetensors", "adapter_model.bin",
                      "pytorch_model.bin", "model.safetensors"))
        if ok and not payload_ok:
            ok, why = False, "no adapter/model payload"
        if ok:
            sf = _os.path.join(p, "trainer_state.json")
            try:
                st = _json.loads(open(sf).read())
                g = st.get("global_step")
                if not isinstance(g, int):
                    ok, why = False, "trainer_state.json: no integer global_step"
                elif g != step:
                    # NOTE: no line may start with '%' or '!' — the
                    # notebook linter strips colab shell lines
                    msg = "step mismatch: folder says %d, state says %d (interrupted?)"
                    ok, why = False, msg % (step, g)
            except Exception as e:
                ok, why = False, "trainer_state.json unreadable: %s" % e
        if ok:
            best = (step, p)
        else:
            skipped.append((n, why))
    return best, skipped
'''
