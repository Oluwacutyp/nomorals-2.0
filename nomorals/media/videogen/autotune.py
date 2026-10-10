"""Profile-gated resolution/duration auto-tune — never design down.

The request dataclasses carry the full-capability workstation values
as defaults. At runtime, when the caller left a field at its default,
the active profile scales it to what the machine can actually run:

- ``termux``: small + short (neural video can't run on a phone anyway —
  this keeps direct backend calls honest on weak hardware)
- ``laptop``: mid resolution, slightly shorter Wan clips / fewer steps
- ``workstation``: full defaults

Explicitly-passed values are NEVER overridden — profile gating only
fills in what the caller didn't choose.
"""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["AUTOTUNE", "autotune_request"]

#: backend → profile → tuned fields (workstation == dataclass defaults)
AUTOTUNE: dict[str, dict[str, dict[str, Any]]] = {
    "ltx": {
        "termux": {"width": 512, "height": 320,
                   "duration_s": 3.0, "fps": 16},
        "laptop": {"width": 768, "height": 512,
                   "duration_s": 5.0, "fps": 24},
        "workstation": {"width": 768, "height": 512,
                        "duration_s": 5.0, "fps": 24},
    },
    "wan": {
        "termux": {"width": 640, "height": 360, "duration_s": 3.0,
                   "fps": 16, "num_inference_steps": 30},
        "laptop": {"width": 960, "height": 540, "duration_s": 4.0,
                   "fps": 24, "num_inference_steps": 40},
        "workstation": {"width": 1280, "height": 720, "duration_s": 5.0,
                        "fps": 24, "num_inference_steps": 50},
    },
}


def autotune_request(backend: str, req: Any, *,
                     profile: str = "") -> Any:
    """Return a request with default-valued fields tuned for the
    runtime profile. Explicit values are preserved; unknown profiles
    fall back to the workstation table (full capability)."""
    from ...core.profiles import get_profile_kind
    kind = (profile or get_profile_kind() or "").lower()
    table = AUTOTUNE.get((backend or "").lower(), {})
    tuned_fields = table.get(kind) or table.get("workstation", {})
    if not tuned_fields:
        return req
    defaults = {f.name: f.default for f in dataclasses.fields(req)
                if f.default is not dataclasses.MISSING}
    out = dataclasses.replace(req)
    for key, val in tuned_fields.items():
        if key in defaults and getattr(req, key, None) == defaults[key]:
            setattr(out, key, val)
    return out
