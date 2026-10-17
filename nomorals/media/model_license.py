"""Model license registry — a LABEL, not a gate.

Correction (applies everywhere): licence is metadata, one field on the
registry. EVERYTHING is usable for personal AI. The licence column
stays so that if a track ever goes public, "was any non-commercial?"
is a QUERY, not an audit — same as the /play licence label: honest
label, not gate. Nothing in this module blocks a generation, a chain
run, or a personal listen. :func:`licence_audit` answers the
public-release question; :func:`can_ship` returns the
``commercial_use`` flag per model.

Registry (verified against upstream licenses):
- ACE-Step 1.5 — Apache-2.0 — commercial_use True (beds).
- DiffSinger — MIT — commercial_use True (vocals; voicebank license
  still applies to the *bank* — a bank trained on a real artist's
  voice does not grant rights to that artist's likeness).
- RVC — MIT — commercial_use True (timbre swap; same likeness caveat).
- Demucs — MIT — commercial_use True (stem separation).
- Matchering — GPL-3.0 — commercial_use True, copyleft label: a
  release embedding Matchering must comply with GPL-3 (source
  availability). Labelled, not blocked.
- MusicGen — CC-BY-NC 4.0 — commercial_use False. Personal AI: fine.
- XTTS v2 — CPML / non-commercial — commercial_use False. Personal: fine.
- Suno / Udio — proprietary, API terms — NOT IN THIS PIPELINE; listed
  so `nm models` can answer "why not Suno?" honestly: raw fidelity
  one-shot pop is where Suno wins, and we don't fight there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ModelLicense",
    "MODEL_LICENSES",
    "can_ship",
    "licence_audit",
    "release_gate",
    "license_report",
    "probe_matchering",
    "matchering_master",
    "MatcheringUnavailable",
    "register",
]

MATCHERING_INSTALL_HINT = (
    "Matchering not installed. Reference mastering needs it:\n"
    "  pip install matchering\n"
    "  (GPL-3.0 — commercial use OK, copyleft applies to the code)\n"
    "Until then, the chain uses the built-in mastering stage "
    "(voice/mastering.py) and reports it honestly."
)


@dataclass(frozen=True)
class ModelLicense:
    model: str
    license: str
    commercial: bool          # may its OUTPUT ship commercially?
    condition: str = ""       # caveat (likeness, copyleft, attribution)
    role: str = ""            # what it does in the pipeline
    source: str = ""          # upstream repo / terms URL


MODEL_LICENSES: dict[str, ModelLicense] = {
    "acestep": ModelLicense(
        "acestep", "Apache-2.0", True,
        "Apache-2.0 — output is yours.",
        "bed (instrumental)",
        "https://github.com/ace-step/ACE-Step-1.5"),
    "diffsinger": ModelLicense(
        "diffsinger", "MIT", True,
        "Voicebank license still applies; a bank of a real artist does "
        "not grant rights to that artist's likeness.",
        "vocal (notes+lyrics → sung audio)",
        "https://github.com/MoonInTheRiver/DiffSinger"),
    "rvc": ModelLicense(
        "rvc", "MIT", True,
        "Swapping in a real artist's timbre without consent is a legal "
        "risk — use own voice / consented / synthetic voices.",
        "timbre swap (preserves prosody/melody)",
        "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"),
    "demucs": ModelLicense(
        "demucs", "MIT", True, "",
        "stem separation", "https://github.com/facebookresearch/demucs"),
    "matchering": ModelLicense(
        "matchering", "GPL-3.0", True,
        "CONDITIONAL: copyleft — a release embedding Matchering must "
        "comply with GPL-3 (source availability).",
        "reference mastering", "https://github.com/sergree/matchering"),
    "musicgen": ModelLicense(
        "musicgen", "CC-BY-NC 4.0", False,
        "commercial_use False: research / personal use only. Personal AI: "
        "fine — the flag only matters if the track goes public.",
        "NOT USED in this pipeline (listed to label it out)",
        "https://github.com/facebookresearch/audiocraft"),
    "xtts": ModelLicense(
        "xtts", "CPML (non-commercial)", False,
        "commercial_use False: private/personal use only. Personal AI: fine.",
        "speech TTS / voice cloning (private surfaces only)",
        "https://github.com/coqui-ai/TTS"),
    "suno": ModelLicense(
        "suno", "Proprietary (API terms)", False,
        "Not in this pipeline. One-shot pop fidelity is where Suno wins; "
        "we don't fight there — we win on tonal singing + stems + spec.",
        "NOT USED — external service",
        "https://suno.com"),
    "udio": ModelLicense(
        "udio", "Proprietary (API terms)", False,
        "Not in this pipeline (same reason as Suno).",
        "NOT USED — external service",
        "https://udio.com"),
}


def can_ship(model: str, audience: str = "commercial") -> dict[str, Any]:
    """Licence LABEL for one model — never a gate.

    Returns {model, licence, commercial_use, personal_use, note,
    condition}. ``personal_use`` is True for every registered model:
    everything is usable for personal AI. ``commercial_use`` is the
    one flag that answers "can this ship publicly?" Unknown models
    are labelled unknown (commercial_use False) — verify the licence
    before going public; personal use is unaffected.
    """
    key = (model or "").strip().lower()
    entry = MODEL_LICENSES.get(key)
    if entry is None:
        return {"model": model, "licence": "unknown",
                "license": "unknown", "commercial_use": False,
                "personal_use": False,
                "reason": f"unknown model {model!r} — not in the license registry; verify before going public",
                "condition": ""}
    return {"model": entry.model, "licence": entry.license,
            "license": entry.license, "commercial_use": entry.commercial,
            "personal_use": True,
            "reason": ("commercial_use True — can ship publicly"
                       if entry.commercial else
                       "commercial_use False — personal AI only; do not ship publicly without clearing"),
            "condition": entry.condition}


def licence_audit(models: list[str]) -> dict[str, Any]:
    """The public-release question, as a QUERY (not a gate).

    "Was any non-commercial (model) used?" Returns
    {any_noncommercial, noncommercial: [...], commercial_use: [...],
    warnings: [...]}. Callers use it to LABEL a track's manifest —
    nothing is blocked. A track with any_noncommercial True simply
    must not go public until cleared.
    """
    noncommercial: list[str] = []
    commercial: list[str] = []
    warnings: list[str] = []
    for m in models or []:
        r = can_ship(m, "commercial")
        if r["commercial_use"]:
            commercial.append(m)
        else:
            noncommercial.append(m)
        if r.get("condition"):
            warnings.append(f"{m}: {r['condition']}")
    return {"any_noncommercial": bool(noncommercial),
            "noncommercial": noncommercial,
            "commercial_use": commercial,
            "warnings": warnings,
            "models": list(models or []),
            "note": ("Label, not gate: personal AI use is fine regardless. "
                     "any_noncommercial=True means do not publish publicly "
                     "until cleared.")}


def release_gate(models: list[str]) -> dict[str, Any]:
    """Back-compat alias for :func:`licence_audit` (label, not gate).

    Kept so existing imports keep working. The old blocking semantics
    are gone: this returns the audit, it never fails a build.
    """
    return licence_audit(models)


def license_report() -> list[dict[str, Any]]:
    """Full registry as `nm models`-style rows: model, license, commercial_use flag."""
    rows = []
    for key in sorted(MODEL_LICENSES):
        e = MODEL_LICENSES[key]
        rows.append({"model": e.model, "license": e.license,
                     "commercial_use": e.commercial, "personal_use": True,
                     "role": e.role,
                     "condition": e.condition, "source": e.source})
    return rows


# ── Matchering wrapper ──────────────────────────────────────────────────────


class MatcheringUnavailable(RuntimeError):
    """Matchering can't run here — install recipe attached."""


def probe_matchering() -> dict[str, Any]:
    """Detect a usable Matchering install. Never raises."""
    try:
        import matchering  # noqa: F401
        return {"ok": True, "method": "python",
                "detail": "matchering package importable"}
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("matchering"):
        return {"ok": True, "method": "cli", "detail": "matchering on PATH"}
    return {"ok": False, "method": "", "detail": MATCHERING_INSTALL_HINT}


def matchering_master(target_path: str, reference_path: str,
                      out_path: str) -> dict[str, Any]:
    """Reference mastering: make target sound like reference (Matchering).

    Fails closed with the install recipe when Matchering is absent.
    Never writes a fake master: on failure the output file is not
    created (or is removed if a partial write happened).
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"no such target: {target_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"no such reference: {reference_path}")
    probe = probe_matchering()
    if not probe["ok"]:
        raise MatcheringUnavailable(probe["detail"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        import matchering as mg
        mg.process(target=target_path, reference=reference_path,
                   results=[(mg.pcm16, out_path)])
    except ImportError:
        r = subprocess.run(["matchering", target_path, reference_path,
                            out_path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        if r.returncode != 0:
            raise MatcheringUnavailable(
                f"matchering CLI failed: {r.stderr[-1000:]}")
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise MatcheringUnavailable(f"matchering failed: {exc}")
    if not os.path.exists(out_path):
        raise MatcheringUnavailable("matchering produced no output")
    return {"ok": True, "path": out_path, "method": probe["method"],
            "note": "reference-mastered with Matchering (GPL-3); "
                    "per-platform loudness still applied by the caller"}


def register(registry: Any) -> None:
    """Tool registration: ``model_license`` — what can ship."""
    from ..core.policy import Capability

    @registry.register(
        "model_license",
        description=(
            "Model license registry (LABEL, not gate): action=list (every "
            "audio/music model, its license, commercial_use flag) | "
            "can-ship (model, audience → licence label) | audit/gate "
            "(models JSON → licence audit: was any non-commercial used?) "
            "| matchering (target, reference, out → reference mastering "
            "when Matchering is installed, honest failure otherwise). "
            "Everything is usable for personal AI; the flag only matters "
            "if a track goes public."
        ),
        capability=Capability.FS_WRITE,
    )
    def model_license(action: str = "list", model: str = "",
                      audience: str = "commercial", models: str = "",
                      target: str = "", reference: str = "",
                      out: str = "") -> dict:
        import json as _json
        action = (action or "list").lower()
        if action == "list":
            return {"ok": True, "models": license_report()}
        if action == "can-ship":
            return {"ok": True, **can_ship(model, audience)}
        if action == "gate" or action == "audit":
            try:
                ms = _json.loads(models) if models else []
            except Exception as exc:  # noqa: BLE001
                from ..core.errors import ToolError
                raise ToolError(f"models must be JSON: {exc}")
            return {"ok": True, **licence_audit(ms)}
        if action == "matchering":
            try:
                return {"ok": True, **matchering_master(target, reference, out)}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
