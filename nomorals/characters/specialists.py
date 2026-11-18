"""Characters' generative specialists — family (b) of the small-models
programme (owner, 2026-10-24/25, register §10).

Family (a) — small classical ML that judges/scores/predicts — lives in
``llm/specialists.py``. This module is family (b): the specialists that
MAKE things, one model per particular thing:

* ``character_voice`` — XTTS cloning speaks the character's line in
  their own voice (``voice/tts.py`` UniversalTTS, backend ``xtts``).
* ``swap_voice`` — the RVC voice-swap bridge per character
  (``voice/rvc_bridge.py``), keyed off the character's ``rvc_model``.
* ``lora_for`` — the persona LoRA adapter slot per character: one
  adapter file per character on one shared base (``llm/local_server.py``
  ``--lora`` plumbing); the adapter is validated on disk before it is
  ever named as available.
* ``swap_face`` — guarded face swap, ONLY where the character appears
  OUTSIDE the game (``media_edit/faceswap_guard.py``). NPCs in-game
  do NOT get faces — his words.

Every function returns ``{"ok": bool, ...}`` and NEVER raises: an
unavailable specialist is an honest ``ok: False`` with a reason naming
the wall (no voice named, no backend installed, no adapter file, no
face reference), and the caller falls back to the code rung (text, the
unswapped render, the shared base, the target image as-is). A model
with no weights is unavailable, never a fake render.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["character_voice", "swap_voice", "lora_for", "lora_speak",
           "swap_face"]

_log = get_logger(__name__)


def character_voice(char: Any, line: str, *, out_path: str = "",
                    audience: str = "private") -> dict[str, Any]:
    """Speak ``line`` in the character's own XTTS voice.

    The character's ``voice_name`` resolves through the voice
    catalogue; the catalogue voice's backend is preferred (``xtts``
    for a cloned voice), and the render goes through UniversalTTS.
    Returns ``{"ok": True, "path": ...}`` or ``{"ok": False,
    "reason": ...}`` naming the wall.
    """
    name = str(getattr(char, "voice_name", "") or "").strip()
    if not name:
        return {"ok": False,
                "reason": "no TTS voice named for this character — "
                          "set voice_name to a catalogue voice first"}
    try:
        from ..voice.tts import UniversalTTS, available_backends

        backends = available_backends()
        if not backends:
            return {"ok": False,
                    "reason": "no TTS backend installed on this box — "
                              "XTTS (and every other TTS backend) is "
                              "unavailable until one is installed — run "
                              "/character provision in Devon's chat to "
                              "install the voice backends"}
        from ..voice.catalogue import VoiceCatalogue

        entry = VoiceCatalogue().get(name)
        if entry is None:
            return {"ok": False,
                    "reason": f"voice {name!r} is not in the voice "
                              f"catalogue — clone it first, then name it"}
        backend = (getattr(entry, "backend", "") or "auto").lower()
        if backend != "auto" and backend not in backends:
            return {"ok": False,
                    "reason": f"TTS backend {backend!r} (the voice's "
                              f"backend) is not installed here — "
                              f"installed: {', '.join(backends)}"}
        tts = UniversalTTS(backend=backend, audience=audience)
        result = tts.speak(line, voice_name=name, out_path=out_path,
                           audience=audience)
        if isinstance(result, dict) and result.get("ok") is False:
            return {"ok": False,
                    "reason": str(result.get("reason") or result)}
        path = ""
        if isinstance(result, dict):
            path = str(result.get("path") or result.get("out_path") or "")
        return {"ok": True, "path": path or out_path, "voice": name,
                "backend": backend}
    except Exception as exc:  # noqa: BLE001 - a voice failure is a wall
        _log.debug("character voice failed", exc_info=True)
        return {"ok": False, "reason": f"TTS render failed: {exc}"}


def swap_voice(char: Any, wav_path: str, *, out_path: str = "",
               f0_up_key: int = 0) -> dict[str, Any]:
    """Swap ``wav_path``'s voice identity into the character's RVC
    model. Returns ``{"ok": True, "path": ...}`` or ``{"ok": False,
    "reason": ...}`` naming the wall (no model named, RVC not
    installed, model file missing)."""
    model = str(getattr(char, "rvc_model", "") or "").strip()
    if not model:
        return {"ok": False,
                "reason": "no RVC model named for this character — "
                          "set rvc_model to a model under the RVC "
                          "models dir first"}
    try:
        from ..voice import rvc_bridge

        probe = rvc_bridge.detect_rvc()
        if not probe.get("ok"):
            return {"ok": False,
                    "reason": f"RVC unavailable: {probe.get('detail')} — "
                              "run /character provision rvc in Devon's "
                              "chat to install the RVC runtime"}
        result = rvc_bridge.convert(wav_path, model,
                                    f0_up_key=f0_up_key,
                                    out_path=out_path)
        return {"ok": True, "path": str(result.get("path", "")),
                "model": model}
    except Exception as exc:  # noqa: BLE001 - a swap failure is a wall
        _log.debug("character RVC swap failed", exc_info=True)
        return {"ok": False, "reason": f"RVC swap failed: {exc}"}


def lora_for(char: Any) -> dict[str, Any]:
    """Resolve the character's persona LoRA adapter slot.

    Returns ``{"ok": True, "path": ...}`` when the adapter file named
    on the character exists on disk and passes the GGUF integrity
    check (the shape ``llm/local_server.py --lora`` takes). Otherwise
    ``{"ok": False, "reason": ...}`` naming the wall: no adapter
    named yet, or the named file is missing/corrupt. No adapter file
    exists for any character today — the slot is the wiring, and the
    wall is honest until one lands.
    """
    path = str(getattr(char, "lora_file", "") or "").strip()
    if not path:
        return {"ok": False,
                "reason": "no LoRA adapter named for this character — "
                          "the shared base carries the persona prompt "
                          "until an adapter file lands"}
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return {"ok": False,
                "reason": f"LoRA adapter file missing: {path}"}
    try:
        from ..llm.local_server import gguf_check

        ok, detail = gguf_check(expanded)
        if not ok:
            return {"ok": False,
                    "reason": f"LoRA adapter fails integrity: {detail}"}
    except Exception:  # noqa: BLE001 - a broken check is a wall, not a crash
        return {"ok": False,
                "reason": f"LoRA adapter could not be checked: {path}"}
    return {"ok": True, "path": expanded}


#: A failed LoRA-server boot is remembered per (base, adapter) so a
#: wedged/OOM server is not re-booted on every spoken line — a boot costs
#: a full model load. The memo is time-boxed (10 minutes): a box fixed
#: in the meantime is picked up again, and the reason is always the last
#: measured one, never a stale guess.
_LORA_DOWN_TTL_S = 600.0
_LORA_DOWN: dict[tuple[str, str], tuple[float, str]] = {}


def reset_lora_down_cache() -> None:
    """Forget remembered LoRA-server failures (tests / a repaired box)."""
    _LORA_DOWN.clear()


def lora_speak(char: Any, prompt: str, *, base_model: str = "",
               timeout_s: float = 120.0, max_tokens: int = 512
               ) -> dict[str, Any]:
    """Generate ``prompt`` through the character's LoRA server.

    The live call site ``lora_for`` was waiting for: the adapter is
    resolved through ``lora_for`` (never bypassing its validation), a
    per-adapter llama-server is ensured on its own port
    (``llm/local_server.py ensure_lora_server`` — a second instance,
    never a restart of the shared server), and the prompt is completed
    through it. Returns ``{"ok": True, "text": ..., "adapter": ...,
    "server": ...}`` or ``{"ok": False, "reason": ...}`` naming the wall
    — no adapter named, no base model configured, no binary, too little
    RAM for a second copy of the base, a failed boot, an empty answer.
    Never raises: an unavailable LoRA rung is a fallback, not a crash.

    ``base_model``: the shared base (a GGUF path or an HF repo id).
    When empty it is read from ``llm.local_model``. There is NO silent
    fallback to a catalog recommendation here — a base the owner never
    configured must never trigger a multi-GB download mid-conversation.
    """
    resolved = lora_for(char)
    if not resolved.get("ok"):
        return {"ok": False,
                "reason": str(resolved.get("reason") or "no LoRA adapter")}
    adapter = str(resolved["path"])
    base = (base_model or "").strip()
    if not base:
        try:
            from ..core.config import get_settings

            base = str(getattr(get_settings().llm, "local_model", "")
                       or "").strip()
        except Exception:  # noqa: BLE001 - no settings, no base
            base = ""
    if not base:
        return {"ok": False,
                "reason": "no local base model configured — set "
                          "llm.local_model (nm models --promote-local "
                          "<base .gguf>) so the LoRA server has a base "
                          "to stack the adapter on"}
    memo_key = (base, adapter)
    memo = _LORA_DOWN.get(memo_key)
    if memo is not None and (time.time() - memo[0]) < _LORA_DOWN_TTL_S:
        return {"ok": False, "reason": memo[1]}
    try:
        from ..llm.local_server import ensure_lora_server

        mgr, problem = ensure_lora_server(adapter, base_model=base)
        if mgr is None:
            _LORA_DOWN[memo_key] = (time.time(), problem)
            return {"ok": False, "reason": problem}
        text = mgr.complete(prompt, timeout_s=timeout_s,
                            max_tokens=max_tokens)
        if not (text or "").strip():
            return {"ok": False,
                    "reason": "LoRA server answered with no text"}
        _LORA_DOWN.pop(memo_key, None)
        return {"ok": True, "text": text.strip()[:600],
                "adapter": adapter, "server": mgr.base_url}
    except Exception as exc:  # noqa: BLE001 - a LoRA failure is a wall
        _log.debug("character LoRA speak failed", exc_info=True)
        return {"ok": False, "reason": f"LoRA generate failed: {exc}"}


def swap_face(char: Any, target_path: str, *,
              vault_dir: str | Path | None = None,
              out_path: str | Path | None = None) -> dict[str, Any]:
    """Guarded face swap for a character appearing OUTSIDE the game.

    The character's ``face_ref`` must name a reference face inside
    the vault; the swap runs through ``media_edit/faceswap_guard.py``
    (identity lock, watermark, sidecar verification). NPCs in-game do
    NOT get faces — a character with no face reference is honestly
    unavailable, and the target image is delivered as-is.
    """
    ref = str(getattr(char, "face_ref", "") or "").strip()
    if not ref:
        return {"ok": False,
                "reason": "no face reference for this character — "
                          "face swap only runs where the character "
                          "appears outside the game, and this "
                          "character has no reference face"}
    try:
        from ..media_edit.faceswap_guard import FaceswapRequest, guarded_swap

        vault = Path(vault_dir) if vault_dir is not None else Path(
            os.path.expanduser("~")) / ".nomorals" / "faces"
        req = FaceswapRequest(
            source_ref=ref, target=target_path, vault_dir=vault,
            identity_id=str(getattr(char, "id", "") or "character"),
            consent=True, out_path=out_path)
        result = guarded_swap(req)
        return {"ok": True, "path": str(result.get("output", "")),
                "sidecar": str(result.get("sidecar", ""))}
    except Exception as exc:  # noqa: BLE001 - a swap failure is a wall
        _log.debug("character face swap failed", exc_info=True)
        reason = f"face swap failed: {exc}"
        if "pip install" in str(exc):
            reason += (" — run /character provision faceswap in Devon's "
                       "chat to install the face-swap backend")
        return {"ok": False, "reason": reason}
