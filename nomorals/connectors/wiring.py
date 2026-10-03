"""Shared streaming-adapter wiring for music playback.

The player (``nomorals.media.playback``, L4) never imports the connector
layer (L5) — adapters are *injected* via the ``spotify_adapter`` /
``soundcloud_adapter`` context attributes.  This module is the one L5
place that performs that injection, so the ``nm music`` CLI (L7) and the
chat runtime (L5) share identical wiring instead of drifting apart.

Idempotent per process; failures are reported, never fatal (local files
still play).
"""

from __future__ import annotations

import os
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["wire_streaming_adapters", "spotify_linked", "spotify_connector"]


def wire_streaming_adapters(context: Any) -> dict[str, str]:
    """Attach SoundCloud/Spotify adapters to ``context`` (best-effort).

    Returns ``{name: "ok" | "skipped: <reason>"}``.
    """
    from .registry import create_connector

    if getattr(context, "spotify_adapter", None) is not None and \
            getattr(context, "soundcloud_adapter", None) is not None:
        return {"spotify": "ok", "soundcloud": "ok"}
    result: dict[str, str] = {}
    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    vault = None
    if passphrase:
        try:
            from ..accounts.vault import CredentialVault
            vault = CredentialVault(
                getattr(context, "db", None), master_passphrase=passphrase)
        except Exception as exc:  # noqa: BLE001 - locked vault, skip
            _log.debug("streaming wiring: vault unavailable: %s", exc)
    # SoundCloud is keyless — the connector never reads the vault, so a
    # missing/locked vault is fine for it.
    if getattr(context, "soundcloud_adapter", None) is None:
        try:
            context.soundcloud_adapter = create_connector(
                "soundcloud", vault)
            result["soundcloud"] = "ok"
        except Exception as exc:  # noqa: BLE001 - streaming is optional
            result["soundcloud"] = f"skipped: {exc}"
    else:
        result["soundcloud"] = "ok"
    if getattr(context, "spotify_adapter", None) is None:
        if vault is None:
            # Spotify's OAuth tokens live in the vault — locked, skip.
            result["spotify"] = "skipped: vault locked (no NM_VAULT_PASSPHRASE)"
        else:
            try:
                context.spotify_adapter = create_connector("spotify", vault)
                result["spotify"] = "ok"
            except Exception as exc:  # noqa: BLE001 - streaming is optional
                result["spotify"] = f"skipped: {exc}"
    else:
        result["spotify"] = "ok"
    return result


def spotify_connector(context: Any) -> Any | None:
    """A Spotify connector instance, or None when it can't be built."""
    adapter = getattr(context, "spotify_adapter", None)
    if adapter is not None:
        return adapter
    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    if not passphrase:
        return None
    try:
        from ..accounts.vault import CredentialVault
        from .registry import create_connector
        vault = CredentialVault(getattr(context, "db", None),
                                master_passphrase=passphrase)
        return create_connector("spotify", vault)
    except Exception:  # noqa: BLE001 - best-effort
        return None


def spotify_linked(context: Any) -> bool:
    """True when Spotify OAuth is usable right now (honest, live check)."""
    adapter = spotify_connector(context)
    if adapter is None:
        return False
    try:
        return bool(adapter.test_connection())
    except Exception:  # noqa: BLE001 - offline / bad token → not linked
        return False


def spotify_link_help() -> str:
    """What to tell the owner when Spotify isn't linked."""
    return (
        "Spotify isn't linked. Link it with "
        "`nm connectors connect --name spotify`, or say “link Spotify” "
        "and I'll drive the OAuth flow (tokens stay in your vault — "
        "never paste them in chat)."
    )
