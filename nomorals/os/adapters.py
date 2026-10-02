"""Read-only adapters: fragmented session concepts → :class:`os.Session`.

Devon grew several independent "session" ideas — the auth sessions in
:mod:`nomorals.accounts.sessions`, the live voice conversations in
:mod:`nomorals.voice.session` — and the OS control plane needs one uniform
view.  These adapters build that view *without editing any of those files*:
they wrap the existing objects and project them into
:class:`nomorals.os.session.Session`.

Defensive by contract: an adapter never raises.  Missing optional pieces
(no manager, unknown service, a half-initialized voice object) degrade to
a minimal :class:`Session` whose ``state`` records what was unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.ids import new_short_id
from .session import Session

_log = logging.getLogger(__name__)

__all__ = ["AccountSessionAdapter", "VoiceSessionAdapter"]


def _safe_view(error: Exception, *, frontend: str, principal: str,
               project_id: str, extra: dict[str, Any] | None = None) -> Session:
    """Fallback Session when the wrapped concept cannot be projected."""
    state: dict[str, Any] = {
        "degraded": True,
        "error": f"{type(error).__name__}: {error}",
    }
    if extra:
        state.update(extra)
    return Session(
        id=new_short_id("sess_"),
        principal=principal,
        frontend=frontend,
        project_id=project_id,
        state=state,
    )


class AccountSessionAdapter:
    """Project an :mod:`accounts` auth session as an OS session.

    Wraps an ``accounts.sessions.SessionManager`` (passed in — this module
    never imports it, so the OS layer stays decoupled from the accounts
    package's import weight).
    """

    frontend = "api"

    def __init__(self, manager: Any = None) -> None:
        self._manager = manager

    def to_os_session(self, service: str, username: str, *,
                      frontend: str | None = None,
                      principal: str = "owner",
                      project_id: str = "") -> Session:
        """Build an OS-session view of the auth session for
        ``(service, username)``.  Never raises."""
        try:
            return self._project(service, username,
                                 frontend=frontend or self.frontend,
                                 principal=principal, project_id=project_id)
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            _log.warning("AccountSessionAdapter degraded: %r", exc)
            return _safe_view(exc, frontend=frontend or self.frontend,
                              principal=principal, project_id=project_id,
                              extra={"service": service, "username": username})

    def _project(self, service: str, username: str, *, frontend: str,
                 principal: str, project_id: str) -> Session:
        manager = self._manager
        if manager is None:
            raise ValueError("no SessionManager was provided")
        get_session = getattr(manager, "get_session", None)
        if not callable(get_session):
            raise AttributeError("manager has no get_session(service, username)")

        acct = get_session(service, username)

        def _opt(name: str, default: Any = None) -> Any:
            try:
                return getattr(acct, name, default)
            except Exception:  # noqa: BLE001 — property may raise
                return default

        is_valid: Any = True
        validator = getattr(acct, "is_valid", None)
        if callable(validator):
            try:
                is_valid = bool(validator())
            except Exception:  # noqa: BLE001
                is_valid = False

        oauth = _opt("oauth_token")
        state: dict[str, Any] = {
            "kind": "account",
            "service": _opt("service", service),
            "username": _opt("username", username),
            "valid": is_valid,
            "has_oauth": oauth is not None,
            "scope": _opt("scope", ""),
            "metadata": _opt("metadata", {}) or {},
            "created_at": _opt("created_at"),
            "last_used": _opt("last_used"),
        }
        return Session(
            id=new_short_id("sess_"),
            principal=principal,
            frontend=frontend,
            project_id=project_id,
            conversation_id=f"{service}:{username}",
            state=state,
        )


class VoiceSessionAdapter:
    """Project a :mod:`voice.session` live conversation as an OS session.

    Wraps a ``VoiceSession`` instance (passed in or given per call).  Only
    reads plain attributes — ``session_id``, ``state``, ``device_id``,
    ``profile`` — via ``getattr`` so a partially-constructed object still
    yields a view instead of an exception.
    """

    frontend = "voice"

    def __init__(self, voice_session: Any = None) -> None:
        self._voice_session = voice_session

    def to_os_session(self, voice_session: Any = None, *,
                      frontend: str | None = None,
                      principal: str = "owner",
                      project_id: str = "") -> Session:
        """Build an OS-session view of a voice conversation.  Never raises."""
        try:
            return self._project(voice_session,
                                 frontend=frontend or self.frontend,
                                 principal=principal, project_id=project_id)
        except Exception as exc:  # noqa: BLE001 — adapters never raise
            _log.warning("VoiceSessionAdapter degraded: %r", exc)
            return _safe_view(exc, frontend=frontend or self.frontend,
                              principal=principal, project_id=project_id,
                              extra={"kind": "voice"})

    def _project(self, voice_session: Any, *, frontend: str,
                 principal: str, project_id: str) -> Session:
        vs = voice_session if voice_session is not None else self._voice_session
        if vs is None:
            raise ValueError("no voice session was provided")

        def _opt(name: str, default: Any = None) -> Any:
            try:
                return getattr(vs, name, default)
            except Exception:  # noqa: BLE001 — property may raise
                return default

        report: dict[str, Any] = {}
        stats = _opt("stats")
        if stats is not None:
            for key in ("turns", "barge_ins", "end_reason", "error"):
                report[key] = _opt(key, getattr(stats, key, None)
                                   if not isinstance(stats, dict)
                                   else stats.get(key))

        state: dict[str, Any] = {
            "kind": "voice",
            "voice_state": _opt("state", ""),
            "device_id": _opt("device_id", ""),
            "profile": _opt("profile", ""),
            "consented": bool(_opt("consent", None) is not None),
            "report": report,
        }
        return Session(
            id=new_short_id("sess_"),
            principal=principal,
            frontend=frontend,
            project_id=project_id,
            conversation_id=str(_opt("session_id", "")),
            state=state,
        )
