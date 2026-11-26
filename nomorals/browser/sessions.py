"""Process-wide registry of live browser sessions — the watch surface.

Every :class:`BrowserService` registers itself here on construction, so
``/browser watch`` can see EVERY live tab in this process: provisioning
drives, takeover tabs, per-job ``/browser run`` services, webshot,
planner, the CLI daemon — not just the two objects the old
``_watchable_sessions`` probe knew about. That probe also called
``SessionHandle.active_tab`` as a method; it is a ``@property``, so its
takeover-service half raised ``TypeError`` on every call and silently
returned nothing (the outer ``except`` swallowed it).

Rules of this module:
- stdlib only; no imports from agents/ or tools/ (browser/ is L4).
- weak references only: a service that is garbage-collected drops out
  by itself; no close() hook is needed anywhere.
- probing never raises: a service whose tabs are gone, whose driver
  died, or whose shape is unexpected is skipped, never fatal.
- records are plain dicts (JSON-safe except the live ``tab`` object
  itself, which callers that need it take from the ``tab`` key).
"""
from __future__ import annotations

import threading
import weakref
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

_services: weakref.WeakSet = weakref.WeakSet()
_labels: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Per-service owner-takeover hooks: a run-job service carries its
# job's bound ``pause``/``resume`` here, so a takeover (chat verb
# OR brain tool — neither of which owns the job manager) can park
# the run before handing the tab to the owner. Stored as weak-keyed
# callables: the service dropping out drops the hooks with it.
_hooks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def _safe_str(value: Any) -> str:
    try:
        return str(value or "").strip()
    except Exception:  # noqa: BLE001 - display text never raises
        return ""


def _default_label(service: Any) -> str:
    mission = _safe_str(getattr(service, "_mission_id", ""))
    if mission:
        return mission
    try:
        name = _safe_str(getattr(getattr(service, "data_dir", None),
                                 "name", ""))
    except Exception:  # noqa: BLE001 - label is cosmetic
        name = ""
    if name:
        return name
    return f"svc-{id(service) % 100000}"


def register_service(service: Any, *, label: str = "",
                     pause: Any = None, resume: Any = None) -> None:
    """Register a live browser service. Never raises.

    ``pause``/``resume`` are the owning run job's bound methods
    (``BrowserRunJob.pause``/``.resume``) when this service drives
    one; re-registering the same service updates the record, so the
    job runner attaches them after the service is built."""
    try:
        with _lock:
            _services.add(service)
            try:
                _labels[service] = (
                    (label or "").strip() or _default_label(service))
            except Exception:  # noqa: BLE001 - label is cosmetic
                _log.debug("session registry label failed", exc_info=True)
            if pause is not None or resume is not None:
                _hooks[service] = (pause, resume)
    except Exception:  # noqa: BLE001 - registration never breaks browsing
        _log.debug("session registry register failed", exc_info=True)


def _service_hooks(service: Any) -> tuple[Any, Any]:
    """(pause, resume) for a service, or (None, None). Never raises."""
    try:
        return _hooks.get(service) or (None, None)
    except Exception:  # noqa: BLE001 - hooks are best-effort
        return (None, None)


def _service_label(service: Any) -> str:
    try:
        return _labels.get(service) or _default_label(service)
    except Exception:  # noqa: BLE001 - label is cosmetic
        return "browser"


def _record(*, service_label: str, session: str, tab: Any,
            kind: str, pause: Any = None, resume: Any = None) -> dict[str, Any]:
    return {
        "service": service_label,
        "session": session,
        "tab_id": _safe_str(getattr(tab, "tab_id", "")),
        "url": _safe_str(getattr(tab, "url", "")),
        "title": _safe_str(getattr(tab, "title", "")),
        "kind": kind,
        "tab": tab,
        # Owner-takeover hooks for a run-job tab: bound
        # ``BrowserRunJob.pause``/``.resume`` (None for a tab no
        # loop is driving). Callers park the run through these,
        # never by reaching for a job manager they do not own.
        "pause": pause,
        "resume": resume,
    }


def live_sessions() -> list[dict[str, Any]]:
    """Every live tab in this process, as records.

    Rendered tabs first (they are the live pages), then plain-HTTP
    session tabs. ``display`` is unique across the list: a bare
    session name, qualified with ``[kind]`` only when the name
    collides. Never raises.
    """
    with _lock:
        services = list(_services)
    out: list[dict[str, Any]] = []
    for svc in services:
        try:
            label = _service_label(svc)
        except Exception:  # noqa: BLE001 - one bad service never sinks
            continue
        pause, resume = _service_hooks(svc)
        try:
            rendered = dict(getattr(svc, "_rendered_tabs", {}) or {})
        except Exception:  # noqa: BLE001 - probe is best-effort
            rendered = {}
        for tab in rendered.values():
            try:
                session = (_safe_str(getattr(tab, "session_name", ""))
                           or "rendered")
                out.append(_record(service_label=label, session=session,
                                   tab=tab, kind="rendered",
                                   pause=pause, resume=resume))
            except Exception:  # noqa: BLE001 - one bad tab never sinks
                continue
        try:
            sessions = dict(getattr(svc, "_sessions", {}) or {})
        except Exception:  # noqa: BLE001 - probe is best-effort
            sessions = {}
        for name, handle in sessions.items():
            try:
                # NOTE: active_tab is a @property — never call it.
                tab = handle.active_tab
            except Exception:  # noqa: BLE001 - no live tab in session
                continue
            try:
                out.append(_record(service_label=label,
                                   session=_safe_str(name), tab=tab,
                                   kind="http", pause=pause,
                                   resume=resume))
            except Exception:  # noqa: BLE001 - one bad tab never sinks
                continue
    seen: dict[str, int] = {}
    for rec in out:
        seen[rec["session"]] = seen.get(rec["session"], 0) + 1
    for rec in out:
        rec["display"] = (rec["session"] if seen[rec["session"]] == 1
                          else f"{rec['session']} [{rec['kind']}]")
    return out


def find_live_tab(ref: str) -> dict[str, Any] | None:
    """Resolve a ref to one live session record, or None.

    Matches, in order: exact session name, ``service:session``,
    tab id, URL substring, title substring (case-insensitive).
    Never raises.
    """
    want = (ref or "").strip()
    if not want:
        return None
    try:
        sessions = live_sessions()
    except Exception:  # noqa: BLE001 - lookup never raises
        return None
    if not sessions:
        return None
    for rec in sessions:
        if (rec["session"] == want
                or f"{rec['service']}:{rec['session']}" == want):
            return rec
    for rec in sessions:
        if rec["tab_id"] and rec["tab_id"] == want:
            return rec
    low = want.lower()
    for rec in sessions:
        if rec["url"] and low in rec["url"].lower():
            return rec
    for rec in sessions:
        if rec["title"] and low in rec["title"].lower():
            return rec
    return None


def is_driveable_tab(tab: Any) -> bool:
    """True when the owner can drive this tab through the takeover page.

    Needs a live frame source (``screenshot_png`` or
    ``screenshot_bytes`` — the takeover adapter bridges the latter)
    plus ``click_norm``. Plain-HTTP tabs have neither: there is no
    live page to drive, and takeover honestly refuses them.
    """
    try:
        has_shot = (callable(getattr(tab, "screenshot_png", None))
                    or callable(getattr(tab, "screenshot_bytes", None)))
        has_click = callable(getattr(tab, "click_norm", None))
        return bool(has_shot and has_click)
    except Exception:  # noqa: BLE001 - undriveable on any doubt
        return False


def describe_sessions(sessions: list[dict[str, Any]]) -> str:
    """Human list of live sessions for chat/CLI replies."""
    if not sessions:
        return "no live browser sessions right now."
    lines = [f"live browser sessions ({len(sessions)}):"]
    for rec in sessions:
        where = rec["url"] or "(no page yet)"
        title = f" — {rec['title']}" if rec["title"] else ""
        lines.append(f"  {rec['display']} — {where} "
                     f"({rec['kind']}){title}")
    return "\n".join(lines)


def live_view() -> dict[str, Any]:
    """JSON-safe live view: the human list plus one plain record per
    tab (no live objects, no hooks). Shared by the CLI (local +
    daemon) so both surfaces describe the same process the same
    way. This is the PROCESS's own live tabs — Devon's runtime
    sessions live in her process and are watched from chat
    (``/browser watch``), never from here."""
    sessions = live_sessions()
    return {"live": describe_sessions(sessions),
            "sessions": [
                {k: rec[k] for k in (
                    "service", "session", "display", "tab_id",
                    "url", "title", "kind")}
                for rec in sessions]}
