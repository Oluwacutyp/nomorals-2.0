"""Live agent window — Hark-style trust UX for browser tasks.

When Devon runs a browser task for the owner, a small live view shows what
she is doing: one chat message whose screenshot updates on each major
action (navigate/click/fill/...), instead of a silent black box.

Cost honesty:
- screenshots themselves are cheap; chat spam is not — updates are
  throttled per profile (termux: 30s, laptop/workstation: 8s) and
  coalesced (rapid actions collapse into the latest description).
- hard cap of ``LIVE_VIEW_MAX_SHOTS`` screenshots per task (default 25);
  the finish frame is always allowed through.
- in-place ``edit_media`` is preferred (one message, updated); adapters
  without it fall back to throttled new photo messages.

Opt-in: nothing happens unless :meth:`LiveView.attach` is called with
``live_view=True`` (the default) — background tasks stay silent.

Every public method never raises: a broken live view must never kill the
browser task it is observing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from ..social.chat.base import ChatAdapter, ChatRef, MediaRef, SendResult

_log = get_logger(__name__)

#: Hard cap on screenshots per task (the finish frame may exceed it by one).
LIVE_VIEW_MAX_SHOTS = 25

#: Minimum seconds between chat updates, by profile. Screenshots are cheap;
#: message spam is not — termux is throttled hardest.
_LIVE_VIEW_INTERVAL_S = {
    "termux": 30.0,
    "laptop": 8.0,
    "workstation": 8.0,
}


def _interval_for(profile: str | None) -> float:
    kind = (profile or get_profile_kind()).strip().lower()
    return _LIVE_VIEW_INTERVAL_S.get(kind, 8.0)


class LiveView:
    """One live window onto a browser task.

    ``shot`` is a zero-arg callable returning a screenshot path (str/Path)
    or ``None`` when the shot failed. It is invoked lazily — only when an
    update is actually due — and its exceptions are swallowed.
    """

    def __init__(
        self,
        shot: Callable[[], Any],
        *,
        profile: str | None = None,
        max_shots: int = LIVE_VIEW_MAX_SHOTS,
        min_interval_s: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._shot = shot
        self._profile = profile
        self._max_shots = max(1, int(max_shots))
        self._interval = (min_interval_s if min_interval_s is not None
                          else _interval_for(profile))
        self._clock = clock or time.monotonic
        self._chat: ChatRef | None = None
        self._adapter: ChatAdapter | None = None
        self._label = ""
        self._message_id = ""
        self._shots = 0
        self._last_sent = 0.0
        self._pending_action = ""
        self._tab: Any = None
        self._active = False

    # -- lifecycle ---------------------------------------------------------
    def start(self, chat: ChatRef, adapter: ChatAdapter,
              *, task_label: str) -> None:
        """Send the opening frame. Never raises."""
        try:
            self._chat = chat
            self._adapter = adapter
            self._label = (task_label or "browsing").strip() or "browsing"
            self._active = True
            path = self._take_shot()
            caption = f"\U0001F50D {self._label}\u2026"
            if path is not None:
                res = adapter.send_media(
                    chat, MediaRef(path=str(path), kind="image"),
                    caption=caption)
            else:
                res = adapter.send(chat, caption)
            if res.ok:
                self._message_id = res.message_id
                self._shots += 1
                self._last_sent = self._clock()
        except Exception:  # noqa: BLE001 - live view never kills the task
            _log.debug("liveview.start failed", exc_info=True)

    def update(self, action: str) -> None:
        """Record a browser action; send a throttled screenshot update."""
        try:
            if not self._active:
                return
            action = (action or "").strip()
            if action:
                self._pending_action = action
            if self._shots >= self._max_shots:
                return  # capped: keep the latest action for the finish frame
            now = self._clock()
            if now - self._last_sent < self._interval:
                return  # throttled: action is coalesced, sent on next window
            path = self._take_shot()
            if path is None:
                return
            caption = self._caption()
            self._send_frame(path, caption)
        except Exception:  # noqa: BLE001
            _log.debug("liveview.update failed", exc_info=True)

    def finish(self, summary: str) -> None:
        """Final frame + summary, then detach. Never raises."""
        try:
            if not self._active:
                return
            path = self._take_shot()
            caption = f"\u2705 {self._label}\n{(summary or '').strip()}"
            if path is not None:
                self._send_frame(path, caption, force=True)
            elif self._adapter is not None and self._chat is not None:
                self._adapter.send(self._chat, caption)
        except Exception:  # noqa: BLE001
            _log.debug("liveview.finish failed", exc_info=True)
        finally:
            self.detach()

    def fail(self, error: str) -> None:
        """Honest failure frame, then detach. Never raises."""
        try:
            if not self._active:
                return
            err = (error or "unknown error").strip()
            caption = f"\u274C {self._label}\nfailed: {err}"
            path = self._take_shot()
            if path is not None:
                self._send_frame(path, caption, force=True)
            elif self._adapter is not None and self._chat is not None:
                self._adapter.send(self._chat, caption)
        except Exception:  # noqa: BLE001
            _log.debug("liveview.fail failed", exc_info=True)
        finally:
            self.detach()

    def detach(self) -> None:
        """Unhook from the tab. Never raises."""
        try:
            self._active = False
            tab, self._tab = self._tab, None
            # Bound methods never compare identical with `is`; compare the
            # underlying function + instance so we only clear our own hook.
            hook = getattr(tab, "on_action", None) if tab is not None else None
            if (hook is not None
                    and getattr(hook, "__self__", None) is self
                    and getattr(hook, "__func__", None) == self.update.__func__):
                tab.on_action = None
        except Exception:  # noqa: BLE001
            _log.debug("liveview.detach failed", exc_info=True)

    # -- internals ---------------------------------------------------------
    def _caption(self) -> str:
        base = f"\U0001F50D {self._label}"
        if self._pending_action:
            return f"{base}\n{self._pending_action}"
        return base + "\u2026"

    def _take_shot(self) -> Path | None:
        try:
            raw = self._shot()
        except Exception:  # noqa: BLE001 - shot failures are silent
            _log.debug("liveview shot failed", exc_info=True)
            return None
        if raw is None:
            return None
        p = Path(str(raw))
        return p if p.is_file() else None

    def _send_frame(self, path: Path, caption: str,
                    *, force: bool = False) -> None:
        adapter, chat = self._adapter, self._chat
        if adapter is None or chat is None:
            return
        media = MediaRef(path=str(path), kind="image")
        res: SendResult | None = None
        # Prefer in-place edit: one message, updated. Fall back to a new
        # photo when the adapter has no edit_media.
        if self._message_id:
            try:
                res = adapter.edit_media(chat, self._message_id, media,
                                         caption=caption)
            except Exception:  # noqa: BLE001
                _log.debug("liveview edit_media raised", exc_info=True)
                res = None
        if res is None or not res.ok:
            res = adapter.send_media(chat, media, caption=caption)
        if res.ok:
            if res.message_id:
                self._message_id = res.message_id
            self._shots += 1
            self._last_sent = self._clock()
            self._pending_action = ""

    # -- attach ------------------------------------------------------------
    @classmethod
    def attach(
        cls,
        tab: Any,
        chat: ChatRef,
        adapter: ChatAdapter,
        *,
        task_label: str,
        shot: Callable[[], Any] | None = None,
        profile: str | None = None,
        live_view: bool = True,
        **kwargs: Any,
    ) -> "LiveView | None":
        """Attach a live window to ``tab``'s action hook.

        ``live_view=False`` is the explicit opt-out (background tasks):
        returns ``None`` and leaves the tab untouched. The tab's
        ``on_action`` is set to the view's ``update``; :meth:`detach`
        (called by ``finish``/``fail``) restores it. Never raises.
        """
        try:
            if not live_view:
                return None
            if shot is None:
                shot = _tab_shot_fn(tab)
            view = cls(shot, profile=profile, **kwargs)
            view.start(chat, adapter, task_label=task_label)
            view._tab = tab
            tab.on_action = view.update
            return view
        except Exception:  # noqa: BLE001
            _log.debug("liveview.attach failed", exc_info=True)
            return None


def _tab_shot_fn(tab: Any) -> Callable[[], Any]:
    """Build a shot callable from a tab's ``screenshot()`` method.

    Handles both shapes: ``RenderedTab.screenshot()`` returning
    ``{"path": ...}`` and anything returning a plain path.
    """
    fn = getattr(tab, "screenshot", None)

    def _shot() -> Any:
        if not callable(fn):
            return None
        try:
            out = fn()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(out, dict):
            return out.get("path")
        return out

    return _shot
