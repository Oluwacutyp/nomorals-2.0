"""The mock-screen service: render + deliver, makers-style.

Every outcome is a plain dict; the service never raises —
a :class:`MockScreensError`'s message IS the user-fit
reason.  Delivery follows the makers contract: the PNG is
saved under the media dir, wrapped in a ``MediaRef`` and
handed to the caller's ``send`` — the runtime passes the
expression send seam, so audience classification (and the
brand mark it drives on public surfaces) behaves exactly
as it does for maker cards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .render import MockScreensError, render_screen
from .render_screens import (
    render_call_history,
    render_call_screen,
    render_list_screen,
    render_status_screen,
    render_status_views,
)
from .schema import (
    CallHistoryScreen,
    CallScreen,
    ListScreen,
    Screen,
    StatusScreen,
    StatusViewsScreen,
)

__all__ = ["MockScreenService", "MockScreensError"]


class MockScreenService:
    """Renders mock screens and delivers them as photos."""

    def __init__(self, *, media_dir: Path | str = "") -> None:
        self.media_dir = Path(media_dir) if media_dir else None

    def _save(self, data: bytes, ext: str) -> Path:
        if self.media_dir is None:
            raise MockScreensError(
                "no media directory on this runtime")
        self.media_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(data).hexdigest()[:16] + ext
        path = self.media_dir / name
        if not path.exists():
            path.write_bytes(data)
        return path

    def render(self, screen: Screen, *,
               avatars: dict | None = None,
               media: dict | None = None,
               audience: str = "",
               caption: str = "",
               send=None) -> dict[str, Any]:
        """Render one screen.  With ``send`` given, also
        deliver it as a photo and report ``sent``; without,
        the result carries the PNG bytes for the caller.
        ``media`` is the real-media hand-off (message id →
        bytes / path / PIL image / MediaRef) forwarded to
        the renderer."""
        try:
            data, drawn = render_screen(
                screen, avatars=avatars, media=media)
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc)}
        return self._serve(
            data, drawn, label=screen.platform,
            title=screen.title, audience=audience,
            caption=caption, send=send)

    # ── Phase 2b screen classes ──────────────────────────

    def _render_with(self, fn, screen, *, label: str,
                     title: str, avatars: dict | None = None,
                     media: dict | None = None,
                     audience: str = "", caption: str = "",
                     send=None) -> dict[str, Any]:
        """One Phase 2b render through its engine, then
        the shared delivery tail."""
        try:
            data, drawn = fn(screen, avatars=avatars,
                             media=media)
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc)}
        return self._serve(
            data, drawn, label=label, title=title,
            audience=audience, caption=caption, send=send)

    def render_list(self, screen: ListScreen, **kw
                    ) -> dict[str, Any]:
        """Render a list screen (chat inbox / call log /
        status list) and deliver it like a chat screen."""
        return self._render_with(
            render_list_screen, screen,
            label=screen.platform, title=screen.title, **kw)

    def render_status(self, screen: StatusScreen, **kw
                      ) -> dict[str, Any]:
        """Render a WhatsApp Status screen."""
        return self._render_with(
            render_status_screen, screen,
            label="whatsapp status", title=screen.owner.name,
            **kw)

    def render_status_views(self, screen: StatusViewsScreen,
                            **kw) -> dict[str, Any]:
        """Render a WhatsApp Status Views screen."""
        return self._render_with(
            render_status_views, screen,
            label="whatsapp status views", title="My Status",
            **kw)

    def render_call(self, screen: CallScreen, **kw
                    ) -> dict[str, Any]:
        """Render a call screen."""
        return self._render_with(
            render_call_screen, screen,
            label=screen.platform + " call",
            title=screen.display_name(), **kw)

    def render_call_history(self, screen: CallHistoryScreen,
                            **kw) -> dict[str, Any]:
        """Render the iPhone Call History screen."""
        return self._render_with(
            render_call_history, screen,
            label="iphone call history",
            title=screen.title_text, **kw)

    # ── the shared delivery tail ─────────────────────────

    def _serve(self, data: bytes, drawn: dict, *,
               label: str, title: str, audience: str = "",
               caption: str = "",
               send=None) -> dict[str, Any]:
        out: dict[str, Any] = {"sent": False, "drawn": drawn,
                               "data": data}
        if send is None:
            return out
        from ..base import MediaRef

        try:
            path = self._save(data, ".png")
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc),
                    "drawn": drawn}
        ref = MediaRef(
            path=str(path), mime="image/png", kind="image",
            name=path.name, audience=audience)
        try:
            result = send(ref, caption
                          or "🖼 " + label + " mock screen — "
                          + title)
        except Exception as exc:  # noqa: BLE001 - send failed
            return {"sent": False,
                    "reason": "the send failed: " + str(exc),
                    "drawn": drawn}
        if result is not None and getattr(result, "ok", True):
            out["sent"] = True
            return out
        out["reason"] = str(
            getattr(result, "error", "")
            or "this surface couldn't send it")
        return out
