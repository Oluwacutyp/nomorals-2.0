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
import io
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from .animation import (
    RecordingError,
    TimelineSpec,
    build_call_recording,
    build_chat_recording,
    build_comment_recording,
    build_hold_recording,
    build_list_recording,
    build_profile_recording,
    build_status_recording,
    build_story_recording,
    build_video_recording,
    record,
)
from .render import MockScreensError, render_screen
from .render_screens import (
    render_call_history,
    render_call_screen,
    render_list_screen,
    render_status_screen,
    render_status_views,
)
from .render_social import (
    render_comment_screen,
    render_post_screen,
    render_profile_screen,
    render_story_screen,
    render_video_screen,
)
from .schema import (
    CallHistoryScreen,
    CallScreen,
    CommentScreen,
    ListScreen,
    PostScreen,
    ProfileScreen,
    Screen,
    StatusScreen,
    StatusViewsScreen,
    StoryScreen,
    VideoScreen,
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

    # ── Phase 3 screen classes ───────────────────────────

    def render_post(self, screen: PostScreen, **kw
                    ) -> dict[str, Any]:
        """Render a social post."""
        return self._render_with(
            render_post_screen, screen,
            label=screen.platform + " post",
            title=screen.author.name, **kw)

    def render_profile(self, screen: ProfileScreen, **kw
                       ) -> dict[str, Any]:
        """Render a profile page."""
        return self._render_with(
            render_profile_screen, screen,
            label=screen.platform + " profile",
            title=screen.identity.name, **kw)

    def render_story(self, screen: StoryScreen, **kw
                     ) -> dict[str, Any]:
        """Render an Instagram Story."""
        return self._render_with(
            render_story_screen, screen,
            label="instagram story",
            title=screen.owner.name, **kw)

    def render_comment(self, screen: CommentScreen, **kw
                       ) -> dict[str, Any]:
        """Render a TikTok comment sheet."""
        return self._render_with(
            render_comment_screen, screen,
            label="tiktok comments",
            title=str(screen.total) + " comments", **kw)

    def render_video(self, screen: VideoScreen, **kw
                     ) -> dict[str, Any]:
        """Render a YouTube video page."""
        return self._render_with(
            render_video_screen, screen,
            label="youtube video",
            title=screen.title, **kw)

    # ── Phase 4 recordings ───────────────────────────────

    def _record_with(self, recording, *, screen=None,
                     label: str, title: str,
                     audience: str = "", caption: str = "",
                     send=None) -> dict[str, Any]:
        """Encode one built recording and deliver the mp4
        through the same send seam stills use.  The file
        is cache-keyed on the screen + timeline spec, so
        re-recording the same screen reuses the encode —
        the key carries the whole screen, not just its
        shape: two chats with the same message count
        must never share a file."""
        out: dict[str, Any] = {
            "sent": False, "drawn": recording.drawn,
            "duration_s": recording.duration_s,
            "timeline": recording.timeline,
            "kind": recording.kind}
        if self.media_dir is None:
            return {**out, "reason":
                    "no media directory on this runtime"}
        try:
            key = hashlib.sha256(json.dumps(
                {"kind": recording.kind,
                 "spec": asdict(recording.spec),
                 "timeline": recording.timeline,
                 "screen": asdict(screen)
                 if screen is not None else None},
                sort_keys=True, default=str).encode(
                    "utf-8")).hexdigest()[:16]
        except Exception:  # noqa: BLE001 - unique key
            key = "recording-" + uuid.uuid4().hex[:16]
        self.media_dir.mkdir(parents=True, exist_ok=True)
        path = self.media_dir / (key + ".mp4")
        if path.exists() and path.stat().st_size > 0:
            # the same screen + timeline was already
            # encoded: reuse it, exactly like the stills
            # reuse their content-hash file
            result = {
                "path": str(path),
                "duration_s": recording.duration_s,
                "expected_s": recording.duration_s,
                "frames": int(round(
                    recording.duration_s
                    * recording.spec.fps)),
                "fps": recording.spec.fps,
                "size": recording.size,
                "kind": recording.kind}
        else:
            try:
                result = record(recording, path)
            except RecordingError as exc:
                return {**out, "reason": str(exc)}
        out.update(result)
        if send is None:
            return out
        from ..base import MediaRef

        ref = MediaRef(
            path=str(path), mime="video/mp4", kind="video",
            name=path.name, audience=audience)
        try:
            sent = send(ref, caption
                        or "🎬 " + label + " mock screen "
                        "recording — " + title)
        except Exception as exc:  # noqa: BLE001 - send failed
            return {**out,
                    "reason": "the send failed: " + str(exc)}
        if sent is not None and getattr(sent, "ok", True):
            out["sent"] = True
            return out
        out["reason"] = str(
            getattr(sent, "error", "")
            or "this surface couldn't send it")
        return out

    def _record(self, builder, screen, *, label: str,
                title: str, spec=None, avatars=None,
                media=None, audience: str = "",
                caption: str = "", send=None) -> dict[str, Any]:
        """Build a recording through ``builder`` and serve
        it; a build failure is the honest reason."""
        try:
            recording = builder(
                screen, spec or TimelineSpec(),
                avatars=avatars, media=media)
        except RecordingError as exc:
            return {"sent": False, "reason": str(exc)}
        return self._record_with(
            recording, screen=screen, label=label,
            title=title, audience=audience, caption=caption,
            send=send)

    def record_chat(self, screen: Screen, **kw
                    ) -> dict[str, Any]:
        """Record a chat screen as a video: messages
        reveal, typing dots, voice notes at true length."""
        return self._record(
            build_chat_recording, screen,
            label=screen.platform, title=screen.title, **kw)

    def record_list(self, screen: ListScreen, **kw
                    ) -> dict[str, Any]:
        """Record a list screen: rows arrive one by one."""
        return self._record(
            build_list_recording, screen,
            label=screen.platform, title=screen.title, **kw)

    def record_status(self, screen: StatusScreen, **kw
                      ) -> dict[str, Any]:
        """Record a WhatsApp Status: the segment fills to
        the user-set point."""
        return self._record(
            build_status_recording, screen,
            label="whatsapp status",
            title=screen.owner.name, **kw)

    def record_story(self, screen: StoryScreen, **kw
                     ) -> dict[str, Any]:
        """Record an Instagram Story: segments advance,
        each bar filling for its set duration."""
        return self._record(
            build_story_recording, screen,
            label="instagram story",
            title=screen.owner.name, **kw)

    def record_call(self, screen: CallScreen, **kw
                    ) -> dict[str, Any]:
        """Record a call screen: an ongoing timer
        advances, an incoming call holds ringing."""
        return self._record(
            build_call_recording, screen,
            label=screen.platform + " call",
            title=screen.display_name(), **kw)

    def record_profile(self, screen: ProfileScreen, **kw
                       ) -> dict[str, Any]:
        """Record a profile page scrolling through it."""
        return self._record(
            build_profile_recording, screen,
            label=screen.platform + " profile",
            title=screen.identity.name, **kw)

    def record_comment(self, screen: CommentScreen, **kw
                       ) -> dict[str, Any]:
        """Record a comment sheet scrolling through it."""
        return self._record(
            build_comment_recording, screen,
            label="tiktok comments",
            title=str(screen.total) + " comments", **kw)

    def record_video(self, screen: VideoScreen, **kw
                     ) -> dict[str, Any]:
        """Record a YouTube video page scrolling."""
        return self._record(
            build_video_recording, screen,
            label="youtube video",
            title=screen.title, **kw)

    def _record_hold(self, frame, *, label: str, title: str,
                     kind: str, drawn: dict, spec=None,
                     audience: str = "", caption: str = "",
                     send=None) -> dict[str, Any]:
        recording = build_hold_recording(
            frame, spec or TimelineSpec(), kind=kind,
            drawn=drawn)
        return self._record_with(
            recording, label=label, title=title,
            audience=audience, caption=caption, send=send)

    def record_post(self, screen: PostScreen, **kw
                    ) -> dict[str, Any]:
        """Record a social post: a hold on the card (a
        post carries no animation choreography of its
        own)."""
        try:
            data, drawn = render_post_screen(screen)
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc)}
        frame = Image.open(io.BytesIO(data)).convert("RGB")
        return self._record_hold(
            frame, label=screen.platform + " post",
            title=screen.author.name, kind="post",
            drawn=drawn, **kw)

    def record_status_views(self, screen: StatusViewsScreen,
                            **kw) -> dict[str, Any]:
        """Record the Status Views sheet: a hold on the
        sheet."""
        try:
            data, drawn = render_status_views(screen)
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc)}
        frame = Image.open(io.BytesIO(data)).convert("RGB")
        return self._record_hold(
            frame, label="whatsapp status views",
            title="My Status", kind="views", drawn=drawn, **kw)

    def record_call_history(self, screen: CallHistoryScreen,
                            **kw) -> dict[str, Any]:
        """Record the iPhone Call History: a hold on the
        Recents screen."""
        try:
            data, drawn = render_call_history(screen)
        except MockScreensError as exc:
            return {"sent": False, "reason": str(exc)}
        frame = Image.open(io.BytesIO(data)).convert("RGB")
        return self._record_hold(
            frame, label="iphone call history",
            title=screen.title_text, kind="history",
            drawn=drawn, **kw)

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
