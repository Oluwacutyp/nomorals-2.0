"""RuntimeMockScreensMixin: the /mock command in the live
event path.

The organ itself (``social/chat/mock_screens/``) is
transport-free; this mixin owns the live wiring: parse the
compact structured form (or a built-in demo), render
through the organ's service, and deliver the PNG as a
photo through the expression send path — success returns
"" because the picture WAS the reply, and the kind rides
expression.MEDIA_DELIVERED_KINDS.

Gate: /mock is owner-only (deliberately NOT in
PUBLIC_COMMANDS, so caller_may_run admits the owner
alone), and the handler repeats the check in the sibling
idiom — a mock screen is authored data about people, so
it is the owner's act, like deciding a proposal.

Media tokens: the compact form's ``attach:`` / ``last:``
references are resolved HERE, against what the live
surface actually holds — ``attach:`` binds the command
message's own media (``ChatMessage.media``, in order),
``last:image`` the newest file in the generated-images
folder the image-generation seam writes (``data/media/
generated``, resolved the same way it resolves it).  An
unresolvable token is an honest text answer, never a
screen with the media silently dropped.

Phase 2b demos (``/mock demo whatsapp-list`` and its
siblings) parse to the organ's list / status / call
screen classes and dispatch to their own service doors
below — the same delivery contract, the picture WAS the
reply.  Phase 3 demos (posts / profiles / stories /
comments / video pages) ride the same dispatch.
"""

from __future__ import annotations

from .failures import record_failure


class RuntimeMockScreensMixin:
    """Mock-screen commands for PartnerRuntime."""

    def _mock_screens_service(self):  # type: ignore[no-untyped-def]
        service = getattr(self, "_mock_screens_obj", None)
        if service is not None:
            return service
        from ...social.chat.mock_screens import MockScreenService

        service = MockScreenService(
            media_dir=self._expression_root() / "mock_media")
        self._mock_screens_obj = service
        return service

    def _mock_generated_image(self):  # type: ignore[no-untyped-def]
        """The newest generated image's path, or "" — the
        folder ``_image_generate`` saves to, resolved the
        same way (``data/media/generated``)."""
        try:
            from pathlib import Path

            root = Path(self.context.settings.resolve(
                "data/media/generated"))
            if not root.is_dir():
                return ""
            files = [p for p in root.iterdir()
                     if p.is_file() and p.suffix.lower() in (
                         ".png", ".jpg", ".jpeg", ".webp")]
            if not files:
                return ""
            newest = max(files, key=lambda p: p.stat().st_mtime)
            return str(newest)
        except Exception:  # noqa: BLE001 - no folder, no last
            return ""

    def _mock_resolve_media(self, screen, message):  # type: ignore[no-untyped-def]
        """Resolve the compact form's media tokens on a
        parsed screen to real paths.  Returns "" when every
        token resolved (or none was present), else the
        honest reply text."""
        refs = list(getattr(message, "media", None) or []) \
            if message is not None else []

        def _of_kind(kinds, prefix):
            return [r for r in refs
                    if getattr(r, "kind", "") in kinds
                    or str(getattr(r, "mime", "") or "")
                    .startswith(prefix)]

        images = _of_kind(("image", "photo"), "image/")
        voices = _of_kind(("voice", "audio"), "audio/")
        for msg in screen.messages:
            token = str(getattr(msg, "media", "") or "")
            if token == "attach:image":
                if not images:
                    return ("you marked a photo but no image "
                            "was attached to your message — "
                            "attach one, or use [photo <path>] "
                            "/ [photo last]")
                msg.media = str(images.pop(0).path)
            elif token == "attach:voice":
                if not voices:
                    return ("you marked a voice note but no "
                            "audio was attached to your "
                            "message — attach one, or use "
                            "[voice <path>]")
                msg.media = str(voices.pop(0).path)
            elif token == "last:image":
                path = self._mock_generated_image()
                if not path:
                    return ("no generated image yet — "
                            "/image <prompt> first, or attach "
                            "one and use [photo]")
                msg.media = path
        return ""

    def _control_mock(self, kind: str, tail: str,
                      message=None) -> str:  # type: ignore[no-untyped-def]
        """One /mock command: author a screen, render it,
        deliver it as a photo.  Text answers only for usage
        and honest failures."""
        if message is not None and not self._is_operator(message):
            return "that one's just for the owner."
        from ...social.chat.mock_screens.authoring import (
            USAGE,
            parse_mock_tail,
        )
        from ...social.chat.mock_screens.render import (
            MockScreensError,
        )

        tail = str(tail or "").strip()
        if not tail:
            return USAGE
        if message is None:
            return ("usage: /mock in a chat — the screen "
                    "needs somewhere to land.")
        try:
            screen, mode = parse_mock_tail(tail)
        except MockScreensError as exc:
            return str(exc)
        service = self._mock_screens_service()
        audience = self._makers_audience(message)
        if mode not in ("demo", "built"):
            # Phase 2b/3 screen classes (lists / status /
            # calls, then posts / profiles / stories /
            # comments / video pages): their own service
            # doors, the same delivery contract — a
            # picture WAS the reply
            door = {
                "demo-list": "render_list",
                "demo-status": "render_status",
                "demo-views": "render_status_views",
                "demo-call": "render_call",
                "demo-history": "render_call_history",
                "demo-post": "render_post",
                "demo-profile": "render_profile",
                "demo-story": "render_story",
                "demo-comment": "render_comment",
                "demo-video": "render_video",
            }[mode]
            if mode == "demo-status":
                title = screen.owner.name
            elif mode == "demo-views":
                title = "My Status"
            elif mode == "demo-call":
                title = screen.display_name()
            elif mode == "demo-history":
                title = screen.title_text
            elif mode == "demo-post":
                title = screen.author.name
            elif mode == "demo-profile":
                title = screen.identity.name
            elif mode == "demo-story":
                title = screen.owner.name
            elif mode == "demo-comment":
                title = str(screen.total) + " comments"
            elif mode == "demo-video":
                title = screen.title
            else:
                title = screen.title
            caption = ("🖼 " + screen.platform
                       + " mock screen — " + title)
            try:
                out = getattr(service, door)(
                    screen, audience=audience, caption=caption,
                    send=lambda ref, cap: self._expression_send(
                        message, ref, caption=cap))
            except Exception as exc:  # noqa: BLE001 - honest
                record_failure("mock_screens.render", exc)
                return ("the mock screen render failed: "
                        + str(exc)[:160])
            if out.get("sent"):
                return ""
            return str(out.get("reason") or "no mock screen "
                                            "right now")
        media_error = self._mock_resolve_media(screen, message)
        if media_error:
            return media_error
        caption = ("🖼 " + screen.platform + " mock screen — "
                   + screen.title)
        try:
            out = service.render(
                screen, audience=audience, caption=caption,
                send=lambda ref, cap: self._expression_send(
                    message, ref, caption=cap))
        except Exception as exc:  # noqa: BLE001 - honest line
            record_failure("mock_screens.render", exc)
            return ("the mock screen render failed: "
                    + str(exc)[:160])
        if out.get("sent"):
            return ""
        return str(out.get("reason") or "no mock screen "
                                        "right now")
