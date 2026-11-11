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
comments / video pages) ride the same dispatch.  Phase 5
(``/mock demo bulk-*``) populates whole screens from one
seed + count through the same doors, and ``/mock data``
answers in chat text with the generated records.
"""

from __future__ import annotations

import re

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
        head = tail.split(None, 1)[0].lower()
        if head == "make":
            # Phase 6: natural-language authoring — a
            # description becomes a structured model before
            # it renders; questions and candidates come
            # back as text, never a guessed screen.
            return self._mock_make(tail[4:].strip(), message)
        if head == "project":
            # Phase 6: saved screen projects — a spec,
            # reopened and revised, never baked pixels.
            return self._mock_project(
                tail[7:].strip(), message)
        try:
            screen, mode = parse_mock_tail(tail)
        except MockScreensError as exc:
            return str(exc)
        service = self._mock_screens_service()
        audience = self._makers_audience(message)
        if mode == "data":
            # Phase 5: /mock data answers in chat text —
            # the export IS the reply (no picture to send).
            return str(screen)
        record = mode.startswith("record-")
        base = mode[len("record-"):] if record else mode
        if base not in ("demo", "built"):
            # Phase 2b/3 screen classes (lists / status /
            # calls, then posts / profiles / stories /
            # comments / video pages): their own service
            # doors, the same delivery contract — a
            # picture WAS the reply, a video WAS the reply
            if record:
                door = {
                    "demo-list": "record_list",
                    "demo-status": "record_status",
                    "demo-views": "record_status_views",
                    "demo-call": "record_call",
                    "demo-history": "record_call_history",
                    "demo-post": "record_post",
                    "demo-profile": "record_profile",
                    "demo-story": "record_story",
                    "demo-comment": "record_comment",
                    "demo-video": "record_video",
                }.get(base)
                if door is None:
                    return ("that screen can't be recorded "
                            "as a video yet.")
            else:
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
                }[base]
            if base == "demo-status":
                title = screen.owner.name
            elif base == "demo-views":
                title = "My Status"
            elif base == "demo-call":
                title = screen.display_name()
            elif base == "demo-history":
                title = screen.title_text
            elif base == "demo-post":
                title = screen.author.name
            elif base == "demo-profile":
                title = screen.identity.name
            elif base == "demo-story":
                title = screen.owner.name
            elif base == "demo-comment":
                title = str(screen.total) + " comments"
            elif base == "demo-video":
                title = screen.title
            else:
                title = screen.title
            if record:
                caption = ("🎬 " + screen.platform
                           + " mock screen recording — " + title)
            else:
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
        if record:
            # a chat recording: resolve the media tokens
            # the same way the still path does, then the
            # recording door (the picture's door is below)
            media_error = self._mock_resolve_media(
                screen, message)
            if media_error:
                return media_error
            caption = ("🎬 " + screen.platform
                       + " mock screen recording — "
                       + screen.title)
            try:
                out = service.record_chat(
                    screen, audience=audience, caption=caption,
                    send=lambda ref, cap: self._expression_send(
                        message, ref, caption=cap))
            except Exception as exc:  # noqa: BLE001 - honest
                record_failure("mock_screens.render", exc)
                return ("the mock screen recording failed: "
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

    # ── Phase 6: NL authoring + saved projects ───────────

    def _mock_project_store(self):  # type: ignore[no-untyped-def]
        store = getattr(self, "_mock_projects_obj", None)
        if store is not None:
            return store
        from ...social.chat.mock_screens.projects import (
            ProjectStore,
        )
        store = ProjectStore(
            getattr(getattr(self, "context", None), "db", None))
        self._mock_projects_obj = store
        return store

    def _mock_render_nl_screen(self, screen, message,
                               caption):  # type: ignore[no-untyped-def]
        """Render an NL/project chat screen: resolve the
        compact form's media tokens, refuse honestly when
        a concrete media path is gone, deliver the PNG —
        the picture WAS the reply."""
        from ...social.chat.mock_screens.projects import (
            missing_media,
        )
        media_error = self._mock_resolve_media(screen, message)
        if media_error:
            return media_error
        gone = missing_media(screen)
        if gone:
            mid, ref = gone[0]
            return ("that screen references media that is "
                    "gone — message " + mid + " wants " + ref
                    + " — re-attach it or revise the message")
        service = self._mock_screens_service()
        audience = self._makers_audience(message)
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

    def _mock_make(self, description,
                   message):  # type: ignore[no-untyped-def]
        """One /mock make: a description becomes a visible
        structured model, then a rendered screen.  A
        trailing ``save <name>`` also stores the project.
        Questions and candidates come back as text."""
        from ...social.chat.mock_screens.nl_authoring import (
            MAKE_USAGE,
            NLAuthoringError,
            parse_nl_description,
        )
        from ...social.chat.mock_screens.projects import (
            MockProject,
        )
        if not description:
            return MAKE_USAGE
        save_name = ""
        save_match = re.search(
            r"\s+save\s+([A-Za-z0-9][A-Za-z0-9 _\-]{0,63})\s*$",
            description)
        if save_match is not None:
            save_name = save_match.group(1)
            description = description[:save_match.start()].strip()
        try:
            result = parse_nl_description(description)
        except NLAuthoringError as exc:
            return str(exc)
        if result.candidates:
            return ("which platform — "
                    + " or ".join(result.candidates)
                    + "? say one of them and I will build it.")
        if result.questions:
            return "\n".join(result.questions)
        screen = result.screen
        caption = ("🖼 mock screen — " + screen.title + "\n"
                   + result.summary)
        reply = self._mock_render_nl_screen(
            screen, message, caption)
        if reply != "" or not save_name:
            return reply
        store = self._mock_project_store()
        if store.exists(save_name):
            return ("project " + save_name + " already exists "
                    "— /mock project edit " + save_name
                    + " <instruction> revises it, or delete "
                    "it first")
        if not store.save(MockProject(
                name=save_name, screen=screen)):
            return "could not save project " + save_name
        return ""

    def _mock_project(self, rest,
                      message):  # type: ignore[no-untyped-def]
        """One /mock project verb: list · show · edit ·
        delete a saved screen project."""
        from ...social.chat.mock_screens.nl_authoring import (
            PROJECT_USAGE,
            NLAuthoringError,
            model_summary,
            revise_screen,
        )
        from ...social.chat.mock_screens.projects import (
            MockProject,
        )
        if not rest:
            return PROJECT_USAGE
        words = rest.split(None, 1)
        verb = words[0].lower()
        tail = words[1].strip() if len(words) > 1 else ""
        store = self._mock_project_store()
        if verb == "list":
            names = store.list_names()
            if not names:
                return ("no saved mock projects yet — /mock "
                        "make ... save <name>")
            return "mock projects:\n" + "\n".join(
                "· " + n for n in names)
        if verb == "show":
            if not tail:
                return "usage: /mock project show <name>"
            project = store.load(tail)
            if project is None:
                return "no mock project called " + tail
            caption = ("🖼 mock screen — "
                       + project.screen.title
                       + " · project " + project.name
                       + " · revision " + str(project.revision)
                       + "\n"
                       + model_summary(project.screen))
            return self._mock_render_nl_screen(
                project.screen, message, caption)
        if verb == "edit":
            parts = tail.split(None, 1)
            if len(parts) < 2:
                return ("usage: /mock project edit <name> "
                        "<instruction>")
            name, instruction = parts
            project = store.load(name)
            if project is None:
                return "no mock project called " + name
            try:
                new_screen, change = revise_screen(
                    project.screen, instruction)
            except NLAuthoringError as exc:
                return str(exc)
            revised = MockProject(
                name=name, screen=new_screen,
                revision=project.revision + 1)
            if not store.save(
                    revised,
                    expect_revision=project.revision):
                return ("that project changed under you — "
                        "open it again and retry the edit")
            caption = ("🖼 mock screen — " + new_screen.title
                       + " · project " + name
                       + " · revision " + str(revised.revision)
                       + " · " + change
                       + "\n" + model_summary(new_screen))
            return self._mock_render_nl_screen(
                new_screen, message, caption)
        if verb == "delete":
            if not tail:
                return "usage: /mock project delete <name>"
            if store.delete(tail):
                return "deleted mock project " + tail
            return "no mock project called " + tail
        return PROJECT_USAGE
