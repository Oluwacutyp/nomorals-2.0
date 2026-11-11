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
            screen, _mode = parse_mock_tail(tail)
        except MockScreensError as exc:
            return str(exc)
        service = self._mock_screens_service()
        audience = self._makers_audience(message)
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
