"""Doc 16 Pass 4 — ``_send_long_checked`` routes through the
output router, and the ``/read`` surface answers its actions.

The runtime seam is the migration point: short outputs keep the
exact legacy behaviour (byte-identical single/chunked sends),
long outputs arrive as one reader introduction whose controls —
driven here through the real handle_control dispatch, the shipped
callback path — recover the complete original text, and any
router-side failure falls back to the legacy chunk path.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatKind, ChatRef

PROSE = (
    "Weekly research digest\n\n"
    + ("Finding: the signal held across every session this week, "
       "with volume confirming the move and no sign of the usual "
       "exhaustion pattern near the highs. " * 12 + "\n\n")
    + ("Risk note: the invalidation sits below the prior swing "
       "low; a close through it ends the thesis and the plan "
       "reverts to observation only. " * 12 + "\n\n")
)
MEDIUM = PROSE * 4            # ~13k chars → reader route
LONG = PROSE * 14             # ~45k chars → reader+edition route
CODE = "def compute(x):\n    return x * 2  # exact bytes\n" * 220
CHAT_KEY = "telegram:42"
TOKEN_RE = re.compile(r"read go ([0-9a-f]{12})")
PAGE_RE = re.compile(r"page (\d+) of (\d+) —")


class _Gateway:
    """Records sends/files the way the wave-F1 harness does, with
    the send_file the document route needs."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[str, object]] = []
        self.files: list[tuple[str, bytes, str]] = []

    def send(self, platform, chat, text, **kwargs):
        self.sent.append((text, kwargs.get("buttons")))
        return SimpleNamespace(ok=self.ok,
                               error="" if self.ok else "down")

    def send_file(self, platform, chat, path, **kwargs):
        data = Path(path).read_bytes()
        self.files.append((kwargs.get("name") or "", data,
                           kwargs.get("caption") or ""))
        return SimpleNamespace(ok=self.ok,
                               error="" if self.ok else "down")

    def typing(self, *args, **kwargs):
        return None


def _page_no(rendered: str) -> tuple[int, int]:
    m = PAGE_RE.search(rendered)
    assert m is not None, rendered[-200:]
    return int(m.group(1)), int(m.group(2))


@pytest.fixture(scope="module")
def home():
    tmp = tempfile.TemporaryDirectory(prefix="nm-d16p4-")
    yield tmp.name
    tmp.cleanup()


def _runtime(home: str, gateway: _Gateway) -> PartnerRuntime:
    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "telegram",
        "partner.owner_chats": "42",
        "partner.owner_sender_ids": "42",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context, gateway=gateway)


def _chat(chat_id: str = "42") -> ChatRef:
    return ChatRef(platform="telegram", chat_id=chat_id,
                   kind=ChatKind.GROUP)


class TestSeam:
    def test_short_output_is_byte_identical(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        note = rt._send_long_checked("telegram", _chat("s1"), "hello")
        assert note == ""
        assert [text for text, _ in gw.sent] == ["hello"]
        assert gw.files == []

    def test_inline_intent_keeps_the_chunk_dump(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        note = rt._send_long_checked(
            "telegram", _chat("s2"), MEDIUM, intent="inline")
        assert note == ""
        assert len(gw.sent) > 1
        assert "".join(text for text, _ in gw.sent) == MEDIUM
        assert gw.files == []

    def test_long_output_arrives_as_intro_and_recovers(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        chat = _chat("s3")
        note = rt._send_long_checked("telegram", chat, MEDIUM,
                                     title="Weekly research digest")
        assert note == ""
        # ONE introduction — no chunk dump
        assert len(gw.sent) == 1 and gw.files == []
        intro = gw.sent[0][0]
        assert "nothing is cut" in intro
        # the intro's own controls page through the whole document,
        # driven through the real dispatch (the shipped callback
        # path: a tap re-enters as the /read command)
        page, total = _page_no(intro)
        assert page == 1 and total > 1
        token = TOKEN_RE.search(intro).group(1)
        last = intro
        for expected in range(2, total + 1):
            last = rt.handle_control(
                f"/read go {token} n", chat_key="telegram:s3")
            assert _page_no(last) == (expected, total)
            token = TOKEN_RE.search(last).group(1)
        assert MEDIUM.rstrip()[-150:] in last
        # and the session the send created holds the full text,
        # recoverable page by page with nothing lost
        from nomorals.social.chat.output_router import OutputRouter
        router = OutputRouter.for_context(rt.context)
        row = router.recent("chat:telegram:s3")[0]
        session = router.reader.session_for(
            "chat:telegram:s3", row["document_id"])
        resolver = router.resolver()
        # the dispatch loop left the session on its last page;
        # rewind, then walk it end to end
        view = router.reader.goto(session.session_id, resolver,
                                  page_index=0)
        bodies = [view.body]
        for _ in range(view.page_count - 1):
            view = router.reader.navigate(
                session.session_id, resolver, "next")
            bodies.append(view.body)
        assert "".join(bodies) == MEDIUM

    def test_read_get_txt_and_html_through_dispatch(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        chat = _chat("s4")
        assert rt._send_long_checked("telegram", chat, LONG) == ""
        token = TOKEN_RE.search(gw.sent[0][0]).group(1)
        reply = rt.handle_control(
            f"/read get {token} txt", chat_key="telegram:s4")
        assert reply.startswith("📄 sent above")
        name, data, _caption = gw.files[-1]
        assert name.endswith(".txt") and data == LONG.encode("utf-8")
        reply = rt.handle_control(
            f"/read get {token} html", chat_key="telegram:s4")
        assert reply.startswith("📄 sent above")
        name, data, _caption = gw.files[-1]
        assert name.endswith(".html")
        html = data.decode("utf-8")
        assert "Powered by peacethefirst" in html
        assert LONG[100:400] in html

    def test_read_list_and_open(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        chat = _chat("s5")
        assert rt._send_long_checked(
            "telegram", chat, MEDIUM,
            title="Weekly research digest") == ""
        board = rt.handle_control("/read", chat_key="telegram:s5")
        assert "Weekly research digest" in board
        assert "/read open 1" in board
        page = rt.handle_control("/read open 1",
                                 chat_key="telegram:s5")
        assert MEDIUM[:200] in page

    def test_read_go_unknown_token_is_honest(self, home):
        rt = _runtime(home, _Gateway())
        reply = rt.handle_control(
            "/read go 0123456789ab n", chat_key="telegram:s6")
        assert "expired" in reply

    def test_code_kind_lands_as_the_raw_document(self, home):
        gw = _Gateway()
        rt = _runtime(home, gw)
        note = rt._send_long_checked(
            "telegram", _chat("s7"), CODE, kind="code")
        assert note == ""
        assert gw.sent == []
        assert len(gw.files) == 1
        name, data, caption = gw.files[0]
        assert data == CODE.encode("utf-8")
        assert name.endswith(".txt")

    def test_total_failure_still_says_so(self, home):
        gw = _Gateway(ok=False)
        rt = _runtime(home, gw)
        note = rt._send_long_checked("telegram", _chat("s8"), MEDIUM)
        assert "delivery failed" in note
        assert "Nothing was delivered" in note


class TestFailOpen:
    """The pinned wave-F1 harness shape: a bare runtime (no
    context) whose gateway has no send_file.  The router cannot
    stand up, so the legacy chunk path carries the output."""

    def test_bare_runtime_chunks_as_before(self):
        class _OkGateway:
            def __init__(self):
                self.sent: list[str] = []

            def send(self, platform, chat, text):
                self.sent.append(text)
                return SimpleNamespace(ok=True, error="")

            def typing(self, platform, chat, seconds=0.0):
                return None

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.settings = SimpleNamespace(
            partner=SimpleNamespace(typing_in_groups=False))
        rt.brain = None
        gw = _OkGateway()
        rt.gateway = gw
        chat = SimpleNamespace(platform="telegram",
                               kind=ChatKind.GROUP, key="telegram:1")
        note = rt._send_long_checked(
            "telegram", chat, "hello " * 2000)
        assert note == ""
        assert len(gw.sent) > 1  # long text went out in chunks
