"""Chat surfaces for the artist + film + effects organs (lane 95).

Failing-first: ``/filmcraft``, ``/effect`` and ``/artist`` are not
registered commands on base — ``CONTROL_COMMANDS`` has no such kinds,
the runtime has no dispatch branch, and the NL parsers do not exist.
(``/film`` stays the VOD film study — it is taken, live, and this lane
does not touch it; the editor verb is ``/filmcraft``.)

The verbs are thin dispatches to the tools the brain already uses
(``filmcraft`` / ``effectcraft`` / ``artist``) — no organ logic is
duplicated here, so a command's output is the tool's output.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    help_text,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.tools.artist import parse_artist_request
from nomorals.tools.filmcraft import (
    parse_effect_request,
    parse_filmcraft_request,
)

OWNER_ID = "5478650254"
STRANGER_ID = "999"

NEW_KINDS = ("filmcraft", "effect", "artist")


def _probe(path: Path) -> dict:
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None

    def run(self, handler) -> None:
        self._handler = handler

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons=None, **kwargs) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _msg(platform: str, text: str, *, sender: str = OWNER_ID) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=sender,
                     kind=ChatKind.DM, peer="owner" if sender == OWNER_ID
                     else "stranger"),
        incoming=True,
        text=text,
        sender="owner" if sender == OWNER_ID else "stranger",
        sender_id=sender,
        meta={},
    )


def _make_runtime():
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-filmsurface-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False)
    adapter = FakeAdapter("telegram")
    gateway = ChatGateway({"telegram": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


class RegistryTests(unittest.TestCase):
    def test_new_kinds_registered_with_help(self) -> None:
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, CONTROL_COMMANDS)
                page = COMMAND_DETAILS.get(kind)
                self.assertIsNotNone(page, f"/{kind} has no help page")
                for field in ("what", "usage", "example"):
                    self.assertTrue(str(page.get(field, "")).strip(),
                                    f"/{kind} DETAILS.{field} is empty")
                self.assertTrue(str(LIST_ONELINERS.get(kind, "")).strip(),
                                f"/{kind} has no oneliner")
                groups = [k for _, ks in LIST_GROUPS for k in ks]
                self.assertEqual(groups.count(kind), 1,
                                 f"/{kind} must sit in exactly one "
                                 "help group")
                cmd = parse_control(f"/{kind}")
                self.assertIsNotNone(cmd)
                self.assertEqual(cmd.kind, kind)

    def test_new_kinds_owner_only_public_count_unchanged(self) -> None:
        # Owner-only by absence from PUBLIC_COMMANDS — the 291 pin
        # does NOT move for owner-only verbs (the brief's 279 -> 282
        # was stale: public commands are what the pin counts).
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertNotIn(kind, PUBLIC_COMMANDS)
                self.assertFalse(caller_may_run(kind, False))
                self.assertTrue(caller_may_run(kind, True))
        self.assertEqual(len(PUBLIC_COMMANDS), 291)

    def test_new_kinds_in_help_and_catalog(self) -> None:
        text = help_text()
        catalog = list_catalog()
        for kind in NEW_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(f"/{kind}", text)
                self.assertIn(f"/{kind}", catalog)
                page = detailed_help(kind, public_only=True)
                self.assertNotIn(f"/{kind}", page,
                                 f"/{kind} must not leak onto the "
                                 "public help page")

    def test_film_still_vod_study(self) -> None:
        # /film is taken, live: VOD film study, not the editor.
        cmd = parse_control("/film analyze match.mp4 chess")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "film")
        self.assertIn("VOD film study",
                      COMMAND_DETAILS["film"]["what"])
        self.assertIn("filmcraft", CONTROL_COMMANDS)


class ParseFilmcraftTests(unittest.TestCase):
    def test_show_render_list(self) -> None:
        known = ["demo", "launch cut"]
        self.assertEqual(
            parse_filmcraft_request("show me the film project demo",
                                    known),
            ("show", "demo"))
        self.assertEqual(
            parse_filmcraft_request("render the film demo", known),
            ("render", "demo"))
        self.assertEqual(
            parse_filmcraft_request("export the film launch cut", known),
            ("render", "launch cut"))
        self.assertEqual(
            parse_filmcraft_request("list my film projects", known),
            ("list", ""))

    def test_no_steal(self) -> None:
        known = ["demo"]
        for text in (
                "edit this video",
                "analyze this match video for mistakes",
                "make a video of my cat",
                "what's the weather",
                "show demo",
                "render the demo",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_filmcraft_request(text, known))


class ParseEffectTests(unittest.TestCase):
    def test_show_render_effects_list(self) -> None:
        known = ["intro"]
        self.assertEqual(
            parse_effect_request("show the motion composition intro",
                                 known),
            ("show", "intro"))
        self.assertEqual(
            parse_effect_request("render the composition intro", known),
            ("render", "intro"))
        self.assertEqual(
            parse_effect_request("what effects can you use", known),
            ("effects", ""))
        self.assertEqual(
            parse_effect_request("list my compositions", known),
            ("list", ""))

    def test_no_steal(self) -> None:
        known = ["intro"]
        for text in (
                "that movie had great effects",
                "the sound effect was loud",
                "show intro",
                "what's the weather",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_effect_request(text, known))


class ParseArtistTests(unittest.TestCase):
    def test_show_compose_video_list(self) -> None:
        known = [("abc123", "Rooftop at dusk"), ("def456", "Market")]
        self.assertEqual(
            parse_artist_request("show the artist scene Rooftop at dusk",
                                 known),
            ("show", "abc123"))
        self.assertEqual(
            parse_artist_request("compose the scene Rooftop at dusk",
                                 known),
            ("compose", "abc123"))
        self.assertEqual(
            parse_artist_request("make a video of the Rooftop at dusk "
                                 "scene",
                                 known),
            ("video", "abc123"))
        self.assertEqual(
            parse_artist_request("list my artist scenes", known),
            ("list", ""))

    def test_no_steal(self) -> None:
        known = [("abc123", "Rooftop at dusk")]
        for text in (
                "the artist painted a scene",
                "stage a scene",
                "show me Rooftop at dusk",
                "what's the weather",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_artist_request(text, known))


class VerbRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        (self.runtime, self.adapter, self.context,
         self._tmp) = _make_runtime()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):  # noqa: BLE001 - teardown
            self.runtime.stop()
        self.context.close()
        self._tmp.cleanup()

    def _drive(self, text: str, *, sender: str = OWNER_ID) -> str:
        before = len(self.adapter.sent)
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=AssertionError(
                                   "brain must not see a slash verb")):
            self.runtime._process(_msg("telegram", text, sender=sender))
        self.assertGreater(len(self.adapter.sent), before,
                           f"{text!r} produced no reply")
        return self.adapter.sent[-1]

    def test_bare_verbs_list_and_are_not_empty(self) -> None:
        # Tour/heal treat an empty reply as BROKEN — a bare verb lists.
        for verb in ("/filmcraft", "/effect", "/artist"):
            with self.subTest(verb=verb):
                reply = self._drive(verb)
                self.assertTrue(reply.strip(), f"{verb} replied empty")

    def test_unknown_subverb_returns_usage(self) -> None:
        for verb in ("/filmcraft frobnicate", "/effect frobnicate",
                     "/artist frobnicate"):
            with self.subTest(verb=verb):
                reply = self._drive(verb)
                self.assertIn("usage:", reply)

    def test_non_owner_refused(self) -> None:
        # The gate lives in the control dispatch, not in _process
        # (a stranger's slash text never reaches the dispatch — it
        # falls through to the brain, as for every owner-only verb).
        for verb in ("/filmcraft", "/effect", "/artist"):
            with self.subTest(verb=verb):
                reply = self.runtime.handle_control(
                    verb, "telegram:999",
                    message=_msg("telegram", verb, sender=STRANGER_ID))
                self.assertEqual(reply, "that one's just for the owner.")

    def test_filmcraft_end_to_end_through_verb(self) -> None:
        reply = self._drive("/filmcraft create demo 160 90 12")
        self.assertIn("film project created: demo", reply)
        reply = self._drive(
            "/filmcraft clip demo V1 #336699 0 1 kind solid")
        self.assertIn("clip c1 added to demo track V1", reply)
        out = self.tmp / "verb-film.mp4"
        reply = self._drive(f"/filmcraft render demo {out}")
        self.assertIn("render", reply.lower())
        self.assertTrue(out.exists(), reply)
        self.assertGreater(out.stat().st_size, 1000)
        info = _probe(out)
        self.assertGreater(float(info["format"]["duration"]), 0.5)
        reply = self._drive("/filmcraft show demo")
        self.assertIn("demo", reply)

    def test_effect_end_to_end_through_verb(self) -> None:
        reply = self._drive("/effect create intro 160 90 12 1")
        self.assertIn("composition created: intro", reply)
        reply = self._drive("/effect layer intro solid color [30,120,200]")
        self.assertIn("layer added to intro", reply)
        out = self.tmp / "verb-comp.mp4"
        reply = self._drive(f"/effect render intro {out}")
        self.assertIn("render", reply.lower())
        self.assertTrue(out.exists(), reply)
        self.assertGreater(out.stat().st_size, 1000)
        info = _probe(out)
        self.assertGreater(float(info["format"]["duration"]), 0.5)
        reply = self._drive("/effect effects")
        self.assertIn("gaussian_blur", reply)

    def test_artist_end_to_end_through_verb(self) -> None:
        reply = self._drive("/artist create Rooftop at dusk")
        self.assertIn("Rooftop", reply)
        match = re.search(r"scene created: (\S+)", reply)
        self.assertIsNotNone(match, reply)
        scene_id = match.group(1)
        reply = self._drive(f"/artist stage {scene_id} hero")
        self.assertIn("staged hero in scene", reply)
        reply = self._drive(f"/artist compose {scene_id}")
        self.assertIn("staged", reply.lower())
        pngs = list((self.tmp / "artist" / "renders").glob("*.png"))
        self.assertTrue(pngs, reply)
        self.assertGreater(pngs[0].stat().st_size, 100)
        reply = self._drive(f"/artist show {scene_id}")
        self.assertIn("Rooftop", reply)

    def test_nl_hook_reaches_filmcraft(self) -> None:
        self._drive("/filmcraft create demo 160 90 12")
        # A hooked phrase must produce a FRESH send — reading
        # sent[-1] without checking freshness would pass on the
        # stale create reply even if the hook never fired.
        before = len(self.adapter.sent)
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=AssertionError(
                                   "brain must not see a hooked phrase")):
            self.runtime._process(
                _msg("telegram", "show me the film project demo"))
        self.assertGreater(len(self.adapter.sent), before,
                           "hooked phrase produced no reply")
        self.assertIn("film demo —", self.adapter.sent[-1])
        before = len(self.adapter.sent)
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=AssertionError(
                                   "brain must not see a hooked phrase")):
            self.runtime._process(
                _msg("telegram", "list my film projects"))
        self.assertGreater(len(self.adapter.sent), before,
                           "hooked list phrase produced no reply")
        self.assertIn("film projects:", self.adapter.sent[-1])

    def test_nl_hook_reaches_effect_and_artist(self) -> None:
        self._drive("/effect create intro 160 90 12 1")
        self._drive("/artist create Rooftop at dusk")
        for text, needle in (
                ("show the motion composition intro",
                 "composition intro —"),
                ("show the artist scene Rooftop at dusk",
                 "Rooftop at dusk")):
            with self.subTest(text=text):
                before = len(self.adapter.sent)
                with mock.patch.object(
                        self.runtime.brain, "handle_message",
                        side_effect=AssertionError(
                            "brain must not see a hooked phrase")):
                    self.runtime._process(_msg("telegram", text))
                self.assertGreater(len(self.adapter.sent), before,
                                   f"{text!r} produced no reply")
                self.assertIn(needle, self.adapter.sent[-1])

    def test_nl_hook_ignores_ordinary_chat(self) -> None:
        self._drive("/filmcraft create demo 160 90 12")
        before = len(self.adapter.sent)
        with mock.patch.object(self.runtime.brain, "handle_message",
                               side_effect=AssertionError(
                                   "brain must see ordinary chat")):
            self.runtime._process(
                _msg("telegram", "what's the weather like today"))
        self.assertEqual(len(self.adapter.sent), before,
                         "hook stole ordinary chat")


if __name__ == "__main__":
    unittest.main()
