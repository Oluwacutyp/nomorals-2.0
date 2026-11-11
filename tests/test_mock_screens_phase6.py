"""Mock screens Phase 6 (FakeDetail port, census §30):
natural-language authoring and saved screen projects.

- a description becomes a VISIBLE structured model before
  it renders (platform, people with stable ids, numbered
  messages, style, defaults named);
- missing facts are asked for, never invented; two
  platforms named = candidates, never a guess;
- follow-up revision touches ONLY the addressed element
  (every other message's dict is byte-identical);
- a project is a saved SPEC (never baked pixels): reopen
  it, revise one message, re-render it — and a project
  whose media is gone fails honestly, naming the message
  and the path.

Network-free: renders are local Pillow work, the delivery
seam is a recording stub — the phase-2b suite's idiom.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.social.chat.mock_screens import (
    MockProject,
    NLAuthoringError,
    NLParseResult,
    ProjectError,
    ProjectStore,
    model_summary,
    parse_nl_description,
    revise_screen,
)
from nomorals.social.chat.mock_screens.authoring import (
    demo_screen,
)
from nomorals.social.chat.mock_screens.projects import (
    missing_media,
    valid_project_name,
)
from nomorals.social.chat.mock_screens.render import (
    render_screen,
)
from nomorals.storage.db import Database


def _png_bytes(color=(30, 120, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


# ── NL parsing: exact specs ──────────────────────────────

def test_parse_whatsapp_chat_with_voice_note_and_dark():
    result = parse_nl_description(
        'a WhatsApp chat with Ada, me: "hey are you coming?" '
        ';; Ada: "on my way" ;; me: "a voice note", dark mode')
    assert result.screen is not None
    assert result.questions == [] and result.candidates == []
    screen = result.screen
    assert screen.platform == "whatsapp"
    assert screen.title == "Ada"
    assert screen.dark is True
    assert [p.id for p in screen.participants] == ["me", "ada"]
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    assert [m.id for m in content] == ["n1", "n2", "n3"]
    assert content[0].text == "hey are you coming?"
    assert content[0].sender == "me"
    assert content[0].ticks == "read"
    assert content[1].text == "on my way"
    assert content[1].sender == "ada"
    assert content[2].kind == "voice"
    assert content[2].media == "attach:voice"
    assert content[2].text == ""
    assert screen.messages[0].kind == "divider"
    assert "times from 9:41" in " ".join(result.defaults)
    assert "Ada" in result.summary and "[n2]" in result.summary


def test_parse_last_one_clause_rekinds_empty_message():
    result = parse_nl_description(
        'a WhatsApp chat with Ada, me: "hey" ;; Ada: "" — '
        'last one is a voice note')
    screen = result.screen
    assert screen is not None
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    assert content[1].kind == "voice"
    assert content[1].media == "attach:voice"


def test_parse_iphone_group_with_reactions_and_photo():
    result = parse_nl_description(
        'a telegram group chat with Ada and Ben iphone, '
        'Ada: "team?" ;; Ben: "yes [react 👍]" ;; '
        'Ada: "a photo"')
    screen = result.screen
    assert screen is not None
    assert screen.platform == "telegram"
    assert screen.group is True
    assert screen.chrome.style == "iphone"
    assert screen.title == "Ada, Ben"
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    assert content[1].reactions == ["👍"]
    assert content[2].kind == "image"
    assert content[2].media == "attach:image"


def test_parse_photo_last_token_and_light_mode():
    result = parse_nl_description(
        'make an instagram chat with Ada, me: "[photo last] '
        'look at this" ;; Ada: "wow", light mode')
    screen = result.screen
    assert screen is not None
    assert screen.platform == "instagram"
    assert screen.dark is False
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    assert content[0].kind == "image"
    assert content[0].media == "last:image"
    assert content[0].caption == "look at this"


def test_parse_says_form_and_ticks():
    result = parse_nl_description(
        'a whatsapp chat with Ada, Ada says "hi" ;; '
        'me says "hello" [sent]')
    screen = result.screen
    assert screen is not None
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    assert content[0].text == "hi"
    assert content[1].text == "hello"
    assert content[1].ticks == "sent"


# ── NL parsing: ask, don't invent; never guess ──────────

def test_parse_two_platforms_is_candidates_not_a_guess():
    result = parse_nl_description(
        'a WhatsApp or Telegram chat with Ada, Ada: "hi"')
    assert result.screen is None
    assert result.candidates == ["whatsapp", "telegram"]


def test_parse_no_platform_asks_with_the_families():
    result = parse_nl_description('a chat with Ada, Ada: "hi"')
    assert result.screen is None
    assert result.questions
    assert "whatsapp" in result.questions[0]


def test_parse_no_messages_asks_instead_of_inventing():
    result = parse_nl_description(
        'a WhatsApp chat with Ada, three messages, dark mode')
    assert result.screen is None
    assert any("what do they say" in q
               for q in result.questions)


def test_parse_no_one_named_asks_who():
    result = parse_nl_description('a whatsapp chat')
    assert result.screen is None
    assert any("who is the chat with" in q
               for q in result.questions)


def test_parse_claimed_count_mismatch_is_a_question():
    result = parse_nl_description(
        'a whatsapp chat with Ada, three messages, '
        'Ada: "one" ;; Ada: "two"')
    assert result.screen is None
    assert any("3 messages but I have 2" in q
               for q in result.questions)


def test_parse_empty_is_a_question():
    result = parse_nl_description("  ")
    assert result.screen is None
    assert result.questions


def test_parse_voice_note_with_words_is_an_error():
    with pytest.raises(NLAuthoringError):
        parse_nl_description(
            'a whatsapp chat with Ada, me: "hello" — last one '
            'is a voice note')


def test_result_type_defaults():
    result = NLParseResult()
    assert result.screen is None
    assert result.questions == []
    assert result.candidates == []
    assert result.defaults == []


# ── follow-up revision: only the addressed element ──────

def _ada_screen():
    result = parse_nl_description(
        'a WhatsApp chat with Ada, me: "hey" ;; '
        'Ada: "on my way" ;; me: "see you"')
    assert result.screen is not None
    return result.screen


def _content_dicts(screen):
    return [m.to_dict() for m in screen.messages
            if m.kind not in ("divider", "system")]


def test_revise_change_message_leaves_others_identical():
    screen = _ada_screen()
    before = _content_dicts(screen)
    new, change = revise_screen(
        screen, 'change message 2 to "almost there"')
    assert "almost there" in change
    after = _content_dicts(new)
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1]["text"] == "almost there"
    assert after[1]["id"] == "n2"
    assert after[1]["time"] == before[1]["time"]


def test_revise_add_message_gets_next_id_and_later_time():
    screen = _ada_screen()
    new, change = revise_screen(
        screen, 'add Ada: "five more minutes"')
    assert "n4" in change
    content = [m for m in new.messages
               if m.kind not in ("divider", "system")]
    assert len(content) == 4
    assert content[3].id == "n4"
    assert content[3].text == "five more minutes"
    assert content[3].sender == "ada"
    assert content[3].time == "9:49 AM"
    assert [m.id for m in content[:3]] == ["n1", "n2", "n3"]


def test_revise_delete_message_keeps_ids_stable():
    screen = _ada_screen()
    new, change = revise_screen(screen, "delete message 1")
    assert "n1" in change
    content = [m for m in new.messages
               if m.kind not in ("divider", "system")]
    assert [m.id for m in content] == ["n2", "n3"]
    assert content[0].text == "on my way"


def test_revise_rename_updates_title_and_participant():
    screen = _ada_screen()
    new, change = revise_screen(
        screen, "rename Ada to Adaeze")
    assert "Adaeze" in change
    assert new.title == "Adaeze"
    assert new.participant("ada").name == "Adaeze"
    assert [m.text for m in new.messages
            if m.kind == "text"] == ["hey", "on my way",
                                     "see you"]


def test_revise_message_to_photo_and_reaction():
    screen = _ada_screen()
    new, _ = revise_screen(screen, "message 2 is a photo")
    content = [m for m in new.messages
               if m.kind not in ("divider", "system")]
    assert content[1].kind == "image"
    assert content[1].caption == "on my way"
    assert content[1].media == "attach:image"
    new, _ = revise_screen(new, "react 😂 to message 1")
    content = [m for m in new.messages
               if m.kind not in ("divider", "system")]
    assert content[0].reactions == ["😂"]


def test_revise_style_and_title():
    screen = _ada_screen()
    new, _ = revise_screen(screen, "make it dark")
    assert new.dark is True
    new, _ = revise_screen(new, 'title "Weekend plans"')
    assert new.title == "Weekend plans"
    new, _ = revise_screen(new, "iphone")
    assert new.chrome.style == "iphone"


def test_revise_unknown_instruction_and_missing_message():
    screen = _ada_screen()
    with pytest.raises(NLAuthoringError):
        revise_screen(screen, "make it sparkle")
    with pytest.raises(NLAuthoringError):
        revise_screen(screen, 'change message 9 to "nope"')
    with pytest.raises(NLAuthoringError):
        revise_screen(screen, "rename Zara to Zed")


def test_revise_never_mutates_the_input_screen():
    screen = _ada_screen()
    before = screen.to_dict()
    revise_screen(screen, 'change message 1 to "changed"')
    assert screen.to_dict() == before


def test_model_summary_numbers_messages_for_revision():
    screen = _ada_screen()
    summary = model_summary(screen)
    assert "1. [n1] you: hey" in summary
    assert "2. [n2] Ada: on my way" in summary
    assert "whatsapp chat — Ada" in summary


# ── projects: the store ──────────────────────────────────

def _project(name="ada-chat"):
    result = parse_nl_description(
        'a WhatsApp chat with Ada, me: "hey" ;; '
        'Ada: "on my way"')
    return MockProject(name=name, screen=result.screen)


def test_project_round_trip_in_memory():
    store = ProjectStore()
    project = _project()
    assert store.save(project) is True
    loaded = store.load("ada-chat")
    assert loaded is not None
    assert loaded.screen.to_dict() == project.screen.to_dict()
    assert loaded.revision == 1
    assert loaded.name == "ada-chat"


def test_project_list_delete_and_exists():
    store = ProjectStore()
    store.save(_project("one"))
    store.save(_project("two"))
    assert store.exists("one") is True
    assert set(store.list_names()) == {"one", "two"}
    assert store.delete("one") is True
    assert store.exists("one") is False
    assert store.delete("one") is False
    assert store.load("missing") is None


def test_project_cas_refuses_a_stale_revision():
    store = ProjectStore()
    project = _project()
    assert store.save(project) is True
    stale = _project()
    stale.revision = 2
    assert store.save(stale, expect_revision=99) is False
    assert store.load("ada-chat").revision == 1
    assert store.save(stale, expect_revision=1) is True
    assert store.load("ada-chat").revision == 2


def test_project_name_rule():
    assert valid_project_name("ada-chat 2") is True
    assert valid_project_name("") is False
    assert valid_project_name("../escape") is False
    assert valid_project_name("a/b") is False
    store = ProjectStore()
    bad = _project("../escape")
    assert store.save(bad) is False


def test_project_payload_carries_the_mock_marker():
    project = _project()
    assert project.to_dict()["mock_screen_project"] is True
    with pytest.raises(ProjectError):
        MockProject.from_dict({"name": "x", "screen": {}})


def test_project_round_trip_on_sqlite(tmp_path):
    db = Database(str(tmp_path / "ctx.db"))
    store = ProjectStore(db)
    assert store.save(_project()) is True
    again = ProjectStore(db)
    loaded = again.load("ada-chat")
    assert loaded is not None
    assert loaded.screen.title == "Ada"
    assert again.list_names() == ["ada-chat"]
    assert again.delete("ada-chat") is True
    assert again.load("ada-chat") is None


def test_missing_media_names_the_message_and_path(tmp_path):
    screen = _ada_screen()
    content = [m for m in screen.messages
               if m.kind not in ("divider", "system")]
    content[1].kind = "image"
    content[1].media = str(tmp_path / "gone.png")
    assert missing_media(screen) == [
        ("n2", str(tmp_path / "gone.png"))]
    content[1].media = "attach:image"
    assert missing_media(screen) == []
    content[1].media = "last:image"
    assert missing_media(screen) == []
    content[1].media = ""
    assert missing_media(screen) == []
    real = tmp_path / "here.png"
    real.write_bytes(_png_bytes())
    content[1].media = str(real)
    assert missing_media(screen) == []


# ── runtime: /mock make + /mock project ─────────────────

class _Runtime(RuntimeMockScreensMixin):
    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _Send()
        self.context = SimpleNamespace(db=None)

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


def test_command_make_renders_with_the_model_in_caption(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock",
        'make a WhatsApp chat with Ada, me: "hey" ;; '
        'Ada: "on my way"',
        _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()
    assert "mock screen" in caption
    assert "whatsapp chat — Ada" in caption
    assert "[n1]" in caption and "[n2]" in caption


def test_command_make_ambiguity_and_questions_are_text(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock",
        'make a WhatsApp or Telegram chat with Ada, '
        'Ada: "hi"',
        _message())
    assert "whatsapp" in reply and "telegram" in reply
    assert runtime.send.calls == []
    reply = runtime._control_mock(
        "mock", "make a WhatsApp chat with Ada",
        _message())
    assert "what do they say" in reply
    assert runtime.send.calls == []
    reply = runtime._control_mock("mock", "make", _message())
    assert reply.startswith("usage: /mock make")
    reply = runtime._control_mock("mock", "project", _message())
    assert reply.startswith("usage: /mock project")


def test_command_make_save_then_show_list_edit_delete(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock",
        'make a WhatsApp chat with Ada, me: "hey" ;; '
        'Ada: "on my way" save ada-chat',
        _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    reply = runtime._control_mock(
        "mock", "project list", _message())
    assert "ada-chat" in reply
    runtime.send.calls.clear()
    reply = runtime._control_mock(
        "mock", "project show ada-chat", _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    assert "revision 1" in runtime.send.calls[0][1]
    runtime.send.calls.clear()
    reply = runtime._control_mock(
        "mock",
        'project edit ada-chat change message 2 to '
        '"almost there"',
        _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    assert "almost there" in runtime.send.calls[0][1]
    assert "revision 2" in runtime.send.calls[0][1]
    reply = runtime._control_mock(
        "mock", "project delete ada-chat", _message())
    assert reply == "deleted mock project ada-chat"
    reply = runtime._control_mock(
        "mock", "project show ada-chat", _message())
    assert reply == "no mock project called ada-chat"
    reply = runtime._control_mock(
        "mock", "project delete ada-chat", _message())
    assert reply == "no mock project called ada-chat"


def test_command_make_save_refuses_an_existing_name(tmp_path):
    runtime = _Runtime(tmp_path)
    tail = ('make a WhatsApp chat with Ada, me: "hey" ;; '
            'Ada: "hi" save ada-chat')
    assert runtime._control_mock("mock", tail, _message()) == ""
    reply = runtime._control_mock("mock", tail, _message())
    assert "already exists" in reply


def test_command_project_stays_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "make a WhatsApp chat with Ada, Ada: \"hi\"",
        _message())
    assert reply == "that one's just for the owner."
    reply = runtime._control_mock(
        "mock", "project list", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_command_missing_media_fails_honestly(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock",
        'make a WhatsApp chat with Ada, me: "[photo '
        + str(tmp_path / "gone.png") + '] look"',
        _message())
    assert "gone" in reply
    assert "n1" in reply
    assert runtime.send.calls == []


# ── regression: existing demos byte-identical ────────────

def test_existing_demos_still_render_deterministically():
    one, _ = render_screen(demo_screen("whatsapp"))
    two, _ = render_screen(demo_screen("whatsapp"))
    assert one == two
