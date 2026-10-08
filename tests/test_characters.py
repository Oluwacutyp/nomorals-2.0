"""Offline tests for conversational story characters (#105)."""

import pytest

from nomorals.audio.characters import (
    Character,
    CharacterStore,
    characters_from_book,
    control_character,
    talk_to,
)


def _make_char(**kw):
    kw.setdefault("name", "Renfield")
    kw.setdefault("book_title", "Dracula")
    return Character(**kw)


# ── character creation ────────────────────────────────────────────────────


def test_creation_defaults():
    c = _make_char()
    assert c.name == "Renfield"
    assert c.knows_up_to() == 1


def test_cutoff_clamped():
    c = _make_char(knowledge_cutoff=0)
    assert c.knows_up_to() == 1
    c.set_cutoff(7)
    assert c.knows_up_to() == 7


def test_personality_clamped():
    c = _make_char(personality={"bravery": 2.0, "greed": -1.0, "bad": "x"})
    assert c.personality["bravery"] == 1.0
    assert c.personality["greed"] == 0.0
    assert "bad" not in c.personality


def test_remember_tags_chapters():
    c = _make_char(knowledge_cutoff=3)
    c.remember(2, "met the count at the castle")
    c.remember(5, "escaped the asylum")
    assert 2 in c.chapter_memories
    assert 5 in c.chapter_memories


# ── spoiler discipline ────────────────────────────────────────────────────


def test_system_prompt_hard_cutoff():
    c = _make_char(knowledge_cutoff=4)
    c.remember(2, "the count is pale")
    c.remember(6, "the count is destroyed at dawn")  # past cutoff
    prompt = c.to_system_prompt()
    assert "chapter 4" in prompt
    assert "NO knowledge of" in prompt
    # past-cutoff memory never reaches the prompt
    assert "destroyed at dawn" not in prompt
    assert "the count is pale" in prompt


def test_cutoff_stated_as_identity():
    c = _make_char(knowledge_cutoff=3)
    prompt = c.to_system_prompt()
    assert "It is currently chapter 3" in prompt
    assert "never read ahead" in prompt


def test_current_chapter_can_narrow_not_widen():
    c = _make_char(knowledge_cutoff=5)
    c.remember(4, "early fact")
    c.remember(5, "later fact")
    prompt = c.to_system_prompt(current_chapter=4)
    assert "chapter 4" in prompt
    assert "later fact" not in prompt
    assert "early fact" in prompt


def test_talk_to_filters_memories():
    c = _make_char(knowledge_cutoff=2)
    c.remember(1, "works at the asylum")
    c.remember(9, "dies in the final battle")
    seen = {}

    def llm(system, question):
        seen["system"] = system
        seen["question"] = question
        return "I am but a humble clerk."

    res = talk_to(c, "what happens to you?", llm_fn=llm)
    assert res.ok
    assert "works at the asylum" in seen["system"]
    assert "dies in the final battle" not in seen["system"]


def test_talk_to_no_engine_refuses():
    c = _make_char()
    res = talk_to(c, "hello?")
    assert not res.ok
    assert res.reason


def test_talk_to_no_character():
    res = talk_to(None, "hello?")
    assert not res.ok


def test_talk_to_empty_question():
    c = _make_char()
    res = talk_to(c, "   ", llm_fn=lambda s, q: "hi")
    assert not res.ok


# ── store ─────────────────────────────────────────────────────────────────


def _tmp_store(tmp_path):
    return CharacterStore(db_path=str(tmp_path / "chars.db"))


def test_store_roundtrip(tmp_path):
    s = _tmp_store(tmp_path)
    c = _make_char(knowledge_cutoff=3, voice="raspy whisper",
                   goals=["serve the count"],
                   speech_patterns=["calls everyone master"],
                   personality={"loyalty": 0.9})
    c.remember(1, "eats flies")
    c.remember(2, "waits in the cell")
    assert s.save(c)
    back = s.get("renfield")  # case-insensitive
    assert back is not None
    assert back.name == "Renfield"
    assert back.knows_up_to() == 3
    assert back.voice == "raspy whisper"
    assert back.goals == ["serve the count"]
    assert back.personality["loyalty"] == 0.9
    assert back.chapter_memories[1] == ["eats flies"]


def test_store_list_and_remove(tmp_path):
    s = _tmp_store(tmp_path)
    s.save(_make_char(name="Mina", book_title="Dracula"))
    s.save(_make_char(name="Harker", book_title="Dracula"))
    assert len(s.list()) == 2
    assert s.remove("mina")
    assert len(s.list()) == 1
    assert not s.remove("nobody")


def test_store_never_raises_on_bad_db(tmp_path):
    s = CharacterStore(db_path=str(tmp_path / "no" / "such" / "dir" / "x.db"))
    assert s.save(_make_char()) is False
    assert s.get("x") is None
    assert s.list() == []


# ── book surface ──────────────────────────────────────────────────────────


class _FakeChapter:
    def __init__(self, number, text):
        self.number = number
        self.text = text


class _FakeBook:
    def __init__(self):
        self.id = "book_1"
        self.title = "Test Tale"
        self.chapters = [
            _FakeChapter(1, "Renfield muttered in the cell. Mina visited."),
            _FakeChapter(2, "Harker arrived at the castle. Renfield laughed."),
            _FakeChapter(3, "Dracula sailed at dawn."),
        ]


def test_characters_from_book():
    chars = characters_from_book(_FakeBook())
    by_name = {c.name: c for c in chars}
    assert "Renfield" in by_name
    assert "Mina" in by_name
    assert "Harker" in by_name
    assert by_name["Renfield"].knows_up_to() == 1  # first seen ch1
    assert by_name["Harker"].knows_up_to() == 2
    # memories tagged per chapter
    assert 2 in by_name["Renfield"].chapter_memories


def test_characters_from_book_none():
    assert characters_from_book(None) == []


def test_characters_from_book_saves(tmp_path):
    s = _tmp_store(tmp_path)
    chars = characters_from_book(_FakeBook(), store=s)
    assert len(chars) == len(s.list())


# ── chat ──────────────────────────────────────────────────────────────────


def _ctl(tail, tmp_path, **kw):
    return control_character(tail, store=_tmp_store(tmp_path), **kw)


def test_chat_help(tmp_path):
    out = _ctl("", tmp_path)
    assert "/character talk" in out


def test_chat_add_list(tmp_path):
    out = _ctl("add Renfield Dracula --cutoff 4", tmp_path)
    assert "talkable now" in out
    out = _ctl("list", tmp_path)
    assert "Renfield" in out
    assert "chapter 4" in out


def test_chat_add_no_name(tmp_path):
    assert "add whom" in _ctl("add", tmp_path)


def test_chat_talk(tmp_path):
    _ctl("add Renfield Dracula", tmp_path)
    out = _ctl("talk Renfield who are you?",
               tmp_path,
               llm_fn=lambda s, q: "I am Renfield, at your service.")
    assert "I am Renfield" in out
    assert "chapter 1" in out


def test_chat_talk_no_engine(tmp_path):
    _ctl("add Renfield", tmp_path)
    out = _ctl("talk Renfield hello", tmp_path)
    assert "no dialogue engine" in out


def test_chat_talk_unknown(tmp_path):
    assert "no character" in _ctl("talk Nobody hi", tmp_path)


def test_chat_cutoff(tmp_path):
    _ctl("add Renfield", tmp_path)
    out = _ctl("cutoff Renfield 6", tmp_path)
    assert "chapter 6" in out
    out = _ctl("list", tmp_path)
    assert "chapter 6" in out


def test_chat_cutoff_bad(tmp_path):
    _ctl("add Renfield", tmp_path)
    assert "must be a number" in _ctl("cutoff Renfield soon", tmp_path)
    assert "no character" in _ctl("cutoff Nobody 3", tmp_path)


def test_chat_forget(tmp_path):
    _ctl("add Renfield", tmp_path)
    assert "is gone" in _ctl("forget Renfield", tmp_path)
    assert "no character" in _ctl("forget Renfield", tmp_path)


def test_chat_unknown_command(tmp_path):
    assert "/character talk" in _ctl("dance Renfield", tmp_path)


def test_chat_never_raises(tmp_path):
    for tail in ["", "talk", "talk Renfield", "add", "cutoff", "cutoff x",
                 "forget", None, "talk \x00 Renfield"]:
        out = control_character(tail, store=_tmp_store(tmp_path))
        assert isinstance(out, str)
