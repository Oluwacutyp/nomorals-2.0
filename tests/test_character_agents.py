"""Character agents: persistence, dialogue, and real game play."""
import tempfile
from pathlib import Path

from nomorals.characters import (
    Character, CharacterStore, converse, character_initiate,
    AgentSeat, run_agent_match, register_with_engine,
    CHAR_KEY_PREFIX, BRAIN_KEY,
)


def stub_suggest(prompt: str) -> str:
    # deterministic stub: picks first option for decisions,
    # short in-character line otherwise
    low = prompt.lower()
    if "legal options" in low:
        for line in prompt.splitlines():
            s = line.strip()
            if s.startswith("- "):
                return s[2:].strip()
    return "stub line, staying in character."


def make_store():
    return CharacterStore(data_dir=Path(tempfile.mkdtemp()))


def test_character_memory():
    c = Character(name="Zara", persona={"witty": 0.9})
    c.remember("Devon likes afrobeats", 0.9)
    c.remember("the sky is blue", 0.1)
    hits = c.recall("what music does Devon like")
    assert any("afrobeats" in h for h in hits)


def test_character_persist():
    store = make_store()
    c = Character(name="Kilo", persona={"bold": 0.8}, voice_name="kilo-voice",
                  goals=["win every game"])
    store.save(c)
    back = store.get(c.id)
    assert back is not None and back.name == "Kilo"
    assert back.voice_name == "kilo-voice"
    assert store.get_by_name("kilo").id == c.id


def test_character_speak():
    c = Character(name="Zara", persona={"witty": 0.9},
                  backstory="a Lagos radio host")
    line = c.speak("the show starts now", stub_suggest)
    assert len(line) > 0


def test_converse_brain_character():
    c = Character(name="Zara", persona={"witty": 0.9})
    dlg = converse("You are Devon.", c, "brain: want to play auction?",
                   stub_suggest, rounds=2)
    assert len(dlg.turns) >= 3
    speakers = {t.speaker for t in dlg.turns}
    assert "Devon" in speakers and "Zara" in speakers
    # character remembered the exchange
    assert len(c.memory) > 0


def test_character_initiate():
    c = Character(name="Kilo", persona={"bold": 0.8},
                  goals=["challenge Devon to a game"])
    line = character_initiate(c, stub_suggest)
    assert len(line) > 0


def test_agent_auction_match_real_state():
    store = make_store()
    zara = Character(name="Zara", persona={"witty": 0.9})
    kilo = Character(name="Kilo", persona={"bold": 0.8})
    store.save(zara)
    store.save(kilo)
    seats = [AgentSeat(kind="brain", name="Devon"),
             AgentSeat(kind="character", character=zara),
             AgentSeat(kind="character", character=kilo)]
    res = run_agent_match("auction", seats, stub_suggest, seed=42,
                          store=store, max_turns=60)
    # real game: multiple moves, winner decided, transcript non-trivial
    assert res.moves > 5, res.transcript[:5]
    assert any("winner:" in t for t in res.transcript)
    assert res.winner in ("Devon", "Zara", "Kilo", "draw")
    # characters remembered playing
    assert any("auction" in m.get("text", "")
               for m in store.get(zara.id).memory)


def test_register_with_engine():
    from nomorals.games.engine import GameEngine

    class _Ctx:
        db = None

    eng = GameEngine(_Ctx())
    store = make_store()
    register_with_engine(eng, store, stub_suggest)
    assert CHAR_KEY_PREFIX in eng.seat_deciders
    assert BRAIN_KEY in eng.seat_deciders
