"""Pass #22 — characters + community.

Verified bugs (all reproduced before the fix):
1. ``characters talk <name>`` returned *Devon's* reply labeled as the
   character's line (rounds=2 -> last turn is Devon's).
2. ``converse()`` built a throwaway RelationshipGraph internally, so the
   character<->brain relationship grown by a talk was silently discarded;
   the tool's saved ``_relationships.json`` never learned it happened.
3. ``Beat.ready()`` ignored ``cooldown_rounds`` for beats without a
   precondition — a custom beat fired every round.
"""
import json
import tempfile
from pathlib import Path

from nomorals.characters import Character, CharacterStore, converse
from nomorals.characters.ensemble import Beat, DramaDirector, Scene
from nomorals.characters.relationships import RelationshipGraph, BRAIN_ID


def _distinguishing_suggest(prompt: str) -> str:
    first = (prompt or "").splitlines()[0]
    if first.startswith("You are Zara"):
        return "ZARA-LINE"
    return "BRAIN-LINE"


def _make_store():
    root = Path(tempfile.mkdtemp())
    return CharacterStore(data_dir=root / "characters")


class _FakeRegistry:
    def __init__(self, suggest):
        from types import SimpleNamespace
        self.context = SimpleNamespace(suggest=suggest)

    def register(self, name, **kwargs):
        def deco(fn):
            self.fn = fn
            return fn
        return deco


def _tool_fn(suggest):
    from nomorals.tools import characters as tool_mod
    reg = _FakeRegistry(suggest)
    tool_mod.register(reg)
    return reg.fn


def test_tool_talk_returns_the_characters_own_line():
    """The talk reply must be the character's line, not Devon's."""
    store = _make_store()
    store.save(Character(name="Zara", persona={"witty": 0.9}))
    # the tool builds its store on the default dir; point it at the
    # same scratch root via NOMORALS_DATA_DIR.
    import os
    os.environ["NOMORALS_DATA_DIR"] = str(store.dir.parent)
    fn = _tool_fn(_distinguishing_suggest)
    out = fn(action="talk", target="Zara", text="hey")
    assert "ZARA-LINE" in out, f"expected the character's line, got: {out!r}"
    assert "BRAIN-LINE" not in out.replace("🎭 Zara:", ""), \
        f"Devon's line was attributed to Zara: {out!r}"


def test_converse_accepts_a_shared_graph():
    """A graph passed into converse() must carry the interaction out."""
    c = Character(name="Zara", persona={"witty": 0.9})
    g = RelationshipGraph()
    converse("You are Devon.", c, "brain: hi", _distinguishing_suggest,
             rounds=1, graph=g)
    edge = g.edge(c.id, BRAIN_ID)
    assert edge.interactions >= 1, "the shared graph should record the talk"


def test_tool_talk_persists_brain_relationship():
    """After a tool talk, _relationships.json knows the brain edge."""
    import os
    store = _make_store()
    store.save(Character(name="Zara", persona={"witty": 0.9}))
    os.environ["NOMORALS_DATA_DIR"] = str(store.dir.parent)
    fn = _tool_fn(_distinguishing_suggest)
    fn(action="talk", target="Zara", text="hey")
    gpath = store.dir / "_relationships.json"
    assert gpath.exists(), "the sidecar should be written after a talk"
    edges = json.loads(gpath.read_text("utf-8"))["edges"]
    brain_edges = [e for e in edges
                   if e.get("a") == store.get_by_name("zara").id
                   and e.get("b") == BRAIN_ID]
    assert brain_edges and brain_edges[0].get("interactions", 0) >= 1, \
        "the char->brain edge should grow with each talk"


def test_beat_cooldown_holds_without_precondition():
    """cooldown_rounds must hold even when a beat has no precondition."""
    beat = Beat("nudge", "steer text", energy_boost=-0.5, cooldown_rounds=5)
    d = DramaDirector(beats=[beat], energy=0.3)
    sc = Scene(title="t", participants=["A", "B"])
    fired = [bool(d.direct(sc, rnd)) for rnd in range(5)]
    assert fired[0], "a beat with no precondition may fire once"
    assert not any(fired[1:]), \
        f"cooldown_rounds=5 must suppress re-fires inside the window: {fired}"
