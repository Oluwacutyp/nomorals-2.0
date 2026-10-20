"""Pass #23 — /charagent talk graph wiring.

Pass #22 fixed the *tool* path (``characters talk``) discarding the
char<->brain relationship growth: ``converse()`` built a throwaway
RelationshipGraph, the shift landed on it, and the saved
``_relationships.json`` never learned the talk happened.

The ``/charagent talk`` command path in the partner runtime had the
same defect: it never loaded the sidecar graph, never passed one to
``converse()``, and never saved one — so char<->brain growth from
command-driven talks was silently discarded on every call.

These tests drive the real handler (RuntimeGamesMixin) against a
scratch character store (via NOMORALS_DATA_DIR).
"""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.characters import Character, CharacterStore
from nomorals.characters.relationships import RelationshipGraph, BRAIN_ID


def _distinguishing_suggest(prompt: str) -> str:
    first = (prompt or "").splitlines()[0]
    if first.startswith("You are Zara"):
        return "ZARA-LINE"
    return "BRAIN-LINE"


def _handler(suggest):
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    class _RT(RuntimeGamesMixin):
        def __init__(self, ctx):
            self.context = ctx

    return _RT(SimpleNamespace(suggest=suggest))


def _scratch_store():
    root = Path(tempfile.mkdtemp())
    os.environ["NOMORALS_DATA_DIR"] = str(root)
    store = CharacterStore()  # honors NOMORALS_DATA_DIR
    assert store.dir == root / "characters"
    return root, store


def _brain_edge(store: CharacterStore, char_id: str):
    sidecar = store.dir / "_relationships.json"
    if not sidecar.exists():
        return None
    g = RelationshipGraph.from_dict(
        json.loads(sidecar.read_text("utf-8")))
    return g.edges.get((char_id, BRAIN_ID))


def test_charagent_talk_persists_brain_relationship_growth():
    """A /charagent talk must grow AND persist the char<->brain edge."""
    root, store = _scratch_store()
    zara = Character(name="Zara", persona={"witty": 0.9})
    store.save(zara)

    rt = _handler(_distinguishing_suggest)
    out = rt._control_charagent("talk Zara hello there", "local:test")

    # the reply is the character's latest line (rounds=1 ends on her)
    assert out == "🎭 Zara: ZARA-LINE"
    # the relationship growth survived: an edge to the brain exists in
    # the persisted sidecar, with the conversation recorded on it
    edge = _brain_edge(store, zara.id)
    assert edge is not None, "no char<->brain edge in _relationships.json"
    assert edge.interactions >= 1
    assert edge.dims["familiarity"] > 0.0


def test_charagent_talk_preserves_existing_sidecar_edges():
    """Loading must not clobber pre-existing edges when saving."""
    root, store = _scratch_store()
    zara = Character(name="Zara", persona={"witty": 0.9})
    elder = Character(name="Elder", persona={"calm": 0.8})
    store.save(zara)
    store.save(elder)
    g = RelationshipGraph()
    g.interact(zara.id, elder.id, "laughed_together")
    (store.dir / "_relationships.json").write_text(
        json.dumps(g.to_dict(), indent=1), encoding="utf-8")

    rt = _handler(_distinguishing_suggest)
    rt._control_charagent("talk Zara a long chat about many things",
                          "local:test")

    saved = RelationshipGraph.from_dict(json.loads(
        (store.dir / "_relationships.json").read_text("utf-8")))
    assert (zara.id, elder.id) in saved.edges  # pre-existing edge kept
    assert (zara.id, BRAIN_ID) in saved.edges   # new brain edge added


def test_charagent_talk_persists_character_memory_and_mood():
    """store.save after converse persists the char's memory + mood."""
    root, store = _scratch_store()
    zara = Character(name="Zara", persona={"witty": 0.9})
    store.save(zara)
    before_valence = zara.mood["valence"]

    rt = _handler(_distinguishing_suggest)
    rt._control_charagent("talk Zara hello there", "local:test")

    fresh = CharacterStore().get_by_name("Zara")
    assert fresh is not None
    notes = " ".join(str(m.get("text", "")) for m in fresh.memory)
    assert "Devon" in notes  # the talk became a character memory
    assert fresh.mood["valence"] > before_valence  # mood shifted + saved
