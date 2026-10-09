"""Character rebuild: relationships, arcs, casting, ensemble, processing."""
import time

from nomorals.characters import (
    Character, CharacterStore,
    RelationshipGraph, Edge,
    add_belief, challenge_belief, drop_belief, milestone, arc_summary,
    cast_for, cast_preset, chemistry,
    run_scene, podcast_episode,
    process_session, SessionEvent, decay_moods,
)


def _mk(name, **kw):
    kw.setdefault("persona", {"witty": 0.8})
    return Character(name=name, **kw)


class TestRelationships:
    def test_edge_defaults(self):
        e = Edge(a="x", b="y")
        assert e.dims["trust"] == 0.5
        assert e.kind == "acquaintance"

    def test_nudge_slow(self):
        e = Edge(a="x", b="y")
        before = e.dims["trust"]
        e.nudge({"trust": 1.0})
        # slow by design: one nudge moves a fraction
        assert e.dims["trust"] - before < 0.2
        assert e.dims["trust"] > before

    def test_kind_evolution(self):
        e = Edge(a="x", b="y")
        for _ in range(30):
            e.nudge({"trust": 1.0, "warmth": 1.0, "familiarity": 1.0})
        assert e.kind in ("friend", "close")

    def test_rivalry(self):
        e = Edge(a="x", b="y")
        for _ in range(30):
            e.nudge({"friction": 1.0, "trust": -1.0})
        assert e.kind == "rival"

    def test_directional(self):
        g = RelationshipGraph()
        g.interact("a", "b", "betrayed", mirror="was_helped_by")
        assert g.edge("a", "b").dims["trust"] < g.edge("b", "a").dims["trust"]

    def test_decay(self):
        e = Edge(a="x", b="y")
        e.dims["friction"] = 0.9
        e.updated_at = time.time() - 86400 * 5
        e.decay()
        assert e.dims["friction"] < 0.9

    def test_serialization(self):
        g = RelationshipGraph()
        g.interact("a", "b", "good_conversation")
        g2 = RelationshipGraph.from_dict(g.to_dict())
        assert g2.edge("a", "b").interactions == 1


class TestArcs:
    def test_belief_lifecycle(self):
        c = _mk("T")
        b = add_belief(c, "The sky is blue.", 0.8)
        assert b["revisions"] == 0
        challenge_belief(c, "The sky is blue.", 0.4, "saw a red sunset")
        assert c.beliefs[0]["revisions"] == 1
        assert c.beliefs[0]["confidence"] == 0.4
        # too alive to drop
        assert drop_belief(c, "The sky is blue.") is False
        challenge_belief(c, "The sky is blue.", 0.1)
        assert drop_belief(c, "The sky is blue.") is True
        assert not c.beliefs

    def test_milestone_and_arc(self):
        c = _mk("T")
        add_belief(c, "I am enough.", 0.5)
        challenge_belief(c, "I am enough.", 0.8, "won the match")
        milestone(c, "First win", "beat everyone")
        s = arc_summary(c)
        assert "revised 1" in s
        assert "MILESTONE" in s


class TestCasting:
    def _chars(self):
        z = _mk("Zara", persona={"witty": 0.9, "curious": 0.9},
                roles=["podcast_host"], skills={"interviewing": 0.9})
        k = _mk("Kilo", persona={"witty": 0.8, "dry": 0.9},
                roles=["podcast_host"], skills={"banter": 0.9})
        j = _mk("Jax", persona={"competitive": 0.9},
                roles=["gamer"], skills={"strategy": 0.85})
        return [z, k, j]

    def test_cast_single(self):
        picks = cast_for(self._chars(), role="podcast_host", n=1)
        assert picks[0][0].name in ("Zara", "Kilo")

    def test_cast_gamer(self):
        picks = cast_preset(self._chars(), "gamer", n=1)
        assert picks[0][0].name == "Jax"

    def test_ensemble_chemistry(self):
        chars = self._chars()
        g = RelationshipGraph()
        g.edge(chars[0].id, chars[1].id).nudge(
            {"warmth": 1.0, "trust": 1.0, "familiarity": 1.0})
        picks = cast_for(chars, n=2, graph=g)
        names = {p[0].name for p in picks}
        # Zara+Kilo have history → picked together
        assert names == {"Zara", "Kilo"}

    def test_chemistry_bounds(self):
        a, b = _mk("A"), _mk("B")
        assert 0.0 <= chemistry(a, b) <= 1.0


class TestProcessing:
    def test_session_updates(self):
        a, b = _mk("A"), _mk("B")
        chars = {a.id: a, b.id: b}
        g = RelationshipGraph()
        before = g.edge(a.id, b.id).dims["trust"]
        process_session(
            [SessionEvent(a.id, b.id, "deep_conversation", salience=0.8,
                          note="talked about life")],
            chars, g)
        assert g.edge(a.id, b.id).dims["trust"] > before
        assert a.mood["valence"] > 0  # mood shifted

    def test_decay_moods(self):
        c = _mk("T")
        c.mood["valence"] = 0.9
        c.last_active = time.time() - 86400 * 2
        decay_moods([c])
        assert c.mood["valence"] < 0.9


class TestEnsemble:
    def test_scene_runs(self):
        a = _mk("A", persona={"witty": 0.9})
        b = _mk("B", persona={"calm": 0.9})
        sc = run_scene([a, b], "Test", "testing", suggest=None,
                       rounds=2, seed=42)
        assert len(sc.turns) == 4
        assert set(sc.participants) == {"A", "B"}

    def test_podcast_episode(self):
        h = _mk("Host", roles=["podcast_host"])
        g = _mk("Guest")
        ep = podcast_episode(h, [g], "test topic", suggest=None,
                             rounds=2, seed=42)
        # opener + 2 guest turns + closer (host steers at i%3==2, not hit)
        assert ep.turns[0].speaker == "Host"
        assert ep.turns[-1].speaker == "Host"
