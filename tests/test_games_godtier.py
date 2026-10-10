"""God-tier games phase: TriviaForge, recommender, wild cards."""
import random

from nomorals.games.gamemaster import WILD_CARDS, GameMaster
from nomorals.games.recommender import GAME_CLUSTERS, GameRecommender
from nomorals.games.trivia_forge import TriviaForge


class _FakeKV:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v):
        self.d[k] = v


def _forge(**kw):
    f = TriviaForge(seed=1234, **kw)
    f._kv = _FakeKV()
    f._seen = []
    return f


def test_forge_deals_unique_questions():
    f = _forge()
    qs = f.deal(count=8)
    assert len(qs) == 8
    questions = [q for q, _ in qs]
    assert len(set(questions)) == 8, "duplicates in one deal"


def test_forge_anti_repeat_across_deals():
    f = _forge()
    q1 = f.deal(count=8)
    q2 = f.deal(count=8)
    s1 = {q for q, _ in q1}
    s2 = {q for q, _ in q2}
    assert not (s1 & s2), "questions repeated across deals"


def test_forge_llm_path():
    def fake_suggest(prompt):
        assert "trivia" in prompt.lower()
        return '[["What is 2+2?", "4"], ["Capital of France?", "Paris"]]'
    f = TriviaForge(suggest=fake_suggest, seed=1)
    f._kv = _FakeKV()
    f._seen = []
    qs = f.deal(count=2)
    assert len(qs) == 2
    assert qs[0] == ("What is 2+2?", "4")


def test_forge_llm_junk_falls_back():
    f = TriviaForge(suggest=lambda p: "not json at all", seed=7)
    f._kv = _FakeKV()
    f._seen = []
    qs = f.deal(count=4)
    assert len(qs) == 4  # template forge filled in


def test_forge_topics_sampled():
    f = _forge()
    topics = f.sample_topics(10, profile={"technology": 5.0})
    assert len(topics) == 10
    assert all(isinstance(t, str) and t for t in topics)


def test_recommender_new_player():
    r = GameRecommender(seed=1)
    player = type("P", (), {"per_game": {}})()
    rec = r.recommend(player, ["trivia", "wordle", "mafia"])
    assert rec is not None
    assert rec["game"] in ("trivia", "wordle", "mafia")


def test_recommender_uses_history():
    r = GameRecommender(seed=1)
    player = type("P", (), {"per_game": {
        "trivia": {"played": 20}, "duel": {"played": 15}}})()
    rec = r.recommend(player, ["wordle", "anagram", "quizduel",
                               "cryptogram"], player_key="u1")
    assert rec is not None
    # trivia cluster → word cluster games suggested
    assert rec["game"] in ("wordle", "anagram", "cryptogram", "quizduel")
    assert "trivia" in rec["reason"] or "duel" in rec["reason"]


def test_recommender_no_nag():
    kv = _FakeKV()
    r = GameRecommender(seed=1)
    r._kv = kv
    player = type("P", (), {"per_game": {"trivia": {"played": 10}}})()
    avail = ["wordle", "anagram", "cryptogram", "twentyquestions"]
    r1 = r.recommend(player, avail, player_key="u2")
    assert r1 is not None
    r2 = r.recommend(player, avail, player_key="u2")
    if r2 is not None:
        # second recommendation shouldn't repeat the first
        assert r1["game"] != r2["game"]


def test_wild_cards_defined():
    assert len(WILD_CARDS) >= 6
    ids = [c[0] for c in WILD_CARDS]
    assert len(set(ids)) == len(ids)


def test_draw_wild_card():
    gm = GameMaster(seed=99)
    card = gm.draw_wild_card("test-game")
    assert card["id"] in [c[0] for c in WILD_CARDS]
    assert card["narration"]
    assert "WILD CARD" in card["narration"] or card["title"] in card["narration"]
