"""The narration specialist: journal -> grade -> retrain.

The dedicated narration specialist (games/narration_learn.py) scores
candidate narration lines and picks the best; GameMaster.narrate
consults it through the DEDICATED narration bridge
(``narration_suggest``) — never the dialogue bridge (``suggest``).
The brain generates candidates; the scorer decides. All offline:
the journal and model store live in tmp_path, the "model" is a stub
bridge callable. No gain numbers from fake models are asserted — a
smoke proves the wiring, never a number.
"""

from __future__ import annotations

import random

from nomorals.games.narration_learn import (
    NARRATION_FEATURES,
    NARRATION_MIN_ROWS,
    NarrationJournal,
    fit_narration_scorer,
    load_narration_scorer,
    narration_line_features,
    score_narration_candidates,
    score_narration_candidates_full,
)
from nomorals.llm.specialists import ModelStore

SEED = 20261114
EVENT = "Ada obliterates the Hollow Warden"
GRIM_LINE = "Ada falls, blood on the stones, the dark closes in."
CHEERFUL_LINE = "Ada wins the fight, joy and laugh fill the hall."
VIVID_LINE = ("Ada obliterates the Hollow Warden, blade blazing, "
              "the crowd roaring.")
PLAIN_LINE = "The fight continues."
RAMBLE_LINE = "Something happens somewhere."


# ── features ─────────────────────────────────────────────────────────────

def test_features_are_bounded_and_event_fit_reflects_event():
    feats = narration_line_features(VIVID_LINE, EVENT, "grim")
    assert set(feats) == set(NARRATION_FEATURES)
    for value in feats.values():
        assert 0.0 <= value <= 1.0, value
    assert feats["event_fit"] == 1.0  # every event word surfaces
    assert feats["vividness"] > 0.0  # blazing, roaring
    assert narration_line_features(PLAIN_LINE, EVENT, "grim")["event_fit"] == 0.0
    # mood fit: a grim line fits a grim mood better than a cheerful one
    grim_fit = narration_line_features(GRIM_LINE, EVENT, "grim")["mood_fit"]
    cheer_fit = narration_line_features(CHEERFUL_LINE, EVENT, "grim")["mood_fit"]
    assert grim_fit > cheer_fit
    # novelty: a line repeated against itself is not novel
    assert narration_line_features(VIVID_LINE, EVENT, "grim",
                                   recent_lines=[VIVID_LINE])["novelty"] == 0.0
    assert narration_line_features(VIVID_LINE, EVENT, "grim")["novelty"] == 1.0
    # length fit: near the 120-char ideal scores near 1
    ideal = "x" * 120
    assert narration_line_features(ideal, EVENT, "grim")["length_fit"] == 1.0
    assert narration_line_features("x", EVENT, "grim")["length_fit"] < 0.1


def test_hand_scorer_picks_most_event_fitting_line():
    cands = [PLAIN_LINE, VIVID_LINE, RAMBLE_LINE]
    scored = score_narration_candidates(cands, EVENT, "grim")
    assert [text for text, _ in scored] == cands  # input order kept
    assert scored[1][1] == max(score for _, score in scored)
    assert scored[1][1] > scored[0][1] and scored[1][1] > scored[2][1]
    # the full form attaches the REAL feature vector per candidate
    full = score_narration_candidates_full(cands, EVENT, "grim")
    assert set(full[1]["features"]) == set(NARRATION_FEATURES)
    assert full[1]["features"]["event_fit"] == 1.0


# ── journal → grade → retrain ────────────────────────────────────────────

def test_journal_grade_rows_round_trip(tmp_path):
    journal = NarrationJournal(tmp_path / "narr")
    scored = score_narration_candidates_full([VIVID_LINE], EVENT, "grim")
    entry_id = journal.journal("arena", line=VIVID_LINE, candidates=scored,
                               chosen=0, event=EVENT, mood="grim")
    assert entry_id
    entries = journal.entries("arena")
    assert len(entries) == 1
    assert entries[0]["line"] == VIVID_LINE
    assert entries[0]["grade"] is None
    assert entries[0]["candidates"][0]["features"]["event_fit"] == 1.0
    assert journal.rows("arena") == []  # ungraded: no training rows
    assert journal.grade("arena", VIVID_LINE, 0.8) is True
    rows = journal.rows("arena")
    assert len(rows) == 1
    assert rows[0]["label"] == 0.8
    assert rows[0]["features"] == scored[0]["features"]
    assert rows[0]["line"] == VIVID_LINE
    # already graded: a second grade on the same line does not land
    assert journal.grade("arena", VIVID_LINE, -0.5) is False
    assert journal.rows("arena")[0]["label"] == 0.8


def _graded_rows(n=80, seed=SEED):
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        feats = {k: rnd.random() for k in NARRATION_FEATURES}
        # grade: event fit + vividness carry the signal — the learned
        # scorer must beat the hand weights, which spread weight over
        # all five features
        grade = (0.6 * feats["event_fit"]
                 + 0.3 * feats["vividness"]
                 + 0.1 * rnd.random() - 0.45)
        rows.append({"features": feats,
                     "label": max(-1.0, min(1.0, grade))})
    return rows


def test_fit_trains_when_signal_beats_hand_weights(tmp_path):
    outcome = fit_narration_scorer(_graded_rows(),
                                   store=ModelStore(tmp_path / "models"))
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.beats_code is True
    loaded = load_narration_scorer(ModelStore(tmp_path / "models"))
    assert loaded is not None
    assert loaded.version == 1
    pred = loaded.predict_one(
        {"event_fit": 0.9, "mood_fit": 0.5, "vividness": 0.9,
         "novelty": 0.5, "length_fit": 0.5})
    assert isinstance(pred, float)
    # the fitted scorer prefers a high event-fit line over a low one
    low = loaded.predict_one(
        {"event_fit": 0.1, "mood_fit": 0.5, "vividness": 0.1,
         "novelty": 0.5, "length_fit": 0.5})
    assert pred > low


def test_fit_insufficient_below_floor(tmp_path):
    outcome = fit_narration_scorer(_graded_rows(5),
                                   store=ModelStore(tmp_path / "models"))
    assert outcome.status == "insufficient"
    assert outcome.model is None
    assert str(NARRATION_MIN_ROWS) in outcome.reason


# ── narrate wiring ───────────────────────────────────────────────────────

def test_narration_on_property():
    from nomorals.games.gamemaster import GameMaster

    assert GameMaster().narration_on is False
    assert GameMaster().model_on is False
    gm = GameMaster(narration_suggest=lambda p: "x")
    assert gm.narration_on is True
    assert gm.model_on is False  # the narration bridge is not the dialogue bridge


def test_narrate_falls_back_to_template_without_bridge():
    from nomorals.games.gamemaster import GameMaster

    gm = GameMaster()
    line = gm.narrate("arena", EVENT, mood="grim")
    assert line
    assert EVENT in line


def test_narrate_never_calls_dialogue_bridge(tmp_path, monkeypatch):
    from nomorals.games.gamemaster import GameMaster

    monkeypatch.setenv("NM_NARRATION_LEARN_DIR", str(tmp_path / "narr"))
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    dialogue_calls: list[str] = []
    narration_calls: list[str] = []

    def dialogue(prompt):
        dialogue_calls.append(prompt)
        return "DIALOGUE LINE"

    def narration(prompt):
        narration_calls.append(prompt)
        return VIVID_LINE

    # narration bridge absent: the dialogue bridge is never consulted
    gm = GameMaster(suggest=dialogue)
    line = gm.narrate("arena", EVENT, mood="grim")
    assert EVENT in line  # template forge carries
    assert dialogue_calls == []
    # narration bridge present: narrate reads ONLY it
    gm = GameMaster(suggest=dialogue, narration_suggest=narration)
    line = gm.narrate("arena", EVENT, mood="grim")
    assert line == VIVID_LINE
    assert dialogue_calls == []
    assert len(narration_calls) == 3  # three candidates asked for


def test_narrate_picks_scored_best_and_journals(tmp_path, monkeypatch):
    from nomorals.games.gamemaster import GameMaster

    monkeypatch.setenv("NM_NARRATION_LEARN_DIR", str(tmp_path / "narr"))
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    cands = [PLAIN_LINE, VIVID_LINE, RAMBLE_LINE]
    calls = {"n": 0}

    def fake_bridge(prompt):
        line = cands[calls["n"] % len(cands)]
        calls["n"] += 1
        return line

    gm = GameMaster(narration_suggest=fake_bridge)
    line = gm.narrate("arena", EVENT, mood="grim")
    assert line == VIVID_LINE  # the most event-fitting, vivid line wins
    journal = NarrationJournal(tmp_path / "narr")
    entries = journal.entries("arena")
    assert len(entries) >= 1
    last = entries[-1]
    assert last["candidates"][last["chosen"]]["text"] == VIVID_LINE
    assert len(last["candidates"]) == 3
    assert last["candidates"][1]["features"]["event_fit"] == 1.0
    assert last["grade"] is None  # ungraded until a grade lands


def test_narrate_fitted_model_picks_line(tmp_path, monkeypatch):
    from nomorals.games.gamemaster import GameMaster

    models_dir = tmp_path / "models"
    outcome = fit_narration_scorer(_graded_rows(),
                                   store=ModelStore(models_dir))
    assert outcome.status == "trained", outcome.reason
    monkeypatch.setenv("NM_NARRATION_LEARN_DIR", str(tmp_path / "narr"))
    monkeypatch.setenv("NM_MODELS_DIR", str(models_dir))
    cands = [PLAIN_LINE, VIVID_LINE, RAMBLE_LINE]
    calls = {"n": 0}

    def fake_bridge(prompt):
        line = cands[calls["n"] % len(cands)]
        calls["n"] += 1
        return line

    gm = GameMaster(narration_suggest=fake_bridge)
    line = gm.narrate("arena", EVENT, mood="grim")
    # the FITTED scorer decides: the expected winner is the one the
    # loaded model scores highest
    loaded = load_narration_scorer(ModelStore(models_dir))
    assert loaded is not None
    scored = score_narration_candidates_full(cands, EVENT, "grim",
                                             model=loaded)
    best = max(scored, key=lambda c: c["score"])
    assert line == best["text"]
