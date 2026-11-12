"""Small models Phase 2 — the REAL specialist models (owner correction).

Phase 1 (31346469) shipped the ladder with the BRAIN as the model rung
— a design-down the owner corrected 2026-11-12. These tests pin the
correction: real learned specialists, trained on real journal rows,
that must BEAT the code rung on a held-out slice to be trusted; below
their floor they are honestly unavailable and the code rung carries.

Every learner test trains on synthetic journal rows from a FIXED seed
(the signal is real — the label correlates with the features) and
asserts the trained model beats the code rung on the held-out slice.
A test that asserts on an untrained model proves nothing.
"""

from __future__ import annotations

import json
import random

import pytest

from nomorals.llm.specialists import (
    BackendUnavailable,
    LearnerSpec,
    ModelStore,
    NotEnoughData,
    available_backends,
    backend_available,
    fit_if_ready,
    train,
)

SEED = 0


def _synthetic_rows(n: int = 80, seed: int = SEED):
    """Label = 1 iff f0 + f1 > 1.0 — a real signal a learner can find."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        f0 = rnd.random()
        f1 = rnd.random()
        rows.append({"features": {"f0": f0, "f1": f1,
                                  "noise": rnd.random()},
                     "label": 1 if f0 + f1 > 1.0 else 0})
    return rows


SPEC = LearnerSpec(job="phase2-test", features=("f0", "f1", "noise"),
                   kind="classifier", backend="gradboost",
                   min_examples=30)


# ── toolkit ─────────────────────────────────────────────────────────────

def test_toolkit_backends_report_honestly():
    avail = available_backends()
    assert avail["gradboost"] == "ok"
    assert avail["logistic"] == "ok"
    assert avail["forest"] == "ok"
    assert avail["lightgbm"] == "ok"  # installed 2026-11-12
    # xgboost / catboost are NOT on this box (disk quota, Errno 122):
    # the rung must say so, naming the package — never a fake model.
    for name in ("xgboost", "catboost"):
        ok, reason = backend_available(name)
        assert ok is False
        assert name in reason.lower() or "not installed" in reason


def test_toolkit_train_beats_baseline_on_held_out():
    model, report = train(_synthetic_rows(), SPEC)
    assert report.n_rows == 80
    assert report.score > report.baseline_score
    assert 0.0 <= model.predict_one({"f0": 0.9, "f1": 0.9}) <= 1.0
    assert model.predict_one({"f0": 0.9, "f1": 0.9}) > 0.5


def test_toolkit_min_examples_guard():
    with pytest.raises(NotEnoughData):
        train(_synthetic_rows(5), SPEC)


def test_toolkit_unknown_backend_is_honest():
    spec = LearnerSpec(job="phase2-test", features=("f0", "f1"),
                       kind="classifier", backend="nope",
                       min_examples=5)
    with pytest.raises(BackendUnavailable):
        train(_synthetic_rows(10), spec)


def test_fit_if_ready_held_when_code_rung_wins():
    outcome = fit_if_ready(_synthetic_rows(), SPEC,
                           code_score_fn=lambda feats, labels: 1.0)
    assert outcome.status == "held"
    assert outcome.model is None
    assert outcome.report is not None
    assert outcome.report.beats_code is False


def test_fit_if_ready_trains_and_saves_when_it_beats_code(tmp_path):
    store = ModelStore(tmp_path / "models")
    outcome = fit_if_ready(_synthetic_rows(), SPEC,
                           code_score_fn=lambda feats, labels: 0.0,
                           store=store)
    assert outcome.status == "trained"
    assert outcome.model is not None
    loaded = store.load("phase2-test")
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.predict_one({"f0": 0.9, "f1": 0.9}) > 0.5
    assert store.latest_version("phase2-test") == 1


def test_fit_if_ready_insufficient_below_floor():
    outcome = fit_if_ready(_synthetic_rows(5), SPEC,
                           code_score_fn=lambda feats, labels: 0.0)
    assert outcome.status == "insufficient"
    assert outcome.model is None
    assert "below" in outcome.reason


def test_fit_if_ready_unavailable_when_backend_missing():
    spec = LearnerSpec(job="phase2-test", features=("f0", "f1"),
                       kind="classifier", backend="catboost",
                       min_examples=5)
    outcome = fit_if_ready(_synthetic_rows(10), spec)
    assert outcome.status == "unavailable"
    assert "catboost" in outcome.reason


def test_store_versions_and_corrupt_load_is_none(tmp_path):
    store = ModelStore(tmp_path / "models")
    fit_if_ready(_synthetic_rows(), SPEC,
                 code_score_fn=lambda feats, labels: 0.0, store=store)
    fit_if_ready(_synthetic_rows(), SPEC,
                 code_score_fn=lambda feats, labels: 0.0, store=store)
    assert store.latest_version("phase2-test") == 2
    loaded = store.load("phase2-test")
    assert loaded is not None and loaded.version == 2
    # corrupt the latest model file: load degrades to None, never raises
    job_dir = tmp_path / "models" / "phase2-test"
    (job_dir / "phase2-test-v2.pkl").write_bytes(b"not a pickle")
    assert store.load("phase2-test") is None


def test_store_manifest_carries_held_out_numbers(tmp_path):
    store = ModelStore(tmp_path / "models")
    outcome = fit_if_ready(_synthetic_rows(), SPEC,
                           code_score_fn=lambda feats, labels: 0.0,
                           store=store)
    manifest = json.loads(
        (tmp_path / "models" / "phase2-test" / "manifest.json")
        .read_text(encoding="utf-8"))
    assert manifest["latest"] == 1
    assert manifest["report"]["score"] == outcome.report.score
    assert manifest["report"]["n_rows"] == 80


# ── NPC learner ─────────────────────────────────────────────────────────

def _npc_profile():
    from nomorals.games.npc import NPCProfile

    return NPCProfile(
        id="mara", name="Mara", game_id="arena",
        personality={"witty": 0.9, "brilliant": 0.7, "sarcastic": 0.7},
        voice_style="dry", mood={"valence": 0.2, "arousal": 0.5})


def test_npc_scorer_picks_most_in_character():
    from nomorals.games.npc_learn import score_npc_candidates

    npc = _npc_profile()
    lines = [
        "The rain is wet today, I suppose.",
        "Oh brilliant, another genius plan. What could possibly go wrong?",
        "Hello friend, how are you doing today?",
    ]
    scored = score_npc_candidates(lines, npc, situation="rainy market")
    assert len(scored) == 3
    assert all(0.0 <= s[1] <= 1.0 for s in scored)
    best = max(scored, key=lambda s: s[1])[0]
    # the dry, sarcastic line is the most in-character for Mara
    assert best == lines[1]


def test_npc_learn_fit_beats_hand_weights(tmp_path):
    from nomorals.games.npc_learn import NPC_FEATURES, fit_npc_scorer

    rnd = random.Random(SEED)
    rows = []
    for _ in range(80):
        feats = {k: rnd.random() for k in NPC_FEATURES}
        # grade: trait overlap + novelty carry the signal
        grade = (0.6 * feats["trait_overlap"]
                 + 0.3 * feats["novelty"]
                 + 0.1 * rnd.random() - 0.45)
        rows.append({"features": feats, "label": max(-1.0, min(1.0, grade))})
    outcome = fit_npc_scorer("mara", rows, store=ModelStore(tmp_path))
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.beats_code is True


def test_npc_learn_fit_insufficient_below_floor(tmp_path):
    from nomorals.games.npc_learn import fit_npc_scorer

    rows = [{"features": {"trait_overlap": 0.5, "mood_fit": 0.5,
                          "situation_fit": 0.5, "novelty": 0.5,
                          "length_fit": 0.5}, "label": 0.1}] * 5
    outcome = fit_npc_scorer("mara", rows, store=ModelStore(tmp_path))
    assert outcome.status == "insufficient"


def test_npc_speak_picks_scored_best_and_journals(tmp_path, monkeypatch):
    from nomorals.games.gamemaster import GameMaster
    from nomorals.games.npc_learn import NPCJournal

    monkeypatch.setenv("NM_NPC_LEARN_DIR", str(tmp_path / "npc"))
    # fake brain: three candidates, the sarcastic one is most in-character
    cands = ["The rain is wet today, I suppose.",
             "Oh brilliant, another genius plan. What could possibly go wrong?",
             "Hello friend, how are you doing today?"]
    calls = {"n": 0}

    def fake_suggest(prompt, **kw):
        line = cands[calls["n"] % len(cands)]
        calls["n"] += 1
        return line

    gm = GameMaster(suggest=fake_suggest)
    npc = _npc_profile()
    line = gm.npc_speak(npc, "rainy market", candidates=3)
    assert line == cands[1]
    journal = NPCJournal(tmp_path / "npc")
    entries = journal.entries("arena", "mara")
    assert len(entries) >= 1
    last = entries[-1]
    assert last["candidates"][last["chosen"]]["text"] == cands[1]
    assert len(last["candidates"]) == 3
    assert last["candidates"][1]["features"]["trait_overlap"] > 0.0


def test_tactics_policy_learns_from_combat_journal(tmp_path):
    from nomorals.games.npc_learn import TacticsJournal, fit_tactics_policy

    rnd = random.Random(SEED)
    journal = TacticsJournal(tmp_path / "tactics")
    # signal: fury wins when the foe is nearly dead, attack otherwise.
    # The journal must cover what was TRIED: fury is tried on a healthy
    # foe 20% of the time (and whiffs) — without those rows the policy
    # has no evidence fury loses there and extrapolates.
    for _ in range(120):
        foe_low = rnd.random() < 0.5
        if foe_low:
            action = "fury" if rnd.random() < 0.8 else "attack"
        else:
            action = "fury" if rnd.random() < 0.2 else "attack"
        won = (action == "fury") if foe_low else (action == "attack")
        journal.record_round(
            "arena", {"self_hp": 0.5, "foe_hp": 0.15 if foe_low else 0.8,
                      "foe_hp_low": 1.0 if foe_low else 0.0,
                      "potions": 1, "fury_ready": 1.0, "focused": 0.0},
            action, payoff=1.0 if won else -1.0, won=won)
    outcome = fit_tactics_policy("arena", journal,
                                 store=ModelStore(tmp_path / "models"))
    assert outcome.status == "trained", outcome.reason
    policy = outcome.model
    assert policy is not None
    # foe nearly dead + fury ready -> the policy picks fury
    assert policy.pick_action(
        {"self_hp": 0.5, "foe_hp": 0.15, "foe_hp_low": 1.0,
         "potions": 1, "fury_ready": 1.0, "focused": 0.0}) == "fury"
    # foe healthy -> attack, not fury
    assert policy.pick_action(
        {"self_hp": 0.5, "foe_hp": 0.8, "foe_hp_low": 0.0,
         "potions": 1, "fury_ready": 1.0, "focused": 0.0}) == "attack"


# ── DJ feedback loop ────────────────────────────────────────────────────

def test_dj_show_outcomes_logged_and_taste_trains(tmp_path, monkeypatch):
    from nomorals.media import dj_engine

    monkeypatch.setattr(dj_engine, "DEFAULT_FEEDBACK_PATH",
                        str(tmp_path / "dj_feedback.jsonl"))
    tracklist = [
        {"track_id": f"t{i}", "title": f"Track {i}",
         "artist": "A", "duration_s": 180,
         "features": {"energy": 0.5 + 0.05 * i, "tempo": 120,
                      "valence": 0.5, "danceability": 0.5,
                      "same_style": 1, "hour": 20, "energy_target": 0.6,
                      "fatigue": 0.0}}
        for i in range(10)]
    kept = {"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"}
    dj_engine.log_show_outcomes(tracklist, kept)
    rows = dj_engine.read_feedback_log()
    assert len(rows) == 10
    assert sum(1 for r in rows if r.get("event") == "played") == 8
    assert sum(1 for r in rows if r.get("event") == "skipped") == 2
    # the taste model now trains from REAL logged rows
    taste = dj_engine.LearnedTaste()
    assert taste.fit_from_log() is True
    assert taste.trained
    assert taste.rows_seen == 8  # the 2 compose skips are not taste rows


def test_dj_feedback_action_logs_owner_like(tmp_path, monkeypatch):
    from nomorals.media import dj as dj_mod
    from nomorals.media import dj_engine

    monkeypatch.setattr(dj_engine, "DEFAULT_FEEDBACK_PATH",
                        str(tmp_path / "dj_feedback.jsonl"))
    actions = dj_mod.register()
    # register() returns {"name", "description", "call": {...}}
    assert "feedback" in actions["call"]
    result = actions["call"]["feedback"](track="t3", verdict="liked")
    assert result.get("ok") is True
    rows = dj_engine.read_feedback_log()
    assert len(rows) == 1
    assert rows[0]["event"] == "liked"
    assert rows[0]["track_id"] == "t3"


# ── PumpDesk win model ──────────────────────────────────────────────────

def _graded_calls(n: int = 100, seed: int = SEED):
    """Label = win iff momentum + smart_money > 1.1 — a real learnable
    edge. Rows use the real graded-call shape (components + outcome)."""
    rnd = random.Random(seed)
    calls = []
    for i in range(n):
        mom = rnd.random()
        smart = rnd.random()
        safety_rug = rnd.random()
        safety_manip = rnd.random()
        call = {
            "id": f"c{i}", "mint": "MINT", "call_id": f"c{i}",
            "components": {"momentum": mom, "smart_money": smart,
                           "rug_risk": 1.0 - safety_rug,
                           "manipulation": 1.0 - safety_manip},
            "composite": (0.30 * mom + 0.15 * smart
                          + 0.30 * safety_rug + 0.25 * safety_manip),
            "entry_mc_sol": 100.0, "entry_curve_pct": 0.5,
            "outcome": "win" if mom + smart > 1.1 else "loss",
        }
        calls.append(call)
    return calls


def test_pumpdesk_win_model_beats_composite_rule(tmp_path):
    from nomorals.agents.pumpdesk import learn as pdl

    calls = _graded_calls()
    outcome = pdl.fit_win_model(calls, store=ModelStore(tmp_path / "m"))
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.beats_code is True


def test_pumpdesk_win_model_insufficient_below_floor(tmp_path):
    from nomorals.agents.pumpdesk import learn as pdl

    outcome = pdl.fit_win_model(_graded_calls(5),
                                store=ModelStore(tmp_path / "m"))
    assert outcome.status == "insufficient"


def test_pumpdesk_evaluate_carries_win_probability(tmp_path):
    from nomorals.agents.pumpdesk import learn as pdl
    from nomorals.agents.pumpdesk.score import evaluate

    outcome = pdl.fit_win_model(_graded_calls(),
                                store=ModelStore(tmp_path / "m"))
    model = outcome.model
    assert model is not None
    base = {
        "mint": "MINT",
        "trades": {"trades": 120, "unique_buyers": 40,
                   "velocity_sol_per_min": 2.0, "sol_per_trade": 0.35,
                   "buy_sell_ratio": 3.0},
        "curve": {"market_cap_sol": 100.0, "price_sol": 0.001,
                  "curve_pct": 0.85},
        "dev": {"launches": 5, "known_outcomes": 5, "rug_ratio": 0.0},
        "authorities": {"mint_authority_live": False,
                        "freeze_authority_live": False},
        "holders": {"top10_share": 0.1},
        "smart": {"weighted": 0.9, "hits": 3},
    }
    verdict = evaluate(base, win_model=model)
    assert verdict.win_probability is not None
    assert 0.0 <= verdict.win_probability <= 1.0
    # a strong dossier reads high; a weak one reads low
    weak = dict(base)
    weak["trades"] = {"trades": 120, "unique_buyers": 40,
                      "velocity_sol_per_min": 0.1, "sol_per_trade": 0.05,
                      "buy_sell_ratio": 1.0}
    weak["curve"] = {"market_cap_sol": 100.0, "price_sol": 0.001,
                     "curve_pct": 0.1}
    weak["smart"] = {"weighted": 0.1, "hits": 0}
    weak_verdict = evaluate(weak, win_model=model)
    assert weak_verdict.win_probability < verdict.win_probability


# ── error-catcher classifier ────────────────────────────────────────────

def _graded_failures(n: int = 80, seed: int = SEED):
    from nomorals.agents.failure_models import failure_features

    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        real = rnd.random() < 0.5
        failure = {
            "site": rnd.choice(["agents/news.py", "media/dj.py",
                                "agents/scheduler.py"]),
            "exc_type": ("TimeoutError" if real else "ValueError"),
            "message": ("x" * 120 if real else "x" * 12),
            "burst_1h": (5 if real else 1),
        }
        rows.append({"features": failure_features(failure),
                     "label": 1 if real else 0})
    return rows


def test_error_classifier_beats_surface_everything(tmp_path):
    from nomorals.agents.failure_models import fit_error_classifier

    outcome = fit_error_classifier(_graded_failures(),
                                   store=ModelStore(tmp_path / "m"))
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.beats_code is True


def test_error_classifier_grade_roundtrip(tmp_path, monkeypatch):
    from nomorals.agents import failure_models as fm

    grades_path = tmp_path / "failure_grades.jsonl"
    monkeypatch.setattr(fm, "DEFAULT_GRADES_PATH", str(grades_path))
    fm.grade_failure(site="agents/news.py", exc_type="TimeoutError",
                     message="poll timed out after 300s", real=True)
    fm.grade_failure(site="agents/news.py", exc_type="ValueError",
                     message="bad row", real=False)
    rows = fm.graded_rows()
    assert len(rows) == 2
    assert rows[0]["label"] == 1
    assert rows[1]["label"] == 0


def test_error_classifier_unavailable_below_floor(tmp_path):
    from nomorals.agents.failure_models import fit_error_classifier

    outcome = fit_error_classifier(_graded_failures(5),
                                   store=ModelStore(tmp_path / "m"))
    assert outcome.status == "insufficient"


# ── morning-pulse ranker ────────────────────────────────────────────────

def _pulse_history(n: int = 100, seed: int = SEED):
    """Early-life sections (few prior views): the hand rule's
    views>=7 threshold never fires there, so it predicts a followup
    for every section while only ~10% get one — exactly where it is
    blind. The features are PRIOR counts (no label leakage: the
    label is whether THIS briefing got a followup)."""
    rnd = random.Random(seed)
    history = []
    for i in range(n):
        section = rnd.choice(["markets", "personal", "news"])
        prior_views = rnd.randint(0, 6)
        base_rate = 0.4 if section == "personal" else 0.05
        prior_followups = sum(1 for _ in range(prior_views)
                              if rnd.random() < base_rate)
        label = 1 if (prior_views >= 3
                      and rnd.random() < base_rate) else 0
        history.append({"section": section,
                        "date": f"2026-11-{1 + i % 10:02d}",
                        "views": prior_views, "followups": prior_followups,
                        "days_since": rnd.randint(0, 7), "n_items": 5,
                        "label": label})
    return history


def test_pulse_ranker_beats_hand_rule(tmp_path):
    from nomorals.agents.pulse_rank import fit_pulse_ranker

    outcome = fit_pulse_ranker(_pulse_history(),
                               store=ModelStore(tmp_path / "m"))
    assert outcome.status == "trained", outcome.reason
    assert outcome.report is not None
    assert outcome.report.beats_code is True


def test_pulse_ranker_unavailable_below_floor(tmp_path):
    from nomorals.agents.pulse_rank import fit_pulse_ranker

    outcome = fit_pulse_ranker(_pulse_history(5),
                               store=ModelStore(tmp_path / "m"))
    assert outcome.status == "insufficient"


# ── characters' generative specialists ──────────────────────────────────

def test_character_specialist_fields_roundtrip():
    from nomorals.characters.character import Character

    char = Character(name="Mara", voice_name="mara-dry",
                     rvc_model="mara-v1", lora_file="/x/mara-lora.gguf",
                     face_ref="/x/mara-face.png")
    data = char.to_dict()
    assert data["rvc_model"] == "mara-v1"
    assert data["lora_file"] == "/x/mara-lora.gguf"
    assert data["face_ref"] == "/x/mara-face.png"
    back = Character.from_dict(data)
    assert back.rvc_model == "mara-v1"
    assert back.lora_file == "/x/mara-lora.gguf"
    assert back.face_ref == "/x/mara-face.png"


def test_character_lora_slot_names_the_wall(tmp_path):
    from nomorals.characters import specialists as cs
    from nomorals.characters.character import Character

    char = Character(name="Mara")
    result = cs.lora_for(char)
    assert result["ok"] is False
    assert "no lora" in result["reason"].lower()
    char.lora_file = str(tmp_path / "missing.gguf")
    result = cs.lora_for(char)
    assert result["ok"] is False
    assert "missing" in result["reason"].lower()
    # an existing adapter file resolves — gguf_check requires the
    # GGUF magic AND >=10 MB (a real adapter, not a stub), so the
    # fixture is a real 10 MB file with the magic at offset 0.
    adapter = tmp_path / "mara.gguf"
    with open(adapter, "wb") as fh:
        fh.write(b"GGUF")
        fh.seek(10 * 1024 * 1024 - 1)
        fh.write(b"\x00")
    char.lora_file = str(adapter)
    result = cs.lora_for(char)
    assert result["ok"] is True
    assert result["path"] == str(adapter)


def test_character_rvc_swap_names_the_wall():
    from nomorals.characters import specialists as cs
    from nomorals.characters.character import Character

    char = Character(name="Mara")
    result = cs.swap_voice(char, "/tmp/line.wav")
    assert result["ok"] is False
    assert "rvc" in result["reason"].lower()
    char.rvc_model = "mara-v1"
    result = cs.swap_voice(char, "/tmp/line.wav")
    # no RVC binary on this box — the wall is named, never a fake swap
    assert result["ok"] is False
    assert "rvc" in result["reason"].lower()


def test_character_xtts_voice_names_the_wall():
    from nomorals.characters import specialists as cs
    from nomorals.characters.character import Character

    char = Character(name="Mara", voice_name="mara-dry")
    result = cs.character_voice(char, "Hello there.")
    assert result["ok"] is False
    # this box has no TTS backend installed — the reason says so
    assert "tts" in result["reason"].lower() or \
        "backend" in result["reason"].lower()


def test_character_face_swap_names_the_wall(tmp_path):
    from nomorals.characters import specialists as cs
    from nomorals.characters.character import Character

    char = Character(name="Mara")
    result = cs.swap_face(char, str(tmp_path / "target.png"))
    assert result["ok"] is False
    assert "face" in result["reason"].lower()


# ── ladder: the brain is NOT a rung ─────────────────────────────────────

def test_ladder_brain_is_generator_not_rung():
    from nomorals.llm.ladder import resolve

    # brain present, specialist NOT trained -> code rung carries
    res = resolve("character_tactics", brain=lambda p: "attack",
                  specialist_available=False)
    assert res.rung == "code"
    assert "policy" in res.reason.lower() or "learned" in res.reason.lower()
    # specialist trained -> the model rung, and the brain is still only
    # the candidate generator, never the rung
    res = resolve("character_tactics", brain=lambda p: "attack",
                  specialist_available=True)
    assert res.rung == "model"
    assert "policy" in res.model_rung.lower()


def test_ladder_arena_narration_never_model_rung():
    from nomorals.llm.ladder import resolve

    res = resolve("arena_narration", brain=lambda p: "a tale",
                  specialist_available=False)
    assert res.rung == "code"
    assert "narration" in res.reason.lower()
    # even with the brain present there is no narration specialist
    res = resolve("arena_narration", brain=lambda p: "a tale")
    assert res.rung == "code"


def test_ladder_dialogue_specialist_replaces_brain_rung():
    from nomorals.llm.ladder import resolve

    res = resolve("character_dialogue", brain=lambda p: "hello",
                  specialist_available=True)
    assert res.rung == "model"
    assert "scorer" in res.model_rung.lower()
    res = resolve("character_dialogue", brain=lambda p: "hello",
                  specialist_available=False)
    assert res.rung == "code"


def test_ladder_new_specialist_jobs_registered():
    from nomorals.llm.ladder import JOBS

    for job in ("npc_line_scorer", "npc_tactics", "pumpdesk_win",
                "error_triage", "morning_rank", "character_voice_xtts",
                "character_voice_rvc", "character_lora",
                "character_face"):
        assert job in JOBS, job
        assert "brain" not in JOBS[job].model_rung.lower()
