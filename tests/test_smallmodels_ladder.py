"""Models-into-classes ladder (Phase 1): registry, per-device selection,
and the four wired jobs — character dialogue, character tactics, DJ sound
selection, arena battle narration.

Failing-first: every test here names behaviour the ladder must provide.
The code rung proves each job deterministically; the model rung proves
selection + honest fallback when no model is available.
"""

from __future__ import annotations

import json

import pytest

from nomorals.core.profile import resolve_profile

# ── fake brain ───────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _FakeBrain:
    """A Brain-shaped stand-in: available() + chat() -> resp.text."""

    def __init__(self, available: bool = True,
                 replies: list[str] | None = None) -> None:
        self._available = available
        self._replies = list(replies or [])
        self.calls: list[dict] = []

    def available(self) -> bool:
        return self._available

    def chat(self, messages, *, task_kind: str = "",
             timeout_s: float | None = None, **_kw):
        self.calls.append({
            "prompt": messages[-1].content if messages else "",
            "task_kind": task_kind,
            "timeout_s": timeout_s,
        })
        text = self._replies.pop(0) if self._replies else ""
        return _FakeResp(text)


# ── registry ─────────────────────────────────────────────────────────────

def test_four_jobs_registered():
    from nomorals.llm.ladder import JOB_NAMES, list_jobs

    # JOB_NAMES is the Phase-1 set (kept stable); the registry grew in
    # Phase 2 with the real specialists (owner correction 2026-11-12:
    # the brain is not a rung — the specialist is).
    assert set(JOB_NAMES) == {
        "character_dialogue",
        "character_tactics",
        "dj_sound_selection",
        "arena_narration",
    }
    jobs = set(list_jobs())
    assert set(JOB_NAMES) <= jobs
    for specialist in ("npc_line_scorer", "npc_tactics", "pumpdesk_win",
                       "error_triage", "morning_rank",
                       "character_voice_xtts", "character_voice_rvc",
                       "character_lora", "character_face"):
        assert specialist in jobs


def test_unknown_job_is_an_error():
    from nomorals.llm.ladder import resolve

    with pytest.raises(ValueError):
        resolve("not_a_job")


def test_register_job_extends_registry():
    from nomorals.llm.ladder import JobSpec, list_jobs, register_job

    register_job(JobSpec(
        name="test_only_job",
        description="a job registered by this test",
        model_rung="Brain chat",
        code_rung="deterministic fallback",
    ))
    assert "test_only_job" in list_jobs()


# ── resolution ───────────────────────────────────────────────────────────

def test_resolve_model_rung_with_brain():
    # Phase 2 correction: the brain alone NEVER promotes a job to the
    # model rung — the rung is decided by the specialist. The brain is
    # the candidate generator (dialogue), wired as ``suggest`` only
    # when the specialist is trained.
    from nomorals.llm.ladder import resolve

    brain = _FakeBrain(available=True, replies=["hello there"])
    res = resolve("character_dialogue", brain=brain,
                  profile=resolve_profile("workstation"))
    assert res.rung == "code"
    # the generator flows on either rung — it is not the rung; the
    # hand-weight scorer picks its candidates on the code rung
    assert res.suggest is not None
    assert res.suggest("say hi") == "hello there"
    res = resolve("character_dialogue", brain=brain,
                  profile=resolve_profile("workstation"),
                  specialist_available=True)
    assert res.rung == "model"
    assert res.suggest is not None
    assert "scorer" in res.model_rung.lower()
    assert brain.calls[-1]["timeout_s"] == 60.0


def test_resolve_model_timeout_is_per_profile():
    from nomorals.llm.ladder import resolve

    brain = _FakeBrain(available=True, replies=["x", "x"])
    termux = resolve("character_dialogue", brain=brain,
                     profile=resolve_profile("termux"),
                     specialist_available=True)
    termux.suggest("hi")
    assert brain.calls[-1]["timeout_s"] == 20.0
    pc = resolve("character_dialogue", brain=brain,
                 profile=resolve_profile("pc"),
                 specialist_available=True)
    pc.suggest("hi")
    assert brain.calls[-1]["timeout_s"] == 30.0


def test_resolve_code_rung_no_brain():
    from nomorals.llm.ladder import resolve

    res = resolve("character_dialogue", brain=None,
                  profile=resolve_profile("pc"))
    assert res.rung == "code"
    assert res.suggest is None
    assert "specialist" in res.reason.lower()


def test_resolve_code_rung_brain_unavailable():
    from nomorals.llm.ladder import resolve

    brain = _FakeBrain(available=False)
    res = resolve("arena_narration", brain=brain,
                  profile=resolve_profile("workstation"))
    assert res.rung == "code"
    assert res.suggest is None


def test_resolve_suggest_never_raises():
    from nomorals.llm.ladder import resolve

    class _BoomBrain(_FakeBrain):
        def chat(self, messages, **_kw):
            raise RuntimeError("provider exploded")

    res = resolve("character_dialogue", brain=_BoomBrain(available=True),
                  profile=resolve_profile("pc"),
                  specialist_available=True)
    assert res.rung == "model"
    assert res.suggest("hi") == ""


def test_resolve_model_available_override():
    from nomorals.llm.ladder import resolve

    # The DJ job's model rung is a learned model, not the brain: the
    # caller supplies its availability, and the rung follows it.
    res = resolve("dj_sound_selection", brain=None,
                  model_available=lambda: True)
    assert res.rung == "model"
    assert res.suggest is None  # no brain -> no suggest fn; the caller
    # owns the learned taste object.
    res2 = resolve("dj_sound_selection", brain=None,
                   model_available=lambda: False)
    assert res2.rung == "code"


# ── local-model recommendation ───────────────────────────────────────────

def test_recommend_local_model_is_per_profile():
    from nomorals.llm.ladder import recommend_local_model

    termux = recommend_local_model(resolve_profile("termux"))
    assert "1.5b" in termux.repo_id.lower() or "1-5b" in termux.repo_id.lower()
    assert termux.params <= 2_000_000_000
    ws = recommend_local_model(resolve_profile("workstation"))
    assert ws.params >= 8_000_000_000
    pc = recommend_local_model(resolve_profile("pc"))
    assert 2_000_000_000 <= pc.params <= 8_000_000_000


def test_local_model_available_is_honest_when_absent(tmp_path):
    from nomorals.llm.ladder import local_model_available

    ok, reason = local_model_available(
        "cognitivecomputations/dolphin-2.9.3-qwen2.5-1.5b",
        cache_dir=str(tmp_path))
    assert ok is False
    assert reason  # names why: no weights on disk


# ── job 1: character dialogue ────────────────────────────────────────────

def _npc(name: str = "Sage Marlowe"):
    from nomorals.games.npc import NPCProfile

    return NPCProfile(id="arena_sage", name=name, game_id="arena")


def test_dialogue_model_line_when_brain_present():
    from nomorals.games.gamemaster import GameMaster

    gm = GameMaster(suggest=lambda _p: "Ahoy, sailor.")
    line = gm.npc_speak(_npc(), "the tavern is on fire")
    assert "Ahoy, sailor." in line


def test_dialogue_template_fallback_when_no_model():
    from nomorals.games.gamemaster import GameMaster

    gm = GameMaster(suggest=None)
    line = gm.npc_speak(_npc(), "the tavern is on fire")
    assert line  # always a voice, never empty
    assert "Sage Marlowe" in line


def test_dialogue_via_ladder_resolve():
    from nomorals.games.gamemaster import GameMaster
    from nomorals.llm.ladder import resolve

    brain = _FakeBrain(available=True, replies=["Well met, traveller."])
    res = resolve("character_dialogue", brain=brain,
                  profile=resolve_profile("pc"),
                  specialist_available=True)
    gm = GameMaster(suggest=res.suggest)
    assert gm.model_on is True
    assert "Well met" in gm.npc_speak(_npc(), "hello")


# ── job 2: character tactics ─────────────────────────────────────────────

def _stats(hp: int = 100, max_hp: int = 100, atk: int = 20, **kw):
    d = {"hp": hp, "max_hp": max_hp, "atk": atk, "def": 10,
         "potions": 0, "fury_cd": 0, "focused": False}
    d.update(kw)
    return d


def test_tactics_math_is_the_code_rung():
    from nomorals.games.ai import GameMind

    mind = GameMind()  # no model: pure math, deterministic
    assert mind.combat_move(_stats(), _stats(hp=50, atk=5)) == {"action": "fury"}
    assert mind.combat_move(_stats(hp=20, potions=1),
                            _stats(hp=50)) == {"action": "potion"}


def test_tactics_model_can_override_math():
    # Phase 2 correction: the combat DECISION is the learned policy, not
    # the brain. A trained policy overrides the math; the brain never
    # picks an action.
    from nomorals.games.ai import GameMind

    mind = GameMind(policy=lambda _state: "defend")
    # healthy + fury ready: math says fury, the policy says defend
    assert mind.combat_move(_stats(), _stats(hp=50, atk=5)) == {"action": "defend"}


def test_tactics_model_garbage_falls_back_to_math():
    from nomorals.games.ai import GameMind

    mind = GameMind(policy=lambda _state: "teleport behind the moon")
    assert mind.combat_move(_stats(), _stats(hp=50, atk=5)) == {"action": "fury"}
    mind2 = GameMind(policy=lambda _state: None)
    assert mind2.combat_move(_stats(), _stats(hp=50, atk=5)) == {"action": "fury"}


def test_tactics_model_never_invents_an_action():
    from nomorals.games.ai import GameMind

    for reply in ("FURY", "  potion ", "Focus!", "ATTACK"):
        mind = GameMind(policy=lambda _state, r=reply: r)
        got = mind.combat_move(_stats(hp=20, potions=1), _stats(hp=50))
        assert got["action"] in ("attack", "focus", "fury", "defend",
                                 "potion")


def test_tactics_brain_never_picks_an_action():
    # The correction, pinned: a brain suggest on the mind does NOT
    # decide combat — the math carries until a policy is trained.
    from nomorals.games.ai import GameMind

    mind = GameMind(suggest=lambda _p: "defend")
    assert mind.combat_move(_stats(), _stats(hp=50, atk=5)) == {"action": "fury"}


# ── job 3: DJ sound selection ────────────────────────────────────────────

def _trk(bpm: float = 120.0, energy: float = 0.5, key: str = "C",
         mode: str = "major"):
    from nomorals.media.dj_engine import TrackAnalysis

    return TrackAnalysis(path="/x/a.wav", title="a", bpm=bpm, key=key,
                         mode=mode, camelot="8B", energy=energy)


def test_dj_learned_untrained_without_data(tmp_path):
    from nomorals.media.dj_engine import LearnedTaste

    taste = LearnedTaste()
    assert taste.trained is False
    assert taste.fit_from_log(str(tmp_path / "nope.jsonl")) is False
    assert taste.trained is False


def _feedback_rows():
    # kept rows: high harmonic fit + blend; skipped rows: poor fit.
    rows = []
    for _ in range(6):
        rows.append({"event": "kept", "features": {
            "bpm_delta_pct": 0.02, "harmonic_score": 0.9,
            "energy_delta": 0.1, "energy_abs_b": 0.7,
            "transition_kind": "blend", "same_style": 1}})
    for _ in range(6):
        rows.append({"event": "skipped", "features": {
            "bpm_delta_pct": 0.25, "harmonic_score": 0.1,
            "energy_delta": -0.4, "energy_abs_b": 0.3,
            "transition_kind": "echo-drop", "same_style": 0}})
    return rows


def test_dj_learned_trains_and_ranks_with_the_data():
    from nomorals.media.dj_engine import (
        HeuristicTaste,
        LearnedTaste,
        plan_transition,
    )

    taste = LearnedTaste()
    assert taste.fit(_feedback_rows()) is True
    assert taste.trained is True
    good = (_trk(bpm=120.0, energy=0.6), _trk(bpm=122.0, energy=0.7))
    bad = (_trk(bpm=120.0, energy=0.6), _trk(bpm=150.0, energy=0.2))
    good_score = taste.score_transition(
        good[0], good[1], plan_transition(good[0], good[1]))
    bad_score = taste.score_transition(
        bad[0], bad[1], plan_transition(bad[0], bad[1]))
    assert 0.0 <= good_score <= 1.0
    assert 0.0 <= bad_score <= 1.0
    assert good_score > bad_score
    # the code rung still scores deterministically either way
    assert 0.0 <= HeuristicTaste().score_transition(
        good[0], good[1], plan_transition(good[0], good[1])) <= 1.0


def test_dj_learned_fit_from_log_file(tmp_path):
    from nomorals.media.dj_engine import LearnedTaste

    path = tmp_path / "taste-feedback.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _feedback_rows()),
                    encoding="utf-8")
    taste = LearnedTaste()
    assert taste.fit_from_log(str(path)) is True
    assert taste.trained is True


def test_dj_select_taste_prefers_learned_when_trained(tmp_path):
    from nomorals.media.dj_engine import select_taste

    path = tmp_path / "taste-feedback.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _feedback_rows()),
                    encoding="utf-8")
    taste, rung = select_taste(feedback_path=str(path))
    assert rung == "model"
    assert taste.trained is True
    from nomorals.media.dj_engine import HeuristicTaste
    taste2, rung2 = select_taste(feedback_path=str(tmp_path / "empty.jsonl"))
    assert rung2 == "code"
    assert isinstance(taste2, HeuristicTaste)


# ── job 4: arena battle narration ────────────────────────────────────────

def test_narration_model_line_when_brain_present():
    # Phase 2 correction: there is NO narration specialist — the
    # template bank carries even when a brain is attached. The brain
    # never narrates as a rung.
    from nomorals.games.gamemaster import GameMaster

    gm = GameMaster(suggest=lambda _p: "The blade sings.")
    line = gm.narrate("arena", "Ada lands a killing crit on the Hollow Warden")
    assert line  # template bank, mood-parameterized, never empty
    assert "The blade sings." not in line


def test_narration_template_bank_when_no_model():
    from nomorals.games.gamemaster import GameMaster

    gm = GameMaster(suggest=None)
    line = gm.narrate("arena", "Ada lands a killing crit on the Hollow Warden")
    assert line  # template bank, mood-parameterized, never empty


def test_narration_via_ladder_resolve():
    # Phase 2 correction: arena_narration is honestly unavailable —
    # no narration specialist exists — so the ladder resolves to the
    # code rung even with a brain present, and the reason says so.
    from nomorals.games.gamemaster import GameMaster
    from nomorals.llm.ladder import resolve

    brain = _FakeBrain(available=True, replies=["Dust settles. Ada stands."])
    res = resolve("arena_narration", brain=brain,
                  profile=resolve_profile("pc"))
    assert res.rung == "code"
    assert "narration" in res.reason.lower()
    assert res.suggest is None
    gm = GameMaster(suggest=res.suggest)
    assert "Dust settles" not in gm.narrate(
        "arena", "Ada lands a killing crit on the Hollow Warden")


def test_narration_code_rung_when_no_brain():
    from nomorals.games.gamemaster import GameMaster
    from nomorals.llm.ladder import resolve

    res = resolve("arena_narration", brain=None,
                  profile=resolve_profile("termux"))
    assert res.rung == "code"
    gm = GameMaster(suggest=res.suggest)
    assert gm.model_on is False
    line = gm.narrate("arena", "Ada lands a killing crit on the Hollow Warden")
    assert line
