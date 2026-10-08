"""Build-map #86: Seer as form coach. All offline (mock vision seam)."""

import os
import tempfile

import pytest

from nomorals.health.form import (
    CHECKPOINTS,
    FIXES,
    FORM_DISCLAIMER,
    GAIT_RISKS,
    MOVEMENTS,
    FormAnalysis,
    FormStore,
    analyze_form,
    analyze_gait,
    augment_today_workout,
    control_form,
    mobility_for,
)
from nomorals.health.training import TrainingCoach, generate_plan


def _img(suffix=".jpg"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    return path


def _flag_vision(flag_cid, observed="knees caving inward on the descent"):
    def vision(path, question):
        lines = []
        for cid in CHECKPOINTS["squat"]:
            if cid == flag_cid:
                lines.append(f"{cid}: FLAG — {observed}")
            else:
                lines.append(f"{cid}: PASS — looks fine")
        return "\n".join(lines)
    return vision


def _pass_vision(exercise):
    def vision(path, question):
        return "\n".join(
            f"{cid}: PASS — looks good" for cid in CHECKPOINTS[exercise])
    return vision


def _gait_vision(flag_rid=None):
    def vision(path, question):
        lines = []
        for rid in GAIT_RISKS:
            if rid == flag_rid:
                lines.append(f"{rid}: FLAG — foot well ahead of hips")
            else:
                lines.append(f"{rid}: PASS — fine")
        return "\n".join(lines)
    return vision


# ── 5 movements ──────────────────────────────────────────────────────────

def test_all_five_movements_have_checkpoints_and_fixes():
    assert set(MOVEMENTS) == {"squat", "deadlift", "push-up", "plank",
                              "lunge"}
    for ex, cps in CHECKPOINTS.items():
        assert len(cps) >= 3, ex
        for cid in cps:
            assert (ex, cid) in FIXES, (ex, cid)


@pytest.mark.parametrize("exercise", list(MOVEMENTS))
def test_clean_form_all_pass(exercise):
    a = analyze_form(_img(), exercise, vision=_pass_vision(exercise))
    assert a.available
    assert a.band == "solid"
    assert not a.issues
    assert len(a.passes) == len(CHECKPOINTS[exercise])
    assert "medical advice" in a.format()


def test_knees_caving_flagged_with_fix():
    a = analyze_form(_img(), "squat",
                     vision=_flag_vision("knees", "knees caving in on rep 3"))
    assert a.available
    assert len(a.issues) == 1
    issue = a.issues[0]
    assert issue.checkpoint == "knees"
    assert "rep 3" in issue.observed
    assert "spread" in issue.fix  # the concrete fix
    out = a.format()
    assert "knees caving in on rep 3" in out
    assert "medical advice" in out


def test_multiple_flags_and_band():
    def vision(path, question):
        return ("knees: FLAG — caving\n"
                "depth: FLAG — high\n"
                "spine: PASS — neutral\n"
                "feet: PASS — planted")
    a = analyze_form(_img(), "squat", vision=vision)
    assert len(a.issues) == 2
    assert a.band == "needs work"  # 2/4 passed


def test_unknown_exercise_honest():
    a = analyze_form(_img(), "burpee", vision=_pass_vision("squat"))
    assert not a.available
    assert "squat" in a.note


def test_missing_video_honest():
    a = analyze_form("/tmp/definitely_not_here_xyz.mp4", "squat",
                     vision=_pass_vision("squat"))
    assert not a.available
    assert "no video found" in a.note


def test_vision_failure_honest():
    def bad_vision(path, question):
        raise RuntimeError("no vision today")
    a = analyze_form(_img(), "squat", vision=bad_vision)
    assert not a.available


def test_garbage_vision_text_honest():
    def junk(path, question):
        return "I see a person exercising, looks fine overall!"
    a = analyze_form(_img(), "squat", vision=junk)
    assert not a.available
    assert "usable checkpoint" in a.note


def test_never_raises_on_garbage():
    a = analyze_form(None, None, vision=None)  # noqa
    assert isinstance(a, FormAnalysis)
    a2 = analyze_gait(None, vision=None)
    assert not a2.available


# ── gait ─────────────────────────────────────────────────────────────────

def test_gait_clean():
    g = analyze_gait(_img(), vision=_gait_vision())
    assert g.available
    assert not g.issues
    assert "medical advice" in g.format()


def test_gait_overstride_flagged():
    g = analyze_gait(_img(), vision=_gait_vision("overstride"))
    assert g.available
    assert len(g.issues) == 1
    assert g.issues[0].risk == "overstride"
    assert "cadence" in g.issues[0].fix
    assert "overstriding" in g.format()


def test_gait_missing_video():
    g = analyze_gait("/tmp/nope_xyz.mp4", vision=_gait_vision())
    assert not g.available


# ── #85 integration ──────────────────────────────────────────────────────

def test_mobility_for_squat_depth():
    a = analyze_form(_img(), "squat",
                     vision=_flag_vision("depth", "well above parallel"))
    mob = mobility_for(a)
    assert mob
    assert any("squat" in e.name.lower() for e in mob)


def test_augment_today_workout_adds_mobility():
    coach = TrainingCoach(
        store_path=os.path.join(tempfile.mkdtemp(), "t.db"))
    a = analyze_form(_img(), "squat",
                     vision=_flag_vision("depth", "above parallel"))
    w = augment_today_workout(coach, a)
    assert w.exercises
    assert w.exercises[0].name == "Deep squat hold"
    assert "form coach" in w.gate_note
    assert "squat" in w.gate_note


def test_augment_clean_form_no_change():
    coach = TrainingCoach(
        store_path=os.path.join(tempfile.mkdtemp(), "t.db"))
    a = analyze_form(_img(), "squat", vision=_pass_vision("squat"))
    w = augment_today_workout(coach, a)
    assert "form coach" not in (w.gate_note or "")


# ── store ────────────────────────────────────────────────────────────────

def test_store_roundtrip():
    store = FormStore(db_path=os.path.join(tempfile.mkdtemp(), "f.db"))
    a = analyze_form(_img(), "deadlift", vision=_pass_vision("deadlift"),
                     store=store)
    assert store.save(a)
    got = store.get(a.id)
    assert got is not None and got.exercise == "deadlift"
    assert store.latest("deadlift").id == a.id
    assert store.get("nope") is None


# ── chat ─────────────────────────────────────────────────────────────────

def _ctx(vision):
    store = FormStore(db_path=os.path.join(tempfile.mkdtemp(), "f.db"))
    return {"vision": vision, "store": store}


def test_chat_help():
    assert "squat" in control_form("")


def test_chat_movements():
    assert "deadlift" in control_form("movements")


def test_chat_analyze():
    out = control_form(
        f"analyze {_img()} squat",
        vision=_flag_vision("knees", "knees caving in on rep 3"))
    assert "knees caving" in out
    assert "analysis id:" in out
    assert "medical advice" in out


def test_chat_analyze_usage():
    assert "usage" in control_form("analyze onlyone")


def test_chat_gait():
    out = control_form(f"gait {_img()}",
                       vision=_gait_vision("overstride"))
    assert "overstriding" in out


def test_chat_apply():
    store = FormStore(db_path=os.path.join(tempfile.mkdtemp(), "f.db"))
    a = analyze_form(_img(), "squat",
                     vision=_flag_vision("depth", "above parallel"),
                     store=store)
    coach = TrainingCoach(
        store_path=os.path.join(tempfile.mkdtemp(), "t.db"))
    out = control_form(f"apply {a.id}", coach=coach, store=store)
    assert "Deep squat hold" in out
    assert "medical advice" in out


def test_chat_apply_unknown_id():
    store = FormStore(db_path=os.path.join(tempfile.mkdtemp(), "f.db"))
    assert "no analysis" in control_form("apply xyz", store=store)


def test_chat_never_raises():
    assert control_form(None) is not None
    assert control_form("analyze", vision=None) is not None


def test_guard_coaching_on_outputs():
    a = analyze_form(_img(), "squat",
                     vision=_flag_vision("spine", "back rounding"))
    out = a.format()
    for phrase in ("you have a ", "you suffer", "disease", "disorder",
                   "syndrome"):
        assert phrase not in out.lower()
