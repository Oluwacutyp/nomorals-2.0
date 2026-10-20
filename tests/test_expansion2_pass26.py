"""Pass #26 — learn / marketing / codews / social-drafts verified bugs.

1. Mistake notebook split-store: mistakes recorded through the tutor's
   per-chat notebook (``get_notebook``) never reached the on-disk store
   that ``/learn flashcards`` reads, and review grades were never saved —
   learning progress vanished while the UI said "noted for review".
2. ExamSim counted every ``answer()`` call, so re-answering one question
   inflated the score past 100% ("answered 5/3", 166.7%).
3. ``SendEngine.campaign_report`` reported ``spent_kobo: 0`` forever —
   the field was never computed from the campaign's sent records.
4. ``CodeWorkspace.clean(force=True)`` deleted files but reported
   ``removed: []`` (its parser only understood dry-run output).
5. ``detect_explicit_post("post it at 6")`` posted IMMEDIATELY — the
   owner's stated time was silently dropped (only the bare
   "post at 6" form was treated as a schedule).
"""

from __future__ import annotations

import subprocess
from datetime import datetime

import pytest


# ── 1. mistake notebook durability ───────────────────────────────────────


@pytest.fixture()
def nb_dir(tmp_path, monkeypatch):
    import nomorals.learn.flashcards as fc

    monkeypatch.setattr(fc, "notebooks_dir", lambda: tmp_path)
    return tmp_path


def test_tutor_notebook_records_reach_the_disk_store(nb_dir):
    from nomorals.learn import tutor as tutor_mod
    from nomorals.learn.flashcards import MistakeNotebook

    tutor_mod._NOTEBOOKS.pop("pass26-chat", None)
    nb = tutor_mod.get_notebook("pass26-chat")
    card = nb.record("What is 2+2?", "5", "4", topic="fractions")

    # /learn flashcards loads a FRESH notebook from disk — it must see it.
    fresh = MistakeNotebook.load()
    assert fresh.count() == 1
    assert fresh.get(card.id) is not None


def test_review_grade_survives_reload(nb_dir):
    from nomorals.learn.flashcards import MistakeNotebook

    nb = MistakeNotebook.load()
    card = nb.record("Capital of Nigeria?", "Lagos", "Abuja")
    assert nb.review(card.id, 4)

    fresh = MistakeNotebook.load()
    got = fresh.get(card.id)
    assert got is not None
    assert got.reviews == 1
    assert got.last_grade == 4
    assert got.fsrs.reps == 1


def test_chat_flashcards_review_persists(nb_dir):
    from nomorals.learn.chat import control_learn
    from nomorals.learn.flashcards import MistakeNotebook

    nb = MistakeNotebook.load()
    card = nb.record("What is 3+3?", "7", "6")
    out = control_learn(f"flashcards review {card.id} 3")
    assert "graded" in out
    assert "saved" in out  # the reply must not claim a throwaway grade

    fresh = MistakeNotebook.load()
    assert fresh.get(card.id).last_grade == 3


# ── 2. ExamSim: one score per question ────────────────────────────────────


def _sim():
    from nomorals.learn.curriculum import ExamSim

    qs = [{"q": "2+2?", "answer": "4"}, {"q": "3+3?", "answer": "6"},
          {"q": "5+5?", "answer": "10"}]
    sim = ExamSim("JAMB", "Mathematics", qs, minutes=30)
    sim.start()
    return sim


def test_exam_sim_reanswering_does_not_inflate_score():
    sim = _sim()
    for _ in range(5):
        sim.answer(0, "4")
    r = sim.finish()
    assert r["answered"] == 1
    assert r["total"] == 3
    assert r["score"] == 1.0
    assert r["pct"] <= 100.0


def test_exam_sim_changed_answer_last_one_counts():
    sim = _sim()
    sim.answer(0, "5")          # wrong first
    r = sim.answer(0, "4")      # changed mind -> correct
    assert r["verdict"] == "correct"
    result = sim.finish()
    assert result["answered"] == 1
    assert result["correct"] == 1
    assert result["score"] == 1.0


# ── 3. campaign_report spend ──────────────────────────────────────────────


def test_campaign_report_spent_kobo_matches_sent_records():
    from nomorals.marketing.send import SendEngine

    eng = SendEngine(":memory:")
    # smtp kind prices at 40 kobo/msg (COST_PER_MSG_KOBO).
    eng.register_provider("p1", kind="smtp", rate_per_minute=1000,
                          sender=lambda to, s, b: True)
    eng.save_template("t1", "Hello", subject="Hi")
    res = eng.schedule_campaign("t1", ["a@x.com", "b@x.com"], "p1", 0.0)
    stats = eng.process()
    assert stats["sent"] == 2
    rep = eng.campaign_report(res["campaign_id"])
    assert rep["by_status"].get("sent") == 2
    assert rep["spent_kobo"] == 80


# ── 4. codews clean(force=True) reports what it removed ───────────────────


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    (repo / "a.txt").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def test_clean_force_reports_removed_files(tmp_path):
    from nomorals.codews.workspace import CodeWorkspace

    repo = _git_repo(tmp_path)
    (repo / "junk.tmp").write_text("x")
    ws = CodeWorkspace(repo)
    out = ws.clean(force=True)
    assert out["dry_run"] is False
    assert out["removed"] == ["junk.tmp"]
    assert not (repo / "junk.tmp").exists()


# ── 5. explicit-post: a stated time is a schedule, even with it/this ──────


def test_post_it_at_a_time_schedules_instead_of_posting_now():
    from nomorals.social.drafts import detect_explicit_post

    for text in ("post it at 6", "post this at 6pm", "post that at 18:30"):
        got = detect_explicit_post(text)
        assert got is not None and got["action"] == "schedule", text
        assert got["at"] is not None, text


def test_post_it_without_a_time_still_posts_now():
    from nomorals.social.drafts import detect_explicit_post

    got = detect_explicit_post("post it")
    assert got is not None and got["action"] == "post"


def test_parse_post_time_allows_it_this_that():
    from nomorals.social.drafts import parse_post_time

    now = datetime(2026, 10, 8, 9, 0)
    ts = parse_post_time("post it at 6", now=now)
    assert ts is not None
    assert datetime.fromtimestamp(ts).strftime("%H:%M") == "18:00"
