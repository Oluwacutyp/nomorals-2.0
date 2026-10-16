"""TEN_PACKAGES Tier 0 wiring: nm learn / nm legal / /learn / /legal /
matching surfaces.

All handlers delegate to existing learn/ / legal/ / matching/
functions (import, don't reimplement). Legal outputs must all carry
the "not legal advice" disclaimer; citations (DoNotPay FTC $193K, NBA
2024 AI guidelines) are surfaced as citations, never as advice.
"""

from __future__ import annotations

import types

import pytest

from nomorals.cmdline.commands.learn import _cmd_learn
from nomorals.cmdline.commands.legal import _cmd_legal
from nomorals.cmdline.dispatch import _COMMANDS
from nomorals.cmdline.parser import _parser
from nomorals.learn.chat import control_learn
from nomorals.legal.chat import control_legal_full
from nomorals.matching import adapter_community as community
from nomorals.matching import adapter_gig as gig
from nomorals.matching import adapter_learning as learning
from nomorals.matching.batches import Candidate, DailyBatchStore
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    parse_control,
)

TENANCY_TEXT = (
    "This tenancy agreement between landlord and tenant: the landlord "
    "may increase the rent at any time without notice, and the tenant "
    "must pay an agency fee of 15 percent of the annual rent, plus a "
    "service charge that is not stated anywhere in this agreement."
)


def _ns(*words: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(task=list(words))


@pytest.fixture()
def _tmp_dbs(monkeypatch, tmp_path):
    """Point the legal SQLite stores + flashcards at tmp paths.

    The legal portfolio/regwatch stores default to ~/.nomorals; the
    flashcards notebook does too. Tests must never touch the real home.
    """
    import nomorals.legal.portfolio as portfolio
    import nomorals.legal.regulatory as regulatory
    import nomorals.learn.flashcards as flashcards

    monkeypatch.setattr(portfolio, "_DEFAULT_DB", str(tmp_path / "p.db"))
    monkeypatch.setattr(regulatory, "_DEFAULT_DB", str(tmp_path / "r.db"))
    monkeypatch.setattr(flashcards, "notebooks_dir", lambda: tmp_path)
    yield tmp_path


# ── CLI registration ─────────────────────────────────────────────────────


def test_cli_parser_has_learn_and_legal():
    parser = _parser()
    args = parser.parse_args(["learn", "waec", "physics"])
    assert args.command == "learn"
    assert args.task == ["waec", "physics"]
    args = parser.parse_args(["legal", "citations"])
    assert args.command == "legal"
    assert args.task == ["citations"]


def test_cli_dispatch_has_learn_and_legal():
    assert "learn" in _COMMANDS
    assert "legal" in _COMMANDS


def test_nm_learn_no_args_is_usage_error(capsys):
    assert _cmd_learn(_ns(), None) == 2
    assert "nm learn" in capsys.readouterr().err


def test_nm_legal_no_args_is_usage_error(capsys):
    assert _cmd_legal(_ns(), None) == 2
    assert "nm legal" in capsys.readouterr().err


def test_nm_learn_waec(capsys):
    assert _cmd_learn(_ns("waec", "physics"), None) == 0
    assert "Physics" in capsys.readouterr().out


def test_nm_legal_citations(capsys):
    assert _cmd_legal(_ns("citations"), None) == 0
    out = capsys.readouterr().out
    assert "DoNotPay" in out and "NBA" in out
    assert "not legal advice" in out


# ── chat registration ────────────────────────────────────────────────────


def test_chat_has_learn_command():
    assert "learn" in CONTROL_COMMANDS
    cmd = parse_control("/learn tutor fractions")
    assert cmd is not None and cmd.kind == "learn"
    assert COMMAND_DETAILS["learn"]["usage"].startswith("/learn tutor")
    # /legal still parses and its catalog entry documents the subcommands.
    cmd = parse_control("/legal aid pcm my landlord don lock me out")
    assert cmd is not None and cmd.kind == "legal"
    assert "contract" in COMMAND_DETAILS["legal"]["usage"]


# ── /learn ───────────────────────────────────────────────────────────────


def test_learn_tutor_start_status_answer():
    key = "tier0-tutor-1"
    out = control_learn("tutor fractions", chat_key=key)
    assert "fractions" in out
    status = control_learn("tutor status", chat_key=key)
    assert "fractions" in status
    ans = control_learn("tutor answer 1/2", chat_key=key)
    assert ans.strip()
    end = control_learn("tutor stop", chat_key=key)
    assert "ended" in end


def test_learn_mastery_without_session_is_honest():
    out = control_learn("mastery", chat_key="tier0-no-session")
    assert "no tutoring session" in out


def test_learn_mastery_after_start():
    key = "tier0-mastery-1"
    control_learn("tutor fractions", chat_key=key)
    out = control_learn("mastery", chat_key=key)
    assert "mastery" in out and "fractions" in out
    control_learn("tutor stop", chat_key=key)


def test_learn_waec_and_jamb():
    waec = control_learn("waec physics")
    assert "Physics" in waec and "WAEC" in waec
    grade = control_learn("waec grade 72")
    assert grade.strip()
    jamb = control_learn("jamb physics")
    assert "JAMB" in jamb
    formats = control_learn("jamb formats")
    assert "JAMB" in formats


def test_learn_quiz_formats_real_but_start_honest():
    formats = control_learn("quiz formats")
    assert "JAMB" in formats or "WAEC" in formats
    start = control_learn("quiz start jamb physics")
    assert "not available" in start
    assert "question bank" in start


def test_learn_flashcards_stats_and_review(_tmp_dbs, tmp_path):
    from nomorals.learn.flashcards import MistakeNotebook

    nb = MistakeNotebook()
    card = nb.record("q1", "wrong answer", "right answer", topic="fractions")
    nb.save(tmp_path / "default.json")
    stats = control_learn("flashcards stats")
    assert "flashcards" in stats
    due = control_learn("flashcards due")
    assert card.id in due
    graded = control_learn(f"flashcards review {card.id} 4")
    assert "graded" in graded
    unknown = control_learn("flashcards review nope-123 3")
    assert "no card" in unknown


def test_learn_unknown_verb_is_usage():
    assert "/learn" in control_learn("frobnicate")


# ── /legal ───────────────────────────────────────────────────────────────


def test_legal_aid_in_five_languages():
    for lang in ("en", "pcm", "yo", "ha", "ig"):
        out = control_legal_full(f"aid {lang} my landlord locked me out")
        assert "not legal advice" in out, lang


def test_legal_bare_question_keeps_aid_behaviour():
    out = control_legal_full("my landlord locked me out")
    assert "not legal advice" in out


def test_legal_contract_types_and_review():
    types = control_legal_full("contract types")
    assert "tenancy" in types
    review = control_legal_full(f"contract review tenancy {TENANCY_TEXT}")
    assert "Grade" in review
    assert "not legal advice" in review
    # Nigerian playbook reference, not a generic template.
    assert "Lagos Tenancy Law" in review or "Lagos" in review


def test_legal_portfolio(_tmp_dbs):
    summary = control_legal_full("portfolio")
    assert "portfolio" in summary.lower()
    assert "not legal advice" in summary
    attention = control_legal_full("portfolio attention 60")
    assert "not legal advice" in attention


def test_legal_regulatory(_tmp_dbs):
    regs = control_legal_full("regulatory regulators")
    assert "CBN" in regs and "SEC" in regs and "NDPA" in regs and "FIRS" in regs
    assert "cbn.gov.ng" in regs
    assert "not legal advice" in regs
    added = control_legal_full("regulatory add CBN fintech")
    assert "Watching CBN" in added
    assert "not legal advice" in added


def test_legal_citations_are_citations_not_advice():
    out = control_legal_full("citations")
    assert "DoNotPay" in out and "193" in out and "NBA" in out
    assert "not legal advice" in out
    # Citations explain the information-only rule; they never claim
    # Devon is a lawyer.
    assert "never claims to be" in out or "never" in out


def test_legal_help_lists_subcommands():
    usage = control_legal_full("help")
    for verb in ("contract", "portfolio", "regulatory", "aid", "citations"):
        assert verb in usage
    assert "not legal advice" in usage


# ── matching: three documented surfaces ──────────────────────────────────


def test_matching_surfaces_are_documented():
    assert gig.SURFACE == "gig"
    assert community.SURFACE == "community"
    assert learning.SURFACE == "learning"


def test_matching_gig_daily_and_stable():
    cands = [
        Candidate(candidate_id="a", title="Gig A", tags=("python",),
                  quality=0.9),
        Candidate(candidate_id="b", title="Gig B", tags=("design",),
                  quality=0.2),
    ]
    picks = gig.daily_picks(cands, {"python": 10.0}, n=1)
    assert [c.candidate_id for c in picks] == ["a"]
    assert "python" in gig.explain(cands[0], {"python": 10.0})
    result = gig.stable_match_for(
        ["f1"], ["c1"], {"f1": ["c1"]}, {"c1": ["f1"]})
    assert result["f1"] == "c1"
    stable, _ = gig.verify(result, {"f1": ["c1"]}, {"c1": ["f1"]})
    assert stable


def test_matching_gig_questionnaire_and_dealbreakers(tmp_path):
    q = gig.new_questionnaire("owner", db_path=str(tmp_path / "g.db"))
    assert q.unanswered()
    q.set_dealbreaker("budget", "rate", "le", 100)
    kept = gig.filter_dealbreakers(
        q, [{"attributes": {"rate": 50}}, {"attributes": {"rate": 500}}])
    assert kept == [{"attributes": {"rate": 50}}]


def test_matching_community_roommates():
    prefs = {
        "a": ["b", "c", "d"], "b": ["a", "c", "d"],
        "c": ["d", "a", "b"], "d": ["c", "a", "b"],
    }
    pairs = community.pair_members(["a", "b", "c", "d"], prefs)
    assert pairs["a"] == "b" and pairs["c"] == "d"
    stable, _ = community.verify_pairing(dict(pairs), prefs)
    assert stable


def test_matching_learning_capacity_and_assignment(tmp_path):
    result = learning.capacity_match(
        ["m1", "m2"], ["mentor"],
        {"m1": ["mentor"], "m2": ["mentor"]},
        {"mentor": ["m1", "m2"]},
        {"mentor": 2})
    assert result["m1"] == "mentor" and result["m2"] == "mentor"
    assign = learning.assign({"s1": {"slot1": 9.0, "slot2": 1.0},
                              "s2": {"slot1": 1.0, "slot2": 9.0}})
    assert assign == {"s1": "slot1", "s2": "slot2"}
    q = learning.new_questionnaire("owner", db_path=str(tmp_path / "l.db"))
    assert q.unanswered()


def test_matching_daily_batch_store_per_surface(tmp_path):
    store = DailyBatchStore(db_path=str(tmp_path / "b.db"))
    for surface, adapter in (("gig", gig), ("community", community),
                             ("learning", learning)):
        assert store.add_candidate(
            Candidate(candidate_id=f"{surface}-1", title=f"{surface} pick",
                      quality=0.8), surface=surface)
        batch = adapter.todays_batch(store, owner="owner", n=5)
        assert [c.candidate_id for c in batch] == [f"{surface}-1"]
        # Stable all day: second call returns the same batch.
        assert adapter.todays_batch(store, owner="owner", n=5) == batch
