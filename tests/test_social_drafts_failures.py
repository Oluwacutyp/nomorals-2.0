"""Tail-93 pins: nomorals/social/drafts.py census sites.

This module is the owner's morning review surface, and its
renderers carry an explicit contract: "Never raises." Every
site is a degradation rung whose fallback is another TRUE
statement — the plain card, a visible "?" score, a bare count
— never an invented value. All five sites are exclusions:

291 — ``DraftQueue.close`` db-close crash -> pass.
Teardown: the connection is being discarded either way and
nothing reads the outcome. No decision consumes close().
374 — ``review_card`` enrichment crash -> falls back to
``propose_post(draft)[0]``, the module's plain review card.
The owner still gets the draft and the standing question;
only the styling/stats layer is lost.
398 — ``render_review_batch`` per-draft scoring crash ->
that draft renders "[?/100 · ?]": the failure is VISIBLE in
the digest line itself, and the draft is still listed with
its preview. Other drafts score normally.
408 — ``render_review_batch`` outer crash -> the bare count
line "N drafts need review." The count is computed from the
real list; no score or hook is claimed for any draft.
572 — ``handle_draft_callback`` variants action: one
suggested hook's ``add_variant`` crash -> pass, remaining
suggestions still attempted, and the reply is rendered from
a RE-READ of the queue — so the list shown is exactly what
actually got stored, never what was attempted.

NOT-recording is pinned with the snapshot method: drafts.py
calls no failures-ledger record function on these paths, so
both the agency and agents ledgers must be unchanged across
each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.social import drafts as D


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 291: queue close crash -> swallowed at teardown ──────────

def test_291_close_failure_does_not_raise(tmp_path):
    before = _snap()
    queue = D.DraftQueue(tmp_path / "d.db")
    queue._db = SimpleNamespace(close=_raise)
    assert queue.close() is None
    assert _snap() == before


# ── 374: review_card crash -> plain propose_post card ────────

def test_374_review_card_falls_back_to_plain_card():
    before = _snap()
    draft = D.Draft(id="t1", content="a perfectly fine draft body",
                    platforms=["x"])
    with mock.patch("nomorals.social.voice.virality_score",
                    side_effect=RuntimeError("score boom")):
        assert D.review_card(draft) == D.propose_post(draft)[0]
    # control: with scoring alive the styled card carries stats
    assert "Virality" in D.review_card(draft)
    assert _snap() == before


# ── 398: per-draft scoring crash -> visible "?" in the line ──

def test_398_batch_marks_unscored_draft_with_question_marks():
    before = _snap()
    draft = D.Draft(id="t1", content="a perfectly fine draft body",
                    platforms=["x"])
    with mock.patch("nomorals.social.voice.virality_score",
                    side_effect=RuntimeError("score boom")):
        out = D.render_review_batch([draft])
    assert "[?/100 · ?]" in out
    assert "a perfectly fine draft body" in out
    # control: a scored draft shows its real numeric score
    out = D.render_review_batch([draft])
    assert "/100" in out and "[?" not in out
    assert _snap() == before


# ── 408: batch outer crash -> truthful bare count line ───────

def test_408_batch_outer_failure_returns_count_line():
    before = _snap()
    draft = D.Draft(id="t1", content="a perfectly fine draft body",
                    platforms=["x"])
    with mock.patch("nomorals.social.chat.style.section",
                    side_effect=RuntimeError("style boom")):
        assert D.render_review_batch([draft]) == "1 drafts need review."
    assert _snap() == before


# ── 572: one variant add crash -> rest survive, reply truthful ─

def test_572_variant_add_failure_keeps_handler_truthful(tmp_path):
    before = _snap()
    queue = D.DraftQueue(tmp_path / "d.db")
    draft = queue.create_draft("variant test draft body", platforms=["x"])
    suggestions = [{"text": "Hook A\nrest"}, {"text": "Hook B\nrest"}]
    with mock.patch("nomorals.social.voice.suggest_hook_upgrades",
                    return_value=suggestions), \
            mock.patch.object(D, "add_variant",
                              side_effect=RuntimeError("add boom")):
        out = D.handle_draft_callback(queue, f"draft:variants:{draft.id}")
    # the handler answers with the variants prompt; because the
    # reply is rendered from a re-read and every add failed, it
    # lists no variant it failed to store
    assert "hook variants" in out
    assert "Hook A" not in out
    assert (queue.get(draft.id).metadata or {}).get("variants") in (None, [])
    assert _snap() == before
