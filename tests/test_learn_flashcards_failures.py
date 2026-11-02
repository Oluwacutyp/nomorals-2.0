"""Tail-103 pins: nomorals/learn/flashcards.py census sites.

All four sites guard the FSRS-lite math (``due`` /
``retrievability``) inside ``MistakeNotebook`` and are held
as exclusions. The class contract is printed in its own
docstring — "Never raises: a broken scheduler must not
break tutoring" — and every failure path degrades
fail-closed or to the module's own honest default:

361 — ``due_cards``: a card whose FSRS due/retrievability
computation raises is skipped for this round. The card is
not presented — less practice, never a fabricated due.

380 — ``notebook``: retrievability computation raises ->
the entry reports 0.0, which is exactly the value the
module's legitimate path reports for a never-reviewed
card (``if card.fsrs.reps else 0.0``). Same default, same
meaning: no measurable retrievability.

388 — ``_due``: ``fsrs.due`` raises -> False (not due).
Fail-closed: the card is withheld, not surfaced early.
Cards with ``reps == 0`` short-circuit to True without
touching the FSRS object at all.

413 — ``stats``: a card whose retrievability raises is
left out of the average's sample — it is not counted as
0.0 and does not drag the mean; the card still counts in
``total`` and in the state breakdown.

NOT-recording is pinned with the snapshot method:
flashcards.py records nothing to either failures ledger at
these sites, so both ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.learn import flashcards as F

NOW = 1_700_000_000.0


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _dead_fsrs():
    return SimpleNamespace(due=_raise, retrievability=_raise,
                           request_retention=0.9)


def _ok_fsrs():
    return SimpleNamespace(due=lambda state, now: True,
                           retrievability=lambda stability, elapsed: 0.5,
                           request_retention=0.9)


def _reviewed_card(nb):
    card = nb.record("2+2?", "5", "4")
    card.fsrs.reps = 1  # reviewed once -> the FSRS paths engage
    return card


# ── 361: a card the FSRS math rejects is not presented ─────────

def test_361_dead_fsrs_card_skipped_from_due():
    before = _snap()
    nb = F.MistakeNotebook(fsrs=_dead_fsrs())
    _reviewed_card(nb)
    assert nb.due_cards(now=NOW) == []
    # control: healthy FSRS -> the reviewed card is due
    nb_ok = F.MistakeNotebook(fsrs=_ok_fsrs())
    card = _reviewed_card(nb_ok)
    assert [c.id for c in nb_ok.due_cards(now=NOW)] == [card.id]
    assert _snap() == before


# ── 380: failed retrievability reads as the no-data default ────

def test_380_dead_fsrs_retrievability_is_zero():
    before = _snap()
    nb = F.MistakeNotebook(fsrs=_dead_fsrs())
    _reviewed_card(nb)
    assert nb.notebook(now=NOW)[0]["retrievability"] == 0.0
    # the identical value a never-reviewed card honestly reports
    fresh = F.MistakeNotebook(fsrs=_ok_fsrs())
    fresh.record("1+1?", "3", "2")
    assert fresh.notebook(now=NOW)[0]["retrievability"] == 0.0
    # control: healthy FSRS -> the computed value flows through
    nb_ok = F.MistakeNotebook(fsrs=_ok_fsrs())
    _reviewed_card(nb_ok)
    assert nb_ok.notebook(now=NOW)[0]["retrievability"] == 0.5
    assert _snap() == before


# ── 388: _due fails closed ──────────────────────────────────────

def test_388_dead_fsrs_due_is_false():
    before = _snap()
    nb = F.MistakeNotebook(fsrs=_dead_fsrs())
    card = _reviewed_card(nb)
    assert nb._due(card, NOW) is False
    # control: healthy FSRS decides
    nb_ok = F.MistakeNotebook(fsrs=_ok_fsrs())
    card_ok = _reviewed_card(nb_ok)
    assert nb_ok._due(card_ok, NOW) is True
    # a never-reviewed card is due without consulting the FSRS
    fresh = nb_ok.record("1+1?", "3", "2")
    assert nb_ok._due(fresh, NOW) is True
    assert _snap() == before


# ── 413: stats leave the broken card out of the average ────────

def test_413_dead_fsrs_card_out_of_average():
    before = _snap()
    nb = F.MistakeNotebook(fsrs=_dead_fsrs())
    _reviewed_card(nb)
    stats = nb.stats(now=NOW)
    assert stats["total"] == 1
    assert stats["avg_retrievability"] == 0.0  # empty sample
    # control: healthy FSRS -> the value is the average's input
    nb_ok = F.MistakeNotebook(fsrs=_ok_fsrs())
    _reviewed_card(nb_ok)
    assert nb_ok.stats(now=NOW)["avg_retrievability"] == 0.5
    assert _snap() == before
