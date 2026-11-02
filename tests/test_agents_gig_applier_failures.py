"""Tail-99 pins: nomorals/agents/gig_applier.py census sites.

The applier's honesty rule is printed on ``submit()``: the
record says what really happened — ``submitted`` only on a
submitter's evidence, everything else an attempted/manual
state with a checklist. Two of the three sites are that law's
supporting machinery and are held as exclusions; the third is
pinned AND flagged (below):

485 — ``submit()``: the cognition representation ledger call
crashes -> swallowed ("ledger never breaks the action"). The
application itself was already transitioned AND saved to the
application store before the ledger is touched; the returned
status is the store's durable fact, and the ledger is a
best-effort side-log of an action that genuinely happened.
576 — ``follow_up_draft()``: the LLM call crashes -> the
offline template. The docstring prints the contract ("Uses
the LLM when available, otherwise an offline template —
follow-ups must work without a model"); the template is a
real, sendable draft built from the application's own fields,
and the follow_up_drafted event is recorded either way.

215 — ``ApplicationStore.load_all()``: a corrupt JSONL line is
skipped. ⚠ FLAGGED — DATA-LOSS SHAPE, targeted-fix candidate,
NOT a batch convert (protocol: save/persist swallows read back
later leave the batch). The load alone loses nothing durable,
but ``save()`` is load-then-rewrite: it rewrites
applications.jsonl from exactly what load_all returned, so
the first save after a line corrupts permanently deletes that
application's record — the same load/save corrupt-gate pair
flagged when this store was first surveyed. The pin below
proves the mechanism exactly as it stands today; the fix
(quarantine unparseable lines instead of dropping them)
belongs to the targeted-fix pass.

NOT-recording is pinned with the snapshot method:
gig_applier.py calls no failures-ledger record function
anywhere, so both the agency and agents ledgers must be
unchanged across each trigger. (The representation ledger at
485 is a cognition log, not a failures ledger — its call is
the trigger, not the recording surface.)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.agents.gig_applier as G
import nomorals.cognition.representation as rep


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _settings(root: Path):
    return SimpleNamespace(resolve=lambda rel: str(root))


def _store(root: Path) -> G.ApplicationStore:
    store = G.ApplicationStore(settings=_settings(root))
    store.data_dir  # resolve + create the directory up front
    return store


# ── 215: corrupt JSONL line skipped (FLAGGED data-loss pair) ────

def test_215_corrupt_line_skipped_then_save_rewrites(tmp_path):
    before = _snap()
    store = _store(tmp_path)
    good = G.Application(gig_id="g1", title="T1", url="http://x/1")
    (store._path()).write_text(
        json.dumps(good.to_dict()) + "\n{corrupt line\n"
        + json.dumps({"gig_id": "g2"}) + "\n")  # missing title/url
    assert sorted(store.load_all()) == ["g1"]
    # the flagged mechanism: save() rewrites from load_all, so
    # the corrupt bytes — and the half-formed g2 — are deleted
    store.save(G.Application(gig_id="g3", title="T3", url="http://x/3"))
    raw = (store._path()).read_text()
    assert "{corrupt line" not in raw
    assert sorted(store.load_all()) == ["g1", "g3"]
    # control: an all-valid file round-trips whole through save
    store2 = _store(tmp_path / "clean")
    a = G.Application(gig_id="a1", title="A", url="http://x/a")
    b = G.Application(gig_id="a2", title="B", url="http://x/b")
    store2.save(a)
    store2.save(b)
    assert sorted(store2.load_all()) == ["a1", "a2"]
    assert _snap() == before


# ── 485: representation ledger crash never breaks submit ────────

def test_485_ledger_crash_leaves_submission_intact(tmp_path):
    before = _snap()
    applier = G.GigApplier(settings=_settings(tmp_path),
                           llm_fn=lambda prompt: "")
    G.register_submitter(
        "tail99board",
        lambda app, draft: {"ok": True, "method": "fake",
                            "evidence": "conf-1"})
    try:
        applier.store.save(G.Application(
            gig_id="g9", title="T9", url="http://x/9",
            source="tail99board"))
        with mock.patch.object(rep, "log_representation_action",
                               _raise):
            out = applier.submit("g9")
        assert out.status == "submitted"
        # the durable record — not the ledger — carries the fact
        assert applier.store.get("g9").status == "submitted"
        # control: with the ledger healthy it is written, once
        calls = []
        applier.store.save(G.Application(
            gig_id="g10", title="T10", url="http://x/10",
            source="tail99board"))
        with mock.patch.object(
                rep, "log_representation_action",
                lambda *a, **k: calls.append((a, k))):
            out = applier.submit("g10")
        assert out.status == "submitted"
        assert len(calls) == 1 and calls[0][0][0] == "apply"
    finally:
        G.SUBMITTERS.pop("tail99board", None)
    assert _snap() == before


# ── 576: LLM crash -> offline template follow-up ────────────────

def test_576_llm_crash_falls_back_to_template(tmp_path):
    before = _snap()
    applier = G.GigApplier(settings=_settings(tmp_path), llm_fn=_raise)
    applier.store.save(G.Application(
        gig_id="g11", title="T11", url="http://x/11", source="board",
        status="submitted"))
    text = applier.follow_up_draft("g11")
    assert text.startswith("Subject: Following up — T11")
    # the draft event is recorded on the application either way
    kinds = [h["kind"] for h in applier.store.get("g11").history]
    assert "follow_up_drafted" in kinds
    # control: a working LLM's text wins over the template
    applier2 = G.GigApplier(
        settings=_settings(tmp_path),
        llm_fn=lambda prompt: "  polished follow-up  ")
    assert applier2.follow_up_draft("g11") == "polished follow-up"
    assert _snap() == before
