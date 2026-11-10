"""Doc 20 P4 — the thymus self-corpus grows from Devon's own history.

Pins, end to end:

* with no harvest data the corpus is the eight shipped seeds and
  nothing else (provenance "seed", corpus version 0);
* harvest admits the incident journal's verified fixes as
  exemplars, each CITING its source record (provenance
  "verified_fix", source ref the fix's signature) — and a second
  harvest over the same journal admits zero and moves no version
  (idempotent; dedupe is by source ref AND by normalized text);
* harvest admits owner-directed police receipts recorded on the
  surface (provenance "owner_receipt") and nothing else from the
  receipt stream;
* negative selection consumes the GROWN corpus: a candidate that
  fires on a harvested exemplar is rejected and the refusal CITES
  the exemplar (kind + source ref), exactly as a seed match cites
  the seed;
* a missing or empty journal harvests zero and says so — never an
  error, never a fabricated exemplar; an unverified recovery is
  not a verified fix and never harvests;
* the corpus file is durable (a fresh store over the same path
  reads the same version + population) and every prior version
  stays readable (corpus_at filters by admission version);
* a corrupt corpus file degrades to seed-only with the reason
  recorded, and the chat surface still answers;
* the /immune corpus read verb reports version + counts by
  provenance; harvested exemplars never become antibodies.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from nomorals.immune.corpus import CorpusStore, normalize_text
from nomorals.immune.thymus import Thymus

# ── planting helpers (the organs' real writers) ──────────────────

def _plant_verified_fix(journal_path, *, detail, subsystem="tools/web",
                        strategy="retry"):
    """One verified fix in a real IncidentJournal at ``journal_path``.

    Returns the fix's signature — the citation harvest must carry.
    """
    from nomorals.core.incidents import IncidentJournal, RecoveryRecord

    journal = IncidentJournal(path=str(journal_path))
    try:
        incident = journal.record_incident(
            ConnectionError("flaky fetch refused"), subsystem=subsystem)
        journal.record_recovery(RecoveryRecord(
            incident_id=incident.id, signature=incident.signature,
            strategy=strategy, detail=detail, verified=True,
            verify_note="re-ran the recovered call: passed"))
        return incident.signature
    finally:
        journal.close()


def _plant_unverified_recovery(journal_path):
    from nomorals.core.incidents import IncidentJournal, RecoveryRecord

    journal = IncidentJournal(path=str(journal_path))
    try:
        incident = journal.record_incident(
            ConnectionError("still broken"), subsystem="tools/web")
        journal.record_recovery(RecoveryRecord(
            incident_id=incident.id, signature=incident.signature,
            strategy="retry", detail="tried, did not verify",
            verified=False))
    finally:
        journal.close()


@pytest.fixture()
def immune_env(tmp_path, monkeypatch):
    """Isolate every shared immune store on tmp files."""
    from nomorals.immune import wiring
    from nomorals.immune.chat import reset_immune_state

    corpus_path = tmp_path / "immune-selfcorpus.json"
    monkeypatch.setenv(wiring.CORPUS_STORE_ENV_VAR, str(corpus_path))
    monkeypatch.setenv(
        wiring.ANTIBODY_STORE_ENV_VAR, str(tmp_path / "immune-antibodies.json"))
    monkeypatch.setenv(
        wiring.STORE_ENV_VAR, str(tmp_path / "immune-baseline.json"))
    wiring.reset_shared_corpus_stores()
    wiring.reset_shared_antibody_stores()
    wiring.reset_shared_monitors()
    key = f"p4-{time.time_ns()}"
    reset_immune_state(key)
    yield key, tmp_path, corpus_path
    reset_immune_state(key)
    wiring.reset_shared_corpus_stores()
    wiring.reset_shared_antibody_stores()
    wiring.reset_shared_monitors()


# ── the seed floor ───────────────────────────────────────────────

class TestSeedFloor:

    def test_seed_only_when_no_harvest_data(self, tmp_path):
        store = CorpusStore(tmp_path / "corpus.json")
        exemplars = store.exemplars()
        assert len(exemplars) == 8
        assert {e.provenance for e in exemplars} == {"seed"}
        assert {e.corpus_version for e in exemplars} == {0}
        assert store.version == 0
        assert store.degraded_reason == ""
        assert store.counts_by_provenance() == {
            "seed": 8, "verified_fix": 0, "owner_receipt": 0}

    def test_missing_file_is_seed_only_not_an_error(self, tmp_path):
        store = CorpusStore(tmp_path / "never-written.json")
        assert len(store.exemplars()) == 8
        assert store.degraded_reason == ""


# ── harvest: verified fixes ──────────────────────────────────────

class TestVerifiedFixHarvest:

    def test_harvest_admits_verified_fixes_with_citations(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        signature = _plant_verified_fix(
            journal_path,
            detail="retry recovered the flaky web fetch after a refusal")
        store = CorpusStore(tmp_path / "corpus.json")
        report = store.harvest(journal_path=journal_path, receipts=[])
        assert report["sources"]["verified_fix"]["found"] == 1
        assert report["sources"]["verified_fix"]["added"] == 1
        assert report["added"] == 1
        assert report["version_before"] == 0
        assert report["version_after"] == 1
        assert store.version == 1
        harvested = [e for e in store.exemplars()
                     if e.provenance == "verified_fix"]
        assert len(harvested) == 1
        exemplar = harvested[0]
        assert exemplar.text.startswith("retry recovered the flaky")
        assert exemplar.source_ref == f"verified_fix:{signature}"
        assert exemplar.corpus_version == 1
        assert exemplar.harvested_ts is not None
        assert store.counts_by_provenance()["verified_fix"] == 1

    def test_second_harvest_is_idempotent(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_verified_fix(journal_path, detail="a verified recovery")
        store = CorpusStore(tmp_path / "corpus.json")
        first = store.harvest(journal_path=journal_path, receipts=[])
        assert first["added"] == 1
        second = store.harvest(journal_path=journal_path, receipts=[])
        assert second["sources"]["verified_fix"]["found"] == 1
        assert second["added"] == 0
        assert second["version_before"] == 1
        assert second["version_after"] == 1
        assert store.version == 1
        assert len(store.exemplars()) == 9

    def test_dedupe_by_normalized_text(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_verified_fix(
            journal_path, detail="Recovered the FLAKY fetch",
            subsystem="tools/web")
        _plant_verified_fix(
            journal_path, detail="recovered   the flaky fetch",
            subsystem="tools/mail")
        store = CorpusStore(tmp_path / "corpus.json")
        report = store.harvest(journal_path=journal_path, receipts=[])
        assert report["sources"]["verified_fix"]["found"] == 2
        assert report["added"] == 1

    def test_missing_journal_harvests_zero_honestly(self, tmp_path):
        store = CorpusStore(tmp_path / "corpus.json")
        report = store.harvest(
            journal_path=tmp_path / "no-such-journal.db", receipts=[])
        assert report["sources"]["verified_fix"]["found"] == 0
        assert report["added"] == 0
        assert "not found" in report["sources"]["verified_fix"]["note"]
        assert store.version == 0

    def test_empty_journal_and_unverified_recovery_harvest_zero(
            self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_unverified_recovery(journal_path)
        store = CorpusStore(tmp_path / "corpus.json")
        report = store.harvest(journal_path=journal_path, receipts=[])
        assert report["sources"]["verified_fix"]["found"] == 0
        assert report["added"] == 0
        assert len(store.exemplars()) == 8


# ── harvest: owner-directed receipts ─────────────────────────────

class TestReceiptHarvest:

    def test_owner_directed_receipts_admitted_others_not(self):
        from nomorals.immune.police import Police

        police = Police(vault_fn=lambda artifact: None)
        owner_receipt = police.respond(
            {"target": "owner-laptop-tunnel"}, owner_directed=True)
        observe_receipt = police.observe({"target": "scan"}, "scan")
        store = CorpusStore()
        report = store.harvest(
            journal_path=None,
            receipts=[owner_receipt, observe_receipt])
        source = report["sources"]["owner_receipt"]
        assert source["found"] == 1
        assert source["added"] == 1
        harvested = [e for e in store.exemplars()
                     if e.provenance == "owner_receipt"]
        assert len(harvested) == 1
        assert harvested[0].text == "owner-laptop-tunnel"
        assert harvested[0].source_ref == \
            f"police_receipt:{owner_receipt.id}"

    def test_dict_shaped_receipts_accepted(self):
        store = CorpusStore()
        report = store.harvest(journal_path=None, receipts=[{
            "id": "receipt-1", "action": "report_only",
            "target": "owner work shape", "reason": "",
            "details": {"blocked": False, "owner_directed": True},
        }])
        assert report["sources"]["owner_receipt"]["added"] == 1

    def test_no_receipts_is_an_honest_zero(self):
        store = CorpusStore()
        report = store.harvest(journal_path=None, receipts=[])
        assert report["sources"]["owner_receipt"]["found"] == 0
        assert report["added"] == 0


# ── negative selection over the grown corpus ─────────────────────

class TestNegativeSelectionCitations:

    def test_candidate_matching_harvested_exemplar_cites_it(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        signature = _plant_verified_fix(
            journal_path,
            detail="retry recovered the flaky web fetch after a refusal")
        store = CorpusStore(tmp_path / "corpus.json")
        store.harvest(journal_path=journal_path, receipts=[])
        thymus = Thymus(store.corpus())
        candidate = thymus.propose(
            "flaky web fetch", "virus", "broad pattern, planted for the test")
        with pytest.raises(ValueError) as excinfo:
            thymus.accept(candidate)
        message = str(excinfo.value)
        assert "autoimmune" in message
        assert "verified_fix" in message
        assert signature in message

    def test_candidate_matching_only_a_seed_cites_the_seed(self):
        store = CorpusStore()
        thymus = Thymus(store.corpus())
        candidate = thymus.propose(
            "plays music", "virus", "broad pattern, planted for the test")
        with pytest.raises(ValueError) as excinfo:
            thymus.accept(candidate)
        message = str(excinfo.value)
        assert "[seed:" in message
        assert "seed-2" in message

    def test_clean_candidate_still_accepted_over_grown_corpus(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_verified_fix(journal_path, detail="a verified recovery")
        store = CorpusStore(tmp_path / "corpus.json")
        store.harvest(journal_path=journal_path, receipts=[])
        thymus = Thymus(store.corpus())
        accepted = thymus.accept(thymus.propose(
            "ignore all previous instructions", "virus",
            "injection pattern"))
        assert accepted.status == "active"


# ── persistence + version history ────────────────────────────────

class TestPersistence:

    def test_corpus_survives_fresh_construction(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_verified_fix(journal_path, detail="a verified recovery")
        corpus_path = tmp_path / "corpus.json"
        CorpusStore(corpus_path).harvest(
            journal_path=journal_path, receipts=[])
        reloaded = CorpusStore(corpus_path)
        assert reloaded.version == 1
        assert len(reloaded.exemplars()) == 9
        assert reloaded.counts_by_provenance()["verified_fix"] == 1

    def test_prior_versions_stay_readable(self, tmp_path):
        journal_path = tmp_path / "error_journal.db"
        _plant_verified_fix(
            journal_path, detail="first verified recovery",
            subsystem="tools/web")
        store = CorpusStore(tmp_path / "corpus.json")
        store.harvest(journal_path=journal_path, receipts=[])
        _plant_verified_fix(
            journal_path, detail="second verified recovery",
            subsystem="tools/mail")
        store.harvest(journal_path=journal_path, receipts=[])
        assert store.version == 2
        assert len(store.corpus_at(0).exemplars) == 8
        assert len(store.corpus_at(1).exemplars) == 9
        assert len(store.corpus_at(2).exemplars) == 10
        # corpus_at(0) is the seed floor negative selection started from.
        thymus = Thymus(store.corpus_at(0))
        candidate = thymus.propose(
            "verified recovery", "virus", "planted")
        assert thymus.accept(candidate).status == "active"

    def test_corrupt_file_degrades_to_seed_only(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text("{ not json at all", encoding="utf-8")
        store = CorpusStore(corpus_path)
        assert len(store.exemplars()) == 8
        assert store.version == 0
        assert "seed-only" in store.degraded_reason

    def test_wrong_format_file_degrades_to_seed_only(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text('{"format": "something.else"}',
                               encoding="utf-8")
        store = CorpusStore(corpus_path)
        assert len(store.exemplars()) == 8
        assert store.degraded_reason != ""


# ── the chat surface ─────────────────────────────────────────────

class _FakeSettings:
    """Settings whose resolve() lands under a tmp home (the wiring
    resolvers' contract), so the chat harvest reads the planted
    journal and every shared store stays in tmp."""

    def __init__(self, home):
        self._home = home
        self.home_path = str(home)

    def resolve(self, rel):
        return str(self._home / rel)


class TestChatSurface:

    def test_corpus_read_verb_reports_counts_and_version(
            self, immune_env):
        key, _tmp, _path = immune_env
        from nomorals.immune.chat import control_immune

        out = control_immune("corpus", chat_key=key)
        assert "self-corpus v0" in out
        assert "8 exemplars" in out
        assert "seed 8" in out
        assert "verified fixes 0" in out
        assert "owner receipts 0" in out

    def test_status_reports_the_corpus(self, immune_env):
        key, _tmp, _path = immune_env
        from nomorals.immune.chat import control_immune

        out = control_immune("status", chat_key=key)
        assert "self-corpus: 8 exemplars (v0" in out

    def test_chat_harvest_grows_the_corpus_the_thymus_uses(
            self, immune_env):
        key, tmp_path, _path = immune_env
        from nomorals.immune.chat import control_immune, get_immune_state

        journal_path = tmp_path / "data" / "error_journal.db"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        signature = _plant_verified_fix(
            journal_path,
            detail="retry recovered the flaky web fetch after a refusal")
        state = get_immune_state(key)
        receipt = state.police.respond(
            {"target": "owner-laptop-tunnel"}, owner_directed=True)
        state._record(receipt)
        antibodies_before = state.store.all()

        context = SimpleNamespace(settings=_FakeSettings(tmp_path))
        out = control_immune(
            "corpus harvest", context=context, chat_key=key)
        assert "+2 exemplar(s)" in out
        assert "corpus v0 → v1" in out
        assert "verified fixes (incident journal): 1 found, +1 admitted" in out
        assert "owner-directed receipts (this surface): 1 found, " \
            "+1 admitted" in out

        # The read verb now reports the grown corpus.
        read = control_immune("corpus", chat_key=key)
        assert "self-corpus v1" in read
        assert "verified fixes 1" in read
        assert "owner receipts 1" in read

        # The state's thymus — built BEFORE the harvest — tests
        # against the grown corpus (the store's live corpus object).
        with pytest.raises(ValueError) as excinfo:
            state.thymus.accept(state.thymus.propose(
                "flaky web fetch", "virus", "planted broad pattern"))
        assert signature in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            state.thymus.accept(state.thymus.propose(
                "owner-laptop", "virus", "planted broad pattern"))
        assert "owner_receipt" in str(excinfo.value)
        assert receipt.id in str(excinfo.value)

        # Harvested exemplars never became antibodies.
        assert state.store.all() == antibodies_before

        # A second chat harvest over the same records adds zero.
        again = control_immune(
            "corpus harvest", context=context, chat_key=key)
        assert "+0 exemplar(s)" in again
        assert "corpus v1 → v1" in again

    def test_corrupt_corpus_file_surface_stays_alive(self, immune_env):
        _key, _tmp, corpus_path = immune_env
        from nomorals.immune import wiring
        from nomorals.immune.chat import control_immune

        corpus_path.write_text("{ corrupt", encoding="utf-8")
        wiring.reset_shared_corpus_stores()
        key = f"p4-corrupt-{time.time_ns()}"
        out = control_immune("status", chat_key=key)
        assert "immune status" in out
        assert "self-corpus: 8 exemplars (v0" in out
        read = control_immune("corpus", chat_key=key)
        assert "seed-only corpus in use" in read


def test_normalize_text_collapses_case_and_whitespace():
    assert normalize_text("  Recovered   the\nFLAKY fetch ") == \
        "recovered the flaky fetch"
    assert normalize_text(None) == ""
