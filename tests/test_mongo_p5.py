"""Mongo program P5: sidecar federation, analytics backend-awareness.

P5 settles the stores that live BESIDE the central database:

* **The registry** — storage.sidecars.SIDECAR_STORES classifies every
  sidecar with a verdict (ported / federated / pinned / residue /
  ephemeral / reader). The drift guard scans the tree mechanically:
  a new raw ``sqlite3.connect`` file or ``VectorStore`` site outside
  ``storage/`` that is not classified fails this suite the day it is
  written.
* **The ports** — challenge-memory, preferences and social-drafts run
  over the DocumentStore seam whenever a store is resolved, behind
  their unchanged APIs, on BOTH seam rungs (SQLite document store and
  a real mongod), with the SQLite default byte-identical. Their
  existing file data imports lazily, once, through the same mapping.
  The residue Batch 1 slice-1 ports (agent-identities,
  audio-audiobooks, audio-characters, audio-overviews,
  audio-keyterms) follow the same pattern below, as do the
  slice-2 ports (health-coach, health-form, life-archive,
  llm-cost-display, marketing-aeo) and the slice-3 ports
  (audio-melodic, property-scam, property-value, social-voice,
  social-voice-notes, calendar-integration — the last with the
  cursor-continuity case its registry entry demands). The residue
  Batch 2 ports: workspace-inbox, health-nutrition and
  commerce-medusa (slice 1), and the B2 port (shopping-deals) is the shared-file shape: two
  engines over one Database handle, six tables, one shared port —
  with the legacy price_snapshots adoption cases on the seam. The
  residue Batch 2 slice-3 ports (email-integration,
  travel-whitelabel, social-whatsapp-cost, planning-estimates)
  follow the same pattern below — email with the cursor-continuity
  case its registry entry demands. The residue Batch 2 slice-6
  ports (matching-batches, os-timeline, planning-route) follow the
  same pattern below — os-timeline with the search-disposition case
  its registry entry demands: no split disposition, fts_available
  honest on the seam, search served by a document scan whose result
  set matches the FTS5 rung's.
  The residue Batch 2 slice-5 ports (marketing-briefs,
  marketing-competitor, marketing-guardrails, commerce-cart-recovery)
  follow the same pattern below.
  The residue Batch 2 slice-8 ports (travel-watchers, social-leads)
  follow the same pattern below.
  The residue Batch 2 slice-7 ports (matching-questionnaire,
  property-passport, social-profiles, triggers-saved-search) follow
  the same pattern below.
  The health-timeline lane follows the same pattern below
  (attached_rows hybrid).
  The B10 lane (immune-corpus, injected port) follows the same
  pattern below.
  The B9 lane (identity-documents, injected port, vault review
  recorded) follows the same pattern below.
  The B28 lane (health-challenges, full streak-lifecycle
  conformance) follows the same pattern below. The residue Batch 3
  C4 port (people — person+notes rollups re-expressed over finds,
  int-id continuity across the lazy import) follows the same
  pattern below.
  The residue Batch 3 C5 port (household — record-then-audit
  composition preserved verbatim) follows the same pattern below.
  The residue Batch 3 C8 port (planning-graph —
  traversal parity: the walks are the module's own Python over
  full-edge loads on both rungs) follows the same pattern below.
  The residue Batch 3 C3 port (planning-congestion — hold expiry
  as document data via the sql_expires_at rename, never the
  store-managed TTL) follows the same pattern below.
  The residue Batch 3 C6 port (travel-groups — one collection
  namespace per group key via the sidecar_port namespace_suffix
  extension; money splits are single-row writes, balances are
  recomputed reads) follows the same pattern below.
  The residue Batch 3 C2 port (marketing-send — the claim as
  seam-only fields fenced by one cas_row, two-engine exactly-once)
  follows the same pattern below.
* **Analytics** — the sync source is backend-aware: for seam-served
  tables the rows come from the seam (identical output on every
  rung); the SQLite passthrough counts them honestly and refuses
  loudly to read their fossil.
* **Coherence** — storage_backend_status answers the whole picture
  in one call: backend + pinned domains + the sidecar federation.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.storage import sidecars
from nomorals.storage.db import Database, open_database
from nomorals.storage.docstore import (
    MongoDocumentStore,
    SQLiteDocumentStore,
    set_ambient_document_store,
    storage_backend_status,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "nomorals"


@pytest.fixture(autouse=True)
def _no_ambient():
    set_ambient_document_store(None)
    yield
    set_ambient_document_store(None)


# ---------------------------------------------------------------------------
# mongod fixture — a real mongod 7.0 on an ephemeral port (P3/P4 pattern)
# ---------------------------------------------------------------------------

def _find_mongod() -> str | None:
    env = os.environ.get("NM_MONGOD_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("mongod")
    if found:
        return found
    candidate = os.path.expanduser("~/workspace/tools/mongodb/bin/mongod")
    return candidate if os.path.exists(candidate) else None


_MONGOD = _find_mongod()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def mongod(tmp_path_factory):
    """Start a throwaway single-node replica-set mongod (or skip)."""
    import contextlib

    pymongo = pytest.importorskip("pymongo")
    if _MONGOD is None:
        pytest.skip("mongod binary not installed")
    data = tmp_path_factory.mktemp("mongo5-data")
    port = _free_port()
    log = open(data / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [_MONGOD, "--dbpath", str(data), "--bind_ip", "127.0.0.1",
         "--port", str(port), "--replSet", "rs0",
         "--wiredTigerCacheSizeGB", "0.25"],
        stdout=log, stderr=log)
    base = f"mongodb://127.0.0.1:{port}"
    try:
        direct = pymongo.MongoClient(base + "/?directConnection=true",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                direct.admin.command("ping")
                break
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    pytest.skip("mongod did not come up")
                time.sleep(0.25)
        with contextlib.suppress(pymongo.errors.OperationFailure):
            direct.admin.command("replSetInitiate")
        client = pymongo.MongoClient(base + "/?replicaSet=rs0",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                if client.admin.command("hello").get("isWritablePrimary"):
                    break
            except Exception:  # noqa: BLE001
                pass
            if time.time() > deadline:
                pytest.skip("mongod replica set did not elect a primary")
            time.sleep(0.25)
        yield base + "/?replicaSet=rs0"
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        log.close()


def _mongo_store(mongod) -> MongoDocumentStore:
    # Short database names: a uuid-suffixed name stays far below the
    # 63-char namespace limit the P4 fixture hit.
    return MongoDocumentStore(mongod, f"mongo5{uuid.uuid4().hex[:10]}")


@pytest.fixture(params=["native", "sqlite_doc", "mongo"])
def rung(request, mongod):
    """One of the three execution rungs a port/analytics path can serve."""
    if request.param == "native":
        yield SimpleNamespace(kind="native", store=None)
    elif request.param == "sqlite_doc":
        scratch = Database(":memory:")
        yield SimpleNamespace(kind="sqlite_doc",
                              store=SQLiteDocumentStore(scratch))
        scratch.close()
    else:
        store = _mongo_store(mongod)
        yield SimpleNamespace(kind="mongo", store=store)
        store.close()


# ---------------------------------------------------------------------------
# the registry: completeness, sanity, drift guard
# ---------------------------------------------------------------------------

class TestSidecarRegistry:
    def test_no_unclassified_sidecar_modules(self):
        uncovered = sidecars.uncovered_modules(PACKAGE_ROOT)
        assert uncovered == {"connect": [], "vector_sites": []}, (
            "a module opens durable state outside the central store "
            f"without a sidecar classification: {uncovered}")

    def test_every_spec_is_complete_and_real(self):
        for name, spec in sidecars.SIDECAR_STORES.items():
            assert spec.verdict in sidecars.VERDICTS, name
            assert spec.reason.strip(), f"{name}: a verdict needs its reason"
            for module in spec.modules:
                assert (PACKAGE_ROOT / module).exists(), \
                    f"{name}: owner module {module} does not exist"
            for module, marker in spec.anchors:
                src = (PACKAGE_ROOT / module).read_text(encoding="utf-8")
                assert marker in src, \
                    f"{name}: anchor marker {marker!r} gone from {module} — reclassify"

    def test_no_module_double_classified(self):
        seen: dict[str, str] = {}
        for name, spec in sidecars.SIDECAR_STORES.items():
            for module in spec.modules:
                assert module not in seen, \
                    f"{module} claimed by both {seen[module]} and {name}"
                seen[module] = name

    def test_ported_set_is_exactly_the_p5_ports(self):
        ported = {n for n, s in sidecars.SIDECAR_STORES.items()
                  if s.verdict == "ported"}
        assert ported == {
            "challenge-memory", "preferences", "social-drafts",
            # residue Batch 1, slice 1:
            "agent-identities", "audio-audiobooks", "audio-characters",
            "audio-overviews", "audio-keyterms",
            # residue Batch 1, slice 3:
            "audio-melodic", "property-scam", "property-value",
            "social-voice", "social-voice-notes",
            "calendar-integration",
            # residue Batch 1, slice 2:
            "health-coach", "health-form", "life-archive",
            "llm-cost-display", "marketing-aeo",
            # residue Batch 2, slice 1:
            "workspace-inbox", "health-nutrition", "commerce-medusa",
            # residue Batch 2, B2 (the shared-file group):
            "shopping-deals",
            # residue Batch 2, slice 3:
            "email-integration", "travel-whitelabel",
            "social-whatsapp-cost", "planning-estimates",
            # residue Batch 2, slice 4:
            "digital-twin", "health-training", "legal-portfolio",
            "legal-regulatory",
            # residue Batch 1, A16 catch-up:
            "travel-itinerary",
            # residue Batch 2, slice 6:
            "matching-batches", "os-timeline", "planning-route",
            # residue Batch 2, slice 5:
            "marketing-briefs", "marketing-competitor",
            "marketing-guardrails", "commerce-cart-recovery",
            # residue Batch 2, slice 8:
            "travel-watchers", "social-leads",
            # residue Batch 2, slice 7:
            "matching-questionnaire", "property-passport",
            "social-profiles", "triggers-saved-search",
            # residue Batch 2, health-timeline lane:
            "health-timeline",
            # residue Batch 2, B10 lane:
            "immune-corpus",
            # residue Batch 2, B9 lane:
            "identity-documents",
            # residue Batch 2, B28 lane:
            "health-challenges",
            # residue Batch 3, C4 lane:
            "people",
            # residue Batch 3, C5 lane:
            "household",
            # residue Batch 3, C8 lane:
            "planning-graph",
            # residue Batch 3, C3 lane:
            "planning-congestion",
            # residue Batch 3, C6 lane:
            "travel-groups",
            # residue Batch 3, C2 lane:
            "marketing-send"}

    def test_pinned_sidecars_follow_real_central_pins(self):
        from nomorals.storage.carry import PINNED_DOMAINS

        for name, spec in sidecars.SIDECAR_STORES.items():
            if spec.verdict == "pinned" and spec.follows:
                assert spec.follows in PINNED_DOMAINS or \
                    spec.follows == "mongo-p6-cutover", \
                    f"{name} follows unknown pin {spec.follows!r}"


class TestStatusCoherence:
    def test_one_call_answers_the_whole_federation(self):
        settings = SimpleNamespace(storage=SimpleNamespace(
            backend="sqlite", mongo_uri="", mongo_db="nomorals",
            mongo_shadow=False))
        db = Database(":memory:")
        try:
            status = storage_backend_status(settings, db)
        finally:
            db.close()
        assert status["active_backend"] == "sqlite"
        assert set(status["pinned_domains"]) == {
            "games-arena", "memory-recall", "fts"}
        federation = status["sidecars"]
        assert set(federation["stores"]) == set(sidecars.SIDECAR_STORES)
        assert sum(federation["counts"].values()) == len(sidecars.SIDECAR_STORES)
        for verdict in ("ported", "federated", "pinned", "residue"):
            assert federation["counts"][verdict] >= 1


# ---------------------------------------------------------------------------
# the ports, per rung
# ---------------------------------------------------------------------------

class TestChallengeMemoryPort:
    def _scenario(self, memory) -> dict:
        first = memory.record(
            site="https://sub.example.com/login", kind="Captcha",
            strategies=[("wait", True, 120), ("click", False, 40)],
            cleared=True, via="wait", ms=900)
        second = memory.record(
            site="example.com", kind="captcha",
            strategies=[("click", True, 30)], cleared=True, via="click",
            takeover_needed=True, ms=300)
        return {
            "ids": (first, second),
            "rank": memory.rank_strategies(
                "example.com", "captcha", ["click", "wait", "solve"]),
            "stats": memory.stats(),
            "stats_site": memory.stats(site="example.com", kind="captcha"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.browser.challenge_memory import ChallengeMemory

        memory = ChallengeMemory(tmp_path / "cm.db", store=rung.store)
        try:
            observed = self._scenario(memory)
        finally:
            memory.close()
        assert observed["ids"] == (1, 2)
        # wait: 1 win / 1 try at the site (smoothed 2/3) outranks
        # click's site record (0/1 → 1/3), even with its global win.
        assert observed["rank"][0] == "wait"
        assert observed["stats"] == {
            "encounters": 2, "cleared": 2, "cleared_rate": 1.0,
            "takeovers": 1, "takeover_rate": 0.5}
        assert observed["stats_site"]["encounters"] == 2

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.browser.challenge_memory import ChallengeMemory

        path = tmp_path / "cm.db"
        native = ChallengeMemory(path)
        native.record(site="example.com", kind="age",
                      strategies=[("confirm", True, 10)], cleared=True)
        native.close()

        memory = ChallengeMemory(path, store=rung.store)
        try:
            assert memory.stats()["encounters"] == 1  # imported, visible
            assert rung.store.count("challenge-memory__encounters") == 1
            doc = rung.store.find("challenge-memory__encounters")[0]
            assert doc["site"] == "example.com" and doc["cleared"] == 1
            # the AUTOINCREMENT counter was seeded by the import:
            assert memory.record(site="example.com", kind="age") == 2
        finally:
            memory.close()
        # second construction: adopted, not re-imported or duplicated
        again = ChallengeMemory(path, store=rung.store)
        try:
            assert again.stats()["encounters"] == 2
            assert rung.store.count("challenge-memory__encounters") == 2
        finally:
            again.close()


class TestPreferenceStorePort:
    def _scenario(self, store) -> dict:
        store.update_weight("music", "energy", "High", 0.5)
        store.update_weight("music", "energy", "high", 0.25)
        store.update_weight("music", "tempo", "fast", -0.5)
        store.log_signal("music", {"energy": "high"}, True, 0.5, source="t1")
        store.log_signal("music", {"tempo": "fast"}, False, -0.5, source="t2")
        store.log_signal("food", {"spice": "hot"}, True, 1.0, source="t3")
        return {
            "weight": store.get_weight("music", "energy", "HIGH"),
            "weights": store.get_weights("music"),
            "signals": [(s["domain"], s["approve"], s["delta"])
                        for s in store.signals()],
            "music_signals": len(store.signals("music")),
        }

    @staticmethod
    def _port_for(path, rung):
        # preferences is L0: the composition layer (this test) builds
        # the port and injects it — the module never imports storage.
        from nomorals.preferences.store import _PREFS_DDL
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("preferences", _PREFS_DDL,
                            ("preferences", "preference_signals"),
                            str(path), store=rung.store)

    def test_conformance_file_variant(self, rung, tmp_path):
        from nomorals.preferences.store import PreferenceStore

        store = PreferenceStore(str(tmp_path / "prefs.db"),
                                port=self._port_for(tmp_path / "prefs.db", rung))
        observed = self._scenario(store)
        store.close()
        assert observed["weight"] == 0.75
        assert observed["weights"] == {
            "energy": {"high": 0.75}, "tempo": {"fast": -0.5}}
        assert observed["signals"] == [
            ("music", True, 0.5), ("music", False, -0.5), ("food", True, 1.0)]
        assert observed["music_signals"] == 2

    def test_json_and_memory_variants_never_port(self, rung, tmp_path):
        from nomorals.preferences.store import PreferenceStore

        port = self._port_for(tmp_path / "prefs.db", rung)
        json_store = PreferenceStore(str(tmp_path / "prefs.json"), port=port)
        assert json_store._port is None
        json_store.update_weight("food", "spice", "hot", 1.0)
        assert json_store.get_weight("food", "spice", "hot") == 1.0
        json_store.close()
        memory_store = PreferenceStore(port=port)
        assert memory_store._port is None
        memory_store.close()

    def test_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.preferences.store import PreferenceStore

        path = tmp_path / "prefs.db"
        native = PreferenceStore(str(path))
        native.update_weight("books", "pace", "slow", 2.0)
        native.log_signal("books", {"pace": "slow"}, True, 2.0)
        native.close()

        ported = PreferenceStore(str(path), port=self._port_for(path, rung))
        try:
            assert ported.get_weight("books", "pace", "slow") == 2.0
            assert len(ported.signals("books")) == 1
            doc = rung.store.find("preferences__preferences")[0]
            assert doc["_id"] == "books\x1fpace\x1fslow"
            # signal ids continue from the imported maximum:
            ported.log_signal("books", {"pace": "slow"}, True, 0.5)
            assert len(ported.signals("books")) == 2
        finally:
            ported.close()


class TestDraftQueuePort:
    def _scenario(self, queue) -> dict:
        import time as _time

        first = queue.create_draft("first post", platforms=["telegram"],
                                   metadata={"n": 1})
        _time.sleep(0.01)  # distinct created_at: pending order is pinned
        second = queue.create_draft("second post")
        queue.propose(first.id)
        queue.approve(first.id)
        queue.schedule(first.id, _time.time() + 3600)
        pending = {d.id: d for d in queue.pending()}
        queue.mark_posted(first.id)
        posted = queue.get(first.id)
        return {
            "pending_contents": sorted(d.content for d in pending.values()),
            "posted_status": posted.status,
            "posted_at_set": posted.posted_at is not None,
            "scheduled_set": pending[first.id].scheduled_time is not None,
            "platforms": pending[first.id].platforms,
            "metadata": pending[first.id].metadata,
            "pending_after": [d.content for d in queue.pending()],
            "second_status": queue.get(second.id).status,
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.social.drafts import DraftQueue

        queue = DraftQueue(tmp_path / "drafts.db", store=rung.store)
        try:
            observed = self._scenario(queue)
        finally:
            queue.close()
        assert observed["pending_contents"] == ["first post", "second post"]
        assert observed["posted_status"] == "posted"
        assert observed["posted_at_set"] is True
        assert observed["scheduled_set"] is True
        assert observed["pending_after"] == ["second post"]
        assert observed["second_status"] == "draft"

    def test_transition_graph_enforced_on_seam(self, rung, tmp_path):
        from nomorals.social.drafts import DraftQueue

        queue = DraftQueue(tmp_path / "drafts.db", store=rung.store)
        try:
            draft = queue.create_draft("x")
            with pytest.raises(ValueError):
                queue.approve(draft.id)  # draft → approved is not a step
            with pytest.raises(KeyError):
                queue.propose("nope")
        finally:
            queue.close()

    def test_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.drafts import DraftQueue

        path = tmp_path / "drafts.db"
        native = DraftQueue(path)
        kept = native.create_draft("from the file days", platforms=["x"])
        native.propose(kept.id)
        native.close()

        queue = DraftQueue(path, store=rung.store)
        try:
            got = queue.get(kept.id)
            assert got is not None and got.status == "pending_review"
            assert [d.id for d in queue.pending()] == [kept.id]
            doc = rung.store.get("social-drafts__social_drafts", kept.id)
            assert doc is not None and doc["content"] == "from the file days"
        finally:
            queue.close()


class TestAgentIdentitiesPort:
    def _scenario(self, store) -> dict:
        import time as _time

        a = store.create("Devon Research",
                         email="devon.research@example.com",
                         handle="@devon_research", purpose="research tasks")
        _time.sleep(0.01)  # distinct created_at: list order is pinned
        b = store.create("Devon Scheduler")
        got = store.get(a.id)
        return {
            "empty_name": store.create("   "),
            "got": (got.name, got.email, got.handle, got.purpose),
            "names": [i.name for i in store.list()],
            "removed": store.remove(a.id),
            "removed_again": store.remove(a.id),
            "after": [i.name for i in store.list()],
            "missing": store.get("nope"),
            "b_id_prefix": b.id.startswith("ident_"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.agents.identities import IdentityStore

        store = IdentityStore(db_path=str(tmp_path / "identities.db"),
                              store=rung.store)
        observed = self._scenario(store)
        assert observed["empty_name"] is None
        assert observed["got"] == (
            "Devon Research", "devon.research@example.com",
            "@devon_research", "research tasks")
        assert observed["names"] == ["Devon Research", "Devon Scheduler"]
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["after"] == ["Devon Scheduler"]
        assert observed["missing"] is None
        assert observed["b_id_prefix"] is True

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.agents.identities import IdentityStore

        path = tmp_path / "identities.db"
        native = IdentityStore(db_path=str(path))
        kept = native.create("Devon Research", handle="@devon_research")

        store = IdentityStore(db_path=str(path), store=rung.store)
        got = store.get(kept.id)
        assert got is not None and got.name == "Devon Research"
        assert rung.store.count("agent-identities__identities") == 1
        doc = rung.store.get("agent-identities__identities", kept.id)
        assert doc is not None and doc["handle"] == "@devon_research"
        # second construction: adopted, not re-imported or duplicated
        again = IdentityStore(db_path=str(path), store=rung.store)
        assert [i.name for i in again.list()] == ["Devon Research"]
        assert rung.store.count("agent-identities__identities") == 1


class TestAudiobookStorePort:
    def _scenario(self, store) -> dict:
        import json as _json

        from nomorals.audio.audiobook import Audiobook

        older = Audiobook(
            book_id="ab_older", title="Old", author="A",
            master_path="/old.wav", duration_s=30.0,
            target_stores=["kobo"],
            disclosure={"kobo": {"text": "Narrator: Synthesised voice."}},
            created_at=100.0)
        newer = Audiobook(
            book_id="ab_newer", title="New", author="B",
            master_path="/new.wav", duration_s=90.5,
            target_stores=["spotify", "kobo"], created_at=200.0)
        saved = (store.save(older), store.save(newer))
        older.title = "Old (2nd ed)"  # INSERT OR REPLACE on re-save
        resaved = store.save(older)
        listed = store.list()
        got = store.get("ab_newer")
        got_older = store.get("ab_older")
        return {
            "saved": saved, "resaved": resaved,
            "listed_ids": [b["book_id"] for b in listed],
            "listed_keys": sorted(listed[0].keys()),
            "listed_title": listed[1]["title"],
            "limit1": [b["book_id"] for b in store.list(limit=1)],
            "got_title": got["title"],
            "got_stores": got["target_stores"],
            "got_disclosure": _json.loads(got_older["disclosure"]),
            "missing": store.get("nope"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.audio.audiobook import AudiobookStore

        store = AudiobookStore(db_path=str(tmp_path / "ab.db"),
                               store=rung.store)
        observed = self._scenario(store)
        assert observed["saved"] == (True, True)
        assert observed["resaved"] is True
        assert observed["listed_ids"] == ["ab_newer", "ab_older"]
        assert observed["listed_keys"] == [
            "author", "book_id", "created_at", "duration_s",
            "master_path", "target_stores", "title"]
        assert observed["listed_title"] == "Old (2nd ed)"
        assert observed["limit1"] == ["ab_newer"]
        assert observed["got_title"] == "New"
        assert observed["got_stores"] == "spotify,kobo"
        assert observed["got_disclosure"] == {
            "kobo": {"text": "Narrator: Synthesised voice."}}
        assert observed["missing"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.audio.audiobook import Audiobook, AudiobookStore

        path = tmp_path / "ab.db"
        native = AudiobookStore(db_path=str(path))
        native.save(Audiobook(book_id="ab_kept", title="Kept",
                              created_at=5.0))

        store = AudiobookStore(db_path=str(path), store=rung.store)
        got = store.get("ab_kept")
        assert got is not None and got["title"] == "Kept"
        assert rung.store.count("audio-audiobooks__audiobooks") == 1
        again = AudiobookStore(db_path=str(path), store=rung.store)
        assert [b["book_id"] for b in again.list()] == ["ab_kept"]
        assert rung.store.count("audio-audiobooks__audiobooks") == 1


class TestCharacterStorePort:
    def _scenario(self, store) -> dict:
        from nomorals.audio.characters import Character

        ren = Character(name="Renfield", book_id="book_1",
                        book_title="Dracula", voice="raspy whisper",
                        knowledge_cutoff=3, goals=["serve the count"],
                        personality={"loyalty": 0.9})
        ren.remember(1, "eats flies")
        ren.remember(2, "waits in the cell")
        mina = Character(name="Mina", book_id="book_1",
                         book_title="Dracula")
        saved = (store.save(ren), store.save(mina))
        by_name = store.get("renfield")  # case-insensitive, as pinned
        by_id = store.get(ren.id)
        return {
            "saved": saved,
            "by_name": (by_name.name, by_name.voice, by_name.knows_up_to(),
                        by_name.goals, by_name.personality,
                        by_name.chapter_memories),
            "same_row": by_id.id == by_name.id,
            "all_names": [c.name for c in store.list()],
            "book_names": [c.name for c in store.list("book_1")],
            "other_book": store.list("book_2"),
            "removed": store.remove("mina"),
            "removed_again": store.remove("mina"),
            "after": [c.name for c in store.list()],
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.audio.characters import CharacterStore

        store = CharacterStore(db_path=str(tmp_path / "chars.db"),
                               store=rung.store)
        observed = self._scenario(store)
        assert observed["saved"] == (True, True)
        assert observed["by_name"] == (
            "Renfield", "raspy whisper", 3, ["serve the count"],
            {"loyalty": 0.9},
            {1: ["eats flies"], 2: ["waits in the cell"]})
        assert observed["same_row"] is True
        assert observed["all_names"] == ["Mina", "Renfield"]
        assert observed["book_names"] == ["Mina", "Renfield"]
        assert observed["other_book"] == []
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["after"] == ["Renfield"]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.audio.characters import Character, CharacterStore

        path = tmp_path / "chars.db"
        native = CharacterStore(db_path=str(path))
        native.save(Character(name="Renfield", book_title="Dracula"))

        store = CharacterStore(db_path=str(path), store=rung.store)
        got = store.get("renfield")
        assert got is not None and got.name == "Renfield"
        assert rung.store.count("audio-characters__characters") == 1
        again = CharacterStore(db_path=str(path), store=rung.store)
        assert [c.name for c in again.list()] == ["Renfield"]
        assert rung.store.count("audio-characters__characters") == 1


class TestOverviewStorePort:
    def _overview(self, oid, title, created):
        from nomorals.audio.overview import (
            AudioOverview,
            Chapter,
            OverviewScript,
            ScriptLine,
        )

        return AudioOverview(
            overview_id=oid, title=title, format="deep-dive", lang="en",
            script=OverviewScript(
                format="deep-dive", lang="en",
                chapters=[Chapter(title="ch1", start_s=0.0)],
                lines=[ScriptLine(speaker="A", text="gold rose [S1]",
                                  cites=[0], chapter="ch1")],
                dropped=1),
            sources=[{"title": "Yaba report", "text": "gold rose"}],
            audio_path="", created_at=created)

    def _scenario(self, store) -> dict:
        saved = (store.save(self._overview("ov_old", "Old", 100.0)),
                 store.save(self._overview("ov_new", "New", 200.0)))
        got = store.get("ov_old")
        return {
            "saved": saved,
            "got": (got.title, got.format, len(got.script.lines),
                    got.script.lines[0].text, got.script.dropped,
                    got.sources[0]["title"]),
            "listed": [o.overview_id for o in store.list()],
            "latest": store.latest().overview_id,
            "missing": store.get("nope"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.audio.overview import OverviewStore

        store = OverviewStore(db_path=str(tmp_path / "ov.db"),
                              store=rung.store)
        observed = self._scenario(store)
        assert observed["saved"] == (True, True)
        assert observed["got"] == (
            "Old", "deep-dive", 1, "gold rose [S1]", 1, "Yaba report")
        assert observed["listed"] == ["ov_new", "ov_old"]
        assert observed["latest"] == "ov_new"
        assert observed["missing"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.audio.overview import OverviewStore

        path = tmp_path / "ov.db"
        native = OverviewStore(db_path=str(path))
        native.save(self._overview("ov_kept", "Kept", 5.0))

        store = OverviewStore(db_path=str(path), store=rung.store)
        got = store.get("ov_kept")
        assert got is not None and got.title == "Kept"
        assert rung.store.count("audio-overviews__overviews") == 1
        again = OverviewStore(db_path=str(path), store=rung.store)
        assert [o.overview_id for o in again.list()] == ["ov_kept"]
        assert rung.store.count("audio-overviews__overviews") == 1


class TestKeytermStorePort:
    def _scenario(self, store) -> dict:
        import time as _time

        adds = []
        # distinct `added` stamps: list order (ORDER BY added) is pinned
        for args in (("Adaeze", "name", ""), ("agbada", "jargon", ""),
                     ("ẹẹm", "dialect", "yo-ekiti"),
                     ("Lekki", "place", "yo")):
            adds.append(store.add(args[0], kind=args[1], lang=args[2]))
            _time.sleep(0.01)
        prompt = store.initial_prompt()
        return {
            "adds": tuple(adds),
            "bad_kind": store.add("hustle", kind="bogus"),
            "all": [(r["term"], r["kind"], r["lang"])
                    for r in store.list()],
            "names": [r["term"] for r in store.list(kind="name")],
            "yo": [r["term"] for r in store.list(lang="yo")],
            "prompt": ("Adaeze" in prompt, "agbada" in prompt,
                       "Lekki" in prompt),
            "removed": store.remove("AGBAda"),  # case-insensitive
            "removed_again": store.remove("agbada"),
            "after": [r["term"] for r in store.list()],
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.audio.pipeline import KeytermStore

        store = KeytermStore(db_path=str(tmp_path / "k.db"),
                             store=rung.store)
        observed = self._scenario(store)
        assert observed["adds"] == (True, True, True, True)
        assert observed["bad_kind"] is True  # unknown kind → jargon
        assert observed["all"] == [
            ("Adaeze", "name", ""), ("agbada", "jargon", ""),
            ("ẹẹm", "dialect", "yo-ekiti"), ("Lekki", "place", "yo"),
            ("hustle", "jargon", "")]
        assert observed["names"] == ["Adaeze"]
        # lang filter: the requested lang plus lang-less terms
        assert observed["yo"] == ["Adaeze", "agbada", "Lekki", "hustle"]
        assert observed["prompt"] == (True, True, True)
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["after"] == ["Adaeze", "ẹẹm", "Lekki", "hustle"]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.audio.pipeline import KeytermStore

        path = tmp_path / "k.db"
        native = KeytermStore(db_path=str(path))
        native.add("Adaeze", kind="name")

        store = KeytermStore(db_path=str(path), store=rung.store)
        assert [r["term"] for r in store.list()] == ["Adaeze"]
        assert rung.store.count("audio-keyterms__keyterms") == 1
        doc = rung.store.get("audio-keyterms__keyterms", "Adaeze")
        assert doc is not None and doc["kind"] == "name"
        again = KeytermStore(db_path=str(path), store=rung.store)
        assert [r["term"] for r in again.list()] == ["Adaeze"]
        assert rung.store.count("audio-keyterms__keyterms") == 1


class TestMelodyCatalogPort:
    _ALPHA = (60, 64, 67, 72, 76, 79, 84, 79, 76, 72)
    _BETA = (48, 55, 62, 59, 53, 65, 50, 57, 64, 60)

    @staticmethod
    def _melody(midis):
        from nomorals.audio.melodic import Melody, MelodyNote

        notes = tuple(MelodyNote(midi=m, start_s=i * 0.4, dur_s=0.4)
                      for i, m in enumerate(midis))
        chroma = [0.0] * 12
        for m in midis:
            chroma[m % 12] += 1.0
        total = sum(chroma)
        return Melody(notes=notes,
                      chroma=tuple(c / total for c in chroma))

    def _patch_extraction(self, monkeypatch):
        import nomorals.audio.melodic as melodic

        melodies = {"alpha.wav": self._melody(self._ALPHA),
                    "beta.wav": self._melody(self._BETA)}

        def fake(path, *, prefer_stem=True):
            return melodies[Path(str(path)).name], "f0-test"

        monkeypatch.setattr(melodic, "melody_from_path", fake)

    def _scenario(self, catalog) -> dict:
        import time as _time

        a = catalog.add_work("alpha.wav", title="Alpha",
                             artist="Fixture")
        _time.sleep(0.02)  # distinct created stamps: order is pinned
        b = catalog.add_work("beta.wav", title="Beta", artist="Fixture")
        _time.sleep(0.02)
        # re-enrolling supersedes (INSERT OR REPLACE): still 2 works,
        # and Alpha's created stamp is now the newest
        again = catalog.add_work("alpha.wav", title="Alpha",
                                 artist="Fixture")
        works = catalog.list_works()
        stats = catalog.stats()
        snap = catalog.catalog_snapshot()
        hyps = catalog.rank_hypotheses(self._melody(self._ALPHA))
        return {
            "ok": (a["ok"], b["ok"], again["ok"]),
            "same_id": a["work_id"] == again["work_id"],
            "titles": [w["title"] for w in works],
            "notes": [w["notes"] for w in works],
            "stats": (stats["works"], stats["intervals"]),
            "snap": (snap["works"], snap["intervals"],
                     snap["version"].split(":")[0]),
            "top": hyps[0]["title"] if hyps else None,
            "removed": catalog.remove_work(b["work_id"]),
            "removed_again": catalog.remove_work(b["work_id"]),
            "after": [w["title"] for w in catalog.list_works()],
        }

    def test_conformance(self, rung, tmp_path, monkeypatch):
        from nomorals.audio.melodic import MelodyCatalog

        self._patch_extraction(monkeypatch)
        catalog = MelodyCatalog(str(tmp_path / "melodies.db"),
                                store=rung.store)
        try:
            observed = self._scenario(catalog)
        finally:
            catalog.close()
        assert observed["ok"] == (True, True, True)
        assert observed["same_id"] is True
        assert observed["titles"] == ["Beta", "Alpha"]
        assert observed["notes"] == [10, 10]
        assert observed["stats"] == (2, 18)
        assert observed["snap"] == (2, 18, "local-melody-v1")
        assert observed["top"] == "Alpha"
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["after"] == ["Alpha"]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path,
                                        monkeypatch):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import json as _json

        from nomorals.audio.melodic import MelodyCatalog

        self._patch_extraction(monkeypatch)
        path = tmp_path / "melodies.db"
        native = MelodyCatalog(str(path))
        kept = native.add_work("alpha.wav", title="Alpha",
                               artist="Fixture")
        native.close()

        catalog = MelodyCatalog(str(path), store=rung.store)
        try:
            assert [w["title"] for w in catalog.list_works()] == ["Alpha"]
            assert rung.store.count("audio-melodic__works") == 1
            doc = rung.store.get("audio-melodic__works", kept["work_id"])
            assert doc is not None and doc["title"] == "Alpha"
            # the segmented melody rides as its JSON text (a field)
            assert _json.loads(doc["notes"])[0][0] == 60
            # doc-24 class separation is structural: nothing landed
            # in the class-1 fingerprint catalog's collections
            assert rung.store.count("audio-fingerprints__tracks") == 0
            assert rung.store.count("audio-fingerprints__hashes") == 0
        finally:
            catalog.close()
        # second construction: adopted, not re-imported or duplicated
        again = MelodyCatalog(str(path), store=rung.store)
        try:
            assert [w["title"] for w in again.list_works()] == ["Alpha"]
            assert rung.store.count("audio-melodic__works") == 1
        finally:
            again.close()


class TestScamStorePort:
    def _scenario(self, store) -> dict:
        store.remember("ph_a", "12 Allen Ave, Ikeja",
                       "http://x/1", 150_000_00)
        store.remember("ph_b", "12 Allen Ave, Ikeja",
                       "http://x/2", 140_000_00)
        store.remember("ph_a", "9 Other Rd, Lekki",
                       "http://x/3", 200_000_00)
        # INSERT OR IGNORE: the same (photo, address, url) triple
        # keeps the first row, price and all
        store.remember("ph_a", "12 Allen Ave, Ikeja",
                       "http://x/1", 999)
        by_photo = store.find_duplicate("ph_a", "")
        by_addr = store.find_duplicate("", "12 Allen Ave, Ikeja")
        both = store.find_duplicate("ph_a", "12 Allen Ave, Ikeja")
        return {
            "photo_urls": sorted(r["url"] for r in by_photo),
            "kept_price": [r["price_kobo"] for r in by_photo
                           if r["url"] == "http://x/1"],
            "addr_urls": sorted(r["url"] for r in by_addr),
            "both_urls": sorted(r["url"] for r in both),
            "none": store.find_duplicate("ph_zz", "nowhere"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.property.scam import ScamStore

        store = ScamStore(db_path=str(tmp_path / "scam.db"),
                          store=rung.store)
        observed = self._scenario(store)
        assert observed["photo_urls"] == ["http://x/1", "http://x/3"]
        assert observed["kept_price"] == [150_000_00]
        assert observed["addr_urls"] == ["http://x/1", "http://x/2"]
        assert observed["both_urls"] == [
            "http://x/1", "http://x/2", "http://x/3"]
        assert observed["none"] == []

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import hashlib as _hashlib

        from nomorals.property.scam import ScamStore

        path = tmp_path / "scam.db"
        native = ScamStore(db_path=str(path))
        native.remember("ph_a", "12 Allen Ave, Ikeja",
                        "http://x/1", 150_000_00)

        store = ScamStore(db_path=str(path), store=rung.store)
        found = store.find_duplicate("ph_a", "")
        assert [r["url"] for r in found] == ["http://x/1"]
        assert rung.store.count("property-scam__seen_listings") == 1
        listing_id = _hashlib.md5(
            b"ph_a|12 Allen Ave, Ikeja|http://x/1"
        ).hexdigest()[:16]
        doc = rung.store.get("property-scam__seen_listings", listing_id)
        assert doc is not None and doc["price_kobo"] == 150_000_00
        again = ScamStore(db_path=str(path), store=rung.store)
        assert len(again.find_duplicate("ph_a", "")) == 1
        assert rung.store.count("property-scam__seen_listings") == 1


class TestValueStorePort:
    @staticmethod
    def _estimate(eid, area, bedrooms, point, created):
        from nomorals.property.value import Comp, ValueEstimate

        return ValueEstimate(
            estimate_id=eid, area=area, bedrooms=bedrooms,
            point_kobo=point, low_kobo=point - 10, high_kobo=point + 10,
            fsd=0.12,
            comps=[Comp(comp_id="c1", price_kobo=point, area=area,
                        bedrooms=bedrooms, title="Listing One",
                        source="seed")],
            kept_ids=["c1"], thin_market=True, created_at=created)

    def _scenario(self, store) -> dict:
        saved = (
            store.save(self._estimate("val_1", "ikeja", "3", 5000, 100.0)),
            store.save(self._estimate("val_2", "ikeja", "3", 7000, 200.0)),
            store.save(self._estimate("val_3", "lekki", "2", 9000, 300.0)))
        # INSERT OR REPLACE: val_1's point moves, its stamp stays
        store.save(self._estimate("val_1", "ikeja", "3", 5500, 100.0))
        got = store.get("val_1")
        trend = store.trend("ikeja", "3")
        return {
            "saved": saved,
            "got": (got.point_kobo, got.area, got.comps[0].title,
                    got.kept_ids, got.thin_market),
            "hist_area": [e.id for e in store.history("ikeja", "3")],
            "hist_all": [e.id for e in store.history()],
            "hist_limit1": [e.id for e in store.history(limit=1)],
            "trend": (trend["ok"], trend["direction"], trend["n"]),
            "missing": store.get("nope"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.property.value import ValueStore

        store = ValueStore(db_path=str(tmp_path / "values.db"),
                           store=rung.store)
        observed = self._scenario(store)
        assert observed["saved"] == (True, True, True)
        assert observed["got"] == (
            5500, "ikeja", "Listing One", ["c1"], True)
        assert observed["hist_area"] == ["val_2", "val_1"]
        assert observed["hist_all"] == ["val_3", "val_2", "val_1"]
        assert observed["hist_limit1"] == ["val_3"]
        assert observed["trend"] == (True, "up", 2)
        assert observed["missing"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.property.value import ValueStore

        path = tmp_path / "values.db"
        native = ValueStore(db_path=str(path))
        native.save(self._estimate("val_kept", "ikeja", "3", 5000, 100.0))

        store = ValueStore(db_path=str(path), store=rung.store)
        got = store.get("val_kept")
        assert got is not None and got.point_kobo == 5000
        assert rung.store.count("property-value__value_estimates") == 1
        doc = rung.store.get("property-value__value_estimates", "val_kept")
        assert doc is not None and "Listing One" in doc["comps_json"]
        again = ValueStore(db_path=str(path), store=rung.store)
        assert [e.id for e in again.history()] == ["val_kept"]
        assert rung.store.count("property-value__value_estimates") == 1


class TestEvergreenQueuePort:
    def _scenario(self, queue) -> dict:
        old = queue.add("Old winner", "tips", engagement=90.0,
                        min_engagement=50.0, cooldown_days=30.0)
        fresh = queue.add("Fresh winner", "tips", engagement=80.0,
                          min_engagement=50.0)
        below = queue.add("Below floor", "tips", engagement=10.0,
                          min_engagement=50.0)
        other = queue.add("Other cat", "wins", engagement=99.0,
                          min_engagement=50.0)
        due_all = queue.next_due()
        due_tips = queue.next_due("tips")
        fresh_post = next(p for p in due_tips
                          if p.content == "Fresh winner")
        queue.mark_posted(old)
        due_after = queue.next_due()
        queue.mark_posted(fresh)
        return {
            "ids_ok": all([old, fresh, below, other]),
            "due_all": sorted(p.content for p in due_all),
            "due_tips": sorted(p.content for p in due_tips),
            "fresh_fields": (fresh_post.category, fresh_post.engagement,
                             fresh_post.last_posted_at,
                             fresh_post.cooldown_days),
            "due_after": sorted(p.content for p in due_after),
            "due_final": [p.content for p in queue.next_due()],
            "limited1": [p.content for p in queue.next_due(limit=1)],
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.social.voice import EvergreenQueue

        queue = EvergreenQueue(tmp_path / "evergreen.db",
                               store=rung.store)
        observed = self._scenario(queue)
        assert observed["ids_ok"] is True
        assert observed["due_all"] == [
            "Fresh winner", "Old winner", "Other cat"]
        assert observed["due_tips"] == ["Fresh winner", "Old winner"]
        assert observed["fresh_fields"] == ("tips", 80.0, None, 30.0)
        assert observed["due_after"] == ["Fresh winner", "Other cat"]
        assert observed["due_final"] == ["Other cat"]
        assert observed["limited1"] == ["Other cat"]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.voice import EvergreenQueue

        path = tmp_path / "evergreen.db"
        native = EvergreenQueue(path)
        kept = native.add("Kept winner", "tips", engagement=90.0,
                          min_engagement=50.0)

        queue = EvergreenQueue(path, store=rung.store)
        assert [p.content for p in queue.next_due()] == ["Kept winner"]
        assert rung.store.count("social-voice__evergreen") == 1
        doc = rung.store.get("social-voice__evergreen", kept)
        assert doc is not None and doc["content"] == "Kept winner"
        queue.mark_posted(kept)
        assert queue.next_due() == []
        again = EvergreenQueue(path, store=rung.store)
        assert again.next_due() == []
        assert rung.store.count("social-voice__evergreen") == 1


class TestVoiceNoteStorePort:
    @staticmethod
    def _stt(path, lang):
        return {"text": f"talk about the market in {lang}",
                "backend": "fake-stt"}

    def _scenario(self, store, audio) -> dict:
        import time as _time

        n1 = store.store_voice_note("chat-1", "Ada", audio,
                                    language="en")
        _time.sleep(0.01)  # distinct created stamps: order is pinned
        n2 = store.store_voice_note("chat-1", "Bola", audio,
                                    language="yo")
        _time.sleep(0.01)
        n3 = store.store_voice_note("chat-2", "Ada", audio,
                                    language="en")
        got = store.get(n1.note_id)
        stats_all = store.stats()
        stats_chat = store.stats("chat-2")
        return {
            "stored": (n1 is not None, n2 is not None, n3 is not None),
            "got": (got.chat_id, got.sender, got.transcript,
                    got.language, got.stt_backend),
            "listed": [n.sender for n in store.list("chat-1")],
            "stats_all": (stats_all["total"], stats_all["transcribed"],
                          stats_all["transcript_coverage"],
                          stats_all["languages"]),
            "stats_chat": (stats_chat["total"],
                           stats_chat["transcribed"]),
            "found": [n.sender for n in
                      store.search_voice_notes("market")],
            "missing": store.get("vn_nope"),
            "bad_chat": store.store_voice_note("  ", "Ada", audio),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.social.voice_notes import VoiceNoteStore

        audio = tmp_path / "note.wav"
        audio.write_bytes(b"fake-audio")
        store = VoiceNoteStore(db_path=str(tmp_path / "vn.db"),
                               vault_dir=str(tmp_path / "vault"),
                               stt=self._stt, store=rung.store)
        observed = self._scenario(store, str(audio))
        assert observed["stored"] == (True, True, True)
        assert observed["got"] == (
            "chat-1", "Ada", "talk about the market in en", "en",
            "fake-stt")
        assert observed["listed"] == ["Ada", "Bola"]
        assert observed["stats_all"] == (
            3, 3, 1.0, {"en": 2, "yo": 1})
        assert observed["stats_chat"] == (1, 1)
        assert observed["found"] == ["Ada", "Bola", "Ada"]
        assert observed["missing"] is None
        assert observed["bad_chat"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.voice_notes import VoiceNoteStore

        audio = tmp_path / "note.wav"
        audio.write_bytes(b"fake-audio")
        path = tmp_path / "vn.db"
        vault = str(tmp_path / "vault")
        native = VoiceNoteStore(db_path=str(path), vault_dir=vault,
                                stt=self._stt)
        kept = native.store_voice_note("chat-1", "Ada", str(audio),
                                       language="en")

        store = VoiceNoteStore(db_path=str(path), vault_dir=vault,
                               stt=self._stt, store=rung.store)
        got = store.get(kept.note_id)
        assert got is not None and got.sender == "Ada"
        assert rung.store.count("social-voice-notes__voice_notes") == 1
        doc = rung.store.get("social-voice-notes__voice_notes",
                             kept.note_id)
        assert doc is not None \
            and doc["transcript"] == "talk about the market in en"
        again = VoiceNoteStore(db_path=str(path), vault_dir=vault,
                               stt=self._stt, store=rung.store)
        assert [n.sender for n in again.list()] == ["Ada"]
        assert rung.store.count("social-voice-notes__voice_notes") == 1


class TestCalendarIntegrationPort:
    @staticmethod
    def _integration(store=None, counter=None):
        from unittest.mock import MagicMock

        import nomorals.integrations.calendar_integration as cal_mod

        integ = cal_mod.CalendarIntegration(MagicMock(), MagicMock(),
                                            store=store)
        integ._detect_backend = lambda account: "google_api"
        calls = []
        tally = counter if counter is not None else [0]

        def fake_request(account, method, url, payload=None, params=None):
            calls.append(dict(params or {}))
            tally[0] += 1
            return {"items": [], "nextSyncToken": f"tok-{tally[0]}"}

        integ._request = fake_request
        return integ, calls

    def test_conformance(self, rung, tmp_path, monkeypatch):
        import asyncio

        import nomorals.integrations.calendar_integration as cal_mod

        monkeypatch.setattr(cal_mod, "_SYNC_DB", str(tmp_path / "sync.db"))
        integ, calls = self._integration(store=rung.store)
        first = asyncio.run(integ.sync_changes("a@gmail.com"))
        second = asyncio.run(integ.sync_changes("a@gmail.com"))
        other = asyncio.run(
            integ.sync_changes("a@gmail.com", calendar_id="work"))
        assert first["full_sync"] is True
        assert calls[0].get("syncToken") is None
        assert second["full_sync"] is False
        assert calls[1].get("syncToken") == "tok-1"
        # a different calendar keeps its own cursor
        assert other["full_sync"] is True
        assert calls[2].get("syncToken") is None

    def test_cursor_continuity_across_the_move(self, rung, tmp_path,
                                               monkeypatch):
        """The registry's continuity pin: a cursor advanced on the
        native path is the cursor the seam resumes from — a reset
        would mean a full re-sync storm against the external
        system's token stream."""
        if rung.store is None:
            pytest.skip("native rung has no move to prove")
        import asyncio

        import nomorals.integrations.calendar_integration as cal_mod

        monkeypatch.setattr(cal_mod, "_SYNC_DB", str(tmp_path / "sync.db"))
        counter = [0]
        # phase 1: native construction advances the cursor to tok-1
        native, _native_calls = self._integration(counter=counter)
        first = asyncio.run(native.sync_changes("a@gmail.com"))
        assert first["full_sync"] is True
        # phase 2: seam construction imports the file's cursor and
        # resumes from it, then advances it in its own collection
        seam, seam_calls = self._integration(store=rung.store,
                                             counter=counter)
        second = asyncio.run(seam.sync_changes("a@gmail.com"))
        assert second["full_sync"] is False
        assert seam_calls[0].get("syncToken") == "tok-1"
        assert rung.store.count(
            "calendar-integration__sync_cursors") == 1
        doc = rung.store.get("calendar-integration__sync_cursors",
                             "a@gmail.com\x1fprimary")
        assert doc is not None and doc["sync_token"] == "tok-2"
        # phase 3: a second seam construction adopts the advanced
        # cursor — never re-imports, never resets
        again, again_calls = self._integration(store=rung.store,
                                               counter=counter)
        third = asyncio.run(again.sync_changes("a@gmail.com"))
        assert third["full_sync"] is False
        assert again_calls[0].get("syncToken") == "tok-2"
        assert rung.store.count(
            "calendar-integration__sync_cursors") == 1


class TestAmbientPickup:
    def test_boot_ambient_routes_sidecars_without_signature_changes(
            self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung: ambient stays empty by design")
        from nomorals.social.drafts import DraftQueue

        set_ambient_document_store(rung.store)
        queue = DraftQueue(tmp_path / "ambient.db")  # no store kwarg
        try:
            assert queue._port is not None
            draft = queue.create_draft("ambient-routed")
            assert rung.store.get(
                "social-drafts__social_drafts", draft.id) is not None
        finally:
            queue.close()
            set_ambient_document_store(None)
        native = DraftQueue(tmp_path / "native.db")
        try:
            assert native._port is None
        finally:
            native.close()


# ---------------------------------------------------------------------------
# analytics backend-awareness
# ---------------------------------------------------------------------------

def _migrated_db(tmp_path) -> Database:
    db, _recovered, _backup = open_database(tmp_path / "nomorals.db")
    return db


def _watcher_rows(db: Database) -> list[dict]:
    """Two watcher rows, materialized through a native insert + read-back.

    Defaults and storage types are whatever the migrated schema really
    produces — the seam rungs must reproduce these dicts exactly.
    The fossil table is emptied again before returning, so a seam
    rung's analytics can only be reading the seam.
    """
    for i, name in enumerate(("gold", "repo")):
        db.insert("watchers", {
            "id": f"w{i + 1}", "name": name, "kind": "price",
            "interval_s": 3600.0, "severity": "info", "state": "active",
        })
    rows = [dict(r) for r in db.query("SELECT * FROM watchers ORDER BY id")]
    db.execute("DELETE FROM watchers")
    return rows


class TestAnalyticsBackendAwareness:
    def test_source_rows_equal_on_every_rung(self, rung, tmp_path):
        from nomorals.storage.analytics import analytics_source_rows
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
                assert analytics_source_rows(db, "watchers") is None
                live = [dict(r) for r in db.query(
                    "SELECT * FROM watchers ORDER BY id")]
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                assert table is not None
                for row in expected:
                    table.put(row)
                live = analytics_source_rows(db, "watchers")
            assert live == expected
        finally:
            db.close()

    def test_sqlite_analytics_counts_and_guard(self, rung, tmp_path):
        from nomorals.core.errors import StorageError
        from nomorals.storage.analytics import SQLiteAnalytics
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            engine = SQLiteAnalytics(db)
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
                assert engine.load_table(db, "watchers") == 2
                assert len(engine.query("SELECT * FROM watchers")) == 2
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                for row in expected:
                    table.put(row)
                # the count is the seam's truth, not the fossil's 0:
                assert engine.load_table(db, "watchers") == 2
                assert engine.load_table(
                    db, "watchers", where="name = ?",
                    params=("gold",)) == 1
                # passthrough SQL over a seam-served table refuses
                # loudly; a pinned/SQLite-served table still reads:
                with pytest.raises(StorageError):
                    engine.query("SELECT * FROM watchers")
                assert engine.query("SELECT * FROM memories") == []
        finally:
            db.close()

    def test_duckdb_sync_reads_the_seam(self, rung, tmp_path):
        pytest.importorskip("duckdb")
        from nomorals.storage.analytics import DuckDBAnalytics
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            engine = DuckDBAnalytics()
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                for row in expected:
                    table.put(row)
            assert engine.load_table(db, "watchers") == 2
            names = [r["name"] for r in engine.query(
                "SELECT name FROM watchers ORDER BY name")]
            assert names == ["gold", "repo"]
            assert engine.load_table(
                db, "watchers", where="name = ?",
                params=("gold",)) == 1
        finally:
            db.close()

    def test_kv_store_rows_come_from_the_seam(self, rung, tmp_path):
        from nomorals.storage.analytics import analytics_source_rows
        from nomorals.storage.kv import KVStore

        db = _migrated_db(tmp_path)
        try:
            if rung.store is None:
                pytest.skip("kv seam shape only exists with a store attached")
            db.document_store = rung.store
            kv = KVStore(db)
            kv.set("analytics:probe", {"v": 7})
            rows = analytics_source_rows(db, "kv_store")
            assert rows is not None and len(rows) == 1
            assert rows[0]["key"].endswith("analytics:probe")
            assert rows[0]["value"] == '{"v": 7}' or rows[0]["value"] == {"v": 7}
        finally:
            db.close()



class _CoachSource:
    """HealthDataSource-shaped fake: steady HRV, good sleep, no strain."""

    def has_data(self):
        return True

    def metrics(self, start, end):
        return [{"heart_rate_variability_ms": 52.0, "step_count": 8000}
                for _ in range(30)]

    def sleep_sessions(self, start, end):
        return [{"sleep_in_bed_duration_sec": 8.2 * 3600,
                 "sleep_awake_duration_sec": 0.3 * 3600,
                 "start_datetime": "2026-10-09T22:30:00-04:00",
                 "end_datetime": "2026-10-10T06:30:00-04:00"}
                for _ in range(7)]

    def workouts(self, start, end):
        return []


class _StubTimeline:
    def __init__(self):
        self.logged = []

    def log(self, kind, text, source=""):
        self.logged.append((kind, text, source))


class TestHealthCoachPort:
    def _coach(self, tmp_path, store):
        from nomorals.health.coach import HealthCoach

        coach = HealthCoach(source=_CoachSource(),
                            timeline=_StubTimeline(), store=store)
        path = str(tmp_path / "coach.db")
        coach._history_path = lambda: path  # keep history out of $HOME
        return coach

    def _scenario(self, coach) -> dict:
        import time as _time

        r1 = coach.readiness(log=True)
        _time.sleep(0.02)  # distinct ts: the log's PK
        r2 = coach.readiness(log=True)
        hist = coach.readiness_history(days=7)
        return {
            "levels": (r1.level, r2.level),
            "scores": (r1.score, r2.score),
            "hist_levels": [p.level for p in hist],
            "hist_scores": [p.score for p in hist],
            "hist_ascending": [p.ts for p in hist] == sorted(
                p.ts for p in hist),
        }

    def test_conformance(self, rung, tmp_path):
        coach = self._coach(tmp_path, rung.store)
        observed = self._scenario(coach)
        assert observed["levels"] == ("high", "high")
        assert observed["hist_levels"] == ["high", "high"]
        assert observed["hist_scores"] == list(observed["scores"])
        assert observed["hist_ascending"] is True

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        native = self._coach(tmp_path, None)
        native.readiness(log=True)

        coach = self._coach(tmp_path, rung.store)
        hist = coach.readiness_history(days=7)
        assert len(hist) == 1 and hist[0].level == "high"
        assert rung.store.count("health-coach__readiness_log") == 1
        doc = rung.store.find("health-coach__readiness_log")[0]
        assert doc["level"] == "high"
        assert doc["numbers_json"]
        # second construction: adopted, not re-imported or duplicated
        again = self._coach(tmp_path, rung.store)
        assert len(again.readiness_history(days=7)) == 1
        assert rung.store.count("health-coach__readiness_log") == 1


class TestHealthFormPort:
    def _scenario(self, store) -> dict:
        from nomorals.health.form import (
            FormAnalysis,
            FormIssue,
            form_trend,
        )

        issue = FormIssue(checkpoint="knees", label="knees",
                          observed="valgus collapse",
                          fix="push knees out")
        a1 = FormAnalysis(id="fa_1", exercise="squat", available=True,
                          score=0.80, band="good", passes=["depth"],
                          frames_used=12, created_at=100.0)
        a2 = FormAnalysis(id="fa_2", exercise="squat", available=True,
                          score=0.90, band="solid", issues=[issue],
                          passes=["depth", "knees"], frames_used=14,
                          created_at=200.0)
        a3 = FormAnalysis(id="fa_3", exercise="deadlift", available=True,
                          score=0.70, band="good", created_at=150.0)
        saved = (store.save(a1), store.save(a2), store.save(a3))
        a1.score = 0.85  # INSERT OR REPLACE on re-save
        resaved = store.save(a1)
        got = store.get("fa_2")
        trend = form_trend(store, "squat")
        return {
            "saved": saved, "resaved": resaved,
            "got": (got.exercise, got.score, got.band,
                    [i.checkpoint for i in got.issues], got.frames_used),
            "latest": store.latest().id,
            "latest_squat": store.latest("squat").id,
            "latest_deadlift": store.latest("deadlift").id,
            "latest_missing": store.latest("plank"),
            "trend": trend,
            "missing": store.get("nope"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.health.form import FormStore

        store = FormStore(db_path=str(tmp_path / "form.db"),
                          store=rung.store)
        observed = self._scenario(store)
        assert observed["saved"] == (True, True, True)
        assert observed["resaved"] is True
        assert observed["got"] == ("squat", 0.90, "solid", ["knees"], 14)
        assert observed["latest"] == "fa_2"
        assert observed["latest_squat"] == "fa_2"
        assert observed["latest_deadlift"] == "fa_3"
        assert observed["latest_missing"] is None
        assert "squat form trend (2 checks)" in observed["trend"]
        assert "85%" in observed["trend"]
        assert "90%" in observed["trend"]
        assert observed["missing"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.form import FormAnalysis, FormStore

        path = tmp_path / "form.db"
        native = FormStore(db_path=str(path))
        native.save(FormAnalysis(id="fa_kept", exercise="squat",
                                 available=True, score=0.75, band="good",
                                 created_at=50.0))

        store = FormStore(db_path=str(path), store=rung.store)
        got = store.get("fa_kept")
        assert got is not None and got.score == 0.75
        assert rung.store.count("health-form__form_analyses") == 1
        doc = rung.store.get("health-form__form_analyses", "fa_kept")
        assert doc is not None and doc["exercise"] == "squat"
        again = FormStore(db_path=str(path), store=rung.store)
        assert again.latest().id == "fa_kept"
        assert rung.store.count("health-form__form_analyses") == 1


class TestLifeArchivePort:
    @staticmethod
    def _port_for(path, rung):
        # life/archive.py is L0: the composition layer (this test)
        # builds the port and injects it — the module never imports
        # storage.
        from nomorals.life.archive import _DDL
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("life-archive", _DDL, ("life_events",),
                            None if path is None else str(path),
                            store=rung.store)

    def _scenario(self, archive) -> dict:
        id1 = archive.record_event(
            "journal", "note", "Morning pages", ts=100.0,
            provenance="journal:2026-11-11", owner=True)
        id2 = archive.record_event(
            "health", "metric", "Slept well", ts=300.0, owner=True)
        id3 = archive.record_event(
            "journal", "idea", "Build the thing", ts=200.0,
            private=False, owner=True)
        got = archive.get(id2, owner=True)
        denied = False
        try:
            archive.get(id1)  # owner defaults False → denied
        except PermissionError:
            denied = True
        return {
            "got": (got["text"], got["kind"], got["private"],
                    got["provenance"]),
            "public_event": archive.get(id3, owner=True)["private"],
            "timeline": [e["id"] for e in archive.timeline(owner=True)],
            "expect_order": [id2, id3, id1],
            "kind_filter": [e["id"] for e in archive.timeline(
                kind="note", owner=True)],
            "search_text": [e["id"] for e in archive.search(
                "slept", owner=True)],
            "search_kind": [e["id"] for e in archive.search(
                "IDEA", owner=True)],
            "missing": archive.get("nope", owner=True),
            "denied": denied,
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.life.archive import LifeArchive

        path = tmp_path / "life.db"
        archive = LifeArchive(db_path=str(path),
                              port=self._port_for(path, rung))
        observed = self._scenario(archive)
        assert observed["got"] == ("Slept well", "metric", True, "")
        assert observed["public_event"] is False
        assert observed["timeline"] == observed["expect_order"]
        assert observed["kind_filter"] == [observed["expect_order"][2]]
        assert observed["search_text"] == [observed["expect_order"][0]]
        assert observed["search_kind"] == [observed["expect_order"][1]]
        assert observed["missing"] is None
        assert observed["denied"] is True
        archive.close()

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.life.archive import LifeArchive

        path = tmp_path / "life.db"
        native = LifeArchive(db_path=str(path))
        kept = native.record_event("journal", "note", "Kept event",
                                   ts=42.0, owner=True)
        native.close()

        archive = LifeArchive(db_path=str(path),
                              port=self._port_for(path, rung))
        got = archive.get(kept, owner=True)
        assert got is not None and got["text"] == "Kept event"
        assert rung.store.count("life-archive__life_events") == 1
        doc = rung.store.get("life-archive__life_events", kept)
        assert doc is not None and doc["kind"] == "note"
        again = LifeArchive(db_path=str(path),
                            port=self._port_for(path, rung))
        assert [e["id"] for e in again.timeline(owner=True)] == [kept]
        assert rung.store.count("life-archive__life_events") == 1
        # a ':memory:' archive on the seam imports nothing, still works
        mem = LifeArchive(port=self._port_for(None, rung))
        mid = mem.record_event("journal", "note", "Seam-only event",
                               owner=True)
        assert mem.get(mid, owner=True)["text"] == "Seam-only event"


class TestCostDisplayPort:
    def _scenario(self, disp) -> dict:
        import time as _time

        return {
            "before": disp.is_enabled("owner"),
            "on": disp.set_enabled(True, "owner"),
            "after_on": disp.is_enabled("owner"),
            "other_user": disp.is_enabled("someone-else"),
            "footer_on": disp.footer_for_turn(_time.time(), "owner"),
            "off": disp.set_enabled(False, "owner"),
            "after_off": disp.is_enabled("owner"),
            "default_user": disp.is_enabled(),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.llm.cost_display import CostDisplay

        disp = CostDisplay(db_path=str(tmp_path / "cost.db"),
                           store=rung.store)
        observed = self._scenario(disp)
        assert observed["before"] is False
        assert observed["on"] is True
        assert observed["after_on"] is True
        assert observed["other_user"] is False
        assert observed["footer_on"].startswith("\n\n_this answer cost")
        assert observed["off"] is True
        assert observed["after_off"] is False
        assert observed["default_user"] is False

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.llm.cost_display import CostDisplay

        path = tmp_path / "cost.db"
        native = CostDisplay(db_path=str(path))
        native.set_enabled(True, "owner")

        disp = CostDisplay(db_path=str(path), store=rung.store)
        assert disp.is_enabled("owner") is True
        assert rung.store.count("llm-cost-display__cost_display") == 1
        doc = rung.store.get("llm-cost-display__cost_display", "owner")
        assert doc is not None and doc["enabled"] == 1
        again = CostDisplay(db_path=str(path), store=rung.store)
        assert again.is_enabled("owner") is True
        assert rung.store.count("llm-cost-display__cost_display") == 1


class TestAEOTrackerPort:
    def _scenario(self, tracker) -> dict:
        from nomorals.marketing.aeo import VisibilityReport

        tracker._record(VisibilityReport(
            report_id="aeo_t1", brand="Devon", prompts=[], engines=[],
            gaps=["q1"], created_at=100.0))
        tracker._record(VisibilityReport(
            report_id="aeo_t2", brand="Devon", prompts=[], engines=[],
            created_at=200.0))
        tracker._record(VisibilityReport(
            report_id="aeo_t3", brand="Other", prompts=[], engines=[],
            gaps=["q9"], created_at=150.0))
        got = tracker.get("aeo_t2")
        hist = tracker.history("Devon")
        return {
            "got": (got.brand, got.gaps, got.created_at),
            "got_gaps": tracker.get("aeo_t1").gaps,
            "latest": tracker.latest().report_id,
            "latest_brand": tracker.latest("Other").report_id,
            "latest_missing": tracker.latest("Nope"),
            "hist_ids": [r["report_id"] for r in hist],
            "hist_keys": sorted(hist[0].keys()),
            "hist_all": [r["report_id"] for r in tracker.history()],
            "missing": tracker.get("nope"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.marketing.aeo import AEOTracker

        tracker = AEOTracker(db_path=str(tmp_path / "aeo.db"),
                             store=rung.store)
        observed = self._scenario(tracker)
        assert observed["got"] == ("Devon", [], 200.0)
        assert observed["got_gaps"] == ["q1"]
        assert observed["latest"] == "aeo_t2"
        assert observed["latest_brand"] == "aeo_t3"
        assert observed["latest_missing"] is None
        assert observed["hist_ids"] == ["aeo_t2", "aeo_t1"]
        assert observed["hist_keys"] == [
            "brand", "confirmed", "created_at", "pairs", "report_id",
            "share"]
        assert observed["hist_all"] == ["aeo_t2", "aeo_t3", "aeo_t1"]
        assert observed["missing"] is None

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.marketing.aeo import AEOTracker, VisibilityReport

        path = tmp_path / "aeo.db"
        native = AEOTracker(db_path=str(path))
        native._record(VisibilityReport(
            report_id="aeo_kept", brand="Devon", prompts=[], engines=[],
            created_at=50.0))

        tracker = AEOTracker(db_path=str(path), store=rung.store)
        got = tracker.get("aeo_kept")
        assert got is not None and got.brand == "Devon"
        assert rung.store.count("marketing-aeo__aeo_reports") == 1
        doc = rung.store.get("marketing-aeo__aeo_reports", "aeo_kept")
        assert doc is not None and doc["brand"] == "Devon"
        again = AEOTracker(db_path=str(path), store=rung.store)
        assert again.latest().report_id == "aeo_kept"
        assert rung.store.count("marketing-aeo__aeo_reports") == 1


class _FakeUrlResp:
    """urlopen stand-in owned by TestShoppingDealsPort: a product
    page carrying one data-price (the failures suite's shape)."""

    def __init__(self, html: str) -> None:
        self._html = html

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._html.encode()


class TestShoppingDealsPort:
    """B2, the shared-file group: two engines over one Database
    handle, six tables, ONE shared port — whichever engine
    constructs first imports the file's history for all six."""

    @staticmethod
    def _engines(path, store, monkeypatch):
        import urllib.request
        from unittest.mock import MagicMock

        from nomorals.integrations.naija_deals import NaijaDealHunter
        from nomorals.integrations.naija_shopping import (
            NaijaShoppingEngine,
        )
        from nomorals.storage.db import Database

        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **k: _FakeUrlResp(
                '<div data-price="700"></div>'))
        db = Database(str(path))
        engine = NaijaShoppingEngine(db, MagicMock(), store=store)
        hunter = NaijaDealHunter(MagicMock(), db, store=store)
        return engine, hunter

    def _scenario(self, engine, hunter) -> dict:
        import asyncio
        import time

        from nomorals.integrations.naija_deals import Deal
        from nomorals.integrations.naija_shopping import (
            PriceSnapshot,
            Steal,
        )

        now = time.time()
        url = "https://www.jumia.com.ng/p/1"
        engine._store_snapshot(PriceSnapshot(
            product_url=url, site="jumia", price_ngn=1000.0,
            title="Phone X", timestamp=now - 300))
        engine._store_snapshot(PriceSnapshot(
            product_url=url, site="jumia", price_ngn=800.0,
            title="Phone X", timestamp=now - 200))
        engine._store_snapshot(PriceSnapshot(
            product_url="https://www.konga.com/p/1", site="konga",
            price_ngn=900.0, title="Phone X", timestamp=now - 100))

        steal = asyncio.run(engine.compute_steal_score(url))
        engine._store_steal(Steal(
            product_url=url, site="jumia", title="Phone X",
            current_price=800.0, median_30d=1000.0,
            cross_site_best=800.0, steal_score=90.0,
            savings_vs_median=20.0, savings_vs_cross_site=0.0,
            discovered_at=now))
        steals = asyncio.run(engine.get_steals(threshold=70))

        watched = asyncio.run(engine.track_url(
            url, "user1", target_price=850.0))
        asyncio.run(engine.track_url(
            "https://www.konga.com/p/1", "user1", target_price=100.0))
        triggered = asyncio.run(engine.check_watchlists())
        remaining = engine.list_watchlists("user1")

        hunter._store_deal(Deal(
            deal_id="deal_1", title="Phone X", vendor="Jumia Nigeria",
            current_price=800.0, original_price=1000.0,
            discount_percent=20.0, url=url, deal_score=90.0,
            discovered_at=now))
        hunter._store_deal(Deal(
            deal_id="deal_2", title="Rice 50kg", vendor="Konga",
            current_price=5000.0, original_price=6000.0,
            discount_percent=16.0, url="https://www.konga.com/p/2",
            deal_score=50.0, discovered_at=now))
        top = asyncio.run(hunter.get_top_deals(limit=10))

        alert = asyncio.run(hunter.track_product(url, 750.0, "user1"))
        history = asyncio.run(hunter.price_history(url))
        fired = asyncio.run(hunter.check_alerts())
        alerts = asyncio.run(hunter.get_user_alerts(
            "user1", active_only=False))
        return {
            "steal": (steal.current_price, steal.median_30d,
                      steal.cross_site_best,
                      round(steal.steal_score, 6)),
            "steals": [(s.product_url, s.steal_score) for s in steals],
            "watched_site": watched.site,
            "triggered": [(w.product_url, w.last_price, w.is_triggered)
                          for w in triggered],
            "remaining": [(r["product_url"], r["target_price"])
                          for r in remaining],
            "top": [(d.deal_id, d.deal_score) for d in top],
            "alert": (alert.current_price, alert.vendor),
            "history": (history.lowest_price, history.highest_price,
                        history.average_price, len(history.prices)),
            "fired": [(a.product_url, a.current_price, a.is_triggered)
                      for a in fired],
            "alerts": [(a.alert_id == alert.alert_id, a.is_triggered,
                        a.current_price) for a in alerts],
        }

    def test_conformance(self, rung, tmp_path, monkeypatch):
        engine, hunter = self._engines(
            tmp_path / "shopping.db", rung.store, monkeypatch)
        # one shared disposition: both engines seam-backed together
        # or both native — never one of each
        assert (engine._port is None) == (rung.store is None)
        assert (hunter._port is None) == (rung.store is None)
        observed = self._scenario(engine, hunter)
        assert observed["steal"] == (800.0, 1000.0, 800.0, 27.0)
        assert observed["steals"] == [
            ("https://www.jumia.com.ng/p/1", 90.0)]
        assert observed["watched_site"] == "jumia"
        assert observed["triggered"] == [
            ("https://www.jumia.com.ng/p/1", 800.0, True)]
        assert observed["remaining"] == [
            ("https://www.konga.com/p/1", 100.0)]
        assert observed["top"] == [("deal_1", 90.0), ("deal_2", 50.0)]
        assert observed["alert"] == (700.0, "jumia")
        assert observed["history"] == (700.0, 700.0, 700.0, 1)
        assert observed["fired"] == [
            ("https://www.jumia.com.ng/p/1", 700.0, True)]
        assert observed["alerts"] == [(True, True, 700.0)]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path,
                                        monkeypatch):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import asyncio
        import time

        from nomorals.integrations.naija_deals import Deal
        from nomorals.integrations.naija_shopping import PriceSnapshot

        path = tmp_path / "shopping.db"
        engine, hunter = self._engines(path, None, monkeypatch)
        now = time.time()
        url = "https://www.jumia.com.ng/p/1"
        engine._store_snapshot(PriceSnapshot(
            product_url=url, site="jumia", price_ngn=1000.0,
            title="Phone X", timestamp=now - 300))
        engine._store_snapshot(PriceSnapshot(
            product_url=url, site="jumia", price_ngn=800.0,
            title="Phone X", timestamp=now - 200))
        watched = asyncio.run(engine.track_url(
            url, "user1", target_price=850.0))
        hunter._store_deal(Deal(
            deal_id="deal_1", title="Phone X", vendor="Jumia Nigeria",
            current_price=800.0, original_price=1000.0,
            discount_percent=20.0, url=url, deal_score=90.0,
            discovered_at=now))
        alert = asyncio.run(hunter.track_product(url, 750.0, "user1"))

        # seam construction imports all six tables' history once —
        # the engine constructs first, the hunter's tables come too
        engine2, _hunter2 = self._engines(path, rung.store, monkeypatch)
        store = rung.store
        assert store.count("shopping-deals__naija_price_snapshots") == 2
        assert store.count("shopping-deals__watchlists") == 1
        assert store.count("shopping-deals__steals") == 0
        assert store.count("shopping-deals__deals") == 1
        assert store.count("shopping-deals__price_alerts") == 1
        assert store.count("shopping-deals__price_history") == 1
        docs = store.find("shopping-deals__naija_price_snapshots")
        assert sorted(d["price_ngn"] for d in docs) == [800.0, 1000.0]
        assert all(d["title"] == "Phone X" for d in docs)
        doc = store.get("shopping-deals__watchlists",
                        watched.watchlist_id)
        assert doc is not None and doc["target_price"] == 850.0
        doc = store.get("shopping-deals__price_alerts", alert.alert_id)
        assert doc is not None and doc["current_price"] == 700.0
        # a second seam construction adopts — never re-imports
        self._engines(path, rung.store, monkeypatch)
        assert store.count("shopping-deals__naija_price_snapshots") == 2
        assert store.count("shopping-deals__price_alerts") == 1
        # and the seam path keeps serving: a new snapshot lands there
        engine2._store_snapshot(PriceSnapshot(
            product_url=url, site="jumia", price_ngn=750.0,
            title="Phone X", timestamp=now - 50))
        assert store.count("shopping-deals__naija_price_snapshots") == 3
        steal = asyncio.run(engine2.compute_steal_score(url))
        assert steal is not None and steal.current_price == 750.0

    def test_legacy_adoption_carries_into_the_seam(
            self, rung, tmp_path, monkeypatch):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.storage.db import Database

        path = tmp_path / "shopping.db"
        db = Database(str(path))
        db.execute(
            """CREATE TABLE price_snapshots (
                 product_url TEXT NOT NULL, site TEXT NOT NULL,
                 price_ngn REAL NOT NULL, title TEXT NOT NULL,
                 timestamp REAL NOT NULL,
                 in_stock INTEGER NOT NULL DEFAULT 1,
                 original_price REAL NOT NULL DEFAULT 0,
                 discount_badge REAL NOT NULL DEFAULT 0,
                 PRIMARY KEY (product_url, timestamp))""")
        db.execute(
            "INSERT INTO price_snapshots VALUES "
            "('https://x.example/1', 'jumia', 100.0, 't', 5.0, 1, 0, 0)")
        db.close()

        self._engines(path, rung.store, monkeypatch)
        # the rename happened in the file (the rule, verbatim) …
        check = Database(str(path))
        try:
            assert check.table_exists("naija_price_snapshots")
            assert not check.table_exists("price_snapshots")
        finally:
            check.close()
        # … and the adopted history is what the seam imported
        assert rung.store.count(
            "shopping-deals__naija_price_snapshots") == 1
        docs = rung.store.find("shopping-deals__naija_price_snapshots")
        assert docs[0]["price_ngn"] == 100.0
        assert docs[0]["site"] == "jumia"

    def test_migrations_shaped_table_left_untouched(
            self, rung, tmp_path, monkeypatch):
        from nomorals.storage.db import Database

        path = tmp_path / "shopping.db"
        db = Database(str(path))
        db.migrate()
        assert db.table_exists("price_snapshots")
        db.close()

        self._engines(path, rung.store, monkeypatch)
        check = Database(str(path))
        try:
            cols = {r["name"]
                    for r in check.table_info("price_snapshots")}
            assert "listing_id" in cols and "product_url" not in cols
            assert check.table_exists("naija_price_snapshots")
        finally:
            check.close()
        if rung.store is not None:
            # the migrations shape is not this engine's data: the
            # import found only the engine's own (empty) table
            assert rung.store.count(
                "shopping-deals__naija_price_snapshots") == 0


# ---------------------------------------------------------------------------
# residue Batch 2, slice 1: workspace-inbox, health-nutrition,
# commerce-medusa
# ---------------------------------------------------------------------------


class TestWorkspaceInboxPort:
    """workspace/inbox.py — the workspace inbox port (residue
    Batch 2, slice 1): a Database-class organ, so the store
    attaches to the Inbox's own Database handle and its three
    tables route via attached_rows (collections carry the table
    names). The pending→processing claim is the seam's
    compare-and-swap."""

    def _scenario(self, inbox) -> dict:
        drop = inbox.dir / "digest-me.txt"
        drop.write_text("alpha beta\ngamma\n")
        item = inbox.intake(drop)
        swept = inbox.sweep()
        got = inbox.get_item(item.id)
        rule = inbox.add_rule("pdfs", {"ext": [".pdf"]}, "summarize",
                              priority=3)
        inbox.set_rule_enabled(rule.id, False)
        inbox.set_priority(item.id, 2)
        inbox.annotate(item.id, "owner note")
        again = inbox.get_item(item.id)
        return {
            "swept_counts": swept["counts"],
            "status": got.status,
            "summary": got.result_summary,
            "intent": got.intent,
            "attempts": got.attempts,
            "priority": again.priority,
            "note_has": "owner note" in again.owner_note,
            "found": [i.name for i in inbox.search("digest-me")],
            "rules": [(r.name, r.intent, r.enabled)
                      for r in inbox.list_rules()],
            "history": [(h["intent"], h["outcome"])
                        for h in inbox.history(item.id)],
            "digest_mentions": "digest-me.txt" in inbox.digest(hours=1.0),
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.workspace.inbox import Inbox

        inbox = Inbox(tmp_path, store=rung.store)
        assert self._scenario(inbox) == {
            "swept_counts": {"done": 1},
            "status": "done",
            "summary": "3 lines, 17 bytes: alpha beta gamma",
            "intent": "summarize",
            "attempts": 1,
            "priority": 2,
            "note_has": True,
            "found": ["digest-me.txt"],
            "rules": [("pdfs", "summarize", False)],
            "history": [("summarize", "processed")],
            "digest_mentions": True,
        }

    def test_lazy_import_from_native_file(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.workspace.inbox import Inbox

        native = Inbox(tmp_path)
        drop = tmp_path / "kept.txt"
        drop.write_text("kept body\n")
        item = native.intake(drop)
        native.sweep()
        native.add_rule("pdfs", {"ext": [".pdf"]}, "summarize")

        seam = Inbox(tmp_path, store=rung.store)
        got = seam.get_item(item.id)
        assert got.status == "done"
        assert got.result_summary.startswith("2 lines")
        assert len(seam.history(item.id)) == 1
        assert [r.name for r in seam.list_rules()] == ["pdfs"]
        assert rung.store.count("inbox_items") == 1
        assert rung.store.count("inbox_history") == 1
        assert rung.store.count("inbox_rules") == 1
        marker = rung.store.get("workspace-inbox__meta", "import")
        assert marker is not None and marker["done"] is True
        # importing never overwrites: a second construction adopts
        again = Inbox(tmp_path, store=rung.store)
        assert again.get_item(item.id).status == "done"
        assert rung.store.count("inbox_items") == 1
        # the duplicate-content guard answers from the seam too:
        # a twin drop with identical bytes resolves as a dropped
        # duplicate — a new 'done' record naming the original,
        # exactly the native contract — and the copy is discarded
        twin = tmp_path / "kept-twin.txt"
        twin.write_text("kept body\n")
        dup = seam.intake(twin)
        assert dup.id != item.id
        assert dup.status == "done"
        assert dup.result_summary.startswith("duplicate of kept.txt (")
        assert item.id in dup.result_summary
        assert not (tmp_path / "inbox" / "kept-twin.txt").exists()
        assert rung.store.count("inbox_items") == 2

    def test_claim_is_a_compare_and_swap(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("the native claim is pinned by test_inbox")
        from nomorals.workspace.inbox import Inbox

        seam = Inbox(tmp_path, store=rung.store)
        drop = tmp_path / "race.txt"
        drop.write_text("x\n")
        item = seam.intake(drop)
        assert seam._claim(item.id) is True
        assert seam._claim(item.id) is False
        other = Inbox(tmp_path, store=rung.store)
        assert other._claim(item.id) is False
        claimed = other.get_item(item.id)
        assert claimed.status == "processing" and claimed.attempts == 1


class TestNutritionStorePort:
    """health/nutrition.py — NutritionStore port (residue Batch 2,
    slice 1): two independent logs behind the unchanged
    never-raises API; the AUTOINCREMENT water ids come from the
    port's atomic counter, seeded from the imported maximum."""

    def _scenario(self, store_) -> dict:
        from nomorals.health.nutrition import MealItem, MealLog

        item = MealItem(name="jollof rice", cal_low=450, cal_high=700,
                        protein_g=(12, 18), carbs_g=(70, 95),
                        fat_g=(15, 25))
        log = MealLog(items=[item], answers={}, cal_low=450,
                      cal_high=700)
        meal_id = store_.save_meal(log, day="2026-11-10")
        store_.log_water(250, day="2026-11-10")
        total = store_.log_water(250, day="2026-11-10")
        meals = store_.meals_on("2026-11-10")
        return {
            "saved": bool(meal_id),
            "meals": len(meals),
            "meal_items": [m["items"][0]["name"] for m in meals],
            "cal": (meals[0]["cal_low"], meals[0]["cal_high"]),
            "water_total": total,
            "water_on": store_.water_on("2026-11-10"),
            "recent": len(store_.recent_meals(5)),
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.health.nutrition import NutritionStore

        store_ = NutritionStore(str(tmp_path / "nutrition.db"),
                                store=rung.store)
        assert self._scenario(store_) == {
            "saved": True,
            "meals": 1,
            "meal_items": ["jollof rice"],
            "cal": (450, 700),
            "water_total": 500,
            "water_on": 500,
            "recent": 1,
        }

    def test_lazy_import_and_water_counter_continuity(self, rung,
                                                      tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.nutrition import NutritionStore

        path = str(tmp_path / "nutrition.db")
        native = NutritionStore(path)
        assert native.log_water(200, day="2026-11-09") == 200
        assert native.log_water(300, day="2026-11-09") == 500

        seam = NutritionStore(path, store=rung.store)
        assert seam.water_on("2026-11-09") == 500
        assert seam.log_water(100, day="2026-11-09") == 600
        docs = rung.store.find("health-nutrition__water")
        assert sorted(d["_id"] for d in docs) == ["1", "2", "3"]
        # a second construction adopts — never re-imports, and the
        # counter continues past the imported maximum
        again = NutritionStore(path, store=rung.store)
        assert again.log_water(50, day="2026-11-09") == 650
        docs = rung.store.find("health-nutrition__water")
        assert sorted(d["_id"] for d in docs) == ["1", "2", "3", "4"]

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.nutrition import NutritionStore

        store_ = NutritionStore(str(tmp_path / "nutrition.db"),
                                store=rung.store)
        assert self._scenario(store_)["saved"] is True
        assert rung.store.count("health-nutrition__meals") == 1
        assert rung.store.count("health-nutrition__water") == 2
        meals = rung.store.find("health-nutrition__meals")
        items = meals[0]["items_json"]
        if isinstance(items, str):  # the sqlite-doc rung stores JSON text
            import json

            items = json.loads(items)
        assert items[0]["name"] == "jollof rice"


class TestMedusaStorePort:
    """commerce/medusa.py — StoreManager port (residue Batch 2,
    slice 1): the registry mirrors an external system's own keys,
    so document ids carry the store ids verbatim."""

    def _scenario(self, manager) -> dict:
        from nomorals.commerce.medusa import Store

        manager._save(Store(
            id="store_0123456789abcdef", name="Suya Place",
            status="ready", engine="medusa", api_url="http://api",
            admin_url="http://api/app",
            api_key_ref="medusa/store_0123456789abcdef/publishable",
            webhook_secret_ref="medusa/store_0123456789abcdef/secret",
            currency="NGN", created_at=1000.0))
        manager._save(Store(
            id="store_fedcba9876543210", name="Beta Shop",
            status="failed", engine="medusa", api_url="",
            admin_url="", api_key_ref="", webhook_secret_ref="",
            currency="USD", created_at=2000.0))
        got = manager.get("store_0123456789abcdef")
        return {
            "got_name": got.name if got else None,
            "got_note": got.webhook_note if got else None,
            "all": [s.id for s in manager.list()],
            "ready": [s.id for s in manager.list(status="ready")],
            "missing": manager.get("store_nope") is None,
            "no_subscriber": manager.get_webhook_subscriber(
                "store_0123456789abcdef") == "",
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.commerce.medusa import StoreManager

        manager = StoreManager(str(tmp_path / "stores.db"),
                               store=rung.store, now=lambda: 0.0)
        assert self._scenario(manager) == {
            "got_name": "Suya Place",
            "got_note": "",
            "all": ["store_0123456789abcdef", "store_fedcba9876543210"],
            "ready": ["store_0123456789abcdef"],
            "missing": True,
            "no_subscriber": True,
        }

    def test_lazy_import_carries_external_ids_verbatim(self, rung,
                                                       tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.commerce.medusa import Store, StoreManager

        path = str(tmp_path / "stores.db")
        native = StoreManager(path, now=lambda: 0.0)
        native._save(Store(
            id="store_0123456789abcdef", name="Suya Place",
            status="ready", engine="medusa", api_url="http://api",
            admin_url="http://api/app",
            api_key_ref="medusa/store_0123456789abcdef/publishable",
            webhook_secret_ref="medusa/store_0123456789abcdef/secret",
            currency="NGN", created_at=1000.0))
        conn = native._db
        with conn:
            conn.execute(
                "INSERT INTO webhook_subscribers (store_id, code,"
                " event, created_at) VALUES (?, ?, ?, ?)",
                ("store_0123456789abcdef", "whsec_code",
                 "cart.updated", 1500.0))

        seam = StoreManager(path, store=rung.store, now=lambda: 0.0)
        got = seam.get("store_0123456789abcdef")
        assert got is not None and got.name == "Suya Place"
        assert seam.get_webhook_subscriber(
            "store_0123456789abcdef") == "whsec_code"
        assert rung.store.count("commerce-medusa__stores") == 1
        assert rung.store.count(
            "commerce-medusa__webhook_subscribers") == 1
        doc = rung.store.get("commerce-medusa__stores",
                             "store_0123456789abcdef")
        assert doc is not None and doc["_id"] == "store_0123456789abcdef"
        again = StoreManager(path, store=rung.store, now=lambda: 0.0)
        assert again.get("store_0123456789abcdef").name == "Suya Place"
        assert rung.store.count("commerce-medusa__stores") == 1


# ---------------------------------------------------------------------------
# residue Batch 2, slice 3: email-integration, travel-whitelabel,
# social-whatsapp-cost, planning-estimates
# ---------------------------------------------------------------------------

class TestEmailIntegrationPort:
    """Helpers (_integration + the fake Gmail transport) live on the
    class, adjacent to the cases that use them."""

    @staticmethod
    def _integration(store=None, hid=None):
        from unittest.mock import MagicMock

        import nomorals.integrations.email_integration as email_mod

        integ = email_mod.EmailIntegration(MagicMock(), MagicMock(),
                                           store=store)
        integ._detect_backend = lambda account: "gmail_api"
        integ._gmail_token = lambda account: "tok"
        state = hid if hid is not None else ["100"]
        calls = []

        def fake_request(account, method, url, payload=None, params=None):
            calls.append((method, url, dict(params or {})))
            if url.endswith("/profile"):
                return {"historyId": state[0]}
            if "/history" in url:
                return {"history": []}
            if url.endswith("/watch"):
                return {"historyId": state[0],
                        "expiration": int((time.time() + 7 * 86400) * 1000)}
            return {}

        integ._gmail_request = fake_request
        return integ, state, calls

    def test_conformance(self, rung, tmp_path, monkeypatch):
        import asyncio

        monkeypatch.setenv("HOME", str(tmp_path))
        hid = ["100"]
        integ, _, calls = self._integration(store=rung.store, hid=hid)
        first = asyncio.run(integ.sync_new("a@gmail.com"))
        assert first == {"new": [], "cursor_seeded": True,
                         "history_id": "100"}
        hid[0] = "150"
        second = asyncio.run(integ.sync_new("a@gmail.com"))
        assert second["cursor_seeded"] is False
        assert second["history_id"] == "150"
        history_calls = [c for c in calls if "/history" in c[1]]
        assert history_calls[-1][2].get("startHistoryId") == "100"
        # watches ride the same store
        asyncio.run(integ.start_watch("projects/p/topic", "a@gmail.com"))
        status = integ.watch_status("a@gmail.com")
        assert status is not None
        assert status["topic"] == "projects/p/topic"
        assert status["history_id"] == "150"
        asyncio.run(integ.stop_watch("a@gmail.com"))
        assert integ.watch_status("a@gmail.com") is None

    def test_cursor_continuity_across_the_move(self, rung, tmp_path,
                                               monkeypatch):
        """The registry's continuity pin (the A17 case applied to
        email): a historyId advanced on the native path is the
        cursor the seam resumes from — a reset would mean a full
        re-sync storm against Gmail's token stream."""
        if rung.store is None:
            pytest.skip("native rung has no move to prove")
        import asyncio

        monkeypatch.setenv("HOME", str(tmp_path))
        hid = ["100"]
        # phase 1: native construction seeds, then advances the
        # file's cursor to 150
        native, _, _ = self._integration(hid=hid)
        first = asyncio.run(native.sync_new("a@gmail.com"))
        assert first["cursor_seeded"] is True
        hid[0] = "150"
        second = asyncio.run(native.sync_new("a@gmail.com"))
        assert second["history_id"] == "150"
        # phase 2: seam construction imports the file's cursor and
        # resumes from it, then advances it in its own collection
        seam, _, seam_calls = self._integration(store=rung.store, hid=hid)
        hid[0] = "200"
        third = asyncio.run(seam.sync_new("a@gmail.com"))
        assert third["cursor_seeded"] is False
        assert third["history_id"] == "200"
        seam_history = [c for c in seam_calls if "/history" in c[1]]
        assert seam_history[-1][2].get("startHistoryId") == "150"
        assert rung.store.count("email-integration__cursors") == 1
        doc = rung.store.get("email-integration__cursors", "a@gmail.com")
        assert doc is not None and doc["history_id"] == "200"
        # phase 3: a second seam construction adopts the advanced
        # cursor — never re-imports, never resets
        again, _, again_calls = self._integration(store=rung.store,
                                                  hid=hid)
        hid[0] = "260"
        fourth = asyncio.run(again.sync_new("a@gmail.com"))
        assert fourth["history_id"] == "260"
        again_history = [c for c in again_calls if "/history" in c[1]]
        assert again_history[-1][2].get("startHistoryId") == "200"
        assert rung.store.count("email-integration__cursors") == 1

    def test_seam_shape_and_lazy_import(self, rung, tmp_path,
                                        monkeypatch):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import asyncio

        monkeypatch.setenv("HOME", str(tmp_path))
        hid = ["100"]
        native, _, _ = self._integration(hid=hid)
        asyncio.run(native.sync_new("a@gmail.com"))
        asyncio.run(native.start_watch("projects/p/topic", "a@gmail.com"))
        seam, _, _ = self._integration(store=rung.store, hid=hid)
        status = seam.watch_status("a@gmail.com")
        assert status is not None
        assert status["topic"] == "projects/p/topic"
        assert rung.store.count("email-integration__cursors") == 1
        assert rung.store.count("email-integration__watches") == 1
        doc = rung.store.get("email-integration__watches", "a@gmail.com")
        assert doc is not None and doc["history_id"] == "100"
        again, _, _ = self._integration(store=rung.store, hid=hid)
        assert again.watch_status("a@gmail.com") is not None
        assert rung.store.count("email-integration__cursors") == 1
        assert rung.store.count("email-integration__watches") == 1


class TestTravelWhitelabelPort:
    @staticmethod
    def _scenario(store, path):
        from nomorals.travel.whitelabel import TravelClientStore

        s = TravelClientStore(str(path), store=store)
        a = s.create("Alpha Travel", whatsapp_number="+234800000000",
                     branding={"tone": "warm"},
                     channels=["whatsapp", "instagram"],
                     languages=["en", "yo"])
        b = s.create("Beta Tours")
        assert a is not None and b is not None
        got = s.get(a.id)
        assert got is not None and got.name == "Alpha Travel"
        assert got.channels == ["whatsapp", "instagram"]
        assert got.languages == ["en", "yo"]
        assert got.branding == {"tone": "warm"}
        assert {c.id for c in s.list()} == {a.id, b.id}
        assert s.deactivate(b.id) is True
        assert s.deactivate("tcl_missing") is False
        assert {c.id for c in s.list()} == {a.id}
        assert {c.id for c in s.list(active_only=False)} == {a.id, b.id}
        s.log_event(a.id, "query", {"q": "flights"})
        s.log_event(a.id, "query", {})
        s.log_event(a.id, "booking", {})
        s.log_event(b.id, "query", {})
        assert s.event_counts(a.id, 0.0) == {"query": 2, "booking": 1}
        return s, a, b

    def test_conformance(self, rung, tmp_path):
        self._scenario(rung.store, tmp_path / "clients.db")

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        path = tmp_path / "clients.db"
        _native, a, _b = self._scenario(None, path)
        from nomorals.travel.whitelabel import (
            TravelClientStore,
            log_client_event,
        )

        seam = TravelClientStore(str(path), store=rung.store)
        assert rung.store.count("travel-whitelabel__travel_clients") == 2
        assert rung.store.count("travel-whitelabel__client_events") == 4
        assert {c.id for c in seam.list()} == {a.id}
        assert seam.event_counts(a.id, 0.0) == {"query": 2, "booking": 1}
        # id continuity: the seam's next event continues the file's
        # AUTOINCREMENT sequence instead of restarting it
        seam.log_event(a.id, "escalation", {})
        docs = rung.store.find("travel-whitelabel__client_events")
        assert len(docs) == 5
        assert max(int(d["id"]) for d in docs) == 5
        # the module-level writer reaches the same seam via the boot
        # ambient, exactly as production wiring resolves it
        set_ambient_document_store(rung.store)
        try:
            assert log_client_event(a.id, "query", {},
                                    db_path=str(path)) is True
        finally:
            set_ambient_document_store(None)
        assert rung.store.count("travel-whitelabel__client_events") == 6
        # a second seam construction adopts — no re-import, no dupes
        again = TravelClientStore(str(path), store=rung.store)
        assert rung.store.count("travel-whitelabel__travel_clients") == 2
        assert rung.store.count("travel-whitelabel__client_events") == 6
        assert again.event_counts(a.id, 0.0) == {
            "query": 3, "booking": 1, "escalation": 1}


class TestWhatsAppCostPort:
    NOW = 1_700_000_000.0

    def _scenario(self, store, path):
        from nomorals.social.whatsapp_cost import (
            RATE_MARKETING_KOBO,
            RATE_SERVICE_KOBO,
            CostTracker,
        )

        t = CostTracker(str(path), now=lambda: self.NOW, store=store)
        assert t.track("08030000000", "service", client="acme",
                       purpose="otp") == RATE_SERVICE_KOBO
        assert t.track("08030000000", "marketing",
                       client="acme") == RATE_MARKETING_KOBO
        assert t.track("08030000001", "service",
                       client="other") == RATE_SERVICE_KOBO
        total = RATE_SERVICE_KOBO + RATE_MARKETING_KOBO
        assert t.spent_since(self.NOW - 3600, client="acme") == total
        assert t.spent_since(self.NOW + 60, client="acme") == 0
        assert t.message_count(client="acme") == 2
        assert t.get_budget("acme") is None
        t.set_budget("acme", total - 1, hard=True)
        assert t.get_budget("acme") == {
            "client": "acme", "amount_kobo": total - 1, "hard": True}
        may, spent, amount = t.check_budget("acme")
        assert (may, spent, amount) == (False, total, total - 1)
        return t

    def test_conformance(self, rung, tmp_path):
        self._scenario(rung.store, tmp_path / "cost.db")

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.whatsapp_cost import CostTracker

        path = tmp_path / "cost.db"
        self._scenario(None, path)
        seam = CostTracker(str(path), now=lambda: self.NOW,
                           store=rung.store)
        assert rung.store.count("social-whatsapp-cost__wa_budgets") == 1
        assert rung.store.count("social-whatsapp-cost__wa_costs") == 3
        budget = rung.store.get("social-whatsapp-cost__wa_budgets",
                                "acme")
        assert budget is not None and budget["hard"] == 1
        # id continuity: the seam's next ledger row continues the
        # file's AUTOINCREMENT sequence
        seam.track("08030000002", "service", client="acme")
        docs = rung.store.find("social-whatsapp-cost__wa_costs")
        assert len(docs) == 4
        assert max(int(d["id"]) for d in docs) == 4
        again = CostTracker(str(path), now=lambda: self.NOW,
                            store=rung.store)
        assert rung.store.count("social-whatsapp-cost__wa_budgets") == 1
        assert rung.store.count("social-whatsapp-cost__wa_costs") == 4
        assert again.message_count(client="acme") == 3


class TestPlanningEstimatesPort:
    @staticmethod
    def _scenario(store, path):
        from nomorals.planning.estimates import EstimateStore

        es = EstimateStore(str(path), store=store)
        assert es.record_actual("email", 30.0, 45.0,
                                {"prep": 12.0}) is True
        assert es.record_actual("email", 30.0, 42.0,
                                {"prep": 11.0}) is True
        assert es.record_actual("email", 30.0, 20.0,
                                {"prep": 9.0}) is True
        assert es._stats("email")["n"] == 3
        seg = es._segment_row("email", "prep")
        assert seg is not None and seg["n"] == 3
        est = es.estimate("email", {"prep": 0.0})
        assert est.task_type == "email"
        rows = es.calibration_rows()
        assert len(rows) == 1
        assert rows[0]["task_type"] == "email" and rows[0]["n"] == 3
        return es

    def test_conformance(self, rung, tmp_path):
        self._scenario(rung.store, tmp_path / "estimates.db")

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.planning.estimates import EstimateStore

        path = tmp_path / "estimates.db"
        self._scenario(None, path)
        seam = EstimateStore(str(path), store=rung.store)
        assert rung.store.count(
            "planning-estimates__estimate_stats") == 1
        assert rung.store.count(
            "planning-estimates__estimate_segments") == 1
        doc = rung.store.get("planning-estimates__estimate_stats",
                             "email")
        assert doc is not None and doc["n"] == 3
        # the stats table ports AS-IS: the seam serves the file's
        # stored rollup, and learning continues on top of it
        assert seam._stats("email")["n"] == 3
        again = EstimateStore(str(path), store=rung.store)
        assert rung.store.count(
            "planning-estimates__estimate_stats") == 1
        assert again.record_actual("email", 30.0, 40.0,
                                   {"prep": 10.0}) is True
        doc = rung.store.get("planning-estimates__estimate_stats",
                             "email")
        assert doc is not None and doc["n"] == 4


class TestDigitalTwinPort:
    """integrations/digital_twin.py — HomeTwin port (residue Batch 2,
    slice 4): desired-state + entity-meta RowTables behind the
    unchanged never-raises API; the AUTOINCREMENT states history
    goes via put_raw with ids from the port's atomic counter,
    seeded from the imported maximum, and the aggregate reads
    re-express as Python rollups ordered by (ts, id)."""

    def _scenario(self, twin) -> dict:
        twin.register_entity("light.kitchen", device_class="light",
                             area="kitchen", friendly_name="Kitchen Light")
        # a second registration with an empty area preserves the first
        twin.register_entity("light.kitchen", unit="bool")
        recorded = [
            twin.ingest("light.kitchen", "on", ts=1000.0),
            twin.ingest("light.kitchen", "on", ts=1001.0),
            twin.ingest("light.kitchen", "off", ts=1002.0),
            twin.ingest("sensor.temp", "21.5", ts=1000.5),
        ]
        diverged = twin.set_desired("light.kitchen", "on")
        desired = {eid: (d["state"], d["set_by"])
                   for eid, d in twin.desired_state().items()}
        out = {
            "recorded": recorded,
            "diverged": diverged,
            "desired": desired,
            "current": twin.current_state(),
            "query": [(e.state, e.ts)
                      for e in twin.query("light.kitchen", 0.0)],
            "ids": twin.entity_ids(),
            "meta": twin.entity_meta("light.kitchen"),
            "history_days": round(twin.history_days(), 8),
        }
        twin.clear_desired("light.kitchen")
        out["desired_after_clear"] = twin.desired_state()
        return out

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.integrations.digital_twin import HomeTwin

        twin = HomeTwin(str(tmp_path / "twin.db"), store=rung.store)
        try:
            assert self._scenario(twin) == {
                "recorded": [True, False, True, True],
                "diverged": True,
                "desired": {"light.kitchen": ("on", "devon")},
                "current": {"light.kitchen": "off",
                            "sensor.temp": "21.5"},
                "query": [("on", 1000.0), ("off", 1002.0)],
                "ids": ["light.kitchen", "sensor.temp"],
                "meta": {"device_class": "light", "unit": "bool",
                         "area": "kitchen",
                         "friendly_name": "Kitchen Light"},
                "history_days": round(2.0 / 86400.0, 8),
                "desired_after_clear": {},
            }
        finally:
            twin.close()

    def test_lazy_import_and_state_counter_continuity(self, rung,
                                                      tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.integrations.digital_twin import HomeTwin

        path = str(tmp_path / "twin.db")
        native = HomeTwin(path)
        native.ingest("light.kitchen", "on", ts=1000.0)
        native.ingest("light.kitchen", "off", ts=1002.0)
        native.register_entity("light.kitchen", area="kitchen")
        native.set_desired("light.kitchen", "on")
        native.close()

        seam = HomeTwin(path, store=rung.store)
        try:
            assert seam.current_state() == {"light.kitchen": "off"}
            assert seam.entity_meta("light.kitchen")["area"] == "kitchen"
            assert seam.ingest("light.kitchen", "on", ts=1003.0) is True
            docs = rung.store.find("digital-twin__states")
            assert sorted(d["_id"] for d in docs) == ["1", "2", "3"]
            assert rung.store.count("digital-twin__desired") == 1
            assert rung.store.count("digital-twin__entity_meta") == 1
        finally:
            seam.close()
        # a second construction adopts — never re-imports
        again = HomeTwin(path, store=rung.store)
        try:
            assert again.current_state() == {"light.kitchen": "on"}
            assert rung.store.count("digital-twin__states") == 3
        finally:
            again.close()

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.integrations.digital_twin import HomeTwin

        twin = HomeTwin(str(tmp_path / "twin.db"), store=rung.store)
        try:
            self._scenario(twin)
        finally:
            twin.close()
        assert rung.store.count("digital-twin__states") == 3
        # ingest auto-registers metadata for sensor.temp too
        assert rung.store.count("digital-twin__entity_meta") == 2
        # the desired row was cleared by the scenario
        assert rung.store.count("digital-twin__desired") == 0
        doc = rung.store.get("digital-twin__entity_meta", "light.kitchen")
        assert doc is not None and doc["_id"] == "light.kitchen"


class TestTrainingCoachPort:
    """health/training.py — TrainingCoach port (residue Batch 2,
    slice 4): plans + workout_log RowTables behind the unchanged
    API; the AUTOINCREMENT set_feedback children of log rows go
    via put_raw with ids from the port's atomic counter, seeded
    from the imported maximum."""

    def _scenario(self, coach) -> dict:
        from nomorals.health.training import Exercise, Workout

        plan = coach.new_plan("strength", "full", 4)
        push = Workout(
            date="2026-11-09", kind="push", plan_id=plan.id,
            exercises=[Exercise(key="bench", name="Bench Press", sets=4,
                                reps="8", rest_secs=90, intensity="hard",
                                load="60")])
        legs = Workout(
            date="2026-11-10", kind="legs", plan_id=plan.id,
            exercises=[Exercise(key="squat", name="Squat", sets=5,
                                reps="5", rest_secs=120, intensity="hard",
                                load="80")])
        logged = [
            coach.log_workout(push, completed=True, rpe=7,
                              loads={"bench": "60"},
                              set_rpes={"bench": 7}),
            coach.log_workout(legs, completed=True, rpe=8,
                              loads={"squat": "80"},
                              set_rpes={"squat": 8}),
        ]
        got = coach.get_plan(plan.id)
        latest = coach.latest_plan()
        return {
            "logged": logged,
            "plan_goal": got.goal if got else None,
            "plan_weeks": got.weeks if got else None,
            "latest_is_plan": latest is not None
            and latest.id == plan.id,
            "history": [(h["date"], h["kind"], h["completed"])
                        for h in coach.history(3650)],
            "prs": [(p.exercise_key, p.date)
                    for p in coach.personal_records()],
            "advice_known": not coach.progression_advice(
                "bench").startswith("no RPE history"),
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.health.training import TrainingCoach

        coach = TrainingCoach(str(tmp_path / "training.db"),
                              store=rung.store)
        assert self._scenario(coach) == {
            "logged": [True, True],
            "plan_goal": "strength",
            "plan_weeks": 4,
            "latest_is_plan": True,
            "history": [("2026-11-10", "legs", 1),
                        ("2026-11-09", "push", 1)],
            "prs": [("squat", "2026-11-10"), ("bench", "2026-11-09")],
            "advice_known": True,
        }

    def test_lazy_import_and_feedback_counter_continuity(self, rung,
                                                         tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.training import Exercise, TrainingCoach, Workout

        path = str(tmp_path / "training.db")
        native = TrainingCoach(path)
        w = Workout(
            date="2026-11-09", kind="push",
            exercises=[Exercise(key="bench", name="Bench Press", sets=4,
                                reps="8", rest_secs=90, intensity="hard",
                                load="60")])
        assert native.log_workout(w, completed=True, rpe=7,
                                  loads={"bench": "60"},
                                  set_rpes={"bench": 7}) is True

        seam = TrainingCoach(path, store=rung.store)
        assert [(h["date"], h["kind"]) for h in seam.history(3650)] == [
            ("2026-11-09", "push")]
        assert seam.log_workout(w, completed=True, rpe=7,
                                loads={"bench": "60"},
                                set_rpes={"bench": 6}) is True
        docs = rung.store.find("health-training__set_feedback")
        assert sorted(d["_id"] for d in docs) == ["1", "2"]
        # a second construction adopts — never re-imports, and the
        # counter continues past the imported maximum
        again = TrainingCoach(path, store=rung.store)
        assert again.log_workout(w, completed=True, rpe=7,
                                 loads={"bench": "60"},
                                 set_rpes={"bench": 6}) is True
        docs = rung.store.find("health-training__set_feedback")
        assert sorted(d["_id"] for d in docs) == ["1", "2", "3"]

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.training import TrainingCoach

        coach = TrainingCoach(str(tmp_path / "training.db"),
                              store=rung.store)
        assert self._scenario(coach)["logged"] == [True, True]
        assert rung.store.count("health-training__plans") == 1
        assert rung.store.count("health-training__workout_log") == 2
        assert rung.store.count("health-training__set_feedback") == 2
        plan_doc = rung.store.find("health-training__plans")[0]
        assert plan_doc["_id"].startswith("plan_")


class TestLegalPortfolioPort:
    """legal/portfolio.py — ContractPortfolio port (residue Batch 2,
    slice 4): contracts + obligations RowTables, and sla_terms
    keeping its composite (contract_id, metric) primary key —
    the seam mapping joins composite keys natively; the breach
    update is a find + merged put. The seam DDL carries the final
    shape the native path reaches via _migrate's ALTERs."""

    _TEXT = (
        "TENANCY AGREEMENT. This agreement commences on 1 January "
        "2027 and the term is for a period of 12 months. The annual "
        "rent is ₦500,000 payable yearly. The tenant shall give "
        "notice in writing at least by 1 November 2027 to "
        "terminate. A service charge of ₦50,000 applies.")

    def _scenario(self, pf) -> dict:
        result = pf.add_raw(self._TEXT, name="Tenancy")
        contract, _review = result
        cid = contract.id
        extracted = pf.obligations(cid)
        pending = pf.unconfirmed_obligations()
        confirmed_one = pf.confirm_obligation(pending[0].id) if pending \
            else False
        oid = pf.track_obligation(cid, "Pay service charge",
                                  1812000000.0)
        sla_ok = pf.track_sla(cid, "uptime", "99.9%")
        breach_ok = pf.report_sla_breach(cid, "uptime", "missed in May")
        breach_missing = pf.report_sla_breach(cid, "latency", "n/a")
        return {
            "name": contract.name,
            "annual_value": contract.annual_value,
            "value": pf.portfolio_value(),
            "extracted_kinds": sorted(o.kind for o in extracted),
            "pending_before": len(pending),
            "confirmed_one": confirmed_one,
            "pending_after": len(pf.unconfirmed_obligations()),
            "tracked": oid is not None,
            "done_ok": pf.mark_done(oid) if oid else False,
            "done_missing": pf.mark_done("obl_nope"),
            "open_after_done": len(pf.obligations(cid)),
            "sla_ok": sla_ok,
            "breach_ok": breach_ok,
            "breach_missing": breach_missing,
            "sla": [(s.metric, s.breached) for s in pf.sla_status(cid)],
            "search": [c.id == cid for c, _s in
                       pf.search_contracts("service charge")],
            "contracts": [c.name for c in pf.list_contracts()],
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.legal.portfolio import ContractPortfolio

        pf = ContractPortfolio(str(tmp_path / "portfolio.db"),
                               store=rung.store)
        assert self._scenario(pf) == {
            "name": "Tenancy",
            "annual_value": 500000.0,
            "value": 500000.0,
            "extracted_kinds": ["commencement", "expiry", "notice"],
            "pending_before": 3,
            "confirmed_one": True,
            "pending_after": 2,
            "tracked": True,
            "done_ok": True,
            "done_missing": False,
            "open_after_done": 3,
            "sla_ok": True,
            "breach_ok": True,
            "breach_missing": False,
            "sla": [("uptime", True)],
            "search": [True],
            "contracts": ["Tenancy"],
        }

    def test_lazy_import_adopts_rows(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.legal.portfolio import ContractPortfolio

        path = str(tmp_path / "portfolio.db")
        native = ContractPortfolio(path)
        result = native.add_raw(self._TEXT, name="Tenancy")
        assert result is not None
        cid = result[0].id
        assert native.track_sla(cid, "uptime", "99.9%") is True

        seam = ContractPortfolio(path, store=rung.store)
        got = seam.get_contract(cid)
        assert got is not None and got.name == "Tenancy"
        assert seam.portfolio_value() == 500000.0
        assert [(s.metric, s.breached) for s in seam.sla_status(cid)] \
            == [("uptime", False)]
        assert rung.store.count("legal-portfolio__contracts") == 1
        assert rung.store.count("legal-portfolio__sla_terms") == 1
        again = ContractPortfolio(path, store=rung.store)
        assert again.get_contract(cid).name == "Tenancy"
        assert rung.store.count("legal-portfolio__contracts") == 1

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.legal.portfolio import ContractPortfolio

        pf = ContractPortfolio(str(tmp_path / "portfolio.db"),
                               store=rung.store)
        out = self._scenario(pf)
        assert out["name"] == "Tenancy"
        assert rung.store.count("legal-portfolio__contracts") == 1
        assert rung.store.count("legal-portfolio__obligations") == 4
        assert rung.store.count("legal-portfolio__sla_terms") == 1
        doc = rung.store.find("legal-portfolio__sla_terms")[0]
        # the composite (contract_id, metric) key, joined natively
        assert doc["_id"].endswith("uptime")
        assert doc["breached"] == 1


class TestRegulatoryWatchPort:
    """legal/regulatory.py — RegulatoryWatch port (residue Batch 2,
    slice 4): watches + items RowTables behind the unchanged
    never-raises API; the seen-state dedup registry is get-by-key
    with insert-ignore on first sight and digest updates on
    change. The degraded :memory: fallback stays native-only."""

    _T0 = 1800000000.0  # fixed clock for the effective-date window

    def _item(self, **kw):
        from nomorals.legal.regulatory import RegulatoryItem

        base = dict(id="note-1", regulator="CBN",
                    title="New payments circular",
                    summary="CBN updates payments rules.",
                    source_url="https://www.cbn.gov.ng",
                    published=100.0,
                    effective_date=self._T0 + 10 * 86400,
                    urgency="normal")
        base.update(kw)
        return RegulatoryItem(**base)

    def _scenario(self, rw) -> dict:
        w = rw.watch("CBN", topics=["payments"])
        alias = rw.watch("ndpc")
        first = rw.note_item(self._item())
        again = rw.note_item(self._item())
        changed_item = self._item(title="New payments circular v2")
        changed = rw.note_item(changed_item)

        def fetcher(_reg):
            return [{"id": "chk-1", "regulator": "CBN",
                     "title": "Fresh payments rule",
                     "summary": "A new payments rule lands.",
                     "source_url": "https://www.cbn.gov.ng",
                     "published": 50.0}]

        seen_first = [i.id for i in rw.check(fetcher=fetcher)]
        seen_second = [i.id for i in rw.check(fetcher=fetcher)]
        dismissed = rw.dismiss("chk-1")
        seen_third = [i.id for i in rw.check(fetcher=fetcher)]
        out = {
            "watch": (w.regulator, w.topics) if w else None,
            "alias": alias.regulator if alias else None,
            "bad_regulator": rw.watch("NOPE") is None,
            "note_first": first,
            "note_again": again,
            "note_changed": changed,
            "changed_flag": changed_item.updated,
            "seen_first": seen_first,
            "seen_second": seen_second,
            "dismissed": dismissed,
            "seen_third": seen_third,
            "watches": sorted((x.regulator, tuple(x.topics))
                              for x in rw.list_watches()),
            "upcoming": [i.id for i in rw.upcoming_effective(
                days=90, now=self._T0)],
            "upcoming_far": [i.id for i in rw.upcoming_effective(
                days=5, now=self._T0)],
            "unwatch": rw.unwatch(alias.id) if alias else False,
            "unwatch_again": rw.unwatch("reg_nope"),
        }
        return out

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.legal.regulatory import RegulatoryWatch

        rw = RegulatoryWatch(str(tmp_path / "regulatory.db"),
                             store=rung.store)
        assert self._scenario(rw) == {
            "watch": ("CBN", ["payments"]),
            "alias": "NDPA",
            "bad_regulator": True,
            "note_first": True,
            "note_again": False,
            "note_changed": True,
            "changed_flag": True,
            "seen_first": ["chk-1"],
            "seen_second": [],
            "dismissed": True,
            "seen_third": [],
            "watches": [("CBN", ("payments",)), ("NDPA", ())],
            "upcoming": ["note-1"],
            "upcoming_far": [],
            "unwatch": True,
            "unwatch_again": False,
        }

    def test_lazy_import_adopts_rows(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.legal.regulatory import RegulatoryWatch

        path = str(tmp_path / "regulatory.db")
        native = RegulatoryWatch(path)
        w = native.watch("SEC", topics=["crypto"])
        assert w is not None
        assert native.note_item(self._item(
            id="sec-1", regulator="SEC",
            title="SEC digital asset rules")) is True

        seam = RegulatoryWatch(path, store=rung.store)
        assert [(x.regulator, tuple(x.topics))
                for x in seam.list_watches()] == [("SEC", ("crypto",))]
        assert [i.id for i in seam.upcoming_effective(
            days=90, now=self._T0)] == ["sec-1"]
        assert rung.store.count("legal-regulatory__reg_watches") == 1
        assert rung.store.count("legal-regulatory__reg_seen") == 1
        assert rung.store.count("legal-regulatory__reg_items") == 1
        again = RegulatoryWatch(path, store=rung.store)
        assert len(again.list_watches()) == 1
        assert rung.store.count("legal-regulatory__reg_watches") == 1

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.legal.regulatory import RegulatoryWatch

        rw = RegulatoryWatch(str(tmp_path / "regulatory.db"),
                             store=rung.store)
        out = self._scenario(rw)
        assert out["note_first"] is True
        # the alias watch was unwatched by the scenario
        assert rung.store.count("legal-regulatory__reg_watches") == 1
        assert rung.store.count("legal-regulatory__reg_items") == 1
        seen = rung.store.get("legal-regulatory__reg_seen", "chk-1")
        assert seen is not None and seen["_id"] == "chk-1"
        assert seen["dismissed"] == 1



# ---------------------------------------------------------------------------
# residue Batch 1, A16 catch-up: travel-itinerary
# ---------------------------------------------------------------------------


class TestTravelItineraryPort:
    @staticmethod
    def _trip(tid, name, pnr, created):
        from nomorals.travel.itinerary import Flight, Trip

        return Trip(
            id=tid, name=name,
            flights=[Flight(airline="British Airways",
                            flight_number="BA075", origin="LOS",
                            destination="LHR",
                            departs="2026-12-01T22:45",
                            arrives="2026-12-02T05:30", pnr=pnr)],
            docs=["email.txt"], created_at=created)

    def _scenario(self, builder) -> dict:
        builder._save(self._trip("trip_a", "LOS → LHR", "ABC123", 100.0))
        builder._save(self._trip("trip_b", "LOS → LHR", "DEF456", 200.0))
        builder._save(self._trip("trip_c", "LOS → LHR", "GHI789", 300.0))
        # INSERT OR REPLACE: trip_a's payload moves, its stamp stays
        moved = self._trip("trip_a", "LOS → LHR", "ABC123", 100.0)
        moved.flights[0].seat = "14A"
        builder._save(moved)
        got = builder.get_trip("trip_a")
        return {
            "got": (got.name, got.flights[0].pnr, got.flights[0].seat,
                    got.docs, got.created_at),
            "listed": [t.id for t in builder.list_trips()],
            "missing": builder.get_trip("nope"),
            "deleted": builder.delete_trip("trip_b"),
            "deleted_again": builder.delete_trip("trip_b"),
            "after_delete": [t.id for t in builder.list_trips()],
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.travel.itinerary import ItineraryBuilder

        builder = ItineraryBuilder(db_path=str(tmp_path / "trips.db"),
                                   vault_dir=str(tmp_path / "vault"),
                                   store=rung.store)
        observed = self._scenario(builder)
        assert observed["got"] == ("LOS → LHR", "ABC123", "14A",
                                   ["email.txt"], 100.0)
        assert observed["listed"] == ["trip_c", "trip_b", "trip_a"]
        assert observed["missing"] is None
        assert observed["deleted"] is True
        assert observed["deleted_again"] is False
        assert observed["after_delete"] == ["trip_c", "trip_a"]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.travel.itinerary import ItineraryBuilder

        path = tmp_path / "trips.db"
        native = ItineraryBuilder(db_path=str(path),
                                  vault_dir=str(tmp_path / "vault"))
        native._save(self._trip("trip_kept", "LOS → LHR", "ABC123",
                                100.0))

        builder = ItineraryBuilder(db_path=str(path),
                                   vault_dir=str(tmp_path / "vault"),
                                   store=rung.store)
        got = builder.get_trip("trip_kept")
        assert got is not None and got.flights[0].pnr == "ABC123"
        assert rung.store.count("travel-itinerary__trips") == 1
        doc = rung.store.get("travel-itinerary__trips", "trip_kept")
        assert doc is not None and "BA075" in doc["data_json"]
        again = ItineraryBuilder(db_path=str(path),
                                 vault_dir=str(tmp_path / "vault"),
                                 store=rung.store)
        assert [t.id for t in again.list_trips()] == ["trip_kept"]
        assert rung.store.count("travel-itinerary__trips") == 1


# ---------------------------------------------------------------------------
# residue Batch 2, slice 6: matching-batches, os-timeline, planning-route
# ---------------------------------------------------------------------------

class TestMatchingBatchesPort:
    """matching/batches.py — DailyBatchStore port (residue Batch 2,
    slice 6): all four tables over the seam behind the unchanged
    never-raises API; shown/feedback are composite-keyed child state
    whose parts live in the joined document id — the feedback
    upserts are read-modify-writes on that id, and the reads that
    filter on key parts scan restored rows, never find on PK fields.
    Helpers live on the class, per the slice-2 placement lesson."""

    _NOW = 1_700_000_000.0  # 2023-11-14 (UTC)

    def _scenario(self, store, path) -> dict:
        import time

        from nomorals.matching.batches import Candidate, DailyBatchStore

        s = DailyBatchStore(str(path), store=store)
        added = [
            s.add_candidate(Candidate(
                candidate_id="c1", title="Gig One", tags=("afrobeats",),
                quality=0.9, attributes={"pay": 500})),
            s.add_candidate(Candidate(
                candidate_id="c2", title="Gig Two", tags=("jazz",),
                quality=0.8)),
            s.add_candidate(Candidate(
                candidate_id="c3", title="Gig Three",
                tags=("afrobeats", "live"), quality=0.7)),
            s.add_candidate(Candidate(
                candidate_id="c4", title="Gig Four", tags=("jazz",),
                quality=0.6)),
        ]
        pool = sorted(
            (c.candidate_id, c.title, c.quality, c.tags, c.attributes)
            for c in s.candidates())
        batch1 = [c.candidate_id for c in s.today(
            now=self._NOW, preferences={"afrobeats": 2.0}, n=2,
            explore=0.0, seed=7)]
        again = [c.candidate_id for c in s.today(
            now=self._NOW, n=2, explore=0.0)]
        stats1 = sorted(s.feedback_stats("gig", "owner").items())
        assert s.record_feedback("gig", "owner", "c1", True)
        assert s.record_feedback("gig", "owner", "c2", False)
        stats2 = sorted(s.feedback_stats("gig", "owner").items())
        # next day: yesterday's shown pair is excluded by the
        # 7-day no-repeat rule, so the batch must move on
        batch2 = [c.candidate_id for c in s.today(
            now=self._NOW + 86400.0, preferences={"afrobeats": 2.0},
            n=2, explore=0.0, seed=7)]
        history = s.batch_history()
        removed = s.remove_candidate("c2")
        removed_again = s.remove_candidate("c2")
        day1 = time.strftime("%Y-%m-%d", time.gmtime(self._NOW))
        day2 = time.strftime("%Y-%m-%d",
                             time.gmtime(self._NOW + 86400.0))
        return {
            "added": added,
            "pool": pool,
            "batch1": batch1,
            "again": again,
            "stats1": stats1,
            "stats2": stats2,
            "batch2": batch2,
            "history": history,
            "day1": day1,
            "day2": day2,
            "removed": removed,
            "removed_again": removed_again,
            "pool_after": sorted(c.candidate_id
                                 for c in s.candidates()),
            "stats_after_remove": sorted(
                s.feedback_stats("gig", "owner").items()),
        }

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "b.db")
        assert observed["added"] == [True, True, True, True]
        assert observed["pool"] == [
            ("c1", "Gig One", 0.9, ("afrobeats",), {"pay": 500}),
            ("c2", "Gig Two", 0.8, ("jazz",), {}),
            ("c3", "Gig Three", 0.7, ("afrobeats", "live"), {}),
            ("c4", "Gig Four", 0.6, ("jazz",), {}),
        ]
        assert observed["batch1"] == ["c1", "c3"]
        # the day's batch is stable: the stored one comes back
        assert observed["again"] == ["c1", "c3"]
        assert observed["stats1"] == [("c1", (0, 1)), ("c3", (0, 1))]
        assert observed["stats2"] == [
            ("c1", (1, 2)), ("c2", (0, 1)), ("c3", (0, 1))]
        assert observed["batch2"] == ["c2", "c4"]
        assert observed["history"] == [
            (observed["day2"], ["c2", "c4"]),
            (observed["day1"], ["c1", "c3"])]
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["pool_after"] == ["c1", "c3", "c4"]
        # remove_candidate swept c2's feedback row too
        assert observed["stats_after_remove"] == [
            ("c1", (1, 2)), ("c3", (0, 1)), ("c4", (0, 1))]

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.matching.batches import Candidate, DailyBatchStore

        path = tmp_path / "b.db"
        native = DailyBatchStore(str(path))
        native.add_candidate(Candidate(
            candidate_id="k1", title="Kept One", tags=("afrobeats",),
            quality=0.9, attributes={"pay": 500}))
        native.add_candidate(Candidate(
            candidate_id="k2", title="Kept Two", tags=("jazz",),
            quality=0.8))
        native.today(now=self._NOW, preferences={"afrobeats": 2.0},
                     n=1, explore=0.0, seed=7)
        native.record_feedback("gig", "owner", "k1", True)

        seam = DailyBatchStore(str(path), store=rung.store)
        assert rung.store.count("matching-batches__candidates") == 2
        assert rung.store.count("matching-batches__batches") == 1
        assert rung.store.count("matching-batches__shown") == 1
        assert rung.store.count("matching-batches__feedback") == 1
        # the composite key lives in the joined document id
        doc = rung.store.get("matching-batches__feedback",
                             "gig\x1fowner\x1fk1")
        assert doc is not None
        assert doc["impressions"] == 2 and doc["likes"] == 1
        # the imported batch is the day's batch on the seam too
        assert [c.candidate_id for c in seam.today(
            now=self._NOW, n=1, explore=0.0)] == ["k1"]
        assert seam.feedback_stats("gig", "owner") == {"k1": (1, 2)}
        # learning continues on the seam, on the imported rows
        assert seam.record_feedback("gig", "owner", "k1", False)
        assert seam.feedback_stats("gig", "owner") == {"k1": (1, 3)}
        again = DailyBatchStore(str(path), store=rung.store)
        assert rung.store.count("matching-batches__candidates") == 2
        assert again.feedback_stats("gig", "owner") == {"k1": (1, 3)}


class TestOsTimelinePort:
    """os/timeline.py — Timeline port (residue Batch 2, slice 6):
    the append-only event log as documents behind the unchanged
    API. THE SEARCH DISPOSITION is pinned here, not just stated:
    fts_available is True on the native rung (real FTS5) and False
    on the seam rungs — honestly, there is no FTS5 index there —
    while search() returns the same result SET on every rung,
    served on the seam by a document scan. A split disposition
    (seam log, FTS reading the fossil file) would fail the second
    half of the seam-shape test: an event recorded on the seam,
    never written to the file, must be findable. Helpers live on
    the class, per the slice-2 placement lesson."""

    @staticmethod
    def _event(event_id, topic, ts, data):
        return SimpleNamespace(event_id=event_id, topic=topic, ts=ts,
                               source="p5-test", data=data)

    def _corpus(self, tl) -> None:
        tl.record(self._event("e1", "mission.transition", 1000.0,
                              {"mission_id": "m1",
                               "note": "alpha started"}))
        tl.record(self._event("e2", "mission.verify", 1001.0,
                              {"mission_id": "m1", "verdict": "pass",
                               "details": "alpha checks"}))
        tl.record(self._event("e3", "session.created", 1002.0,
                              {"session_id": "s1", "frontend": "cli"}))
        tl.record(self._event("e4", "artifact.created", 1003.0,
                              {"artifact_id": "a1",
                               "uri": "artifact://a1",
                               "mission_id": "m1"}))
        tl.record(self._event("e5", "task.done", 1004.0,
                              {"note": "beta finished",
                               "session_id": "s1"}))

    @staticmethod
    def _ids(rows) -> list:
        return sorted(r["event_id"] for r in rows)

    def _scenario(self, tl) -> dict:
        self._corpus(tl)
        out = {
            "fts_available": tl.fts_available,
            "all_topics": [r["topic"] for r in tl.query(limit=100)],
            "by_mission": [r["topic"]
                           for r in tl.query(mission_id="m1",
                                             limit=100)],
            "glob": sorted(r["topic"]
                           for r in tl.query(topic="mission.*",
                                             limit=100)),
            "since": len(tl.query(since=1002.0, limit=100)),
            "search_alpha": self._ids(tl.search("alpha")),
            "search_phrase": self._ids(tl.search('"alpha started"')),
            "search_or": self._ids(tl.search("beta OR verdict")),
            "search_prefix": self._ids(tl.search("miss*")),
            "counts": tl.topic_counts(),
            "stats": tl.stats(),
        }
        out["pruned_keep"] = tl.prune_keep_latest(2)
        out["after_keep"] = [r["event_id"] for r in tl.query(limit=100)]
        out["pruned_old"] = tl.prune_older_than(1003.0)
        out["after_old"] = [r["event_id"] for r in tl.query(limit=100)]
        return out

    def test_conformance(self, rung, tmp_path):
        from nomorals.os.timeline import Timeline

        tl = Timeline(str(tmp_path / "tl.db"), store=rung.store)
        try:
            observed = self._scenario(tl)
        finally:
            tl.close()
        # the honesty pin: FTS5 exists only on the native rung
        assert observed["fts_available"] is (rung.store is None)
        assert observed["all_topics"] == [
            "task.done", "artifact.created", "session.created",
            "mission.verify", "mission.transition"]
        assert observed["by_mission"] == [
            "artifact.created", "mission.verify", "mission.transition"]
        assert observed["glob"] == [
            "mission.transition", "mission.verify"]
        assert observed["since"] == 3
        # identical search RESULT SETS on every rung — the bar the
        # disposition is held to (mechanism differs, results don't)
        assert observed["search_alpha"] == ["e1", "e2"]
        assert observed["search_phrase"] == ["e1"]
        assert observed["search_or"] == ["e2", "e5"]
        assert observed["search_prefix"] == ["e1", "e2", "e4"]
        assert observed["counts"] == [
            ("artifact.created", 1), ("mission.transition", 1),
            ("mission.verify", 1), ("session.created", 1),
            ("task.done", 1)]
        stats = observed["stats"]
        assert stats["events"] == 5 and stats["topics"] == 5
        assert stats["span_from"] == 1000.0
        assert stats["span_to"] == 1004.0
        assert stats["fts_available"] is (rung.store is None)
        assert observed["pruned_keep"] == 3
        assert observed["after_keep"] == ["e5", "e4"]
        assert observed["pruned_old"] == 1
        assert observed["after_old"] == ["e5"]

    def test_export_import_round_trip(self, rung, tmp_path):
        from nomorals.os.timeline import Timeline

        tl = Timeline(str(tmp_path / "tl.db"), store=rung.store)
        try:
            self._corpus(tl)
            dump = tmp_path / "dump.jsonl"
            assert tl.export_jsonl(dump) == 5
            # re-import upserts by event_id: nothing duplicates
            assert tl.import_jsonl(dump) == 5
            assert tl.stats()["events"] == 5
        finally:
            tl.close()

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.os.timeline import Timeline

        path = str(tmp_path / "tl.db")
        native = Timeline(path)
        self._corpus(native)
        native.close()

        seam = Timeline(path, store=rung.store)
        try:
            assert rung.store.count("os-timeline__event_log") == 5
            doc = rung.store.get("os-timeline__event_log", "e1")
            assert doc is not None and doc["topic"] == \
                "mission.transition"
            assert "alpha started" in doc["data_json"]
            # honest on the seam: no FTS5 index exists there
            assert seam.fts_available is False
            # imported events are searchable through the scan
            assert self._ids(seam.search("alpha")) == ["e1", "e2"]
            # an event recorded ON THE SEAM — never in the fossil
            # file — is searchable too: the disposition does not split
            seam.record(self._event(
                "e6", "task.done", 1005.0,
                {"note": "zephyr uniqueoneseam"}))
            assert self._ids(seam.search("uniqueoneseam")) == ["e6"]
            assert seam.stats()["events"] == 6
        finally:
            seam.close()
        again = Timeline(path, store=rung.store)
        try:
            assert rung.store.count("os-timeline__event_log") == 6
            assert again.stats()["events"] == 6
        finally:
            again.close()


class TestPlanningRoutePort:
    """planning/route.py — CostModel + RoutePlanner port (residue
    Batch 2, slice 6): the module's four raw connect sites are one
    store — both classes build a port facade over the same
    namespaced collections. The route_globals rollups port as-is
    (served, never re-derived); route_actuals' freshness-decayed
    EMA re-expresses as a read-modify-write on the joined
    (cell_a, cell_b) document id. Helpers live on the class, per
    the slice-2 placement lesson."""

    @staticmethod
    def _pair():
        from nomorals.planning.route import Stop

        return (Stop("a", "Alpha", 6.50, 3.20),
                Stop("b", "Beta", 6.60, 3.30))

    def _scenario(self, store, path) -> dict:
        from nomorals.planning.route import CostModel, RoutePlanner, Stop

        a, b = self._pair()
        cm = CostModel(str(path), store=store)
        recorded = [cm.record_actual(a, b, 30.0, 5000),
                    cm.record_actual(a, b, 40.0, 6000)]
        learned = cm._learned(a, b)
        detail = cm.estimate_detail(a, b)
        unknown = cm.estimate_detail(
            Stop("c", "Gamma", 9.0, 9.0), Stop("d", "Delta", 9.5, 9.5))
        stats = cm.stats()
        out = {
            "recorded": recorded,
            "learned": (round(learned[0], 3), round(learned[1], 3)),
            "detail": (round(detail[0], 3), detail[1], detail[2]),
            "unknown_source": unknown[2],
            "trips": stats["recorded_trips"],
            "recent": stats["recent_trips_30d"],
            "source": stats["source"],
            "speed": stats["speed_kmh"],
            "speed_global": round(
                cm._get_global("speed_kmh", -1.0), 6),
        }
        rp = RoutePlanner(str(path), store=store)
        s1 = rp.add_stop("Market", lat=6.51, lng=3.21,
                         dwell_minutes=20)
        rp.add_stop("Office", lat=6.61, lng=3.31)
        out["window_set"] = rp.set_window("Market", "09:00", "17:00")
        out["stop_id_shape"] = bool(
            s1 and s1.stop_id.startswith("stop_"))
        # a fresh planner over the same store sees the persisted
        # registry — stops, windows and the learned model alike
        rp2 = RoutePlanner(str(path), store=store)
        market = rp2.find_stop("market")
        out["labels"] = sorted(s.label for s in rp2.list_stops())
        out["dwell"] = market.dwell_minutes if market else None
        out["window_hours"] = (
            round((market.window[1] - market.window[0]) / 3600.0, 3)
            if market and market.window else None)
        out["trips_via_planner"] = rp2.record_trip(a, b, 50.0, 7000)
        out["trips_after"] = rp2.cost.stats()["recorded_trips"]
        learned2 = rp2.cost._learned(a, b)
        out["learned_after"] = (round(learned2[0], 3),
                                 round(learned2[1], 3))
        return out

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "route.db")
        assert observed["recorded"] == [True, True]
        # freshness-decayed EMA, w=0.35 on a fresh pair:
        # minutes 30 -> 30*.65 + 40*.35 = 33.5; cost likewise 5350
        assert observed["learned"] == (33.5, 5350.0)
        assert observed["detail"] == (33.5, 5350, "learned")
        assert observed["unknown_source"] == "heuristic"
        assert observed["trips"] == 2
        assert observed["recent"] == 2
        assert observed["source"] == "learned"
        # the learned speed is a STORED rollup: the global the model
        # reports is the global it persisted, on every rung
        assert observed["speed"] == round(observed["speed_global"], 1)
        assert observed["labels"] == ["Market", "Office"]
        assert observed["stop_id_shape"] is True
        assert observed["window_set"] is True
        assert observed["dwell"] == 20.0
        assert observed["window_hours"] == 8.0
        assert observed["trips_via_planner"] is True
        assert observed["trips_after"] == 3
        # third observation: 33.5*.65 + 50*.35 = 39.275; cost 5927.5
        assert observed["learned_after"] == (39.275, 5927.5)

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.planning.route import CostModel, RoutePlanner

        a, b = self._pair()
        path = str(tmp_path / "route.db")
        native = CostModel(path)
        assert native.record_actual(a, b, 30.0, 5000)
        native_planner = RoutePlanner(path)
        native_planner.add_stop("Market", lat=6.51, lng=3.21,
                                dwell_minutes=20)

        seam = CostModel(path, store=rung.store)
        assert rung.store.count("planning-route__route_actuals") == 1
        # speed + cost-per-km rollups were learned and stored natively
        assert rung.store.count("planning-route__route_globals") == 2
        assert rung.store.count("planning-route__route_stops") == 1
        doc = rung.store.get("planning-route__route_globals",
                             "speed_kmh")
        assert doc is not None
        assert doc["value"] == pytest.approx(
            native._get_global("speed_kmh", -1.0))
        # the imported actual is served as-is, learning continues
        assert seam._learned(a, b) == (30.0, 5000.0)
        assert seam.record_actual(a, b, 40.0, 6000)
        doc = rung.store.get("planning-route__route_actuals",
                             seam._pair_key(a, b)[0] + "\x1f"
                             + seam._pair_key(a, b)[1])
        assert doc is not None and doc["n"] == 2
        seam_planner = RoutePlanner(path, store=rung.store)
        assert [s.label for s in seam_planner.list_stops()] == ["Market"]
        again = CostModel(path, store=rung.store)
        assert rung.store.count("planning-route__route_actuals") == 1
        assert again.stats()["recorded_trips"] == 2


class TestMarketingBriefsPort:
    """marketing/briefs.py — BriefStore port (residue Batch 2,
    slice 5): briefs + drafts behind the unchanged never-raises
    API; the PK-less score_feedback log rides put_raw with
    caller-shaped <draft_id>:<ordinal> ids."""

    @staticmethod
    def _search(topic):
        return [{"title": "solar power guide",
                 "snippet": "solar panels save money on bills",
                 "text": "how do solar panels work in practice",
                 "url": "https://example.invalid/solar"}]

    def _scenario(self, store_) -> dict:
        brief = store_.build_brief("solar panels",
                                   search_fn=self._search)
        assert brief is not None
        draft = store_.draft_from_brief(brief)
        assert draft is not None and draft.score > 0
        got = store_.get_brief(brief.brief_id)
        latest = store_.latest_brief("solar")
        got_draft = store_.get_draft(draft.draft_id)
        found = store_.find_by_content(draft.content)
        for _ in range(5):
            assert store_.record_outcome(
                draft.draft_id, {"engagement_score": 60}) is True
        return {
            "brief_topic": got.topic if got else None,
            "latest_topic": latest.topic if latest else None,
            "draft_ok": got_draft is not None
                        and got_draft.brief_id == brief.brief_id,
            "found": len(found),
            "calibration": store_._calibration() is not None,
            "stale": len(store_.stale_drafts(days=30)),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.marketing.briefs import BriefStore

        store_ = BriefStore(str(tmp_path / "briefs.db"),
                            store=rung.store)
        assert self._scenario(store_) == {
            "brief_topic": "solar panels",
            "latest_topic": "solar panels",
            "draft_ok": True,
            "found": 1,
            "calibration": True,
            "stale": 0,
        }

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.marketing.briefs import BriefStore

        path = str(tmp_path / "briefs.db")
        native = BriefStore(path)
        self._scenario(native)

        seam = BriefStore(path, store=rung.store)
        assert rung.store.count("marketing-briefs__briefs") == 1
        assert rung.store.count("marketing-briefs__drafts") == 1
        assert rung.store.count("marketing-briefs__score_feedback") == 5
        doc = rung.store.find("marketing-briefs__briefs")[0]
        assert doc["topic"] == "solar panels"
        assert "term_bands" in doc
        draft_id = rung.store.find("marketing-briefs__drafts")[0]["_id"]
        fb_ids = sorted(d["_id"] for d in
                        rung.store.find("marketing-briefs__score_feedback"))
        assert fb_ids == [f"{draft_id}:{n}" for n in range(1, 6)]
        # a second construction adopts — never re-imports — and the
        # per-draft feedback ordinal continues on the seam
        again = BriefStore(path, store=rung.store)
        assert rung.store.count("marketing-briefs__score_feedback") == 5
        assert again.record_outcome(
            draft_id, {"engagement_score": 42}) is True
        assert rung.store.count("marketing-briefs__score_feedback") == 6
        assert rung.store.get("marketing-briefs__score_feedback",
                              f"{draft_id}:6") is not None
        assert seam.get_brief(doc["_id"]) is not None


class TestMarketingCompetitorPort:
    """marketing/competitor.py — CompetitorStore port (residue
    Batch 2, slice 5): tracked's composite (account, platform)
    key lives only in the joined document id; snapshots are
    append documents behind the unchanged never-raises API."""

    @staticmethod
    def _posts(n, likes):
        from nomorals.marketing.competitor import Post

        return [Post(platform="instagram", account="rival",
                     post_id=f"p{i}", text="new video drop",
                     format="video", likes=likes, comments=2,
                     shares=1)
                for i in range(n)]

    def _scenario(self, store_) -> dict:
        assert store_.track("Rival", "instagram") is True
        assert store_.track("rival", "instagram") is True  # dup ignored
        first = store_.analyze(
            "rival", "instagram",
            scrape_fn=lambda a, p: self._posts(2, 10))
        second = store_.analyze(
            "rival", "instagram",
            scrape_fn=lambda a, p: self._posts(4, 40))
        return {
            "tracked": [(t["account"], t["platform"])
                        for t in store_.list_tracked()],
            "posts": (first.post_count, second.post_count),
            "digest": "competitor digest" in store_.digest(
                "rival", "instagram"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.marketing.competitor import CompetitorStore

        store_ = CompetitorStore(str(tmp_path / "competitor.db"),
                                 store=rung.store)
        assert self._scenario(store_) == {
            "tracked": [("rival", "instagram")],
            "posts": (2, 4),
            "digest": True,
        }
        assert store_.untrack("rival") is True
        assert store_.list_tracked() == []
        assert store_.untrack("rival") is False

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.marketing.competitor import CompetitorStore

        path = str(tmp_path / "competitor.db")
        native = CompetitorStore(path)
        self._scenario(native)

        seam = CompetitorStore(path, store=rung.store)
        assert rung.store.count("marketing-competitor__tracked") == 1
        assert rung.store.count("marketing-competitor__snapshots") == 2
        # the composite key lives only in the joined document id
        doc = rung.store.get("marketing-competitor__tracked",
                             "rival\x1finstagram")
        assert doc is not None
        assert "account" not in doc and "platform" not in doc
        assert seam.list_tracked()[0]["account"] == "rival"
        # a second construction adopts — never re-imports
        again = CompetitorStore(path, store=rung.store)
        assert rung.store.count("marketing-competitor__tracked") == 1
        assert again.untrack("rival", "instagram") is True
        assert rung.store.count("marketing-competitor__tracked") == 0


class TestMarketingGuardrailsPort:
    """marketing/guardrails.py — GuardrailStore port (residue
    Batch 2, slice 5): rules behind the unchanged never-raises
    API; the AUTOINCREMENT audit log rides put_raw with ids from
    the port's atomic counter, seeded from the imported maximum,
    and PK-less metric_history rides <adset_id>:<counter> ids."""

    def _scenario(self, store_) -> dict:
        from nomorals.marketing.guardrails import (
            FiredAction,
            fatigue_signal,
            parse_rule,
            record_metrics,
        )

        rule = parse_rule(
            "pause if cpa > ₦50000 for 3 days on adset123 via meta")
        assert rule is not None
        rule = store_.add_rule(rule)
        assert rule is not None and rule.rule_id
        got = store_.get_rule(rule.rule_id)
        for ts, detail in ((1000.0, "first"), (2000.0, "second")):
            assert store_.log_audit(FiredAction(
                rule_id=rule.rule_id, adset_id="adset123",
                action="pause", detail=detail, ok=True,
                ts=ts)) is True
        for ctr, freq in ((10.0, 1.0), (10.0, 1.0),
                          (5.0, 3.0), (5.0, 3.0)):
            assert record_metrics(
                store_, "adset123",
                {"ctr": ctr, "frequency": freq}) is True
        sig = fatigue_signal(store_, "adset123")
        return {
            "metric": got.metric, "threshold": got.threshold,
            "days": got.days, "action": got.action,
            "audit": [(a["detail"], a["ts"])
                      for a in store_.audit_log(10)],
            "listed": len(store_.list_rules()),
            "fatigued": sig["fatigued"],
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.marketing.guardrails import GuardrailStore

        store_ = GuardrailStore(str(tmp_path / "guardrails.db"),
                                store=rung.store)
        assert self._scenario(store_) == {
            "metric": "cpa", "threshold": 50000 * 100,
            "days": 3, "action": "pause",
            "audit": [("second", 2000.0), ("first", 1000.0)],
            "listed": 1,
            "fatigued": True,
        }

    def test_seam_shape_and_counter_continuity(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.marketing.guardrails import FiredAction, GuardrailStore, record_metrics

        path = str(tmp_path / "guardrails.db")
        native = GuardrailStore(path)
        self._scenario(native)

        seam = GuardrailStore(path, store=rung.store)
        assert rung.store.count("marketing-guardrails__rules") == 1
        audit_ids = sorted(d["_id"] for d in
                           rung.store.find("marketing-guardrails__audit"))
        assert audit_ids == ["1", "2"]
        assert rung.store.count(
            "marketing-guardrails__metric_history") == 4
        # the audit counter continues past the imported maximum
        rules = seam.list_rules()
        assert seam.log_audit(FiredAction(
            rule_id=rules[0].rule_id, adset_id="adset123",
            action="pause", detail="third", ok=True,
            ts=3000.0)) is True
        assert rung.store.get("marketing-guardrails__audit",
                              "3") is not None
        assert record_metrics(seam, "adset123",
                              {"ctr": 4.0, "frequency": 3.5}) is True
        assert rung.store.count(
            "marketing-guardrails__metric_history") == 5
        # a second construction adopts — never re-imports
        again = GuardrailStore(path, store=rung.store)
        assert rung.store.count("marketing-guardrails__audit") == 3
        assert len(again.audit_log(10)) == 3


class TestCommerceCartRecoveryPort:
    """commerce/cart_recovery.py — CartRecovery port (residue
    Batch 2, slice 5): carts + recoveries behind the unchanged
    API; cart_steps are child rows on put_raw under
    <cart_id>:<step> ids, and the SQL rollups re-express as
    finds plus Python rollups."""

    @staticmethod
    def _engine(path, clock, store=None):
        from nomorals.commerce.cart_recovery import CartRecovery

        return CartRecovery(str(path),
                            sender=lambda phone, text: True,
                            now=lambda: clock[0], store=store)

    def _scenario(self, engine, clock) -> dict:
        cart = engine.cart_abandoned(
            "c1", "+2348000000000", [{"name": "shoes", "qty": 1}],
            50000, True, store="shop")
        clock[0] += 73 * 3600  # past every step's delay
        sent = engine.run_due_steps()
        assert engine.mark_recovered("c1",
                                     recovered_kobo=50000) is True
        stats = engine.stats()
        return {
            "cart": cart.cart_id,
            "sent": sent,
            "recovered": stats["recovered_7d"],
            "kobo": stats["recovered_kobo_7d"],
            "sent_msgs": stats["messages_sent"],
            "via": stats["recovered_via_step_7d"],
            "weekly": "recovered" in engine.weekly_report(),
            "funnel": "recovered" in engine.funnel_report(),
        }

    def test_conformance(self, rung, tmp_path):
        clock = [1_700_000_000.0]
        engine = self._engine(tmp_path / "carts.db", clock,
                              store=rung.store)
        assert self._scenario(engine, clock) == {
            "cart": "c1",
            "sent": 3,
            "recovered": 1,
            "kobo": 50000,
            "sent_msgs": 3,
            "via": {"3": 1},
            "weekly": True,
            "funnel": True,
        }

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        path = tmp_path / "carts.db"
        clock = [1_700_000_000.0]
        native = self._engine(path, clock)
        native.cart_abandoned(
            "c1", "+2348000000000", [{"name": "shoes", "qty": 1}],
            50000, True, store="shop")

        seam = self._engine(path, clock, store=rung.store)
        assert rung.store.count("commerce-cart-recovery__carts") == 1
        assert rung.store.count(
            "commerce-cart-recovery__recoveries") == 0
        step_ids = sorted(
            d["_id"] for d in rung.store.find(
                "commerce-cart-recovery__cart_steps"))
        assert step_ids == ["c1:1", "c1:2", "c1:3"]
        # the imported sequence runs on the seam
        clock[0] += 73 * 3600
        assert seam.run_due_steps() == 3
        assert seam.mark_recovered("c1",
                                   recovered_kobo=50000) is True
        assert rung.store.count(
            "commerce-cart-recovery__recoveries") == 1
        doc = rung.store.get("commerce-cart-recovery__recoveries",
                             "c1")
        assert doc is not None and doc["recovered_via_step"] == 3
        # a second construction adopts — never re-imports
        again = self._engine(path, clock, store=rung.store)
        assert rung.store.count("commerce-cart-recovery__carts") == 1
        assert again.stats()["recovered_7d"] == 1


class TestTravelWatchersPort:
    """travel/watchers.py — PriceWatcher port (residue Batch 2,
    slice 8): watches are documents by text id; the PK-less
    price_history / alert_log children ride put_raw with
    caller-shaped <watch_id>:<n> ids from one atomic counter per
    table on the meta document, seeded by the lazy import so seam
    ids stay clear of imported ones (and of the history prune).
    Helpers live on the class, per the slice-2 placement lesson."""

    @staticmethod
    def _duffel(prices):
        from types import SimpleNamespace

        class _Duffel:
            def __init__(self):
                self.prices = list(prices)

            def search_offers(self, origin, destination, date):
                amount = self.prices.pop(0)
                return [SimpleNamespace(total_amount=amount,
                                        total_currency="NGN", raw={})]

        return _Duffel()

    def _scenario(self, store, path) -> dict:
        from nomorals.travel.watchers import PriceWatcher

        clock = [1_700_000_000.0]
        duffel = self._duffel(["5000.00", "4500.00", "4000.00"])
        w = PriceWatcher(str(path), duffel=duffel,
                         now=lambda: clock[0], store=store)
        watch = w.watch("LOS", "LHR", "2026-12-01",
                        target_kobo=460_000)
        out: dict = {"id_shape": watch.id.startswith("watch_")}
        # first check establishes the baseline — no alert yet
        out["first_quiet"] = w.check(watch) is None
        clock[0] += 3600
        alert = w.check(watch)  # ₦5000 → ₦4500: 10% drop, target crossed
        out["drop"] = (alert is not None and alert.kind == "drop"
                       and alert.current_kobo == 450_000
                       and alert.previous_kobo == 500_000)
        clock[0] += 3600
        alert2 = w.check(watch)  # ₦4500 → ₦4000: a fresh low
        out["second_drop"] = (alert2 is not None
                              and alert2.current_kobo == 400_000)
        out["history"] = [p.amount_kobo for p in w.history(watch.id)]
        got = w.get(watch.id)
        out["state"] = (got.last_kobo, got.alerted_kobo, got.active,
                        got.notify_on, got.cooldown_h)
        out["listed"] = [x.id for x in w.list_watches()] == [watch.id]
        out["unwatch"] = w.unwatch(watch.id)
        out["after_unwatch"] = (len(w.list_watches()),
                                len(w.list_watches(active_only=False)))
        return out

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "watches.db")
        assert observed["id_shape"] is True
        assert observed["first_quiet"] is True
        assert observed["drop"] is True
        assert observed["second_drop"] is True
        assert observed["history"] == [500_000, 450_000, 400_000]
        assert observed["state"] == (400_000, 400_000, True, "drop", 24.0)
        assert observed["listed"] is True
        assert observed["unwatch"] is True
        assert observed["after_unwatch"] == (0, 1)

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.travel.watchers import PriceWatcher

        clock = [1_700_000_000.0]
        path = str(tmp_path / "watches.db")
        native = PriceWatcher(path, now=lambda: clock[0])
        watch = native.watch("LOS", "LHR", "2026-12-01")
        native._record_point(watch.id, 500_000, "NGN", clock[0])
        native._log_alert(watch.id, "drop", 500_000, clock[0])
        native._db.commit()  # check() commits on the real native flow

        seam = PriceWatcher(path, now=lambda: clock[0], store=rung.store)
        assert rung.store.count("travel-watchers__watches") == 1
        assert rung.store.count("travel-watchers__price_history") == 1
        assert rung.store.count("travel-watchers__alert_log") == 1
        doc = rung.store.get("travel-watchers__watches", watch.id)
        assert doc is not None and doc["_id"] == watch.id
        point = rung.store.get("travel-watchers__price_history",
                               f"{watch.id}:0")
        assert point is not None and point["kobo"] == 500_000
        # the imported alert still enforces the cooldown on the seam
        assert seam._recent_alert(watch.id, "drop", clock[0] + 60,
                                  24.0) is True
        # the seeded counter keeps seam point ids clear of imported ones
        clock[0] += 3600
        seam._record_point(watch.id, 450_000, "NGN", clock[0])
        assert rung.store.count("travel-watchers__price_history") == 2
        assert [p.amount_kobo for p in seam.history(watch.id)] == \
            [500_000, 450_000]
        again = PriceWatcher(path, now=lambda: clock[0], store=rung.store)
        assert rung.store.count("travel-watchers__watches") == 1
        assert [p.amount_kobo for p in again.history(watch.id)] == \
            [500_000, 450_000]


# ---------------------------------------------------------------------------
# residue Batch 2, slice 8: social-leads
# ---------------------------------------------------------------------------

class TestSocialLeadsPort:
    """social/leads.py — LeadStore port (residue Batch 2, slice 8):
    triggers and leads are documents by text id; comment_seen and
    trigger_meta are keyed by composite primary keys whose parts
    live in the joined document id (membership is a get by joined
    id, never a find on key fields); the PK-less lead_comments log
    rides put_raw with caller-shaped <post_id>:<n> ids from one
    atomic counter on the meta document, seeded by the lazy import;
    funnel_stats' GROUP BY / COUNT DISTINCT rollups re-express as
    finds + Python rollups. Helpers live on the class, per the
    slice-2 placement lesson."""

    def _scenario(self, store, path) -> dict:
        from nomorals.social.leads import (
            CommentEvent,
            LeadStore,
            handle_comment,
            is_lead_magnet,
            mark_lead_magnet,
        )

        s = LeadStore(str(path), store=store)
        trig = s.add_trigger("instagram", "post1", "PRICE")
        out: dict = {"armed": trig is not None
                     and trig.keyword == "PRICE"}
        out["meta"] = (
            s.set_public_reply(trig.trigger_id,
                               "Sent {name} — check your DMs"),
            s.set_qualifier(trig.trigger_id, "Training now?"))
        ev = CommentEvent(platform="instagram", post_id="post1",
                          commenter="Ada", commenter_contact="@ada",
                          text="PRICE please")
        res = handle_comment(s, ev)
        out["first"] = [(r["dmed"], r["reason"]) for r in res]
        out["public"] = res[0].get("public_reply")
        out["qualifier"] = res[0].get("qualifier")
        res2 = handle_comment(s, ev)  # same commenter: already DMed
        out["again"] = [(r["dmed"], r["reason"]) for r in res2]
        ev2 = CommentEvent(platform="instagram", post_id="post1",
                           commenter="Bola", commenter_contact="@bola",
                           text="PRICE!")
        handle_comment(s, ev2)
        out["commenters"] = sorted(s.commenters("post1"),
                                    key=lambda d: d["commenter"])
        out["funnel"] = s.funnel_stats(trig.trigger_id)
        out["magnet"] = (mark_lead_magnet(s, trig.trigger_id),
                         is_lead_magnet(s, trig.trigger_id))
        lead = s.list_leads()[0]
        out["lead_status"] = (s.set_lead_status(lead.lead_id,
                                                "converted"),
                              s.get_lead(lead.lead_id).status)
        out["stop"] = (s.stop_trigger(trig.trigger_id),
                       s.get_trigger(trig.trigger_id).active)
        return out

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "leads.db")
        assert observed["armed"] is True
        assert observed["meta"] == (True, True)
        assert observed["first"] == [(True, "sent")]
        assert observed["public"] == "Sent Ada — check your DMs"
        assert observed["qualifier"] == "Training now?"
        assert observed["again"] == [(False, "already DMed")]
        assert observed["commenters"] == [
            {"commenter": "Ada", "contact": "@ada"},
            {"commenter": "Bola", "contact": "@bola"}]
        funnel = observed["funnel"]
        assert funnel["comments"] == 2 and funnel["dmed"] == 2
        assert funnel["dm_rate"] == 1.0
        assert funnel["leads_by_status"] == {"dmed": 2}
        assert funnel["converted"] == 0
        assert funnel["qualifier"] == "Training now?"
        assert funnel["public_reply"] is True
        assert observed["magnet"] == (True, True)
        assert observed["lead_status"] == (True, "converted")
        assert observed["stop"] == (True, False)

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.leads import CommentEvent, LeadStore, handle_comment, mark_lead_magnet

        path = str(tmp_path / "leads.db")
        native = LeadStore(path)
        trig = native.add_trigger("instagram", "post1", "PRICE")
        native.set_qualifier(trig.trigger_id, "Training now?")
        ev = CommentEvent(platform="instagram", post_id="post1",
                          commenter="Ada", commenter_contact="@ada",
                          text="PRICE please")
        handle_comment(native, ev)
        mark_lead_magnet(native, trig.trigger_id)

        seam = LeadStore(path, store=rung.store)
        assert rung.store.count("social-leads__lead_triggers") == 1
        assert rung.store.count("social-leads__leads") == 1
        assert rung.store.count("social-leads__comment_seen") == 1
        assert rung.store.count("social-leads__lead_comments") == 1
        assert rung.store.count("social-leads__trigger_meta") == 1
        assert rung.store.count("social-leads__lead_magnets") == 1
        doc = rung.store.get("social-leads__lead_comments", "post1:0")
        assert doc is not None and doc["commenter"] == "Ada"
        # composite-PK membership survives the import: the same
        # commenter is recognised as already DMed, not re-DMed
        assert seam.already_dmed("instagram", "post1", "Ada",
                                 trig.trigger_id) is True
        res = handle_comment(seam, ev)
        assert res[0]["reason"] == "already DMed"
        # handle_comment logs every comment event before the dedup
        # check (the native contract too — funnel_stats is what
        # dedups, via COUNT DISTINCT), so the re-handle added a
        # second log row for Ada; the seeded counter kept its id
        # clear of the imported post1:0
        assert rung.store.count("social-leads__lead_comments") == 2
        ev2 = CommentEvent(platform="instagram", post_id="post1",
                           commenter="Bola", commenter_contact="@bola",
                           text="PRICE!")
        handle_comment(seam, ev2)
        assert rung.store.count("social-leads__lead_comments") == 3
        assert seam.funnel_stats(trig.trigger_id)["comments"] == 2
        again = LeadStore(path, store=rung.store)
        assert rung.store.count("social-leads__leads") == 2
        assert again.funnel_stats(trig.trigger_id)["dmed"] == 2


class TestMatchingQuestionnairePort:
    """matching/questionnaire.py — Questionnaire port (residue
    Batch 2, slice 7): all four tables over the seam behind the
    unchanged never-raises API. Every table is composite-keyed by
    owner, so the key parts live only in the joined document id —
    point reads fetch by it, owner-scoped reads scan restored rows.
    Helpers live on the class, per the slice-2 placement lesson."""

    def _scenario(self, store, path) -> dict:
        from nomorals.matching.questionnaire import Questionnaire

        q = Questionnaire("owner", str(path), store=store)
        starters = sorted(x.question_id for x in q.ensure_starters("gig"))
        assert q.answer("gig:remote", 5, "yes")
        assert q.set_acceptable("gig:remote", ["yes", "hybrid"])
        assert q.answer("gig:speed", 2, "fast")
        unanswered = [x.question_id for x in q.unanswered()]
        unanswered_b = [x.question_id
                        for x in q.unanswered("boundaries")]
        boundary = [x.question_id for x in q.boundary_questions()]
        assert q.answer("gig:budget_fit", 3, "strict")
        ghost_answer = q.answer("gig:ghost", 4, "x")
        ghost_accept = q.set_acceptable("gig:ghost", ["x"])
        answered = [x.question_id for x in q.answered()]
        breakdown = sorted(
            (r["question_id"], r["weight"], r["matched"])
            for r in q.match_breakdown(
                {"remote": "yes", "speed": "fast",
                 "budget_fit": "loose"}))
        satisfaction = round(q.satisfaction(
            {"remote": "yes", "speed": "fast",
             "budget_fit": "loose"}), 6)
        score = q.score({"attributes": {"remote": "yes"}})
        assert q.set_dealbreaker("budget", "budget", "le", 5000)
        assert q.set_dealbreaker("when", "start", "in", ["now", "soon"])
        breakers = sorted(
            (d.name, d.attribute, d.predicate,
             tuple(d.value) if isinstance(d.value, list) else d.value)
            for d in q.dealbreakers())
        kept = q.filter_dealbreakers([
            {"attributes": {"budget": 4000, "start": "now"}},
            {"attributes": {"budget": 9000, "start": "now"}},
            {"attributes": {"budget": 100, "start": "later"}}])
        removed = q.remove_dealbreaker("when")
        removed_again = q.remove_dealbreaker("when")
        assert q.set_ranking("cand-a", "proposer", ["rev-1", "rev-2"])
        assert q.set_capacity("rev-1", 3)
        other = Questionnaire("other", str(path), store=store)
        other.ensure_starters("gig")
        assert other.answer("gig:remote", 4, "yes")
        assert other.set_acceptable("gig:remote", ["yes"])
        return {
            "starters": starters,
            "unanswered": unanswered,
            "unanswered_b": unanswered_b,
            "boundary": boundary,
            "ghost_answer": ghost_answer,
            "ghost_accept": ghost_accept,
            "answered": answered,
            "weights": q.weights(),
            "okcupid": q.okcupid_weights(),
            "breakdown": breakdown,
            "satisfaction": satisfaction,
            "score": score,
            "breakers": breakers,
            "kept": kept,
            "removed": removed,
            "removed_again": removed_again,
            "rankings": q.rankings("proposer"),
            "capacities": q.capacities(),
            "match_percent": q.match_percent(other),
        }

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "q.db")
        assert observed["starters"] == [
            "gig:budget_fit", "gig:experience", "gig:remote", "gig:speed"]
        assert observed["unanswered"] == ["gig:budget_fit",
                                          "gig:experience"]
        assert observed["unanswered_b"] == ["gig:budget_fit"]
        assert observed["boundary"] == ["gig:budget_fit"]
        assert observed["ghost_answer"] is False
        assert observed["ghost_accept"] is False
        # most-recently-updated first (answers were set in this order)
        assert observed["answered"] == [
            "gig:budget_fit", "gig:speed", "gig:remote"]
        assert observed["weights"] == {
            "budget_fit": 3.0, "remote": 5.0, "speed": 2.0}
        assert observed["okcupid"] == {
            "budget_fit": 50, "remote": 250, "speed": 10}
        assert observed["breakdown"] == [
            ("gig:budget_fit", 50, False), ("gig:remote", 250, True),
            ("gig:speed", 10, True)]
        assert observed["satisfaction"] == round(260 / 310, 6)
        assert observed["score"] == 1.0
        assert observed["breakers"] == [
            ("budget", "budget", "le", 5000),
            ("when", "start", "in", ("now", "soon"))]
        assert observed["kept"] == [
            {"attributes": {"budget": 4000, "start": "now"}}]
        assert observed["removed"] is True
        assert observed["removed_again"] is False
        assert observed["rankings"] == {"cand-a": ["rev-1", "rev-2"]}
        assert observed["capacities"] == {"rev-1": 3}
        assert observed["match_percent"] == 100.0

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.matching.questionnaire import Questionnaire

        path = tmp_path / "q.db"
        native = Questionnaire("owner", str(path))
        native.ensure_starters("gig")
        native.answer("gig:remote", 5, "yes")
        native.set_dealbreaker("budget", "budget", "le", 5000)
        native.set_ranking("cand-a", "proposer", ["rev-1"])
        native.set_capacity("rev-1", 3)

        seam = Questionnaire("owner", str(path), store=rung.store)
        assert rung.store.count("matching-questionnaire__answers") == 4
        assert rung.store.count(
            "matching-questionnaire__dealbreakers") == 1
        assert rung.store.count("matching-questionnaire__ranks") == 1
        assert rung.store.count(
            "matching-questionnaire__capacities") == 1
        # the composite key lives in the joined document id only
        doc = rung.store.get("matching-questionnaire__answers",
                             "owner\x1fgig:remote")
        assert doc is not None and doc["importance"] == 5
        assert "owner" not in doc and "question_id" not in doc
        # the imported state is the live state on the seam
        assert seam.get("gig:remote").importance == 5
        assert seam.weights() == {"remote": 5.0}
        assert seam.rankings("proposer") == {"cand-a": ["rev-1"]}
        # answering continues on the seam, on the imported rows
        assert seam.answer("gig:speed", 2, "fast")
        again = Questionnaire("owner", str(path), store=rung.store)
        assert rung.store.count("matching-questionnaire__answers") == 4
        assert again.get("gig:speed").importance == 2


class TestPropertyPassportPort:
    """property/passport.py — PassportStore port (residue Batch 2,
    slice 7): the passport aggregate over the seam behind the
    unchanged never-raises API. The port brief warned of JSON-blob
    document rows; the live store carries none — history / refs /
    doc_refs are typed single-PK child rows keyed by
    module-allocated ids, assembled per passport exactly as the
    native SELECTs assembled them, and the store still holds vault
    references only. Time is a fake clock (the module's own time
    source) so attestation/list observables are exact on every
    rung. Helpers live on the class, per the slice-2 placement
    lesson."""

    class _Clock:
        """Deterministic time.time: +60s per call, both rungs."""

        def __init__(self):
            self.t = 1_700_000_000.0

        def time(self):
            value = self.t
            self.t += 60.0
            return value

    def _scenario(self, store, path) -> dict:
        from nomorals.property.passport import PassportStore, referencing_readiness

        s = PassportStore(str(path), store=store)
        p = s.generate("Ada Tenant", memory={
            "income_band": "250k-500k", "kyc_summary": "NIN verified",
            "rental_history": [{
                "address": "12 Yaba Rd", "landlord": "Mr Ojo",
                "years": "2022-2024", "note": "good tenant"}]})
        pid = p.passport_id
        band_bad = s.set_income_band(pid, "bogus")
        band_ok = s.set_income_band(pid, "500k-1m")
        # native parity: the setters report True for a missing
        # passport (the UPDATE's rowcount is never checked)
        ghost_band = s.set_income_band("pp_ghost", "500k-1m")
        kyc_ok = s.set_kyc(pid, "NIN + BVN verified")
        hist_ok = s.add_history(pid, "3 Marina Close", "Mrs Bello",
                                "2020-2022")
        hist_ghost = s.add_history("pp_ghost", "Nowhere")
        ref_ok = s.add_reference(pid, "Mr Ojo", "former landlord",
                                 "vault:ref1")
        doc_ok = s.add_doc_ref(pid, "vault:doc_nin_2024", "NIN slip")
        doc2_ok = s.add_doc_ref(pid, "vault:util_2024",
                                "recent utility bill")
        s.generate("Bola Second")
        order1 = [x.owner_name for x in s.list()]
        assert s.set_kyc(pid, "NIN + BVN + utility verified")
        order2 = [x.owner_name for x in s.list()]
        reatt = s.reattest(pid)
        got = s.get(pid)
        readiness = referencing_readiness(got)
        return {
            "generated": (p.owner_name, p.income_band, p.kyc_summary,
                          [(h.address, h.landlord, h.years, h.note)
                           for h in p.history]),
            "band_bad": band_bad, "band_ok": band_ok,
            "ghost_band": ghost_band, "kyc_ok": kyc_ok,
            "hist_ok": hist_ok, "hist_ghost": hist_ghost,
            "ref_ok": ref_ok, "doc_ok": doc_ok, "doc2_ok": doc2_ok,
            "order1": order1, "order2": order2,
            "default": s.default().owner_name,
            "reatt": reatt,
            "got": (got.owner_name, got.income_band, got.kyc_summary,
                    sorted((h.address, h.landlord, h.years, h.note)
                           for h in got.history),
                    sorted((r.name, r.relationship, r.contact_ref)
                           for r in got.references),
                    sorted((d.vault_id, d.label)
                           for d in got.doc_refs),
                    got.is_fresh()),
            "export": s.export_text(pid),
            "readiness": (readiness["score"], readiness["ok"],
                          readiness["verdict"], readiness["missing"]),
        }

    def test_conformance(self, rung, tmp_path, monkeypatch):
        import nomorals.property.passport as passport_mod

        monkeypatch.setattr(passport_mod, "time",
                            SimpleNamespace(time=self._Clock().time))
        observed = self._scenario(rung.store, tmp_path / "pp.db")
        assert observed["generated"] == (
            "Ada Tenant", "250k-500k", "NIN verified",
            [("12 Yaba Rd", "Mr Ojo", "2022-2024", "good tenant")])
        assert observed["band_bad"] is False
        assert observed["band_ok"] is True
        assert observed["ghost_band"] is True
        assert observed["kyc_ok"] is True
        assert observed["hist_ok"] is True
        assert observed["hist_ghost"] is False
        assert observed["ref_ok"] is True
        assert observed["doc_ok"] is True and observed["doc2_ok"] is True
        assert observed["order1"] == ["Bola Second", "Ada Tenant"]
        assert observed["order2"] == ["Ada Tenant", "Bola Second"]
        assert observed["default"] == "Ada Tenant"
        assert observed["reatt"] is True
        assert observed["got"] == (
            "Ada Tenant", "500k-1m", "NIN + BVN + utility verified",
            [("12 Yaba Rd", "Mr Ojo", "2022-2024", "good tenant"),
             ("3 Marina Close", "Mrs Bello", "2020-2022", "")],
            [("Mr Ojo", "former landlord", "vault:ref1")],
            [("vault:doc_nin_2024", "NIN slip"),
             ("vault:util_2024", "recent utility bill")],
            True)
        assert observed["readiness"] == (
            100, True, "referencing-ready — share the passport",
            ["kyc: employment letter / staff ID, "
             "3 months' bank statement, passport photograph"])
        export = observed["export"]
        assert "Income band: ₦500,000–₦1,000,000/mo" in export
        assert "vault:doc_nin_2024" in export
        assert "attested 0d ago — fresh for" in export

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.property.passport import PassportStore

        path = tmp_path / "pp.db"
        native = PassportStore(str(path))
        p = native.generate("Ada Tenant", memory={
            "income_band": "250k-500k",
            "rental_history": [{"address": "12 Yaba Rd"}]})
        native.add_reference(p.passport_id, "Mr Ojo",
                             "former landlord", "vault:ref1")
        native.add_doc_ref(p.passport_id, "vault:doc_nin_2024",
                           "NIN slip")

        seam = PassportStore(str(path), store=rung.store)
        assert rung.store.count("property-passport__passports") == 1
        assert rung.store.count("property-passport__history") == 1
        assert rung.store.count("property-passport__refs") == 1
        assert rung.store.count("property-passport__doc_refs") == 1
        doc = rung.store.get("property-passport__passports",
                             p.passport_id)
        assert doc is not None and doc["income_band"] == "250k-500k"
        got = seam.get(p.passport_id)
        assert got.owner_name == "Ada Tenant"
        assert [h.address for h in got.history] == ["12 Yaba Rd"]
        assert [r.contact_ref for r in got.references] == ["vault:ref1"]
        # the aggregate keeps growing on the seam, on imported rows
        assert seam.add_history(p.passport_id, "3 Marina Close")
        assert rung.store.count("property-passport__history") == 2
        again = PassportStore(str(path), store=rung.store)
        assert rung.store.count("property-passport__passports") == 1
        assert len(again.get(p.passport_id).history) == 2

    def test_legacy_file_import(self, rung, tmp_path):
        """A passports file that predates the attested_at migration
        (never opened by the current organ) imports cleanly: the
        missing column lands as the NULL a native ALTER leaves."""
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import sqlite3

        from nomorals.property.passport import PassportStore

        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE passports (id TEXT PRIMARY KEY, "
            "owner_name TEXT, kyc_summary TEXT, income_band TEXT, "
            "created_at REAL, updated_at REAL)")
        conn.execute(
            "INSERT INTO passports VALUES ('pp_old', 'Old Timer', "
            "'kyc', '100k-250k', 10.0, 20.0)")
        conn.execute(
            "CREATE TABLE history (id TEXT PRIMARY KEY, "
            "passport_id TEXT, address TEXT, landlord TEXT, "
            "years TEXT, note TEXT)")
        conn.execute(
            "INSERT INTO history VALUES ('h_old', 'pp_old', "
            "'1 Old St', '', '', '')")
        conn.execute(
            "CREATE TABLE refs (id TEXT PRIMARY KEY, "
            "passport_id TEXT, name TEXT, relationship TEXT, "
            "contact_ref TEXT)")
        conn.execute(
            "CREATE TABLE doc_refs (id TEXT PRIMARY KEY, "
            "passport_id TEXT, vault_id TEXT, label TEXT)")
        conn.commit()
        conn.close()

        seam = PassportStore(str(path), store=rung.store)
        old = seam.get("pp_old")
        assert old is not None and old.owner_name == "Old Timer"
        assert old.attested_at == 0.0
        assert [h.address for h in old.history] == ["1 Old St"]


class TestSocialProfilesPort:
    """social/profiles.py — ProfileStore port (residue Batch 2,
    slice 7): profiles + elements + likes over the seam behind the
    unchanged never-raises API. Likes are composite-keyed (profile,
    element, liker) with the parts in the joined document id, so
    the JOIN reads re-express as finds and restored-row scans —
    get() counts likes per element id from the restored rows
    (element_id is a key part, never a find field), matches()
    walks owner → elements → likes with the native join's
    semantics. Helpers live on the class, per the slice-2
    placement lesson."""

    def _scenario(self, store, path, tmp) -> dict:
        from nomorals.social.profiles import ProfileStore

        vault = tmp / "vault"
        vault.mkdir(exist_ok=True)
        original_vault = ProfileStore.voice_vault_dir
        ProfileStore.voice_vault_dir = staticmethod(lambda: str(vault))
        try:
            s = ProfileStore(str(path), store=store)
            p = s.create("ada", "gig", "Ada")
            pid = p.profile_id
            coerced = s.create("bola", "nope-surface", "Bola")
            a1 = s.answer_prompt(pid, "best_work",
                                 "Built a market app")
            a_bad = s.answer_prompt(pid, "nope", "x")
            el1 = s.add_element(pid, "project", "Market App",
                                "the content")
            el2 = s.add_element(pid, "photo", "Studio Shot", "",
                                "http://x")
            el_ghost = s.add_element("prof_ghost", "project", "Ghost")
            lk1 = s.like(pid, el1.element_id, "bola", "nice work")
            lk_dup = s.like(pid, el1.element_id, "bola", "second")
            s.like(pid, el1.element_id, "chidi", "")
            s.like(pid, el2.element_id, "bola", "great pic")
            lk_bad = s.like(pid, "elem_ghost", "bola")
            got = s.get(pid)
            wav = tmp / "intro.wav"
            wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
            vi = s.set_voice_intro(
                pid, str(wav), language="en",
                stt=lambda pth, lang: {"text": "hello from ada",
                                       "backend": "fake-stt"})
            vi_seen = s.get_voice_intro(pid)
            vi_in_get = (got.voice_intro is None
                         and s.get(pid).voice_intro.transcript)
            vi_deleted = s.delete_voice_intro(pid)
            vi_gone = s.get_voice_intro(pid)
            return {
                "created": (p.owner, p.surface, p.display_name),
                "coerced_surface": coerced.surface,
                "a1": a1, "a_bad": a_bad,
                "el_ghost": el_ghost is None,
                "lk1": (lk1.liker, lk1.comment) if lk1 else None,
                "lk_dup_kept_first": (
                    lk_dup.comment if lk_dup else None),
                "lk_bad": lk_bad is None,
                "prompts": got.prompts,
                "elements": sorted(
                    (e.title, e.type, e.like_count)
                    for e in got.elements),
                "completeness": got.completeness()["score"],
                "unanswered": [pid_ for pid_, _ in s.unanswered(pid)],
                "likes_el1": [(lk.liker, lk.comment)
                              for lk in s.likes_for(el1.element_id)],
                "matches": [(lk.liker, lk.comment)
                            for lk in s.matches("ada")],
                "listed": sorted((x.owner, x.surface)
                                 for x in s.list()),
                "listed_business": s.list("business"),
                "voice": (vi.transcript, vi.language, vi.stt_backend,
                          vi.audio_path.endswith(pid + ".wav")),
                "vi_seen": vi_seen.transcript if vi_seen else None,
                "vi_in_get": vi_in_get,
                "vi_deleted": vi_deleted,
                "vi_gone": vi_gone is None,
            }
        finally:
            ProfileStore.voice_vault_dir = original_vault

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "p.db",
                                  tmp_path)
        assert observed["created"] == ("ada", "gig", "Ada")
        assert observed["coerced_surface"] == "gig"
        assert observed["a1"] is True and observed["a_bad"] is False
        assert observed["el_ghost"] is True
        assert observed["lk1"] == ("bola", "nice work")
        # liking twice keeps the first like — the returned like is
        # the new attempt, the stored one is still "nice work"
        assert observed["lk_dup_kept_first"] == "second"
        assert observed["lk_bad"] is True
        assert observed["prompts"] == {"best_work": "Built a market app"}
        assert observed["elements"] == [
            ("Market App", "project", 2), ("Studio Shot", "photo", 1)]
        assert observed["completeness"] == 43
        assert observed["unanswered"] == [
            "looking_for", "ask_me", "superpower"]
        assert observed["likes_el1"] == [
            ("bola", "nice work"), ("chidi", "")]
        assert observed["matches"] == [
            ("bola", "nice work"), ("chidi", ""),
            ("bola", "great pic")]
        assert observed["listed"] == [("ada", "gig"), ("bola", "gig")]
        assert observed["listed_business"] == []
        assert observed["voice"] == (
            "hello from ada", "en", "fake-stt", True)
        assert observed["vi_seen"] == "hello from ada"
        assert observed["vi_in_get"] == "hello from ada"
        assert observed["vi_deleted"] is True
        assert observed["vi_gone"] is True

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.profiles import ProfileStore

        path = tmp_path / "p.db"
        native = ProfileStore(str(path))
        p = native.create("ada", "gig", "Ada")
        el1 = native.add_element(p.profile_id, "project", "Market App")
        native.like(p.profile_id, el1.element_id, "bola", "nice work")
        native.like(p.profile_id, el1.element_id, "chidi", "")

        seam = ProfileStore(str(path), store=rung.store)
        assert rung.store.count("social-profiles__profiles") == 1
        assert rung.store.count("social-profiles__elements") == 1
        assert rung.store.count("social-profiles__likes") == 2
        # the composite like key lives in the joined document id
        doc = rung.store.get(
            "social-profiles__likes",
            f"{p.profile_id}\x1f{el1.element_id}\x1fbola")
        assert doc is not None and doc["comment"] == "nice work"
        got = seam.get(p.profile_id)
        assert got.elements[0].like_count == 2
        # likes continue on the seam, on the imported rows
        assert seam.like(p.profile_id, el1.element_id, "dara", "third")
        assert seam.get(p.profile_id).elements[0].like_count == 3
        again = ProfileStore(str(path), store=rung.store)
        assert rung.store.count("social-profiles__profiles") == 1
        assert len(again.likes_for(el1.element_id)) == 3

    def test_legacy_file_import(self, rung, tmp_path):
        """A profiles file that predates the voice-intro columns
        imports cleanly: the lazy import backfills them, and the
        profile reads as having no voice intro."""
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import sqlite3

        from nomorals.social.profiles import ProfileStore

        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE profiles (profile_id TEXT PRIMARY KEY, "
            "owner TEXT, surface TEXT, display_name TEXT, "
            "prompts_json TEXT, created_at REAL)")
        conn.execute(
            "INSERT INTO profiles VALUES ('prof_old', 'ada', 'gig', "
            "'Old Ada', '{}', 5.0)")
        conn.execute(
            "CREATE TABLE elements (element_id TEXT PRIMARY KEY, "
            "profile_id TEXT, type TEXT, title TEXT, content TEXT, "
            "url TEXT)")
        conn.execute(
            "CREATE TABLE likes (profile_id TEXT, element_id TEXT, "
            "liker TEXT, comment TEXT, created_at REAL, "
            "PRIMARY KEY (profile_id, element_id, liker))")
        conn.commit()
        conn.close()

        seam = ProfileStore(str(path), store=rung.store)
        old = seam.get("prof_old")
        assert old is not None and old.display_name == "Old Ada"
        assert old.voice_intro is None


class TestTriggersSavedSearchPort:
    """triggers/saved_search.py — SavedSearchStore port (residue
    Batch 2, slice 7): searches + seen dedup + the digest queue
    over the seam behind the unchanged API. seen_matches is a
    dedup registry composite-keyed (search, dedup key) — existence
    is a joined-id get. pending_digest is AUTOINCREMENT-keyed: its
    rows ride put_raw under their file seq and new rows allocate
    from the meta counter seeded by the import, so seq order is
    queue order on both rungs.

    THE CLAIM DISPOSITION, stated explicitly per the census: the
    digest queue's claims are plain single-process CRUD — take is
    a read-then-delete by the one host loop that runs the hourly
    check, and no cross-process claim semantics exist anywhere in
    the module — so this port does NOT need, and does not build,
    the C2 generation-fencing treatment; single-document writes
    are the whole requirement. Helpers live on the class, per the
    slice-2 placement lesson."""

    def _scenario(self, store, path) -> dict:
        import nomorals.triggers.saved_search as ss

        s = ss.SavedSearchStore(str(path), store=store)
        w1 = s.create("property", "2bed Yaba",
                      {"area": "Yaba", "beds": "2"}, now=1000.0)
        w2 = s.create("gig", "logo design", {}, now=2000.0)
        bad = s.create("mars", "nothing watchable")
        g1 = s.get(w1.id)
        s.mark_seen(w1.id, "ad:abc|100", now=1500.0)
        seen_yes = s.seen(w1.id, "ad:abc|100")
        seen_no = s.seen(w1.id, "ad:other|1")
        s.mark_run(w1.id, 3, now=3000.0)
        after_run = s.get(w1.id)
        snz = s.snooze(w1.id, 2, now=4000.0)
        snz_until = s.get(w1.id).snooze_until
        snz_ghost = s.snooze("watch_ghost", 2, now=4000.0)
        unsnz = s.unsnooze(w1.id)
        dig = s.set_digest_at(w1.id, "7:30")
        dig_val = s.get(w1.id).digest_at
        dig_bad = s.set_digest_at(w1.id, "25:00")
        s.queue_digest(ss.Match(
            search_id=w1.id, title="Flat A", price_kobo=150000000,
            area="Yaba", address="1 Yaba Rd", beds="2",
            url="http://a", found_at=5000.0))
        s.queue_digest(ss.Match(
            search_id=w1.id, title="Flat B", price_kobo=120000000,
            area="Yaba", found_at=6000.0))
        s.queue_digest(ss.Match(
            search_id=w2.id, title="Gig X", found_at=5500.0))
        cnt1 = s.pending_digest_count(w1.id)
        taken = s.take_digest(w1.id)
        listed = [(x.domain, x.query) for x in s.list(
            active_only=False, include_expired=True)]
        w3 = s.create("flight", "LOS to ABV", {}, ttl_days=7,
                      now=1000.0)
        purged = s.purge_expired(now=1000.0 + 8 * 86400)
        rm1 = s.remove(w2.id)
        rm2 = s.remove(w2.id)
        return {
            "bad_domain": bad is None,
            "g1": (g1.domain, g1.query, g1.filters, g1.cadence_s,
                   g1.ttl_days, g1.created_at, g1.expires_at,
                   g1.last_run, g1.match_count, g1.active,
                   g1.snooze_until, g1.digest_at),
            "seen_yes": seen_yes, "seen_no": seen_no,
            "after_run": (after_run.last_run, after_run.match_count),
            "snz": snz, "snz_until": snz_until, "snz_ghost": snz_ghost,
            "unsnz": unsnz, "dig": dig, "dig_val": dig_val,
            "dig_bad": dig_bad,
            "cnt1": cnt1,
            "taken": [(m.title, m.price_kobo, m.found_at)
                      for m in taken],
            "cnt_after": s.pending_digest_count(w1.id),
            "cnt_w2": s.pending_digest_count(w2.id),
            "listed": listed,
            "purged": purged,
            "w3_gone": s.get(w3.id) is None,
            "rm1": rm1, "rm2": rm2,
            "seen_after_rm": s.seen(w1.id, "ad:abc|100"),
        }

    def test_conformance(self, rung, tmp_path):
        observed = self._scenario(rung.store, tmp_path / "ss.db")
        assert observed["bad_domain"] is True
        assert observed["g1"] == (
            "property", "2bed Yaba", {"area": "Yaba", "beds": "2"},
            21600, 14, 1000.0, 1000.0 + 14 * 86400, 0.0, 0, True,
            0.0, "")
        assert observed["seen_yes"] is True
        assert observed["seen_no"] is False
        assert observed["after_run"] == (3000.0, 3)
        assert observed["snz"] is True
        assert observed["snz_until"] == 4000.0 + 2 * 3600
        assert observed["snz_ghost"] is False
        assert observed["unsnz"] is True
        assert observed["dig"] is True
        assert observed["dig_val"] == "07:30"
        assert observed["dig_bad"] is False
        assert observed["cnt1"] == 2
        assert observed["taken"] == [
            ("Flat A", 150000000, 5000.0), ("Flat B", 120000000, 6000.0)]
        assert observed["cnt_after"] == 0
        assert observed["cnt_w2"] == 1
        # newest first (read taken before the purge/remove segment)
        assert observed["listed"] == [
            ("gig", "logo design"), ("property", "2bed Yaba")]
        assert observed["purged"] == 1
        assert observed["w3_gone"] is True
        assert observed["rm1"] is True and observed["rm2"] is False
        assert observed["seen_after_rm"] is True

    def test_check_search_dedup_flow(self, rung, tmp_path):
        """The module-level check flow (match → filter → dedup →
        remember) behaves identically on every rung."""
        import time as _time

        import nomorals.triggers.saved_search as ss

        s = ss.SavedSearchStore(str(tmp_path / "ss.db"),
                                store=rung.store)
        w = s.create("property", "2bed Yaba", {"area": "Yaba"})

        def matcher(domain, query, filters):
            return [
                {"title": "New Flat", "price_kobo": 90000000,
                 "area": "Yaba", "address": "1 Yaba Rd", "beds": "2"},
                {"title": "New Flat", "price_kobo": 90000000,
                 "area": "Yaba", "address": "1 Yaba Rd", "beds": "2"},
                {"title": "Far Flat", "price_kobo": 90000000,
                 "area": "Ikeja", "address": "9 Far Rd", "beds": "2"},
            ]

        fresh = ss.check_search(s, s.get(w.id), matcher=matcher,
                                now=_time.time())
        again = ss.check_search(s, s.get(w.id), matcher=matcher,
                                now=_time.time())
        assert [(m.title, m.dedup_key) for m in fresh] == [
            ("New Flat", "ad:yaba|90000000")]
        assert again == []
        assert s.get(w.id).match_count == 1

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import nomorals.triggers.saved_search as ss

        path = tmp_path / "ss.db"
        native = ss.SavedSearchStore(str(path))
        w = native.create("property", "2bed Yaba", {"area": "Yaba"},
                          now=1000.0)
        native.mark_seen(w.id, "ad:abc|100", now=1500.0)
        native.mark_seen(w.id, "ad:def|200", now=1600.0)
        native.queue_digest(ss.Match(
            search_id=w.id, title="Flat A", found_at=5000.0))
        native.queue_digest(ss.Match(
            search_id=w.id, title="Flat B", found_at=6000.0))

        seam = ss.SavedSearchStore(str(path), store=rung.store)
        assert rung.store.count(
            "triggers-saved-search__saved_searches") == 1
        assert rung.store.count(
            "triggers-saved-search__seen_matches") == 2
        assert rung.store.count(
            "triggers-saved-search__pending_digest") == 2
        first = rung.store.get(
            "triggers-saved-search__pending_digest", "1")
        assert first is not None and first["title"] == "Flat A"
        # the seen registry answers by joined id on the seam
        assert seam.seen(w.id, "ad:abc|100") is True
        # seq continuity: the next seam-queued match takes seq 3
        seam.queue_digest(ss.Match(
            search_id=w.id, title="Flat C", found_at=7000.0))
        third = rung.store.get(
            "triggers-saved-search__pending_digest", "3")
        assert third is not None and third["title"] == "Flat C"
        counters = rung.store.get(
            "triggers-saved-search__meta", "counters")
        assert counters["pending_digest"] == 3
        taken = seam.take_digest(w.id)
        assert [m.title for m in taken] == ["Flat A", "Flat B", "Flat C"]
        again = ss.SavedSearchStore(str(path), store=rung.store)
        assert rung.store.count(
            "triggers-saved-search__saved_searches") == 1
        assert again.pending_digest_count(w.id) == 0


class TestHealthTimelinePort:
    """health/timeline.py — the health timeline port (residue
    Batch 2, health-timeline lane): a Database-class organ with a
    single table, so the store attaches to the timeline's own
    Database handle and health_events routes via attached_rows
    (collection carries the table name — the workspace-inbox
    pattern, not the shopping shared-file deviation: one module,
    one handle, one table, a name no other store owns). No
    claim/CAS flow exists in the module; log() is a put by id and
    the reads are finds + Python ordering/filtering."""

    @staticmethod
    def _noon(days_ago: float) -> float:
        """Local noon, ``days_ago`` days before today — a timestamp
        whose local calendar date is stable in every timezone."""
        lt = time.localtime(time.time() - days_ago * 86400.0)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                            12, 0, 0, 0, 0, -1))

    def _scenario(self, tl) -> dict:
        tl.log("symptom", "knee hurt today", severity=3,
               ts=self._noon(2))
        tl.log("measurement", "bp reading", value="120/80",
               unit="mmHg", ts=self._noon(1))
        tl.log("note", "clinic called", ts=self._noon(1) + 60,
               pharmacy="green cross")
        tl.log("symptom", "knee hurt again", severity=4,
               ts=self._noon(0))
        tl.log("mood", "feeling okay", severity=2, ts=self._noon(0) + 30)
        return {
            "timeline": [(e.event_type, e.text, e.severity)
                         for e in tl.timeline()],
            "symptoms": [(e.text, e.severity)
                         for e in tl.timeline(event_type="symptom")],
            "since": [e.text
                      for e in tl.timeline(since=self._noon(1))],
            "search": [e.text for e in tl.search("knee")],
            "search_case": [e.text for e in tl.search("KNEE")],
            "streak": tl.logging_streak(now=self._noon(0) + 120),
            "vitals": [(t.metric, t.latest) for t in tl.vitals_trend()],
            "stats": [(s.symptom, s.count, s.avg_severity)
                      for s in tl.symptom_stats()],
            "extra": [e.extra for e in tl.timeline(event_type="note")],
            "summary_has": "knee hurt again" in tl.summary(),
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.health.timeline import HealthTimeline

        tl = HealthTimeline(db_path=str(tmp_path / "timeline.db"),
                            store=rung.store)
        assert self._scenario(tl) == {
            "timeline": [("symptom", "knee hurt today", 3),
                         ("measurement", "bp reading", None),
                         ("note", "clinic called", None),
                         ("symptom", "knee hurt again", 4),
                         ("mood", "feeling okay", 2)],
            "symptoms": [("knee hurt today", 3),
                         ("knee hurt again", 4)],
            "since": ["bp reading", "clinic called",
                      "knee hurt again", "feeling okay"],
            "search": ["knee hurt again", "knee hurt today"],
            "search_case": ["knee hurt again", "knee hurt today"],
            "streak": 3,
            "vitals": [("blood_pressure_diastolic", 80.0),
                       ("blood_pressure_systolic", 120.0)],
            "stats": [("knee", 2, 3.5)],
            "extra": [{"pharmacy": "green cross"}],
            "summary_has": True,
        }

    def test_lazy_import_from_native_file(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.timeline import HealthTimeline

        path = str(tmp_path / "timeline.db")
        native = HealthTimeline(db_path=path)
        native.log("symptom", "knee hurt today", severity=3,
                   ts=self._noon(2))
        native.log("measurement", "bp reading", value="120/80",
                   unit="mmHg", ts=self._noon(1))
        native.log("visit", "saw the doctor", ts=self._noon(0))
        native.close()

        seam = HealthTimeline(db_path=path, store=rung.store)
        assert [e.text for e in seam.timeline()] == [
            "knee hurt today", "bp reading", "saw the doctor"]
        assert rung.store.count("health_events") == 3
        marker = rung.store.get("health-timeline__meta", "import")
        assert marker is not None and marker["done"] is True
        assert marker["tables"]["health_events"] == 3
        # logging continues on the seam after the import
        seam.log("note", "clinic called", ts=self._noon(0) + 60)
        assert rung.store.count("health_events") == 4
        # a second construction adopts — never re-imports
        again = HealthTimeline(db_path=path, store=rung.store)
        assert [e.text for e in again.timeline()] == [
            "knee hurt today", "bp reading", "saw the doctor",
            "clinic called"]
        assert rung.store.count("health_events") == 4

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.timeline import HealthTimeline

        tl = HealthTimeline(db_path=str(tmp_path / "timeline.db"),
                            store=rung.store)
        ev = tl.log("note", "clinic called", ts=self._noon(0),
                    pharmacy="green cross")
        tl.log("symptom", "knee hurt today", severity=3,
               ts=self._noon(0) + 30)
        assert rung.store.count("health_events") == 2
        doc = rung.store.get("health_events", ev.id)
        assert doc is not None and doc["_id"] == ev.id
        assert doc["event_type"] == "note"
        assert doc["text"] == "clinic called"
        # the extra payload rides as the same JSON string the
        # native INSERT writes — no data re-shaping on the seam
        assert doc["extra"] == '{"pharmacy": "green cross"}'


class TestImmuneCorpusPort:
    """immune/corpus.py — CorpusStore port (residue Batch 2, B10):
    the self-corpus's harvested exemplars as raw documents
    (immune-corpus__exemplars, the exemplar dict verbatim) behind
    the unchanged API, with corpus versions allocated by one
    atomic inc on the meta counters document — the lane's design
    point. corpus.py is L0: the composition layer (this test)
    builds the port and injects it — the module never imports
    storage. Intra-version admission order is source order
    natively and (harvested_ts, source_ref) order on seam
    reloads, so the scenario compares per-version sorted views.
    """

    @staticmethod
    def _port_for(path, rung):
        from nomorals.immune.corpus import _EXEMPLAR_SHAPE_DDL
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("immune-corpus", _EXEMPLAR_SHAPE_DDL, (),
                            None if path is None else str(path),
                            store=rung.store)

    @staticmethod
    def _plant_fix(journal_path, *, detail, subsystem="tools/web"):
        """One verified fix in a real IncidentJournal (the doc-20
        P4 suite's planting shape); returns the fix's signature —
        the citation harvest must carry."""
        from nomorals.core.incidents import IncidentJournal, RecoveryRecord

        journal = IncidentJournal(path=str(journal_path))
        try:
            incident = journal.record_incident(
                ConnectionError("flaky fetch refused"), subsystem=subsystem)
            journal.record_recovery(RecoveryRecord(
                incident_id=incident.id, signature=incident.signature,
                strategy="retry", detail=detail, verified=True,
                verify_note="re-ran the recovered call: passed"))
            return incident.signature
        finally:
            journal.close()

    @staticmethod
    def _receipt(receipt_id, target):
        return {"id": receipt_id, "action": "report_only",
                "target": target, "reason": "",
                "details": {"owner_directed": True}}

    @staticmethod
    def _harvested_view(store) -> list:
        """The harvested population as comparable tuples, sorted
        within each admission version (seeds excluded)."""
        rows = [
            (e.corpus_version, e.provenance, e.source_ref, e.text)
            for e in store.exemplars() if e.provenance != "seed"]
        return sorted(rows)

    def _scenario(self, store, journal_path) -> dict:
        sig1 = self._plant_fix(
            journal_path, detail="retry recovered the flaky web fetch")
        receipt = self._receipt("rcpt-1", "owner-laptop-tunnel")
        first = store.harvest(journal_path=journal_path,
                              receipts=[receipt])
        second = store.harvest(journal_path=journal_path,
                               receipts=[receipt])
        sig2 = self._plant_fix(
            journal_path, detail="patched the mail sync stall",
            subsystem="tools/mail")
        third = store.harvest(journal_path=journal_path, receipts=[])
        return {
            "first": (first["added"], first["version_before"],
                      first["version_after"],
                      first["sources"]["verified_fix"]["added"],
                      first["sources"]["owner_receipt"]["added"]),
            "second_added": second["added"],
            "third": (third["added"], third["version_after"]),
            "version": store.version,
            "harvested": self._harvested_view(store),
            "expect": sorted([
                (1, "verified_fix", f"verified_fix:{sig1}",
                 "retry recovered the flaky web fetch"),
                (1, "owner_receipt", "police_receipt:rcpt-1",
                 "owner-laptop-tunnel"),
                (2, "verified_fix", f"verified_fix:{sig2}",
                 "patched the mail sync stall")]),
            "counts": store.counts_by_provenance(),
            "at": (len(store.corpus_at(0).exemplars),
                   len(store.corpus_at(1).exemplars),
                   len(store.corpus_at(2).exemplars)),
            "degraded": store.degraded_reason,
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.immune.corpus import CorpusStore

        journal_path = tmp_path / "error_journal.db"
        corpus_path = tmp_path / "corpus.json"
        store = CorpusStore(corpus_path,
                            port=self._port_for(corpus_path, rung))
        observed = self._scenario(store, journal_path)
        assert observed["first"] == (2, 0, 1, 1, 1)
        assert observed["second_added"] == 0
        assert observed["third"] == (1, 2)
        assert observed["version"] == 2
        assert observed["harvested"] == observed["expect"]
        assert observed["counts"] == {
            "seed": 8, "verified_fix": 2, "owner_receipt": 1}
        assert observed["at"] == (8, 10, 11)
        assert observed["degraded"] == ""

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.immune.corpus import CorpusStore

        path = tmp_path / "corpus.json"
        journal_path = tmp_path / "error_journal.db"
        sig1 = self._plant_fix(
            journal_path, detail="first verified recovery")
        native = CorpusStore(path)
        native.harvest(journal_path=journal_path, receipts=[])
        self._plant_fix(journal_path, detail="second verified recovery",
                        subsystem="tools/mail")
        native.harvest(journal_path=journal_path, receipts=[])
        assert native.version == 2

        seam = CorpusStore(path, port=self._port_for(path, rung))
        assert seam.version == 2
        assert rung.store.count("immune-corpus__exemplars") == 2
        marker = rung.store.get("immune-corpus__meta", "import")
        assert marker is not None and marker["done"] is True
        assert marker["tables"] == {"exemplars": 2}
        counters = rung.store.get("immune-corpus__meta", "counters")
        assert counters is not None and counters["corpus_version"] == 2
        by_ref = {e.source_ref: e for e in seam.exemplars()}
        doc = rung.store.get(
            "immune-corpus__exemplars", by_ref[f"verified_fix:{sig1}"].id)
        assert doc is not None and doc["corpus_version"] == 1
        assert doc["provenance"] == "verified_fix"
        assert doc["source_ref"] == f"verified_fix:{sig1}"
        # the counter was seeded at the file's version: the next
        # allocation is 3 — the imported max + 1, never a reuse
        self._plant_fix(journal_path, detail="third verified recovery",
                        subsystem="tools/calendar")
        report = seam.harvest(journal_path=journal_path, receipts=[])
        assert (report["version_before"], report["version_after"]) == (2, 3)
        assert rung.store.count("immune-corpus__exemplars") == 3
        # second construction adopts: no re-import, no duplicates
        again = CorpusStore(path, port=self._port_for(path, rung))
        assert again.version == 3
        assert len(again.exemplars()) == 11
        assert rung.store.count("immune-corpus__exemplars") == 3
        # a path-less seam store imports nothing of its own — it
        # adopts the live seam state and still works
        mem = CorpusStore(port=self._port_for(None, rung))
        assert mem.version == 3
        assert len(mem.exemplars()) == 11

    def test_atomic_version_allocation(self, rung, tmp_path):
        """The B10 design point: two CorpusStore instances over
        one backend, interleaved harvests — versions are distinct
        and monotone, an exemplar admitted by one instance is
        never re-admitted by the other, and a fresh instance
        reads the shared truth."""
        if rung.store is None:
            pytest.skip("native rung has no seam to allocate on")
        from nomorals.immune.corpus import CorpusStore

        journal_path = tmp_path / "error_journal.db"
        path = tmp_path / "corpus.json"
        alpha = CorpusStore(path, port=self._port_for(path, rung))
        beta = CorpusStore(path, port=self._port_for(path, rung))
        self._plant_fix(journal_path, detail="alpha's verified recovery")
        ra = alpha.harvest(journal_path=journal_path, receipts=[])
        self._plant_fix(journal_path, detail="beta's verified recovery",
                        subsystem="tools/mail")
        rb = beta.harvest(journal_path=journal_path, receipts=[])
        self._plant_fix(journal_path, detail="alpha again, third recovery",
                        subsystem="tools/calendar")
        ra2 = alpha.harvest(journal_path=journal_path, receipts=[])
        assert [ra["version_after"], rb["version_after"],
                ra2["version_after"]] == [1, 2, 3]
        # beta's harvest deduped against alpha's admission (the
        # reload at harvest start): each fix admitted exactly once
        assert (ra["added"], rb["added"], ra2["added"]) == (1, 1, 1)
        assert rung.store.count("immune-corpus__exemplars") == 3
        admitted = sorted(
            doc["corpus_version"]
            for doc in rung.store.find("immune-corpus__exemplars"))
        assert admitted == [1, 2, 3]
        fresh = CorpusStore(path, port=self._port_for(path, rung))
        assert fresh.version == 3
        assert len(fresh.corpus_at(1).exemplars) == 9
        assert len(fresh.corpus_at(2).exemplars) == 10
        assert len(fresh.exemplars()) == 11

    def test_seam_import_of_corrupt_file_degrades(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.immune.corpus import CorpusStore

        path = tmp_path / "corpus.json"
        path.write_text("{ not json at all", encoding="utf-8")
        seam = CorpusStore(path, port=self._port_for(path, rung))
        assert seam.version == 0
        assert len(seam.exemplars()) == 8
        assert "seed-only" in seam.degraded_reason
        assert rung.store.count("immune-corpus__exemplars") == 0


class TestIdentityDocumentsPort:
    NOW = 1_800_000_000.0  # fixed reference time for expiry math
    DAY = 86400.0

    @staticmethod
    def _port_for(path, rung):
        # identity/ is layer L0: the composition layer (this test)
        # builds the port and injects it — the module never imports
        # storage.
        from nomorals.identity.documents import _DDL
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("identity-documents", _DDL,
                            ("identity_documents",),
                            None if path is None else str(path),
                            store=rung.store)

    @classmethod
    def _doc(cls, doc_id, expiry, doc_type="passport",
             owner_id="owner"):
        from nomorals.identity.documents import IdentityDocument

        return IdentityDocument(
            id=doc_id, owner_id=owner_id, doc_type=doc_type,
            issuing_country="NG",
            number_ref=f"vault:identity/{doc_id}/1",
            expiry_ts=expiry, issued_ts=cls.NOW - 365 * cls.DAY,
            notes="")

    def _scenario(self, tracker) -> dict:
        from nomorals.identity.documents import IdentityDocument

        tracker.add(self._doc("soon", self.NOW + 10 * self.DAY),
                    owner=True)
        tracker.add(self._doc("mid", self.NOW + 45 * self.DAY,
                              doc_type="visa", owner_id="other"),
                    owner=True)
        tracker.add(self._doc("far", self.NOW + 400 * self.DAY),
                    owner=True)
        tracker.add(self._doc("old", self.NOW - 20 * self.DAY),
                    owner=True)
        got = tracker.get("soon", owner=True)
        duplicate = False
        try:
            tracker.add(self._doc("soon", self.NOW + 10 * self.DAY),
                        owner=True)
        except ValueError:
            duplicate = True
        denied = False
        try:
            tracker.get("soon")  # owner defaults False → denied
        except PermissionError:
            denied = True
        raw_rejected = False
        try:
            IdentityDocument(
                id="raw", owner_id="owner", doc_type="passport",
                issuing_country="NG", number_ref="A12345678",
                expiry_ts=self.NOW + self.DAY)
        except ValueError:
            raw_rejected = True
        checklist = tracker.renewal_checklist("far", owner=True)
        return {
            "got": (got.doc_type, got.number_ref, got.expiry_ts),
            "list": [d.id for d in tracker.list(owner=True)],
            "list_owner": [d.id for d in tracker.list(
                "other", owner=True)],
            "within_30": [d.id for d in tracker.expiring_within(
                30, now=self.NOW, owner=True)],
            "within_60": [d.id for d in tracker.expiring_within(
                60, now=self.NOW, owner=True)],
            "overdue": [d.id for d in tracker.overdue(
                now=self.NOW, owner=True)],
            "checklist_orders": [s["order"] for s in checklist],
            "missing": tracker.get("nope", owner=True),
            "duplicate": duplicate,
            "denied": denied,
            "raw_rejected": raw_rejected,
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.identity.documents import IdentityTracker

        path = tmp_path / "identity.db"
        tracker = IdentityTracker(str(path),
                                  port=self._port_for(path, rung))
        observed = self._scenario(tracker)
        assert observed["got"] == (
            "passport", "vault:identity/soon/1",
            self.NOW + 10 * self.DAY)
        assert observed["list"] == ["old", "soon", "mid", "far"]
        assert observed["list_owner"] == ["mid"]
        assert observed["within_30"] == ["old", "soon"]
        assert observed["within_60"] == ["old", "soon", "mid"]
        assert observed["overdue"] == ["old"]
        assert observed["checklist_orders"] == [1, 2, 3, 4]
        assert observed["missing"] is None
        assert observed["duplicate"] is True
        assert observed["denied"] is True
        assert observed["raw_rejected"] is True
        tracker.close()

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.identity.documents import IdentityTracker

        path = tmp_path / "identity.db"
        native = IdentityTracker(str(path))
        native.add(self._doc("kept", self.NOW + 90 * self.DAY),
                   owner=True)
        native.close()

        tracker = IdentityTracker(str(path),
                                  port=self._port_for(path, rung))
        got = tracker.get("kept", owner=True)
        assert got is not None
        assert got.number_ref == "vault:identity/kept/1"
        assert rung.store.count(
            "identity-documents__identity_documents") == 1
        doc = rung.store.get(
            "identity-documents__identity_documents", "kept")
        assert doc is not None and doc["_id"] == "kept"
        # the vault review's claim, pinned mechanically: the stored
        # document carries exactly the table's non-PK columns plus
        # the store's managed fields — the vault reference rides
        # verbatim and no raw-number field exists on the seam,
        # exactly as in the native file.
        assert doc["number_ref"] == "vault:identity/kept/1"
        assert set(doc) == {
            "_id", "owner_id", "doc_type", "issuing_country",
            "number_ref", "expiry_ts", "issued_ts", "notes",
            "expires_at"}
        # the document's own expiry_ts is DATA, never the store's
        # managed TTL: the seam never invents an expires_at, so an
        # overdue document stays readable on the seam exactly as
        # it does in the native file.
        assert doc["expires_at"] is None
        again = IdentityTracker(str(path),
                                port=self._port_for(path, rung))
        assert [d.id for d in again.list(owner=True)] == ["kept"]
        assert rung.store.count(
            "identity-documents__identity_documents") == 1
        # a ':memory:' tracker on the seam imports nothing, still works
        mem = IdentityTracker(port=self._port_for(None, rung))
        mem.add(self._doc("mem", self.NOW + 5 * self.DAY), owner=True)
        assert mem.get("mem", owner=True).number_ref == \
            "vault:identity/mem/1"


class TestHealthChallengesPort:
    """health/challenges.py — the health-challenges port (residue
    Batch 2, B28, its own lane): all eight tables are primary-keyed
    (challenge_members, completions and challenge_pbs compositely),
    so every one routes through a RowTable; the composite key parts
    live only in the joined document id. The registry reserved this
    store a domain pass of its own, so the scenario below is the
    full streak lifecycle, not per-table CRUD: create → join →
    workouts to completion (with personal bests and the board) →
    daily check-ins through checkin's own now= seam → the 7-day
    freeze milestone → a missed day the freeze covers → a missed
    day that resets → at-risk reads — identical observables on
    every rung."""

    @staticmethod
    def _noon(days_ago: float) -> float:
        """Local noon, ``days_ago`` days before today — a timestamp
        whose local calendar date is stable in every timezone."""
        lt = time.localtime(time.time() - days_ago * 86400.0)
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                            12, 0, 0, 0, 0, -1))

    def _scenario(self, cs) -> dict:
        day = 86400.0
        base = self._noon(0)
        c = cs.create_challenge("Morning Miles", "workout-count", 30, 3,
                                proof_method="honor", created_by="amy")
        joined = (cs.join_challenge(c.id, "amy"),
                  cs.join_challenge(c.id, "bob"))
        logs = [cs.log_workout(c.id, "amy") for _ in range(3)]
        proof = cs.submit_proof(c.id, "bob", "")
        monthly = cs.monthly_challenge("distance", created_by="amy")
        squad = cs.create_squad("Dawn Patrol", ["amy", "bob", "cy"])
        # the streak lifecycle, driven through checkin's now= seam
        amy_week = [cs.checkin("amy", now=base + i * day)
                    for i in range(7)]
        amy = {
            "week": amy_week,
            "freezes_after_week": cs.freezes("amy"),
            "same_day": cs.checkin("amy", now=base + 6 * day + 3600),
            "gap_covered": cs.checkin("amy", now=base + 8 * day),
            "freezes_after_gap": cs.freezes("amy"),
            "reset": cs.checkin("amy", now=base + 10 * day),
        }
        bob_seq = [cs.checkin("bob", now=base + i * day)
                   for i in (0, 1, 2, 4, 5, 6)]
        got = cs.get(c.id)
        return {
            "created": (c.name, c.type, c.target, c.duration_days,
                        c.proof_method, c.created_by),
            "got": (got.name, got.target, got.badge == f"challenge:{c.id}"),
            "joined": joined,
            "members": sorted(cs.members(c.id)),
            "logs": logs,
            "progress": (cs.progress(c.id, "amy"),
                         cs.progress(c.id, "bob")),
            "pb_amy": cs.personal_best(c.id, "amy"),
            "board": cs.leaderboard(c.id),
            "proof": (proof.verified, proof.verdict),
            "monthly": (monthly.badge,
                        cs.get(monthly.id).badge == monthly.badge),
            "active_names": sorted(
                ch.name for ch in cs.list_challenges()),
            "squad": (squad.name, squad.members),
            "squad_board": cs.squad_leaderboard(squad.id, c.id),
            "amy": amy,
            "bob_seq": bob_seq,
            "streaks": (cs.streak("amy"), cs.streak("bob")),
            "at_risk": cs.at_risk(now=base + 7 * day),
            "nudge_bob": cs.risk_nudge("bob"),
            "nudge_amy": cs.risk_nudge("amy"),
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        from nomorals.health.challenges import ChallengeStore

        cs = ChallengeStore(db_path=str(tmp_path / "challenges.db"),
                            store=rung.store)
        month = time.strftime("%B")
        assert self._scenario(cs) == {
            "created": ("Morning Miles", "workout-count", 3, 30,
                        "honor", "amy"),
            "got": ("Morning Miles", 3, True),
            "joined": (True, True),
            "members": ["amy", "bob"],
            "logs": [1, 2, 3],
            "progress": ((3, 3), (1, 3)),
            "pb_amy": 3,
            "board": [("amy", 3, True), ("bob", 1, False)],
            "proof": (True, "honor system — trusted "
                             "(workout #1 logged)"),
            "monthly": (f"monthly:{time.strftime('%Y-%m')}:distance",
                        True),
            "active_names": sorted(["Morning Miles",
                                    f"{month} Distance Challenge"]),
            "squad": ("Dawn Patrol", ["amy", "bob", "cy"]),
            "squad_board": [("amy", 3), ("bob", 1), ("cy", 0)],
            "amy": {"week": [1, 2, 3, 4, 5, 6, 7],
                    "freezes_after_week": 1,
                    "same_day": 7,
                    "gap_covered": 8,
                    "freezes_after_gap": 0,
                    "reset": 1},
            "bob_seq": [1, 2, 3, 1, 2, 3],
            "streaks": (1, 3),
            "at_risk": [("bob", 3)],
            "nudge_bob": ("🔥 bob: your 3-day streak is at risk! "
                          "Check in today to keep it alive."),
            "nudge_amy": "",
        }

    def test_lazy_import_from_native_file(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.challenges import ChallengeStore

        path = str(tmp_path / "challenges.db")
        day = 86400.0
        base = self._noon(0)
        native = ChallengeStore(db_path=path)
        c = native.create_challenge("Imported Miles", "workout-count",
                                    30, 2, proof_method="honor")
        native.join_challenge(c.id, "amy")
        native.join_challenge(c.id, "bob")
        native.log_workout(c.id, "amy")
        for i in range(7):
            native.checkin("amy", now=base + i * day)
        squad = native.create_squad("Import Squad", ["amy", "bob", "cy"])
        assert native.streak("amy") == 7 and native.freezes("amy") == 1

        seam = ChallengeStore(db_path=path, store=rung.store)
        assert seam.streak("amy") == 7
        assert seam.freezes("amy") == 1
        assert seam.progress(c.id, "amy") == (1, 2)
        assert sorted(seam.members(c.id)) == ["amy", "bob"]
        assert seam.get_squad(squad.id).members == ["amy", "bob", "cy"]
        counts = {t: rung.store.count(f"health-challenges__{t}")
                  for t in ("challenges", "challenge_members", "proofs",
                            "completions", "streaks", "squads",
                            "streak_freezes", "challenge_pbs")}
        assert counts == {"challenges": 1, "challenge_members": 2,
                          "proofs": 0, "completions": 2, "streaks": 1,
                          "squads": 1, "streak_freezes": 1,
                          "challenge_pbs": 1}
        marker = rung.store.get("health-challenges__meta", "import")
        assert marker is not None and marker["done"] is True
        assert marker["tables"]["streaks"] == 1
        # the lifecycle continues across the import boundary: the
        # imported freeze covers the missed day 7, exactly as the
        # native store would have spent it
        assert seam.checkin("amy", now=base + 8 * day) == 8
        assert seam.freezes("amy") == 0
        # a second construction adopts — never re-imports
        again = ChallengeStore(db_path=path, store=rung.store)
        assert again.streak("amy") == 8
        assert rung.store.count("health-challenges__streaks") == 1

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.health.challenges import ChallengeStore

        cs = ChallengeStore(db_path=str(tmp_path / "challenges.db"),
                            store=rung.store)
        c = cs.create_challenge("Shape Check", "workout-count", 30, 2,
                                proof_method="honor")
        cs.join_challenge(c.id, "amy")
        cs.log_workout(c.id, "amy")
        cs.submit_proof(c.id, "amy", "")
        cs.checkin("amy", now=self._noon(0))
        assert rung.store.count("health-challenges__challenges") == 1
        assert rung.store.count("health-challenges__completions") == 1
        assert rung.store.count("health-challenges__proofs") == 1
        assert rung.store.count("health-challenges__streaks") == 1
        doc = rung.store.get("health-challenges__challenges", c.id)
        assert doc is not None and doc["_id"] == c.id
        assert doc["name"] == "Shape Check" and doc["ctype"] == "workout-count"
        # composite-keyed rows: the key parts live only in the joined
        # document id, never in the body
        comp = rung.store.get("health-challenges__completions",
                              f"{c.id}\x1famy")
        assert comp is not None and comp["_id"] == f"{c.id}\x1famy"
        assert comp["count"] == 2 and comp["completed"] == 1
        assert "challenge_id" not in comp and "member" not in comp
        proof = rung.store.find("health-challenges__proofs")[0]
        assert proof["verdict"] == "honor system — trusted"
        assert proof["verified"] == 1


class TestPeoplePort:
    """people/store.py — the people port (residue Batch 3, C4):
    persons are documents in people__persons; the four
    AUTOINCREMENT children ride put_raw under caller-shaped ids
    str(next_id(<table>)), seeded by the lazy import, and child
    writes return int(doc_id) — the public int-id contract. The
    module's only JOIN (search's LEFT JOIN notes) re-expresses as
    a find-all + notes grouped by person_id with literal substring
    match. No FK layer: the module never deletes and every child
    write ensures the person first, on both rungs. The scenario is
    the rollup-equivalence bar: one fixture population, identical
    get / top_contacts / needs_reconnect / search / gifts_for /
    obligations_for / links_for observables on every rung, all
    interaction time driven through the methods' own at= seams."""

    T0 = 1_700_000_000.0

    @staticmethod
    def _port_for(path, rung):
        # people/store.py is L0: the composition layer (this test)
        # builds the port and injects it — the module never imports
        # storage.
        from nomorals.people.store import _SCHEMA
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("people", _SCHEMA, ("persons",),
                            str(path), store=rung.store)

    @staticmethod
    def _view(person) -> dict:
        # The cross-rung person view: every field the interaction
        # at= seams determine. created_at / updated_at / first_seen
        # are excluded — child writes (add_note & friends) ensure
        # with the wall clock. warmth is rounded to 6 places: the
        # EMA is unrounded in the store (0.6000000000000001), the
        # rounding here is the view's, not the data's.
        return {
            "person_id": person["person_id"],
            "display_name": person["display_name"],
            "closeness": person["closeness"],
            "warmth": round(person["warmth"], 6),
            "inbound_count": person["inbound_count"],
            "outbound_count": person["outbound_count"],
            "owner_initiated_count": person["owner_initiated_count"],
            "last_inbound": person["last_inbound"],
            "last_outbound": person["last_outbound"],
            "last_topic": person["last_topic"],
            "notes": person["notes"],
        }

    def _scenario(self, store) -> dict:
        t0 = self.T0
        store.ensure("ada", "Ada Lovelace", at=t0)
        store.record_inbound("ada", topic="interview prep", warm=True,
                             at=t0 + 60)
        store.record_outbound("ada", owner_initiated=True, at=t0 + 120)
        store.record_outbound("ada", owner_initiated=True, at=t0 + 180)
        note_ids = (
            store.add_note("ada", "Birthday in June"),
            store.add_note("ada", "Scored 100% on review"),
            store.add_note("ada", "Birthday in June"),  # duplicate
        )
        gift_id = store.add_gift(
            "ada", "Book", "given", value_minor=2500, currency="NGN",
            date="2026-10-01")
        obligation_ids = (
            store.link_obligation("ada", "c-1", "Send the draft",
                                  deadline="2026-10-20"),
            store.link_obligation("ada", "c-1", "Send the draft"),
            store.link_obligation("ada", None, "Water the plants"),
        )
        link_ids = (
            store.add_link("ada", "grace", "colleague"),
            store.add_link("ada", "grace", "colleague"),  # duplicate
        )
        store.ensure("grace", "Grace Hopper", at=t0 + 200)
        grace_note = store.add_note("grace", "Loves chess")
        # closeness tie: identical interactions, person_id breaks it
        store.record_outbound("mia", owner_initiated=True, at=t0 + 30)
        store.record_outbound("nia", owner_initiated=True, at=t0 + 30)
        for _ in range(5):
            store.record_outbound("old", owner_initiated=True,
                                  at=t0 - 40 * 86400)
        store.record_inbound("stranger", at=t0 - 40 * 86400)
        # orphan pin: a child write for an unknown person creates them
        orphan_note = store.add_note("newbie", "First fact")
        record = {
            "person_id": "imported", "display_name": "Imported Person",
            "closeness": 0.25, "warmth": 0.7, "inbound_count": 3,
            "outbound_count": 1, "owner_initiated_count": 1,
            "first_seen": t0 - 1000, "last_inbound": t0 - 500,
            "last_outbound": 0.0, "last_topic": "migration",
            "notes": ["Migrated note"],
        }
        imported = (store.import_record(record),
                    store.import_record(record))
        return {
            "ids": {"notes": note_ids, "gift": gift_id,
                    "obligations": obligation_ids, "links": link_ids,
                    "grace_note": grace_note, "orphan_note": orphan_note},
            "imported_flags": imported,
            "ada": self._view(store.get("ada")),
            "grace": self._view(store.get("grace")),
            "newbie": self._view(store.get("newbie")),
            "imported": self._view(store.get("imported")),
            "missing": store.get("nobody"),
            "top": [p["person_id"] for p in store.top_contacts()],
            "top2": [p["person_id"] for p in store.top_contacts(limit=2)],
            "reconnect": [
                p["person_id"]
                for p in store.needs_reconnect(days=30, now=t0)],
            "search": {
                q: [p["person_id"] for p in store.search(q)]
                for q in ("lovelace", "interview", "chess", "100%",
                          "birthday", "zzz", "a")
            },
            "gifts": [
                (g["id"], g["description"], g["direction"],
                 g["value_minor"], g["currency"], g["date"])
                for g in store.gifts_for("ada")],
            "obligations": [
                (o["id"], o["commitment_id"], o["what"], o["deadline"],
                 o["status"])
                for o in store.obligations_for("ada")],
            "links_ada": [
                (link["id"], link["from_person"], link["to_person"],
                 link["kind"])
                for link in store.links_for("ada")],
            "links_grace": [
                (link["id"], link["from_person"], link["to_person"],
                 link["kind"])
                for link in store.links_for("grace")],
            "gifts_grace": store.gifts_for("grace"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.people import PersonStore

        path = tmp_path / "people.db"
        store = PersonStore(path, port=self._port_for(path, rung))
        try:
            observed = self._scenario(store)
        finally:
            store.close()
        t0 = self.T0
        assert observed == {
            "ids": {"notes": (1, 2, 1), "gift": 1,
                    "obligations": (1, 1, 2), "links": (1, 1),
                    "grace_note": 3, "orphan_note": 4},
            "imported_flags": (True, False),
            "ada": {
                "person_id": "ada", "display_name": "Ada Lovelace",
                "closeness": 0.49, "warmth": 0.6, "inbound_count": 1,
                "outbound_count": 2, "owner_initiated_count": 2,
                "last_inbound": t0 + 60, "last_outbound": t0 + 180,
                "last_topic": "interview prep",
                "notes": ["Birthday in June", "Scored 100% on review"],
            },
            "grace": {
                "person_id": "grace", "display_name": "Grace Hopper",
                "closeness": 0.0, "warmth": 0.5, "inbound_count": 0,
                "outbound_count": 0, "owner_initiated_count": 0,
                "last_inbound": 0.0, "last_outbound": 0.0,
                "last_topic": "", "notes": ["Loves chess"],
            },
            "newbie": {
                "person_id": "newbie", "display_name": "",
                "closeness": 0.0, "warmth": 0.5, "inbound_count": 0,
                "outbound_count": 0, "owner_initiated_count": 0,
                "last_inbound": 0.0, "last_outbound": 0.0,
                "last_topic": "", "notes": ["First fact"],
            },
            "imported": {
                "person_id": "imported",
                "display_name": "Imported Person",
                "closeness": 0.25, "warmth": 0.7, "inbound_count": 3,
                "outbound_count": 1, "owner_initiated_count": 1,
                "last_inbound": t0 - 500, "last_outbound": 0.0,
                "last_topic": "migration", "notes": ["Migrated note"],
            },
            "missing": None,
            "top": ["old", "ada", "mia", "nia", "imported", "stranger",
                    "grace", "newbie"],
            "top2": ["old", "ada"],
            "reconnect": ["old"],
            "search": {
                "lovelace": ["ada"], "interview": ["ada"],
                "chess": ["grace"], "100%": ["ada"],
                "birthday": ["ada"], "zzz": [],
                "a": ["ada", "mia", "nia", "imported", "stranger",
                      "grace", "newbie"],
            },
            "gifts": [(1, "Book", "given", 2500, "NGN", "2026-10-01")],
            "obligations": [
                (1, "c-1", "Send the draft", "2026-10-20", "open"),
                (2, None, "Water the plants", "", "open"),
            ],
            "links_ada": [(1, "ada", "grace", "colleague")],
            "links_grace": [(1, "ada", "grace", "colleague")],
            "gifts_grace": [],
        }

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.people import PersonStore

        path = tmp_path / "people.db"
        native = PersonStore(path)
        native.ensure("ada", "Ada Lovelace")
        for text in ("n1", "n2", "n3"):
            native.add_note("ada", text)
        native.add_gift("ada", "Book", "given", value_minor=100)
        native.close()

        store = PersonStore(path, port=self._port_for(path, rung))
        try:
            # the file's history is the seam's history
            assert store.get("ada")["notes"] == ["n1", "n2", "n3"]
            assert [g["id"] for g in store.gifts_for("ada")] == [1]
            # int-id continuity: imported notes 1..3, the next
            # add_note returns 4 — the file's sequence, continued
            assert store.add_note("ada", "n4") == 4
            assert store.add_gift("ada", "Pen", "received") == 2
            assert rung.store.count("people__notes") == 4
            assert rung.store.count("people__persons") == 1
            marker = rung.store.get("people__meta", "import")
            assert marker is not None and marker["done"] is True
            assert marker["tables"]["notes"] == 3
            counters = rung.store.get("people__meta", "counters")
            assert counters["notes"] == 4 and counters["gifts"] == 2
            # the person key lives only in the document id
            doc = rung.store.get("people__persons", "ada")
            assert doc is not None and doc["_id"] == "ada"
            assert "person_id" not in doc
            # a second construction adopts — never re-imports
            again = PersonStore(path, port=self._port_for(path, rung))
            try:
                assert again.get("ada")["notes"] == ["n1", "n2", "n3", "n4"]
                assert again.add_note("ada", "n5") == 5
            finally:
                again.close()
            assert rung.store.count("people__notes") == 5
        finally:
            store.close()


class TestHouseholdPort:
    """household/household.py — the household port (residue Batch 3,
    C5): all four tables are TEXT-PK, so every one routes through a
    RowTable — mechanically the plainest port in the batch. The
    design content is the write composition: natively every mutation
    is two sequential commits (the record INSERT, then the audit
    INSERT), never a transaction, so an audit failure propagates
    with the record already durable — and delegated member ops
    audit only after the family registry call returns. The seam
    preserves that composition verbatim (record document first,
    audit document second, no atomicity invented), and the pins
    below include the failure injection that proves it on every
    rung. Membership itself stays in family; only the shared
    records move. household/ is layer L0, so the port is INJECTED:
    the composition layer (here, this test) builds it via
    sidecar_port and hands it to the constructor."""

    @staticmethod
    def _port_for(tmp_path, rung):
        # household/ is layer L0: the composition layer (this test)
        # builds the port and injects it — the module never imports
        # storage.
        from nomorals.household.household import _DDL, _TABLES
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("household", _DDL, _TABLES,
                            str(tmp_path / "household_shared.db"),
                            store=rung.store)

    @staticmethod
    def _registry(tmp_path):
        from nomorals.family.registry import family_registry

        registry = family_registry(str(tmp_path))
        # Bootstrap: first member must be an owner.
        registry.call("family_person_add", actor_id="owner1",
                      actor_role="owner", person_id="owner1",
                      display_name="Owner", role="owner")
        registry.call("family_person_add", actor_id="owner1",
                      actor_role="owner", person_id="adult1",
                      display_name="Adult", role="adult")
        registry.call("family_person_add", actor_id="owner1",
                      actor_role="owner", person_id="teen1",
                      display_name="Teen", role="teen")
        return registry

    @classmethod
    def _household(cls, registry, tmp_path, rung=None):
        from nomorals.household.household import Household

        if rung is None or rung.store is None:
            return Household(registry=registry)
        return Household(registry=registry,
                         port=cls._port_for(tmp_path, rung))

    def _scenario(self, hh) -> dict:
        commitment = hh.add_shared_commitment(
            "owner1", "Pay the rent", owner_person="adult1",
            due=1_800_000_000.0)
        cal = hh.add_shared_calendar(
            "adult1", "Dentist", 1_800_000_100.0, person_id="teen1")
        asset = hh.add_shared_asset("teen1", "Board game")
        member = hh.add_member(
            actor_id="owner1", actor_role="owner", person_id="child1",
            display_name="Kid", role="child")
        hh.revoke_member(actor_id="owner1", actor_role="owner",
                         person_id="teen1", reason="moved out")
        log = hh.audit_log("owner1")
        return {
            "ids": (commitment["id"], cal["id"], asset["id"]),
            "commitment": (commitment["what"],
                           commitment["owner_person"],
                           commitment["due"], commitment["status"]),
            "calendar": (cal["title"], cal["ts"], cal["person_id"]),
            "asset": (asset["name"], asset["owner_person"]),
            "member": (member["person_id"], member["role"]),
            "commitments": [c["what"] for c in
                            hh.list_shared_commitments("adult1")],
            "calendar_list": [c["title"] for c in
                              hh.list_shared_calendar("adult1")],
            "assets": [a["name"] for a in
                       hh.list_shared_assets("adult1")],
            "audit": [(e["action"], e["target"]) for e in log],
            "audit_details": [e["detail"] for e in log],
        }

    def test_scenario_matches_across_rungs(self, rung, tmp_path):
        hh = self._household(self._registry(tmp_path),
                             tmp_path, rung)
        got = self._scenario(hh)
        ids = got.pop("ids")
        assert got.pop("audit") == [
            ("shared_commitment_add", ids[0]),
            ("shared_calendar_add", ids[1]),
            ("shared_asset_add", ids[2]),
            ("person_add", "child1"),
            ("person_revoke", "teen1"),
        ]
        assert got == {
            "commitment": ("Pay the rent", "adult1",
                           1_800_000_000.0, "open"),
            "calendar": ("Dentist", 1_800_000_100.0, "teen1"),
            "asset": ("Board game", "teen1"),
            "member": ("child1", "child"),
            "commitments": ["Pay the rent"],
            "calendar_list": ["Dentist"],
            "assets": ["Board game"],
            "audit_details": ["Pay the rent", "Dentist", "Board game",
                              "role=child", "moved out"],
        }

    def test_audit_failure_leaves_record_durable(self, rung, tmp_path):
        """The composition pin: forcing the audit write to raise —
        injected at the organ's own _audit seam, identically on
        every rung — leaves the record durable, surfaces the
        exception, and leaves zero audit entries. A port that made
        the pair atomic would fail this on the seam rungs."""
        hh = self._household(self._registry(tmp_path),
                             tmp_path, rung)

        def _boom(*args, **kwargs):
            raise RuntimeError("audit write unavailable")

        original = hh._audit
        hh._audit = _boom
        try:
            with pytest.raises(RuntimeError):
                hh.add_shared_commitment("owner1", "Pay the rent")
        finally:
            hh._audit = original
        assert [c["what"] for c in
                hh.list_shared_commitments("owner1")] == ["Pay the rent"]
        assert hh.audit_log("owner1") == []

    def test_failed_family_call_leaves_no_audit(self, rung, tmp_path):
        """The delegation pin: audit is written only after the
        family registry call returns — revoking the last owner
        fails inside family (a ValueError), so nothing is logged;
        a successful delegated add then audits exactly once."""
        hh = self._household(self._registry(tmp_path),
                             tmp_path, rung)
        with pytest.raises(ValueError):
            hh.revoke_member(actor_id="owner1", actor_role="owner",
                             person_id="owner1")
        assert hh.audit_log("owner1") == []
        hh.add_member(actor_id="owner1", actor_role="owner",
                      person_id="child1", display_name="Kid",
                      role="child")
        log = hh.audit_log("owner1")
        assert [(e["action"], e["target"]) for e in log] == [
            ("person_add", "child1")]

    def test_lazy_import_from_native_file(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        registry = self._registry(tmp_path)
        native = self._household(registry, tmp_path)
        commitment = native.add_shared_commitment(
            "owner1", "Imported rent", owner_person="adult1")
        asset = native.add_shared_asset("owner1", "Imported car")
        native.add_shared_calendar("adult1", "Imported dentist",
                                   1_800_000_200.0, person_id="teen1")
        native.close()

        seam = self._household(registry, tmp_path, rung)
        assert [c["what"] for c in
                seam.list_shared_commitments("owner1")] == [
                    "Imported rent"]
        assert [a["name"] for a in
                seam.list_shared_assets("owner1")] == ["Imported car"]
        assert [(e["action"], e["target"]) for e in
                seam.audit_log("owner1")] == [
                    ("shared_commitment_add", commitment["id"]),
                    ("shared_asset_add", asset["id"]),
                    ("shared_calendar_add",
                     seam.list_shared_calendar("owner1")[0]["id"])]
        counts = {t: rung.store.count(f"household__{t}")
                  for t in ("shared_assets", "shared_calendar",
                            "shared_commitments", "household_audit")}
        assert counts == {"shared_assets": 1, "shared_calendar": 1,
                          "shared_commitments": 1, "household_audit": 3}
        marker = rung.store.get("household__meta", "import")
        assert marker is not None and marker["done"] is True
        assert marker["tables"]["shared_commitments"] == 1
        # a second construction adopts — never re-imports
        again = self._household(registry, tmp_path, rung)
        assert [c["what"] for c in
                again.list_shared_commitments("owner1")] == [
                    "Imported rent"]
        assert rung.store.count("household__household_audit") == 3

    def test_seam_shape(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        hh = self._household(self._registry(tmp_path),
                             tmp_path, rung)
        entry = hh.add_shared_commitment(
            "owner1", "Shape rent", owner_person="adult1",
            due=1_800_000_000.0)
        assert rung.store.count("household__shared_commitments") == 1
        doc = rung.store.get("household__shared_commitments",
                             entry["id"])
        assert doc is not None and doc["_id"] == entry["id"]
        assert doc["what"] == "Shape rent"
        assert doc["owner_person"] == "adult1"
        # the audit document is a separate write in its own
        # collection, targeting the record's id
        assert rung.store.count("household__household_audit") == 1
        audit = rung.store.find("household__household_audit")[0]
        assert audit["action"] == "shared_commitment_add"
        assert audit["target"] == entry["id"]


class TestPlanningGraphPort:
    """planning/graph.py — the planning-graph port (residue Batch 3,
    C8): the HOLD this store carried is discharged by evidence —
    the module has no SQL traversals (no WITH RECURSIVE, no JOINs),
    so nodes and edges are documents and every analysis
    (dependents/_propagate BFS, critical_path DFS, find_cycles,
    path_between, neighborhood, weak_links Brandes,
    schedule_waves) runs its existing Python over seam-loaded
    edges, identical on both rungs by construction. The composite
    edge key lives only in the joined document id: the
    single-node edges() form and remove_node scan restored rows
    and filter in Python — never a find on key parts.
    external_ref dedup is the code's find-first on both rungs.
    Helpers live on the class."""

    @staticmethod
    def _build(graph) -> dict:
        ids = {}
        spec = [("schedule", "Flight"), ("commitment", "Meeting"),
                ("commitment", "Dinner"), ("deadline", "Report"),
                ("person", "Ada"), ("commitment", "Cycle A"),
                ("commitment", "Cycle B"), ("asset", "Vault")]
        for type_, label in spec:
            node = graph.add_node(
                type_, label,
                {"due_ts": 100.0} if label == "Report" else None,
                external_ref="ext:vault" if label == "Vault" else "")
            assert node is not None
            ids[label] = node.node_id
        graph.add_edge(ids["Meeting"], ids["Flight"], "depends_on")
        graph.add_edge(ids["Dinner"], ids["Meeting"], "depends_on")
        graph.add_edge(ids["Report"], ids["Dinner"], "depends_on")
        graph.add_edge(ids["Meeting"], ids["Ada"], "owned_by")
        graph.add_edge(ids["Vault"], ids["Meeting"], "blocks")
        graph.add_edge(ids["Cycle A"], ids["Cycle B"], "depends_on")
        graph.add_edge(ids["Cycle B"], ids["Cycle A"], "depends_on")
        return ids

    def _scenario(self, graph) -> dict:
        from types import SimpleNamespace

        ids = self._build(graph)
        by_id = {n.node_id: n.label for n in graph.list_nodes()}
        out = {
            "dependents": [n.label for n in graph.dependents(ids["Flight"])],
            "dependencies": [n.label
                             for n in graph.dependencies(ids["Dinner"])],
            "breaks": graph.what_breaks_if(ids["Flight"])["count"],
            "critical": [n.label for n in graph.critical_path()],
            "cycles": sorted(
                sorted(by_id[i] for i in cyc) for cyc in graph.find_cycles()),
            "path": [n.label for n in graph.path_between(
                ids["Flight"], ids["Report"])],
            "hood": [n.label for n in graph.neighborhood(ids["Meeting"], 2)],
            "waves": [[n.label for n in w] for w in graph.schedule_waves()],
            "weak": [(w["label"], w["betweenness"])
                     for w in graph.weak_links()],
            "summary": graph.summary(),
            "ext_dedup": graph.add_node(
                "asset", "Vault Again",
                external_ref="ext:vault").node_id == ids["Vault"],
            "edges": len(graph.edges()),
            "edges_of_meeting": len(graph.edges(ids["Meeting"])),
            "nodes": len(graph.list_nodes()),
        }
        # projection compositions: counts equal on both rungs, and
        # re-projection is idempotent via external_ref dedup
        goal = SimpleNamespace(
            id="g1", title="Ship it", status="active", progress=0.5,
            deadline=0,
            steps=[SimpleNamespace(id="s1", description="Step one",
                                   status="open"),
                   SimpleNamespace(id="s2", description="Step two",
                                   status="open")])
        out["goal_first"] = graph.project_goal(goal)
        out["goal_second"] = graph.project_goal(goal)

        class Rec:
            def __init__(self, rid, text, kind, tags):
                self.record_id = rid
                self.text = text
                self.kind = kind
                self.tags = tags

        class Result:
            records = [Rec("r1", "Rent due Friday", "commitment",
                           "deadline"),
                       Rec("r2", "Flight to London", "plan", "flight")]

        class FakeMemory:
            def recall(self, query, limit=None):
                return Result()

        out["mem_first"] = graph.sync_from_memory(FakeMemory())
        out["mem_second"] = graph.sync_from_memory(FakeMemory())
        out["nodes_after_sync"] = len(graph.list_nodes())
        # disruption: final-state equivalence (affected set, flags,
        # the note, the summary), then a partial clear
        graph.mark_disrupted(ids["Flight"], note="cancelled")
        out["disrupted"] = sorted(n.label for n in graph.disrupted())
        out["note"] = graph.get(ids["Meeting"]).attrs.get(
            "disruption_note")
        out["summary_disrupted"] = graph.summary()
        graph.clear_disruption(ids["Flight"])
        out["after_clear"] = sorted(n.label for n in graph.disrupted())
        # removal: edges first, then the node — final state pinned
        out["remove_cycle"] = graph.remove_node(ids["Cycle A"])
        out["edges_after_remove"] = len(graph.edges())
        out["cycles_after_remove"] = len(graph.find_cycles())
        out["removed_gone"] = graph.get(ids["Cycle A"]) is None
        out["remove_missing"] = graph.remove_node("wg_missing")
        return out

    def test_conformance(self, rung, tmp_path):
        from nomorals.planning.graph import WorldGraph

        graph = WorldGraph(str(tmp_path / "world.db"), store=rung.store)
        assert self._scenario(graph) == {
            "dependents": ["Dinner", "Meeting", "Report"],
            "dependencies": ["Meeting"],
            "breaks": 3,
            "critical": ["Report", "Dinner", "Meeting", "Flight"],
            "cycles": [["Cycle A", "Cycle B"]],
            "path": ["Flight", "Meeting", "Dinner", "Report"],
            "hood": ["Dinner", "Flight", "Meeting", "Report", "Vault"],
            "waves": [["Vault", "Ada", "Flight"], ["Meeting"],
                      ["Dinner"], ["Report"], ["Cycle A", "Cycle B"]],
            "weak": [("Meeting", 0.333), ("Dinner", 0.2),
                     ("Vault", 0.0), ("Flight", 0.0), ("Cycle A", 0.0),
                     ("Cycle B", 0.0), ("Report", 0.0)],
            "summary": "🌐 world graph: 8 nodes, 7 edges",
            "ext_dedup": True,
            "edges": 7,
            "edges_of_meeting": 4,
            "nodes": 8,
            "goal_first": 3,
            "goal_second": 3,
            "mem_first": 2,
            "mem_second": 2,
            "nodes_after_sync": 13,
            "disrupted": ["Dinner", "Flight", "Meeting", "Report"],
            "note": "cancelled",
            "summary_disrupted":
                "🌐 world graph: 13 nodes, 10 edges, ⚠️ 4 disrupted",
            "after_clear": ["Dinner", "Meeting", "Report"],
            "remove_cycle": True,
            "edges_after_remove": 8,
            "cycles_after_remove": 0,
            "removed_gone": True,
            "remove_missing": False,
        }

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.planning.graph import WorldGraph

        path = str(tmp_path / "world.db")
        native = WorldGraph(path)
        flight = native.add_node("schedule", "Flight")
        meet = native.add_node("commitment", "Meeting")
        vault = native.add_node("asset", "Vault",
                                external_ref="ext:vault")
        native.add_edge(meet.node_id, flight.node_id, "depends_on")
        native.add_edge(vault.node_id, meet.node_id, "blocks")
        native.mark_disrupted(flight.node_id, note="cancelled")

        seam = WorldGraph(path, store=rung.store)
        assert rung.store.count("planning-graph__graph_nodes") == 3
        assert rung.store.count("planning-graph__graph_edges") == 2
        marker = rung.store.get("planning-graph__meta", "import")
        assert marker is not None and marker["done"] is True
        assert marker["tables"] == {"graph_nodes": 3, "graph_edges": 2}
        # the node key lives only in the document id
        doc = rung.store.get("planning-graph__graph_nodes",
                             flight.node_id)
        assert doc is not None and doc["_id"] == flight.node_id
        assert "node_id" not in doc
        assert doc["disrupted"] == 1
        # the edge key parts live only in the joined document id
        joined = "\x1f".join((meet.node_id, flight.node_id,
                              "depends_on"))
        edoc = rung.store.get("planning-graph__graph_edges", joined)
        assert edoc is not None and edoc["_id"] == joined
        assert "from_id" not in edoc and "to_id" not in edoc
        assert "type" not in edoc
        # a traversal spot-check over the imported state
        assert [n.label for n in seam.dependents(flight.node_id)] == \
            ["Meeting"]
        assert seam.get(meet.node_id).attrs["disruption_note"] == \
            "cancelled"
        assert seam.add_node("asset", "Vault Again",
                             external_ref="ext:vault").node_id == \
            vault.node_id
        # seam writes continue; a second construction adopts
        extra = seam.add_node("person", "Ada")
        seam.add_edge(extra.node_id, meet.node_id, "owned_by")
        assert rung.store.count("planning-graph__graph_nodes") == 4
        assert rung.store.count("planning-graph__graph_edges") == 3
        again = WorldGraph(path, store=rung.store)
        assert [n.label for n in again.dependents(flight.node_id)] == \
            ["Meeting"]
        assert rung.store.count("planning-graph__graph_nodes") == 4

    def test_large_graph_walks(self, rung, tmp_path):
        """The dossier's volume case: a graph at the top of the
        personal-world-model range (300 nodes) — the walks are the
        module's own Python on both rungs, so their results (not
        just the small fixture's) are identical."""
        from nomorals.planning.graph import WorldGraph

        graph = WorldGraph(str(tmp_path / "big.db"), store=rung.store)
        chain = []
        for i in range(150):
            node = graph.add_node("commitment", f"N{i:03d}")
            assert node is not None
            chain.append(node.node_id)
            if i:
                graph.add_edge(node.node_id, chain[i - 1], "depends_on")
        for j in range(150):
            leaf = graph.add_node("commitment", f"L{j:03d}")
            assert leaf is not None
            graph.add_edge(leaf.node_id, chain[-1], "depends_on")
        assert len(graph.list_nodes()) == 300
        assert len(graph.edges()) == 299
        assert len(graph.dependents(chain[0])) == 299
        # leaf -> ... -> chain head: 150 chain nodes + the leaf
        assert len(graph.critical_path()) == 151
        assert graph.find_cycles() == []


class TestPlanningCongestionPort:
    """planning/congestion.py — the planning-congestion port
    (residue Batch 3, C3): resources, holds and breakers are
    documents in planning-congestion__congestion_resources /
    __congestion_holds / __congestion_breakers; the AUTOINCREMENT
    events ride allocated ids via put_raw in __congestion_events,
    the counter seeded by the lazy import. THE lease pin (design
    dossier §C3): a hold's expires_at is document DATA — it rides
    in the body as sql_expires_at through the carry layer's
    _RESERVED rename and the managed expires_at stays None on
    every seam write — so 0 still means NO expiry, an expired
    hold stays stored until reap_stale counts it and emits its
    expire event, and expiry is evaluated by the organ's own
    predicates (live: expires_at <= 0 OR expires_at > now;
    reapable: expires_at > 0 AND expires_at <= now), identically
    on every rung. Multi-document write sets write the state
    document first and the event document second."""

    @staticmethod
    def _monitor(path, rung):
        # planning is L3: the monitor wires the factory itself
        # behind the keyword-only store= — no injected port here.
        from nomorals.planning.congestion import ContentionMonitor

        if rung.store is None:
            return ContentionMonitor(str(path))
        return ContentionMonitor(str(path), store=rung.store)

    @staticmethod
    def _wait_until(predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return predicate()

    def _scenario(self, path, rung) -> dict:
        mon = self._monitor(path, rung)
        out: dict = {}
        a = mon.register("lease-a", kind="api_key", capacity=2)
        b = mon.register("lease-b", kind="endpoint", capacity=1)
        out["resources"] = [r.name for r in mon.list_resources()]
        out["kinds"] = [mon.get("lease-a").kind, mon.get("lease-b").kind]
        out["by_id"] = mon.get(a.resource_id).name
        a2 = mon.register("lease-a", kind="api_key", capacity=3)
        out["reregister_same_id"] = a2.resource_id == a.resource_id
        out["capacity_after"] = mon.get("lease-a").capacity
        out["alternate"] = mon.register_alternate("lease-a", "lease-b")
        out["alternates"] = mon.get("lease-a").alternates == [b.resource_id]
        # the never-expiring hold + the 600 s lease (the depth window)
        h_anchor = mon.acquire("lease-a", "anchor", ttl_s=0)
        h_lease = mon.acquire("lease-a", "lessee", ttl_s=600)
        out["holds_shape"] = (h_anchor is not None, h_lease is not None)
        t0 = h_lease.acquired_at
        out["depth_t0"] = mon.queue_depth("lease-a", now=t0)
        out["depth_599"] = mon.queue_depth("lease-a", now=t0 + 599)
        out["depth_601"] = mon.queue_depth("lease-a", now=t0 + 601)
        age = mon.oldest_hold_age("lease-a", now=t0 + 601)
        out["age_601_live"] = bool(age is not None and age > 600)
        # a full resource advises backoff; an expired-but-unreaped
        # hold frees it again (the advise-side lease pin)
        mon.acquire("lease-b", "squatter", ttl_s=0)
        out["advise_full"] = mon.advise("lease-b")["action"]
        out["release_squatter"] = mon.release("lease-b", "squatter")
        h_temp = mon.acquire("lease-b", "temp", ttl_s=0.05)
        tb = h_temp.acquired_at
        out["depth_b_live"] = mon.queue_depth("lease-b", now=tb)
        out["depth_b_expired_unreaped"] = mon.queue_depth(
            "lease-b", now=tb + 10)
        assert self._wait_until(lambda: mon.queue_depth("lease-b") == 0)
        out["advise_freed"] = mon.advise("lease-b")["action"]
        # the reap: exactly the one expired hold, counted once, and
        # the never-expiring / long-lease holds on lease-a untouched
        out["reaped"] = mon.reap_stale()
        out["reaped_again"] = mon.reap_stale()
        out["depth_b_post_reap"] = mon.queue_depth("lease-b", now=tb + 10)
        # release flow + rates over the event corpus
        out["release_lessee"] = mon.release("lease-a", "lessee")
        out["release_missing"] = mon.release("lease-a", "lessee")
        out["rates_a"] = mon._rates(a.resource_id, now=time.time())
        out["wait_clear"] = mon.wait_estimate("lease-a", now=t0)
        # breaker: five jam observations open it; advise fails fast
        states = [mon._breaker_observe(a.resource_id, True)
                  for _ in range(5)]
        out["breaker_progression"] = states
        out["breaker_state"] = mon.breaker_state("lease-a")
        advice_open = mon.advise("lease-a")
        out["advise_open"] = (advice_open["action"], advice_open["breaker"])
        out["status"] = [(r["name"], r["depth"]) for r in mon.status()]
        out["predict_shape"] = (
            mon.predict_jam("lease-a")[1] is None,
            mon.predict_jam("ghost-resource"))
        return out

    def test_conformance(self, rung, tmp_path):
        out = self._scenario(tmp_path / "congestion.db", rung)
        assert out == {
            "resources": ["lease-a", "lease-b"],
            "kinds": ["api_key", "endpoint"],
            "by_id": "lease-a",
            "reregister_same_id": True,
            "capacity_after": 3,
            "alternate": True,
            "alternates": True,
            "holds_shape": (True, True),
            "depth_t0": 2,
            "depth_599": 2,
            "depth_601": 1,
            "age_601_live": True,
            "advise_full": "backoff",
            "release_squatter": True,
            "depth_b_live": 1,
            "depth_b_expired_unreaped": 0,
            "advise_freed": "proceed",
            "reaped": 1,
            "reaped_again": 0,
            "depth_b_post_reap": 0,
            "release_lessee": True,
            "release_missing": False,
            "rates_a": (0.4, 0.2),
            "wait_clear": 0.0,
            "breaker_progression": ["closed", "closed", "closed",
                                    "closed", "open"],
            "breaker_state": "open",
            "advise_open": ("backoff", "open"),
            "status": [("lease-a", 1), ("lease-b", 0)],
            "predict_shape": (True, (0.0, None)),
        }

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.planning.congestion import ContentionMonitor

        path = tmp_path / "congestion.db"
        native = ContentionMonitor(str(path))
        native.register("imp-a", kind="api_key", capacity=5)
        h_holder = native.acquire("imp-a", "holder", ttl_s=600)
        native.acquire("imp-a", "anchor", ttl_s=0)
        assert native.release("imp-a", "anchor")
        # events in the file: acquire(1), acquire(2), release(3)

        res_c = "planning-congestion__congestion_resources"
        holds_c = "planning-congestion__congestion_holds"
        events_c = "planning-congestion__congestion_events"
        mon = self._monitor(path, rung)
        assert rung.store.count(res_c) == 1
        assert rung.store.count(holds_c) == 1
        assert rung.store.count(events_c) == 3
        marker = rung.store.get("planning-congestion__meta", "import")
        assert marker is not None and marker["done"] is True
        # THE seam-shape pin: the lease is document data under the
        # carry rename; the managed TTL field is present and None
        doc = rung.store.get(holds_c, h_holder.hold_id,
                             include_expired=True)
        assert doc is not None
        assert set(doc) == {"_id", "resource_id", "agent",
                            "acquired_at", "sql_expires_at",
                            "expires_at"}
        assert doc["expires_at"] is None
        assert doc["sql_expires_at"] == doc["acquired_at"] + 600.0
        # event ids continue the file's sequence (import max = 3)
        assert rung.store.get(events_c, "3")["kind"] == "release"
        h_late = mon.acquire("imp-a", "late", ttl_s=0)
        assert rung.store.count(events_c) == 4
        assert rung.store.get(events_c, "4")["kind"] == "acquire"
        late_doc = rung.store.get(holds_c, h_late.hold_id,
                                  include_expired=True)
        assert late_doc["sql_expires_at"] == 0  # ttl<=0: NO expiry
        # an expired hold stays PRESENT until the organ reaps it
        h_temp = mon.acquire("imp-a", "temp", ttl_s=0.05)
        assert rung.store.count(holds_c) == 3
        assert mon.queue_depth("imp-a", now=time.time() + 10) == 2
        assert self._wait_until(lambda: mon.reap_stale() == 1)
        assert rung.store.count(holds_c) == 2
        assert rung.store.get(holds_c, h_temp.hold_id,
                              include_expired=True) is None
        # exactly one expire event, emitted by the reap
        expires = rung.store.find(events_c, {"kind": "expire"},
                                  include_expired=True)
        assert len(expires) == 1
        # no seam document for this store EVER carries a managed TTL
        for coll in (res_c, holds_c, events_c,
                     "planning-congestion__congestion_breakers"):
            for stored in rung.store.find(coll, {},
                                          include_expired=True):
                assert stored.get("expires_at") is None, (
                    f"{coll}/{stored.get('_id')} carries a managed "
                    "expires_at — hold expiry must stay document data")
        # a second construction adopts — never re-imports
        again = self._monitor(path, rung)
        assert again.queue_depth("imp-a") == 2
        assert rung.store.count(events_c) == 6


# ---------------------------------------------------------------------------
# travel-groups (residue Batch 3, C6)
# ---------------------------------------------------------------------------

class TestTravelGroupsPort:
    """travel/groups.py — the travel-groups port (residue Batch 3,
    C6): one file per group chat becomes one collection namespace
    per group key — ``travel-groups--<sanitized group
    key>__<table>`` — with its own import marker, so each group's
    file lazy-imports once into its own collections and no group
    can see another's trips, expenses or polls. Composite PKs
    (votes (poll_id, member), budgets (trip_id, member)) ride the
    joined document id; their parts live only in _id, so per-poll
    vote reads scan the namespaced table and filter in Python.
    Money disposition (design dossier §C6): splits are single-row
    writes with shares precomputed by the unchanged pure
    compute_shares, and balances / settlement / member_totals /
    category_totals stay recomputed reads over finds, never stored
    state."""

    @staticmethod
    def _store(path, rung, key="test:group1"):
        # travel is L2: the store wires the factory itself behind
        # the keyword-only store= — no injected port here.
        from nomorals.travel.groups import GroupTripStore

        if rung.store is None:
            return GroupTripStore(key, db_path=str(path))
        return GroupTripStore(key, db_path=str(path), store=rung.store)

    def _scenario(self, path, rung) -> dict:
        from nomorals.travel.itinerary import Flight

        store = self._store(path, rung)
        trip = store.create_trip(
            "Lagos", ["Ada", "Mama", "Emeka"], created_by="Ada",
            destination="Lagos", start_date="2026-12-01",
            end_date="2026-12-05")
        out: dict = {}
        out["trip"] = (trip.name, trip.members, trip.destination,
                       trip.start_date, trip.end_date)
        out["fetched"] = (store.get_trip(trip.id).name,
                          store.get_trip(trip.id).members)
        out["trips"] = [t.name for t in store.list_trips()]
        store.set_trip_details(trip.id, destination="Abuja")
        out["details"] = store.get_trip(trip.id).destination
        # equal (with dust) / percent / exact splits
        e1 = store.add_expense(trip.id, "Ada", 10000, None, "suya",
                               category="food")
        out["e1_shares"] = e1.shares_kobo
        e2 = store.add_expense(trip.id, "Mama", 9000, ["Ada", "Mama"],
                               "taxi", split="percent", shares=[70, 30],
                               category="transport")
        out["e2_shares"] = e2.shares_kobo
        # exact: the payer is auto-added to for_whom, so the
        # shares list covers all three (Emeka's own share is 0)
        e3 = store.add_expense(trip.id, "Emeka", 5000,
                               ["Ada", "Mama", "Emeka"],
                               "fuel", split="exact",
                               shares=[2000, 3000, 0])
        out["e3_shares"] = e3.shares_kobo
        out["expenses"] = [(e.who_paid, e.amount_kobo, e.category)
                           for e in store.list_expenses(trip.id)]
        out["balances"] = store.balances(trip.id)
        out["settlement"] = store.settlement(trip.id)
        pay = store.record_payment(trip.id, "Ada", "Emeka", 1634)
        out["pay"] = (pay.from_member, pay.to_member, pay.amount_kobo)
        out["balances_after_pay"] = store.balances(trip.id)
        out["settlement_after_pay"] = store.settlement(trip.id)
        out["payments"] = [(p.from_member, p.to_member, p.amount_kobo)
                           for p in store.list_payments(trip.id)]
        store.set_budget(trip.id, "Ada", 20000)
        store.set_budget(trip.id, "Mama", 15000)
        out["budget_ada"] = store.get_budget(trip.id, "Ada")
        out["budget_mama"] = store.get_budget(trip.id, "Mama")
        out["budget_missing"] = store.get_budget(trip.id, "Nobody")
        out["spending_ada"] = store.spending(trip.id, "Ada")
        out["spending_mama"] = store.spending(trip.id, "Mama")
        out["member_totals"] = store.member_totals(trip.id)
        out["category_totals"] = store.category_totals(trip.id)
        # polls: votes, replace, deadline, too-few-options
        poll = store.create_poll(
            trip.id, "Beach or city?", ["Beach", "City", "Mountains"],
            deadline=time.time() + 3600, created_by="Ada")
        store.vote(poll.id, "Ada", "Beach")
        store.vote(poll.id, "Mama", 1)
        store.vote(poll.id, "Emeka", "Beach")
        store.vote(poll.id, "Ada", "City")  # replace: one vote per member
        res = store.poll_result(poll.id)
        out["poll_counts"] = res["counts"]
        out["poll_winner"] = res["winner"]
        out["poll_total"] = res["total"]
        out["voters"] = res["voters"]
        poll2 = store.create_poll(trip.id, "Second?", ["X", "Y"])
        res2 = store.poll_result(poll2.id)
        out["poll2"] = (res2["counts"], res2["winner"], res2["total"])
        closed = store.create_poll(trip.id, "Late?", ["A", "B"],
                                   deadline=time.time() - 10)
        out["deadline_vote"] = store.vote(closed.id, "Ada", "A")
        out["short_poll"] = store.create_poll(trip.id, "One?", ["Only"])
        out["closing_soon"] = sorted(
            p.question for p in store.polls_closing_soon(trip.id))
        # Zara joins after the money: she has not voted yet
        store.add_member(trip.id, "Zara")
        out["after_add"] = store.get_trip(trip.id).members
        out["nonvoters"] = store.poll_nonvoters(poll.id)
        # legs + proposals
        leg = store.add_leg(trip.id, "Ada", Flight(
            airline="BA", flight_number="BA075", origin="LOS",
            destination="LHR", departs="2026-12-01T22:45",
            arrives="2026-12-02T05:30", pnr="ABC123"))
        out["legs"] = [(lg.member, lg.flight.get("flight_number"))
                       for lg in store.list_legs(trip.id)]
        assert leg is not None
        out["itinerary_has_flight"] = (
            "BA075" in store.merged_itinerary(trip.id))
        prop = store.propose(trip.id, "Mama", "Sunset cruise")
        assert prop is not None
        out["proposals_open_before"] = [
            p.idea for p in store.list_proposals(trip.id, "proposed")]
        agreed = store.agree(trip.id, prop.id)
        assert agreed is not None and agreed.status == "agreed"
        out["proposals_open"] = [
            p.idea for p in store.list_proposals(trip.id, "proposed")]
        out["proposals_agreed"] = [
            p.idea for p in store.list_proposals(trip.id, "agreed")]
        out["bad_expense"] = (
            store.add_expense(trip.id, "Ada", 1000, ["Ada", "Mama"],
                              "x", split="percent", shares=[50, 40])
            is None,
            store.add_expense(trip.id, "Ada", 0, None, "x") is None)
        out["missing"] = (store.get_trip("nope") is None,
                          store.get_poll("nope") is None,
                          store.poll_result("nope") is None)
        out["overview"] = ("🧳 Lagos" in store.trip_overview(trip.id)
                           and "settle up" in store.trip_overview(trip.id))
        # settle-guard: a nonzero balance cannot leave; zero can
        out["leave_settler"] = store.leave_trip(trip.id, "Mama")
        out["leave_clean"] = store.leave_trip(trip.id, "Ada")
        out["members_after_leave"] = store.get_trip(trip.id).members
        return out

    def test_conformance(self, rung, tmp_path):
        out = self._scenario(tmp_path / "g.db", rung)
        assert out == {
            "trip": ("Lagos", ["Ada", "Mama", "Emeka"], "Lagos",
                     "2026-12-01", "2026-12-05"),
            "fetched": ("Lagos", ["Ada", "Mama", "Emeka"]),
            "trips": ["Lagos"],
            "details": "Abuja",
            "e1_shares": [3334, 3333, 3333],
            "e2_shares": [6300, 2700],
            "e3_shares": [2000, 3000, 0],
            "expenses": [("Ada", 10000, "food"),
                         ("Mama", 9000, "transport"),
                         ("Emeka", 5000, "")],
            "balances": {"Ada": -1634, "Mama": -33, "Emeka": 1667},
            "settlement": [("Ada", "Emeka", 1634),
                           ("Mama", "Emeka", 33)],
            "pay": ("Ada", "Emeka", 1634),
            "balances_after_pay": {"Ada": 0, "Mama": -33, "Emeka": 33},
            "settlement_after_pay": [("Mama", "Emeka", 33)],
            "payments": [("Ada", "Emeka", 1634)],
            "budget_ada": 20000,
            "budget_mama": 15000,
            "budget_missing": 0,
            "spending_ada": 11634,
            "spending_mama": 9033,
            "member_totals": {
                "Ada": {"spent": 11634, "paid": 10000, "balance": 0,
                        "budget": 20000, "remaining": 8366},
                "Emeka": {"spent": 3333, "paid": 5000, "balance": 33,
                          "budget": 0, "remaining": 0},
                "Mama": {"spent": 9033, "paid": 9000, "balance": -33,
                         "budget": 15000, "remaining": 5967},
            },
            "category_totals": {"food": 10000, "transport": 9000,
                                "other": 5000},
            "poll_counts": [1, 2, 0],
            "poll_winner": 1,
            "poll_total": 3,
            "voters": {"Ada": 1, "Emeka": 0, "Mama": 1},
            "poll2": ([0, 0], None, 0),
            "deadline_vote": False,
            "short_poll": None,
            "closing_soon": ["Beach or city?"],
            "after_add": ["Ada", "Mama", "Emeka", "Zara"],
            "nonvoters": ["Zara"],
            "legs": [("Ada", "BA075")],
            "itinerary_has_flight": True,
            "proposals_open_before": ["Sunset cruise"],
            "proposals_open": [],
            "proposals_agreed": ["Sunset cruise"],
            "bad_expense": (True, True),
            "missing": (True, True, True),
            "overview": True,
            "leave_settler": (False,
                              "settle up first — you still owe ₦0."),
            "leave_clean": (True, "Ada left Lagos."),
            "members_after_leave": ["Mama", "Emeka", "Zara"],
        }

    def test_two_group_isolation(self, rung, tmp_path):
        """THE partitioning pin (design dossier §C6.2): two stores
        with different group keys over ONE seam backend share
        nothing — and group B's file still lazy-imports after group
        A is live on the seam (the collision the un-namespaced
        factory would have served group A's trips to group B)."""
        a = self._store(tmp_path / "a.db", rung, key="telegram:alpha")
        trip_a = a.create_trip("Lagos", ["Ada", "Mama"], created_by="Ada")
        a.add_expense(trip_a.id, "Ada", 10000, None, "suya",
                      category="food")
        a.set_budget(trip_a.id, "Ada", 20000)
        poll_a = a.create_poll(trip_a.id, "Go?", ["Yes", "No"])
        a.vote(poll_a.id, "Ada", "Yes")

        b = self._store(tmp_path / "b.db", rung, key="telegram:beta")
        assert b.list_trips() == []
        assert b.get_trip(trip_a.id) is None
        assert b.balances(trip_a.id) == {}
        assert b.poll_result(poll_a.id) is None
        assert b.get_budget(trip_a.id, "Ada") == 0
        trip_b = b.create_trip("Abuja", ["Zara"], created_by="Zara")
        assert [t.name for t in a.list_trips()] == ["Lagos"]
        assert [t.name for t in b.list_trips()] == ["Abuja"]
        assert a.get_trip(trip_b.id) is None

        # group C's file, pre-built natively, opens on the seam
        # AFTER A is live — its own import, its own namespace.
        c_path = tmp_path / "c.db"
        c_nat = self._store(c_path, SimpleNamespace(kind="native",
                                                     store=None),
                            key="telegram:gamma")
        trip_c = c_nat.create_trip("Enugu", ["Obi"], created_by="Obi")
        c_nat.add_expense(trip_c.id, "Obi", 5000, None, "fuel",
                          category="transport")
        c = self._store(c_path, rung, key="telegram:gamma")
        assert [t.name for t in c.list_trips()] == ["Enugu"]
        assert c.balances(trip_c.id) == {"Obi": 0}
        assert c.get_budget(trip_c.id, "Obi") == 0
        # A is untouched by C's import
        assert [t.name for t in a.list_trips()] == ["Lagos"]
        assert a.balances(trip_a.id) == {"Ada": 5000, "Mama": -5000}
        if rung.store is not None:
            assert rung.store.count(
                "travel-groups--telegram_alpha__group_trips") == 1
            assert rung.store.count(
                "travel-groups--telegram_beta__group_trips") == 1
            assert rung.store.count(
                "travel-groups--telegram_gamma__group_trips") == 1
            assert rung.store.count(
                "travel-groups--telegram_alpha__expenses") == 1
            assert rung.store.count(
                "travel-groups--telegram_gamma__expenses") == 1

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        import sqlite3

        from nomorals.travel.groups import GroupTripStore

        # a LEGACY group file: pre-ALTER shape (no destination /
        # dates on group_trips, no split / shares_json / category
        # on expenses) — the import must take those columns from
        # the seam DDL defaults, exactly as the native ALTERs do.
        legacy = tmp_path / "legacy.db"
        con = sqlite3.connect(str(legacy))
        con.executescript("""
            CREATE TABLE group_trips (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                members_json TEXT NOT NULL DEFAULT '[]',
                created_by TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE polls (id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                question TEXT NOT NULL, options_json TEXT NOT NULL DEFAULT '[]',
                deadline REAL NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE votes (poll_id TEXT NOT NULL, member TEXT NOT NULL,
                option_idx INTEGER NOT NULL, PRIMARY KEY (poll_id, member));
            CREATE TABLE expenses (id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                who_paid TEXT NOT NULL, amount_kobo INTEGER NOT NULL,
                for_whom_json TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE payments (id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                from_member TEXT NOT NULL, to_member TEXT NOT NULL,
                amount_kobo INTEGER NOT NULL,
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE budgets (trip_id TEXT NOT NULL, member TEXT NOT NULL,
                amount_kobo INTEGER NOT NULL, PRIMARY KEY (trip_id, member));
            CREATE TABLE legs (id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                member TEXT NOT NULL, flight_json TEXT NOT NULL DEFAULT '{}',
                note TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL DEFAULT 0);
            CREATE TABLE proposals (id TEXT PRIMARY KEY, trip_id TEXT NOT NULL,
                member TEXT NOT NULL, idea TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at REAL NOT NULL DEFAULT 0);
        """)
        con.execute(
            "INSERT INTO group_trips VALUES (?,?,?,?,?)",
            ("gtrip_old", "Old Trip", '["Ada", "Mama"]', "Ada", 1000.0))
        con.execute(
            "INSERT INTO expenses VALUES (?,?,?,?,?,?,?)",
            ("exp_old", "gtrip_old", "Ada", 10000, '["Ada", "Mama"]',
             "suya", 1001.0))
        con.execute(
            "INSERT INTO votes VALUES (?,?,?)", ("poll_old", "Ada", 0))
        con.execute(
            "INSERT INTO budgets VALUES (?,?,?)",
            ("gtrip_old", "Ada", 20000))
        con.commit()
        con.close()

        ns = "travel-groups--telegram_legacy"
        store = GroupTripStore("telegram:legacy", db_path=str(legacy),
                               store=rung.store)
        trip = store.get_trip("gtrip_old")
        assert trip is not None and trip.name == "Old Trip"
        assert trip.destination == "" and trip.start_date == ""
        assert trip.end_date == ""
        exps = store.list_expenses("gtrip_old")
        assert len(exps) == 1
        # legacy row: no shares stored — the equal split is
        # recomputed, exactly as on the native path
        assert exps[0].split == "equal"
        assert exps[0].shares_kobo == [5000, 5000]
        assert exps[0].category == ""
        assert store.get_budget("gtrip_old", "Ada") == 20000
        assert store.balances("gtrip_old") == {"Ada": 5000, "Mama": -5000}
        marker = rung.store.get(f"{ns}__meta", "import")
        assert marker is not None and marker["done"] is True
        assert rung.store.count(f"{ns}__group_trips") == 1
        assert rung.store.count(f"{ns}__expenses") == 1
        assert rung.store.count(f"{ns}__votes") == 1
        assert rung.store.count(f"{ns}__budgets") == 1
        # the imported trip document carries the final shape
        doc = rung.store.get(f"{ns}__group_trips", "gtrip_old")
        assert doc["destination"] == "" and doc["start_date"] == ""
        # a second construction adopts — never re-imports
        again = GroupTripStore("telegram:legacy", db_path=str(legacy),
                               store=rung.store)
        assert len(again.list_trips()) == 1
        assert rung.store.count(f"{ns}__expenses") == 1
        # seam-side writes land in the same namespace
        again.add_expense("gtrip_old", "Mama", 2000, None, "water")
        assert rung.store.count(f"{ns}__expenses") == 2
        assert again.balances("gtrip_old") == {"Ada": 4000, "Mama": -4000}


class TestTravelGroupsNamespace:
    """The Batch 3 factory extension itself
    (``sidecar_port(..., namespace_suffix=...)``), pinned in the
    C6 commit: the namespace is validated against the document
    store's collection-name alphabet and rejected loudly; the
    registry lookup and verdict check still resolve against the
    base entry; a port without a namespace is exactly the old
    shape (effective name == base name)."""

    def test_namespace_rejected_loudly(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung resolves no store")
        from nomorals.storage.sidecars import sidecar_port
        from nomorals.travel.groups import _DDL, _TABLES

        with pytest.raises(ValueError, match="not a valid collection"):
            sidecar_port("travel-groups", _DDL, _TABLES,
                         str(tmp_path / "x.db"), store=rung.store,
                         namespace_suffix="bad key!")
        with pytest.raises(ValueError, match="not a valid collection"):
            sidecar_port("travel-groups", _DDL, _TABLES,
                         str(tmp_path / "x.db"), store=rung.store,
                         namespace_suffix=".leading-dot")

    def test_verdict_still_checked_through_namespace(self, rung,
                                                    tmp_path):
        if rung.store is None:
            pytest.skip("native rung resolves no store")
        from nomorals.storage.sidecars import sidecar_port
        from nomorals.travel.groups import _DDL, _TABLES

        # NOTE (parent re-pin, C2 ship): this test used to pin
        # "marketing-send" as the residue example; C2 ports it, so
        # the residue example is now "audio-fingerprints" (the last
        # Batch 3 residue until C7 lands).
        with pytest.raises(ValueError, match="classified 'residue'"):
            sidecar_port("audio-fingerprints", _DDL, _TABLES,
                         str(tmp_path / "y.db"), store=rung.store,
                         namespace_suffix="telegram_999")

    def test_namespace_effective_name(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung resolves no store")
        from nomorals.storage.sidecars import sidecar_port
        from nomorals.travel.groups import _DDL, _TABLES

        port = sidecar_port("travel-groups", _DDL, _TABLES,
                            str(tmp_path / "z.db"), store=rung.store,
                            namespace_suffix="telegram_123")
        assert port.store_name == "travel-groups--telegram_123"
        assert port.meta_collection == \
            "travel-groups--telegram_123__meta"
        assert port.tables["group_trips"].collection == \
            "travel-groups--telegram_123__group_trips"

    def test_non_namespaced_port_unaffected(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung resolves no store")
        from nomorals.storage.sidecars import sidecar_port

        port = sidecar_port(
            "planning-congestion",
            "CREATE TABLE congestion_resources "
            "(resource_id TEXT PRIMARY KEY)",
            ("congestion_resources",), str(tmp_path / "w.db"),
            store=rung.store)
        assert port.store_name == "planning-congestion"
        assert port.meta_collection == "planning-congestion__meta"


class TestMarketingSendPort:
    """marketing/send.py — the marketing-send port (residue Batch 3,
    C2): the coordination port. The nine tables are documents in
    ``marketing-send__<table>``; the AUTOINCREMENT log/cost rows
    ride allocated ids via put_raw, the counters seeded by the
    one-time lazy import. THE claim pin (design dossier §C2): the
    queue's claim lives in two seam-only document fields,
    claim_token and claim_until (CLAIM_TTL 300s) — process() claims
    each due row with ONE cas_row fenced on {status, attempts,
    next_attempt_at, claim_token as-read}, every outcome write is
    fenced on the claim token the attempt holds, and a lost fence
    fails closed (refused + journaled, the row left for its claim
    to expire); deferrals release the claim so the row stays
    immediately re-attemptable. Native rows never carry claim
    fields and the native path serves the SQLite default
    byte-for-byte; a mixed fleet is NOT a supported composition."""

    @staticmethod
    def _engine(path, rung, **kw):
        from nomorals.marketing.send import SendEngine

        if rung.store is None:
            return SendEngine(str(path), **kw)
        return SendEngine(str(path), store=rung.store, **kw)

    @staticmethod
    def _fast_forward(engine, rung, path, send_id):
        """Make a queued row due again on every rung.

        Native: SQL UPDATE of next_attempt_at. Seam: the RowTable
        update clears the claim too — the claim is live until
        CLAIM_TTL, and this simulates its expiry."""
        if rung.store is None:
            import sqlite3

            conn = sqlite3.connect(str(path))
            try:
                conn.execute(
                    "UPDATE send_queue SET next_attempt_at = 0 "
                    "WHERE send_id = ?", (send_id,))
                conn.commit()
            finally:
                conn.close()
        else:
            engine._port.tables["send_queue"].update(send_id, {
                "next_attempt_at": 0.0, "claim_token": "",
                "claim_until": 0.0})

    @staticmethod
    def _queue_rows(engine, rung, path, campaign_id):
        """Queue rows for a campaign, as plain dicts, on every rung."""
        if rung.store is None:
            import sqlite3

            conn = sqlite3.connect(str(path))
            try:
                cur = conn.execute(
                    "SELECT * FROM send_queue WHERE campaign_id = ?",
                    (campaign_id,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r, strict=True))
                        for r in cur.fetchall()]
            finally:
                conn.close()
        return engine._port.tables["send_queue"].find(
            {"campaign_id": campaign_id})

    def _scenario(self, path, rung) -> dict:
        e = self._engine(path, rung)
        out: dict = {}
        e.register_provider("mock", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        e.register_provider("flaky", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: False)
        e.register_provider("raiser", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: (_ for _ in ()).throw(
                                RuntimeError("boom")))
        e.register_provider("slow", kind="mock", rate_per_minute=1,
                            sender=lambda to, s, b: True)
        e.register_provider("warm", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        out["templates_saved"] = (
            e.save_template("launch", "Hi {{ name }}!", "Launch day"),
            e.save_template("other", "Other {{ name }}", "Other"))
        out["template_names"] = [t["name"] for t in e.list_templates()]
        out["template_body"] = e.get_template("launch")["body"]
        out["provider_names"] = [p["name"] for p in e.list_providers()]
        out["variants_saved"] = (
            e.save_variant("launch", "A", "A {{ name }}", "Sub A"),
            e.save_variant("launch", "B", "B {{ name }}", "Sub B"))
        out["variant_names"] = [v["variant_name"]
                                for v in e.list_variants("launch")]
        out["variant_body"] = e.get_variant("launch", "A")["body"]
        out["estimate"] = e.estimate_campaign(10, "mock")
        out["warmup_none"] = e.warmup_status("mock")
        # blocklist / suppression refusals at enqueue time
        out["blocklisted"] = e.blocklist("bad@x.com", "test")
        out["is_blocklisted"] = e.is_blocklisted("bad@x.com")
        out["unsubscribed"] = e.unsubscribe("gone@x.com")
        out["is_suppressed"] = e.is_suppressed("gone@x.com")
        out["enqueue_blocked"] = e.enqueue("bad@x.com", "launch", {}, "mock")
        out["enqueue_suppressed"] = e.enqueue("gone@x.com", "launch",
                                              {}, "mock")
        out["suppression_count"] = e.suppression_count()
        e.enqueue("a@x.com", "launch", {"name": "Ada"}, "mock")
        out["queued_after_enqueue"] = e.queue_depth()
        out["process_basic"] = e.process()
        out["depth_sent"] = e.queue_depth("sent")
        out["depth_queued"] = e.queue_depth()
        st = e.status()
        out["status_basic"] = (st["queued"], st["sent_24h"], st["failed"],
                               st["dead"], st["blocklisted"],
                               st["suppressed"], st["spent_24h_kobo"])
        # provider-false → backoff → dead at MAX_ATTEMPTS (5)
        sid_f = e.enqueue("f@x.com", "launch", {}, "flaky")
        out["retry_stats"] = []
        for _ in range(5):
            out["retry_stats"].append(e.process())
            self._fast_forward(e, rung, path, sid_f)
        out["depth_dead"] = e.queue_depth("dead")
        # raising sender → the same retry/dead shape
        sid_r = e.enqueue("r@x.com", "launch", {}, "raiser")
        out["raise_stats"] = []
        for _ in range(5):
            out["raise_stats"].append(e.process())
            self._fast_forward(e, rung, path, sid_r)
        out["depth_dead_2"] = e.queue_depth("dead")
        # rate deferral: 1/min, two rows → one sent, one deferred,
        # and the deferred row is re-attemptable at once
        e.enqueue("s1@x.com", "launch", {}, "slow")
        e.enqueue("s2@x.com", "launch", {}, "slow")
        out["process_rate"] = e.process()
        out["process_rate_again"] = e.process()
        out["queued_after_rate"] = e.queue_depth()
        # warmup deferral: target 10 → day-1 cap 2. The slow row
        # from the rate step rides every later drain as a second
        # deferral (its rate window is still full).
        out["warmup_started"] = e.set_warmup("warm", 10)
        e.enqueue("w1@x.com", "launch", {}, "warm")
        e.enqueue("w2@x.com", "launch", {}, "warm")
        e.enqueue("w3@x.com", "launch", {}, "warm")
        out["process_warmup"] = e.process()
        out["process_warmup_again"] = e.process()
        out["warmup_status"] = e.warmup_status("warm")
        # blocklist after enqueue → blocked at attempt time (the two
        # deferred rows above are attempted again in this drain)
        e.enqueue("late@x.com", "launch", {}, "mock")
        e.blocklist("late@x.com", "late")
        out["process_blocked"] = e.process()
        out["depth_blocked"] = e.queue_depth("blocked")
        # A/B campaign: A gets 3, B gets 1 — the winner is
        # deterministic on every rung (lowest fail rate, most sent)
        res = e.create_ab_campaign("ab1", "launch", ["A", "A", "B"],
                                   ["c1@x.com", "c2@x.com", "c3@x.com",
                                    "c4@x.com"], "mock")
        out["ab_campaign"] = (bool(res["campaign_id"]),
                              res["per_variant"])
        out["process_ab"] = e.process()
        cid = res["campaign_id"]
        out["ab_report"] = e.ab_report(cid)
        out["ab_winner"] = e.ab_winner(cid)
        out["campaign_report"] = e.campaign_report(cid)
        # engagement signal prefers the most-engaged variant
        rows = self._queue_rows(e, rung, path, cid)
        a_rows = [r for r in rows if r["variant"] == "A"]
        for r in a_rows[:2]:
            e._log(r["send_id"], "mock", "engaged")
        out["ab_winner_engaged"] = e.ab_winner(cid)
        st = e.status()
        out["status_final"] = (
            st["queued"], st["sent_24h"], st["failed"], st["dead"],
            st["blocklisted"], st["suppressed"], st["spent_24h_kobo"],
            [(p["name"], p["kind"], p["rate_per_minute"],
              p["cost_kobo"], p["enabled"]) for p in st["providers"]])
        return out

    def test_conformance(self, rung, tmp_path):
        out = self._scenario(tmp_path / "send.db", rung)
        assert out == {
            "templates_saved": (True, True),
            "template_names": ["launch", "other"],
            "template_body": "Hi {{ name }}!",
            "provider_names": ["flaky", "mock", "raiser", "slow", "warm"],
            "variants_saved": (True, True),
            "variant_names": ["A", "B"],
            "variant_body": "A {{ name }}",
            "estimate": {
                "count": 10, "per_kobo": 40, "total_kobo": 400,
                "text": "this campaign will cost ₦4 for 10 recipients "
                        "(mock, ₦0.40 each)"},
            "warmup_none": {},
            "blocklisted": True,
            "is_blocklisted": True,
            "unsubscribed": True,
            "is_suppressed": True,
            "enqueue_blocked": "",
            "enqueue_suppressed": "",
            "suppression_count": 1,
            "queued_after_enqueue": 1,
            "process_basic": {"sent": 1, "failed": 0, "deferred": 0,
                              "blocked": 0},
            "depth_sent": 1,
            "depth_queued": 0,
            "status_basic": (0, 1, 0, 0, 1, 1, 0),
            "retry_stats": [
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 1, "deferred": 0, "blocked": 0}],
            "depth_dead": 1,
            "raise_stats": [
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 0, "deferred": 1, "blocked": 0},
                {"sent": 0, "failed": 1, "deferred": 0, "blocked": 0}],
            "depth_dead_2": 2,
            "process_rate": {"sent": 1, "failed": 0, "deferred": 1,
                             "blocked": 0},
            "process_rate_again": {"sent": 0, "failed": 0, "deferred": 1,
                                   "blocked": 0},
            "queued_after_rate": 1,
            "warmup_started": True,
            "process_warmup": {"sent": 2, "failed": 0, "deferred": 2,
                               "blocked": 0},
            "process_warmup_again": {"sent": 0, "failed": 0, "deferred": 2,
                                     "blocked": 0},
            "warmup_status": {"provider": "warm", "day": 1, "cap": 2,
                              "used": 2, "remaining": 0},
            "process_blocked": {"sent": 0, "failed": 0, "deferred": 2,
                                "blocked": 1},
            "depth_blocked": 1,
            "ab_campaign": (True, {"A": 3, "B": 1}),
            "process_ab": {"sent": 4, "failed": 0, "deferred": 2,
                           "blocked": 0},
            "ab_report": {
                "campaign_id": out["ab_report"]["campaign_id"],
                "variants": {
                    "A": {"sent": 3, "failed": 0, "dead": 0, "queued": 0,
                          "blocked": 0, "fail_rate": 0.0},
                    "B": {"sent": 1, "failed": 0, "dead": 0, "queued": 0,
                          "blocked": 0, "fail_rate": 0.0}}},
            "ab_winner": {"winner": "A",
                          "reason": "lowest fail rate (0.0%)",
                          "by": "fail_rate"},
            "campaign_report": {
                "campaign_id": out["campaign_report"]["campaign_id"],
                "total": 4, "by_status": {"sent": 4},
                "by_provider": {"mock": {"sent": 4}}, "spent_kobo": 0},
            "ab_winner_engaged": {"winner": "A",
                                  "reason": "most engagement (2)",
                                  "by": "engagement"},
            "status_final": (
                2, 8, 0, 2, 2, 1, 0,
                [("flaky", "mock", 1000, 0, 1),
                 ("mock", "mock", 1000, 0, 1),
                 ("raiser", "mock", 1000, 0, 1),
                 ("slow", "mock", 1, 0, 1),
                 ("warm", "mock", 1000, 0, 1)]),
        }

    def test_two_engine_exactly_once(self, rung, tmp_path):
        """THE coordination pin: two engines over one seam store
        process() concurrently — every send_id is attempted by
        exactly one engine, total sends == queue size."""
        if rung.store is None:
            pytest.skip("the claim race is a seam property; "
                        "native has no claim")
        import threading
        from types import SimpleNamespace

        from nomorals.storage.db import Database
        from nomorals.storage.docstore import SQLiteDocumentStore

        # the fixture's sqlite_doc store is an in-memory Database
        # whose connections are thread-local: a worker thread would
        # see an EMPTY database, not the shared queue. The race
        # needs a store both threads actually share, so the
        # sqlite_doc rung runs on a file-backed Database (same file,
        # same rows, per-thread connections); the mongo rung is a
        # real server and is shared as-is.
        if rung.kind == "sqlite_doc":
            shared = SimpleNamespace(
                kind="sqlite_doc",
                store=SQLiteDocumentStore(
                    Database(str(tmp_path / "store.db"))))
        else:
            shared = rung

        path = tmp_path / "send.db"
        calls: list[str] = []
        lock = threading.Lock()

        def sender(to, subj, body):
            with lock:
                calls.append(to)
            return True

        a = self._engine(path, shared)
        b = self._engine(path, shared)
        for eng in (a, b):
            eng.register_provider("mock", kind="mock", rate_per_minute=1000,
                                  sender=sender)
            eng.save_template("t", "Hi {{ name }}!", "Hey")
        sids = [a.enqueue(f"u{i}@x.com", "t", {}, "mock") for i in range(6)]
        assert all(sids) and a.queue_depth() == 6
        results: dict[str, dict] = {}

        def run(name, eng):
            results[name] = eng.process()

        ta = threading.Thread(target=run, args=("a", a))
        tb = threading.Thread(target=run, args=("b", b))
        ta.start()
        tb.start()
        ta.join()
        tb.join()
        total_sent = results["a"]["sent"] + results["b"]["sent"]
        assert total_sent == 6, results
        assert len(calls) == 6 and len(set(calls)) == 6, calls
        assert a.queue_depth("sent") == 6
        assert b.queue_depth("sent") == 6

    def test_stale_claim_reclaimed(self, rung, tmp_path):
        """A row hand-claimed with a past claim_until — a crashed
        engine's row — is reclaimed and sent."""
        if rung.store is None:
            pytest.skip("the claim is a seam property; native has "
                        "no claim")
        import time

        e = self._engine(tmp_path / "send.db", rung)
        e.register_provider("mock", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        e.save_template("t", "Hi", "Hey")
        sid = e.enqueue("s@x.com", "t", {}, "mock")
        e._port.tables["send_queue"].update(sid, {
            "claim_token": "clm_dead", "claim_until": time.time() - 10})
        assert e.process() == {"sent": 1, "failed": 0, "deferred": 0,
                               "blocked": 0}
        assert e.queue_depth("sent") == 1

    def test_fence_loss_refuses_outcome(self, rung, tmp_path):
        """Mutating the claim token between claim and outcome makes
        the outcome write refuse: the row stays queued and the loss
        is journaled."""
        if rung.store is None:
            pytest.skip("the claim fence is a seam property")
        from nomorals.marketing.failures import failure_count, recent_failures

        e = self._engine(tmp_path / "send.db", rung)
        e.register_provider("mock", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        e.save_template("t", "Hi", "Hey")
        sid = e.enqueue("f@x.com", "t", {}, "mock")
        row = e._port.tables["send_queue"].get(sid)
        before = failure_count()
        assert e._set_status_seam(row, "clm_stale_token", "sent") is False
        assert e.queue_depth("queued") == 1
        assert e.queue_depth("sent") == 0
        assert failure_count() > before
        assert any(f.site == "send.SendEngine._set_status.fence"
                   for f in recent_failures())
        before = failure_count()
        assert e._retry_or_dead_seam(row, "clm_stale_token", "boom") \
            == "failed"
        assert e.queue_depth("queued") == 1
        assert failure_count() > before
        assert any(f.site == "send.SendEngine._retry_or_dead.fence"
                   for f in recent_failures())

    def test_cost_deferral_leaves_row_queued(self, rung, tmp_path):
        """A budget block defers (and releases the claim, so the row
        is re-attemptable at once) on every rung."""
        from types import SimpleNamespace

        e = self._engine(tmp_path / "send.db", rung)
        # kind=smtp costs 40 kobo: a zero-cost provider never
        # reaches the budget check (_cost_ok returns early).
        e.register_provider("mock", kind="smtp", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        e.save_template("t", "Hi", "Hey")
        e._tracker = SimpleNamespace(
            check_budget=lambda client: (False, 0, 0))
        e.enqueue("c@x.com", "t", {}, "mock")
        assert e.process() == {"sent": 0, "failed": 0, "deferred": 1,
                               "blocked": 0}
        assert e.queue_depth("queued") == 1
        assert e.process() == {"sent": 0, "failed": 0, "deferred": 1,
                               "blocked": 0}

    def test_schedule_and_seed(self, rung, tmp_path):
        """Scheduled sends wait for their time; seeds go at once."""
        import time

        e = self._engine(tmp_path / "send.db", rung)
        e.register_provider("mock", kind="mock", rate_per_minute=1000,
                            sender=lambda to, s, b: True)
        e.save_template("t", "Hi", "Hey")
        res = e.schedule_campaign("t", ["s1@x.com", "s2@x.com"], "mock",
                                  time.time() + 3600)
        assert res["queued"] == 2 and res["campaign_id"]
        assert e.process() == {"sent": 0, "failed": 0, "deferred": 0,
                               "blocked": 0}
        assert e.queue_depth() == 2
        assert e.send_seed("t", ["seed@x.com"], "mock") == {"seeded": 1,
                                                            "of": 1}
        assert e.process()["sent"] == 1
        assert e.queue_depth() == 2

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.marketing.send import SendEngine

        path = tmp_path / "send.db"
        native = SendEngine(str(path))
        native.save_template("imp", "Hello {{ name }}", "Hi")
        native.register_provider("mock", kind="mock", rate_per_minute=100,
                                 sender=lambda to, s, b: True)
        sid = native.enqueue("imp@x.com", "imp", {"name": "I"}, "mock")
        assert native.process()["sent"] == 1
        native.blocklist("b@x.com", "test")
        native.unsubscribe("u@x.com")
        native.set_warmup("mock", 50)
        native.save_variant("imp", "V", "Variant body", "VS")

        e = self._engine(path, rung)
        assert rung.store.count("marketing-send__send_templates") == 1
        assert rung.store.count("marketing-send__send_queue") == 1
        assert rung.store.count("marketing-send__send_log") == 1
        assert rung.store.count("marketing-send__send_costs") == 1
        assert rung.store.count("marketing-send__send_providers") == 1
        assert rung.store.count("marketing-send__send_blocklist") == 1
        assert rung.store.count("marketing-send__send_suppressions") == 1
        assert rung.store.count("marketing-send__send_warmup") == 1
        assert rung.store.count("marketing-send__send_variants") == 1
        marker = rung.store.get("marketing-send__meta", "import")
        assert marker is not None and marker["done"] is True
        # THE seam-shape pin: the claim fields are document data;
        # the managed TTL field is present and None
        doc = rung.store.get("marketing-send__send_queue", sid,
                             include_expired=True)
        assert doc is not None
        assert doc["claim_token"] == "" and doc["claim_until"] == 0
        assert doc.get("expires_at") is None
        # log ids continue the file's sequence (import max = 1)
        assert rung.store.get("marketing-send__send_log", "1")["event"] \
            == "sent"
        e.register_provider("mock", kind="mock", rate_per_minute=100,
                            sender=lambda to, s, b: True)
        e.enqueue("second@x.com", "imp", {}, "mock")
        assert e.process()["sent"] == 1
        assert rung.store.get("marketing-send__send_log", "2")["event"] \
            == "sent"
        # a second construction adopts — never re-imports
        again = self._engine(path, rung)
        assert again.queue_depth("sent") == 2
        assert rung.store.count("marketing-send__send_log") == 2

    def test_native_file_has_no_claim_columns(self, tmp_path):
        """Native rows never carry claim fields: the file's
        send_queue is the final 14-col shape, nothing more."""
        import sqlite3

        from nomorals.marketing.send import SendEngine

        path = tmp_path / "send.db"
        SendEngine(str(path))
        cols = [r[1] for r in sqlite3.connect(str(path)).execute(
            "PRAGMA table_info(send_queue)")]
        assert len(cols) == 14
        assert "claim_token" not in cols and "claim_until" not in cols
