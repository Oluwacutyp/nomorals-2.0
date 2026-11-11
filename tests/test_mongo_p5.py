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
  with the legacy price_snapshots adoption cases on the seam.
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
            "shopping-deals"}

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
