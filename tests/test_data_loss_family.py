"""Data-loss family (targeted-fix Pass 1) — production-path proofs.

One repair pattern across nine stores: unreadable is not empty.
When a store's persisted state exists but cannot be read, the bytes
are quarantined, the failure is logged + ledgered, and a rewrite
from the empty in-memory image is refused (or, for the JSONL store,
the unreadable lines are preserved verbatim through the rewrite;
for the never-raises community stores, the load quarantines and
save results become honest bools).

Per site: (a) corruption → quarantine + honest error + no silent
overwrite [the defect — these fail on the pre-fix tree], (b) missing
state → normal empty behavior [preserved-behavior pin], (c) valid
state → round-trips exactly as before [preserved-behavior pin].
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import nomorals.agents.failures as agents_fail
import nomorals.books.failures as books_fail
import nomorals.community.failures as community_fail
import nomorals.core.failures as core_fail
from nomorals.core.storeguard import StoreUnreadableError


def _sites(module) -> list[str]:
    return [f.site for f in module.recent_failures(limit=200)]


# ── finance/alerts (2B.1) ─────────────────────────────────────────

from nomorals.finance import alerts as A  # noqa: E402


def test_alerts_corrupt_quarantined_and_mutation_refused(tmp_path):
    path = tmp_path / "alerts.json"
    store = A.AlertStore(path)
    good = A.add_alert("price_above", "BTC", 100.0, store=store)
    original = path.read_text(encoding="utf-8")
    assert good.id in original
    path.write_text("{corrupt json", encoding="utf-8")
    before = _sites(core_fail)
    # reads proceed from the empty image, loudly
    assert store.list() == []
    # the corrupt bytes were quarantined, not overwritten
    assert not path.exists()
    assert (tmp_path / "alerts.json.corrupt").read_text(
        encoding="utf-8") == "{corrupt json"
    assert "finance.alerts._load" in _sites(core_fail)
    assert _sites(core_fail) != before
    # a mutation on the poisoned instance refuses honestly
    with pytest.raises(StoreUnreadableError):
        A.add_alert("price_below", "ETH", 50.0, store=store)
    # a fresh instance starts clean; the quarantine still holds the data
    fresh = A.AlertStore(path)
    new = A.add_alert("price_below", "ETH", 50.0, store=fresh)
    assert [a.id for a in fresh.list()] == [new.id]
    assert (tmp_path / "alerts.json.corrupt").exists()


def test_alerts_wrong_shape_is_corruption(tmp_path):
    path = tmp_path / "alerts.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    store = A.AlertStore(path)
    assert store._load() == {}
    assert (tmp_path / "alerts.json.corrupt").exists()


def test_alerts_missing_and_valid_roundtrip(tmp_path):
    path = tmp_path / "alerts.json"
    store = A.AlertStore(path)
    assert store.list() == []  # missing → legitimately empty
    a = A.add_alert("price_above", "BTC", 100.0, store=store)
    assert store.get(a.id) is not None
    assert A.AlertStore(path).get(a.id).target == "BTC"
    assert not (tmp_path / "alerts.json.corrupt").exists()


# ── finance/goals (2B.2) ──────────────────────────────────────────

from nomorals.finance.goals import Goal, GoalStore  # noqa: E402


def test_goals_corrupt_quarantined_and_add_refused(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    store.add(Goal(id="a", name="Alpha", target_kobo=100))
    store.path.write_text("{corrupt json", encoding="utf-8")
    before = _sites(core_fail)
    assert store.list() == []
    assert not store.path.exists()
    assert (tmp_path / "goals.json.corrupt").read_text(
        encoding="utf-8") == "{corrupt json"
    assert "finance.goals._load" in _sites(core_fail)
    assert _sites(core_fail) != before
    with pytest.raises(StoreUnreadableError):
        store.add(Goal(id="b", name="Beta", target_kobo=200))
    # fresh instance recovers; quarantine preserved
    fresh = GoalStore(tmp_path / "goals.json")
    fresh.add(Goal(id="b", name="Beta", target_kobo=200))
    assert [g.id for g in fresh.list()] == ["b"]
    assert (tmp_path / "goals.json.corrupt").exists()


def test_goals_missing_and_valid_roundtrip(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    assert store.list() == []
    store.add(Goal(id="a", name="Alpha", target_kobo=100))
    assert [g.id for g in GoalStore(store.path).list()] == ["a"]
    assert store.mark_done("a") is True
    assert GoalStore(store.path).get("a").done is True


# ── core/trust (2B.3) ─────────────────────────────────────────────

from nomorals.core import trust as tru  # noqa: E402


class _FakeKV:
    """The get/set_raw pairing SourceTrust uses."""

    def __init__(self):
        self.raw: dict[str, str] = {}

    def get(self, key):
        value = self.raw.get(key)
        return json.loads(value) if value is not None else None

    def set_raw(self, key, raw, kind="json"):
        self.raw[key] = raw


class _CrashKV(_FakeKV):
    def get(self, key):
        raise RuntimeError("kv read boom")


def test_trust_crashed_load_never_overwrites_history():
    store = _FakeKV()
    healthy = tru.SourceTrust(None, store=store)
    healthy.feedback("https://a.com/x", ok=True)
    healthy.feedback("https://b.com/y", ok=False)
    persisted = store.raw[tru.SourceTrust._KV]
    crashing = tru.SourceTrust(None, store=_CrashKV())
    crashing._store.raw = store.raw  # same backing bytes, reads crash
    before = _sites(core_fail)
    crashing.feedback("https://c.com/z", ok=True)
    # the persisted history is byte-identical: the crashed load's
    # single-host image was refused, not written over it
    assert store.raw[tru.SourceTrust._KV] == persisted
    assert "core.trust._load" in _sites(core_fail)
    assert _sites(core_fail) != before


def test_trust_wrong_shape_quarantined_and_refused():
    store = _FakeKV()
    store.raw[tru.SourceTrust._KV] = json.dumps(["not", "a", "dict"])
    trust = tru.SourceTrust(None, store=store)
    trust.feedback("https://a.com/x", ok=True)
    # original value untouched in place; a copy sits under .corrupt
    assert json.loads(store.raw[tru.SourceTrust._KV]) == ["not", "a", "dict"]
    assert tru.SourceTrust._KV + ".corrupt" in store.raw


def test_trust_block_pins_survive_crashed_load():
    store = _CrashKV()
    store.raw[tru.SourceTrust._BLOCK_KV] = json.dumps(["old.com"])
    trust = tru.SourceTrust(None, store=store)
    trust.block("https://new.com/")
    # persisted pins byte-identical; the new pin holds for the session
    assert json.loads(store.raw[tru.SourceTrust._BLOCK_KV]) == ["old.com"]
    assert trust.is_blocked("https://new.com/") is True


def test_trust_missing_and_valid_roundtrip():
    store = _FakeKV()
    trust = tru.SourceTrust(None, store=store)
    assert trust._history("https://a.com/x") == (0, 0)  # missing → empty
    trust.feedback("https://a.com/x", ok=True)
    trust.block("https://bad.com/")
    second = tru.SourceTrust(None, store=store)
    assert second._history("https://a.com/x") == (1, 0)
    assert second.is_blocked("https://bad.com/") is True


# ── books/serial (2B.4) ───────────────────────────────────────────

import nomorals.books.serial as SER  # noqa: E402


def _serial_manager(root: Path) -> SER.SerialManager:
    mgr = object.__new__(SER.SerialManager)
    mgr.fiction = SimpleNamespace(story_dir=lambda slug: root / slug)
    return mgr


def test_serial_corrupt_meta_quarantined_and_save_refused(tmp_path):
    mgr = _serial_manager(tmp_path)
    (tmp_path / "s1").mkdir(parents=True)
    meta_file = tmp_path / "s1" / "serial.json"
    meta_file.write_text(
        '{"slug": "s1", "drift_log": [{"mark": "KEEPME"}], "wis')
    before = _sites(books_fail)
    loaded = mgr._load_meta("s1")
    assert loaded == {"slug": "s1", "drift_log": [], "wisdom_motifs": []}
    assert not meta_file.exists()
    assert "KEEPME" in (tmp_path / "s1" / "serial.json.corrupt").read_text()
    assert "books.serial._load_meta" in _sites(books_fail)
    assert _sites(books_fail) != before
    with pytest.raises(SER.SerialError):
        mgr._save_meta("s1", loaded)


def test_serial_missing_and_valid_meta(tmp_path):
    mgr = _serial_manager(tmp_path)
    (tmp_path / "s1").mkdir(parents=True)
    # missing → the default meta, and saving it works
    default = mgr._load_meta("s1")
    assert default["drift_log"] == []
    mgr._save_meta("s1", {**default, "extra": 3})
    # valid → loads whole, nothing dropped
    assert mgr._load_meta("s1")["extra"] == 3
    assert not (tmp_path / "s1" / "serial.json.corrupt").exists()


# ── agents/gig_applier (2B.5) ─────────────────────────────────────

import nomorals.agents.gig_applier as G  # noqa: E402


def _gig_store(root: Path) -> G.ApplicationStore:
    store = G.ApplicationStore(
        settings=SimpleNamespace(resolve=lambda rel: str(root)))
    assert store.data_dir == root  # resolve + create up front
    return store


def test_gig_corrupt_lines_preserved_through_save(tmp_path):
    store = _gig_store(tmp_path)
    good = G.Application(gig_id="g1", title="T1", url="http://x/1")
    (store._path()).write_text(
        json.dumps(good.to_dict()) + "\n{corrupt line\n"
        + json.dumps({"gig_id": "g2"}) + "\n")
    before = _sites(agents_fail)
    assert sorted(store.load_all()) == ["g1"]
    assert "agents.gig_applier.load_all" in _sites(agents_fail)
    assert _sites(agents_fail) != before
    store.save(G.Application(gig_id="g3", title="T3", url="http://x/3"))
    raw = (store._path()).read_text()
    # the unreadable records survive the rewrite verbatim
    assert "{corrupt line" in raw
    assert json.dumps({"gig_id": "g2"}) in raw
    assert sorted(store.load_all()) == ["g1", "g3"]


def test_gig_unreadable_file_refuses_save(tmp_path):
    store = _gig_store(tmp_path)
    (store._path()).mkdir()  # a directory where the file belongs
    assert store.load_all() == {}
    with pytest.raises(RuntimeError):
        store.save(G.Application(gig_id="g1", title="T", url="http://x/1"))


def test_gig_missing_and_valid_roundtrip(tmp_path):
    store = _gig_store(tmp_path)
    assert store.load_all() == {}
    a = G.Application(gig_id="a1", title="A", url="http://x/a")
    b = G.Application(gig_id="a2", title="B", url="http://x/b")
    store.save(a)
    store.save(b)
    assert sorted(store.load_all()) == ["a1", "a2"]


# ── agents/reflection (2B.7) ──────────────────────────────────────

import nomorals.agents.reflection as R  # noqa: E402


class _PatchFailKV:
    """First set() lands; the consistency-patch set() raises."""

    store: dict = {}
    set_calls = 0

    def __init__(self, *a, **k):
        pass

    def get(self, key, default=None):
        value = self.store.get(key, default)
        return dict(value) if isinstance(value, dict) else value

    def set(self, key, value):
        type(self).set_calls += 1
        if type(self).set_calls >= 2:
            raise RuntimeError("patch write boom")
        self.store[key] = value


class _FakeKG:
    def __init__(self, *a, **k):
        pass

    def upsert_node(self, *a, **k):
        pass

    def link(self, *a, **k):
        pass


class _FakeSkills:
    def __init__(self, *a, **k):
        pass

    def save(self, name, **k):
        return SimpleNamespace(id="sk-42", name=name)


def test_reflection_patch_failure_is_surfaced(monkeypatch):
    _PatchFailKV.store = {}
    _PatchFailKV.set_calls = 0
    monkeypatch.setattr(R, "KVStore", _PatchFailKV)
    monkeypatch.setattr("nomorals.agents.kg.KnowledgeGraph", _FakeKG)
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _FakeSkills)
    reflector = R.GoalReflector(SimpleNamespace(db=object()))
    goal = SimpleNamespace(id="g9", title="Ship it", finished_at=160.0)
    reflection = {
        "what_worked": ["w"], "what_didnt_work": [], "lessons": ["l"],
        "reusable_skill": {"name": "s-name", "kind": "strategy",
                           "body": "body"},
    }
    before = _sites(agents_fail)
    meta = reflector._persist(goal, reflection, "heuristic")
    # the skill WAS saved and meta says so…
    assert meta["skill_id"] == "sk-42"
    # …but the meta also says the stored record was NOT synced, and
    # the failure is in the ledger instead of swallowed
    assert meta["skill_id_persisted"] is False
    assert "agents.reflection._persist.skill_link" in _sites(agents_fail)
    assert _sites(agents_fail) != before
    # the physical record keeps the placeholder — now a known,
    # reported fact rather than a silent divergence
    assert _PatchFailKV.store["goal.reflection.g9"]["skill_id"] == ""


# ── tools/music save trio (2B.8) ──────────────────────────────────

import nomorals.media.song_draft as song_draft_mod  # noqa: E402
import nomorals.media.taste as taste_mod  # noqa: E402
from nomorals.tools import music as M  # noqa: E402


def _raise_save(*args, **kwargs):
    raise OSError("disk full")


def test_music_draft_save_failure_is_honest():
    key = "dataloss-draft"
    draft = SimpleNamespace(topic="lagos nights")
    before = _sites(core_fail)
    try:
        with mock.patch.object(
                taste_mod, "TasteStore",
                lambda: SimpleNamespace(profile=SimpleNamespace(
                    liked_styles=[], disliked_styles=[]))), \
             mock.patch.object(song_draft_mod, "draft_song",
                               lambda *a, **k: draft), \
             mock.patch.object(song_draft_mod, "save_draft", _raise_save), \
             mock.patch.object(song_draft_mod, "render_notebook",
                               lambda d: "NOTEBOOK"):
            out = M._draft("lagos nights", "pop", key, None)
        assert out.startswith("NOTEBOOK")
        assert "couldn't save this draft" in out
        assert "tools.music.save_draft" in _sites(core_fail)
        assert _sites(core_fail) != before
        # the session draft itself is untouched
        assert M._ACTIVE_DRAFTS[key] is draft
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)


def test_music_revise_save_failure_is_honest():
    key = "dataloss-revise"
    M._ACTIVE_DRAFTS[key] = SimpleNamespace(topic="rain")
    revised = SimpleNamespace(topic="rain, darker")
    try:
        with mock.patch.object(song_draft_mod, "revise_draft",
                               lambda *a, **k: revised), \
             mock.patch.object(song_draft_mod, "save_draft", _raise_save), \
             mock.patch.object(song_draft_mod, "render_notebook",
                               lambda d: "NOTEBOOK2"):
            out = M._revise("make it darker", key, None)
        assert out.startswith("NOTEBOOK2")
        assert "couldn't save this draft" in out
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)


def test_music_save_success_has_no_warning():
    key = "dataloss-ok"
    M._ACTIVE_DRAFTS[key] = SimpleNamespace(topic="rain")
    revised = SimpleNamespace(topic="rain, darker")
    try:
        with mock.patch.object(song_draft_mod, "revise_draft",
                               lambda *a, **k: revised), \
             mock.patch.object(song_draft_mod, "save_draft",
                               lambda d: None), \
             mock.patch.object(song_draft_mod, "render_notebook",
                               lambda d: "NOTEBOOK3"):
            out = M._revise("make it darker", key, None)
        assert out == "NOTEBOOK3"  # happy path byte-identical
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)


# ── tools/ssh_socks (2B.10) ───────────────────────────────────────

from nomorals.core.errors import ToolError  # noqa: E402
from nomorals.storage.db import Database  # noqa: E402
from nomorals.storage.kv import KVStore  # noqa: E402
from nomorals.storage.migrations import MIGRATIONS  # noqa: E402
from nomorals.storage.schema import MigrationRunner  # noqa: E402
from nomorals.tools import ssh_socks as ss  # noqa: E402


def _ssh_db():
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


def _seed_profiles(db) -> dict:
    seed = {"old": ss.SshSocksProfile(
        name="old", host="example.invalid", user="u").to_dict()}
    KVStore(db).set_raw(ss._PROFILES_KEY, json.dumps(seed), "json")
    return seed


def test_ssh_crashed_init_load_refuses_upsert_and_preserves():
    db = _ssh_db()
    _seed_profiles(db)
    before = _sites(core_fail)
    with mock.patch.object(KVStore, "get",
                           side_effect=RuntimeError("kv read dead")):
        mgr = ss.SshSocksManager(SimpleNamespace(db=db))
    assert mgr._profiles == {}  # the crashed load's degraded image
    assert "tools.ssh_socks._load_json" in _sites(core_fail)
    assert _sites(core_fail) != before
    with pytest.raises(ToolError):
        mgr.upsert_profile(name="new", host="new.invalid", user="y")
    # the stored set is byte-preserved: still exactly the seeded profile
    assert set(KVStore(db).get(ss._PROFILES_KEY)) == {"old"}


def test_ssh_missing_and_valid_roundtrip():
    db = _ssh_db()
    mgr = ss.SshSocksManager(SimpleNamespace(db=db))
    assert mgr._profiles == {}  # missing → legitimately empty
    mgr.upsert_profile(name="t1", host="example.invalid", user="u")
    fresh = ss.SshSocksManager(SimpleNamespace(db=db))
    assert set(fresh._profiles) == {"t1"}


# ── community/miniapps stores (2B.11) ─────────────────────────────

import nomorals.community.miniapps as mp  # noqa: E402


def test_miniapp_corrupt_file_quarantined(tmp_path):
    d = tmp_path / "miniapps"
    d.mkdir()
    (d / "telegram_123.json").write_text("{not json", encoding="utf-8")
    store = mp.MiniAppStore(data_dir=d)
    before = _sites(community_fail)
    assert store.load("telegram:123") == []  # never raises, reads empty
    assert not (d / "telegram_123.json").exists()
    assert (d / "telegram_123.json.corrupt").read_text(
        encoding="utf-8") == "{not json"
    assert "community.miniapps.MiniAppStore.load" in _sites(community_fail)
    assert _sites(community_fail) != before
    # the store recovers for new writes; the quarantine is untouched
    app = mp.create_miniapp("poll", "telegram:123", "Q?",
                            options=["a", "b"])
    assert store.put(app) is True
    assert [a.id for a in store.load("telegram:123")] == [app.id]
    assert (d / "telegram_123.json.corrupt").exists()


def test_miniapp_save_failure_is_honest(tmp_path):
    store = mp.MiniAppStore(data_dir=tmp_path)
    app = mp.create_miniapp("poll", "telegram:123", "Q?",
                            options=["a", "b"])
    assert store.put(app) is True
    # break the write: the .tmp path becomes a directory
    (tmp_path / "telegram_123.tmp").mkdir()
    before = _sites(community_fail)
    assert store.remove("telegram:123", app.id) is False
    assert "community.miniapps.MiniAppStore.save" in _sites(community_fail)
    assert _sites(community_fail) != before
    # nothing was removed — the honest False matches the durable fact
    assert [a.id for a in store.load("telegram:123")] == [app.id]


def test_panel_corrupt_file_quarantined_and_save_honest(tmp_path):
    d = tmp_path / "panels"
    d.mkdir()
    (d / "owner.json").write_text("[1, 2", encoding="utf-8")
    store = mp.PanelStore(data_dir=d)
    assert store.load("owner") == []
    assert (d / "owner.json.corrupt").exists()
    panel = mp.create_panel("Marathon", "owner", "strava", store=store)
    assert [p.id for p in store.load("owner")] == [panel.id]
    (d / "owner.tmp").mkdir()
    assert store.remove("owner", panel.id) is False
    assert [p.id for p in store.load("owner")] == [panel.id]


def test_miniapp_missing_and_valid_roundtrip(tmp_path):
    store = mp.MiniAppStore(data_dir=tmp_path)
    assert store.load("telegram:123") == []  # missing → empty
    app = mp.create_miniapp("poll", "telegram:123", "Q?",
                            options=["a", "b"])
    assert store.put(app) is True
    assert [a.id for a in store.load("telegram:123")] == [app.id]
    assert store.remove("telegram:123", app.id) is True
    assert store.load("telegram:123") == []
    assert not list(tmp_path.glob("*.corrupt"))
