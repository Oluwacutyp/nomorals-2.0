"""Tail-98 pins — nomorals/agents/reflection.py silent-failure sites.

101 E104 — ``existing()``: KV read crash -> None, the documented
"no stored record" shape. Consumer ``reflect()`` treats None as
"run a fresh pass" and returns the fresh result with cached=False —
it redoes the work, it never claims a cache hit it did not get.
EXCLUSION.

108 E104 — ``list()``: KV scan crash -> [], the absent shape for a
listing surface. EXCLUSION.

169 E103 — ``_evidence()``: upstream-knowledge crash -> the
UPSTREAM section is simply omitted from the evidence text; the
reflection still runs on the goal's own step history. Degradation,
not fabrication. EXCLUSION.

342 E103 — ``_persist()`` final consistency patch: re-read/merge of
the real skill id into the stored record crashes -> the record keeps
its documented ``skill_id: ""`` placeholder (the first write carries
skill_name, and the caller-visible meta dict carries the true id).
WEAKEST EXCLUSION of this batch: the durable record is read back
later by ``existing()``/``list()``, so the placeholder can persist —
a persist-consistency shape, which the track protocol shelves for a
targeted fix rather than converting inside a census batch. The
counter-argument for converting (the code comment states the intent
"keep the record consistent") is noted for parent adjudication; the
placeholder is never a *wrong* id, and the skill itself is saved and
findable by name. EXCLUSION, flagged targeted-fix candidate.
"""
from types import SimpleNamespace

import nomorals.agents.reflection as R


class _BoomKV:
    """KVStore stand-in whose every operation raises."""

    def __init__(self, *a, **k):
        pass

    def get(self, *a, **k):
        raise RuntimeError("kv get boom")

    def scan(self, *a, **k):
        raise RuntimeError("kv scan boom")

    def set(self, *a, **k):
        raise RuntimeError("kv set boom")


class _DictKV:
    """KVStore stand-in backed by a plain dict (control)."""

    store: dict = {}

    def __init__(self, *a, **k):
        pass

    def get(self, key, default=None):
        return self.store.get(key, default)

    def scan(self, prefix, limit=20):
        return [(k, v) for k, v in self.store.items()
                if k.startswith(prefix)][:limit]

    def set(self, key, value):
        self.store[key] = value


def _reflector():
    return R.GoalReflector(SimpleNamespace(db=object()))


# ── 101: existing() read crash -> None ────────────────────────────

def test_101_existing_read_crash_returns_none(monkeypatch):
    monkeypatch.setattr(R, "KVStore", _BoomKV)
    assert _reflector().existing("goal-1") is None
    # control: a healthy store returns the record as-is
    _DictKV.store = {"goal.reflection.goal-1": {"lessons": ["x"]}}
    monkeypatch.setattr(R, "KVStore", _DictKV)
    assert _reflector().existing("goal-1") == {"lessons": ["x"]}


# ── 108: list() scan crash -> [] ──────────────────────────────────

def test_108_list_scan_crash_returns_empty(monkeypatch):
    monkeypatch.setattr(R, "KVStore", _BoomKV)
    assert _reflector().list() == []
    # control: healthy scan yields rows newest-first with goal_id
    _DictKV.store = {"goal.reflection.g7": {"ts": 5.0, "lessons": []}}
    monkeypatch.setattr(R, "KVStore", _DictKV)
    rows = _reflector().list()
    assert len(rows) == 1 and rows[0]["goal_id"] == "g7"


# ── 169: _evidence upstream crash -> section omitted ──────────────

def _goal():
    return SimpleNamespace(
        id="g1", title="Ship it", description="do the thing", heals=0,
        created_at=100.0, updated_at=160.0, finished_at=160.0, steps=[])


def test_169_evidence_upstream_crash_omits_section():
    class _GS:
        def upstream_knowledge(self, goal_id):
            raise RuntimeError("kg boom")

    ev = _reflector()._evidence(_GS(), _goal())
    assert "GOAL: Ship it" in ev
    assert "UPSTREAM KNOWLEDGE" not in ev
    # control: a healthy upstream read is included
    class _GS2:
        def upstream_knowledge(self, goal_id):
            return "prior facts"

    ev2 = _reflector()._evidence(_GS2(), _goal())
    assert "UPSTREAM KNOWLEDGE" in ev2 and "prior facts" in ev2


# ── 342: _persist skill-id patch crash -> placeholder stays ───────

class _PatchFailKV:
    """First set() lands; the consistency-patch set() raises."""

    store: dict = {}
    set_calls = 0

    def __init__(self, *a, **k):
        pass

    def get(self, key, default=None):
        value = self.store.get(key, default)
        # the real KVStore deserializes per read: hand back a copy so
        # in-place edits are invisible until a set() actually lands
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


def test_342_persist_patch_crash_keeps_placeholder(monkeypatch):
    _PatchFailKV.store = {}
    _PatchFailKV.set_calls = 0
    monkeypatch.setattr(R, "KVStore", _PatchFailKV)
    monkeypatch.setattr("nomorals.agents.kg.KnowledgeGraph", _FakeKG)
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _FakeSkills)
    goal = SimpleNamespace(id="g9", title="Ship it", finished_at=160.0)
    reflection = {
        "what_worked": ["w"], "what_didnt_work": [], "lessons": ["l"],
        "reusable_skill": {"name": "s-name", "kind": "strategy",
                           "body": "body"},
    }
    meta = _reflector()._persist(goal, reflection, "heuristic")
    # caller-visible meta is honest: the skill WAS saved, id is real
    assert meta["skill_id"] == "sk-42"
    # the durable record keeps the documented placeholder — the patch
    # swallow's exact, pinned effect (flagged for a targeted fix)
    stored = _PatchFailKV.store["goal.reflection.g9"]
    assert stored["skill_id"] == ""
    assert stored["skill_name"] == "s-name"
