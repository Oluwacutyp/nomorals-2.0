"""Tail-21 pins: DevonAgent failure values + NOT-recording.

nomorals/agents/devon.py has 9 true sites. Eight are exclusions
(documented bonus-context fallbacks, best-effort recording, honest
suffix encoding, documented generic fallback); one is a parent convert:

- Site 1201 (CONVERT): _tool_chat_stats used to fall through to
  "no chats recorded" on ANY db failure when no gateway was wired — a
  false empty indistinguishable from a fresh DB. It now appends a
  "chat stats failed: ..." line, the same honesty pattern the gateway
  block in the same function already uses. The convert pin fails on
  the old target (old target returns "no chats recorded").

Every pin drives its handler into the failure path and pins the exact
fallback value plus the fact that the agents failures ledger is
unchanged (snapshot before/after, tail-8 method). devon.py never calls
record_failure, and none of these failures may surface as a crash or
a success claim. No except handlers in this file (try/finally only,
for patch restore).
"""

import sys
from types import SimpleNamespace

import nomorals.agents.devon as devmod
from nomorals.agents import failures as agents_fail
from nomorals.agents.devon import DevonAgent


def _ledger():
    return list(agents_fail.recent_failures())


def _agent(ctx=None, gateway=None):
    return DevonAgent(ctx if ctx is not None else SimpleNamespace(),
                      gateway=gateway)


class _FakeBrain:
    def __init__(self, text):
        self._text = text

    def chat(self, messages, params, task_kind="chat"):
        return SimpleNamespace(ok=True, text=self._text, model="fake")


def _block_import(modname):
    real = sys.modules.get(modname)
    sys.modules[modname] = None
    return real


def _restore_import(modname, real):
    if real is not None:
        sys.modules[modname] = real
    else:
        sys.modules.pop(modname, None)


def test_llm_plan_orientation_failure_still_plans():
    # Site 476: orientation is a bonus; the plan proceeds without it.
    brain = _FakeBrain('{"steps":[{"tool":"git_status","args":{},"why":"check"}]}')
    real_brain = devmod.brain_for
    devmod.brain_for = lambda ctx: brain
    real_mod = _block_import("nomorals.agents.orientation")
    before = _ledger()
    try:
        ag = _agent(SimpleNamespace(router=object()))
        steps, _, err = ag._llm_plan("check git status", "")
    finally:
        devmod.brain_for = real_brain
        _restore_import("nomorals.agents.orientation", real_mod)
    assert steps and err == ""
    assert _ledger() == before


def test_llm_plan_type_hint_failure_still_plans():
    # Site 502: the task-type hint is a bonus; the plan proceeds.
    brain = _FakeBrain('{"steps":[{"tool":"git_status","args":{},"why":"check"}]}')
    real_brain = devmod.brain_for
    devmod.brain_for = lambda ctx: brain
    real_mod = _block_import("nomorals.agents.task_type")
    before = _ledger()
    try:
        ag = _agent(SimpleNamespace(router=object()))
        steps, _, err = ag._llm_plan("build a test script", "")
    finally:
        devmod.brain_for = real_brain
        _restore_import("nomorals.agents.task_type", real_mod)
    assert steps and err == ""
    assert _ledger() == before


def test_heuristic_plan_classifier_failure_falls_back_to_keywords():
    # Site 587: classifier is a bonus; the keyword plan still runs and
    # simply skips the build-escalation shortcut.
    real_mod = _block_import("nomorals.agents.task_type")
    before = _ledger()
    try:
        ag = _agent()
        trig = ag._heuristic_plan("build a test script")
    finally:
        _restore_import("nomorals.agents.task_type", real_mod)
    tools = [s["tool"] for s in trig]
    assert tools and "mission" not in tools
    assert _ledger() == before


def test_execute_record_failure_does_not_break_the_step():
    # Site 662: macro recording is best-effort; the step still parks
    # and the outcome claims nothing about recording.
    import nomorals.tools.macros as macros

    class FakeTools:
        def names(self):
            return ["git_status"]

    def _boom(name, args):
        raise RuntimeError("recorder down")

    real_record = macros.record_step
    macros.record_step = _boom
    before = _ledger()
    try:
        ag = _agent(SimpleNamespace(tools=FakeTools()))
        out = ag._execute("git_status", {})
    finally:
        macros.record_step = real_record
    assert out.parked
    assert "record" not in out.observation.lower()
    assert _ledger() == before


def test_llm_digest_orientation_failure_still_digests():
    # Site 821: orientation is a bonus; the digest still renders.
    brain = _FakeBrain("Found three things.")
    real_brain = devmod.brain_for
    devmod.brain_for = lambda ctx: brain
    real_mod = _block_import("nomorals.agents.orientation")
    before = _ledger()
    try:
        ag = _agent(SimpleNamespace(router=object()))
        text, model = ag._llm_digest("summarize", "git_status: clean", "", "")
    finally:
        devmod.brain_for = real_brain
        _restore_import("nomorals.agents.orientation", real_mod)
    assert text == "Found three things." and model == "fake"
    assert _ledger() == before


class _FailDB:
    def __init__(self, exc):
        self._exc = exc

    def query(self, sql, params=()):
        raise self._exc

    def query_one(self, sql, params=()):
        raise self._exc


class _RowsDB:
    def query(self, sql, params=()):
        if "game_sessions" in sql:
            return []
        return [{"platform": "telegram", "kind": "dm", "n": 3,
                 "last": 1700000000}]

    def query_one(self, sql, params=()):
        return None


def test_chat_stats_db_failure_is_surfaced_not_empty():
    # Site 1201 CONVERT: a db failure used to fall through to
    # "no chats recorded" when no gateway was wired. It now surfaces.
    before = _ledger()
    ag = _agent(SimpleNamespace(db=_FailDB(RuntimeError("db down"))))
    out = ag._tool_chat_stats({})
    assert "chat stats failed:" in out
    assert "no chats recorded" not in out
    ag_ok = _agent(SimpleNamespace(db=_RowsDB()))
    out_ok = ag_ok._tool_chat_stats({})
    assert "chats by platform:" in out_ok
    assert "chat stats failed:" not in out_ok
    assert _ledger() == before


class _TraceDB:
    def __init__(self, mood_ok):
        self._mood_ok = mood_ok

    def query(self, sql, params=()):
        if "mood_history" in sql:
            if not self._mood_ok:
                raise RuntimeError("mood table down")
            return [{"ts": 1700000000, "label": "calm", "event": "ok"}]
        return [{"role": "user", "name": "owner", "model": "",
                 "content": "hello", "created_at": 1700000000}]

    def query_one(self, sql, params=()):
        return {"id": "tg:1"}


def test_trace_mood_failure_omits_only_the_mood_section():
    # Site 1314: mood is a bonus section; the trace itself is intact.
    before = _ledger()
    ag = _agent(SimpleNamespace(db=_TraceDB(mood_ok=False)))
    out = ag._tool_trace({"chat_key": "tg:1", "limit": 8})
    assert "trace of tg:1" in out
    assert "recent mood" not in out
    assert _ledger() == before


def test_connector_guide_failure_falls_back_to_generic_guidance():
    # Site 1989: the failure value equals the documented no-instructions
    # value (generic link guidance), never a fabricated specific guide.
    import nomorals.connectors as conns

    class FakeConn:
        name = "FakeConn"
        description = "A fake connector."
        auth_methods = [SimpleNamespace(value="api_key")]
        PROVISIONABLE = ["spotify"]

        @staticmethod
        def connect_instructions(inst):
            raise RuntimeError("instructions broken")

    real_get = conns.get_connector
    conns.get_connector = lambda cid: FakeConn
    before = _ledger()
    try:
        ag = _agent()
        out = ag._tool_connector({"id": "fakeconn", "action": "info"})
    finally:
        conns.get_connector = real_get
    assert 'say "link fakeconn"' in out
    assert _ledger() == before


class _SpeakTools:
    def call(self, name, **kw):
        assert name == "speak"
        return SimpleNamespace(ok=True, value={"path": "/tmp/x.wav",
                                                "engine": "fake"})


class _SendGW:
    def __init__(self, ok):
        self._ok = ok

    def send_file(self, platform, ref, path, caption=""):
        if not self._ok:
            raise RuntimeError("send down")
        return SimpleNamespace(ok=True)


def test_speak_send_failure_omits_the_sent_suffix():
    # Site 2062: synthesis succeeded (file on disk); delivery failed.
    # The output encodes delivery with the "(sent to you)" suffix, so
    # its absence IS the honest signal — no delivery is claimed.
    before = _ledger()
    ag = _agent(SimpleNamespace(tools=_SpeakTools()),
                gateway=_SendGW(ok=False))
    ag._last_chat_key = "telegram:123"
    out = ag._tool_speak({"text": "hello"})
    assert out == "spoken with fake: /tmp/x.wav"
    ag_ok = _agent(SimpleNamespace(tools=_SpeakTools()),
                   gateway=_SendGW(ok=True))
    ag_ok._last_chat_key = "telegram:123"
    out_ok = ag_ok._tool_speak({"text": "hello"})
    assert out_ok == "spoken with fake: /tmp/x.wav (sent to you)"
    assert _ledger() == before
