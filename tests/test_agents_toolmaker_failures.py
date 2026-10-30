"""Tail-66 pins for nomorals/agents/toolmaker.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's own call log. The file contains
no exception handlers, so it adds zero census sites.

toolmaker.py records on the module logger for its model
paths (design/generate bind + debug-log) but calls no
failures-ledger record function at any census site, so
NOT-recording is pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-66 SURVEY.md):
- 267: temp cleanup in test()'s finally; the probe result
  is already decided and returned by the guarded body.
- 310: the mid-task check is advisory by contract
  (reasoning.mid_task_check: "advisory reasoning never
  blocks"); install never consults it, and a crashed check
  lands on the same None the field shows for an empty one.
  The evolution.py twin guard is pinned the same way
  (test_agents_evolution_failures: mid_task_check_silent).
- 355: unregister is best-effort live-registry hygiene;
  remove()'s contract is the file outcome, and the real
  ToolRegistry.unregister is a dict pop that cannot raise
  for an absent name.
- 398 / 421: suggest()'s ledger queries degrade to "no
  signal from this source" — both tables are migration-
  created, so a pre-migration DB raises on the query; the
  docstring contract is "Empty list when there is no
  signal" and each source degrades independently.
- 540: a spec_json parse boundary — every failure lands on
  the same honest {"ok": False, "error": "invalid
  spec_json"} return; nothing is installed on that path.
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as agent_fail
from nomorals.agents import reasoning as reasoning_mod
from nomorals.agents import toolmaker as tm


def _snap():
    return list(agent_fail.recent_failures(200))


_CODE = (
    "def register(registry):\n"
    "    @registry.register('probe_tool', description='p', "
    "capability='io', parameters={})\n"
    "    def probe_tool():\n"
    "        return {'ok': True}\n"
)


def _maker(db=None, tools=None):
    return tm.ToolMaker(SimpleNamespace(router=None, db=db, tools=tools))


def test_267_test_finally_cleanup_failure_keeps_result():
    before = _snap()
    rmdir_calls = []

    def boom_rmdir(self):
        rmdir_calls.append(str(self))
        raise OSError("cleanup dead")

    with mock.patch.object(tm.Path, "rmdir", boom_rmdir):
        res = _maker().test(_CODE)
    assert res.get("ok") is True and res.get("tools") == ["probe_tool"]
    assert len(rmdir_calls) == 1
    assert _snap() == before


def test_310_install_mid_task_check_failure_reports_none(tmp_path):
    before = _snap()
    calls = []

    class BoomAgent:
        def __init__(self, context):
            pass

        def mid_task_check(self, *a, **k):
            calls.append(1)
            raise RuntimeError("reasoning dead")

    good = {"action": "x", "risks": [], "advice": "", "proceed": True}

    class GoodAgent:
        def __init__(self, context):
            pass

        def mid_task_check(self, *a, **k):
            calls.append(1)
            return good

    maker = _maker()
    with mock.patch.object(tm, "_CUSTOM_DIR", tmp_path):
        with mock.patch.object(reasoning_mod, "ReasoningAgent", BoomAgent):
            res = maker.install("probe310", _CODE, tested=True)
        assert res["ok"] is True and res["mid_task_check"] is None
        assert calls == [1]
        with mock.patch.object(reasoning_mod, "ReasoningAgent", GoodAgent):
            res2 = maker.install("probe310b", _CODE, tested=True)
        assert res2["mid_task_check"] == good
        assert calls == [1, 1]
    assert _snap() == before


def test_355_remove_unregister_failure_keeps_file_outcome(tmp_path):
    before = _snap()
    unreg = []

    class BadRegistry:
        def unregister(self, name):
            unreg.append(name)
            raise RuntimeError("registry dead")

    class GoodRegistry:
        def unregister(self, name):
            unreg.append(name)
            return True

    with mock.patch.object(tm, "_CUSTOM_DIR", tmp_path):
        (tmp_path / "gone.py").write_text("x = 1\n")
        maker = _maker(tools=BadRegistry())
        assert maker.remove("gone") is True
        assert unreg == ["gone"]
        assert not (tmp_path / "gone.py").exists()
        maker2 = _maker(tools=GoodRegistry())
        assert maker2.remove("gone") is False
        assert unreg == ["gone", "gone"]
    assert _snap() == before


class _SuggestDB:
    """Serves/raises per-table so each suggest() source is driven
    independently; logs every SQL string it was asked to run."""

    def __init__(self, failures_rows=None, failures_raise=False,
                 calls_rows=None, calls_raise=False):
        self.sqls = []
        self.failures_rows = failures_rows or []
        self.failures_raise = failures_raise
        self.calls_rows = calls_rows or []
        self.calls_raise = calls_raise

    def query(self, sql, params=()):
        self.sqls.append(sql)
        if "FROM failures" in sql:
            if self.failures_raise:
                raise RuntimeError("no such table: failures")
            return self.failures_rows
        if "FROM tool_calls" in sql:
            if self.calls_raise:
                raise RuntimeError("no such table: tool_calls")
            return self.calls_rows
        return []


def test_398_suggest_failures_ledger_failure_keeps_other_source():
    before = _snap()
    db = _SuggestDB(
        failures_raise=True,
        calls_rows=[{"tool": "flaky", "total": 10, "errs": 9}])
    out = _maker(db=db).suggest()
    assert any("FROM failures" in s for s in db.sqls)
    assert [c["name"] for c in out] == ["harden_flaky"]
    assert _snap() == before


def test_421_suggest_tool_calls_failure_keeps_other_source():
    before = _snap()
    db = _SuggestDB(
        failures_rows=[{"family": "net", "n": 3, "sample": "boom sample"}],
        calls_raise=True)
    out = _maker(db=db).suggest()
    assert any("FROM tool_calls" in s for s in db.sqls)
    assert [c["name"] for c in out] == ["guard_net"]
    assert _snap() == before


def test_540_tool_create_generate_bad_spec_json_is_honest_error():
    before = _snap()

    class FakeReg:
        def __init__(self, context):
            self.context = context
            self.fns = {}

        def register(self, name, **kw):
            def dec(fn):
                self.fns[name] = fn
                return fn
            return dec

    reg = FakeReg(SimpleNamespace(router=None, db=None, tools=None))
    tm.register(reg)
    fn = reg.fns["tool_create"]
    assert fn(action="generate", spec_json="{broken json") == {
        "ok": False, "error": "invalid spec_json"}
    seen = []
    real_from_data = tm.ToolSpec.from_data

    def spy_from_data(data):
        seen.append(data)
        return real_from_data(data)

    with mock.patch.object(tm.ToolSpec, "from_data", spy_from_data):
        assert fn(action="generate", spec_json="[1, 2]") == {
            "ok": False, "error": "invalid spec_json"}
    assert seen == [[1, 2]]
    assert fn(action="generate",
              spec_json='{"name": "t1", "description": "d",'
                        ' "parameters": {}}') == {"ok": True, "code": ""}
    assert _snap() == before
