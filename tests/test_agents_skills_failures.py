"""Tail-41 pins for nomorals/agents/skills.py — 0 converts / 8 exclusions.

8 tests for 8 sites (counts equal). Each test pins one exclusion site's
fallback behavior plus a control half proving the except path was taken
(trigger == fallback AND control != fallback), asserting via the db
double's call log that the failing operation itself was attempted.
The file itself contains no exception handlers, so it adds zero census
sites.

skills.py calls no failures-ledger record function anywhere
(grep-proven: no `record_failure` import or call; imports are ids +
logging only), so NOT-recording is pinned with the tail-8 snapshot
method: agents recent_failures() before/after each trigger must be
unchanged.

Site 196's pin records OBSERVED behavior only, not endorsed: it is
FLAGGED for parent adjudication (see SURVEY.md) — on a ledger-query
failure, capture returns [] and the tool wrapper reports
{"ok": True, "captured": [], "count": 0}, byte-identical to both the
supported no-ledger-yet case and a genuine zero-recovery scan. Do not
read this pin as blessing that indistinguishability.

Verdict notes (full grounds in SURVEY.md):
- 93: from_row tags parse failure -> []; metadata-only degradation,
  every other field of from_row fails loud by contrast.
- 196: FLAGGED (see above).
- 258: operator_context inner failure drops only the prevention-lessons
  part; the skills + traps parts still return.
- 260: operator_context total failure -> "" under the function's
  explicit "Never raises" contract; an internal context string,
  byte-identical to the genuine no-prior-art value.
- 443: context_block tool_sequence JSON failure -> raw body preview;
  content preserved, only the "tool -> tool" rendering is lost.
- 472/526/559: recall-surface list failure -> [] (documented
  best-effort); the decisions of every caller are identical to the
  callers' own designed failure paths.
"""
import sys
from unittest.mock import patch

sys.path.insert(0, "/home/hatch/workspace/nm-tail41-tree")

from nomorals.agents import failures as ag_fail
from nomorals.agents import skills as sk
from nomorals.agents.failure import categorize


def _snap():
    return list(ag_fail.recent_failures())


def skill_row(**over):
    row = {
        "id": "skill1", "name": "plain skill", "description": "desc",
        "kind": "strategy", "body": "body text", "tags": "alpha,beta",
        "source": "", "success_count": 3, "failure_count": 1, "uses": 4,
        "version": 1, "created_at": 1000.0, "updated_at": 1000.0,
        "last_used": 0.0, "pruned": 0,
    }
    row.update(over)
    return row


class FakeDB:
    """Scripted db double with a full call log."""

    def __init__(self, *, query_rows=None, query_exc=None):
        self.calls = []
        self.query_rows = query_rows if query_rows is not None else []
        self.query_exc = query_exc

    def query(self, sql, params=()):
        self.calls.append(("query", sql, tuple(params)))
        if self.query_exc is not None:
            raise self.query_exc
        return list(self.query_rows)

    def query_one(self, sql, params=()):
        self.calls.append(("query_one", sql, tuple(params)))
        return None

    def execute(self, sql, params=()):
        self.calls.append(("execute", sql, tuple(params)))
        return None


def test_93_from_row_tags_failure_yields_empty_tags():
    before = _snap()
    bad = sk.Skill.from_row(skill_row(tags=12345))
    assert bad.tags == []
    assert bad.name == "plain skill"  # the row otherwise survives intact
    good = sk.Skill.from_row(skill_row(tags="alpha, beta"))
    assert good.tags == ["alpha", "beta"]
    assert _snap() == before


def test_196_capture_from_ledger_query_failure_returns_empty():
    # FLAGGED — observed behavior only, not endorsed (see docstring).
    before = _snap()
    db = FakeDB(query_exc=RuntimeError("no such table: tool_calls"))
    out = sk.SkillLibrary(db).capture_from_ledger()
    assert out == []
    # the failing operation itself was attempted, with the ledger SQL
    assert any(c[0] == "query" and "tool_calls" in c[1] for c in db.calls)
    # control: a real recovery pair in the ledger IS captured
    fam = categorize("boom failure")
    name = "retry sometool after " + fam + " error"
    rows = [
        {"actor": "a1", "tool": "sometool", "status": "ok", "error": "",
         "created_at": 1010.0},
        {"actor": "a1", "tool": "sometool", "status": "error",
         "error": "boom failure", "created_at": 1000.0},
    ]
    db2 = FakeDB(query_rows=rows)
    saved_row = skill_row(id="skillX", name=name, kind="solution")

    def q1(sql, params=()):
        db2.calls.append(("query_one", sql, tuple(params)))
        if "WHERE name" in sql:
            return None  # not yet saved -> INSERT path
        return saved_row  # get(skill_id) after save

    db2.query_one = q1
    out2 = sk.SkillLibrary(db2).capture_from_ledger()
    assert out2 == [name]
    assert any(c[0] == "execute" and "INSERT INTO skills" in c[1]
               for c in db2.calls)
    assert _snap() == before


def test_258_operator_context_enrich_failure_drops_only_lessons_part():
    before = _snap()
    row = skill_row(name="proven approach", body="do the thing")
    lib = sk.SkillLibrary(FakeDB(query_rows=[row]))
    with patch("nomorals.agents.failure.enrich_with_lessons",
               side_effect=RuntimeError("lessons down")):
        out = lib.operator_context("proven approach")
    assert "Relevant proven skills" in out  # other parts survive
    assert "PREVENTION" not in out
    with patch("nomorals.agents.failure.enrich_with_lessons",
               return_value="PREVENTION BLOCK"):
        out2 = lib.operator_context("proven approach")
    assert "PREVENTION BLOCK" in out2  # control: part present when it works
    assert _snap() == before


def test_260_operator_context_failure_returns_empty_string():
    before = _snap()
    db = FakeDB(query_exc=RuntimeError("db down"))
    out = sk.SkillLibrary(db).operator_context("anything")
    assert out == ""
    assert any(c[0] == "query" and "FROM skills" in c[1] for c in db.calls)
    row = skill_row(name="proven approach", body="do the thing")
    out2 = sk.SkillLibrary(FakeDB(query_rows=[row])).operator_context(
        "proven approach")
    assert "Relevant proven skills" in out2  # control differs
    assert _snap() == before


def test_443_context_block_bad_tool_sequence_json_keeps_raw_preview():
    before = _snap()
    row = skill_row(name="seq skill", kind="tool_sequence",
                    body='[{"tool": "search"}, {"tool": "fetch"}]')
    out = sk.SkillLibrary(FakeDB(query_rows=[row])).context_block("seq skill")
    assert "search → fetch" in out  # control: parsed preview
    bad = skill_row(name="seq skill", kind="tool_sequence",
                    body='corrupt body [{"tool": ')
    out2 = sk.SkillLibrary(FakeDB(query_rows=[bad])).context_block("seq skill")
    assert "corrupt body" in out2  # raw preview survives
    assert "→" not in out2
    assert _snap() == before


def test_472_systemic_traps_list_failure_returns_empty():
    before = _snap()
    db = FakeDB(query_exc=RuntimeError("db down"))
    out = sk.SkillLibrary(db).systemic_traps()
    assert out == []
    assert any(c[0] == "query" and "FROM skills" in c[1] for c in db.calls)
    rows = [skill_row(id=f"s{i}", name="fix-modulenotfounderror-task",
                      kind="code", source="coding_session")
            for i in range(3)]
    out2 = sk.SkillLibrary(FakeDB(query_rows=rows)).systemic_traps()
    assert len(out2) == 1 and out2[0]["family"] == "modulenotfounderror"
    assert _snap() == before


def test_526_match_errors_list_failure_returns_empty():
    before = _snap()
    db = FakeDB(query_exc=RuntimeError("db down"))
    out = sk.SkillLibrary(db).match_errors("ModuleNotFoundError: No module")
    assert out == []
    assert any(c[0] == "query" and "FROM skills" in c[1] for c in db.calls)
    row = skill_row(name="fix-modulenotfounderror-foo", kind="code",
                    source="coding_session")
    out2 = sk.SkillLibrary(FakeDB(query_rows=[row])).match_errors(
        "ModuleNotFoundError: No module named foo")
    assert [s.name for s in out2] == ["fix-modulenotfounderror-foo"]
    assert _snap() == before


def test_559_match_preventions_list_failure_returns_empty():
    before = _snap()
    db = FakeDB(query_exc=RuntimeError("db down"))
    out = sk.SkillLibrary(db).match_preventions("delete production database")
    assert out == []
    assert any(c[0] == "query" and "FROM skills" in c[1] for c in db.calls)
    row = skill_row(name="abort delete production database", kind="prevention",
                    description="stop",
                    body="never delete production database")
    out2 = sk.SkillLibrary(FakeDB(query_rows=[row])).match_preventions(
        "delete production database")
    assert [s.name for s in out2] == ["abort delete production database"]
    assert _snap() == before
