"""Pass tail-118 pins — nomorals/tools/database.py (3 exclusions).

The module is the read-only tool plane over the state DB (its docstring:
writes stay with the owning subsystems) — there is no write path and no
load-then-rewrite shape in it. All three sites are per-table COUNT(*)
guards in the listing tools:

- 90: ``db_tables`` — a broken table's count raises -> the table is
  still listed with the visible sentinel ``rows: -1`` (the in-code
  comment prints the contract: "a broken table still gets listed").
- 125: ``db_schema`` — same crash -> ``rows: -1`` while the columns
  (the tool's actual payload) are returned in full.
- 170: ``db_counts`` — same crash -> the table is OMITTED from the
  orientation list (the count field honestly reports what was
  measured). Inconsistent with the siblings' -1 sentinel — a shelved
  consistency sketch, not a convert: orientation display only, no
  consumer decides on it.
"""

from __future__ import annotations

import sqlite3

import pytest

from nomorals.tools import database as dbtool


class _Registry:
    def __init__(self, context):
        self.context = context
        self.handlers = {}

    def register(self, name, **meta):
        def deco(fn):
            self.handlers[name] = fn
            return fn
        return deco


class _Ctx:
    pass


class _DB:
    def __init__(self, conn, broken_in_scalar=(), count_broken=()):
        self._conn = conn
        self._broken = set(broken_in_scalar)
        self._count_broken = set(count_broken)

    def query(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def scalar(self, sql, params=(), default=None):
        for name in self._broken:
            if name in sql:
                raise RuntimeError("table locked/corrupt")
        for name in self._count_broken:
            if name in sql and "COUNT" in sql:
                raise RuntimeError("count boom")
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row else default


def _handlers(db):
    ctx = _Ctx()
    ctx.db = db
    reg = _Registry(ctx)
    dbtool.register(reg)
    return reg.handlers


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE good (id INTEGER)")
    c.executemany("INSERT INTO good VALUES (?)", [(1,), (2,), (3,)])
    c.execute("CREATE TABLE broken (id INTEGER)")
    c.commit()
    yield c
    c.close()


def test_healthy_controls(conn):
    h = _handlers(_DB(conn))
    tables = {t["name"]: t["rows"] for t in h["db_tables"]()["tables"]}
    assert tables == {"broken": 0, "good": 3}
    assert h["db_schema"]("good")["rows"] == 3
    counts = {t["name"] for t in h["db_counts"]()["tables"]}
    assert counts == {"broken", "good"}


def test_90_broken_table_listed_with_minus_one(conn):
    h = _handlers(_DB(conn, broken_in_scalar={"broken"}))
    tables = {t["name"]: t["rows"] for t in h["db_tables"]()["tables"]}
    assert tables["broken"] == -1
    assert tables["good"] == 3


def test_125_count_crash_rows_minus_one_columns_intact(conn):
    h = _handlers(_DB(conn, count_broken={"good"}))
    schema = h["db_schema"]("good")
    assert schema["rows"] == -1
    assert [c["name"] for c in schema["columns"]] == ["id"]


def test_170_broken_table_omitted_from_counts(conn):
    h = _handlers(_DB(conn, broken_in_scalar={"broken"}))
    counts = h["db_counts"]()
    names = {t["name"] for t in counts["tables"]}
    assert "broken" not in names and "good" in names
    assert counts["count"] == len(counts["tables"])
