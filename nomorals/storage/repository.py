"""Generic repository layer.

A thin typed CRUD wrapper so callers stop hand-writing SQL for the 80% case,
while still dropping to raw SQL for the 20%. Every method returns plain dicts so
nothing forces a dataclass shape on you.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from .db import Database

__all__ = ["InsertBuilder", "Query", "Repository"]


class Query:
    """Small composable SELECT builder. Deliberately not an ORM."""

    def __init__(self, table: str) -> None:
        self.table = table
        self._columns: list[str] = ["*"]
        self._where: list[str] = []
        self._params: list[Any] = []
        self._order: list[str] = []
        self._limit: int | None = None
        self._offset: int | None = None

    def select(self, *columns: str) -> "Query":
        self._columns = list(columns) or ["*"]
        return self

    def where(self, clause: str, *params: Any) -> "Query":
        self._where.append(f"({clause})")
        self._params.extend(params)
        return self

    def order_by(self, clause: str) -> "Query":
        self._order.append(clause)
        return self

    def limit(self, count: int) -> "Query":
        self._limit = max(0, int(count))
        return self

    def offset(self, count: int) -> "Query":
        self._offset = max(0, int(count))
        return self

    def build(self) -> tuple[str, list[Any]]:
        sql = f'SELECT {", ".join(self._columns)} FROM "{self.table}"'
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        if self._order:
            sql += " ORDER BY " + ", ".join(self._order)
        if self._limit is not None:
            sql += " LIMIT ?"
            params = [*self._params, self._limit]
            if self._offset:
                sql += " OFFSET ?"
                params.append(self._offset)
            return sql, params
        if self._offset is not None:
            sql += " LIMIT -1 OFFSET ?"
            return sql, [*self._params, self._offset]
        return sql, list(self._params)

    def count_sql(self) -> tuple[str, list[Any]]:
        sql = f'SELECT COUNT(*) AS n FROM "{self.table}"'
        if self._where:
            sql += " WHERE " + " AND ".join(self._where)
        return sql, list(self._params)


class InsertBuilder:
    """Multi-row INSERT batching."""

    def __init__(self, table: str, columns: Sequence[str], *, batch_size: int = 500) -> None:
        if not columns:
            raise ValidationError("insert requires at least one column")
        self.table = table
        self.columns = list(columns)
        self.batch_size = batch_size
        self._rows: list[Sequence[Any]] = []
        self.inserted = 0

    def add(self, *values: Any) -> "InsertBuilder":
        if len(values) != len(self.columns):
            raise ValidationError(
                f"expected {len(self.columns)} values, got {len(values)}"
            )
        self._rows.append(values)
        return self

    def add_many(self, rows: Iterable[Sequence[Any]]) -> "InsertBuilder":
        for row in rows:
            self.add(*row)
        return self

    def flush(self, db: Database) -> int:
        if not self._rows:
            return 0
        quoted = ", ".join('"' + c + '"' for c in self.columns)
        placeholders = ", ".join("?" for _ in self.columns)
        sql = f'INSERT INTO "{self.table}" ({quoted}) VALUES ({placeholders})'
        db.executemany(sql, self._rows)
        count = len(self._rows)
        self.inserted += count
        self._rows.clear()
        return count

    def __len__(self) -> int:
        return len(self._rows)


@dataclass
class Repository:
    """Table-scoped CRUD helper.

        repo = Repository(db, "memories", pk="id")
        row = repo.create({"kind": "fact", "content": "…"})
        repo.update(row["id"], {"importance": 0.9})
    """

    db: Database
    table: str
    pk: str = "id"
    auto_id: bool = True
    timestamp_columns: tuple[str, ...] = ("created_at", "updated_at")
    json_columns: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, key: Any) -> dict[str, Any] | None:
        row = self.db.query_one(f'SELECT * FROM "{self.table}" WHERE "{self.pk}" = ?', (key,))
        return self._decode(row) if row else None

    def require(self, key: Any) -> dict[str, Any]:
        row = self.get(key)
        if row is None:
            raise NotFound(f"{self.table} {self.pk}={key!r} not found")
        return row

    def exists(self, key: Any) -> bool:
        return (
            self.db.scalar(
                f'SELECT 1 FROM "{self.table}" WHERE "{self.pk}" = ?', (key,)
            )
            is not None
        )

    def all(self, *, order_by: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        query = self.query()
        if order_by:
            query.order_by(order_by)
        if limit is not None:
            query.limit(limit)
        sql, params = query.build()
        return [self._decode(row) for row in self.db.query(sql, params)]

    def find(self, **filters: Any) -> list[dict[str, Any]]:
        query = self.query()
        for column, value in filters.items():
            if value is None:
                query.where(f'"{column}" IS NULL')
            else:
                query.where(f'"{column}" = ?', value)
        sql, params = query.build()
        return [self._decode(row) for row in self.db.query(sql, params)]

    def find_one(self, **filters: Any) -> dict[str, Any] | None:
        rows = self.find(**filters)
        return rows[0] if rows else None

    def count(self, **filters: Any) -> int:
        query = self.query()
        for column, value in filters.items():
            if value is None:
                query.where(f'"{column}" IS NULL')
            else:
                query.where(f'"{column}" = ?', value)
        sql, params = query.count_sql()
        return int(self.db.scalar(sql, params, default=0))

    def query(self) -> Query:
        return Query(self.table)

    def paginate(self, page: int, per_page: int, *, order_by: str = "rowid") -> list[dict[str, Any]]:
        if page < 1 or per_page < 1:
            raise ValidationError("page and per_page must be >= 1")
        query = self.query().order_by(order_by).limit(per_page).offset((page - 1) * per_page)
        sql, params = query.build()
        return [self._decode(row) for row in self.db.query(sql, params)]

    # ── writes ───────────────────────────────────────────────────────────────
    def create(self, values: Mapping[str, Any], *, commit: bool = True) -> dict[str, Any]:
        payload = {**self.defaults, **values}
        if self.auto_id and self.pk not in payload:
            payload[self.pk] = new_id()
        now = time.time()
        for column in self.timestamp_columns:
            payload.setdefault(column, now)
        row_id = self.db.insert(self.table, self._encode(payload))
        key = payload.get(self.pk) or row_id
        if not commit:
            return {**payload, self.pk: key}
        return self.require(key)

    def create_many(self, rows: Iterable[Mapping[str, Any]]) -> list[str]:
        """Bulk insert inside one transaction. Returns the new primary keys."""
        materialized = list(rows)
        if not materialized:
            return []
        now = time.time()
        prepared: list[dict[str, Any]] = []
        for values in materialized:
            payload = {**self.defaults, **values}
            if self.auto_id and self.pk not in payload:
                payload[self.pk] = new_id()
            for column in self.timestamp_columns:
                payload.setdefault(column, now)
            prepared.append(payload)
        columns = sorted({key for payload in prepared for key in payload})
        builder = InsertBuilder(self.table, columns)
        with self.db.transaction():
            for payload in prepared:
                encoded = self._encode(payload)
                builder.add(*[encoded.get(c) for c in columns])
            builder.flush(self.db)
        return [payload[self.pk] for payload in prepared]

    def update(self, key: Any, changes: Mapping[str, Any]) -> int:
        payload = dict(changes)
        if "updated_at" in self.timestamp_columns:
            payload["updated_at"] = time.time()
        return self.db.update(self.table, self._encode(payload), f'"{self.pk}" = ?', (key,))

    def upsert(self, values: Mapping[str, Any]) -> dict[str, Any]:
        payload = {**self.defaults, **values}
        if self.auto_id and self.pk not in payload:
            payload[self.pk] = new_id()
        key = payload[self.pk]
        existing = self.exists(key)
        now = time.time()
        if existing:
            payload["updated_at"] = now
            self.update(key, {k: v for k, v in payload.items() if k != self.pk})
        else:
            for column in self.timestamp_columns:
                payload.setdefault(column, now)
            self.db.insert(self.table, self._encode(payload))
        return self.require(key)

    def delete(self, key: Any) -> int:
        return self.db.delete(self.table, f'"{self.pk}" = ?', (key,))

    def delete_where(self, clause: str, params: Sequence[Any] = ()) -> int:
        return self.db.delete(self.table, clause, params)

    def truncate(self) -> int:
        before = self.count()
        self.db.execute(f'DELETE FROM "{self.table}"')
        return before

    # ── encoding ─────────────────────────────────────────────────────────────
    def _encode(self, values: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in values.items():
            if key in self.json_columns or isinstance(value, (dict, list, tuple)):
                out[key] = json.dumps(value, default=str, ensure_ascii=False)
            elif isinstance(value, bool):
                out[key] = 1 if value else 0
            else:
                out[key] = value
        return out

    def _decode(self, row: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for column in self.json_columns:
            if column in out and isinstance(out[column], str) and out[column]:
                try:
                    out[column] = json.loads(out[column])
                except json.JSONDecodeError:
                    pass
        return out

    # ── bulk helpers ─────────────────────────────────────────────────────────
    def batch(self, columns: Sequence[str], *, batch_size: int = 500) -> InsertBuilder:
        return InsertBuilder(self.table, columns, batch_size=batch_size)

    def ids(self, *, limit: int | None = None) -> list[Any]:
        query = Query(self.table).select(self.pk)
        if limit is not None:
            query.limit(limit)
        sql, params = query.build()
        return [row[self.pk] for row in self.db.query(sql, params)]

    def stats(self) -> dict[str, Any]:
        return {"table": self.table, "rows": self.count()}
