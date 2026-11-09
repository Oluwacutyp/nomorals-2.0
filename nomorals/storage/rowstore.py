"""Row-shaped access to the document seam (Mongo program P3).

Domain modules speak rows: dicts keyed by column name, one primary key,
``ORDER BY`` + ``LIMIT`` reads. The seam speaks documents. This module is
the adapter between the two — and it is deliberately thin:

* Writes go through :func:`nomorals.storage.carry.row_to_document`, so a
  document written by a live port is *the same document* the P2 carry
  writes for the same row. One mapping, owned by
  :mod:`nomorals.storage.carry`, never a second ad-hoc shape.
* Reads come back through :func:`nomorals.storage.carry.document_to_row`,
  so callers receive exactly the row dicts the native SQL path returns
  (primary key restored, reserved names un-renamed, BLOBs decoded).
* Filters are expressed in row-field terms and translated here: the
  primary-key column addresses ``_id``, reserved columns rename, and
  every other field passes through to the seam's dict-filter language
  (equality plus ``$gt/$gte/$lt/$lte/$ne/$in/$exists``). Ordering beyond
  ``_id`` is the caller's job (sort the returned rows) — the seam's
  ``find`` is ``_id``-ordered by contract.
* What the adapter does NOT do: transactions. A native-path method that
  composed several writes inside ``db.transaction()`` gets several
  single-document writes here — atomic per document on both rungs,
  exactly the guarantee the seam advertises (see
  :mod:`nomorals.storage.docstore`). Domain ports state that difference
  where it matters; domains whose correctness *is* the composition stay
  on native SQL and are recorded in :data:`nomorals.storage.carry.PORT_RESIDUE`.

The pickup rule is KV's (:mod:`nomorals.storage.kv`): a domain constructs
its tables via :func:`attached_rows` with its ``Database`` handle; when
boot attached a store (Mongo, or the shadow pair) the tables are
seam-backed, and on the SQLite default — where ``db.document_store`` is
``None`` — the helper returns ``None`` and the domain runs its native SQL
path byte-for-byte as before. Resolution is lazy (per call), so an organ
built before boot attachment still routes correctly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from . import carry
from .carry import Column, TableMapping
from .failures import record_failure

__all__ = ["RowTable", "attached_rows", "mapping_for",
           "repository_encode", "repository_decode"]

#: Row-field name → document-field name for the store-managed fields.
#: Mirrors carry._RESERVED; kept as data here so filters can translate.
_RESERVED = {"_id": "sql__id", "expires_at": "sql_expires_at"}


def _default_literal(raw: Any) -> Any:
    """Parse a PRAGMA ``dflt_value`` literal to its Python value.

    Handles the literal forms this schema uses (quoted strings, numbers,
    NULL). Expression defaults (CURRENT_TIMESTAMP and friends) have no
    static value; they materialize as None here and are named as a limit
    of the adapter — a caller writing such a column passes it explicitly.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if text.upper() == "NULL":
        return None
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def column_defaults(db: Any, table: str) -> dict[str, Any]:
    """Column → DDL default value, from PRAGMA (only columns that have one)."""
    out: dict[str, Any] = {}
    for r in db.query(f'PRAGMA table_info("{table}")'):
        if r["dflt_value"] is not None:
            out[r["name"]] = _default_literal(r["dflt_value"])
    return out


def mapping_for(db: Any, table: str, *,
                extra_columns: tuple[str, ...] = ()) -> TableMapping:
    """The carry mapping for one live table (``sqlite_master`` + PRAGMA).

    Derived from the attached SQLite database's actual schema — the same
    derivation :func:`nomorals.storage.carry.inventory` performs — so a
    port can never drift from the schema the migrations (plus the owning
    module's runtime DDL, which still runs against the SQLite handle)
    actually built. ``extra_columns`` declares columns a module adds at
    runtime after this mapping is derived; they append as TEXT columns.
    """
    rows = db.query(f'PRAGMA table_info("{table}")')
    if not rows:
        raise KeyError(f"no such table in the attached database: {table!r}")
    columns = [
        Column(name=r["name"], type=r["type"] or "", pk=int(r["pk"] or 0))
        for r in rows
    ]
    known = {c.name for c in columns}
    for name in extra_columns:
        if name not in known:
            columns.append(Column(name=name, type="TEXT"))
    return TableMapping(table=table, collection=table, columns=columns)


class RowTable:
    """One table's rows over a :class:`DocumentStore`, in carry-v1 shape."""

    def __init__(self, store: Any, mapping: TableMapping, *,
                 defaults: Mapping[str, Any] | None = None) -> None:
        if mapping.table in carry.TRANSFORMS:
            raise ValueError(
                f"table {mapping.table!r} has a declared carry transform; "
                "its seam shape is owned by its own port (kv_store "
                "precedent), not by the generic row adapter")
        self._store = store
        self._mapping = mapping
        self._defaults = dict(defaults or {})
        self.collection = mapping.collection
        self._pk = mapping.pk_columns

    def _complete(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Materialize a full row: every column present, DDL defaults
        filled for the ones the caller omitted — the same row SQLite
        would store (and the carry would read back) for a native INSERT
        that lists only some columns. An explicit None stays None."""
        full: dict[str, Any] = {}
        for col in self._mapping.columns:
            if col.name in row:
                full[col.name] = row[col.name]
            elif col.name in self._defaults:
                full[col.name] = self._defaults[col.name]
        for name, value in row.items():
            if name not in full:
                full[name] = value
        return full

    # ── field translation ────────────────────────────────────────────────
    def _doc_field(self, row_field: str) -> str:
        if len(self._pk) == 1 and row_field == self._pk[0]:
            return "_id"
        return _RESERVED.get(row_field, row_field)

    def _translate_filter(self, filter: Mapping[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (filter or {}).items():
            if isinstance(value, (bytes, bytearray, memoryview)):
                value = carry.encode_value(value)
            out[self._doc_field(key)] = value
        return out

    def _pk_value(self, row: Mapping[str, Any]) -> str:
        pk = self._pk
        if len(pk) == 1:
            return str(row[pk[0]])
        if pk:
            return carry._ID_SEP.join(str(row[c]) for c in pk)
        raise ValueError(
            f"table {self._mapping.table!r} has no primary key; the row "
            "adapter cannot address its rows (carry uses rowid ids, which "
            "have no live-write meaning)")

    # ── writes ───────────────────────────────────────────────────────────
    def put(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Insert or replace one row (the seam's ``put`` is an upsert;
        call sites that need insert-or-ignore semantics guard with
        :meth:`get` first, exactly as they would read before writing)."""
        doc_id, fields, expires_at = carry.row_to_document(
            self._mapping, self._complete(row))
        self._store.put(self.collection, doc_id, fields,
                        expires_at=expires_at)
        return dict(row)

    def insert_ignore(self, row: Mapping[str, Any]) -> bool:
        """``INSERT OR IGNORE``: write only when the id is absent."""
        doc_id, fields, expires_at = carry.row_to_document(
            self._mapping, self._complete(row))
        existing = self._store.get(self.collection, doc_id,
                                   include_expired=True)
        if existing is not None:
            return False
        self._store.put(self.collection, doc_id, fields,
                        expires_at=expires_at)
        return True

    def update(self, pk_value: Any, fields: Mapping[str, Any]) -> bool:
        """``UPDATE ... WHERE pk = ?`` over row-field names."""
        translated: dict[str, Any] = {}
        for name, value in fields.items():
            if self._pk and name in self._pk:
                if str(value) != str(pk_value):
                    raise ValueError(
                        f"update may not move the primary key of "
                        f"{self._mapping.table!r}")
                continue
            translated[self._doc_field(name)] = carry.encode_value(value)
        return self._store.update_fields(
            self.collection, str(pk_value), translated)

    def inc(self, pk_value: Any, field: str, delta: float = 1.0) -> float:
        return self._store.inc(
            self.collection, str(pk_value), self._doc_field(field), delta)

    def delete(self, pk_value: Any) -> bool:
        return self._store.delete(self.collection, str(pk_value))

    def delete_where(self, filter: Mapping[str, Any] | None = None) -> int:
        return self._store.delete_many(
            self.collection, self._translate_filter(filter))

    # ── reads ────────────────────────────────────────────────────────────
    def get(self, pk_value: Any) -> dict[str, Any] | None:
        doc = self._store.get(self.collection, str(pk_value))
        return carry.document_to_row(self._mapping, doc) if doc else None

    def find(self, filter: Mapping[str, Any] | None = None, *,
             limit: int | None = None) -> list[dict[str, Any]]:
        docs = self._store.find(
            self.collection, self._translate_filter(filter), limit=limit)
        return [carry.document_to_row(self._mapping, d) for d in docs]

    def find_one(self, filter: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        rows = self.find(filter, limit=1)
        return rows[0] if rows else None

    def all(self) -> list[dict[str, Any]]:
        return self.find(None)

    def count(self, filter: Mapping[str, Any] | None = None) -> int:
        return self._store.count(
            self.collection, self._translate_filter(filter))


def attached_rows(db: Any, table: str, *,
                  extra_columns: tuple[str, ...] = ()) -> RowTable | None:
    """The seam-backed table for ``db`` + ``table``, or ``None``.

    ``None`` is the SQLite default (no store attached): the caller runs
    its native SQL path. When a store IS attached, the returned
    :class:`RowTable` is cached on the Database handle per
    (table, extra columns) so repeated calls share one mapping.
    """
    store = getattr(db, "document_store", None)
    if store is None:
        return None
    cache = getattr(db, "_row_tables", None)
    if cache is None:
        cache = {}
        try:
            db._row_tables = cache
        except Exception:  # noqa: BLE001 - a handle that refuses attrs
            cache = {}
    key = (table, extra_columns, id(store))
    table_obj = cache.get(key)
    if table_obj is None:
        defaults = column_defaults(db, table)
        for name in extra_columns:
            defaults.setdefault(name, None)
        table_obj = RowTable(
            store, mapping_for(db, table, extra_columns=extra_columns),
            defaults=defaults)
        cache[key] = table_obj
    return table_obj


# ── Repository value coding ──────────────────────────────────────────────────
#
# Organs built on storage.repository.Repository store dict/list values as
# JSON text and bools as 1/0 (Repository._encode/_decode). A registry that
# ports at registry level — its Repository stays the native-path writer —
# encodes rows for the seam with the SAME rules, so a document written by
# the port and a row written by the Repository are byte-identical. These
# two helpers are that rule, shared: they mirror Repository._encode and
# Repository._decode exactly, including keep-raw-on-bad-JSON.


def repository_encode(values: Mapping[str, Any],
                      json_columns: tuple[str, ...]) -> dict[str, Any]:
    """Encode a row dict for storage, Repository-style: values in
    ``json_columns`` — and any dict/list/tuple value anywhere — become
    JSON text; bools become 1/0; everything else passes through."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key in json_columns or isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, default=str, ensure_ascii=False)
        elif isinstance(value, bool):
            out[key] = 1 if value else 0
        else:
            out[key] = value
    return out


def repository_decode(row: Mapping[str, Any],
                      json_columns: tuple[str, ...]) -> dict[str, Any]:
    """Decode a stored row dict, Repository-style: JSON text in
    ``json_columns`` parses back to values; unparseable text is kept raw
    (and journaled), exactly as Repository._decode keeps it."""
    out = dict(row)
    for column in json_columns:
        if column in out and isinstance(out[column], str) and out[column]:
            try:
                out[column] = json.loads(out[column])
            except json.JSONDecodeError as exc:  # keep the raw value
                record_failure("rowstore.repository_decode", exc)
    return out
