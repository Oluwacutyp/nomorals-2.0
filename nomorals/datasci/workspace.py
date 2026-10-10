"""Data-science workspace: named pandas datasets with provenance.

A ``DataWorkspace`` holds named :class:`pandas.DataFrame` objects loaded
from CSV/JSON/JSONL/TSV/Excel files. Every dataset records its provenance
(source path, sha256, loaded_at, rows, columns) so analyses are traceable.
Querying uses pandas' own ``query``/``eval`` or read-only SQL (run through
an in-memory SQLite database — stdlib only, no extra dependency).

Beyond loading, the workspace offers dataset algebra (``query``, ``sql``,
``aggregate``, ``merge``, ``clean``), data-quality reporting
(``validate``), comparison (``compare``), export (``export``), and
matplotlib charting (``plot``) with composable style themes.

Heavy pandas/numpy imports stay inside this module; importing
``nomorals.datasci`` does not require them until a workspace is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    DatasetExists,
    DatasetNotFound,
    DataSciError,
    ExportError,
    LoadError,
    QueryError,
)
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the data workspace (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

__all__ = [
    "Dataset",
    "DataWorkspace",
    "SUPPORTED_SUFFIXES",
    "EXPORT_FORMATS",
]

SUPPORTED_SUFFIXES = (".csv", ".json", ".jsonl", ".tsv", ".xlsx", ".xls")
EXPORT_FORMATS = ("csv", "json", "jsonl", "xlsx")

# Read-only SQL: only SELECT / WITH statements may run against a dataset.
_SQL_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

# ydata-profiling-style alert thresholds (see DATASCI_SWEEP_MINING.md).
_ALERT_MISSING_PCT = 20.0
_ALERT_ZEROS_PCT = 50.0
_ALERT_SKEW = 1.0
_ALERT_CORR = 0.9
_ALERT_HIGH_CARDINALITY_PCT = 90.0

# Persistence file suffixes (parquet preferred when pyarrow is present).
_PERSIST_SUFFIXES = (".parquet", ".csv", ".meta.json")


@dataclass
class Dataset:
    """A named DataFrame plus its provenance."""

    name: str
    frame: Any  # pandas.DataFrame (imported lazily)
    source: str
    sha256: str
    loaded_at: float = field(default_factory=time.time)

    @property
    def rows(self) -> int:
        return int(self.frame.shape[0])

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def memory_bytes(self) -> int:
        try:
            return int(self.frame.memory_usage(index=True, deep=True).sum())
        except Exception:  # noqa: BLE001 - best-effort
            return 0

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "sha256": self.sha256,
            "loaded_at": self.loaded_at,
            "rows": self.rows,
            "columns": self.columns,
        }


def _pd():
    try:
        import pandas as pd
    except ImportError as exc:
        raise LoadError(
            "pandas is required for the data-science workspace "
            "(pip install pandas)") from exc
    return pd


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _sha256_frame(frame: Any) -> str:
    """Content hash for provenance. Falls back to shape+dtypes when the
    frame can't be hashed (exotic dtypes), never to a constant."""
    try:
        import pandas as pd
        hashed = pd.util.hash_pandas_object(frame, index=True)
        return hashlib.sha256(hashed.values.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 - best-effort, keep some provenance
        return hashlib.sha256(
            repr((frame.shape, [str(t) for t in frame.dtypes])).encode()
        ).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    """Normalize a scalar for JSON output (NaN/NaT/inf → None)."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    import pandas as _pd_mod
    try:
        is_na = bool(_pd_mod.isna(value))
    except (ValueError, TypeError):
        is_na = False
    if is_na:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return str(value)
    return value


def _frame_records(frame: Any, n: int | None = None) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts."""
    view = frame if n is None else frame.head(n)
    return [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in view.to_dict(orient="records")
    ]


class DataWorkspace:
    """Registry of named datasets for analysis.

    Datasets live in memory, and are persisted to ``root`` so separate CLI
    invocations share them: parquet when pyarrow is installed (exact dtypes,
    compressed), otherwise CSV plus a JSON sidecar carrying dtypes and
    provenance.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._sets: dict[str, Dataset] = {}
        self._root = Path(root) if root else None
        if self._root:
            self._root.mkdir(parents=True, exist_ok=True)
            self._restore()

    # ── persistence ───────────────────────────────────────────────────
    def _restore(self) -> None:
        """Reload persisted datasets (best-effort; corrupt files are skipped)."""
        if not self._root:
            return
        try:
            import pandas as pd
        except ImportError:
            return
        for meta_path in self._root.glob("*.meta.json"):
            name = meta_path.name[:-len(".meta.json")]
            data_path = None
            for suffix in (".parquet", ".csv"):
                cand = self._root / f"{name}{suffix}"
                if cand.is_file():
                    data_path = cand
                    break
            if data_path is None:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if data_path.suffix == ".parquet":
                    frame = pd.read_parquet(data_path)
                else:
                    frame = pd.read_csv(data_path)
                    # Restore dtypes where possible (CSV loses them); a column
                    # that won't convert keeps its inferred dtype.
                    for col, dtype in meta.get("dtypes", {}).items():
                        if col in frame.columns:
                            try:
                                frame[col] = frame[col].astype(dtype)
                            except (ValueError, TypeError):
                                continue
                self._sets[name] = Dataset(
                    name=name, frame=frame, source=meta.get("source", "?"),
                    sha256=meta.get("sha256", "?"),
                    loaded_at=meta.get("loaded_at", time.time()))
            except Exception as exc:
                # Corrupt entry: skip it but say why (debug log), don't
                # crash the whole workspace restore.
                _log.debug("skipping corrupt dataset %s: %s", name, exc)
                continue

    def _persist(self, ds: Dataset) -> None:
        if not self._root:
            return
        data_path: Path
        if _has_pyarrow():
            data_path = self._root / f"{ds.name}.parquet"
            ds.frame.to_parquet(data_path, index=False)
            # Remove a stale CSV twin from before pyarrow was installed.
            stale = self._root / f"{ds.name}.csv"
            if stale.is_file():
                stale.unlink()
        else:
            # CSV for portability (no pyarrow dependency); dtypes in the
            # sidecar.
            data_path = self._root / f"{ds.name}.csv"
            ds.frame.to_csv(data_path, index=False)
        meta = ds.provenance()
        meta["dtypes"] = {str(c): str(t) for c, t in ds.frame.dtypes.items()}
        meta["format"] = data_path.suffix.lstrip(".")
        (self._root / f"{ds.name}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8")

    def _register(self, name: str, frame: Any, source: str, *,
                  overwrite: bool = False) -> Dataset:
        """Register a derived frame with provenance + persistence + event."""
        if name in self._sets and not overwrite:
            raise DatasetExists(
                f"dataset {name!r} already loaded; drop it first or pass "
                "overwrite=True")
        if getattr(frame, "empty", True):
            raise LoadError("refusing empty result frame")
        ds = Dataset(name=name, frame=frame, source=source,
                     sha256=_sha256_frame(frame))
        self._sets[name] = ds
        self._persist(ds)
        return ds

    # ── loading ───────────────────────────────────────────────────────
    def load(self, name: str, path: str | Path, *,
             overwrite: bool = False) -> Dataset:
        """Load a CSV/JSON/JSONL/TSV/Excel file as a named dataset.

        Raises :exc:`DatasetExists` if ``name`` is taken (unless
        ``overwrite``), :exc:`LoadError` for unsupported or unreadable
        files. Fail fast — never a partial/empty dataset on error.
        """
        pd = _pd()
        p = Path(path)
        if not p.is_file():
            raise LoadError(f"not a file: {p}")
        suffix = p.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise LoadError(
                f"unsupported format {suffix!r}; supported: "
                + ", ".join(SUPPORTED_SUFFIXES))
        if name in self._sets and not overwrite:
            raise DatasetExists(
                f"dataset {name!r} already loaded; drop it first or pass "
                "overwrite=True")
        try:
            if suffix == ".csv":
                frame = pd.read_csv(p)
            elif suffix == ".tsv":
                frame = pd.read_csv(p, sep="\t")
            elif suffix in (".json", ".jsonl"):
                frame = pd.read_json(p, lines=(suffix == ".jsonl"))
            elif suffix in (".xlsx", ".xls"):
                try:
                    frame = pd.read_excel(p)
                except ImportError as exc:
                    raise LoadError(
                        "reading Excel files needs openpyxl "
                        "(pip install openpyxl)") from exc
            else:  # pragma: no cover — guarded above
                raise LoadError(f"unsupported format {suffix!r}")
        except DataSciError:
            raise
        except Exception as exc:
            raise LoadError(f"could not load {p}: {exc}") from exc
        if frame.empty:
            raise LoadError(f"{p} loaded 0 rows; refusing empty dataset")
        ds = Dataset(name=name, frame=frame, source=str(p),
                     sha256=_sha256_file(p))
        self._sets[name] = ds
        self._persist(ds)
        _emit("datasci.dataset.loaded", {
            "name": name,
            "source": str(p),
            "rows": ds.rows,
            "columns": ds.columns,
        })
        return ds

    def load_frame(self, name: str, frame: Any, *,
                   source: str = "<dataframe>",
                   overwrite: bool = False) -> Dataset:
        """Register an existing DataFrame directly (e.g., from ta/data)."""
        _pd()  # fail fast if pandas is missing
        if name in self._sets and not overwrite:
            raise DatasetExists(
                f"dataset {name!r} already loaded; drop it first or pass "
                "overwrite=True")
        if getattr(frame, "empty", True):
            raise LoadError("refusing empty DataFrame")
        ds = self._register(name, frame, source, overwrite=overwrite)
        _emit("datasci.dataset.loaded", {
            "name": name,
            "source": source,
            "rows": ds.rows,
            "columns": ds.columns,
        })
        return ds

    # ── inspection ────────────────────────────────────────────────────
    def get(self, name: str) -> Dataset:
        try:
            return self._sets[name]
        except KeyError:
            raise DatasetNotFound(
                f"no dataset {name!r}; loaded: {sorted(self._sets)}") from None

    def list_datasets(self) -> list[dict[str, Any]]:
        return [ds.provenance() for ds in self._sets.values()]

    def drop(self, name: str) -> None:
        if name not in self._sets:
            raise DatasetNotFound(f"no dataset {name!r}")
        del self._sets[name]
        if self._root:
            for suffix in _PERSIST_SUFFIXES:
                p = self._root / f"{name}{suffix}"
                if p.is_file():
                    p.unlink()
        _emit("datasci.dataset.dropped", {"name": name})

    def rename(self, old: str, new: str) -> Dataset:
        """Rename a dataset (in memory and on disk)."""
        ds = self.get(old)
        if new in self._sets:
            raise DatasetExists(
                f"dataset {new!r} already loaded; drop it first")
        del self._sets[old]
        ds.name = new
        self._sets[new] = ds
        if self._root:
            for suffix in _PERSIST_SUFFIXES:
                src = self._root / f"{old}{suffix}"
                if src.is_file():
                    src.rename(self._root / f"{new}{suffix}")
        _emit("datasci.dataset.renamed", {"old": old, "new": new})
        return ds

    def describe(self, name: str) -> dict[str, Any]:
        """Summary statistics + dtypes + data-quality overview.

        JSON-serializable. Beyond pandas ``describe()`` this adds an
        ``overview`` block (memory footprint, duplicate rows, missing
        cells — the ydata-profiling *Overview* section), a Pearson
        ``correlations`` matrix over numeric columns, and ``alerts``.
        """
        ds = self.get(name)
        frame = ds.frame
        desc = frame.describe(include="all")
        table: dict[str, dict[str, Any]] = {}
        for col in desc.columns:
            table[str(col)] = {str(idx): _json_safe(val)
                               for idx, val in desc[col].items()}
        quality = self.validate(name)
        overview = {
            "memory_bytes": ds.memory_bytes,
            "duplicate_rows": quality["duplicate_rows"],
            "missing_cells": quality["missing_cells"],
            "missing_pct": quality["missing_pct"],
            "numeric_columns": [
                c for c in ds.columns
                if str(frame[c].dtype) != "object"
                and _is_numeric_dtype(frame[c])],
            "categorical_columns": [
                c for c in ds.columns if not _is_numeric_dtype(frame[c])],
        }
        return {
            "name": name,
            "rows": ds.rows,
            "columns": ds.columns,
            "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
            "nulls": {str(c): int(frame[c].isna().sum())
                      for c in frame.columns},
            "summary": table,
            "overview": overview,
            "correlations": quality["correlations"],
            "alerts": quality["alerts"],
            "provenance": ds.provenance(),
        }

    def head(self, name: str, n: int = 5) -> list[dict[str, Any]]:
        """First ``n`` rows as plain dicts (JSON-safe)."""
        return _frame_records(self.get(name).frame, n)

    def tail(self, name: str, n: int = 5) -> list[dict[str, Any]]:
        """Last ``n`` rows as plain dicts (JSON-safe)."""
        ds = self.get(name)
        return _frame_records(ds.frame.tail(n))

    def sample(self, name: str, n: int = 5, *,
               seed: int | None = None) -> list[dict[str, Any]]:
        """Random sample of ``n`` rows (JSON-safe). ``seed`` for
        reproducibility."""
        ds = self.get(name)
        if n > ds.rows:
            raise QueryError(
                f"sample n={n} larger than dataset ({ds.rows} rows)")
        return _frame_records(ds.frame.sample(n=n, random_state=seed))

    def value_counts(self, name: str, column: str,
                     n: int = 10) -> list[dict[str, Any]]:
        """Top-``n`` values of a column with counts (JSON-safe)."""
        ds = self.get(name)
        if column not in ds.columns:
            raise QueryError(
                f"column {column!r} not in {ds.columns}")
        vc = ds.frame[column].astype(str).fillna("∅").value_counts().head(n)
        return [{"value": _json_safe(v), "count": int(c)}
                for v, c in vc.items()]

    # ── data quality ──────────────────────────────────────────────────
    def validate(self, name: str) -> dict[str, Any]:
        """Data-quality report: per-column missing/uniqueness/zeros/outlier
        stats plus ydata-profiling-style ``alerts`` (CONSTANT, MISSING,
        HIGH_CARDINALITY, SKEWED, ZEROS, HIGH_CORRELATION, DUPLICATES).

        JSON-serializable. Pure observation — never mutates the dataset.
        """
        ds = self.get(name)
        frame = ds.frame
        rows = ds.rows
        columns: dict[str, dict[str, Any]] = {}
        alerts: list[dict[str, Any]] = []

        def _alert(code: str, column: str | None, message: str) -> None:
            alerts.append({"code": code, "column": column,
                           "message": message})

        for col in frame.columns:
            s = frame[col]
            cname = str(col)
            missing = int(s.isna().sum())
            missing_pct = round(missing / rows * 100, 2) if rows else 0.0
            unique = int(s.nunique(dropna=True))
            unique_pct = round(unique / rows * 100, 2) if rows else 0.0
            info: dict[str, Any] = {
                "dtype": str(s.dtype),
                "missing": missing,
                "missing_pct": missing_pct,
                "unique": unique,
                "unique_pct": unique_pct,
            }
            if _is_numeric_dtype(s):
                sv = s.dropna()
                zeros = int((sv == 0).sum())
                zeros_pct = (round(zeros / len(sv) * 100, 2)
                             if len(sv) else 0.0)
                q1, q3 = sv.quantile(0.25), sv.quantile(0.75)
                iqr = q3 - q1
                if iqr and iqr == iqr:  # not NaN
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = int(((sv < lo) | (sv > hi)).sum())
                else:
                    outliers = 0
                skew = _json_safe(sv.skew())
                info.update({
                    "zeros": zeros,
                    "zeros_pct": zeros_pct,
                    "outliers_iqr": outliers,
                    "skew": skew,
                })
                if zeros_pct >= _ALERT_ZEROS_PCT and len(sv):
                    _alert("ZEROS", cname,
                           f"{zeros_pct}% of values are zero")
                if isinstance(skew, (int, float)) and abs(skew) >= _ALERT_SKEW:
                    _alert("SKEWED", cname, f"skew={skew:.2f}")
            else:
                top = s.astype(str).fillna("∅").value_counts().head(5)
                info["top_values"] = [
                    {"value": _json_safe(v), "count": int(c)}
                    for v, c in top.items()]
                if unique_pct >= _ALERT_HIGH_CARDINALITY_PCT and rows >= 10:
                    _alert("HIGH_CARDINALITY", cname,
                           f"{unique} distinct values in {rows} rows")
            if unique <= 1:
                _alert("CONSTANT", cname, "single distinct value")
            if missing_pct >= _ALERT_MISSING_PCT:
                _alert("MISSING", cname, f"{missing_pct}% missing")
            columns[cname] = info

        dup_rows = int(frame.duplicated().sum())
        if dup_rows:
            _alert("DUPLICATES", None, f"{dup_rows} duplicate rows")

        correlations: dict[str, dict[str, float | None]] = {}
        num = frame.select_dtypes(include="number")
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            for c1 in corr.columns:
                correlations[str(c1)] = {
                    str(c2): _json_safe(corr.loc[c1, c2])
                    for c2 in corr.columns}
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1:]:
                    val = corr.loc[c1, c2]
                    if val == val and abs(float(val)) >= _ALERT_CORR:
                        _alert("HIGH_CORRELATION", None,
                               f"{c1} ↔ {c2}: r={float(val):.3f}")

        missing_cells = int(frame.isna().sum().sum())
        total_cells = rows * len(ds.columns)
        return {
            "name": name,
            "rows": rows,
            "duplicate_rows": dup_rows,
            "missing_cells": missing_cells,
            "missing_pct": (round(missing_cells / total_cells * 100, 2)
                            if total_cells else 0.0),
            "columns": columns,
            "correlations": correlations,
            "alerts": alerts,
            "provenance": ds.provenance(),
        }

    def compare(self, a: str, b: str) -> dict[str, Any]:
        """Compare two datasets: shapes, column adds/drops, dtype changes
        (ydata-profiling's *Compare datasets*, lightweight and JSON-safe)."""
        da, db = self.get(a), self.get(b)
        cols_a, cols_b = set(da.columns), set(db.columns)
        dtype_changes = {
            c: [str(da.frame[c].dtype), str(db.frame[c].dtype)]
            for c in sorted(cols_a & cols_b)
            if str(da.frame[c].dtype) != str(db.frame[c].dtype)}
        return {
            "a": a, "b": b,
            "rows": {"a": da.rows, "b": db.rows},
            "columns": {"a": da.columns, "b": db.columns},
            "only_in_a": sorted(cols_a - cols_b),
            "only_in_b": sorted(cols_b - cols_a),
            "dtype_changes": dtype_changes,
            "common_columns": len(cols_a & cols_b),
        }

    # ── querying / algebra ────────────────────────────────────────────
    def query(self, name: str, expr: str, *,
              as_name: str | None = None) -> Dataset | Any:
        """Run a pandas ``query()`` filter.

        If ``as_name`` is given, the filtered frame is registered as a new
        dataset (provenance notes the parent + expression); otherwise the
        filtered DataFrame is returned directly.
        """
        ds = self.get(name)
        try:
            out = ds.frame.query(expr)
        except Exception as exc:
            raise QueryError(f"query {expr!r} failed: {exc}") from exc
        if as_name is None:
            return out
        child = self._register(
            as_name, out, f"{ds.source} | query({expr!r})")
        _emit("datasci.dataset.derived", {
            "name": as_name,
            "parent": name,
            "expr": expr,
            "rows": child.rows,
            "columns": child.columns,
        })
        return child

    def sql(self, name: str, statement: str, *,
            as_name: str | None = None) -> Dataset | Any:
        """Run a read-only SQL statement over a dataset.

        The frame is loaded into an in-memory SQLite database as table
        ``dataset`` (stdlib only — no DuckDB needed). Only ``SELECT`` /
        ``WITH`` statements are accepted; anything else raises
        :exc:`QueryError`. ``as_name`` registers the result as a dataset.
        """
        ds = self.get(name)
        if not _SQL_READONLY.match(statement or ""):
            raise QueryError(
                "only read-only SELECT/WITH statements are allowed")
        pd = _pd()
        try:
            with sqlite3.connect(":memory:") as conn:
                ds.frame.to_sql("dataset", conn, index=False,
                                if_exists="replace")
                out = pd.read_sql_query(statement, conn)
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"sql failed: {exc}") from exc
        if as_name is None:
            return out
        child = self._register(
            as_name, out, f"{ds.source} | sql({statement!r})")
        _emit("datasci.dataset.sql", {
            "name": as_name, "parent": name,
            "rows": child.rows, "columns": child.columns,
        })
        return child

    def aggregate(self, name: str, by: str | list[str],
                  agg: dict[str, str] | str, *,
                  as_name: str | None = None) -> Dataset | Any:
        """Group-by aggregation. ``by`` is a column (or list); ``agg`` is
        either ``{column: func}`` (funcs like ``"mean"``, ``"sum"``,
        ``"count"``, ``"nunique"``) or a single func applied to all numeric
        columns."""
        ds = self.get(name)
        by_cols = [by] if isinstance(by, str) else list(by)
        for col in by_cols:
            if col not in ds.columns:
                raise QueryError(f"group column {col!r} not in {ds.columns}")
        try:
            if isinstance(agg, str):
                num = ds.frame.select_dtypes(include="number").columns
                if not len(num):
                    raise QueryError("no numeric columns to aggregate")
                out = ds.frame.groupby(by_cols)[num].agg(agg)
            else:
                for col in agg:
                    if col not in ds.columns:
                        raise QueryError(
                            f"agg column {col!r} not in {ds.columns}")
                out = ds.frame.groupby(by_cols).agg(agg)
            out = out.reset_index()
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"aggregate failed: {exc}") from exc
        if as_name is None:
            return out
        child = self._register(
            as_name, out,
            f"{ds.source} | aggregate(by={by_cols!r}, agg={agg!r})")
        _emit("datasci.dataset.aggregated", {
            "name": as_name, "parent": name,
            "rows": child.rows, "columns": child.columns,
        })
        return child

    def merge(self, left: str, right: str, *, on: str | list[str],
              how: str = "inner", as_name: str) -> Dataset:
        """Join two datasets on shared key column(s) (pandas ``merge``).
        ``as_name`` is required — the join is always registered."""
        dl, dr = self.get(left), self.get(right)
        on_cols = [on] if isinstance(on, str) else list(on)
        for col in on_cols:
            if col not in dl.columns or col not in dr.columns:
                raise QueryError(
                    f"join key {col!r} must exist in both datasets")
        if how not in ("inner", "left", "right", "outer", "cross"):
            raise QueryError(f"unknown join type {how!r}")
        try:
            out = dl.frame.merge(dr.frame, on=on_cols if on_cols else None,
                                 how=how,
                                 suffixes=(f"_{left}", f"_{right}"))
        except Exception as exc:
            raise QueryError(f"merge failed: {exc}") from exc
        child = self._register(
            as_name, out,
            f"merge({left}, {right}, on={on_cols!r}, how={how!r})")
        _emit("datasci.dataset.merged", {
            "name": as_name, "left": left, "right": right,
            "rows": child.rows, "columns": child.columns,
        })
        return child

    def clean(self, name: str, *, drop_duplicates: bool = False,
              drop_na: str = "none",
              coerce: dict[str, str] | None = None,
              as_name: str | None = None) -> Dataset | Any:
        """Clean a dataset (pyjanitor-style verbs, explicit and reversible).

        - ``drop_duplicates``: drop fully-duplicate rows.
        - ``drop_na``: ``"none"`` | ``"any"`` (drop rows with any NA) |
          ``"all"`` (drop rows/cols that are entirely NA).
        - ``coerce``: ``{column: "numeric"|"datetime"|"string"}`` type fixes.
        Returns the cleaned frame, or registers it when ``as_name`` given.
        """
        ds = self.get(name)
        if drop_na not in ("none", "any", "all"):
            raise QueryError(f"bad drop_na mode {drop_na!r}")
        pd = _pd()
        frame = ds.frame.copy()
        steps: list[str] = []
        try:
            if drop_duplicates:
                before = len(frame)
                frame = frame.drop_duplicates()
                steps.append(f"drop_duplicates(-{before - len(frame)})")
            if drop_na != "none":
                frame = frame.dropna(how=drop_na)
                frame = frame.dropna(axis=1, how="all")
                steps.append(f"drop_na({drop_na})")
            for col, kind in (coerce or {}).items():
                if col not in frame.columns:
                    raise QueryError(
                        f"coerce column {col!r} not in {ds.columns}")
                if kind == "numeric":
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                elif kind == "datetime":
                    frame[col] = pd.to_datetime(frame[col], errors="coerce")
                elif kind == "string":
                    frame[col] = frame[col].astype("string")
                else:
                    raise QueryError(f"bad coerce kind {kind!r}")
                steps.append(f"coerce({col}->{kind})")
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"clean failed: {exc}") from exc
        if as_name is None:
            return frame
        child = self._register(
            as_name, frame,
            f"{ds.source} | clean({', '.join(steps) or 'noop'})")
        _emit("datasci.dataset.cleaned", {
            "name": as_name, "parent": name,
            "rows": child.rows, "columns": child.columns,
        })
        return child

    # ── export ────────────────────────────────────────────────────────
    def export(self, name: str, path: str | Path, *,
               format: str | None = None) -> Path:
        """Write a dataset to ``path`` (csv/json/jsonl/xlsx). Format is
        inferred from the suffix unless ``format`` is given."""
        ds = self.get(name)
        p = Path(path)
        fmt = (format or p.suffix.lstrip(".")).lower()
        if fmt not in EXPORT_FORMATS:
            raise ExportError(
                f"unsupported export format {fmt!r}; "
                f"supported: {', '.join(EXPORT_FORMATS)}")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "csv":
                ds.frame.to_csv(p, index=False)
            elif fmt == "json":
                ds.frame.to_json(p, orient="records", indent=2)
            elif fmt == "jsonl":
                ds.frame.to_json(p, orient="records", lines=True)
            elif fmt == "xlsx":
                try:
                    ds.frame.to_excel(p, index=False, engine="openpyxl")
                except ImportError as exc:
                    raise ExportError(
                        "xlsx export needs openpyxl "
                        "(pip install openpyxl)") from exc
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError(f"export to {p} failed: {exc}") from exc
        _emit("datasci.dataset.exported", {
            "name": name, "path": str(p), "format": fmt,
            "rows": ds.rows,
        })
        return p

    # ── plotting ──────────────────────────────────────────────────────
    def plot(self, name: str, kind: str, *, x: str = "", y: str = "",
             title: str = "", theme: str = "light",
             figsize: tuple[float, float] = (10, 6), dpi: int = 100,
             bins: int = 30, grid: bool = True, top_n: int = 20,
             trend: bool = False) -> bytes:
        """Render a dataset as a PNG chart (see :mod:`.plots`).

        Returns PNG bytes; the caller stores them as an artifact.
        Emits ``datasci.plot.rendered`` with the dataset provenance so
        charts stay traceable to their source data.
        """
        from .plots import render_plot
        ds = self.get(name)
        png = render_plot(ds.frame, kind, x=x, y=y, title=title,
                          theme=theme, figsize=figsize, dpi=dpi, bins=bins,
                          grid=grid, top_n=top_n, trend=trend)
        _emit("datasci.plot.rendered", {
            "name": name,
            "kind": kind,
            "x": x,
            "y": y,
            "theme": theme,
            "sha256": ds.sha256,
            "bytes": len(png),
        })
        return png


def _is_numeric_dtype(s: Any) -> bool:
    try:
        import pandas as pd
        return bool(pd.api.types.is_numeric_dtype(s.dtype))
    except Exception:  # noqa: BLE001 - best-effort
        return False
