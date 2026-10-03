"""Data-science workspace: named pandas datasets with provenance.

A ``DataWorkspace`` holds named :class:`pandas.DataFrame` objects loaded
from CSV/JSON files. Every dataset records its provenance (source path,
sha256, loaded_at, rows, columns) so analyses are traceable. Querying
uses pandas' own ``query``/``eval`` — no custom expression language.

Heavy pandas/numpy imports stay inside this module; importing
``nomorals.datasci`` does not require them until a workspace is used.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DatasetExists, DatasetNotFound, LoadError, QueryError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "Dataset",
    "DataWorkspace",
    "SUPPORTED_SUFFIXES",
]

SUPPORTED_SUFFIXES = (".csv", ".json", ".jsonl", ".tsv")


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DataWorkspace:
    """Registry of named datasets for analysis.

    Datasets live in memory, and can be persisted to ``root`` as
    parquet files so separate CLI invocations share them.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._sets: dict[str, Dataset] = {}
        self._root = Path(root) if root else None
        if self._root:
            self._root.mkdir(parents=True, exist_ok=True)
            self._restore()

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
            csv_path = self._root / f"{name}.csv"
            if not csv_path.is_file():
                continue
            try:
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                frame = pd.read_csv(csv_path)
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
        import json
        # CSV for portability (no pyarrow dependency); dtypes in the sidecar.
        ds.frame.to_csv(self._root / f"{ds.name}.csv", index=False)
        meta = ds.provenance()
        meta["dtypes"] = {str(c): str(t) for c, t in ds.frame.dtypes.items()}
        (self._root / f"{ds.name}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8")

    # ── loading ───────────────────────────────────────────────────────
    def load(self, name: str, path: str | Path, *,
             overwrite: bool = False) -> Dataset:
        """Load a CSV/JSON/JSONL/TSV file as a named dataset.

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
        h = hashlib.sha256(repr(frame.shape).encode()).hexdigest()
        ds = Dataset(name=name, frame=frame, source=source, sha256=h)
        self._sets[name] = ds
        self._persist(ds)
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
            for suffix in (".csv", ".meta.json"):
                p = self._root / f"{name}{suffix}"
                if p.is_file():
                    p.unlink()

    def describe(self, name: str) -> dict[str, Any]:
        """Summary statistics + dtypes, JSON-serializable."""
        ds = self.get(name)
        frame = ds.frame
        desc = frame.describe(include="all")
        # describe() may contain NaN/NaT — normalize for JSON.
        import math
        def _clean(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            # pd.isna() raises on some exotic scalars (lists, sets); only
            # call it for scalar-like values.
            import pandas as _pd_mod
            try:
                is_na = bool(_pd_mod.isna(v))
            except (ValueError, TypeError):
                is_na = False
            if is_na:
                return None
            return v() if callable(getattr(v, "item", None)) else v
        table: dict[str, dict[str, Any]] = {}
        for col in desc.columns:
            table[str(col)] = {str(idx): _clean(val)
                               for idx, val in desc[col].items()}
        return {
            "name": name,
            "rows": ds.rows,
            "columns": ds.columns,
            "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
            "nulls": {str(c): int(frame[c].isna().sum())
                      for c in frame.columns},
            "summary": table,
            "provenance": ds.provenance(),
        }

    # ── querying ──────────────────────────────────────────────────────
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
        child = Dataset(name=as_name, frame=out,
                        source=f"{ds.source} | query({expr!r})",
                        sha256=ds.sha256)
        if as_name in self._sets:
            raise DatasetExists(
                f"dataset {as_name!r} already loaded; drop it first")
        self._sets[as_name] = child
        return child

    def head(self, name: str, n: int = 5) -> list[dict[str, Any]]:
        """First ``n`` rows as plain dicts (JSON-safe)."""
        ds = self.get(name)
        return ds.frame.head(n).to_dict(orient="records")
