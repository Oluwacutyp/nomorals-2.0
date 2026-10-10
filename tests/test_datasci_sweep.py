"""Sweep tests for nomorals/datasci: new features mined from ydata-profiling,
matplotlib gallery, pandas IO guides, pyjanitor/GE vocabulary, Datasette UX.

Covers: excel loading, sql(), tail/sample/value_counts, aggregate, merge,
clean, validate alerts, compare, export, rename, describe upgrades,
new plot kinds + themes + knobs. Pre-existing behavior is covered by
tests/test_datasci.py (run alongside).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nomorals.datasci import (
    THEMES,
    DataSciError,
    DatasetExists,
    DatasetNotFound,
    DataWorkspace,
    ExportError,
    LoadError,
    PlotError,
    QueryError,
    render_plot,
)


def _make_ws(tmp: str) -> DataWorkspace:
    return DataWorkspace(Path(tmp) / "ws")


def _sales(tmp: str) -> tuple[DataWorkspace, Path]:
    ws = _make_ws(tmp)
    p = Path(tmp) / "sales.csv"
    p.write_text(
        "region,product,units,price\n"
        "north,widget,10,2.5\n"
        "south,widget,5,2.5\n"
        "north,gadget,7,9.99\n"
        "south,gadget,3,9.99\n"
        "north,widget,10,2.5\n",  # duplicate row on purpose
        encoding="utf-8",
    )
    ws.load("sales", p)
    return ws, p


class ExcelLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_xlsx(self):
        p = Path(self.tmp.name) / "d.xlsx"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(
            p, index=False, engine="openpyxl")
        ws = _make_ws(self.tmp.name)
        ds = ws.load("d", p)
        self.assertEqual(ds.rows, 2)
        self.assertEqual(ds.columns, ["a", "b"])


class SqlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sql_select(self):
        out = self.ws.sql(
            "sales", "SELECT region, SUM(units) AS u FROM dataset "
            "GROUP BY region ORDER BY region")
        self.assertEqual(list(out["region"]), ["north", "south"])
        self.assertEqual(list(out["u"]), [27, 8])

    def test_sql_with_cte(self):
        out = self.ws.sql(
            "sales", "WITH x AS (SELECT * FROM dataset WHERE units > 6) "
            "SELECT COUNT(*) AS n FROM x")
        self.assertEqual(int(out["n"][0]), 3)

    def test_sql_rejects_write(self):
        with self.assertRaises(QueryError):
            self.ws.sql("sales", "DELETE FROM dataset")
        with self.assertRaises(QueryError):
            self.ws.sql("sales", "DROP TABLE dataset")

    def test_sql_bad_syntax_raises(self):
        with self.assertRaises(QueryError):
            self.ws.sql("sales", "SELECT FROM WHERE")

    def test_sql_as_name_registers(self):
        child = self.ws.sql(
            "sales", "SELECT * FROM dataset WHERE price > 5",
            as_name="expensive")
        self.assertEqual(child.rows, 2)
        self.assertIn("sql", child.source)
        # persists across instances
        ws2 = _make_ws(self.tmp.name)
        self.assertEqual(ws2.get("expensive").rows, 2)


class InspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tail(self):
        rows = self.ws.tail("sales", 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["region"], "north")

    def test_sample(self):
        rows = self.ws.sample("sales", 3, seed=42)
        self.assertEqual(len(rows), 3)
        rows2 = self.ws.sample("sales", 3, seed=42)
        self.assertEqual(rows, rows2)  # deterministic with seed

    def test_sample_too_big_raises(self):
        with self.assertRaises(QueryError):
            self.ws.sample("sales", 999)

    def test_value_counts(self):
        vc = self.ws.value_counts("sales", "region")
        self.assertEqual(vc[0], {"value": "north", "count": 3})
        self.assertEqual(vc[1], {"value": "south", "count": 2})

    def test_value_counts_bad_column_raises(self):
        with self.assertRaises(QueryError):
            self.ws.value_counts("sales", "nope")

    def test_head_tail_json_safe(self):
        json.dumps(self.ws.head("sales"))
        json.dumps(self.ws.tail("sales"))
        json.dumps(self.ws.sample("sales", 2))


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_aggregate_dict(self):
        out = self.ws.aggregate(
            "sales", "region", {"units": "sum", "price": "mean"})
        by_region = {r["region"]: r for r in
                     out.to_dict(orient="records")}
        self.assertEqual(by_region["north"]["units"], 27)
        self.assertAlmostEqual(by_region["south"]["price"], 6.245)

    def test_aggregate_str_func(self):
        out = self.ws.aggregate("sales", "product", "sum")
        self.assertIn("units", out.columns)

    def test_aggregate_as_name(self):
        child = self.ws.aggregate("sales", "region", {"units": "sum"},
                                  as_name="by_region")
        self.assertEqual(child.rows, 2)
        self.assertIn("aggregate", child.source)

    def test_aggregate_bad_column_raises(self):
        with self.assertRaises(QueryError):
            self.ws.aggregate("sales", "nope", {"units": "sum"})


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)
        self.ws.load_frame(
            "prices", pd.DataFrame({"product": ["widget", "gadget"],
                                    "cost": [1.0, 4.0]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_inner(self):
        m = self.ws.merge("sales", "prices", on="product", as_name="joined")
        self.assertEqual(m.rows, 5)
        self.assertIn("cost", m.columns)

    def test_merge_needs_as_name(self):
        with self.assertRaises(TypeError):
            self.ws.merge("sales", "prices", on="product")  # type: ignore

    def test_merge_bad_key_raises(self):
        with self.assertRaises(QueryError):
            self.ws.merge("sales", "prices", on="nope", as_name="x")

    def test_merge_duplicate_name_raises(self):
        self.ws.merge("sales", "prices", on="product", as_name="j")
        with self.assertRaises(DatasetExists):
            self.ws.merge("sales", "prices", on="product", as_name="j")


class CleanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = _make_ws(self.tmp.name)
        self.ws.load_frame("d", pd.DataFrame({
            "a": [1, 1, 2, None],
            "b": ["1.5", "1.5", "x", "2.5"],
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_drop_duplicates(self):
        out = self.ws.clean("d", drop_duplicates=True)
        self.assertEqual(len(out), 3)

    def test_clean_drop_na_any(self):
        out = self.ws.clean("d", drop_na="any")
        self.assertEqual(len(out), 3)
        self.assertFalse(out.isna().any().any())

    def test_clean_coerce_numeric(self):
        out = self.ws.clean("d", coerce={"b": "numeric"})
        self.assertTrue(pd.api.types.is_numeric_dtype(out["b"].dtype))
        self.assertTrue(out["b"].isna().sum() >= 1)  # "x" -> NaN

    def test_clean_as_name_registers(self):
        child = self.ws.clean("d", drop_duplicates=True, as_name="dc")
        self.assertEqual(child.rows, 3)
        self.assertIn("clean", child.source)

    def test_clean_bad_mode_raises(self):
        with self.assertRaises(QueryError):
            self.ws.clean("d", drop_na="sometimes")

    def test_clean_bad_coerce_raises(self):
        with self.assertRaises(QueryError):
            self.ws.clean("d", coerce={"b": "quantum"})


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = _make_ws(self.tmp.name)
        self.ws.load_frame("d", pd.DataFrame({
            "const": [7] * 10,
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # skewed + outlier
            "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 2000],  # ~perfect corr
            "z": [0] * 6 + [1, 2, 3, 4],  # 60% zeros
            "cat": list("abcdefghij"),  # high cardinality
            "miss": [1, 2, None, None, None, 6, 7, 8, 9, 10],  # 30% missing
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def test_validate_json_serializable(self):
        json.dumps(self.ws.validate("d"))

    def test_validate_alerts(self):
        codes = {a["code"] for a in self.ws.validate("d")["alerts"]}
        for expected in ("CONSTANT", "SKEWED", "ZEROS", "HIGH_CORRELATION",
                         "HIGH_CARDINALITY", "MISSING"):
            self.assertIn(expected, codes, f"missing alert {expected}")

    def test_validate_duplicates(self):
        self.ws.load_frame("e", pd.DataFrame({"a": [1, 1, 2]}))
        rep = self.ws.validate("e")
        self.assertEqual(rep["duplicate_rows"], 1)
        self.assertIn("DUPLICATES",
                      {a["code"] for a in rep["alerts"]})

    def test_validate_column_stats(self):
        col = self.ws.validate("d")["columns"]["miss"]
        self.assertEqual(col["missing"], 3)
        self.assertEqual(col["missing_pct"], 30.0)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = _make_ws(self.tmp.name)
        self.ws.load_frame("a", pd.DataFrame({"x": [1], "y": ["a"]}))
        self.ws.load_frame("b", pd.DataFrame({"x": [1.5], "z": [2]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_compare(self):
        rep = self.ws.compare("a", "b")
        self.assertEqual(rep["only_in_a"], ["y"])
        self.assertEqual(rep["only_in_b"], ["z"])
        self.assertIn("x", rep["dtype_changes"])
        self.assertEqual(rep["common_columns"], 1)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_csv_roundtrip(self):
        p = Path(self.tmp.name) / "out.csv"
        self.ws.export("sales", p)
        df = pd.read_csv(p)
        self.assertEqual(len(df), 5)

    def test_export_json_jsonl(self):
        for fmt in ("json", "jsonl"):
            p = Path(self.tmp.name) / f"out.{fmt}"
            self.ws.export("sales", p, format=fmt)
            self.assertTrue(p.is_file() and p.stat().st_size > 0)

    def test_export_xlsx_roundtrip(self):
        p = Path(self.tmp.name) / "out.xlsx"
        self.ws.export("sales", p)
        df = pd.read_excel(p, engine="openpyxl")
        self.assertEqual(len(df), 5)

    def test_export_bad_format_raises(self):
        with self.assertRaises(ExportError):
            self.ws.export("sales", Path(self.tmp.name) / "out.yaml")

    def test_export_missing_dataset_raises(self):
        with self.assertRaises(DatasetNotFound):
            self.ws.export("nope", Path(self.tmp.name) / "out.csv")


class RenameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rename(self):
        self.ws.rename("sales", "orders")
        with self.assertRaises(DatasetNotFound):
            self.ws.get("sales")
        self.assertEqual(self.ws.get("orders").rows, 5)

    def test_rename_persists(self):
        self.ws.rename("sales", "orders")
        ws2 = _make_ws(self.tmp.name)
        self.assertEqual(ws2.get("orders").rows, 5)
        with self.assertRaises(DatasetNotFound):
            ws2.get("sales")

    def test_rename_collision_raises(self):
        self.ws.load_frame("other", pd.DataFrame({"a": [1]}))
        with self.assertRaises(DatasetExists):
            self.ws.rename("sales", "other")


class DescribeUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe_has_overview(self):
        info = self.ws.describe("sales")
        ov = info["overview"]
        self.assertGreater(ov["memory_bytes"], 0)
        self.assertEqual(ov["duplicate_rows"], 1)
        self.assertIn("units", ov["numeric_columns"])
        self.assertIn("region", ov["categorical_columns"])

    def test_describe_has_correlations(self):
        info = self.ws.describe("sales")
        self.assertIn("units", info["correlations"])
        r = info["correlations"]["units"]["price"]
        self.assertIsInstance(r, float)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)

    def test_describe_has_alerts(self):
        info = self.ws.describe("sales")
        self.assertIsInstance(info["alerts"], list)
        codes = {a["code"] for a in info["alerts"]}
        self.assertIn("DUPLICATES", codes)

    def test_describe_json_serializable(self):
        json.dumps(self.ws.describe("sales"))


class PlotUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws, _ = _sales(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _png(self, kind, **kw):
        png = self.ws.plot("sales", kind, **kw)
        self.assertTrue(png.startswith(b"\x89PNG"), kind)
        self.assertGreater(len(png), 1000, kind)
        return png

    def test_new_kinds(self):
        self._png("barh", x="region", y="units")
        self._png("kde", x="units")
        self._png("box", x="region", y="units")
        self._png("violin", x="region", y="units")
        self._png("box", y="units")  # ungrouped
        self._png("heatmap")
        self._png("area", x="units", y="price")
        self._png("pie", x="region")

    def test_pie_rejects_y(self):
        # legacy contract: pie takes only --x
        with self.assertRaises(PlotError):
            render_plot(self.ws.get("sales").frame, "pie",
                        x="region", y="units")

    def test_themes(self):
        self.assertEqual(set(THEMES), {"light", "dark", "ninja", "minimal"})
        for theme in THEMES:
            self._png("line", x="units", y="price", theme=theme)

    def test_bad_theme_raises(self):
        with self.assertRaises(PlotError):
            render_plot(self.ws.get("sales").frame, "line",
                        x="units", y="price", theme="nope")

    def test_render_knobs(self):
        png_big = self.ws.plot("sales", "line", x="units", y="price",
                               figsize=(16, 9), dpi=150)
        png_small = self.ws.plot("sales", "line", x="units", y="price",
                                 figsize=(6, 4), dpi=60)
        self.assertGreater(len(png_big), len(png_small))

    def test_scatter_trend(self):
        self._png("scatter", x="units", y="price", trend=True)

    def test_bar_caps_categories(self):
        df = pd.DataFrame({"c": [f"cat{i}" for i in range(50)],
                           "v": range(50)})
        png = render_plot(df, "bar", x="c", y="v", top_n=10)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_kde_needs_numeric(self):
        with self.assertRaises(PlotError):
            render_plot(self.ws.get("sales").frame, "kde", x="region")

    def test_heatmap_needs_two_numeric(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        with self.assertRaises(PlotError):
            render_plot(df, "heatmap")

    def test_plot_emits_theme_in_event(self):
        seen = []

        from nomorals.core.events import global_bus
        def _catch(event):
            if event.topic == "datasci.plot.rendered":
                seen.append(event.data)
        sub = global_bus.subscribe("datasci.plot.rendered", _catch)
        try:
            self.ws.plot("sales", "line", x="units", y="price",
                         theme="ninja")
        finally:
            global_bus.unsubscribe(sub)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["theme"], "ninja")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_derived_dataset_persists(self):
        ws = _make_ws(self.tmp.name)
        ws.load_frame("d", pd.DataFrame({"a": [1, 2, 3]}))
        ws.query("d", "a > 1", as_name="big")
        ws2 = _make_ws(self.tmp.name)
        self.assertEqual(ws2.get("big").rows, 2)

    def test_drop_removes_all_persisted_files(self):
        ws = _make_ws(self.tmp.name)
        ws.load_frame("d", pd.DataFrame({"a": [1]}))
        root = Path(self.tmp.name) / "ws"
        ws.drop("d")
        leftovers = [p for p in root.iterdir()
                     if p.name.startswith("d.")]
        self.assertEqual(leftovers, [])

    def test_parquet_roundtrip_when_pyarrow(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow not installed")
        ws = _make_ws(self.tmp.name)
        df = pd.DataFrame({"a": pd.array([1, 2], dtype="int32"),
                           "b": ["x", "y"]})
        ws.load_frame("d", df)
        root = Path(self.tmp.name) / "ws"
        self.assertTrue((root / "d.parquet").is_file())
        ws2 = _make_ws(self.tmp.name)
        ds = ws2.get("d")
        self.assertEqual(str(ds.frame["a"].dtype), "int32")


class ErrorSurfaceTests(unittest.TestCase):
    def test_export_error_is_datasci_error(self):
        self.assertTrue(issubclass(ExportError, DataSciError))


if __name__ == "__main__":
    unittest.main()
