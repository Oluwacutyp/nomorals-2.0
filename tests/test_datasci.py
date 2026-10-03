"""Tests for the data-science workspace (nomorals/datasci)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.datasci import (
    DataSciError,
    DatasetExists,
    DatasetNotFound,
    DataWorkspace,
    LoadError,
    QueryError,
    render_plot,
)
from nomorals.datasci.errors import PlotError


def _csv(path: Path, rows: int = 10) -> Path:
    lines = ["a,b,c"]
    for i in range(rows):
        lines.append(f"{i},{i * 2},{i * 0.5}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = DataWorkspace(Path(self.tmp.name) / "ws")

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_csv(self):
        p = _csv(Path(self.tmp.name) / "d.csv")
        ds = self.ws.load("d", p)
        self.assertEqual(ds.rows, 10)
        self.assertEqual(ds.columns, ["a", "b", "c"])
        self.assertEqual(len(ds.sha256), 64)

    def test_load_missing_file_raises(self):
        with self.assertRaises(LoadError):
            self.ws.load("x", "/nonexistent/file.csv")

    def test_load_unsupported_format_raises(self):
        p = Path(self.tmp.name) / "d.txt"
        p.write_text("hello", encoding="utf-8")
        with self.assertRaises(LoadError):
            self.ws.load("x", p)

    def test_load_duplicate_raises(self):
        p = _csv(Path(self.tmp.name) / "d.csv")
        self.ws.load("d", p)
        with self.assertRaises(DatasetExists):
            self.ws.load("d", p)

    def test_load_duplicate_overwrite(self):
        p = _csv(Path(self.tmp.name) / "d.csv")
        self.ws.load("d", p)
        ds = self.ws.load("d", p, overwrite=True)
        self.assertEqual(ds.rows, 10)

    def test_load_empty_csv_raises(self):
        p = Path(self.tmp.name) / "empty.csv"
        p.write_text("a,b,c\n", encoding="utf-8")
        with self.assertRaises(LoadError):
            self.ws.load("e", p)

    def test_persistence_across_workspaces(self):
        p = _csv(Path(self.tmp.name) / "d.csv")
        self.ws.load("d", p)
        ws2 = DataWorkspace(Path(self.tmp.name) / "ws")
        ds = ws2.get("d")
        self.assertEqual(ds.rows, 10)

    def test_drop(self):
        p = _csv(Path(self.tmp.name) / "d.csv")
        self.ws.load("d", p)
        self.ws.drop("d")
        with self.assertRaises(DatasetNotFound):
            self.ws.get("d")

    def test_drop_missing_raises(self):
        with self.assertRaises(DatasetNotFound):
            self.ws.drop("nope")

    def test_get_missing_raises(self):
        with self.assertRaises(DatasetNotFound):
            self.ws.get("nope")


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = DataWorkspace(Path(self.tmp.name) / "ws")
        self.ws.load("d", _csv(Path(self.tmp.name) / "d.csv"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe_shape(self):
        info = self.ws.describe("d")
        self.assertEqual(info["rows"], 10)
        self.assertEqual(len(info["columns"]), 3)
        self.assertIn("a", info["dtypes"])
        self.assertIn("provenance", info)

    def test_describe_json_serializable(self):
        import json
        info = self.ws.describe("d")
        json.dumps(info)  # must not raise

    def test_head(self):
        rows = self.ws.head("d", 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["a"], 0)


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = DataWorkspace(Path(self.tmp.name) / "ws")
        self.ws.load("d", _csv(Path(self.tmp.name) / "d.csv", rows=10))

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_filter(self):
        out = self.ws.query("d", "a > 5")
        self.assertEqual(len(out), 4)  # a = 6,7,8,9

    def test_query_bad_expr_raises(self):
        with self.assertRaises(QueryError):
            self.ws.query("d", "nonexistent_column > 5")

    def test_query_save_as(self):
        child = self.ws.query("d", "a > 5", as_name="big")
        self.assertEqual(child.rows, 4)
        self.assertIn("query", child.source)

    def test_query_save_duplicate_raises(self):
        self.ws.query("d", "a > 5", as_name="big")
        with self.assertRaises(DatasetExists):
            self.ws.query("d", "a > 1", as_name="big")


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = DataWorkspace(Path(self.tmp.name) / "ws")
        self.ws.load("d", _csv(Path(self.tmp.name) / "d.csv"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_line_plot(self):
        ds = self.ws.get("d")
        png = render_plot(ds.frame, "line", x="a", y="b", title="t")
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertGreater(len(png), 1000)

    def test_bar_scatter_hist(self):
        ds = self.ws.get("d")
        for kind, kw in [("bar", {"x": "a", "y": "b"}),
                         ("scatter", {"x": "a", "y": "b"}),
                         ("hist", {"x": "a"})]:
            png = render_plot(ds.frame, kind, **kw)
            self.assertTrue(png.startswith(b"\x89PNG"))

    def test_unknown_kind_raises(self):
        ds = self.ws.get("d")
        with self.assertRaises(PlotError):
            render_plot(ds.frame, "pie", x="a", y="b")

    def test_missing_column_raises(self):
        ds = self.ws.get("d")
        with self.assertRaises(PlotError):
            render_plot(ds.frame, "line", x="nope", y="b")

    def test_missing_xy_raises(self):
        ds = self.ws.get("d")
        with self.assertRaises(PlotError):
            render_plot(ds.frame, "line", x="a")


if __name__ == "__main__":
    unittest.main()
