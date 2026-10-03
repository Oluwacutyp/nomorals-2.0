"""Universal wave: free-dataset catalog additions.

Four new verified-free sources (schemas + licenses + row counts checked
live against the HuggingFace API on 2026-10-03):

* mlabonne/FineTome-100k — dense instruction SFT (100K, conversations)
* nvidia/OpenMathInstruct-2 — math reasoning (13.9M, problem/solution)
* open-thoughts/OpenThoughts-114k — reasoning traces (114K, system+conv)
* HuggingFaceFW/fineweb-edu — pretraining web text (1.5B docs, text)

Tests run fully offline: the fetch path is exercised with mocked HTTP
so the normalizers are proven against the REAL row shapes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nomorals.training import free_datasets
from nomorals.training.free_datasets import (
    FREE_DATASET_CATALOG,
    _NORMALIZERS,
    _norm_fineweb,
    _norm_openmath,
    _norm_thoughts,
    fetch_dataset,
    lookup,
)


class CatalogTests(unittest.TestCase):
    def test_new_entries_present(self) -> None:
        names = {e["name"] for e in FREE_DATASET_CATALOG}
        for name in ("finetome-100k", "openmathinstruct-2",
                     "open-thoughts-114k", "fineweb-edu"):
            self.assertIn(name, names)

    def test_new_entries_verified_and_licensed(self) -> None:
        by_name = {e["name"]: e for e in FREE_DATASET_CATALOG}
        self.assertEqual(by_name["openmathinstruct-2"]["license"], "cc-by-4.0")
        self.assertEqual(by_name["open-thoughts-114k"]["license"], "apache-2.0")
        self.assertEqual(by_name["fineweb-edu"]["license"], "odc-by")
        for name in ("finetome-100k", "openmathinstruct-2",
                     "open-thoughts-114k", "fineweb-edu"):
            self.assertTrue(by_name[name]["verified"], name)
            self.assertIn(by_name[name]["normalize"], _NORMALIZERS)

    def test_lookup_by_name_and_id(self) -> None:
        self.assertEqual(lookup("finetome-100k")["id"], "mlabonne/FineTome-100k")
        self.assertEqual(lookup("nvidia/OpenMathInstruct-2")["name"],
                         "openmathinstruct-2")
        self.assertEqual(lookup("fineweb-edu")["kind"], "pretrain")
        self.assertEqual(lookup("open-thoughts/OpenThoughts-114k")["kind"],
                         "reasoning")

    def test_kinds_cover_the_missing_layers(self) -> None:
        kinds = {e["kind"] for e in FREE_DATASET_CATALOG}
        self.assertIn("math", kinds)
        self.assertIn("reasoning", kinds)
        self.assertIn("pretrain", kinds)


class NormalizerTests(unittest.TestCase):
    def test_openmath(self) -> None:
        row = {
            "problem": "Ava has 5 granola bars. She eats 2. How many left?",
            "generated_solution": "5 - 2 = 3. Ava has 3 left.",
            "expected_answer": "3",
            "problem_source": "augmented_gsm8k",
        }
        out = _norm_openmath(row)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("granola", out["instruction"])
        self.assertIn("5 - 2 = 3", out["output"])
        self.assertEqual(out["answer"], "3")

    def test_openmath_rejects_empties(self) -> None:
        self.assertIsNone(_norm_openmath({"problem": "", "generated_solution": "x"}))
        self.assertIsNone(_norm_openmath({"problem": "p", "generated_solution": ""}))

    def test_thoughts_keeps_system(self) -> None:
        row = {
            "system": "Think step by step before answering.",
            "conversations": [
                {"from": "user", "value": "What is 2+2?"},
                {"from": "assistant", "value": "Let me think... 4."},
            ],
        }
        out = _norm_thoughts(row)
        self.assertIsNotNone(out)
        assert out is not None
        conv = out["conversations"]
        self.assertEqual(conv[0]["from"], "system")
        self.assertIn("Think step by step", conv[0]["value"])
        self.assertEqual(conv[1]["from"], "human")
        self.assertEqual(conv[2]["from"], "gpt")

    def test_thoughts_rejects_short(self) -> None:
        self.assertIsNone(_norm_thoughts({"conversations": [{"from": "user",
                                                             "value": "hi"}]}))

    def test_fineweb(self) -> None:
        out = _norm_fineweb({"text": "word " * 100})
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("text", out)
        # too short to be useful pretraining text
        self.assertIsNone(_norm_fineweb({"text": "tiny"}))


class FetchOfflineTests(unittest.TestCase):
    """The full fetch pipeline for a NEW catalog entry, with the HTTP
    layer mocked — proves wiring (lookup → normalizer → JSONL+manifest)
    without network."""

    def _fake_http(self, url: str):
        if url.startswith("https://huggingface.co/api/datasets/"):
            return {"id": "nvidia/OpenMathInstruct-2", "gated": False,
                    "private": False, "disabled": False}
        if "/info?" in url:
            return {"configs": [{"config_name": "default",
                                 "data": {"train": {}}}]}
        if "/rows?" in url:
            import urllib.parse
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            offset = int(query.get("offset", ["0"])[0])
            if offset > 0:
                return {"features": [], "num_rows_total": 13972791, "rows": []}
            return {
                "features": [{"name": "problem"}, {"name": "generated_solution"},
                             {"name": "expected_answer"}, {"name": "problem_source"}],
                "num_rows_total": 13972791,
                "rows": [
                    {"row": ["Ava has 5 bars. She eats 2. Left?",
                             "5 - 2 = 3.", "3", "augmented_gsm8k"]},
                    {"row": ["", "no problem", "", "x"]},  # dropped
                ],
            }
        raise AssertionError(f"unexpected url {url}")

    def test_fetch_openmath_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with patch.object(free_datasets, "_http_json",
                              side_effect=self._fake_http):
                out = fetch_dataset("openmathinstruct-2", Path(d), max_rows=10)
            lines = Path(out["path"]).read_text(encoding="utf-8").splitlines()
        self.assertTrue(out["ok"])
        self.assertEqual(out["rows"], 1)  # the empty row was dropped
        self.assertEqual(out["dropped"], 1)
        self.assertEqual(out["license"], "cc-by-4.0")
        row = json.loads(lines[0])
        self.assertIn("Ava", row["instruction"])
        self.assertEqual(row["answer"], "3")

    def test_fetch_unknown_still_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                fetch_dataset("definitely-not-a-real-dataset-xyz", Path(d),
                              max_rows=5)


if __name__ == "__main__":
    unittest.main()
