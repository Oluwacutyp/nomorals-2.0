"""Wave 69b — zero-phone-data Colab pipeline.

Covers:
* dataset CONFIG-suffix fetches (`org/name:Config`) — the UltraData
  Code/General/Search split, per-config files, no clobber;
* the self-contained Colab NOTEBOOK generator (streams datasets inside
  Colab, persona on every row, abliterated base, LoRA GGUF out) with
  valid-python cells and the exact settings baked in;
* `nm data mix` now emitting the notebook next to the bundle;
* the local llama.cpp server's first-class LoRA support (`--lora`
  argv, missing-file diagnostics) and `NM_LLM_LOCAL_LORA`;
* `nm models --promote-local <base> --lora <file>` writing .env.

All tests are hermetic (no network); CLI e2e runs in a private NM_HOME.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nomorals.core.config import load_settings
from nomorals.training.finetune import (COLAB_BASE_MODELS, DEFAULT_COLAB_BASE,
                                        DEFAULT_COLAB_SOURCES,
                                        write_colab_notebook,
                                        write_colab_script)
from nomorals.training import free_datasets

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) for c in nb["cells"])


# ── dataset config-suffix fetches ────────────────────────────────────────────


class ConfigSuffixTest(unittest.TestCase):
    def test_split_config(self) -> None:
        self.assertEqual(free_datasets._split_config("org/repo"),
                         ("org/repo", ""))
        self.assertEqual(free_datasets._split_config("org/repo:Code-Agent"),
                         ("org/repo", "Code-Agent"))
        self.assertEqual(
            free_datasets._split_config(" openbmb/UltraData-SFT-Agent-2609 : General-Agent "),
            ("openbmb/UltraData-SFT-Agent-2609", "General-Agent"))

    def test_lookup_carries_config(self) -> None:
        entry = free_datasets.lookup("ultra-data-agent:Code-Agent")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["config"], "Code-Agent")
        self.assertEqual(entry["id"], "openbmb/UltraData-SFT-Agent-2609")
        plain = free_datasets.lookup("ultra-data-agent")
        self.assertIsNotNone(plain)
        self.assertNotIn("config", plain)

    def test_ad_hoc_carries_config(self) -> None:
        entry = free_datasets._ad_hoc("openbmb/UltraData-SFT-Agent-2609:Code-Agent")
        self.assertEqual(entry["id"], "openbmb/UltraData-SFT-Agent-2609")
        self.assertEqual(entry["config"], "Code-Agent")
        self.assertEqual(free_datasets._ad_hoc("bad ref"), {})

    def test_server_splits_selects_requested_config(self) -> None:
        info = {"configs": [
            {"config_name": "Code-Agent",
             "data": {"train": {"num_rows": 22770}}},
            {"config_name": "General-Agent",
             "data": {"train": {"num_rows": 38581}}},
            {"config_name": "Tool-Use",
             "data": {"train": {"num_rows": 999, "partial": True}}},
        ]}
        original = free_datasets._http_json

        def fake(url: str, timeout: float = 30.0) -> Any:
            self.assertIn("/info?", url)
            return info

        free_datasets._http_json = fake
        try:
            self.assertEqual(free_datasets._server_splits(
                "openbmb/UltraData-SFT-Agent-2609", "General-Agent"),
                ("General-Agent", "train"))
            self.assertEqual(free_datasets._server_splits(
                "openbmb/UltraData-SFT-Agent-2609"),
                ("Code-Agent", "train"))  # first config when unrequested
            with self.assertRaises(ValueError):
                free_datasets._server_splits(
                    "openbmb/UltraData-SFT-Agent-2609", "Nope-Agent")
        finally:
            free_datasets._http_json = original

    def test_fetch_writes_config_suffixed_file(self) -> None:
        """Two configs of one dataset must land in two different files."""
        dest = Path(tempfile.mkdtemp(prefix="nm-w69b-fetch-"))
        rows_by_config: dict[str, list[dict[str, Any]]] = {
            "Code-Agent": [
                {"messages": json.dumps([
                    {"role": "user", "content": "fix the parser bug"},
                    {"role": "assistant", "content": "patch applied and tested"}
                ]), "tools": "[]"},
            ] * 3,
        }
        state = {"offset": 0}

        def fake_info(dataset_id: str) -> Any:
            self.assertEqual(dataset_id, "openbmb/UltraData-SFT-Agent-2609")
            return {"gated": None, "private": False, "disabled": False,
                    "siblings": []}

        def fake_splits(dataset_id: str, config_name: str = "") -> tuple[str, str]:
            self.assertEqual(dataset_id, "openbmb/UltraData-SFT-Agent-2609")
            self.assertEqual(config_name, "Code-Agent")
            return "Code-Agent", "train"

        def fake_rows(dataset_id: str, config: str, split: str,
                      offset: int, length: int) -> dict[str, Any]:
            self.assertEqual(config, "Code-Agent")
            rows = rows_by_config["Code-Agent"]
            page = rows[offset:offset + length]
            state["offset"] = offset + length
            # `messages` is already a JSON *string* (the rows API shape)
            return {"features": [{"name": "messages"}, {"name": "tools"}],
                    "num_rows_total": len(rows),
                    "rows": [{"row": [r["messages"], r["tools"]]}
                              for r in page]}

        orig_info = free_datasets._api_info
        orig_splits = free_datasets._server_splits
        orig_rows = free_datasets._server_rows
        free_datasets._api_info = fake_info
        free_datasets._server_splits = fake_splits
        free_datasets._server_rows = fake_rows
        try:
            out = free_datasets.fetch_dataset(
                "openbmb/UltraData-SFT-Agent-2609:Code-Agent",
                dest, max_rows=3)
        finally:
            free_datasets._api_info = orig_info
            free_datasets._server_splits = orig_splits
            free_datasets._server_rows = orig_rows
        self.assertEqual(out["rows"], 3)
        self.assertEqual(out["config"], "Code-Agent")
        # the file must carry the config — no clobber of other configs
        self.assertTrue(out["path"].endswith(
            "ultradata-sft-agent-2609_code-agent.jsonl")
            or "code-agent" in Path(out["path"]).name)
        self.assertTrue(Path(out["path"]).is_file())
        manifest = json.loads(
            (Path(out["path"]).with_suffix(".manifest.json")).read_text())
        self.assertEqual(manifest["config"], "Code-Agent")


# ── wave 69e: verified-scorcha schema + catalog corrections ─────────────────


class VerifiedSchemaTest(unittest.TestCase):
    def test_norm_hermes_reads_current_conversations_schema(self) -> None:
        # verified 2026-09-12: the CURRENT OpenHermes-2.5 release is
        # ShareGPT `conversations` (+ system_prompt) — the old
        # prompt/completion-only path fetched 0 rows
        row = {
            "conversations": [
                {"from": "system", "value": "sys prompt here"},
                {"from": "human", "value": "broad topic question"},
                {"from": "gpt", "value": "a real broad answer"},
                {"from": "human", "value": "follow-up question"},
                {"from": "gpt", "value": "the follow-up answer"},
            ],
            "system_prompt": "sys prompt here",
            "topic": "science",
        }
        out = free_datasets._norm_hermes(row)
        self.assertIsNotNone(out, "current schema must produce rows")
        conv = out["conversations"]
        # the phone-side normalizer KEEPS the source system turn (the mix
        # builder's apply_persona replaces it downstream) — but the
        # row-level system_prompt must not be DOUBLE-inserted
        self.assertEqual([c["from"] for c in conv],
                         ["system", "human", "gpt", "human", "gpt"])
        self.assertEqual(
            sum(1 for c in conv if c["from"] == "system"), 1)
        self.assertEqual(conv[1]["value"], "broad topic question")
        self.assertEqual(conv[3]["value"], "follow-up question")

    def test_norm_hermes_reads_legacy_prompt_completion(self) -> None:
        row = {"system": "legacy sys", "prompt": "old question",
               "prompt2": "old follow", "completion": "old answer"}
        out = free_datasets._norm_hermes(row)
        self.assertIsNotNone(out)
        conv = out["conversations"]
        self.assertEqual([c["from"] for c in conv],
                         ["system", "human", "human", "gpt"])
        self.assertEqual(conv[-1]["value"], "old answer")

    def test_norm_hermes_conversations_without_system_prompt(self) -> None:
        row = {"conversations": [
            {"from": "human", "value": "question without system"},
            {"from": "gpt", "value": "answer without system"},
        ], "system_prompt": ""}
        out = free_datasets._norm_hermes(row)
        self.assertIsNotNone(out)
        self.assertEqual([c["from"] for c in out["conversations"]],
                         ["human", "gpt"])

    def test_dolphin_catalog_points_at_verified_mirror(self) -> None:
        # the original cognitivecomputations/dolphin2.9 dataset was
        # deleted; Skorcht/dolphin2.9 is the verified mirror (456,361
        # rows, conversations schema)
        entry = free_datasets.lookup("dolphin-2.9")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["id"], "Skorcht/dolphin2.9")
        self.assertTrue(entry["verified"])
        self.assertIn("456,361", entry["size"])

    def test_open_hermes_catalog_verified_size(self) -> None:
        entry = free_datasets.lookup("open-hermes-25")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["id"], "teknium/OpenHermes-2.5")
        self.assertTrue(entry["verified"])
        self.assertIn("1,001,551", entry["size"])

    def test_norm_m4_reads_sharegpt_dolphin_rows(self) -> None:
        # Skorcht/dolphin2.9 rows: conversations = [{from, value}]
        row = {"conversations": [
            {"from": "system", "value": "dolphin system"},
            {"from": "human", "value": "dolphin style question"},
            {"from": "gpt", "value": "no-lecture dolphin style answer"},
        ]}
        out = free_datasets._norm_m4(row)
        self.assertIsNotNone(out)
        self.assertEqual(len(out["conversations"]), 3)
        self.assertEqual(out["conversations"][1]["value"],
                         "dolphin style question")


# ── the self-contained Colab notebook ────────────────────────────────────────


class NotebookGenerationTest(unittest.TestCase):
    def _generate(self, **kwargs: Any) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-nb-"))
        path = Path(write_colab_notebook(tmp / "persona-mix", **kwargs))
        self.assertTrue(path.is_file())
        nb = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(len(nb["cells"]), 16)
        return path

    def test_reference_doc_in_branch_is_generated_shape(self) -> None:
        doc = REPO_ROOT / "docs" / "nomorals-qlora.ipynb"
        self.assertTrue(doc.is_file())
        nb = json.loads(doc.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        text = _notebook_text(doc)
        for needle in (DEFAULT_COLAB_BASE, "streaming=True",
                       "convert_lora_to_gguf", "promote-local",
                       "persona-lora.gguf"):
            self.assertIn(needle, text)

    def test_default_base_is_abliterated_v2(self) -> None:
        # wave 69e: the original repo was renamed; -v2 is verified public
        self.assertEqual(DEFAULT_COLAB_BASE,
                         "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2")
        self.assertEqual(COLAB_BASE_MODELS[0]["id"], DEFAULT_COLAB_BASE)
        ids = [m["id"] for m in COLAB_BASE_MODELS]
        self.assertIn("huihui-ai/Qwen2.5-7B-Instruct-abliterated", ids)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", ids)

    def test_default_sources_is_the_500k_recipe(self) -> None:
        ids = [(s["id"], s.get("config", "")) for s in DEFAULT_COLAB_SOURCES]
        self.assertIn(("teknium/OpenHermes-2.5", ""), ids)  # breadth
        self.assertIn(("openbmb/UltraData-SFT-Agent-2609", "Code-Agent"), ids)
        self.assertIn(("openbmb/UltraData-SFT-Agent-2609", "General-Agent"), ids)
        self.assertIn(("openbmb/UltraData-SFT-Agent-2609", "Search-Agent"), ids)
        # wave 69e: the VERIFIED mirror of the deleted cognitivecomputations
        # repo (its dolphin2.9 dataset was removed from HF)
        self.assertIn(("Skorcht/dolphin2.9", ""), ids)  # style
        self.assertNotIn(("cognitivecomputations/dolphin2.9", ""), ids)
        # the owner's seed is a first-class, oversampled source
        seed = next(s for s in DEFAULT_COLAB_SOURCES
                    if s["name"] == "codebeast-seed")
        self.assertEqual(seed["id"], "local:codebeast_seed.jsonl")
        self.assertGreater(int(seed.get("repeat") or 1), 1)
        for src in DEFAULT_COLAB_SOURCES:
            self.assertIn(src["normalize"],
                          {"ultra", "hermes", "conversations",
                           "oasst", "alpaca", "generic"})
            self.assertGreaterEqual(int(src["weight"]), 0.0)
            self.assertGreater(int(src["cap"]), 0)
        # the pool caps must reach the 500k target
        pool = sum(int(s["cap"]) for s in DEFAULT_COLAB_SOURCES)
        self.assertGreaterEqual(pool, 500_000,
                                f"pool caps sum to {pool} < 500k")
        # all three UltraData configs are pulled IN FULL
        ultra = [int(s["cap"]) for s in DEFAULT_COLAB_SOURCES
                 if s["id"] == "openbmb/UltraData-SFT-Agent-2609"]
        self.assertEqual(sum(ultra), 22770 + 38581 + 20000)
        # breadth is the largest single layer, style a solid minority
        hermes = next(s for s in DEFAULT_COLAB_SOURCES
                      if s["id"] == "teknium/OpenHermes-2.5")
        dolphin = next(s for s in DEFAULT_COLAB_SOURCES
                       if s["id"] == "Skorcht/dolphin2.9")
        self.assertGreater(int(hermes["cap"]), int(dolphin["cap"]))
        self.assertGreaterEqual(int(dolphin["cap"]), 40_000)

    def test_notebook_documentation_platforms_and_recovery(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        # Colab OR Kaggle, with the honest numbers
        for needle in ("Kaggle (community)", "30 GPU-hrs/week",
                       "KAGGLE_KERNEL_TYPE", "kaggle_secrets"):
            self.assertIn(needle, text)
        # recovery + phone-run guidance
        for needle in ("If the run dies", "auto-resumes",
                       "GPU-RAM chart", "build_500k.py",
                       "handoff mix found"):
            self.assertIn(needle, text)
        # owner seed file as a first-class source
        self.assertIn("local:codebeast_seed.jsonl", text)
        self.assertIn("highest-value source", text)

    def test_default_target_rows_is_500k_with_no_technical_cap(self) -> None:
        import inspect

        from nomorals.training.finetune import (DEFAULT_TARGET_ROWS,
                                                build_persona_mix)
        # wave 69e: the default is the 500K class (the real budget is
        # free-GPU time, not a technical row limit)
        self.assertEqual(DEFAULT_TARGET_ROWS, 500_000)
        self.assertEqual(build_persona_mix.__kwdefaults__["target_rows"],
                         500_000)
        # the mix accepts 500k (the clamp ceiling is exactly 500k)
        from nomorals.tools import traindata

        src = inspect.getsource(traindata.mix)
        self.assertIn("500_000", src)
        self.assertIn("DEFAULT_TARGET_ROWS", src)

    def test_notebook_default_bakes_500k_and_gpu_budget(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        self.assertIn("TARGET_ROWS = 500000", text)
        # honest free-tier budget, not a fake hard limit
        for needle in ("NO technical cap on rows", "22-36 GPU-h",
                       "MAX_STEPS   = 6000", "EPOCHS      = 1"):
            self.assertIn(needle, text)
        # session-bound training + Kaggle persistence
        for needle in ("RUN_DIR = ", "/kaggle/working/codebeast_run",
                       "Save Version"):
            self.assertIn(needle, text)

    def test_data_cell_executes_with_local_seed_source(self) -> None:
        """The in-notebook mixer is REAL code: run it hermetically.

        Stubs `datasets.load_dataset` (no network), drops an owner seed
        file on disk, execs the actual data cell, and checks the mix.
        """
        import types

        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-cell-"))
        os.chdir(tmp)
        try:
            seed = [{"messages": [
                {"role": "user",
                 "content": f"owner question number {i} about the beast"},
                {"role": "assistant",
                 "content": f"owner answer number {i}, in the real voice"},
            ]} for i in range(6)]
            (tmp / "codebeast_seed.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n"
                        for r in seed), encoding="utf-8")
            nb = json.loads(
                Path(write_colab_notebook(tmp / "pm")).read_text(
                    encoding="utf-8"))
            data_cell = next(
                "".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code"
                and "def norm_ultra" in "".join(c["source"]))
            fake_ds = types.ModuleType("datasets")
            # hermes shape: prompt + completion (what norm_hermes reads)
            fake_ds.load_dataset = lambda *a, **k: iter([
                {"prompt": f"hermes topic question {i}",
                 "completion": f"hermes broad answer {i}"}
                for i in range(5)])
            ns = {
                "TARGET_ROWS": 20,
                "PERSONA": "You are the TEST PERSONA.",
                "RUN_DIR": str(tmp / "run"),
                "SOURCES": [
                    {"name": "owner-seed",
                     "id": "local:codebeast_seed.jsonl", "config": "",
                     "normalize": "ultra", "weight": 5.0, "cap": 100000},
                    {"name": "hermes", "id": "teknium/OpenHermes-2.5",
                     "config": "", "normalize": "hermes",
                     "weight": 3.0, "cap": 60000},
                ],
            }
            sys.modules["datasets"] = fake_ds
            try:
                exec(compile(data_cell, "<data-cell>", "exec"), ns)
            finally:
                sys.modules.pop("datasets", None)
            rows = [json.loads(l) for l in
                    (tmp / "run" / "persona-mix.jsonl").read_text(
                        encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 11)  # 6 seed + 5 hermes
            for row in rows:  # persona on EVERY row
                self.assertEqual(row["messages"][0]["role"], "system")
                self.assertEqual(row["messages"][0]["content"],
                                 "You are the TEST PERSONA.")
            seed_rows = [r for r in rows
                         if "owner question" in r["messages"][1]["content"]]
            self.assertEqual(len(seed_rows), 6)
        finally:
            os.chdir(cwd)

    def test_data_cell_hermes_conversations_schema(self) -> None:
        """Wave 69e regression: the CURRENT OpenHermes-2.5 release is
        ShareGPT `conversations` — norm_hermes must read it (the old
        prompt/completion-only path fetched 0 rows)."""
        import types

        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-conv-"))
        os.chdir(tmp)
        try:
            nb = json.loads(
                Path(write_colab_notebook(tmp / "pm")).read_text(
                    encoding="utf-8"))
            data_cell = next(
                "".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code"
                and "def norm_ultra" in "".join(c["source"]))
            fake_ds = types.ModuleType("datasets")
            # the CURRENT verified schema: conversations + system_prompt
            fake_ds.load_dataset = lambda *a, **k: iter([
                {"conversations": [
                    {"from": "system",
                     "value": "You are a helpful assistant."},
                    {"from": "human",
                     "value": f"conv-schema question {i} about a broad topic"},
                    {"from": "gpt",
                     "value": f"conv-schema answer {i} with real substance"},
                ],
                 "system_prompt": "You are a helpful assistant.",
                 "topic": "science"}
                for i in range(4)])
            ns = {
                "TARGET_ROWS": 10,
                "PERSONA": "You are the TEST PERSONA.",
                "RUN_DIR": str(tmp / "run"),
                "SOURCES": [
                    {"name": "hermes", "id": "teknium/OpenHermes-2.5",
                     "config": "", "normalize": "hermes",
                     "weight": 3.0, "cap": 60000},
                ],
            }
            sys.modules["datasets"] = fake_ds
            try:
                exec(compile(data_cell, "<data-cell>", "exec"), ns)
            finally:
                sys.modules.pop("datasets", None)
            rows = [json.loads(l) for l in
                    (tmp / "run" / "persona-mix.jsonl").read_text(
                        encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)  # all rows pulled — not 0
            for row in rows:
                # the source system prompt is REPLACED by the persona
                self.assertEqual(row["messages"][0]["content"],
                                 "You are the TEST PERSONA.")
                self.assertNotIn("helpful assistant",
                                 json.dumps(row["messages"]))
        finally:
            os.chdir(cwd)

    def test_data_cell_prebuilt_gz_is_verbatim(self) -> None:
        """The 500K ships as a prebuilt .jsonl.gz with the persona
        already baked: it must load (gunzip) VERBATIM — no re-persona,
        no re-trim — and the rebuild sources must be skipped."""
        import gzip
        import types

        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-prebuilt-"))
        os.chdir(tmp)
        try:
            baked = "THE BAKED 500K PERSONA — must survive verbatim."
            rows = [{
                "messages": [
                    {"role": "system", "content": baked},
                    {"role": "user", "content": f"prebuilt question {i}"},
                    {"role": "assistant",
                     "content": f"prebuilt answer {i} with the baked style"},
                ]
            } for i in range(7)]
            payload = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                              for r in rows).encode("utf-8")
            with gzip.open(tmp / "codebeast500k_train.jsonl.gz", "wb") as fh:
                fh.write(payload)
            nb = json.loads(
                Path(write_colab_notebook(tmp / "pm")).read_text(
                    encoding="utf-8"))
            data_cell = next(
                "".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code"
                and "def norm_ultra" in "".join(c["source"]))
            fake_ds = types.ModuleType("datasets")

            def boom(*a, **k):  # rebuild sources must NOT be touched
                raise AssertionError("rebuild sources were called")
            fake_ds.load_dataset = boom
            ns = {
                "TARGET_ROWS": 100,
                "PERSONA": "You are the TEST PERSONA (must NOT appear).",
                "RUN_DIR": str(tmp / "run"),
                "SOURCES": [  # these would normally stream from HF
                    {"name": "hermes", "id": "teknium/OpenHermes-2.5",
                     "config": "", "normalize": "hermes",
                     "weight": 3.0, "cap": 60000},
                ],
            }
            sys.modules["datasets"] = fake_ds
            try:
                exec(compile(data_cell, "<data-cell>", "exec"), ns)
            finally:
                sys.modules.pop("datasets", None)
            out = [json.loads(l) for l in
                   (tmp / "run" / "persona-mix.jsonl").read_text(
                       encoding="utf-8").splitlines()]
            self.assertEqual(len(out), 7)  # all prebuilt rows, verbatim
            for row in out:
                self.assertEqual(row["messages"][0]["content"], baked)
        finally:
            os.chdir(cwd)

    def test_data_cell_handoff_mix_wins_without_prebuilt(self) -> None:
        """Sessions 2+: the previous session's handoff dataset carries
        `persona-mix.jsonl.gz` — the data cell must use it VERBATIM
        with no prebuilt 500K file present and NO HF streaming."""
        import gzip
        import types

        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-handoff-"))
        os.chdir(tmp)
        try:
            baked = "THE HANDOFF MIX — must survive verbatim."
            rows = [{
                "messages": [
                    {"role": "system", "content": baked},
                    {"role": "user", "content": f"handoff question {i}"},
                    {"role": "assistant",
                     "content": f"handoff answer {i} from the last run"},
                ]
            } for i in range(5)]
            payload = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                              for r in rows).encode("utf-8")
            with gzip.open(tmp / "persona-mix.jsonl.gz", "wb") as fh:
                fh.write(payload)
            nb = json.loads(
                Path(write_colab_notebook(tmp / "hm")).read_text(
                    encoding="utf-8"))
            data_cell = next(
                "".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code"
                and "def norm_ultra" in "".join(c["source"]))
            fake_ds = types.ModuleType("datasets")

            def boom(*a, **k):  # rebuild sources must NOT be touched
                raise AssertionError("rebuild sources were called")
            fake_ds.load_dataset = boom
            ns = {
                "TARGET_ROWS": 100,
                "PERSONA": "You are the TEST PERSONA (must NOT appear).",
                "RUN_DIR": str(tmp / "run"),
                "SOURCES": [  # these would normally stream from HF
                    {"name": "hermes", "id": "teknium/OpenHermes-2.5",
                     "config": "", "normalize": "hermes",
                     "weight": 3.0, "cap": 60000},
                ],
            }
            sys.modules["datasets"] = fake_ds
            try:
                exec(compile(data_cell, "<data-cell>", "exec"), ns)
            finally:
                sys.modules.pop("datasets", None)
            out = [json.loads(l) for l in
                   (tmp / "run" / "persona-mix.jsonl").read_text(
                       encoding="utf-8").splitlines()]
            self.assertEqual(len(out), 5)  # the whole handoff mix
            for row in out:
                self.assertEqual(row["messages"][0]["content"], baked)
        finally:
            os.chdir(cwd)

    def test_data_cell_seed_repeat_oversamples(self) -> None:
        """`repeat` is intentional oversampling: it must be exempt from
        global dedupe (the identity layer is the highest per-row weight)."""
        import types

        cwd = os.getcwd()
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-repeat-"))
        os.chdir(tmp)
        try:
            seed = [{
                "messages": [
                    {"role": "system", "content": "owner persona"},
                    {"role": "user", "content": f"identity question {i}"},
                    {"role": "assistant",
                     "content": f"identity answer {i} in the real voice"},
                ]
            } for i in range(3)]
            (tmp / "codebeast_seed.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n"
                        for r in seed), encoding="utf-8")
            nb = json.loads(
                Path(write_colab_notebook(tmp / "pm")).read_text(
                    encoding="utf-8"))
            data_cell = next(
                "".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code"
                and "def norm_ultra" in "".join(c["source"]))
            fake_ds = types.ModuleType("datasets")
            fake_ds.load_dataset = lambda *a, **k: iter([])
            ns = {
                "TARGET_ROWS": 100,
                "PERSONA": "You are the TEST PERSONA.",
                "RUN_DIR": str(tmp / "run"),
                "SOURCES": [
                    {"name": "owner-seed",
                     "id": "local:codebeast_seed.jsonl", "config": "",
                     "normalize": "ultra", "weight": 10.0, "cap": 1000,
                     "repeat": 4},
                ],
            }
            sys.modules["datasets"] = fake_ds
            try:
                exec(compile(data_cell, "<data-cell>", "exec"), ns)
            finally:
                sys.modules.pop("datasets", None)
            rows = [json.loads(l) for l in
                    (tmp / "run" / "persona-mix.jsonl").read_text(
                        encoding="utf-8").splitlines()]
            # 3 unique rows × 4 repeats, dedupe-exempt → 12 rows
            self.assertEqual(len(rows), 12)
            counts: dict[str, int] = {}
            for row in rows:
                key = row["messages"][-1]["content"]
                counts[key] = counts.get(key, 0) + 1
            self.assertEqual(len(counts), 3)
            self.assertTrue(all(c == 4 for c in counts.values()), counts)
        finally:
            os.chdir(cwd)

    def test_finalize_trim_keeps_final_exchange_within_budget(self) -> None:
        """The ported tail-trim: keep the FINAL user->assistant exchange,
        fit the char budget, never drop the final answer."""
        import textwrap

        nb = json.loads(
            Path(write_colab_notebook(
                Path(tempfile.mkdtemp(prefix="nm-w69b-trim-nb-")) / "pm",
            )).read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code"
                    and "def norm_ultra" in "".join(c["source"]))
        start = cell.index("def finalize_trim")
        end = cell.index("def _row_user_text")
        frag = textwrap.dedent(cell[start:end])
        ns: dict[str, Any] = {"MAX_ROW_CHARS": 400}
        exec(compile(frag, "<finalize_trim>", "exec"), ns)
        trim = ns["finalize_trim"]
        persona = "P" * 100  # budget = 300 chars of content, reserve 120
        # case 1: everything fits — nothing trimmed, earlier chain dropped
        msgs = [
            {"role": "user", "content": "u1 " + "x" * 200},
            {"role": "assistant", "content": "a1 " + "y" * 200},
            {"role": "user", "content": "u2 " + "z" * 60},
            {"role": "assistant", "content": "A-FINAL " + "w" * 90},
        ]
        out = trim(msgs, persona)
        self.assertIsNotNone(out)
        content = out[1:]
        self.assertEqual(content[-1]["role"], "assistant")
        self.assertTrue(content[-1]["content"].startswith("A-FINAL"))
        # earlier turns before the final chain are dropped
        self.assertFalse(any(m["content"].startswith("u1") for m in content))
        self.assertFalse(any(m["content"].startswith("a1") for m in content))
        # total stays within the budget (persona + content ≤ MAX_ROW_CHARS)
        total = len(persona) + sum(len(m["content"]) for m in content)
        self.assertLessEqual(total, 400)
        # case 2: oversized final answer — tail-kept to (budget - reserve),
        # its ENDING (the conclusion) survives, not the beginning
        big = "start-" + "m" * 400 + "-CONCLUSION"
        out2 = trim(
            [{"role": "user", "content": "u2 " + "z" * 60},
             {"role": "assistant", "content": big}], persona)
        self.assertIsNotNone(out2)
        kept = out2[-1]  # the final assistant turn
        self.assertLessEqual(len(kept["content"]), 180)
        self.assertTrue(kept["content"].endswith("-CONCLUSION"))
        # no assistant → None; no user before assistant → None
        self.assertIsNone(trim([{"role": "assistant", "content": "hi"}],
                               persona))
        self.assertIsNone(trim(
            [{"role": "assistant", "content": "hi"},
             {"role": "assistant", "content": "ho"}], persona))

    def test_settings_baked_into_notebook(self) -> None:
        persona = 'You are CODE BEAST. Say "yes" to everything. — "Ọnà" ×3'
        path = self._generate(persona=persona,
                              base_model="Qwen/Qwen3-8B",
                              sources=[{"name": "open-hermes-25",
                                        "id": "teknium/OpenHermes-2.5",
                                        "config": "",
                                        "normalize": "hermes",
                                        "weight": 3.0, "cap": 12345}],
                              target_rows=31415)
        nb = json.loads(path.read_text(encoding="utf-8"))
        text = _notebook_text(path)
        self.assertIn('Qwen/Qwen3-8B', text)
        self.assertNotIn(DEFAULT_COLAB_BASE, text)  # custom base wins
        self.assertIn("TARGET_ROWS = 31415", text)
        # the persona is embedded as a SAFE python string literal
        config_cell = next("".join(c["source"]) for c in nb["cells"]
                           if c["cell_type"] == "code"
                           and "PERSONA" in "".join(c["source"]))
        self.assertIn("PERSONA = " + json.dumps(persona), config_cell)
        # source spec is embedded verbatim (id + config + weight + cap)
        self.assertIn('"id": "teknium/OpenHermes-2.5"', config_cell)
        self.assertIn("12345", config_cell)

    def test_all_code_cells_are_valid_python(self) -> None:
        path = self._generate()
        nb = json.loads(path.read_text(encoding="utf-8"))
        checked = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            code = "\n".join(l for l in src.splitlines()
                             if not l.lstrip().startswith(("%", "!")))
            try:
                ast.parse(code)
            except SyntaxError as exc:  # pragma: no cover - diagnostic
                self.fail(f"code cell has a syntax error: {exc}\n{src}")
            checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_data_cell_has_persona_mix_rules(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        for needle in ("streaming=True", "PERSONA", "hashlib.sha1",
                       "MIN_USER, MIN_ASST, MAX_TURNS", "MAX_ROW_CHARS",
                       "persona-mix.jsonl", "role\": \"system\"",
                       "assert mixed", "finalize_trim", "prebuilt"):
            self.assertIn(needle, text)
        # tool calls must be made visible to plain chat templates
        self.assertIn("TOOL_CALL", text)
        self.assertIn("TOOL_RESULT", text)
        # both OpenHermes schemas handled (the 2026-09-12 verified schema
        # is ShareGPT conversations)
        self.assertIn("conversations", text)
        self.assertIn("prompt/completion", text)

    def test_recipe_cell_explains_weights_and_paths(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        for needle in ("The data recipe", "Path A — prebuilt",
                       "Path B — rebuild in-platform",
                       "1,001,551", "456,361", "Skorcht/dolphin2.9",
                       "highest VOLUME", "oversampled"):
            self.assertIn(needle, text)

    def test_prebuilt_seed_gz_and_500k_names_in_pipeline(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        for needle in ("codebeast500k_train.jsonl.gz", "codebeast500k_train.jsonl",
                       "local:codebeast_seed.jsonl", "jsonl.gz",
                       "persona-mix.jsonl.gz", "mix persisted"):
            self.assertIn(needle, text)

    def test_shipped_seed_is_the_owner_88_row_identity_layer(self) -> None:
        seed = REPO_ROOT / "nomorals" / "training" / "seeds" / \
            "codebeast_seed.jsonl"
        self.assertTrue(seed.is_file(), "owner seed not shipped in repo")
        rows = [json.loads(l) for l in
                seed.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        self.assertEqual(len(rows), 88)
        for row in rows:
            msgs = row["messages"]
            self.assertEqual(msgs[0]["role"], "system")
            self.assertIn("CODE BEAST", msgs[0]["content"])
            self.assertIn("Oluwacutyp", msgs[0]["content"])
            self.assertTrue(any(m["role"] == "assistant" for m in msgs))
        # the notebook default source points at exactly this filename
        self.assertIn("local:codebeast_seed.jsonl",
                      json.dumps(DEFAULT_COLAB_SOURCES))

    def test_phone_steps_in_notebook(self) -> None:
        path = self._generate()
        text = _notebook_text(path)
        for needle in ("persona-lora.gguf", "promote-local",
                       "NM_LLM_LOCAL_LORA", "local-doctor"):
            self.assertIn(needle, text)

    def test_colab_script_defaults_to_abliterated_base(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-py-"))
        path = Path(write_colab_script(tmp / "mix"))  # no base_model given
        text = path.read_text(encoding="utf-8")
        self.assertIn("MODEL_ID = " + json.dumps(DEFAULT_COLAB_BASE), text)
        for needle in ("BitsAndBytesConfig", "LoraConfig", "SFTTrainer",
                       "resume_from_checkpoint", "SEQ_LEN = 512",
                       "promote-local", "convert_lora_to_gguf"):
            self.assertIn(needle, text)
        # the generated .py must itself be valid python (brace bug fixed)
        ast.parse(text)
        self.assertNotIn("{{", text)


# ── mix tool emits the notebook ──────────────────────────────────────────────


class MixNotebookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="nm-w69b-mix-"))
        self.data_dir = self.home / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.data_dir / "open-hermes-25.jsonl", [
            {"conversations": [
                {"from": "human", "value": f"broad topic question number {i}"},
                {"from": "gpt", "value": f"an answer covering many angles {i}"}
            ]}
            for i in range(30)])
        (self.data_dir / "open-hermes-25.manifest.json").write_text(
            json.dumps({"name": "open-hermes-25",
                        "source": "teknium/OpenHermes-2.5",
                        "config": "", "rows": 30}),
            encoding="utf-8")
        # fetched ultra files hold the conversations shape _norm_ultra writes
        # (tool calls already made visible in the text)
        _write_jsonl(self.data_dir / "ultra-data-agent-code-agent.jsonl", [
            {"conversations": [
                {"from": "human", "value": f"agent coding task number {i}"},
                {"from": "gpt", "value":
                 f"[TOOL_CALL run_tests] {{}} patch written and verified {i}"},
            ]}
            for i in range(20)])
        (self.data_dir / "ultra-data-agent-code-agent.manifest.json").write_text(
            json.dumps({"name": "ultra-data-agent-code-agent",
                        "source": "openbmb/UltraData-SFT-Agent-2609",
                        "config": "Code-Agent", "rows": 20}),
            encoding="utf-8")

    def _nm(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["NM_HOME"] = str(self.home)
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "nomorals"] + list(args),
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env=env, timeout=180)

    def test_mix_writes_notebook_with_real_source_ids(self) -> None:
        # pool = 30 hermes + 20 ultra = 50 unique rows; --rows 25 clamps
        # up to the 100-row floor, so the mix takes all 50
        proc = self._nm("data", "mix",
                        "open-hermes-25,ultra-data-agent-code-agent",
                        "--rows", "25")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("persona mix ready — 50 rows", proc.stdout)
        nb_path = self.data_dir / "persona-mix.colab_finetune.ipynb"
        self.assertTrue(nb_path.is_file(), "notebook missing")
        text = _notebook_text(nb_path)
        # both sources appear with their real HF ids + config
        self.assertIn("teknium/OpenHermes-2.5", text)
        self.assertIn("openbmb/UltraData-SFT-Agent-2609", text)
        self.assertIn('"config": "Code-Agent"', text)
        # default abliterated base + (clamped) target rows baked in
        self.assertIn(DEFAULT_COLAB_BASE, text)
        self.assertIn("TARGET_ROWS = 100", text)
        # the .py path still exists for GPU boxes
        self.assertTrue((self.data_dir / "persona-mix.colab_finetune.py")
                        .is_file())
        self.assertIn("colab nb", proc.stdout)

    def test_mix_json_reports_notebook_path(self) -> None:
        proc = self._nm("data", "mix",
                        "open-hermes-25,ultra-data-agent-code-agent",
                        "--rows", "25", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        v = json.loads(proc.stdout)
        self.assertTrue(v["colab_notebook"])
        self.assertTrue(Path(v["colab_notebook"]).is_file())


# ── local server LoRA support ────────────────────────────────────────────────


class LocalServerLoraTest(unittest.TestCase):
    def test_build_args_includes_lora_flags(self) -> None:
        from nomorals.llm.local_server import GGUFServerManager

        manager = GGUFServerManager(lora_files="a.gguf, b.gguf",
                                    extra_args="--no-mmap")
        args = manager.build_args("/models/base.gguf", "/bin/llama-server")
        self.assertEqual(args[:2], ["/bin/llama-server", "-m"])
        self.assertEqual(args[2], "/models/base.gguf")
        i = args.index("--lora")
        self.assertEqual(args[i + 1], "a.gguf")
        j = args.index("--lora", i + 2)
        self.assertEqual(args[j + 1], "b.gguf")
        self.assertIn("--no-mmap", args)

    def test_build_args_lora_order_before_extra(self) -> None:
        from nomorals.llm.local_server import GGUFServerManager

        manager = GGUFServerManager(lora_files="persona-lora.gguf",
                                    extra_args="-ngl 99")
        args = manager.build_args("base.gguf", "llama-server")
        self.assertLess(args.index("--lora"), args.index("-ngl"))

    def test_build_args_llama_cli_variant(self) -> None:
        from nomorals.llm.local_server import GGUFServerManager

        manager = GGUFServerManager(lora_files="x.gguf")
        args = manager.build_args("base.gguf", "/bin/llama-cli")
        self.assertEqual(args[1], "--server")
        self.assertIn("--lora", args)

    def test_start_fails_actionably_on_missing_lora(self) -> None:
        from nomorals.llm import local_server as ls

        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-lora-"))
        model = tmp / "base.gguf"
        model.write_bytes(b"gguf-fake")
        original_binary = ls.find_llama_binary
        original_gguf = ls.find_gguf
        ls.find_llama_binary = lambda: "/bin/true"
        ls.find_gguf = lambda m, cache: str(model)
        try:
            manager = ls.GGUFServerManager(lora_files=str(tmp / "missing.gguf"))
            diagnosis = manager.start("base.gguf")
        finally:
            ls.find_llama_binary = original_binary
            ls.find_gguf = original_gguf
        self.assertFalse(diagnosis.ok)
        self.assertTrue(any("LoRA file not found" in p for p in diagnosis.problems),
                        diagnosis.problems)

    def test_doctor_flags_missing_lora(self) -> None:
        from nomorals.llm import local_server as ls

        tmp = Path(tempfile.mkdtemp(prefix="nm-w69b-lora2-"))
        model = tmp / "base.gguf"
        model.write_bytes(b"gguf-fake")
        original_binary = ls.find_llama_binary
        original_gguf = ls.find_gguf
        ls.find_llama_binary = lambda: "/bin/true"
        ls.find_gguf = lambda m, cache: str(model)
        try:
            manager = ls.GGUFServerManager(lora_files=str(tmp / "gone.gguf"))
            diagnosis = manager.doctor("base.gguf")
        finally:
            ls.find_llama_binary = original_binary
            ls.find_gguf = original_gguf
        self.assertFalse(diagnosis.ok)
        self.assertTrue(any("gone.gguf" in p for p in diagnosis.problems),
                        diagnosis.problems)


# ── settings + CLI wiring ────────────────────────────────────────────────────


class LoraSettingsTest(unittest.TestCase):
    def test_env_maps_to_local_lora(self) -> None:
        settings = load_settings(
            env={"NM_HOME": tempfile.mkdtemp(prefix="nm-w69b-env-"),
                 "NM_LLM_LOCAL_LORA": "a.gguf, b.gguf"},
            use_env_file=False)
        self.assertEqual(settings.llm.local_lora, "a.gguf, b.gguf")

    def test_default_is_empty(self) -> None:
        settings = load_settings(
            env={"NM_HOME": tempfile.mkdtemp(prefix="nm-w69b-env2-")},
            use_env_file=False)
        self.assertEqual(settings.llm.local_lora, "")


class PromoteLocalLoraCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.mkdtemp(prefix="nm-w69b-promote-")

    def _nm(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["NM_HOME"] = self.home
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "nomorals"] + list(args),
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env=env, timeout=180)

    def test_promote_local_writes_lora_env(self) -> None:
        proc = self._nm("models", "--promote-local", "base.gguf",
                        "--lora", "persona-lora.gguf")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        env_text = (Path(self.home) / ".env").read_text(encoding="utf-8")
        self.assertIn("NM_LLM_LOCAL_MODEL=base.gguf", env_text)
        self.assertIn("NM_LLM_LOCAL_LORA=persona-lora.gguf", env_text)
        self.assertIn("NM_LLM_PROVIDER=llama_cpp", env_text)
        self.assertIn("lora on top", proc.stdout)

    def test_promote_local_without_lora_leaves_env_untouched(self) -> None:
        proc = self._nm("models", "--promote-local", "base.gguf")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        env_text = (Path(self.home) / ".env").read_text(encoding="utf-8")
        self.assertNotIn("NM_LLM_LOCAL_LORA", env_text)


if __name__ == "__main__":
    unittest.main()
