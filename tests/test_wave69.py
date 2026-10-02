"""Wave 69 — the fine-tune pipeline: persona mix, Colab script,
persona self-distillation, the verified free-dataset catalog, and the
base-model shortlist.

Hermetic: synthetic fetched JSONL files stand in for network fetches;
a scripted router stands in for the model in the sample generator;
the CLI e2e runs in a subprocess with a private NM_HOME.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.training.dataset import decode_example
from nomorals.training.finetune import (COLAB_BASE_MODELS,
                                        DEFAULT_PERSONA, MixSource,
                                        apply_persona, build_persona_mix,
                                        generate_persona_samples,
                                        load_persona, write_colab_script)
from nomorals.training.free_datasets import (_norm_hermes, _norm_ultra,
                                             catalog, lookup)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class FakeRouter:
    def __init__(self, bad_first: bool = False) -> None:
        self.n = 0
        self.bad_first = bad_first

    def chat(self, messages, params=None, **kw) -> LLMResponse:
        self.n += 1
        if self.bad_first and self.n == 1:
            return LLMResponse(text="sorry, no json today", model="fake")
        pairs = [{"question": f"English question {self.n}-{k}",
                  "answer": f"Ey! Yoruba answer {self.n}-{k} — I'm on it, o."}
                 for k in range(5)]
        return LLMResponse(text="Sure: " + json.dumps({"pairs": pairs}),
                           model="fake")


# ── catalog (verified flags) ─────────────────────────────────────────────────


class CatalogTest(unittest.TestCase):
    def test_new_entries_present_with_honest_flags(self) -> None:
        # wave 69e: dolphin-2.9 re-verified live (Skorcht mirror,
        # 456,361 rows) — the original cognitivecomputations repo was
        # deleted, the verified mirror is now the catalog target
        for name, verified in (("ultra-data-agent", True),
                               ("open-hermes-25", True),
                               ("yoruba-bbc-topics", True),
                               ("dolphin-2.9", True)):
            entry = lookup(name)
            self.assertIsNotNone(entry, name)
            self.assertIs(entry["verified"], verified, name)

    def test_ultra_entry_is_apache_and_agent(self) -> None:
        entry = lookup("ultra-data-agent")
        self.assertIn("apache-2.0", entry["license"])
        self.assertEqual(entry["kind"], "agent")

    def test_base_model_shortlist_covers_the_phone_and_colab(self) -> None:
        ids = [m["id"] for m in COLAB_BASE_MODELS]
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", ids)
        self.assertTrue(any("abliterated" in i for i in ids))
        self.assertTrue(all(m["license"] for m in COLAB_BASE_MODELS))


# ── normalizers ──────────────────────────────────────────────────────────────


class NormalizerTest(unittest.TestCase):
    def test_ultra_parsers_messages_and_tools(self) -> None:
        row = {
            "uuid": "u1", "domain": "code",
            "messages": json.dumps([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant",
                 "tool_calls": [{"function": {
                     "name": "run_tests", "arguments": '{"k": 1}'}}]},
                {"role": "tool", "name": "run_tests", "content": "2 failed"},
                {"role": "assistant", "content": "fixed it"},
            ]),
        }
        out = _norm_ultra(row)
        conv = out["conversations"]
        self.assertEqual(conv[0]["from"], "system")
        self.assertIn("TOOL CALL: run_tests", conv[2]["value"])
        self.assertIn("[run_tests result]", conv[3]["value"])
        self.assertEqual(conv[-1]["from"], "gpt")

    def test_ultra_rejects_unfinished_conversations(self) -> None:
        row = {"messages": json.dumps([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": []},
        ])}
        self.assertIsNone(_norm_ultra(row))

    def test_hermes_multi_prompt(self) -> None:
        out = _norm_hermes({"system": "s", "prompt": "q1",
                            "prompt2": "q2", "completion": "a"})
        conv = out["conversations"]
        self.assertEqual([c["from"] for c in conv],
                         ["system", "human", "human", "gpt"])

    def test_sharegpt_decoder_preserves_system_role(self) -> None:
        # the mix depends on this: source system prompts must arrive as
        # system turns so apply_persona can replace them
        example = decode_example({
            "conversations": [
                {"from": "system", "value": "old sys"},
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ]
        })
        roles = [t.role for t in example.turns]
        self.assertEqual(roles, ["system", "user", "assistant"])


# ── persona application + mix builder ────────────────────────────────────────


class PersonaMixTest(unittest.TestCase):
    PERSONA = "You are CODE BEAST — test persona."

    def _sources_dir(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        hermes = [
            {"conversations": [
                {"from": "system", "value": "old sys"},
                {"from": "human", "value": f"tell me about topic {i} in detail"},
                {"from": "gpt", "value": f"topic {i} is broad and rich."}]}
            for i in range(10)
        ] + [{"conversations": [
            {"from": "human", "value": "tell me about topic 0 in detail"},
            {"from": "gpt", "value": "topic 0 is broad and rich."}]}]  # dupe
        _write_jsonl(tmp / "hermes.jsonl", hermes)
        agent = [
            {"conversations": [
                {"from": "human", "value": f"agent job {i}"},
                {"from": "gpt", "value": "TOOL CALL: x({})"},
                {"from": "human", "value": "[x result]\nok"},
                {"from": "gpt", "value": f"job {i} done"}]}
            for i in range(6)
        ]
        _write_jsonl(tmp / "agent.jsonl", agent)
        alpaca = [{"instruction": f"style question {i}", "input": "",
                   "output": f"direct answer {i}"} for i in range(4)]
        _write_jsonl(tmp / "style.jsonl", alpaca)
        return tmp

    def test_persona_replaces_source_system_prompt(self) -> None:
        tmp = self._sources_dir()
        out = tmp / "mix"
        manifest = build_persona_mix(
            [MixSource("hermes", str(tmp / "hermes.jsonl")),
             MixSource("agent", str(tmp / "agent.jsonl")),
             MixSource("style", str(tmp / "style.jsonl"))],
            self.PERSONA, out, target_rows=50)
        rows = [json.loads(l) for l in Path(
            manifest["outputs"]["messages_jsonl"]).read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 20)  # 10+6+4, 1 dupe removed
        self.assertEqual(manifest["deduped"], 1)
        for row in rows:
            self.assertEqual(row["messages"][0]["role"], "system")
            self.assertEqual(row["messages"][0]["content"], self.PERSONA)
        self.assertNotIn("old sys", json.dumps(rows))

    def test_target_caps_the_mix(self) -> None:
        tmp = self._sources_dir()
        manifest = build_persona_mix(
            [MixSource("hermes", str(tmp / "hermes.jsonl"))],
            self.PERSONA, tmp / "mix2", target_rows=5)
        self.assertEqual(manifest["rows"], 5)

    def test_filters_drop_tiny_and_huge_rows(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rows = [
            {"conversations": [
                {"from": "human", "value": "hi"},  # too short
                {"from": "gpt", "value": "hello friend, how are you today"}]},
            {"conversations": [
                {"from": "human", "value": "x" * 20000},  # too big
                {"from": "gpt", "value": "y" * 20000}]},
            {"conversations": [
                {"from": "human", "value": "a properly sized question here"},
                {"from": "gpt", "value": "a properly sized answer here"}]},
        ]
        _write_jsonl(tmp / "mixed.jsonl", rows)
        manifest = build_persona_mix(
            [MixSource("mixed", str(tmp / "mixed.jsonl"))],
            self.PERSONA, tmp / "mix3", target_rows=50)
        self.assertEqual(manifest["rows"], 1)
        self.assertEqual(manifest["filtered"], 2)

    def test_missing_source_is_skipped_not_fatal(self) -> None:
        tmp = self._sources_dir()
        manifest = build_persona_mix(
            [MixSource("ghost", str(tmp / "ghost.jsonl")),
             MixSource("style", str(tmp / "style.jsonl"))],
            self.PERSONA, tmp / "mix4", target_rows=50)
        self.assertEqual(manifest["rows"], 4)
        self.assertEqual(manifest["per_source"]["ghost"]["rows"], 0)

    def test_all_four_formats_written(self) -> None:
        tmp = self._sources_dir()
        manifest = build_persona_mix(
            [MixSource("hermes", str(tmp / "hermes.jsonl")),
             MixSource("agent", str(tmp / "agent.jsonl"))],
            self.PERSONA, tmp / "mix5", target_rows=50)
        for key in ("messages_jsonl", "alpaca", "sharegpt", "chatml"):
            self.assertTrue(Path(manifest["outputs"][key]).is_file(), key)
        chatml = [json.loads(l) for l in Path(
            manifest["outputs"]["chatml"]).read_text(
            encoding="utf-8").splitlines()]
        self.assertIn("CODE BEAST", chatml[0]["text"])
        alpaca = json.loads(Path(manifest["outputs"]["alpaca"]).read_text())
        self.assertTrue(alpaca and alpaca[0]["instruction"])
        sharegpt = json.loads(Path(manifest["outputs"]["sharegpt"]).read_text())
        self.assertNotIn("system",
                         [c["from"] for c in sharegpt[0]["conversation"]])

    def test_deterministic_with_seed(self) -> None:
        tmp = self._sources_dir()
        srcs = lambda: [MixSource("hermes", str(tmp / "hermes.jsonl")),
                        MixSource("agent", str(tmp / "agent.jsonl"))]
        m1 = build_persona_mix(srcs(), self.PERSONA, tmp / "d1",
                               target_rows=50)
        m2 = build_persona_mix(srcs(), self.PERSONA, tmp / "d2",
                               target_rows=50)
        r1 = Path(m1["outputs"]["messages_jsonl"]).read_text()
        r2 = Path(m2["outputs"]["messages_jsonl"]).read_text()
        self.assertEqual(r1, r2)

    def test_apply_persona_keeps_conversation_only(self) -> None:
        from nomorals.training.dataset import Example, Turn

        example = Example(turns=[Turn("user", "q"), Turn("assistant", "a")])
        out = apply_persona(example, "P")
        self.assertEqual([t.role for t in out.turns],
                         ["system", "user", "assistant"])
        # a conversation with no user turn is left untouched
        weird = Example(turns=[Turn("assistant", "a")])
        self.assertEqual(apply_persona(weird, "P").turns, weird.turns)


# ── Colab script ─────────────────────────────────────────────────────────────


class ColabScriptTest(unittest.TestCase):
    def test_script_is_complete_and_sized_for_free_tier(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="nm-w69-colab-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "data.jsonl").write_text(
            json.dumps({"messages": []}) + "\n", encoding="utf-8")
        path = write_colab_script(tmp / "mix",
                                  base_model="Qwen/Qwen2.5-7B-Instruct")
        text = Path(path).read_text(encoding="utf-8")
        for needle in ("Qwen/Qwen2.5-7B-Instruct", "BitsAndBytesConfig",
                       "LoraConfig", "SFTTrainer", "resume_from_checkpoint",
                       "SEQ_LEN = 512", "promote-local"):
            self.assertIn(needle, text, needle)
        self.assertTrue(path.endswith("mix.colab_finetune.py"))


# ── persona self-distillation ────────────────────────────────────────────────


class PersonaGenerationTest(unittest.TestCase):
    def test_generates_persona_rows_in_language(self) -> None:
        out = Path(tempfile.mkdtemp()) / "yo.jsonl"
        self.addCleanup(shutil.rmtree, out.parent, ignore_errors=True)
        result = generate_persona_samples(
            FakeRouter(), "You are CODE BEAST.",
            language="Yoruba (Yoruba)", n=12, per_call=5, out_path=out)
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], 12)
        self.assertGreaterEqual(result["calls"], 3)
        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            messages = row["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], "You are CODE BEAST.")
            self.assertEqual([m["role"] for m in messages[1:]],
                             ["user", "assistant"])

    def test_degrades_never_raises_on_garbage(self) -> None:
        out = Path(tempfile.mkdtemp()) / "yo2.jsonl"
        self.addCleanup(shutil.rmtree, out.parent, ignore_errors=True)
        result = generate_persona_samples(
            FakeRouter(bad_first=True), "p",
            language="Yoruba (Yoruba)", n=12, per_call=5, out_path=out)
        self.assertEqual(result["rows"], 12)  # recovered after call 1

    def test_load_persona_file_wins(self) -> None:
        f = Path(tempfile.mkdtemp()) / "persona.txt"
        self.addCleanup(shutil.rmtree, f.parent, ignore_errors=True)
        f.write_text("My own persona text.", encoding="utf-8")
        self.assertEqual(load_persona(str(f)), "My own persona text.")
        self.assertEqual(load_persona(""), DEFAULT_PERSONA)
        self.assertEqual(load_persona(str(Path("/nope/none.txt"))),
                         DEFAULT_PERSONA)


# ── CLI e2e (private NM_HOME, subprocess, no network) ────────────────────────


class DataCliE2ETest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w69-cli-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.data_dir = Path(self.home) / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(self.data_dir / "open-hermes-25.jsonl", [
            {"conversations": [
                {"from": "human", "value": f"topic question {i} about the world"},
                {"from": "gpt", "value": f"a broad answer covering many angles {i}"}]}
            for i in range(20)])
        _write_jsonl(self.data_dir / "ultra-data-agent.jsonl", [
            {"conversations": [
                {"from": "human", "value": f"agent job {i}"},
                {"from": "gpt", "value": f"TOOL CALL: x({{}}) done {i}"}]}
            for i in range(15)])

    def _nm(self, *args):
        env = dict(os.environ)
        env["NM_HOME"] = self.home
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            [sys.executable, "-m", "nomorals"] + list(args),
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env=env, timeout=180)

    def test_catalog_lists_new_entries(self) -> None:
        proc = self._nm("data", "catalog")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for name in ("ultra-data-agent", "open-hermes-25",
                     "dolphin-2.9", "yoruba-bbc-topics"):
            self.assertIn(name, proc.stdout)
        self.assertIn("[?]", proc.stdout)  # unverified entries marked

    def test_models_shortlist(self) -> None:
        proc = self._nm("data", "models")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", proc.stdout)
        self.assertIn("abliterated", proc.stdout)

    def test_mix_e2e_writes_bundle_and_colab(self) -> None:
        proc = self._nm("data", "mix", "open-hermes-25,ultra-data-agent",
                        "--rows", "25")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("persona mix ready — 25 rows", proc.stdout)
        for name in ("persona-mix.jsonl", "persona-mix.alpaca.json",
                     "persona-mix.sharegpt.json", "persona-mix.chatml.jsonl",
                     "persona-mix.manifest.json",
                     "persona-mix.colab_finetune.py"):
            self.assertTrue((self.data_dir / name).is_file(), name)
        rows = [json.loads(l) for l in
                (self.data_dir / "persona-mix.jsonl").read_text(
                    encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 25)
        for row in rows:  # built-in default persona applied
            self.assertIn("CODE BEAST", row["messages"][0]["content"])

    def test_mix_json_and_usage_and_missing(self) -> None:
        proc = self._nm("data", "mix", "open-hermes-25,ultra-data-agent",
                        "--rows", "25", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        v = json.loads(proc.stdout)
        self.assertEqual(v["rows"], 25)

        proc = self._nm("data", "mix")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", proc.stdout)

        proc = self._nm("data", "mix", "ghost-dataset")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("no fetched files", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
