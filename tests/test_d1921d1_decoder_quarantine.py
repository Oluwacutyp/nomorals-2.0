"""D1 — the decoder leg (docs 19+21): artifact quarantine, the limits
display, and owner-gating honesty.

Pins:
* a binary decode lands in the ArtifactStore as an inert,
  provenance-recorded, unverified quarantine artifact — the store's
  managed copy is the ONLY copy (no loose ``decodes/`` file), the
  decode record carries the artifact ref, and identical bytes reuse
  their quarantine row;
* the never-executes property as a living pin: an AST scan of the
  four decoder modules fails if an execution primitive appears;
* every clause of the limits display is backed — by a behavioral
  test (truncation flag, auto-attack default) or a structural pin
  (the depth clamp, the store's verification state);
* the hash auto-attack is OFF by default (the owner's
  ``NM_DECODER_AUTO_CRACK`` setting or an explicit ``auto_crack=True``
  turn it on; ``mode="crack"`` is untouched in power);
* the /decode chat surface gates what touches the owner's world —
  ``file:`` reads, the decode archive, and binary saves are
  owner-only — while inline decoding stays public, and the tool
  layer's limits record states plainly that the registry cannot
  identify the chat caller.

Hermetic: fake routers, in-process fixtures, no network.
"""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.core.config import load_settings
from nomorals.core.decoder import analyze
from nomorals.core.policy import CapabilitySet
from tests.test_search_and_model import _FakeRouter

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
SCRIPT = b"#!/bin/sh\necho pwned\n"
# md5("1357") — identified, NOT in the built-in known-secrets table,
# and inside the bounded auto-attack's digits pass (cracks in <1s).
CRACKABLE_MD5 = hashlib.md5(b"1357").hexdigest()
UNKNOWN_MD5 = hashlib.md5(b"some-unique-string-xyz").hexdigest()

_DECODER_MODULES = (
    "nomorals/core/decoder.py",
    "nomorals/tools/decoder.py",
    "nomorals/agents/decoder.py",
    "nomorals/tools/decoder_agent.py",
)


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _ctx(tmp: str) -> Any:
    context = build_context(_settings(tmp), with_executor=False,
                            with_tools=True, with_router=False)
    context.router = _FakeRouter()
    return context


def _reg(ctx: Any) -> Any:
    from nomorals.tools.registry import ToolRegistry

    if not hasattr(ctx, "_d1_reg"):
        ctx._d1_reg = ToolRegistry(ctx).register_builtins()
    return ctx._d1_reg


def _call(ctx: Any, name: str, **kw: Any) -> dict[str, Any]:
    out = _reg(ctx).call(name, capabilities=CapabilitySet.all(), **kw)
    assert out.ok, out.error
    return out.unwrap()


def _no_auto_crack_env() -> Any:
    env = dict(os.environ)
    env.pop("NM_DECODER_AUTO_CRACK", None)
    return mock.patch.dict(os.environ, env, clear=True)


# ────────────────────────── never executes ─────────────────────────────

class NeverExecutesPinTests(unittest.TestCase):
    """The eval verified never-executes by grep; this is the living
    pin — it fails the day any decoder module grows an execution
    primitive."""

    def test_no_execution_primitives_in_decoder_modules(self) -> None:
        root = Path(__file__).resolve().parent.parent
        banned_imports = {"subprocess", "pty", "ctypes", "commands"}
        banned_calls = {"exec", "eval", "compile", "__import__"}
        banned_attrs = {"system", "popen", "execv", "execve", "execl",
                        "spawnl", "spawnlp", "spawnv", "spawnvp"}
        for rel in _DECODER_MODULES:
            src = (root / rel).read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(
                            alias.name.split(".")[0], banned_imports,
                            f"{rel}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[0]
                    self.assertNotIn(mod, banned_imports,
                                     f"{rel}:{node.lineno} from {mod}")
                elif isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name):
                        self.assertNotIn(
                            fn.id, banned_calls,
                            f"{rel}:{node.lineno} calls {fn.id}()")
                    elif isinstance(fn, ast.Attribute):
                        self.assertNotIn(
                            fn.attr, banned_attrs,
                            f"{rel}:{node.lineno} calls .{fn.attr}()")


# ────────────────────────── quarantine wire ────────────────────────────

class QuarantineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="d1-quar-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.agents.decoder import DecoderAgent

        self.agent = DecoderAgent(context=self.ctx, name="t")

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def _store(self) -> Any:
        from nomorals.agents.decoder import _artifact_store_for

        store = _artifact_store_for(self.ctx)
        self.assertIsNotNone(store)
        return store

    def test_binary_decode_lands_in_artifact_store(self) -> None:
        r = self.agent.run({"data": base64.b64encode(PNG).decode(),
                            "save": True})
        self.assertTrue(r.ok, r.error)
        art = r.output["artifact"]
        self.assertIsNotNone(art)
        self.assertEqual(r.output["saved_to"], art["uri"])
        self.assertEqual(art["kind"], "decoded-binary")
        self.assertEqual(art["quarantine"], "decoded-untrusted")
        self.assertEqual(art["sha256"], hashlib.sha256(PNG).hexdigest())
        self.assertEqual(art["size"], len(PNG))
        self.assertEqual(art["verification_state"], "unverified")
        # the decode RECORD carries the artifact ref
        self.assertEqual(r.output["report"]["artifact"]["id"], art["id"])
        self.assertEqual(r.output["report"]["saved_to"], art["uri"])
        # the store holds the bytes, marked inert + provenance-recorded
        store = self._store()
        stored = store.get(art["id"])
        self.assertIsNotNone(stored)
        self.assertEqual(store.read(art["id"]), PNG)
        self.assertEqual(stored.creator, "decoder")
        self.assertEqual(stored.metadata["quarantine"],
                         "decoded-untrusted")
        self.assertEqual(stored.metadata["inert"], True)
        self.assertEqual(stored.metadata["release"],
                         "owner-action-only")
        self.assertEqual(stored.provenance.source_id, "decoder")
        self.assertEqual(stored.provenance.verification_state,
                         "unverified")
        self.assertTrue(store.verify(art["id"]))

    def test_no_loose_decodes_file_anywhere(self) -> None:
        r = self.agent.run({"data": base64.b64encode(PNG).decode(),
                            "save": True})
        self.assertTrue(r.ok, r.error)
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp.name, "workspace",
                                        "decodes")))
        loose = [p for p in Path(self.tmp.name).rglob("decoded-*")
                 if "blobs" not in p.parts]
        self.assertEqual(loose, [])

    def test_script_payload_decodes_as_text_and_is_never_written(
            self) -> None:
        # A script is text-shaped, so the engine reports it as text:
        # it is never written to disk or the store at all (and the
        # AST pin above covers execution).  Only genuinely binary
        # outputs take the quarantine path.
        r = self.agent.run({"data": base64.b64encode(SCRIPT).decode(),
                            "save": True})
        self.assertTrue(r.ok, r.error)
        self.assertIsNone(r.output["artifact"])
        self.assertEqual(
            r.output["report"]["best"]["output"], SCRIPT.decode())
        self.assertEqual(self._store().search(creator="decoder"), [])
        loose = [p for p in Path(self.tmp.name).rglob("decoded-*")
                 if "blobs" not in p.parts]
        self.assertEqual(loose, [])

    def test_elf_payload_quarantined_with_magic_ext(self) -> None:
        elf = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 40
        r = self.agent.run({"data": base64.b64encode(elf).decode(),
                            "save": True})
        self.assertTrue(r.ok, r.error)
        art = r.output["artifact"]
        self.assertIsNotNone(art)
        self.assertTrue(art["name"].endswith(".elf"), art["name"])
        self.assertEqual(art["magic"], "elf")
        self.assertEqual(self._store().read(art["id"]), elf)

    def test_identical_bytes_reuse_quarantine_row(self) -> None:
        enc = base64.b64encode(PNG).decode()
        first = self.agent.run({"data": enc, "save": True})
        second = self.agent.run({"data": enc, "save": True})
        self.assertEqual(first.output["artifact"]["id"],
                         second.output["artifact"]["id"])
        self.assertEqual(
            len(self._store().search(creator="decoder")), 1)

    def test_save_false_stores_nothing(self) -> None:
        r = self.agent.run({"data": base64.b64encode(PNG).decode(),
                            "save": False})
        self.assertTrue(r.ok, r.error)
        self.assertIsNone(r.output["artifact"])
        self.assertIsNone(r.output["saved_to"])
        self.assertEqual(self._store().search(creator="decoder"), [])

    def test_no_store_means_no_persistence_and_the_record_says_so(
            self) -> None:
        from nomorals.agents.decoder import DecoderAgent

        agent = DecoderAgent(
            context=SimpleNamespace(db=None, router=None), name="t")
        r = agent.run({"data": base64.b64encode(PNG).decode(),
                       "save": True})
        self.assertTrue(r.ok, r.error)
        self.assertIsNone(r.output["artifact"])
        self.assertIsNone(r.output["saved_to"])
        self.assertIn("not persisted",
                      r.output["report"].get("storage", ""))

    def test_agent_output_carries_limits(self) -> None:
        r = self.agent.run({"data": base64.b64encode(b"hello").decode()})
        self.assertTrue(r.ok, r.error)
        self.assertIn("decoder limits:", r.output["limits"]["text"])


# ────────────────────────── limits display ─────────────────────────────

class LimitsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="d1-lim-")
        self.ctx = _ctx(self.tmp.name)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_limits_record_shape_and_text(self) -> None:
        from nomorals.tools.decoder import decoder_limits

        with _no_auto_crack_env():
            limits = decoder_limits()
        self.assertIs(limits["executes_decoded_content"], False)
        self.assertIs(limits["scanned"], False)
        self.assertEqual(limits["verification_state"], "unverified")
        self.assertEqual(limits["binary_storage"], "artifact-quarantine")
        self.assertEqual(limits["max_materialized_input_bytes"],
                         33_554_432)
        self.assertIsNone(limits["file_input_bytes"])
        self.assertEqual(limits["max_chain_depth"], 6)
        self.assertIs(limits["auto_crack_default"], False)
        self.assertEqual(limits["auto_crack_max_candidates"], 800_000)
        text = limits["text"]
        for clause in ("never executed", "quarantined in the artifact "
                       "store", "not a safety certification",
                       "streamed whole", "33,554,432",
                       "depth 6", "800,000", "mode=crack"):
            self.assertIn(clause, text)
        # the tool layer's honesty about caller identity
        self.assertIn("cannot identify the chat caller",
                      limits["caller_gating"])

    def test_depth_clamp_exists_in_tool(self) -> None:
        # structural pin for the "depth 6" clause: the registered
        # decoder tool clamps max_depth with a min(..., 6) ceiling.
        root = Path(__file__).resolve().parent.parent
        tree = ast.parse((root / "nomorals/tools/decoder.py").read_text(
            encoding="utf-8"))
        found = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "min"
                    and any(isinstance(a, ast.Constant) and a.value == 6
                            for a in node.args)):
                found = True
        self.assertTrue(found, "no min(..., 6) depth clamp in the tool")

    def test_tool_result_carries_limits(self) -> None:
        out = _call(self.ctx, "decoder",
                    data=base64.b64encode(b"hello").decode())
        self.assertIn("decoder limits:", out["limits"]["text"])
        out = _call(self.ctx, "decoder", mode="decoders")
        self.assertIn("decoder limits:", out["limits"]["text"])

    def test_file_inputs_are_never_truncated(self) -> None:
        # the reworked input door (2026-11-10): the old 2,000,000-
        # byte file cap is gone — a larger file is analyzed WHOLE
        # (streamed), the report's forensics carry its full size,
        # and the flag says not truncated.
        ws = os.path.join(self.tmp.name, "workspace")
        os.makedirs(ws, exist_ok=True)
        payload = gzip.compress(os.urandom(2_100_000))
        self.assertGreater(len(payload), 2_000_000)
        with open(os.path.join(ws, "big.bin"), "wb") as fh:
            fh.write(payload)
        out = _call(self.ctx, "decoder", path="big.bin")
        self.assertIs(out["truncated"], False)
        self.assertEqual(out["report"]["forensics"]["bytes"],
                         len(payload))
        self.assertEqual(
            out["limits"]["max_materialized_input_bytes"], 33_554_432)
        self.assertIsNone(out["limits"]["file_input_bytes"])


# ────────────────────────── auto-crack honesty ─────────────────────────

class AutoCrackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="d1-crack-")
        self.ctx = _ctx(self.tmp.name)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    def test_analyze_does_not_auto_attack_by_default(self) -> None:
        with _no_auto_crack_env():
            out = _call(self.ctx, "decoder", data=UNKNOWN_MD5)
        self.assertNotIn("auto_crack", out)
        self.assertIs(out["limits"]["auto_crack_default"], False)

    def test_explicit_auto_crack_still_cracks(self) -> None:
        with _no_auto_crack_env():
            out = _call(self.ctx, "decoder", data=CRACKABLE_MD5,
                        auto_crack=True)
        self.assertTrue(out["auto_crack"]["cracked"])
        self.assertEqual(out["auto_crack"]["plaintext"], "1357")

    def test_owner_env_setting_reenables_auto_attack(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NM_DECODER_AUTO_CRACK": "1"}):
            out = _call(self.ctx, "decoder", data=CRACKABLE_MD5)
        self.assertTrue(out["auto_crack"]["cracked"])
        self.assertEqual(out["auto_crack"]["plaintext"], "1357")
        self.assertIs(out["limits"]["auto_crack_default"], True)

    def test_explicit_crack_mode_unchanged(self) -> None:
        with _no_auto_crack_env():
            out = _call(self.ctx, "decoder", data=CRACKABLE_MD5,
                        mode="crack")
        self.assertTrue(out["cracked"])
        self.assertEqual(out["plaintext"], "1357")
        self.assertIn("decoder limits:", out["limits"]["text"])


# ────────────────────────── chat-surface gating ────────────────────────

class ChatGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="d1-gate-")
        self.ctx = _ctx(self.tmp.name)
        from nomorals.agents.partner_runtime import PartnerRuntime

        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = self.ctx
        rt._owner_chats = {"telegram:1"}
        rt._owner_sender_ids = set()
        self.sent: list[dict[str, Any]] = []

        def _send(platform: str, chat: Any, path: str, **kw: Any) -> Any:
            self.sent.append({"path": path,
                              "bytes": Path(path).read_bytes(), **kw})
            return SimpleNamespace(ok=True)

        rt.gateway = SimpleNamespace(send_file=_send)
        self.rt = rt

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmp.cleanup()

    @staticmethod
    def _msg(chat_key: str) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(key=chat_key),
                               text="", meta={}, sender_id="")

    def test_stranger_inline_decode_stays_public(self) -> None:
        reply = self.rt._control_decode(
            base64.b64encode(b"hello code beast").decode(),
            "telegram:999", message=self._msg("telegram:999"))
        self.assertIn("hello code beast", reply)
        self.assertIn("decoder limits:", reply)
        self.assertNotIn("owner-only", reply)

    def test_stranger_file_read_is_owner_only(self) -> None:
        reply = self.rt._control_decode(
            "file:secrets.txt", "telegram:999",
            message=self._msg("telegram:999"))
        self.assertIn("owner-only", reply)

    def test_stranger_history_is_owner_only(self) -> None:
        reply = self.rt._control_decode(
            "history", "telegram:999",
            message=self._msg("telegram:999"))
        self.assertIn("owner-only", reply)

    def test_stranger_binary_decode_stores_nothing(self) -> None:
        reply = self.rt._control_decode(
            base64.b64encode(PNG).decode(), "telegram:999",
            message=self._msg("telegram:999"))
        self.assertIn("nothing was stored", reply)
        self.assertEqual(self.sent, [])
        from nomorals.agents.decoder import _artifact_store_for

        store = _artifact_store_for(self.ctx)
        self.assertEqual(store.search(creator="decoder"), [])

    def test_owner_history_passes_the_gate(self) -> None:
        reply = self.rt._control_decode(
            "history", "telegram:1", message=self._msg("telegram:1"))
        self.assertNotIn("owner-only", reply)

    def test_owner_binary_decode_quarantines_and_sends(self) -> None:
        reply = self.rt._control_decode(
            base64.b64encode(PNG).decode(), "telegram:1",
            message=self._msg("telegram:1"))
        self.assertIn("decoder limits:", reply)
        # exactly one send, carrying the exact decoded bytes, through
        # a transient export that is gone afterwards (the store's
        # copy is the only durable one)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["bytes"], PNG)
        self.assertFalse(os.path.exists(self.sent[0]["path"]))
        from nomorals.agents.decoder import _artifact_store_for

        store = _artifact_store_for(self.ctx)
        rows = store.search(creator="decoder")
        self.assertEqual(len(rows), 1)
        self.assertEqual(store.read(rows[0].id), PNG)


# ────────────────────────── doc-19 §8 fixtures ─────────────────────────

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    pad = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * pad + (out or "")


class DocFixtureTests(unittest.TestCase):
    """The eval's D1 proof set: nested chains, PEM, JWT, base58,
    malformed input, an archive bomb, a script payload — all decoded
    as data, none executed."""

    def test_base64_gzip_json_chain(self) -> None:
        # The chain walker follows text outputs; a byte stage ends
        # the walk and the next analyze() continues it — the nested
        # fixture therefore lands in two honest steps.
        payload = json.dumps({"k": "v"}).encode()
        enc = base64.b64encode(gzip.compress(payload)).decode()
        report = analyze(enc)
        self.assertIsNotNone(report.best)
        self.assertEqual(report.best["chain"][0], "base64")
        stage = report.best["output"]
        self.assertIsInstance(stage, (bytes, bytearray))
        from nomorals.core.decoder import identify_magic

        self.assertEqual(identify_magic(bytes(stage)), "gzip")
        inner = analyze(bytes(stage))
        self.assertIsNotNone(inner.best)
        out = inner.best["output"]
        self.assertEqual(
            out if isinstance(out, str) else out.decode(), '{"k": "v"}')

    def test_pem_and_jwt_and_base58(self) -> None:
        pem = ("-----BEGIN TEST-----\n"
               + base64.b64encode(b"pem body").decode()
               + "\n-----END TEST-----")
        self.assertIsNotNone(analyze(pem).best)
        jwt = ".".join([
            base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("="),
            base64.urlsafe_b64encode(b'{"sub":"1234"}').decode().rstrip("="),
            "sig"])
        report = analyze(jwt)
        self.assertIsNotNone(report.jwt)
        report = analyze(_b58encode(b"hello"))
        self.assertIsNotNone(report.best)

    def test_malformed_input_never_raises(self) -> None:
        report = analyze("!!! not anything !!!")
        self.assertIsInstance(report.hits, list)

    def test_archive_bomb_decodes_as_data_only(self) -> None:
        bomb = gzip.compress(b"A" * 5_000_000)
        report = analyze(bomb)
        self.assertIsNotNone(report.best)
        out = report.best["output"]
        size = out["bytes"] if isinstance(out, dict) else len(out)
        self.assertEqual(size, 5_000_000)


if __name__ == "__main__":
    unittest.main()
