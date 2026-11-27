"""Burn-after-reading notes (Privnote build-new, 52-websites #25).

The organ seals a note under a fresh random key, stores only the
ciphertext, and hands the key back inside the link — the key never
lives in the store, the plaintext never lives anywhere but the one
read. The first read returns the text and deletes the row; a second
read finds nothing. A wrong code or a wrong password refuses and
leaves the note sealed. An expired note refuses and is deleted.
"""

from __future__ import annotations

import argparse
import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.memory.ephemeral_notes import (
    BURN_NOTE_JOB,
    BURN_NOTE_SPEC,
    BURN_NOTE_TOOL,
    DEFAULT_TTL_S,
    BurnNoteStore,
    ensure_burn_note_job,
    list_notes,
    make_link,
    parse_expiry,
    parse_link,
    purge_note,
    read_note,
    run_burn_note_tick_once,
    seal_note,
    sweep_expired,
)


class _DecoRegistry:
    """A registry double that captures register() calls."""

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.registered: dict[str, Any] = {}

    def register(self, name: str, fn: Any = None, **_kw: Any) -> Any:
        def deco(f: Any) -> Any:
            self.registered[name] = f
            return f

        if fn is not None:
            return deco(fn)
        return deco


def _store(tmp: str) -> BurnNoteStore:
    return BurnNoteStore(tmp)


class OrganTests(unittest.TestCase):
    def test_seal_read_once_then_gone(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            sealed = seal_note(store, "meet at the mango tree at six",
                               now=1_000_000.0)
            self.assertTrue(sealed["ok"])
            link = sealed["link"]
            self.assertTrue(link.startswith("burn:"))
            out = read_note(store, link, now=1_000_100.0)
            self.assertTrue(out["ok"])
            self.assertEqual("meet at the mango tree at six", out["text"])
            # The row is GONE from the store, not flagged.
            self.assertEqual([], list_notes(store))
            second = read_note(store, link, now=1_000_200.0)
            self.assertFalse(second["ok"])
            self.assertIn("gone", second["error"])
            # And the burn is on the record.
            burned = store.burned_log()
            self.assertEqual(1, len(burned))
            self.assertEqual(sealed["note_id"], burned[0]["note_id"])
            self.assertEqual("read", burned[0]["reason"])

    def test_plaintext_and_key_never_in_store_file(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            secret = "the vault code is 48151623"
            sealed = seal_note(store, secret, now=1_000_000.0)
            raw = Path(store.path).read_bytes()
            self.assertNotIn(secret.encode(), raw)
            _note_id, key = parse_link(sealed["link"])
            self.assertNotIn(base64.urlsafe_b64encode(key), raw)
            # Ciphertext is not the plaintext.
            row = store._data["notes"][sealed["note_id"]]
            self.assertNotEqual(secret, row["ciphertext"])
            self.assertIn("nmc1:", row["ciphertext"])

    def test_same_text_seals_differently(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            a = seal_note(store, "same words", now=1_000_000.0)
            b = seal_note(store, "same words", now=1_000_000.0)
            self.assertNotEqual(a["link"], b["link"])
            self.assertNotEqual(a["note_id"], b["note_id"])
            ra = store._data["notes"][a["note_id"]]
            rb = store._data["notes"][b["note_id"]]
            self.assertNotEqual(ra["ciphertext"], rb["ciphertext"])

    def test_wrong_key_refuses_and_note_survives(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            sealed = seal_note(store, "still here", now=1_000_000.0)
            note_id, _key = parse_link(sealed["link"])
            forged = make_link(note_id, b"\x00" * 32)
            out = read_note(store, forged, now=1_000_100.0)
            self.assertFalse(out["ok"])
            self.assertIn("wrong", out["error"].lower())
            # The note is still sealed and readable with the real link.
            again = read_note(store, sealed["link"], now=1_000_200.0)
            self.assertTrue(again["ok"])
            self.assertEqual("still here", again["text"])

    def test_malformed_link_refuses(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            for bad in ("", "burn:", "burn:abc", "burn:abc:!!!",
                        "http://example.com/x", "burn:abc:AAAA:extra"):
                out = read_note(store, bad, now=1_000_000.0)
                self.assertFalse(out["ok"], bad)
                self.assertIn("code", out["error"].lower())

    def test_password_gates_the_read(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            sealed = seal_note(store, "behind a word", password="mango",
                               now=1_000_000.0)
            self.assertTrue(sealed["has_password"])
            # No password, wrong password: refused, note survives.
            self.assertFalse(read_note(store, sealed["link"],
                                      now=1_000_100.0)["ok"])
            self.assertFalse(read_note(store, sealed["link"],
                                      password="banana",
                                      now=1_000_100.0)["ok"])
            self.assertEqual(1, len(list_notes(store)))
            # Right password: reads once, then gone.
            out = read_note(store, sealed["link"], password="mango",
                            now=1_000_200.0)
            self.assertTrue(out["ok"])
            self.assertEqual("behind a word", out["text"])
            self.assertEqual([], list_notes(store))
            # The password hash is not the password.
            raw = Path(store.path).read_bytes()
            self.assertNotIn(b"mango", raw)

    def test_expired_note_refuses_and_is_deleted(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            sealed = seal_note(store, "too late", expires_in_s=60,
                               now=1_000_000.0)
            out = read_note(store, sealed["link"], now=1_000_061.0)
            self.assertFalse(out["ok"])
            self.assertIn("expir", out["error"].lower())
            self.assertEqual([], list_notes(store))
            burned = store.burned_log()
            self.assertEqual("expired", burned[0]["reason"])

    def test_sweep_deletes_expired_only(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            old = seal_note(store, "old", expires_in_s=60,
                            now=1_000_000.0)
            live = seal_note(store, "live", expires_in_s=3600,
                             now=1_000_000.0)
            self.assertEqual(1, sweep_expired(store, now=1_000_061.0))
            self.assertEqual(0, sweep_expired(store, now=1_000_061.0))
            ids = [n["note_id"] for n in list_notes(store)]
            self.assertEqual([live["note_id"]], ids)
            self.assertNotIn(old["note_id"], ids)

    def test_purge_deletes_unread(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            sealed = seal_note(store, "changed my mind",
                               now=1_000_000.0)
            self.assertTrue(purge_note(store, sealed["note_id"]))
            self.assertFalse(purge_note(store, sealed["note_id"]))
            self.assertEqual([], list_notes(store))
            self.assertEqual("purged",
                             store.burned_log()[0]["reason"])

    def test_list_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            seal_note(store, "hidden words", password="x",
                      now=1_000_000.0)
            (meta,) = list_notes(store)
            self.assertNotIn("text", meta)
            self.assertNotIn("ciphertext", meta)
            self.assertNotIn("key", meta)
            self.assertTrue(meta["has_password"])
            self.assertEqual(1, meta["reads_left"])
            self.assertIn("expires_at", meta)

    def test_parse_expiry(self):
        self.assertEqual(3600.0, parse_expiry("1h"))
        self.assertEqual(90.0, parse_expiry("90s"))
        self.assertEqual(2 * 86400.0, parse_expiry("2d"))
        self.assertEqual(7 * 86400.0, parse_expiry("1w"))
        self.assertIsNone(parse_expiry("never"))
        self.assertIsNone(parse_expiry(""))
        self.assertEqual(DEFAULT_TTL_S, parse_expiry(None))
        for bad in ("0", "-1h", "1x", "hour", "1.5.2h"):
            with self.assertRaises(ValueError, msg=bad):
                parse_expiry(bad)

    def test_seal_rejects_empty_and_huge(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            with self.assertRaises(ValueError):
                seal_note(store, "   ", now=1_000_000.0)
            with self.assertRaises(ValueError):
                seal_note(store, "x" * 100_001, now=1_000_000.0)

    def test_store_survives_reload(self):
        with tempfile.TemporaryDirectory() as home:
            sealed = seal_note(_store(home), "across restarts",
                               now=1_000_000.0)
            again = BurnNoteStore(home)
            out = read_note(again, sealed["link"], now=1_000_100.0)
            self.assertTrue(out["ok"])
            self.assertEqual("across restarts", out["text"])


class ToolTests(unittest.TestCase):
    def test_tool_registered_by_builtins(self):
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("burn_note", reg._tools)

    def test_seal_and_read_through_tool(self):
        from nomorals.tools.burn_note import register

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            deco = _DecoRegistry(context=ctx)
            register(deco)
            self.assertIn("burn_note", deco.registered)
            fn = deco.registered["burn_note"]
            sealed = fn(action="seal", text="through the tool")
            self.assertTrue(sealed["ok"])
            link = sealed["link"]
            out = fn(action="read", code=link)
            self.assertTrue(out["ok"])
            self.assertEqual("through the tool", out["text"])
            self.assertFalse(fn(action="read", code=link)["ok"])
            listed = fn(action="list")
            self.assertEqual([], listed["notes"])

    def test_tool_status_and_unknown_action(self):
        from nomorals.tools.burn_note import register

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            deco = _DecoRegistry(context=ctx)
            register(deco)
            fn = deco.registered["burn_note"]
            status = fn(action="status")
            self.assertTrue(status["ok"])
            self.assertEqual(0, status["live"])
            self.assertEqual(BURN_NOTE_JOB, status["job"])
            with self.assertRaises(ValueError):
                fn(action="explode")


class TickTests(unittest.TestCase):
    def test_tick_tool_registered_and_never_raises(self):
        from nomorals.tools.autonomy import register

        deco = _DecoRegistry(context=None)
        register(deco)
        self.assertIn("__burn_note_tick__", deco.registered)
        out = deco.registered["__burn_note_tick__"]()
        self.assertTrue(out["ok"])
        self.assertEqual(0, out["swept"])

    def test_job_shape_and_ensure(self):
        self.assertEqual("burn note sweep", BURN_NOTE_JOB)
        self.assertEqual("0 * * * *", BURN_NOTE_SPEC)
        self.assertEqual("__burn_note_tick__", BURN_NOTE_TOOL)

        class _Sched:
            def __init__(self) -> None:
                self.jobs: list[dict[str, Any]] = []

            def list_jobs(self) -> list[dict[str, Any]]:
                return list(self.jobs)

            def add(self, name: str, spec: str, kind: str,
                    payload: dict[str, Any], **_kw: Any) -> dict[str, Any]:
                job = {"name": name, "spec": spec, "kind": kind,
                       "payload": payload}
                self.jobs.append(job)
                return job

            def remove(self, job_id: Any) -> None:
                self.jobs = [j for j in self.jobs
                             if j.get("id") != job_id]

        sched = _Sched()
        self.assertIsNone(ensure_burn_note_job(None))
        first = ensure_burn_note_job(sched)
        self.assertEqual(BURN_NOTE_JOB, first["name"])
        self.assertEqual(BURN_NOTE_SPEC, first["spec"])
        self.assertEqual({"tool": BURN_NOTE_TOOL, "args": {}},
                         first["payload"])
        second = ensure_burn_note_job(sched)
        self.assertIs(second, first)

    def test_tick_sweeps_expired(self):
        with tempfile.TemporaryDirectory() as home:
            store = _store(home)
            seal_note(store, "stale", expires_in_s=1, now=1_000_000.0)
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            out = run_burn_note_tick_once(ctx, now=1_000_002.0)
            self.assertTrue(out["ok"])
            self.assertEqual(1, out["swept"])
            self.assertEqual(0, out["live"])


class ChatRegistrationTests(unittest.TestCase):
    def test_registered_everywhere_owner_only(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("burn", CONTROL_COMMANDS)
        self.assertIn("burn", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["burn"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("burn", grouped)
        self.assertEqual(grouped.count("burn"), 1)
        self.assertIn("/burn seal <text>",
                      COMMAND_DETAILS["burn"]["usage"])
        self.assertNotIn("burn", PUBLIC_COMMANDS)
        self.assertEqual(287, len(PUBLIC_COMMANDS))
        self.assertFalse(caller_may_run("burn", False))
        self.assertTrue(caller_may_run("burn", True))
        self.assertIn("/burn", list_catalog())
        cmd = parse_control("/burn seal hello")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "burn")
        self.assertEqual(cmd.tail, "seal hello")

    def test_handler_seals_reads_and_never_raises(self):
        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

        with tempfile.TemporaryDirectory() as home:
            runtime = RuntimeMetaMixin.__new__(RuntimeMetaMixin)
            runtime.context = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            handler = runtime._control_burn
            self.assertIn("usage", handler("").lower())
            sealed = handler("seal the eagle has landed")
            self.assertIn("burn:", sealed)
            link = sealed.split("burn:", 1)[1].split()[0]
            code = f"burn:{link}"
            out = handler(f"read {code}")
            self.assertIn("the eagle has landed", out)
            self.assertIn("gone", handler(f"read {code}").lower())
            self.assertIn("usage", handler("read").lower())


class CliTests(unittest.TestCase):
    def _args(self, **kw: Any) -> argparse.Namespace:
        base = {"burn_action": "", "burn_rest": [], "json": False}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_nm_burn_end_to_end(self) -> None:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from nomorals.cmdline.commands.burn import _cmd_burn

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_burn(self._args(
                    burn_action="seal",
                    burn_rest=["the", "eagle", "has", "landed"]), ctx)
            self.assertEqual(0, rc)
            out = buf.getvalue()
            self.assertIn("burn:", out)
            link = out.split("burn:", 1)[1].split()[0]
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_burn(self._args(
                    burn_action="read",
                    burn_rest=[f"burn:{link}"]), ctx)
            self.assertEqual(0, rc)
            self.assertIn("the eagle has landed", buf.getvalue())
            buf = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(buf), redirect_stderr(err):
                rc = _cmd_burn(self._args(
                    burn_action="read",
                    burn_rest=[f"burn:{link}"]), ctx)
            self.assertEqual(1, rc)
            self.assertIn("gone", err.getvalue().lower())

    def test_nm_burn_usage_and_list(self) -> None:
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.burn import _cmd_burn

        with tempfile.TemporaryDirectory() as home:
            ctx = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(2, _cmd_burn(self._args(), ctx))
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(0, _cmd_burn(
                    self._args(burn_action="list"), ctx))
            self.assertIn("no live burn notes", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
