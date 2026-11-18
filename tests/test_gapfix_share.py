"""Expiring file shares (File.io build, 52-websites #26).

The organ mints a share over a file Devon already holds and hands
back a link that dies on its own: after the download budget is
spent or the expiry passes, the link 404s and the bytes are gone
from storage. The first serve returns the bytes and spends the
budget; the second finds nothing. A wrong token refuses and leaves
the share alive. An expired share dies on the sweep without any
fetch. The token never appears in the journal.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.storage.share import (
    DEFAULT_TTL_S,
    SHARE_JOB,
    SHARE_SPEC,
    SHARE_TOOL,
    ShareRecord,
    ShareStore,
    ensure_share_job,
    list_shares,
    mint_share,
    parse_share_expiry,
    revoke_share,
    run_share_tick_once,
    serve_share,
    sweep_shares,
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


def _blobs(tmp: str) -> tuple[Database, BlobStore]:
    db = Database(":memory:")
    db.migrate()
    return db, BlobStore(db, Path(tmp) / "blobs")


def _store(home: str, blobs: BlobStore) -> ShareStore:
    return ShareStore(home, blob_store=blobs)


def _file(tmp: str, name: str = "report.pdf", data: bytes = b"%PDF-1.4 fake\n") -> str:
    path = Path(tmp) / name
    path.write_bytes(data)
    return str(path)


class OrganTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db, self.blobs = _blobs(self.tmp.name)
        self.store = _store(self.tmp.name, self.blobs)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_mint_from_path_stores_once_and_links(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, now=1_000_000.0)
        self.assertTrue(out["ok"])
        self.assertEqual("report.pdf", out["filename"])
        self.assertEqual(len(b"%PDF-1.4 fake\n"), out["size"])
        self.assertEqual(1, out["downloads_left"])
        self.assertEqual(1_000_000.0 + DEFAULT_TTL_S, out["expires_at"])
        link = out["link"]
        self.assertTrue(link.startswith("/share/"))
        token = link[len("/share/"):]
        self.assertEqual(48, len(token))
        self.assertEqual(token, token.lower())
        self.assertTrue(all(c in "0123456789abcdef" for c in token))
        # the record holds the ref; the blob is really in the store
        rec = self.store.by_token(token)
        assert rec is not None
        self.assertEqual(1, self.blobs.info(rec.sha256).refcount)
        self.assertIsNotNone(self.blobs.info(rec.sha256))

    def test_mint_same_file_twice_dedupes(self) -> None:
        src = _file(self.tmp.name)
        first = mint_share(self.store, src, now=1_000_000.0)
        second = mint_share(self.store, src, now=1_000_000.0)
        rec_a = self.store.get(first["share_id"])
        rec_b = self.store.get(second["share_id"])
        assert rec_a is not None and rec_b is not None
        self.assertEqual(rec_a.sha256, rec_b.sha256)
        self.assertEqual(2, self.blobs.info(rec_a.sha256).refcount)
        self.assertEqual(1, self.blobs.describe()["blobs"])

    def test_serve_once_then_gone_and_bytes_dead(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, now=1_000_000.0)
        token = out["link"][len("/share/"):]
        status, payload = serve_share(self.store, token, now=1_000_001.0)
        self.assertEqual("ok", status)
        self.assertEqual(b"%PDF-1.4 fake\n", payload["bytes"])
        self.assertEqual(0, payload["downloads_left"])
        status, payload = serve_share(self.store, token, now=1_000_002.0)
        self.assertEqual("gone", status)
        rec = self.store.get(out["share_id"])
        self.assertIsNone(rec)
        # the bytes are gone from storage, not just flagged
        self.assertEqual(0, self.blobs.describe()["blobs"])

    def test_budget_three_serves_exactly_three(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=3, now=1_000_000.0)
        token = out["link"][len("/share/"):]
        for left in (2, 1, 0):
            status, payload = serve_share(self.store, token, now=1_000_001.0)
            self.assertEqual("ok", status)
            self.assertEqual(left, payload["downloads_left"])
        status, _ = serve_share(self.store, token, now=1_000_001.0)
        self.assertEqual("gone", status)

    def test_expired_share_refuses_and_is_deleted(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, expires_in_s=1, now=1_000_000.0)
        token = out["link"][len("/share/"):]
        status, _ = serve_share(self.store, token, now=1_000_002.0)
        self.assertEqual("gone", status)
        self.assertIsNone(self.store.get(out["share_id"]))
        self.assertEqual(0, self.blobs.describe()["blobs"])

    def test_sweep_deletes_expired_only_without_fetch(self) -> None:
        live = _file(self.tmp.name, "live.txt", b"live")
        stale = _file(self.tmp.name, "stale.txt", b"stale")
        mint_share(self.store, live, expires_in_s=10_000, now=1_000_000.0)
        dead = mint_share(self.store, stale, expires_in_s=1, now=1_000_000.0)
        swept = sweep_shares(self.store, now=1_000_002.0)
        self.assertEqual(1, swept)
        self.assertIsNone(self.store.get(dead["share_id"]))
        self.assertEqual(1, len(list_shares(self.store, now=1_000_002.0)))
        self.assertEqual(1, self.blobs.describe()["blobs"])

    def test_revoke_kills_early(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=5, now=1_000_000.0)
        self.assertTrue(revoke_share(self.store, out["share_id"]))
        self.assertFalse(revoke_share(self.store, out["share_id"]))
        status, _ = serve_share(self.store, out["link"][len("/share/"):],
                                now=1_000_001.0)
        self.assertEqual("gone", status)
        self.assertEqual(0, self.blobs.describe()["blobs"])

    def test_forged_token_refuses_and_share_survives(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=5, now=1_000_000.0)
        journal_before = len(self.store.journal_entries())
        status, _ = serve_share(self.store, "not-a-token", now=1_000_001.0)
        self.assertEqual("gone", status)
        status, _ = serve_share(self.store, "0" * 48, now=1_000_001.0)
        self.assertEqual("gone", status)
        self.assertEqual(journal_before, len(self.store.journal_entries()))
        rec = self.store.get(out["share_id"])
        assert rec is not None
        self.assertEqual(5, rec.downloads_left)

    def test_list_is_live_only(self) -> None:
        live = _file(self.tmp.name, "live.txt", b"live")
        stale = _file(self.tmp.name, "stale.txt", b"stale")
        mint_share(self.store, live, expires_in_s=10_000, now=1_000_000.0)
        dead = mint_share(self.store, stale, expires_in_s=1, now=1_000_000.0)
        listed = list_shares(self.store, now=1_000_002.0)
        self.assertEqual(1, len(listed))
        self.assertIsNone(self.store.get(dead["share_id"]))

    def test_parse_share_expiry(self) -> None:
        now = 1_000_000.0
        self.assertEqual(now + DEFAULT_TTL_S, parse_share_expiry(None, now=now))
        self.assertEqual(now + DEFAULT_TTL_S, parse_share_expiry("", now=now))
        self.assertIsNone(parse_share_expiry("never", now=now))
        self.assertEqual(now + 90, parse_share_expiry("90s", now=now))
        self.assertEqual(now + 2 * 86400, parse_share_expiry("2d", now=now))
        self.assertEqual(now + 604800, parse_share_expiry("1w", now=now))
        # 1_000_000.0 is a Monday (1970-01-12); friday is 4 days on, 23:59:59
        friday = parse_share_expiry("friday", now=now)
        assert friday is not None
        import datetime

        self.assertEqual(datetime.datetime(1970, 1, 16, 23, 59, 59).timestamp(),
                         friday)
        iso = parse_share_expiry("2026-12-25", now=now)
        assert iso is not None
        self.assertEqual(datetime.datetime(2026, 12, 25, 23, 59, 59).timestamp(),
                         iso)
        with self.assertRaises(ValueError):
            parse_share_expiry("soon", now=now)
        with self.assertRaises(ValueError):
            parse_share_expiry("2026-13-01", now=now)

    def test_mint_rejects_bad_source_and_budget(self) -> None:
        out = mint_share(self.store, "/no/such/file.bin", now=1_000_000.0)
        self.assertFalse(out["ok"])
        self.assertIn("no such file", out["error"])
        out = mint_share(self.store, "blob:" + "0" * 64, now=1_000_000.0)
        self.assertFalse(out["ok"])
        self.assertIn("no blob", out["error"])
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=0, now=1_000_000.0)
        self.assertFalse(out["ok"])
        out = mint_share(self.store, src, expires_in_s=-5, now=1_000_000.0)
        self.assertFalse(out["ok"])

    def test_blob_source_bumps_refcount_and_survives(self) -> None:
        info = self.blobs.put_bytes(b"shared bytes", mime="text/plain")
        out = mint_share(self.store, f"blob:{info.sha256}", downloads=1,
                         now=1_000_000.0)
        self.assertTrue(out["ok"])
        self.assertEqual(2, self.blobs.info(info.sha256).refcount)
        status, payload = serve_share(self.store, out["link"][len("/share/"):],
                                      now=1_000_001.0)
        self.assertEqual("ok", status)
        self.assertEqual(b"shared bytes", payload["bytes"])
        # the share died, but the original holder's blob survives
        self.assertEqual(1, self.blobs.info(info.sha256).refcount)
        self.assertEqual(b"shared bytes",
                         self.blobs.get_bytes(info.sha256))

    def test_never_expiry(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, never=True, now=1_000_000.0)
        self.assertTrue(out["ok"])
        self.assertIsNone(out["expires_at"])
        status, _ = serve_share(self.store, out["link"][len("/share/"):],
                                now=1_000_000.0 + 10 * 365 * 86400)
        self.assertEqual("ok", status)

    def test_journal_never_contains_the_token(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=1, now=1_000_000.0)
        token = out["link"][len("/share/"):]
        serve_share(self.store, token, now=1_000_001.0)
        sweep_shares(self.store, now=1_000_002.0)
        blob = json.dumps(self.store.journal_entries())
        self.assertNotIn(token, blob)

    def test_store_survives_reload(self) -> None:
        src = _file(self.tmp.name)
        out = mint_share(self.store, src, downloads=3, now=1_000_000.0)
        again = ShareStore(self.tmp.name, blob_store=self.blobs)
        rec = again.get(out["share_id"])
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(3, rec.downloads_left)

    def test_corrupt_store_quarantined_not_silent(self) -> None:
        home = Path(self.tmp.name)
        (home / "data").mkdir(parents=True, exist_ok=True)
        (home / "data" / "shares.json").write_text("{not json", encoding="utf-8")
        store = ShareStore(self.tmp.name, blob_store=self.blobs)
        self.assertEqual({}, store.shares)
        quarantined = list((home / "data").glob("shares.json.corrupt-*"))
        self.assertEqual(1, len(quarantined))

    def test_record_validation(self) -> None:
        with self.assertRaises(ValueError):
            ShareRecord(share_id="x", file_ref="path/to/file", filename="f",
                        mime="text/plain", size=1, token="a" * 48,
                        downloads_left=1)
        with self.assertRaises(ValueError):
            ShareRecord(share_id="x", file_ref="blob:" + "a" * 64,
                        filename="f", mime="text/plain", size=1,
                        token="UPPERCASE", downloads_left=1)


class ToolTests(unittest.TestCase):
    def test_tool_registered_by_builtins(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register_builtins()
        self.assertIn("share_file", reg._tools)

    def _tool(self, home: str):
        from nomorals.tools.share import register

        ctx = SimpleNamespace(settings=SimpleNamespace(
            home=home,
            api=SimpleNamespace(host="127.0.0.1", port=8731)))
        deco = _DecoRegistry(context=ctx)
        register(deco)
        return deco.registered["share_file"]

    def test_mint_list_revoke_through_tool(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            src = _file(home, "notes.txt", b"hello share")
            out = fn(action="mint", file=src, downloads=2, expire="2d")
            self.assertTrue(out["ok"])
            self.assertTrue(out["link"].startswith(
                "http://127.0.0.1:8731/share/"))
            self.assertIn("anyone holding this link", out["note"])
            listed = fn(action="list")
            self.assertEqual(1, len(listed["shares"]))
            self.assertEqual("notes.txt", listed["shares"][0]["filename"])
            revoked = fn(action="revoke", share_id=out["share_id"])
            self.assertTrue(revoked["ok"])
            self.assertEqual([], fn(action="list")["shares"])

    def test_tool_status_and_unknown_action(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            status = fn(action="status")
            self.assertTrue(status["ok"])
            self.assertEqual(0, status["live"])
            self.assertEqual(SHARE_JOB, status["job"])
            self.assertEqual(SHARE_SPEC, status["spec"])
            with self.assertRaises(ValueError):
                fn(action="explode")
            with self.assertRaises(ValueError):
                fn(action="mint", file="/nope.bin", expire="soon")

    def test_tool_mint_missing_file_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            fn = self._tool(home)
            out = fn(action="mint", file="/no/such/file.bin")
            self.assertFalse(out["ok"])
            self.assertIn("no such file", out["error"])


class TickTests(unittest.TestCase):
    def test_tick_tool_registered_and_never_raises(self) -> None:
        from nomorals.tools.autonomy import register

        deco = _DecoRegistry(context=None)
        register(deco)
        self.assertIn("__share_tick__", deco.registered)
        out = deco.registered["__share_tick__"]()
        self.assertTrue(out["ok"])
        self.assertEqual(0, out["swept"])

    def test_job_shape_and_ensure(self) -> None:
        self.assertEqual("share sweep", SHARE_JOB)
        self.assertEqual("0 * * * *", SHARE_SPEC)
        self.assertEqual("__share_tick__", SHARE_TOOL)

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
        self.assertIsNone(ensure_share_job(None))
        first = ensure_share_job(sched)
        self.assertEqual(SHARE_JOB, first["name"])
        self.assertEqual(SHARE_SPEC, first["spec"])
        self.assertEqual({"tool": SHARE_TOOL, "args": {}},
                         first["payload"])
        second = ensure_share_job(sched)
        self.assertIs(second, first)

    def test_tick_sweeps_expired(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            db, blobs = _blobs(home)
            try:
                store = _store(home, blobs)
                src = _file(home, "stale.txt", b"stale")
                mint_share(store, src, expires_in_s=1, now=1_000_000.0)
                ctx = SimpleNamespace(
                    settings=SimpleNamespace(home=home))
                out = run_share_tick_once(ctx, now=1_000_002.0)
                self.assertTrue(out["ok"])
                self.assertEqual(1, out["swept"])
                self.assertEqual(0, out["live"])
            finally:
                db.close()


class ChatRegistrationTests(unittest.TestCase):
    def test_registered_everywhere_owner_only(self) -> None:
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
            parse_control,
        )

        self.assertIn("share", CONTROL_COMMANDS)
        self.assertIn("share", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["share"].get("what"))
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertIn("share", grouped)
        self.assertEqual(grouped.count("share"), 1)
        self.assertIn("/share <file>",
                      COMMAND_DETAILS["share"]["usage"])
        self.assertNotIn("share", PUBLIC_COMMANDS)
        self.assertEqual(279, len(PUBLIC_COMMANDS))
        self.assertFalse(caller_may_run("share", False))
        self.assertTrue(caller_may_run("share", True))
        self.assertIn("/share", list_catalog())
        cmd = parse_control("/share list")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "share")
        self.assertEqual(cmd.tail, "list")

    def test_handler_mints_lists_revokes_and_never_raises(self) -> None:
        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

        with tempfile.TemporaryDirectory() as home:
            db_path = Path(home) / "data" / "nomorals.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db = Database(str(db_path))
            try:
                db.migrate()
            finally:
                db.close()
            src = _file(home, "deck.pdf", b"%PDF-1.4 deck\n")
            runtime = RuntimeMetaMixin.__new__(RuntimeMetaMixin)
            runtime.context = SimpleNamespace(settings=SimpleNamespace(
                home=home,
                api=SimpleNamespace(host="127.0.0.1", port=8731)))
            handler = runtime._control_share
            self.assertIn("usage", handler("").lower())
            minted = handler(f"{src} downloads 2 --expire 2d")
            self.assertIn("http://127.0.0.1:8731/share/", minted)
            self.assertIn("2 download(s) left", minted)
            listed = handler("list")
            self.assertIn("deck.pdf", listed)
            self.assertIn("2 left", listed)
            share_id = listed.split("• ", 1)[1].split(" ", 1)[0]
            self.assertIn("revoked", handler(f"revoke {share_id}").lower())
            self.assertIn("no live shares", handler("list").lower())
            self.assertIn("bad share expiry",
                          handler(f"{src} --expire soon").lower())
            self.assertIn("could not share",
                          handler("/no/such/file.bin").lower())


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        db_path = Path(self.home) / "data" / "nomorals.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(str(db_path))
        self.db.migrate()
        self.store = ShareStore(self.home)
        src = _file(self.home, "deck.pdf", b"%PDF-1.4 deck\n")
        out = mint_share(self.store, src, downloads=1)
        assert out["ok"]
        self.link = out["link"]
        from nomorals.api.server import APIServer, _make_handler

        ctx = SimpleNamespace(
            db=None, router=None, tools=None, bus=None, memory=None,
            settings=SimpleNamespace(
                api=SimpleNamespace(token="", max_body_mb=0),
                home=self.home))
        srv = APIServer(ctx, token="")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                         _make_handler(srv))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.db.close()
        self.tmp.cleanup()

    def _get(self, path: str):
        return urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10)

    def test_download_once_then_404(self) -> None:
        with self._get(self.link) as resp:
            self.assertEqual(200, resp.status)
            self.assertEqual(b"%PDF-1.4 deck\n", resp.read())
            self.assertEqual("application/pdf",
                             resp.headers.get_content_type())
            disp = resp.headers.get("Content-Disposition") or ""
            self.assertIn("attachment", disp)
            self.assertIn("deck.pdf", disp)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get(self.link)
        self.assertEqual(404, caught.exception.code)

    def test_unknown_token_404s(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/share/" + "0" * 48)
        self.assertEqual(404, caught.exception.code)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/share/not-a-token")
        self.assertEqual(404, caught.exception.code)
