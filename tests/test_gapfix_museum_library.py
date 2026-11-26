"""gapfix74: Museum + Library — artifacts by human name, zip/unzip on
demand, class shelves, a central library, and interaction between them.

Failing-first: on the ship base none of `Museum.search/zip_card/
unzip_card/attach_companion/cite`, `Library.classify/classes/
central_library/export_book_pdf`, the `museum_library` brain tool, the
`/library` command, or the new `/museum` verbs exist.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.museum import Museum
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    list_catalog,
    parse_control,
)
from nomorals.social.chat.telegram_menu import validate_menu
from tests.test_books_library import BOOK_TEXT
from tests.test_gapfix_museum_command import _msg
from tests.test_gapfix_museum_send import _SendRecordingGateway

SCRATCH = Path.home() / "workspace" / "museum-library-verify"


def _settings(tmp: str):
    return load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.owner_chats": "5478650254",
        "partner.owner_sender_ids": "5478650254",
    })


def _context(tmp: str):
    settings = _settings(tmp)
    return build_context(settings, with_executor=False, with_router=False,
                         with_tools=True)


def _runtime(ctx):
    class _Router:
        def chat(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("test must not call the model")

    ctx.router = _Router()
    return PartnerRuntime(ctx, gateway=_SendRecordingGateway())


class _FakeLibrary:
    """Minimal library stand-in for pure-Museum unit tests."""

    def __init__(self, slugs: list[str]) -> None:
        self._slugs = slugs

    def list_books(self) -> list[dict[str, Any]]:
        return [{"slug": s, "title": s, "chapters": 1, "tags": []}
                for s in self._slugs]


def _ns(**kwargs: Any) -> argparse.Namespace:
    base = {"action": "list", "target": "", "find": "", "title": "",
            "author": "", "chapter": 0, "offset": 0, "limit": 8,
            "label": "", "quote": "", "note_text": "", "name": "",
            "create": False, "delete": False, "add": "", "remove": "",
            "set": None, "stars": 0, "id": 0, "list": False,
            "chapters": 1, "from_chapter": 0, "direction": "",
            "premise": "", "genre": "fantasy", "for_whom": "",
            "theme": "", "status": False, "import_": False, "path": "",
            "book_class": "", "json": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _museum_ns(**kwargs: Any) -> argparse.Namespace:
    base = {"museum_action": None, "museum_rest": [], "json": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _seed_downloads(downloads: Path) -> None:
    downloads.mkdir(parents=True, exist_ok=True)
    (downloads / "golden-hour.wav").write_bytes(b"RIFF" + b"\0" * 100)
    (downloads / "market-anthem.mp3").write_bytes(b"ID3" + b"\0" * 200)
    (downloads / "field-notes.txt").write_text(
        "notes from the field\n", encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inside.txt", "packed away\n")
    (downloads / "old-pack.zip").write_bytes(buf.getvalue())


class MuseumUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-museum-lib-", dir=str(SCRATCH))
        self.root = Path(self.tmp.name)
        self.downloads = self.root / "data" / "downloads"
        _seed_downloads(self.downloads)
        self.store = self.root / "museum"
        self.archives = self.root / "workspace" / "archives"
        self.museum = Museum(
            downloads_root=self.downloads,
            archives_root=self.archives,
            museum_root=self.store,
            library=_FakeLibrary(["dragon-codex"]))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_downloads_exhibitor_kinds_and_shelves(self) -> None:
        res = self.museum.census()
        kinds = sorted(c["kind"] for c in res["artifacts"])
        self.assertIn("audio-wav", kinds)
        self.assertIn("audio-music", kinds)
        self.assertIn("download", kinds)
        self.assertIn("archive", kinds)
        shelves = res["by_shelf"]
        self.assertIn("downloads-audio", shelves)
        self.assertIn("downloads-files", shelves)
        wav = next(c for c in res["artifacts"]
                   if c["kind"] == "audio-wav")
        self.assertEqual(wav["title"], "golden-hour.wav")
        self.assertTrue(wav["sendable"])
        self.assertTrue(any(t.startswith("date:20")
                            for t in wav["tags"]))

    def test_search_by_name_kind_date(self) -> None:
        by_name = self.museum.search("golden-hour")
        self.assertEqual(by_name["count"], 1)
        self.assertEqual(by_name["artifacts"][0]["kind"], "audio-wav")
        by_kind = self.museum.search("", kind="audio-music")
        self.assertEqual(by_kind["count"], 1)
        self.assertEqual(by_kind["artifacts"][0]["title"],
                         "market-anthem.mp3")
        date = next(c for c in self.museum.census()["artifacts"]
                    if c["kind"] == "audio-wav")["tags"]
        day = next(t[5:] for t in date if t.startswith("date:"))
        by_date = self.museum.search("", date=day)
        self.assertGreaterEqual(by_date["count"], 3)
        self.assertIn("golden-hour.wav",
                      [c["title"] for c in by_date["artifacts"]])
        none = self.museum.search("no-such-artifact-xyz")
        self.assertEqual(none["count"], 0)
        self.assertIn("no artifacts match", none["card_text"])

    def test_zip_unzip_round_trip(self) -> None:
        zipped = self.museum.zip_card("downloads-audio:golden-hour.wav")
        zpath = Path(zipped["zip"])
        self.assertTrue(zpath.is_file())
        self.assertTrue(str(zpath).startswith(str(self.store / "zips")))
        self.assertEqual(zipped["files"], 1)
        # The zip itself is now a museum card on the next census.
        res = self.museum.census()
        self.assertIn("museum-zips", res["by_shelf"])
        zcard = next(c for c in res["artifacts"]
                     if c["kind"] == "museum-zip")
        out = self.museum.unzip_card(zcard["id"])
        dest = Path(out["destination"])
        self.assertTrue((dest / "golden-hour.wav").is_file())
        self.assertEqual(out["files"], 1)
        self.assertEqual(out["skipped"], [])

    def test_zip_refuses_exhibit_only_and_unknown(self) -> None:
        with self.assertRaises(ValueError):
            self.museum.zip_card("no-such-card")
        with self.assertRaises(ValueError):
            self.museum.unzip_card("downloads-audio:golden-hour.wav")

    def test_unzip_skips_zip_slip_members(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "escape\n")
            zf.writestr("/abs.txt", "absolute\n")
            zf.writestr("good.txt", "stays\n")
        evil = self.downloads / "evil.zip"
        evil.write_bytes(buf.getvalue())
        out = self.museum.unzip_card("downloads-files:evil.zip")
        self.assertEqual(out["files"], 1)
        self.assertEqual(len(out["skipped"]), 2)
        self.assertFalse((self.root / "evil.txt").exists())
        dest = Path(out["destination"])
        self.assertTrue((dest / "good.txt").is_file())

    def test_companions_attach_cite_detach(self) -> None:
        ref = "downloads-audio:golden-hour.wav"
        attached = self.museum.attach_companion(
            "dragon-codex", ref, note="theme music")
        self.assertEqual(attached["book"], "dragon-codex")
        self.assertEqual(attached["artifact"], ref)
        cited = self.museum.cite("dragon-codex")
        self.assertEqual(cited["count"], 1)
        self.assertEqual(cited["artifacts"][0]["kind"], "audio-wav")
        listed = self.museum.companions_for_book("dragon-codex")
        self.assertEqual(listed[0]["note"], "theme music")
        self.assertFalse(listed[0]["missing"])
        # Persisted: a fresh Museum over the same store sees it.
        again = Museum(downloads_root=self.downloads,
                       museum_root=self.store,
                       library=_FakeLibrary(["dragon-codex"]))
        self.assertEqual(again.cite("dragon-codex")["count"], 1)
        detached = self.museum.detach_companion("dragon-codex", ref)
        self.assertEqual(detached["removed"], ref)
        self.assertEqual(self.museum.cite("dragon-codex")["count"], 0)

    def test_companions_refuse_unknown_book_and_artifact(self) -> None:
        with self.assertRaises(ValueError):
            self.museum.attach_companion(
                "no-such-book", "downloads-audio:golden-hour.wav")
        with self.assertRaises(ValueError):
            self.museum.attach_companion(
                "dragon-codex", "downloads-audio:nope.wav")

    def test_cite_reports_vanished_artifact_as_missing(self) -> None:
        ref = "downloads-audio:golden-hour.wav"
        self.museum.attach_companion("dragon-codex", ref)
        (self.downloads / "golden-hour.wav").unlink()
        cited = self.museum.cite("dragon-codex")
        self.assertEqual(cited["count"], 0)
        self.assertEqual(cited["missing"], [ref])


class LibraryClassesTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-lib-class-", dir=str(SCRATCH))
        self.ctx = _context(self.tmp.name)
        book = Path(self.tmp.name) / "dragon.txt"
        book.write_text(BOOK_TEXT, encoding="utf-8")
        ingested = self.ctx.tools.call(
            "library_ingest", path=str(book), title="Dragon Codex")
        self.assertTrue(ingested.ok, ingested)
        self.slug = ingested.value["slug"]

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.close()
        self.tmp.cleanup()

    def _lib(self):
        from nomorals.books.library import Library

        return Library(self.ctx)

    def test_classify_classes_shelf_central(self) -> None:
        lib = self._lib()
        out = lib.classify(self.slug, "Fantasy")
        self.assertEqual(out["class"], "fantasy")
        self.assertEqual(out["collection"], "class:fantasy")
        self.assertEqual(lib.book_classes(self.slug), ["fantasy"])
        self.assertEqual(
            [c["class"] for c in lib.classes()], ["fantasy"])
        shelf = lib.class_shelf("fantasy")
        self.assertEqual(shelf["class"], "fantasy")
        self.assertEqual([b["slug"] for b in shelf["books"]],
                         [self.slug])
        central = lib.central_library()
        self.assertEqual(central["total_books"], 1)
        self.assertEqual(central["classes"][0]["class"], "fantasy")
        self.assertEqual(central["unclassified"], [])
        # A second book stays unclassified until it is classed.
        book2 = Path(self.tmp.name) / "second.txt"
        book2.write_text(BOOK_TEXT.replace("Dragon", "Phoenix"),
                         encoding="utf-8")
        ing2 = self.ctx.tools.call(
            "library_ingest", path=str(book2), title="Phoenix Codex")
        self.assertTrue(ing2.ok, ing2)
        central = lib.central_library()
        self.assertEqual(central["total_books"], 2)
        self.assertEqual(len(central["unclassified"]), 1)
        # Declassify returns the book to unclassified; the empty
        # shelf itself stays (like remove_from_collection).
        lib.declassify(self.slug, "fantasy")
        self.assertEqual(lib.book_classes(self.slug), [])
        self.assertEqual(lib.classes(),
                         [{"class": "fantasy", "books": 0,
                           "collection": "class:fantasy"}])
        self.assertEqual(lib.class_shelf("fantasy")["books"], [])

    def test_classify_unknown_book_refused(self) -> None:
        from nomorals.books.library import LibraryError

        with self.assertRaises(LibraryError):
            self._lib().classify("no-such-book", "fantasy")

    def test_export_book_pdf_is_real_pdf(self) -> None:
        lib = self._lib()
        out = lib.export_book_pdf(self.slug)
        pdf = Path(out["pdf"])
        self.assertTrue(pdf.is_file())
        self.assertGreater(out["pdf_bytes"], 500)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        self.assertEqual(out["slug"], self.slug)
        self.assertEqual(out["chapters"],
                         len(lib.load(self.slug)["chapters"]))


class MuseumLibraryToolTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ml-tool-", dir=str(SCRATCH))
        self.ctx = _context(self.tmp.name)
        downloads = (Path(self.tmp.name) / "data" / "downloads")
        _seed_downloads(downloads)
        book = Path(self.tmp.name) / "dragon.txt"
        book.write_text(BOOK_TEXT, encoding="utf-8")
        ingested = self.ctx.tools.call(
            "library_ingest", path=str(book), title="Dragon Codex")
        self.assertTrue(ingested.ok, ingested)
        self.slug = ingested.value["slug"]

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.close()
        self.tmp.cleanup()

    def _call(self, **kwargs: Any):
        outcome = self.ctx.tools.call("museum_library", **kwargs)
        self.assertTrue(outcome.ok, outcome)
        return outcome.value

    def test_tool_count_is_800_with_museum_library(self) -> None:
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        # 785 after C.8 added the robot tool; 786 after gapfix74 added museum_library; 786 after gapfix74
        # added museum_library.
        self.assertEqual(len(registry._tools), 802)  # trendclip adds 5 tools
        self.assertIn("museum_library", registry._tools)

    def test_unified_search_reaches_museum_and_library(self) -> None:
        res = self._call(action="search", query="golden-hour")
        self.assertEqual(len(res["museum_cards"]), 1)
        self.assertEqual(res["museum_cards"][0]["kind"], "audio-wav")
        res = self._call(action="search", query="Dragon Codex")
        self.assertTrue(any(b["slug"] == self.slug
                            for b in res["library_books"]))
        self.assertIn("Dragon Codex", res["card_text"])

    def test_find_zip_unzip_through_tool(self) -> None:
        found = self._call(action="find",
                           ref="downloads-audio:golden-hour.wav")
        self.assertEqual(found["card"]["kind"], "audio-wav")
        zipped = self._call(action="zip",
                            ref="downloads-audio:golden-hour.wav")
        self.assertTrue(Path(zipped["zip"]).is_file())
        res = self._call(action="search", query="golden-hour",
                         kind="museum-zip")
        self.assertEqual(len(res["museum_cards"]), 1)
        out = self._call(action="unzip",
                         ref=res["museum_cards"][0]["id"])
        self.assertEqual(out["files"], 1)

    def test_companion_cite_shelves_classify_through_tool(self) -> None:
        self._call(action="classify", slug=self.slug,
                   class_name="Fantasy")
        shelves = self._call(action="shelves")
        self.assertEqual(shelves["classes"][0]["class"], "fantasy")
        self.assertIn("fantasy", shelves["card_text"])
        attached = self._call(
            action="companion", verb="attach", slug=self.slug,
            ref="downloads-audio:golden-hour.wav", note="theme")
        self.assertEqual(attached["artifact"],
                         "downloads-audio:golden-hour.wav")
        cited = self._call(action="cite", slug=self.slug)
        self.assertEqual(cited["count"], 1)
        self.assertEqual(cited["artifacts"][0]["title"],
                         "golden-hour.wav")
        listed = self._call(action="companion", verb="list",
                            slug=self.slug)
        self.assertEqual(len(listed["companions"]), 1)

    def test_unknown_action_refused(self) -> None:
        outcome = self.ctx.tools.call("museum_library",
                                      action="dance")
        self.assertFalse(outcome.ok)


class MuseumLibraryChatTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ml-chat-", dir=str(SCRATCH))
        self.ctx = _context(self.tmp.name)
        downloads = (Path(self.tmp.name) / "data" / "downloads")
        _seed_downloads(downloads)
        book = Path(self.tmp.name) / "dragon.txt"
        book.write_text(BOOK_TEXT, encoding="utf-8")
        ingested = self.ctx.tools.call(
            "library_ingest", path=str(book), title="Dragon Codex")
        self.assertTrue(ingested.ok, ingested)
        self.slug = ingested.value["slug"]
        self.runtime = _runtime(self.ctx)
        self.gateway = self.runtime.gateway

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        with contextlib.suppress(Exception):
            self.ctx.close()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True) -> str:
        return self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, owner=owner))

    def test_library_registered_everywhere(self) -> None:
        self.assertIn("library", CONTROL_COMMANDS)
        self.assertIn("library", COMMAND_DETAILS)
        self.assertTrue(COMMAND_DETAILS["library"].get("what"))
        self.assertIn("library", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("library"), 1)
        self.assertIn("/library", COMMAND_DETAILS["library"]["usage"])
        self.assertNotIn("library", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("library", False))
        self.assertTrue(caller_may_run("library", True))
        self.assertIn("/library", list_catalog())
        ok, problems = validate_menu()
        self.assertTrue(ok, problems)

    def test_museum_arity_widened_for_new_verbs(self) -> None:
        self.assertEqual(CONTROL_COMMANDS["museum"], (0, None))
        cmd = parse_control("/museum search golden hour")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "museum")

    def test_library_bare_and_shelf(self) -> None:
        reply = self._run("/library")
        self.assertIn("Central Library", reply)
        self.assertIn("Dragon Codex", reply)
        self.assertIn("unclassified", reply)
        self._run(f"/library class {self.slug} fantasy")
        reply = self._run("/library shelf fantasy")
        self.assertIn("Dragon Codex", reply)
        self.assertIn("fantasy", reply)
        reply = self._run("/library")
        self.assertIn("fantasy", reply)
        self.assertNotIn("unclassified: 1", reply)

    def test_library_send_delivers_real_pdf(self) -> None:
        reply = self._run(f"/library send {self.slug}")
        self.assertIn("sent", reply)
        self.assertEqual(len(self.gateway.files), 1)
        sent = Path(self.gateway.files[0]["path"])
        self.assertTrue(sent.is_file())
        self.assertTrue(sent.read_bytes().startswith(b"%PDF"))
        self.assertEqual(sent.suffix, ".pdf")

    def test_museum_search_zip_attach_cite(self) -> None:
        reply = self._run("/museum search golden-hour")
        self.assertIn("golden-hour.wav", reply)
        self.assertIn("audio-wav", reply)
        reply = self._run("/museum zip downloads-audio:golden-hour.wav")
        self.assertIn("zipped", reply)
        self.assertIn(".zip", reply)
        reply = self._run(
            f"/museum attach {self.slug} "
            "downloads-audio:golden-hour.wav theme music")
        self.assertIn("attached", reply)
        reply = self._run(f"/museum cite {self.slug}")
        self.assertIn("golden-hour.wav", reply)
        reply = self._run(f"/museum companions {self.slug}")
        self.assertIn("theme music", reply)

    def test_books_library_verb_is_central_view(self) -> None:
        reply = self._run("/books library")
        self.assertIn("Central Library", reply)
        self.assertIn("Dragon Codex", reply)
        self.assertNotIn("writing", reply)

    def test_non_owner_denied_library(self) -> None:
        reply = self._run("/library", owner=False)
        self.assertIn("just for the owner", reply)
        self.assertNotIn("Central Library", reply)


class MuseumLibraryCliTest(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ml-cli-", dir=str(SCRATCH))
        self.ctx = _context(self.tmp.name)
        downloads = (Path(self.tmp.name) / "data" / "downloads")
        _seed_downloads(downloads)
        book = Path(self.tmp.name) / "dragon.txt"
        book.write_text(BOOK_TEXT, encoding="utf-8")
        ingested = self.ctx.tools.call(
            "library_ingest", path=str(book), title="Dragon Codex")
        self.assertTrue(ingested.ok, ingested)
        self.slug = ingested.value["slug"]

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.close()
        self.tmp.cleanup()

    def test_nm_museum_search_zip_unzip(self) -> None:
        from nomorals.cmdline.commands.museum import _cmd_museum

        rc = _cmd_museum(_museum_ns(museum_action="search",
                                    museum_rest=["golden-hour"]),
                         self.ctx)
        self.assertEqual(rc, 0)
        rc = _cmd_museum(_museum_ns(
            museum_action="zip",
            museum_rest=["downloads-audio:golden-hour.wav"]), self.ctx)
        self.assertEqual(rc, 0)
        out = self.ctx.tools.call("museum_library", action="search",
                                  query="golden-hour", kind="museum-zip")
        self.assertTrue(out.ok, out)
        zid = out.value["museum_cards"][0]["id"]
        rc = _cmd_museum(_museum_ns(museum_action="unzip",
                                    museum_rest=[zid]), self.ctx)
        self.assertEqual(rc, 0)

    def test_nm_books_class_central_export(self) -> None:
        from nomorals.cmdline.commands.books import _cmd_books

        self.assertEqual(_cmd_books(
            _ns(action="class", target=self.slug,
                book_class="Fantasy"), self.ctx), 0)
        self.assertEqual(_cmd_books(
            _ns(action="classes"), self.ctx), 0)
        self.assertEqual(_cmd_books(
            _ns(action="central"), self.ctx), 0)
        self.assertEqual(_cmd_books(
            _ns(action="export", target=self.slug), self.ctx), 0)
        lib_root = Path(self.tmp.name) / "workspace" / "library"
        pdf = lib_root / self.slug / f"{self.slug}.pdf"
        self.assertTrue(pdf.is_file())
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
