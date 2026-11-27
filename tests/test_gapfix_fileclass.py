"""File class (gapfix122) — zip/unzip, artifact pack/unpack, and the
museum/library connection, end to end over the same stores.

Real-build queue §5a (Nov 9, seq 191644): "The file class the zip
unzip and artifacts class and more like it... getting connected and
God tier." The organ is nomorals/files/: a File class over the
workspace's real shelves, packs built on the Archivist's bundle seam
(nomorals/archives.py), and a files exhibitor that rides the Museum
census — never a parallel file system.

Harness: the museum test harness — load_settings with a throwaway
home + build_context (no executor, no router), so every test runs
against real stores under a temp dir.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.cmdline.commands.file import _cmd_file
from nomorals.core.config import load_settings
from nomorals.files.chat import control_files, file_action
from nomorals.files.kinds import kind_for, register_kind
from nomorals.files.packs import pack_files, pack_info
from nomorals.files.store import FileStore

_OWNER = "5478650254"


def _ctx(tmp: Path):
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.owner_chats": _OWNER,
        "partner.owner_sender_ids": _OWNER,
    })
    return build_context(settings, with_executor=False, with_router=False,
                         with_tools=True)


def _seed_downloads(tmp: Path) -> None:
    dl = tmp / "data" / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    (dl / "song.mp3").write_bytes(b"ID3" + b"\x00" * 5000)
    (dl / "notes.txt").write_text("field notes\n" * 20)
    (dl / "codex.md").write_text("# The Dragon Codex\n\nA short book.\n" * 10)


def _pack_via_chat(ctx, title="codex-pack"):
    out = file_action(
        ctx, "pack",
        refs=("files:files-music/song.mp3", "downloads:notes.txt"),
        title=title)
    assert out.get("ok"), out
    return (out["pack"] or {}).get("path", "")


class KindTests(unittest.TestCase):
    def test_extensions_map_to_kinds(self):
        # kind_for returns a (kind, shelf) tuple (kinds.py docstring).
        self.assertEqual(kind_for(Path("a.mp3")),
                         ("audio-music", "files-music"))
        self.assertEqual(kind_for(Path("a.wav")),
                         ("audio-wav", "files-audio"))
        self.assertEqual(kind_for(Path("a.zip")),
                         ("archive", "files-archives"))
        self.assertEqual(kind_for(Path("a.pdf")),
                         ("document", "files-documents"))
        # .md is in the document extension set, not the book set
        # (book is .epub/.mobi/.azw3/.fb2/.cbz/.cbr).
        self.assertEqual(kind_for(Path("a.md")),
                         ("document", "files-documents"))
        self.assertEqual(kind_for(Path("a.png")),
                         ("image", "files-images"))
        self.assertEqual(kind_for(Path("a.py")),
                         ("code", "files-code"))
        # .csv is in the document set, not the data set
        # (data is .db/.sqlite/.parquet/.arrow/.npy/.npz/.pkl/.joblib).
        self.assertEqual(kind_for(Path("a.csv")),
                         ("document", "files-documents"))
        self.assertEqual(kind_for(Path("a.xyz")),
                         ("download", "files-other"))

    def test_magic_sniffs_archive_without_extension(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mystery"
            p.write_bytes(b"PK\x03\x04" + b"\x00" * 40)
            self.assertEqual(kind_for(p), ("archive", "files-archives"))

    def test_register_kind_opens_the_taxonomy(self):
        # register_kind(kind, shelf, extensions) — three args.
        register_kind("blueprint", "files-blueprints", (".bp",))
        self.assertEqual(kind_for(Path("house.bp")),
                         ("blueprint", "files-blueprints"))


class StoreTests(unittest.TestCase):
    def test_shelve_moves_download_onto_kind_shelf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            store = FileStore(ctx)
            f = store.shelve("downloads:song.mp3")
            self.assertEqual(f.shelf, "files-music")
            self.assertEqual(f.kind, "audio-music")
            self.assertFalse((tmp / "data/downloads/song.mp3").exists())
            self.assertTrue(f.path.is_file())
            listed = store.shelf("files-music")
            self.assertEqual([x.name for x in listed], ["song.mp3"])

    def test_unshelve_moves_back_to_downloads(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            store = FileStore(ctx)
            store.shelve("downloads:song.mp3")
            back = store.unshelve("files:files-music/song.mp3")
            self.assertTrue((tmp / "data/downloads/song.mp3").exists())
            # unshelve preserves the file's shelf: song.mp3 is
            # files-music under kind_for.
            self.assertEqual(back.shelf, "files-music")

    def test_shelves_list_and_downloads_intake(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            store = FileStore(ctx)
            # downloads() returns File objects, not name strings.
            self.assertEqual([f.name for f in store.downloads()],
                             ["codex.md", "notes.txt", "song.mp3"])
            store.shelve("downloads:song.mp3")
            # shelves() returns shelf dicts, not name strings.
            self.assertIn("files-music",
                          [s["shelf"] for s in store.shelves()])

    def test_all_files_capped(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            store = FileStore(ctx)
            store.shelve("downloads:song.mp3")
            self.assertEqual(len(store.all_files(cap=500)), 1)


class ResolveTests(unittest.TestCase):
    def test_plain_path_inside_workspace_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            out = file_action(ctx, "info",
                              ref=str(tmp / "data/downloads/song.mp3"))
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["file"]["kind"], "audio-music")

    def test_outside_roots_is_a_named_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            out = file_action(ctx, "info", ref="/etc/passwd")
            self.assertFalse(out["ok"])
            self.assertIn("outside the file roots", out["error"])

    def test_unknown_museum_ref_refused(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            out = file_action(ctx, "info", ref="museum:nope")
            self.assertFalse(out["ok"])
            self.assertIn("no museum card matches", out["error"])

    def test_archive_member_ref_reads_inside_the_pack(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            file_action(ctx, "shelve", ref="downloads:song.mp3")
            pack = _pack_via_chat(ctx)
            out = file_action(
                ctx, "info",
                ref=f"archive:{pack}::files-music/song.mp3")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["file"]["size"], 5003)
            # Archive members are deliberately kinded archive-member
            # (file.py:337), not the member's own extension kind.
            self.assertEqual(out["file"]["kind"], "archive-member")

    def test_archive_member_missing_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            file_action(ctx, "shelve", ref="downloads:song.mp3")
            pack = _pack_via_chat(ctx)
            out = file_action(ctx, "info",
                              ref=f"archive:{pack}::nope.txt")
            self.assertFalse(out["ok"])
            self.assertIn("no member", out["error"])


class PackTests(unittest.TestCase):
    def _packed(self, tmp, container="zip"):
        _seed_downloads(tmp)
        ctx = _ctx(tmp)
        file_action(ctx, "shelve", ref="downloads:song.mp3")
        out = file_action(
            ctx, "pack",
            refs=("files:files-music/song.mp3", "downloads:notes.txt"),
            title="codex-pack", container=container)
        assert out.get("ok"), out
        return ctx, (out["pack"] or {}).get("path", "")

    def test_zip_round_trip_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx, pack = self._packed(tmp, "zip")
            # pack_info(archive) — one arg; its dict has no "ok" key
            # and no pack["members"] (the member list is
            # info["members"], from the bundle manifest).
            info = pack_info(pack)
            self.assertFalse(info["legacy"], info)
            names = [m["name"] for m in info["members"]]
            self.assertEqual(names, ["files-music/song.mp3",
                                     "files-documents/notes.txt",
                                     ".files.json"])
            out = file_action(ctx, "unpack", ref=pack,
                              shelf="files-documents")
            self.assertTrue(out["ok"], out)
            self.assertIn("unpacked 4 file(s)", out["text"])
            landed = tmp / "data/files/files-documents"
            self.assertEqual(
                (landed / "files-music/song.mp3").read_bytes(),
                b"ID3" + b"\x00" * 5000)
            self.assertTrue(all(v["ok"] for v in out["unpack"]["verified"]))

    def test_tar_gz_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx, pack = self._packed(tmp, "tar.gz")
            self.assertTrue(pack.endswith(".tar.gz"))
            out = file_action(ctx, "unpack", ref=pack,
                              shelf="files-documents")
            self.assertTrue(out["ok"], out)
            self.assertIn("unpacked 4 file(s)", out["text"])

    def test_tampered_zip_refused_naming_the_member(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx, pack = self._packed(tmp, "zip")
            with zipfile.ZipFile(pack, "a") as zf:
                zf.writestr("files-music/song.mp3", b"tampered")
            out = file_action(ctx, "unpack", ref=pack,
                              shelf="files-other")
            self.assertFalse(out["ok"])
            self.assertIn("refused", out["error"])
            self.assertIn("files-music/song.mp3", out["error"])
            self.assertIn("size mismatch", out["error"])

    def test_tampered_tar_gz_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ctx, pack = self._packed(tmp, "tar.gz")
            # Rewrite the pack with the song's tar member bytes changed.
            with tempfile.TemporaryDirectory() as td2:
                work = Path(td2)
                with tarfile.open(pack, "r:gz") as tf:
                    tf.extractall(work)
                song = work / "files-music/song.mp3"
                song.write_bytes(b"tampered")
                with tarfile.open(pack, "w:gz") as tf:
                    for member in sorted(work.rglob("*")):
                        if member.is_file():
                            tf.add(member, arcname=str(
                                member.relative_to(work)))
            out = file_action(ctx, "unpack", ref=pack,
                              shelf="files-other")
            self.assertFalse(out["ok"])
            self.assertIn("refused", out["error"])

    def test_unsafe_member_refuses_the_whole_pack(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            # Hand-craft the bundle (create_bundle itself refuses
            # unsafe member names at build time, so the escape has
            # to be smuggled in): a zip whose manifest honestly
            # lists ../evil.txt with a matching hash, so the
            # refusal comes from verification's unsafe-member
            # check (verify_bundle), not from extraction.
            # The pack must sit UNDER the file roots: resolve()
            # refuses a pack outside them before verification is
            # ever reached.
            import hashlib
            evil = b"escape\n"
            manifest = {
                "format": "nomorals-bundle", "version": 1,
                "created_by": "test", "kind": "files-artifact",
                "members": [{"name": "../evil.txt", "size": len(evil),
                             "sha256": hashlib.sha256(evil).hexdigest(),
                             "is_dir": False}],
            }
            pack = tmp / "data" / "downloads" / "evil.zip"
            with zipfile.ZipFile(pack, "w") as zf:
                zf.writestr("../evil.txt", evil)
                zf.writestr("bundle.manifest.json",
                            json.dumps(manifest).encode())
            out = file_action(ctx, "unpack", ref="downloads:evil.zip",
                              shelf="files-other")
            self.assertFalse(out["ok"])
            self.assertIn("unsafe member", out["error"])
            self.assertFalse((tmp / "evil.txt").exists())
            self.assertNotIn("evil.txt",
                             [p.name for p in
                              (tmp / "data/files/files-other").glob("*")])

    def test_extract_bomb_budget_trips(self):
        from nomorals.archives import create_bundle, extract_bundle
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            srcs = []
            for i in range(3):
                p = tmp / f"f{i}.txt"
                p.write_text("x")
                srcs.append((f"f{i}.txt", p))
            pack = tmp / "many.zip"
            create_bundle(pack, srcs, created_by="test",
                          kind="files-artifact", container="zip")
            with self.assertRaises(Exception) as cm:
                extract_bundle(pack, tmp / "out", max_files=2)
            self.assertIn("2", str(cm.exception))

    def test_pack_files_direct_and_pack_info(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            store = FileStore(ctx)
            store.shelve("downloads:song.mp3")
            # pack_files takes refs: list[str] and returns one dict.
            manifest = pack_files(
                ctx, ["files:files-music/song.mp3"], title="direct",
                container="zip", store=store)
            dest = Path(manifest["path"])
            self.assertTrue(dest.is_file())
            self.assertEqual(manifest["kind"], "files-artifact")
            # pack_info(archive) — one arg, and its dict has no "ok"
            # key (open_bundle shape: path/format/legacy/manifest/
            # members, plus pack/files added by pack_info).
            info = pack_info(dest)
            self.assertFalse(info["legacy"], info)
            self.assertEqual(info["pack"]["title"], "direct")


class ChatTests(unittest.TestCase):
    def test_control_files_verbs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            out = control_files("shelve downloads:song.mp3", context=ctx)
            self.assertIn("shelved song.mp3", out)
            out = control_files("list files-music", context=ctx)
            self.assertIn("song.mp3", out)
            out = control_files("info files:files-music/song.mp3",
                                context=ctx)
            self.assertIn("audio-music", out)
            out = control_files("pack files:files-music/song.mp3 "
                                "downloads:notes.txt --title t1",
                                context=ctx)
            self.assertIn("packed 2 file(s)", out)
            pack = str(tmp / "data/files/packs/t1.zip")
            out = control_files(f"unpack {pack} --shelf files-documents",
                                context=ctx)
            self.assertIn("unpacked 4 file(s)", out)
            out = control_files(f"packinfo {pack}", context=ctx)
            # The packinfo reply names the pack and its member count,
            # not the member list.
            self.assertIn("3 member(s)", out)
            out = control_files("exhibit files-music", context=ctx)
            self.assertIn("song.mp3", out)
            out = control_files("bogusverb", context=ctx)
            # An unknown verb falls through to the usage line
            # (chat.py:274); the "unknown file action" error lives
            # in file_action's dict, not in the chat reply.
            self.assertIn("usage: /files", out)

    def test_control_files_bare_lists(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            out = control_files("", context=ctx)
            self.assertIn(
                "the file store is empty — shelve a file with "
                "/files shelve <ref>", out)

    def test_files_command_is_registered_owner_only(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS, caller_may_run
        self.assertIn("files", CONTROL_COMMANDS)
        kind, arity = CONTROL_COMMANDS["files"]
        self.assertEqual((kind, arity), (0, None))
        self.assertFalse(caller_may_run("files", False))
        self.assertTrue(caller_may_run("files", True))

    def test_send_rides_the_live_file_send(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            file_action(ctx, "shelve", ref="downloads:song.mp3")
            sent = []

            class _Tools:
                def call(self, name, **kwargs):
                    sent.append((name, kwargs))
                    # The send path reads outcome.value (chat.py:115);
                    # a real registry call result carries .value.
                    return types.SimpleNamespace(ok=True, value={})

            ctx.tools = _Tools()
            out = file_action(ctx, "send",
                              ref="files:files-music/song.mp3",
                              platform="telegram", chat_id="123",
                              caption="for you")
            self.assertTrue(out["ok"], out)
            self.assertEqual(sent[0][0], "file_send")
            self.assertEqual(sent[0][1]["chat_id"], "123")
            self.assertTrue(sent[0][1]["path"].endswith("song.mp3"))

    def test_send_without_tool_caller_is_a_named_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            file_action(ctx, "shelve", ref="downloads:song.mp3")
            ctx.tools = None
            out = file_action(ctx, "send",
                              ref="files:files-music/song.mp3",
                              platform="telegram", chat_id="123")
            self.assertFalse(out["ok"])
            self.assertIn("no tool caller attached", out["error"])


class MuseumLibraryTests(unittest.TestCase):
    def test_exhibit_sees_shelved_files_in_the_census(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            file_action(ctx, "shelve", ref="downloads:song.mp3")
            out = file_action(ctx, "exhibit")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["count"], 1)
            self.assertEqual(out["files"][0]["title"], "song.mp3")
            self.assertEqual(out["files"][0]["shelf"], "files-music")

    def test_ingest_book_into_library_then_library_ref_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            out = file_action(ctx, "ingest", ref="downloads:codex.md")
            self.assertTrue(out["ok"], out)
            self.assertIn("ingested codex.md", out["text"])
            slug = out["book"].get("slug", "")
            self.assertTrue(slug, out)
            resolved = file_action(ctx, "info", ref=f"library:{slug}")
            self.assertTrue(resolved["ok"], resolved)
            # The library ref is kinded by extension: codex.md is a
            # document. (Parent decision: kind-ing library refs as
            # "book" would be a design change, not ordered here.)
            self.assertEqual(resolved["file"]["kind"], "document")

    def test_ingest_non_book_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            out = file_action(ctx, "ingest", ref="downloads:song.mp3")
            self.assertFalse(out["ok"])
            self.assertIn("is a audio-music", out["error"])


class CliTests(unittest.TestCase):
    def _run(self, ctx, task, as_json=False):
        args = types.SimpleNamespace(task=task, json=as_json)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_file(args, ctx)
        return rc, buf.getvalue()

    def test_cli_shelve_list_pack(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            rc, out = self._run(ctx, ["shelve", "downloads:song.mp3"])
            self.assertEqual(rc, 0)
            self.assertIn("shelved song.mp3", out)
            rc, out = self._run(ctx, ["list", "files-music"])
            self.assertEqual(rc, 0)
            self.assertIn("song.mp3", out)
            rc, out = self._run(ctx, ["pack", "files:files-music/song.mp3",
                                      "downloads:notes.txt",
                                      "--dest", "cli-pack.zip"])
            self.assertEqual(rc, 0, out)
            self.assertIn("packed 2 file(s)", out)

    def test_cli_bare_lists_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            rc, out = self._run(ctx, [])
            self.assertEqual(rc, 0)
            # Bare list on an empty store returns the JSON dict
            # ({"ok": true, "shelves": []}) — no text reply.
            self.assertEqual(json.loads(out)["shelves"], [])
            rc, out = self._run(ctx, ["list"], as_json=True)
            self.assertEqual(rc, 0)
            # Bare list returns {"ok": true, "shelves": [...]} — no
            # "count" key (count only appears for a named shelf).
            self.assertEqual(json.loads(out)["shelves"], [])

    def test_cli_unknown_verb_is_usage_error(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(Path(td))
            rc, _ = self._run(ctx, ["bogusverb"])
            self.assertEqual(rc, 2)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nomorals.tools.registry import ToolRegistry
        cls._ns = types.SimpleNamespace()
        cls.registry = ToolRegistry(cls._ns)
        cls.registry.register_builtins()

    def test_tool_is_registered_and_count_is_787(self):
        # 786 after the audit N4 lane added build_swarm;
        # 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control. 809 after this lane (moneyprinter) added make_short. 810 after this lane (VoiceStudio) added voicestudio; 811 after this lane (Animato) added animato_spark.
        self.assertIn("file", self.registry._tools)
        self.assertEqual(len(self.registry._tools), 810)

    def test_tool_info_list_pack(self):
        from nomorals.core.policy import CapabilitySet
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _seed_downloads(tmp)
            ctx = _ctx(tmp)
            outcome = self.registry.call(
                "file", actor="test", capabilities=CapabilitySet.all(),
                context=ctx, action="shelve", ref="downloads:song.mp3")
            self.assertTrue(outcome.ok, outcome)
            value = outcome.unwrap()
            self.assertTrue(value["ok"], value)
            self.assertIn("shelved song.mp3", value["text"])
            outcome = self.registry.call(
                "file", actor="test", capabilities=CapabilitySet.all(),
                context=ctx, action="list", shelf="files-music")
            self.assertTrue(outcome.ok, outcome)
            self.assertEqual(outcome.unwrap()["count"], 1)


if __name__ == "__main__":
    unittest.main()
