"""WS1-P6 — the file family's two connection fixes.

(A) ``archives.extract_for_downloader`` was dead code (zero
production callers).  It is now wired into the media downloader's
one success funnel (``MediaDownloader._succeed``): a download that
lands as an archive container is unpacked beside the original —
inspect-first, all-members-safe or refused whole — and the outcome
rides ``DownloadReport.archive`` / the summary.  The original
archive is retained at ``report.path``.

(B) Inbound chat files were never registered as artifacts.  The
chat gateway's inbound funnel (the seam every adapter's messages
pass through, beside the identity/session side-registrations) now
registers every saved file — real bytes on disk — in the injected
ArtifactStore with chat provenance, deduped by content, so the
/artifact surface (P4) reaches it.  Registration is fail-open by
design: a broken store never drops a message.

All offline: fake adapters, a patched direct-download stage, real
stores over temp dirs.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.media.downloader import MediaDownloader
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    MediaRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

# ── shared fixtures ────────────────────────────────────────────────


def _dl_ctx(tmp: str):
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=tmp), db=None,
        home=Path(tmp))


def _make_store(tmp: str) -> ArtifactStore:
    db = Database(":memory:")
    db.migrate()
    return ArtifactStore(db, BlobStore(db, Path(tmp) / "blobs"))


class _FakeAdapter(ChatAdapter):
    name = "fake"

    def run(self, handler):
        pass

    def send(self, chat: ChatRef, text: str, *,
             reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name)


def _file_msg(path: Path, *, kind: str = "document",
              name: str = "", platform: str = "whatsapp",
              chat_id: str = "2348012345678", message_id: str = "m-1",
              sender: str = "Ada", sender_id: str = "2348099999999",
              view_once: bool = False, meta: dict | None = None,
              mime: str = "") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id,
                     kind=ChatKind.DM),
        incoming=True, text="", sender=sender, sender_id=sender_id,
        media=[MediaRef(path=str(path), kind=kind,
                        name=name or path.name, view_once=view_once,
                        mime=mime)],
        ts=1_730_000_000.0, message_id=message_id,
        meta=dict(meta or {}))


# ── (A) downloader archive wire ────────────────────────────────────


class DownloaderArchiveWireTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_dl_ctx(self.tmp), direct_retries=1,
                                  max_proxies=0)

    def _land(self, payload_name: str, payload: bytes):
        """A direct stage that 'downloads' the given bytes."""
        def direct(url, dest, **k):
            p = Path(dest) / payload_name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
            return {"url": url, "path": str(p), "bytes": len(payload),
                    "title": payload_name, "extractor": "fake",
                    "seconds": 0.1}
        return direct

    @staticmethod
    def _zip_bytes(members: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def test_downloaded_zip_arrives_extracted(self):
        payload = self._zip_bytes({
            "notes.txt": b"hello archive",
            "docs/readme.md": b"# inside\n"})
        with patch("nomorals.tools.media.download",
                   side_effect=self._land("bundle.zip", payload)):
            report = self.dl.download("https://x.test/bundle.zip")
        self.assertTrue(report.ok, report.summary())
        # The original archive is retained at report.path.
        self.assertTrue(Path(report.path).is_file())
        self.assertEqual(Path(report.path).name, "bundle.zip")
        # …and it arrived extracted, additively, beside it.
        self.assertIsNotNone(report.archive)
        assert report.archive is not None
        self.assertFalse(report.archive["blocked"])
        self.assertEqual(report.archive["extracted"], 2)
        dest = Path(report.archive["dest"])
        self.assertEqual((dest / "notes.txt").read_bytes(),
                         b"hello archive")
        self.assertEqual((dest / "docs" / "readme.md").read_bytes(),
                         b"# inside\n")
        # The outcome is reported, never silent.
        self.assertIn("archive: extracted 2 file(s)",
                      report.summary())

    def test_downloaded_targz_arrives_extracted(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = b"tar member body"
            info = tarfile.TarInfo("inner.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        with patch("nomorals.tools.media.download",
                   side_effect=self._land("pack.tar.gz",
                                          buf.getvalue())):
            report = self.dl.download("https://x.test/pack.tar.gz")
        self.assertTrue(report.ok, report.summary())
        assert report.archive is not None
        self.assertFalse(report.archive["blocked"])
        dest = Path(report.archive["dest"])
        self.assertEqual((dest / "inner.txt").read_bytes(),
                         b"tar member body")

    def test_hostile_zip_is_refused_whole_and_named(self):
        payload = self._zip_bytes({
            "good.txt": b"fine",
            "../evil.txt": b"escape"})
        with patch("nomorals.tools.media.download",
                   side_effect=self._land("evil.zip", payload)):
            report = self.dl.download("https://x.test/evil.zip")
        # The download itself landed; the ARCHIVE was refused, and
        # the refusal is on the report, not silent.
        self.assertTrue(report.ok, report.summary())
        assert report.archive is not None
        self.assertTrue(report.archive["blocked"])
        self.assertEqual(report.archive["extracted"], 0)
        self.assertIn("unsafe", report.archive["reason"])
        self.assertIn("NOT extracted", report.summary())
        # Nothing escaped, and not even the good member extracted:
        # a hostile whole is refused whole.
        self.assertFalse((Path(self.tmp) / "evil.txt").exists())
        self.assertFalse(
            (Path(report.path).parent / "evil.zip.extracted").exists())

    def test_non_archive_download_has_no_archive_block(self):
        with patch("nomorals.tools.media.download",
                   side_effect=self._land("song.mp3", b"data")):
            report = self.dl.download("https://x.test/a.mp3")
        self.assertTrue(report.ok, report.summary())
        self.assertIsNone(report.archive)


# ── (B) inbound files → artifacts ──────────────────────────────────


class GatewayArtifactRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)
        self.media_dir = Path(self.tmp) / "media"
        self.media_dir.mkdir()

    def _gateway(self, store=None) -> ChatGateway:
        gw = ChatGateway({"fake": _FakeAdapter()},
                         artifact_store=store)
        self.seen: list[ChatMessage] = []
        gw._inbound = self.seen.append
        return gw

    def _saved_file(self, name: str, body: bytes) -> Path:
        p = self.media_dir / name
        p.write_bytes(body)
        return p

    def test_inbound_file_lands_as_artifact_with_provenance(self):
        path = self._saved_file("notes.txt",
                                b"the zebra ledger balances\n")
        gw = self._gateway(self.store)
        gw._on_inbound(_file_msg(path))
        # The message flowed to the handler untouched…
        self.assertEqual(len(self.seen), 1)
        # …and the file is an artifact now, carrying its provenance.
        arts = self.store.search(creator="chat_inbound")
        self.assertEqual(len(arts), 1)
        art = arts[0]
        self.assertEqual(self.store.read(art.id),
                         b"the zebra ledger balances\n")
        md = art.metadata
        self.assertEqual(md["origin"], "chat_inbound")
        self.assertEqual(md["platform"], "whatsapp")
        self.assertEqual(md["chat"], "whatsapp:2348012345678")
        self.assertEqual(md["sender"], "Ada")
        self.assertEqual(md["sender_id"], "2348099999999")
        self.assertEqual(md["message_id"], "m-1")
        self.assertEqual(md["kind"], "document")
        self.assertEqual(md["filename"], "notes.txt")
        self.assertEqual(art.provenance.source_type, "chat")
        self.assertEqual(art.provenance.source_id,
                         "whatsapp:2348012345678")
        # The id is stamped on the message for downstream consumers.
        self.assertEqual(self.seen[0].meta["artifact_ids"], [art.id])
        # P4's surface reaches it: content search finds the body.
        hits = self.store.search_content("zebra")
        self.assertEqual([h.artifact.id for h in hits], [art.id])

    def test_duplicate_send_dedups_to_one_row(self):
        body = b"same bytes, sent twice\n"
        first = self._saved_file("first.txt", body)
        second = self._saved_file("second.txt", body)
        gw = self._gateway(self.store)
        gw._on_inbound(_file_msg(first, message_id="m-1"))
        gw._on_inbound(_file_msg(second, message_id="m-2",
                                 name="second.txt"))
        arts = self.store.search(creator="chat_inbound")
        self.assertEqual(len(arts), 1)
        # Both arrivals resolve to the SAME artifact id.
        self.assertEqual(self.seen[0].meta["artifact_ids"],
                         self.seen[1].meta["artifact_ids"])
        # The blob layer deduped too (one stored body).
        self.assertEqual(self.store.blobs.stats["dedup_hits"], 0)
        rows = self.store.find_by_content_hash(arts[0].content_hash)
        self.assertEqual(len(rows), 1)

    def test_registration_failure_never_breaks_the_message(self):
        class _BrokenStore:
            def find_by_content_hash(self, digest):
                raise RuntimeError("db on fire")

        path = self._saved_file("notes.txt", b"payload\n")
        gw = self._gateway(_BrokenStore())
        gw._on_inbound(_file_msg(path))  # must not raise
        self.assertEqual(len(self.seen), 1)
        self.assertNotIn("artifact_ids", self.seen[0].meta)

    def test_no_store_means_no_registration_and_no_breakage(self):
        path = self._saved_file("notes.txt", b"payload\n")
        gw = self._gateway(None)
        gw._on_inbound(_file_msg(path))
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.store.search(creator="chat_inbound"),
                         [])

    def test_view_once_and_service_media_stay_with_the_vault(self):
        vo = self._saved_file("once.jpg", b"jpeg-bytes")
        gw = self._gateway(self.store)
        gw._on_inbound(_file_msg(vo, kind="image", view_once=True))
        status = self._saved_file("status.jpg", b"jpeg-bytes-2")
        gw._on_inbound(_file_msg(
            status, kind="image", meta={"service": "status"}))
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(self.store.search(creator="chat_inbound"),
                         [])

    def test_missing_or_empty_file_registers_nothing(self):
        gw = self._gateway(self.store)
        ghost = self.media_dir / "gone.pdf"
        gw._on_inbound(_file_msg(ghost))
        empty = self._saved_file("empty.txt", b"")
        gw._on_inbound(_file_msg(empty, message_id="m-2"))
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(self.store.search(creator="chat_inbound"),
                         [])


# ── (B) the /artifact surface itself reaches the inbound file ─────


class ArtifactCommandReachesInboundFileTests(unittest.TestCase):
    """End-to-end at the chat surface: a file arriving through a
    gateway wired with the runtime's store is found by the real
    /artifact search command (P4) on that same store."""

    def setUp(self):
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings

        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ws1p6-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a, **k):
                raise AssertionError("test must not call the model")

        self.context.router = _Router()

    def tearDown(self):
        with suppress(Exception):
            self.context.close()
        self.tmp.cleanup()

    def test_artifact_search_finds_the_inbound_file(self):
        from pathlib import Path as _P

        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.documents import artifact_body_text
        from nomorals.storage.artifacts import ArtifactStore
        from nomorals.storage.blob import BlobStore

        db = self.context.db
        store = ArtifactStore(
            db, BlobStore(db, _P(db.path).parent / "blobs"),
            text_extractor=artifact_body_text)

        class _NullGateway:
            adapters: dict = {}

            def status(self):
                return {}

            def stop(self):
                return None

            def set_rate_limit(self, *a, **k) -> None:
                pass

            def typing(self, *a, **k) -> None:
                pass

            def send(self, platform, chat, text, **kwargs):
                return SimpleNamespace(ok=True, error="")

        runtime = PartnerRuntime(self.context, gateway=_NullGateway())
        try:
            media = _P(self.tmp.name) / "inbound-report.txt"
            media.write_bytes(b"the zebra ledger balances at dawn\n")
            gw = ChatGateway({"fake": _FakeAdapter()},
                             artifact_store=store)
            seen: list[ChatMessage] = []
            gw._inbound = seen.append
            gw._on_inbound(_file_msg(
                media, platform="telegram", chat_id="999",
                message_id="tg-1", sender="owner",
                sender_id="5478650254"))
            self.assertEqual(len(seen), 1)
            query = _file_msg(
                media, platform="telegram", chat_id="999",
                message_id="q-1", sender="owner",
                sender_id="5478650254")
            query.media = []
            query.meta["is_owner"] = True
            out = runtime.handle_control(
                "/artifact search zebra", "telegram:999",
                message=query)
            self.assertIn("zebra", out)
            arts = store.search(creator="chat_inbound")
            self.assertEqual(len(arts), 1)
            self.assertIn(arts[0].id, out)
        finally:
            runtime.stop()

    def test_runtime_wires_the_store_into_its_own_gateway(self):
        # The production wiring (agents/partner/runtime.py): when
        # the runtime builds its own gateway, it injects an
        # ArtifactStore over the context db — the registration
        # above is live in production, not just in this suite.
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.artifacts import ArtifactStore

        runtime = PartnerRuntime(self.context)
        try:
            self.assertIsNotNone(runtime.gateway)
            store = runtime.gateway.artifact_store
            self.assertIsInstance(store, ArtifactStore)
            self.assertIs(store.db, self.context.db)
        finally:
            runtime.stop()


if __name__ == "__main__":
    unittest.main()
