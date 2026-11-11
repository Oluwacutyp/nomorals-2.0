"""Doc 16 — ``/read get pdf|epub``: the routed editions.

P4 shipped the output router with two file actions (the branded
HTML edition, the raw original) and named the PDF/EPUB renditions
a router-owned later pass: a routed long document could be read
and downloaded as HTML/txt, but its PDF or EPUB rendition — the
forms /books export already produced for stories — could not be
pulled through /read.  This suite pins the closing of that gap:

- ``OutputRouter.file_for(doc, "pdf")`` renders the branded PDF
  through the SAME generator the /books export uses
  (``core.pdf.render_pdf`` with the brand mark resolved through
  ``core.branding``): completeness line as front matter, the line
  + the manifest revision stamp in the /Info metadata, the
  designed watermark on the pages — and caches it beside the
  HTML edition, content-addressed;
- ``file_for(doc, "epub")`` renders through the SAME EPUB
  generator (``books.forge.epub_from_chapters`` — injected into
  the router by the /read surface, because books sits a layer
  above social): title page carrying the completeness line + the
  brand line, the revision + completeness stamps in the OPF;
- ``/read get <token> pdf|epub`` delivers the rendition as a file
  through the gateway, refuses in words for an unknown token, a
  document gone from the store, and a generation failure (the
  generator's own error, secret-redacted);
- the long route's introduction and buttons advertise all four
  actions, and /read stays owner-only, exactly as gated before.

Fake data only; every store lives under a per-test tmp path.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.pdf import read_pdf_text
from nomorals.social.chat import output_router as or_module
from nomorals.social.chat.output_router import OutputRouter

KEY = "chat:telegram:42"
CHAT_KEY = "telegram:42"

PARAS = [
    f"Rendition paragraph {i:02d} carries its own marker R{i:02d} "
    "and a full sentence of prose to read."
    for i in range(60)
]
LONG = (
    "Weekly research digest\n\n"
    + "\n\n".join(PARAS)
    + "\n\nClosing note. RENDITION-TAIL-SENTINEL"
)
LONG_ROUTE = (LONG + "\n\n") * 12  # past the reader+edition band


def _norm(text: str) -> str:
    return " ".join(text.split())


def _router(tmp_path) -> OutputRouter:
    return OutputRouter(tmp_path / "longdoc")


def _production_router(tmp_path) -> OutputRouter:
    """A router with the production EPUB renderer installed, the
    way the /read surface installs it."""
    from nomorals.agents.partner.runtime_search import _read_epub_bytes

    router = _router(tmp_path)
    router.epub_renderer = _read_epub_bytes
    return router


def _streams(data: bytes) -> list[bytes]:
    """Decompressed content streams of a rendered PDF."""
    import zlib

    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            out.append(zlib.decompress(m.group(1)))
        except zlib.error:
            continue
    return out


# ── the PDF rendition ─────────────────────────────────────────────


class TestPdfRendition:
    def test_pdf_file_is_a_real_branded_pdf(self, tmp_path):
        router = _router(tmp_path)
        manifest = router.store_text(LONG)
        found = router.file_for(manifest.document_id, "pdf")
        assert found is not None
        path, name = found
        assert name.endswith(".pdf")
        data = path.read_bytes()
        assert data.startswith(b"%PDF")
        text = _norm(read_pdf_text(data))
        # the whole body rides: sampled paragraphs + the tail
        for para in (PARAS[0], PARAS[17], PARAS[42], PARAS[59]):
            assert _norm(para) in text
        assert "RENDITION-TAIL-SENTINEL" in text
        # the stamps: completeness in the body, the line + the
        # manifest revision in the /Info metadata (uncompressed)
        assert "Complete edition" in text
        assert "1 of 1 parts" in text
        assert b"Complete edition - 1 of 1 parts" in data
        assert b"nomorals-longdoc" in data
        assert manifest.revision_ref.encode() in data

    def test_pdf_carries_the_designed_watermark(self, tmp_path):
        router = _router(tmp_path)
        manifest = router.store_text(LONG)
        path, _name = router.file_for(manifest.document_id, "pdf")
        data = path.read_bytes()
        assert b"/GSW" in data and b"/ca 0.15" in data
        streams = _streams(data)
        assert streams
        assert any(b"Powered by peacethefirst" in s for s in streams)

    def test_pdf_is_the_export_generator_with_export_stamps(
            self, tmp_path, monkeypatch):
        # freeze the clock: render_pdf stamps a /CreationDate and a
        # footer date, the only volatile bytes in its output
        monkeypatch.setattr(
            time, "strftime", lambda fmt, t=None: "FROZEN")
        calls = []
        real_render = or_module.render_pdf

        def spy(text, **kwargs):
            calls.append((text, kwargs))
            return real_render(text, **kwargs)

        monkeypatch.setattr(or_module, "render_pdf", spy)
        router = _router(tmp_path)
        manifest = router.store_text(LONG)
        path, _name = router.file_for(manifest.document_id, "pdf")
        assert len(calls) == 1
        body, kwargs = calls[0]
        # the export seam's own arguments: book layout, the stamps,
        # the mark resolved from the manifest's brand policy
        assert kwargs["headings"] is True
        assert kwargs["chapter_break"] is True
        assert kwargs["toc"] is True
        assert kwargs["watermark"].text == "Powered by peacethefirst"
        assert kwargs["metadata"]["subject"] == \
            "Complete edition - 1 of 1 parts."
        assert kwargs["metadata"]["keywords"] == \
            "nomorals-longdoc " + manifest.revision_ref
        # the completeness line rides as front matter in the body
        # the generator was fed; the original follows whole
        assert "Complete edition — 1 of 1 parts." in body
        assert PARAS[0] in body and "RENDITION-TAIL-SENTINEL" in body
        # and the file is byte-for-byte that generator's output for
        # those inputs — no parallel PDF writer anywhere
        assert real_render(body, **kwargs) == path.read_bytes()

    def test_pdf_rendition_is_cached(self, tmp_path):
        router = _router(tmp_path)
        manifest = router.store_text(LONG)
        first = router.file_for(manifest.document_id, "pdf")
        mtime = first[0].stat().st_mtime_ns
        second = router.file_for(manifest.document_id, "pdf")
        assert second[0] == first[0]
        assert second[0].stat().st_mtime_ns == mtime


# ── the EPUB rendition ────────────────────────────────────────────


class TestEpubRendition:
    def test_epub_without_a_renderer_is_honestly_unavailable(
            self, tmp_path):
        router = _router(tmp_path)
        manifest = router.store_text(LONG)
        assert router.edition_epub(manifest.document_id) is None
        assert router.file_for(manifest.document_id, "epub") is None
        editions = tmp_path / "longdoc" / "editions"
        assert not editions.exists() or not list(
            editions.glob("*.epub"))

    def test_epub_file_is_a_real_branded_epub(self, tmp_path):
        router = _production_router(tmp_path)
        manifest = router.store_text(LONG)
        found = router.file_for(manifest.document_id, "epub")
        assert found is not None
        path, name = found
        assert name.endswith(".epub")
        z = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
        assert z.testzip() is None
        names = z.namelist()
        assert names[0] == "mimetype"
        # the title page carries the completeness truth + the
        # brand line, exactly as a story export's does
        title_page = z.read("OEBPS/title.xhtml").decode()
        assert "Complete edition" in title_page
        assert "1 of 1 parts" in title_page
        assert "Powered by peacethefirst" in title_page
        # the machine stamps ride the OPF metadata
        opf = z.read("OEBPS/content.opf").decode()
        assert 'property="nomorals:revision_ref"' in opf
        assert manifest.revision_ref in opf
        assert 'property="nomorals:completeness"' in opf
        assert "complete 1 of 1" in opf
        # the chapter holds the whole routed text
        chapter = z.read("OEBPS/ch001.xhtml").decode()
        assert PARAS[0] in chapter
        assert "RENDITION-TAIL-SENTINEL" in chapter

    def test_epub_is_the_export_generator(self, tmp_path):
        from nomorals.books.forge import (
            _epub_stamp_meta,
            epub_from_chapters,
        )
        from nomorals.core.branding import resolve_brand_mark
        from nomorals.core.edition import completeness_line

        router = _production_router(tmp_path)
        manifest = router.store_text(LONG)
        path, _name = router.file_for(manifest.document_id, "epub")
        line = completeness_line(manifest)
        mark = resolve_brand_mark(manifest.presentation.brand_policy)
        expected = epub_from_chapters(
            slug="read-" + manifest.document_id.split(":", 1)[-1],
            title=manifest.title,
            chapters=[(1, manifest.title, LONG.strip(), "")],
            front_note=f"{line} {mark.text}.",
            extra_meta=_epub_stamp_meta(manifest))
        mine = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
        theirs = zipfile.ZipFile(io.BytesIO(expected))
        assert mine.namelist() == theirs.namelist()
        # every member is byte-identical except the OPF and the
        # NCX, whose only volatile fields are the book's uuid and
        # its dates
        volatile = re.compile(
            r"urn:uuid:[0-9a-f-]+|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        for member in mine.namelist():
            a, b = mine.read(member), theirs.read(member)
            if member.endswith(("content.opf", "toc.ncx")):
                assert volatile.sub("X", a.decode()) == \
                    volatile.sub("X", b.decode())
            else:
                assert a == b, member

    def test_epub_rendition_is_cached(self, tmp_path):
        router = _production_router(tmp_path)
        manifest = router.store_text(LONG)
        first = router.file_for(manifest.document_id, "epub")
        mtime = first[0].stat().st_mtime_ns
        second = router.file_for(manifest.document_id, "epub")
        assert second[0] == first[0]
        assert second[0].stat().st_mtime_ns == mtime


# ── refusals at the router ────────────────────────────────────────


class TestRouterRefusals:
    def test_unknown_document_has_no_rendition(self, tmp_path):
        router = _production_router(tmp_path)
        assert router.file_for("text:0000000000000000", "pdf") is None
        assert router.file_for("text:0000000000000000", "epub") is None

    def test_rootless_router_serves_no_files(self, tmp_path):
        router = OutputRouter(None)
        manifest = router.store_text(LONG)
        assert router.file_for(manifest.document_id, "pdf") is None
        assert router.file_for(manifest.document_id, "epub") is None


# ── the route advertises the new actions ──────────────────────────


class TestAdvertising:
    def test_long_route_offers_all_four_file_actions(self, tmp_path):
        router = _router(tmp_path)
        plan = router.plan(LONG_ROUTE, platform="telegram",
                           reader_key=KEY)
        assert plan.decision.kind == "reader+edition"
        for fmt in ("html", "pdf", "epub", "txt"):
            assert f"/read get {plan.token} {fmt}" in plan.intro_text
        payloads = [p for row in (plan.buttons or []) for _, p in row]
        for fmt in ("html", "pdf", "epub", "txt"):
            assert f"read get {plan.token} {fmt}" in payloads
        assert "PDF" in plan.decision.reason
        assert "EPUB" in plan.decision.reason


# ── the /read get verb ────────────────────────────────────────────


def _harness(tmp_path):
    """The /read surface on a bare mixin, per-test tmp workspace —
    the BooksExportVerbTests shape."""
    from nomorals.agents.partner.runtime_search import RuntimeSearchMixin

    ctx = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        extras={})
    m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
    m.context = ctx
    m._ref_from_key = lambda key: SimpleNamespace(platform="telegram")
    sent: dict = {}

    def _send_file(platform, chat, path, caption="", name=""):
        sent.update(platform=platform, path=path, caption=caption,
                    name=name, data=Path(path).read_bytes())
        return SimpleNamespace(ok=True)

    m.gateway = SimpleNamespace(send_file=_send_file)
    return m, ctx, sent


def _open_routed(ctx, text=LONG) -> str:
    router = OutputRouter.for_context(ctx)
    plan = router.plan(text, platform="telegram", reader_key=KEY)
    return plan.token


class TestReadGetRenditions:
    def test_get_pdf_sends_the_branded_file(self, tmp_path):
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        reply = m._control_read(f"get {token} pdf", CHAT_KEY)
        assert reply.startswith("📄 sent above — ")
        assert reply.endswith(".pdf")
        assert "PDF edition" in sent["caption"]
        assert sent["data"].startswith(b"%PDF")
        text = _norm(read_pdf_text(sent["data"]))
        assert "Complete edition" in text
        assert "RENDITION-TAIL-SENTINEL" in text

    def test_get_epub_sends_the_branded_file(self, tmp_path):
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        reply = m._control_read(f"get {token} epub", CHAT_KEY)
        assert reply.startswith("📄 sent above — ")
        assert reply.endswith(".epub")
        assert "EPUB edition" in sent["caption"]
        z = zipfile.ZipFile(io.BytesIO(sent["data"]))
        assert z.namelist()[0] == "mimetype"
        title_page = z.read("OEBPS/title.xhtml").decode()
        assert "Powered by peacethefirst" in title_page

    def test_get_txt_and_html_behave_as_before(self, tmp_path):
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        reply = m._control_read(f"get {token} txt", CHAT_KEY)
        assert reply.startswith("📄 sent above")
        assert sent["data"] == LONG.encode("utf-8")
        reply = m._control_read(f"get {token} html", CHAT_KEY)
        assert reply.startswith("📄 sent above")
        assert "Powered by peacethefirst" in sent["data"].decode()

    def test_get_unknown_token_refuses_in_words(self, tmp_path):
        m, _ctx, sent = _harness(tmp_path)
        reply = m._control_read("get 0123456789ab pdf", CHAT_KEY)
        assert "that file control has expired" in reply
        assert sent == {}

    def test_get_bad_format_is_a_usage_line(self, tmp_path):
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        reply = m._control_read(f"get {token} docx", CHAT_KEY)
        assert reply.startswith("usage: /read get")
        assert "html|pdf|epub|txt" in reply
        assert sent == {}

    def test_get_document_gone_from_the_store_refuses(self, tmp_path):
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        content = tmp_path / "longdoc" / "content"
        for f in content.iterdir():
            f.unlink()
        reply = m._control_read(f"get {token} pdf", CHAT_KEY)
        assert "no longer in the store" in reply
        assert sent == {}

    def test_generation_failure_is_named_and_redacted(
            self, tmp_path, monkeypatch):
        import nomorals.agents.partner.runtime_search as rs

        def _boom(manifest, text):
            raise RuntimeError(
                "epub render blew up: sk-deadbeefcafe1234567890")

        monkeypatch.setattr(rs, "_read_epub_bytes", _boom)
        m, ctx, sent = _harness(tmp_path)
        token = _open_routed(ctx)
        reply = m._control_read(f"get {token} epub", CHAT_KEY)
        assert reply.startswith("the epub edition could not be built")
        assert "deadbeefcafe" not in reply
        assert "sk-***" in reply
        assert sent == {}


# ── gating + help parity ──────────────────────────────────────────


class TestGatingParity:
    def test_read_stays_owner_only(self):
        from nomorals.social.chat.control import (
            CONTROL_COMMANDS,
            PUBLIC_COMMANDS,
            caller_may_run,
        )

        assert "read" in CONTROL_COMMANDS
        assert "read" not in PUBLIC_COMMANDS
        assert caller_may_run("read", True)
        assert not caller_may_run("read", False)

    def test_help_names_the_new_formats(self):
        from nomorals.social.chat.control import COMMAND_DETAILS

        usage = COMMAND_DETAILS["read"]["usage"]
        assert "get <token> <html|pdf|epub|txt>" in usage
        assert "PDF and EPUB" in COMMAND_DETAILS["read"]["what"]
