"""Devon Studio native photo editor — the PhotoCraft port (lane 76).

The gap, failing-first on the pristine base: there is no photo-editor
organ in the repo. ``nomorals/media/imggen/document.py`` is an
op-list image document (no layers, no tiles, no PSD), ``git grep psd``
finds no decoder, and there is no /photo verb, no photo_editor tool
and no Studio photo route. This lane builds the foundation:

* ``nomorals/media/photo_editor/psd.py`` — the PSD format crate
  (PhotoCraft ``crates/psd`` decomposition, reimplemented in Python
  from Adobe's public Photoshop File Formats spec): header, image
  resources, layer records, channel data (raw / PackBits RLE / ZIP /
  ZIP-predicted), composite image data, and an encoder that writes
  PSD v1 back from the document model.
* ``nomorals/media/photo_editor/document.py`` — the COW-tile
  document model (PhotoCraft ``crates/doc`` + ``crates/raster``):
  layers over copy-on-write 256px tiles, history with undo/redo,
  edits through the EXISTING ladder (media_edit.images /
  imggen.develop), and a bridge to ImageDocument.
* ``nomorals/media/photo_editor/oracle.py`` — the corpus-oracle
  harness: every real PSD in the corpus is decoded, flattened, and
  compared against Photoshop's own rendering of the same file (the
  merged composite the file itself carries), plus an encode→decode
  round-trip check.
* ``nomorals/media/photo_editor/store.py`` + ``page.py`` — the
  durable session store and the Studio photo page (plain HTML, no
  scripts — the API's CSP allows nothing).
* ``nomorals/tools/photo_editor.py`` — the ``photo_editor`` brain
  tool; ``/photo`` chat verb; ``GET /studio/photo/<id>`` on the API.

Every test here drives the PRODUCTION path against REAL PSD files
— the vendored fixtures in tests/fixtures/photocraft/ are genuine
Adobe Photoshop files from the dual MIT/Apache PhotoCraft corpus
(see the fixtures README). A decoder proven only against fixtures
written by the same hand as the decoder proves nothing; the oracle
is the check.
"""

from __future__ import annotations

import http.client
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "photocraft"
RGB8 = FIXTURES / "rgb8-baseline-layer-copy.psd"
RGB8_ADJ = FIXTURES / "rgb8-brightness-contrast.psd"
CMYK8 = FIXTURES / "cmyk8-baseline-layer-copy.psd"
GRAY8 = FIXTURES / "gray8-baseline-layer-copy.psd"

# Full 258-file corpus: the mining checkout on this box, or the env
# override. The vendored fixtures above always run; the full-corpus
# test skips honestly when the checkout is absent.
CORPUS = Path(os.environ.get(
    "NM_PHOTOCRAFT_CORPUS",
    str(Path.home() / "workspace" / "echelon-mining" / "repos"
        / "storytold" / "photocraft-corpus" / "photoshop")))


def _read(path: Path) -> bytes:
    return path.read_bytes()


class PsdDecodeTests(unittest.TestCase):
    def test_real_rgb8_psd_opens_with_layers(self) -> None:
        from nomorals.media.photo_editor.psd import PsdFile

        psd = PsdFile.from_bytes(_read(RGB8))
        self.assertEqual(psd.mode, "RGB")
        self.assertEqual(psd.depth, 8)
        self.assertEqual((psd.width, psd.height), (96, 96))
        self.assertEqual(len(psd.layers), 2)
        names = [layer.name for layer in psd.layers]
        self.assertIn("Background", names)
        self.assertTrue(all(layer.rect for layer in psd.layers))

    def test_flatten_matches_photoshop_composite_exactly(self) -> None:
        """The oracle in miniature: this fixture has no adjustment
        layers, so Devon's normal-blend flatten must match Photoshop's
        own rendering of the same stack — the merged composite the
        file itself carries — pixel for pixel."""
        from nomorals.media.photo_editor.psd import PsdFile

        psd = PsdFile.from_bytes(_read(RGB8))
        flat = psd.flatten_layers()
        comp = psd.composite_rgba()
        self.assertEqual(flat.size, comp.size)
        self.assertEqual(flat.mode, "RGBA")
        diff = _max_diff(flat, comp)
        self.assertEqual(diff, 0, f"flatten differs from Photoshop's "
                         f"composite by up to {diff} per channel")

    def test_cmyk_and_gray_decode_to_rgb(self) -> None:
        from nomorals.media.photo_editor.psd import PsdFile

        for path, mode in ((CMYK8, "CMYK"), (GRAY8, "L")):
            psd = PsdFile.from_bytes(_read(path))
            self.assertEqual(psd.mode, mode)
            flat = psd.flatten_layers()
            self.assertEqual(flat.size, (96, 96))
            self.assertEqual(flat.mode, "RGBA")
            comp = psd.composite_rgba()
            diff = _max_diff(flat, comp)
            # CMYK conversion is the standard naive formula and Lab
            # is exact math; both must land within 2 levels of the
            # vendor rendering, never wildly off.
            self.assertLessEqual(diff, 2, f"{path.name}: flatten "
                                 f"differs from composite by {diff}")

    def test_adjustment_layer_is_parsed_and_rendered(self) -> None:
        """An adjustment layer is data, not a silent drop: it is
        parsed, its kind recorded, and the flatten RENDERS it —
        lane 97 closed the lane-76 gap, so the brightness-contrast
        adjustment is applied and the flatten lands within one
        level of Photoshop's own composite."""
        from nomorals.media.photo_editor.psd import PsdFile

        psd = PsdFile.from_bytes(_read(RGB8_ADJ))
        adj = [layer for layer in psd.layers
               if layer.adjustment_keys]
        self.assertTrue(adj, "expected an adjustment layer")
        flat = psd.flatten_layers()
        self.assertEqual(flat.size, (96, 96))
        self.assertEqual(psd.unrendered_adjustments(), [],
                         "the adjustment is rendered, not reported")
        comp = psd.composite_rgba()
        self.assertLessEqual(_max_diff(flat, comp), 1)

    def test_roundtrip_preserves_pixels_and_layers(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.psd import PsdFile, encode_psd

        doc = PhotoDocument.from_psd_bytes(_read(RGB8))
        blob = encode_psd(doc)
        again = PsdFile.from_bytes(blob)
        self.assertEqual(again.mode, "RGB")
        self.assertEqual((again.width, again.height), (96, 96))
        self.assertEqual([layer.name for layer in again.layers],
                         [layer.name for layer in
                          PsdFile.from_bytes(_read(RGB8)).layers])
        diff = _max_diff(again.flatten_layers(),
                         PsdFile.from_bytes(_read(RGB8)).flatten_layers())
        self.assertEqual(diff, 0)

    def test_rejects_non_psd_and_truncated(self) -> None:
        from nomorals.media.photo_editor.psd import PsdError, PsdFile

        with self.assertRaises(PsdError):
            PsdFile.from_bytes(b"not a photoshop file at all" * 4)
        with self.assertRaises(PsdError):
            PsdFile.from_bytes(_read(RGB8)[:40])
        with self.assertRaises(PsdError):
            PsdFile.from_bytes(b"")

    def test_unknown_mode_refuses_honestly(self) -> None:
        """A multichannel file is refused with the mode named —
        never decoded as if it were RGB."""
        from nomorals.media.photo_editor.psd import PsdError, PsdFile

        raw = bytearray(_read(RGB8))
        raw[24:26] = (7).to_bytes(2, "big")  # color mode: multichannel
        with self.assertRaises(PsdError) as caught:
            PsdFile.from_bytes(bytes(raw))
        self.assertIn("ultichannel", str(caught.exception))


def _max_diff(a: Any, b: Any) -> int:
    """Max per-channel absolute difference between two RGBA images."""
    import numpy as np

    x = np.asarray(a.convert("RGBA")).astype(int)
    y = np.asarray(b.convert("RGBA")).astype(int)
    if x.shape != y.shape:
        raise AssertionError(f"shape mismatch {x.shape} vs {y.shape}")
    return int(abs(x - y).max()) if x.size else 0


class DocumentTests(unittest.TestCase):
    def _doc(self) -> Any:
        from nomorals.media.photo_editor.document import PhotoDocument

        return PhotoDocument.from_psd_bytes(_read(RGB8))

    def test_document_opens_with_layers_and_history(self) -> None:
        doc = self._doc()
        self.assertEqual((doc.width, doc.height), (96, 96))
        self.assertEqual(len(doc.layers), 2)
        self.assertEqual(doc.history, [])
        flat = doc.flatten()
        self.assertEqual(flat.size, (96, 96))
        self.assertEqual(flat.mode, "RGBA")

    def test_cow_tiles_share_until_written(self) -> None:
        """Copy-on-write: an edit clones only the tiles it touches.
        Untouched layers keep the very same tile objects; undo
        restores the original tile objects."""
        doc = self._doc()
        bottom_before = doc.layers[0].surface.tiles
        top_before = doc.layers[1].surface.tiles
        doc.apply_edit("enhance", layer=1, brightness=1.5)
        # the bottom layer was not touched: same tile objects
        self.assertIs(doc.layers[0].surface.tiles[(0, 0)],
                      bottom_before[(0, 0)])
        # the edited layer got new tile objects
        self.assertIsNot(doc.layers[1].surface.tiles[(0, 0)],
                         top_before[(0, 0)])
        doc.undo()
        self.assertIs(doc.layers[1].surface.tiles[(0, 0)],
                      top_before[(0, 0)])

    def test_crop_and_brightness_through_the_ladder(self) -> None:
        """One end-to-end edit: crop the canvas (every layer) then
        brighten the active layer — both through the EXISTING ladder
        (media_edit.images), never a parallel implementation. The
        ops are journaled as data; undo replays the journal back."""
        doc = self._doc()
        original = doc.flatten()
        doc.apply_edit("crop", box=(0, 0, 64, 64))
        self.assertEqual((doc.width, doc.height), (64, 64))
        self.assertEqual(doc.flatten().size, (64, 64))
        self.assertEqual(len(doc.layers), 2)
        before = doc.flatten()
        doc.apply_edit("enhance", brightness=1.5)
        after = doc.flatten()
        self.assertEqual(after.size, (64, 64))
        import numpy as np

        mean_before = float(np.asarray(before).mean())
        mean_after = float(np.asarray(after).mean())
        self.assertGreater(mean_after, mean_before,
                           "brightness 1.5 must brighten the flatten")
        self.assertEqual([op["op"] for op in doc.history],
                         ["crop", "enhance"])
        doc.undo()
        self.assertEqual(doc.flatten().size, (64, 64))
        doc.undo()
        self.assertEqual((doc.width, doc.height), (96, 96))
        self.assertEqual(_max_diff(doc.flatten(), original), 0)
        doc.redo()
        self.assertEqual((doc.width, doc.height), (64, 64))

    def test_develop_edit_runs_through_imggen(self) -> None:
        doc = self._doc()
        doc.apply_edit("develop", brightness=0.5)
        self.assertEqual(doc.history[-1]["op"], "develop")
        self.assertEqual(doc.flatten().size, (96, 96))

    def test_save_and_reopen_psd(self) -> None:
        doc = self._doc()
        doc.apply_edit("crop", box=(16, 16, 80, 80))
        doc.apply_edit("enhance", brightness=1.2)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "edited.psd")
            doc.save_psd(out)
            reopened = self._open(out)
            self.assertEqual((reopened.width, reopened.height),
                             (64, 64))
            self.assertEqual(
                _max_diff(reopened.flatten(), doc.flatten()), 0)

    def _open(self, path: str) -> Any:
        from nomorals.media.photo_editor.document import PhotoDocument

        return PhotoDocument.open_psd(path)

    def test_bridge_to_image_document(self) -> None:
        """The two document families bridge instead of duplicating:
        a photo document compiles to an ImageDocument (same flatten)
        and rebuilds from one as a single layer."""
        from nomorals.media.photo_editor.document import PhotoDocument

        doc = self._doc()
        doc.apply_edit("enhance", brightness=1.1)
        with tempfile.TemporaryDirectory() as tmp:
            imgdoc = doc.to_image_document(
                Path(tmp) / "flat.png")
            rendered = imgdoc.render()
            self.assertEqual(
                _max_diff(rendered.image, doc.flatten()), 0)
            back = PhotoDocument.from_image_document(imgdoc)
            self.assertEqual(len(back.layers), 1)
            self.assertEqual(
                _max_diff(back.flatten(), doc.flatten()), 0)


class OracleTests(unittest.TestCase):
    def test_oracle_over_vendored_fixtures(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        report = run_oracle(str(FIXTURES))
        summary = report["summary"]
        self.assertEqual(summary["files"], 4)
        self.assertEqual(summary["failed"], 0, report["files"])
        self.assertEqual(summary["exact"], 3, report["files"])
        # The adjustment-layer fixture is rendered now (lane 97):
        # parsed, applied, and within one level of Photoshop's
        # composite — a render gap of a single level, not an
        # unrendered adjustment.
        adj = [f for f in report["files"]
               if f["adjustments"]]
        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0]["status"], "differs")
        self.assertLessEqual(adj[0]["max_diff"], 1)
        self.assertTrue(adj[0]["adjustments_rendered"])

    def test_oracle_full_corpus_when_present(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        if not CORPUS.is_dir():
            self.skipTest(f"no corpus checkout at {CORPUS}")
        report = run_oracle(str(CORPUS))
        summary = report["summary"]
        self.assertEqual(summary["files"], 258)
        # No file may crash the decoder: every file parses or fails
        # with a named reason.
        self.assertEqual(summary["failed"], 0, [
            f for f in report["files"] if f["status"] == "failed"])
        # Every file without an adjustment layer, a non-normal
        # blend, a layer effect, live text or a smart object
        # flattens exactly onto Photoshop's own composite.
        exact_expected = [f for f in report["files"]
                          if not f["adjustments"] and not f["blends"]
                          and not f["effects"] and not f["text"]
                          and not f["smartobjects"]
                          and f["mode"] == "RGB"
                          and f["depth"] == 8]
        self.assertTrue(exact_expected)
        bad = [f for f in exact_expected if not f["exact"]]
        self.assertEqual(bad, [], f"{len(bad)} RGB8 files without "
                         "adjustments differ from the composite")

    def test_oracle_roundtrip_all_fixtures(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        report = run_oracle(str(FIXTURES))
        for entry in report["files"]:
            self.assertTrue(entry["roundtrip_ok"], entry)


class WiringTests(unittest.TestCase):
    def test_tool_registered(self) -> None:
        from types import SimpleNamespace

        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            context = SimpleNamespace(
                settings=SimpleNamespace(home=tmp))
            registry = ToolRegistry(context)
            registry.register_builtins()
            names = {schema["name"] for schema in registry.schemas()}
            self.assertIn("photo_editor", names)
            outcome = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="open", path=str(RGB8))
            self.assertTrue(outcome.ok, outcome)
            result = outcome.unwrap()
            self.assertTrue(result["ok"])
            self.assertEqual(result["layers"], 2)
            self.assertEqual(result["size"], [96, 96])

    def test_photo_verb_registered(self) -> None:
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
        )

        self.assertIn("photo", CONTROL_COMMANDS)
        details = COMMAND_DETAILS["photo"]
        self.assertTrue(details["what"])
        self.assertTrue(details["usage"])
        self.assertTrue(details["example"])
        self.assertIn("photo", LIST_ONELINERS)
        slots = [group for group, kinds in LIST_GROUPS
                 if "photo" in kinds]
        self.assertEqual(len(slots), 1,
                         f"photo needs exactly one LIST_GROUPS slot, "
                         f"found {slots}")
        # owner-only like /board: absent from PUBLIC_COMMANDS.
        self.assertNotIn("photo", PUBLIC_COMMANDS)

    def test_runtime_dispatches_photo(self) -> None:
        from pathlib import Path as _Path

        src = (_Path(__file__).resolve().parent.parent
               / "nomorals" / "agents" / "partner" / "runtime.py"
               ).read_text()
        self.assertIn('kind == "photo"', src)

    def test_studio_page_renders(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.page import photo_page_html

        doc = PhotoDocument.from_psd_bytes(_read(RGB8))
        html = photo_page_html(doc)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("Background", html)
        self.assertIn("/photo", html)
        self.assertNotIn("<script", html)

    def test_store_roundtrip(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.store import PhotoStore

        with tempfile.TemporaryDirectory() as tmp:
            store = PhotoStore(home=tmp)
            doc = PhotoDocument.from_psd_bytes(_read(RGB8))
            doc_id = store.save(doc, title="baseline")
            self.assertTrue(doc_id)
            found = store.matches(doc_id[:8])
            self.assertEqual(found, [doc_id])
            back = store.load(doc_id)
            self.assertIsNotNone(back)
            assert back is not None
            self.assertEqual((back.width, back.height), (96, 96))
            self.assertEqual(len(store.list_docs()), 1)


class _LiveServer:
    """Real HTTP server on loopback; http.client for full control."""

    def __init__(self, api: Any) -> None:
        from nomorals.api.server import _make_handler

        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str, token: str | None = None
            ) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=10)
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn.request("GET", path, headers=headers)
        return conn.getresponse()


class HttpRouteTests(unittest.TestCase):
    def test_studio_photo_route(self) -> None:
        from nomorals.api.server import APIServer
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.store import PhotoStore
        from nomorals.storage.db import Database
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(
                home=tmp,
                api=SimpleNamespace(token="", max_body_mb=0),
                backup_dir=tmp,
                workspace_dir=tmp,
            )
            context = SimpleNamespace(
                db=Database(":memory:"), settings=settings,
                tools=ToolRegistry(), bus=None, memory=None,
            )
            store = PhotoStore(home=tmp)
            doc = PhotoDocument.from_psd_bytes(_read(RGB8))
            doc_id = store.save(doc, title="baseline")
            api = APIServer(context, token="test-token",
                            max_body_bytes=64 * 1024)
            server = _LiveServer(api)
            try:
                no_auth = server.get(f"/studio/photo/{doc_id}")
                self.assertEqual(no_auth.status, 401)
                no_auth.read()
                resp = server.get(f"/studio/photo/{doc_id}",
                                  token="test-token")
                payload = resp.read()
                self.assertEqual(resp.status, 200, payload)
                body = payload.decode("utf-8")
                self.assertIn("data:image/png;base64,", body)
                self.assertIn("Background", body)
                missing = server.get("/studio/photo/no-such-doc",
                                     token="test-token")
                self.assertEqual(missing.status, 404)
                missing.read()
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
