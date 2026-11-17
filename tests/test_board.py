"""Board organ tests: the Excalidraw/Tldraw whiteboard (52-websites
#18 + #44) — typed shape records, op-based editing, deterministic
render, live editable page, brain tool, chat verb surface, CLI.

The board is the editable cousin of the scene organ's diagrams: the
brain and the owner manipulate it as data, and it renders to PNG or
to a live page they can keep drawing on.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from nomorals.media.edit_engine.board import (
    THEMES,
    BoardDocument,
    BoardError,
    BoardShape,
    BoardStore,
    board_from_diagram,
    board_live_html,
    render_board,
    write_board_page,
)


def _shape(shape_id: str, kind: str = "rect", x: float = 0.0,
           y: float = 0.0, w: float = 200.0, h: float = 120.0,
           **props: object) -> dict:
    shape: dict = {"id": shape_id, "type": kind, "x": x, "y": y,
                   "w": w, "h": h}
    if props:
        shape["props"] = dict(props)
    return shape


def _board_with_flow() -> BoardDocument:
    doc = BoardDocument.create("flow", theme="clean")
    doc.add_shape(_shape("a", "rect", 0, 0, text="Start"))
    doc.add_shape(_shape("b", "ellipse", 400, 0, text="End"))
    doc.bind("a", "b", label="then")
    return doc


class BoardModelTests(unittest.TestCase):
    def test_create_empty(self) -> None:
        doc = BoardDocument.create("untitled")
        self.assertEqual(doc.title, "untitled")
        self.assertEqual(doc.theme, "clean")
        self.assertEqual(doc.shapes, [])
        self.assertEqual(doc.bindings, [])
        self.assertEqual(doc.ops, [])
        self.assertEqual(doc.version, 1)

    def test_bad_theme_rejected(self) -> None:
        with self.assertRaises(BoardError) as ctx:
            BoardDocument.create("x", theme="neon")
        self.assertIn("theme", str(ctx.exception))

    def test_bad_shape_type_rejected(self) -> None:
        doc = BoardDocument.create("x")
        with self.assertRaises(BoardError) as ctx:
            doc.add_shape(_shape("s", "triangle", 0, 0))
        self.assertIn("type", str(ctx.exception))

    def test_duplicate_shape_id_rejected(self) -> None:
        doc = BoardDocument.create("x")
        doc.add_shape(_shape("s", "rect", 0, 0))
        with self.assertRaises(BoardError) as ctx:
            doc.add_shape(_shape("s", "rect", 10, 10))
        self.assertIn("duplicate", str(ctx.exception))

    def test_add_update_remove(self) -> None:
        doc = BoardDocument.create("x")
        added = doc.add_shape(_shape("s", "rect", 0, 0, text="hi"))
        self.assertEqual(added.props["text"], "hi")
        self.assertEqual(len(doc.ops), 1)
        moved = doc.update_shape("s", x=50.0, y=60.0)
        self.assertEqual((moved.x, moved.y), (50.0, 60.0))
        self.assertEqual(len(doc.ops), 2)
        doc.remove_shape("s")
        self.assertEqual(doc.shapes, [])
        self.assertEqual(len(doc.ops), 3)

    def test_update_unknown_shape_rejected(self) -> None:
        doc = BoardDocument.create("x")
        with self.assertRaises(BoardError) as ctx:
            doc.update_shape("ghost", x=1.0)
        self.assertIn("ghost", str(ctx.exception))

    def test_update_rejects_bad_field(self) -> None:
        doc = BoardDocument.create("x")
        doc.add_shape(_shape("s", "rect", 0, 0))
        with self.assertRaises(BoardError) as ctx:
            doc.update_shape("s", nope=1)
        self.assertIn("nope", str(ctx.exception))

    def test_round_trip_exact(self) -> None:
        doc = _board_with_flow()
        doc.add_shape(_shape("d", "draw", 0, 300,
                             points=[[0, 300], [50, 320], [100, 300]]))
        restored = BoardDocument.from_dict(doc.to_dict())
        self.assertEqual(restored.to_dict(), doc.to_dict())
        self.assertEqual(restored.compute_hash(), doc.compute_hash())

    def test_binding_integrity_on_delete(self) -> None:
        doc = _board_with_flow()
        binding_id = doc.bindings[0].id
        self.assertEqual(len(doc.bindings), 1)
        result = doc.remove_shape("a")
        self.assertEqual(doc.bindings, [])
        # The report names the *binding* that dropped, not the shape.
        self.assertEqual(result.get("removed_bindings"), [binding_id])
        # The doc still replays: the op list is the truth.
        replayed = BoardDocument.replay(doc.ops, title="flow")
        self.assertEqual(replayed.to_dict()["shapes"],
                         doc.to_dict()["shapes"])

    def test_bind_unknown_shape_rejected(self) -> None:
        doc = BoardDocument.create("x")
        doc.add_shape(_shape("a", "rect", 0, 0))
        with self.assertRaises(BoardError) as ctx:
            doc.bind("a", "ghost")
        self.assertIn("ghost", str(ctx.exception))

    def test_unbind(self) -> None:
        doc = _board_with_flow()
        binding = doc.bindings[0]
        doc.unbind(binding.id)
        self.assertEqual(doc.bindings, [])

    def test_op_replay_determinism(self) -> None:
        doc = _board_with_flow()
        ops = [dict(op) for op in doc.ops]
        first = BoardDocument.replay(ops, title="flow",
                                     board_id=doc.id)
        second = BoardDocument.replay(ops, title="flow",
                                      board_id=doc.id)
        self.assertEqual(first.compute_hash(), second.compute_hash())
        self.assertEqual(first.compute_hash(), doc.compute_hash())

    def test_unknown_op_rejected(self) -> None:
        doc = BoardDocument.create("x")
        with self.assertRaises(BoardError) as ctx:
            doc.apply({"op": "teleport", "id": "a"})
        self.assertIn("teleport", str(ctx.exception))

    def test_undo_redo(self) -> None:
        doc = BoardDocument.create("x")
        doc.add_shape(_shape("s", "rect", 0, 0, text="one"))
        doc.add_shape(_shape("t", "rect", 300, 0, text="two"))
        self.assertTrue(doc.can_undo)
        doc.undo()
        self.assertEqual([s.id for s in doc.shapes], ["s"])
        self.assertTrue(doc.can_redo)
        doc.redo()
        self.assertEqual([s.id for s in doc.shapes], ["s", "t"])
        doc.undo()
        doc.undo()
        self.assertEqual(doc.shapes, [])
        self.assertFalse(doc.can_undo)
        with self.assertRaises(BoardError):
            doc.undo()

    def test_undo_restores_binding(self) -> None:
        doc = _board_with_flow()
        doc.remove_shape("a")
        self.assertEqual(doc.bindings, [])
        doc.undo()
        self.assertEqual(len(doc.bindings), 1)
        # The undo brings back the node AND its arrow shape.
        self.assertEqual(len(doc.shapes), 3)
        self.assertIn("a", [s.id for s in doc.shapes])

    def test_seal_bumps_version_and_hash(self) -> None:
        doc = BoardDocument.create("x")
        before = doc.compute_hash()
        doc.add_shape(_shape("s", "rect", 0, 0))
        self.assertGreater(doc.version, 1)
        self.assertNotEqual(doc.compute_hash(), before)
        self.assertEqual(doc.doc_hash, doc.compute_hash())

    def test_shape_validation_names_field(self) -> None:
        with self.assertRaises(BoardError) as ctx:
            BoardShape(id="s", type="rect", x=0, y=0, w=-5, h=10)
        self.assertIn("w", str(ctx.exception))
        with self.assertRaises(BoardError) as ctx:
            BoardShape(id="s", type="draw", x=0, y=0, w=0, h=0,
                       props={"points": [[0]]})
        self.assertIn("points", str(ctx.exception))

    def test_board_from_diagram(self) -> None:
        spec = {"kind": "diagram", "title": "map",
                "nodes": [{"id": "a", "label": "A"},
                          {"id": "b", "label": "B", "shape": "circle"}],
                "edges": [{"from": "a", "to": "b", "label": "goes"}]}
        doc = board_from_diagram(spec, title="map")
        # Two nodes + the arrow shape the edge becomes.
        self.assertEqual([s.id for s in doc.shapes][:2], ["a", "b"])
        self.assertEqual(len(doc.shapes), 3)
        self.assertEqual(doc.shapes[2].type, "arrow")
        self.assertEqual(doc.shapes[1].type, "ellipse")
        self.assertEqual(len(doc.bindings), 1)
        self.assertEqual(doc.bindings[0].label, "goes")
        # Nodes without coordinates get the deterministic layout.
        self.assertNotEqual((doc.shapes[0].x, doc.shapes[0].y),
                            (doc.shapes[1].x, doc.shapes[1].y))

    def test_board_from_diagram_rejects_non_diagram(self) -> None:
        with self.assertRaises(BoardError):
            board_from_diagram({"kind": "overlay", "annotations": []})

    def test_bounds_empty(self) -> None:
        doc = BoardDocument.create("x")
        self.assertEqual(doc.bounds(), (0.0, 0.0, 0.0, 0.0))


class BoardStoreTests(unittest.TestCase):
    def test_save_load_list_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BoardStore(home=tmp)
            doc = _board_with_flow()
            path = store.save(doc)
            self.assertTrue(Path(path).is_file())
            loaded = store.load(doc.id)
            assert loaded is not None
            self.assertEqual(loaded.to_dict(), doc.to_dict())
            rows = store.list_boards()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], doc.id)
            self.assertEqual(rows[0]["shapes"], 3)
            self.assertTrue(store.delete(doc.id))
            self.assertIsNone(store.load(doc.id))
            self.assertEqual(store.list_boards(), [])
            self.assertFalse(store.delete(doc.id))

    def test_find_by_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BoardStore(home=tmp)
            doc = _board_with_flow()
            store.save(doc)
            found = store.find(doc.id[:8])
            assert found is not None
            self.assertEqual(found.id, doc.id)
            self.assertIsNone(store.find("zzzz-no-such-board"))

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BoardStore(home=tmp)
            self.assertIsNone(store.load("nope"))

    def test_save_after_edit_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = BoardStore(home=tmp)
            doc = _board_with_flow()
            store.save(doc)
            doc.update_shape("a", x=123.0)
            store.save(doc)
            loaded = store.load(doc.id)
            assert loaded is not None
            shape = loaded.shape("a")
            assert shape is not None
            self.assertEqual(shape.x, 123.0)

    def test_undo_survives_save_load(self) -> None:
        # The real catch: every tool/CLI/chat call loads fresh, so
        # undo history must persist beside the board, not just in
        # memory — or undo is dead in practice.
        with tempfile.TemporaryDirectory() as tmp:
            store = BoardStore(home=tmp)
            doc = _board_with_flow()
            store.save(doc)
            doc.update_shape("a", x=500.0)
            store.save(doc)
            loaded = store.load(doc.id)
            assert loaded is not None
            self.assertTrue(loaded.can_undo)
            loaded.undo()
            shape = loaded.shape("a")
            assert shape is not None
            self.assertEqual(shape.x, 0.0)
            self.assertTrue(loaded.can_redo)


class BoardRenderTests(unittest.TestCase):
    def test_render_png_real_bytes(self) -> None:
        doc = _board_with_flow()
        with tempfile.TemporaryDirectory() as tmp:
            out = render_board(doc, out_dir=tmp)
            path = Path(out["path"])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(out["shapes"], 3)
            self.assertEqual(out["missing_images"], [])
            self.assertEqual(out["size"], [1280, 720])
            with path.open("rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")

    def test_render_deterministic(self) -> None:
        doc = _board_with_flow()
        with tempfile.TemporaryDirectory() as tmp:
            first = render_board(doc, out_dir=tmp)
            second = render_board(doc, out_dir=tmp)
            self.assertEqual(first["render_hash"], second["render_hash"])
            self.assertEqual(
                hashlib.sha256(
                    Path(first["path"]).read_bytes()).hexdigest(),
                first["render_hash"])

    def test_render_themes_differ(self) -> None:
        doc = _board_with_flow()
        with tempfile.TemporaryDirectory() as tmp:
            clean = render_board(doc, theme="clean", out_dir=tmp)
            sketch = render_board(doc, theme="sketch", out_dir=tmp)
            self.assertNotEqual(clean["render_hash"],
                                sketch["render_hash"])

    def test_render_sketch_theme(self) -> None:
        doc = _board_with_flow()
        with tempfile.TemporaryDirectory() as tmp:
            out = render_board(doc, theme="sketch", out_dir=tmp)
            self.assertTrue(Path(out["path"]).is_file())
            self.assertEqual(out["theme"], "sketch")

    def test_render_empty_board(self) -> None:
        doc = BoardDocument.create("blank")
        with tempfile.TemporaryDirectory() as tmp:
            out = render_board(doc, out_dir=tmp)
            self.assertTrue(Path(out["path"]).is_file())

    def test_render_missing_image_is_honest(self) -> None:
        doc = BoardDocument.create("x")
        doc.add_shape(_shape("img", "image", 0, 0, 200, 150,
                             path="/no/such/picture.png"))
        with tempfile.TemporaryDirectory() as tmp:
            out = render_board(doc, out_dir=tmp)
            self.assertIn("/no/such/picture.png", out["missing_images"])
            self.assertTrue(Path(out["path"]).is_file())

    def test_render_real_image_shape(self) -> None:
        from PIL import Image

        doc = BoardDocument.create("x")
        with tempfile.TemporaryDirectory() as tmp:
            pic = Path(tmp) / "pic.png"
            Image.new("RGB", (60, 40), "#22C55E").save(pic)
            doc.add_shape(_shape("img", "image", 0, 0, 200, 150,
                                 path=str(pic)))
            out = render_board(doc, out_dir=tmp)
            self.assertEqual(out["missing_images"], [])
            self.assertTrue(Path(out["path"]).is_file())

    def test_render_bad_theme_rejected(self) -> None:
        doc = BoardDocument.create("x")
        with self.assertRaises(BoardError):
            render_board(doc, theme="neon")

    def test_live_page_is_editable(self) -> None:
        doc = _board_with_flow()
        html = board_live_html(doc)
        self.assertIn(doc.id, html)
        self.assertIn("Start", html)
        self.assertIn("canvas", html)
        # The editor surface: drag, add, delete, download.
        for marker in ("data-add", "Download board", "mousedown",
                       "dblclick", "Delete"):
            self.assertIn(marker, html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_write_board_page(self) -> None:
        doc = _board_with_flow()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_board_page(doc, Path(tmp) / "board.html")
            self.assertTrue(Path(path).is_file())
            self.assertIn(doc.id, Path(path).read_text(encoding="utf-8"))

    def test_themes_listed(self) -> None:
        self.assertEqual(set(THEMES), {"clean", "sketch"})


class BoardToolTests(unittest.TestCase):
    def _registry(self, home: str):
        from types import SimpleNamespace

        from nomorals.tools.board import register as register_board
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(
            settings=SimpleNamespace(home=home))
        reg = ToolRegistry(context=context, enforce=False)
        register_board(reg)
        return reg

    def test_tool_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            self.assertIn("board", reg._tools)

    def test_tool_create_add_render_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            created = reg.call("board", action="create",
                               title="plan", theme="sketch")
            self.assertTrue(created.ok, created.error)
            board_id = created.value["id"]
            added = reg.call(
                "board", action="add", board_id=board_id,
                shape={"id": "n1", "type": "rect", "x": 0, "y": 0,
                       "w": 200, "h": 120, "props": {"text": "Idea"}})
            self.assertTrue(added.ok, added.error)
            rendered = reg.call("board", action="render",
                                board_id=board_id)
            self.assertTrue(rendered.ok, rendered.error)
            self.assertTrue(Path(rendered.value["path"]).is_file())
            shown = reg.call("board", action="show", board_id=board_id)
            self.assertTrue(shown.ok, shown.error)
            self.assertEqual(shown.value["shapes"], 1)

    def test_tool_create_from_diagram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            created = reg.call(
                "board", action="create", title="map",
                diagram={"kind": "diagram", "title": "map",
                         "nodes": [{"id": "a"}, {"id": "b"}],
                         "edges": [{"from": "a", "to": "b"}]})
            self.assertTrue(created.ok, created.error)
            # Two nodes + the arrow shape the edge becomes.
            self.assertEqual(created.value["shapes"], 3)
            self.assertEqual(created.value["bindings"], 1)

    def test_tool_move_and_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            created = reg.call("board", action="create", title="x")
            board_id = created.value["id"]
            reg.call("board", action="add", board_id=board_id,
                     shape={"id": "s", "type": "rect", "x": 0, "y": 0,
                            "w": 100, "h": 80})
            moved = reg.call("board", action="move", board_id=board_id,
                             shape_id="s", x=55.0, y=66.0)
            self.assertTrue(moved.ok, moved.error)
            self.assertEqual((moved.value["x"], moved.value["y"]),
                             (55.0, 66.0))
            undone = reg.call("board", action="undo", board_id=board_id)
            self.assertTrue(undone.ok, undone.error)
            shown = reg.call("board", action="show", board_id=board_id)
            shape = shown.value["shape_list"][0]
            self.assertEqual((shape["x"], shape["y"]), (0.0, 0.0))

    def test_tool_unknown_board_errors_in_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            outcome = reg.call("board", action="show",
                               board_id="no-such-board")
            self.assertFalse(outcome.ok)
            self.assertIn("no-such-board", str(outcome.error))

    def test_tool_list_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            first = reg.call("board", action="create", title="one")
            reg.call("board", action="create", title="two")
            listed = reg.call("board", action="list")
            self.assertTrue(listed.ok, listed.error)
            self.assertEqual(len(listed.value["boards"]), 2)
            deleted = reg.call("board", action="delete",
                               board_id=first.value["id"])
            self.assertTrue(deleted.ok, deleted.error)
            listed = reg.call("board", action="list")
            self.assertEqual(len(listed.value["boards"]), 1)

    def test_tool_unknown_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            outcome = reg.call("board", action="teleport")
            self.assertFalse(outcome.ok)


class BoardCliTests(unittest.TestCase):
    def _context(self, home: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home=home))

    def test_cli_new_render_list(self) -> None:
        import argparse

        from nomorals.cmdline.commands.board import cmd_board

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            args = argparse.Namespace(board_action="new", title="cli",
                                      theme="clean", diagram_file="")
            self.assertEqual(cmd_board(args, ctx), 0)
            store = BoardStore(home=tmp)
            boards = store.list_boards()
            self.assertEqual(len(boards), 1)
            board_id = boards[0]["id"]
            args = argparse.Namespace(
                board_action="add", board_ref=board_id, shape_type="rect",
                label="Box", x=None, y=None, w=None, h=None,
                color="", points="")
            self.assertEqual(cmd_board(args, ctx), 0)
            args = argparse.Namespace(board_action="render",
                                      board_ref=board_id, theme="",
                                      out="", size="")
            self.assertEqual(cmd_board(args, ctx), 0)
            args = argparse.Namespace(board_action="list")
            self.assertEqual(cmd_board(args, ctx), 0)

    def test_cli_unknown_board_fails_cleanly(self) -> None:
        import argparse

        from nomorals.cmdline.commands.board import cmd_board

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            args = argparse.Namespace(board_action="show",
                                      board_ref="no-such-board")
            self.assertEqual(cmd_board(args, ctx), 1)


class BoardLayeringTests(unittest.TestCase):
    def test_board_exports(self) -> None:
        """The board organ is exported from the edit-engine package —
        the same objects, not a parallel copy."""
        import nomorals.media.edit_engine as edit_engine
        from nomorals.media.edit_engine import board as board_mod

        self.assertIs(edit_engine.BoardDocument,
                      board_mod.BoardDocument)
        self.assertIs(edit_engine.BoardShape, board_mod.BoardShape)
        self.assertIs(edit_engine.BoardBinding,
                      board_mod.BoardBinding)
        self.assertIs(edit_engine.BoardStore, board_mod.BoardStore)
        self.assertIs(edit_engine.render_board, board_mod.render_board)


if __name__ == "__main__":
    unittest.main()
