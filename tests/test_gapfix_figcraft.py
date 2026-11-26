"""FigCraft port — failing-first tests (lane 105, census §3A.12).

A real interface-design + multiplayer organ: .figcraft documents
(ZIP with document.json, upstream file shape), auto layout,
components/variants, prototyping, a state-based collab core with a
server-ordered merge, presence, history, and a PIL frame renderer.
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "figcraft"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class ModelTests(unittest.TestCase):
    def test_open_simple_fixture(self) -> None:
        from nomorals.media.design import DesignDocument
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        self.assertEqual(doc.name, "Simple Fixture")
        self.assertEqual(doc.pages, [1])
        self.assertEqual(len(doc.nodes), 4)
        frame = doc.node(2)
        self.assertEqual(frame["type"], "frame")
        self.assertEqual(frame["width"], 400.0)
        self.assertEqual(doc.node(4)["characters"], "Hello FigCraft")

    def test_open_real_upstream_shape_fields(self) -> None:
        from nomorals.media.design import DesignDocument
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        self.assertEqual(doc.format, "figcraft")
        self.assertEqual(doc.version, 1)
        self.assertEqual(doc.current_page, 1)
        self.assertEqual(doc.next_id, 5)

    def test_unedited_save_keeps_document_json(self) -> None:
        from nomorals.media.design import DesignDocument
        raw = _read("simple.figcraft")
        doc = DesignDocument.from_bytes(raw)
        out = DesignDocument.from_bytes(doc.to_bytes())
        self.assertEqual(out.to_dict(), doc.to_dict())

    def test_broken_files_refused_with_kind(self) -> None:
        from nomorals.media.design import DesignDocument, DesignError
        for name, kind in (("broken.figcraft", "bad_file"),
                           ("notdesign.figcraft", "bad_format"),
                           ("corrupt.figcraft", "bad_format")):
            with self.assertRaises(DesignError) as ctx:
                DesignDocument.from_bytes(_read(name))
            self.assertEqual(ctx.exception.kind, kind, name)

    def test_create_blank_document(self) -> None:
        from nomorals.media.design import DesignDocument
        doc = DesignDocument.blank("Blank Test")
        self.assertEqual(doc.name, "Blank Test")
        self.assertEqual(doc.pages, [1])
        self.assertEqual(len(doc.nodes), 1)
        self.assertEqual(doc.node(1)["type"], "page")

    def test_alloc_id_uses_client_space(self) -> None:
        from nomorals.media.design import DesignDocument
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        first = doc.alloc_id(client_id=7)
        second = doc.alloc_id(client_id=7)
        self.assertEqual(first, (7 << 32) | 0)
        self.assertEqual(second, (7 << 32) | 1)
        self.assertNotEqual(first, second)
        other = doc.alloc_id(client_id=9)
        self.assertEqual(other >> 32, 9)

    def test_children_order(self) -> None:
        from nomorals.media.design import DesignDocument
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        self.assertEqual([n["id"] for n in doc.children(2)], [3, 4])
        self.assertEqual([n["id"] for n in doc.children(1)], [2])

    def test_node_not_found(self) -> None:
        from nomorals.media.design import DesignDocument, DesignError
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        with self.assertRaises(DesignError) as ctx:
            doc.node(999)
        self.assertEqual(ctx.exception.kind, "not_found")


class LayoutTests(unittest.TestCase):
    def test_vertical_stack_with_gap_and_padding(self) -> None:
        from nomorals.media.design import DesignDocument, compute_layout
        doc = DesignDocument.from_bytes(_read("layout.figcraft"))
        boxes = compute_layout(doc, 2)
        # Text hug height: 16pt * 140% = 22.4; frame hug height is
        # 16 + 22.4 + 8 + 40 + 8 + 50 + 16 = 160.4.
        self.assertAlmostEqual(boxes[2]["h"], 160.4, places=1)
        self.assertAlmostEqual(boxes[3]["y"], 16.0, places=1)
        self.assertAlmostEqual(boxes[3]["h"], 22.4, places=1)
        self.assertAlmostEqual(boxes[4]["y"], 16.0 + 22.4 + 8.0, places=1)
        self.assertAlmostEqual(boxes[5]["y"], 16.0 + 22.4 + 8.0 + 40.0 + 8.0, places=1)

    def test_fill_child_takes_content_width(self) -> None:
        from nomorals.media.design import DesignDocument, compute_layout
        doc = DesignDocument.from_bytes(_read("layout.figcraft"))
        boxes = compute_layout(doc, 2)
        # Frame width 400, padding 16 each side -> content width 368.
        self.assertAlmostEqual(boxes[4]["w"], 368.0, places=1)
        self.assertAlmostEqual(boxes[4]["x"], 16.0, places=1)
        self.assertAlmostEqual(boxes[5]["w"], 100.0, places=1)

    def test_text_hug_width_measured(self) -> None:
        from nomorals.media.design import measure_text
        w, h = measure_text("Layout", "Inter", 16.0, bold=True)
        self.assertGreater(w, 30.0)
        self.assertLess(w, 120.0)
        self.assertAlmostEqual(h, 22.4, places=1)

    def test_fixed_frame_keeps_size(self) -> None:
        from nomorals.media.design import DesignDocument, compute_layout
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        boxes = compute_layout(doc, 2)
        self.assertAlmostEqual(boxes[2]["w"], 400.0)
        self.assertAlmostEqual(boxes[2]["h"], 300.0)
        self.assertAlmostEqual(boxes[3]["x"], 16.0)
        self.assertAlmostEqual(boxes[3]["y"], 16.0)


class ComponentTests(unittest.TestCase):
    def test_variants_listed(self) -> None:
        from nomorals.media.design import DesignDocument, variants
        doc = DesignDocument.from_bytes(_read("components.figcraft"))
        names = sorted(v["name"] for v in variants(doc, 2))
        self.assertEqual(names, ["State=Default", "State=Hover"])

    def test_instance_resolves_variant_by_props(self) -> None:
        from nomorals.media.design import DesignDocument, resolve_instance
        doc = DesignDocument.from_bytes(_read("components.figcraft"))
        resolved = resolve_instance(doc, 9)
        self.assertEqual(resolved["component"], 6)
        bg = [n for n in resolved["tree"] if n["name"] == "Background"][0]
        fill = bg["fills"][0]["color"]
        # Hover background is #2f5597.
        self.assertAlmostEqual(fill["r"], 0x2F / 255.0, places=3)
        self.assertAlmostEqual(fill["g"], 0x55 / 255.0, places=3)

    def test_instance_override_applies(self) -> None:
        from nomorals.media.design import DesignDocument, resolve_instance
        doc = DesignDocument.from_bytes(_read("components.figcraft"))
        resolved = resolve_instance(doc, 10)
        label = [n for n in resolved["tree"] if n["name"] == "Label"][0]
        self.assertEqual(label["characters"], "Go now")

    def test_detach_instance_becomes_plain_nodes(self) -> None:
        from nomorals.media.design import DesignDocument, detach_instance
        doc = DesignDocument.from_bytes(_read("components.figcraft"))
        root = detach_instance(doc, 10)
        node = doc.node(root)
        self.assertEqual(node["type"], "frame")
        self.assertNotIn("component", node)
        labels = [n for n in doc.children(root) if n["name"] == "Label"]
        self.assertEqual(labels[0]["characters"], "Go now")


class ProtoTests(unittest.TestCase):
    def test_flows_reach_all_screens(self) -> None:
        from nomorals.media.design import DesignDocument, flows
        doc = DesignDocument.from_bytes(_read("proto.figcraft"))
        self.assertEqual(flows(doc, 2), [2, 5, 8])

    def test_wire_interaction_and_validate(self) -> None:
        from nomorals.media.design import (
            DesignDocument,
            add_interaction,
            flows,
            validate,
        )
        doc = DesignDocument.from_bytes(_read("proto.figcraft"))
        add_interaction(doc, 8, "on-click", "navigate", 2, "push")
        self.assertEqual(validate(doc), [])
        self.assertEqual(flows(doc, 8), [8, 2, 5])

    def test_dangling_destination_refused(self) -> None:
        from nomorals.media.design import (
            DesignDocument,
            DesignError,
            add_interaction,
        )
        doc = DesignDocument.from_bytes(_read("proto.figcraft"))
        with self.assertRaises(DesignError) as ctx:
            add_interaction(doc, 8, "on-click", "navigate", 999)
        self.assertEqual(ctx.exception.kind, "not_found")


class CollabTests(unittest.TestCase):
    def _two_replicas(self):
        from nomorals.media.design import DesignDocument, Replica
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        return Replica(doc.to_dict()), Replica(doc.to_dict())

    def test_frac_keys_order(self) -> None:
        from nomorals.media.design.collab import frac_between
        a = frac_between(None, None)
        b = frac_between(a, None)
        c = frac_between(a, b)
        self.assertTrue(a < b)
        self.assertTrue(a < c < b)

    def test_diff_and_apply_round_trip(self) -> None:
        from nomorals.media.design import DesignDocument
        from nomorals.media.design.collab import apply_batch
        replica, other = self._two_replicas()
        edited = DesignDocument.from_bytes(_read("simple.figcraft"))
        edited.node(4)["characters"] = "Changed"
        batch = replica.diff(edited.to_dict())
        self.assertTrue(any(ch["kind"] == "set" and ch["id"] == 4 for ch in batch))
        fixes = apply_batch(other, batch)
        self.assertEqual(fixes, [])
        self.assertEqual(other.materialize()["nodes"]["4"]["characters"], "Changed")

    def test_last_writer_wins_in_server_order(self) -> None:
        from nomorals.media.design.collab import apply_batch
        _, replica = self._two_replicas()
        first = [{"kind": "set", "id": 4, "prop": "characters", "value": "First"}]
        second = [{"kind": "set", "id": 4, "prop": "characters", "value": "Second"}]
        apply_batch(replica, first)
        apply_batch(replica, second)
        self.assertEqual(replica.materialize()["nodes"]["4"]["characters"], "Second")

    def test_cycle_rejected_with_fix(self) -> None:
        from nomorals.media.design.collab import apply_batch
        _, replica = self._two_replicas()
        # Frame 2 is the parent of rect 3; making 2 a child of 3 cycles.
        batch = [{"kind": "set", "id": 2, "prop": "parent", "value": 3}]
        fixes = apply_batch(replica, batch, strict=True)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0]["prop"], "parent")
        self.assertEqual(replica.materialize()["nodes"]["2"]["parent"], 1)

    def test_edit_of_deleted_node_rejected_lenient_skips(self) -> None:
        from nomorals.media.design.collab import apply_batch
        _, replica = self._two_replicas()
        apply_batch(replica, [{"kind": "delete", "id": 3}])
        fixes = apply_batch(
            replica, [{"kind": "set", "id": 3, "prop": "name", "value": "Ghost"}],
            strict=False)
        self.assertEqual(fixes, [])
        self.assertNotIn("3", replica.materialize()["nodes"])

    def test_create_then_delete_round_trip(self) -> None:
        from nomorals.media.design.collab import apply_batch
        _, replica = self._two_replicas()
        create = [{"kind": "create", "id": 50, "node_type": "rectangle",
                   "parent": 2, "position": None,
                   "props": {"name": "New", "width": 10.0, "height": 10.0}}]
        self.assertEqual(apply_batch(replica, create), [])
        self.assertIn("50", replica.materialize()["nodes"])
        self.assertEqual(apply_batch(replica, [{"kind": "delete", "id": 50}]), [])
        self.assertNotIn("50", replica.materialize()["nodes"])

    def test_history_undo_redo(self) -> None:
        from nomorals.media.design.collab import History, apply_batch
        _, replica = self._two_replicas()
        history = History()
        batch = [{"kind": "set", "id": 4, "prop": "characters", "value": "V2"}]
        inverse = replica.inverse(batch)
        apply_batch(replica, batch)
        history.record(1, "alice", batch, inverse)
        undo = history.undo_for("alice")
        self.assertIsNotNone(undo)
        apply_batch(replica, undo)
        self.assertEqual(replica.materialize()["nodes"]["4"]["characters"],
                         "Hello FigCraft")
        redo = history.redo_for("alice")
        apply_batch(replica, redo)
        self.assertEqual(replica.materialize()["nodes"]["4"]["characters"], "V2")

    def test_revert_to_seq(self) -> None:
        from nomorals.media.design.collab import History, apply_batch
        _, replica = self._two_replicas()
        history = History()
        for i, value in enumerate(("One", "Two", "Three")):
            batch = [{"kind": "set", "id": 4, "prop": "characters", "value": value}]
            inverse = replica.inverse(batch)
            apply_batch(replica, batch)
            history.record(i + 1, "alice", batch, inverse)
        compensating = history.revert_to(1)
        for batch in compensating:
            apply_batch(replica, batch)
        self.assertEqual(replica.materialize()["nodes"]["4"]["characters"], "One")


class SessionTests(unittest.TestCase):
    def _server(self):
        from nomorals.media.design import DesignDocument
        from nomorals.media.design.session import SessionServer
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        return SessionServer(doc)

    def test_two_editors_converge(self) -> None:
        from nomorals.media.design.session import SessionClient
        server = self._server()
        alice = SessionClient(server, "alice")
        bob = SessionClient(server, "bob")
        alice.connect()
        bob.connect()
        alice.set_text(4, "Alice text")
        bob.set_fill(3, "#ff0000")
        alice.receive()
        bob.receive()
        alice.receive()
        bob.receive()
        self.assertEqual(alice.doc.node(4)["characters"], "Alice text")
        self.assertEqual(bob.doc.node(4)["characters"], "Alice text")
        self.assertEqual(alice.doc.node(3)["fills"][0]["color"]["r"], 1.0)
        self.assertEqual(bob.doc.node(3)["fills"][0]["color"]["r"], 1.0)

    def test_conflicting_write_server_order_wins(self) -> None:
        from nomorals.media.design.session import SessionClient
        server = self._server()
        alice = SessionClient(server, "alice")
        bob = SessionClient(server, "bob")
        alice.connect()
        bob.connect()
        alice.set_text(4, "Alice wins")
        bob.set_text(4, "Bob wins")
        alice.receive()
        bob.receive()
        alice.receive()
        bob.receive()
        # Bob submitted second, so the server order puts Bob last.
        self.assertEqual(alice.doc.node(4)["characters"], "Bob wins")
        self.assertEqual(bob.doc.node(4)["characters"], "Bob wins")

    def test_view_link_cannot_edit(self) -> None:
        from nomorals.media.design import DesignError
        from nomorals.media.design.session import SessionClient
        server = self._server()
        link = server.create_link("view")
        carol = SessionClient(server, "carol")
        carol.connect(token=link)
        with self.assertRaises(DesignError):
            carol.set_text(4, "Sneaky")
        self.assertEqual(carol.doc.node(4)["characters"], "Hello FigCraft")

    def test_join_code_resolves(self) -> None:
        server = self._server()
        code = server.create_code("edit")
        self.assertEqual(len(code), 9)
        file_id, access = server.resolve_code(code)
        self.assertEqual(access, "edit")
        self.assertEqual(file_id, server.file_id)

    def test_presence_is_ephemeral(self) -> None:
        from nomorals.media.design.session import SessionClient
        server = self._server()
        alice = SessionClient(server, "alice")
        bob = SessionClient(server, "bob")
        alice.connect()
        bob.connect()
        alice.presence(cursor=(10.0, 20.0), selection=[4])
        peers = bob.receive_presence()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["cursor"], [10.0, 20.0])
        # Presence never lands in the document.
        self.assertNotIn("presence", alice.doc.to_dict())
        self.assertNotIn("presence", server.doc.to_dict())

    def test_offline_queue_reconnects(self) -> None:
        from nomorals.media.design.session import SessionClient
        server = self._server()
        alice = SessionClient(server, "alice")
        bob = SessionClient(server, "bob")
        alice.connect()
        bob.connect()
        alice.go_offline()
        alice.set_text(4, "While offline")
        self.assertEqual(bob.doc.node(4)["characters"], "Hello FigCraft")
        alice.reconnect()
        bob.receive()
        self.assertEqual(bob.doc.node(4)["characters"], "While offline")


class RenderTests(unittest.TestCase):
    def test_render_frame_png(self) -> None:
        from nomorals.media.design import DesignDocument, render_frame
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        png, info = render_frame(doc, 2)
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertEqual((info["width"], info["height"]), (400, 300))
        self.assertEqual(info["unrendered"], [])
        image = io.BytesIO(png)
        from PIL import Image
        img = Image.open(image)
        # Accent rect #4472c4 at (16..136, 16..64): sample its middle.
        self.assertEqual(img.getpixel((76, 40))[:3], (0x44, 0x72, 0xC4))

    def test_render_gradient_and_star(self) -> None:
        from nomorals.media.design import DesignDocument, render_frame
        doc = DesignDocument.from_bytes(_read("styled.figcraft"))
        png, info = render_frame(doc, 2)
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertEqual((info["width"], info["height"]), (400, 300))
        # Drop shadow is honestly reported, never faked.
        self.assertTrue(any("shadow" in u for u in info["unrendered"]))

    def test_render_text_present(self) -> None:
        from nomorals.media.design import DesignDocument, render_frame
        doc = DesignDocument.from_bytes(_read("simple.figcraft"))
        png, _ = render_frame(doc, 2)
        from PIL import Image
        img = Image.open(io.BytesIO(png)).convert("RGB")
        # The title text draws dark pixels inside its box.
        dark = sum(1 for x in range(16, 216) for y in range(80, 102)
                   if sum(img.getpixel((x, y))) < 200)
        self.assertGreater(dark, 20)


class EditorTests(unittest.TestCase):
    def test_edit_round_trip_fidelity(self) -> None:
        from nomorals.media.design import DesignDocument, DesignEditor
        raw = _read("simple.figcraft")
        doc = DesignDocument.from_bytes(raw)
        editor = DesignEditor(doc)
        editor.set_text(4, "Edited title")
        editor.add_rect(2, "New rect", 200.0, 200.0, 40.0, 30.0, "#ff0000")
        report = editor.fidelity_report(raw)
        self.assertEqual(report["untouched_changed"], 0)
        self.assertEqual(report["changed_ids"], [4])
        self.assertEqual(len(report["added"]), 1)
        saved = DesignDocument.from_bytes(editor.to_bytes())
        self.assertEqual(saved.node(4)["characters"], "Edited title")

    def test_add_component_and_instance(self) -> None:
        from nomorals.media.design import DesignDocument, DesignEditor
        doc = DesignDocument.from_bytes(_read("components.figcraft"))
        editor = DesignEditor(doc)
        comp = editor.add_component(1, "Chip", 120.0, 32.0)
        inst = editor.add_instance(1, comp, {"State": "Default"})
        self.assertEqual(doc.node(inst)["type"], "instance")
        self.assertEqual(doc.node(inst)["source"], comp)


class ParityTests(unittest.TestCase):
    def test_parity_runs(self) -> None:
        from nomorals.media.design import run_parity
        result = run_parity()
        self.assertEqual(result["failed"], [])
        self.assertGreaterEqual(result["passed"], 10)

    def test_missing_fixture_fails(self) -> None:
        import tempfile

        from nomorals.media.design import run_parity
        toml = ('schema_version = 1\n\n[[cases]]\n'
                'id = "ghost"\nfixture = "no-such.figcraft"\n'
                'ops = ["inspect"]\n[cases.expect]\nnodes = 1\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ghost.toml"
            path.write_text(toml)
            result = run_parity(path=path)
        self.assertGreater(len(result["failed"]), 0)
        self.assertIn("ghost", result["failed"])


class AttributionTests(unittest.TestCase):
    def test_attribution_verifies(self) -> None:
        from nomorals.media.design import verify_attribution
        result = verify_attribution()
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["skipped"], [])
        self.assertGreater(result["verified"], 0)


class ToolTests(unittest.TestCase):
    def _registry(self):
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        return registry

    def test_tool_registered_and_count(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry.get("figcraft"))
        # 723 after the deckcraft port, 724 after the designcraft
        # port (lane 104), 725 after the figcraft port (lane 105),
        # 726 after the soundcraft port (lane 107), 727 after the
        # cadcraft port (lane 106), 728 after the materialcraft
        # port (lane 109), 729 after the texturecraft port
        # (lane 108), 730 after the anicraft port (lane 110),
        # 731 after the craft-anything port (lane 111), 732 after the
        # craftlauncher port (lane 112), 733 after the craft-libs
        # port (lane 113) added the craftlibs tool, 734 after the
        # craft-fonts port (lane 114) added the craftfonts tool.
        # 735 after the indd-utils port (lane 117) added the inddutils tool.
        # 748 after the auth-plugin wiring added 5 tools (session_forge, session_verify, payload_png_bomb, payload_svg_bomb, payload_keylogger) on top of the bughunter port (lane 121). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet).
        self.assertEqual(len(registry._tools), 800)

    def test_tool_inspect_and_render(self) -> None:
        import tempfile
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.figcraft"
            src.write_bytes(_read("simple.figcraft"))
            out = registry.call("figcraft", action="inspect", path=str(src))
            self.assertTrue(out.ok)
            self.assertEqual(out.value["name"], "Simple Fixture")
            png = Path(tmp) / "frame.png"
            result = registry.call("figcraft", action="render", path=str(src),
                                   node=2, out=str(png))
            self.assertTrue(result.ok)
            self.assertTrue(png.exists())
            self.assertGreater(png.stat().st_size, 100)

    def test_tool_unknown_action_is_error_outcome(self) -> None:
        registry = self._registry()
        out = registry.call("figcraft", action="explode")
        self.assertFalse(out.ok)
        self.assertIn("unknown", str(out.error).lower())


if __name__ == "__main__":
    unittest.main()
