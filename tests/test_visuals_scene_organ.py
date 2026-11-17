"""Scene-spec organ: spec validation, renderer ladder, brain tool.

The organ is first a tool for Devon herself — she composes a scene
spec and uses it to drive her own outputs (PNG images, scene specs
handed to the code-video rung) — and second the live display ladder
from the owner's 2026-10-31 Farza/Clicky link. Every renderer ends
in something she can send or hand to the next stage: a PNG file, a
live HTML page file, or the validated spec itself.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nomorals.visuals.codegraph import callgraph_scene
from nomorals.visuals.impossible import (
    dossier_scene,
    render_dossier_png,
)
from nomorals.visuals.profile import pick_target
from nomorals.visuals.render import (
    render_live_html,
    render_png,
    render_scene,
    render_text,
    write_live_page,
)
from nomorals.visuals.spec import (
    SceneSpecError,
    auto_layout,
    diagram_from_parts,
    validate_scene_spec,
)


def _diagram(**overrides):
    spec = {
        "kind": "diagram",
        "title": "test diagram",
        "nodes": [
            {"id": "a", "label": "Alpha", "x": 0.2, "y": 0.5},
            {"id": "b", "label": "Beta", "x": 0.8, "y": 0.5},
        ],
        "edges": [{"from": "a", "to": "b", "label": "flows"}],
    }
    spec.update(overrides)
    return spec


def _dossier():
    return {
        "formulation": {
            "raw": "a car that never breaks",
            "goal_function": "transport without failure",
            "constraints": ["budget under $5000", "no exotic materials"],
            "context": "",
        },
        "classification": {"primary": "engineering"},
        "gate": {"verdict": "bounded", "why": "cost binds"},
        "reframed": {
            "reframed_goal": "failures predictable, cheap, scheduled",
            "verdict": "bounded",
            "ache": "unexpected breakdowns",
        },
        "prior_art": {
            "query": "a car that never breaks",
            "results": {
                "arxiv": [
                    {"status": "found", "title": "Reliability by design"},
                    {"status": "found", "title": "Predictive maintenance"},
                ],
                "openalex": [{"status": "found", "title": "Failure modes"}],
                "patentsview": [{"status": "unavailable"}],
            },
            "failed_attempts": [
                {"status": "found", "title": "Self-healing paint trial"},
            ],
            "summary": "3 prior-art hits across free sources; 1 failed-attempt leads.",
        },
        "claim_id": "test-claim-1",
    }


class SpecValidationTests(unittest.TestCase):
    def test_valid_diagram_keeps_explicit_coords(self):
        spec = validate_scene_spec(_diagram())
        self.assertEqual(spec["kind"], "diagram")
        self.assertEqual(spec["nodes"][0]["x"], 0.2)
        self.assertEqual(spec["nodes"][1]["shape"], "box")

    def test_auto_layout_fills_missing_coords_deterministically(self):
        raw = {
            "kind": "diagram",
            "title": "laid out",
            "nodes": [{"id": "src"}, {"id": "mid"}, {"id": "sink"}],
            "edges": [
                {"from": "src", "to": "mid"},
                {"from": "mid", "to": "sink"},
            ],
        }
        first = validate_scene_spec(raw)
        second = validate_scene_spec(raw)
        self.assertEqual(first, second)
        xs = [n["x"] for n in first["nodes"]]
        self.assertLess(xs[0], xs[1])
        self.assertLess(xs[1], xs[2])
        for n in first["nodes"]:
            self.assertTrue(0.0 <= n["x"] <= 1.0)
            self.assertTrue(0.0 <= n["y"] <= 1.0)

    def test_auto_layout_keeps_explicit_coords(self):
        raw = {
            "kind": "diagram",
            "nodes": [
                {"id": "a", "x": 0.1, "y": 0.9},
                {"id": "b"},
            ],
            "edges": [{"from": "a", "to": "b"}],
        }
        spec = validate_scene_spec(raw)
        self.assertEqual(spec["nodes"][0]["x"], 0.1)
        self.assertEqual(spec["nodes"][0]["y"], 0.9)

    def test_not_a_dict(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec(["diagram"])

    def test_unknown_kind(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec({"kind": "hologram", "nodes": []})

    def test_diagram_needs_nodes(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec({"kind": "diagram", "nodes": []})

    def test_duplicate_node_id(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec(_diagram(nodes=[
                {"id": "a", "x": 0.2, "y": 0.5},
                {"id": "a", "x": 0.8, "y": 0.5},
            ]))

    def test_node_coord_out_of_range(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec(_diagram(nodes=[
                {"id": "a", "x": 1.5, "y": 0.5},
                {"id": "b", "x": 0.8, "y": 0.5},
            ]))

    def test_edge_to_unknown_node(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec(_diagram(edges=[
                {"from": "a", "to": "ghost"},
            ]))

    def test_bad_shape(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec(_diagram(nodes=[
                {"id": "a", "x": 0.2, "y": 0.5, "shape": "pyramid"},
                {"id": "b", "x": 0.8, "y": 0.5},
            ]))

    def test_overlay_annotations_validated(self):
        spec = validate_scene_spec({
            "kind": "overlay",
            "title": "pins",
            "annotations": [
                {"x": 0.5, "y": 0.5, "text": "look here", "kind": "cursor"},
                {"x": 0.1, "y": 0.1, "text": "zone", "kind": "rect"},
            ],
        })
        self.assertEqual(len(spec["annotations"]), 2)
        self.assertEqual(spec["annotations"][0]["kind"], "cursor")

    def test_overlay_annotation_needs_coords(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec({
                "kind": "overlay",
                "annotations": [{"text": "nowhere"}],
            })

    def test_overlay_bad_annotation_kind(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec({
                "kind": "overlay",
                "annotations": [
                    {"x": 0.5, "y": 0.5, "kind": "teleport"},
                ],
            })

    def test_scene_objects_validated(self):
        spec = validate_scene_spec({
            "kind": "scene",
            "title": "room",
            "objects": [
                {"type": "box", "position": [0, 0, 0], "size": 1},
                {"type": "sphere", "position": [2, 0, 0], "size": 0.5},
            ],
        })
        self.assertEqual(len(spec["objects"]), 2)

    def test_scene_bad_object_type(self):
        with self.assertRaises(SceneSpecError):
            validate_scene_spec({
                "kind": "scene",
                "objects": [{"type": "torus-knot"}],
            })

    def test_input_dict_is_never_mutated(self):
        raw = _diagram()
        before = json.dumps(raw, sort_keys=True)
        validate_scene_spec(raw)
        self.assertEqual(json.dumps(raw, sort_keys=True), before)

    def test_diagram_from_parts_lays_out_for_the_brain(self):
        spec = diagram_from_parts(
            "flow", ["one", "two", "three"],
            [("one", "two"), ("two", "three")],
        )
        self.assertEqual(spec["kind"], "diagram")
        self.assertEqual([n["id"] for n in spec["nodes"]],
                         ["one", "two", "three"])
        xs = [n["x"] for n in spec["nodes"]]
        self.assertLess(xs[0], xs[1])
        self.assertLess(xs[1], xs[2])

    def test_auto_layout_is_pure(self):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [{"from": "a", "to": "b"}]
        first = auto_layout(nodes, edges)
        second = auto_layout(nodes, edges)
        self.assertEqual(first, second)
        self.assertNotIn("x", nodes[0])


class RendererTests(unittest.TestCase):
    def test_live_html_diagram_is_self_contained(self):
        html = render_live_html(_diagram())
        self.assertIn("<canvas", html)
        self.assertIn("Alpha", html)
        self.assertIn("Beta", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_live_html_scene_uses_three_page(self):
        spec = validate_scene_spec({
            "kind": "scene",
            "title": "room",
            "objects": [{"type": "box", "position": [0, 0, 0], "size": 1}],
        })
        html = render_live_html(spec)
        self.assertIn("three", html.lower())
        self.assertIn("room", html)

    def test_write_live_page_writes_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_live_page(_diagram(), Path(tmp) / "scene.html")
            self.assertTrue(Path(path).is_file())
            self.assertIn("<canvas", Path(path).read_text())

    def test_png_diagram_is_a_real_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = render_png(_diagram(), Path(tmp) / "d.png")
            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG"))
            self.assertGreater(len(data), 1000)

    def test_png_overlay_marks_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {
                "kind": "overlay",
                "title": "pins",
                "annotations": [
                    {"x": 0.5, "y": 0.5, "text": "look here",
                     "kind": "cursor"},
                ],
            }
            path = render_png(spec, Path(tmp) / "o.png")
            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG"))
            self.assertGreater(len(data), 1000)

    def test_png_scene_refuses_honestly(self):
        spec = validate_scene_spec({
            "kind": "scene",
            "objects": [{"type": "box", "position": [0, 0, 0], "size": 1}],
        })
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SceneSpecError) as caught:
                render_png(spec, Path(tmp) / "s.png")
            self.assertIn("live", str(caught.exception).lower())

    def test_text_diagram_names_nodes_and_edges(self):
        text = render_text(_diagram())
        self.assertIn("Alpha", text)
        self.assertIn("Beta", text)
        self.assertIn("a -> b", text)

    def test_text_overlay_lists_annotations(self):
        text = render_text({
            "kind": "overlay",
            "annotations": [
                {"x": 0.25, "y": 0.75, "text": "here", "kind": "rect"},
            ],
        })
        self.assertIn("here", text)
        self.assertIn("rect", text)

    def test_text_scene_lists_objects(self):
        text = render_text({
            "kind": "scene",
            "objects": [{"type": "box", "position": [0, 0, 0], "size": 2}],
        })
        self.assertIn("box", text)


class ProfilePickTests(unittest.TestCase):
    def test_auto_diagram_goes_to_chat_png(self):
        self.assertEqual(pick_target(_diagram()), "chat")

    def test_auto_overlay_goes_to_chat_png(self):
        spec = {"kind": "overlay", "annotations": [
            {"x": 0.5, "y": 0.5, "text": "x"}]}
        self.assertEqual(pick_target(spec), "chat")

    def test_auto_scene_goes_live(self):
        spec = {"kind": "scene", "objects": [
            {"type": "box", "position": [0, 0, 0], "size": 1}]}
        self.assertEqual(pick_target(spec), "live")

    def test_explicit_surface_wins(self):
        self.assertEqual(pick_target(_diagram(), surface="terminal"),
                         "terminal")
        self.assertEqual(pick_target(_diagram(), surface="live"), "live")

    def test_profile_kind_is_recorded_not_a_clamp(self):
        self.assertEqual(
            pick_target(_diagram(), profile_kind="termux"), "chat")
        self.assertEqual(
            pick_target(_diagram(), profile_kind="workstation"), "chat")


class RenderSceneTests(unittest.TestCase):
    def test_render_scene_writes_png_and_returns_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = render_scene(_diagram(), out_dir=tmp)
            self.assertEqual(record["engine"], "visuals")
            self.assertEqual(record["target"], "chat")
            self.assertEqual(record["format"], "png")
            self.assertTrue(Path(record["path"]).is_file())
            self.assertEqual(record["spec"]["kind"], "diagram")
            self.assertEqual(record["nodes"], 2)
            self.assertEqual(record["edges"], 1)

    def test_render_scene_terminal_returns_text_no_file(self):
        record = render_scene(_diagram(), target="terminal")
        self.assertEqual(record["format"], "text")
        self.assertIsNone(record["path"])
        self.assertIn("Alpha", record["text"])

    def test_render_scene_live_writes_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = render_scene(_diagram(), target="live", out_dir=tmp)
            self.assertEqual(record["format"], "html")
            self.assertTrue(str(record["path"]).endswith(".html"))
            self.assertIn("<canvas",
                          Path(record["path"]).read_text())

    def test_render_scene_hands_spec_to_next_stage(self):
        # The validated spec in the record IS the handoff to the
        # code-video rung: the next stage consumes it, never re-authors it.
        with tempfile.TemporaryDirectory() as tmp:
            record = render_scene(_diagram(), out_dir=tmp)
            again = validate_scene_spec(record["spec"])
            self.assertEqual(again, record["spec"])


class ToolTests(unittest.TestCase):
    def test_render_scene_tool_is_registered(self):
        from nomorals.tools import visuals as visuals_tool
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(enforce=False)
        visuals_tool.register(registry)
        self.assertIn("render_scene", registry.names())

    def test_render_scene_tool_round_trip(self):
        from nomorals.tools import visuals as visuals_tool
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(enforce=False)
            visuals_tool.register(registry)
            outcome = registry.call(
                "render_scene", spec=_diagram(), out_dir=tmp)
            self.assertTrue(outcome.ok, outcome.error)
            record = outcome.value
            self.assertEqual(record["engine"], "visuals")
            self.assertTrue(Path(record["path"]).is_file())


class ImpossibleConsumerTests(unittest.TestCase):
    def test_prior_art_map_names_sources_and_failed_leads(self):
        spec = dossier_scene(_dossier(), kind="prior_art")
        labels = [n["label"] for n in spec["nodes"]]
        self.assertTrue(any("arxiv" in label for label in labels))
        self.assertTrue(any("openalex" in label for label in labels))
        self.assertTrue(any("failed" in label for label in labels))
        self.assertTrue(any("car" in label for label in labels))

    def test_prior_art_map_empty_state_is_honest(self):
        dossier = _dossier()
        dossier["prior_art"] = {
            "query": "x", "results": {}, "failed_attempts": [],
            "summary": "0 prior-art hits.",
        }
        spec = dossier_scene(dossier, kind="prior_art")
        labels = [n["label"] for n in spec["nodes"]]
        self.assertTrue(any("no prior art" in label.lower()
                            for label in labels))

    def test_constraint_diagram_names_constraints(self):
        spec = dossier_scene(_dossier(), kind="constraint")
        labels = [n["label"] for n in spec["nodes"]]
        self.assertTrue(any("budget" in label for label in labels))
        self.assertTrue(any("exotic" in label for label in labels))

    def test_reframe_diagram_is_before_after(self):
        spec = dossier_scene(_dossier(), kind="reframe")
        labels = [n["label"] for n in spec["nodes"]]
        self.assertTrue(any("car" in label for label in labels))
        self.assertTrue(any("predictable" in label for label in labels))
        self.assertTrue(any("ache" in label for label in labels))
        self.assertTrue(any(e["from"] == "before"
                            and e["to"] == "after"
                            and e["label"] == "reframe"
                            for e in spec["edges"]))

    def test_unknown_dossier_kind(self):
        with self.assertRaises(SceneSpecError):
            dossier_scene(_dossier(), kind="timeline")

    def test_render_dossier_png_writes_a_real_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = render_dossier_png(_dossier(), kind="prior_art",
                                      out_dir=tmp)
            data = Path(path).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG"))
            self.assertGreater(len(data), 1000)


class CodegraphConsumerTests(unittest.TestCase):
    def _tiny_repo(self, root: Path):
        (root / "a.py").write_text(
            "def foo():\n    return bar()\n\n\ndef bar():\n    return 1\n")
        (root / "b.py").write_text(
            "from a import foo\n\n\ndef main():\n    foo()\n")

    def test_callgraph_scene_from_a_real_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._tiny_repo(root)
            spec = callgraph_scene(str(root), "a.foo",
                                   db_path=str(Path(tmp) / "g.db"))
            ids = {n["id"] for n in spec["nodes"]}
            self.assertIn("a.foo", ids)
            self.assertIn("a.bar", ids)
            self.assertIn("b.main", ids)
            self.assertTrue(any(e["from"] == "a.foo"
                                and e["to"] == "a.bar"
                                for e in spec["edges"]))
            self.assertTrue(any(e["from"] == "b.main"
                                and e["to"] == "a.foo"
                                for e in spec["edges"]))

    def test_callgraph_unknown_symbol_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            self._tiny_repo(root)
            with self.assertRaises(SceneSpecError):
                callgraph_scene(str(root), "nope.missing",
                                db_path=str(Path(tmp) / "g.db"))


if __name__ == "__main__":
    unittest.main()
