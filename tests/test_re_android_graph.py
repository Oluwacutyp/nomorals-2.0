"""RE Pass 2 — the Android reference graph over jadx output.

The fixture is a small hand-written Java tree (an INPUT — jadx's
output shape: package dirs of .java sources). The graph's
answers over it are exact: who calls Gamma.spin, what Alpha.run
touches, and the traps — a call mentioned in a comment and in a
string literal must NOT become an edge, because the scanner
strips both before it reads.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.reverse import ReEngine, android_graph
from nomorals.storage.db import Database

ALPHA = """package com.example.app;

import com.example.util.Beta;
import com.example.util.Gamma;

public class Alpha {
    private Beta beta;
    private String note = "Gamma.spin() in prose must not count";

    public void run() {
        Beta.poke();
        Gamma g = new Gamma();
        g.spin();
        local();
    }

    private void local() {
        beta.poke();
    }

    public static void reset() {
    }
}
"""

BETA = """package com.example.util;

public class Beta {
    // Gamma.spin() in a comment must not count either
    public static void poke() {
        Gamma.spin();
    }
}
"""

GAMMA = """package com.example.util;

public class Gamma {
    public void spin() {
        Alpha.reset();
    }
}
"""

FOO_A = """package com.one;

public class Foo {
    public void act() {
    }
}
"""

FOO_B = """package com.two;

public class Foo {
    public void act() {
    }
}
"""


def _write_tree(root: Path) -> Path:
    sources = root / "sources"
    files = {
        "com/example/app/Alpha.java": ALPHA,
        "com/example/util/Beta.java": BETA,
        "com/example/util/Gamma.java": GAMMA,
        "com/one/Foo.java": FOO_A,
        "com/two/Foo.java": FOO_B,
    }
    for rel, text in files.items():
        dest = sources / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return sources


class _TmpHome:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-graph-")
        return Path(self.dir)

    def __exit__(self, *exc):
        return False


class GraphBuildTest(unittest.TestCase):
    def test_graph_answers_are_exact(self):
        with _TmpHome() as home:
            sources = _write_tree(home)
            graph = android_graph.build_graph(sources)
        stats = graph["stats"]
        self.assertEqual(stats["classes"], 5)
        self.assertIn("spin", graph["classes"]
                      ["com.example.util.Gamma"]["methods"])
        self.assertEqual(graph["collisions"], ["Foo"])

        callers = android_graph.callers(graph, "Gamma.spin")
        froms = {(c["from_class"], c["from_method"])
                 for c in callers["callers"]}
        self.assertEqual(froms, {
            ("com.example.util.Beta", "poke"),
            ("com.example.app.Alpha", "run"),
        })
        self.assertEqual(callers["references"], 2)

        # the comment + string traps did not become edges
        beta_callers = android_graph.callers(graph, "Beta.poke")
        self.assertEqual(beta_callers["references"], 2)
        froms = {(c["from_class"], c["from_method"])
                 for c in beta_callers["callers"]}
        self.assertEqual(froms, {
            ("com.example.app.Alpha", "run"),
            ("com.example.app.Alpha", "local"),
        })

        callees = android_graph.callees(graph, "Alpha.run")
        tos = {(c["to_class"], c["to_method"], c["kind"])
               for c in callees["callees"]}
        self.assertIn(("com.example.util.Beta", "poke", "call"),
                      tos)
        self.assertIn(("com.example.util.Gamma", "spin", "call"),
                      tos)
        self.assertIn(("com.example.util.Gamma", "Gamma",
                       "instantiates"), tos)
        self.assertIn(("com.example.app.Alpha", "local", "call"),
                      tos)

        profile = android_graph.class_profile(graph, "Gamma")
        self.assertEqual(profile["class"]["fqcn"],
                         "com.example.util.Gamma")
        self.assertGreaterEqual(profile["inbound_count"], 3)

        found = android_graph.search(graph, "spin")
        self.assertIn("com.example.util.Gamma",
                      [hit["class"] for hit in found["methods"]])

    def test_ambiguous_simple_name_is_an_honest_error(self):
        with _TmpHome() as home:
            sources = _write_tree(home)
            graph = android_graph.build_graph(sources)
            with self.assertRaises(android_graph.GraphError):
                android_graph.callers(graph, "Foo.act")
            ok = android_graph.callers(graph, "com.one.Foo.act")
            self.assertEqual(ok["resolved_class"], "com.one.Foo")

    def test_sources_root_resolution(self):
        with _TmpHome() as home:
            sources = _write_tree(home)
            # the jadx dir itself (sources/ inside) resolves
            self.assertEqual(android_graph.sources_root_of(home),
                             sources)
            self.assertEqual(android_graph.sources_root_of(sources),
                             sources)
            empty = home / "empty"
            empty.mkdir()
            with self.assertRaises(android_graph.GraphError):
                android_graph.sources_root_of(empty)


class GraphEngineTest(unittest.TestCase):
    def _engine(self) -> ReEngine:
        return ReEngine(db=Database(":memory:"))

    def test_build_and_query_land_as_evidence(self):
        engine = self._engine()
        with _TmpHome() as home:
            _write_tree(home)
            built = engine.android_graph_build(str(home))
            self.assertNotIn("error", built)
            self.assertEqual(built["stats"]["classes"], 5)
            out = engine.android_query(str(home), "callers",
                                       "Gamma.spin")
        self.assertNotIn("error", out)
        self.assertEqual(out["answer"]["caller_count"], 2)
        rec = engine.store.get(out["evidence_id"])
        self.assertEqual(rec.predicate_type, "android.references")
        self.assertEqual(rec.operation, "graph.callers_of")
        self.assertEqual(rec.provider_id, "devon.android-graph")
        self.assertTrue(any("reflection" in lim
                            for lim in rec.limitations))
        graph_rec = engine.store.get(out["graph_evidence_id"])
        self.assertEqual(graph_rec.predicate_type,
                         "android.class_graph")
        self.assertIn(graph_rec.evidence_id, rec.links)

    def test_query_unknown_class_is_an_honest_error(self):
        engine = self._engine()
        with _TmpHome() as home:
            _write_tree(home)
            out = engine.android_query(str(home), "callers",
                                       "Nope.missing")
        self.assertIn("error", out)
        self.assertIn("no class matches", out["error"])


class GraphSurfaceTest(unittest.TestCase):
    def test_registry_graph_action(self):
        from nomorals.agents import reverse
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(db=Database(":memory:"),
                                  memory=None, settings=None)
        registry = ToolRegistry(context, enforce=False)
        reverse.register(registry)
        with _TmpHome() as home:
            _write_tree(home)
            outcome = registry.call(
                "re", action="graph", path=str(home),
                query="callees", target="Alpha.run")
        self.assertTrue(outcome.ok, outcome)
        self.assertEqual(outcome.value["query"], "callees")
        self.assertGreaterEqual(
            outcome.value["answer"]["callee_count"], 3)

    def test_cli_graph_and_parser(self):
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.re import _cmd_re
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        args = parser.parse_args(["re", "provision"])
        self.assertEqual(args.re_action, "provision")
        args = parser.parse_args(["re", "inspect", "/x", "--deep"])
        self.assertTrue(args.deep)
        args = parser.parse_args(
            ["re", "graph", "/apk", "callers", "Foo.bar"])
        self.assertEqual(args.re_action, "graph")
        self.assertEqual(args.query, "callers")

        context = SimpleNamespace(db=Database(":memory:"),
                                  memory=None, settings=None)
        with _TmpHome() as home:
            _write_tree(home)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_re(SimpleNamespace(
                    re_action="graph", path=str(home),
                    query="callers", target="Gamma.spin"), context)
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("callers", out)
        self.assertIn("Beta", out)


if __name__ == "__main__":
    unittest.main()
