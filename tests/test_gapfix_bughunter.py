"""BugHunter port tests (lane 121, Claude-BugHunter -> Devon).

Covers: the upstream self-tests (scope / skill map / closure /
state / memory / agent / recon, all offline), the machine-checked
parity TOML, the SHA-256 attribution record, the 8-tool surface
(registry count 774; 775 after this lane (B.19) added scene_spawn; 776 after this lane (B.20) added cloudcam), the 7-Question Gate, the closure ledger, and 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control.
the engine end to end on a scoped fixture HTTP server — with scope
enforcement demonstrated (an out-of-scope URL is refused in code
and never tested).

No real-format secret literals appear in this file: the one secret
the fixture JS carries is assembled at runtime from concatenated
parts, per the programme rule.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

#: AWS example key, assembled at runtime — never a literal in source.
_AWS_EXAMPLE = "AKIA" + "IOSFODNN7EXAMPLE"

STRONG_FINDING = {
    "url": "https://app.example.com/api/account?id=6",
    "param": "id",
    "vuln_class": "idor",
    "severity": "high",
    "asset": "https://app.example.com",
    "evidence": (
        "curl https://app.example.com/api/account?id=6 HTTP/1.1 — "
        "unauthenticated, previously unknown, leaked PII for other users"
    ),
}

WEAK_FINDING = {
    "url": "https://app.example.com/search?q=x",
    "param": "q",
    "vuln_class": "xss",
    "evidence": "",
}


def _registry():
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


def _call(registry, name, **kwargs):
    # registry.call returns an Outcome, not the dict — unwrap .value
    # the same way the craftlauncher test does.
    outcome = registry.call(name, **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


class _FixtureHandler(BaseHTTPRequestHandler):
    """A tiny scoped target: HTML + JS + robots + OpenAPI."""

    def log_message(self, *args):  # noqa: D102 - silence the server
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server naming
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = (
                b'<html><head><script src="/app.js"></script></head><body>'
                b'<a href="/api/account?id=1">account</a>'
                b'<a href="/search?q=x">search</a>'
                b'<a href="/graphql">graphql</a>'
                b'<a href="https://evil.example.com/x">out of scope</a>'
                b'<form action="/search"><input name="q"></form>'
                b"</body></html>"
            )
            self._send(body, "text/html")
        elif parsed.path == "/app.js":
            body = (
                'fetch("/api/account?id=" + id);\n'
                'fetch("/search?q=test");\n'
                'const other = "https://evil.example.com/x";\n'
                f'const key = "{_AWS_EXAMPLE}";\n'
            ).encode()
            self._send(body, "application/javascript")
        elif parsed.path == "/robots.txt":
            self._send(b"User-agent: *\nDisallow: /admin/\n",
                       "text/plain")
        elif parsed.path == "/openapi.json":
            self._send(json.dumps({
                "openapi": "3.0.0",
                "info": {"title": "fixture", "version": "1"},
                "paths": {
                    "/api/account": {
                        "get": {"security": [{"bearerAuth": []}],
                                "responses": {"200": {"description": "ok"}}}
                    },
                },
            }).encode(), "application/json")
        elif parsed.path == "/api/account":
            self._send(b'{"id": 5, "name": "user5"}',
                       "application/json")
        elif parsed.path == "/search":
            self._send(b"<html><body>results</body></html>", "text/html")
        elif parsed.path == "/graphql":
            self._send(b'{"data": {}}', "application/json")
        else:
            self.send_response(404)
            self.end_headers()


def _canned_suggest(prompt: str) -> str:
    """A deterministic judge: idor is real, everything else is not."""
    if "Adversarially verify" in prompt:
        if "idor" in prompt:
            return (
                "```json\n{\"real\": true, \"severity\": \"high\", "
                "\"reason\": \"confirmed PII exposure for arbitrary id\"}\n```"
            )
        return ("```json\n{\"real\": false, "
                "\"reason\": \"not reproducible\"}\n```")
    if "idor" in prompt:
        return (
            "```json\n{\"vulnerable\": true, \"severity\": \"high\", "
            "\"evidence\": \"id=6 returns another user's PII\", "
            "\"request\": \"GET /api/account?id=6\", "
            "\"hosts_contacted\": [], \"proof_note\": \"\"}\n```"
        )
    return "```json\n{\"vulnerable\": false}\n```"


class SelfTestTest(unittest.TestCase):
    def test_upstream_selftests_all_pass(self):
        from nomorals.security.bughunter.agent import agent_selftest
        from nomorals.security.bughunter.closure import closure_selftest
        from nomorals.security.bughunter.memory import memory_selftest
        from nomorals.security.bughunter.recon_map import recon_selftest
        from nomorals.security.bughunter.scope import scope_selftest
        from nomorals.security.bughunter.skillmap import skill_map_selftest
        from nomorals.security.bughunter.state import state_selftest

        self.assertTrue(scope_selftest())
        self.assertTrue(closure_selftest())
        self.assertTrue(state_selftest())
        self.assertTrue(memory_selftest())
        self.assertTrue(agent_selftest())
        self.assertTrue(recon_selftest())
        result = skill_map_selftest()
        self.assertEqual(result["errors"], [])


class ParityTest(unittest.TestCase):
    def test_parity_all_cases_pass(self):
        from nomorals.security.bughunter.parity import run_parity

        result = run_parity()
        failed = [c for c in result["cases"] if not c["passed"]]
        self.assertEqual(result["failed"], 0, failed)
        self.assertEqual(result["passed"], 16)


class AttributionTest(unittest.TestCase):
    def test_attribution_verified(self):
        from nomorals.security.bughunter.attribution import (
            verify_attribution,
        )

        result = verify_attribution()
        self.assertEqual(result["failed"], [], result["failed"])
        self.assertEqual(result["skipped"], [], result["skipped"])
        # 161 vendored upstream files + 1 devon organ entry.
        self.assertEqual(result["verified"], 162)

    def test_vendored_counts_on_disk(self):
        from nomorals.security.bughunter.classify import (
            library_dir,
        )

        lib = library_dir()
        skills = [p for p in (lib / "skills").iterdir() if p.is_dir()]
        reports = sorted((lib / "disclosed-reports").glob("hunt-*.md"))
        commands = sorted((lib / "commands").glob("*.md"))
        # The auditable unit is the on-disk count: 83 skills, 36
        # pattern files (hunt-*.md) plus a README, 15 commands — not
        # the repo's own 681/574+.
        self.assertEqual(len(skills), 83)
        self.assertEqual(len(reports), 36)
        self.assertTrue((lib / "disclosed-reports" / "README.md").is_file())
        self.assertEqual(len(commands), 15)
        self.assertTrue((lib / "skill_index.json").is_file())
        self.assertTrue((lib / "THIRD_PARTY_NOTICES.md").is_file())


class ToolCountTest(unittest.TestCase):
    def test_tool_count_is_787_after_bughunter(self):
        # 735 at base 62ae286d; the bughunter port (lane 121) adds
        # 8 tools (scope_check, classify, recon, surface, triage,
        # report, engage, hackerone) — the security organ's 27
        # existing tools are separate registry entries and there is
        # no single action-dispatched "security" tool to extend. 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after B.19 added scene_spawn; 776 after B.20 added cloudcam; 777 after B.22 added show; 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control.
        registry = _registry()
        self.assertEqual(len(registry._tools), 808)
        for name in ("bughunter_scope_check", "bughunter_classify",
                     "bughunter_recon", "bughunter_surface",
                     "bughunter_triage", "bughunter_report",
                     "bughunter_engage", "bughunter_hackerone"):
            self.assertIn(name, registry._tools)

    def test_scope_check_tool(self):
        registry = _registry()
        out = _call(registry, "bughunter_scope_check",
                    url="https://app.example.com/x",
                    in_scope="*.example.com",
                    out_of_scope="evil.example.com")
        self.assertTrue(out["ok"])
        self.assertTrue(out["in_scope"])
        out = _call(registry, "bughunter_scope_check",
                    url="https://evil.example.com/x",
                    in_scope="*.example.com",
                    out_of_scope="evil.example.com")
        self.assertFalse(out["in_scope"])
        self.assertIn("out-of-scope", out["reason"])

    def test_classify_tool(self):
        registry = _registry()
        out = _call(registry, "bughunter_classify",
                    url="https://app.example.com/api/account?id=5")
        self.assertTrue(out["ok"])
        self.assertIn("hunt-idor", out["matches"])
        self.assertEqual(len(out["available_reports"]), 36)

    def test_triage_tool(self):
        registry = _registry()
        out = _call(registry, "bughunter_triage",
                    finding=json.dumps(STRONG_FINDING))
        self.assertTrue(out["ok"])
        self.assertEqual(out["verdict"], "PASS")
        out = _call(registry, "bughunter_triage",
                    finding=json.dumps(WEAK_FINDING))
        self.assertEqual(out["verdict"], "KILL")

    def test_hackerone_tool_refuses_login_without_credentials(self):
        registry = _registry()
        out = _call(registry, "bughunter_hackerone",
                    action="my_programs")
        self.assertFalse(out["ok"])
        self.assertIn("owner's hand", out["error"])


class GateTest(unittest.TestCase):
    def test_gate_passes_strong_finding(self):
        from nomorals.security.bughunter.triage import gate_finding

        result = gate_finding(STRONG_FINDING)
        self.assertEqual(result["verdict"], "PASS")

    def test_gate_kills_weak_finding(self):
        from nomorals.security.bughunter.triage import gate_finding

        result = gate_finding(WEAK_FINDING)
        self.assertEqual(result["verdict"], "KILL")

    def test_gate_applies_to_devon_audit_findings(self):
        # The gate is Devon's reporting standard, not just a bounty
        # filter: an audit_web finding with real evidence passes,
        # one without it is killed before any disclosure draft.
        from nomorals.security.bughunter.triage import gate_finding

        audit_finding = {
            "source": "audit_web", "kind": "idor",
            "title": "unauthenticated endpoint",
            "location": "https://app.example.com/api/account?id=6",
            "detail": "curl https://app.example.com/api/account?id=6 "
                      "HTTP/1.1 — unauthenticated, previously unknown, "
                      "leaked PII for other users",
            "meta": {"severity": "high"},
        }
        self.assertEqual(gate_finding(audit_finding)["verdict"], "PASS")


class ClosureTest(unittest.TestCase):
    def test_untested_item_is_a_gap_not_coverage(self):
        from nomorals.security.bughunter.closure import evaluate

        state = {
            "surface": [
                {"url": "https://app.example.com/api/account?id=5",
                 "param": "id", "vuln_class": "idor"},
                {"url": "https://app.example.com/search?q=x",
                 "param": "q", "vuln_class": "xss"},
            ],
            "tested": ["https://app.example.com/api/account?id=5|id|idor"],
            "coverage_by_item": {
                "https://app.example.com/api/account?id=5|id|idor": {
                    "verdict": "finding"},
            },
        }
        decision = evaluate(state)
        self.assertFalse(decision["clean"])
        self.assertEqual(len(decision["gaps"]), 1)
        self.assertEqual(decision["gaps"][0]["vuln_class"], "xss")
        self.assertEqual(decision["gaps"][0]["state"], "never_tested")

    def test_blocked_is_a_gap_not_coverage(self):
        from nomorals.security.bughunter.closure import evaluate

        state = {
            "surface": [
                {"url": "https://app.example.com/api/account?id=5",
                 "param": "id", "vuln_class": "idor"},
            ],
            "tested": [],
            "coverage_by_item": {
                "https://app.example.com/api/account?id=5|id|idor": {
                    "verdict": "blocked", "note": "no-model"},
            },
        }
        decision = evaluate(state)
        self.assertFalse(decision["clean"])
        self.assertEqual(decision["gaps"][0]["state"], "blocked")


class EngineEndToEndTest(unittest.TestCase):
    def test_engine_end_to_end_on_fixture_target(self):
        from nomorals.security.bughunter.engine import run_engagement

        server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        thread = threading.Thread(target=server.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            seed = f"http://127.0.0.1:{port}/"
            scope = {"name": "fixture", "in_scope": ["127.0.0.1"],
                     "out_of_scope": ["evil.example.com"],
                     "seeds": [seed]}
            with tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp) / "home"
                out = run_engagement(scope, hunt=True,
                                     suggest=_canned_suggest,
                                     max_hunts=50, home=home)
                summary = out["summary"]
                # The canned judge confirms exactly the idor.
                self.assertEqual(summary["confirmed"], 1)
                self.assertEqual(summary["candidates"], 1)
                self.assertGreaterEqual(summary["surface"], 3)
                self.assertEqual(summary["tested"], summary["surface"])
                self.assertTrue(out["coverage"]["clean"])
                report_path = Path(out["report_path"])
                self.assertTrue(report_path.is_file())
                text = report_path.read_text(encoding="utf-8")
                self.assertIn("IDOR", text)
                self.assertIn("confirmed: 1", text)
                # Scope enforcement: no out-of-scope host was ever
                # surfaced, tested, or confirmed.
                eng_dir = Path(out["dir"])
                state = json.loads(
                    (eng_dir / "state.json").read_text(encoding="utf-8"))
                urls = [s["url"] for s in state["surface"]]
                self.assertTrue(all("evil.example.com" not in u
                                    for u in urls))
                self.assertTrue(any(s["vuln_class"] == "idor"
                                    and "/api/account" in s["url"]
                                    for s in state["surface"]))
                self.assertTrue(any(s["vuln_class"] == "xss"
                                    and "/search" in s["url"]
                                    for s in state["surface"]))
                # The secured OpenAPI GET returning 200 unauthenticated
                # is an auth-bypass work unit.
                self.assertTrue(any(s["vuln_class"] == "auth-bypass"
                                    for s in state["surface"]))
                # Evidence pack: the confirmed finding is hash-chained
                # under the home dir with secrets masked at rest.
                chain = home / "evidence" / "fixture" / "chain.jsonl"
                self.assertTrue(chain.is_file())
        finally:
            server.shutdown()
            server.server_close()

    def test_hunt_refuses_out_of_scope_item_in_code(self):
        from nomorals.security.bughunter.engine import Engine
        from nomorals.security.bughunter.scope import Scope

        seen: list[str] = []

        def _recording(prompt: str) -> str:
            seen.append(prompt)
            return "```json\n{\"vulnerable\": false}\n```"

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            scope = Scope(["127.0.0.1"], ["evil.example.com"],
                          ["http://127.0.0.1/"], "refusal")
            eng = Engine(scope, suggest=_recording, home=home)
            eng.eng.add_surface([
                {"url": "http://evil.example.com/x", "param": "",
                 "vuln_class": "sqli"},
                {"url": "http://127.0.0.1/y", "param": "",
                 "vuln_class": "info-leak"},
            ])
            eng.hunt()
            self.assertTrue(all("evil.example.com" not in p
                                for p in seen))
            key = "http://evil.example.com/x||sqli"
            self.assertEqual(
                eng.eng.state["coverage_by_item"][key]["verdict"],
                "out_of_scope")

    def test_no_model_means_blocked_never_clean(self):
        from nomorals.security.bughunter.engine import Engine
        from nomorals.security.bughunter.scope import Scope

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            scope = Scope(["127.0.0.1"], [],
                          ["http://127.0.0.1/"], "nomodel")
            eng = Engine(scope, suggest=None, home=home)
            eng.eng.add_surface([
                {"url": "http://127.0.0.1/y", "param": "id",
                 "vuln_class": "idor"},
            ])
            eng.hunt()
            key = "http://127.0.0.1/y|id|idor"
            self.assertEqual(
                eng.eng.state["coverage_by_item"][key]["verdict"],
                "blocked")
            self.assertEqual(eng.eng.state["confirmed"], [])


if __name__ == "__main__":
    unittest.main()
