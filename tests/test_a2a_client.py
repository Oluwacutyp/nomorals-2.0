"""A2A client tests — Devon talking to other agents over Agent2Agent.

A fake in-process A2A server (ThreadingHTTPServer) speaks both
protocol versions: 1.0 (PascalCase JSON-RPC methods, SCREAMING task
states, kind-less parts, supportedInterfaces card) and 0.3 (legacy
method names, lowercase states, kind-based parts, top-level url
card). A third mode serves the card only at the legacy
``/.well-known/agent.json`` path; a fourth demands a bearer token;
bad-card / bad-JSON / no-JSONRPC-interface modes pin the honest
failures. Registry/config/command-surface pins ride along.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.config import A2AAgentEntry, A2ASettings, Settings  # noqa: E402
from nomorals.core.errors import ConfigError  # noqa: E402
from nomorals.core.policy import CapabilitySet  # noqa: E402
from nomorals.social.chat.control import (  # noqa: E402
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
)
from nomorals.tools import a2a_client as ac  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402


def _skills() -> list[dict[str, Any]]:
    return [{
        "id": "echo-skill",
        "name": "Echo",
        "description": "Echoes text back.",
        "tags": ["echo"],
        "inputModes": ["text/plain"],
        "outputModes": ["text/plain"],
        "examples": ["Echo hello"],
    }]


def _card_v10(base: str) -> dict[str, Any]:
    return {
        "name": "fake-a2a",
        "description": "A fake A2A agent for tests.",
        "version": "0.1",
        "supportedInterfaces": [{
            "url": base + "/rpc",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }],
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": _skills(),
    }


def _card_v03(base: str) -> dict[str, Any]:
    return {
        "name": "fake-a2a",
        "description": "A fake A2A agent for tests.",
        "version": "0.1",
        "url": base + "/rpc",
        "protocolVersion": "0.3",
        "capabilities": {"streaming": True, "pushNotifications": False,
                         "stateTransitionHistory": True},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": _skills(),
    }


def _card_auth(base: str) -> dict[str, Any]:
    card = _card_v10(base)
    card["securitySchemes"] = {
        "bearerAuth": {"httpAuthSecurityScheme": {"scheme": "bearer"}},
    }
    card["securityRequirements"] = [{"schemes": {"bearerAuth": {"list": []}}}]
    return card


def _part_text_v10(text: str) -> dict[str, Any]:
    return {"text": text, "mediaType": "text/plain"}


def _part_text_v03(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def _task_v10(task_id: str, state: str, text: str | None,
              context_id: str = "ctx-1") -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_" + state.upper().replace("-", "_")},
    }
    if text is not None:
        task["artifacts"] = [{
            "artifactId": "a1",
            "parts": [_part_text_v10(text)],
        }]
    return task


def _task_v03(task_id: str, state: str, text: str | None,
              context_id: str = "ctx-1") -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state},
    }
    if state == "input-required":
        task["status"]["message"] = {
            "messageId": "msg-q",
            "role": "agent",
            "parts": [_part_text_v03(text or "")],
        }
    elif text is not None:
        task["artifacts"] = [{
            "artifactId": "a1",
            "parts": [_part_text_v03(text)],
        }]
    return task


def _a2a_handler(mode: str, state: dict[str, Any]):
    """A fake A2A server. Modes: v10 | v03 | legacy-path | auth |
    badcard | badjson | nojsonrpc."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _base(self) -> str:
            port = self.server.server_address[1]
            return f"http://127.0.0.1:{port}"

        def _send_json(self, payload: Any, code: int = 200,
                       ctype: str = "application/json") -> None:
            raw = payload if isinstance(payload, bytes) else json.dumps(
                payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802 - stdlib name
            if self.path == "/.well-known/agent-card.json":
                if mode == "badcard":
                    self._send_json(b"<html>not json</html>", ctype="text/html")
                    return
                if mode == "legacy-path":
                    self.send_response(404)
                    self.end_headers()
                    return
                if mode == "v03":
                    self._send_json(_card_v03(self._base()))
                elif mode == "auth":
                    self._send_json(_card_auth(self._base()))
                elif mode == "nojsonrpc":
                    card = _card_v10(self._base())
                    card["supportedInterfaces"] = [{
                        "url": self._base() + "/grpc",
                        "protocolBinding": "GRPC",
                        "protocolVersion": "1.0",
                    }]
                    self._send_json(card)
                else:
                    self._send_json(_card_v10(self._base()))
                return
            if self.path == "/.well-known/agent.json":
                if mode == "legacy-path":
                    self._send_json(_card_v03(self._base()))
                    return
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802 - stdlib name
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if mode == "badjson":
                self._send_json(b"{not json")
                return
            try:
                msg = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"jsonrpc": "2.0", "id": None, "error": {
                    "code": -32700, "message": "parse error"}})
                return
            if mode == "auth":
                auth = self.headers.get("Authorization") or ""
                if auth != "Bearer test-token-123":
                    self._send_json({"jsonrpc": "2.0", "id": msg.get("id"),
                                     "error": {"code": -32001,
                                               "message": "unauthorized"}})
                    return
            method = msg.get("method")
            mid = msg.get("id")
            params = msg.get("params") or {}
            result = self._dispatch(mode, state, method, params)
            if isinstance(result, tuple):
                # (status, payload) — an error envelope.
                code, payload = result
                self._send_json(payload, code=code)
                return
            if isinstance(result, bytes):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
                return
            self._send_json({"jsonrpc": "2.0", "id": mid, "result": result})

        def _dispatch(self, mode: str, state: dict[str, Any],
                      method: str, params: dict[str, Any]) -> Any:
            legacy = mode in ("v03", "legacy-path")
            send = "message/send" if legacy else "SendMessage"
            stream = "message/stream" if legacy else "SendStreamingMessage"
            get = "tasks/get" if legacy else "GetTask"
            list_m = "tasks/list" if legacy else "ListTasks"
            cancel = "tasks/cancel" if legacy else "CancelTask"
            task_fn = _task_v03 if legacy else _task_v10
            if method == send:
                message = params.get("message") or {}
                parts = message.get("parts") or []
                text = ""
                for part in parts:
                    if isinstance(part, dict):
                        text += str(part.get("text") or "")
                state["seq"] += 1
                task_id = f"task-{state['seq']}"
                ctx = message.get("contextId") or "ctx-1"
                if text.startswith("fail"):
                    return (200, {"jsonrpc": "2.0", "id": "x", "error": {
                        "code": -32602, "message": "boom"}})
                if text.startswith("slow"):
                    task = task_fn(task_id, "working", None, ctx)
                    state["tasks"][task_id] = {"task": task, "polls": 0}
                    return task
                if text.startswith("need-more"):
                    task = task_fn(task_id, "input-required",
                                   "Which city?", ctx)
                    state["tasks"][task_id] = {"task": task, "polls": 0}
                    return task
                task = task_fn(task_id, "completed", f"echo: {text}", ctx)
                state["tasks"][task_id] = {"task": task, "polls": 0}
                return task
            if method == stream:
                state["seq"] += 1
                task_id = f"task-{state['seq']}"
                if legacy:
                    frames = [
                        {"task": _task_v03(task_id, "submitted", None)},
                        {"statusUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "status": {"state": "working"}}},
                        {"artifactUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "artifact": {"artifactId": "a1", "parts": [
                                _part_text_v03("streamed hello")]},
                            "lastChunk": True}},
                        {"statusUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "status": {"state": "completed"}}},
                    ]
                else:
                    frames = [
                        {"task": _task_v10(task_id, "submitted", None)},
                        {"statusUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "status": {"state": "TASK_STATE_WORKING"}}},
                        {"artifactUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "artifact": {"artifactId": "a1", "parts": [
                                _part_text_v10("streamed hello")]},
                            "lastChunk": True}},
                        {"statusUpdate": {
                            "taskId": task_id, "contextId": "ctx-1",
                            "status": {"state": "TASK_STATE_COMPLETED"}}},
                    ]
                out = b""
                for frame in frames:
                    raw = json.dumps(
                        {"jsonrpc": "2.0", "id": "x",
                         "result": frame}).encode("utf-8")
                    out += b"data: " + raw + b"\n\n"
                return out
            if method == get:
                task_id = params.get("id") or params.get("taskId")
                entry = state["tasks"].get(task_id)
                if entry is None:
                    return (200, {"jsonrpc": "2.0", "id": "x", "error": {
                        "code": -32602, "message": "unknown task"}})
                task = entry["task"]
                if task["status"]["state"] in (
                        "working", "TASK_STATE_WORKING"):
                    entry["polls"] += 1
                    if entry["polls"] >= 2:
                        task = task_fn(task_id, "completed", "slow done",
                                       task.get("contextId") or "ctx-1")
                        entry["task"] = task
                return task
            if method == list_m:
                return {"tasks": [e["task"]
                                  for e in state["tasks"].values()]}
            if method == cancel:
                task_id = params.get("id") or params.get("taskId")
                entry = state["tasks"].get(task_id)
                if entry is None:
                    return (200, {"jsonrpc": "2.0", "id": "x", "error": {
                        "code": -32602, "message": "unknown task"}})
                task = task_fn(task_id, "canceled", None,
                               entry["task"].get("contextId") or "ctx-1")
                entry["task"] = task
                return task
            return (200, {"jsonrpc": "2.0", "id": "x", "error": {
                "code": -32601, "message": f"method not found: {method}"}})

    return Handler


def _serve(mode: str) -> tuple[ThreadingHTTPServer, dict[str, Any]]:
    state: dict[str, Any] = {"tasks": {}, "seq": 0}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _a2a_handler(mode, state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def _base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _settings_with_fake(url: str, *, name: str = "fake",
                        enabled: bool = True, timeout_s: float = 30.0,
                        api_key_env: str = "") -> Settings:
    settings = Settings()
    settings.a2a.enabled = True
    settings.a2a.agents[name] = A2AAgentEntry(
        url=url, enabled=enabled, timeout_s=timeout_s,
        api_key_env=api_key_env)
    return settings

class A2ADefaultsTests(unittest.TestCase):
    def test_master_switch_off_by_default(self):
        settings = Settings()
        self.assertFalse(settings.a2a.enabled)

    def test_no_default_agents_shipped(self):
        # No public A2A endpoint is stable enough to ship as a
        # default; the registry is built from config only.
        self.assertEqual(ac.DEFAULT_A2A_AGENTS, {})
        self.assertEqual(ac.effective_a2a_agents(None), {})

    def test_config_entry_adds_an_agent(self):
        settings = Settings()
        settings.a2a.enabled = True
        settings.a2a.agents["peer"] = {
            "url": "http://127.0.0.1:9", "enabled": True}
        agents = ac.effective_a2a_agents(settings)
        self.assertIn("peer", agents)
        self.assertTrue(agents["peer"].enabled)
        self.assertEqual(agents["peer"].url, "http://127.0.0.1:9")

    def test_config_unknown_agent_key_is_rejected(self):
        with self.assertRaises(ConfigError):
            A2ASettings(agents={"peer": {"url": "http://x", "nope": 1}})

    def test_config_agent_without_url_is_rejected(self):
        with self.assertRaises(ac.A2AClientError):
            ac.A2AAgentConfig(name="peer", url="").validate()

    def test_from_dict_rejects_unknown_keys(self):
        with self.assertRaises(ac.A2AClientError):
            ac.A2AAgentConfig.from_dict("peer", {"url": "http://x",
                                                 "bogus": True})


class A2ACardTests(unittest.TestCase):
    def test_fetch_v10_card(self):
        server, _state = _serve("v10")
        try:
            card = ac.fetch_agent_card(_base_url(server), timeout_s=5.0)
            self.assertEqual(card.name, "fake-a2a")
            self.assertEqual(card.protocol_version, "1.0")
            self.assertTrue(card.streaming)
            self.assertEqual([s["id"] for s in card.skills], ["echo-skill"])
            self.assertTrue(card.endpoint_url.endswith("/rpc"))
        finally:
            server.shutdown()
            server.server_close()

    def test_fetch_v03_card(self):
        server, _state = _serve("v03")
        try:
            card = ac.fetch_agent_card(_base_url(server), timeout_s=5.0)
            self.assertEqual(card.protocol_version, "0.3")
            self.assertEqual([s["id"] for s in card.skills], ["echo-skill"])
        finally:
            server.shutdown()
            server.server_close()

    def test_legacy_card_path_fallback(self):
        server, _state = _serve("legacy-path")
        try:
            card = ac.fetch_agent_card(_base_url(server), timeout_s=5.0)
            self.assertEqual(card.protocol_version, "0.3")
            self.assertEqual(card.card_path, "/.well-known/agent.json")
        finally:
            server.shutdown()
            server.server_close()

    def test_malformed_card_is_an_error(self):
        server, _state = _serve("badcard")
        try:
            with self.assertRaises(ac.A2AClientError):
                ac.fetch_agent_card(_base_url(server), timeout_s=5.0)
        finally:
            server.shutdown()
            server.server_close()

    def test_unreachable_card_is_an_error(self):
        with self.assertRaises(ac.A2AClientError):
            ac.fetch_agent_card("http://127.0.0.1:1", timeout_s=2.0)

    def test_card_without_jsonrpc_interface_is_an_error(self):
        server, _state = _serve("nojsonrpc")
        try:
            with self.assertRaises(ac.A2AClientError) as caught:
                ac.fetch_agent_card(_base_url(server), timeout_s=5.0)
            self.assertIn("JSONRPC", str(caught.exception))
        finally:
            server.shutdown()
            server.server_close()


class A2AClientV10Tests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("v10")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _client(self) -> ac.A2AClient:
        cfg = ac.A2AAgentConfig(name="fake", url=_base_url(self.server),
                                enabled=True)
        return ac.A2AClient(cfg, timeout_s=5.0)

    def test_send_completes_with_artifact_text(self):
        client = self._client()
        try:
            result = client.send_message("hello")
            self.assertEqual(ac.task_state(result), "completed")
            self.assertEqual(ac.task_text(result), "echo: hello")
        finally:
            client.close()

    def test_slow_task_waits_to_terminal(self):
        client = self._client()
        try:
            result = client.send_message("slow please")
            self.assertEqual(ac.task_state(result), "working")
            final = client.wait_for_task(result["id"], timeout_s=10.0,
                                         poll_s=0.05)
            self.assertEqual(ac.task_state(final), "completed")
            self.assertEqual(ac.task_text(final), "slow done")
        finally:
            client.close()

    def test_send_and_wait_one_call(self):
        client = self._client()
        try:
            out = client.send_and_wait("slow please", wait_s=10.0,
                                       poll_s=0.05)
            self.assertEqual(out["state"], "completed")
            self.assertEqual(out["text"], "slow done")
        finally:
            client.close()

    def test_input_required_is_surfaced_and_answerable(self):
        client = self._client()
        try:
            result = client.send_message("need-more info")
            self.assertEqual(ac.task_state(result), "input-required")
            self.assertIn("Which city?", ac.task_text(result))
            reply = client.send_message(
                "Lagos", task_id=result["id"],
                context_id=result.get("contextId"))
            self.assertEqual(ac.task_state(reply), "completed")
            self.assertEqual(ac.task_text(reply), "echo: Lagos")
        finally:
            client.close()

    def test_streaming_events_end_completed(self):
        client = self._client()
        try:
            events = list(client.send_message_stream("hello"))
            kinds = [next(iter(e)) for e in events]
            self.assertEqual(kinds, ["task", "statusUpdate",
                                     "artifactUpdate", "statusUpdate"])
            self.assertEqual(
                events[-1]["statusUpdate"]["status"]["state"],
                "TASK_STATE_COMPLETED")
        finally:
            client.close()

    def test_get_list_cancel_task(self):
        client = self._client()
        try:
            sent = client.send_message("hello")
            got = client.get_task(sent["id"])
            self.assertEqual(got["id"], sent["id"])
            listed = client.list_tasks()
            self.assertEqual([t["id"] for t in listed], [sent["id"]])
            cancelled = client.cancel_task(sent["id"])
            self.assertEqual(ac.task_state(cancelled), "canceled")
        finally:
            client.close()

    def test_unknown_skill_is_refused_before_send(self):
        client = self._client()
        try:
            with self.assertRaises(ac.A2AClientError) as caught:
                client.send_message("hello", skill="no-such-skill")
            self.assertIn("echo-skill", str(caught.exception))
        finally:
            client.close()

    def test_server_jsonrpc_error_is_raised(self):
        client = self._client()
        try:
            with self.assertRaises(ac.A2AClientError) as caught:
                client.send_message("fail now")
            self.assertIn("boom", str(caught.exception))
        finally:
            client.close()

    def test_unknown_method_surfaces_server_error(self):
        client = self._client()
        try:
            client.connect()
            with self.assertRaises(ac.A2AClientError):
                client._rpc("Nope", {})
        finally:
            client.close()


class A2AClientV03Tests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("v03")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _client(self) -> ac.A2AClient:
        cfg = ac.A2AAgentConfig(name="fake", url=_base_url(self.server),
                                enabled=True)
        return ac.A2AClient(cfg, timeout_s=5.0)

    def test_send_completes_over_legacy_methods(self):
        client = self._client()
        try:
            result = client.send_message("hello")
            self.assertEqual(client.protocol_version, "0.3")
            self.assertEqual(ac.task_state(result), "completed")
            self.assertEqual(ac.task_text(result), "echo: hello")
        finally:
            client.close()

    def test_slow_task_waits_over_legacy_poll(self):
        client = self._client()
        try:
            out = client.send_and_wait("slow please", wait_s=10.0,
                                       poll_s=0.05)
            self.assertEqual(out["state"], "completed")
            self.assertEqual(out["text"], "slow done")
        finally:
            client.close()

    def test_streaming_over_legacy_methods(self):
        client = self._client()
        try:
            events = list(client.send_message_stream("hello"))
            kinds = [next(iter(e)) for e in events]
            self.assertEqual(kinds, ["task", "statusUpdate",
                                     "artifactUpdate", "statusUpdate"])
            texts = [ac.task_text({"artifacts": [e["artifactUpdate"]["artifact"]]})
                     if "artifactUpdate" in e else ""
                     for e in events]
            self.assertIn("streamed hello", texts)
        finally:
            client.close()


class A2AAuthTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("auth")
        os.environ["A2A_FAKE_TOKEN"] = "test-token-123"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.environ.pop("A2A_FAKE_TOKEN", None)

    def test_auth_required_without_token_is_honest(self):
        cfg = ac.A2AAgentConfig(name="fake", url=_base_url(self.server),
                                enabled=True)
        client = ac.A2AClient(cfg, timeout_s=5.0)
        try:
            with self.assertRaises(ac.A2AClientError) as caught:
                client.send_message("hello")
            self.assertIn("api_key_env", str(caught.exception))
        finally:
            client.close()

    def test_auth_with_token_from_env_works(self):
        cfg = ac.A2AAgentConfig(name="fake", url=_base_url(self.server),
                                enabled=True,
                                api_key_env="A2A_FAKE_TOKEN")
        client = ac.A2AClient(cfg, timeout_s=5.0)
        try:
            result = client.send_message("hello")
            self.assertEqual(ac.task_text(result), "echo: hello")
        finally:
            client.close()


class A2AManagerTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("v10")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_refuses_when_master_off(self):
        settings = Settings()
        settings.a2a.agents["fake"] = A2AAgentEntry(
            url=_base_url(self.server), enabled=True)
        manager = ac.A2AClientManager.from_settings(settings)
        with self.assertRaises(ac.A2AClientError) as caught:
            manager.connect("fake")
        self.assertIn("a2a.enabled", str(caught.exception))

    def test_refuses_disabled_agent(self):
        settings = _settings_with_fake(_base_url(self.server),
                                       enabled=False)
        manager = ac.A2AClientManager.from_settings(settings)
        with self.assertRaises(ac.A2AClientError) as caught:
            manager.connect("fake")
        self.assertIn("disabled", str(caught.exception))

    def test_discover_send_and_status(self):
        settings = _settings_with_fake(_base_url(self.server))
        manager = ac.A2AClientManager.from_settings(settings)
        try:
            card = manager.discover("fake")
            self.assertEqual(card["name"], "fake-a2a")
            self.assertEqual(card["protocol_version"], "1.0")
            self.assertEqual([s["id"] for s in card["skills"]],
                             ["echo-skill"])
            out = manager.send_and_wait("fake", "hello", wait_s=10.0)
            self.assertEqual(out["text"], "echo: hello")
            status = manager.status()
            self.assertTrue(status["enabled"])
            self.assertEqual(status["connected"], 1)
        finally:
            manager.close()

    def test_status_makes_no_connections(self):
        settings = _settings_with_fake(_base_url(self.server))
        manager = ac.A2AClientManager.from_settings(settings)
        status = manager.status()
        self.assertEqual(status["connected"], 0)
        self.assertFalse(status["agents"]["fake"]["connected"])


class A2ARegistryTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("v10")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _registry(self, settings: Settings) -> ToolRegistry:
        context = SimpleNamespace(settings=settings)
        registry = ToolRegistry(context=context)
        registry.register_builtins()
        return registry

    def test_tools_registered_with_net_out(self):
        registry = self._registry(Settings())
        for name in ("a2a_discover", "a2a_send", "a2a_task",
                     "a2a_cancel", "a2a_status"):
            spec = registry.get(name)
            self.assertIsNotNone(spec, name)
            self.assertEqual(str(spec.capability), "net.out")

    def test_discover_through_registry(self):
        settings = _settings_with_fake(_base_url(self.server))
        registry = self._registry(settings)
        outcome = registry.call(
            "a2a_discover", actor="test", capabilities=CapabilitySet.all(),
            agent="fake")
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertTrue(value["ok"])
        self.assertEqual(value["agents"]["fake"]["name"], "fake-a2a")

    def test_send_through_registry(self):
        settings = _settings_with_fake(_base_url(self.server))
        registry = self._registry(settings)
        outcome = registry.call(
            "a2a_send", actor="test", capabilities=CapabilitySet.all(),
            agent="fake", text="hello", wait_s=10.0)
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertTrue(value["ok"], value)
        self.assertEqual(value["text"], "echo: hello")
        self.assertEqual(value["state"], "completed")

    def test_send_refuses_disabled_agent(self):
        settings = _settings_with_fake(_base_url(self.server),
                                       enabled=False)
        registry = self._registry(settings)
        outcome = registry.call(
            "a2a_send", actor="test", capabilities=CapabilitySet.all(),
            agent="fake", text="hello")
        value = outcome.unwrap()
        self.assertFalse(value["ok"])
        self.assertIn("disabled", value["error"])

    def test_status_tool_makes_no_connections(self):
        settings = _settings_with_fake(_base_url(self.server))
        registry = self._registry(settings)
        outcome = registry.call(
            "a2a_status", actor="test", capabilities=CapabilitySet.all())
        value = outcome.unwrap()
        self.assertTrue(value["ok"])
        self.assertEqual(value["status"]["connected"], 0)


class A2ACommandTests(unittest.TestCase):
    def setUp(self):
        self.server, self.state = _serve("v10")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_command_registered_with_help(self):
        self.assertIn("a2a", CONTROL_COMMANDS)
        self.assertEqual(CONTROL_COMMANDS["a2a"], (0, None))
        details = COMMAND_DETAILS["a2a"]
        for key in ("what", "usage", "example", "related"):
            self.assertTrue(str(details.get(key) or "").strip(), key)
        self.assertIn("a2a", LIST_ONELINERS)
        self.assertTrue(any("a2a" in g for _, g in LIST_GROUPS),
                        "a2a missing from LIST_GROUPS")

    def test_command_status_and_list(self):
        settings = _settings_with_fake(_base_url(self.server))
        out = ac.control_a2a("", settings)
        self.assertIn("A2A client: on", out)
        self.assertIn("fake", out)
        out = ac.control_a2a("list", settings)
        self.assertIn("fake", out)
        self.assertIn("http", out)

    def test_command_send_and_task(self):
        settings = _settings_with_fake(_base_url(self.server))
        out = ac.control_a2a("send fake hello", settings)
        self.assertIn("echo: hello", out)
        self.assertIn("task-1", out)


if __name__ == "__main__":
    unittest.main()
