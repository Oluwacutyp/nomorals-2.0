"""Wave-40 coverage: the seven-item feature build.

* scheduler — spec parsing, add/tick/run_now, one-shot self-disable,
  tool/command payloads, CRUD (hermetic temp-db context)
* notifier — 600s dedupe window + critical bypass
* connectors — hermetic registry calls, error → ToolError, unknown names
* audio — engine detection with no binaries present, wav audio_info
* vision — the screen-reader prompt and its structured output
* swarm — decomposition heuristics, model decomposition, result shape
* registry — all new tools present with their capabilities
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
import unittest.mock
import wave
from pathlib import Path

from tests.test_partner_runtime import _make_context

from nomorals.agents.notifier import Notifier
from nomorals.agents.scheduler import Scheduler, parse_schedule_spec
from nomorals.core.errors import ToolError
from nomorals.core.policy import Capability
from nomorals.llm.base import LLMResponse
from nomorals.tools import audio as audio_module
from nomorals.tools import connectors as connectors_module
from nomorals.tools import vision as vision_module
from nomorals.tools.registry import ToolRegistry


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeOutcome:
    def __init__(self, ok: bool, value=None, error=None) -> None:
        self.ok = ok
        self.value = value
        self.error = error


class FakeError:
    def __init__(self, message: str) -> None:
        self.message = message


class FakeTools:
    """Stands in for the tool registry on contexts built without tools."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, *, capabilities=None, **kwargs):  # noqa: ANN001
        self.calls.append((name, kwargs))
        if name == "shell_run":
            return FakeOutcome(True, {"exit_code": 0, "stdout": "fake-out", "stderr": ""})
        if name == "mood":
            return FakeOutcome(True, {"mood": "calm"})
        if name == "boom":
            return FakeOutcome(False, None, FakeError("exploded on purpose"))
        return FakeOutcome(True, {"echo": kwargs})


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200
        self.ok = True
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    """Just enough of core.http.HttpClient for the connector fns."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self._nm_api_settings = None  # the connector settings slot register() uses

    def get(self, url: str, **kwargs):  # noqa: ANN001
        self.requests.append(("GET", url))
        return FakeHttpResponse(self._payload_for(url))

    def post_json(self, url: str, body, **kwargs):  # noqa: ANN001
        self.requests.append(("POST", url))
        return FakeHttpResponse({"ok": True, "echo": body})

    def _payload_for(self, url: str) -> dict:
        if "geocoding-api.open-meteo.com" in url:
            return {"results": [{"latitude": 6.5, "longitude": 3.4, "name": "Lagos"}]}
        if "api.open-meteo.com" in url:
            return {"current": {"temperature_2m": 31.2, "relative_humidity_2m": 78,
                                "apparent_temperature": 33.0, "wind_speed_10m": 9.5,
                                "weather_code": 2}}
        if "open.er-api" in url:
            return {"rates": {"USD": 1.0, "NGN": 1500.0}}
        return {"ok": True}


# ── schedule spec parsing ────────────────────────────────────────────────────


class ScheduleSpecParseTest(unittest.TestCase):
    def test_at_iso(self) -> None:
        kind, detail = parse_schedule_spec("at 2026-12-25 09:00")
        self.assertEqual(kind, "at")
        self.assertGreater(detail, 0)

    def test_at_bare_timestamp(self) -> None:
        kind, _ = parse_schedule_spec("2026-12-25T09:00")
        self.assertEqual(kind, "at")

    def test_every_prefix(self) -> None:
        kind, detail = parse_schedule_spec("every 30m")
        self.assertEqual((kind, detail), ("every", 1800.0))

    def test_every_bare_interval(self) -> None:
        kind, detail = parse_schedule_spec("45s")
        self.assertEqual((kind, detail), ("every", 45.0))

    def test_every_hours(self) -> None:
        kind, detail = parse_schedule_spec("2h")
        self.assertEqual((kind, detail), ("every", 7200.0))

    def test_daily_prefix(self) -> None:
        kind, detail = parse_schedule_spec("daily 22:00")
        self.assertEqual((kind, detail), ("daily", "22:00"))

    def test_daily_bare_hhmm(self) -> None:
        kind, detail = parse_schedule_spec("22:00")
        self.assertEqual((kind, detail), ("daily", "22:00"))

    def test_daily_pads_single_digits(self) -> None:
        _, detail = parse_schedule_spec("9:5")
        self.assertEqual(detail, "09:05")

    def test_rejects_too_small_interval(self) -> None:
        with self.assertRaises(ValueError):
            parse_schedule_spec("every 5s")

    def test_rejects_bad_time(self) -> None:
        with self.assertRaises(ValueError):
            parse_schedule_spec("25:99")

    def test_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            parse_schedule_spec("whenever")
        with self.assertRaises(ValueError):
            parse_schedule_spec("")


# ── scheduler behavior ───────────────────────────────────────────────────────


class SchedulerBehaviorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.context.tools = FakeTools()
        self.sched = Scheduler(self.context, tick_seconds=60.0)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _force_due(self, job_id: str) -> None:
        with self.context.db.transaction():
            self.context.db.execute(
                "UPDATE schedule_jobs SET next_run = 1 WHERE id = ?", (job_id,)
            )

    def test_message_job_fires_and_self_disables(self) -> None:
        job = self.sched.add("onetime", "at 2099-12-25 23:59",
                             "message", {"text": "goodnight"})
        self._force_due(job["id"])
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])
        self.assertIn("goodnight", outcomes[0]["result"])
        row = self.context.db.query(
            "SELECT enabled, next_run FROM schedule_jobs WHERE id = ?", (job["id"],)
        )[0]
        self.assertEqual(row["enabled"], 0)  # one-shot disabled itself
        self.assertIsNone(row["next_run"])

    def test_every_job_reschedules(self) -> None:
        import time

        job = self.sched.add("beat", "every 60m", "message", {"text": "tick"})
        self._force_due(job["id"])
        before = time.time()
        outcomes = self.sched.tick()
        self.assertEqual(len(outcomes), 1)
        row = self.context.db.query(
            "SELECT enabled, next_run FROM schedule_jobs WHERE id = ?", (job["id"],)
        )[0]
        self.assertEqual(row["enabled"], 1)
        self.assertGreaterEqual(row["next_run"], before)  # scheduled again

    def test_tool_payload_calls_the_registry(self) -> None:
        self.sched.add("t", "at 2099-12-31 23:00", "tool", {"tool": "mood", "args": {}})
        outcome = self.sched.run_now("t")
        self.assertTrue(outcome["ok"])
        self.assertIn("mood", outcome["result"])
        self.assertEqual(self.context.tools.calls[0][0], "mood")

    def test_tool_payload_failure_is_reported_not_fatal(self) -> None:
        self.sched.add("t2", "at 2099-12-31 23:00", "tool", {"tool": "boom", "args": {}})
        outcome = self.sched.run_now("t2")
        self.assertFalse(outcome["ok"])
        self.assertIn("exploded", outcome["result"])

    def test_command_payload_runs_shell(self) -> None:
        self.sched.add("c", "at 2099-12-31 23:00", "command", {"command": "echo hi"})
        outcome = self.sched.run_now("c")
        self.assertTrue(outcome["ok"])
        self.assertIn("exit=0", outcome["result"])

    def test_crud(self) -> None:
        job = self.sched.add("crud", "every 1h", "message", {"text": "x"})
        jobs = self.sched.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "crud")

        off = self.sched.set_enabled("crud", False)
        self.assertFalse(off["enabled"])
        self._force_due(job["id"])
        self.sched.tick()  # disabled job must not fire
        row = self.context.db.query(
            "SELECT last_run FROM schedule_jobs WHERE id = ?", (job["id"],)
        )[0]
        self.assertIsNone(row["last_run"])

        on = self.sched.set_enabled("crud", True)
        self.assertTrue(on["enabled"])

        self.assertTrue(self.sched.remove("crud"))
        self.assertFalse(self.sched.remove("crud"))
        self.assertEqual(self.sched.list_jobs(), [])

    def test_rejects_past_one_shot(self) -> None:
        with self.assertRaises(ValueError):
            self.sched.add("old", "at 2000-01-01 00:00", "message", {"text": "x"})

    def test_outcome_published_to_notifications(self) -> None:
        self.sched.add("alerty", "at 2099-12-31 23:00", "message", {"text": "look at this"})
        self.sched.run_now("alerty")
        rows = self.context.db.query(
            "SELECT kind, title FROM notifications WHERE kind = 'schedule'"
        )
        self.assertTrue(any("alerty" in str(r["title"]) for r in rows))


# ── notifier dedupe ──────────────────────────────────────────────────────────


class NotifierDedupeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.notifier = Notifier(self.context, gateway=None)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def count(self, title: str) -> int:
        return self.context.db.scalar(
            "SELECT COUNT(*) AS n FROM notifications WHERE title = ?", (title,)
        ) or 0

    def test_second_identical_alert_is_deduped(self) -> None:
        first = self.notifier.publish("schedule", "job1 done", "sent: hi")
        self.assertFalse(first.get("deduped"))
        second = self.notifier.publish("schedule", "job1 done", "sent: hi")
        self.assertTrue(second.get("deduped"))
        self.assertEqual(self.count("job1 done"), 1)

    def test_different_title_not_deduped(self) -> None:
        self.notifier.publish("schedule", "title A", "")
        out = self.notifier.publish("schedule", "title B", "")
        self.assertFalse(out.get("deduped"))
        self.assertEqual(self.count("title B"), 1)

    def test_critical_bypasses_dedupe(self) -> None:
        self.notifier.publish("security", "CRIT: port open", "port 22 open")
        out = self.notifier.publish("security", "CRIT: port open", "still open",
                                    critical=True)
        self.assertFalse(out.get("deduped"))
        self.assertEqual(self.count("CRIT: port open"), 2)


# ── connectors ───────────────────────────────────────────────────────────────


class ConnectorFrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.http = FakeHttpClient()
        self.registry = connectors_module.ConnectorRegistry()
        self.builtins = connectors_module.connectors

    def test_builtins_registered(self) -> None:
        names = connectors_module.connectors.names()
        for name in ("weather", "fx", "ip_info", "github", "wikipedia", "http_api"):
            self.assertIn(name, names)

    def test_call_returns_connector_data(self) -> None:
        result = self.builtins.call("fx", {"from": "USD", "to": "NGN"}, self.http)
        self.assertEqual(result["rates"]["NGN"], 1500.0)

    def test_error_becomes_tool_error(self) -> None:
        class BadHttp:
            def get(self, url, **kw):
                raise RuntimeError("no network")

        with self.assertRaises(ToolError):
            self.builtins.call("fx", {"from": "USD"}, BadHttp())

    def test_unknown_connector_raises(self) -> None:
        with self.assertRaises(ToolError):
            self.builtins.call("does_not_exist", {}, self.http)

    def test_custom_decorator_registers(self) -> None:
        @self.registry.register_connector("joke", description="a joke",
                                       params={"category": "str (optional)"})
        def _joke(params: dict, http) -> dict:
            return {"joke": "why did the joke go to therapy?"}

        result = self.registry.call("joke", {}, self.http)
        self.assertIn("therapy", result["joke"])

    def test_http_api_get(self) -> None:
        result = self.builtins.call("http_api",
                                    {"url": "https://example.com/latest"}, self.http)
        self.assertEqual(result["status"], 200)

    def test_weather_geocode_then_forecast(self) -> None:
        result = self.builtins.call("weather", {"city": "Lagos"}, self.http)
        self.assertEqual(result["temperature_c"], 31.2)
        self.assertEqual(result["location"], "Lagos")
        self.assertEqual(len(self.http.requests), 2)  # geocode + forecast


# ── audio ────────────────────────────────────────────────────────────────────


class AudioToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _no_engines(self):
        """shutil.which finds nothing AND edge_tts is not importable."""
        return (
            unittest.mock.patch("shutil.which", return_value=None),
            unittest.mock.patch.dict(sys.modules, {"edge_tts": None}),
        )

    def test_detect_with_no_engines(self) -> None:
        patch_which, patch_module = self._no_engines()
        with patch_which, patch_module:
            self.assertEqual(audio_module.detect_tts_engines(), [])

    def test_audio_info_on_real_wav(self) -> None:
        wav_path = self.tmp_path / "tone.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(b"\x00\x00" * 8000)  # 1 second of silence
        info = audio_module.audio_info(wav_path)
        self.assertEqual(info["format"], "wav")
        self.assertEqual(info["channels"], 1)
        self.assertAlmostEqual(info["duration"], 1.0, delta=0.05)

    def test_speak_without_engines_fails_cleanly(self) -> None:
        patch_which, patch_module = self._no_engines()
        with patch_which, patch_module:
            with self.assertRaises(ToolError) as ctx:
                audio_module.tts("hello world", out_dir=self.tmp_path)
        self.assertIn("no TTS engine", str(ctx.exception))

    def test_transcribe_missing_file(self) -> None:
        with self.assertRaises(ToolError):
            audio_module.stt(self.tmp_path / "nope.ogg")


# ── vision screen reader ─────────────────────────────────────────────────────


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class VisionScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        workspace = Path(self.context.settings.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        self.img = workspace / "shot.png"
        self.img.write_bytes(_PNG_1X1)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _screen_result(self, prompt_capture: dict, focus: str = "") -> dict:
        reg = ToolRegistry()
        reg.context = self.context  # captured by the register() closures
        vision_module.register(reg)
        with unittest.mock.patch.object(
            vision_module, "describe",
            lambda ctx, data, prompt, *, cache=None, ocr=False, mime="image/png": (
                prompt_capture.update(prompt=prompt) or
                {"sha256": "x", "width": 1, "height": 1, "format": "png",
                 "description": "ok", "provider": "fake", "prompt": prompt,
                 "seconds": 0.0, "ocr_text": None}
            ),
        ):
            spec = reg.get("vision_screen")
            assert spec is not None
            return spec.fn(path=str(self.img), focus=focus)

    def test_screen_prompt_structure(self) -> None:
        captured: dict[str, str] = {}
        result = self._screen_result(captured, focus="login form")
        prompt = captured["prompt"]
        for marker in ("TEXT:", "ELEMENTS:", "STATE:", "NOTE:", "FOCUS:"):
            self.assertIn(marker, prompt)
        self.assertIn("login form", prompt)
        self.assertEqual(result["analysis"], "screen")

    def test_screen_result_flagged(self) -> None:
        result = self._screen_result({})
        self.assertEqual(result["analysis"], "screen")
        self.assertEqual(result["format"], "png")


# ── swarm ───────────────────────────────────────────────────────────────────


class SwarmTest(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.swarm import SwarmAgent

        self.context, self.tmp = _make_context()
        self.context.router = None
        self.context.tools = FakeTools()
        self.swarm = SwarmAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_heuristic_conjunction(self) -> None:
        subs = self.swarm._heuristic_decompose(
            "check the logs and run the tests", 2)
        self.assertEqual(len(subs), 2)
        subs, planner = self.swarm.decompose("check the logs and run the tests", 2)
        self.assertEqual(planner, "heuristic")

    def test_heuristic_numbered(self) -> None:
        goal = "do these:\n1. read the config\n2. check the database"
        subs = self.swarm._heuristic_decompose(goal, 3)
        self.assertEqual(len(subs), 2)

    def test_heuristic_perspectives_fallback(self) -> None:
        subs = self.swarm._heuristic_decompose("launch the product", 3)
        self.assertEqual(len(subs), 3)
        self.assertIn("evidence", subs[0])

    def test_model_decompose_uses_llm(self) -> None:
        class LLMRouter:
            def stats_snapshot(self):
                return {"active": "groq"}

            def chat(self, messages, params=None, **kw):
                return LLMResponse(text='[ "part one", "part two" ]', model="fake")

        self.context.router = LLMRouter()
        subs, planner = self.swarm.decompose("do A and B", 3)
        self.assertEqual(planner, "llm")
        self.assertEqual(subs, ["part one", "part two"])

    def test_model_decompose_falls_back_on_bad_json(self) -> None:
        class BadRouter:
            def stats_snapshot(self):
                return {"active": "groq"}

            def chat(self, messages, params=None, **kw):
                return LLMResponse(text="i refuse to use json", model="fake")

        self.context.router = BadRouter()
        subs, planner = self.swarm.decompose("launch the product", 2)
        self.assertEqual(planner, "heuristic")
        self.assertEqual(len(subs), 2)

    def test_no_model_is_heuristic(self) -> None:
        self.context.router = None
        subs, planner = self.swarm.decompose("launch the product", 2)
        self.assertEqual(planner, "heuristic")


# ── runtime wiring: the new /commands end to end ─────────────────────────────


class RuntimeCommandsTest(unittest.TestCase):
    """The new control commands through the real PartnerRuntime dispatch."""

    def setUp(self) -> None:
        from tests.test_partner_runtime import FakeAdapter, FakeRouter
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.social.chat.base import ChatKind, ChatRef
        from nomorals.social.chat.gateway import ChatGateway

        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()
        self.registry = ToolRegistry()
        self.registry.context = self.context
        self.registry.register_builtins()
        self.context.tools = self.registry
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.chat = ChatRef(platform="local", chat_id="console",
                            kind=ChatKind.DM, peer="you")
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.key = self.chat.key

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_db_tables(self) -> None:
        reply = self.runtime.handle_control("/db tables", self.key)
        self.assertIn("tables (", reply)
        self.assertIn("memories", reply)

    def test_db_schema_and_query(self) -> None:
        schema = self.runtime.handle_control("/db schema messages", self.key)
        self.assertIn("messages", schema)
        self.assertIn("role", schema)
        reply = self.runtime.handle_control(
            "/db query SELECT role FROM messages LIMIT 1", self.key)
        self.assertIn("role", reply)
        refused = self.runtime.handle_control(
            "/db query DELETE FROM messages", self.key)
        self.assertIn("failed", refused)

    def test_db_counts(self) -> None:
        reply = self.runtime.handle_control("/db counts", self.key)
        self.assertIn("largest tables", reply)

    def test_api_list(self) -> None:
        reply = self.runtime.handle_control("/api list", self.key)
        self.assertIn("API connectors", reply)
        for name in ("weather", "fx", "github", "http_api"):
            self.assertIn(name, reply)

    def test_api_unknown_name(self) -> None:
        reply = self.runtime.handle_control("/api nope{}", self.key)
        self.assertIn("api failed", reply)

    def test_schedule_add_list_rm(self) -> None:
        reply = self.runtime.handle_control(
            "/schedule add goodnight 22:00 message goodnight 🌙", self.key)
        self.assertIn("scheduled goodnight", reply)

        listing = self.runtime.handle_control("/schedule list", self.key)
        self.assertIn("goodnight", listing)
        self.assertIn("daily 22:00", listing)

        removed = self.runtime.handle_control("/schedule rm goodnight", self.key)
        self.assertEqual(removed, "removed.")
        listing = self.runtime.handle_control("/schedule list", self.key)
        self.assertIn("no scheduled jobs", listing)

    def test_schedule_add_tool_payload(self) -> None:
        reply = self.runtime.handle_control(
            '/schedule add hourly 60m tool mood {}', self.key)
        self.assertIn("scheduled hourly", reply)
        run = self.runtime.handle_control("/schedule run hourly", self.key)
        self.assertIn("mood", run)

    def test_schedule_usage_and_bad_spec(self) -> None:
        bare = self.runtime.handle_control("/schedule", self.key)
        self.assertIn("no scheduled jobs", bare)
        bad = self.runtime.handle_control(
            "/schedule add x whenever message hi", self.key)
        self.assertIn("scheduling failed", bad)
        short = self.runtime.handle_control("/schedule add x 22:00", self.key)
        self.assertIn("usage", short)

    def test_stt_missing_file(self) -> None:
        reply = self.runtime.handle_control("/stt /nope/voice.ogg", self.key)
        self.assertIn("stt failed", reply)

    def test_tts_speaks_or_fails_cleanly(self) -> None:
        reply = self.runtime.handle_control("/tts hello there", self.key)
        self.assertTrue(reply.startswith("🔊") or reply.startswith("tts failed"),
                        msg=reply)

    def test_look_reads_screen(self) -> None:
        workspace = Path(self.context.settings.workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)
        shot = workspace / "shot.png"
        shot.write_bytes(_PNG_1X1)
        self.runtime.handle_control(f"/look {shot} the login form", self.key)
        sent = "".join(str(m.text if hasattr(m, "text") else m)
                       for m in self.adapter.sent)
        self.assertIn("screen read", sent)

    def test_swarm_runs_and_reports(self) -> None:
        self.runtime.handle_control(
            "/swarm check the database tables 2", self.key)
        sent = [m for m in self.adapter.sent]
        joined = "\n".join(str(m.text if hasattr(m, "text") else m) for m in sent)
        self.assertIn("swarm: check the database tables", joined)
        self.assertIn("swarm done", joined)

    def test_voice_note_without_backend_stays_a_note(self) -> None:
        from nomorals.social.chat.base import ChatMessage, MediaRef

        media = MediaRef(path="/tmp/nm-nope.ogg", mime="audio/ogg",
                         kind="audio", name="voice.ogg")
        msg = ChatMessage(chat=self.chat, incoming=True, text="", media=[media])
        notes = self.runtime.brain._media_notes(msg)
        # no STT backend configured in the test env → plain audio note, no crash
        self.assertTrue(any(n.startswith("audio:") for n in notes), notes)

    def test_help_mentions_new_commands(self) -> None:
        help_text = self.runtime.handle_control("/help", self.key)
        for token in ("/tts", "/stt", "/look", "/schedule", "/db", "/api", "/swarm"):
            self.assertIn(token, help_text)


# ── registry wave-40 smoke ───────────────────────────────────────────────────


class RegistryWave40Test(unittest.TestCase):
    def test_new_tools_present(self) -> None:
        reg = ToolRegistry().register_builtins()
        expected = {
            "api_call": Capability.NET_OUT,
            "api_list": Capability.NET_OUT,
            "speak": Capability.EXEC_SHELL,
            "transcribe": Capability.EXEC_SHELL,
            "audio_info": Capability.FS_READ,
            "db_query": Capability.DB_READ,
            "db_tables": Capability.DB_READ,
            "db_schema": Capability.DB_READ,
            "db_counts": Capability.DB_READ,
            "vision_screen": Capability.MODEL_CALL,
        }
        for name, cap in expected.items():
            spec = reg.get(name)
            self.assertIsNotNone(spec, f"tool {name} not registered")
            self.assertEqual(spec.capability, cap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
