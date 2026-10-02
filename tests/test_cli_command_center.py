"""Wave D Stream F — the ``nm`` command center.

Covers: the alias map (resolution, no collisions), help completeness
(every registered command has help text; ``nm help cli`` / ``nm help <cmd>``),
and the informative status surfaces (``nm status``, ``nm mind``,
``nm missions`` with crash-resume info).
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.cli import (
    CLI_ALIASES,
    _canonical_command,
    _cli_command_help,
    _cli_overview,
    _cli_subparsers,
    _cmd_mind,
    _cmd_missions,
    _cmd_status,
    _parser,
)
from nomorals.core.config import Settings
from nomorals.core.errors import NotFound, ValidationError
from nomorals.missions import MissionStore


def _canonical_commands() -> dict[str, argparse.ArgumentParser]:
    """canonical name → subparser, deduped (aliases share the parser object)."""
    sub = _cli_subparsers()
    assert sub is not None
    out: dict[str, argparse.ArgumentParser] = {}
    for name, target in sub.choices.items():
        canonical = target.prog.split()[-1]
        out.setdefault(canonical, target)
    return out


def _help_text_for(sub: object, canonical: str) -> str:
    for choice_action in sub._choices_actions:  # noqa: SLF001 - argparse internals
        if choice_action.dest.strip() == canonical:
            return choice_action.help or ""
    return ""


class AliasMapTests(unittest.TestCase):
    def test_every_alias_resolves_to_its_canonical(self):
        for canonical, aliases in CLI_ALIASES.items():
            self.assertEqual(_canonical_command(canonical), canonical)
            for alias in aliases:
                self.assertEqual(_canonical_command(alias), canonical,
                                 f"alias {alias!r} should resolve to {canonical!r}")

    def test_no_alias_collides_with_a_command_or_another_alias(self):
        canonicals = set(_canonical_commands())
        seen: dict[str, str] = {}
        for canonical, aliases in CLI_ALIASES.items():
            self.assertIn(canonical, canonicals,
                          f"CLI_ALIASES key {canonical!r} is not a real command")
            for alias in aliases:
                self.assertNotIn(alias, canonicals,
                                 f"alias {alias!r} collides with a command name")
                self.assertNotIn(alias, seen,
                                 f"alias {alias!r} claimed by both "
                                 f"{seen.get(alias)!r} and {canonical!r}")
                seen[alias] = canonical

    def test_aliases_parse_end_to_end(self):
        parser = _parser()
        # some commands require positionals; give them dummies
        dummy_args = {"run": ["dummy-goal"], "ask": ["dummy prompt"],
                      "briefing": ["status"],
                      "simulate": ["risk", "echo hi"],
                      "arena": ["status"], "trial": ["list"],
                      "research-loop": ["status"], "hub": ["status"],
                      "skill": ["list"], "book": ["list"],
                      # wave F3: every command has an alias now, so every
                      # command with a required subaction needs dummy args
                      "bet": ["bankroll"], "captcha": ["detect"],
                      "improve": ["status"], "inbox": ["list"],
                      "media": ["probe", "x"], "room": ["new", "x"],
                      "studio": ["presets"], "swarm": ["run", "x"],
                      "trade": ["analyze", "x"], "vision": ["describe", "x"],
                      "voice": ["stats"], "weather": ["now"]}
        for canonical, aliases in CLI_ALIASES.items():
            for alias in aliases:
                argv = [alias] + dummy_args.get(canonical, [])
                try:
                    args = parser.parse_args(argv)
                except SystemExit as exc:
                    self.fail(f"nm {' '.join(argv)} did not parse "
                              f"(exit {exc.code})")
                self.assertEqual(_canonical_command(args.command), canonical,
                                 f"nm {alias} did not dispatch to {canonical}")

    def test_expected_shortcuts_exist(self):
        for canonical, alias in (("status", "st"), ("mind", "m"),
                                 ("missions", "ms"), ("mission", "mi"),
                                 ("doctor", "dr"), ("memory", "mem"),
                                 ("help", "h"), ("queue", "q")):
            self.assertIn(alias, CLI_ALIASES[canonical])


class HelpCompletenessTests(unittest.TestCase):
    def test_every_command_has_help_text(self):
        sub = _cli_subparsers()
        missing = [name for name in _canonical_commands()
                   if not _help_text_for(sub, name).strip()]
        self.assertEqual(missing, [],
                         f"commands without help text: {missing}")

    def test_stub_commands_say_so(self):
        # Wave E: the six former stubs are real now — their help must NOT
        # claim "not implemented" anymore; each names its real module.
        sub = _cli_subparsers()
        for name in ("book", "hub", "arena", "trial", "skill", "simulate",
                     "research-loop"):
            text = _help_text_for(sub, name)
            self.assertNotIn("not implemented", text.lower(),
                             f"{name}: help still claims a stub: {text!r}")
            self.assertTrue(text.strip(), f"{name}: help text is empty")

    def test_help_cli_overview_lists_commands_and_aliases(self):
        page = _cli_overview()
        for canonical, aliases in CLI_ALIASES.items():
            self.assertIn(canonical, page)
            for alias in aliases:
                self.assertIn(alias, page)
        # every canonical command appears, not just aliased ones
        for name in _canonical_commands():
            self.assertIn(name, page)

    def test_help_for_command_topic(self):
        page = _cli_command_help("missions")
        self.assertIsNotNone(page)
        self.assertIn("usage: nm missions", page)
        self.assertIn("--resume", page)

    def test_help_for_alias_topic(self):
        page = _cli_command_help("st")
        self.assertIsNotNone(page)
        self.assertIn("usage: nm status", page)

    def test_help_for_unknown_topic_is_none(self):
        self.assertIsNone(_cli_command_help("no-such-command-xyz"))


class CommandCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="nm-cmdctr-")
        cls._ctx_mgr = build_context(Settings(home=cls.home))
        cls.ctx = cls._ctx_mgr.__enter__()
        cls.addClassCleanup(cls._ctx_mgr.__exit__, None, None, None)
        cls.addClassCleanup(shutil.rmtree, cls.home, True)

    def _run(self, fn, **kwargs):
        args = argparse.Namespace(json=False, **kwargs)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = fn(args, self.ctx)
        return rc, buf.getvalue()

    # ── nm status ─────────────────────────────────────────────────────────
    def test_status_shows_real_sections(self):
        rc, out = self._run(_cmd_status)
        self.assertEqual(rc, 0)
        for section in ("system:", "database:", "queue:", "missions:",
                        "memory:", "proactive:", "power:"):
            self.assertIn(section, out, f"status missing {section} section")
        self.assertIn("schema v", out)
        self.assertNotIn("unavailable", out,
                         f"fresh context should have no gaps:\n{out}")

    def test_status_json_has_sections(self):
        args = argparse.Namespace(json=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_status(args, self.ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIn("sections", payload)
        for name in ("system", "database", "queue", "missions",
                     "memory", "proactive", "power"):
            self.assertIn(name, payload["sections"])
        self.assertTrue(payload["sections"]["database"]["available"])
        self.assertGreaterEqual(
            payload["sections"]["database"]["schema_version"], 1)

    def test_status_degrades_honestly(self):
        # A context whose DB is gone must say "unavailable", not print zeros.
        broken = argparse.Namespace(json=False)
        ctx = _BrokenContext(self.ctx)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_status(broken, ctx)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("unavailable", out)

    # ── nm mind ───────────────────────────────────────────────────────────
    def _seed_coremind(self):
        state_dir = Path(self.ctx.settings.resolve("data/coremind"))
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pending": {
                "telegram:123": {
                    "kind": "research", "action": "run", "target": "q4 prices",
                    "question": "which quarter did you mean — Q3 or Q4?",
                    "created": time.time(),
                }
            },
            "jobs": [
                {"id": "abc123", "kind": "research", "target": "q4 prices",
                 "route": "research", "status": "done",
                 "created": time.time() - 60, "note": "3 sources cited",
                 "finished": time.time() - 30, "mem": ""},
                {"id": "def456", "kind": "build", "target": "price watcher",
                 "route": "build", "status": "failed",
                 "created": time.time() - 3600, "note": "sandbox timeout",
                 "finished": time.time() - 3500, "mem": ""},
            ],
            "last_objective": {"text": "track q4 prices", "route": "research",
                               "job": "abc123", "created": time.time() - 60},
        }
        (state_dir / "state.json").write_text(json.dumps(payload))

    def test_mind_reports_unavailable_honestly(self):
        rc, out = self._run(_cmd_mind, action="status")
        self.assertEqual(rc, 0)
        self.assertIn("pending clarifications", out)
        self.assertIn("recent jobs", out)
        # persisted router telemetry (migration 65): with no decisions
        # recorded yet, nm mind says so — never a fake 0
        self.assertIn("router calls: none recorded yet", out)
        self.assertIn("last plan_error: none recorded", out)

    def test_mind_shows_seeded_clarifications_and_jobs(self):
        self._seed_coremind()
        rc, out = self._run(_cmd_mind, action="status")
        self.assertEqual(rc, 0)
        self.assertIn("pending clarifications: 1", out)
        self.assertIn("which quarter did you mean", out)
        self.assertIn("abc123 [done] research", out)
        self.assertIn("def456 [failed] build", out)
        self.assertIn("sandbox timeout", out)
        self.assertIn("last objective: track q4 prices", out)

    def test_mind_json(self):
        self._seed_coremind()
        args = argparse.Namespace(json=True, action="status")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_mind(args, self.ctx)
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(len(payload["pending_clarifications"]), 1)
        self.assertEqual(len(payload["recent_jobs"]), 2)
        # router telemetry is persisted (migration 65), not per-process
        self.assertIn("per_route", payload["router_calls"])
        self.assertIn("total", payload["router_calls"])
        self.assertIn("model_consults", payload["router_calls"])
        self.assertIsNone(payload["last_plan_error"])

    # ── nm missions ───────────────────────────────────────────────────────
    def _mission_args(self, **kwargs):
        base = dict(start="", resume="", resume_all=False, status="",
                    show="", max_iterations=8, budget_wall=0.0,
                    budget_tokens=0, no_reflect=True, pause="", cancel="",
                    resume_status="", json=False)
        base.update(kwargs)
        return argparse.Namespace(**base)

    def test_missions_list_shows_crash_resume_info(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("interrupted work")
        store.set_status(mission.id, "running", "")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_missions(self._mission_args(), self.ctx)
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("interrupted (resumable)", out)
        self.assertIn("↺", out)
        self.assertIn(mission.id, out)
        self.assertIn("nm missions --resume <id>", out)

    def test_missions_pause_and_cancel_roundtrip(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("pausable work")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_missions(self._mission_args(pause=mission.id), self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("paused", buf.getvalue())
        self.assertEqual(store.get(mission.id).status, "paused")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_missions(self._mission_args(cancel=mission.id), self.ctx)
        self.assertEqual(rc, 0)
        self.assertEqual(store.get(mission.id).status, "cancelled")

    def test_missions_pause_unknown_id_errors_cleanly(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_missions(self._mission_args(pause="no-such-id"), self.ctx)
        self.assertEqual(rc, 2)

    def test_missions_show_marks_resumable(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("show me")
        store.set_status(mission.id, "paused", "")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_missions(self._mission_args(show=mission.id), self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("resumable: yes", buf.getvalue())


class MissionStoreSetStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="nm-setstatus-")
        cls._ctx_mgr = build_context(Settings(home=cls.home))
        cls.ctx = cls._ctx_mgr.__enter__()
        cls.addClassCleanup(cls._ctx_mgr.__exit__, None, None, None)
        cls.addClassCleanup(shutil.rmtree, cls.home, True)

    def test_set_status_roundtrip(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("flip me")
        store.set_status(mission.id, "paused", "taking a break")
        got = store.get(mission.id)
        self.assertEqual(got.status, "paused")
        self.assertEqual(got.state.get("status_note"), "taking a break")

    def test_set_status_unknown_id_raises_not_found(self):
        store = MissionStore(self.ctx.db)
        with self.assertRaises(NotFound):
            store.set_status("missing-id", "paused", "")

    def test_set_status_bad_status_raises(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("bad flip")
        with self.assertRaises(ValidationError):
            store.set_status(mission.id, "exploding", "")

    def test_terminal_mission_cannot_be_reactivated(self):
        store = MissionStore(self.ctx.db)
        mission = store.create_new("done work")
        store.set_status(mission.id, "cancelled", "")
        with self.assertRaises(ValidationError):
            store.set_status(mission.id, "running", "")


class _BrokenContext:
    """Context whose DB/memory are gone: status must degrade, not crash.

    ``extras`` is a fresh dict (not the real one) so probes that cache
    per-context state cannot pollute the wrapped context.
    """

    def __init__(self, real: object):
        self._real = real
        self._extras: dict[str, object] = {}

    def __getattr__(self, name: str):
        if name == "db":
            raise RuntimeError("db gone")
        if name == "memory":
            raise RuntimeError("memory gone")
        if name == "extras":
            return self._extras
        return getattr(self._real, name)


if __name__ == "__main__":
    unittest.main()
