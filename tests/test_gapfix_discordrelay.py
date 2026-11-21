"""Failing-first tests for the discord-github-ticket-relay port (lane 81).

A Discord ticket becomes a GitHub issue and the loop closes: issue
comments and state changes flow back to the Discord channel, Discord
replies flow to the issue as comments. Ported contract from
storytold/discord-github-ticket-relay (MIT): the issue title/body
shape, the submission marker, @-mention neutralisation, duplicate
protection, retry/failed state. The Discord panel, AI triage,
moderation queue and analytics of the upstream app are NOT ported —
Devon has no public webhook endpoint and no panel; the trigger here
is the /ticket verb in a channel Devon serves plus the brain tool.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.relay import content as relay_content
from nomorals.relay import jobs as relay_jobs
from nomorals.relay import surface as relay_surface
from nomorals.relay.relay import (
    RELAY_COMMENT_FOOTER,
    RelayEngine,
)
from nomorals.relay.store import (
    RelayError,
    TicketLink,
    TicketRelayStore,
    ticket_relay_path,
)

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402


class FakeGitHub:
    """A dict-backed stand-in for GitHubConnector's issue surface."""

    def __init__(self) -> None:
        self.issues: dict[tuple[str, int], dict[str, Any]] = {}
        self.comments: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.next_number = 1
        self.next_comment_id = 100
        self.fail_create: str = ""
        self.fail_comment: str = ""
        self.created = 0

    def create_issue(self, full_name: str, title: str, *, body: str = "",
                     labels: list[str] | None = None,
                     assignees: list[str] | None = None) -> dict[str, Any]:
        if self.fail_create:
            raise RuntimeError(self.fail_create)
        number = self.next_number
        self.next_number += 1
        self.created += 1
        issue = {
            "number": number, "title": title, "state": "open",
            "is_pull_request": False,
            "labels": list(labels or []), "author": "devon",
            "comments": 0, "created_at": "2026-11-18T00:00:00Z",
            "updated_at": "2026-11-18T00:00:00Z",
            "url": f"https://github.com/{full_name}/issues/{number}",
            "body": body[:2000],
        }
        self.issues[(full_name, number)] = issue
        self.comments[(full_name, number)] = []
        return dict(issue)

    def get_issue(self, full_name: str, number: int) -> dict[str, Any]:
        issue = self.issues.get((full_name, number))
        if issue is None:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        out = dict(issue)
        out["comments"] = len(self.comments.get((full_name, number), []))
        return out

    def list_issue_comments(self, full_name: str,
                            number: int) -> list[dict[str, Any]]:
        return [dict(c) for c in self.comments.get((full_name, number), [])]

    def comment_issue(self, full_name: str, number: int,
                      body: str) -> dict[str, Any]:
        if self.fail_comment:
            raise RuntimeError(self.fail_comment)
        if (full_name, number) not in self.issues:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        comment = {
            "id": self.next_comment_id, "body": body,
            "user": {"login": "devon"},
            "created_at": "2026-11-18T00:00:00Z",
        }
        self.next_comment_id += 1
        self.comments[(full_name, number)].append(comment)
        return dict(comment)

    def close_issue(self, full_name: str, number: int) -> dict[str, Any]:
        issue = self.issues.get((full_name, number))
        if issue is None:
            raise RuntimeError(f"issue {full_name}#{number} not found")
        issue["state"] = "closed"
        return dict(issue)

    def add_outside_comment(self, full_name: str, number: int, *,
                            author: str, body: str) -> dict[str, Any]:
        """A comment from a human on GitHub (not via the relay)."""
        comment = {
            "id": self.next_comment_id, "body": body,
            "user": {"login": author},
            "created_at": "2026-11-18T00:00:00Z",
        }
        self.next_comment_id += 1
        self.comments[(full_name, number)].append(comment)
        return dict(comment)


class FakeSend:
    """Records gateway sends as (chat_key, text)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail: str = ""

    def __call__(self, chat_key: str, text: str) -> dict[str, Any]:
        if self.fail:
            return {"ok": False, "error": self.fail}
        self.sent.append((chat_key, text))
        return {"ok": True}


class _StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-relay-81-")
        self.path = Path(self.tmp.name) / "relay" / "tickets.json"
        self.store = TicketRelayStore(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _open(self, gh: FakeGitHub | None = None, **kw: Any) -> dict[str, Any]:
        gh = gh if gh is not None else FakeGitHub()
        send = FakeSend()
        engine = RelayEngine(gh, send=send)
        out = engine.open_ticket(
            self.store, repo=kw.pop("repo", "acme/widgets"),
            title=kw.pop("title", "Widget crashes on launch"),
            body=kw.pop("body", "Steps: open it, it dies."),
            reporter=kw.pop("reporter", "Ada"),
            reporter_id=kw.pop("reporter_id", "111"),
            origin_platform=kw.pop("origin_platform", "discord"),
            origin_chat=kw.pop("origin_chat", "discord:222"),
            **kw,
        )
        assert out["ok"], out
        return out


class ContentTests(unittest.TestCase):
    def test_title_carries_app_and_type_prefix(self) -> None:
        title = relay_content.build_issue_title(
            "widgets", "Widget crashes on launch", type_prefix="bug")
        self.assertEqual(title, "[widgets] [bug] Widget crashes on launch")

    def test_title_is_capped_and_single_line(self) -> None:
        title = relay_content.build_issue_title("widgets", "x" * 300)
        self.assertLessEqual(len(title), 256)
        self.assertNotIn("\n", title)

    def test_body_carries_marker_and_metadata_rows(self) -> None:
        body = relay_content.build_issue_body(
            ref="T-abcdef01", app_name="widgets",
            title="Widget crashes on launch",
            description="Steps: open it, it dies.",
            reporter_display_name="Ada",
            reporter_username="ada_l",
            reporter_discord_id="111",
            submitted_at=datetime(2026, 11, 18, 14, 0, tzinfo=UTC),
            guild_name="Devon", guild_id="999", channel_id="222",
            ticket_type="bug",
        )
        self.assertIn("<!-- ticket-relay:submission=T-abcdef01 -->", body)
        self.assertIn("Ada", body)
        self.assertIn("111", body)
        self.assertIn("Devon", body)
        self.assertIn("222", body)
        self.assertIn("bug", body)
        self.assertIn("Steps: open it, it dies.", body)

    def test_neutralize_mentions_outside_fences_only(self) -> None:
        text = "@ada please look\n```\n@ada in code\n```\n`@ada` inline"
        out = relay_content.neutralize_github_mentions(text)
        self.assertIn("@\u200bada please look", out)
        self.assertIn("@ada in code", out)
        self.assertIn("`@ada` inline", out)

    def test_neutralize_leaves_closing_keywords_alone(self) -> None:
        # Parity with upstream: only @-mentions are neutralised.
        text = "fixes #123 and closes #4"
        self.assertEqual(
            relay_content.neutralize_github_mentions(text), text)


class StoreTests(_StoreCase):
    def test_path_lives_under_home_relay(self) -> None:
        settings = SimpleNamespace(home_path=Path(self.tmp.name))
        self.assertEqual(
            ticket_relay_path(settings),
            Path(self.tmp.name) / "relay" / "tickets.json")

    def test_add_get_list_round_trip(self) -> None:
        link = TicketLink(
            ref="T-abcdef01", repo="acme/widgets", issue_number=3,
            issue_url="https://github.com/acme/widgets/issues/3",
            title="Crash", reporter="Ada", reporter_id="111",
            origin_platform="discord", origin_chat="discord:222")
        self.store.add(link)
        got = self.store.get("T-abcdef01")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.issue_number, 3)
        self.assertEqual(self.store.get("T-abcd").ref, "T-abcdef01")
        self.assertEqual(len(self.store.list_all()), 1)
        reloaded = TicketRelayStore(self.path)
        self.assertEqual(len(reloaded.list_all()), 1)

    def test_duplicate_ref_is_refused(self) -> None:
        link = TicketLink(
            ref="T-abcdef01", repo="acme/widgets", issue_number=3,
            issue_url="https://github.com/acme/widgets/issues/3",
            title="Crash", reporter="Ada", reporter_id="111",
            origin_platform="discord", origin_chat="discord:222")
        self.store.add(link)
        with self.assertRaises(RelayError):
            self.store.add(link)

    def test_unknown_ref_answers_none(self) -> None:
        self.assertIsNone(self.store.get("T-nope"))
        self.assertIsNone(self.store.find_by_issue("acme/widgets", 99))

    def test_record_failure_is_durable(self) -> None:
        out = self._open()
        self.store.record_failure(out["ref"], "github said no")
        got = self.store.get(out["ref"])
        assert got is not None
        self.assertEqual(got.failures[-1]["reason"], "github said no")
        self.assertEqual(got.state, "failed")


class EngineTests(_StoreCase):
    def test_open_creates_one_issue_and_link(self) -> None:
        gh = FakeGitHub()
        out = self._open(gh)
        self.assertEqual(out["issue_number"], 1)
        self.assertIn("github.com/acme/widgets/issues/1", out["issue_url"])
        link = self.store.get(out["ref"])
        assert link is not None
        self.assertEqual(link.state, "open")
        self.assertEqual(link.last_issue_state, "open")
        issue = gh.issues[("acme/widgets", 1)]
        self.assertIn(f"ticket-relay:submission={out['ref']}",
                      issue["body"])
        self.assertIn("Ada", issue["body"])

    def test_open_with_same_ref_never_files_twice(self) -> None:
        gh = FakeGitHub()
        out = self._open(gh)
        engine = RelayEngine(gh, send=FakeSend())
        again = engine.open_ticket(
            self.store, repo="acme/widgets",
            title="Widget crashes on launch", reporter="Ada",
            reporter_id="111", origin_chat="discord:222",
            ref=out["ref"])
        self.assertTrue(again["ok"])
        self.assertEqual(again["issue_number"], 1)
        self.assertEqual(gh.created, 1)

    def test_open_requires_repo_and_title(self) -> None:
        gh = FakeGitHub()
        engine = RelayEngine(gh, send=FakeSend())
        bad = engine.open_ticket(self.store, repo="", title="x")
        self.assertFalse(bad["ok"])
        self.assertIn("repo", bad["reason"])
        bad = engine.open_ticket(self.store, repo="acme/widgets", title=" ")
        self.assertFalse(bad["ok"])
        self.assertIn("title", bad["reason"])
        self.assertEqual(gh.created, 0)

    def test_open_failure_is_recorded_never_fabricated(self) -> None:
        gh = FakeGitHub()
        gh.fail_create = "github 422: Validation Failed"
        engine = RelayEngine(gh, send=FakeSend())
        out = engine.open_ticket(
            self.store, repo="acme/widgets", title="Crash",
            reporter="Ada", reporter_id="111", origin_chat="discord:222")
        self.assertFalse(out["ok"])
        self.assertIn("422", out["reason"])
        self.assertNotIn("issue_url", out)
        link = self.store.get(out["ref"])
        assert link is not None
        self.assertEqual(link.state, "failed")
        self.assertEqual(link.failures[-1]["reason"], out["reason"])

    def test_comment_flows_to_github_with_footer(self) -> None:
        gh = FakeGitHub()
        out = self._open(gh)
        engine = RelayEngine(gh, send=FakeSend())
        done = engine.post_comment(
            self.store, out["ref"], author="Ada",
            text="Still crashing on 2.1")
        self.assertTrue(done["ok"])
        comments = gh.list_issue_comments("acme/widgets", 1)
        self.assertEqual(len(comments), 1)
        self.assertIn("Still crashing on 2.1", comments[0]["body"])
        self.assertIn(RELAY_COMMENT_FOOTER, comments[0]["body"])

    def test_comment_on_unknown_ticket_answers_honestly(self) -> None:
        engine = RelayEngine(FakeGitHub(), send=FakeSend())
        done = engine.post_comment(self.store, "T-nope", text="hi")
        self.assertFalse(done["ok"])
        self.assertIn("no ticket", done["reason"])

    def test_sync_posts_new_comment_and_state_once(self) -> None:
        gh = FakeGitHub()
        send = FakeSend()
        out = self._open(gh)
        gh.add_outside_comment("acme/widgets", 1, author="maintainer",
                               body="Reproduced, fix in flight")
        gh.issues[("acme/widgets", 1)]["state"] = "closed"
        engine = RelayEngine(gh, send=send)
        synced = engine.sync_ticket(self.store, out["ref"])
        self.assertTrue(synced["ok"])
        self.assertEqual(synced["posted"], 2)
        texts = [t for _, t in send.sent]
        self.assertTrue(any("Reproduced" in t for t in texts))
        self.assertTrue(any("closed" in t for t in texts))
        self.assertTrue(all(k == "discord:222" for k, _ in send.sent))
        again = engine.sync_ticket(self.store, out["ref"])
        self.assertEqual(again["posted"], 0)
        self.assertEqual(len(send.sent), 2)

    def test_sync_skips_the_relays_own_comments(self) -> None:
        gh = FakeGitHub()
        send = FakeSend()
        out = self._open(gh)
        engine = RelayEngine(gh, send=send)
        engine.post_comment(self.store, out["ref"], author="Ada",
                            text="Still crashing")
        synced = engine.sync_ticket(self.store, out["ref"])
        self.assertEqual(synced["posted"], 0)
        self.assertEqual(send.sent, [])

    def test_sync_without_a_send_queues_pending(self) -> None:
        gh = FakeGitHub()
        out = self._open(gh)
        gh.add_outside_comment("acme/widgets", 1, author="maintainer",
                               body="Looking into it")
        engine = RelayEngine(gh, send=None)
        synced = engine.sync_ticket(self.store, out["ref"])
        self.assertTrue(synced["ok"])
        self.assertEqual(synced["posted"], 0)
        link = self.store.get(out["ref"])
        assert link is not None
        self.assertEqual(len(link.pending_posts), 1)
        self.assertIn("Looking into it", link.pending_posts[0])
        send = FakeSend()
        engine2 = RelayEngine(gh, send=send)
        again = engine2.sync_ticket(self.store, out["ref"])
        self.assertEqual(again["posted"], 1)
        self.assertEqual(len(send.sent), 1)
        link = self.store.get(out["ref"])
        assert link is not None
        self.assertEqual(link.pending_posts, [])

    def test_sync_all_isolates_a_dead_ticket(self) -> None:
        gh = FakeGitHub()
        out1 = self._open(gh, title="First")
        out2 = self._open(gh, title="Second")
        del gh.issues[("acme/widgets", 2)]
        engine = RelayEngine(gh, send=FakeSend())
        done = engine.sync_all(self.store)
        self.assertTrue(done["ok"])
        self.assertEqual(len(done["results"]), 2)
        by_ref = {r["ref"]: r for r in done["results"]}
        self.assertTrue(by_ref[out1["ref"]]["ok"])
        self.assertFalse(by_ref[out2["ref"]]["ok"])

    def test_close_closes_issue_and_notes_discord(self) -> None:
        gh = FakeGitHub()
        send = FakeSend()
        out = self._open(gh)
        engine = RelayEngine(gh, send=send)
        done = engine.close_ticket(self.store, out["ref"], by="owner")
        self.assertTrue(done["ok"])
        self.assertEqual(
            gh.issues[("acme/widgets", 1)]["state"], "closed")
        link = self.store.get(out["ref"])
        assert link is not None
        self.assertEqual(link.state, "closed")
        self.assertTrue(any("closed" in t for _, t in send.sent))
        again = engine.close_ticket(self.store, out["ref"], by="owner")
        self.assertTrue(again["ok"])
        self.assertTrue(again.get("already"))


class JobsTests(unittest.TestCase):
    def test_ensure_registers_hourly_tool_job(self) -> None:
        added: list[tuple[str, str, str, dict]] = []

        class _Scheduler:
            # Mirrors the real Scheduler.add signature, including
            # ``overlap_policy`` (scheduler.py) — the relay ensure
            # registers overlap skip, so a fake without it would fail
            # the registration instead of testing it.
            def list_jobs(self) -> list:
                return []

            def add(self, name: str, spec: str, kind: str,
                    payload: dict, overlap_policy: str = "concurrent"
                    ) -> dict:
                added.append((name, spec, kind, payload))
                return {"name": name, "overlap_policy": overlap_policy}

        job = relay_jobs.ensure_ticket_relay_job(_Scheduler())
        self.assertIsNotNone(job)
        self.assertEqual(len(added), 1)
        name, spec, kind, payload = added[0]
        self.assertEqual(name, relay_jobs.TICKET_RELAY_JOB)
        self.assertEqual(kind, "tool")
        self.assertEqual(payload["tool"], relay_jobs.TICKET_RELAY_TOOL)
        self.assertIn("1h", spec)

    def test_ensure_is_idempotent(self) -> None:
        added: list[str] = []

        class _Scheduler:
            def list_jobs(self) -> list:
                return [{"name": relay_jobs.TICKET_RELAY_JOB,
                         "spec": relay_jobs.TICKET_RELAY_SPEC}]

            def add(self, *a: Any, **k: Any) -> None:
                added.append("x")

        relay_jobs.ensure_ticket_relay_job(_Scheduler())
        self.assertEqual(added, [])


class SurfaceTests(_StoreCase):
    def test_board_lists_open_tickets(self) -> None:
        out = self._open()
        board = relay_surface.render_ticket_board(self.store.list_all())
        self.assertIn(out["ref"], board)
        self.assertIn("acme/widgets#1", board)
        self.assertIn("open", board)

    def test_empty_board_answers_honestly(self) -> None:
        board = relay_surface.render_ticket_board([])
        self.assertIn("no tickets", board.lower())


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-relay-81-")
        self.gh = FakeGitHub()
        self.send = FakeSend()
        from nomorals.tools import registry as tool_registry
        from nomorals.tools import ticket_relay as ticket_tool

        self.registry = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=Path(self.tmp.name)),
            _ticket_github=self.gh,
            _ticket_send=self.send,
        ))
        ticket_tool.register(self.registry)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _call(self, **kw: Any) -> dict[str, Any]:
        tool = self.registry._tools["ticket_relay"].fn
        return tool(**kw)

    def test_tool_registers(self) -> None:
        self.assertIsNotNone(self.registry.get("ticket_relay"))
        self.assertIsNotNone(self.registry.get("__ticket_relay_sync__"))

    def test_tool_open_lists_and_status(self) -> None:
        opened = self._call(action="open", repo="acme/widgets",
                            title="Crash", body="it dies",
                            reporter="Ada", reporter_id="111",
                            origin_chat="discord:222")
        self.assertTrue(opened["ok"], opened)
        listed = self._call(action="list")
        self.assertEqual(len(listed["tickets"]), 1)
        status = self._call(action="status", ref=opened["ref"])
        self.assertEqual(status["ticket"]["issue_number"], 1)

    def test_tool_open_without_github_is_honest(self) -> None:
        from nomorals.tools import registry as tool_registry
        from nomorals.tools import ticket_relay as ticket_tool

        reg = tool_registry.ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home_path=Path(self.tmp.name))))
        ticket_tool.register(reg)
        out = reg._tools["ticket_relay"].fn(
            action="open", repo="acme/widgets", title="Crash")
        self.assertFalse(out["ok"])
        self.assertIn("github", out["error"].lower())

    def test_tool_sync_tick_posts_back(self) -> None:
        opened = self._call(action="open", repo="acme/widgets",
                            title="Crash", reporter="Ada",
                            reporter_id="111", origin_chat="discord:222")
        self.assertTrue(opened["ok"])
        self.gh.add_outside_comment("acme/widgets", 1, author="maintainer",
                                    body="Fixed in 2.2")
        tick = self.registry._tools["__ticket_relay_sync__"].fn()
        self.assertTrue(tick["ok"])
        self.assertEqual(tick["posted"], 1)
        self.assertTrue(any("Fixed in 2.2" in t for _, t in self.send.sent))


class ChatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings

        self.tmp = tempfile.TemporaryDirectory(prefix="nm-relay-81-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(
            self.context, gateway=_RecordingGateway())
        self.gh = FakeGitHub()
        self.runtime._ticket_github_client = self.gh
        self.store = TicketRelayStore(
            Path(self.tmp.name) / "relay" / "tickets.json")
        self.runtime._ticket_relay_store = lambda: self.store

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str, chat_key: str = "discord:222") -> str:
        reply = self.runtime.handle_control(
            text, chat_key, message=_msg(text, chat_key=chat_key))
        return reply.text if hasattr(reply, "text") else str(reply)

    def test_open_board_comment_sync_close_end_to_end(self) -> None:
        out = self._run(
            "/ticket open acme/widgets | Widget crashes | it dies on launch")
        self.assertIn("acme/widgets#1", out)
        self.assertIn("github.com/acme/widgets/issues/1", out)
        board = self._run("/ticket")
        self.assertIn("acme/widgets#1", board)
        self.assertIn("open", board)
        ref = self.store.list_all()[0].ref
        commented = self._run(f"/ticket comment {ref} | still on 2.1")
        self.assertIn("posted", commented.lower())
        self.assertEqual(
            len(self.gh.list_issue_comments("acme/widgets", 1)), 1)
        self.gh.add_outside_comment(
            "acme/widgets", 1, author="maintainer", body="Fix shipped")
        synced = self._run("/ticket sync")
        self.assertIn("1", synced)
        closed = self._run(f"/ticket close {ref}")
        self.assertIn("closed", closed.lower())
        self.assertEqual(
            self.gh.issues[("acme/widgets", 1)]["state"], "closed")

    def test_unknown_ticket_answers_honestly(self) -> None:
        out = self._run("/ticket status T-nope")
        self.assertIn("no ticket", out.lower())

    def test_open_without_repo_answers_honestly(self) -> None:
        out = self._run("/ticket open | No repo here")
        self.assertIn("repo", out.lower())
        self.assertEqual(self.gh.created, 0)


if __name__ == "__main__":
    unittest.main()
