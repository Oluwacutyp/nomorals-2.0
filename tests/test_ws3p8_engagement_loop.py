"""WS3 P8 — the engagement loop.

Pins, per the shipped platform truth:
  (a) the poll ingests a capable adapter's items exactly once
      across two polls (dedup on the platform's item id); a
      failing platform is recorded by name and the others still
      land; an incapable platform is NAMED unavailable in the
      poll report — never read as zero engagement;
  (b) the board renders needs-him ordering (new, oldest first)
      with no internal ids anywhere; ignore/done transitions
      stick and remove the item from the board;
  (c) reply <n> lands ONE pending_review draft carrying the
      engagement linkage metadata + the recorded drafter rung;
      approving + posting it through the real queue path
      (handle_draft_callback → execute_post) threads the reply
      onto the original post on a reply-capable adapter and
      flips the item to replied; a read-capable adapter WITHOUT
      reply support yields the honest plain-draft shape (no
      executable reply_to, the note says why);
  (d) outcomes: a post's engagement volume feeds the briefs
      loop through the spine's exact join — exactly one linkable
      brief draft gets the outcome; an ambiguous join records
      ``ambiguous`` and feeds nothing; an unlinked post records
      ``unlinked``;
  (e) the live gate: with the owner's stop set, the personal
      poll path makes ZERO read calls on the adapter and
      records the platform gated with the gate's words;
  (f) the shipped adapters' structural truth: Mastodon + X are
      read-capable and reply-capable; LinkedIn + both personal
      adapters are not read-capable (the survey, as code);
  (g) the poll job registers via the ensure_* pattern,
      idempotently, as a routine tool job on the spine's 6h
      cadence.

No network: every adapter is a fake; the shipped adapters
appear only in capability-shape pins (constructor + MRO).
"""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

from nomorals.core.ids import new_id
from nomorals.social.base import PlatformAdapter, PostResult
from nomorals.social.engagement import (
    STATE_DONE,
    STATE_IGNORED,
    STATE_NEW,
    STATE_REPLIED,
    STATE_TRIAGED,
    EngagementStore,
    adapter_supports_engagement,
    board_items,
    draft_reply,
    platform_truth,
    poll_engagement,
    reconcile_replied,
    render_board,
)
from nomorals.social.manager import SocialManager

NOW_DT = datetime(2026, 11, 9, 12, 0)  # a Monday
NOW = NOW_DT.timestamp()
HOUR = 3600.0
DAY = 86400.0


# ── fakes + harness ──────────────────────────────────────────────────


class EngageAdapter(PlatformAdapter):
    """Capable adapter: canned engagement, native reply threading."""

    supports_reply = True

    def __init__(self, name: str, results: dict | None = None):
        self.name = name
        self.results = dict(results or {})
        self.read_calls: list[str] = []
        self.posts: list[dict] = []

    def post(self, account, content, **kwargs):
        self.posts.append({"content": content,
                           "reply_to": kwargs.get("reply_to")})
        return PostResult(platform=self.name, ok=True,
                          external_id=f"{self.name}-reply-"
                                      f"{len(self.posts)}")

    def read_engagement(self, account, external_id):
        self.read_calls.append(external_id)
        return [dict(i) for i in self.results.get(external_id, [])]


class ReadOnlyEngageAdapter(PlatformAdapter):
    """Reads engagement but cannot thread a reply (supports_reply
    stays False — the honest plain-draft shape)."""

    def __init__(self, name: str, results: dict | None = None):
        self.name = name
        self.results = dict(results or {})

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="plain-1")

    def read_engagement(self, account, external_id):
        return [dict(i) for i in self.results.get(external_id, [])]


class NoEngageAdapter(PlatformAdapter):
    """Incapable adapter: inherits the base read_engagement."""

    def __init__(self, name: str):
        self.name = name

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")


class BoomEngageAdapter(PlatformAdapter):
    """Capable on paper; the read explodes."""

    def __init__(self, name: str):
        self.name = name

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")

    def read_engagement(self, account, external_id):
        raise RuntimeError("platform exploded")


class PersonalEngageAdapter(PlatformAdapter):
    """Personal-shaped: carries the live gate + a read method.

    The poll must consult the gate BEFORE the read; when the gate
    speaks, the read must never happen.
    """

    def __init__(self, name: str, gate: str = "",
                 results: dict | None = None):
        self.name = name
        self.gate_text = gate
        self.results = dict(results or {})
        self.read_calls: list[str] = []

    def post(self, account, content, **kwargs):
        return PostResult(platform=self.name, ok=True,
                          external_id="unused")

    def engagement_gate(self):
        return self.gate_text

    def read_engagement(self, account, external_id):
        self.read_calls.append(external_id)
        return [dict(i) for i in self.results.get(external_id, [])]


def _item(item_id, author="Ada Lovelace", text="great post, thanks",
          kind="reply", created_at=NOW - 2 * HOUR):
    return {"item_id": item_id, "author": author, "text": text,
            "kind": kind, "created_at": created_at}


def _db():
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    return db


def _manager(db, adapters) -> SocialManager:
    ctx = SimpleNamespace(db=db, tools=None, extras={})
    return SocialManager(ctx, adapters=adapters, enforce_policy=False)


def _seed_post(manager, account, *, content="hello world",
               external_id="ext-1", posted_at=NOW - 2 * HOUR,
               platform=None):
    row = manager.posts.create({
        "id": new_id(), "platform": platform or account.platform,
        "account_id": account.id, "content": content,
        "media_paths": [], "status": "posted",
        "external_id": external_id, "posted_at": posted_at,
        "created_at": posted_at, "metrics": {},
    })
    return row["id"]


def _queue(tmp_path):
    from nomorals.social.drafts import DraftQueue

    return DraftQueue(str(tmp_path / "drafts.db"))


def _brief_store(tmp_path):
    from nomorals.marketing.briefs import BriefStore

    return BriefStore(db_path=str(tmp_path / "briefs.db"))


_MOCK_RESULTS = [
    {"title": "Lagos rents explained",
     "snippet": "How much is rent in Lagos? 2 bedroom flats in "
                "Yaba cost around 2m per year. What should you "
                "look for when renting? Always inspect before "
                "paying."},
    {"title": "Yaba rental guide",
     "snippet": "Why are Yaba rents rising? The area is close to "
                "tech hubs. How do you negotiate with Lagos "
                "landlords? Get everything in writing."},
]


def _brief_draft(store):
    brief = store.build_brief("Lagos rents",
                              search_fn=lambda topic: _MOCK_RESULTS)
    draft = store.draft_from_brief(brief)
    assert draft is not None
    return draft


# ── (a) the poll: ingest, dedup, failure, unavailable ───────────────


class TestPoll:
    def test_ingests_once_across_two_polls(self):
        adapter = EngageAdapter("mastodon", {
            "ext-1": [_item("r-1"), _item("r-2", author="Bo")]})
        manager = _manager(_db(), {"mastodon": adapter})
        account = manager.connect("mastodon", "@me@example.social")
        _seed_post(manager, account)

        first = poll_engagement(manager, now=NOW, feed_outcomes=False)
        assert first["new_items"] == 2
        assert first["duplicates"] == 0

        second = poll_engagement(
            manager, now=NOW + HOUR, feed_outcomes=False)
        assert second["new_items"] == 0
        assert second["duplicates"] == 2

        store = EngagementStore(manager.db)
        items = store.items()
        assert [i["item_external_id"] for i in items] == ["r-1", "r-2"]
        assert all(i["state"] == STATE_NEW for i in items)
        assert adapter.read_calls == ["ext-1", "ext-1"]

    def test_failing_platform_recorded_others_still_land(self):
        good = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        bad = BoomEngageAdapter("x")
        manager = _manager(_db(), {"mastodon": good, "x": bad})
        a1 = manager.connect("mastodon", "@me@example.social")
        a2 = manager.connect("x", "@me")
        _seed_post(manager, a1, external_id="ext-1")
        _seed_post(manager, a2, external_id="ext-2")

        report = poll_engagement(manager, now=NOW, feed_outcomes=False)
        assert report["new_items"] == 1
        assert report["failed"] == 1
        assert report["failures"][0]["platform"] == "x"
        assert report["failures"][0]["external_id"] == "ext-2"
        assert "exploded" in report["failures"][0]["error"]

    def test_incapable_platform_named_unavailable(self):
        adapter = NoEngageAdapter("linkedin")
        manager = _manager(_db(), {"linkedin": adapter})
        account = manager.connect("linkedin", "me")
        _seed_post(manager, account)

        report = poll_engagement(manager, now=NOW, feed_outcomes=False)
        assert report["unavailable"] == 1
        assert report["unavailable_platforms"] == ["linkedin"]
        assert report["new_items"] == 0
        store = EngagementStore(manager.db)
        assert store.items() == []


# ── (e) the live gate ────────────────────────────────────────────────


class TestLiveGate:
    def test_owner_stop_means_zero_read_calls(self):
        adapter = PersonalEngageAdapter(
            "whatsapp-status",
            gate="whatsapp is under your standing stop — it stays "
                 "halted until you start it again; nothing was "
                 "posted",
            results={"ext-1": [_item("r-1")]})
        manager = _manager(_db(), {"whatsapp-status": adapter})
        account = manager.connect("whatsapp-status", "me")
        _seed_post(manager, account)

        report = poll_engagement(manager, now=NOW, feed_outcomes=False)
        assert adapter.read_calls == []  # never even asked
        assert report["gated"] == 1
        assert report["gated_platforms"] == ["whatsapp-status"]
        assert "standing stop" in report["gated_reasons"][
            "whatsapp-status"]
        assert report["new_items"] == 0

    def test_gate_open_reads_normally(self):
        adapter = PersonalEngageAdapter(
            "telegram", gate="", results={"ext-1": [_item("r-9")]})
        manager = _manager(_db(), {"telegram": adapter})
        account = manager.connect("telegram", "@chan")
        _seed_post(manager, account)

        report = poll_engagement(manager, now=NOW, feed_outcomes=False)
        assert report["gated"] == 0
        assert report["new_items"] == 1
        assert adapter.read_calls == ["ext-1"]

    def test_shipped_personal_adapters_gate_verdict(self):
        """The real personal adapters expose the gate: with no
        live surface in this process the verdict refuses in words
        (zero platform contact — there is nothing to contact)."""
        from nomorals.social.adapters.personal import TelegramChannelAdapter, WhatsAppStatusAdapter

        for adapter in (WhatsAppStatusAdapter(context=None),
                        TelegramChannelAdapter(context=None)):
            verdict = adapter.engagement_gate()
            assert "no live" in verdict


# ── (b) the board ────────────────────────────────────────────────────


def _store_with_items(db):
    store = EngagementStore(db)
    store.add_item(
        platform="mastodon", post_id="p-1", post_external_id="ext-1",
        account_id="a-1", item_external_id="r-old", author="Old Ana",
        text="first reply, still waiting", kind="reply",
        arrived_at=NOW - 3 * DAY)
    store.add_item(
        platform="x", post_id="p-2", post_external_id="ext-2",
        account_id="a-2", item_external_id="r-new", author="New Bo",
        text="a fresher reply", kind="comment",
        arrived_at=NOW - HOUR)
    store.add_item(
        platform="mastodon", post_id="p-1", post_external_id="ext-1",
        account_id="a-1", item_external_id="r-seen", author="Seen Cy",
        text="already looked at this one", kind="reply",
        arrived_at=NOW - 2 * DAY)
    seen = [i for i in store.items()
            if i["item_external_id"] == "r-seen"][0]
    store.set_state(seen["id"], STATE_TRIAGED)
    return store


class TestBoard:
    def test_needs_him_ordering_oldest_first(self):
        store = _store_with_items(_db())
        items = board_items(store)
        assert [i["item_external_id"] for i in items] == [
            "r-old", "r-new", "r-seen"]

    def test_board_has_no_internal_ids(self):
        store = _store_with_items(_db())
        text = render_board(store)
        for item in store.items():
            assert item["id"] not in text
            assert item["post_id"] not in text
        assert "Old Ana" in text and "New Bo" in text
        assert "needs you" in text

    def test_board_names_unavailable_platforms(self):
        store = _store_with_items(_db())
        truth = {"mastodon": {"read": True, "reply": True},
                 "linkedin": {"read": False, "reply": False}}
        text = render_board(store, truth=truth)
        assert "watching: mastodon" in text
        assert "linkedin" in text
        assert "no comments read" in text

    def test_ignore_and_done_transitions_stick(self):
        store = _store_with_items(_db())
        items = board_items(store)
        store.set_state(items[0]["id"], STATE_IGNORED)
        store.set_state(items[1]["id"], STATE_DONE)
        remaining = board_items(store)
        assert [i["item_external_id"] for i in remaining] == ["r-seen"]
        assert store.get(items[0]["id"])["state"] == STATE_IGNORED
        assert store.get(items[1]["id"])["state"] == STATE_DONE

    def test_empty_board_is_quiet_and_honest(self):
        text = render_board(EngagementStore(_db()))
        assert "nothing is waiting" in text


# ── (c) reply drafts through the queue + #43 ─────────────────────────


class TestReplyDrafts:
    def _setup(self, tmp_path, adapter, platform="mastodon"):
        manager = _manager(_db(), {platform: adapter})
        account = manager.connect(platform, "@me")
        _seed_post(manager, account,
                   posted_at=time.time() - 2 * HOUR)
        poll_engagement(manager, now=time.time(),
                        feed_outcomes=False)
        store = EngagementStore(manager.db)
        item = store.items()[0]
        return manager, store, _queue(tmp_path), item

    def test_reply_draft_shape_and_brain_rung(self, tmp_path):
        adapter = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        manager, store, queue, item = self._setup(tmp_path, adapter)
        result = draft_reply(
            store, queue, item["id"],
            llm_fn=lambda prompt: "Thank you, Ada — more soon.",
            adapters=manager.adapters)
        assert result["ok"] is True
        assert result["reply_native"] is True
        assert result["drafter"] == "brain"
        draft = queue.get(result["draft_id"])
        assert draft.status == "pending_review"
        assert draft.platforms == ["mastodon"]
        meta = draft.metadata
        assert meta["engagement_item"] == item["id"]
        assert meta["platform"] == "mastodon"
        assert meta["in_reply_to"] == "ext-1"
        assert meta["reply_to"] == "ext-1"
        assert meta["author"] == "Ada Lovelace"
        assert meta["drafter"] == "brain"
        # Nothing posted; the item waits in hand, draft linked.
        assert adapter.posts == []
        after = store.get(item["id"])
        assert after["state"] == STATE_TRIAGED
        assert after["reply_draft_id"] == draft.id

    def test_template_fallback_rung_recorded(self, tmp_path):
        adapter = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        _manager1, store, queue, item = self._setup(tmp_path, adapter)
        result = draft_reply(store, queue, item["id"], llm_fn=None,
                             adapters={"mastodon": adapter})
        assert result["drafter"] == "template"
        draft = queue.get(result["draft_id"])
        assert draft.metadata["drafter"] == "template"
        assert "Ada" in draft.content

    def test_posting_through_queue_path_flips_replied(self, tmp_path):
        from nomorals.social.drafts import handle_draft_callback

        adapter = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        manager, store, queue, item = self._setup(tmp_path, adapter)
        result = draft_reply(
            store, queue, item["id"],
            llm_fn=lambda prompt: "Thanks, Ada!",
            adapters=manager.adapters)
        reply = handle_draft_callback(
            queue, f"draft:post:{result['draft_id']}", manager=manager)
        assert reply == "posted to 1 platform(s)."
        # The adapter saw the thread link — the reply lands ON the
        # original post, not beside it.
        assert adapter.posts[0]["reply_to"] == "ext-1"
        after = store.get(item["id"])
        assert after["state"] == STATE_REPLIED
        assert after["reply_post_external_id"] == "mastodon-reply-1"

    def test_no_reply_capability_plain_draft_shape(self, tmp_path):
        adapter = ReadOnlyEngageAdapter("bluesky",
                                        {"ext-1": [_item("r-1")]})
        manager, store, queue, item = self._setup(
            tmp_path, adapter, platform="bluesky")
        result = draft_reply(store, queue, item["id"], llm_fn=None,
                             adapters=manager.adapters)
        assert result["ok"] is True
        assert result["reply_native"] is False
        draft = queue.get(result["draft_id"])
        assert "reply_to" not in draft.metadata  # nothing executable
        assert draft.metadata["in_reply_to"] == "ext-1"
        assert "standalone" in draft.metadata["reply_note"]

    def test_reconcile_flips_from_queue_truth(self, tmp_path):
        adapter = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        _m, store, queue, item = self._setup(tmp_path, adapter)
        result = draft_reply(store, queue, item["id"], llm_fn=None,
                             adapters={"mastodon": adapter})
        queue.mark_posted(result["draft_id"])  # posted elsewhere
        assert reconcile_replied(store, queue) == 1
        assert store.get(item["id"])["state"] == STATE_REPLIED

    def test_terminal_item_refuses_a_draft(self, tmp_path):
        adapter = EngageAdapter("mastodon", {"ext-1": [_item("r-1")]})
        _m, store, queue, item = self._setup(tmp_path, adapter)
        store.set_state(item["id"], STATE_IGNORED)
        result = draft_reply(store, queue, item["id"],
                             adapters={"mastodon": adapter})
        assert result["ok"] is False
        assert queue.pending() == []


# ── (d) outcomes into the spine ──────────────────────────────────────


class TestOutcomes:
    def _manager_with_post(self, tmp_path, *, content, posted_days=2):
        # Brief drafts carry platform "x" (the BriefStore default),
        # so the join is exercised on x — the spine's own shape.
        adapter = EngageAdapter("x", {
            "ext-1": [_item("r-1"), _item("r-2", author="Bo")]})
        manager = _manager(_db(), {"x": adapter})
        account = manager.connect("x", "@me")
        post_id = _seed_post(manager, account, content=content,
                             posted_at=NOW - posted_days * DAY)
        return manager, post_id

    def test_volume_feeds_the_one_linkable_brief(self, tmp_path):
        briefs = _brief_store(tmp_path)
        draft = _brief_draft(briefs)
        manager, post_id = self._manager_with_post(
            tmp_path, content=draft.content)
        report = poll_engagement(manager, now=NOW, brief_store=briefs)
        assert report["outcomes_fed"] == 1
        fed = briefs.get_draft(draft.draft_id)
        assert fed.outcome_metrics["engagement_items"] == 2
        assert fed.outcome_metrics["engagement_replies"] == 0
        store = EngagementStore(manager.db)
        outcome = store.outcome_for(post_id)
        assert outcome["verdict"] == "fed"
        assert outcome["items_received"] == 2
        # Once only: the next poll records nothing new.
        again = poll_engagement(manager, now=NOW + HOUR,
                                brief_store=briefs)
        assert again["outcomes_fed"] == 0

    def test_ambiguous_join_records_and_feeds_nothing(self, tmp_path):
        briefs = _brief_store(tmp_path)
        first = _brief_draft(briefs)
        second = _brief_draft(briefs)  # same content, second draft
        assert first.content == second.content
        manager, post_id = self._manager_with_post(
            tmp_path, content=first.content)
        report = poll_engagement(manager, now=NOW, brief_store=briefs)
        assert report["outcomes_ambiguous"] == 1
        assert report["outcomes_fed"] == 0
        for draft in (first, second):
            assert not briefs.get_draft(draft.draft_id).outcome_metrics
        store = EngagementStore(manager.db)
        assert store.outcome_for(post_id)["verdict"] == "ambiguous"

    def test_unlinked_post_records_unlinked(self, tmp_path):
        briefs = _brief_store(tmp_path)  # no brief drafts at all
        manager, post_id = self._manager_with_post(
            tmp_path, content="a post no brief drafted")
        report = poll_engagement(manager, now=NOW, brief_store=briefs)
        assert report["outcomes_unlinked"] == 1
        store = EngagementStore(manager.db)
        assert store.outcome_for(post_id)["verdict"] == "unlinked"

    def test_young_post_waits_for_a_fair_age(self, tmp_path):
        briefs = _brief_store(tmp_path)
        draft = _brief_draft(briefs)
        manager, post_id = self._manager_with_post(
            tmp_path, content=draft.content, posted_days=0)
        # Re-seed at 2h old: engagement exists but the outcome
        # stamp waits for the spine's 24h fairness rule.
        report = poll_engagement(manager, now=NOW, brief_store=briefs)
        assert report["outcomes_fed"] == 0
        assert report["outcomes_pending_age"] == 1
        store = EngagementStore(manager.db)
        assert store.outcome_for(post_id) is None


# ── (f) shipped adapters: the survey as structural truth ─────────────


class TestShippedAdapterTruth:
    def test_capability_shapes(self):
        from nomorals.social.adapters.linkedin import LinkedInAdapter
        from nomorals.social.adapters.mastodon import Adapter as M
        from nomorals.social.adapters.personal import TelegramChannelAdapter, WhatsAppStatusAdapter
        from nomorals.social.adapters.x import XAdapter

        assert adapter_supports_engagement(M()) is True
        assert adapter_supports_engagement(XAdapter()) is True
        assert M.supports_reply is True
        assert XAdapter.supports_reply is True
        assert adapter_supports_engagement(LinkedInAdapter()) is False
        assert LinkedInAdapter.supports_reply is False
        for cls in (WhatsAppStatusAdapter, TelegramChannelAdapter):
            assert adapter_supports_engagement(cls(context=None)) \
                is False
            assert cls.supports_reply is False

    def test_platform_truth_reads_structurally(self):
        from nomorals.social.adapters.mastodon import Adapter as M
        from nomorals.social.adapters.x import XAdapter

        truth = platform_truth({"mastodon": M(), "x": XAdapter()})
        assert truth == {
            "mastodon": {"read": True, "reply": True},
            "x": {"read": True, "reply": True},
        }


class TestShippedReaders:
    """The shipped Mastodon/X reads, exercised over fake transports
    (the P3 suites' fake-http shape) — parsing is the contract."""

    def test_mastodon_context_parsing(self):
        import json as _json

        from nomorals.social.adapters import mastodon as mastodon_mod
        from nomorals.social.base import Account

        payload = {
            "ancestors": [],
            "descendants": [
                {"id": "900", "content": "<p>mine</p>",
                 "account": {"acct": "me", "display_name": "Me"},
                 "created_at": "2026-11-09T10:00:00.000Z"},
                {"id": "901", "content": "<p>great &amp; true</p>",
                 "account": {"acct": "ada@social",
                             "display_name": "Ada"},
                 "created_at": "2026-11-09T11:00:00.000Z"},
            ],
        }

        class Resp:
            ok = True
            status = 200
            text = _json.dumps(payload)
            headers = {}

        class Client:
            def get(self, url):
                assert url.endswith("/api/v1/statuses/ext-1/context")
                return Resp()

        adapter = mastodon_mod.Adapter()
        adapter._client = lambda account: Client()
        account = Account(platform="mastodon",
                          handle="me@example.social",
                          credentials="env:UNSET")
        items = adapter.read_engagement(account, "ext-1")
        # His own thread follow-up is skipped; Ada's reply lands
        # with its HTML stripped to text.
        assert [i["item_id"] for i in items] == ["901"]
        assert items[0]["author"] == "Ada"
        assert items[0]["text"] == "great & true"
        assert items[0]["kind"] == "reply"
        assert items[0]["created_at"] > 0

    def test_x_conversation_parsing(self):
        from nomorals.social.adapters.x import XAdapter
        from nomorals.social.base import Account

        payload = {
            "data": [
                {"id": "ext-1", "text": "the post itself",
                 "author_id": "u-me"},
                {"id": "t-1", "text": "a reply",
                 "author_id": "u-ada",
                 "referenced_tweets": [{"type": "replied_to",
                                         "id": "ext-1"}],
                 "created_at": "2026-11-09T11:00:00.000Z"},
                {"id": "t-2", "text": "RT @me the post itself",
                 "author_id": "u-bo",
                 "referenced_tweets": [{"type": "retweeted",
                                         "id": "ext-1"}]},
                {"id": "t-3", "text": "my own follow-up",
                 "author_id": "u-me",
                 "referenced_tweets": [{"type": "replied_to",
                                         "id": "ext-1"}]},
            ],
            "includes": {"users": [
                {"id": "u-me", "username": "me", "name": "Me"},
                {"id": "u-ada", "username": "ada", "name": "Ada"},
                {"id": "u-bo", "username": "bo", "name": "Bo"},
            ]},
        }

        class Resp:
            ok = True
            status = 200
            text = ""

            def json(self):
                return payload

        class Http:
            def __init__(self):
                self.calls = []

            def request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                assert "conversation_id:ext-1" in kwargs["params"]["query"]
                return Resp()

        http = Http()
        adapter = XAdapter(http=http)
        account = Account(platform="x", handle="@me",
                          credentials="tok")
        items = adapter.read_engagement(account, "ext-1")
        # The post itself, the bare RT and his own follow-up are
        # all skipped; only Ada's reply is engagement.
        assert [i["item_id"] for i in items] == ["t-1"]
        assert items[0]["author"] == "Ada"
        assert items[0]["kind"] == "reply"


# ── the /engage surface, driven end to end ───────────────────────────


class TestEngageSurface:
    def _runtime(self, tmp_path, db):
        from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
        from nomorals.social.drafts import DraftQueue

        queue_path = str(tmp_path / "drafts.db")

        class FakeRuntime(RuntimeSystemMixin):
            def __init__(self):
                self.context = SimpleNamespace(db=db)

            def _draft_queue(self):
                return DraftQueue(queue_path)

        return FakeRuntime(), queue_path

    def test_board_show_reply_ignore_flow(self, tmp_path):
        from nomorals.social.drafts import DraftQueue

        db = _db()
        store = EngagementStore(db)
        store.add_item(
            platform="mastodon", post_id="p-1",
            post_external_id="ext-1", account_id="a-1",
            item_external_id="r-1", author="Ada Lovelace",
            text="loved this — will you write the follow-up?",
            kind="reply", arrived_at=NOW - HOUR)
        runtime, queue_path = self._runtime(tmp_path, db)

        board = runtime._control_engage("")
        assert "Ada Lovelace" in board
        assert "needs you" in board
        item_id = store.items()[0]["id"]
        assert item_id not in board

        card = runtime._control_engage("show 1")
        assert "will you write the follow-up?" in card
        assert store.items()[0]["state"] == STATE_TRIAGED

        reply = runtime._control_engage("reply 1")
        assert "reply drafted" in reply
        queue = DraftQueue(queue_path)
        try:
            waiting = queue.pending()
        finally:
            queue.close()
        assert len(waiting) == 1
        assert waiting[0].status == "pending_review"
        assert waiting[0].metadata["engagement_item"] == item_id
        assert waiting[0].metadata["reply_to"] == "ext-1"
        assert waiting[0].metadata["drafter"] == "template"

        done = runtime._control_engage("ignore 1")
        assert "ignored" in done
        assert board_items(store) == []
        usage = runtime._control_engage("frobnicate")
        assert usage.startswith("usage: /engage")


# ── (g) the job ──────────────────────────────────────────────────────


class TestJobRegistration:
    def test_ensure_registers_routine_6h_tool_job(self):
        from nomorals.social.jobs import ENGAGEMENT_POLL_JOB, ensure_social_engagement_jobs

        class FakeScheduler:
            def __init__(self):
                self.jobs = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"id": f"job-{len(self.jobs)}", "name": name,
                       "spec": spec, "kind": kind, "payload": payload,
                       "routine": routine}
                self.jobs.append(job)
                return job

        sched = FakeScheduler()
        out = ensure_social_engagement_jobs(sched)
        assert out[0]["scheduled"] is True
        job = sched.jobs[0]
        assert job["name"] == ENGAGEMENT_POLL_JOB
        assert job["spec"] == "every 6h"
        assert job["routine"] is True
        assert job["payload"] == {
            "tool": "social_engagement_poll", "args": {}}
        again = ensure_social_engagement_jobs(sched)
        assert again[0]["already_scheduled"] is True
        assert len(sched.jobs) == 1
