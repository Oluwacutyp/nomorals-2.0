"""Channels (§16/P4a) + leaderboards (§14).

Pins:
- a channel is claimed like a group (bot-admin stamp / owner
  designation); an UNCLAIMED channel's queue never posts;
- the posting queue posts due entries on the cycle through a
  send callback — one-shot consumed, daily/weekly advanced to
  their next local occurrence, cadence floor respected, a
  failed send keeps the entry due and journals the failure;
- every post is journaled with its source; channel posts are
  typography announcements carrying the stamp;
- triggered posts: a season podium in a linked, claimed group
  cross-posts into a claimed channel; an unclaimed source
  group fires nothing;
- analytics honesty: no samples = "cannot read" (subscribers
  None), a sampled 0 is a real zero; count sampling is
  throttled; growth deltas come from samples only;
- boards per scope × metrics from seeded state; movement
  markers (▲/▼/—/new) read the last PUBLISHED snapshot and a
  peek never advances it; snapshots bounded;
- cross-platform stitch: one username on two platforms ranks
  once, platforms noted, values summed;
- reputation is never a board metric;
- board publishing rides the message cycle at the group's
  hour and holds through silent hours like other cycle posts;
- the console lists channels beside groups and runs their
  queue/post/analytics/settings sections; /boards renders by
  name; a group turn drives due channel posts (no daemon).

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; RuntimeGroupMixin stubs like the console suite.
"""

from __future__ import annotations

import calendar
import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

G = "telegram-bot:-100111"          # the group of record
C = "telegram-bot:-100222"          # a channel Devon admins
C2 = "telegram:-100333"             # a foreign channel (userbot)
D = "discord:777"                   # a discord group
KEY = G

NOW = time.time()


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _seed(surface, uid, name, username, amount, *, day0=0,
          key=G):
    """Real sparks through the real faucet (presence verifies;
    chunks spread over past days under the earn caps)."""
    surface.handle_message(key, uid, username=username,
                           name=name, text="hello group")
    base = time.time() - day0 * 86400
    remaining, step = amount, 0
    while remaining > 0:
        chunk = min(150, remaining)
        surface.game_award(key, uid, chunk, "ttt",
                           now=base - step * 2 * 86400)
        remaining -= chunk
        step += 1
    return surface.economy.balance(key, uid)


class _Send:
    """A fake channel transport: records posts, can fail."""

    def __init__(self, fail=False):
        self.sent: list = []
        self.fail = fail

    def __call__(self, text, entry):
        if self.fail:
            return False, "boom"
        self.sent.append((text, dict(entry)))
        return True, f"m{len(self.sent)}"


def _claimed_channel(surface, key=C, title="Gold Signals"):
    surface.note_channel_seen(key, title=title)
    surface.note_bot_admin(key, True)
    assert surface.claim_state(key)["claimed"]
    return key


# ── channels: claim + queue + cadence ────────────────────────────


def test_channel_claim_and_queue_post_on_cadence(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    send = _Send()
    surface.channel_enqueue(C, {
        "kind": "once", "text": "gold is moving",
        "due": NOW - 5, "source": "dm", "created_by": "99"})
    surface.channel_enqueue(C, {
        "kind": "once", "text": "not yet",
        "due": NOW + 3600, "source": "dm"})
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1 and len(send.sent) == 1
    assert posts[0]["source"] == "dm"
    assert posts[0]["kind"] == "once"
    # a posted one-shot leaves the queue; the future one stays
    queue = surface.channel_queue(C)
    assert len(queue) == 1 and queue[0]["text"] == "not yet"
    # the post journal carries the record (audit + posts ring)
    kinds = [a["kind"] for a in surface.audit(C, limit=50)]
    assert "channel_post" in kinds and "channel_queued" in kinds
    analytics = surface.channel_analytics(C)
    assert analytics["posts"] == 1
    # a second cycle posts nothing new
    assert surface.channel_cycle(C, now=NOW + 60, send=send,
                                 force=True) == []


def test_channel_post_is_stamped_announcement(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    send = _Send()
    surface.channel_enqueue(C, {
        "kind": "once", "text": "gold is moving", "due": NOW - 5,
        "source": "dm"})
    surface.channel_cycle(C, now=NOW, send=send, force=True)
    text = send.sent[0][0]
    assert "gold is moving" in text
    assert "Powered by peacethefirst" in text  # presented surface


def test_unclaimed_channel_refuses(tmp_path):
    # a USERBOT-side channel: the account merely sits there, so
    # nothing is claimed until the owner designates it (§33 —
    # a telegram-bot key would be claimed by the bot's own
    # presence, like a bot group)
    surface = _surface(tmp_path)
    surface.note_channel_seen(C2, title="Foreign")
    send = _Send()
    surface.channel_enqueue(C2, {
        "kind": "once", "text": "should never post",
        "due": NOW - 5})
    assert surface.channel_cycle(C2, now=NOW, send=send,
                                 force=True) == []
    assert send.sent == []
    # designation claims it — then the same queue posts
    surface.set_designated(C2, True, by="owner")
    posts = surface.channel_cycle(C2, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1 and len(send.sent) == 1


def test_channel_cadence_floor_spaces_posts(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    surface.update_settings(C, channel={
        "cadence_min": 60, "links": {}, "analytics": True})
    send = _Send()
    for i in range(2):
        surface.channel_enqueue(C, {
            "kind": "once", "text": f"post {i}", "due": NOW - 5,
            "source": "dm"})
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1  # the floor holds the second back
    posts = surface.channel_cycle(C, now=NOW + 30 * 60,
                                  send=send, force=True)
    assert posts == []      # still inside the hour
    posts = surface.channel_cycle(C, now=NOW + 61 * 60,
                                  send=send, force=True)
    assert len(posts) == 1


def test_channel_recurring_posts_advance_due(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    send = _Send()
    surface.channel_enqueue(C, {
        "kind": "daily", "text": "morning read", "time": "08:00",
        "due": NOW - 5, "source": "dm"})
    surface.channel_enqueue(C, {
        "kind": "weekly", "text": "week ahead", "time": "09:00",
        "weekday": 0, "due": NOW - 5, "source": "dm"})
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 2
    queue = {e["kind"]: e for e in surface.channel_queue(C)}
    # recurring entries stay queued, due moved past now
    assert queue["daily"]["due"] > NOW
    assert queue["daily"]["due"] - NOW <= 86400 + 60
    assert queue["weekly"]["due"] > NOW
    assert queue["weekly"]["due"] - NOW <= 7 * 86400 + 60
    parts = time.gmtime(queue["weekly"]["due"])
    assert parts.tm_wday == 0  # next run lands on a Monday


def test_channel_failed_send_keeps_entry_due(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    surface.channel_enqueue(C, {
        "kind": "once", "text": "retry me", "due": NOW - 5,
        "source": "dm"})
    posts = surface.channel_cycle(C, now=NOW, send=_Send(True),
                                  force=True)
    assert posts == []
    assert len(surface.channel_queue(C)) == 1  # still due
    kinds = [a["kind"] for a in surface.audit(C, limit=50)]
    assert "channel_post_failed" in kinds
    # and a later healthy cycle delivers it
    send = _Send()
    posts = surface.channel_cycle(C, now=NOW + 60, send=send,
                                  force=True)
    assert len(posts) == 1


def test_trigger_season_podium_crossposts(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    surface.update_settings(C, channel={
        "cadence_min": 0, "analytics": True,
        "links": {G: ["season_podium"]}})
    _seed(surface, "55", "Bola", "bola", 300)
    econ = surface.economy
    from nomorals.community.group_economy import season_key

    old = time.time() - 40 * 7 * 86400
    econ.award(G, "55", 50, "old season work", now=old)
    extras = surface._extras(G)
    extras["season_seen"] = season_key(old)
    note = surface.season_cycle(G, now=time.time())
    assert note and "champions" in note
    queue = surface.channel_queue(C)
    assert len(queue) == 1
    assert queue[0]["kind"] == "trigger"
    assert queue[0]["trigger"] == "season_podium"
    assert queue[0]["source_group"] == G
    send = _Send()
    posts = surface.channel_cycle(C, now=time.time(), send=send,
                                  force=True)
    assert len(posts) == 1
    assert posts[0]["source"] == "trigger:season_podium"
    assert "champions" in send.sent[0][0]
    assert "Powered by peacethefirst" in send.sent[0][0]


def test_trigger_respects_link_and_source_claim(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    # no link for war_result on this channel: a war trigger
    # from the group lands nowhere
    surface._channel_trigger(G, "war_result", "war text",
                             now=NOW)
    assert surface.channel_queue(C) == []
    # an unclaimed (userbot) source fires nothing even when
    # linked — §33 applies to the source group
    user_key = "telegram:-100999"
    surface.update_settings(C, channel={
        "cadence_min": 0, "analytics": True,
        "links": {user_key: ["milestone"]}})
    surface._channel_trigger(user_key, "milestone", "a record",
                             now=NOW)
    assert surface.channel_queue(C) == []


# ── analytics honesty ────────────────────────────────────────────


def test_analytics_absence_is_not_zero(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    a = surface.channel_analytics(C)
    assert a["subscribers"] is None       # cannot read ≠ 0
    assert a["samples"] == 0
    assert a["views_total"] is None       # views not exposed
    # a real sample that reads zero IS zero
    assert surface.note_channel_count(C, 0, now=NOW)
    a = surface.channel_analytics(C)
    assert a["subscribers"] == 0 and a["samples"] == 1


def test_analytics_sampling_throttle_and_deltas(tmp_path):
    surface = _surface(tmp_path)
    _claimed_channel(surface)
    assert surface.note_channel_count(C, 100, now=NOW)
    # inside the sample gap, another read is not recorded
    assert not surface.note_channel_count(C, 101, now=NOW + 60)
    assert surface.note_channel_count(C, 130, now=NOW + 7 * 3600)
    assert surface.note_channel_count(C, 150,
                                      now=NOW + 25 * 3600)
    a = surface.channel_analytics(C)
    assert a["subscribers"] == 150 and a["samples"] == 3
    assert a["delta_24h"] == 50           # 150 vs the 100 base
    assert a["delta_7d"] == 50


# ── boards: scopes × metrics ─────────────────────────────────────


def test_group_boards_spread(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, "55", "Bola", "bola", 500)
    _seed(surface, "66", "Ada", "ada", 300)
    rows = surface.board_rows("group:sparks", group_key=G)
    assert [r["label"] for r in rows] == ["Bola", "Ada"]
    assert rows[0]["value"] > rows[1]["value"]
    # messages: presence + activity counted real messages
    surface.handle_message(G, "55", username="bola", name="Bola",
                           text="one more")
    rows = surface.board_rows("group:messages", group_key=G)
    by_label = {r["label"]: r["value"] for r in rows}
    assert by_label["Bola"] == 2 and by_label["Ada"] == 1
    # gems ride the estate organ
    surface.estate.credit_gems(G, "66", 40, source="season",
                               reason="seed", now=NOW)
    rows = surface.board_rows("group:gems", group_key=G)
    assert [(r["label"], r["value"]) for r in rows] == [("Ada", 40)]
    # reviewed contributions read the counted tally only
    members = surface.store.group(G)["members"]
    members["55"]["contributions"] = {"counted": 7}
    members["66"]["contributions"] = {"counted": 3}
    rows = surface.board_rows("group:contributions", group_key=G)
    assert [(r["label"], r["value"]) for r in rows] == [
        ("Bola", 7), ("Ada", 3)]
    # gangs rank by the conflict organ's strength
    cfg = dict(surface.store.settings(G)["conflict"])
    cfg["mode"] = "crime"
    surface.update_settings(G, conflict=cfg)
    out = surface.gang_create(G, "55", "Bola Crew", "street",
                              now=NOW)
    assert out["ok"], out
    rows = surface.board_rows("group:gangs", group_key=G)
    assert rows and rows[0]["label"] == "Bola Crew"
    assert rows[0]["value"] > 0


def test_board_movement_marks_and_peek_never_publishes(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, "55", "Bola", "bola", 500)
    _seed(surface, "66", "Ada", "ada", 300)
    text = surface.render_board("group:sparks", group_key=G,
                                publish=True)
    assert "▲" not in text and "new" not in text.split("Ada")[0]
    # Ada passes Bola; a third member appears
    _seed(surface, "66", "Ada", "ada", 400, day0=30)
    _seed(surface, "77", "Cy", "cy", 100, day0=30)
    text = surface.render_board("group:sparks", group_key=G)
    assert "▲1" in text and "▼1" in text and "new" in text, text
    # the peek moved nothing: the same marks render again
    text2 = surface.render_board("group:sparks", group_key=G)
    assert "▲1" in text2 and "new" in text2
    # publishing settles the baseline — next peek is steady
    surface.render_board("group:sparks", group_key=G,
                         publish=True)
    text3 = surface.render_board("group:sparks", group_key=G)
    assert "▲" not in text3 and "▼" not in text3


def test_board_snapshots_bounded(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, "55", "Bola", "bola", 100)
    for i in range(7):
        surface.render_board("group:sparks", group_key=G,
                             publish=True, now=NOW + i)
    book = surface._board_book()
    snaps = book.data["boards"]["group:sparks"]["snapshots"]
    assert len(snaps) == 4  # SNAPSHOTS_KEPT
    assert book.data["boards"]["group:sparks"]["published"] == 7


def test_cross_platform_stitch_ranks_once(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(G, "55", username="bola", name="Bola",
                           text="tg one")
    surface.handle_message(G, "55", username="bola", name="Bola",
                           text="tg two")
    surface.handle_message(D, "900", username="bola",
                           name="Bola D", text="dc one")
    surface.handle_message(D, "901", username="solo",
                           name="Solo", text="dc two")
    rows = surface.board_rows("global:messages")
    bola = [r for r in rows if "Bola" in r["label"]]
    assert len(bola) == 1                      # ranked ONCE
    assert bola[0]["value"] == 3               # summed
    assert "telegram-bot" in bola[0]["note"]
    assert "discord" in bola[0]["note"]        # platforms noted
    # platform scope counts only that platform's rooms
    rows = surface.board_rows("platform:discord:messages")
    assert {r["label"] for r in rows} == {"Bola D", "Solo"}
    assert all("telegram" not in r["note"] for r in rows)


def test_community_board_reads_the_tag(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, "55", "Bola", "bola", 200)
    surface.update_settings(G, community_id="gold")
    rows = surface.board_rows("community:gold:sparks")
    assert rows and rows[0]["label"] == "Bola"
    # an untagged community is the honest empty board
    rows = surface.board_rows("community:other:sparks")
    assert rows == []
    text = surface.render_board("community:other:sparks")
    assert "no standings yet" in text


def test_game_boards_from_provider_rows(tmp_path):
    surface = _surface(tmp_path)
    games_rows = [
        {"player_key": "telegram-bot:55", "name": "Bola",
         "platform": "telegram-bot", "coins": 900, "xp": 1200,
         "level": 5, "wins": 9, "mastery": 40,
         "per_game": {"chess": {"wins": 4, "points": 300,
                                "played": 6, "elo": 1450,
                                "elo_games": 6}}},
        {"player_key": "telegram-bot:66", "name": "Ada",
         "platform": "telegram-bot", "coins": 100, "xp": 300,
         "level": 2, "wins": 2, "mastery": 5,
         "per_game": {"chess": {"wins": 1, "points": 50,
                                "played": 2, "elo": 1000,
                                "elo_games": 0}}},
    ]
    rows = surface.board_rows("games:xp", games_rows=games_rows)
    assert [r["label"] for r in rows] == ["Bola", "Ada"]
    assert rows[0]["sub"] == "level 5"
    # ELO ranks only rated players — unrated is not a standing
    rows = surface.board_rows("game:chess:elo",
                              games_rows=games_rows)
    assert [(r["label"], r["value"]) for r in rows] == [
        ("Bola", 1450)]
    rows = surface.board_rows("game:chess:wins",
                              games_rows=games_rows)
    assert [r["label"] for r in rows] == ["Bola", "Ada"]
    # without a provider the board is honestly unavailable
    assert surface.board_rows("games:xp") is None
    text = surface.render_board("games:xp",
                                games_rows=games_rows)
    assert "Bola" in text and "Powered by peacethefirst" in text


def test_reputation_is_never_a_board(tmp_path):
    surface = _surface(tmp_path)
    from nomorals.social.chat.group_boards import parse_board_id

    assert parse_board_id("global:reputation") is None
    assert parse_board_id("group:flags") is None
    assert surface.render_board("group:reputation",
                                group_key=G) is None
    assert surface.render_board("nonsense", group_key=G) is None


# ── boards: the publish beat + silent hours ──────────────────────


def _ts(y, mo, d, h=0, mi=0):
    return float(calendar.timegm((y, mo, d, h, mi, 0, 0, 0, 0)))


def test_boards_cycle_publishes_on_cadence(tmp_path):
    surface = _surface(tmp_path)
    _seed(surface, "55", "Bola", "bola", 200)
    surface.update_settings(G, boards={
        "daily": ["sparks"], "weekly": [], "hour": 0})
    day = _ts(2023, 11, 14, 12, 0)
    result = surface.handle_message(
        G, "55", username="bola", name="Bola", text="beat",
        message_id="b1", ts=day)
    notices = "\n".join(result.get("notices") or [])
    assert "sparks held" in notices
    # same local day: the beat does not fire twice
    result = surface.handle_message(
        G, "55", username="bola", name="Bola", text="again",
        message_id="b2", ts=day + 600)
    assert "sparks held" not in "\n".join(
        result.get("notices") or [])
    # next local day: it fires again
    result = surface.handle_message(
        G, "55", username="bola", name="Bola", text="next day",
        message_id="b3", ts=day + 86400)
    assert "sparks held" in "\n".join(result.get("notices") or [])


def test_boards_cycle_holds_through_silent_hours(tmp_path):
    surface = _surface(tmp_path)
    # seed inside the test's own 2023 ts-space: the rhythm
    # cycle's throttle reads real message timestamps, and a
    # 2026-dated seed would park the 2023 cycle as "just ran"
    base = _ts(2023, 11, 10, 12, 0)
    surface.handle_message(G, "55", username="bola", name="Bola",
                           text="hello group", message_id="s0",
                           ts=base)
    remaining, step = 200, 0
    while remaining > 0:
        chunk = min(150, remaining)
        surface.game_award(G, "55", chunk, "ttt",
                           now=base - step * 2 * 86400)
        remaining -= chunk
        step += 1
    surface.update_settings(G, boards={
        "daily": ["sparks"], "weekly": [], "hour": 20})
    rhy = dict(surface.store.settings(G)["rhythm"])
    rhy.update(silent_enabled=True, silent_start="22:00",
               silent_end="07:00", tz_offset_min=0)
    surface.update_settings(G, rhythm=rhy)
    night = _ts(2023, 11, 14, 22, 30)
    result = surface.handle_message(
        G, "55", username="bola", name="Bola", text="late one",
        message_id="n1", ts=night)
    assert "sparks held" not in "\n".join(
        result.get("notices") or [])
    held = surface._extras(G).get("rhythm_held") or []
    assert any("sparks held" in str(h.get("text") or "")
               for h in held)
    morning = _ts(2023, 11, 15, 7, 30)
    result = surface.handle_message(
        G, "55", username="bola", name="Bola", text="morning",
        message_id="n2", ts=morning)
    digest = "\n".join(result.get("notices") or [])
    assert "While the room was quiet" in digest
    assert "sparks held" in digest


# ── runtime: console channels, drive, /boards ────────────────────


class _Chat:
    def __init__(self, key=G, platform="telegram-bot",
                 kind="group", title="Builders"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""
        self.ts = kw.get("ts", 0.0)

    def first_media(self):
        return None


class _Adapter:
    """Gate-suite shape: bot standing + human admins; count
    read only when the test gives the adapter one."""

    def __init__(self, admin_ids=("99",),
                 bot_status="administrator", count=None):
        self.calls = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self._count = count

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def admin_member_count(self, chat):
        if self._count is None:
            return {"ok": False, "error": "no counts here"}
        return {"ok": True, "count": self._count}


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        return SimpleNamespace(ok=True, message_id="1",
                               error="")

    def send_file(self, platform, chat, path, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)),
            f"FILE {path} :: {kw.get('caption', '')}"))
        return SimpleNamespace(ok=True, message_id="2",
                               error="")


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter,
                 platform="telegram-bot", operator=True):
        self.sent = []
        self.gateway = _Gateway({platform: adapter})
        self._operator = operator
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return self._operator

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _dm_msg(sender_id="99", sender="Boss"):
    chat = _Chat(key="telegram-bot:99", kind="dm", title="")
    return _Msg(chat=chat, sender_id=sender_id, sender=sender,
                sender_username="boss")


def _text(out):
    return getattr(out, "text", out)


def test_console_channel_flow(tmp_path):
    adapter = _Adapter(count=42)
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.note_channel_seen(C, title="Gold Signals")
    surface.store.group(G)  # a group exists beside the channel
    home = _text(rt._control_console("", _dm_msg()))
    assert "Your channels" in home and "Gold Signals" in home
    menu = _text(rt._control_console("c1", _dm_msg()))
    # a telegram-bot channel is claimed by the bot's own
    # presence there (§33 — the same derivation as bot groups)
    assert "claimed (bot)" in menu
    # designation is the owner's extra stamp; it round-trips
    out = _text(rt._control_console(
        "c1 settings designate on", _dm_msg()))
    assert "designation on" in out
    out = _text(rt._control_console(
        "c1 post now gold update for the week", _dm_msg()))
    assert "posted now" in out
    posted = [t for (_p, k, t) in rt.gateway.sent if k == C]
    assert len(posted) == 1
    assert "gold update for the week" in posted[0]
    assert "Powered by peacethefirst" in posted[0]
    # analytics: the drive sampled the adapter's real count
    out = _text(rt._control_console("c1 analytics", _dm_msg()))
    assert "42" in out and "subscribers" in out
    # queue section: queue then cancel by position
    rt._control_console("c1 post at 23:59 late note", _dm_msg())
    assert len(surface.channel_queue(C)) == 1
    out = _text(rt._control_console("c1 queue cancel 1",
                                    _dm_msg()))
    assert "cancelled" in out
    assert surface.channel_queue(C) == []
    # trigger link by group handle from the same session
    rt._control_console("", _dm_msg())
    out = _text(rt._control_console(
        "c1 settings link g1 season_podium", _dm_msg()))
    assert "linked" in out
    links = surface.channel_cfg(C)["links"]
    assert links[G] == ["season_podium"]


def test_console_channel_analytics_honest_without_counts(tmp_path):
    rt = _Runtime(tmp_path, _BareAdapter())
    surface = rt._group_surface_obj
    surface.note_channel_seen(C, title="Gold Signals")
    rt._control_console("", _dm_msg())
    out = _text(rt._control_console("c1 analytics", _dm_msg()))
    assert "exposes no count" in out
    assert "0 subscribers" not in out


class _BareAdapter:
    """An adapter with no count method at all (WhatsApp /
    Discord shape): analytics must render absence."""

    _bot_id = 4242

    def admin_member(self, chat, user_id):
        status = "administrator" if str(user_id) in (
            "99", "4242") else "member"
        return {"ok": True, "result": {"status": status}}


def test_group_turn_drives_channel_posts(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    _claimed_channel(surface)
    surface.channel_enqueue(C, {
        "kind": "once", "text": "driven by the group",
        "due": time.time() - 5, "source": "dm"})
    chat = _Chat(key=G, kind="group")
    rt.turn(_Msg(chat=chat, sender_id="7", text="hello group"))
    posted = [t for (_p, k, t) in rt.gateway.sent if k == C]
    assert len(posted) == 1
    assert "driven by the group" in posted[0]


def test_channel_message_notes_registry_and_drives(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.set_designated(C, True, by="owner")
    surface.note_channel_seen(C, title="Gold Signals")
    surface.channel_enqueue(C, {
        "kind": "once", "text": "on channel activity",
        "due": time.time() - 5, "source": "dm"})
    chat = _Chat(key=C, kind="channel", title="Gold Signals")
    rt._channel_note_and_drive(_Msg(chat=chat, sender_id="5",
                                    text="a channel post"))
    posted = [t for (_p, k, t) in rt.gateway.sent if k == C]
    assert len(posted) == 1
    assert surface.is_channel(C)


def test_board_games_rows_provider_shape(tmp_path):
    """The runtime provider shapes real PlayerStore profiles
    into board rows (db-less here: ELO/mastery stay zero —
    honest absence, never invented)."""
    rt = _Runtime(tmp_path, _Adapter())
    prof = SimpleNamespace(
        key="telegram-bot:55", name="Bola",
        platform="telegram-bot", coins=900, xp=1200, wins=9,
        per_game={"chess": {"wins": 4, "points": 300,
                            "played": 6}})
    engine = SimpleNamespace(
        store=SimpleNamespace(all=lambda limit=500: [prof]),
        db=None)
    rt._game_engine = lambda: engine
    rows = rt._board_games_rows()
    assert rows[0]["player_key"] == "telegram-bot:55"
    assert rows[0]["level"] >= 1
    assert rows[0]["per_game"]["chess"]["wins"] == 4
    assert rows[0]["per_game"]["chess"]["elo_games"] == 0
    board = surface_rows = rt._group_surface_obj.board_rows(
        "games:xp", games_rows=rows)
    assert board[0]["label"] == "Bola"
    # no engine in the process → no rows, honestly
    rt2 = _Runtime(tmp_path, _Adapter())
    assert rt2._board_games_rows() is None


def test_boards_command_and_gsettings_grammar(tmp_path):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    _seed(surface, "55", "Bola", "bola", 200)
    admin = _Msg(sender_id="99", sender="Boss")
    index = rt._control_boards("", admin)
    assert "this group publishes" in _text(index)
    # Buttons completion repair (2026-11-11): board renders are
    # ActionableMessages now — the index's buttons open the boards
    # it lists, a board's buttons walk its metrics / hop scope.
    assert index.buttons
    board = rt._control_boards("sparks", admin)
    assert "Bola" in _text(board)
    assert board.buttons
    assert "no board called" in _text(rt._control_boards("bogus",
                                                         admin))
    out = rt._control_gsettings("boards daily sparks messages",
                                admin)
    assert "daily boards" in out
    cfg = surface.store.settings(G)["boards"]
    assert cfg["daily"] == ["sparks", "messages"]
    card = _text(rt._control_gsettings("", admin))
    assert "Boards" in card
    # console boards section reads + writes the same block
    rt._control_console("", _dm_msg())
    out = _text(rt._control_console("g1 boards hour 9",
                                    _dm_msg()))
    assert "publish hour" in out
    assert surface.store.settings(G)["boards"]["hour"] == 9
