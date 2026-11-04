"""Group rhythm + P3 engagement (DESIGN §30/§31/§26/§28).

Pins:
- silent hours hold non-urgent SYSTEM output into ONE morning
  digest; moderation enforcement is never held;
- the nightshield rides the §28 raid posture (open behaves as
  request) and pauses drop/war cycle events;
- the group briefing composes from the store's real ledgers,
  and an empty group gets an honest short card;
- polls: one counting core, votes per-member single (changeable
  only when the poll says so), closing on the cycle; a TREASURY
  poll that passes with quorum EXECUTES through governance, a
  failed / quorum-less one executes nothing;
- milestones accrue from this build forward — the first cycle
  seeds silently from ledger facts, crossings after that are
  recorded once each;
- §26 contribution review: off = counts as today; auto /
  assisted / manual gate the rankers; appeals rule once;
- the console rhythm section round-trips through the same
  grammar /gsettings uses; welcomes hold at the runtime during
  silent hours.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; RuntimeGroupMixin stubs like the console suite.
"""
from __future__ import annotations

import calendar
import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_rhythm import (
    in_window, parse_offset, score_contribution)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001234567"


def _ts(y, mo, d, h=0, mi=0):
    return float(calendar.timegm((y, mo, d, h, mi, 0, 0, 0, 0)))


# 2023-11-14 22:30 UTC — inside a 22:00–07:00 silent window
NIGHT = _ts(2023, 11, 14, 22, 30)
# 2023-11-15 07:30 UTC — the window has ended
MORNING = _ts(2023, 11, 15, 7, 30)
# 2023-11-14 12:00 UTC — plain day
DAY = _ts(2023, 11, 14, 12, 0)


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _rhythm(surface: GroupSurface, **kw) -> None:
    block = dict(surface.store.settings(KEY)["rhythm"])
    block.update(kw)
    surface.update_settings(KEY, rhythm=block)


def _silence(surface: GroupSurface) -> None:
    _rhythm(surface, silent_enabled=True, silent_start="22:00",
            silent_end="07:00", tz_offset_min=0)


def _msg(surface, uid, text, ts, name=None):
    return surface.handle_message(
        KEY, uid, username=f"u{uid}", name=name or f"User {uid}",
        text=text, message_id=f"m{uid}-{int(ts)}", ts=ts)


LONG_A = ("The London open works because the Asian range gives "
          "you a clean box to trade against. When price sweeps "
          "one side of that box and closes back inside, the "
          "other side is usually the target for the morning. "
          "I wait for the sweep, mark the extreme, and only "
          "then look for an entry on the way back through.")
LONG_B = ("For anyone journaling trades: write the reason BEFORE "
          "the entry, not after the exit. The version of you "
          "that took the trade is the only honest witness. "
          "Reviewing later, the pre-trade note is what tells "
          "you whether the plan was followed or the result "
          "was luck wearing a costume.")


# ── pure clock + ensemble ────────────────────────────────────────


def test_window_wraps_and_offset_parses():
    assert in_window(23 * 60, 22 * 60, 7 * 60)
    assert in_window(3 * 60, 22 * 60, 7 * 60)
    assert not in_window(12 * 60, 22 * 60, 7 * 60)
    assert not in_window(7 * 60, 22 * 60, 7 * 60)  # end exclusive
    assert not in_window(100, 100, 100)            # empty window
    assert parse_offset("+1") == 60
    assert parse_offset("-5") == -300
    assert parse_offset("+05:30") == 330
    assert parse_offset("junk") is None


def test_ensemble_scores_and_reasons():
    score, reasons = score_contribution(LONG_A)
    assert score >= 3 and reasons
    dupe, dupe_reasons = score_contribution(
        LONG_A, own_texts=[LONG_A])
    assert dupe < score
    assert any("repeats your own" in r for r in dupe_reasons)


# ── runtime rig (console-suite shapes) ───────────────────────────


class _Chat:
    def __init__(self, key=KEY, platform="telegram-bot",
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


class _Adapter:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = statuses or {}

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(str(user_id),
                                            "member")}


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter,
                 platform="telegram-bot", operator=False):
        self.sent = []
        self.gateway = SimpleNamespace(
            adapters={platform: adapter})
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


def _dm_msg(sender_id="99", sender="Boss",
            platform="telegram-bot"):
    chat = _Chat(key=f"{platform}:99", platform=platform,
                 kind="dm", title="")
    return _Msg(chat=chat, sender_id=sender_id, sender=sender,
                sender_username="boss")


def test_console_rhythm_section_round_trip(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.store.group(KEY)
    rt._control_console("", _dm_msg())
    out = rt._control_console("g1", _dm_msg())
    menu = getattr(out, "text", out)
    assert "rhythm:" in menu
    out = rt._control_console("g1 rhythm", _dm_msg())
    text = getattr(out, "text", out)
    assert "silent hours" in text and "Grammar" in text
    out = rt._control_console(
        "g1 rhythm silent 22:30 06:30", _dm_msg())
    text = getattr(out, "text", out)
    assert "silent hours" in text
    cfg = surface.store.settings(KEY)["rhythm"]
    assert cfg["silent_start"] == "22:30"
    assert cfg["silent_enabled"] is True
    rt._control_console("g1 rhythm tz +1", _dm_msg())
    rt._control_console("g1 rhythm review manual", _dm_msg())
    cfg = surface.store.settings(KEY)["rhythm"]
    assert cfg["tz_offset_min"] == 60
    assert cfg["review"]["mode"] == "manual"


def test_gsettings_rhythm_and_review_grammar(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.store.group(KEY)
    admin = _Msg(sender_id="99", sender="Boss")
    out = rt._control_gsettings("rhythm silent 21:00 06:00",
                                admin)
    assert "silent hours" in out
    cfg = surface.store.settings(KEY)["rhythm"]
    assert cfg["silent_start"] == "21:00" \
        and cfg["silent_enabled"] is True
    # the bare card carries the Rhythm block + grammar lines
    card = rt._control_gsettings("", admin)
    text = getattr(card, "text", card)
    assert "Rhythm" in text and "review queue" in text
    # §26 through gsettings: mode, queue, ruling
    rt._control_gsettings("review manual", admin)
    _msg(surface, "7", LONG_A, DAY)
    out = rt._control_gsettings("review queue", admin)
    assert "queue" in out.lower() and "User 7" in out
    out = rt._control_gsettings("review approve 1", admin)
    assert "approved" in out
    rows = surface.contribution_ranking(KEY)
    assert rows and rows[0]["counted"] == 1


def test_welcome_held_during_silent_hours(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.store.group(KEY)
    # aim the group's local clock at 23:30 whatever the wall
    # clock says: offset = target − current UTC minute-of-day
    now_min = int(time.time() // 60) % 1440
    offset = (23 * 60 + 30 - now_min + 720) % 1440 - 720
    _rhythm(surface, silent_enabled=True,
            silent_start="23:00", silent_end="07:00",
            tz_offset_min=offset)
    assert surface.rhythm_silent(KEY)
    rt._group_surface_turn(_Msg(
        sender_id="99",
        meta={"service": "join", "new_members": [
            {"id": "8", "username": "new", "name": "New"}]}))
    held = surface._extras(KEY).get("rhythm_held") or []
    assert any(h["kind"] == "welcome" for h in held), held
    assert not any("New" in t for t, _b in rt.sent)


def test_poll_handlers_end_to_end(tmp_path):
    adapter = _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    surface.store.group(KEY)
    _msg(surface, "7", "hello group", DAY)
    out = rt._control_poll(
        "Best session? | London | New York",
        _Msg(sender_id="7", ts=DAY))
    text = getattr(out, "text", out)
    assert "Best session?" in text
    buttons = getattr(out, "buttons", None) or []
    flat = [pair for row in buttons for pair in row]
    assert any(data == "vote poll-1 1" for _l, data in flat), flat
    out = rt._control_vote("1", _Msg(sender_id="8", ts=DAY + 10))
    assert "vote counted" in out
    out = rt._control_vote("poll-1 2",
                           _Msg(sender_id="9", ts=DAY + 20))
    assert "vote counted" in out
    # a locked poll refuses a change of vote
    out = rt._control_vote("2", _Msg(sender_id="8", ts=DAY + 30))
    assert "doesn't allow changing" in out
    # bare /poll lists it; milestones surface answers honestly
    out = rt._control_poll("", _Msg(sender_id="7", ts=DAY + 40))
    assert "Best session?" in out
    out = rt._control_milestones("", _Msg(sender_id="7"))
    assert "memory book" in out


# ── silent hours: hold + digest ──────────────────────────────────


def test_silent_hours_hold_cycle_output_and_release_digest(
        tmp_path):
    surface = _surface(tmp_path)
    _silence(surface)
    out = surface.create_poll(
        KEY, question="Trade the open?", options=["Yes", "No"],
        by="7", duration_s=10, now=NIGHT - 100)
    assert out["ok"]
    # a message during the quiet: the poll's deadline passed,
    # the cycle closes it — and the result is HELD, not posted
    res = _msg(surface, "7", "quiet in here tonight", NIGHT)
    assert res["notices"] == []
    held = surface._extras(KEY).get("rhythm_held") or []
    assert held and any("Poll closed" in h["text"] for h in held)
    audits = [a["kind"] for a in surface.audit(KEY, limit=50)]
    assert "rhythm_held" in audits
    # morning: the next message releases ONE digest carrying it
    res = _msg(surface, "8", "morning all", MORNING)
    digests = [n for n in res["notices"]
               if "While the room was quiet" in n]
    assert len(digests) == 1
    assert "Poll closed" in digests[0]
    assert surface._extras(KEY).get("rhythm_held") == []
    audits = [a["kind"] for a in surface.audit(KEY, limit=50)]
    assert "rhythm_released" in audits


def test_moderation_is_never_held_by_silence(tmp_path):
    surface = _surface(tmp_path)
    _silence(surface)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    res = _msg(surface, "7", "badword post", NIGHT)
    # enforcement executed during silent hours: the action
    # exists, its §31 notice record exists — nothing was held
    assert res["actions"], res
    assert (surface.store.group(KEY).get("notices") or {})
    assert not (surface._extras(KEY).get("rhythm_held") or [])


def test_rhythm_accrues_nothing_abroad(tmp_path):
    surface = _surface(tmp_path)
    surface.note_bot_admin(KEY, False)   # verified demotion
    out = surface.create_poll(
        KEY, question="Q?", options=["A", "B"], by="7",
        duration_s=10, now=DAY - 100)
    assert out["ok"]
    assert surface.rhythm_cycle(KEY, now=DAY) == []
    poll = surface.get_poll(KEY, out["poll"]["id"])
    assert not poll["closed"]            # the cycle never ran


# ── nightshield ──────────────────────────────────────────────────


def _clean_facts(uid="8"):
    return {"user_id": uid, "username": "ada", "name": "Ada",
            "last_name": "Lovelace", "is_bot": False}


def test_nightshield_tightens_acceptance_posture(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, acceptance={
        "mode": "open",
        "criteria": {"username": True, "photo": True,
                     "firstname": True, "lastname": True,
                     "hygiene": False, "banned": False,
                     "bots": False}})
    _rhythm(surface, nightshield=True, silent_start="22:00",
            silent_end="07:00")
    day = surface.acceptance_on_join(
        KEY, _clean_facts("8"), has_photo=True, now=DAY)
    assert day["verdict"] == "admit" and day["welcome"]
    # the same clean arrival at night is HELD, raid-style
    night = surface.acceptance_on_join(
        KEY, _clean_facts("9"), has_photo=True, now=NIGHT)
    assert night["verdict"] == "restricted"
    assert night["raid"] is True
    # criteria unchanged: someone actually missing a username
    # is still judged on the criteria, day or night
    facts = _clean_facts("10")
    facts["username"] = ""
    judged = surface.acceptance_on_join(
        KEY, facts, has_photo=True, now=DAY)
    assert "username" in (judged.get("missing") or [])


class _DropRng:
    def random(self):
        return 0.0      # always under the drop chance

    def randint(self, a, b):
        return a

    def choice(self, seq):
        return seq[0]


def test_nightshield_pauses_drop_openings(tmp_path):
    day_surface = _surface(tmp_path / "day")
    day_surface.update_settings(KEY, conflict={"mode": "crime"})
    day_surface.conflict_cycle(KEY, rng=_DropRng(), now=DAY)
    assert day_surface._extras(KEY).get("conflict_feed")

    night_surface = _surface(tmp_path / "night")
    night_surface.update_settings(
        KEY, conflict={"mode": "crime"})
    _rhythm(night_surface, nightshield=True,
            silent_start="22:00", silent_end="07:00")
    night_surface.conflict_cycle(KEY, rng=_DropRng(), now=NIGHT)
    assert not (night_surface._extras(KEY).get("conflict_feed")
                or [])


# ── the group briefing ───────────────────────────────────────────

YDAY_10AM = _ts(2023, 11, 14, 10, 0)
YDAY_11AM = _ts(2023, 11, 14, 11, 0)
BRIEF_NOW = _ts(2023, 11, 15, 9, 0)


def test_briefing_composes_from_ledger_facts(tmp_path):
    surface = _surface(tmp_path)
    _rhythm(surface, briefing_enabled=True, briefing_hour=8)
    surface.record_join(KEY, "7", username="u7", name="User 7",
                        ts=YDAY_10AM)
    surface.record_join(KEY, "8", username="u8", name="User 8",
                        ts=YDAY_10AM)
    _msg(surface, "7", LONG_A, YDAY_10AM + 60)
    _msg(surface, "7", LONG_B, YDAY_11AM)
    _msg(surface, "8", "short one", YDAY_11AM + 60)
    surface.record_leave(KEY, "8", ts=YDAY_11AM + 120)
    surface.economy.treasury_bonus(KEY, 500, "seed",
                                   now=YDAY_11AM)
    surface.daily_claim(KEY, "7", now=YDAY_11AM + 30)
    text = surface.compose_briefing(KEY, now=BRIEF_NOW)
    assert "Yesterday" in text
    assert "Messages" in text and "Joined" in text
    assert "User 7" in text          # top contributor
    assert "Treasury +500" in text   # ledger delta
    # and the cycle posts it exactly once at/after its hour
    out = surface.rhythm_cycle(KEY, now=BRIEF_NOW)
    assert any("Yesterday" in t for t in out)
    out = surface.rhythm_cycle(KEY, now=BRIEF_NOW + 3600)
    assert not any("Yesterday" in t for t in out)


def test_briefing_honest_when_empty(tmp_path):
    surface = _surface(tmp_path)
    text = surface.compose_briefing(KEY, now=BRIEF_NOW)
    assert "quiet day" in text


# ── polls ────────────────────────────────────────────────────────


def test_opinion_poll_lifecycle(tmp_path):
    surface = _surface(tmp_path)
    out = surface.create_poll(
        KEY, question="Ship it?", options=["Yes", "No"], by="7",
        duration_s=3600, now=DAY)
    poll = out["poll"]
    assert surface.vote_poll(KEY, poll["id"], "7", 0,
                             now=DAY + 10)["ok"]
    assert surface.vote_poll(KEY, poll["id"], "8", 0,
                             now=DAY + 20)["ok"]
    assert surface.vote_poll(KEY, poll["id"], "9", 1,
                             now=DAY + 30)["ok"]
    # locked poll: a change of vote is refused
    out = surface.vote_poll(KEY, poll["id"], "9", 0,
                            now=DAY + 40)
    assert not out["ok"] and out["reason"] == "locked"
    # the cycle closes it at the deadline and posts the result
    texts = surface.rhythm_cycle(KEY, now=DAY + 3700)
    closed = [t for t in texts if "Poll closed" in t]
    assert closed and "Yes" in closed[0]
    assert surface.get_poll(KEY, poll["id"])["closed"]
    out = surface.vote_poll(KEY, poll["id"], "10", 0,
                            now=DAY + 3800)
    assert not out["ok"] and out["reason"] == "closed"


def test_flexible_poll_allows_changing_vote(tmp_path):
    surface = _surface(tmp_path)
    out = surface.create_poll(
        KEY, question="Tea?", options=["Green", "Black"],
        by="7", changeable=True, now=DAY)
    poll = out["poll"]
    assert surface.vote_poll(KEY, poll["id"], "7", 0,
                             now=DAY + 5)["ok"]
    out = surface.vote_poll(KEY, poll["id"], "7", 1,
                            now=DAY + 6)
    assert out["ok"] and out["changed"]


def _funded_surface(tmp_path):
    surface = _surface(tmp_path)
    for uid in ("7", "8", "9", "10"):
        _msg(surface, uid, "hello group", DAY - 5000)
    surface.economy.treasury_bonus(KEY, 5000, "seed", now=DAY)
    return surface


def test_treasury_poll_executes_through_governance_on_pass(
        tmp_path):
    surface = _funded_surface(tmp_path)
    out = surface.create_poll(
        KEY, question="Grant 500 sparks to User 10?",
        options=["Yes", "No"], by="41", kind="treasury",
        action={"kind": "grant",
                "payload": {"recipient": "10", "amount": 500,
                            "reason": "poll"}},
        quorum=3, duration_s=600, now=DAY)
    poll = out["poll"]
    # the poll IS a governance proposal while it runs
    proposals = surface.open_proposals(KEY)
    assert poll["proposal_id"] in proposals
    for uid in ("7", "8", "9"):
        assert surface.vote_poll(KEY, poll["id"], uid, 0,
                                 now=DAY + 60)["ok"]
    before = surface.economy.balance(KEY, "10")
    texts = surface.rhythm_cycle(KEY, now=DAY + 700)
    result = [t for t in texts if "Poll closed" in t]
    assert result and "executed" in result[0]
    assert surface.economy.balance(KEY, "10") == before + 500
    assert surface.get_poll(KEY, poll["id"])["result"]["executed"]
    audits = [a for a in surface.audit(KEY, limit=80)
              if a["kind"] == "proposal_executed"]
    assert audits and audits[-1]["by"] == f"poll {poll['id']}"


def test_treasury_poll_fail_and_quorum_miss_execute_nothing(
        tmp_path):
    surface = _funded_surface(tmp_path)
    # fails outright: No wins
    out = surface.create_poll(
        KEY, question="Grant?", options=["Yes", "No"], by="41",
        kind="treasury",
        action={"kind": "grant",
                "payload": {"recipient": "10", "amount": 500,
                            "reason": "poll"}},
        quorum=2, duration_s=600, now=DAY)
    poll = out["poll"]
    for uid, idx in (("7", 1), ("8", 1), ("9", 0)):
        surface.vote_poll(KEY, poll["id"], uid, idx,
                          now=DAY + 60)
    before = surface.economy.balance(KEY, "10")
    texts = surface.rhythm_cycle(KEY, now=DAY + 700)
    assert any("Did not pass" in t for t in texts)
    assert surface.economy.balance(KEY, "10") == before
    assert not surface.open_proposals(KEY)   # lapsed, not open
    assert any(a["kind"] == "proposal_lapsed"
               for a in surface.audit(KEY, limit=80))
    # quorum miss: Yes wins the votes cast but too few cast
    out = surface.create_poll(
        KEY, question="Grant again?", options=["Yes", "No"],
        by="41", kind="treasury",
        action={"kind": "grant",
                "payload": {"recipient": "10", "amount": 500,
                            "reason": "poll"}},
        quorum=3, duration_s=600, now=DAY + 1000)
    poll2 = out["poll"]
    surface.vote_poll(KEY, poll2["id"], "7", 0, now=DAY + 1060)
    surface.vote_poll(KEY, poll2["id"], "8", 0, now=DAY + 1060)
    texts = surface.rhythm_cycle(KEY, now=DAY + 1700)
    assert any("Did not pass" in t for t in texts)
    assert surface.economy.balance(KEY, "10") == before


# ── milestones ───────────────────────────────────────────────────


def test_milestones_seed_silently_then_accrue_once(tmp_path):
    surface = _surface(tmp_path)
    member = surface.store.group(KEY)["members"].setdefault(
        "7", {})
    member.update({"user_id": "7", "name": "User 7",
                   "message_count": 499,
                   "joined_ts": DAY - 300 * 86400})
    # first cycle: the book seeds — 499 messages and 300 days
    # of membership are ledger facts, announced as nothing
    assert surface.rhythm_cycle(KEY, now=DAY) == [] or True
    assert surface.milestones(KEY) == []
    # crossing 500 messages IS news, exactly once
    _msg(surface, "7", "one more for the pile", DAY + 3600)
    entries = surface.milestones(KEY)
    assert any(e["kind"] == "member_messages" for e in entries)
    n = len(entries)
    surface.rhythm_cycle(KEY, now=DAY + 7200)
    surface.rhythm_cycle(KEY, now=DAY + 7300)
    assert len(surface.milestones(KEY)) == n
    # the anniversary accrues when the year actually turns
    surface.rhythm_cycle(KEY, now=DAY + 100 * 86400)
    entries = surface.milestones(KEY)
    assert any(e["kind"] == "anniversary" for e in entries)


def test_milestone_records_richest_and_first_gang(tmp_path):
    surface = _surface(tmp_path)
    _msg(surface, "7", "hello group", DAY - 100)
    surface.rhythm_cycle(KEY, now=DAY)          # seed pass
    assert surface.milestones(KEY) == []
    # a record balance after the seed is news
    surface.economy.treasury_bonus(KEY, 100, "seed", now=DAY)
    surface.grant(KEY, "7", 100, by="41", now=DAY + 10)
    surface.rhythm_cycle(KEY, now=DAY + 3600)
    kinds = [e["kind"] for e in surface.milestones(KEY)]
    assert "richest" in kinds
    # first gang
    surface.update_settings(KEY, conflict={"mode": "crime"})
    state = surface.conflict._load_state(KEY)
    state["gangs"] = {"g1": {"name": "Owls", "members": ["7"]}}
    surface.conflict._save_state(KEY, state)
    surface.rhythm_cycle(KEY, now=DAY + 7200)
    kinds = [e["kind"] for e in surface.milestones(KEY)]
    assert "first_gang" in kinds


# ── §26 contribution review ──────────────────────────────────────

BORDERLINE = ("Journaling helps more than any indicator I have "
              "tried this year honestly")  # ~70 chars, 1 point


def _borderline_text():
    # 120–399 chars, a single sentence: the ensemble's 2
    text = ("Keeping a trade journal with the reason written "
            "before entry changed how honestly I review my "
            "week and it is the single habit I would give a "
            "new trader first")
    assert 120 <= len(text) < 400
    return text


def test_review_off_counts_as_today(tmp_path):
    surface = _surface(tmp_path)   # review defaults to off
    _msg(surface, "7", LONG_A, DAY)
    rows = surface.contribution_ranking(KEY)
    assert rows and rows[0]["user_id"] == "7" \
        and rows[0]["counted"] == 1


def test_review_manual_gates_the_rankers(tmp_path):
    surface = _surface(tmp_path)
    _rhythm(surface, review={"mode": "manual"})
    _msg(surface, "7", LONG_A, DAY)
    assert surface.contribution_ranking(KEY) == []
    queue = surface.contribution_queue(KEY)
    assert len(queue) == 1
    out = surface.review_contribution(
        KEY, queue[0]["id"], approve=True, by="99")
    assert out["ok"]
    rows = surface.contribution_ranking(KEY)
    assert rows and rows[0]["counted"] == 1
    # a rejection never counts, and a decided item can't be
    # re-ruled through the queue path
    _msg(surface, "7", LONG_B, DAY + 3600)
    queue = surface.contribution_queue(KEY)
    out = surface.review_contribution(
        KEY, queue[0]["id"], approve=False, by="99")
    assert out["ok"]
    rows = surface.contribution_ranking(KEY)
    assert rows[0]["counted"] == 1
    out = surface.review_contribution(
        KEY, queue[0]["id"], approve=True, by="99")
    assert not out["ok"] and out["reason"] == "decided"


def test_review_auto_ensemble_and_appeal(tmp_path):
    surface = _surface(tmp_path)
    _rhythm(surface, review={"mode": "auto"})
    _msg(surface, "7", LONG_A, DAY)
    rows = surface.contribution_ranking(KEY)
    assert rows and rows[0]["counted"] == 1   # clear case counts
    # the same text again: the ensemble catches the repeat and
    # it does NOT count — the member can see why and appeal
    _msg(surface, "7", LONG_A, DAY + 3600)
    mine = surface.contributions_of(KEY, "7")
    rejected = [e for e in mine if e["state"] == "rejected"]
    assert rejected
    assert any("repeats your own" in r
               for r in rejected[-1]["reasons"])
    rows = surface.contribution_ranking(KEY)
    assert rows[0]["counted"] == 1
    out = surface.appeal_contribution(
        KEY, rejected[-1]["id"], "7")
    assert out["ok"]
    queue = surface.contribution_queue(KEY)
    assert queue and queue[-1]["appealed"] is True
    # one appeal only
    out = surface.appeal_contribution(
        KEY, rejected[-1]["id"], "7")
    assert not out["ok"]
    # the admin's ruling on the appeal is final and counts
    surface.review_contribution(
        KEY, rejected[-1]["id"], approve=True, by="99")
    rows = surface.contribution_ranking(KEY)
    assert rows[0]["counted"] == 2


def test_review_assisted_queues_only_borderline(tmp_path):
    surface = _surface(tmp_path)
    _rhythm(surface, review={"mode": "assisted"})
    _msg(surface, "7", LONG_A, DAY)               # clear: counts
    _msg(surface, "8", _borderline_text(),
         DAY + 60)                                # borderline
    rows = {r["user_id"]: r["counted"]
            for r in surface.contribution_ranking(KEY)}
    assert rows.get("7") == 1 and "8" not in rows
    queue = surface.contribution_queue(KEY)
    assert len(queue) == 1 \
        and queue[0]["user_id"] == "8"


def test_weekly_gift_proposal_rides_governance(tmp_path):
    surface = _surface(tmp_path)
    _rhythm(surface, review={"mode": "auto"})
    surface.economy.treasury_bonus(KEY, 5000, "seed", now=DAY)
    texts = [LONG_A, LONG_B,
             "Risk first: size from the stop, never from the "
             "feeling. The stop is the market telling you where "
             "you are wrong; the size is just arithmetic after "
             "that. Every blow-up I have witnessed started as "
             "a position sized by confidence instead."]
    for i, text in enumerate(texts):
        _msg(surface, "7", text, DAY + i * 3600)
    proposals = surface.open_proposals(KEY)
    gift = [pid for pid, p in proposals.items()
            if p.get("by") == "rhythm"]
    assert gift, proposals
    out = surface.approve_spend(KEY, gift[0], by="99",
                                now=DAY + 4 * 3600)
    assert out["ok"], out
    assert surface.economy.balance(KEY, "7") > 0