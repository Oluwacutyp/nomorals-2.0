"""Join acceptance & admission rules (DESIGN §28).

Pins: every criterion's evaluation (username / photo / first /
last name / name hygiene / previously banned / bots+whitelist),
actions + per-criterion overrides, the open-mode join flow
(admit + welcome / restricted + fix-it note / decline),
request mode (auto-approve clean, auto-decline failing, queue
the borderline, /joinq + buttons deciding, bulk decisions),
the restricted lifecycle (grace expiry for kick actions,
recheck pass/fail), profile-change re-engagement, raid
tightening, the /glog trail, /gsettings display + grammar, and
the honesty rules (unverifiable photo queues or admits-with-
journal, never fails a candidate; no account-age criterion
anywhere).
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_acceptance import (
    ACCEPTANCE_DEFAULTS,
    evaluate,
    name_hygiene_failures,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"
NOW = 1_800_000_000.0


# ── fakes ─────────────────────────────────────────────────────


class _BotAdapter:
    """Bot API surface: admin actions + join requests + photo."""

    name = "telegram-bot"

    def __init__(self, *, photo=None, photo_ok=True):
        self.calls: list = []
        self._photo = photo
        self._photo_ok = photo_ok

    def _out(self, name, *args, **kw):
        self.calls.append((name, args, kw))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        return {"status": "administrator"}

    def admin_restrict(self, chat, user_id, *, until=0, **perms):
        return self._out("restrict", user_id, until=until)

    def admin_unrestrict(self, chat, user_id):
        return self._out("unrestrict", user_id)

    def admin_ban(self, chat, user_id):
        return self._out("ban", user_id)

    def admin_unban(self, chat, user_id):
        return self._out("unban", user_id)

    def admin_delete(self, chat, message_id):
        return self._out("delete", message_id)

    def admin_join_request(self, chat, user_id, approve):
        return self._out("join_request", user_id, approve=approve)

    def admin_member_photo(self, chat, user_id):
        self.calls.append(("photo", (user_id,), {}))
        if not self._photo_ok:
            return {"ok": False, "error": "lookup failed"}
        return {"ok": True, "has_photo": bool(self._photo)}

    def kinds(self):
        return [c[0] for c in self.calls]


def _surface(tmp_path: Path, **acceptance) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if acceptance:
        surface.update_settings(KEY, acceptance=acceptance)
    return surface


def _criteria(**on):
    crit = {name: False for name in (
        "username", "photo", "firstname", "lastname", "hygiene",
        "banned", "bots")}
    crit.update(on)
    return crit


def _facts(**kw):
    facts = {"user_id": "42", "username": "ada",
             "first_name": "Ada", "last_name": "Lovelace",
             "name": "Ada", "is_bot": False, "has_photo": True,
             "previously_banned": False, "whitelisted": False,
             "filter_words": []}
    facts.update(kw)
    return facts


def _cfg(**kw):
    import copy

    cfg = copy.deepcopy(ACCEPTANCE_DEFAULTS)
    cfg.update(kw)
    return cfg


def _audit_kinds(surface, kind):
    return [a for a in surface.audit(KEY, limit=500)
            if a.get("kind") == kind]


def _texts(sent):
    """The text argument of every gateway.send the rig made
    (send(platform, chat, text, buttons=…, parse_mode=…))."""
    return [str(a[2]) for a, _k in sent]


def _plain(out):
    """Unwrap an ActionableMessage reply to its text."""
    return getattr(out, "text", out)


# ── the evaluator ─────────────────────────────────────────────


def test_username_criterion():
    cfg = _cfg(criteria=_criteria(username=True))
    ev = evaluate(cfg, _facts(username=""))
    assert ev.verdict == "restricted"
    assert ev.missing == ["username"]
    assert "set a username" in ev.reasons[0]
    assert evaluate(cfg, _facts()).clean


def test_photo_criterion_and_the_unknown_rule():
    cfg = _cfg(criteria=_criteria(photo=True))
    ev = evaluate(cfg, _facts(has_photo=False))
    assert ev.missing == ["photo"]
    # UNKNOWN is not a failure: it queues for a human.
    ev = evaluate(cfg, _facts(has_photo=None))
    assert ev.missing == [] and ev.unknown == ["photo"]
    assert ev.verdict == "queue"


def test_firstname_emoji_only_fails():
    cfg = _cfg(criteria=_criteria(firstname=True))
    assert evaluate(cfg, _facts(first_name="🎉🎉")).missing == [
        "firstname"]
    assert evaluate(cfg, _facts(first_name="", name="")).missing == [
        "firstname"]
    assert evaluate(cfg, _facts(first_name="Ada")).clean


def test_lastname_criterion():
    cfg = _cfg(criteria=_criteria(lastname=True))
    assert evaluate(cfg, _facts(last_name="")).missing == [
        "lastname"]
    assert evaluate(cfg, _facts()).clean


def test_name_hygiene_links_handles_and_filter_words():
    assert name_hygiene_failures("earn at t.me/freemoney", "",
                                 [])
    assert name_hygiene_failures("Ada @otherbot", "", [])
    assert name_hygiene_failures("Ada", "", ["spamword"]) == []
    assert name_hygiene_failures("spamword fan", "",
                                 ["spamword"])
    cfg = _cfg(criteria=_criteria(hygiene=True))
    ev = evaluate(cfg, _facts(name="visit www.easycash.example",
                              filter_words=[]))
    assert ev.missing == ["hygiene"]


def test_previously_banned_and_bot_criteria():
    cfg = _cfg(criteria=_criteria(banned=True, bots=True))
    ev = evaluate(cfg, _facts(previously_banned=True,
                              ban_reason="spam"))
    assert ev.missing == ["banned"] and "spam" in ev.reasons[0]
    assert evaluate(cfg, _facts(is_bot=True)).missing == ["bots"]
    # A whitelisted bot is a legitimate member.
    assert evaluate(cfg, _facts(is_bot=True,
                                whitelisted=True)).clean


def test_action_resolution_and_overrides():
    cfg = _cfg(criteria=_criteria(username=True, photo=True),
               action="restricted",
               overrides={"photo": "decline"})
    ev = evaluate(cfg, _facts(username="", has_photo=False))
    # Strongest per-criterion action wins.
    assert ev.verdict == "decline" and ev.action == "decline"
    ev = evaluate(cfg, _facts(username="", has_photo=True))
    assert ev.verdict == "restricted"


def test_no_account_age_anywhere():
    assert "age" not in str(ACCEPTANCE_DEFAULTS).lower()
    assert "age" not in str(ACCEPTANCE_DEFAULTS["criteria"])


# ── open-mode joins (surface) ─────────────────────────────────


def test_clean_join_admits_with_welcome(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(username=True))
    out = surface.acceptance_on_join(KEY, _facts(), now=NOW)
    assert out["verdict"] == "admit" and out["welcome"] is True
    assert out["actions"] == []
    assert _audit_kinds(surface, "acceptance_eval")


def test_failing_join_restricted_by_default(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(username=True))
    out = surface.acceptance_on_join(
        KEY, _facts(username=""), now=NOW)
    assert out["verdict"] == "restricted"
    assert out["missing"] == ["username"]
    assert [a["kind"] for a in out["actions"]] == [
        "acceptance_restrict"]
    member = surface.store.group(KEY)["members"]["42"]
    assert member["acceptance"]["state"] == "restricted"
    assert member["acceptance"]["welcome_pending"] is True


def test_kick_action_join_gets_grace_not_instant_kick(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(username=True),
        action="kick", grace_minutes=30)
    out = surface.acceptance_on_join(
        KEY, _facts(username=""), now=NOW)
    assert out["verdict"] == "restricted"
    assert out["pending_kick"] is True
    # Inside the grace: no removal.  Past it: the cycle kicks.
    assert surface.acceptance_cycle(KEY, now=NOW + 60) == []
    actions = surface.acceptance_cycle(KEY, now=NOW + 31 * 60)
    assert [a["kind"] for a in actions] == ["kick"]
    assert _audit_kinds(surface, "acceptance_grace_kick")


def test_decline_join_kicks_and_banned_rebans(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(username=True),
        action="decline")
    out = surface.acceptance_on_join(
        KEY, _facts(username=""), now=NOW)
    assert [a["kind"] for a in out["actions"]] == ["kick"]
    # A previously-banned returner meets the ban again.
    surface2 = _surface(tmp_path / "b", criteria=_criteria(
        banned=True), action="decline")
    surface2.store.group(KEY)["audit"].append({
        "kind": "admin_action", "action": "ban", "user_id": "9",
        "ts": NOW - 100})
    out = surface2.acceptance_on_join(
        KEY, _facts(user_id="9"), now=NOW)
    assert out["missing"] == ["banned"]
    assert [a["kind"] for a in out["actions"]] == ["ban"]
    # …but an unban on the record clears the history.
    surface2.store.group(KEY)["audit"].append({
        "kind": "admin_action", "action": "unban", "user_id": "9",
        "ts": NOW - 50})
    out = surface2.acceptance_on_join(
        KEY, _facts(user_id="9"), now=NOW)
    assert out["verdict"] == "admit"


def test_banlist_counts_as_previously_banned(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(banned=True))
    surface.bans.ban("9", reason="raid spam", group_id=KEY)
    out = surface.acceptance_on_join(
        KEY, _facts(user_id="9"), now=NOW)
    assert out["missing"] == ["banned"]


def test_unverifiable_photo_open_mode_admits_and_journals(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(photo=True))
    out = surface.acceptance_on_join(
        KEY, _facts(), has_photo=None, now=NOW)
    assert out["verdict"] == "admit" and out["welcome"] is True
    assert _audit_kinds(surface, "acceptance_unverifiable")


def test_inert_defaults_stay_inert(tmp_path):
    surface = _surface(tmp_path)   # shipped defaults
    out = surface.acceptance_on_join(
        KEY, _facts(username=""), now=NOW)
    # Hygiene + banned are on by default; a clean name with no
    # ban history passes even with no username configured.
    assert out["verdict"] == "admit"


# ── request mode ──────────────────────────────────────────────


def test_request_clean_auto_approves(tmp_path):
    surface = _surface(
        tmp_path, mode="request", criteria=_criteria(username=True))
    out = surface.acceptance_on_request(KEY, _facts(), now=NOW)
    assert out["verdict"] == "approve"
    admitted = surface.acceptance_request_admitted(
        KEY, out["request_id"], now=NOW)
    assert admitted["welcome"] is True
    assert admitted["restricted"] is False


def test_request_restricted_approves_restricted(tmp_path):
    surface = _surface(
        tmp_path, mode="request", criteria=_criteria(username=True))
    out = surface.acceptance_on_request(
        KEY, _facts(username=""), now=NOW)
    assert out["verdict"] == "approve_restricted"
    admitted = surface.acceptance_request_admitted(
        KEY, out["request_id"], now=NOW)
    assert admitted["restricted"] is True
    member = surface.store.group(KEY)["members"]["42"]
    assert member["acceptance"]["state"] == "restricted"


def test_request_decline_action_declines(tmp_path):
    surface = _surface(
        tmp_path, mode="request", criteria=_criteria(username=True),
        action="decline")
    out = surface.acceptance_on_request(
        KEY, _facts(username=""), now=NOW)
    assert out["verdict"] == "decline"
    assert surface.acceptance_queue(KEY) == []


def test_request_no_auto_approve_queues_for_a_human(tmp_path):
    surface = _surface(
        tmp_path, mode="request", criteria=_criteria(username=True),
        auto_approve=False)
    out = surface.acceptance_on_request(KEY, _facts(), now=NOW)
    assert out["verdict"] == "queued"
    queue = surface.acceptance_queue(KEY)
    assert len(queue) == 1 and queue[0]["id"] == out["request_id"]
    decided = surface.acceptance_decide(
        KEY, out["request_id"], approve=True, by="admin1", now=NOW)
    assert decided["ok"] and decided["execute"] == "join_request"
    assert _audit_kinds(surface, "acceptance_decision")
    # Double-deciding is a no-op, honestly reported.
    again = surface.acceptance_decide(
        KEY, out["request_id"], approve=True, by="admin1", now=NOW)
    assert again["ok"] is False


def test_request_unverifiable_photo_queues(tmp_path):
    surface = _surface(
        tmp_path, mode="request", criteria=_criteria(photo=True))
    out = surface.acceptance_on_request(
        KEY, _facts(), has_photo=None, now=NOW)
    assert out["verdict"] == "queued"


# ── the restricted lifecycle ──────────────────────────────────


def _restricted_surface(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(username=True))
    surface.acceptance_on_join(KEY, _facts(username=""), now=NOW)
    return surface


def test_recheck_fail_then_pass_releases(tmp_path):
    surface = _restricted_surface(tmp_path)
    still = surface.acceptance_recheck(KEY, "42", now=NOW + 60)
    assert still["state"] == "still"
    assert still["missing"] == ["username"]
    # The member sets a username; the ledger learns it on their
    # next identity observation, and the recheck taps it fresh.
    passed = surface.acceptance_recheck(
        KEY, "42", username="ada", now=NOW + 120)
    assert passed["state"] == "passed"
    assert passed["welcome"] is True
    assert passed["was_restricted"] is True
    member = surface.store.group(KEY)["members"]["42"]
    assert "acceptance" not in member
    kinds = [a for a in surface.audit(KEY, limit=500)
             if a.get("kind") == "acceptance_recheck"]
    assert [k["passed"] for k in kinds] == [False, True]


def test_recheck_when_not_restricted(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(username=True))
    out = surface.acceptance_recheck(KEY, "42", now=NOW)
    assert out["state"] == "not_restricted"


def test_admin_release_via_queue_hold(tmp_path):
    # A raid hold decides through the same queue machinery.
    surface = _surface(tmp_path, criteria=_criteria())
    for i in range(8):
        surface.acceptance_on_join(
            KEY, _facts(user_id=str(100 + i),
                        username=f"u{i}"), now=NOW + i)
    queue = surface.acceptance_queue(KEY)
    assert queue and queue[-1]["kind"] == "hold"
    hold = queue[-1]
    decided = surface.acceptance_decide(
        KEY, hold["id"], approve=True, by="admin1", now=NOW + 100)
    assert decided["execute"] == "release"
    assert decided["record"]["welcome"] is True
    member = surface.store.group(KEY)["members"][hold["user_id"]]
    assert "acceptance" not in member


# ── raid tightening ───────────────────────────────────────────


def test_raid_holds_clean_joins_and_shrinks_grace(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(username=True),
        grace_minutes=60, raid_burst=4, raid_window_s=120,
        raid_grace_minutes=5)
    for i in range(3):
        out = surface.acceptance_on_join(
            KEY, _facts(user_id=str(200 + i), username=f"p{i}"),
            now=NOW + i * 10)
        assert out["verdict"] == "admit" and out["raid"] is False
    held = surface.acceptance_on_join(
        KEY, _facts(user_id="299", username="late"), now=NOW + 40)
    assert held["raid"] is True
    assert held["verdict"] == "restricted"
    assert held["reasons"] and "join rush" in held["reasons"][0]
    cfg, raid = surface.acceptance_cfg(KEY, now=NOW + 50)
    assert raid is True
    assert cfg["auto_approve"] is False
    assert cfg["grace_minutes"] == 5
    # A raid hold does NOT self-release via recheck mid-raid.
    still = surface.acceptance_recheck(
        KEY, "299", username="late", now=NOW + 60)
    assert still["state"] == "still"


# ── profile-change re-engagement ──────────────────────────────


def test_username_drop_warns_then_restricts(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(username=True),
        grace_minutes=30)
    surface.acceptance_on_join(KEY, _facts(), now=NOW)
    # The member messages with a live username: nothing.
    assert surface.acceptance_profile_check(
        KEY, "42", username="ada", now=NOW + 10) is None
    # …then drops it: warning + grace.
    event = surface.acceptance_profile_check(
        KEY, "42", username="", now=NOW + 20)
    assert event and event["warn"] is True
    # Restoring it inside the window clears silently.
    back = surface.acceptance_profile_check(
        KEY, "42", username="ada2", now=NOW + 30)
    assert back == {"recovered": True}
    # Drop again and let the grace run out: restriction engages.
    surface.acceptance_profile_check(KEY, "42", username="",
                                     now=NOW + 40)
    actions = surface.acceptance_cycle(KEY, now=NOW + 40 + 31 * 60)
    assert [a["kind"] for a in actions] == ["acceptance_restrict"]
    member = surface.store.group(KEY)["members"]["42"]
    assert member["acceptance"]["state"] == "restricted"
    assert member["acceptance"]["via"] == "profile"


def test_profile_check_exempts_admins_and_never_had(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria(username=True))
    surface.acceptance_on_join(KEY, _facts(), now=NOW)
    assert surface.acceptance_profile_check(
        KEY, "42", username="", privileged=True, now=NOW) is None
    # A member who never had a username on record: the drop
    # can't be told from absence — no engagement.
    surface.acceptance_on_join(
        KEY, _facts(user_id="77", username=""), now=NOW)
    assert surface.acceptance_profile_check(
        KEY, "77", username="", now=NOW) is None


def test_cycle_holds_kicks_while_autonomy_off(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(username=True),
        action="kick", grace_minutes=10)
    surface.acceptance_on_join(KEY, _facts(username=""), now=NOW)
    assert surface.acceptance_cycle(
        KEY, now=NOW + 11 * 60, enforce=False) == []
    expired = _audit_kinds(surface, "acceptance_grace_expired")
    assert expired and expired[0]["suppressed"] == "autonomy_off"
    # Audited once, not per message.
    assert surface.acceptance_cycle(
        KEY, now=NOW + 12 * 60, enforce=False) == []
    assert len(_audit_kinds(surface,
                            "acceptance_grace_expired")) == 1
    # Autonomy back: the held kick lands.
    actions = surface.acceptance_cycle(KEY, now=NOW + 13 * 60)
    assert [a["kind"] for a in actions] == ["kick"]


# ── runtime wiring ────────────────────────────────────────────


def _msg(*, sender_id="admin1", sender="Admin",
         sender_username="boss", meta=None, key=KEY):
    chat_id = key.split(":", 1)[1] if ":" in key else ""
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform="telegram-bot",
                             kind="group", title="Builders",
                             chat_id=chat_id),
        sender_id=sender_id, sender=sender,
        sender_username=sender_username,
        text="", meta=meta or {}, reply_to="", message_id="900")


def _rig(tmp_path: Path, *, bot=None, acceptance=None):
    from nomorals.agents.partner.runtime_group import \
        RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if acceptance:
        surface.update_settings(KEY, acceptance=acceptance)
    bot = bot if bot is not None else _BotAdapter()
    sent: list = []
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore
    mixin._group_adapter = (  # type: ignore
        lambda platform: {"telegram-bot": bot}.get(platform))
    mixin.gateway = SimpleNamespace(  # type: ignore
        send=lambda *a, **k: sent.append((a, k)),
        adapters={"telegram-bot": bot})
    return mixin, surface, bot, sent


def _join_msg(members):
    return _msg(meta={"service": "join", "new_members": members})


def test_runtime_clean_join_welcomes(tmp_path):
    mixin, surface, bot, sent = _rig(
        tmp_path, acceptance={"criteria": _criteria(username=True)})
    mixin._group_surface_turn(_join_msg([
        {"id": "42", "username": "ada", "name": "Ada",
         "first_name": "Ada", "last_name": "L", "is_bot": False}]))
    texts = _texts(sent)
    assert any("Ada" in t for t in texts)
    assert "restrict" not in bot.kinds()


def test_runtime_restricted_join_restricts_and_notes(tmp_path):
    mixin, surface, bot, sent = _rig(
        tmp_path, acceptance={"criteria": _criteria(username=True)})
    mixin._group_surface_turn(_join_msg([
        {"id": "42", "username": "", "name": "Ada",
         "first_name": "Ada", "last_name": "", "is_bot": False}]))
    assert ("restrict", ("42",), {"until": 0}) in bot.calls
    texts = _texts(sent)
    assert any("set a username" in t for t in texts)
    buttons = [k.get("buttons") for _a, k in sent if k.get("buttons")]
    assert any(b and b[0][0][1] == "recheck" for b in buttons)


def test_runtime_inert_defaults_take_the_legacy_path(tmp_path):
    mixin, surface, bot, sent = _rig(tmp_path)   # no acceptance cfg
    mixin._group_surface_turn(_join_msg([
        {"id": "42", "username": "", "name": "Ada"}]))
    texts = _texts(sent)
    assert any("Ada" in t for t in texts)      # welcomed as ever
    assert _audit_kinds(surface, "acceptance_eval") == []


def test_runtime_join_request_auto_approve(tmp_path):
    bot = _BotAdapter()
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot,
        acceptance={"mode": "request",
                    "criteria": _criteria(username=True)})
    msg = _msg(meta={"service": "join_request", "requester": {
        "id": "42", "username": "ada", "name": "Ada",
        "first_name": "Ada", "last_name": "L", "is_bot": False}})
    mixin._group_surface_turn(msg)
    assert ("join_request", ("42",), {"approve": True}) in bot.calls
    texts = _texts(sent)
    assert any("Ada" in t for t in texts)      # the welcome fired


def test_runtime_join_request_decline_and_queue(tmp_path):
    bot = _BotAdapter()
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot,
        acceptance={"mode": "request", "action": "decline",
                    "criteria": _criteria(username=True)})
    msg = _msg(meta={"service": "join_request", "requester": {
        "id": "42", "username": "", "name": "Ada",
        "first_name": "Ada", "last_name": "", "is_bot": False}})
    mixin._group_surface_turn(msg)
    assert ("join_request", ("42",), {"approve": False}) in bot.calls
    # With auto-approve off, a clean request queues + notifies.
    mixin2, surface2, bot2, sent2 = _rig(
        tmp_path / "q", bot=_BotAdapter(),
        acceptance={"mode": "request", "auto_approve": False,
                    "criteria": _criteria(username=True)})
    msg2 = _msg(meta={"service": "join_request", "requester": {
        "id": "43", "username": "bob", "name": "Bob",
        "first_name": "Bob", "last_name": "", "is_bot": False}})
    mixin2._group_surface_turn(msg2)
    assert "join_request" not in bot2.kinds()
    texts = _texts(sent2)
    assert any("Join request" in t or "wants in" in t
               for t in texts)
    buttons = [k.get("buttons") for _a, k in sent2
               if k.get("buttons")]
    flat = [pair for rows in buttons for row in rows for pair in row]
    assert any(data.startswith("joinq approve jr-")
               for _label, data in flat)


def test_runtime_photo_criterion_uses_adapter_lookup(tmp_path):
    bot = _BotAdapter(photo=False)
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot,
        acceptance={"criteria": _criteria(photo=True)})
    mixin._group_surface_turn(_join_msg([
        {"id": "42", "username": "ada", "name": "Ada",
         "first_name": "Ada", "last_name": "L", "is_bot": False}]))
    assert ("photo", ("42",), {}) in bot.calls
    assert "restrict" in bot.kinds()
    # A failed lookup is UNKNOWN: admitted, journaled.
    bot2 = _BotAdapter(photo_ok=False)
    mixin2, surface2, bot2, sent2 = _rig(
        tmp_path / "u", bot=bot2,
        acceptance={"criteria": _criteria(photo=True)})
    mixin2._group_surface_turn(_join_msg([
        {"id": "42", "username": "ada", "name": "Ada",
         "first_name": "Ada", "last_name": "L", "is_bot": False}]))
    assert "restrict" not in bot2.kinds()
    assert _audit_kinds(surface2, "acceptance_unverifiable")


def test_control_recheck_releases_and_welcomes(tmp_path):
    bot = _BotAdapter()
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot,
        acceptance={"criteria": _criteria(username=True)})
    surface.acceptance_on_join(
        KEY, _facts(username=""), now=time.time())
    bot.calls.clear()
    out = mixin._control_recheck(_msg(
        sender_id="42", sender="Ada", sender_username="ada"))
    assert ("unrestrict", ("42",), {}) in bot.calls
    assert "Ada" in out          # the welcome text is the reply
    out = mixin._control_recheck(_msg(
        sender_id="42", sender="Ada", sender_username="ada"))
    assert "nothing to check" in out


def test_control_recheck_still_failing(tmp_path):
    mixin, surface, bot, sent = _rig(
        tmp_path,
        acceptance={"criteria": _criteria(username=True)})
    surface.acceptance_on_join(
        KEY, _facts(username=""), now=time.time())
    out = mixin._control_recheck(_msg(
        sender_id="42", sender="Ada", sender_username=""))
    assert "set a username" in _plain(out)
    member = surface.store.group(KEY)["members"]["42"]
    assert member["acceptance"]["state"] == "restricted"


def test_control_joinq_lists_and_decides(tmp_path):
    bot = _BotAdapter()
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot,
        acceptance={"mode": "request", "auto_approve": False,
                    "criteria": _criteria(username=True)})
    out = surface.acceptance_on_request(KEY, _facts(), now=NOW)
    rid = out["request_id"]
    listing = _plain(mixin._control_joinq("", _msg()))
    assert "Ada" in listing
    reply = mixin._control_joinq(f"approve {rid}", _msg())
    assert "approved" in reply
    assert ("join_request", ("42",), {"approve": True}) in bot.calls
    assert _plain(mixin._control_joinq("", _msg())).startswith(
        "the join queue is empty")
    # Unknown ids are answered honestly, never invented.
    reply = mixin._control_joinq("approve jr-9999", _msg())
    assert "isn't waiting anymore" in reply


def test_control_joinq_decline_hold_kicks(tmp_path):
    bot = _BotAdapter()
    mixin, surface, bot, sent = _rig(
        tmp_path, bot=bot, acceptance={"criteria": _criteria()})
    for i in range(8):
        surface.acceptance_on_join(
            KEY, _facts(user_id=str(300 + i), username=f"r{i}"),
            now=NOW + i)
    hold = surface.acceptance_queue(KEY)[-1]
    reply = mixin._control_joinq(f"decline {hold['id']}", _msg())
    assert "removed" in reply
    assert ("ban", (hold["user_id"],), {}) in bot.calls


def test_gsettings_panel_and_grammar(tmp_path):
    mixin, surface, bot, sent = _rig(tmp_path)
    panel = _plain(mixin._control_gsettings("", _msg()))
    assert "Acceptance" in panel
    assert "mode: open" in panel
    out = mixin._control_gsettings(
        "acceptance username on", _msg())
    assert "username: on" in out
    crit = surface.store.settings(KEY)["acceptance"]["criteria"]
    assert crit["username"] is True
    out = mixin._control_gsettings(
        "acceptance mode request", _msg())
    assert "join mode: request" in out
    out = mixin._control_gsettings(
        "acceptance override username decline", _msg())
    assert "decline" in out
    acc = surface.store.settings(KEY)["acceptance"]
    assert acc["overrides"]["username"] == "decline"
    out = mixin._control_gsettings(
        "acceptance whitelist add @helperbot", _msg())
    assert "@helperbot" in out
    acc = surface.store.settings(KEY)["acceptance"]
    assert acc["bot_whitelist"] == ["@helperbot"]
    out = mixin._control_gsettings("acceptance grace 15", _msg())
    assert "15 minutes" in out
    panel = _plain(mixin._control_gsettings("", _msg()))
    assert "username" in panel and "queue: 0 waiting" in panel


def test_bot_whitelisted_bot_admitted(tmp_path):
    surface = _surface(
        tmp_path, criteria=_criteria(bots=True),
        bot_whitelist=["@helperbot"])
    out = surface.acceptance_on_join(
        KEY, _facts(user_id="55", username="helperbot",
                    is_bot=True), now=NOW)
    assert out["verdict"] == "admit"
    out = surface.acceptance_on_join(
        KEY, _facts(user_id="56", username="spambot",
                    is_bot=True), now=NOW)
    assert out["missing"] == ["bots"]


# ── internal ids never print (owner, 2026-11-03) ──────────────
# Folded into this build from the owner's live /math testing:
# the rendered quiz showed "· id <problem_id>" — an internal
# tracking token members never need (the bot holds the open
# problem per member; that's how /math answer matches).  The
# sweep covered every partner runtime render; these pins hold
# the math surface.


def _render_math(result):
    from nomorals.agents.partner.runtime_games import \
        RuntimeGamesMixin

    return RuntimeGamesMixin._render_math(result)


def test_rendered_problem_prints_no_problem_id():
    problem = {"problem_id": "prob_secret_123", "tier": "easy",
               "topic": "fractions", "question": "1/2 + 1/4 = ?",
               "time_limit_s": 60}
    result = {"ok": True, "kind": "problem", "problem": problem}
    text = _render_math(result)
    assert "· id" not in text
    assert "prob_secret_123" not in text
    # The problem itself still renders whole…
    assert "1/2 + 1/4 = ?" in text and "⏱ 60s" in text
    # …and the id stays in state exactly as before.
    assert result["problem"]["problem_id"] == "prob_secret_123"


def test_rendered_next_problem_prints_no_problem_id():
    nxt = {"problem_id": "prob_next_9", "tier": "hard",
           "topic": "algebra", "question": "2x = 10, x = ?",
           "time_limit_s": 90}
    result = {"ok": True, "kind": "result", "correct": True,
              "answer": "5", "next_problem": nxt}
    text = _render_math(result)
    assert "· id" not in text and "prob_next_9" not in text
    assert "2x = 10, x = ?" in text


def test_duel_header_prints_no_duel_id():
    result = {"ok": True, "kind": "duel", "duel_id": "duel_abc",
              "players": ["ada", "bob"],
              "problems": [{"question": "3 × 7 = ?"}]}
    text = _render_math(result)
    header = text.splitlines()[0]
    assert "duel_abc" not in header
    assert header.startswith("⚔️ Duel — ada vs bob")
    # The one surviving print is the answer grammar itself:
    # today's duel_answer command takes the id as an argument
    # (an id a human must quote back to act — the stated
    # exception), so the instruction line keeps carrying it
    # until the matching itself is state-held.
    assert "/math duel_answer duel_abc" in text


def test_audit_trail_covers_the_decision_chain(tmp_path):
    surface = _surface(
        tmp_path, mode="request", auto_approve=False,
        criteria=_criteria(username=True))
    out = surface.acceptance_on_request(
        KEY, _facts(username=""), now=NOW)
    assert out["verdict"] == "queued"
    surface.acceptance_decide(KEY, out["request_id"],
                              approve=False, by="admin1", now=NOW)
    kinds = [a["kind"] for a in surface.audit(KEY, limit=500)]
    assert "acceptance_eval" in kinds
    assert "acceptance_decision" in kinds
    decision = [a for a in surface.audit(KEY, limit=500)
                if a["kind"] == "acceptance_decision"][0]
    assert decision["by"] == "admin1"
    assert decision["approve"] is False
