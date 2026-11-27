"""Join pipeline (Rose phase 2): CAPTCHA wiring + AntiRaid + Approval.

Pins: the CaptchaGate's shared-pending wiring (merge, status,
release, the attempts cap), the surface's captcha hold/answer/
admit across button/math/question modes, the queue merge
(cap-<user_id> records decidable from /joinq), the approval hold
for request-mode walk-ins (and the recheck that can't release
it), the raid-action switch (hold/captcha/kick), the burst
sensor (including 0 = off), the raid auto-lift, the runtime's
challenge send + intercept + /captcha + raid commands, the
/gsettings captcha/raid grammar, and the registry counts.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001"
NOW = 1_800_000_000.0


# ── fakes ─────────────────────────────────────────────────────

class _Adapter:
    """Bot API surface: admin actions + join requests + photo."""

    name = "telegram-bot"

    def __init__(self, *, admin_ids=("admin1",), refuse=False):
        self.calls: list = []
        self._admin_ids = set(admin_ids)
        self._refuse = refuse

    def _out(self, name, *args, **kw):
        self.calls.append((name, args, kw))
        return {"ok": not self._refuse,
                "error": "nope" if self._refuse else ""}

    def admin_member(self, chat, user_id):
        return {"status": "administrator"
                if str(user_id) in self._admin_ids else "member"}

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

    def kinds(self):
        return [c[0] for c in self.calls]


def _msg(*, sender_id="admin1", sender="Admin",
         username="boss", text="", message_id="900",
         sender_is_bot=False):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="telegram-bot",
                             kind="group", title="Builders"),
        sender_id=sender_id, sender=sender,
        sender_username=username, text=text, meta={},
        reply_to="", message_id=message_id,
        sender_is_bot=sender_is_bot)


def _surface(tmp_path: Path, **settings) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if settings:
        surface.update_settings(KEY, **settings)
    return surface


def _facts(user_id="42", **kw):
    facts = {"user_id": user_id, "username": "ada",
             "first_name": "Ada", "last_name": "Lovelace",
             "name": "Ada", "is_bot": False, "has_photo": True,
             "previously_banned": False, "whitelisted": False,
             "filter_words": []}
    facts.update(kw)
    return facts


def _rig(tmp_path: Path, **adapter_kw):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _Adapter(**adapter_kw)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    mixin._probe_bot_claim = lambda *a, **k: None  # type: ignore[method-assign]
    sent: list = []

    def _send(chat, text, *, buttons=None, parse_mode=""):
        sent.append({"chat": chat, "text": text,
                     "buttons": buttons, "parse_mode": parse_mode})
        return SimpleNamespace(message_id="m123", ok=True)

    mixin._send = _send  # type: ignore[method-assign]
    mixin._system_send_ok = lambda *a, **k: True  # type: ignore[method-assign]
    return mixin, surface, adapter, sent


def _hold(surface, user_id="42", mode="button", **kw):
    surface.update_settings(KEY, captcha_mode=mode)
    return surface.captcha_hold(
        KEY, user_id, name="Ada", username="ada",
        facts=_facts(user_id), now=NOW, **kw)


def _audit_kinds(surface):
    entry = surface.store.group(KEY)
    return [str(r.get("kind") or "")
            for r in (entry.get("audit") or [])]


def _plain(out):
    """Unwrap an ActionableMessage reply to its text."""
    return getattr(out, "text", out)


# ── the gate, wired ───────────────────────────────────────────

def test_captcha_gate_shared_pending_merges(tmp_path):
    from nomorals.community.welcome_moderation import CaptchaGate

    records = {"u1": {"mode": "button", "challenge": "tap",
                      "expected": "verify", "facts": {"a": 1}}}
    gate = CaptchaGate(pending=records)
    res = gate.request_join("u1", identity={"user_id": "u1"})
    assert res["state"] == "pending_captcha"
    # The merge keeps the surface's own fields on the hold.
    assert records["u1"]["mode"] == "button"
    assert records["u1"]["expected"] == "verify"
    assert records["u1"]["facts"] == {"a": 1}
    assert records["u1"]["attempts"] == 0


def test_captcha_gate_status_and_release(tmp_path):
    from nomorals.community.welcome_moderation import CaptchaGate

    gate = CaptchaGate()
    assert gate.status("u1")["state"] == "unknown"
    gate.request_join("u1")
    assert gate.status("u1")["state"] == "pending_captcha"
    assert gate.status("u1")["remaining"] == 3
    released = gate.release("u1")
    assert released["released"] is True
    assert gate.status("u1")["state"] == "admitted"
    assert gate.release("nobody")["released"] is False


def test_captcha_gate_cap_blocks_on_fourth_attempt(tmp_path):
    from nomorals.community.welcome_moderation import CaptchaGate

    class _Pass:
        def check(self, user_id, answer):
            return answer == "correct"

    gate = CaptchaGate(_Pass())
    gate.request_join("u1")
    for _ in range(3):
        res = gate.verify("u1", "wrong")
        assert res["state"] == "pending_captcha"
    assert gate.status("u1")["attempts"] == 3
    # The fourth attempt is blocked without even checking —
    # even the right answer no longer admits them.
    assert gate.verify("u1", "correct")["state"] == "blocked"
    assert gate.status("u1")["state"] == "blocked"


# ── the surface's captcha ─────────────────────────────────────

def test_captcha_hold_button_mode(tmp_path):
    surface = _surface(tmp_path)
    held = _hold(surface, mode="button")
    assert held["held"] is True
    assert held["mode"] == "button"
    assert "human" in str(held["challenge"]).lower()
    assert held["options"] is None
    rec = surface.store.group(KEY)["captcha"]["42"]
    assert rec["expected"] == "verify"
    assert rec["name"] == "Ada"
    assert rec["facts"]["user_id"] == "42"
    assert "captcha_held" in _audit_kinds(surface)


def test_captcha_hold_math_mode(tmp_path):
    surface = _surface(tmp_path)
    held = _hold(surface, mode="math")
    assert held["held"] is True
    m = re.fullmatch(r"What is (\d+) \+ (\d+)\?",
                     str(held["challenge"]))
    assert m is not None
    rec = surface.store.group(KEY)["captcha"]["42"]
    assert rec["expected"] == str(int(m.group(1))
                                  + int(m.group(2)))
    assert len(held["options"]) == 4
    assert rec["expected"] in held["options"]


def test_captcha_hold_question_mode(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, captcha_questions=[
        {"question": "What colour is the sky?",
         "answer": "blue"},
        {"question": "Second?", "answer": "two"}])
    held = _hold(surface, mode="question")
    assert held["held"] is True
    assert "sky" in str(held["challenge"])
    rec = surface.store.group(KEY)["captcha"]["42"]
    assert rec["expected"] == "blue"
    # The first question is asked; the list order is the order.
    assert held["options"] is None


def test_captcha_hold_off_and_empty_question(tmp_path):
    surface = _surface(tmp_path)
    assert _hold(surface, mode="off")["held"] is False
    assert _hold(surface, mode="off")["reason"] == "off"
    no_q = _hold(surface, mode="question")
    assert no_q["held"] is False
    assert no_q["reason"] == "no_challenge"
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})


def test_captcha_answer_flow(tmp_path):
    surface = _surface(tmp_path)
    _hold(surface, mode="button")
    wrong = surface.captcha_answer(KEY, "42", "nope", now=NOW)
    assert wrong["state"] == "pending_captcha"
    assert wrong["attempts"] == 1
    assert wrong["remaining"] == 2
    assert "captcha_failed" in _audit_kinds(surface)
    right = surface.captcha_answer(KEY, "42", "verify", now=NOW)
    assert right["state"] == "admitted"
    assert right["record"]["name"] == "Ada"
    assert right["record"]["facts"]["user_id"] == "42"
    assert "captcha_verified" in _audit_kinds(surface)
    # The hold is gone — a second answer is unknown.
    assert surface.captcha_answer(KEY, "42", "verify")[
        "state"] == "unknown"
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})


def test_captcha_answer_blocks_after_cap(tmp_path):
    surface = _surface(tmp_path)
    _hold(surface, mode="math")
    rec = surface.store.group(KEY)["captcha"]["42"]
    expected = str(rec["expected"])
    for _ in range(3):
        res = surface.captcha_answer(KEY, "42", "wrong", now=NOW)
        assert res["state"] == "pending_captcha"
    blocked = surface.captcha_answer(KEY, "42", expected, now=NOW)
    assert blocked["state"] == "blocked"
    assert "captcha_blocked" in _audit_kinds(surface)
    # Still pending for a moderator — the record stays.
    assert "42" in (surface.store.group(KEY).get("captcha") or {})


def test_captcha_admit_by_moderator(tmp_path):
    surface = _surface(tmp_path)
    _hold(surface, mode="button")
    admitted = surface.captcha_admit(KEY, "42", by="admin1")
    assert admitted["admitted"] is True
    assert admitted["record"]["username"] == "ada"
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})
    assert surface.captcha_admit(KEY, "nobody")[
        "admitted"] is False


def test_captcha_status(tmp_path):
    surface = _surface(tmp_path)
    assert surface.captcha_status(KEY, "42")["state"] == "unknown"
    _hold(surface, mode="math")
    status = surface.captcha_status(KEY, "42")
    assert status["state"] == "pending_captcha"
    assert status["mode"] == "math"
    assert status["attempts"] == 0
    assert "What is" in str(status["challenge"])


# ── the queue merge ───────────────────────────────────────────

def test_captcha_queue_merge_and_decide(tmp_path):
    surface = _surface(tmp_path)
    _hold(surface, user_id="42", mode="button")
    _hold(surface, user_id="43", mode="math")
    queue = surface.acceptance_queue(KEY)
    caps = [r for r in queue if r.get("kind") == "captcha"]
    assert {r["id"] for r in caps} == {"cap-42", "cap-43"}
    assert all(r["state"] == "pending" for r in caps)
    approve = surface.acceptance_decide(
        KEY, "cap-42", approve=True, by="admin1", now=NOW)
    assert approve["ok"] is True
    assert approve["execute"] == "captcha_admit"
    decline = surface.acceptance_decide(
        KEY, "cap-43", approve=False, by="admin1", now=NOW)
    assert decline["ok"] is True
    assert decline["execute"] == "kick"
    # The declined hold is cleared; the approved one waits for
    # the caller's admit.
    assert "43" not in (surface.store.group(KEY).get("captcha")
                        or {})
    assert "42" in (surface.store.group(KEY).get("captcha") or {})
    assert surface.acceptance_decide(
        KEY, "cap-99", approve=True, by="admin1")[
        "ok"] is False


# ── approval ──────────────────────────────────────────────────

def test_approval_hold_walkin_and_recheck(tmp_path):
    surface = _surface(tmp_path, acceptance={"mode": "request"})
    out = surface.acceptance_on_join(KEY, _facts(), now=NOW)
    assert out["verdict"] == "restricted"
    assert out["reasons"] == ["waiting for an admin's approval"]
    queue = surface.acceptance_queue(KEY)
    assert len(queue) == 1
    assert queue[0]["kind"] == "hold"
    member = surface.store.group(KEY)["members"]["42"]
    assert member["acceptance"]["via"] == "approval"
    # The "check me again" button can't self-release an
    # approval hold — only an admin decides.
    again = surface.acceptance_recheck(
        KEY, "42", username="ada", name="Ada", now=NOW)
    assert again["state"] == "still"
    assert again["welcome"] is False
    assert "approval" in " ".join(again["reasons"])
    # …but the admin's approve does release them.
    decided = surface.acceptance_decide(
        KEY, queue[0]["id"], approve=True, by="admin1", now=NOW)
    assert decided["execute"] == "release"


def test_approval_hold_gets_no_welcome(tmp_path):
    surface = _surface(tmp_path, acceptance={"mode": "request"})
    out = surface.acceptance_on_join(KEY, _facts(), now=NOW)
    assert out["welcome"] is False


# ── antiraid ──────────────────────────────────────────────────

def _arm_raid(surface, n=8, start=NOW):
    outs = []
    for i in range(n):
        outs.append(surface.acceptance_on_join(
            KEY, _facts(str(100 + i), username=f"u{100 + i}"),
            now=start + i))
    return outs


def test_raid_action_hold_is_the_default(tmp_path):
    surface = _surface(tmp_path)
    outs = _arm_raid(surface)
    assert all(o["verdict"] == "admit" for o in outs[:7])
    assert outs[7]["verdict"] == "restricted"
    assert outs[7]["request_id"].startswith("jr-")
    assert "join rush" in " ".join(outs[7]["reasons"])
    assert "acceptance_raid_hold" in _audit_kinds(surface)


def test_raid_action_captcha(tmp_path):
    surface = _surface(
        tmp_path, acceptance={"raid_action": "captcha"})
    outs = _arm_raid(surface)
    assert outs[7]["verdict"] == "captcha"
    assert outs[7]["actions"] == []
    # No queue, no restriction — the caller captcha-holds them.
    assert surface.acceptance_queue(KEY) == []
    member = surface.store.group(KEY)["members"]["107"]
    assert not member.get("acceptance")


def test_raid_action_kick(tmp_path):
    surface = _surface(tmp_path, acceptance={"raid_action": "kick"})
    outs = _arm_raid(surface)
    assert outs[7]["verdict"] == "kick"
    assert outs[7]["actions"] == [{"kind": "kick",
                                    "user_id": "107",
                                    "rule": "acceptance"}]
    assert "raid_kick" in _audit_kinds(surface)


def test_raid_burst_zero_never_arms(tmp_path):
    surface = _surface(tmp_path, acceptance={"raid_burst": 0})
    outs = _arm_raid(surface, n=10)
    assert all(o["verdict"] == "admit" for o in outs)
    assert all(o["raid"] is False for o in outs)


def test_raid_lift_releases_after_window(tmp_path):
    surface = _surface(tmp_path)
    _arm_raid(surface)
    held = [r for r in surface.acceptance_queue(KEY)
            if r.get("kind") == "hold"]
    assert len(held) == 1
    # While the window stands, nothing lifts.
    assert surface.acceptance_raid_lift(KEY, now=NOW + 100) == []
    lifted = surface.acceptance_raid_lift(KEY, now=NOW + 700)
    assert len(lifted) == 1
    assert lifted[0]["execute"] == "release"
    assert lifted[0]["record"]["state"] == "approved"
    assert "raid_auto_end" in _audit_kinds(surface)
    # Once lifted the posture stays down until a new burst.
    assert surface.acceptance_raid_lift(KEY, now=NOW + 701) == []
    assert surface.acceptance_queue(KEY) == []


# ── the runtime: /captcha ─────────────────────────────────────

def test_control_captcha_verify_admits(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    line = mixin._control_captcha(
        "verify 42", _msg(sender_id="42", sender="Ada",
                          username="ada"))
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})
    # The reply IS the welcome (like /recheck) — one message.
    assert "Ada" in line or "verified" in line


def test_control_captcha_wrong_then_right(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="math")
    rec = surface.store.group(KEY)["captcha"]["42"]
    expected = str(rec["expected"])
    line = mixin._control_captcha(
        "answer 42 nope", _msg(sender_id="42", sender="Ada",
                               username="ada"))
    assert "not quite" in line
    assert "2 attempt(s) left" in line
    line = mixin._control_captcha(
        f"answer 42 {expected}", _msg(sender_id="42", sender="Ada",
                                     username="ada"))
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})


def test_control_captcha_other_member_refused(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    line = mixin._control_captcha(
        "verify 42", _msg(sender_id="43", sender="Mallory",
                          username="mal"))
    assert "only the held member" in line
    assert "42" in (surface.store.group(KEY).get("captcha") or {})


def test_control_captcha_blocked_after_cap(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    msg = _msg(sender_id="42", sender="Ada", username="ada")
    for _ in range(3):
        line = mixin._control_captcha("answer 42 nope", msg)
        assert "not quite" in line
    line = mixin._control_captcha("answer 42 nope", msg)
    assert "moderator" in line
    status = mixin._control_captcha("status 42", _msg())
    assert "blocked" in status


def test_control_captcha_status_reads(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    assert "no captcha hold" in mixin._control_captcha("", _msg())
    _hold(surface, mode="math")
    own = mixin._control_captcha(
        "", _msg(sender_id="42", sender="Ada", username="ada"))
    assert "waiting on the math captcha" in own
    admin = mixin._control_captcha("status 42", _msg())
    assert "waiting on the math captcha" in admin
    stranger = mixin._control_captcha(
        "status 42", _msg(sender_id="43", sender="Mal",
                          username="mal"))
    assert "admin read" in stranger


# ── the runtime: raid commands ────────────────────────────────

def test_raid_commands_set_and_report(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    assert "600s" in mixin._control_raidtime("", _msg())
    assert "1800s" in mixin._control_raidtime("1800", _msg())
    assert mixin._control_raidtime("30", _msg()) \
        .startswith("raid time is 60")
    assert "10m" in mixin._control_raidactiontime("600", _msg())
    assert mixin._control_raidactiontime("30", _msg()) \
        .startswith("raid grace is 60")
    assert "5 joins" in mixin._control_autoantiraid("5", _msg())
    assert "off" in mixin._control_autoantiraid("0", _msg())
    assert mixin._control_autoantiraid("65", _msg()) \
        .startswith("auto-raid is 0")
    assert mixin._control_autoantiraid("x", _msg()) \
        .startswith("that's not a number")
    settings = surface.store.settings(KEY)
    assert settings["acceptance"]["raid_cooldown_s"] == 1800
    assert settings["acceptance"]["raid_grace_minutes"] == 10
    assert settings["acceptance"]["raid_burst"] == 0
    assert "raid_config" in _audit_kinds(surface)


def test_raid_commands_refuse_non_admin(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    msg = _msg(sender_id="42", sender="Ada", username="ada")
    assert "admin call" in mixin._control_raidtime("1800", msg)
    assert "admin call" in mixin._control_autoantiraid("5", msg)


# ── the runtime: /gsettings grammar ───────────────────────────

def test_gsettings_captcha_grammar(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    assert "math" in mixin._control_gsettings(
        "captcha mode math", _msg())
    assert surface.store.settings(KEY)["captcha_mode"] == "math"
    added = mixin._control_gsettings(
        "captcha question add What is 2+2?|4", _msg())
    assert "added" in added
    listed = mixin._control_gsettings(
        "captcha question list", _msg())
    assert "What is 2+2?" in listed
    removed = mixin._control_gsettings(
        "captcha question remove 1", _msg())
    assert "removed" in removed
    assert surface.store.settings(KEY)["captcha_questions"] == []
    assert "usage" in mixin._control_gsettings(
        "captcha mode sideways", _msg())
    assert "gsettings" in _audit_kinds(surface)


def test_gsettings_acceptance_raid(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    line = mixin._control_gsettings(
        "acceptance raid captcha", _msg())
    assert "captcha" in line
    assert surface.store.settings(KEY)["acceptance"][
        "raid_action"] == "captcha"
    assert "usage" in mixin._control_gsettings(
        "acceptance raid ban", _msg()) \
        or "grammar" in mixin._control_gsettings(
            "acceptance raid ban", _msg())
    mode = mixin._control_gsettings(
        "acceptance mode request", _msg())
    assert "wait for an admin's approval" in mode


# ── the runtime: /joinq captcha branch ────────────────────────

def test_joinq_captcha_approve_admits(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    line = mixin._control_joinq("approve cap-42", _msg())
    assert "admitted" in line
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})
    assert sent  # the deferred welcome went out


def test_joinq_captcha_decline_kicks(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    line = mixin._control_joinq("decline cap-42", _msg())
    assert "removed" in line
    assert "ban" in adapter.kinds()
    assert "unban" in adapter.kinds()
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})


def test_joinq_lists_captcha_holds(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, user_id="42", mode="math")
    out = mixin._control_joinq("", _msg())
    assert "captcha (math) pending" in _plain(out)
    # The decision rides the buttons, with the cap- id in the
    # callback — the list text carries the label + the reason.
    assert any("joinq approve cap-42" in str(btn)
               for row in (out.buttons or []) for btn in row)
    assert any("joinq decline cap-42" in str(btn)
               for row in (out.buttons or []) for btn in row)


# ── the runtime: hold + intercept ─────────────────────────────

def test_hold_member_sends_challenge(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    surface.update_settings(KEY, captcha_mode="button")
    held = mixin._captcha_hold_member(
        surface, KEY, _msg(sender_id="42", sender="Ada",
                           username="ada"),
        "42", _facts("42"))
    assert held is True
    assert len(sent) == 1
    assert "human" in sent[0]["text"].lower()
    assert sent[0]["buttons"]
    rec = surface.store.group(KEY)["captcha"]["42"]
    assert rec["challenge_msg_id"] == "m123"


def test_hold_member_off_mode_joins_normally(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    held = mixin._captcha_hold_member(
        surface, KEY, _msg(sender_id="42", sender="Ada",
                           username="ada"),
        "42", _facts("42"))
    assert held is False
    assert sent == []
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})


def test_hold_member_send_gate_refusal_skips_hold(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    mixin._system_send_ok = lambda *a, **k: False  # type: ignore[method-assign]
    surface.update_settings(KEY, captcha_mode="button")
    held = mixin._captcha_hold_member(
        surface, KEY, _msg(sender_id="42", sender="Ada",
                           username="ada"),
        "42", _facts("42"))
    assert held is False
    assert sent == []
    # No silent ban: no hold opened, the skip audited.
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})
    assert "captcha_skipped" in _audit_kinds(surface)


def test_intercept_deletes_pending_message(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    msg = _msg(sender_id="42", sender="Ada", username="ada",
               text="hello everyone", message_id="777")
    consumed = mixin._group_surface_turn(msg)
    assert consumed is True
    assert "delete" in adapter.kinds()
    assert "captcha_hold" in _audit_kinds(surface)
    assert "42" in (surface.store.group(KEY).get("captcha") or {})


def test_intercept_question_mode_answers(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    surface.update_settings(KEY, captcha_questions=[
        {"question": "What colour is the sky?",
         "answer": "blue"}])
    _hold(surface, mode="question")
    msg = _msg(sender_id="42", sender="Ada", username="ada",
               text="blue", message_id="778")
    consumed = mixin._group_surface_turn(msg)
    assert consumed is True
    assert "42" not in (surface.store.group(KEY).get("captcha")
                        or {})
    assert sent  # the deferred welcome went out
    assert "delete" not in adapter.kinds()


def test_intercept_ignores_commands_and_admins(tmp_path):
    mixin, surface, adapter, sent = _rig(tmp_path)
    _hold(surface, mode="button")
    cmd = _msg(sender_id="42", sender="Ada", username="ada",
               text="/captcha status 42", message_id="779")
    assert mixin._captcha_intercept_message(
        surface, KEY, cmd, "42") is False
    admin = _msg(sender_id="admin1", sender="Admin",
                 username="boss", text="hello", message_id="780")
    assert mixin._captcha_intercept_message(
        surface, KEY, admin, "admin1") is False
    assert "delete" not in adapter.kinds()


# ── registry ──────────────────────────────────────────────────

def test_registry_counts_include_join_pipeline():
    from nomorals.social.chat.control import (
        _HELP_GROUPS,
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        PUBLIC_COMMANDS,
    )

    for kind in ("captcha", "raidtime", "raidactiontime",
                 "autoantiraid"):
        assert kind in PUBLIC_COMMANDS
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert any(kind in kinds for _, kinds in _HELP_GROUPS)
        assert any(kind in kinds for _, kinds in LIST_GROUPS)
    assert len(PUBLIC_COMMANDS) == 291
    assert len(CONTROL_COMMANDS) == 691
