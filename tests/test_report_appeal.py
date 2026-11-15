"""Mass-report + appeal system (DESIGN §32): reports as durable
records, evidence-gated mass reporting through Telegram's own
reporting routes, and appeals reviewed in the console with the
full record attached.

Pins: BanList durability (bans.json beside the group store —
quarantine-and-raise on corruption, the ReputationBook's rule),
/report filing a record with its evidence, the console one-tap
review verbs (warn/mute/ban/dismiss), /appeal DM-first filing +
admin review with the platform unban on approve, the
evidence-gated candidate set (never a free-form list), the
20s-paced /massreport run with its cap and journaling, and the
admin_report_peer verb through a fake Telethon client (Telethon
is not installed on this box).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    WHATSAPP_UNSUPPORTED_KINDS,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

TG_KEY = "telegram:-1001"
OTHER_KEY = "telegram:-1002"


class _Adapter:
    """Fake group adapter: admin_member + the mod verbs + the
    report verb. Calls are (name, args, kwargs) — the moderation
    tests' shape."""

    def __init__(self, *, admin=True, refuse=False, verbs=True):
        self.calls: list = []
        self._admin = admin
        self._refuse = refuse
        if not verbs:
            self.admin_report_peer = None  # type: ignore[method-assign]

    def _out(self, name, *args, **kw):
        self.calls.append((name, args, kw))
        return {"ok": not self._refuse,
                "error": "nope" if self._refuse else ""}

    def admin_member(self, chat, user_id):
        return {"status": "administrator" if self._admin
                else "member"}

    def admin_restrict(self, chat, user_id, *, minutes=60, until=0):
        return self._out("restrict", user_id, minutes=minutes)

    def admin_unrestrict(self, chat, user_id):
        return self._out("unrestrict", user_id)

    def admin_ban(self, chat, user_id):
        return self._out("ban", user_id)

    def admin_unban(self, chat, user_id):
        return self._out("unban", user_id)

    def admin_report_peer(self, user_id, reason, message):
        return self._out("report_peer", user_id, reason, message)


def _msg(*, key=TG_KEY, platform="telegram", kind="group",
         sender_id="admin1", reply_to="", reply_sender=None,
         reply_to_text=""):
    meta = {}
    if reply_sender:
        meta["reply_to_sender"] = reply_sender
    if reply_to_text:
        meta["reply_to_text"] = reply_to_text
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform, kind=kind,
                             title="Builders"),
        sender_id=sender_id, sender="Admin", sender_username="boss",
        text="", meta=meta, reply_to=reply_to, message_id="900")


def _dm_msg(*, sender_id="555"):
    return SimpleNamespace(
        chat=SimpleNamespace(key="", platform="telegram", kind="dm",
                             title=""),
        sender_id=sender_id, sender="Spammer", sender_username="spammy",
        text="", meta={}, reply_to="", message_id="901")


def _rig(tmp_path: Path, **adapter_kw):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _Adapter(**adapter_kw)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    return mixin, surface, adapter


def _text(out):
    return getattr(out, "text", out)


def _person(user_id, username="spammy", name="Spam"):
    return {"ids": {"telegram": str(user_id)},
            "usernames": [username], "names": [name]}


def _seed_flag(surface, key, user_id, category="scam",
               note="caught scamming"):
    return surface.stage_flag(
        key, user_id=str(user_id), person=_person(user_id),
        category=category, note=note, by="admin1",
        by_label="Admin", direct=False, source="sensor")


# ── registration ─────────────────────────────────────────────

def test_appeal_and_massreport_registered_everywhere():
    assert CONTROL_COMMANDS["appeal"] == (0, None)
    assert CONTROL_COMMANDS["massreport"] == (0, None)
    assert "appeal" in PUBLIC_COMMANDS
    assert "massreport" in PUBLIC_COMMANDS
    # massreport is Telegram-only machinery (userbot
    # account.reportPeer); appeals are platform-agnostic.
    assert "massreport" in WHATSAPP_UNSUPPORTED_KINDS
    assert "appeal" not in WHATSAPP_UNSUPPORTED_KINDS
    assert "/appeal" in COMMAND_DETAILS["appeal"]["usage"]
    assert "evidence-gated" in COMMAND_DETAILS["massreport"]["what"]
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    assert "appeal" in RuntimeGroupMixin._P5_SCOPE_KINDS
    assert "massreport" in RuntimeGroupMixin._P5_SCOPE_KINDS


# ── BanList durability ───────────────────────────────────────

def test_banlist_persists_across_reload(tmp_path):
    from nomorals.community.welcome_moderation import BanList

    first = BanList(tmp_path / "bans.json")
    ban = first.ban("555", reason="spam", group_id=TG_KEY,
                    banned_by="admin1")
    appeal = first.file_appeal("555", ban.id, "was my brother")
    second = BanList(tmp_path / "bans.json")
    assert [b.id for b in second.bans_for("555")] == [ban.id]
    assert second.get_appeal(appeal.id).status == "open"
    assert second.is_banned("555", TG_KEY) is not None


def test_banlist_corrupt_quarantines_and_raises(tmp_path):
    from nomorals.community.welcome_moderation import BanList, BanListStoreError

    path = tmp_path / "bans.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BanListStoreError):
        BanList(path)
    assert not path.exists()
    assert list(tmp_path.glob("bans.corrupt-*"))


def test_banlist_second_open_appeal_refused(tmp_path):
    from nomorals.community.welcome_moderation import BanList

    bans = BanList()
    ban = bans.ban("555", reason="spam", group_id=TG_KEY)
    bans.file_appeal("555", ban.id, "sorry")
    with pytest.raises(ValueError, match="already has an open appeal"):
        bans.file_appeal("555", ban.id, "sorry again")


def test_banlist_in_memory_still_works():
    from nomorals.community.welcome_moderation import BanList

    bans = BanList()
    ban = bans.ban("555", reason="spam", group_id=TG_KEY)
    assert bans.is_banned("555", TG_KEY) is not None
    bans.lift(ban.id, by="admin1")
    assert bans.is_banned("555", TG_KEY) is None


# ── reports as records ───────────────────────────────────────

def test_file_report_creates_record_and_audit(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    report = surface.file_report(
        TG_KEY, user_id="555", label="@spammy", by="member9",
        by_label="Member", reason="hiring spam",
        evidence_text="free money!!!", message_id="31337")
    assert report["id"] == "rp-0001"
    assert report["status"] == "open"
    assert report["evidence_text"] == "free money!!!"
    assert report["message_id"] == "31337"
    assert [r["id"] for r in surface.open_reports(TG_KEY)] == ["rp-0001"]
    assert surface.report_count_for(TG_KEY, "555") == 1
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "report_filed"
               and e["report"] == "rp-0001" for e in audit)


def test_decide_report_dismiss_closes(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    out = surface.decide_report(TG_KEY, "rp-0001",
                                action="dismissed", by="admin1")
    assert out["ok"] is True
    assert surface.open_reports(TG_KEY) == []
    assert surface.report_count_for(TG_KEY, "555") == 0


def test_decide_report_rejects_bad_action_and_closed(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    assert surface.decide_report(TG_KEY, "rp-0001", action="nuke",
                                 by="admin1")["ok"] is False
    surface.decide_report(TG_KEY, "rp-0001", action="dismissed",
                          by="admin1")
    assert surface.decide_report(TG_KEY, "rp-0001",
                                 action="dismissed",
                                 by="admin1")["ok"] is False


# ── the evidence-gated candidate set ─────────────────────────

def test_candidates_empty_on_fresh_group(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    surface.handle_message(TG_KEY, "555", username="spammy",
                           text="hi")
    assert surface.mass_report_candidates(TG_KEY) == []


def test_candidates_from_confirmed_record(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    prop = _seed_flag(surface, TG_KEY, "555")
    surface.decide_flag(TG_KEY, prop["id"], approve=True, by="admin1")
    cands = surface.mass_report_candidates(TG_KEY)
    assert len(cands) == 1
    assert cands[0]["user_id"] == "555"
    assert cands[0]["category"] == "scam"
    assert cands[0]["source"] == "reputation"


def test_candidates_from_staged_flag(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    _seed_flag(surface, TG_KEY, "555", category="raid")
    cands = surface.mass_report_candidates(TG_KEY)
    assert len(cands) == 1
    assert cands[0]["source"] == "flag"
    assert cands[0]["category"] == "raid"


def test_candidates_need_three_open_reports(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    for _ in range(2):
        surface.file_report(TG_KEY, user_id="555", label="@spammy",
                            by="member9", reason="spam")
    assert surface.mass_report_candidates(TG_KEY) == []
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member8", reason="spam again")
    cands = surface.mass_report_candidates(TG_KEY)
    assert len(cands) == 1
    assert cands[0]["source"] == "reports"
    assert "3 open reports" in cands[0]["evidence"]


def test_candidates_from_sensor_ban_audit(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    surface._audit(TG_KEY, "admin_action", action="ban",
                   user_id="555", by="sensor", reason="flood")
    surface.store.save()
    cands = surface.mass_report_candidates(TG_KEY)
    assert len(cands) == 1
    assert cands[0]["source"] == "sensor-ban"


def test_candidates_dedupe_one_row_per_account(tmp_path):
    _mixin, surface, _adapter = _rig(tmp_path)
    prop = _seed_flag(surface, TG_KEY, "555")
    surface.decide_flag(TG_KEY, prop["id"], approve=True, by="admin1")
    for _ in range(3):
        surface.file_report(TG_KEY, user_id="555", label="@spammy",
                            by="member9", reason="spam")
    cands = surface.mass_report_candidates(TG_KEY)
    assert len(cands) == 1
    assert cands[0]["source"] == "reputation"


# ── /report filing ───────────────────────────────────────────

def test_report_filing_creates_record_with_evidence(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    msg = _msg(sender_id="member9",
               reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"},
               reply_to_text="free money!!!", reply_to="31337")
    out = mixin._control_report("hiring spam", msg)
    assert "reported" in out
    reports = surface.open_reports(TG_KEY)
    assert len(reports) == 1
    assert reports[0]["user_id"] == "555"
    assert reports[0]["evidence_text"] == "free money!!!"
    assert reports[0]["message_id"] == "31337"
    # The legacy audit row is untouched, and the flag proposal
    # still stages — the record is additive, not a replacement.
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "report" and e["user_id"] == "555"
               for e in audit)
    assert surface.flag_proposals(TG_KEY)


def test_report_queue_lists_open_reports(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="hiring spam")
    surface.file_report(TG_KEY, user_id="777", label="@loud",
                        by="member9", reason="caps")
    out = mixin._control_report("queue", _msg())
    text = _text(out)
    assert "rp-0001" in text and "rp-0002" in text
    assert "@spammy" in text


def test_report_queue_empty(tmp_path):
    mixin, _surface, _adapter = _rig(tmp_path)
    assert "no open reports" in mixin._control_report("queue", _msg())


# ── /report review in-group ──────────────────────────────────

def _filed_report(mixin, surface):
    msg = _msg(sender_id="member9",
               reply_sender={"id": "555", "username": "spammy",
                             "name": "Spam"})
    mixin._control_report("hiring spam", msg)
    return surface.open_reports(TG_KEY)[0]["id"]


def test_report_review_dismiss(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} dismiss", _msg())
    assert "dismissed" in out
    assert surface.open_reports(TG_KEY) == []


def test_report_review_ban_records_banlist(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} ban", _msg())
    assert "report closed" in out
    assert ("ban", ("555",), {}) in adapter.calls
    assert surface.bans.is_banned("555", TG_KEY) is not None
    assert surface.open_reports(TG_KEY) == []
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "report_decided"
               and e["action"] == "acted" for e in audit)


def test_report_review_mute_one_hour(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} mute", _msg())
    assert "report closed" in out
    assert ("restrict", ("555",), {"minutes": 60}) in adapter.calls
    assert surface.open_reports(TG_KEY) == []


def test_report_review_warn_records_strike(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} warn", _msg())
    assert "warning 1/3" in out
    assert "report closed" in out
    assert surface.open_reports(TG_KEY) == []


def test_report_review_refusal_leaves_report_open(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, refuse=True)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} ban", _msg())
    assert "didn't go through" in out
    assert "stays open" in out
    assert len(surface.open_reports(TG_KEY)) == 1
    assert surface.bans.is_banned("555", TG_KEY) is None


def test_report_review_non_admin_refused(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, admin=False)
    rid = _filed_report(mixin, surface)
    out = mixin._control_report(f"{rid} ban", _msg())
    assert "admin" in out
    assert adapter.calls == []
    assert len(surface.open_reports(TG_KEY)) == 1


# ── /ban + /unban write the BanList ──────────────────────────

def test_ban_records_banlist_and_unban_lifts(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    surface.handle_message(TG_KEY, "777", username="loud", text="hi")
    out = mixin._control_ban("@loud spamming", _msg())
    assert "banned" in out
    ban = surface.bans.is_banned("777", TG_KEY)
    assert ban is not None and ban.reason == "spamming"
    out = mixin._control_unban("@loud", _msg())
    assert "unbanned" in out
    assert surface.bans.is_banned("777", TG_KEY) is None


# ── /appeal ──────────────────────────────────────────────────

def _banned(mixin, surface, user_id="555"):
    surface.handle_message(TG_KEY, user_id, username="spammy",
                           text="hi")
    mixin._control_ban("@spammy spamming", _msg())
    return surface.bans.is_banned(user_id, TG_KEY)


def test_appeal_dm_lists_bans(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    out = mixin._control_appeal("", _dm_msg())
    assert ban.id in out
    assert "spamming" in out


def test_appeal_dm_no_ban_plain(tmp_path):
    mixin, _surface, _adapter = _rig(tmp_path)
    out = mixin._control_appeal("", _dm_msg())
    assert "no ban on record" in out
    assert "before this system shipped" in out


def test_appeal_dm_files_and_audits(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    out = mixin._control_appeal(f"{ban.id} my brother used my phone",
                                _dm_msg())
    assert "appeal filed" in out
    appeals = surface.bans.appeals(status="open")
    assert len(appeals) == 1
    assert appeals[0].ban_id == ban.id
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "appeal_filed"
               and e["appeal"] == appeals[0].id for e in audit)


def test_appeal_dm_second_open_refused(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    mixin._control_appeal(f"{ban.id} first", _dm_msg())
    out = mixin._control_appeal(f"{ban.id} second", _dm_msg())
    assert "already has an open appeal" in out


def test_appeal_in_group_files_own_ban(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    msg = _msg(sender_id="555")
    out = mixin._control_appeal(f"{ban.id} it was a mistake", msg)
    assert "appeal filed" in out


def test_appeal_approve_unbans_and_lifts(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    out = mixin._control_appeal(f"ok {appeal.id}", _msg())
    assert "approved" in out
    assert ("unban", ("555",), {}) in adapter.calls
    assert surface.bans.is_banned("555", TG_KEY) is None
    assert surface.bans.get_appeal(appeal.id).status == "approved"
    audit = surface.audit(TG_KEY, limit=8)
    assert any(e["kind"] == "appeal_reviewed"
               and e["decision"] == "approved" for e in audit)


def test_appeal_deny_keeps_ban(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    out = mixin._control_appeal(f"no {appeal.id}", _msg())
    assert "denied" in out
    assert surface.bans.is_banned("555", TG_KEY) is not None
    assert surface.bans.get_appeal(appeal.id).status == "rejected"
    assert not any(c[0] == "unban" for c in adapter.calls)


def test_appeal_approve_refusal_keeps_appeal_open(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    adapter._refuse = True
    out = mixin._control_appeal(f"ok {appeal.id}", _msg())
    assert "didn't go through" in out
    assert "stays open" in out
    assert surface.bans.get_appeal(appeal.id).status == "open"
    assert surface.bans.is_banned("555", TG_KEY) is not None


def test_appeal_cross_group_refused(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    ban = _banned(mixin, surface)
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    out = mixin._appeal_decide_console(
        surface, OTHER_KEY, appeal.id, approve=True, by="admin1")
    assert "belongs to another group" in out
    assert surface.bans.get_appeal(appeal.id).status == "open"


def test_appeal_global_ban_approves_without_platform_unban(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    ban = surface.bans.ban("555", reason="cross-group spam",
                           scope="global", banned_by="admin1")
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    out = mixin._appeal_decide_console(
        surface, TG_KEY, appeal.id, approve=True, by="admin1")
    assert "approved" in out
    assert not any(c[0] == "unban" for c in adapter.calls)
    assert surface.bans.get_appeal(appeal.id).status == "approved"


# ── /massreport ──────────────────────────────────────────────

def _seed_two_candidates(surface):
    _seed_flag(surface, TG_KEY, "555", category="scam",
               note="fake giveaway")
    _seed_flag(surface, TG_KEY, "777", category="spam-ring",
               note="ring of 40 accounts")


def test_massreport_list_shows_candidates_with_evidence(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    _seed_two_candidates(surface)
    out = mixin._control_massreport("", _msg())
    text = _text(out)
    assert "555" in text and "777" in text
    assert "fake giveaway" in text
    assert "evidence-gated" in text


def test_massreport_dry_submits_nothing(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    _seed_two_candidates(surface)
    out = mixin._control_massreport("dry", _msg())
    assert "Dry run" in _text(out)
    assert adapter.calls == []


def test_massreport_empty_set(tmp_path):
    mixin, _surface, _adapter = _rig(tmp_path)
    out = mixin._control_massreport("", _msg())
    assert "nothing to mass-report" in out


def test_massreport_non_telegram_refused(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    _seed_two_candidates(surface)
    msg = _msg(key="whatsapp:1203630123456@g.us",
               platform="whatsapp")
    out = mixin._control_massreport("", msg)
    assert "no equivalent" in out


def test_massreport_run_submits_with_reason_map(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    mixin, surface, adapter = _rig(tmp_path)
    _seed_two_candidates(surface)
    out = mixin._control_massreport("run", _msg())
    assert "2 submitted, 0 failed" in out
    reports = [c for c in adapter.calls if c[0] == "report_peer"]
    assert len(reports) == 2
    by_user = {c[1][0]: c[1] for c in reports}
    assert by_user["555"][1] == "fake"
    assert by_user["777"][1] == "spam"
    assert "fake giveaway" in by_user["555"][2]
    audit = surface.audit(TG_KEY, limit=8)
    assert sum(1 for e in audit
               if e["kind"] == "mass_report_submitted") == 2
    runs = surface.store.group(TG_KEY).get("mass_report_runs") or []
    assert len(runs) == 1 and runs[0]["submitted"] == 2
    assert "Telegram never reports back" in out


def test_massreport_run_caps_at_ten(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    mixin, surface, adapter = _rig(tmp_path)
    for n in range(12):
        _seed_flag(surface, TG_KEY, str(1000 + n), category="scam")
    out = mixin._control_massreport("run", _msg())
    assert "10 submitted" in out
    assert "2 skipped (cap 10 per run" in out
    assert len([c for c in adapter.calls
                if c[0] == "report_peer"]) == 10


def test_massreport_run_refusal_journaled(monkeypatch, tmp_path):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    mixin, surface, _adapter = _rig(tmp_path, refuse=True)
    _seed_flag(surface, TG_KEY, "555", category="scam")
    out = mixin._control_massreport("run", _msg())
    assert "0 submitted, 1 failed" in out
    audit = surface.audit(TG_KEY, limit=5)
    assert any(e["kind"] == "mass_report_failed"
               and e["user_id"] == "555" for e in audit)


def test_massreport_no_userbot_verb_refuses(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path, verbs=False)
    _seed_two_candidates(surface)
    out = mixin._control_massreport("run", _msg())
    assert "userbot path can't report" in out
    assert "Nothing was submitted" in out


def test_massreport_non_admin_refused(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, admin=False)
    _seed_two_candidates(surface)
    out = mixin._control_massreport("", _msg())
    assert "admin" in out
    assert adapter.calls == []


# ── the adapter verb (fake Telethon) ─────────────────────────

def _stub_telethon(monkeypatch):
    """Telethon is not installed on this box; stub the
    account.reportPeer request + the ten reason types."""
    telethon = types.ModuleType("telethon")
    tl = types.ModuleType("telethon.tl")
    functions = types.ModuleType("telethon.tl.functions")
    account = types.ModuleType("telethon.tl.functions.account")
    types_mod = types.ModuleType("telethon.tl.types")

    class ReportPeerRequest:
        def __init__(self, peer, reason, message):
            self.peer = peer
            self.reason = reason
            self.message = message

    account.ReportPeerRequest = ReportPeerRequest
    functions.account = account
    tl.functions = functions
    for name in ("spam", "violence", "pornography", "child_abuse",
                 "copyright", "fake", "illegal_drugs",
                 "personal_details", "geo_irrelevant", "other"):
        cls_name = "InputReportReason" + "".join(
            w.title() for w in name.split("_"))
        setattr(types_mod, cls_name,
                type(cls_name, (), {}))
    tl.types = types_mod
    telethon.tl = tl
    for name, mod in (("telethon", telethon),
                      ("telethon.tl", tl),
                      ("telethon.tl.functions", functions),
                      ("telethon.tl.functions.account", account),
                      ("telethon.tl.types", types_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return ReportPeerRequest


class _FakeClient:
    def __init__(self):
        self.requests: list = []
        self.entities: dict = {}

    async def get_entity(self, user_id):
        user = SimpleNamespace(id=int(user_id))
        self.entities[int(user_id)] = user
        return user

    async def __call__(self, request):
        self.requests.append(request)
        return True


def _live_adapter(monkeypatch):
    from nomorals.social.chat.telegram import TelegramAdapter

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._loop = loop
    adapter._client = _FakeClient()
    adapter._thread = thread
    return adapter


def test_admin_report_peer_builds_report_peer_request(monkeypatch):
    ReportPeerRequest = _stub_telethon(monkeypatch)
    adapter = _live_adapter(monkeypatch)
    out = adapter.admin_report_peer(
        "555", "fake", "reputation: fake giveaway")
    assert out == {"ok": True}
    req = adapter._client.requests[0]
    assert isinstance(req, ReportPeerRequest)
    assert req.peer.id == 555
    assert type(req.reason).__name__ == "InputReportReasonFake"
    assert req.message == "reputation: fake giveaway"
    adapter._loop.call_soon_threadsafe(adapter._loop.stop)


def test_admin_report_peer_rejects_unknown_reason(monkeypatch):
    _stub_telethon(monkeypatch)
    adapter = _live_adapter(monkeypatch)
    out = adapter.admin_report_peer("555", "nonsense", "x")
    assert out["ok"] is False
    assert "unknown report reason" in out["error"]
    adapter._loop.call_soon_threadsafe(adapter._loop.stop)


# ── the console ──────────────────────────────────────────────

def test_console_reputation_lists_reports_and_appeals(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="hiring spam")
    ban = surface.bans.ban("777", reason="raid", group_id=TG_KEY,
                           banned_by="admin1")
    appeal = surface.bans.file_appeal("777", ban.id, "wasn't me")
    out = mixin._console_section(surface, TG_KEY, _msg(), "t1",
                                 "reputation")
    text = _text(out)
    assert "rp-0001" in text
    assert "open reports here: 1" in text
    assert "open appeals: 1" in text
    assert appeal.id in text


def test_console_report_dismiss(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    out = mixin._apply_report_appeal_console(
        surface, TG_KEY, _msg(), "admin1",
        ["report", "rp-0001", "dismiss"])
    assert "dismissed" in out
    assert surface.open_reports(TG_KEY) == []


def test_console_report_ban(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    out = mixin._apply_report_appeal_console(
        surface, TG_KEY, _msg(), "admin1",
        ["report", "rp-0001", "ban"])
    assert "banned" in out
    assert ("ban", ("555",), {}) in adapter.calls
    assert surface.bans.is_banned("555", TG_KEY) is not None
    assert surface.open_reports(TG_KEY) == []


def test_console_report_warn_lands_ladder_step(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    out = mixin._apply_report_appeal_console(
        surface, TG_KEY, _msg(), "admin1",
        ["report", "rp-0001", "warn"])
    assert "warning 1" in out
    assert surface.open_reports(TG_KEY) == []


def test_console_report_non_admin_refused(tmp_path):
    mixin, surface, adapter = _rig(tmp_path, admin=False)
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    out = mixin._apply_report_appeal_console(
        surface, TG_KEY, _msg(), "member9",
        ["report", "rp-0001", "dismiss"])
    assert "admin decision" in out
    assert len(surface.open_reports(TG_KEY)) == 1


def test_console_appeal_approve(tmp_path):
    mixin, surface, adapter = _rig(tmp_path)
    ban = surface.bans.ban("555", reason="spam", group_id=TG_KEY,
                           banned_by="admin1")
    appeal = surface.bans.file_appeal("555", ban.id, "sorry")
    out = mixin._apply_report_appeal_console(
        surface, TG_KEY, _msg(), "admin1",
        ["appeal", "ok", appeal.id])
    assert "approved" in out
    assert ("unban", ("555",), {}) in adapter.calls
    assert surface.bans.is_banned("555", TG_KEY) is None


# ── /groups waiting ──────────────────────────────────────────

def test_groups_waiting_names_reports(tmp_path):
    mixin, surface, _adapter = _rig(tmp_path)
    assert mixin._groups_waiting(surface, TG_KEY) == ""
    surface.file_report(TG_KEY, user_id="555", label="@spammy",
                        by="member9", reason="spam")
    assert "reports 1" in mixin._groups_waiting(surface, TG_KEY)


# ── BanList wired into the default surface ───────────────────

def test_default_surface_bans_are_durable(tmp_path):
    from nomorals.social.chat.group_surface import default_surface

    first = default_surface(tmp_path)
    ban = first.bans.ban("555", reason="spam", group_id=TG_KEY,
                         banned_by="admin1")
    second = default_surface(tmp_path)
    assert second.bans.is_banned("555", TG_KEY) is not None
    assert second.bans.is_banned("555", TG_KEY).id == ban.id
