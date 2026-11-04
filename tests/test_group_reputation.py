"""Cross-group reputation — the notorious-user watchlist
(DESIGN §32).

Pins: creation discipline (a record exists ONLY by an admin's
confirmation — proposals from /flag, /report and the defense
sensors brand nobody until confirmed, and unconfirmed proposals
expire), identity stitching via §29 (a changed handle does not
shed the record; a shared username bridges platforms), the join
warning in all three modes through the §31 notice layer, §28
acceptance deciding admission exactly as before, local dismiss,
source-group retraction, category aging, §33 suppression in
unclaimed rooms, and worst-first single-warning ordering.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; the RuntimeGroupMixin subclass rig from the
escalation/console suite.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.group_reputation import (
    PROPOSAL_TTL_S,
    ReputationBook,
    ReputationStoreError,
)
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
)

A = "telegram-bot:-1001111"
B = "telegram-bot:-1002222"
C = "telegram-bot:-1003333"
D = "discord:555111"
U = "telegram:-1009999"          # the owner's userbot surface
NOW = 1_800_000_000.0
DAY = 86400.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _confirm_flag(surface: GroupSurface, key: str = A, *,
                  user_id: str = "42", username: str = "ada",
                  name: str = "Ada", category: str = "scam",
                  note: str = "took deposits", by: str = "99",
                  now: float = NOW) -> dict:
    """Ledger the member, stage + confirm a flag — the only
    road to a record."""
    surface.record_join(key, user_id, username=username,
                        name=name, group_name="Src")
    person = surface.reputation_person(
        key, user_id, username=username, name=name)
    prop = surface.stage_flag(
        key, user_id=user_id, person=person, category=category,
        note=note, by=by, by_label="Boss", direct=True,
        source="command", now=now)
    out = surface.decide_flag(key, prop["id"], approve=True,
                              by=by, by_label="Boss", now=now)
    assert out["ok"] and out["record"]
    return out["record"]


# ── creation discipline ───────────────────────────────────────


def test_no_record_without_admin_confirmation(tmp_path):
    surface = _surface(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    person = surface.reputation_person(A, "42")
    prop = surface.stage_flag(
        A, user_id="42", person=person, category="scam",
        note="smells", by="99", direct=True, source="command",
        now=NOW)
    # staged is not branded: the book is empty, joins are clean
    assert surface.reputation.stats(now=NOW)["live"] == 0
    assert not surface.reputation_join_check(
        A, "42", now=NOW)["found"]
    # discard leaves nothing but the audit line
    out = surface.decide_flag(A, prop["id"], approve=False,
                              by="99", now=NOW)
    assert out["ok"] and out["record"] is None
    assert surface.reputation.stats(now=NOW)["total"] == 0
    kinds = [a["kind"] for a in surface.audit(A)]
    assert "reputation_proposed" in kinds
    assert "reputation_proposal_discarded" in kinds


def test_confirmed_flag_is_attributed(tmp_path):
    surface = _surface(tmp_path)
    record = _confirm_flag(surface)
    assert record["status"] == "live"
    assert record["category"] == "scam"
    assert record["severity"] == 5
    assert record["source_group"] == A
    assert record["source_platform"] == "telegram-bot"
    assert record["confirmed_by"] == "99"
    assert record["person"]["ids"] == {"telegram-bot": "42"}
    assert record["person"]["usernames"] == ["ada"]
    assert [e["kind"] for e in record["events"]] == ["created"]
    kinds = [a["kind"] for a in surface.audit(A)]
    assert "reputation_flag" in kinds


def test_proposals_expire_unbranded(tmp_path):
    surface = _surface(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    prop = surface.stage_flag(
        A, user_id="42", person=surface.reputation_person(A, "42"),
        category="raid", note="", by="99", now=NOW)
    assert surface.flag_proposals(A, now=NOW + 60)
    later = NOW + PROPOSAL_TTL_S + 60
    assert surface.flag_proposals(A, now=later) == []
    assert surface.reputation.stats(now=later)["total"] == 0
    # a late decision finds nothing to decide
    assert surface.decide_flag(A, prop["id"], approve=True,
                               by="99", now=later)["ok"] is False
    kinds = [a["kind"] for a in surface.audit(A)]
    assert "reputation_proposal_expired" in kinds


def test_duplicate_proposal_does_not_stack(tmp_path):
    surface = _surface(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    person = surface.reputation_person(A, "42")
    one = surface.stage_flag(A, user_id="42", person=person,
                             category="scam", note="", by="99",
                             now=NOW)
    two = surface.stage_flag(A, user_id="42", person=person,
                             category="scam", note="", by="7",
                             now=NOW)
    assert two["id"] == one["id"] and two["duplicate"]
    assert len(surface.flag_proposals(A, now=NOW)) == 1


def test_book_persists_and_quarantines_corruption(tmp_path):
    surface = _surface(tmp_path)
    _confirm_flag(surface)
    # a fresh surface over the same files sees the record
    again = _surface(tmp_path)
    assert again.reputation.stats()["live"] == 1
    assert again.reputation_join_check(B, "42")["found"]
    # corruption is quarantined + raised, never silently reset
    path = tmp_path / "reputation.json"
    path.write_text("{torn", encoding="utf-8")
    with pytest.raises(ReputationStoreError):
        ReputationBook(path)
    assert not path.exists()
    assert list(tmp_path.glob("reputation.corrupt-*"))


# ── identity stitching (§29) ──────────────────────────────────


def test_username_change_does_not_shed_the_record(tmp_path):
    surface = _surface(tmp_path)
    record = _confirm_flag(surface)          # @ada, id 42
    # the same id reappears in another group under a NEW handle
    surface.record_join(B, "42", username="ada_new", name="Ada")
    check = surface.reputation_join_check(
        B, "42", username="ada_new", name="Ada")
    assert check["found"] and check["record"]["id"] == record["id"]
    # the stitch landed in the book: both handles name them now
    book = surface.reputation.get(record["id"])
    assert book["person"]["usernames"] == ["ada", "ada_new"]
    by_old = surface.reputation.live_for(
        {"ids": {}, "usernames": ["ada"], "names": []})
    by_new = surface.reputation.live_for(
        {"ids": {}, "usernames": ["ada_new"], "names": []})
    assert by_old and by_new


def test_username_bridges_platforms(tmp_path):
    surface = _surface(tmp_path)
    record = _confirm_flag(surface)          # telegram id 42 @ada
    # the same handle on Discord, a different numeric id
    surface.record_join(D, "99", username="ada", name="Ada L")
    check = surface.reputation_join_check(
        D, "99", username="ada", name="Ada L")
    assert check["found"] and check["record"]["id"] == record["id"]
    book = surface.reputation.get(record["id"])
    assert book["person"]["ids"]["discord"] == "99"
    # and the discord id alone finds them from now on
    assert surface.reputation_join_check(D, "99")["found"]


def test_display_name_alone_is_not_identity(tmp_path):
    surface = _surface(tmp_path)
    _confirm_flag(surface)
    # a stranger who merely shares the display name is NOT them
    surface.record_join(B, "77", username="someoneelse",
                        name="Ada")
    assert not surface.reputation_join_check(
        B, "77", username="someoneelse", name="Ada")["found"]


# ── aging + ordering + retraction ─────────────────────────────


def test_categories_age_at_their_own_retention(tmp_path):
    surface = _surface(tmp_path)
    spam = _confirm_flag(surface, category="spam-ring", now=NOW)
    scam = _confirm_flag(surface, user_id="43", username="max",
                         name="Max", category="scam", now=NOW)
    # 100 days on: the spam-ring flag (90d) has aged out, the
    # scam flag (730d) has not
    later = NOW + 100 * DAY
    assert not surface.reputation_join_check(
        B, "42", username="ada", now=later)["found"]
    assert surface.reputation.get(spam["id"])["status"] == "aged"
    assert surface.reputation_join_check(
        B, "43", username="max", now=later)["found"]
    assert surface.reputation.get(scam["id"])["status"] == "live"


def test_worst_record_first_one_warning(tmp_path):
    surface = _surface(tmp_path)
    _confirm_flag(surface, category="spam-ring", now=NOW)
    _confirm_flag(surface, category="scam", now=NOW + 10)
    _confirm_flag(surface, category="raid", now=NOW + 20)
    check = surface.reputation_join_check(B, "42", username="ada")
    assert check["found"]
    assert check["record"]["category"] == "scam"   # severity 5
    assert check["more"] == 2
    assert [r["category"] for r in check["records"]] == [
        "scam", "raid", "spam-ring"]


def test_source_retraction_clears_everywhere(tmp_path):
    surface = _surface(tmp_path)
    record = _confirm_flag(surface)
    assert surface.reputation_join_check(B, "42")["found"]
    # another group cannot retract what it did not source
    assert surface.reputation_retract_for(B, "42", by="7") == []
    assert surface.reputation_join_check(B, "42")["found"]
    # the source group can — and the record stops warning
    gone = surface.reputation_retract_for(A, "42", by="99")
    assert [r["id"] for r in gone] == [record["id"]]
    assert surface.reputation.get(record["id"])["status"] \
        == "retracted"
    assert not surface.reputation_join_check(B, "42")["found"]
    assert not surface.reputation_join_check(C, "42")["found"]
    kinds = [a["kind"] for a in surface.audit(A)]
    assert "reputation_retract" in kinds


# ── runtime rig ───────────────────────────────────────────────


class _Chat:
    def __init__(self, key=A, platform="telegram-bot",
                 kind="group", title="Src"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "99")
        self.sender = kw.get("sender", "Boss")
        self.sender_username = kw.get("sender_username", "boss")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""


class _Adapter:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = statuses or {}

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(str(user_id),
                                            "member")}

    def admin_restrict(self, chat, user_id, *, until=0, **kw):
        self.calls.append(("restrict", user_id))
        return {"ok": True}

    def admin_unrestrict(self, chat, user_id):
        self.calls.append(("unrestrict", user_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_unban(self, chat, user_id):
        self.calls.append(("unban", user_id))
        return {"ok": True}

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def kinds(self):
        return [c[0] for c in self.calls]


from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter):
        self.sent = []          # (text, buttons)
        self.gateway = SimpleNamespace(adapters={
            "telegram-bot": adapter, "telegram": adapter,
            "discord": adapter})
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return False


def _rig(tmp_path, adapter=None):
    adapter = adapter or _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    return rt, rt._group_surface_obj, adapter


def _join(rt, key, member, *, platform="telegram-bot",
          title="Room"):
    chat = _Chat(key=key, platform=platform, title=title)
    rt._group_surface_turn(_Msg(
        chat=chat, sender_id="99",
        meta={"service": "join", "new_members": [member]}))


def _texts(rt):
    return [t for t, _b in rt.sent]


def _audit_kinds(surface, key, kind):
    return [a for a in surface.audit(key, limit=200)
            if a["kind"] == kind]


ADA = {"id": "42", "username": "ada", "name": "Ada",
       "first_name": "Ada", "last_name": "L", "is_bot": False}


# ── join warnings: the three modes ────────────────────────────


def test_join_warning_public_mode(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    record = _confirm_flag(surface)
    _join(rt, B, ADA, title="Buyers")
    texts = _texts(rt)
    warned = [t for t in texts if "⚠️" in t and "scam" in t]
    assert len(warned) == 1
    assert "Ada" in warned[0] and "[42]" in warned[0]
    assert A in warned[0]                    # the source group
    # the buttons carry Dismiss with the record id on it
    buttons = [b for _t, b in rt.sent if b]
    flat = [pair for kb in buttons for row in kb for pair in row]
    assert any(data == f"repdismiss {record['id']}"
               for _label, data in flat)
    # journaled in the book and in the group's /glog
    events = surface.reputation.get(record["id"])["events"]
    assert any(e["kind"] == "warned" and e["group"] == B
               for e in events)
    assert _audit_kinds(surface, B, "reputation_warned")


def test_join_warning_admins_mode_is_log_only(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    _confirm_flag(surface)
    surface.update_settings(B, reputation={"warning": "admins"})
    _join(rt, B, ADA)
    assert not any("scam" in t for t in _texts(rt))
    warned = _audit_kinds(surface, B, "reputation_warned")
    assert warned and warned[-1]["mode"] == "admins"


def test_join_warning_restricted_mode_and_clear(tmp_path):
    rt, surface, adapter = _rig(tmp_path)
    _confirm_flag(surface)
    surface.update_settings(B,
                            reputation={"warning": "restricted"})
    _join(rt, B, ADA)
    assert ("restrict", "42") in adapter.calls
    state = surface.reputation_restricted(B, "42")
    assert state and state["category"] == "scam"
    assert any("restricted" in t and "⚠️" in t
               for t in _texts(rt))
    # an admin taps Clear (typed form of the same affordance)
    out = rt._control_repclear(
        "42", _Msg(chat=_Chat(key=B), sender_id="99"))
    assert "cleared" in out
    assert ("unrestrict", "42") in adapter.calls
    assert surface.reputation_restricted(B, "42") is None


def test_join_warning_off_mode_stays_silent(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    _confirm_flag(surface)
    surface.update_settings(B, reputation={"warning": "off"})
    _join(rt, B, ADA)
    assert not any("scam" in t for t in _texts(rt))
    supp = _audit_kinds(surface, B, "reputation_suppressed")
    assert supp and supp[-1]["suppressed"] == "mode_off"


def test_flagged_joiner_still_judged_by_acceptance(tmp_path):
    """§28 decides admission on the group's own criteria — the
    record changes nothing about the verdict."""
    rt, surface, adapter = _rig(tmp_path)
    _confirm_flag(surface)
    surface.update_settings(B, acceptance={
        "criteria": {"username": True, "photo": False,
                     "firstname": False, "lastname": False,
                     "hygiene": False, "banned": False,
                     "bots": False}})
    _join(rt, B, ADA)          # clean criteria + a live record
    assert "restrict" not in adapter.kinds()      # admitted whole
    assert any("Ada" in t for t in _texts(rt))    # welcomed
    assert any("scam" in t for t in _texts(rt))   # …and warned
    # a criteria failure restricts for the DOOR's reasons, with
    # the record noted alongside — not instead of them
    rt2, surface2, adapter2 = _rig(tmp_path / "two")
    _confirm_flag(surface2)
    surface2.update_settings(B, acceptance={
        "criteria": {"username": True, "photo": False,
                     "firstname": False, "lastname": False,
                     "hygiene": False, "banned": False,
                     "bots": False}})
    _join(rt2, B, {**ADA, "username": ""})
    assert "restrict" in adapter2.kinds()
    assert surface2.reputation_restricted(B, "42") is None


def test_gate_suppresses_warning_in_unclaimed_room(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    _confirm_flag(surface)
    # the owner's userbot merely SITS in this room: §33 — bot
    # systems neither speak nor brand anyone there
    _join(rt, U, ADA, platform="telegram", title="Foreign")
    assert not any("⚠️" in t or "scam" in t for t in _texts(rt))
    supp = _audit_kinds(surface, U, "reputation_suppressed")
    assert supp and supp[-1]["suppressed"] == "gate"


def test_local_dismiss_stops_future_warnings_only_here(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    record = _confirm_flag(surface)
    _join(rt, B, ADA)
    assert any("scam" in t for t in _texts(rt))
    out = rt._control_repdismiss(
        record["id"], _Msg(chat=_Chat(key=B), sender_id="99"))
    assert "dismissed" in out
    rt.sent.clear()
    _join(rt, B, ADA)          # rejoins B: silence now
    assert not any("scam" in t for t in _texts(rt))
    supp = _audit_kinds(surface, B, "reputation_suppressed")
    assert supp and supp[-1]["suppressed"] == "dismissed"
    # …but the record stands globally: group C still hears it
    _join(rt, C, ADA)
    assert any("scam" in t for t in _texts(rt))
    assert surface.reputation.get(record["id"])["status"] \
        == "live"


def test_dismiss_releases_a_reputation_hold(tmp_path):
    rt, surface, adapter = _rig(tmp_path)
    record = _confirm_flag(surface)
    surface.update_settings(B,
                            reputation={"warning": "restricted"})
    _join(rt, B, ADA)
    assert surface.reputation_restricted(B, "42")
    rt._control_repdismiss(
        record["id"], _Msg(chat=_Chat(key=B), sender_id="99"))
    assert surface.reputation_restricted(B, "42") is None
    assert ("unrestrict", "42") in adapter.calls


# ── commands: /flag, /rep, proposals from report + sensors ────


def test_flag_command_confirm_flow(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    msg = _Msg(chat=_Chat(key=A), sender_id="99")
    out = rt._control_flag("@ada scam took deposits", msg)
    text = getattr(out, "text", out)
    assert "Flag proposal" in text
    # nothing recorded yet
    assert surface.reputation.stats()["live"] == 0
    buttons = getattr(out, "buttons", None) or []
    fid = buttons[0][0][1].split()[-1]
    done = rt._control_flag(f"ok {fid}", msg)
    assert "recorded" in done and "scam" in done
    assert surface.reputation.stats()["live"] == 1
    # a non-admin cannot even stage one
    denied = rt._control_flag(
        "@ada scam", _Msg(chat=_Chat(key=A), sender_id="7"))
    assert "admin" in denied
    assert surface.reputation.stats()["live"] == 1


def test_flag_command_retract(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    _confirm_flag(surface)
    msg = _Msg(chat=_Chat(key=A), sender_id="99")
    out = rt._control_flag("retract @ada", msg)
    assert "retracted" in out
    assert surface.reputation.stats()["live"] == 0


def test_rep_view_and_book_shape(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    _confirm_flag(surface)
    msg = _Msg(chat=_Chat(key=A), sender_id="99")
    bare = rt._control_rep("", msg)
    assert "1 live" in bare
    out = rt._control_rep("@ada", msg)
    text = getattr(out, "text", out)
    assert "scam" in text and A in text
    clean = rt._control_rep(
        "@ada", _Msg(chat=_Chat(key=B), sender_id="99"))
    assert "scam" in getattr(clean, "text", clean)


def test_report_proposes_a_flag(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    out = rt._control_report(
        "@ada pushing a fake presale",
        _Msg(chat=_Chat(key=A), sender_id="7", sender="Mem"))
    assert "reported" in out
    props = surface.flag_proposals(A)
    assert len(props) == 1
    assert props[0]["category"] == "other"
    assert props[0]["source"] == "report"
    assert "presale" in props[0]["note"]
    # proposed is not branded
    assert surface.reputation.stats()["live"] == 0


def test_sensor_ban_proposes_a_flag(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    surface.record_join(A, "42", username="ada", name="Ada")
    rt._sensor_flag_proposal(surface, A, {
        "kind": "ban", "rule": "escalation",
        "_strike_rule": "antichannel", "user_id": "42"})
    props = surface.flag_proposals(A)
    assert len(props) == 1
    assert props[0]["category"] == "spam-ring"
    assert props[0]["source"] == "sensor"
    assert surface.reputation.stats()["live"] == 0
    # acceptance-driven bans never propose
    rt._sensor_flag_proposal(surface, A, {
        "kind": "ban", "rule": "acceptance", "user_id": "42"})
    assert len(surface.flag_proposals(A)) == 1


def test_gsettings_reputation_grammar(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    msg = _Msg(chat=_Chat(key=B), sender_id="99")
    out = rt._control_gsettings("reputation admins", msg)
    assert "admins only" in out
    settings = surface.store.settings(B)
    assert settings["reputation"]["warning"] == "admins"
    card = rt._control_gsettings("", msg)
    text = getattr(card, "text", card)
    assert "Reputation" in text and "admins" in text


def test_console_reputation_section(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    surface.store.group(A)
    dm = _Msg(chat=_Chat(key="telegram-bot:99", kind="dm",
                         title=""), sender_id="99")
    rt._control_console("", dm)
    out = rt._control_console("g1 reputation", dm)
    text = getattr(out, "text", out)
    assert "join warnings" in text
    done = rt._control_console("g1 reputation restricted", dm)
    assert "restricted" in getattr(done, "text", done)
    settings = surface.store.settings(A)
    assert settings["reputation"]["warning"] == "restricted"
