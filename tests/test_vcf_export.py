"""/vcf — the group contacts export (owner ask, 2026-11-08).

Each number with its name, like the people on the group:
bare /vcf in a group exports THAT group's people as one
import-ready .vcf; /vcf <group name> from a DM exports the
named group; the manual shape cards people with no group in
common.  The roster merges EVERY route — WhatsApp bridge
roster + member ledger; Telegram member ledger + bot-API
admin list + the personal surface's member fetch — deduped,
best name winning (saved contact > push/display > ledger >
username).  People whose number no source can see are
counted in the reply and left uncarded; nothing is invented
(a Telegram id is never a number; an unresolved WhatsApp
LID is never a number).

Fixtures are the REAL shapes: bridge participant rows
(Go participantRow / Baileys participantRow), MTProto member
rows, ledger rows from the real GroupSurface, the pairing
registry + personal vault for the journal.  The produced
file bytes round-trip through a strict little vCard parser
in this file.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.social.chat import control  # noqa: E402
from nomorals.social.chat import vcf as vcf_organ  # noqa: E402
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef  # noqa: E402
from nomorals.social.chat.chatops import CHATOPS_COMMANDS  # noqa: E402

WA_GROUP = ChatRef(platform="whatsapp", chat_id="123@g.us",
                   kind=ChatKind.GROUP, title="Family Group")
WA_DM = ChatRef(platform="whatsapp",
                chat_id="2348011111111@s.whatsapp.net",
                kind=ChatKind.DM, title="Ada Lovelace",
                peer="2348011111111@s.whatsapp.net")
TG_GROUP = ChatRef(platform="telegram", chat_id="-100123",
                   kind=ChatKind.GROUP, title="TG Group")
TG_DM = ChatRef(platform="telegram", chat_id="777",
                kind=ChatKind.DM, title="Bob", peer="777")

OWNER = "owner-1"


def _msg(chat, *, sender_id=OWNER, text=""):
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender="Owner", sender_id=sender_id,
                       message_id="c1", reply_to="", mentioned=False,
                       meta={})


def parse_vcf(text: str) -> list[dict]:
    """A strict little vCard reader: BEGIN/END must balance,
    every card needs VERSION + FN, TELs collected verbatim."""
    assert "\n" not in text.replace("\r\n", ""), "bare LF found"
    lines: list[str] = []
    for raw in text.split("\r\n"):
        if raw.startswith(" ") and lines:
            lines[-1] += raw[1:]  # unfolding (RFC 2425)
        elif raw:
            lines.append(raw)
    cards: list[dict] = []
    cur: dict | None = None
    for line in lines:
        if line == "BEGIN:VCARD":
            assert cur is None, "nested BEGIN"
            cur = {"fn": "", "tels": []}
        elif line == "END:VCARD":
            assert cur is not None, "END without BEGIN"
            assert cur["fn"], "card without FN"
            cards.append(cur)
            cur = None
        elif cur is not None and line.startswith("FN:"):
            cur["fn"] = line[3:]
        elif cur is not None and line.startswith("TEL"):
            cur["tels"].append(line.split(":", 1)[1])
        elif cur is not None:
            assert line.startswith(("VERSION:", "N:")), line
    assert cur is None, "unterminated card"
    return cards


# ── the organ ────────────────────────────────────────────


def test_number_normalization():
    n = vcf_organ.normalize_number
    assert n("+234 801 234 5678") == ("+2348012345678",
                                      "2348012345678")
    assert n("+1 (202) 555-0134") == ("+12025550134",
                                      "12025550134")
    # Local format (trunk 0) is kept exactly as dialed.
    assert n("0803 222 3333") == ("08032223333", "08032223333")
    assert n("12345") == ("", "")
    assert n("") == ("", "")
    assert n("not a number") == ("", "")


def test_whatsapp_jid_numbers():
    j = vcf_organ.whatsapp_number_from_jid
    assert j("2348012345678@s.whatsapp.net") == (
        "+2348012345678", "2348012345678")
    assert j("2348012345678:4@s.whatsapp.net")[1] == "2348012345678"
    assert j("9988776655@lid") == ("", "")       # anonymous id
    assert j("123@g.us") == ("", "")            # a group, not a person
    assert j("2348012345678")[1] == "2348012345678"


def test_vcard_shape_escaping_and_crlf():
    card = vcf_organ.build_vcard("Doe, Jane; Dr.",
                                 ["+2348012345678"])
    assert card.startswith("BEGIN:VCARD\r\nVERSION:3.0\r\n")
    assert "FN:Doe\\, Jane\\; Dr.\r\n" in card
    assert "TEL;TYPE=CELL:+2348012345678\r\n" in card
    assert card.endswith("END:VCARD\r\n")
    cards = parse_vcf(card)
    assert len(cards) == 1 and cards[0]["tels"] == ["+2348012345678"]


def test_vcard_multi_number_and_nameless_fallback():
    card = vcf_organ.build_vcard("", ["+11111111", "+22222222"])
    cards = parse_vcf(card)
    assert cards[0]["fn"] == "+11111111"  # FN falls back to number
    assert cards[0]["tels"] == ["+11111111", "+22222222"]
    assert vcf_organ.build_vcard("Ada", []) == ""


def test_vcard_folding_keeps_lines_within_75_octets():
    name = "A" * 90 + " Éclair"  # multibyte char past the fold
    card = vcf_organ.build_vcard(name, ["+2348012345678"])
    for raw in card.split("\r\n"):
        assert len(raw.encode("utf-8")) <= 75, raw
    cards = parse_vcf(card)
    assert cards[0]["fn"] == name  # unfolding restores it whole


def test_merge_people_dedup_best_name_and_counts():
    rows = [
        # Same Telegram person in two sources: the display
        # name beats the ledger's shorter observed name.
        {"platform_id": "555", "name": "boss",
         "name_rank": vcf_organ.RANK_LEDGER, "username": "boss",
         "numbers": [], "source": "member ledger"},
        {"platform_id": "555", "name": "Boss Man",
         "name_rank": vcf_organ.RANK_DISPLAY, "username": "boss",
         "numbers": ["+234 805 555 5555"],
         "source": "personal member list"},
        # Two WhatsApp rows, one person, met by NUMBER.
        {"platform_id": "2348066666666@s.whatsapp.net",
         "name": "", "name_rank": vcf_organ.RANK_CONTACT,
         "numbers": ["+2348066666666"], "source": "WhatsApp roster"},
        {"platform_id": "2348066666666", "name": "Sade",
         "name_rank": vcf_organ.RANK_LEDGER,
         "numbers": ["2348066666666"], "source": "member ledger"},
        {"platform_id": "777", "name": "Botty", "bot": True,
         "numbers": ["+10000000000"], "source": "personal member list"},
        {"platform_id": "778", "name": "", "deleted": True,
         "numbers": [], "source": "personal member list"},
        {"platform_id": "779", "name": "No Number",
         "name_rank": vcf_organ.RANK_DISPLAY, "numbers": [],
         "source": "personal member list"},
    ]
    out = vcf_organ.merge_people(rows)
    assert out["dropped_bots"] == 1
    assert out["dropped_deleted"] == 1
    people = {p["name"]: p for p in out["people"]}
    assert set(people) == {"Boss Man", "Sade", "No Number"}
    assert people["Boss Man"]["numbers"] == ["+2348055555555"]
    assert people["Boss Man"]["sources"] == [
        "member ledger", "personal member list"]
    assert people["Sade"]["numbers"] == ["+2348066666666"]
    assert people["No Number"]["numbers"] == []


def test_manual_entries_grammar():
    ok = vcf_organ.parse_manual_entries(
        "Ada Lovelace | +234 801 111 1111 | 0803 222 3333\n"
        "Doe, Jane | +1 202 555 0134")
    assert ok["ok"]
    ada, jane = ok["entries"]
    assert ada["numbers"] == ["+2348011111111", "08032223333"]
    assert jane["name"] == "Doe, Jane"
    bad = vcf_organ.parse_manual_entries("Ada | not-a-number")
    assert not bad["ok"] and "phone number" in bad["error"]
    bad2 = vcf_organ.parse_manual_entries("Ada Lovelace")
    assert not bad2["ok"] and "Name | number" in bad2["error"]
    bad3 = vcf_organ.parse_manual_entries("Ada | ")
    assert not bad3["ok"]


def test_registry_presence_and_owner_only():
    assert "vcf" in CHATOPS_COMMANDS
    assert "vcf" in control.CONTROL_COMMANDS
    assert "vcf" in control.COMMAND_DETAILS
    assert "vcf" in control.LIST_ONELINERS
    assert "vcf" not in control.PUBLIC_COMMANDS
    homes = sum(1 for _label, kinds in control.LIST_GROUPS
                if "vcf" in kinds)
    assert homes == 1


# ── fake surfaces ────────────────────────────────────────


class _FakeWA:
    name = "whatsapp"
    user_jid = "2348012345678@s.whatsapp.net"

    def __init__(self):
        self.roster: list[dict] = []
        self.info: dict = {}
        self.member_status = "member"

    def chats(self, limit=30):
        return [{"id": "123@g.us", "title": "Family Group"},
                {"id": "2348011111111@s.whatsapp.net",
                 "title": "Ada Lovelace"}]

    def group_participants(self, ref):
        return list(self.roster)

    def group_info(self, ref):
        return dict(self.info)

    def admin_member(self, ref, user_id):
        return {"ok": True, "status": self.member_status}


class _FakeTGPersonal:
    name = "telegram"

    def __init__(self):
        self.members: list[dict] = []

    def admin_members_full(self, ref, limit=20000):
        return {"ok": True, "members": list(self.members),
                "count": len(self.members)}

    def admin_members(self, ref, limit=100):
        return {"ok": True, "members": list(self.members),
                "count": len(self.members)}

    def admin_member(self, ref, user_id):
        return {"ok": True, "status": "member"}


class _FakeTGBot:
    name = "telegram-bot"

    def __init__(self):
        self.admins: list[dict] = []
        self.status_by_user: dict[str, str] = {}

    def admin_administrators(self, ref):
        return {"ok": True, "administrators": list(self.admins)}

    def admin_member(self, ref, user_id):
        return {"ok": True, "status": self.status_by_user.get(
            str(user_id), "member")}

    def admin_member_count(self, ref):
        return {"ok": True, "count": 3}


class _FakeGateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent_files: list[dict] = []

    def send_file(self, platform, chat, path, caption="",
                  name="", **kwargs):
        self.sent_files.append({
            "platform": platform, "chat": chat, "path": path,
            "caption": caption, "name": name,
            "bytes": Path(path).read_bytes()})
        return SimpleNamespace(ok=True, error="")

    def send(self, platform, chat, text):
        return SimpleNamespace(ok=True, error="")


def _runtime(tmp_path, wa=None, tgp=None, tgb=None):
    from nomorals.agents.partner.runtime_chatops import RuntimeChatOpsMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    wa = wa if wa is not None else _FakeWA()
    tgp = tgp if tgp is not None else _FakeTGPersonal()
    tgb = tgb if tgb is not None else _FakeTGBot()
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.upsert(AccountRecord(
        id="telegram:777000", platform="telegram",
        identity="777000", status="linked", created_at=1.0))
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    gateway = _FakeGateway({"whatsapp": wa, "telegram": tgp,
                            "telegram-bot": tgb})
    surface = GroupSurface(
        GroupStore(tmp_path / "groups.json").load())

    class _RT(RuntimeChatOpsMixin, RuntimeProvisionMixin,
              RuntimeGroupMixin):
        pass

    rt = object.__new__(_RT)
    rt.context = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = gateway
    rt._pair_personal_svc = {
        "registry": registry, "personal": personal}
    rt._housekeeping_beat = lambda: None
    rt._group_surface_obj = surface
    rt._is_operator = (  # type: ignore[method-assign]
        lambda message: str(message.sender_id) == OWNER)
    return rt, wa, tgp, tgb, gateway, surface, personal


def _seed_wa(rt, surface):
    wa = rt.gateway.adapters["whatsapp"]
    wa.info = {"subject": "Family Group", "size": 4,
               "admins": ["2348022222222@s.whatsapp.net"]}
    wa.roster = [
        {"jid": "2348011111111@s.whatsapp.net",
         "name": "Ada Lovelace", "role": ""},
        {"jid": "2348022222222@s.whatsapp.net",
         "name": "", "role": "admin"},
        {"jid": "9988776655@lid", "name": "Ghost Lid",
         "role": ""},
    ]
    key = "whatsapp:123@g.us"
    surface.note_join(key, "2348022222222", name="Bola",
                      ts=1000.0)
    surface.note_join(key, "2348033333333", name="Chidi",
                      ts=1000.0)
    return wa


# ── the handler, composed ────────────────────────────────


def test_wa_group_export_end_to_end(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, surface, personal = \
        _runtime(tmp_path)
    _seed_wa(rt, surface)
    out = rt._control_chatops("vcf", "", _msg(WA_GROUP))
    assert out == ("📇 sent family-group-contacts.vcf — "
                   "3 contact cards for Family Group.")
    assert len(gateway.sent_files) == 1
    sent = gateway.sent_files[0]
    assert sent["name"] == "family-group-contacts.vcf"
    assert sent["chat"].key == WA_GROUP.key
    caption = sent["caption"]
    assert "Family Group — 4 people · 3 contact cards written" \
        in caption
    assert "1 without a visible number" in caption
    assert "WhatsApp roster" in caption
    assert "member ledger" in caption
    cards = parse_vcf(sent["bytes"].decode("utf-8"))
    by_fn = {c["fn"]: c["tels"] for c in cards}
    assert by_fn == {
        "Ada Lovelace": ["+2348011111111"],
        "Bola": ["+2348022222222"],       # name from the ledger
        "Chidi": ["+2348033333333"],      # ledger-only person
    }
    rows = [r for r in personal.vault.journal()
            if r.get("kind") == "vcf_exported"]
    assert len(rows) == 1
    assert rows[0]["people"] == 4 and rows[0]["cards"] == 3
    assert rows[0]["hidden"] == 1
    assert rows[0]["counterpart_id"] == "whatsapp:123@g.us"


def test_tg_group_export_merges_every_route(tmp_path):
    rt, _wa, tgp, tgb, gateway, surface, personal = \
        _runtime(tmp_path)
    tgp.members = [
        {"id": "555", "name": "Boss Man", "username": "boss",
         "phone": "+2348055555555", "bot": False,
         "deleted": False},
        {"id": "556", "name": "Hidden Hana", "username": "hana",
         "phone": "", "bot": False, "deleted": False},
        {"id": "557", "name": "Botty", "username": "botty",
         "phone": "+10000000000", "bot": True, "deleted": False},
        {"id": "558", "name": "", "username": "",
         "phone": "", "bot": False, "deleted": True},
    ]
    tgb.admins = [{"id": "555", "name": "Boss",
                   "username": "boss", "status": "creator"}]
    key = "telegram:-100123"
    surface.note_join(key, "555", name="boss", username="boss",
                      ts=1000.0)
    surface.note_join(key, "559", name="Ledger Lara",
                      ts=1000.0)
    out = rt._control_chatops("vcf", "", _msg(TG_GROUP))
    assert out == ("📇 sent tg-group-contacts.vcf — "
                   "1 contact card for TG Group.")
    sent = gateway.sent_files[0]
    caption = sent["caption"]
    # People: Boss (carded), Hana + Lara (no visible number).
    # The bot and the deleted account are not people.
    assert "TG Group — 3 people · 1 contact card written" \
        in caption
    assert "2 without a visible number" in caption
    for label in ("personal member list", "admin list",
                  "member ledger"):
        assert label in caption, label
    cards = parse_vcf(sent["bytes"].decode("utf-8"))
    assert len(cards) == 1
    # One card only: merged across three sources, best name.
    assert cards[0]["fn"] == "Boss Man"
    assert cards[0]["tels"] == ["+2348055555555"]
    rows = [r for r in personal.vault.journal()
            if r.get("kind") == "vcf_exported"]
    assert len(rows) == 1 and rows[0]["platform"] == "telegram"
    assert rows[0]["cards"] == 1 and rows[0]["hidden"] == 2


def test_vcf_stranger_refused_group_admin_allowed(tmp_path):
    rt, wa, _tgp, _tgb, gateway, surface, _p = _runtime(tmp_path)
    _seed_wa(rt, surface)
    out = rt._control_chatops(
        "vcf", "", _msg(WA_GROUP, sender_id="member-9"))
    assert "group admin's" in out
    assert gateway.sent_files == []
    wa.member_status = "administrator"
    out = rt._control_chatops(
        "vcf", "", _msg(WA_GROUP, sender_id="admin-7"))
    assert out.startswith("📇 sent family-group-contacts.vcf")
    assert len(gateway.sent_files) == 1
    # An admin's export is the GROUP's record, not the
    # owner's personal journal.
    audit = surface.audit("whatsapp:123@g.us")
    assert any(r.get("kind") == "vcf_exported" for r in audit)


def test_vcf_manual_rung(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, _s, _p = _runtime(tmp_path)
    tail = ("Ada Lovelace | +234 801 111 1111 | 0803 222 3333\n"
            "Doe, Jane | +1 202 555 0134")
    out = rt._control_chatops("vcf", tail, _msg(WA_DM))
    assert out == ("📇 sent contacts.vcf — 2 contact cards "
                   "for manual contacts.")
    sent = gateway.sent_files[0]
    assert sent["name"] == "contacts.vcf"
    cards = parse_vcf(sent["bytes"].decode("utf-8"))
    by_fn = {c["fn"]: c["tels"] for c in cards}
    assert by_fn["Ada Lovelace"] == ["+2348011111111",
                                     "08032223333"]
    assert by_fn["Doe\\, Jane"] == ["+12025550134"]


def test_vcf_manual_bad_number_refuses_whole_batch(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, _s, _p = _runtime(tmp_path)
    out = rt._control_chatops(
        "vcf", "Ada | not-a-number", _msg(WA_DM))
    assert "phone number" in out
    assert gateway.sent_files == []


def test_vcf_by_group_name_from_dm(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, surface, _p = \
        _runtime(tmp_path)
    _seed_wa(rt, surface)
    out = rt._control_chatops(
        "vcf", "Family Group", _msg(WA_DM))
    assert out.startswith("📇 sent family-group-contacts.vcf")
    # The file lands with the OWNER (this DM), not in the group.
    assert gateway.sent_files[0]["chat"].key == WA_DM.key


def test_vcf_no_visible_numbers_no_file(tmp_path):
    rt, _wa, tgp, _tgb, gateway, _s, _p = _runtime(tmp_path)
    tgp.members = [
        {"id": "556", "name": "Hidden Hana", "username": "hana",
         "phone": "", "bot": False, "deleted": False},
    ]
    out = rt._control_chatops("vcf", "", _msg(TG_GROUP))
    assert "1 without a visible number" in out
    assert "no file" in out
    assert gateway.sent_files == []


def test_vcf_bare_in_dm_answers_how(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, _s, _p = _runtime(tmp_path)
    out = rt._control_chatops("vcf", "", _msg(WA_DM))
    assert out.startswith("run /vcf inside a group")
    assert gateway.sent_files == []


def test_vcf_unknown_platform_line(tmp_path):
    rt, _wa, _tgp, _tgb, gateway, _s, _p = _runtime(tmp_path)
    chat = ChatRef(platform="discord", chat_id="42",
                   kind=ChatKind.GROUP, title="Discord Room")
    out = rt._control_chatops("vcf", "", _msg(chat))
    assert "WhatsApp and Telegram" in out
    assert gateway.sent_files == []


# ── the Telegram adapter's phone rung ────────────────────


def test_telegram_member_rows_carry_phone_when_visible():
    from nomorals.social.chat.telegram import TelegramAdapter

    users = [
        SimpleNamespace(
            id=555, first_name="Boss", last_name="Man",
            username="boss", bot=False, deleted=False,
            phone="2348055555555", status=None),
        SimpleNamespace(
            id=556, first_name="Hidden", last_name="Hana",
            username="hana", bot=False, deleted=False,
            phone=None, status=None),
    ]

    class FakeClient:
        async def iter_participants(self, entity, limit=100):
            for u in users:
                yield u

    adapter = object.__new__(TelegramAdapter)
    adapter._client = FakeClient()

    async def fake_resolve(chat):
        return "ENT"

    adapter._resolve = fake_resolve  # type: ignore[method-assign]
    rows = asyncio.run(adapter._members_coro(TG_GROUP, 10))
    assert rows[0]["phone"] == "2348055555555"
    assert rows[1]["phone"] == ""  # hidden stays empty, honestly
