"""PARITY G5 — TELEGRAM USERBOT IDENTITY (platform parity
census bundle #8, 2026-11-08).

The census graded the whole bundle ABSENT: the Telegram
personal (userbot) surface could change nothing about its own
account — no profile name/about, no profile photo, no
username, no privacy settings, no way to add a contact —
while WhatsApp gained exactly that family in G3.  This build
lands the Telethon verbs in TelegramAdapter (account.
UpdateProfile / UpdateUsername, photos.UploadProfilePhoto /
DeletePhotos, account.Get/SetPrivacy, contacts.AddContact)
and drives them through the SAME /myprofile + /privacy
grammar G3 built, plus /addcontact.

The pins follow the truth-pass standard: composed tests drive
the real chain — owner command → handle_control →
runtime_chatops → the real TelegramAdapter → a fake Telethon
client that RECORDS the exact TL request objects (types and
fields asserted — a well-shaped lie cannot pass) — with fake
Telethon modules at the sys.modules seam (the house pattern;
no network, no real Telethon).  A capability counts only
when its request FIRES with the right shape, a landed change
journals, a refusal surfaces verbatim, and the owner's
standing stop gates the setters.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

ROOT = Path(__file__).resolve().parents[1]
TG_OWNER = "4242424242"        # the owner's Telegram id (fixture)
CONTACT_ID = 555001            # the addable contact's user id
PHONE = "+2348000000000"


# ── fake Telethon seam ─────────────────────────────────────

class _TLReq:
    """One fake TL request: kwargs land as attributes so the
    pins assert exact fields on the exact types."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

    def __repr__(self):
        fields = ", ".join(
            f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"{type(self).__name__}({fields})"


class UpdateProfileRequest(_TLReq):
    def __init__(self, first_name=None, last_name=None,
                 about=None):
        super().__init__(first_name=first_name,
                         last_name=last_name, about=about)


class UpdateUsernameRequest(_TLReq):
    def __init__(self, username):
        super().__init__(username=username)


class GetPrivacyRequest(_TLReq):
    def __init__(self, key):
        super().__init__(key=key)


class SetPrivacyRequest(_TLReq):
    def __init__(self, key, rules):
        super().__init__(key=key, rules=rules)


class GetFullUserRequest(_TLReq):
    def __init__(self, id):
        super().__init__(id=id)


class UploadProfilePhotoRequest(_TLReq):
    def __init__(self, file=None, **kw):
        super().__init__(file=file, **kw)


class DeletePhotosRequest(_TLReq):
    def __init__(self, id):
        super().__init__(id=id)


class GetUserPhotosRequest(_TLReq):
    def __init__(self, user_id, offset, max_id, limit):
        super().__init__(user_id=user_id, offset=offset,
                         max_id=max_id, limit=limit)


class ResolvePhoneRequest(_TLReq):
    def __init__(self, phone):
        super().__init__(phone=phone)


class AddContactRequest(_TLReq):
    def __init__(self, id, first_name, last_name, phone, **kw):
        super().__init__(id=id, first_name=first_name,
                         last_name=last_name, phone=phone, **kw)


class _NoArg:
    def __init__(self, *args, **kw):
        self.args = args
        self.kwargs = kw


def _key_class(name):
    return type(name, (_NoArg,), {})


# Input privacy keys (names mirror Telethon's exactly — the
# adapter resolves them by class name).
InputPrivacyKeyStatusTimestamp = _key_class(
    "InputPrivacyKeyStatusTimestamp")
InputPrivacyKeyProfilePhoto = _key_class(
    "InputPrivacyKeyProfilePhoto")
InputPrivacyKeyAbout = _key_class("InputPrivacyKeyAbout")
InputPrivacyKeyPhoneNumber = _key_class(
    "InputPrivacyKeyPhoneNumber")
InputPrivacyKeyChatInvite = _key_class("InputPrivacyKeyChatInvite")
InputPrivacyKeyPhoneCall = _key_class("InputPrivacyKeyPhoneCall")
InputPrivacyKeyPhoneP2P = _key_class("InputPrivacyKeyPhoneP2P")
InputPrivacyKeyForwards = _key_class("InputPrivacyKeyForwards")
InputPrivacyKeyVoiceMessages = _key_class(
    "InputPrivacyKeyVoiceMessages")
InputPrivacyKeyBirthday = _key_class("InputPrivacyKeyBirthday")

# Input privacy rules (the SET shapes).
InputPrivacyValueAllowAll = _key_class("InputPrivacyValueAllowAll")
InputPrivacyValueAllowContacts = _key_class(
    "InputPrivacyValueAllowContacts")
InputPrivacyValueDisallowAll = _key_class(
    "InputPrivacyValueDisallowAll")


class InputPhoto:
    def __init__(self, id, access_hash, file_reference):
        self.id = id
        self.access_hash = access_hash
        self.file_reference = file_reference


# Returned privacy rules (the READ shapes) — the adapter maps
# them by class name, so these names are the contract.
PrivacyValueAllowAll = _key_class("PrivacyValueAllowAll")
PrivacyValueAllowContacts = _key_class("PrivacyValueAllowContacts")
PrivacyValueDisallowAll = _key_class("PrivacyValueDisallowAll")
PrivacyValueAllowCloseFriends = _key_class(
    "PrivacyValueAllowCloseFriends")
PrivacyValueAllowPremium = _key_class("PrivacyValueAllowPremium")


class PrivacyValueAllowUsers(_NoArg):
    def __init__(self, users):
        self.users = users


class PrivacyValueDisallowUsers(_NoArg):
    def __init__(self, users):
        self.users = users


class UsernameOccupiedError(Exception):
    """Mirror of Telethon's RPC error for a taken handle."""


@pytest.fixture()
def fake_telethon(monkeypatch):
    types_mod = SimpleNamespace(
        InputPrivacyKeyStatusTimestamp=InputPrivacyKeyStatusTimestamp,
        InputPrivacyKeyProfilePhoto=InputPrivacyKeyProfilePhoto,
        InputPrivacyKeyAbout=InputPrivacyKeyAbout,
        InputPrivacyKeyPhoneNumber=InputPrivacyKeyPhoneNumber,
        InputPrivacyKeyChatInvite=InputPrivacyKeyChatInvite,
        InputPrivacyKeyPhoneCall=InputPrivacyKeyPhoneCall,
        InputPrivacyKeyPhoneP2P=InputPrivacyKeyPhoneP2P,
        InputPrivacyKeyForwards=InputPrivacyKeyForwards,
        InputPrivacyKeyVoiceMessages=InputPrivacyKeyVoiceMessages,
        InputPrivacyKeyBirthday=InputPrivacyKeyBirthday,
        InputPrivacyValueAllowAll=InputPrivacyValueAllowAll,
        InputPrivacyValueAllowContacts=InputPrivacyValueAllowContacts,
        InputPrivacyValueDisallowAll=InputPrivacyValueDisallowAll,
        InputPhoto=InputPhoto,
    )
    account_mod = SimpleNamespace(
        UpdateProfileRequest=UpdateProfileRequest,
        UpdateUsernameRequest=UpdateUsernameRequest,
        GetPrivacyRequest=GetPrivacyRequest,
        SetPrivacyRequest=SetPrivacyRequest)
    users_mod = SimpleNamespace(
        GetFullUserRequest=GetFullUserRequest)
    photos_mod = SimpleNamespace(
        UploadProfilePhotoRequest=UploadProfilePhotoRequest,
        DeletePhotosRequest=DeletePhotosRequest,
        GetUserPhotosRequest=GetUserPhotosRequest)
    contacts_mod = SimpleNamespace(
        AddContactRequest=AddContactRequest,
        ResolvePhoneRequest=ResolvePhoneRequest)
    functions_mod = SimpleNamespace(
        account=account_mod, users=users_mod,
        photos=photos_mod, contacts=contacts_mod)
    tl_mod = SimpleNamespace(functions=functions_mod,
                             types=types_mod)
    telethon_mod = SimpleNamespace(tl=tl_mod)
    for name, mod in (
            ("telethon", telethon_mod),
            ("telethon.tl", tl_mod),
            ("telethon.tl.functions", functions_mod),
            ("telethon.tl.functions.account", account_mod),
            ("telethon.tl.functions.users", users_mod),
            ("telethon.tl.functions.photos", photos_mod),
            ("telethon.tl.functions.contacts", contacts_mod),
            ("telethon.tl.types", types_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return types_mod


# ── fake client: canned account, recorded asks ─────────────

class _FakeClient:
    def __init__(self):
        self.requests: list = []
        self.me = SimpleNamespace(
            id=int(TG_OWNER), first_name="Peace",
            last_name="First", username="peacethefirst",
            phone=PHONE,
            photo=SimpleNamespace(photo_id=9001, dc_id=2))
        self.about = "building while others flex"
        self.about_error: Exception | None = None
        self.username_error: Exception | None = None
        self.privacy: dict[str, list] = {}
        self.privacy_error: Exception | None = None
        self.resolve_users = [SimpleNamespace(
            id=CONTACT_ID, first_name="Ada", last_name="")]
        self.user_photos = [SimpleNamespace(
            id=9001, access_hash=777,
            file_reference=b"ref-bytes")]
        self.uploaded: list[str] = []

    async def __call__(self, request):
        self.requests.append(request)
        kind = type(request).__name__
        if kind == "UpdateProfileRequest":
            if request.about is not None:
                self.about = request.about
            if request.first_name is not None:
                self.me.first_name = request.first_name
            if request.last_name is not None:
                self.me.last_name = request.last_name
            return self.me
        if kind == "UpdateUsernameRequest":
            if self.username_error is not None:
                raise self.username_error
            self.me.username = request.username
            return self.me
        if kind == "GetFullUserRequest":
            if self.about_error is not None:
                raise self.about_error
            return SimpleNamespace(
                full_user=SimpleNamespace(about=self.about),
                users=[self.me], chats=[])
        if kind == "GetPrivacyRequest":
            if self.privacy_error is not None:
                raise self.privacy_error
            return SimpleNamespace(
                rules=list(self.privacy.get(
                    type(request.key).__name__, [])),
                chats=[], users=[])
        if kind == "SetPrivacyRequest":
            twin = {
                "InputPrivacyValueAllowAll": PrivacyValueAllowAll,
                "InputPrivacyValueAllowContacts":
                    PrivacyValueAllowContacts,
                "InputPrivacyValueDisallowAll":
                    PrivacyValueDisallowAll,
            }
            landed = [twin[type(rule).__name__]()
                      for rule in request.rules]
            self.privacy[type(request.key).__name__] = landed
            return SimpleNamespace(rules=landed, chats=[],
                                   users=[])
        if kind == "UploadProfilePhotoRequest":
            return SimpleNamespace(
                photo=SimpleNamespace(id=9002), users=[self.me])
        if kind == "GetUserPhotosRequest":
            return SimpleNamespace(
                count=len(self.user_photos),
                photos=list(self.user_photos))
        if kind == "DeletePhotosRequest":
            return [9001]
        if kind == "ResolvePhoneRequest":
            return SimpleNamespace(
                peer=SimpleNamespace(user_id=CONTACT_ID),
                users=list(self.resolve_users), chats=[])
        if kind == "AddContactRequest":
            return SimpleNamespace(updates=[])
        raise AssertionError(kind)

    async def get_me(self):
        return self.me

    async def get_input_entity(self, ref):
        if ref == "me":
            return SimpleNamespace(_input="me",
                                   id=int(TG_OWNER))
        return SimpleNamespace(
            _input="user", id=getattr(ref, "id", 0))

    async def upload_file(self, path):
        self.uploaded.append(path)
        return SimpleNamespace(_input_file=path)


def _adapter(client):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = client
    adapter._loop = object()  # non-None: "connected"
    adapter._me_id = int(TG_OWNER)
    adapter._me_username = "peacethefirst"
    adapter._me_first_name = "Peace"
    adapter._sent_ids = set()
    adapter._seen_inbound = set()
    adapter._stopped = threading.Event()

    def _run(coro, timeout=60.0):
        return asyncio.new_event_loop().run_until_complete(coro)

    adapter._run_on_loop = _run  # type: ignore[method-assign]
    return adapter


def _reqs(client, kind):
    return [r for r in client.requests
            if type(r).__name__ == kind]


# ── adapter-level pins ─────────────────────────────────────

def test_adapter_get_profile_shape(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.get_profile()
    assert verdict["ok"] is True
    assert verdict["name"] == "Peace First"
    assert verdict["first_name"] == "Peace"
    assert verdict["last_name"] == "First"
    assert verdict["username"] == "peacethefirst"
    assert verdict["about"] == "building while others flex"
    assert verdict["has_photo"] is True
    asks = _reqs(client, "GetFullUserRequest")
    assert len(asks) == 1 and asks[0].id == "me"


def test_adapter_get_profile_about_failure_is_named(fake_telethon):
    client = _FakeClient()
    client.about_error = RuntimeError("full user read refused")
    adapter = _adapter(client)
    verdict = adapter.get_profile()
    assert verdict["ok"] is True
    assert "about" not in verdict
    assert verdict["info_error"] == "full user read refused"


def test_adapter_set_profile_name_request(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_profile_name("Devon", "Prime")
    assert verdict["ok"] is True
    assert verdict["name"] == "Devon Prime"
    asks = _reqs(client, "UpdateProfileRequest")
    assert len(asks) == 1
    assert asks[0].first_name == "Devon"
    assert asks[0].last_name == "Prime"
    assert asks[0].about is None
    # an empty first name refuses BEFORE traffic
    assert adapter.set_profile_name("  ")["ok"] is False
    assert len(_reqs(client, "UpdateProfileRequest")) == 1


def test_adapter_set_about_request(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_about("shipping daily")
    assert verdict["ok"] is True
    asks = _reqs(client, "UpdateProfileRequest")
    assert len(asks) == 1
    assert asks[0].about == "shipping daily"
    assert asks[0].first_name is None
    assert asks[0].last_name is None


def test_adapter_set_username_and_occupied_verbatim(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_username("@newhandle")
    assert verdict["ok"] is True
    assert verdict["username"] == "newhandle"
    asks = _reqs(client, "UpdateUsernameRequest")
    assert len(asks) == 1 and asks[0].username == "newhandle"
    client.username_error = UsernameOccupiedError(
        "USERNAME_OCCUPIED: the username is already taken")
    verdict = adapter.set_username("taken")
    assert verdict["ok"] is False
    assert "USERNAME_OCCUPIED" in verdict["error"]


def test_adapter_set_profile_photo_uploads(fake_telethon, tmp_path):
    pic = tmp_path / "me.jpg"
    pic.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 16)
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_profile_photo(str(pic))
    assert verdict["ok"] is True
    assert verdict["picture_id"] == "9002"
    assert client.uploaded == [str(pic)]
    asks = _reqs(client, "UploadProfilePhotoRequest")
    assert len(asks) == 1
    assert asks[0].file._input_file == str(pic)
    # a missing file refuses before any upload
    verdict = adapter.set_profile_photo(str(tmp_path / "gone.jpg"))
    assert verdict["ok"] is False
    assert len(_reqs(client, "UploadProfilePhotoRequest")) == 1


def test_adapter_remove_profile_photo_request(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_profile_photo(remove=True)
    assert verdict["ok"] is True
    asks = _reqs(client, "GetUserPhotosRequest")
    assert len(asks) == 1
    assert asks[0].offset == 0 and asks[0].limit == 1
    assert asks[0].user_id._input == "me"
    dels = _reqs(client, "DeletePhotosRequest")
    assert len(dels) == 1
    assert len(dels[0].id) == 1
    photo = dels[0].id[0]
    assert isinstance(photo, InputPhoto)
    assert photo.id == 9001
    assert photo.access_hash == 777
    assert photo.file_reference == b"ref-bytes"
    # no photo set = a refusal in words, no delete fired
    client.user_photos = []
    verdict = adapter.remove_profile_photo()
    assert verdict["ok"] is False
    assert "no profile photo" in verdict["error"]
    assert len(_reqs(client, "DeletePhotosRequest")) == 1


def test_adapter_privacy_read_mapping(fake_telethon):
    client = _FakeClient()
    client.privacy = {
        "InputPrivacyKeyStatusTimestamp":
            [PrivacyValueAllowContacts()],
        "InputPrivacyKeyProfilePhoto": [PrivacyValueAllowAll()],
        "InputPrivacyKeyPhoneNumber": [PrivacyValueDisallowAll()],
        "InputPrivacyKeyChatInvite": [
            PrivacyValueAllowContacts(),
            PrivacyValueDisallowUsers(users=[7])],
        "InputPrivacyKeyAbout": [PrivacyValueAllowUsers(users=[7])],
        "InputPrivacyKeyBirthday":
            [PrivacyValueAllowCloseFriends()],
        "InputPrivacyKeyForwards": [PrivacyValueAllowPremium()],
    }
    adapter = _adapter(client)
    verdict = adapter.get_privacy()
    assert verdict["ok"] is True
    settings = verdict["settings"]
    assert settings["lastseen"] == "contacts"
    assert settings["profile"] == "all"
    assert settings["phone"] == "none"
    assert settings["groupadd"] == "contact_blacklist"
    assert settings["about"] == "contact_allowlist"
    # unrepresentable shapes are NAMED, never squeezed
    assert "close friends" in settings["birthday"]
    assert "premium" in settings["forwards"]
    # keys the fake had no rules for read as Telegram's default
    assert "default" in settings["calls"]
    asks = _reqs(client, "GetPrivacyRequest")
    assert {type(a.key).__name__ for a in asks} == {
        "InputPrivacyKeyStatusTimestamp",
        "InputPrivacyKeyProfilePhoto", "InputPrivacyKeyAbout",
        "InputPrivacyKeyPhoneNumber", "InputPrivacyKeyChatInvite",
        "InputPrivacyKeyPhoneCall", "InputPrivacyKeyPhoneP2P",
        "InputPrivacyKeyForwards", "InputPrivacyKeyVoiceMessages",
        "InputPrivacyKeyBirthday"}


def test_adapter_privacy_set_request_and_readback(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.set_privacy("lastseen", "none")
    assert verdict["ok"] is True
    assert verdict["settings"] == {"lastseen": "none"}
    asks = _reqs(client, "SetPrivacyRequest")
    assert len(asks) == 1
    assert isinstance(asks[0].key, InputPrivacyKeyStatusTimestamp)
    assert len(asks[0].rules) == 1
    assert isinstance(asks[0].rules[0], InputPrivacyValueDisallowAll)
    # out-of-vocabulary pairs refuse BEFORE traffic
    assert adapter.set_privacy("lastseen", "somepeople")["ok"] is False
    assert adapter.set_privacy("bogus", "all")["ok"] is False
    assert len(_reqs(client, "SetPrivacyRequest")) == 1


def test_adapter_add_contact_flow(fake_telethon):
    client = _FakeClient()
    adapter = _adapter(client)
    verdict = adapter.add_contact("Ada", PHONE, "Lovelace")
    assert verdict["ok"] is True
    assert verdict["user_id"] == str(CONTACT_ID)
    assert verdict["name"] == "Ada Lovelace"
    asks = _reqs(client, "ResolvePhoneRequest")
    assert len(asks) == 1 and asks[0].phone == PHONE
    adds = _reqs(client, "AddContactRequest")
    assert len(adds) == 1
    assert adds[0].id.id == CONTACT_ID
    assert adds[0].first_name == "Ada"
    assert adds[0].last_name == "Lovelace"
    assert adds[0].phone == PHONE
    # an unresolvable number refuses in words, nothing added
    client.resolve_users = []
    verdict = adapter.add_contact("Ghost", "+2348999999999")
    assert verdict["ok"] is False
    assert "no Telegram account answers" in verdict["error"]
    assert len(_reqs(client, "AddContactRequest")) == 1
    # empty shapes refuse before traffic
    before = len(client.requests)
    assert adapter.add_contact("", PHONE)["ok"] is False
    assert adapter.add_contact("Ada", "")["ok"] is False
    assert len(client.requests) == before


def test_adapter_identity_when_disconnected():
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = None
    adapter._loop = None
    verdict = adapter.get_profile()
    assert verdict["ok"] is False
    assert "not connected" in verdict["error"]


def test_privacy_tables_agree_across_layers():
    from nomorals.social.chat.chatops import TG_PRIVACY_CATEGORIES
    from nomorals.social.chat.telegram import TG_PRIVACY_KEY_NAMES

    assert set(TG_PRIVACY_CATEGORIES) == set(TG_PRIVACY_KEY_NAMES)
    for values in TG_PRIVACY_CATEGORIES.values():
        assert values == ("all", "contacts", "none")


def test_catalog_registers_addcontact():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
    )

    assert "addcontact" in CHATOPS_COMMANDS
    assert "addcontact" in CONTROL_COMMANDS
    assert "addcontact" in COMMAND_DETAILS
    assert "addcontact" in LIST_ONELINERS
    assert any("addcontact" in names
               for _title, names in LIST_GROUPS)
    # the family it extends is still registered, unduplicated
    for kind in ("myprofile", "privacy"):
        assert kind in CHATOPS_COMMANDS
        assert kind in CONTROL_COMMANDS


# ── composed pins: the real command handlers ───────────────

@pytest.fixture()
def rig(fake_telethon, tmp_path):
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    client = _FakeClient()
    adapter = _adapter(client)
    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "partner.owner_sender_ids": TG_OWNER,
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway(
        {"local": LocalAdapter(), "telegram": adapter},
        db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    rec = sup.registry.upsert(AccountRecord(
        id=f"telegram:{TG_OWNER}", platform="telegram",
        identity=TG_OWNER, creds_dir="", status="linked",
        created_at=1.0))
    wa_rec = sup.registry.upsert(AccountRecord(
        id="whatsapp:2348000000000", platform="whatsapp",
        identity="2348000000000", creds_dir="", status="linked",
        created_at=1.0))
    svc = runtime._pairing_personal()
    env = {
        "runtime": runtime, "client": client,
        "adapter": adapter, "sup": sup, "rec": rec,
        "wa_rec": wa_rec, "svc": svc,
        "personal": svc["personal"], "vault": svc["personal"].vault,
        "tmp": tmp_path,
    }
    yield env
    runtime.stop()
    runtime.gateway.stop()


def _op(rig, text, *, platform="telegram", chat_id=TG_OWNER,
        sender_id=None, meta_extra=None):
    if sender_id is None:
        sender_id = TG_OWNER
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.DM, title="", peer=chat_id)
    meta = {"owner_command": True}
    if meta_extra:
        meta.update(meta_extra)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=sender_id,
        message_id=f"op-{time.time()}",
        ts=time.time(), meta=meta)
    return rig["runtime"].handle_control(
        text, chat.key, message=msg)


def _journal_rows(rig, account_id):
    return rig["vault"].journal(account_id)


def test_composed_profile_read(rig):
    out = _op(rig, "/myprofile")
    assert "Your Telegram identity:" in out
    assert "name: Peace First" in out
    assert "about: building while others flex" in out
    assert "username: @peacethefirst" in out
    assert "photo: set" in out


def test_composed_name_set_splits_first_last_and_journals(rig):
    out = _op(rig, "/myprofile name Devon Prime")
    assert "your Telegram name is now “Devon Prime”" in out
    asks = _reqs(rig["client"], "UpdateProfileRequest")
    assert len(asks) == 1
    assert asks[0].first_name == "Devon"
    assert asks[0].last_name == "Prime"
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "name"]
    assert rows and rows[-1]["text"] == "Devon Prime"
    assert rows[-1]["platform"] == "telegram"
    # a single word lands as first name, last name cleared
    out = _op(rig, "/myprofile name Devon")
    assert "“Devon”" in out
    asks = _reqs(rig["client"], "UpdateProfileRequest")
    assert asks[-1].first_name == "Devon"
    assert asks[-1].last_name == ""


def test_composed_about_set_and_clear(rig):
    out = _op(rig, "/myprofile about shipping daily")
    assert "shipping daily" in out
    asks = _reqs(rig["client"], "UpdateProfileRequest")
    assert asks[-1].about == "shipping daily"
    out = _op(rig, "/myprofile about clear")
    assert "cleared" in out
    asks = _reqs(rig["client"], "UpdateProfileRequest")
    assert asks[-1].about == ""
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "about"]
    assert rows and rows[-1]["text"] == ""


def test_composed_username_set_clear_and_journal(rig):
    out = _op(rig, "/myprofile username devonprime")
    assert "@devonprime" in out
    asks = _reqs(rig["client"], "UpdateUsernameRequest")
    assert len(asks) == 1 and asks[0].username == "devonprime"
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "username"]
    assert rows and rows[-1]["text"] == "devonprime"
    out = _op(rig, "/myprofile username clear")
    assert "removed" in out
    asks = _reqs(rig["client"], "UpdateUsernameRequest")
    assert asks[-1].username == ""


def test_composed_username_occupied_is_verbatim(rig):
    rig["client"].username_error = UsernameOccupiedError(
        "USERNAME_OCCUPIED: the username is already taken")
    out = _op(rig, "/myprofile username takename")
    assert "USERNAME_OCCUPIED" in out
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "username"]
    assert not rows  # a refused change journals nothing


def test_composed_username_on_whatsapp_is_honest(rig, monkeypatch):
    # the WhatsApp surface answers the username ask itself —
    # WhatsApp accounts have none (no adapter call is made:
    # the stub adapter would raise on any verb)
    stub = SimpleNamespace()
    monkeypatch.setattr(rig["runtime"], "_adapter_for",
                        lambda platform: stub)
    chat = ChatRef(platform="whatsapp",
                   chat_id="2348000000000@s.whatsapp.net",
                   kind=ChatKind.DM, title="",
                   peer="2348000000000@s.whatsapp.net")
    msg = ChatMessage(
        chat=chat, incoming=True,
        text="/myprofile username foo", sender="Owner",
        sender_id=TG_OWNER, message_id=f"op-{time.time()}",
        ts=time.time(), meta={"owner_command": True,
                              "caller_platform": "whatsapp"})
    out = rig["runtime"]._chatops_myprofile(
        "username foo", msg)
    assert "usernames are a Telegram thing" in out


def test_composed_photo_set_and_remove(rig, tmp_path):
    pic = tmp_path / "me.jpg"
    pic.write_bytes(b"\xff\xd8\xff\xe0" + b"x" * 16)
    out = _op(rig, "/myprofile photo", meta_extra={
        "replied_media_local": {"path": str(pic),
                                "kind": "photo"}})
    assert "photo is up" in out
    asks = _reqs(rig["client"], "UploadProfilePhotoRequest")
    assert len(asks) == 1
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "photo"]
    assert rows and rows[-1]["text"] == "9002"
    out = _op(rig, "/myprofile photo remove")
    assert "removed" in out
    assert _reqs(rig["client"], "DeletePhotosRequest")
    # no picture attached = a refusal in words, no upload
    before = len(_reqs(rig["client"], "UploadProfilePhotoRequest"))
    out = _op(rig, "/myprofile photo")
    assert "reply to a picture" in out
    assert len(_reqs(
        rig["client"], "UploadProfilePhotoRequest")) == before


def test_composed_privacy_read_plain_words_and_labels(rig):
    rig["client"].privacy = {
        "InputPrivacyKeyStatusTimestamp":
            [PrivacyValueAllowContacts()],
        "InputPrivacyKeyBirthday":
            [PrivacyValueAllowCloseFriends()],
    }
    out = _op(rig, "/privacy")
    assert "Your Telegram privacy:" in out
    assert "lastseen: my contacts (contacts)" in out
    assert "birthday: close friends only" in out
    assert "calls: not set" in out  # the default, named as such
    assert _reqs(rig["client"], "GetPrivacyRequest")


def test_composed_privacy_set_fires_and_journals(rig):
    out = _op(rig, "/privacy phone none")
    assert "phone is now nobody" in out
    asks = _reqs(rig["client"], "SetPrivacyRequest")
    assert len(asks) == 1
    assert isinstance(asks[0].key, InputPrivacyKeyPhoneNumber)
    assert isinstance(asks[0].rules[0], InputPrivacyValueDisallowAll)
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"
            and r.get("field") == "privacy:phone"]
    assert rows and rows[-1]["text"] == "none"
    assert rows[-1]["platform"] == "telegram"


def test_composed_privacy_bad_pair_refuses_before_traffic(rig):
    out = _op(rig, "/privacy lastseen contact_blacklist")
    assert "lastseen can be: all, contacts, none" in out
    out = _op(rig, "/privacy readreceipts all")
    assert "Telegram privacy categories are:" in out
    assert not _reqs(rig["client"], "SetPrivacyRequest")


def test_composed_setters_honor_owner_stop(rig):
    rig["sup"].registry.set_owner_stopped("telegram", True)
    try:
        out = _op(rig, "/myprofile name Halted")
        assert "standing stop" in out
        out = _op(rig, "/privacy lastseen all")
        assert "standing stop" in out
        out = _op(rig, f"/addcontact Ada | {PHONE}")
        assert "standing stop" in out
        assert not _reqs(rig["client"], "UpdateProfileRequest")
        assert not _reqs(rig["client"], "SetPrivacyRequest")
        assert not _reqs(rig["client"], "AddContactRequest")
    finally:
        rig["sup"].registry.set_owner_stopped("telegram", False)


def test_composed_addcontact_fires_and_journals(rig):
    out = _op(rig, f"/addcontact Ada Lovelace | {PHONE}")
    assert "Ada Lovelace is in your Telegram contacts" in out
    adds = _reqs(rig["client"], "AddContactRequest")
    assert len(adds) == 1
    assert adds[0].first_name == "Ada"
    assert adds[0].last_name == "Lovelace"
    assert adds[0].phone == PHONE
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "contact_added"]
    assert rows and rows[-1]["counterpart_id"] == str(CONTACT_ID)
    assert "Ada Lovelace" in rows[-1]["text"]
    assert PHONE in rows[-1]["text"]


def test_composed_addcontact_usage_and_unresolvable(rig):
    out = _op(rig, "/addcontact Ada")
    assert "usage: /addcontact" in out
    rig["client"].resolve_users = []
    out = _op(rig, "/addcontact Ghost | +2348999999999")
    assert "no Telegram account answers" in out
    assert not _reqs(rig["client"], "AddContactRequest")


def test_composed_addcontact_on_whatsapp_is_honest(rig):
    out = _op(rig, f"/addcontact Ada | {PHONE}",
              platform="whatsapp",
              chat_id="2348000000000@s.whatsapp.net")
    assert "phone's address book" in out
    assert not _reqs(rig["client"], "AddContactRequest")


def test_composed_stranger_is_refused(rig):
    # a stranger's message carries NONE of the owner stamps —
    # no adapter owner_command verdict (that stamp is the
    # paired account's own provenance), no allowlisted sender
    # id — so the operator gate refuses before the handler
    out = _op(rig, "/myprofile name Stranger",
              sender_id="9999999999",
              meta_extra={"owner_command": False})
    assert "your Telegram name is now" not in str(out)
    assert not _reqs(rig["client"], "UpdateProfileRequest")
    rows = [r for r in _journal_rows(rig, rig["rec"].id)
            if r.get("kind") == "profile_changed"]
    assert not rows
