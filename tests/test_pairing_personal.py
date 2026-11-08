"""Pairing slices 2–3 + personal surfaces (group build §15).

The registry's new shape (per-account toggles, routing primary,
dropped counts), the supervision sweep flipping states to the
truth (fake bridge processes, fake probes), the Telegram pairing
flow state machine (fake MTProto client: code → linked, 2FA path,
abandoned flow expiry), the routing primitive, and the personal
surfaces organ: status saving, view-once preservation, DM
antidelete/antiedit recovery — owner-only by construction (the
organ has no send path at all), caps enforced, toggles honored.
Plus the transports: the WhatsApp adapter carrying the bridge's
new status_post / view-once events, and the pairing page served
by the API behind the sys.config capability.
"""
from __future__ import annotations

import http.client
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
    PoolSupervision,
    resolve_account,
)
from nomorals.social.personal import (
    PersonalStoreError,
    PersonalSurfaces,
    PersonalVault,
)
from nomorals.social.telegram_pairing import (
    PairingFlowError,
    TelegramPairer,
    TwoFactorRequired,
)


# ── fakes ───────────────────────────────────────────────────────────

class FakeProc:
    def __init__(self, lines=()):
        self.stdout = iter(list(lines))
        self.terminated = False
        self._poll = None

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0


def make_supervisor(tmp_path, lines=()):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn)
    return sup, registry, spawns


class FakeTGClient:
    """The MTProto surface the pairer drives, without Telethon."""

    def __init__(self, *, needs_2fa=False, authorized=True,
                 user_id="777001"):
        self.needs_2fa = needs_2fa
        self.authorized = authorized
        self.user_id = user_id
        self.connected = False
        self.signed_in = False
        self.code_sent_to = ""

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def send_code(self, phone):
        self.code_sent_to = phone
        return "hash-abc"

    async def sign_in_code(self, phone, code, code_hash):
        if code != "12345":
            raise ValueError("bad code")
        if self.needs_2fa:
            raise TwoFactorRequired("2fa on")
        self.signed_in = True

    async def sign_in_password(self, password):
        if password != "hunter2":
            raise ValueError("bad password")
        self.signed_in = True

    async def is_authorized(self):
        return self.authorized and (self.signed_in or self.authorized)

    async def me_id(self):
        return self.user_id


def make_pairer(tmp_path, client=None, clock=None, **kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    client = client or FakeTGClient()
    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "tg_sessions",
        api_id=12345, api_hash="hash",
        client_factory=lambda stem, i, h: client,
        clock=clock or time.time, **kw)
    return pairer, registry, client


# ── registry: toggles / routing / counts ────────────────────────────

def test_toggles_round_trip_and_unknown_raises(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    assert rec.toggle("antidelete") is False  # default OFF
    registry.set_toggles(rec.id, antidelete=True, view_once=True)
    again = PairingRegistry(tmp_path / "paired.json")
    got = again.get(rec.id)
    assert got.toggle("antidelete") is True
    assert got.toggle("view_once") is True
    assert got.toggle("status_saving") is False
    with pytest.raises(ValueError):
        registry.set_toggles(rec.id, antdelete=True)  # typo ≠ success


def test_routing_resolution(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    assert resolve_account(registry, "whatsapp") is None
    sup, registry, _ = make_supervisor(tmp_path)
    first = sup.pair_whatsapp("2348012345678")
    second = sup.pair_whatsapp("2349098765432")
    # neither linked → first registered
    assert resolve_account(registry, "whatsapp").id == first.id
    # first linked wins
    registry.set_status(second.id, "linked")
    assert resolve_account(registry, "whatsapp").id == second.id
    # named primary beats first-linked
    registry.set_primary(first.id)
    assert resolve_account(registry, "whatsapp").id == first.id
    # exactly one primary per platform
    registry.set_primary(second.id)
    assert registry.get(first.id).primary is False
    assert registry.get(second.id).primary is True
    # explicit wins; explicit from the wrong platform resolves nothing
    assert resolve_account(registry, "whatsapp", first.id).id == first.id
    assert resolve_account(registry, "telegram", first.id) is None
    assert resolve_account(registry, "telegram") is None


def test_counts_include_dropped(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    registry.set_status(rec.id, "down")
    counts = sup.counts(probe=False)
    assert counts == {"registered": 1, "connected": 0, "dropped": 1}


# ── supervision sweep ───────────────────────────────────────────────

def test_sweep_flips_dead_bridge_down_and_relaunches(tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    registry.set_status(rec.id, "linked")
    # the bridge process dies (poll reports an exit)
    spawns[0][2]._poll = 0
    clock = [1000.0]
    sweep = PoolSupervision(registry, sup, clock=lambda: clock[0])
    changes = sweep.sweep()
    assert registry.get(rec.id).status == "down"
    assert any("down" in c["change"] or "relaunched" in c["change"]
               for c in changes)
    # relaunched through the supervisor's own start (spawn #2)
    assert len(spawns) == 2
    # the relaunched process is alive → next sweep is quiet
    assert sweep.sweep() == []


def test_sweep_leaves_healthy_and_stopped_alone(tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    registry.set_status(rec.id, "linked")
    sweep = PoolSupervision(registry, sup)
    assert sweep.sweep() == []          # process alive: bridge owns it
    assert len(spawns) == 1
    sup.stop(rec.id)
    assert sweep.sweep() == []          # stopped is a owner's word


def test_sweep_telegram_probe_flips_down(tmp_path):
    pairer, registry, client = make_pairer(tmp_path)
    pairer.start("2348012345678")
    pairer.confirm_code("telegram:2348012345678", "12345")
    assert registry.get("telegram:2348012345678").status == "linked"
    client.authorized = False
    # probe needs a session file to exist
    stem = registry.get("telegram:2348012345678").session_path
    Path(stem + ".session").write_text("x", encoding="utf-8")
    sweep = PoolSupervision(registry, None, pairer,
                            tg_probe_interval=0)
    changes = sweep.sweep()
    assert registry.get("telegram:2348012345678").status == "down"
    assert changes and "down" in changes[0]["change"]


# ── telegram pairing flow ───────────────────────────────────────────

def test_telegram_pairing_happy_path(tmp_path):
    pairer, registry, client = make_pairer(tmp_path)
    rec = pairer.start("+234 801 234 5678", "second line")
    assert rec.platform == "telegram"
    assert rec.id == "telegram:2348012345678"
    assert rec.session_path.endswith("tg-2348012345678")
    assert registry.get(rec.id).status == "pairing"
    assert client.code_sent_to == "2348012345678"
    pending = pairer.pending(rec.id)
    assert pending["state"] == "code_sent"
    out = pairer.confirm_code(rec.id, "12345")
    assert out["status"] == "linked"
    got = registry.get(rec.id)
    assert got.status == "linked"
    assert got.remote_id == "777001"
    assert got.label == "second line"
    assert pairer.pending(rec.id) is None  # flow consumed


def test_telegram_pairing_two_factor(tmp_path):
    pairer, registry, _ = make_pairer(
        tmp_path, client=FakeTGClient(needs_2fa=True))
    rec = pairer.start("2348012345678")
    out = pairer.confirm_code(rec.id, "12345")
    assert out["status"] == "password_needed"
    assert pairer.pending(rec.id)["state"] == "password_needed"
    assert registry.get(rec.id).status == "pairing"  # not yet linked
    out = pairer.confirm_password(rec.id, "hunter2")
    assert out["status"] == "linked"
    assert registry.get(rec.id).status == "linked"


def test_telegram_pairing_no_flow_and_bad_number(tmp_path):
    pairer, _, _ = make_pairer(tmp_path)
    with pytest.raises(PairingFlowError):
        pairer.confirm_code("telegram:2348012345678", "12345")
    with pytest.raises(PairingFlowError):
        pairer.start("123")


def test_telegram_pairing_missing_credentials_fails_loudly(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "s",
        api_id="", api_hash="",
        client_factory=lambda stem, i, h: FakeTGClient())
    with pytest.raises(PairingFlowError, match="api_id"):
        pairer.start("2348012345678")


def test_telegram_abandoned_flow_expires(tmp_path):
    clock = [1000.0]
    pairer, registry, _ = make_pairer(
        tmp_path, clock=lambda: clock[0], flow_ttl=600)
    rec = pairer.start("2348012345678")
    clock[0] += 601
    assert pairer.expire_flows() == [rec.id]
    assert registry.get(rec.id).status == "registered"
    with pytest.raises(PairingFlowError, match="expired|no live"):
        pairer.confirm_code(rec.id, "12345")


def test_telegram_probe_truth(tmp_path):
    pairer, registry, client = make_pairer(tmp_path)
    rec = pairer.start("2348012345678")
    # no session file yet → down, honestly
    assert pairer.probe(rec) == "down"
    Path(rec.session_path + ".session").write_text(
        "x", encoding="utf-8")
    assert pairer.probe(rec) == "linked"
    client.authorized = False
    client.signed_in = False
    assert pairer.probe(rec) == "down"


# ── personal surfaces ───────────────────────────────────────────────

def make_personal(tmp_path, **vault_kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal", **vault_kw)
    personal = PersonalSurfaces(registry, vault)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    return personal, registry, vault, rec


def test_antidelete_recovers_original_owner_only(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, antidelete=True)
    personal.observe_message(
        "whatsapp", chat_key="whatsapp:dm:111@s.whatsapp.net",
        counterpart_id="111@s.whatsapp.net", sender_id="111",
        sender_name="Ada", message_id="m1", text="meet at 7",
        ts=10.0)
    out = personal.observe_delete(
        "whatsapp", chat_key="whatsapp:dm:111@s.whatsapp.net",
        counterpart_id="111@s.whatsapp.net",
        deleted_ids=["m1"], ts=20.0)
    assert len(out) == 1
    entry = out[0]
    assert entry["recovered"] is True
    assert entry["original_text"] == "meet at 7"
    assert entry["counterpart_id"] == "111@s.whatsapp.net"
    assert entry["account"] == rec.id
    # journaled for the owner; the cache no longer holds it
    assert vault.recoveries(rec.id)[0]["original_text"] == "meet at 7"
    out2 = personal.observe_delete(
        "whatsapp", chat_key="whatsapp:dm:111@s.whatsapp.net",
        deleted_ids=["m1"], ts=21.0)
    assert out2[0]["recovered"] is False  # honest: already gone
    # a delete we never cached is journaled as unrecovered, not faked
    out3 = personal.observe_delete(
        "whatsapp", chat_key="whatsapp:dm:111@s.whatsapp.net",
        deleted_ids=["never-seen"], ts=22.0)
    assert out3[0]["recovered"] is False
    assert out3[0]["original_text"] == ""


def test_antidelete_toggle_off_means_no_cache_no_recovery(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    # toggle stays OFF
    personal.observe_message(
        "whatsapp", chat_key="k", sender_id="111", message_id="m1",
        text="secret", ts=1.0)
    assert vault.get_message(rec.id, "k", "m1") is None
    assert personal.observe_delete(
        "whatsapp", chat_key="k", deleted_ids=["m1"]) == []
    assert vault.recoveries() == []


def test_antiedit_journals_pair_and_cache_moves(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, antidelete=True)
    personal.observe_message(
        "whatsapp", chat_key="k", sender_id="111", message_id="m1",
        text="hello", ts=1.0)
    entry = personal.observe_edit(
        "whatsapp", chat_key="k", message_id="m1",
        new_text="hello!", ts=2.0)
    assert entry["original_text"] == "hello"
    assert entry["new_text"] == "hello!"
    # a second edit compares against the NEWEST text
    entry2 = personal.observe_edit(
        "whatsapp", chat_key="k", message_id="m1",
        new_text="hey", ts=3.0)
    assert entry2["original_text"] == "hello!"
    # no-op edit journals nothing
    assert personal.observe_edit(
        "whatsapp", chat_key="k", message_id="m1",
        new_text="hey", ts=4.0) is None


def _media_file(tmp_path, name="pic.jpg", data=b"\xff\xd8jpeg"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_view_once_preserved_before_it_vanishes(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, view_once=True)
    src = _media_file(tmp_path)
    out = personal.observe_message(
        "whatsapp", chat_key="k", counterpart_id="111",
        sender_name="Ada", message_id="m9", text="",
        media=[{"path": src, "kind": "image"}], view_once=True,
        ts=5.0)
    assert len(out) == 1
    entry = out[0]
    assert entry["kind"] == "view_once"
    assert entry["counterpart_id"] == "111"
    saved = Path(entry["path"])
    assert saved.exists() and saved.read_bytes() == b"\xff\xd8jpeg"
    assert saved != Path(src)  # a copy in the owner's store
    # telegram shape: ttl_seconds marks it too
    src2 = _media_file(tmp_path, "pic2.jpg")
    out2 = personal.observe_message(
        "whatsapp", chat_key="k", message_id="m10",
        media=[{"path": src2, "kind": "image"}], ttl_seconds=30,
        ts=6.0)
    assert len(out2) == 1
    # ordinary media is NOT preserved
    src3 = _media_file(tmp_path, "pic3.jpg")
    assert personal.observe_message(
        "whatsapp", chat_key="k", message_id="m11",
        media=[{"path": src3, "kind": "image"}], ts=7.0) == []


def test_view_once_missed_is_journaled_not_faked(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, view_once=True)
    out = personal.observe_message(
        "whatsapp", chat_key="k", message_id="m1",
        media=[{"path": str(tmp_path / "gone.jpg"), "kind": "image"}],
        view_once=True, ts=1.0)
    assert out == []  # nothing saved…
    rows = vault.journal(rec.id)
    # …said so — and (2026-11-08 truth fix) the arrival itself
    # traces too: every view-once arrival leaves a
    # view_once_trace row after its outcome rows.
    assert [r["kind"] for r in rows] == [
        "view_once_missed", "view_once_trace"]
    assert rows[1]["outcome"] == "missed"


def test_status_saving_and_dedupe(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    # toggle off: nothing
    src = _media_file(tmp_path)
    assert personal.observe_status_post(
        "whatsapp", poster_id="222", poster_name="Bo",
        media=[{"path": src, "kind": "image"}], source_id="s1",
        ts=1.0) == []
    registry.set_toggles(rec.id, status_saving=True)
    out = personal.observe_status_post(
        "whatsapp", poster_id="222", poster_name="Bo",
        media=[{"path": src, "kind": "image"}], source_id="s1",
        ts=1.0)
    assert len(out) == 1 and out[0]["kind"] == "status"
    assert out[0]["counterpart_id"] == "222"
    # the same post never saves twice
    assert personal.observe_status_post(
        "whatsapp", poster_id="222",
        media=[{"path": src, "kind": "image"}], source_id="s1",
        ts=2.0) == []
    # text status journals its text
    out = personal.observe_status_post(
        "whatsapp", poster_id="222", text="big news", source_id="s2",
        ts=3.0)
    assert out[0]["text"] == "big news"


def test_story_cycle_with_fetcher(tmp_path):
    personal, registry, vault, rec = make_personal(tmp_path)
    registry.set_toggles(rec.id, status_saving=True)
    src = _media_file(tmp_path)
    stories = [{"poster_id": "555", "poster_name": "Cy",
                "story_id": "st1", "path": src, "kind": "image",
                "ts": 9.0}]
    personal.story_fetcher = lambda account: stories
    out = personal.cycle_stories("whatsapp", rec.id)
    assert len(out) == 1 and out[0]["kind"] == "story"
    assert personal.cycle_stories("whatsapp", rec.id) == []  # dedupe
    # no fetcher → no feed → nothing (honest absence)
    personal.story_fetcher = None
    assert personal.cycle_stories("whatsapp", rec.id) == []


def test_saved_cap_prunes_oldest(tmp_path):
    personal, registry, vault, rec = make_personal(
        tmp_path, saved_per_account=3)
    for i in range(5):
        src = _media_file(tmp_path, f"p{i}.jpg", data=b"x" * (i + 1))
        vault.save_media(rec.id, "status", src, platform="whatsapp",
                         source_id=f"s{i}", ts=100.0 + i)
    rows = [r for r in vault.saved_entries(rec.id) if r["path"]]
    existing = [r for r in rows if Path(r["path"]).exists()]
    assert len(existing) == 3
    assert {r["source_id"] for r in existing} == {"s2", "s3", "s4"}
    assert any(r["kind"] == "pruned" for r in vault.journal(rec.id))


def test_dm_cache_ring_cap(tmp_path):
    vault = PersonalVault(tmp_path / "personal", dm_per_chat=3)
    for i in range(5):
        vault.note_message("a", "k", {"message_id": f"m{i}",
                                      "text": str(i), "ts": float(i)})
    assert vault.get_message("a", "k", "m0") is None
    assert vault.get_message("a", "k", "m4")["text"] == "4"


def test_dm_cache_corrupt_quarantines_and_raises(tmp_path):
    root = tmp_path / "personal"
    root.mkdir()
    (root / "dm_cache.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(PersonalStoreError):
        PersonalVault(root)
    assert list(root.glob("dm_cache.corrupt-*.json"))


def test_observation_never_creates_registry_records(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault)
    personal.observe_message("whatsapp", chat_key="k",
                             message_id="m1", text="hi")
    personal.observe_status_post("whatsapp", poster_id="1",
                                 text="status", source_id="s1")
    assert registry.all() == []  # observation is not an owner act
    rec = personal.ensure_primary("whatsapp")
    assert rec.id == "whatsapp:primary"
    assert [r.id for r in registry.all()] == ["whatsapp:primary"]


# ── whatsapp adapter transports ─────────────────────────────────────

def _wa_adapter():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter
    return WhatsAppAdapter(outbox_enabled=False)


def test_adapter_carries_status_post_and_view_once():
    adapter = _wa_adapter()
    got = []
    adapter._dispatch(
        {"type": "status_post", "id": "s1",
         "from": {"id": "222@s.whatsapp.net", "name": "Bo"},
         "text": "look", "ts": 1700000000000,
         "media": [{"path": "/tmp/x.jpg", "mime": "image/jpeg",
                    "kind": "image"}]},
        handler=got.append)
    assert len(got) == 1
    msg = got[0]
    assert msg.meta["service"] == "status_post"
    assert msg.sender_id == "222@s.whatsapp.net"
    assert msg.media[0].path == "/tmp/x.jpg"
    got.clear()
    adapter._dispatch(
        {"type": "message", "id": "m1",
         "chat": {"id": "111@s.whatsapp.net", "kind": "dm",
                  "title": "Ada"},
         "from": {"id": "111@s.whatsapp.net", "name": "Ada"},
         "text": "", "view_once": True, "ts": 1700000000000,
         "media": [{"path": "/tmp/y.jpg", "mime": "image/jpeg",
                    "kind": "image"}]},
        handler=got.append)
    assert got[0].meta["view_once"] is True


def test_adapter_revoke_still_flows():
    adapter = _wa_adapter()
    got = []
    adapter._dispatch(
        {"type": "revoke", "id": "m7",
         "chat": {"id": "111@s.whatsapp.net", "kind": "dm"},
         "from": {"id": "111@s.whatsapp.net"}, "ts": 1700000000000},
        handler=got.append)
    assert got[0].meta["service"] == "delete"
    assert got[0].meta["deleted_ids"] == ["m7"]


# ── the runtime hook (mixin, bare instance) ─────────────────────────

def _bare_runtime(tmp_path):
    from nomorals.agents.partner.runtime_provision import \
        RuntimeProvisionMixin

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash="")))
    return rt


def _msg(platform, kind, *, text="", sender_id="", sender="",
         message_id="", meta=None, media=None, peer="",
         incoming=True):
    from nomorals.social.chat.base import ChatMessage, ChatRef

    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=peer or "chat1",
                     kind=kind, title="", peer=peer),
        incoming=incoming, text=text, sender=sender,
        sender_id=sender_id, message_id=message_id,
        media=media or [], meta=meta or {}, ts=100.0)


def test_runtime_hook_consumes_and_observes(tmp_path):
    rt = _bare_runtime(tmp_path)
    svc = rt._pairing_personal()
    personal = svc["personal"]
    rec = personal.ensure_primary("whatsapp")
    personal.set_toggle(rec.id, "antidelete", True)
    personal.set_toggle(rec.id, "status_saving", True)

    dm = _msg("whatsapp", "dm", text="meet at 7", sender_id="111",
              message_id="m1", peer="111@s.whatsapp.net")
    assert rt._personal_surface_turn(dm) is False  # flows on
    delete = _msg("whatsapp", "dm", sender_id="111", peer="111@"
                  "s.whatsapp.net",
                  meta={"service": "delete", "deleted_ids": ["m1"]})
    assert rt._personal_surface_turn(delete) is True  # consumed
    rows = personal.vault.recoveries(rec.id)
    assert rows and rows[0]["original_text"] == "meet at 7"

    # status post consumed + saved
    src = _media_file(tmp_path, "st.jpg")
    from nomorals.social.chat.base import MediaRef
    post = _msg("whatsapp", "dm", sender_id="222", message_id="s1",
                meta={"service": "status_post"},
                media=[MediaRef(path=src, kind="image")],
                peer="222")
    assert rt._personal_surface_turn(post) is True
    assert personal.vault.saved_entries(rec.id)

    # groups and the bot surface are never the personal surface
    grp = _msg("whatsapp", "group", text="hi", message_id="g1",
               peer="999@g.us")
    assert rt._personal_surface_turn(grp) is False
    bot = _msg("telegram-bot", "dm", text="hi", message_id="b1")
    assert rt._personal_surface_turn(bot) is False


# ── pairing page ────────────────────────────────────────────────────

def test_pairing_page_renders_state_escaped(tmp_path):
    from nomorals.social.pairing_page import pairing_page_html

    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", label="<b>main</b>",
        status="pairing", created_at=1.0))
    page = pairing_page_html(
        registry, codes={"whatsapp:2348012345678": ("WXYZ-1234", 12)})
    assert "Registered: <b>1</b>" in page
    assert "WXYZ-1234" in page
    assert "&lt;b&gt;main&lt;/b&gt;" in page  # escaped, never raw
    assert "<b>main</b>" not in page


def test_pairing_page_served_owner_only(tmp_path):
    from nomorals.api.server import APIServer, Principal, _make_handler
    from nomorals.core.policy import Capability, CapabilitySet
    from nomorals.storage.db import Database

    context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0),
            backup_dir=str(tmp_path), workspace_dir=str(tmp_path),
            chat=SimpleNamespace(telegram_api_id="",
                                 telegram_api_hash="")),
        tools=None, bus=None, memory=None)
    api = APIServer(
        context, token="owner-token",
        principals={"ro-token": Principal(
            "ro", CapabilitySet.of(Capability.DB_READ))},
        max_body_bytes=64 * 1024)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path, token=""):
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=10)
            headers = {"Authorization": f"Bearer {token}"} \
                if token else {}
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", "replace")
            conn.close()
            return resp.status, body

        status, _ = get("/pair")
        assert status == 401  # no token, no page
        status, _ = get("/pair", "ro-token")
        assert status == 403  # a reader is not the owner
        status, body = get("/pair", "owner-token")
        assert status == 200
        assert "Paired accounts" in body
        assert "No accounts paired yet" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
