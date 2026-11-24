"""View-once second live failure (2026-11-12): the diagnostic lane.

The owner sent a view-once photo in a WhatsApp DM (12:59 AM) and
/ vv at 1:01 AM answered "no view-once captures yet" while the
toggle sat ON — the SECOND live re-demonstration (first 2026-11-08).
The chain on main is complete on paper (Go unwrapInbound, Python
_dispatch stamping, _personal_surface_turn, observe_message's
always-trace), so a zero-row journal means either the RUNNING
bridge binary predates the Nov 8 fix — unprovable from the Python
side, because the bridge's version string is a build-invariant
constant and the supervisor's adoption check keys on that same
constant — or the arrival never reached the observer at all.

This file pins the diagnostic instruments, through the real seams:

1. The bridge reports build identity (commit + built time + the
   unwrap_inbound capability) in its status payload, and the
   Python adapter records it (health() surfaces it).
2. The supervisor's bridge identity carries the build through
   verification, so an adopted orphan's age is visible.
3. An empty /vv reply distinguishes the three states FROM THE
   RECORD: no arrival ever observed (zero trace rows) vs arrivals
   observed but none saved (last trace named) vs captures saved.
4. The /pair verdict for view_once names the middle state too
   (arrivals seen, none saved) instead of "NO capture ever
   recorded", via toggle_activity's view_once_arrival key.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from tests.frozen_clock import MIDDAY_TS

# Clock policy (batch H32): every seed below is MIDDAY_TS (noon
# 2026-10-14, tests.frozen_clock). The trace ts is carried, never
# compared: observe_message stamps it into the view_once_trace row
# (personal.py:798), toggle_activity takes the journal max per kind
# with no clock comparison, _chatops_viewonce renders the last trace
# with no age filter, and toggle_live_verdicts formats the arrival
# ts with no recency gate — so a frozen seed changes no verdict.
# Message ids stay unique via a counter, never frozen: _process
# scopes effects by chat + message id (H11 pattern).
_MSG_SEQ = itertools.count(1)

OWNER_JID = "2348012345678@s.whatsapp.net"
REC_ID = "whatsapp:2348012345678"
CHAT_A = "2347111111111@s.whatsapp.net"
CHAT_C = "2347333333333@s.whatsapp.net"

BUILD = {
    "program": "gobridge/1.0.0",
    "commit": "abc1234",
    "built": "2026-11-12T00:00:00Z",
    "unwrap_inbound": True,
    "whatsmeow": "v0.0.0-20261102152715-54038813c47f",
}


def _status_event(**extra):
    evt = {"type": "status", "state": "open", "user": OWNER_JID,
           "program": "gobridge/1.0.0", "build": dict(BUILD)}
    evt.update(extra)
    return evt


# ── 1. the adapter records bridge build identity ─────────────


def test_adapter_status_records_bridge_build():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    adapter._dispatch(_status_event(), lambda msg: None)
    assert adapter.bridge_program == "gobridge/1.0.0"
    assert adapter.bridge_build["unwrap_inbound"] is True
    assert adapter.bridge_build["commit"] == "abc1234"
    assert adapter.bridge_build["built"] == "2026-11-12T00:00:00Z"
    health = adapter.health()
    assert health["bridge_program"] == "gobridge/1.0.0"
    assert health["bridge_build"]["commit"] == "abc1234"


def test_adapter_status_without_build_reads_unknown():
    """An older bridge sends no build block: the adapter records
    that absence instead of inventing an identity."""
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    evt = {"type": "status", "state": "open", "user": OWNER_JID,
           "program": "gobridge/1.0.0"}
    adapter._dispatch(evt, lambda msg: None)
    assert adapter.bridge_program == "gobridge/1.0.0"
    assert adapter.bridge_build == {}
    assert adapter.health()["bridge_build"] == {}


def test_go_status_payload_carries_build_block():
    """Source contract: the Go bridge's status payload includes
    the build block (the runtime-readable half of the fix)."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent
            / "bridge" / "gobridge" / "main.go").read_text(
                encoding="utf-8")
    assert '"build":' in text and "bridgeBuildInfo()" in text
    assert "unwrap_inbound" in text


def test_node_status_payload_carries_build_block():
    """Same contract on the Node engine: its status carries a
    build block with the same keys, so the Python surface is
    uniform across engines."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent
            / "bridge" / "whatsapp-bridge.mjs").read_text(
                encoding="utf-8")
    assert "unwrap_inbound" in text
    assert '"build"' in text


# ── 2. the supervisor identity carries the build ─────────────


def test_bridge_identity_carries_build(tmp_path):
    from nomorals.social.pairing import (
        AccountRecord,
        BridgeSupervisor,
        PairingRegistry,
    )

    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp", identity="2348012345678",
        status="linked", created_at=1.0, port=8787))
    sup = BridgeSupervisor.__new__(BridgeSupervisor)
    sup._inspector = SimpleNamespace(
        listener_pid=lambda port: 1234,
        process_image=lambda pid: (
            "whatsapp-bridge.exe", "--port 8787"))
    sup._status_reply = lambda r: {
        "type": "status", "state": "open", "user": OWNER_JID,
        "program": "gobridge/1.0.0", "build": dict(BUILD)}
    ident = sup.verify_bridge_identity(registry.get(rec.id))
    assert ident.verified is True
    assert ident.engine == "go"
    assert ident.build["unwrap_inbound"] is True
    assert ident.build["commit"] == "abc1234"


def test_bridge_identity_proves_view_once_only_from_build():
    """The 2026-11-14 vvfix: the capability proof comes from the
    running bridge's build block, never from the program marker.
    A pre-2026-11-12 binary sends no build block at all, and a
    build without the flag proves nothing — both fail the proof
    even though the marker reads "gobridge/1.0.0" in each case."""
    from nomorals.social.pairing import BridgeIdentity

    assert BridgeIdentity(build=dict(BUILD)).proves_view_once is True
    assert BridgeIdentity(build={}).proves_view_once is False
    no_flag = dict(BUILD)
    del no_flag["unwrap_inbound"]
    assert BridgeIdentity(build=no_flag).proves_view_once is False
    false_flag = dict(BUILD)
    false_flag["unwrap_inbound"] = False
    assert BridgeIdentity(build=false_flag).proves_view_once is False


# ── 3. /vv distinguishes the three states ────────────────────


def _chatops_runtime(tmp_path, **toggles):
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin,
    )
    from nomorals.social.pairing import (
        AccountRecord,
        PairingRegistry,
    )

    class RT(RuntimeProvisionMixin, RuntimeChatOpsMixin):
        pass

    rt = RT.__new__(RT)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    if toggles:
        registry.set_toggles(rec.id, **toggles)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    svc = rt._pairing_personal()
    return rt, svc


def _dm_message(chat_jid, reply_to=""):
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_jid,
                     kind=ChatKind.DM, peer=chat_jid),
        incoming=True, text="/vv", sender="Owner",
        sender_id=OWNER_JID, reply_to=reply_to,
        message_id=f"cmd-{next(_MSG_SEQ)}", ts=MIDDAY_TS)


def _missed_arrival(personal, *, ts, message_id="m-missed"):
    """A view-once arrival observed but never saved: media
    missing at arrival, toggle on — the trace row is the whole
    record of it."""
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{CHAT_A}",
        counterpart_id=CHAT_A, sender_id=CHAT_A,
        sender_name="Ada", message_id=message_id, text="",
        media=[], view_once=True, media_missing=True, ts=ts)


def test_vv_empty_names_never_observed(tmp_path):
    """State (a): zero trace rows — the reply says no arrival
    was ever observed, not just that nothing was saved."""
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    reply = rt._chatops_viewonce("viewonce", "", _dm_message(CHAT_C))
    assert "no view-once captures yet" in reply
    assert "No view-once arrival has ever been observed" in reply
    assert "zero trace rows" in reply


def test_vv_empty_names_last_arrival_when_none_saved(tmp_path):
    """State (b): arrivals observed, none saved — the reply
    names the last trace row and its outcome."""
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    ts = MIDDAY_TS - 3600
    _missed_arrival(personal, ts=ts)
    reply = rt._chatops_viewonce("viewonce", "", _dm_message(CHAT_C))
    assert "no view-once captures yet" in reply
    assert "observed" in reply
    assert "missed" in reply
    from datetime import datetime

    when = datetime.fromtimestamp(ts).strftime("%d %b %H:%M")
    assert when in reply


def test_vv_empty_names_standin_account_trace(tmp_path):
    """State (b), cross-runtime shape: the arrivals were traced
    under the unpersisted whatsapp:primary stand-in — the
    observing runtime had no registry account — so the reply
    names that account instead of claiming nothing arrived."""
    from nomorals.social.pairing import PairingRegistry
    from nomorals.social.personal import PersonalSurfaces

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    bare_registry = PairingRegistry(tmp_path / "bare-paired.json")
    bare = PersonalSurfaces(bare_registry, personal.vault)
    bare.observe_message(
        "whatsapp", chat_key=f"whatsapp:{CHAT_A}",
        counterpart_id=CHAT_A, sender_id=CHAT_A,
        sender_name="Ada", message_id="m-standin", text="",
        media=[], view_once=True, media_missing=True,
        ts=MIDDAY_TS - 600)
    assert [r for r in personal.vault.journal("whatsapp:primary")
            if r.get("kind") == "view_once_trace"]
    reply = rt._chatops_viewonce("viewonce", "", _dm_message(CHAT_C))
    assert "whatsapp:primary" in reply


def test_vv_empty_no_longer_tells_owner_to_turn_on_when_on(tmp_path):
    """The old empty reply told the owner to turn the toggle on
    even while it was ON — the screenshot's exact confusion."""
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    reply = rt._chatops_viewonce("viewonce", "", _dm_message(CHAT_C))
    assert "Turn it on with /pair toggle" not in reply


# ── 4. /pair verdict names the middle state ──────────────────


def test_toggle_activity_reports_last_arrival(tmp_path):
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    ts = MIDDAY_TS - 3600
    _missed_arrival(personal, ts=ts)
    activity = personal.toggle_activity(REC_ID)
    assert "view_once" not in activity
    assert activity["view_once_arrival"] == ts


def test_pair_verdict_names_arrivals_seen_none_saved(tmp_path):
    from nomorals.social.pairing import toggle_live_verdicts

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    ts = MIDDAY_TS - 3600
    _missed_arrival(personal, ts=ts)
    activity = personal.toggle_activity(REC_ID)
    rec = svc["registry"].get(REC_ID)
    verdicts = toggle_live_verdicts(
        rec, stopped=False, running=True, attached=True,
        connected=True, activity=activity)
    verdict = verdicts["view_once"]
    assert "arrivals seen" in verdict
    assert "none saved" in verdict


# ── 5. /platforms names the running bridge build ─────────────


def test_platforms_line_carries_bridge_build(tmp_path):
    """The /platforms listing names the running bridge's build —
    the owner-facing half of the diagnostic instrument: the next
    live demo can prove which binary is running from /platforms
    alone."""
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="viewonce-diag-platforms-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    rt = PartnerRuntime(context)
    fake = SimpleNamespace(health=lambda: {
        "connected": True, "user": OWNER_JID,
        "bridge": "127.0.0.1:8787", "bridge_state": "open",
        "bridge_program": "gobridge/1.0.0",
        "bridge_build": dict(BUILD)})
    rt.gateway._known["whatsapp"] = fake
    rt.gateway.adapters["whatsapp"] = fake
    try:
        reply = rt._handle_control_inner("/platforms", "local:owner")
    finally:
        rt.gateway.adapters.pop("whatsapp", None)
        rt.gateway._known.pop("whatsapp", None)
    assert "whatsapp: running" in reply
    assert "commit abc1234" in reply
    assert "built 2026-11-12T00:00:00Z" in reply


def test_platforms_line_without_build_says_unknown(tmp_path):
    """An older bridge sends no build block: the line says the
    build is unknown instead of borrowing the program marker."""
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="viewonce-diag-platforms-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    rt = PartnerRuntime(context)
    fake = SimpleNamespace(health=lambda: {
        "connected": True, "user": OWNER_JID,
        "bridge": "127.0.0.1:8787", "bridge_state": "open",
        "bridge_program": "gobridge/1.0.0",
        "bridge_build": {}})
    rt.gateway._known["whatsapp"] = fake
    rt.gateway.adapters["whatsapp"] = fake
    try:
        reply = rt._handle_control_inner("/platforms", "local:owner")
    finally:
        rt.gateway.adapters.pop("whatsapp", None)
        rt.gateway._known.pop("whatsapp", None)
    assert "whatsapp: running" in reply
    assert "(build unknown)" in reply
    assert "commit" not in reply


# ── 6. the vault fix (2026-11-23): group capture, status kind,
#    7-day retention, redownload, today + reply fallback ──────
#
# The owner's Septorch screenshots proved the old approach dead:
# the reply command ("No view-once media found") fails on a
# textbook-correct invocation, and the new approach is capture
# at arrival into a vault — no command needed.  Devon's capture
# primitive already was arrival-based; what was broken around
# it: group view-once was dropped before the vault, nothing was
# kept past the count/bytes caps, a failed arrival download had
# no second chance, and the only viewing surface was one-file
# /viewonce.  These tests pin the closed-up chain through the
# real seams.  NO dashboard page: the viewing surface is the
# chat commands only (owner correction 2026-11-23).

GROUP_JID = "120363012345678@g.us"
GROUP_SENDER = "2347222222222@s.whatsapp.net"


def _vo_media_file(tmp_path, name="vo.jpg"):
    path = tmp_path / name
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(path)


def _now_ms():
    import time

    return int(time.time() * 1000)


def _provision_runtime(tmp_path, adapter, **toggles):
    """The truthfix rig: bare provision mixin + fake bridge
    supervisor + the scripted adapter, organ clock frozen on
    MIDDAY_TS, housekeeping beat pinned quiet."""
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin,
    )
    from nomorals.social.pairing import (
        AccountRecord,
        PairingRegistry,
    )

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    if toggles:
        registry.set_toggles(rec.id, **toggles)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    rt._adapter_for = lambda platform: adapter
    svc = rt._pairing_personal()
    svc["personal"]._clock = lambda: MIDDAY_TS
    svc["last_cycle"] = 1999999999.0  # beat stays quiet
    return rt, svc, registry.get(rec.id)


class _NullSurfaceAdapter:
    """The status verbs, scripted to succeed, plus a scripted
    redownload: the intake's one second chance at a failed
    arrival download."""

    def __init__(self, redownload=None):
        self.redownload_reply = redownload
        self.redownload_calls: list = []

    def view_status(self, poster_id, message_id, ts=0.0):
        return {"ok": True, "receipt_mode": "read"}

    def react_status(self, poster_id, message_id, emoji):
        return {"ok": True}

    def redownload_message(self, message_id):
        self.redownload_calls.append(message_id)
        if isinstance(self.redownload_reply, dict):
            return dict(self.redownload_reply)
        return {"ok": False, "error": "message not in cache: "
                + str(message_id)}


def _real_wa_adapter():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    adapter._dispatch(
        {"type": "status", "state": "open", "user": OWNER_JID},
        lambda msg: None)
    return adapter


def _feed_wire(rt, wa, wire):
    """One bridge event through the REAL adapter dispatch into
    the REAL runtime personal-surface turn."""
    delivered: list = []
    wa._dispatch(wire, delivered.append)
    assert len(delivered) == 1
    rt._personal_surface_turn(delivered[0])
    return delivered[0]


def _group_vo_wire(mid, media, chat_id=GROUP_JID):
    return {"type": "message", "id": mid,
            "chat": {"id": chat_id, "kind": "group",
                     "title": "Crew"},
            "from": {"id": GROUP_SENDER, "name": "Ada"},
            "text": "", "view_once": True, "media": media,
            "ts": _now_ms()}


def _own_group_vo_wire(mid, media, chat_id=GROUP_JID):
    return {"type": "own_message", "id": mid,
            "chat": {"id": chat_id, "kind": "group",
                     "title": "Crew"},
            "text": "", "view_once": True, "media": media,
            "ts": _now_ms()}


def _status_vo_wire(sid, media):
    return {"type": "status_post", "id": sid,
            "from": {"id": CHAT_A, "name": "Ada"},
            "text": "", "view_once": True, "media": media,
            "ts": _now_ms()}


def _saved_viewonce(personal, rec_id=REC_ID):
    return [r for r in personal.vault.saved_entries(rec_id)
            if r.get("kind") == "view_once"]


def _rows(personal, kind, rec_id=REC_ID):
    return [r for r in personal.vault.journal(rec_id)
            if r["kind"] == kind]


def test_group_viewonce_arrival_captures_and_traces(tmp_path):
    """THE group hole: a view-once in a group used to be
    dropped in _personal_surface_turn before the personal
    organ ever saw it — the DM-only gate ran before
    observe_message.  Now it captures exactly like a DM's."""
    surface = _NullSurfaceAdapter()
    rt, svc, rec = _provision_runtime(
        tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa = _real_wa_adapter()
    src = _vo_media_file(tmp_path)
    _feed_wire(rt, wa, _group_vo_wire("gvo1", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}]))
    assert len(_saved_viewonce(personal)) == 1
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["outcome"] == "saved"
    assert traces[0]["media_present"] is True


def test_group_viewonce_own_send_captures(tmp_path):
    """A view-once the owner sends from another device into a
    group: the bridge downloads it on the own-message path and
    the group lift carries it into the vault — before this fix
    the own path stamped the flag and never fetched, and the
    group gate dropped it anyway."""
    surface = _NullSurfaceAdapter()
    rt, svc, rec = _provision_runtime(
        tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa = _real_wa_adapter()
    src = _vo_media_file(tmp_path, "own-vo.jpg")
    _feed_wire(rt, wa, _own_group_vo_wire("gvo2", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}]))
    assert len(_saved_viewonce(personal)) == 1
    assert _rows(personal, "view_once_trace")[0]["outcome"] \
        == "saved"


def test_status_viewonce_saves_with_view_once_kind(tmp_path):
    """A view-once STATUS saves with kind "view_once" — so
    /viewonce finds it — and the arrival ledger carries the
    flag.  Before this fix it saved as "status" and /viewonce
    never listed it."""
    surface = _NullSurfaceAdapter()
    rt, svc, rec = _provision_runtime(
        tmp_path, surface, status_saving=True, view_once=True)
    personal = svc["personal"]
    wa = _real_wa_adapter()
    src = _vo_media_file(tmp_path, "st-vo.jpg")
    _feed_wire(rt, wa, _status_vo_wire("svo1", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}]))
    saved = _saved_viewonce(personal)
    assert len(saved) == 1
    arrivals = [r for r in personal.vault.journal(rec.id)
                if r.get("kind") == "status_arrival"
                and r.get("source_id") == "svo1"]
    assert len(arrivals) == 1
    assert arrivals[0].get("view_once") is True


def test_seven_day_prune_removes_old_keeps_fresh(
        tmp_path, monkeypatch):
    """The retention rule: a capture older than 7 days is
    pruned even when the count/bytes caps are nowhere near
    full, and the pruned row names the policy.  A capture
    inside the window survives the same pass."""
    import os
    import time

    from nomorals.social.personal import PersonalVault

    vault = PersonalVault(tmp_path / "vault")
    now = time.time()
    old_src = _vo_media_file(tmp_path, "old.jpg")
    fresh_src = _vo_media_file(tmp_path, "fresh.jpg")
    # save_media prunes on write with the real clock — pin it
    # quiet while seeding, so the explicit call below is the
    # pass under test.
    monkeypatch.setattr(vault, "_prune_account",
                        lambda *a, **k: None)
    vault.save_media(REC_ID, "view_once", old_src,
                     platform="whatsapp", counterpart_id=CHAT_A,
                     poster_name="Old", media_kind="image",
                     source_id="old1", ts=now - 8 * 86400.0)
    vault.save_media(REC_ID, "view_once", fresh_src,
                     platform="whatsapp", counterpart_id=CHAT_A,
                     poster_name="Fresh", media_kind="image",
                     source_id="fresh1", ts=now - 3600.0)
    monkeypatch.undo()
    before = {r.get("source_id"): r.get("path")
              for r in vault.saved_entries(REC_ID)
              if r.get("kind") == "view_once"}
    assert set(before) == {"old1", "fresh1"}
    # Age is the vault file's mtime, not the row's ts: backdate
    # the old file's mtime 8 days, leave the fresh one at now.
    os.utime(before["old1"], (now - 8 * 86400.0, now - 8 * 86400.0))
    PersonalVault._prune_account(vault, REC_ID, now=now)
    rows = vault.journal(REC_ID)
    pruned = [r for r in rows if r.get("kind") == "pruned"]
    assert len(pruned) == 1
    assert "7-day" in str(pruned[0].get("note") or "")
    assert pruned[0].get("source_id") == "old1"
    # The journal is append-only, so the pruned row still lists
    # — the proof is the files: the old one is gone, the fresh
    # one is still on disk.
    from pathlib import Path

    assert not Path(str(before["old1"])).exists()
    assert Path(str(before["fresh1"])).exists()


def test_prune_count_cap_still_bites_as_backstop(
        tmp_path, monkeypatch):
    """The 7-day rule does not retire the caps: with the
    window satisfied, the count cap still prunes oldest-first
    and the row says "cap"."""
    import time

    from nomorals.social.personal import PersonalVault

    vault = PersonalVault(tmp_path / "vault",
                          saved_per_account=1)
    now = time.time()
    first = _vo_media_file(tmp_path, "first.jpg")
    second = _vo_media_file(tmp_path, "second.jpg")
    monkeypatch.setattr(vault, "_prune_account",
                        lambda *a, **k: None)
    vault.save_media(REC_ID, "view_once", first,
                     platform="whatsapp", counterpart_id=CHAT_A,
                     poster_name="First", media_kind="image",
                     source_id="cap1", ts=now - 3600.0)
    vault.save_media(REC_ID, "view_once", second,
                     platform="whatsapp", counterpart_id=CHAT_A,
                     poster_name="Second", media_kind="image",
                     source_id="cap2", ts=now - 1800.0)
    monkeypatch.undo()
    before = {r.get("source_id"): r.get("path")
              for r in vault.saved_entries(REC_ID)
              if r.get("kind") == "view_once"}
    assert set(before) == {"cap1", "cap2"}
    PersonalVault._prune_account(vault, REC_ID, now=now)
    pruned = [r for r in vault.journal(REC_ID)
              if r.get("kind") == "pruned"]
    assert len(pruned) == 1
    assert pruned[0].get("source_id") == "cap1"
    assert "cap prune" in str(pruned[0].get("note") or "")
    assert "7-day" not in str(pruned[0].get("note") or "")
    # The journal is append-only, so the pruned row still lists
    # — the proof is the files: the oldest is gone, the newest
    # is still on disk.
    from pathlib import Path

    assert not Path(str(before["cap1"])).exists()
    assert Path(str(before["cap2"])).exists()


def test_redownload_success_recovers_missed_arrival(tmp_path):
    """One second chance: a view-once arrival whose download
    failed at arrival is re-fetched from the bridge cache and
    saved — the trace reads "saved", not "missed"."""
    src = _vo_media_file(tmp_path, "redl.jpg")
    surface = _NullSurfaceAdapter(redownload={
        "ok": True, "media": {"path": src, "kind": "image",
                              "mime": "image/jpeg"}})
    rt, svc, rec = _provision_runtime(
        tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa = _real_wa_adapter()
    _feed_wire(rt, wa, _group_vo_wire("gvo3", [
        {"kind": "image",
         "download_error": "410 Gone: media expired"}]))
    assert surface.redownload_calls == ["gvo3"]
    assert len(_saved_viewonce(personal)) == 1
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["outcome"] == "saved"
    assert _rows(personal, "view_once_missed") == []


def test_redownload_failure_journals_missed_with_reason(tmp_path):
    """A failed second chance still journals the miss — and the
    miss names the bridge's reason instead of a bare "download
    failed"."""
    surface = _NullSurfaceAdapter(redownload={
        "ok": False, "error": "message not in cache: gvo4"})
    rt, svc, rec = _provision_runtime(
        tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa = _real_wa_adapter()
    _feed_wire(rt, wa, _group_vo_wire("gvo4", [
        {"kind": "image",
         "download_error": "410 Gone: media expired"}]))
    assert surface.redownload_calls == ["gvo4"]
    assert _saved_viewonce(personal) == []
    missed = _rows(personal, "view_once_missed")
    assert len(missed) == 1
    assert "410 Gone: media expired" in missed[0]["note"]
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["outcome"] == "missed"
    assert traces[0].get("download_error") \
        == "410 Gone: media expired"


def test_missed_names_download_reason_at_the_organ(tmp_path):
    """The organ half, direct: observe_message with a
    download_error journals the miss with the reason in the
    note and on the trace row."""
    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{CHAT_A}",
        counterpart_id=CHAT_A, sender_id=CHAT_A,
        sender_name="Ada", message_id="m-reason", text="",
        media=[], view_once=True, media_missing=True,
        download_error="410 Gone: media expired",
        ts=MIDDAY_TS)
    missed = _rows(personal, "view_once_missed")
    assert len(missed) == 1
    assert "410 Gone: media expired" in missed[0]["note"]
    traces = _rows(personal, "view_once_trace")
    assert traces[0].get("download_error") \
        == "410 Gone: media expired"


def test_adapter_redownload_message_wrapper(monkeypatch):
    """The adapter's redownload_message never raises: a bridge
    reply passes through, a transport failure comes back as
    ok:False, and an empty id never touches the socket."""
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter()
    calls: list = []

    def _ok(cmd, *, timeout=25.0):
        calls.append(cmd)
        return {"ok": True, "media": {"path": "/x.jpg"}}

    monkeypatch.setattr(adapter, "_send_cmd", _ok)
    assert adapter.redownload_message("VO1") == {
        "ok": True, "media": {"path": "/x.jpg"}}
    assert calls == [{"cmd": "redownload", "message_id": "VO1"}]

    def _boom(cmd, *, timeout=25.0):
        raise RuntimeError("socket gone")

    monkeypatch.setattr(adapter, "_send_cmd", _boom)
    reply = adapter.redownload_message("VO1")
    assert reply["ok"] is False
    assert "socket gone" in reply["error"]
    assert adapter.redownload_message("")["ok"] is False
    assert adapter.redownload_message("   ")["ok"] is False


def _seed_capture_at(tmp_path, personal, *, message_id, chat_jid,
                     sender, ts):
    src = _vo_media_file(tmp_path, f"{message_id}.jpg")
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{chat_jid}",
        counterpart_id=chat_jid, sender_id=chat_jid,
        sender_name=sender, message_id=message_id, text="",
        media=[{"path": src, "kind": "image"}],
        view_once=True, ts=ts)
    saved = [r for r in personal.vault.saved_entries(REC_ID)
             if r.get("kind") == "view_once"
             and r.get("source_id") == message_id]
    assert len(saved) == 1
    return saved[0]


def test_viewonce_today_lists_last_24h_only(tmp_path):
    """`/viewonce today` lists the day's captures, newest
    first — the 25-hour-old capture is not in it."""
    import time

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    now = time.time()
    _seed_capture_at(tmp_path, personal, message_id="t-recent",
                     chat_jid=CHAT_A, sender="Recent Ada",
                     ts=now - 3600.0)
    _seed_capture_at(tmp_path, personal, message_id="t-old",
                     chat_jid=CHAT_A, sender="Old Bob",
                     ts=now - 90000.0)
    reply = rt._chatops_viewonce(
        "viewonce", "today", _dm_message(CHAT_C))
    assert "last 24 hours" in reply
    assert "Recent Ada" in reply
    assert "Old Bob" not in reply


def test_viewonce_today_empty_day_says_so_honestly(tmp_path):
    """An empty day does not read as "none ever": the reply
    names how many older captures the vault still holds."""
    import time

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    now = time.time()
    _seed_capture_at(tmp_path, personal, message_id="t-old2",
                     chat_jid=CHAT_A, sender="Old Bob",
                     ts=now - 90000.0)
    reply = rt._chatops_viewonce(
        "viewonce", "today", _dm_message(CHAT_C))
    assert "no view-once captures in the last 24 hours" in reply
    assert "still holds 1 older" in reply
    rows = _rows(personal, "viewonce_reveal")
    assert rows[-1]["outcome"] == "empty"
    assert rows[-1]["form"] == "today"


def test_viewonce_reply_fallback_uses_same_chat_recent(tmp_path):
    """The reply fallback: the reply names no capture exactly
    (a stale quoted id), so the newest capture in THIS chat
    from the hour before the reply is sent — and the reply
    SAYS it fell back.  The other chat's newer capture is
    never handed back."""
    import time

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    now = time.time()
    ada = _seed_capture_at(tmp_path, personal,
                           message_id="fb-ada", chat_jid=CHAT_A,
                           sender="Ada", ts=now - 1800.0)
    _seed_capture_at(tmp_path, personal,
                     message_id="fb-bob", chat_jid=CHAT_C,
                     sender="Bob", ts=now - 900.0)
    sent: list = []

    def send_file(platform, chat, path, caption="", kind=""):
        sent.append(path)
        return SimpleNamespace(ok=True)

    rt.gateway = SimpleNamespace(send_file=send_file)
    msg = _dm_message(CHAT_A, reply_to="NO-SUCH-MSG")
    msg.ts = now
    reply = rt._chatops_viewonce("viewonce", "", msg)
    assert "fell back" in reply
    assert len(sent) == 1
    assert sent[0] == ada["path"]
    rows = _rows(personal, "viewonce_reveal")
    assert rows[-1]["outcome"] == "found"
    assert rows[-1]["form"] == "reply_fallback"


def test_viewonce_reply_fallback_miss_when_no_recent(tmp_path):
    """No capture in this chat inside the hour: the fallback
    finds nothing and the honest miss answers — it never
    reaches into another chat for a substitute."""
    import time

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    now = time.time()
    _seed_capture_at(tmp_path, personal,
                     message_id="fb-old", chat_jid=CHAT_A,
                     sender="Ada", ts=now - 90000.0)
    msg = _dm_message(CHAT_C, reply_to="NO-SUCH-MSG")
    msg.ts = now
    reply = rt._chatops_viewonce("viewonce", "", msg)
    assert "fell back" not in reply
    assert "no view-once capture" in reply
    rows = _rows(personal, "viewonce_reveal")
    assert rows[-1]["outcome"] == "miss"
    assert rows[-1]["form"] == "reply"
