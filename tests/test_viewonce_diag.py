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

import time
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

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
        message_id=f"cmd-{time.time()}", ts=time.time())


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
    ts = time.time() - 3600
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
        ts=time.time() - 600)
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
    ts = time.time() - 3600
    _missed_arrival(personal, ts=ts)
    activity = personal.toggle_activity(REC_ID)
    assert "view_once" not in activity
    assert activity["view_once_arrival"] == ts


def test_pair_verdict_names_arrivals_seen_none_saved(tmp_path):
    from nomorals.social.pairing import toggle_live_verdicts

    rt, svc = _chatops_runtime(tmp_path, view_once=True)
    personal = svc["personal"]
    ts = time.time() - 3600
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
