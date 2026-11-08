"""Personal-surface truth fix (2026-11-08) — the five repairs,
pinned through the REAL paths (the truth-pass blind spot was
hand-built fixtures bypassing the seams that were broken):

1. DM command promotion: the owner's own messages reach the
   WhatsApp adapter's own_message branch as bridge events; a
   REGISTERED command typed as-him in a 1:1 DM now promotes to
   an owner turn exactly like a group command (the /viewonce
   silence: every chatops command died in third-party DMs).
   Non-command DM text, unknown slashes and Devon's own send
   echoes stay persist-only.

2. Status coverage: the funnel reconciles in the journal —
   every observed post writes a status_arrival row before any
   toggle is read; cycle_statuses sweeps the ledger on the
   housekeeping beat for posts still owed a view/react
   attempt (toggle off at arrival, verb refused, late batch);
   landed views/reactions seed from the journal at boot, so a
   restart never re-fires a verb at a post already done.

3. View+react are ONE act: with both toggles on, one observe
   pass fires both verbs; each verdict stays independent.

4. View-once arrival truth: every ephemeral DM arrival
   journals a view_once_trace (toggle state, media present?,
   outcome) — toggle OFF included — and every /viewonce
   reveal attempt journals viewonce_reveal.

5. /pair verdicts carry evidence: a plain-live verdict for
   view_once / status_saving / status_viewing / status_react
   names the last activity, or says in words that there has
   never been any.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.social.pairing import (
    AccountRecord,
    PairingRegistry,
    toggle_live_verdicts,
)
from nomorals.social.personal import PersonalSurfaces, PersonalVault

OWNER_JID = "2348012345678@s.whatsapp.net"
OWNER_LID = "105382746195847@lid"
TARGET_JID = "2347099999999@s.whatsapp.net"
POSTER = "2347055554444@s.whatsapp.net"
REC_ID = "whatsapp:2348012345678"


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── rigs ────────────────────────────────────────────────────

def _wa_adapter():
    """The real WhatsApp adapter with the paired account's
    identity learned (the status event the bridge sends)."""
    adapter = WhatsAppAdapter()
    msgs: list = []
    adapter._dispatch(
        {"type": "status", "state": "open", "user": OWNER_JID,
         "lid": OWNER_LID}, msgs.append)
    return adapter, msgs


def _own(text, *, kind="dm", chat_id=TARGET_JID, mid="m1",
         ts=None, reply_to=""):
    return {"type": "own_message", "id": mid,
            "chat": {"id": chat_id, "kind": kind, "title": "Crew"},
            "text": text, "ts": _now_ms() if ts is None else ts,
            "reply_to": reply_to}


class _FakeSurfaceAdapter:
    """The WhatsApp adapter's status verbs, scripted: records
    every call; verdicts settable per test."""

    def __init__(self, view_verdict=None, react_verdict=None):
        self.view_verdict = view_verdict or {
            "ok": True, "receipt_mode": "read"}
        self.react_verdict = react_verdict or {"ok": True}
        self.view_calls: list[tuple] = []
        self.react_calls: list[tuple] = []

    def view_status(self, poster_id, message_id, ts=0.0):
        self.view_calls.append((poster_id, message_id, ts))
        return dict(self.view_verdict)

    def react_status(self, poster_id, message_id, emoji):
        self.react_calls.append((poster_id, message_id, emoji))
        return dict(self.react_verdict)


def _runtime(tmp_path, adapter, **toggles):
    """The statusfix rig: bare provision mixin + fake bridge
    supervisor + the scripted surface adapter, with the
    housekeeping beat pinned quiet (tests drive the sweep
    explicitly, except the one that tests the beat itself)."""
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin,
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
    svc["last_cycle"] = time.time()      # the beat stays quiet
    return rt, svc, registry.get(rec.id)


def _rows(personal, kind, rec_id=REC_ID):
    return [r for r in personal.vault.journal(rec_id)
            if r["kind"] == kind]


def _status_wire(sid, *, ts_ms=None, poster=POSTER):
    return {"type": "status_post", "id": sid,
            "from": {"id": poster, "name": "Poster"},
            "text": "", "media": [],
            "ts": _now_ms() if ts_ms is None else ts_ms}


def _feed_status(rt, wa, wire):
    """One bridge status event through the REAL adapter
    dispatch into the REAL runtime personal-surface turn."""
    delivered: list = []
    wa._dispatch(wire, delivered.append)
    assert len(delivered) == 1
    assert rt._personal_surface_turn(delivered[0]) is True
    return delivered[0]


# ── 1. DM command promotion (real adapter dispatch) ─────────

def test_dm_registered_command_promotes_to_owner_turn():
    adapter, msgs = _wa_adapter()
    adapter._dispatch(_own("/viewonce"), msgs.append)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.incoming is True
    assert msg.chat.kind == ChatKind.DM
    assert msg.meta.get("from_owner") is True
    assert msg.meta.get("owner_command") is True
    assert "own_message" not in msg.meta
    assert msg.sender_id == OWNER_JID


def test_dm_viewonce_reply_form_promotes_with_reply_to():
    """The owner's actual dead case: /vv typed as a REPLY to
    the view-once photo inside the sender's DM."""
    adapter, msgs = _wa_adapter()
    adapter._dispatch(
        _own("/vv", reply_to="ORIG-MSG"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is True
    assert msgs[0].meta.get("owner_command") is True
    assert msgs[0].reply_to == "ORIG-MSG"


def test_dm_plain_text_stays_persist_only():
    adapter, msgs = _wa_adapter()
    adapter._dispatch(_own("on my way, give me ten"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is False
    assert msgs[0].meta.get("own_message") is True
    assert "owner_command" not in msgs[0].meta


def test_dm_unknown_slash_stays_persist_only():
    adapter, msgs = _wa_adapter()
    adapter._dispatch(_own("/definitelynotacommand"), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is False
    assert msgs[0].meta.get("own_message") is True


def test_dm_stale_command_promoted_but_stamped_stale():
    adapter, msgs = _wa_adapter()
    adapter._dispatch(
        _own("/viewonce", ts=1_700_000_000_000), msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is True
    assert msgs[0].meta.get("owner_command") is True
    assert msgs[0].meta.get("stale_replay") is True


def test_dm_own_send_echo_never_promotes():
    adapter, msgs = _wa_adapter()
    adapter.connected.set()
    adapter._send_cmd = lambda cmd, timeout=25.0: {
        "id": cmd.get("id"), "ok": True, "message_id": "ECHO-DM"}
    chat = ChatRef(platform="whatsapp", chat_id=TARGET_JID,
                   kind=ChatKind.DM, peer=TARGET_JID)
    result = adapter.send(chat, "/viewonce")
    assert result.ok
    adapter._dispatch(_own("/viewonce", mid="ECHO-DM"),
                      msgs.append)
    assert len(msgs) == 1
    assert msgs[0].incoming is False
    assert msgs[0].meta.get("own_message") is True


# ── 2. status coverage: the funnel reconciles ───────────────

def test_batch_of_five_all_attempted_and_funnel_reconciles(
        tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True, status_react=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    for i in range(5):
        _feed_status(rt, wa, _status_wire(f"S{i}"))
    # every observed post got exactly one view AND one react
    assert sorted(c[1] for c in surface.view_calls) == \
        [f"S{i}" for i in range(5)]
    assert sorted(c[1] for c in surface.react_calls) == \
        [f"S{i}" for i in range(5)]
    # the funnel reconciles from the journal alone:
    # arrivals 5 → live attempts 5 → landed 5
    assert len(_rows(personal, "status_arrival")) == 5
    traces = _rows(personal, "status_view_trace")
    assert len(traces) == 5
    assert all(t["view_attempted"] for t in traces)
    assert all(t["stage"] == "live" for t in traces)
    assert len(_rows(personal, "status_react")) == 5


def test_toggle_off_arrivals_swept_when_toggle_turns_on(
        tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface)   # viewing OFF
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    for i in range(3):
        _feed_status(rt, wa, _status_wire(f"T{i}"))
    assert surface.view_calls == []
    # arrivals recorded even though nothing was attempted —
    # the ledger exists precisely for this backlog
    assert len(_rows(personal, "status_arrival")) == 3
    traces = _rows(personal, "status_view_trace")
    assert len(traces) == 3
    assert all(not t["view_attempted"] for t in traces)
    # the owner turns viewing AND reacting on later the same
    # evening: the sweep gives every still-live post its ONE
    # ACT — the view and the reaction together, like a live
    # arrival would have had
    svc["registry"].set_toggles(rec.id, status_viewing=True,
                                status_react=True)
    personal.cycle_statuses("whatsapp")
    assert sorted(c[1] for c in surface.view_calls) == \
        ["T0", "T1", "T2"]
    assert sorted(c[1] for c in surface.react_calls) == \
        ["T0", "T1", "T2"]
    sweep_traces = [t for t in _rows(personal,
                                     "status_view_trace")
                    if t.get("stage") == "sweep"]
    assert len(sweep_traces) == 3
    assert all(t["view_attempted"] for t in sweep_traces)
    # arrivals never duplicate
    assert len(_rows(personal, "status_arrival")) == 3


def test_stale_timestamped_batch_still_attempted(tmp_path):
    """A reconnect/history batch carries OLD post timestamps;
    the live path must still attempt every one (no freshness
    window eats status posts)."""
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    old_ms = _now_ms() - 2 * 3600 * 1000      # two hours old
    for i in range(3):
        _feed_status(rt, wa, _status_wire(f"OLD{i}",
                                          ts_ms=old_ms))
    assert sorted(c[1] for c in surface.view_calls) == \
        ["OLD0", "OLD1", "OLD2"]
    assert len(_rows(personal, "status_arrival")) == 3


def test_expired_posts_are_not_swept(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface)   # viewing OFF
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    dead_ms = _now_ms() - 25 * 3600 * 1000     # past the 24h life
    _feed_status(rt, wa, _status_wire("DEAD", ts_ms=dead_ms))
    svc["registry"].set_toggles(rec.id, status_viewing=True)
    personal.cycle_statuses("whatsapp")
    assert surface.view_calls == []


def test_refused_view_retried_next_process_not_same(tmp_path):
    calls: list = []

    def refusing_viewer(rec, item):
        calls.append(item["source_id"])
        return False

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    organ1 = PersonalSurfaces(registry, vault,
                              status_viewer=refusing_viewer)
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=True)
    organ1.observe_status_post(
        "whatsapp", poster_id=POSTER, poster_name="Poster",
        source_id="R1", ts=time.time())
    assert calls == ["R1"]
    # same process: the sweep does NOT beat on a refused verb
    organ1.cycle_statuses("whatsapp")
    assert calls == ["R1"]
    # next process (fresh organ over the same vault): the
    # journal says attempted-but-not-landed, so it retries once
    organ2 = PersonalSurfaces(
        PairingRegistry(tmp_path / "paired.json"),
        PersonalVault(tmp_path / "personal"),
        status_viewer=refusing_viewer)
    organ2.cycle_statuses("whatsapp")
    assert calls == ["R1", "R1"]
    # and the funnel still reconciles: one arrival, two traces
    assert len(_rows(organ2, "status_arrival")) == 1
    stages = sorted(t.get("stage") or ""
                    for t in _rows(organ2, "status_view_trace"))
    assert stages == ["live", "sweep"]


def test_landed_view_never_refired_after_restart(tmp_path):
    calls: list = []

    def landing_viewer(rec, item):
        calls.append(item["source_id"])
        item["bridge_verdict"] = {"ok": True,
                                  "receipt_mode": "read"}
        return True

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    organ1 = PersonalSurfaces(registry, vault,
                              status_viewer=landing_viewer)
    rec = registry.upsert(AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=True)
    organ1.observe_status_post(
        "whatsapp", poster_id=POSTER, poster_name="Poster",
        source_id="L1", ts=time.time())
    assert calls == ["L1"]
    # a new process over the same journal: redelivery AND the
    # sweep both leave the landed post alone — the viewed set
    # seeded from the journal's landed trace
    organ2 = PersonalSurfaces(
        PairingRegistry(tmp_path / "paired.json"),
        PersonalVault(tmp_path / "personal"),
        status_viewer=landing_viewer)
    organ2.observe_status_post(
        "whatsapp", poster_id=POSTER, poster_name="Poster",
        source_id="L1", ts=time.time())
    organ2.cycle_statuses("whatsapp")
    assert calls == ["L1"]


def test_housekeeping_beat_drives_the_status_sweep(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface)   # viewing OFF
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("B1"))
    assert surface.view_calls == []
    svc["registry"].set_toggles(rec.id, status_viewing=True)
    # the surface reconcile is not what this pin exercises
    rt._sync_whatsapp_surface = lambda: ""
    svc["last_cycle"] = 0.0
    rt._housekeeping_beat()
    assert [c[1] for c in surface.view_calls] == ["B1"]


# ── 3. view+react as one act ────────────────────────────────

def test_failed_view_does_not_block_react(tmp_path):
    surface = _FakeSurfaceAdapter(
        view_verdict={"ok": False, "error": "bridge says no"})
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True, status_react=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("F1"))
    assert len(surface.view_calls) == 1
    assert len(surface.react_calls) == 1
    assert len(_rows(personal, "status_react")) == 1
    assert len(_rows(personal, "status_view_failed")) == 1


def test_failed_react_does_not_block_view(tmp_path):
    surface = _FakeSurfaceAdapter(
        react_verdict={"ok": False, "error": "react refused"})
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True, status_react=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("F2"))
    assert len(surface.view_calls) == 1
    assert len(surface.react_calls) == 1
    assert (rec.id, "F2") in personal._viewed_status
    assert len(_rows(personal, "status_react_failed")) == 1


def test_only_react_toggle_views_nothing_reacts(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_react=True)    # viewing OFF
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("RO"))
    assert surface.view_calls == []
    assert len(surface.react_calls) == 1
    traces = _rows(personal, "status_view_trace")
    assert len(traces) == 1
    assert traces[0]["view_attempted"] is False


def test_only_viewing_toggle_reacts_nothing(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True)  # react OFF
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("VO"))
    assert len(surface.view_calls) == 1
    assert surface.react_calls == []


# ── 4. view-once arrival truth ──────────────────────────────

def _media_file(tmp_path, name="vo.jpg"):
    path = tmp_path / name
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(path)


def _vo_wire(mid, media):
    return {"type": "message", "id": mid,
            "chat": {"id": TARGET_JID, "kind": "dm",
                     "title": "Ada"},
            "from": {"id": TARGET_JID, "name": "Ada"},
            "text": "", "view_once": True, "media": media,
            "ts": _now_ms()}


def _feed_dm(rt, wa, wire):
    delivered: list = []
    wa._dispatch(wire, delivered.append)
    assert len(delivered) == 1
    rt._personal_surface_turn(delivered[0])
    return delivered[0]


def test_view_once_saved_arrival_traced(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    src = _media_file(tmp_path)
    _feed_dm(rt, wa, _vo_wire("vo1", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}]))
    saved = [r for r in personal.vault.saved_entries(rec.id)
             if r.get("kind") == "view_once"]
    assert len(saved) == 1
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["view_once_toggle"] is True
    assert traces[0]["media_present"] is True
    assert traces[0]["outcome"] == "saved"


def test_view_once_toggle_off_arrival_traced_no_save(tmp_path):
    """THE silent hole: capture off at arrival used to write
    no row of any kind — a lifetime of arrivals read exactly
    like a lifetime of nothing arriving."""
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface)   # view_once OFF
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    src = _media_file(tmp_path)
    _feed_dm(rt, wa, _vo_wire("vo2", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}]))
    assert [r for r in personal.vault.saved_entries(rec.id)
            if r.get("kind") == "view_once"] == []
    assert _rows(personal, "view_once_missed") == []
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["view_once_toggle"] is False
    assert traces[0]["outcome"] == "skipped_toggle_off"


def test_view_once_media_missing_traced_as_missed(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    # the media entry carries no path: the download never
    # landed — the adapter stamps the miss, the organ journals
    # BOTH the missed row and the arrival trace
    _feed_dm(rt, wa, _vo_wire("vo3", [{"kind": "image"}]))
    assert len(_rows(personal, "view_once_missed")) == 1
    traces = _rows(personal, "view_once_trace")
    assert len(traces) == 1
    assert traces[0]["media_missing"] is True
    assert traces[0]["outcome"] == "missed"


def test_view_once_redelivery_writes_one_trace(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(tmp_path, surface, view_once=True)
    personal = svc["personal"]
    wa, _ = _wa_adapter()
    src = _media_file(tmp_path)
    wire = _vo_wire("vo4", [
        {"path": src, "mime": "image/jpeg", "kind": "image"}])
    _feed_dm(rt, wa, wire)
    _feed_dm(rt, wa, wire)
    assert len(_rows(personal, "view_once_trace")) == 1


# ── 4b. /viewonce reveal attempts journal ───────────────────

def _chatops_runtime(tmp_path, **toggles):
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )
    from nomorals.agents.partner.runtime_provision import (
        RuntimeProvisionMixin,
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
    return rt, svc, registry.get(rec.id)


def _chat_msg(reply_to=""):
    return SimpleNamespace(
        reply_to=reply_to,
        chat=SimpleNamespace(platform="whatsapp"))


def _seed_capture(tmp_path, rt, svc):
    personal = svc["personal"]
    src = _media_file(tmp_path, "seed.jpg")
    personal.observe_message(
        "whatsapp", chat_key=f"whatsapp:{TARGET_JID}",
        counterpart_id=TARGET_JID, sender_id=TARGET_JID,
        sender_name="Ada", message_id="seedmsg", text="",
        media=[{"path": src, "kind": "image"}],
        view_once=True, ts=time.time())
    saved = [r for r in personal.vault.saved_entries()
             if r.get("kind") == "view_once"]
    assert len(saved) == 1
    return saved[0]


def test_reveal_on_empty_vault_journals_empty(tmp_path):
    rt, svc, rec = _chatops_runtime(tmp_path, view_once=True)
    reply = rt._chatops_viewonce("viewonce", "", _chat_msg())
    assert "no view-once captures yet" in reply
    rows = _rows(svc["personal"], "viewonce_reveal")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "empty"
    assert rows[0]["form"] == "bare"


def test_reveal_reply_miss_journals_miss(tmp_path):
    rt, svc, rec = _chatops_runtime(tmp_path, view_once=True)
    _seed_capture(tmp_path, rt, svc)
    reply = rt._chatops_viewonce(
        "vv", "", _chat_msg(reply_to="NO-SUCH-MSG"))
    assert "no view-once capture" in reply
    rows = _rows(svc["personal"], "viewonce_reveal")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "miss"
    assert rows[0]["form"] == "reply"


def test_reveal_found_journals_found_and_sends(tmp_path):
    rt, svc, rec = _chatops_runtime(tmp_path, view_once=True)
    _seed_capture(tmp_path, rt, svc)
    sent: list = []

    def send_file(platform, chat, path, caption=""):
        sent.append(path)
        return SimpleNamespace(ok=True)

    rt.gateway = SimpleNamespace(send_file=send_file)
    reply = rt._chatops_viewonce("viewonce", "", _chat_msg())
    assert "straight from your vault" in reply
    assert len(sent) == 1
    rows = _rows(svc["personal"], "viewonce_reveal")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "found"
    assert rows[0]["form"] == "bare"
    assert rows[0]["poster_name"] == "Ada"


def test_reveal_usage_error_is_not_an_attempt(tmp_path):
    rt, svc, rec = _chatops_runtime(tmp_path, view_once=True)
    _seed_capture(tmp_path, rt, svc)
    reply = rt._chatops_viewonce("viewonce", "abc", _chat_msg())
    assert reply.startswith("usage:")
    assert _rows(svc["personal"], "viewonce_reveal") == []


# ── 5. verdicts carry evidence ──────────────────────────────

def _verdict_rec(**toggles):
    return AccountRecord(
        id=REC_ID, platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0,
        toggles=toggles)


def test_verdict_evidence_pure_shapes():
    rec = _verdict_rec(view_once=True, status_viewing=True,
                       status_react=True, status_saving=True)
    base = dict(stopped=False, running=True, attached=True,
                connected=True)
    # no activity map (a caller that cannot see the journal):
    # the historic naked verdict, unchanged
    assert toggle_live_verdicts(
        rec, **base)["view_once"] == "✓ live"
    # activity present: the fact rides the verdict
    when = 1794100000.0
    out = toggle_live_verdicts(
        rec, activity={"view_once": when,
                       "status_react": when}, **base)
    assert out["view_once"].startswith("✓ live · last capture ")
    assert out["status_react"].startswith(
        "✓ live · last reaction ")
    # zero lifetime activity SAYS so, in words
    assert out["status_viewing"] == \
        "✓ live · NO view ever landed"
    assert out["status_saving"] == \
        "✓ live · nothing ever saved"


def test_pair_toggle_live_carries_landed_view_evidence(
        tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True, status_react=True)
    wa, _ = _wa_adapter()
    _feed_status(rt, wa, _status_wire("E1"))
    verdicts = rt._pair_toggle_live(rec, svc=svc)
    assert verdicts["status_viewing"].startswith(
        "✓ live · last view landed ")
    assert verdicts["status_react"].startswith(
        "✓ live · last reaction ")


def test_pair_toggle_live_names_zero_activity(tmp_path):
    surface = _FakeSurfaceAdapter()
    rt, svc, rec = _runtime(
        tmp_path, surface, status_viewing=True, view_once=True)
    verdicts = rt._pair_toggle_live(rec, svc=svc)
    assert verdicts["status_viewing"] == \
        "✓ live · NO view ever landed"
    assert verdicts["view_once"] == \
        "✓ live · NO capture ever recorded"
