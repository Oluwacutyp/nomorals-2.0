"""Status view/react fix (2026-11-07) — ARGUMENT-SHAPE pins.

The mining (~/workspace/statusmining/REPORT.md) found why two
"working" passes shipped dead features: the composed harness's
recording fake bridge accepted any well-shaped call, so the
bridge layer's real shapes were never exercised.  These pins
assert shapes, not "verb fired":

- the WhatsApp adapter's status_view / status_react wire
  payloads, exactly (the poster's JID verbatim — phone form
  and LID form; no post timestamp plumbed into the view);
- the poster JID's form end to end: a bridge-shaped status_post
  wire object (Go emit shape) reaches the delivered message's
  sender_id byte-identical — no bare-number reduction, no
  alt-form swap, no device suffix;
- the composed seam + organ: a bridge verdict of
  receipt_mode "read-self" journals status_view_ineffective
  (once per post per process) and never marks the post viewed,
  while a refusal still journals status_view_failed;
- the live trace: EVERY observed post journals one
  status_view_trace row — resolved record, the toggle value
  read from that record, the exact poster/source/ts handed to
  the seam, the bridge's raw verdict — on success too;
- the /pair verdict: status_viewing + a known read-self mode
  names the OFF read-receipts setting instead of "✓ live";
- the Baileys bridge source: the status_react send carries
  { statusJidList: [poster] } and the status_view reply
  carries receipt_mode;
- /chanstory language: WhatsApp says "channel post", never
  dresses a post as a story (Channel Status is a walled,
  distinct feature — see test_chanstories for the reply pins).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.social.pairing import AccountRecord, PairingRegistry, toggle_live_verdicts
from nomorals.social.personal import PersonalSurfaces, PersonalVault

_REPO = Path(__file__).resolve().parent.parent
PN = "2348012345678@s.whatsapp.net"
LID = "123456789012345@lid"
POST_TS = 1794100000.0


# ── 1. the adapter's wire shapes ────────────────────────────

def _wa_adapter_capturing(responder=None):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter.connected.set()
    sent = []

    def fake_send_cmd(cmd, *, timeout=25.0):
        sent.append(dict(cmd))
        if responder is not None:
            return responder(cmd)
        return {"ok": True}

    adapter._send_cmd = fake_send_cmd
    return adapter, sent


def test_view_status_payload_is_exactly_id_and_from():
    adapter, sent = _wa_adapter_capturing()
    adapter.view_status(PN, "POST1", ts=POST_TS)
    # Exactly {cmd, id, from}: the poster's JID verbatim, and
    # NO ts — the bridges stamp the receipt at read time, like
    # the phone; plumbing the post's ts through was divergence
    # D5 of the mining.
    assert sent == [{"cmd": "status_view", "id": "POST1",
                     "from": PN}]


def test_view_status_payload_lid_poster_verbatim():
    adapter, sent = _wa_adapter_capturing()
    adapter.view_status(LID, "POST2", ts=POST_TS)
    assert sent == [{"cmd": "status_view", "id": "POST2",
                     "from": LID}]


def test_react_status_payload_exact_both_poster_forms():
    adapter, sent = _wa_adapter_capturing()
    adapter.react_status(PN, "POST1", "❤️")
    adapter.react_status(LID, "POST2", "🔥")
    assert sent == [
        {"cmd": "status_react", "id": "POST1", "from": PN,
         "emoji": "❤️"},
        {"cmd": "status_react", "id": "POST2", "from": LID,
         "emoji": "🔥"}]


def test_view_status_verdict_carries_receipt_mode_through():
    adapter, _ = _wa_adapter_capturing(
        responder=lambda cmd: {"ok": True,
                                "receipt_mode": "read-self"})
    res = adapter.view_status(PN, "POST1")
    assert res["ok"] is True
    assert res["receipt_mode"] == "read-self"


# ── 2. the poster JID's form, wire → delivered message ──────

def _dispatch_status(adapter, wire_from):
    delivered = []
    adapter._dispatch(
        {"type": "status_post", "id": "POST1",
         "from": wire_from, "text": "", "media": [],
         "ts": 1794100000000},
        delivered.append)
    assert len(delivered) == 1
    return delivered[0]


def test_sender_id_is_the_wire_from_id_verbatim_lid():
    # Go bridge emit shape: from.id is the full JID (device
    # stripped by ToNonAD), from.alt the OTHER form — the
    # delivered sender_id must be from.id itself, never the
    # alt, never bare digits.
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_status(adapter, {
        "id": LID, "alt": PN, "name": "Poster"})
    assert msg.sender_id == LID
    assert msg.chat.peer == LID
    assert msg.message_id == "POST1"
    assert msg.meta["service"] == "status_post"


def test_sender_id_is_the_wire_from_id_verbatim_pn():
    # Baileys emit shape: from.id = jidNormalizedUser(participant).
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    msg = _dispatch_status(adapter, {"id": PN, "name": "Poster"})
    assert msg.sender_id == PN
    assert msg.chat.peer == PN


# ── 3. the composed seam + organ (real runtime closures) ────

class _FakeViewAdapter:
    """The WhatsApp adapter's view_status, scripted."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls: list[tuple] = []

    def view_status(self, poster_id, message_id, ts=0.0):
        self.calls.append((poster_id, message_id, ts))
        return dict(self.verdict)


def _runtime(tmp_path, adapter, *, viewing=True):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    rt = RuntimeProvisionMixin.__new__(RuntimeProvisionMixin)
    rt.context = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path), chat=None))
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=viewing)
    rt._pair_supervisor = SimpleNamespace(
        registry=registry, is_running=lambda rid: True)
    rt._adapter_for = lambda platform: adapter
    svc = rt._pairing_personal()
    return rt, svc, registry.get(rec.id)


def _rows(personal, rec_id, kind):
    return [r for r in personal.vault.journal(rec_id)
            if r["kind"] == kind]


def _observe(personal, source_id="POST1"):
    personal.observe_status_post(
        "whatsapp", poster_id=PN, poster_name="Poster",
        source_id=source_id, ts=POST_TS)


def test_viewed_post_trace_and_dedupe(tmp_path):
    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    personal = svc["personal"]
    _observe(personal)
    _observe(personal)  # redelivery: deduped, traced once
    assert adapter.calls == [(PN, "POST1", POST_TS)]
    assert (rec.id, "POST1") in personal._viewed_status
    assert _rows(personal, rec.id, "status_view_failed") == []
    assert _rows(personal, rec.id,
                 "status_view_ineffective") == []
    traces = _rows(personal, rec.id, "status_view_trace")
    assert len(traces) == 1
    row = traces[0]
    assert row["account"] == rec.id
    assert row["status_viewing_toggle"] is True
    assert row["seam_poster_id"] == PN
    assert row["seam_source_id"] == "POST1"
    assert row["seam_ts"] == POST_TS
    assert row["view_attempted"] is True
    assert row["bridge_verdict"]["ok"] is True
    assert row["bridge_verdict"]["receipt_mode"] == "read"


def test_refused_post_trace_carries_the_verdict_verbatim(tmp_path):
    adapter = _FakeViewAdapter({"ok": False,
                                "error": "bridge says no"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    personal = svc["personal"]
    _observe(personal)
    _observe(personal)  # a refusal is retried on redelivery…
    assert len(adapter.calls) == 2
    # …journaled failed once, and traced once with the raw
    # verdict — ok and the bridge's own error string, verbatim.
    assert len(_rows(personal, rec.id,
                     "status_view_failed")) == 1
    assert _rows(personal, rec.id,
                 "status_view_ineffective") == []
    traces = _rows(personal, rec.id, "status_view_trace")
    assert len(traces) == 1
    assert traces[0]["bridge_verdict"] == {
        "ok": False, "error": "bridge says no"}
    assert traces[0]["view_attempted"] is True
    assert (rec.id, "POST1") not in personal._viewed_status


def test_read_self_journals_ineffective_never_viewed(tmp_path):
    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read-self"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    personal = svc["personal"]
    _observe(personal)
    _observe(personal)
    # Not landed for dedupe (the poster is never notified):
    # the verb retries on redelivery…
    assert len(adapter.calls) == 2
    assert (rec.id, "POST1") not in personal._viewed_status
    # …the ineffective fact journals once, the refusal entry
    # never fires (the bridge did NOT refuse)…
    assert len(_rows(personal, rec.id,
                     "status_view_ineffective")) == 1
    assert _rows(personal, rec.id, "status_view_failed") == []
    # …the mode is remembered, and the trace shows the landed-
    # but-downgraded verdict for what it is.
    assert rt._status_receipt_modes[rec.id] == "read-self"
    traces = _rows(personal, rec.id, "status_view_trace")
    assert traces[0]["bridge_verdict"]["ok"] is True
    assert traces[0]["bridge_verdict"]["receipt_mode"] \
        == "read-self"


def test_trace_fires_with_toggle_off_and_no_attempt(tmp_path):
    adapter = _FakeViewAdapter({"ok": True})
    rt, svc, rec = _runtime(tmp_path, adapter, viewing=False)
    personal = svc["personal"]
    _observe(personal)
    assert adapter.calls == []
    traces = _rows(personal, rec.id, "status_view_trace")
    assert len(traces) == 1
    # The diagnostic case: the journal shows the toggle was OFF
    # on the record the event resolved to — no guessing.
    assert traces[0]["status_viewing_toggle"] is False
    assert traces[0]["view_attempted"] is False
    assert traces[0]["bridge_verdict"] is None
    assert traces[0]["seam_poster_id"] == PN


def test_pair_verdict_names_receipts_off_once_known(tmp_path):
    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read-self"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    _observe(svc["personal"])
    verdict = rt._pair_toggle_live(rec)["status_viewing"]
    assert "can't register" in verdict
    assert "read receipts are OFF" in verdict
    assert "Settings → Privacy → Read receipts" in verdict


def test_pair_verdict_live_when_receipts_read(tmp_path):
    adapter = _FakeViewAdapter({"ok": True,
                                "receipt_mode": "read"})
    rt, svc, rec = _runtime(tmp_path, adapter)
    _observe(svc["personal"])
    assert rt._pair_toggle_live(rec)["status_viewing"] \
        == "✓ live"


# ── 4. toggle_live_verdicts, pure ───────────────────────────

def _verdict_rec():
    return AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0,
        toggles={"status_viewing": True})


def _verdicts(**kw):
    return toggle_live_verdicts(
        _verdict_rec(), stopped=False, running=True,
        attached=True, connected=True, **kw)


def test_verdict_read_self_names_the_setting():
    verdict = _verdicts(
        status_receipt_mode="read-self")["status_viewing"]
    assert "can't register" in verdict
    assert "read receipts are OFF" in verdict


def test_verdict_read_and_unknown_and_unset_stay_live():
    # "read" is a working view; "unknown" never cries wolf; no
    # kwarg at all (the web rung, older callers) is unchanged.
    assert _verdicts(status_receipt_mode="read")[
        "status_viewing"] == "✓ live"
    assert _verdicts(status_receipt_mode="unknown")[
        "status_viewing"] == "✓ live"
    assert _verdicts()["status_viewing"] == "✓ live"


# ── 5. the organ alone: trace + ineffective discipline ──────

def _organ(tmp_path, viewer):
    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault,
                                status_viewer=viewer)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    registry.set_toggles(rec.id, status_viewing=True)
    return personal, registry.get(rec.id)


def test_organ_traces_even_when_the_seam_raises(tmp_path):
    def boom(rec, item):
        raise RuntimeError("kaboom")

    personal, rec = _organ(tmp_path, boom)
    _observe(personal)
    traces = _rows(personal, rec.id, "status_view_trace")
    assert len(traces) == 1
    assert traces[0]["view_attempted"] is True
    # The seam raised before annotating: no verdict is claimed.
    assert traces[0]["bridge_verdict"] is None
    assert len(_rows(personal, rec.id,
                     "status_view_failed")) == 1


# ── 6. the Baileys bridge source shapes ─────────────────────

def _mjs_case(source: str, name: str) -> str:
    start = source.index(f"case '{name}':")
    nxt = source.index("case '", start + 10)
    return source[start:nxt]


def test_baileys_status_react_carries_the_poster_audience():
    src = (_REPO / "bridge" / "whatsapp-bridge.mjs").read_text(
        encoding="utf-8")
    block = _mjs_case(src, "status_react")
    # The mined death of this verb: no statusJidList ⇒ Baileys
    # encrypts the reaction for zero devices and resolves ok.
    assert "{ statusJidList: [poster] }" in block
    assert "participant: poster" in block


def test_baileys_status_view_reports_receipt_mode():
    src = (_REPO / "bridge" / "whatsapp-bridge.mjs").read_text(
        encoding="utf-8")
    block = _mjs_case(src, "status_view")
    assert "receipt_mode: receiptMode" in block
    assert "fetchPrivacySettings" in block
    assert "readreceipts === 'all'" in block


# ── 7. /chanstory language ──────────────────────────────────

def test_chanstory_catalog_says_channel_post_not_story():
    from nomorals.social.chat.control import COMMAND_DETAILS

    what = COMMAND_DETAILS["chanstory"]["what"]
    assert "channel post" in what
    assert "ARE its story format" not in what
    # the wall is named in the same breath
    assert "Channel Status" in what
