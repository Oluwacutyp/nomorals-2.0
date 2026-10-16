"""GROUPS Tier 1: WA provisioner, /mix surface, welcome/moderation, digest.

Invariants under test:
* provisioner refuses unauthenticated calls (every sensitive method);
* a mix is reproducible from its descriptor (same descriptor →
  same fingerprint + byte-identical render plan);
* moderation always offers an appeal path;
* the digest picks decisions + unanswered questions by deterministic
  scoring — never random, never chatter.
"""

import json
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nomorals.community.digest import (
    DEFAULT_RITUALS,
    Ritual,
    RitualScheduler,
    build_digest,
    next_occurrences,
    parse_rrule,
    render_digest,
    render_ritual,
)
from nomorals.community.welcome_moderation import (
    BanList,
    CaptchaGate,
    IdentityError,
    ModerationConfig,
    Moderator,
    Roster,
    WelcomeConfig,
    render_goodbye,
    render_welcome,
    validate_identity,
    welcome_flow,
)
from nomorals.media.mix_surface import (
    MixDescriptor,
    MixTrack,
    deliver,
    fingerprint,
    format_mix_card,
    load_descriptor,
    render_mix,
    render_plan,
    share_descriptor,
    with_track_added,
    with_tracks_reordered,
)
from nomorals.social.wa_provisioner import (
    ProvisioningError,
    UnauthenticatedError,
    WhatsAppProvisioner,
    mask_phone,
)

TOKEN = "test-single-tenant-token"


# ── WA provisioner ─────────────────────────────────────────────────────

@pytest.fixture
def prov(tmp_path):
    return WhatsAppProvisioner(TOKEN, state_path=tmp_path / "registry.json")


def test_provisioner_refuses_unauthenticated(prov):
    with pytest.raises(UnauthenticatedError):
        prov.create_session(None, method="qr")
    with pytest.raises(UnauthenticatedError):
        prov.create_session("wrong-token", method="qr")
    with pytest.raises(UnauthenticatedError):
        prov.list_sessions(None)
    with pytest.raises(UnauthenticatedError):
        prov.list_sessions("")
    # Authenticated works.
    rec = prov.create_session(TOKEN, method="qr", session_id="s1")
    assert rec.status == "pending_qr"
    with pytest.raises(UnauthenticatedError):
        prov.get_qr(None, "s1")
    with pytest.raises(UnauthenticatedError):
        prov.get_qr("nope", "s1")
    with pytest.raises(UnauthenticatedError):
        prov.revoke("nope", "s1")
    with pytest.raises(UnauthenticatedError):
        prov.confirm_linked(None, "s1")
    with pytest.raises(UnauthenticatedError):
        prov.request_pairing_code(None, "s1")
    with pytest.raises(UnauthenticatedError):
        prov.verify_pairing_code(None, "s1", "00000000")
    with pytest.raises(UnauthenticatedError):
        prov.cleanup_expired(None)
    with pytest.raises(UnauthenticatedError):
        prov.get_session("wrong", "s1")


def test_provisioner_requires_token_and_localhost_default(tmp_path):
    with pytest.raises(ProvisioningError):
        WhatsAppProvisioner("", state_path=tmp_path / "r.json")
    with pytest.raises(ProvisioningError):
        WhatsAppProvisioner(
            TOKEN, bind_host="0.0.0.0", state_path=tmp_path / "r.json"
        )
    # Explicit exposure + token is allowed.
    ok = WhatsAppProvisioner(
        TOKEN, bind_host="0.0.0.0", explicit_allow_remote=True,
        state_path=tmp_path / "r.json",
    )
    assert ok.bind_host == "0.0.0.0"


def test_provisioner_qr_flow_and_registry(prov):
    rec = prov.create_session(TOKEN, method="qr", session_id="qr1")
    qr = prov.get_qr(TOKEN, "qr1")
    assert qr["qr_payload"].startswith("devon-wa-link:qr1:")
    assert qr["ttl_s"] > 0
    assert "/wa/link?session=qr1" in qr["provisioning_url"]
    assert "PHONE browser" in qr["instructions"]
    assert "not SSH" in qr["instructions"]
    html = prov.provisioning_page_html("qr1")
    assert "qr1" in html and "Do not paste codes into SSH" in html
    linked = prov.confirm_linked(TOKEN, "qr1", jid="23480...@s.whatsapp.net")
    assert linked.status == "linked" and linked.linked_at is not None
    rows = prov.list_sessions(TOKEN)
    assert [(r.id, r.status) for r in rows] == [("qr1", "linked")]
    assert rows[0].created > 0 and rows[0].last_seen >= rows[0].created
    revoked = prov.revoke(TOKEN, "qr1")
    assert revoked.status == "revoked"


def test_provisioner_pairing_code_8_digits(prov):
    prov.create_session(TOKEN, method="pairing_code", session_id="pc1",
                        phone_number="+2348012345678")
    first = prov.request_pairing_code(TOKEN, "pc1")
    code = first["pairing_code"]
    assert len(code) == 8 and code.isdigit()
    assert first["formatted"] == f"{code[:4]}-{code[4:]}"
    assert "8-digit" in first["instructions"]
    # Masked phone in the registry, never the full number.
    assert prov.get_session(TOKEN, "pc1").phone_hint == mask_phone("+2348012345678")
    assert "8012345678" not in prov.get_session(TOKEN, "pc1").phone_hint
    assert prov.verify_pairing_code(TOKEN, "pc1", "00000000") is False
    assert prov.verify_pairing_code(TOKEN, "pc1", code) is True
    # A re-issued code differs (counter bumped) and verifies.
    second = prov.request_pairing_code(TOKEN, "pc1")
    assert second["pairing_code"] != code
    assert prov.verify_pairing_code(TOKEN, "pc1", second["pairing_code"]) is True


def test_provisioner_registry_persists(tmp_path):
    p1 = WhatsAppProvisioner(TOKEN, state_path=tmp_path / "r.json")
    p1.create_session(TOKEN, method="qr", session_id="persist1")
    p2 = WhatsAppProvisioner(TOKEN, state_path=tmp_path / "r.json")
    assert [r.id for r in p2.list_sessions(TOKEN)] == ["persist1"]


def test_provisioner_expired_qr_cleanup(prov):
    prov.create_session(TOKEN, method="qr", session_id="old1")
    rec = prov.get_session(TOKEN, "old1")
    rec.qr_expires_at = 1.0  # force past deadline
    assert prov.cleanup_expired(TOKEN) == ["old1"]
    assert prov.get_session(TOKEN, "old1").status == "expired"


# ── /mix surface ───────────────────────────────────────────────────────

def _descriptor() -> MixDescriptor:
    return MixDescriptor(
        id="mix-1",
        title="Friday warm-up",
        tracks=[
            MixTrack(path="", title="Sunset drive", bpm=120.0, key="C",
                     mode="major", energy=0.4, duration_s=180.0),
            MixTrack(path="", title="Peak hour", bpm=122.0, key="G",
                     mode="major", energy=0.8, duration_s=200.0),
            MixTrack(path="", title="Cool down", bpm=118.0, key="A",
                     mode="minor", energy=0.5, duration_s=160.0),
        ],
        energy_curve="late_peak",
        seed=7,
    )


def test_mix_reproducible_from_descriptor():
    d = _descriptor()
    assert fingerprint(d) == fingerprint(MixDescriptor.from_dict(d.to_dict()))
    plan_a = render_plan(d)
    plan_b = render_plan(MixDescriptor.from_dict(json.loads(share_descriptor(d))))
    assert plan_a == plan_b
    assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
    # Editing changes the fingerprint; re-render is still deterministic.
    edited = with_track_added(d, MixTrack(title="Encore", bpm=120.0, duration_s=60.0))
    assert edited.version == d.version + 1
    assert fingerprint(edited) != fingerprint(d)
    assert render_plan(edited) == render_plan(edited)
    reordered = with_tracks_reordered(d, [2, 0, 1])
    assert [t.title for t in reordered.tracks] == [
        "Cool down", "Sunset drive", "Peak hour"]
    assert fingerprint(reordered) != fingerprint(d)


def test_mix_share_load_roundtrip_and_card():
    d = _descriptor()
    loaded = load_descriptor(share_descriptor(d))
    assert loaded.to_dict() == d.to_dict()
    card = format_mix_card(d)
    assert "Friday warm-up" in card
    assert "Sunset drive" in card and "Peak hour" in card
    assert "late_peak" in card
    with pytest.raises(ValueError):
        load_descriptor('{"not": "a mix"}')


def _write_sine_wav(path: Path, freq: float, seconds: float = 1.0, sr: int = 22050):
    n = int(sr * seconds)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(frames)


def test_mix_render_audio_and_resumable(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_sine_wav(a, 440.0)
    _write_sine_wav(b, 660.0)
    d = MixDescriptor(
        id="mix-audio",
        title="Audio test",
        tracks=[
            MixTrack(path=str(a), title="A", bpm=120.0, key="C",
                     mode="major", energy=0.5, duration_s=1.0),
            MixTrack(path=str(b), title="B", bpm=120.0, key="C",
                     mode="major", energy=0.6, duration_s=1.0),
        ],
    )
    out = tmp_path / "out.wav"
    ckpt = tmp_path / "ckpt.json"
    first = render_mix(d, out_path=out, checkpoint_path=ckpt)
    assert first.audio_path == str(out) and Path(out).exists()
    assert first.completed_indices == [0, 1]
    assert json.loads(ckpt.read_text())["fingerprint"] == first.fingerprint
    # Second run resumes from the checkpoint.
    second = render_mix(d, out_path=out, checkpoint_path=ckpt, resume=True)
    assert second.resumed_from_checkpoint is True
    assert second.fingerprint == first.fingerprint
    assert second.plan == first.plan


def test_mix_render_missing_tracks_is_plan_only():
    d = _descriptor()  # paths are "" — but give a missing path to one track
    d.tracks[0].path = "/nonexistent/nope.wav"
    rendered = render_mix(d)
    assert rendered.audio_path is None
    assert rendered.missing_tracks  # honest, no fake audio
    assert rendered.plan["n_tracks"] == 3


class _FakeSender:
    def __init__(self):
        self.calls = []

    def send_audio(self, chat, path, **kwargs):
        self.calls.append(("audio", str(chat), path, kwargs))
        return {"ok": True, "message_id": "m1"}

    def send_voice(self, chat, path, **kwargs):
        self.calls.append(("voice", str(chat), path, kwargs))
        return {"ok": True, "message_id": "m2"}


def test_mix_deliver_via_injected_sender(tmp_path):
    a = tmp_path / "a.wav"
    _write_sine_wav(a, 440.0)
    d = MixDescriptor(
        id="mix-deliver",
        tracks=[MixTrack(path=str(a), title="A", bpm=120.0, duration_s=1.0)],
    )
    rendered = render_mix(d, out_path=tmp_path / "o.wav",
                          checkpoint_path=tmp_path / "c.json")
    sender = _FakeSender()
    receipt = deliver("group-123", rendered, sender=sender, as_voice_note=True)
    assert receipt.kind == "voice_note"
    assert receipt.chat == "group-123"
    assert receipt.fingerprint == rendered.fingerprint
    assert sender.calls and sender.calls[0][0] == "voice"
    receipt2 = deliver("group-123", rendered, sender=sender)
    assert receipt2.kind == "audio"
    # Plan-only mix: deliver refuses instead of fake success.
    plan_only = render_mix(_descriptor())
    with pytest.raises(ValueError):
        deliver("group-123", plan_only, sender=sender)


# ── welcome / moderation ───────────────────────────────────────────────

def test_welcome_flow_rules_intro_role():
    cfg = WelcomeConfig(
        group_name="Lagos Devs",
        rules=["No spam", "Stay on topic"],
        default_role="member",
    )
    steps = welcome_flow("ada", cfg)
    assert [s["step"] for s in steps] == ["rules", "intro", "role"]
    assert [s["order"] for s in steps] == [1, 2, 3]
    assert "No spam" in steps[0]["text"]
    assert steps[2]["role"] == "member"
    text = render_welcome("ada", cfg)
    assert "ada" in text and "Lagos Devs" in text and "No spam" in text
    assert "ada" in render_goodbye("ada", cfg)


def test_identity_boundary_never_owner_real():
    with pytest.raises(IdentityError):
        validate_identity({"owner_real_email": "me@example.com"})
    with pytest.raises(IdentityError):
        validate_identity({"real_phone": "+2348000000000"})
    with pytest.raises(IdentityError):
        validate_identity({"is_owner": True})
    with pytest.raises(IdentityError):
        validate_identity({"disable_identity_check": True})
    # Self-generated identities pass.
    assert validate_identity({"handle": "trial-user-4821"})["handle"] == "trial-user-4821"


class _PassChecker:
    def check(self, user_id, answer):
        return answer == "correct"


def test_captcha_gate_on_join():
    gate = CaptchaGate(_PassChecker())
    assert gate.request_join("u1")["state"] == "pending_captcha"
    assert gate.verify("u1", "wrong")["state"] == "pending_captcha"
    assert gate.verify("u1", "correct")["state"] == "admitted"
    assert gate.is_admitted("u1")
    # Identity boundary enforced at the gate too.
    with pytest.raises(IdentityError):
        gate.request_join("u2", identity={"owner_real_phone": "x"})


def test_moderation_flood_link_repeated():
    mod = Moderator(config=ModerationConfig(
        flood_window_s=60.0, flood_max=3,
        link_allowlist=["example.com"], repeat_threshold=3,
    ))
    # Flood: 4th message inside the window mutes.
    assert mod.check_message("g1", "u1", "hi", ts=1000.0) == []
    assert mod.check_message("g1", "u1", "hi2", ts=1001.0) == []
    assert mod.check_message("g1", "u1", "hi3", ts=1002.0) == []
    flood = mod.check_message("g1", "u1", "hi4", ts=1003.0)
    assert flood and flood[0].kind == "mute" and flood[0].rule == "flood"
    # Link allowlist: allowed domain passes, other domain deleted.
    assert mod.check_message("g1", "u2", "see https://example.com/x", ts=2000.0) == []
    bad = mod.check_message("g1", "u2", "see https://evil.example.net/x", ts=2001.0)
    assert bad and bad[0].kind == "delete" and bad[0].rule == "link_allowlist"
    # Repeated: warn first, then delete. Timestamps spaced > flood
    # window (60s) apart but inside the repeat window (300s), so the
    # repeat rule — not flood — fires.
    assert mod.check_message("g1", "u3", "buy now", ts=3000.0) == []
    assert mod.check_message("g1", "u3", "BUY  NOW", ts=3070.0) == []
    third = mod.check_message("g1", "u3", "buy now", ts=3140.0)
    assert third and third[0].kind == "warn"
    fourth = mod.check_message("g1", "u3", "buy now", ts=3210.0)
    assert fourth and fourth[0].kind == "delete"


def test_moderation_cross_group_ban_and_appeal_path():
    bans = BanList()
    mod = Moderator(ban_list=bans)
    # Group-scoped ban does not leak into another group…
    ban = mod.ban_member("g1", "spammer", reason="spam", scope="group")
    assert bans.is_banned("spammer", "g1") is not None
    assert bans.is_banned("spammer", "g2") is None
    # …but a global ban applies everywhere.
    mod.ban_member("", "spammer", reason="cross-group spam", scope="global")
    assert bans.is_banned("spammer", "g2") is not None
    blocked = mod.check_message("g2", "spammer", "hello")
    assert blocked and blocked[0].kind == "ban"
    # Appeal path exists: file → review → approved lifts the global ban.
    global_ban = bans.is_banned("spammer", "g2")
    appeal = mod.appeal("spammer", global_ban.id, "that was my sibling")
    assert appeal.status == "open"
    assert bans.get_appeal(appeal.id) is not None
    reviewed = bans.review_appeal(appeal.id, approve=True, reviewer="admin1")
    assert reviewed.status == "approved"
    # Global ban lifted; the g1 group ban still stands.
    assert bans.is_banned("spammer", "g2") is None
    assert bans.is_banned("spammer", "g1") is not None
    # Rejected appeal keeps the ban.
    appeal2 = mod.appeal("spammer", ban.id, "please")
    bans.review_appeal(appeal2.id, approve=False, reviewer="admin1")
    assert bans.is_banned("spammer", "g1") is not None
    assert bans.appeals(status="open") == []


def test_roster_intelligence_buckets():
    roster = Roster()
    now = 1_800_000_000.0
    roster.record_join("active1", ts=now - 1000)
    roster.record_message("active1", ts=now - 100)
    roster.record_join("silent1", ts=now - 30 * 86400)
    roster.record_join("new1", ts=now - 1000)
    roster.record_join("left1", ts=now - 5000)
    roster.record_leave("left1", ts=now - 100)
    roster.record_join("answerer", ts=now - 1000)
    roster.record_message("answerer", ts=now - 100, reply_to_question=True)
    report = roster.report(now)
    assert report["buckets"]["active"] == ["active1", "answerer"]
    assert report["buckets"]["silent"] == ["silent1"]
    assert report["buckets"]["new"] == ["new1"]
    assert report["buckets"]["left"] == ["left1"]
    assert report["buckets"]["answering"] == ["answerer"]
    assert report["top_answerers"][0]["user_id"] == "answerer"


# ── digest ─────────────────────────────────────────────────────────────

def _msgs():
    base = 1_800_000_000.0
    return [
        {"id": "q1", "sender": "ada", "text": "Should we move the meetup to Saturday?",
         "ts": base + 1, "reply_to": ""},
        {"id": "q2", "sender": "bob", "text": "What time works for everyone?",
         "ts": base + 2, "reply_to": ""},
        {"id": "a2", "sender": "ada", "text": "Saturday 10am works ✅",
         "ts": base + 3, "reply_to": "q2"},
        {"id": "c1", "sender": "cy", "text": "lol nice photo", "ts": base + 4, "reply_to": ""},
        {"id": "m1", "sender": "ada", "text": "@owner what do you think about the venue?",
         "ts": base + 5, "reply_to": ""},
        {"id": "c2", "sender": "bob", "text": "random chatter about lunch",
         "ts": base + 6, "reply_to": ""},
    ]


def test_digest_picks_decisions_and_unanswered_not_random():
    msgs = _msgs()
    d1 = build_digest(msgs, group_name="Lagos Devs", me="owner")
    d2 = build_digest(list(reversed(msgs)), group_name="Lagos Devs", me="owner")
    # Order-independent: same input set → same digest.
    assert [m["id"] for m in d1.decisions] == [m["id"] for m in d2.decisions]
    assert [m["id"] for m in d1.unanswered] == [m["id"] for m in d2.unanswered]
    # q1 unanswered; q2 answered in-thread (resolved); chatter excluded.
    assert {m["id"] for m in d1.unanswered} == {"q1", "m1"}
    assert {m["id"] for m in d1.decisions} == {"q1", "m1"}
    # Resolved lists *decision* messages that got settled (q2); the
    # answer itself (a2) is not a decision, so it is not listed.
    assert {m["id"] for m in d1.resolved} == {"q2"}
    assert {m["id"] for m in d1.mentions} == {"m1"}
    assert d1.needs_attention is True
    text = render_digest(d1)
    assert "While you were away" in text
    assert "move the meetup" in text
    assert "random chatter" not in text
    # All quiet → no nagging.
    quiet = build_digest(
        [{"id": "x", "sender": "a", "text": "nice day", "ts": 1_800_000_000.0}],
        me="owner",
    )
    assert quiet.needs_attention is False
    assert "all quiet" in render_digest(quiet)


def test_digest_top_20_limit():
    base = 1_800_000_000.0
    msgs = [
        {"id": f"q{i}", "sender": "u", "text": f"Should we do option {i}?",
         "ts": base + i, "reply_to": ""}
        for i in range(25)
    ]
    d = build_digest(msgs)
    assert len(d.decisions) == 20
    assert len(d.unanswered) == 20


def test_recurrence_rrule_and_rituals():
    daily = parse_rrule("FREQ=DAILY;BYHOUR=7;BYMINUTE=0")
    assert daily.freq == "DAILY" and daily.byhour == (7,)
    after = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)  # Wednesday
    nxt = next_occurrences(daily, after, 2, start=after)
    assert nxt[0] == datetime(2026, 10, 15, 7, 0, tzinfo=timezone.utc)
    friday = parse_rrule("FREQ=WEEKLY;BYDAY=FR;BYHOUR=16;BYMINUTE=0")
    nxt_fri = next_occurrences(friday, after, 1, start=after)
    assert nxt_fri[0] == datetime(2026, 10, 16, 16, 0, tzinfo=timezone.utc)
    monthly = parse_rrule("FREQ=MONTHLY;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0")
    nxt_m = next_occurrences(monthly, after, 1, start=after)
    assert nxt_m[0] == datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
    # Round-trip: descriptor → RRULE → descriptor.
    assert parse_rrule(daily.to_rrule()).to_rrule() == daily.to_rrule()
    kinds = {r.kind for r in DEFAULT_RITUALS}
    assert {"daily_verse", "market_open", "good_morning", "friday_vibes", "weekly_poll"} <= kinds
    verse = next(r for r in DEFAULT_RITUALS if r.kind == "daily_verse")
    assert "verse" in render_ritual(verse, after).lower()


def test_ritual_scheduler_due_and_delivered():
    now = datetime(2026, 10, 16, 7, 0, tzinfo=timezone.utc)  # Friday 07:00
    ritual = Ritual(
        name="Good morning", kind="good_morning",
        rrule="FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
        template="Good morning {date}",
    )
    sched = RitualScheduler(rituals=[ritual])
    assert sched.due(now) == [ritual]
    sched.mark_delivered(ritual, now)
    assert sched.due(now) == []  # delivered → not due again for this occurrence
    # Disabled rituals never fire.
    ritual.enabled = False
    assert sched.due(now + __import__("datetime").timedelta(days=1)) == []
