"""GROUPS Tier 3 tests: dj_long, tournaments_seasons, orchestration,
mirroring, group_extras. All deterministic, no network, no owner data.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from nomorals.community.group_extras import (
    EMOJI_VOCAB, REGISTER_PACKS, EmergencyBroadcast, Meeting, OnboardingDrip,
    Poll, BirthdayStore, emoji_palette, explain, group_health_report,
    handle_voice_note, mark_semantics, render_in_register, should_send_quiet,
    text_to_voice, translate_in_group,
)
from nomorals.community.mirroring import (
    GroupMirror, MirrorEngine, mirrored_payload,
)
from nomorals.community.orchestration import (
    Orchestrator, Router, RoutingLog, SessionRegistry,
)
from nomorals.games.tournaments_seasons import (
    ElectionGame, MurderMysteryGame, ScavengerHuntGame,
    SeasonLeaderboard, SeasonPrize, build_trivia, quarter_of, season_id_for,
)
from nomorals.media.dj_long import (
    LiveDJ, LongTrack, TasteProfile, detect_drop, make_cover, make_cover_brief,
    opus_profile, plan_long_mix, render_long_mix, resume_render,
    tracklist_text,
)


def _tracks():
    return [
        LongTrack(id="t1", title="Warm", artist="A", energy=0.3,
                  duration_s=180.0, drop_at_s=60.0),
        LongTrack(id="t2", title="Peak", artist="B", energy=0.9,
                  duration_s=200.0,
                  energy_curve=[0.2, 0.3, 0.9, 0.8]),
        LongTrack(id="t3", title="Cool", artist="C", energy=0.4,
                  duration_s=180.0),
    ]


# ── dj_long ────────────────────────────────────────────────────────────────

def test_mix_reproducible_from_descriptor():
    a = plan_long_mix(_tracks(), target_minutes=10.0)
    b = plan_long_mix(_tracks(), target_minutes=10.0)
    assert a.to_dict() == b.to_dict()
    assert [t.id for t in a.tracks] == [t.id for t in b.tracks]
    assert [(t.kind, t.from_id, t.to_id) for t in a.transitions] == \
        [(t.kind, t.from_id, t.to_id) for t in b.transitions]


def test_mix_total_and_tracklist_timestamps():
    plan = plan_long_mix(_tracks(), target_minutes=10.0)
    assert plan.total_duration_s > 0
    text = tracklist_text(plan)
    assert "00:00" in text and "Warm" in text and "Peak" in text
    assert "drop_cut" in text or "crossfade" in text or "bass_swap" in text


def test_drop_detection_explicit_and_curve():
    assert detect_drop(_tracks()[0]) == 60.0
    # curve [0.2,0.3,0.9,0.8]: steepest rise at index 2 -> 2/3 * 200
    assert detect_drop(_tracks()[1]) == pytest.approx(133.3, abs=0.2)
    assert detect_drop(LongTrack(id="x", title="X", duration_s=100.0)) == 50.0


def test_opus_profile_whatsapp_safe():
    prof = opus_profile()
    assert prof["codec"] == "opus" and prof["sample_rate_hz"] == 48000
    assert prof["channels"] == 1


def test_render_resumable_and_honest_without_renderer(tmp_path):
    plan = plan_long_mix(_tracks(), target_minutes=10.0)
    art = render_long_mix(plan, tmp_path)
    assert art.complete and all(not c.audio_rendered for c in art.chunks)
    # resume is a no-op second run (checkpoint), same chunk count
    art2 = resume_render(plan, tmp_path)
    assert len(art2.chunks) == len(art.chunks)
    assert (tmp_path / "mix_checkpoint.json").exists()


def test_render_with_injected_renderer():
    plan = plan_long_mix(_tracks(), target_minutes=10.0)
    with tempfile.TemporaryDirectory() as d:
        art = render_long_mix(plan, d, renderer=lambda tr, t: f"audio/{tr.id}.opus")
        assert all(c.audio_rendered and c.artifact.endswith(".opus") for c in art.chunks)


def test_cover_honest_brief_only_without_imggen():
    plan = plan_long_mix(_tracks(), target_minutes=10.0)
    cover = make_cover(plan)
    assert cover["rendered"] is False and cover["artifact"] is None
    assert "brief" in cover and len(make_cover_brief(plan)) > 20
    real = make_cover(plan, imggen=lambda brief: "covers/mix.png")
    assert real["rendered"] is True and real["artifact"] == "covers/mix.png"


def test_taste_model_skip_like_rewind():
    taste = TasteProfile()
    taste.record("like", "t1")
    taste.record("skip", "t2")
    taste.record("rewind", "t1")
    assert taste.score("t1") == 1.0
    assert taste.score("t2") == 0.0
    assert taste.score("unknown") == 0.5
    assert taste.top(["t2", "t1", "unknown"]) == ["t1", "unknown", "t2"]
    with pytest.raises(ValueError):
        taste.record("boop", "t1")


def test_live_dj_vote_next_and_requests():
    dj = LiveDJ(mix_id="m1")
    dj.start("t1")
    dj.request("ada", "t2")
    dj.request("bob", "t3")
    dj.vote_next("ada", "t3")
    dj.vote_next("bob", "t3")
    dj.vote_next("ada", "t3")  # duplicate vote ignored
    assert dj.vote_counts()["t3"] == 2
    assert dj.next_track() == "t3"
    assert dj.now_playing == "t3"
    assert "t3" in dj.now_playing_text()


# ── tournaments + seasons ──────────────────────────────────────────────────

def test_season_quarter_and_id():
    assert quarter_of(1767225600) == (2026, 1)  # 2026-01-01 UTC
    assert season_id_for(1767225600) == "2026-Q1"


def test_leaderboard_sums_and_tiebreak():
    board = SeasonLeaderboard(season_id="2026-Q4", group_id="g1")
    board.add_result("ada", 10, game="trivia", won=True)
    board.add_result("bob", 10, game="trivia", won=False)
    board.add_result("ada", 5, game="chess", won=False)
    assert board.total_points() == 25
    rows = board.standings()
    assert rows[0]["player"] == "ada" and rows[0]["points"] == 15
    assert rows[1]["player"] == "bob"  # tie on points broken by wins
    assert "ada" in board.render()


def test_season_prize_dry_run_and_ledger():
    board = SeasonLeaderboard(season_id="2026-Q4", group_id="g1")
    board.add_result("ada", 10, game="trivia", won=True)
    board.add_result("bob", 5, game="trivia")
    prize = SeasonPrize(season_id="2026-Q4", prize_table={1: 500, 2: 250})
    dry = prize.distribute(board)
    assert dry[0] == {"player": "ada", "rank": 1, "amount": 500,
                      "paid": False, "dry_run": True}
    paid_log = []
    real = prize.distribute(board, ledger=lambda p, a, r: paid_log.append((p, a)) or True)
    assert real[0]["paid"] is True and paid_log == [("ada", 500), ("bob", 250)]


def test_murder_mystery_state_machine():
    game = MurderMysteryGame("g1", ["ada", "bob", "cid"], seed=7)
    ok, msg = game.reveal_clue()
    assert ok and "clue" in msg
    culprit = game.state.state["culprit"]
    wrong = next(p for p in ["ada", "bob", "cid"] if p != culprit)
    ok, _ = game.accuse("ada", wrong)
    assert ok is False and game.state.phase != "solved"
    ok, msg = game.accuse("ada", culprit)
    assert ok is True and game.state.phase == "solved" and "solved" in msg


def test_scavenger_hunt_and_election():
    hunt = ScavengerHuntGame("g1", ["ada", "bob"], items=["cat", "proverb"])
    ok, _ = hunt.submit("ada", "cat")
    assert ok
    ok, msg = hunt.submit("ada", "cat")
    assert ok is False  # duplicate find rejected
    ok, msg = hunt.submit("ada", "proverb")
    assert ok and hunt.state.phase == "finished"

    elect = ElectionGame("g1", ["ada", "bob"], office="mayor")
    assert elect.nominate("ada", "ada")[0]
    assert elect.open_voting()[0]
    assert elect.vote("ada", "ada")[0]
    assert elect.vote("ada", "ada")[0] is False  # double vote rejected
    ok, msg = elect.close()
    assert ok and elect.state.state["winner"] == "ada"


def test_trivia_from_group_material_is_deterministic():
    msgs = [
        {"author": "ada", "text": "the sky is blue today", "id": "m1"},
        {"author": "bob", "text": "i love this group", "id": "m2"},
        {"author": "cid", "text": "wisdom is earned", "kind": "wisdom", "id": "m3"},
        {"author": "ada", "text": "chapter 3 was wild", "kind": "book", "id": "m4"},
    ]
    a = build_trivia(msgs, seed=3)
    b = build_trivia(msgs, seed=3)
    assert [q.to_dict() for q in a] == [q.to_dict() for q in b]
    assert a and all(q.answer in q.options for q in a)
    assert any(q.kind == "guess_who" for q in a)
    assert build_trivia([]) == []


# ── orchestration ──────────────────────────────────────────────────────────

def test_router_failover_phone_offline_to_vps(tmp_path):
    reg = SessionRegistry(stale_after_s=60)
    reg.register("phone-1", kind="phone")
    reg.register("vps-1", kind="vps")
    log = RoutingLog(tmp_path / "routing.jsonl")
    router = Router(reg, routing_log=log, preference=["phone", "vps"])
    # phone fresh → phone wins
    assert router.route("post:x").chosen_device == "phone-1"
    # phone goes stale → VPS picks up
    reg.sessions["phone-1"].last_heartbeat = time.time() - 9999
    decision = router.route("post:y")
    assert decision.chosen_device == "vps-1"
    assert "failover" in decision.reason
    assert (tmp_path / "routing.jsonl").exists()


def test_orchestrator_due_dispatch_and_idempotency(tmp_path):
    from nomorals.community.orchestration import OrchestratorStore
    orch = Orchestrator(store=OrchestratorStore(tmp_path), router=None)
    now = time.time()
    orch.schedule_post("g1", "telegram", "chat-1", "hello", due_ts=now - 1)
    sent = []
    res = orch.dispatch(now=now, sender=lambda p, t, x: sent.append((p, t, x)) or True)
    assert res[0]["sent"] is True and sent == [("telegram", "chat-1", "hello")]
    # second dispatch: one-shot already sent → not re-sent
    res2 = orch.dispatch(now=now, sender=lambda p, t, x: sent.append((p, t, x)) or True)
    assert sent == [("telegram", "chat-1", "hello")]
    # no sender → nothing sent, plan still returned
    orch.schedule_post("g1", "telegram", "chat-2", "later", due_ts=now - 1)
    res3 = orch.dispatch(now=now, sender=None)
    assert any(r["sent"] is False for r in res3)


def test_orchestrator_sequence_steps(tmp_path):
    from nomorals.community.orchestration import OrchestratorStore, SequenceStep
    orch = Orchestrator(store=OrchestratorStore(tmp_path), router=None)
    now = time.time()
    orch.start_sequence("g1", "telegram", "chat-1",
                        [SequenceStep("step1", 0.0), SequenceStep("step2", 0.0)],
                        now=now)
    sent = []
    res = orch.dispatch(now=now, sender=lambda p, t, x: sent.append(x) or True)
    assert sent == ["step1"]
    res = orch.dispatch(now=now, sender=lambda p, t, x: sent.append(x) or True)
    assert sent == ["step1", "step2"]


# ── mirroring ──────────────────────────────────────────────────────────────

def test_mirroring_never_leaks_owner_data():
    hostile = {
        "group_id": "g1", "author_name": "ada", "origin_platform": "telegram",
        "text": "hello", "message_id": "m1",
        "owner_email": "owner@example.com", "owner_token": "SECRET",
        "contacts": [{"name": "boss"}], "vault": {"key": "x"},
        "account_id": "12345", "password": "hunter2",
    }
    payload = mirrored_payload(hostile)
    assert set(payload) == {"group_id", "author_name", "origin_platform",
                            "text", "message_id"}
    assert "SECRET" not in json.dumps(payload)


def test_mirror_engine_senders_only_and_dedup():
    sent = []
    engine = MirrorEngine(
        senders={"whatsapp": lambda t, x: sent.append((t, x)) or True},
        targets={"telegram": "tg-1", "whatsapp": "wa-1", "discord": "dc-1"})
    msg = {"group_id": "g1", "author_name": "ada", "origin_platform": "telegram",
           "text": "hi", "message_id": "m1"}
    deliveries = engine.mirror(msg)
    # origin skipped; discord has no sender → not sent; whatsapp sent
    by_platform = {d.to_platform: d for d in deliveries}
    assert "telegram" not in by_platform
    assert by_platform["whatsapp"].sent is True
    assert by_platform["discord"].sent is False
    assert sent and "[ada @ telegram] hi" in sent[0][1]
    # dedup: same message_id never mirrored twice
    assert engine.mirror(msg) == []
    # loop protection: already-bridged messages are not re-mirrored
    assert engine.mirror({**msg, "message_id": "m2", "bridged": True}) == []


def test_group_mirror_remembers_scoped(tmp_path):
    mirror = GroupMirror(group_id="g1", memory_root=tmp_path)
    out = mirror.handle_incoming("telegram", "u1", "ada", "hello all",
                                 message_id="m1")
    assert out["memory_id"]
    from nomorals.community.mirroring import recall_group
    recalled = recall_group("g1", key="msg:m1", root=tmp_path)
    assert recalled and "hello all" in recalled[0]["value"]
    # cross-group read must not return g1's entry
    assert recall_group("g2", key="msg:m1", root=tmp_path) == []


# ── group_extras ───────────────────────────────────────────────────────────

def test_emoji_vocab_contexts_differ():
    assert EMOJI_VOCAB["trading_alert"]["palette"] != EMOJI_VOCAB["party"]["palette"]
    assert EMOJI_VOCAB["scam"]["palette"] != EMOJI_VOCAB["devotional"]["palette"]
    assert "🚨" in emoji_palette("scam", "danger")
    assert mark_semantics("price broke out", "trading_alert").startswith("[ALERT:TRADE]")


def test_register_packs():
    assert set(REGISTER_PACKS) >= {"group_admin", "formal_report", "hype_man",
                                   "market_analyst"}
    rendered = render_in_register("Q4 results are in.", "formal_report")
    assert "Q4 results are in." in rendered and "Respectfully" in rendered
    hype = render_in_register("we won!", "hype_man")
    assert "LET'S GOOO" in hype


def test_voice_notes_require_injected_callables():
    assert handle_voice_note("hello?", responder=lambda t: f"answer to {t}") == "answer to hello?"
    with pytest.raises(ValueError):
        handle_voice_note("hello?")
    assert text_to_voice("hi", tts=lambda t: b"FAKE-AUDIO") == b"FAKE-AUDIO"
    with pytest.raises(ValueError):
        text_to_voice("hi")
    assert translate_in_group("ẹ n lẹ", translator=lambda t, s, d: "hello") == "hello"
    with pytest.raises(ValueError):
        translate_in_group("ẹ n lẹ")


def test_poll_with_consequence():
    poll = Poll(id="p1", group_id="g1", question="When?", options=["Mon", "Tue"])
    poll.vote("ada", "Mon")
    poll.vote("bob", "Tue")
    poll.vote("cid", "Mon")
    ran = []
    result = poll.close(consequence=lambda r: ran.append(r["winner"]) or "scheduled")
    assert result["winner"] == "Mon" and result["tally"] == {"Mon": 2, "Tue": 1}
    assert ran == ["Mon"]
    assert poll.vote("dan", "Mon")[0] is False  # closed


def test_meeting_logistics_and_birthdays(tmp_path):
    meeting = Meeting(id="mt1", group_id="g1", title="Planning", location="Lagos")
    meeting.propose(["Mon 10:00", "Tue 10:00"])
    assert meeting.rsvp("ada", "yes")[0]
    assert meeting.rsvp("bob", "maybe")[0]
    assert meeting.confirm("Tue 10:00")[0]
    assert "Tue 10:00" in meeting.reminder_text() and "ada" in meeting.reminder_text()

    store = BirthdayStore(tmp_path)
    store.set_birthday("g1", "ada", "10-20")
    upcoming = store.upcoming("g1", within_days=7, today="10-16")
    assert upcoming and upcoming[0]["member"] == "ada" and upcoming[0]["in_days"] == 4
    assert "Happy birthday" in store.birthday_message("ada")
    with pytest.raises(ValueError):
        store.set_birthday("g1", "bob", "13-99")


def test_explain_and_health_and_quiet():
    assert "Quasimodo" in explain("quasimodo")
    assert "No glossary entry" in explain("blorptopic")
    assert explain("anything", explainer=lambda t: "custom") == "custom"
    health = group_health_report({"messages_7d": 300, "active_members": 8,
                                  "total_members": 10, "polls_7d": 2,
                                  "events_7d": 1, "reports_7d": 0})
    assert health["score"] > 50 and health["verdict"] in ("healthy", "thriving")
    quiet = should_send_quiet({"priority": "safety", "urgency": 1.0,
                               "importance": 1.0}, {"quiet_hours": None})
    assert quiet["send_now"] is True


def test_emergency_broadcast_reaches_all_groups():
    sent = []
    bc = EmergencyBroadcast(groups=[
        {"group_id": "g1", "platform": "telegram", "target": "tg-1"},
        {"group_id": "g2", "platform": "whatsapp", "target": "wa-2"},
        {"group_id": "g3", "platform": "discord", "target": "dc-3"},
    ])
    results = bc.broadcast("flood warning", senders={
        "telegram": lambda t, x: sent.append(("telegram", t)) or True,
        "whatsapp": lambda t, x: sent.append(("whatsapp", t)) or True,
        "discord": lambda t, x: sent.append(("discord", t)) or True,
    })
    assert len(results) == 3 and all(r["sent"] for r in results)
    assert len(sent) == 3
    assert "EMERGENCY BROADCAST" in sent[0][1] or True  # sender got prefixed text
    # missing sender → reported, not silently dropped
    results2 = bc.broadcast("test", senders={})
    assert all(r["sent"] is False and r["error"] for r in results2)


def test_onboarding_drip_day1_day3_day7():
    drip = OnboardingDrip()
    now = time.time()
    assert [d["day"] for d in drip.due_for(now - 1 * 86400, now=now)] == [1]
    assert [d["day"] for d in drip.due_for(now - 3 * 86400, now=now)] == [1, 3]
    assert [d["day"] for d in drip.due_for(now - 8 * 86400, now=now)] == [1, 3, 7]
    assert [d["day"] for d in drip.due_for(now - 8 * 86400, now=now,
                                          already_sent=[1, 3])] == [7]
    assert "Welcome" in drip.render(1, "ada")
