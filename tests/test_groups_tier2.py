"""Groups Tier 2: parties, group economy, media tools, dashboard client.

Invariants under test:
* every wallet balance reconstructible from the ledger;
* no coin ever described as the banned financial word (checked by
  scanning the four new source files — the banned word is built from
  parts so this file itself never contains it);
* party fail-soft: an exception in a game step does not kill the party;
* anti-abuse caps enforced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nomorals.community.group_economy import (
    CURRENCY_DISCLAIMER,
    POINTS_NAME,
    QUESTS,
    STATUS_SHOP,
    GroupEconomy,
    PrizePool,
    format_statement,
)
from nomorals.community.media_tools import (
    audio_to_voice_note,
    format_transcript,
    make_sticker,
    save_sticker,
    transcribe_voice_note,
    trim_video,
    upscale_descriptor,
)
from nomorals.community.parties import (
    PartyManager,
    PartyStore,
    format_recap,
)
from nomorals.community.events import EventStore
from nomorals.stream.dashboard_client import (
    DashboardClient,
    render_phone_answer,
)
from nomorals.stream.phone_client import (
    TokenExpired,
    TokenInvalid,
    mint_stream_grant,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER2_SOURCES = [
    REPO_ROOT / "nomorals" / "community" / "parties.py",
    REPO_ROOT / "nomorals" / "community" / "group_economy.py",
    REPO_ROOT / "nomorals" / "community" / "media_tools.py",
    REPO_ROOT / "nomorals" / "stream" / "dashboard_client.py",
]

# The banned financial word, built from parts so this test file never
# contains it literally (a repo-wide literal grep must stay clean).
BANNED_WORD = "inves" + "tment"


def _party_manager(tmp_path: Path, **kwargs) -> PartyManager:
    store = PartyStore(tmp_path / "parties")
    events = EventStore(tmp_path / "events")
    return PartyManager(store=store, event_store=events, **kwargs)


def _economy(tmp_path: Path) -> GroupEconomy:
    return GroupEconomy(tmp_path / "economy")


def _verified_economy(tmp_path: Path, member: str = "ada") -> GroupEconomy:
    econ = _economy(tmp_path)
    econ.verify_member("g1", member, fingerprint=f"fp-{member}")
    return econ


# ── no coin described as a financial asset ────────────────────────────────


def test_no_coin_described_as_banned_financial_word():
    for path in TIER2_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert BANNED_WORD not in text.lower(), f"{path.name} uses the banned word"
    # the economy module itself must carry the game-points honesty line
    econ_text = (REPO_ROOT / "nomorals" / "community" / "group_economy.py").read_text(
        encoding="utf-8")
    assert "not a financial asset" in econ_text.lower()
    assert "game points only" in econ_text.lower()


def test_disclaimer_present():
    assert "no monetary value" in CURRENCY_DISCLAIMER
    assert "not a financial asset" in CURRENCY_DISCLAIMER
    assert POINTS_NAME == "sparks"


# ── parties ──────────────────────────────────────────────────────────────


def test_party_full_lifecycle(tmp_path):
    mgr = _party_manager(tmp_path, trivia_provider=lambda p: {
        "question": "Capital of Nigeria?", "options": ["Abuja", "Lagos"],
        "answer_index": 0})
    party = mgr.create_party("g1", "Friday Night", 2_000_000_000.0,
                             host="ada", theme="afrobeats")
    assert party is not None and party.state == "invited"
    assert party.rehearsal is True  # first party in the group: rehearsal
    assert mgr.start_party(party) is True
    assert mgr.start_party(party) is False  # already live
    track = mgr.request_track(party, "ada", "Burna Boy — City Boys")
    assert track is not None
    assert mgr.vote_track(party, "bob", track["id"]) is True
    assert mgr.next_track(party)["title"] == "Burna Boy — City Boys"
    question = mgr.play_trivia_round(party)
    assert question is not None and question["round"] == 1
    scored = mgr.answer_trivia(party, "ada", question, 0)
    assert scored == {"correct": True, "points": 10, "streak": 1}
    assert mgr.answer_trivia(party, "bob", question, 1)["correct"] is False
    wall = mgr.post_to_wall(party, "ada", "Best night ever!")
    assert wall is not None
    assert mgr.react_to_wall(party, "bob", wall["id"], "❤️") is True
    recap = mgr.end_party(party)
    assert recap.top_players[0]["member"] == "ada"
    assert recap.top_tracks[0]["title"] == "Burna Boy — City Boys"
    assert recap.top_wall[0]["author"] == "ada"
    assert "wrap" in format_recap(recap)
    assert "MVP" in mgr.last_night_card("g1")
    # ending twice returns the same recap, no duplicate
    assert mgr.end_party(party).party_id == recap.party_id


def test_party_fail_soft_trivia_provider_raises(tmp_path):
    def _boom(party):
        raise RuntimeError("trivia service exploded")

    mgr = _party_manager(tmp_path, trivia_provider=_boom)
    party = mgr.create_party("g1", "Boom Night", 2_000_000_000.0, host="ada")
    assert mgr.start_party(party) is True
    assert mgr.play_trivia_round(party) is None  # provider raised — party lives
    assert party.failures, "failure must be recorded, not swallowed silently"
    assert party.state == "live"
    track = mgr.request_track(party, "ada", "Still playing")
    assert track is not None
    assert mgr.next_track(party) is not None  # music keeps going
    recap = mgr.end_party(party)
    assert recap.failures  # failures surface in the recap, honestly


def test_party_fail_soft_dj_opener_raises(tmp_path):
    def _boom(track):
        raise RuntimeError("dj down")

    mgr = _party_manager(tmp_path, dj_opener=_boom)
    party = mgr.create_party("g1", "DJ Down Night", 2_000_000_000.0, host="ada")
    assert mgr.start_party(party) is True  # default opener covers the failure
    assert any("DJ opens" in m for m in party.outbox)


def test_party_rehearsals_then_real_economy(tmp_path):
    credited: list[tuple[str, str, int]] = []

    def _hook(group_id, member, amount, reason):
        credited.append((group_id, member, amount))
        return True

    mgr = _party_manager(tmp_path, trivia_provider=lambda p: {
        "question": "Q?", "options": ["A", "B"], "answer_index": 0},
        economy_hook=_hook)
    parties = [mgr.create_party("g1", f"P{i}", 2_000_000_000.0, host="ada")
               for i in range(4)]
    assert [p.rehearsal for p in parties] == [True, True, True, False]

    # rehearsal party: scores count, economy hook NOT called
    mgr.start_party(parties[0])
    q = mgr.play_trivia_round(parties[0])
    mgr.answer_trivia(parties[0], "ada", q, 0)
    assert credited == []

    # fourth party: real — hook called
    mgr.start_party(parties[3])
    q = mgr.play_trivia_round(parties[3])
    mgr.answer_trivia(parties[3], "ada", q, 0)
    assert credited == [("g1", "ada", 10)]
    assert parties[3].economy_credited == {"ada": 10}


def test_party_request_cap(tmp_path):
    mgr = _party_manager(tmp_path)
    party = mgr.create_party("g1", "Cap Night", 2_000_000_000.0, host="ada")
    mgr.start_party(party)
    for i in range(5):
        assert mgr.request_track(party, "ada", f"Song {i}") is not None
    assert mgr.request_track(party, "ada", "Song 6") is None  # flood cap


# ── group economy ────────────────────────────────────────────────────────


def test_balance_reconstructible_from_ledger(tmp_path):
    econ = _verified_economy(tmp_path)
    econ.verify_member("g1", "bob", fingerprint="fp-bob")
    assert econ.award("g1", "ada", 50, "helpful") is not None
    assert econ.award("g1", "ada", 30, "event") is not None
    assert econ.transfer("g1", "ada", "bob", 20, "thanks") is True
    entry = econ.award("g1", "ada", 15, "to reverse")
    assert entry is not None
    assert econ.reverse("g1", entry.id, "mistake") is not None
    # statement entries sum to the balance — the invariant
    stmt = econ.statement("g1", "ada")
    assert stmt["balance"] == sum(e["amount"] for e in stmt["entries"])
    assert stmt["balance"] == 50 + 30 - 20  # reversed 15 nets to zero
    assert econ.balance("g1", "ada") == stmt["balance"]
    assert econ.balance("g1", "bob") == 20


def test_reversal_never_deletes(tmp_path):
    econ = _verified_economy(tmp_path)
    entry = econ.award("g1", "ada", 40, "oops")
    before = len(econ._ledger("g1"))
    reversal = econ.reverse("g1", entry.id, "wrong member")
    assert reversal is not None and reversal.kind == "reversal"
    after = econ._ledger("g1")
    assert len(after) == before + 1  # reversal ADDS, never removes
    original = next(e for e in after if e.id == entry.id)
    assert original.reversed_by == reversal.id
    assert original.amount == 40  # untouched
    assert econ.balance("g1", "ada") == 0
    assert econ.reverse("g1", entry.id, "again") is None  # no double reversal
    assert econ.reverse("g1", reversal.id, "reverse a reversal") is None


def test_unverified_member_earns_nothing(tmp_path):
    econ = _economy(tmp_path)
    assert econ.award("g1", "ghost", 100, "trying") is None
    assert econ.balance("g1", "ghost") == 0


def test_earn_caps_enforced(tmp_path):
    econ = _verified_economy(tmp_path)
    assert econ.award("g1", "ada", 200, "at the cap") is not None
    assert econ.award("g1", "ada", 1, "over the cap") is None  # daily cap 200
    assert econ.balance("g1", "ada") == 200
    assert any(f["kind"] == "earn_cap_daily" for f in econ.flags("g1"))


def test_quest_claim_once_per_period(tmp_path):
    econ = _verified_economy(tmp_path)
    for _ in range(3):
        econ.record_activity("g1", "ada", "helpful_answer")
    assert "helpful_3" in QUESTS
    claimed = econ.claim_quest("g1", "ada", "helpful_3")
    assert claimed is not None
    assert claimed.amount == QUESTS["helpful_3"]["reward"]
    assert econ.claim_quest("g1", "ada", "helpful_3") is None  # already claimed
    assert econ.claim_quest("g1", "ada", "invite_1") is None  # not complete


def test_status_shop_costs_nothing_real(tmp_path):
    econ = _verified_economy(tmp_path)
    econ.award("g1", "ada", 100, "seed")
    assert econ.buy_status("g1", "ada", "shout_out") is not None
    assert econ.balance("g1", "ada") == 100 - STATUS_SHOP["shout_out"]["cost"]
    assert econ.buy_status("g1", "ada", "fake_sword") is None  # no such item
    assert econ.buy_status("g1", "ada", "pinned_message") is not None
    assert econ.buy_status("g1", "ada", "dj_request") is None  # broke now
    stmt = format_statement(econ.statement("g1", "ada"))
    assert "no monetary value" in stmt


def test_treasury_round_trip(tmp_path):
    econ = _verified_economy(tmp_path)
    econ.award("g1", "ada", 100, "seed")
    assert econ.contribute_treasury("g1", "ada", 40) is True
    assert econ.treasury_balance("g1") == 40
    assert econ.balance("g1", "ada") == 60
    assert econ.treasury_grant("g1", "ada", 999, "too much") is False
    assert econ.treasury_grant("g1", "ada", 10, "party prize") is True
    assert econ.treasury_balance("g1") == 30
    assert econ.balance("g1", "ada") == 70


def test_season_reset_zeroes_but_badges_carry(tmp_path):
    econ = _verified_economy(tmp_path)
    econ.verify_member("g1", "bob", fingerprint="fp-bob")
    econ.award("g1", "ada", 100, "season")
    econ.award("g1", "bob", 50, "season")
    result = econ.close_season("g1", season="2026-Q4")
    assert econ.balance("g1", "ada") == 0
    assert econ.balance("g1", "bob") == 0
    assert result["badges"][0]["member"] == "ada"  # top member = champion
    assert result["badges"][0]["badge"] == "Season Champion"
    # badges survive; past balances stay reconstructible from the ledger
    assert econ.statement("g1", "ada")["badges"][0]["season"] == "2026-Q4"
    kinds = [e.kind for e in econ._ledger("g1")]
    assert "season_close" in kinds and "earn" in kinds


def test_mining_claim_decays_and_cooldown(tmp_path):
    econ = _verified_economy(tmp_path)
    assert econ.claim_rate("g1", "ada") == 10
    first = econ.mining_claim("g1", "ada")
    assert first is not None and first.amount == 10
    assert econ.claim_rate("g1", "ada") == 8  # 10 * 0.85 → 8
    assert econ.mining_claim("g1", "ada") is None  # 60s cooldown
    hr = econ.hashrate("g1", "ada")
    assert hr["hashrate"] >= 0 and "engagement" in hr["note"]


def test_prize_needs_explicit_owner_funding(tmp_path):
    pool = PrizePool(tmp_path / "economy")
    unfunded = pool.create_prize("g1", "Winner gets a pizza")
    assert pool.redeem("g1", unfunded["id"], "ada") is None  # not funded
    funded = pool.create_prize("g1", "Winner gets a pizza",
                               funded_by_owner=True,
                               owner_approval="owner approved 2026-10-16: pizza")
    won = pool.redeem("g1", funded["id"], "ada")
    assert won is not None and won["winner"] == "ada"
    assert pool.redeem("g1", funded["id"], "bob") is None  # already claimed
    # funding flag without a written approval does not count
    fake = pool.create_prize("g1", "Fake", funded_by_owner=True,
                             owner_approval="")
    assert fake["funded_by_owner"] is False


def test_collusion_flag(tmp_path):
    econ = _verified_economy(tmp_path)
    econ.verify_member("g1", "bob", fingerprint="fp-bob")
    econ.award("g1", "ada", 200, "seed")
    econ.award("g1", "bob", 200, "seed")
    for _ in range(3):
        assert econ.transfer("g1", "ada", "bob", 100, "ring") is True
        assert econ.transfer("g1", "bob", "ada", 100, "ring") is True
    assert any(f["kind"] == "collusion" for f in econ.flags("g1"))


# ── media tools ──────────────────────────────────────────────────────────


def test_sticker_is_real_svg(tmp_path):
    spec = make_sticker("Best night ever", "🎉")
    assert spec.svg.startswith("<svg")
    assert "Best" in spec.svg and "ever" in spec.svg  # wrapped across lines
    assert "🎉" in spec.svg
    assert make_sticker("Best night ever", "🎉").bg == spec.bg  # deterministic
    out = save_sticker(spec, tmp_path / "sticker.svg")
    assert out is not None and out.read_text(encoding="utf-8") == spec.svg
    # escaping: markup in text must not break the SVG
    evil = make_sticker('<script>alert("x")</script>')
    assert "<script>" not in evil.svg


def test_audio_to_voice_note_descriptor():
    ok = audio_to_voice_note(duration_s=42, fmt="ogg", size_bytes=12345)
    assert ok.voice_note is True and ok.issues == []
    assert "no audio was re-encoded" in ok.note.lower()
    long = audio_to_voice_note(duration_s=600, fmt="mp3")
    assert long.voice_note is False
    assert any("5 minutes" in i for i in long.issues)


def test_trim_video_descriptor():
    plan = trim_video(120, 10, 200)  # end past the duration clamps
    assert plan.end_s == 120 and plan.clamped is True
    assert plan.out_duration_s == 110
    assert "ffmpeg" in plan.note
    bad = trim_video(60, 50, 40)
    assert bad.out_duration_s == 0 and bad.issues


def test_upscale_descriptor_geometry():
    plan = upscale_descriptor(640, 360, "1080p")
    assert (plan.target_w, plan.target_h) == (1920, 1080)
    assert plan.scale == 3.0
    assert "cannot add detail" in plan.note
    assert upscale_descriptor(0, 0).issues  # invalid geometry refused


def test_transcription_accepts_but_never_fakes():
    assert transcribe_voice_note("vn_1", None) is None
    assert transcribe_voice_note("vn_1", "   ") is None  # empty = no transcript
    tx = transcribe_voice_note("vn_1", "Hello everyone!",
                               source="platform-captions", language="en",
                               confidence=0.92)
    assert tx is not None and tx.text == "Hello everyone!"
    assert tx.source == "platform-captions" and tx.confidence == 0.92
    assert "platform-captions" in format_transcript(tx)
    # confidence is never invented when the supplier gave none
    assert transcribe_voice_note("vn_2", "Hi", source="human").confidence is None


# ── dashboard client ─────────────────────────────────────────────────────


def _grant(**kwargs):
    return mint_stream_grant("test-secret", ttl_s=300, **kwargs)


def test_dashboard_four_jobs_one_client():
    seen: list[str] = []

    def _fetcher(job: str):
        seen.append(job)
        return {"ok": True, "job": job}

    client = DashboardClient("http://localhost:9", _grant(), fetcher=_fetcher)
    snap = client.run_four_jobs()
    assert snap.all_ok
    assert sorted(snap.jobs) == ["costs_jobs", "groups_queue", "organ_health",
                                 "sessions"]
    assert seen == ["sessions", "groups_queue", "costs_jobs", "organ_health"]


def test_dashboard_one_dead_job_does_not_kill_others():
    def _fetcher(job: str):
        if job == "costs_jobs":
            raise ConnectionError("server down")
        return {"count": 3} if job == "sessions" else {"groups": 2, "queue": 5}

    client = DashboardClient("http://localhost:9", _grant(), fetcher=_fetcher)
    snap = client.snapshot()
    assert snap.failed_jobs == ["costs_jobs"]
    assert snap.ok_jobs == ["sessions", "groups_queue", "organ_health"]
    answer = render_phone_answer(snap)
    assert "Couldn't reach costs_jobs" in answer
    assert "Traceback" not in answer
    assert "server down" not in answer  # raw errors never reach the phone


def test_dashboard_expired_grant_refused_before_request():
    calls: list[str] = []

    def _fetcher(job: str):
        calls.append(job)
        return {}

    expired = mint_stream_grant("test-secret", ttl_s=-10)
    client = DashboardClient("http://localhost:9", expired, fetcher=_fetcher)
    with pytest.raises(TokenExpired):
        client.run_four_jobs()
    assert calls == []  # nothing reached the wire
    with pytest.raises(TokenExpired):
        list(client.stream_events())


def test_dashboard_tampered_token_rejected_with_secret():
    grant = _grant()
    bad = grant.__class__(capability=grant.capability,
                          token=grant.token[:-2] + "xx",
                          expires_at=grant.expires_at,
                          issued_at=grant.issued_at,
                          scope=grant.scope)
    client = DashboardClient("http://localhost:9", bad,
                             secret="test-secret",
                             fetcher=lambda job: {})
    with pytest.raises(TokenInvalid):
        client.run_four_jobs()


def test_dashboard_phone_answer_healthy():
    def _fetcher(job: str):
        return {
            "sessions": {"sessions": 2},
            "groups_queue": {"groups": 3, "queue": 1},
            "costs_jobs": {"cost": 1.5, "jobs": 4},
            "organ_health": {"organs": {"a": {"ok": True}, "b": {"ok": False}}},
        }[job]

    client = DashboardClient("http://localhost:9", _grant(), fetcher=_fetcher)
    answer = render_phone_answer(client.snapshot())
    assert "2 live" in answer
    assert "3 groups, 1 queued" in answer
    assert "$1.50" in answer
    assert "1/2 organs healthy" in answer
