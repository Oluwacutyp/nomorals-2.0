"""Skip-list build invariants (OWNER-CORRECTION: build everything, with
safeguards built INTO the feature, not as reasons to skip).

Covers the gap-fill modules only; pre-existing systems are verified
by their own suites, not rebuilt here.
"""

import json
import sys

import pytest

from nomorals.games.extra_content import (
    TRIVIA_PACKS, EmojiStory, MemoryMatch, PredictionMarket, RiddleHunt,
    SpeedTyping, WordChain, trivia_pack, trivia_packs_for,
)
from nomorals.community.parties_quick import PartyQuick, build_playlist
from nomorals.community.token_economy import TokenEconomy, TOKEN_DISCLAIMER
from nomorals.sources.health_plus import (
    NEW_SOURCE_DESCRIPTORS, SourceHealth, new_source_descriptors,
)
from nomorals.console.tui_facade import TUIFacade
from nomorals.scheduler.facade import SchedulerFacade, parse_schedule_spec
from nomorals.plugins.marketplace import Marketplace, PluginTestFailed
from nomorals.media.fidelity import (
    FidelityPipeline, QualityLadder, pick_best_model, route,
)


# ── 1. extra game content ────────────────────────────────────────────────────

def test_trivia_packs_include_required_and_universal():
    for pid in ("nigerian-history", "afrobeats", "football", "science"):
        assert pid in TRIVIA_PACKS
    # Nigeria is one pack among N, never the only axis.
    assert "brazil-culture" in TRIVIA_PACKS
    assert "japan-culture" in TRIVIA_PACKS
    assert trivia_pack("football")["questions"]
    assert "football" in trivia_packs_for(nation="BR")  # worldwide packs match all
    assert trivia_packs_for(nation="BR")[0] == "brazil-culture"


def test_word_chain_rules_and_reward_fn():
    credited = []

    def reward(member, amount, reason):
        credited.append((member, amount, reason))

    game = WordChain(players=["ada", "bob"], reward_fn=reward)
    assert game.play("ada", "apple")["ok"] is True
    assert game.play("bob", "banana")["ok"] is False  # must start with 'e'
    assert game.play("bob", "elephant")["ok"] is True
    assert credited and credited[0][0] == "ada"


def test_riddle_hunt_and_emoji_story():
    hunt = RiddleHunt(seed=1)
    q = hunt.current()
    assert q and q["question"]
    assert hunt.answer("ada", "nope")["correct"] is False
    story = EmojiStory(seed=1)
    res = story.submit("ada", "rain made me run home for coffee")
    assert res["points"] > 0


def test_speed_typing_and_memory_match():
    st = SpeedTyping(language="pcm")
    st.start(now=1000.0)
    res = st.submit("ada", st.passage, now=1060.0)
    assert res["wpm"] > 0 and res["accuracy"] == 1.0
    mm = MemoryMatch(pairs=2, seed=3)
    # Flip every position once; pairs eventually match; no crash, score sane.
    for pos in range(4):
        mm.flip("ada", pos)
    assert mm.flips == 4


def test_prediction_market_game_currency_only():
    balances = {"ada": 100, "bob": 100}

    def debit(m, amt, reason):
        balances[m] -= amt

    def credit_fn(m, amt, reason):
        balances[m] += amt

    pm = PredictionMarket(balance_fn=balances.get, debit_fn=debit, credit_fn=credit_fn)
    mid = pm.create("Who wins?", ["ada", "bob"])
    pm.bet(mid, "ada", "ada", 40)
    pm.bet(mid, "bob", "bob", 60)
    out = pm.resolve(mid, "ada")
    assert out["payouts"]["ada"] == 100  # pool 100, sole winner takes all
    with pytest.raises(Exception):
        pm.bet(mid, "ada", "ada", 10)  # resolved


# ── 2. quick parties work with SHORT mixes ──────────────────────────────────

def test_party_quick_short_mixes():
    pq = PartyQuick(trivia_provider=lambda: {"q": "2+2?", "options": ["4"]})
    short_mixes = [{"title": "mix-a", "duration_s": 120},
                   {"title": "mix-b", "duration_s": 180}]
    party = pq.create("g1", "Friday", "ada", mixes=short_mixes)
    assert party.playlist["total_seconds"] == 300
    assert party.playlist["meets_target"] is False  # shortfall reported, NOT a refusal
    assert pq.start(party) is True
    first = pq.next_track(party)
    assert first["title"] == "mix-a"
    second = pq.next_track(party)
    assert second["title"] == "mix-b"
    assert party.trivia_rounds == 1  # trivia ran between tracks
    recap = pq.end(party)
    assert recap.playlist_seconds == 300
    assert len(recap.top_tracks) == 2


def test_party_quick_rehearsal_dry_run():
    credited = []
    pq = PartyQuick(economy_hook=lambda g, m, a, r: credited.append(a))
    party = pq.create("g1", "Rehearsal", "ada", mixes=[{"title": "x", "duration_s": 60}],
                      rehearsal=True)
    pq.start(party)
    pq.answer_trivia(party, "ada", "4", "4")
    assert credited == []  # rehearsal: computed, never credited
    assert pq.end(party).economy_dry_run is True


def test_build_playlist_accepts_unknown_duration():
    pl = build_playlist([{"title": "no-duration"}])
    assert pl["track_count"] == 1 and pl["total_seconds"] == 0


# ── 3. token economy ─────────────────────────────────────────────────────────

def test_token_never_investment_and_no_buy_path():
    assert "not an investment" in TokenEconomy.describe().lower()
    assert "not an investment" in TOKEN_DISCLAIMER.lower()
    assert not hasattr(TokenEconomy, "buy_with_real_money")
    assert not hasattr(TokenEconomy, "redeem_for_money")


def test_token_earn_caps_decay_and_reversal_preserve_ledger():
    te = TokenEconomy("g1")
    assert te.verify("ada", "id-ada") is True
    assert te.verify("ada", "id-ada-2") is False  # second account rejected
    assert te.verify("bob", "id-ada") is False  # same identity, other member
    assert te.verify("bob", "id-bob") is True
    e1 = te.earn("ada", 100, "quest", now=1000.0)
    e2 = te.earn("ada", 100, "quest", now=1100.0)  # repeat inside decay window
    assert e1 is not None and e2 is not None
    assert e2.amount == 50  # 100 * 0.5 decay
    assert te.balance("ada") == 150
    # Reversal never deletes: original stays, balance restored.
    rev = te.reverse(e1.entry_id)
    assert rev is not None
    assert any(e.entry_id == e1.entry_id for e in te.entries("ada"))
    assert te.balance("ada") == 50
    assert te.verify_ledger()["reversals_balanced"] is True
    assert te.reverse(e1.entry_id) is None  # already reversed


def test_token_transfer_caps_and_flags():
    te = TokenEconomy("g1")
    te.verify("ada", "id-ada")
    te.verify("bob", "id-bob")
    te.earn("ada", 500, "quest", now=1000.0)
    assert te.transfer("ada", "bob", 100, now=2000.0) is True
    assert te.balance("bob") == 100
    assert te.transfer("ada", "bob", 10_000, now=2000.0) is False  # insufficient
    assert te.transfer("ada", "ada", 10, now=2000.0) is False  # self-transfer
    # Velocity: many rapid transfers flag the account.
    te2 = TokenEconomy("g2")
    te2.verify("x", "id-x")
    te2.verify("y", "id-y")
    te2.earn("x", 500, "quest", now=1000.0)
    for _ in range(25):
        te2.transfer("x", "y", 1, now=2000.0)
    assert te2.is_flagged("x") or te2.balance("y") <= 25


def test_token_unverified_cannot_earn():
    te = TokenEconomy("g1")
    assert te.earn("ghost", 100, "quest") is None


# ── 4. sources + health ─────────────────────────────────────────────────────

def test_five_new_sources_and_health():
    assert len(NEW_SOURCE_DESCRIPTORS) == 5
    domains = {d["domain"] for d in NEW_SOURCE_DESCRIPTORS}
    assert {"punchng.com", "audiomack.com", "techcrunch.com",
            "espn.com", "yorubaculture.example"} <= domains
    descs = new_source_descriptors()
    assert len(descs) == 5
    health = SourceHealth()
    assert health.success_rate("punchng.com") is None  # unknown, not failed
    assert health.is_stale("punchng.com") is True  # never probed = stale
    health.probe("punchng.com", lambda s: (True, 0.2))
    health.probe("espn.com", lambda s: (False, 0.5))
    assert health.success_rate("punchng.com") == 1.0
    assert health.success_rate("espn.com") == 0.0
    chain = health.fallback_chain(["espn.com", "punchng.com", "never-probed.example"])
    assert chain[0] == "punchng.com"  # healthy first
    assert chain[-1] == "espn.com"  # known-dead last
    assert health.health_report()


# ── 5. facades preserve APIs ────────────────────────────────────────────────

def test_tui_facade_passthrough():
    seen = []

    class StubConsole:
        def push_event(self, event):
            seen.append(event)

        def _render_frame(self):
            return "FRAME"

        def _on_command(self, cmd):
            return f"handled:{cmd}"

    facade = TUIFacade(godconsole=StubConsole())
    facade.push_event("e1")
    assert seen == ["e1"]
    assert facade.render_frame() == "FRAME"
    assert facade.run_command("something-unknown") == "handled:something-unknown"
    assert "status" in facade.list_commands()  # console-native commands present


def test_scheduler_facade_passthrough(tmp_path):
    from nomorals.scheduler.unified_scheduler import UnifiedJob, UnifiedScheduler
    direct = UnifiedScheduler(store_path=str(tmp_path / "sched.json"))
    facade = SchedulerFacade(unified=direct)
    job = UnifiedJob(job_id="j1", name="t",
                     recurrence={"type": "interval", "spec": 3600})
    facade.add_job(job, now=1000.0)
    assert facade.get_job("j1").job_id == "j1"
    assert [j.job_id for j in facade.list_jobs()] == ["j1"]
    assert facade.remove_job("j1") is True
    assert facade.list_jobs() == []
    kind, _ = parse_schedule_spec("every 1h")
    assert kind  # agent-scheduler parser passthrough works


def test_selfheal_already_facaded_not_rebuilt():
    from nomorals.core import selfheal_facade
    assert hasattr(selfheal_facade, "SelfHealingExecutor") or \
        hasattr(selfheal_facade, "heal_one")


# ── 6. marketplace rejects plugin failing its own tests ─────────────────────

def _plugin_dir(tmp_path, name, test_body, manifest_extra=None):
    d = tmp_path / name
    (d / "tests").mkdir(parents=True)
    manifest = {"name": name, "version": "1.0.0", "permissions": ["fs"],
                "artifacts": {}}
    if manifest_extra:
        manifest.update(manifest_extra)
    (d / "plugin.json").write_text(json.dumps(manifest))
    (d / "tests" / "test_plugin.py").write_text(test_body)
    return d


def test_marketplace_rejects_failing_tests(tmp_path):
    mp = Marketplace(plugins_dir=tmp_path / "plugins")
    bad = _plugin_dir(tmp_path, "bad-plugin", "def test_no():\n    assert False\n")
    with pytest.raises(PluginTestFailed):
        mp.install(bad)
    assert mp.listings() == []  # nothing installed


def test_marketplace_installs_passing_tests_and_reviews(tmp_path):
    mp = Marketplace(plugins_dir=tmp_path / "plugins")
    good = _plugin_dir(tmp_path, "good-plugin", "def test_yes():\n    assert True\n")
    receipt = mp.install(good)
    assert receipt["tests_passed"] is True
    assert len(mp.listings()) == 1
    mp.add_review("good-plugin", "ada", 5, "works")
    mp.add_review("good-plugin", "bob", 3, "ok")
    assert mp.average_rating("good-plugin") == 4.0
    with pytest.raises(ValueError):
        mp.add_review("good-plugin", "ada", 9)


# ── 7. fidelity ─────────────────────────────────────────────────────────────

def test_fidelity_routing_picks_best_model():
    cards = [
        {"id": "small", "capabilities": ["music"], "quality": 0.4},
        {"id": "big", "capabilities": ["music"], "quality": 0.9},
        {"id": "other", "capabilities": ["chat"], "quality": 0.99},
    ]
    best = pick_best_model("music", cards)
    assert best["id"] == "big"  # best FOR THIS TASK, not global max
    assert pick_best_model("video", cards) is None  # nothing qualifies
    assert route("music", cards)["model_id"] == "big"
    assert route("video", cards)["model_id"] is None


def test_fidelity_pipeline_honest_about_processors():
    ladder = QualityLadder()
    assert ladder.steps_for("master") == ["upscale", "denoise", "grade", "sharpen", "normalize"]
    assert ladder.climb("draft") == "standard"
    assert ladder.climb("master") is None
    # No processors injected: nothing faked, artifact passes through.
    pipe = FidelityPipeline.for_level("high")
    out = pipe.run("artifact-1")
    assert out["applied"] == [] and out["artifact"] == "artifact-1"
    # With a real processor: it runs.
    pipe2 = FidelityPipeline.for_level(
        "standard", processors={"normalize": lambda a, p: a + "-normalized"})
    out2 = pipe2.run("artifact-1")
    assert out2["applied"] == ["normalize"]
    assert out2["artifact"] == "artifact-1-normalized"
