"""Voice-trained brand voice + virality + evergreen + auto-plug (build-map #44)."""
import json

import pytest

from nomorals.social.voice import (
    VoiceProfile,
    apply_voice,
    virality_score,
    check_virality,
    VIRALITY_WARN_THRESHOLD,
    EvergreenQueue,
    AutoPlug,
)


@pytest.fixture()
def profile():
    p = VoiceProfile()
    p.train([
        "You don't need motivation. You need a system. Here's mine 🧵",
        "Stop optimizing your tools. Start shipping. Nobody cares about your stack.",
        "Hot take: meetings are just procrastination with a calendar invite 😅",
        "I asked my users what they wanted. Then I built the opposite. Revenue doubled?",
    ])
    return p


def test_train_extracts_signals(profile):
    assert profile.n_posts == 4
    assert profile.avg_sentence_len > 0
    assert profile.emoji_per_100 > 0  # two posts have emojis
    assert profile.question_rate == 0.25  # one of four ends with ?
    assert profile.opener_patterns, "openers extracted"
    assert profile.vocab, "vocab fingerprint extracted"


def test_train_empty_noop():
    p = VoiceProfile().train([])
    assert p.n_posts == 0


def test_describe_readable(profile):
    d = profile.describe()
    assert "words per sentence" in d


def test_save_load_roundtrip(profile, tmp_path):
    path = tmp_path / "vp.json"
    profile.save(path)
    loaded = VoiceProfile.load(path)
    assert loaded.n_posts == profile.n_posts
    assert loaded.vocab == profile.vocab


def test_load_corrupt_fresh(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert VoiceProfile.load(bad).n_posts == 0


def test_apply_voice_llm(profile):
    out = apply_voice("We are pleased to announce our new feature release.",
                      profile, llm_fn=lambda p: "We just shipped. Go try it.")
    assert out == "We just shipped. Go try it."


def test_apply_voice_rule_fallback_hashtag_trim():
    p = VoiceProfile()
    p.train(["short post #a", "another one #b"])
    # profile hashtag mean ~1; draft has 4 → trimmed
    out = apply_voice("launch day is here #a #b #c #d", p)
    assert out.count("#") <= 2, out


def test_apply_voice_no_profile_passthrough():
    assert apply_voice("hello", VoiceProfile()) == "hello"


def test_apply_voice_llm_failure_falls_back(profile):
    def boom(prompt):
        raise RuntimeError("llm down")
    out = apply_voice("hello world #a #b #c #d #e #f", profile, llm_fn=boom)
    assert isinstance(out, str) and out  # rule fallback, not a crash


def test_feedback_nudges():
    p = VoiceProfile()
    p.train(["This is a reasonably long sentence with quite a few words in it indeed."])
    before = p.avg_sentence_len
    p.feedback("too formal, make it punchier")
    assert p.avg_sentence_len < before
    assert p.corrections == ["too formal, make it punchier"]


def test_feedback_stores_fact():
    p = VoiceProfile()
    seen = []
    class FakeFacts:
        def add_fact(self, text, confidence=0.7):
            seen.append(text)
    p.feedback("more emojis please", facts=FakeFacts())
    assert seen and "more emojis please" in seen[0]


# ── virality ──

def test_virality_strong_vs_weak():
    strong = virality_score(
        "You don't need motivation. You need a system. What's yours?",
        platform="x")
    weak = virality_score(
        "Excited to announce our quarterly synergy alignment initiative that "
        "will leverage cross-functional stakeholder engagement to drive "
        "paradigm-shifting outcomes across the organization going forward!!! "
        "#business #synergy #growth #leadership #innovation #disruption",
        platform="x")
    assert strong.score > weak.score, (strong.score, weak.score)
    assert strong.score >= 55
    assert weak.score < VIRALITY_WARN_THRESHOLD


def test_virality_reasons_listed():
    vs = virality_score("Stop scrolling. Read this. What's your excuse?",
                        platform="x")
    assert vs.reasons
    assert vs.grade in ("strong", "decent", "weak", "likely to flop")


def test_virality_empty():
    assert virality_score("").score == 0


def test_check_virality_warns_not_blocks():
    warn = check_virality(
        "excited to announce our quarterly synergy alignment initiative that "
        "leverages cross-functional stakeholders across multiple business "
        "units to drive paradigm-shifting outcomes going forward #a #b #c "
        "#d #e #f",
        platform="x")
    assert warn is not None
    assert "might underperform" in warn
    assert "punch-up" in warn
    assert check_virality(
        "You don't need motivation. You need a system. What's yours?",
        platform="x") is None


def test_virality_novelty_vs_winners():
    base = "My morning routine: coffee, code, chaos."
    dup = virality_score(base, past_winners=[base])
    fresh = virality_score(base, past_winners=["Completely unrelated post."])
    assert dup.score < fresh.score


# ── evergreen ──

def test_evergreen_add_and_due(tmp_path):
    q = EvergreenQueue(tmp_path / "e.db")
    q.add("Tip: ship daily.", "tips", engagement=200, min_engagement=50,
          cooldown_days=30)
    due = q.next_due("tips")
    assert len(due) == 1
    assert due[0].content == "Tip: ship daily."
    q.mark_posted(due[0].id)
    assert q.next_due("tips") == []  # cooldown now


def test_evergreen_below_floor_not_due(tmp_path):
    q = EvergreenQueue(tmp_path / "e.db")
    q.add("meh post", "tips", engagement=10, min_engagement=50)
    assert q.next_due("tips") == []


def test_evergreen_category_filter(tmp_path):
    q = EvergreenQueue(tmp_path / "e.db")
    q.add("tip one", "tips", engagement=100)
    q.add("win one", "wins", engagement=100)
    assert [p.content for p in q.next_due("tips")] == ["tip one"]
    assert len(q.next_due()) == 2  # no filter → all


def test_evergreen_never_raises(tmp_path):
    q = EvergreenQueue(tmp_path / "bad.db")
    q.db_path = tmp_path / "nonexistent-dir" / "x.db"  # unwritable
    assert q.add("x", "tips") == ""
    assert q.next_due("tips") == []
    q.mark_posted("nope")  # no crash


# ── auto-plug ──

def test_autoplug_fires_at_threshold():
    calls = []
    ap = AutoPlug(cta="Get the guide: example.com/guide", default_threshold=100)
    assert ap.check("p1", 150, reply_fn=lambda pid, cta: calls.append((pid, cta)))
    assert calls == [("p1", "Get the guide: example.com/guide")]


def test_autoplug_below_threshold_silent():
    ap = AutoPlug(cta="example.com", default_threshold=100)
    assert ap.check("p1", 50, reply_fn=lambda *_: (_ for _ in ()).throw(
        AssertionError("should not fire"))) is False


def test_autoplug_no_cta_never_fires():
    ap = AutoPlug(default_threshold=1)  # even trivial threshold
    assert ap.check("p1", 9999, reply_fn=lambda *_: (_ for _ in ()).throw(
        AssertionError("CTA must never be invented"))) is False


def test_autoplug_fires_once():
    calls = []
    ap = AutoPlug(cta="example.com", default_threshold=10)
    assert ap.check("p1", 50, reply_fn=lambda pid, c: calls.append(pid))
    assert ap.check("p1", 5000, reply_fn=lambda pid, c: calls.append(pid)) is False
    assert calls == ["p1"]


def test_autoplug_per_platform_threshold():
    ap = AutoPlug(cta="example.com",
                  thresholds={"linkedin": 500}, default_threshold=100)
    assert ap.threshold_for("linkedin") == 500
    assert ap.threshold_for("x") == 100
    # 200 < linkedin 500 → no fire; but ≥ x 100 → fire
    assert ap.check("p1", 200, platform="linkedin",
                    reply_fn=lambda *_: True) is False
    assert ap.check("p2", 200, platform="x",
                    reply_fn=lambda *_: True) is True


def test_autoplug_never_raises():
    ap = AutoPlug(cta="example.com")
    def boom(pid, cta):
        raise RuntimeError("network down")
    assert ap.check("p1", 999, reply_fn=boom) is False
