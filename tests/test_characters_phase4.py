"""Phase 4: character memory backend, voice fingerprint, context builder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.characters.character import Character
from nomorals.characters.voice import (
    VoiceFingerprint, fingerprint, consistency, distinctness,
)
from nomorals.characters.context import CharacterContextBuilder
from nomorals.characters.memory import CharacterMemory


def _zara() -> Character:
    return Character(
        name="Zara",
        persona={"witty": 0.9, "curious": 0.9, "bold": 0.75},
        backstory="Lagos-born radio kid.",
        core_motive="To ask the question nobody else will ask.",
        expression={"catchphrases": ["ok real", "hear me out"],
                    "emoji_rate": 0.4},
        spine=0.8,
        beliefs=[{"text": "everyone has a story", "confidence": 0.9,
                  "revisions": 0}],
    )


# ── voice fingerprint ────────────────────────────────────────────────
def test_fingerprint_basic():
    utterances = [
        "ok real, that's actually wild. tell me more?",
        "hear me out — what if we just ask them directly?",
        "i love this energy! who else is coming??",
        "ok real, no way. that's the best thing i've heard all week!",
        "so what's the real story here? give me the gist.",
        "hear me out, i think there's more to this!",
    ]
    fp = fingerprint(utterances, ["ok real", "hear me out"])
    assert fp.n_samples == 6
    assert fp.question_rate > 0.3
    assert fp.catchphrase_hits > 0.3
    assert fp.mean_sentence_len > 0


def test_fingerprint_empty():
    fp = fingerprint([])
    assert fp.n_samples == 0
    assert consistency(fp, "hello") == 0.5  # not enough data


def test_consistency_in_voice():
    utterances = [
        "ok real, that's wild. tell me everything?",
        "hear me out — what if we flip it around?",
        "no way! that's incredible, who did this?",
        "ok real, i'm obsessed with this story.",
        "so what's the angle here? give me details?",
        "hear me out, let's dig deeper into this!",
    ] * 2
    fp = fingerprint(utterances, ["ok real", "hear me out"])
    score = consistency(fp, "ok real, that's amazing? tell me more!",
                        ["ok real", "hear me out"])
    assert score > 0.5, f"in-voice utterance scored {score}"


def test_consistency_out_of_voice():
    utterances = [
        "ok real, that's wild. tell me everything?",
        "hear me out — what if we flip it around?",
        "no way! that's incredible, who did this?",
        "ok real, i'm obsessed with this story.",
        "so what's the angle here? give me details?",
        "hear me out, let's dig deeper into this!",
    ] * 2
    fp = fingerprint(utterances, ["ok real", "hear me out"])
    # formal, no questions, no catchphrases — should score lower
    score = consistency(
        fp,
        "The quarterly financial report indicates a seventeen percent "
        "increase in operational efficiency across all departments.",
        ["ok real", "hear me out"])
    in_voice = consistency(fp, "ok real, that's amazing? tell me more!",
                           ["ok real", "hear me out"])
    assert score < in_voice, f"out-of-voice {score} >= in-voice {in_voice}"


def test_distinctness():
    a_utts = ["ok real, that's wild? tell me more!"] * 8
    b_utts = ["The data suggests otherwise. Furthermore, analysis shows."] * 8
    fa = fingerprint(a_utts, ["ok real"])
    fb = fingerprint(b_utts, [])
    d = distinctness(fa, fb)
    assert d > 0.4, f"distinct voices scored {d}"
    d_same = distinctness(fa, fa)
    assert d_same < 0.3, f"same voice scored {d_same}"


def test_fingerprint_roundtrip():
    fp = fingerprint(["hello world, how are you?"] * 6, [])
    d = fp.to_dict()
    fp2 = VoiceFingerprint.from_dict(d)
    assert fp2.n_samples == fp.n_samples
    assert abs(fp2.mean_sentence_len - fp.mean_sentence_len) < 0.01


# ── character memory (flat fallback) ─────────────────────────────────
def test_character_memory_flat():
    char = _zara()
    mem = CharacterMemory.for_character(char, db=None)
    assert not mem.deep
    mem.remember("Zara promised to call the senator tomorrow", 0.8)
    mem.remember("Elder likes palm wine", 0.4)
    results = mem.recall("What did Zara promise?")
    assert any("senator" in r for r in results)
    # proactive + contradictions degrade gracefully without a DB
    assert mem.proactive("hello") == []
    assert mem.check_contradiction("x") == []


def test_character_memory_empty_query():
    char = _zara()
    mem = CharacterMemory.for_character(char, db=None)
    mem.remember("", 0.5)  # no-op
    assert mem.recall("") == []


# ── context builder ──────────────────────────────────────────────────
def test_context_builder_identity():
    char = _zara()
    builder = CharacterContextBuilder()
    prompt = builder.build(char)
    assert "Zara" in prompt
    assert "Lagos-born radio kid" in prompt
    assert "ask the question nobody else will ask" in prompt
    assert "witty" in prompt.lower() or "very witty" in prompt
    # spine >= 0.7 → push-back instruction
    assert "spine" in prompt.lower() or "disagree" in prompt.lower()


def test_context_builder_expression():
    char = _zara()
    builder = CharacterContextBuilder()
    prompt = builder.build(char)
    assert "ok real" in prompt  # catchphrases rendered


def test_context_builder_mood():
    char = _zara()
    char.mood = {"valence": 0.8, "arousal": 0.9, "trust": 0.9}
    builder = CharacterContextBuilder()
    prompt = builder.build(char)
    assert "feeling good" in prompt
    assert "high energy" in prompt


def test_context_builder_memory():
    char = _zara()
    builder = CharacterContextBuilder()
    prompt = builder.build(
        char, memories=["Zara interviewed the senator in 2024"])
    assert "senator" in prompt


def test_context_builder_beliefs():
    char = _zara()
    builder = CharacterContextBuilder()
    prompt = builder.build(char)
    assert "everyone has a story" in prompt  # strong belief rendered


def test_context_builder_budget():
    char = _zara()
    builder = CharacterContextBuilder(total_budget=1200)
    long_mems = [f"memory number {i} with lots of filler text " * 20
                 for i in range(20)]
    prompt = builder.build(char, memories=long_mems)
    # identity must survive even under a tight budget
    assert "Zara" in prompt
    assert "Reply as the character" in prompt


def test_character_serialization_with_fingerprint():
    char = _zara()
    char.voice_fingerprint = {"n_samples": 10, "mean_sentence_len": 8.5}
    d = char.to_dict()
    assert d["voice_fingerprint"]["n_samples"] == 10
    char2 = Character.from_dict(d)
    assert char2.voice_fingerprint["n_samples"] == 10


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
