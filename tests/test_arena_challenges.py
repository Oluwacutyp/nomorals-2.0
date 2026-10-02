"""Wave B challenge packs: volume, format, and verifiability.

Covers the four challenge packs in nomorals/agents/arena/challenge_packs/:
- every pack registers and validates clean via validate_pack
- the merged bank carries ~400 challenges on top of the core topics
- every challenge entry has a verifiable acceptance criterion + kind
- no duplicate challenge texts anywhere in the bank
- each new category has a real difficulty spread (the sampler's ramp)
- the adaptive sampler can draw challenge entries with verify/kind intact
"""
from __future__ import annotations

import unittest

from nomorals.agents.arena import challenge_packs  # noqa: F401  (side-effect import)
from nomorals.agents.arena.challenges import is_challenge, validate_pack
from nomorals.agents.arena.topics import (
    TOPIC_BANK,
    bank_size,
    topic_packs,
    topic_text,
    topics_in,
)
from nomorals.agents.arena import sampling

PACK_MODULES = {
    "chall_systems": ("nomorals.agents.arena.challenge_packs.systems",
                      ["systems_design", "distributed", "concurrency",
                       "databases", "networking", "compilers",
                       "runtimes", "packaging"]),
    "chall_web": ("nomorals.agents.arena.challenge_packs.web",
                  ["web_fullstack", "apis", "auth", "realtime",
                   "mobile", "termux", "performance", "memory"]),
    "chall_ml": ("nomorals.agents.arena.challenge_packs.ml",
                 ["security_eng", "threat_modeling", "secure_defaults",
                  "ml_ops", "evals", "finetune", "rag", "agents"]),
    "chall_product": ("nomorals.agents.arena.challenge_packs.product",
                      ["ux_chat", "latency", "routing", "media_pipelines",
                       "ocr", "vision_edit", "research_methods",
                       "source_trust", "distillation"]),
}


def _pack_dict(pack_name: str) -> dict:
    mod_path = PACK_MODULES[pack_name][0]
    mod = __import__(mod_path, fromlist=["_PACK"])
    return mod._PACK


class PackRegistrationTests(unittest.TestCase):
    def test_all_packs_registered(self):
        for name in PACK_MODULES:
            self.assertIn(name, topic_packs(), f"pack {name} not registered")

    def test_bank_volume(self):
        # 168 core topics + ~396 challenges
        self.assertGreaterEqual(bank_size(), 500)

    def test_all_packs_validate_clean(self):
        for name in PACK_MODULES:
            errors = validate_pack(name, _pack_dict(name),
                                   min_per_category=12)
            self.assertEqual(errors, [], f"pack {name}: {errors[:5]}")

    def test_categories_present_with_volume(self):
        for name, (_, cats) in PACK_MODULES.items():
            for cat in cats:
                entries = topics_in(cat)
                self.assertGreaterEqual(
                    len(entries), 12, f"{name}/{cat}: only {len(entries)}")


class ChallengeFormatTests(unittest.TestCase):
    def test_every_challenge_has_verify_and_kind(self):
        checked = 0
        for name in PACK_MODULES:
            for cat, entries in _pack_dict(name).items():
                for e in entries:
                    self.assertTrue(is_challenge(e),
                                    f"{name}/{cat}: missing verify: "
                                    f"{topic_text(e)[:60]}")
                    self.assertIn(e.get("kind"), ("code", "research", "build"))
                    self.assertTrue(e.get("tags"), f"{name}/{cat}: no tags")
                    checked += 1
        self.assertGreaterEqual(checked, 390)

    def test_no_duplicate_texts_in_bank(self):
        seen: set[str] = set()
        dups: list[str] = []
        for entries in TOPIC_BANK.values():
            for e in entries:
                t = topic_text(e)
                if t in seen:
                    dups.append(t)
                seen.add(t)
        self.assertEqual(dups, [], f"duplicates: {dups[:3]}")

    def test_difficulty_spread_per_category(self):
        for name, (_, cats) in PACK_MODULES.items():
            for cat in cats:
                ds = {int(e.get("d", 2)) for e in topics_in(cat)}
                self.assertGreaterEqual(
                    len(ds), 2, f"{cat}: no difficulty spread {sorted(ds)}")


class SamplerIntegrationTests(unittest.TestCase):
    def test_sample_challenge_returns_challenge_entry(self):
        # db=None: pure sampler, no persistence — draw from a challenge cat
        import random
        rng = random.Random(7)
        seen_verify = 0
        for _ in range(30):
            cat, text, entry = sampling.sample_challenge(
                None, category="rag", rng=rng, anti_repeat=0)
            self.assertEqual(cat, "rag")
            self.assertTrue(text)
            if entry.get("verify"):
                seen_verify += 1
                self.assertIn(entry.get("kind"), ("code", "research", "build"))
        self.assertGreater(seen_verify, 0)


if __name__ == "__main__":
    unittest.main()
