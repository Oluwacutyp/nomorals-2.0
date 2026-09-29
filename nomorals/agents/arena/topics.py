"""Topic sampling for the self-improvement arena.

The arena picks a random subject from a curated bank of real tech topics
(white and grey: systems, security fundamentals, networking, data, AI,
automation, history) — never one it has already digested (checked against
``arena_knowledge``), so the stream stays fresh.
"""

from __future__ import annotations

import random
from typing import Any

__all__ = ["CATEGORIES", "TOPIC_BANK", "sample_topic"]

CATEGORIES = (
    "web",
    "security",
    "networking",
    "data",
    "ai",
    "automation",
    "systems",
    "history",
)

TOPIC_BANK: dict[str, list[str]] = {
    "web": [
        "how CDN cache invalidation actually works",
        "why HTTP/3 switched to QUIC and what broke",
        "serverless cold starts and the ways people hide them",
        "how DNS-over-HTTPS changed what ISPs can see",
        "the state of HTTP field order and performance tuning in 2026",
        "how browsers decide a site is slow before it loads",
    ],
    "security": [
        "passkeys and how WebAuthn replaced passwords under the hood",
        "how eBPF changed rootkit and process monitoring games",
        "the history of format string attacks and why they still matter",
        "how sandbox escapes in mobile runtimes typically chain",
        "certificate transparency and what it hides from attackers",
        "why most 'quantum-safe' migration advice is premature",
    ],
    "networking": [
        "how BGP hijacks happen and how route origin validation stops them",
        "the inside of a 10Gbps home link: what the CPU actually does",
        "how Tor's onion routing stays anonymous against a compromised relay",
        "MPLS vs SD-WAN: what enterprises actually run in 2026",
        "how DNS tunneling works and how it gets detected",
        "the real cost of TLS handshakes on 2G and how 0-RTT helps",
    ],
    "data": [
        "how LSM trees beat B-trees in modern storage engines",
        "columnar storage and why analytics engines love it",
        "how vector databases index meaning and where they fail",
        "compaction strategies in key-value stores",
        "how data lakes quietly turn into data swamps",
        "the economics of object storage vs block storage",
    ],
    "ai": [
        "how mixture-of-experts models route tokens",
        "the history of GANs and where they lost to diffusion",
        "how inference engines quantize without wrecking quality",
        "RLHF vs DPO: what the training loops actually optimize",
        "how small language models beat big ones with distillation",
        "the engineering of prompt caching and KV-cache reuse",
    ],
    "automation": [
        "how CI systems survive flaky tests without lying",
        "the design of idempotent pipelines and why retries break them",
        "how feature flag systems avoid config drift at scale",
        "self-healing infrastructure and where it backfires",
        "how schedulers place jobs on heterogeneous hardware",
        "the art of observable dead-letter queues",
    ],
    "systems": [
        "how Linux namespaces and cages compose into containers",
        "the memory barrier problem and why ARM code surprises x86 devs",
        "how JIT compilers balance speed of compilation vs speed of code",
        "NUMA awareness and why multi-socket servers lie about locality",
        "how the page cache decides what to evict",
        "the history of the Unix process and why threads exist",
    ],
    "history": [
        "how ARPANET routing evolved into today's internet",
        "the history of the terminal and why ANSI colors exist",
        "how mainframe job schedulers shaped modern batch systems",
        "the rise and fall of groupware and what it taught us",
        "how punch cards actually stored programs",
        "the history of open-source licensing wars",
    ],
}


def _used_topics(db: Any) -> set[str]:
    used: set[str] = set()
    try:
        rows = db.query("SELECT topic FROM arena_knowledge")
        for row in rows:
            used.add(str(row.get("topic", "")))
    except Exception:  # noqa: BLE001 - no table yet on fresh test DBs
        pass
    return used


def sample_topic(db: Any = None, category: str | None = None, rng: random.Random | None = None) -> tuple[str, str]:
    """Pick ``(category, topic)`` — random category unless one is given.

    Repeats within a category are allowed; a topic already in
    ``arena_knowledge`` is skipped (with a bounded retry).
    """
    rng = rng or random.Random()
    used = _used_topics(db)
    cats = [category] if category else list(CATEGORIES)
    # Retry bound sized so a nearly-exhausted category still lands on the
    # unused topic with overwhelming probability (pool 6, one free: (5/6)^64 ≈ 5e-5).
    for _ in range(64):
        cat = rng.choice(cats)
        pool = TOPIC_BANK.get(cat, [])
        if not pool:
            continue
        topic = rng.choice(pool)
        if topic not in used:
            return cat, topic
    cat = cats[0]
    return cat, rng.choice(TOPIC_BANK.get(cat) or ["general computing"])
