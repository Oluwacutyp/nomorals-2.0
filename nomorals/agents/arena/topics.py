"""Dynamic topic tables for the self-improvement arena.

The arena used to pick from 48 static one-liners. Now:

* **Rich topic tables** — ~170 topics across 14 categories, each with a
  difficulty grade (1 = accessible, 2 = practitioner, 3 = deep) and
  keyword tags. ``/arena topics`` renders the full table;
  ``/arena topics <category>`` drills into one.
* **Topic packs** — the bank is a registry of packs. The built-in
  tables ship as the ``"core"`` pack; new packs register with
  ``register_topic_pack(name, topics, weights)`` where ``topics`` is
  ``{category: [entries]}`` (entries are ``_t`` dicts or plain
  strings) and ``weights`` is an optional ``{category: multiplier}``.
  ``TOPIC_BANK`` is the merged view of every registered pack.
* **Personalized sampling** — ``sample_topic`` takes an interest
  ``profile`` (category → weight) built by
  ``nomorals/agents/arena/activity.py`` from what the user actually
  does: recent searches, arena digest history, custom topics, goals,
  command usage. Categories the user engages with get picked more
  often; pack weights multiply on top.
* **Random rotation** — ``surprise_topic`` ignores the profile
  entirely (pure random, optional seed). Every sampler skips topics
  already in ``arena_knowledge``, deprioritizes recently served
  categories, and enforces a **topic anti-repeat window**: no topic
  repeats within the last N served (default 10, configurable via
  ``anti_repeat`` / ``set_anti_repeat_window`` / the
  ``arena.anti_repeat_window`` setting). The window is persisted in
  ``kv_store`` so it survives restarts; when the eligible pool is
  exhausted the window reshuffles (clears) instead of failing.

``TOPIC_BANK`` values are dicts (``t`` = text, ``d`` = difficulty,
``tags`` = keywords); the topic *text* stays the dedup identity so
old ``arena_knowledge`` rows keep working.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CATEGORIES",
    "DEFAULT_ANTI_REPEAT",
    "TOPIC_BANK",
    "TopicPack",
    "all_categories",
    "anti_repeat_window",
    "bank_size",
    "category_stats",
    "category_table",
    "clear_recent_topics",
    "recent_topics",
    "register_topic_pack",
    "sample_topic",
    "set_anti_repeat_window",
    "surprise_topic",
    "topic_packs",
    "topic_text",
    "topics_in",
    "topics_table",
    "unregister_topic_pack",
]

CATEGORIES = (
    "web",
    "security",
    "networking",
    "data",
    "ai",
    "automation",
    "systems",
    "history",
    "crypto",
    "mobile",
    "devtools",
    "os",
    "hardware",
    "languages",
)


def _t(text: str, difficulty: int, *tags: str) -> dict[str, Any]:
    return {"t": text, "d": difficulty, "tags": tags,
            "verify": "", "kind": "code"}


def topic_text(entry: dict[str, Any]) -> str:
    """The dedup identity of a topic entry — its text."""
    return str(entry.get("t", ""))


# The built-in tables, shipped as the "core" topic pack (registered below).
_CORE_TOPICS: dict[str, list[dict[str, Any]]] = {
    "web": [
        _t("how CDN cache invalidation actually works", 2, "cdn", "caching", "edge"),
        _t("why HTTP/3 switched to QUIC and what broke", 2, "http", "quic", "protocol"),
        _t("serverless cold starts and the ways people hide them", 2, "serverless", "lambda"),
        _t("how DNS-over-HTTPS changed what ISPs can see", 2, "dns", "privacy"),
        _t("the state of HTTP field order and performance tuning in 2026", 3, "http", "performance"),
        _t("how browsers decide a site is slow before it loads", 2, "browser", "performance"),
        _t("islands architecture: partial hydration without the framework wars", 2, "frontend", "ssr"),
        _t("how WebSockets stay alive through hostile middleboxes", 2, "websocket", "networking"),
        _t("cookie-less tracking: fingerprinting surfaces browsers actually expose", 3, "privacy", "tracking"),
        _t("edge compute vs origin: where the logic really lives in 2026", 2, "edge", "cdn"),
        _t("how certificate stapling and OCSP keep TLS handshakes fast", 2, "tls", "security"),
        _t("the anatomy of a service worker cache poisoning attack", 3, "pwa", "security", "caching"),
    ],
    "security": [
        _t("passkeys and how WebAuthn replaced passwords under the hood", 2, "auth", "webauthn"),
        _t("how eBPF changed rootkit and process monitoring games", 3, "ebpf", "kernel"),
        _t("the history of format string attacks and why they still matter", 2, "exploit", "c"),
        _t("how sandbox escapes in mobile runtimes typically chain", 3, "mobile", "sandbox"),
        _t("certificate transparency and what it hides from attackers", 2, "tls", "pki"),
        _t("why most 'quantum-safe' migration advice is premature", 2, "crypto", "pqc"),
        _t("how supply-chain attacks poison build pipelines undetected", 2, "supply-chain", "ci"),
        _t("memory-safe languages vs hardened C: the 2026 scoreboard", 2, "rust", "memory-safety"),
        _t("how phishing kits bypass MFA with adversary-in-the-middle proxies", 3, "phishing", "mfa"),
        _t("the economics of bug bounties: what actually gets paid", 1, "bug-bounty"),
        _t("how YARA rules catch malware families without signatures", 2, "malware", "detection"),
        _t("side-channel attacks on shared cloud CPUs and the mitigations that stuck", 3, "side-channel", "cloud"),
    ],
    "networking": [
        _t("how BGP hijacks happen and how route origin validation stops them", 2, "bgp", "routing"),
        _t("the inside of a 10Gbps home link: what the CPU actually does", 2, "ethernet", "performance"),
        _t("how Tor's onion routing stays anonymous against a compromised relay", 3, "tor", "privacy"),
        _t("MPLS vs SD-WAN: what enterprises actually run in 2026", 2, "wan", "enterprise"),
        _t("how DNS tunneling works and how it gets detected", 3, "dns", "exfiltration"),
        _t("the real cost of TLS handshakes on 2G and how 0-RTT helps", 2, "tls", "mobile"),
        _t("how anycast makes one IP live on every continent", 2, "anycast", "dns"),
        _t("bufferbloat: why your gigabit line feels slow and what FQ-CoDel does", 2, "qos", "latency"),
        _t("how QUIC connection migration survives switching from wifi to 5G", 2, "quic", "mobile"),
        _t("the quiet death of NAT and what replaced it at the edge", 2, "nat", "ipv6"),
        _t("how submarine cables are repaired without breaking the internet", 1, "infrastructure"),
        _t("VXLAN and overlay networks: why datacenters stopped trusting VLANs", 3, "datacenter", "sdn"),
    ],
    "data": [
        _t("how LSM trees beat B-trees in modern storage engines", 2, "storage", "lsm"),
        _t("columnar storage and why analytics engines love it", 2, "analytics", "parquet"),
        _t("how vector databases index meaning and where they fail", 2, "vector", "ai"),
        _t("compaction strategies in key-value stores", 3, "lsm", "rocksdb"),
        _t("how data lakes quietly turn into data swamps", 1, "lakehouse"),
        _t("the economics of object storage vs block storage", 2, "s3", "storage"),
        _t("how CRDTs let replicas agree without a leader", 3, "distributed", "consistency"),
        _t("stream processing exactly-once semantics: the checkpoint trick", 3, "kafka", "flink"),
        _t("how Postgres MVCC makes concurrent writes not fight", 2, "postgres", "mvcc"),
        _t("the rise of zero-ETL: databases that read each other's files", 2, "etl", "lakehouse"),
        _t("how time-series databases compress a year of metrics into megabytes", 2, "timeseries"),
        _t("sharding vs partitioning: picking the split that won't haunt you", 2, "scaling", "sql"),
    ],
    "ai": [
        _t("how mixture-of-experts models route tokens", 3, "moe", "llm"),
        _t("the history of GANs and where they lost to diffusion", 2, "gan", "diffusion"),
        _t("how inference engines quantize without wrecking quality", 2, "quantization", "inference"),
        _t("RLHF vs DPO: what the training loops actually optimize", 3, "rlhf", "training"),
        _t("how small language models beat big ones with distillation", 2, "distillation", "slm"),
        _t("the engineering of prompt caching and KV-cache reuse", 2, "inference", "caching"),
        _t("how retrieval-augmented generation fails at the retrieval step", 2, "rag", "embeddings"),
        _t("speculative decoding: free tokens from a draft model", 3, "inference", "speed"),
        _t("how diffusion models actually denoise, step by step", 3, "diffusion"),
        _t("the data wall: where training corpora come from in 2026", 2, "training", "data"),
        _t("how tool-calling agents plan multi-step work without drifting", 2, "agents"),
        _t("evaluation hell: why LLM benchmarks disagree with each other", 2, "eval", "benchmarks"),
    ],
    "automation": [
        _t("how CI systems survive flaky tests without lying", 2, "ci", "testing"),
        _t("the design of idempotent pipelines and why retries break them", 2, "pipelines"),
        _t("how feature flag systems avoid config drift at scale", 2, "feature-flags"),
        _t("self-healing infrastructure and where it backfires", 2, "sre", "kubernetes"),
        _t("how schedulers place jobs on heterogeneous hardware", 3, "scheduling", "gpu"),
        _t("the art of observable dead-letter queues", 2, "queues", "messaging"),
        _t("how GitOps reconciles the cluster you have with the one you declared", 2, "gitops", "kubernetes"),
        _t("canary deploys vs blue-green: failure modes compared", 2, "deploy", "sre"),
        _t("how cron replacements handle millions of scheduled jobs", 2, "scheduling"),
        _t("the anatomy of a Terraform state meltdown and recovery", 2, "terraform", "iac"),
        _t("how build caches make 10-minute pipelines take 40 seconds", 2, "ci", "caching"),
        _t("chaos engineering beyond the hype: what actually breaks first", 2, "chaos", "sre"),
    ],
    "systems": [
        _t("how Linux namespaces and cages compose into containers", 2, "containers", "linux"),
        _t("the memory barrier problem and why ARM code surprises x86 devs", 3, "concurrency", "arm"),
        _t("how JIT compilers balance speed of compilation vs speed of code", 3, "jit", "compilers"),
        _t("NUMA awareness and why multi-socket servers lie about locality", 3, "numa", "performance"),
        _t("how the page cache decides what to evict", 2, "linux", "memory"),
        _t("the history of the Unix process and why threads exist", 1, "unix", "history"),
        _t("how io_uring killed the async I/O debate on Linux", 3, "linux", "io"),
        _t("cgroups v2: how your container's CPU limit is actually enforced", 2, "containers", "linux"),
        _t("how debuggers freeze a running process without its cooperation", 3, "debugging", "ptrace"),
        _t("the real reason fork() still exists in 2026", 2, "unix", "process"),
        _t("how memory allocators fragment and what jemalloc does about it", 3, "memory", "malloc"),
        _t("seccomp filters: the syscall firewall every container runs", 2, "linux", "security"),
    ],
    "history": [
        _t("how ARPANET routing evolved into today's internet", 1, "internet"),
        _t("the history of the terminal and why ANSI colors exist", 1, "terminal"),
        _t("how mainframe job schedulers shaped modern batch systems", 1, "mainframe"),
        _t("the rise and fall of groupware and what it taught us", 1, "collaboration"),
        _t("how punch cards actually stored programs", 1, "retro"),
        _t("the history of open-source licensing wars", 1, "oss", "licensing"),
        _t("how the Morris worm accidentally invented incident response", 2, "worm", "1988"),
        _t("why the QWERTY layout survived a century of better ideas", 1, "keyboards"),
        _t("the browser wars: how Netscape lost and the web won", 1, "browser"),
        _t("how Usenet invented every social media argument by 1993", 1, "usenet", "culture"),
        _t("the 640K myth and the real limits of early PCs", 1, "retro", "dos"),
        _t("how shareware distribution built the first indie software market", 1, "shareware", "business"),
    ],
    "crypto": [
        _t("how zero-knowledge proofs verify without revealing", 3, "zk", "cryptography"),
        _t("UTXO vs account model: why blockchains count money differently", 2, "bitcoin", "ethereum"),
        _t("how MEV bots reorder your transactions for profit", 3, "mev", "defi"),
        _t("the anatomy of a bridge hack: where the trust actually lives", 3, "bridge", "defi"),
        _t("how proof-of-stake finality differs from proof-of-work", 2, "pos", "consensus"),
        _t("stablecoin mechanics: collateralized vs algorithmic, post-mortem", 2, "stablecoin"),
        _t("how wallets derive infinite keys from one seed phrase", 2, "bip39", "wallets"),
        _t("rollups: how L2s inherit L1 security without L1 costs", 3, "l2", "rollup"),
        _t("the oracle problem: why smart contracts can't see the weather", 2, "oracle", "defi"),
        _t("how mixers and privacy pools actually obscure trails", 3, "privacy", "mixer"),
        _t("post-quantum signatures: what changes for blockchains first", 3, "pqc", "signatures"),
        _t("how on-chain forensics clusters wallets into people", 2, "forensics", "chainalysis"),
    ],
    "mobile": [
        _t("how iOS and Android sandbox apps differently", 2, "ios", "android"),
        _t("the anatomy of a push notification's 4-second journey", 2, "push", "apns"),
        _t("how mobile GPUs render 120fps without melting the battery", 3, "gpu", "battery"),
        _t("background execution limits: what your app can do while asleep", 2, "ios", "android"),
        _t("how app thinning ships one binary to every device", 2, "ios", "build"),
        _t("the real cost of cross-platform frameworks in 2026", 2, "flutter", "react-native"),
        _t("how biometric auth stays on-device and out of the cloud", 2, "biometrics", "security"),
        _t("mobile deep links vs app links: the routing wars", 2, "deeplink"),
        _t("how offline-first apps sync without losing writes", 3, "sync", "crdt"),
        _t("the 5G reality check: what actually got faster", 1, "5g"),
        _t("how foldables broke every assumption in layout engines", 2, "foldable", "ui"),
        _t("battery chemistry vs software: where standby drain really goes", 2, "battery"),
    ],
    "devtools": [
        _t("how language servers answer 'go to definition' in milliseconds", 2, "lsp", "ide"),
        _t("the design of incremental builds that actually stay correct", 3, "build", "bazel"),
        _t("how debuggers map optimized machine code back to your source", 3, "debugging", "dwarf"),
        _t("tree-sitter: how one parser powers every editor's highlighting", 2, "parsing", "editor"),
        _t("how package managers resolve dependency hell without SAT solvers choking", 3, "npm", "cargo"),
        _t("the anatomy of a good CLI: flags, pipes, and exit codes", 1, "cli", "ux"),
        _t("how formatters end style wars without anyone noticing", 1, "formatting", "prettier"),
        _t("remote dev environments: why the editor left your laptop", 2, "codespaces", "remote"),
        _t("how test runners parallelize without flaking", 2, "testing", "pytest"),
        _t("the rise of the AI pair programmer inside the IDE", 2, "copilot", "ai"),
        _t("how profilers sample a running program without slowing it much", 3, "profiling"),
        _t("monorepo tooling: how thousands of engineers share one repo", 2, "monorepo", "bazel"),
    ],
    "os": [
        _t("how a bootloader hands off to the kernel without breaking", 2, "boot", "kernel"),
        _t("virtual memory: how every process thinks it owns the RAM", 2, "memory", "mmu"),
        _t("how schedulers decide which thread runs next, 1000 times a second", 3, "scheduling", "kernel"),
        _t("the anatomy of a context switch: what it really costs", 3, "kernel", "performance"),
        _t("how filesystems journal their way out of a power cut", 2, "filesystem", "ext4"),
        _t("copy-on-write: the trick behind instant snapshots", 2, "btrfs", "zfs"),
        _t("how signals interrupt a process mid-syscall", 3, "unix", "signals"),
        _t("the /proc filesystem: the kernel's confession booth", 1, "linux", "proc"),
        _t("how containers share one kernel without seeing each other", 2, "containers", "namespaces"),
        _t("microkernels vs monoliths: the debate that never died", 2, "kernel", "design"),
        _t("how hibernation freezes a whole machine to disk", 2, "power", "acpi"),
        _t("the init wars: what systemd actually replaced", 1, "systemd", "linux"),
    ],
    "hardware": [
        _t("how a CPU pipeline predicts branches and recovers from lies", 3, "cpu", "branch-prediction"),
        _t("the memory hierarchy: why L1 cache is 1ns and RAM is 100ns", 2, "cpu", "cache"),
        _t("how SSDs wear-level without the OS ever knowing", 2, "ssd", "flash"),
        _t("the physics of why clock speeds stopped growing", 2, "cpu", "moore"),
        _t("how GPUs turned into AI accelerators by accident", 2, "gpu", "cuda"),
        _t("chiplet design: why CPUs are Lego bricks now", 2, "cpu", "packaging"),
        _t("how ECC memory catches the bit flips cosmic rays cause", 2, "ram", "reliability"),
        _t("the anatomy of a data center power failure", 2, "datacenter", "power"),
        _t("RISC-V: the open ISA's road from hobby to hyperscaler", 2, "riscv", "isa"),
        _t("how network cards bypass the CPU with RDMA", 3, "nic", "rdma"),
        _t("the thermals of a rack: where 40kW of heat actually goes", 2, "datacenter", "cooling"),
        _t("how Apple Silicon unified memory changed the performance math", 2, "arm", "soc"),
    ],
    "languages": [
        _t("how Rust's borrow checker proves memory safety at compile time", 3, "rust", "memory-safety"),
        _t("garbage collector designs: from mark-sweep to ZGC's colored pointers", 3, "gc", "jvm"),
        _t("how Python's GIL survived and what free-threading changes", 2, "python", "gil"),
        _t("the lambda calculus hiding inside every functional language", 3, "functional", "theory"),
        _t("how TypeScript's type system erases itself before runtime", 2, "typescript", "types"),
        _t("why Go chose goroutines over async/await", 2, "go", "concurrency"),
        _t("the actor model: how Erlang survives failures that kill other systems", 3, "erlang", "actors"),
        _t("how Zig does comptime metaprogramming without macros", 3, "zig", "metaprogramming"),
        _t("pattern matching: the feature every language is stealing", 2, "types", "design"),
        _t("how interpreters, compilers, and JITs form a spectrum, not a ladder", 2, "compilers", "interpreters"),
        _t("the economics of language adoption: why better rarely wins", 1, "history", "design"),
        _t("effect systems: the next big idea after async/await", 3, "types", "functional"),
    ],
}


# ── topic packs ───────────────────────────────────────────────────────────
# The bank is a registry. Every pack contributes {category: [entries]};
# TOPIC_BANK is the merged view (mutated in place so `from ... import
# TOPIC_BANK` holders always see the current merge). Topic text is the
# dedup identity across packs — first pack registered wins.

@dataclass
class TopicPack:
    """One registered topic pack.

    ``topics``: ``{category: [entries]}`` — entries are ``_t`` dicts
    (``t``/``d``/``tags``) or plain strings (normalized to
    difficulty 2, no tags). ``weights``: optional
    ``{category: multiplier}`` applied on top of the interest profile
    during sampling.
    """

    name: str
    topics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)


_PACKS: dict[str, TopicPack] = {}
_PACK_LOCK = threading.Lock()

#: Merged view of every registered pack. Rebuilt in place on
#: register/unregister — never rebind this name.
TOPIC_BANK: dict[str, list[dict[str, Any]]] = {}


def _normalize_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            raise ValueError("topic pack entry must not be an empty string")
        return {"t": text, "d": 2, "tags": ()}
    if isinstance(entry, dict):
        text = str(entry.get("t", "")).strip()
        if not text:
            raise ValueError("topic pack entry dict needs a 't' text")
        d = entry.get("d", 2)
        try:
            d = min(3, max(1, int(d)))
        except (TypeError, ValueError):
            d = 2
        tags = entry.get("tags", ())
        tags = tuple(str(t) for t in tags) if tags else ()
        verify = str(entry.get("verify", "") or "").strip()
        kind = str(entry.get("kind", "code") or "code").strip().lower()
        if kind not in ("code", "research", "build"):
            kind = "code"
        return {"t": text, "d": d, "tags": tags,
                "verify": verify, "kind": kind}
    raise ValueError(
        f"topic pack entries must be str or dict, got {type(entry).__name__}")


def _rebuild_bank() -> None:
    merged: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for pack in _PACKS.values():  # registration order — first pack wins
        for cat, entries in pack.topics.items():
            bucket = merged.setdefault(str(cat), [])
            for e in entries:
                text = topic_text(e)
                if text and text not in seen:
                    seen.add(text)
                    bucket.append(e)
    TOPIC_BANK.clear()
    TOPIC_BANK.update(merged)


def register_topic_pack(name: str,
                        topics: dict[str, list[Any]],
                        weights: dict[str, float] | None = None) -> TopicPack:
    """Register (or replace) a topic pack; returns the stored pack.

    ``topics`` maps category → list of entries (``_t`` dicts or plain
    strings). ``weights`` maps category → sampling multiplier
    (default 1.0). Re-registering a name replaces that pack's
    contribution; the bank is rebuilt from all packs in registration
    order.
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("topic pack needs a non-empty name")
    if not isinstance(topics, dict) or not topics:
        raise ValueError("topic pack needs a non-empty {category: [entries]} mapping")
    norm_topics: dict[str, list[dict[str, Any]]] = {}
    for cat, entries in topics.items():
        cat = str(cat or "").strip().lower()
        if not cat:
            raise ValueError("topic pack category names must be non-empty")
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError(f"topic pack category {cat!r} needs a non-empty entry list")
        norm_topics[cat] = [_normalize_entry(e) for e in entries]
    norm_weights: dict[str, float] = {}
    for cat, w in (weights or {}).items():
        try:
            norm_weights[str(cat).strip().lower()] = max(0.0, float(w))
        except (TypeError, ValueError):
            raise ValueError(f"topic pack weight for {cat!r} must be numeric")
    pack = TopicPack(name=name, topics=norm_topics, weights=norm_weights)
    with _PACK_LOCK:
        _PACKS[name] = pack
        _rebuild_bank()
    return pack


def unregister_topic_pack(name: str) -> bool:
    """Remove a pack by name. The built-in ``"core"`` pack is protected."""
    name = str(name or "").strip()
    if name == "core":
        return False
    with _PACK_LOCK:
        if name not in _PACKS:
            return False
        del _PACKS[name]
        _rebuild_bank()
    return True


def topic_packs() -> list[str]:
    """Names of registered packs, in registration order."""
    with _PACK_LOCK:
        return list(_PACKS)


def all_categories() -> tuple[str, ...]:
    """Every category in the merged bank: core order, then pack extras."""
    extra = sorted(c for c in TOPIC_BANK if c not in CATEGORIES)
    return tuple(CATEGORIES) + tuple(extra)


def _pack_weight(category: str) -> float:
    """Max sampling multiplier any pack assigns to a category (≥ 0)."""
    best = 0.0
    found = False
    for pack in _PACKS.values():
        if category in pack.topics:
            found = True
            best = max(best, float(pack.weights.get(category, 1.0)))
    return best if found else 1.0


register_topic_pack("core", _CORE_TOPICS)


def topics_in(category: str) -> list[dict[str, Any]]:
    """All topic entries for a category (empty list for unknown ones)."""
    return list(TOPIC_BANK.get((category or "").lower(), []))


def bank_size() -> int:
    """Total topics across every category."""
    return sum(len(v) for v in TOPIC_BANK.values())


def category_stats() -> dict[str, dict[str, int]]:
    """Per-category counts plus difficulty spread."""
    out: dict[str, dict[str, int]] = {}
    for cat, entries in TOPIC_BANK.items():
        spread = {1: 0, 2: 0, 3: 0}
        for e in entries:
            d = int(e.get("d", 2))
            spread[min(3, max(1, d))] += 1
        out[cat] = {"topics": len(entries), "easy": spread[1],
                    "medium": spread[2], "deep": spread[3]}
    return out


# ── sampling ──────────────────────────────────────────────────────────────

#: Default anti-repeat window: no topic repeats within the last N served.
DEFAULT_ANTI_REPEAT = 10

_RECENT_TOPICS_KEY = "arena.recent_topics"
_ANTI_REPEAT_WINDOW_KEY = "arena.anti_repeat_window"


def _used_topics(db: Any) -> set[str]:
    used: set[str] = set()
    try:
        rows = db.query("SELECT topic FROM arena_knowledge")
        for row in rows:
            used.add(str(row.get("topic", "")))
    except Exception:  # noqa: BLE001 - no table yet on fresh test DBs
        pass
    return used


def _recent_categories(db: Any, n: int = 3) -> list[str]:
    """Categories served most recently (anti-repeat window)."""
    if db is None:
        return []
    try:
        row = db.query_one(
            "SELECT value FROM kv_store WHERE key = 'arena.recent_categories'")
        if row:
            data = json.loads(row["value"]).get("cats", [])
            return [str(c) for c in data[:n] if str(c)]
    except Exception:  # noqa: BLE001
        pass
    return []


def _record_category(db: Any, category: str) -> None:
    if db is None:
        return
    try:
        cats = [category] + [c for c in _recent_categories(db, 9)
                             if c != category]
        with db.transaction():
            db.execute(
                """INSERT INTO kv_store (key, value, kind, updated_at)
                   VALUES (?, ?, 'json', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                ("arena.recent_categories", json.dumps({"cats": cats[:10]}),
                 time.time()),
            )
    except Exception:  # noqa: BLE001
        pass


# ── topic anti-repeat window (persisted, survives restarts) ────────────────

def anti_repeat_window(db: Any = None,
                       default: int = DEFAULT_ANTI_REPEAT) -> int:
    """Effective anti-repeat window: kv override → default. 0 disables."""
    if db is not None:
        try:
            row = db.query_one(
                "SELECT value FROM kv_store WHERE key = ?",
                (_ANTI_REPEAT_WINDOW_KEY,))
            if row:
                return max(0, min(int(json.loads(row["value"]).get("window", default)),
                                  10000))
        except Exception:  # noqa: BLE001
            pass
    return max(0, int(default))


def set_anti_repeat_window(db: Any, n: int) -> bool:
    """Persist the anti-repeat window (sessions without a topic repeat)."""
    if db is None:
        return False
    try:
        with db.transaction():
            db.execute(
                """INSERT INTO kv_store (key, value, kind, updated_at)
                   VALUES (?, ?, 'json', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (_ANTI_REPEAT_WINDOW_KEY, json.dumps({"window": max(0, int(n))}),
                 time.time()),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def recent_topics(db: Any, n: int | None = None) -> list[str]:
    """Most-recently served topic texts, newest first (persisted)."""
    if db is None:
        return []
    if n is None:
        n = anti_repeat_window(db)
    try:
        row = db.query_one(
            "SELECT value FROM kv_store WHERE key = ?", (_RECENT_TOPICS_KEY,))
        if row:
            data = json.loads(row["value"]).get("topics", [])
            return [str(t) for t in data[:max(0, int(n))] if str(t)]
    except Exception:  # noqa: BLE001
        pass
    return []


def _record_topic(db: Any, topic: str, window: int) -> None:
    if db is None or window <= 0 or not topic:
        return
    try:
        topics = [topic] + [t for t in recent_topics(db, window - 1)
                            if t != topic]
        with db.transaction():
            db.execute(
                """INSERT INTO kv_store (key, value, kind, updated_at)
                   VALUES (?, ?, 'json', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (_RECENT_TOPICS_KEY, json.dumps({"topics": topics[:window]}),
                 time.time()),
            )
    except Exception:  # noqa: BLE001
        pass


def clear_recent_topics(db: Any) -> int:
    """Clear the anti-repeat window (the reshuffle). Returns topics cleared."""
    if db is None:
        return 0
    cleared = len(recent_topics(db, 10000))
    try:
        with db.transaction():
            db.execute("DELETE FROM kv_store WHERE key = ?",
                       (_RECENT_TOPICS_KEY,))
    except Exception:  # noqa: BLE001
        pass
    return cleared


def _weighted_categories(profile: dict[str, float] | None,
                         exclude: set[str]) -> list[tuple[str, float]]:
    cats = [(c, float((profile or {}).get(c, 1.0)) * _pack_weight(c))
            for c in all_categories() if c not in exclude]
    if not cats:  # anti-repeat window ate everything — fall back to all
        cats = [(c, float((profile or {}).get(c, 1.0)) * _pack_weight(c))
                for c in all_categories()]
    return [(c, max(w, 0.01)) for c, w in cats]


def _pick_weighted(rng: random.Random,
                   weighted: list[tuple[str, float]]) -> str:
    total = sum(w for _, w in weighted)
    roll = rng.random() * total
    acc = 0.0
    for cat, w in weighted:
        acc += w
        if roll <= acc:
            return cat
    return weighted[-1][0]


def sample_topic(db: Any = None, category: str | None = None,
                 rng: random.Random | None = None,
                 profile: dict[str, float] | None = None,
                 difficulty: int | None = None,
                 anti_repeat: int | None = None) -> tuple[str, str]:
    """Pick ``(category, topic)``.

    * ``category`` forces one; otherwise the pick is weighted by
      ``profile`` (category → weight; see ``activity.interest_profile``)
      multiplied by any pack weights.
    * The last few served categories are deprioritized (anti-repeat),
      and no *topic* repeats within the last ``anti_repeat`` served
      (default: ``kv arena.anti_repeat_window`` → 10; 0 disables).
    * Topics already digested into ``arena_knowledge`` are skipped
      (bounded retry).
    * ``difficulty`` (1–3) restricts the pool when given.
    * When the anti-repeat window leaves no eligible topic anywhere,
      the window reshuffles (clears) once and sampling retries —
      recent history is persisted in ``kv_store``, so it survives
      restarts.
    """
    rng = rng or random.Random()
    used = _used_topics(db)
    window = (max(0, int(anti_repeat)) if anti_repeat is not None
              else anti_repeat_window(db))
    recent_cats = set(_recent_categories(db))
    recent = set(recent_topics(db, window)) if window > 0 else set()

    def pool_for(cat: str) -> list[dict[str, Any]]:
        pool = topics_in(cat)
        if difficulty in (1, 2, 3):
            # Strict: a category with nothing at this grade is skipped
            # for this round (the retry loop picks another category).
            pool = [e for e in pool if int(e.get("d", 2)) == difficulty]
        return [e for e in pool
                if topic_text(e) not in used and topic_text(e) not in recent]

    # Retry bound: a nearly-exhausted bank still terminates.
    for _ in range(128):
        if category:
            cat = category
        else:
            cat = _pick_weighted(rng, _weighted_categories(profile, recent_cats))
        pool = pool_for(cat)
        if not pool:
            if category:
                break  # forced category is dry — reshuffle/fallback below
            recent_cats.discard(cat)  # let the weighted pick try elsewhere
            continue
        entry = rng.choice(pool)
        text = topic_text(entry)
        _record_category(db, cat)
        _record_topic(db, text, window)
        return cat, text

    # The window ate the whole eligible pool → reshuffle once and retry.
    if window > 0 and clear_recent_topics(db) > 0:
        return sample_topic(db=db, category=category, rng=rng,
                            profile=profile, difficulty=difficulty,
                            anti_repeat=window)

    # Fallback: anything unused, anywhere (never fails callers).
    for cat in all_categories():
        pool = pool_for(cat)
        if pool:
            entry = rng.choice(pool)
            text = topic_text(entry)
            _record_category(db, cat)
            _record_topic(db, text, window)
            return cat, text
    cat = category or CATEGORIES[0]
    _record_category(db, cat)
    return cat, "general computing"


def surprise_topic(db: Any = None, rng: random.Random | None = None,
                   seed: int | None = None,
                   anti_repeat: int | None = None) -> tuple[str, str]:
    """Pure random topic — ignores the interest profile.

    ``seed`` makes the surprise reproducible. The topic anti-repeat
    window still applies (pass ``anti_repeat=0`` to disable).
    """
    if seed is not None:
        rng = random.Random(seed)
    return sample_topic(db=db, rng=rng or random.Random(), profile=None,
                        anti_repeat=anti_repeat)


# ── display ───────────────────────────────────────────────────────────────

_DIFF_LABEL = {1: "easy", 2: "medium", 3: "deep"}


def topics_table(profile: dict[str, float] | None = None,
                 db: Any = None) -> str:
    """The full topic table, multi-line, with counts and difficulty spread.

    When a profile is given, the top-weighted categories are flagged —
    that's the personalization made visible.
    """
    stats = category_stats()
    order = sorted(stats, key=lambda c: -float((profile or {}).get(c, 1.0)))
    pack_bits = []
    for name in topic_packs():
        n = sum(len(v) for v in _PACKS[name].topics.values())
        pack_bits.append(f"{name} ({n})")
    lines = [f"arena topic tables — {bank_size()} topics, "
             f"{len(stats)} categories, packs: {', '.join(pack_bits)}:"]
    for cat in order:
        st = stats[cat]
        star = " ★" if profile and float(profile.get(cat, 1.0)) >= 2.0 else ""
        lines.append(
            f"  {cat:<11} {st['topics']:>3} topics "
            f"(easy {st['easy']} · med {st['medium']} · deep {st['deep']}){star}")
    if db is not None:
        used = _used_topics(db)
        lines.append(f"digested so far: {len(used)} — the sampler skips those.")
    lines.append("usage: /arena topics <category> · /arena run [topic] · "
                 "/arena surprise [seed]")
    return "\n".join(lines)


def category_table(category: str, limit: int = 20) -> str:
    """Drill into one category: topics with difficulty grades."""
    entries = topics_in(category)
    if not entries:
        return (f"no category {category!r} — pick from: "
                + ", ".join(CATEGORIES))
    lines = [f"{category} — {len(entries)} topics:"]
    for e in entries[:max(1, limit)]:
        d = int(e.get("d", 2))
        label = _DIFF_LABEL.get(min(3, max(1, d)), "?")
        lines.append(f"  [{label:<6}] {topic_text(e)}")
    if len(entries) > limit:
        lines.append(f"  … and {len(entries) - limit} more")
    return "\n".join(lines)
