"""Hash cracker — offline password-hash verification engine.

Ported and integrated from the user's engine.py ("BruteForce Engine"):
multi-modal offline cracking of md5 · sha1 · sha256 · sha512 · ntlm with a
mutation rule engine, a character-level Markov-chain generator, a 4-phase
hybrid pipeline, a threaded worker pool, and checkpoint/resume.

Scope
-----
This tool is strictly offline. It never opens a socket and never touches a
live system: it hashes candidate plaintexts and compares digests against
hash strings *you already hold*. Only run it on hashes you are authorized
to test — your own data exports, CTF challenges, or systems covered by
explicit written permission.

Modes
-----
brute     → exhaustive combinatorial over a charset (guaranteed, slow)
wordlist  → dictionary attack from a file
markov    → statistically generated candidates (human-like patterns)
hybrid    → wordlist → mutations → Markov → brute (recommended)
"""

from __future__ import annotations
import re

import hashlib
import itertools
import json
import os
import queue
import random
import string
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Set

from ..core.policy import Capability

__all__ = [
    "ALGORITHMS",
    "Engine",
    "CrackResult",
    "MarkovChain",
    "auto_detect",
    "benchmark",
    "crack_hash",
    "get_hash_fn",
    "mutate",
    "register",
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. HASH LAYER — md5 · sha1 · sha256 · sha512 · ntlm
# ═══════════════════════════════════════════════════════════════════════════

_M32 = 0xFFFFFFFF
_R2 = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
_R3 = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
_S1 = (3, 7, 11, 19)
_S2 = (3, 5, 9, 13)
_S3 = (3, 9, 11, 15)


def _rotl(x: int, n: int) -> int:
    x &= _M32
    return ((x << n) | (x >> (32 - n))) & _M32


def md4(data: bytes) -> str:
    """MD4 (RFC 1320), pure Python.

    OpenSSL 3 removed md4 from hashlib, and NTLM is defined as
    MD4(UTF-16LE(password)) — so we carry our own copy.
    """

    def F(x: int, y: int, z: int) -> int:
        return (x & y) | (~x & z)

    def G(x: int, y: int, z: int) -> int:
        return (x & y) | (x & z) | (y & z)

    def H(x: int, y: int, z: int) -> int:
        return x ^ y ^ z

    msg = bytearray(data)
    rem = len(msg) % 64
    pad_len = (55 - rem) % 64  # zeros after the 0x80 marker (total ≡ 56 mod 64)
    msg += b"\x80" + b"\x00" * pad_len + (len(data) * 8).to_bytes(8, "little")

    a, b, c, d = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
    for off in range(0, len(msg), 64):
        x = [int.from_bytes(msg[off + 4 * i : off + 4 * i + 4], "little") for i in range(16)]
        A, B, C, D = a, b, c, d
        for i in range(48):  # MD4 has three rounds of 16 steps (RFC 1320 App. A)
            if i < 16:
                k, f, g, s = i, F, 0, _S1[i % 4]
            elif i < 32:
                j = i - 16
                k, f, g, s = _R2[j], G, 0x5A827999, _S2[j % 4]
            else:
                j = i - 32
                k, f, g, s = _R3[j], H, 0x6ED9EBA1, _S3[j % 4]
            t = _rotl((A + f(B, C, D) + x[k] + g) & _M32, s)
            A, B, C, D = D, t, B, C
        a = (a + A) & _M32
        b = (b + B) & _M32
        c = (c + C) & _M32
        d = (d + D) & _M32
    return (
        a.to_bytes(4, "little")
        + b.to_bytes(4, "little")
        + c.to_bytes(4, "little")
        + d.to_bytes(4, "little")
    ).hex()


ALGORITHMS: dict[str, Callable[[str], str]] = {
    "md5": lambda p: hashlib.md5(p.encode()).hexdigest(),
    "sha1": lambda p: hashlib.sha1(p.encode()).hexdigest(),
    "sha256": lambda p: hashlib.sha256(p.encode()).hexdigest(),
    "sha512": lambda p: hashlib.sha512(p.encode()).hexdigest(),
    "ntlm": lambda p: md4(p.encode("utf-16-le")),
}
_LEN_MAP: dict[int, str] = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}


def auto_detect(h: str) -> Optional[str]:
    """Guess algorithm from hash string length."""
    return _LEN_MAP.get(len(h.strip().lower()))


def get_hash_fn(algo: str) -> Callable[[str], str]:
    fn = ALGORITHMS.get(algo.lower())
    if not fn:
        raise ValueError(f"Unknown algorithm '{algo}'. Valid: {sorted(ALGORITHMS)}")
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# 2. RULE ENGINE — mutations applied to every base candidate
# ═══════════════════════════════════════════════════════════════════════════

_LEET = str.maketrans("aeiost", "@3!0$+")


def mutate(word: str) -> List[str]:
    """Common password transformation rules: leet-speak, capitalization,
    reversal, suffixes/prefixes, doubling, year appending."""
    variants = [
        word,  # original
        word.capitalize(),  # Password
        word.upper(),  # PASSWORD
        word.lower(),  # password
        word[::-1],  # drowssap
        word.translate(_LEET),  # p@$$w0rd
        word + "1",
        word + "123",
        word + "!",
        word + "2024",
        word + "2025",
        "1" + word,
        (word[0].upper() + word[1:] + "1") if word else word,
        word * 2,
    ]
    seen: Set[str] = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. MARKOV CHAIN GENERATOR
#    Character-level n-gram model trained on common password patterns.
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_CORPUS: List[str] = [
    "password", "qwerty", "letmein", "dragon", "master", "monkey",
    "sunshine", "princess", "football", "baseball", "welcome", "shadow",
    "superman", "batman", "trustno1", "abc123", "pass123", "admin",
    "login", "hello", "iloveyou", "mustang", "access", "hunter",
    "michael", "jessica", "charlie", "thomas", "pepper", "andrew",
    "tigger", "cheese", "buster", "liverpool", "thunder", "ranger",
    "silver", "cookie", "matrix", "secret", "ninja",
]


class MarkovChain:
    """Order-n character Markov chain for password-pattern generation.

    Training builds transition tables from password patterns; generation
    walks those tables stochastically, producing strings that follow
    learned character sequences — much closer to human choices than
    exhaustive combinatorics.
    """

    def __init__(self, order: int = 2) -> None:
        self.order = order
        self.chain: dict[str, list[str]] = defaultdict(list)
        self.starts: list[str] = []

    def train(self, corpus: List[str]) -> "MarkovChain":
        pad = "^" * self.order
        for word in corpus:
            s = pad + word + "$"
            self.starts.append(s[: self.order + 1])
            for i in range(len(s) - self.order):
                self.chain[s[i : i + self.order]].append(s[i + self.order])
        return self

    def train_default(self) -> "MarkovChain":
        return self.train(_DEFAULT_CORPUS)

    def train_from_file(self, path: str, limit: int = 100_000) -> "MarkovChain":
        words: List[str] = []
        with open(path, errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                w = line.strip()
                if 3 <= len(w) <= 16:
                    words.append(w)
        return self.train(words)

    def _gen_one(self, min_len: int, max_len: int) -> Optional[str]:
        start = random.choice(self.starts)
        word = start[self.order :]  # strip padding, keep first char
        state = start[1:]  # sliding window
        for _ in range(max_len):
            choices = self.chain.get(state)
            if not choices:
                break
            ch = random.choice(choices)
            if ch == "$":
                break
            word += ch
            state = state[1:] + ch
        return word if min_len <= len(word) <= max_len else None

    def generate(self, n: int = 10_000, min_len: int = 4, max_len: int = 10) -> List[str]:
        seen: Set[str] = set()
        results: List[str] = []
        attempts = 0
        while len(results) < n and attempts < n * 20:
            attempts += 1
            w = self._gen_one(min_len, max_len)
            if w and w not in seen:
                seen.add(w)
                results.append(w)
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. RESULTS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CrackResult:
    targets: list[str]
    found: dict[str, str] = field(default_factory=dict)  # hash → plaintext
    remaining: list[str] = field(default_factory=list)
    algo: str = ""
    mode: str = ""
    tested: int = 0
    elapsed: float = 0.0
    capped: bool = False  # True when max_candidates stopped the run early
    backend: str = "python"  # "c" when the native hash layer did the work
    learned: bool = False  # True when results were written back (known-
                           # hash chain + learned corpus, wave 77)

    @property
    def rate(self) -> float:
        return self.tested / max(self.elapsed, 1e-9)

    @property
    def cracked(self) -> int:
        return len(self.found)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "mode": self.mode,
            "backend": self.backend,
            "cracked": self.cracked,
            "targets": len(self.targets),
            "found": self.found,
            "remaining": self.remaining,
            "tested": self.tested,
            "elapsed": round(self.elapsed, 3),
            "rate": round(self.rate, 1),
            "capped": self.capped,
            "complete": not self.remaining,
            "learned": self.learned,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class Engine:
    """Multi-modal offline hash cracker.

    Architecture
    ────────────
    Main thread      → pumps candidates into a bounded work queue (backpressure)
    Worker threads   → drain the queue, hash each candidate, test targets
    Found lock       → thread-safe result recording; stops all on full crack

    Hybrid pipeline order (most efficient → least efficient)
    ────────────────────────────────────────────────────────
    Phase 1  raw wordlist             (direct dictionary hits)
    Phase 2  wordlist + mutations     (leet, caps, suffixes, …)
    Phase 3  Markov + mutations       (statistically generated)
    Phase 4  pure brute force         (exhaustive fallback)
    """

    def __init__(
        self,
        targets: List[str],
        algo: str = "",
        mode: str = "hybrid",
        charset: str = "",
        min_len: int = 1,
        max_len: int = 6,
        threads: int = 0,
        wordlist: str = "",
        use_rules: bool = True,
        use_markov: bool = True,
        checkpoint: str = "",
        max_candidates: int = 0,
        quiet: bool = False,
        db: Any = None,
    ) -> None:
        self._targets_orig = set(t.lower().strip() for t in targets if t.strip())
        # wave 77 learning loop: when a database is present the corpus is
        # BUILTIN + LEARNED (every cracked/taught word, migration 28) and
        # every fresh crack is written straight back into the store.
        self.db = db
        try:
            from ..core.corpus import BUILTIN_WORDS, base_words

            self._corpus: List[str] = base_words(db)
        except Exception:  # noqa: BLE001 — a broken learned layer must
            #              #  never kill cracking; fall back to builtins
            from ..core.corpus import BUILTIN_WORDS as _BW

            self._corpus = list(_BW)
        self.remaining: Set[str] = set(self._targets_orig)
        # per-target algorithm: a batch may mix md5/sha1/sha256/… and each
        # digest is then tested under its OWN algorithm (length implies it).
        self._target_algos: dict[str, str] = {
            t: (algo.lower() if algo else auto_detect(t) or "md5")
            for t in self._targets_orig}
        self.algo = (algo.lower() if algo
                     else (auto_detect(targets[0]) if targets else "md5")
                     or "md5")
        # The inner loop is the speed-critical path in the whole project.
        # The C layer is offered only where it genuinely wins (see
        # native.loader.FAST_ALGOS — today that's NTLM, since OpenSSL 3
        # removed MD4 from hashlib); everything else stays on hashlib.
        # one hash fn per DISTINCT algorithm present in the batch (usually
        # just one); the common single-algo case stays a single call.
        self._hash_fns: dict[str, Callable[[str], str]] = {}
        self.hash_backend = "python"
        algos = sorted(set(self._target_algos.values()))
        if self.algo not in algos:
            algos.append(self.algo)  # pinned algo (or an empty-target run)
        for a in algos:
            fn = get_hash_fn(a)
            try:
                from .native.loader import fast_hash_fn

                fast = fast_hash_fn(a)
                if fast is not None:
                    fn = fast
                    self.hash_backend = "c"
            except Exception:  # noqa: BLE001 - native is an accelerator, not a dependency
                pass
            self._hash_fns[a] = fn
        self._hash = self._hash_fns[self.algo]
        self.mode = mode
        self.charset = charset or (string.ascii_lowercase + string.digits)
        self.min_len = min_len
        self.max_len = max_len
        self.threads = threads or (os.cpu_count() or 4)
        self.wordlist = wordlist
        self.use_rules = use_rules
        self.use_markov = use_markov
        self.checkpoint_path = checkpoint
        self.max_candidates = max_candidates
        self.quiet = quiet
        self.found: dict[str, str] = {}  # hash → plaintext
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._capped = False
        self._interrupted = False
        self.t0 = time.time()
        self._tested = 0
        self._load_checkpoint()

    # ── hash testing ────────────────────────────────────────────────────────
    def _record(self, h: str, candidate: str) -> bool:
        with self._lock:
            if h in self.remaining:  # re-check under lock
                self.remaining.discard(h)
                self.found[h] = candidate
                if not self.quiet:
                    print(f"\n  ✓   {h[:48]}  →  '{candidate}'")
                # wave 77: a cracked plaintext is written back into the
                # learned corpus immediately, so the next run finds it in
                # one hash.  Must never let the learning layer take down
                # the crack (the result is already recorded above).
                if self.db is not None:
                    try:
                        from ..core.corpus import learn_phrase

                        learn_phrase(self.db, candidate, source="cracked")
                    except Exception:  # noqa: BLE001 — learning is best-effort
                        pass
                if not self.remaining:
                    self._stop.set()  # all targets cracked
                return True
        return False

    def _test(self, candidate: str) -> bool:
        # single algorithm (the common case): one hash, one lookup
        if len(self._hash_fns) == 1:
            h = self._hash(candidate)
            return self._record(h, candidate) if h in self.remaining else False
        # mixed-algorithm batch: test the candidate under each distinct
        # algorithm that still has live targets
        hit = False
        for a in sorted(self._hash_fns):
            if not any(t in self.remaining
                       for t, ta in self._target_algos.items()
                       if ta == a):
                continue
            h = self._hash_fns[a](candidate)
            if h in self.remaining and self._record(h, candidate):
                hit = True
        return hit

    # ── candidate generators ────────────────────────────────────────────────
    def _gen_brute(self) -> Generator[str, None, None]:
        for length in range(self.min_len, self.max_len + 1):
            for combo in itertools.product(self.charset, repeat=length):
                if self._stop.is_set():
                    return
                yield "".join(combo)

    def _gen_wordlist(self) -> Generator[str, None, None]:
        p = Path(self.wordlist)
        if not p.exists():
            if not self.quiet:
                print(f"  [!] Wordlist not found: {self.wordlist}")
            return
        with p.open(errors="ignore") as f:
            for line in f:
                if self._stop.is_set():
                    return
                yield line.rstrip("\n")

    def _markov(self) -> "MarkovChain":
        # Markov trained on the default corpus PLUS the effective corpus
        # (learned words shape the next generation of candidates too).
        mc = MarkovChain()
        mc.train(_DEFAULT_CORPUS + [w for w in self._corpus
                                    if w not in _DEFAULT_CORPUS])
        return mc

    def _gen_markov(self) -> Generator[str, None, None]:
        mc = self._markov()
        if self.wordlist and Path(self.wordlist).exists():
            mc.train_from_file(self.wordlist)
        for w in mc.generate(20_000, self.min_len, self.max_len):
            if self._stop.is_set():
                return
            yield w
            if self.use_rules:
                yield from mutate(w)

    def _gen_builtin(self) -> Generator[str, None, None]:
        # Phase 1 (no explicit wordlist): the EFFECTIVE corpus raw —
        # builtins + everything the system has cracked or been taught
        # (wave 77 learning loop; a cracked word is found in one hash).
        for w in self._corpus:
            if self._stop.is_set():
                return
            yield w

    def _gen_builtin_rules(self) -> Generator[str, None, None]:
        # Phase 2 (no explicit wordlist): effective corpus x rockyou-style
        # rules — leet, caps, reversals, digit/leetspeak suffixes, years.
        from ..core.corpus import rule_stream

        for cand in rule_stream(self._corpus, stop=self._stop):
            if self._stop.is_set():
                return
            yield cand

    def _gen_hybrid(self) -> Generator[str, None, None]:
        seen: Set[str] = set()

        def unique(gen: Generator) -> Generator[str, None, None]:
            for item in gen:
                if self._stop.is_set():
                    return
                if item not in seen:
                    seen.add(item)
                    yield item

        have_wordlist = bool(self.wordlist) and Path(self.wordlist).exists()
        # Phase 1 — raw wordlist (explicit file) or the bundled corpus
        if have_wordlist:
            yield from unique(self._gen_wordlist())
        else:
            yield from unique(self._gen_builtin())
        # Phase 2 — wordlist + mutations (file) or corpus + rules (bundled)
        if self.use_rules and have_wordlist:
            def _wl_mutated() -> Generator[str, None, None]:
                with Path(self.wordlist).open(errors="ignore") as f:
                    for line in f:
                        yield from mutate(line.rstrip("\n"))

            yield from unique(_wl_mutated())
        elif self.use_rules:
            yield from unique(self._gen_builtin_rules())
        # Phase 3 — Markov chain + mutations
        if self.use_markov:
            mc = self._markov()
            if have_wordlist:
                mc.train_from_file(self.wordlist)

            def _markov_with_rules() -> Generator[str, None, None]:
                for w in mc.generate(20_000, self.min_len, self.max_len):
                    yield w
                    if self.use_rules:
                        yield from mutate(w)

            yield from unique(_markov_with_rules())
        # Phase 4 — brute force fallback
        yield from unique(self._gen_brute())

    def _pipeline(self) -> Generator[str, None, None]:
        dispatch = {
            "brute": self._gen_brute,
            "wordlist": self._gen_wordlist,
            "markov": self._gen_markov,
            "hybrid": self._gen_hybrid,
        }
        for candidate in dispatch.get(self.mode, self._gen_hybrid)():
            yield candidate
            if self.max_candidates and self._count >= self.max_candidates:
                self._capped = True
                return

    def _pipeline_counted(self) -> Generator[str, None, None]:
        self._count = 0
        for candidate in self._pipeline():
            self._count += 1
            yield candidate

    # ── worker threads ──────────────────────────────────────────────────────
    def _worker(self, q: "queue.Queue") -> None:
        # A worker runs until its poison pill. After _stop is set it keeps
        # draining the queue (discarding, not hashing) so the producer can
        # never block forever on a full queue with no consumers left.
        while True:
            try:
                candidate = q.get(timeout=0.3)
            except queue.Empty:
                continue
            if candidate is None:  # poison pill — shut down
                q.task_done()
                break
            if self._stop.is_set():
                q.task_done()  # already cracked — discard the rest
                continue
            self._test(candidate)
            self._tested += 1
            q.task_done()

    # ── run ─────────────────────────────────────────────────────────────────
    def run(self) -> CrackResult:
        """Candidates flow into the work queue one at a time; the bounded
        queue is the backpressure (a full queue blocks the generator), so a
        crack found early stops the whole pipeline promptly — including in
        markov/brute phases that would otherwise run away."""
        self.t0 = time.time()
        if not self.quiet:
            self._header()
        wq: "queue.Queue" = queue.Queue(maxsize=self.threads * 8)
        pool = [threading.Thread(target=self._worker, args=(wq,), daemon=True) for _ in range(self.threads)]
        for t in pool:
            t.start()
        interrupted = False
        try:
            for candidate in self._pipeline_counted():
                if self._stop.is_set():
                    break
                wq.put(candidate)  # blocks while the workers catch up
                if not self.quiet and self._count % 2000 == 0:
                    self._render()
                    sys.stdout.flush()
            for _ in pool:
                wq.put(None)
            wq.join()
        except KeyboardInterrupt:
            interrupted = True
            self._interrupted = True
            self._stop.set()
            if not self.quiet:
                print("\n\n  [!] Interrupted — saving checkpoint…")
        if interrupted and self.checkpoint_path:
            self._save_checkpoint()
        return self._result()

    def _result(self) -> CrackResult:
        elapsed = time.time() - self.t0
        return CrackResult(
            targets=sorted(self._targets_orig),
            found=dict(self.found),
            remaining=sorted(self.remaining),
            algo=self.algo,
            mode=self.mode,
            tested=self._tested,
            elapsed=elapsed,
            capped=self._capped or self._interrupted,
            backend=getattr(self, "hash_backend", "python"),
        )

    # ── benchmark ───────────────────────────────────────────────────────────
    def benchmark(self, secs: float = 1.0) -> list[tuple[str, int, int]]:
        """Raw hashing throughput for every supported algorithm."""
        rows: list[tuple[str, int, int]] = []
        sample = "BenchmarkString42!"
        for name, fn in ALGORITHMS.items():
            n = 0
            deadline = time.time() + secs
            while time.time() < deadline:
                fn(sample)
                n += 1
            rows.append((name, int(n / secs), n))
        return rows

    # ── checkpoint ──────────────────────────────────────────────────────────
    def _save_checkpoint(self) -> None:
        data = {
            "tested": self._tested,
            "found": self.found,
            "remaining": list(self.remaining),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        if not self.quiet:
            print(f"  Checkpoint saved → {self.checkpoint_path}")

    def _load_checkpoint(self) -> None:
        p = Path(self.checkpoint_path) if self.checkpoint_path else None
        if p is None or not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            self.found = data.get("found", {})
            self.remaining = set(data.get("remaining", self._targets_orig))
            self._tested = data.get("tested", 0)
            if not self.quiet and self._tested:
                print(f"  Resumed — {self._tested:,} candidates already tested\n")
        except Exception:
            pass

    # ── display ────────────────────────────────────────────────────────────
    def _header(self) -> None:
        print(f"""
  ┌──────────────────────────────────────────────────┐
  │  Hash Cracker  ·  offline, no network            │
  └──────────────────────────────────────────────────┘
  Algorithm : {self.algo.upper()}
  Mode      : {self.mode}
  Threads   : {self.threads}
  Targets   : {len(self._targets_orig)}
  Charset   : {self.charset[:24]}{"…" if len(self.charset) > 24 else ""}
  Range     : {self.min_len}–{self.max_len} chars
""")

    def _render(self, width: int = 30) -> None:
        rate = self._tested / max(time.time() - self.t0, 1e-9)
        filled = (self._tested % 10_000) * width // 10_000
        bar = "█" * filled + "░" * (width - filled)
        print(
            f"\r  [{bar}]  {rate:>11,.0f} H/s  tested {self._tested:>11,}  "
            f"cracked {len(self.found)}/{len(self._targets_orig)}  "
            f"{time.time() - self.t0:>6.1f}s  ",
            end="",
        )

    def summary(self) -> str:
        lines = [
            f"  Targets cracked : {len(self.found)} / {len(self._targets_orig)}",
            f"  Candidates tried: {self._tested:,}",
            f"  Elapsed         : {time.time() - self.t0:.2f}s",
        ]
        if self.found:
            lines.append("\n  ── Cracked ──────────────────────────── ")
            for h, p in self.found.items():
                lines.append(f"    {h[:48]}  →  '{p}'")
        elif not self.found:
            lines.append("\n  No targets cracked in this run.")
        if self._capped:
            lines.append(f"  (stopped at max_candidates={self.max_candidates:,})")
        return "\n".join(lines)


def crack_hash(
    target: str,
    *,
    algo: str = "",
    mode: str = "hybrid",
    wordlist: str = "",
    charset: str = "",
    min_len: int = 1,
    max_len: int = 6,
    max_candidates: int = 200_000,
    threads: int = 0,
    quiet: bool = True,
    db: Any = None,
    use_markov: bool = True,
) -> CrackResult:
    """Crack one digest — or a batch (comma/newline separated) — with sane
    agent-friendly defaults: capped, quiet, checkpoint disabled.

    * **batch** — every candidate is hashed once and checked against ALL
      targets simultaneously (the parallel multi-digest attack).
    * **known-hash chain** — with ``db``, each target is first checked
      against the built-in common-secrets table AND the learned store
      (every digest this system has ever cracked).  Hits short-circuit to
      an instant zero-candidate result with ``backend="known-hash"``.
    * **bundled corpus** — hybrid mode with no wordlist runs the packaged
      dictionary + rockyou-style mutation rules before Markov/brute.
    """
    # ``target`` may be one digest or a batch (comma/newline separated) —
    # the Engine tests every candidate against ALL of them at once
    # (one hash computation, N digests), which is the parallel multi-digest
    # attack: a corpus pass is N times as valuable when it serves N hashes.
    targets = [t.strip().lower()
               for t in re.split(r"[,\s]+", (target or "").strip())
               if t.strip()]
    if not targets:
        raise ValueError("crack_hash needs at least one target digest")
    found: dict[str, str] = {}
    remaining: list[str] = []
    if db is not None:
        from ..core.decoder import known_hash_lookup_chained

        for t in targets:
            hit = known_hash_lookup_chained(db, t)
            if hit is not None:
                found[t] = hit["plaintext"]
                # the known-hash chain is a *hit*, not a fresh crack —
                # still make sure the word is in the learned layer
                try:
                    from ..core.corpus import learn_phrase

                    learn_phrase(db, hit["plaintext"],
                                 source="known-hash")
                except Exception:  # noqa: BLE001
                    pass
            else:
                remaining.append(t)
    else:
        remaining = list(targets)
    if not remaining:
        return CrackResult(
            targets=targets, found=found, remaining=[],
            algo="", mode="known-hash", tested=0, elapsed=0.0,
            capped=False, backend="known-hash",
        )
    engine = Engine(
        remaining,
        algo=algo,
        mode=mode,
        charset=charset,
        min_len=min_len,
        max_len=max_len,
        threads=threads,
        wordlist=wordlist,
        max_candidates=max_candidates,
        quiet=quiet,
        db=db,
        use_markov=use_markov,
    )
    res = engine.run()
    if found:
        res.found = {**found, **res.found}
        res.targets = targets
    # chain EVERY fresh crack into the known-hash store, not just
    # tool-layer calls — CLI, agent and library users all get the
    # "cracked once, known forever" guarantee from the same place.
    if db is not None and res.found:
        from ..core.decoder import learn_hash

        for digest, plain in res.found.items():
            learn_hash(db, digest, plain, algorithm=res.algo,
                       source=f"hash_crack:{res.backend}")
        res.learned = True
    return res


def benchmark(secs: float = 1.0) -> list[tuple[str, int, int]]:
    return Engine([], algo="md5").benchmark(secs)


def generate_combinations(
    out_path: str,
    *,
    charset: str = "",
    min_len: int = 1,
    max_len: int = 6,
    max_candidates: int = 5_000_000,
) -> int:
    """Write the brute-force combination space to a wordlist file.

    The PDF's generator.py: materialize charset x lengths so a second
    run can consume it as a wordlist (``nm crack --mode wordlist
    --wordlist FILE``).  Writes are chunked so a 5M-entry file does not
    hold a giant string in memory.
    """
    chars = charset or (string.ascii_lowercase + string.digits)
    count = 0
    fp = Path(out_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        buf: list[str] = []
        for length in range(max(1, min_len), max(1, max_len) + 1):
            for combo in itertools.product(chars, repeat=length):
                if count >= max_candidates:
                    break
                buf.append("".join(combo))
                count += 1
                if len(buf) >= 10_000:
                    f.write("\n".join(buf) + "\n")
                    buf = []
            if count >= max_candidates:
                break
        if buf:
            f.write("\n".join(buf) + "\n")
    return count


# ═══════════════════════════════════════════════════════════════════════════
# 6. REGISTRY TOOL
# ═══════════════════════════════════════════════════════════════════════════


def register(registry: Any) -> None:
    """Attach the hash cracker to a registry."""
    context = registry.context

    @registry.register(
        "hash_crack",
        description=(
            "Crack password hashes offline (md5/sha1/sha256/sha512/ntlm) by "
            "testing candidate plaintexts against the digest(s) you provide. "
            "target= may hold ONE digest or a BATCH (comma-separated) — "
            "every candidate is tested against all of them at once "
            "(parallel multi-digest attack). Hybrid mode = the EFFECTIVE "
            "corpus (bundled + everything previously cracked or taught) "
            "+ rockyou-style rules, then Markov, then brute. Every fresh "
            "crack is auto-learned back into the corpus so the same word "
            "is found in one hash next time (use the `corpus` tool to "
            "teach/list/forget words). Strictly offline — no network. Use "
            "only for hashes you are authorized to test: your own data "
            "exports, CTF challenges, or explicit written permission."
        ),
        capability=Capability.EXEC_CODE,
    )
    def hash_crack(
        target: str,
        *,
        algo: str = "",
        mode: str = "hybrid",
        wordlist: str = "",
        charset: str = "",
        min_len: int = 1,
        max_len: int = 6,
        max_candidates: int = 200_000,
    ) -> dict[str, Any]:
        wl = wordlist
        if wl and context is not None and not Path(wl).is_absolute():
            from .filesystem import safe_path

            wl = str(safe_path(context, wl))
        db = getattr(context, "db", None) if context is not None else None
        result = crack_hash(
            target,
            algo=algo,
            mode=mode,
            wordlist=wl,
            charset=charset,
            min_len=min_len,
            max_len=max_len,
            max_candidates=max_candidates,
            db=db,
        )
        out = result.as_dict()
        # Note: the known-hash chain + learned-corpus write-back now
        # happens inside crack_hash itself (single source of truth), so
        # CLI, agent and library users all chain identically.  `out`
        # already carries `learned` from result.as_dict().
        return out

    @registry.register(
        "corpus",
        description=(
            "The cracking corpus + its learning loop. action=stats (size "
            "of bundled + learned layers) | learn (word|phrase — teach a "
            "word so future cracks find it instantly) | list (learned "
            "words) | forget (word). Cracked plaintexts are learned "
            "automatically; use learn to add known-good words manually. "
            "Wordlist words shorter than 3 chars are rejected (brute "
            "force owns the short space)."
        ),
        capability=Capability.MEM_WRITE,
    )
    def corpus(
        action: str = "stats",
        word: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return {"ok": False, "error": "no database for the learned "
                    "corpus"}
        from ..core.corpus import (corpus_stats, forget_word,
                                   learn_word, learned_words)

        if action == "learn":
            added = learn_word(db, word, source="manual")
            st = corpus_stats(db)
            return {"ok": True, "added": added, "word": word.lower(),
                    "learned_words": st.get("learned_words", 0),
                    "effective_base_words": st.get("effective_base_words", 0)}
        if action == "forget":
            gone = forget_word(db, word)
            return {"ok": True, "removed": gone, "word": word.lower()}
        if action == "list":
            words = learned_words(db, max(1, int(limit or 100)))
            return {"ok": True, "learned": len(words), "words": words}
        # default: stats
        st = corpus_stats(db)
        st["ok"] = True
        return st
