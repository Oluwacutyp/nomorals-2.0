"""Bundled cracking corpus + rockyou-style mutation rules.

This is the dictionary layer of the hash crackers:

* :data:`BUILTIN_WORDS` — a real, bundled base wordlist (top-of-the-top
  common passwords, common phrases, surnames/given names, tech terms).
  It ships inside the package so offline cracking works with zero setup.
* :func:`apply_rules` — one word → its rule-expanded variants (leet,
  caps, reversals, digit suffixes, years, doubling…), the same shape as
  hashcat rule files (``+123``, ``l0``…), just expressed in Python.
* :func:`rule_stream` — the deterministic candidate generator: every base
  word × every rule, in a stable order (raw first, then cheap rules),
  deduplicated.  Feeds ``Engine`` phase 1.5 so hybrid mode gets a real
  dictionary before Markov/brute force.
* :func:`export_wordlist` — materialize the expanded corpus to a file
  (``nm crack --corpus-out FILE``) for reuse by any cracker.

The corpus is deliberately real, not a placeholder: it is trained on the
same distribution actual passwords follow (short, leet-flavored,
year-suffixed, name-based), which is exactly what raises live-crack hit
rates on the digests the decoder hands us.
"""
from __future__ import annotations

import itertools
import re
import time
from typing import Any, Generator, Iterable, List, Sequence

__all__ = [
    "BUILTIN_WORDS", "LEET", "YEARS", "RULE_ORDER",
    "apply_rules", "rule_stream", "export_wordlist", "corpus_stats",
    "learn_word", "learn_phrase", "learned_words", "forget_word",
    "base_words",
]


#: Leet map (a subset of the classic substitutions people actually use).
LEET: dict[str, str] = {
    "a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "b": "8",
    "g": "9", "t": "+", "l": "|",
}

#: Years most common in passwords (recent ± a working lifetime).
YEARS: tuple[int, ...] = tuple(range(2035, 1969, -1))


def _leet_all(word: str) -> str:
    low = word.lower()
    out = []
    for ch in low:
        out.append(LEET.get(ch, ch))
    return "".join(out)


def _leet_tail(word: str, n: int = 2) -> str:
    """Leet only the last n leetable chars (password1337 style)."""
    low = word.lower()
    idxs = [i for i, ch in enumerate(low) if ch in LEET][-n:]
    chars = list(low)
    for i in idxs:
        chars[i] = LEET[chars[i]]
    return "".join(chars)


def apply_rules(word: str, rule: str | None = None) -> List[str]:
    """Expand one base word into variants.

    ``rule=None`` yields the full, ordered rule set (deduplicated, the
    original word first).  ``rule`` names a single transform — see
    :data:`RULE_ORDER` — for callers that want one pass only.
    """
    w = (word or "").strip()
    if not w:
        return []
    low = w.lower()

    def y(years: Sequence[int] = YEARS) -> Generator[str, None, None]:
        for y_ in years:
            yield f"{low}{y_}"

    RULES: dict[str, List[Any]] = {
        "raw": [low],
        "cap": [low.capitalize()],
        "upper": [low.upper()],
        "rev": [low[::-1]],
        "rev_cap": [low[::-1].capitalize()],
        "leet": [_leet_all(low)],
        "leet_tail": [_leet_tail(low)],
        "leet_cap": [_leet_all(low.capitalize())],
        "d1": [f"{low}1", f"{low}12", f"{low}123", f"{low}1234",
               f"{low}!", f"{low}!!", f"{low}!"],
        "p1": ["1" + low, "!" + low],
        "double": [low * 2],
        "year": list(y()),
        "year_rev": [f"{y_}" + low for y_ in YEARS[:16]],
        "cap_year": [f"{low.capitalize()}{y_}" for y_ in YEARS[:16]],
        "year_cap": [f"{y_}{low.capitalize()}" for y_ in YEARS[:16]],
    }
    # canonical order: cheapest/most-likely first
    RULE_ORDER: list[str] = ["raw", "cap", "d1", "year", "upper",
                             "leet_tail", "leet", "rev", "double",
                             "p1", "cap_year", "year_rev", "leet_cap",
                             "rev_cap", "year_cap"]
    if rule is not None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r} (known: "
                             f"{', '.join(RULE_ORDER)})")
        return list(RULES[rule])
    seen: set[str] = set()
    out: List[str] = []
    for name in RULE_ORDER:
        for v in RULES[name]:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return out


#: The stable rule order (re-exported for introspection / tests).
RULE_ORDER: tuple[str, ...] = ("raw", "cap", "d1", "year", "upper",
                               "leet_tail", "leet", "rev", "double",
                               "p1", "cap_year", "year_rev", "leet_cap",
                               "rev_cap", "year_cap")


def rule_stream(words: Iterable[str], *, rules: Sequence[str] | None = None,
                max_per_word: int = 0,
                stop: Any = None) -> Generator[str, None, None]:
    """Deterministic candidate generator: each base word × each rule.

    ``max_per_word`` caps variants per word (0 = all).  ``stop`` is an
    optional :class:`threading.Event` for early termination (the engine
    sets it on a crack).  Never yields duplicates across the whole stream.
    """
    wanted = list(rules) if rules else list(RULE_ORDER)
    seen: set[str] = set()
    for word in words:
        word = (word or "").strip()
        if not word:
            continue
        variants = apply_rules(word)
        if rules:
            # keep only the requested rules, in canonical order
            keep: set[str] = set()
            for r in wanted:
                keep.update(v for v in apply_rules(word, r))
            variants = [v for v in variants if v in keep]
        if max_per_word and len(variants) > max_per_word:
            variants = variants[:max_per_word]
        for v in variants:
            if stop is not None and stop.is_set():
                return
            if v not in seen:
                seen.add(v)
                yield v


# ── the bundled base wordlist ────────────────────────────────────────────────
# Real top-of-the-top common passwords + common phrases + names + tech terms.
# Curated (public-domain / widely-published frequency lists), not scraped.
_BUILTIN_RAW: tuple[str, ...] = (
    # classic top-100 style
    "password", "123456", "12345678", "123456789", "12345", "1234",
    "123450", "1234567", "111111", "000000", "password1", "iloveyou",
    "qwerty", "abc123", "123123", "admin", "welcome", "monkey", "dragon",
    "letmein", "login", "princess", "football", "shadow", "sunshine",
    "master", "666666", "superman", "654321", "1qaz2wsx", "qazwsx",
    "123qwe", "killer", "trustno1", "jordan", "harley", "ranger",
    "iwantu", "0000000", "silver", "samantha", "azerty", "austin",
    "thomas", "hockey", "hunter", "fuckme", "25802", "batman", "tigger",
    "charlie", "robert", "thunder", "snoopy", "1q2w3e", "george", "computer",
    "michael", "jessica", "pepper", "zxcvbnm", "daniel", "access",
    "123abc", "horse", "mustang", "112233", "gauntlet", "starwars",
    "asshole", "buster", "andrew", "super", "yankees", "1qazxsw2",
    "jordan23", "whatever", "zaq12wsx", "maggie", "hello", "hotstuff",
    "dallas", "joshua", "cheese", "summer", "corvette", "taylor",
    "matrix", "cookie", "bigdog", "sexy", "david", "javier", "987654321",
    "aa123456", "anthony", "121212", "1q2w3e4r", "eagle", "321123",
    "ender", "test", "freedom", "hello1", "aaron", "merlin", "diamond",
    "1234qwer", "11111", "00000", "111111111", "12341234", "55555",
    "159753", "7777777", "1234567890", "0123456789", "232323", "999999",
    "147258", "159357", "135790", "11223344", "131313", "13579",
    "987654", "98765432", "777777", "654321", "54321", "789456123",
    "123321", "122334", "123654", "147896325", "246810", "1357911",
    "137911", "119377", "18273645", "258369", "369258", "852456",
    "5263748", "987123", "456789", "123789", "12987", "741852963",
    "159951", "753951", "589632", "369852", "147852", "28465", "8642",
    # common phrases / leetspeak
    "letmein", "iloveyou", "iloveyou1", "iloveyou2", "passw0rd",
    "p@ssw0rd", "password!", "password123", "pass123", "admin123",
    "admin!", "root", "toor", "changeme", "changeme123", "default",
    "welcome1", "welcome123", "qwerty123", "qwer1234", "asdf1234",
    "zxcvbn", "zxc123", "asdfgh", "1q2w3e4r5t", "1qaz2wsx3edc",
    "qazwsxedcrfv", "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "hunter2", "hunter22", "letmein1", "letmein123", "monkey123",
    "dragon123", "princess1", "princess123", "shadow1", "sunshine1",
    "master1", "superman1", "batman1", "superman123", "batman123",
    "football1", "hockey1", "baseball1", "yankees1", "cowboys",
    "cowboys1", "dolphins", "eagles", "lakers", "celtics", "warriors",
    "steelers", "packers", "jaguars", "panthers", "thunder1", "ranger1",
    "tigger1", "cookie1", "cookie123", "matrix1", "cheese1", "cheese123",
    "cookie!", "letmein!", "welcome!", "princess!", "sunshine!",
    "password!", "secret", "secret1", "secret123", "secret!",
    "secure", "secure1", "safety", "safety1", "pass123!", "temp",
    "temp123", "demo", "demo123", "test123", "test1234", "testing",
    "testing123", "sample", "example", "placeholder", "changeme!",
    "temp1", "tmp", "temp1234", "asdf123", "qwer123", "qwert", "asdfg",
    "zxcvb", "poiu", "mnbvc", "lkjh", "gfdsa", "rewq", "tgbn",
    # names (western + nigerian, both common in real creds)
    "michael", "jessica", "jennifer", "joshua", "jordan", "david",
    "andrew", "thomas", "charlie", "robert", "javier", "anthony",
    "austin", "dallas", "maggie", "hello", "aaron", "merlin", "diamond",
    "jordan23", "michael1", "jessica1", "jennifer1", "joshua1",
    "david1", "andrew1", "thomas1", "charlie1", "robert1", "javier1",
    "anthony1", "austin1", "dallas1", "maggie1", "aaron1", "merlin1",
    "oluwaseun", "adebayo", "oluwadamilare", "chinedu", "emeka",
    "chiamaka", "tolu", "femi", "kelechi", "obi", "ngozi", "ada",
    "yemi", "tunde", "seun", "bode", "kunle", "funke", "ola", "ife",
    "olamide", "wale", "tobi", "zainab", "amara", "chika", "nkechi",
    "seun123", "yemi123", "tunde123", "ada123", "ada.lovelace",
    "oluwaseun123", "chinedu123", "emeka123", "ngozi123",
    # tech / service defaults
    "admin", "administrator", "root", "toor", "pi", "raspberry",
    "raspberry1", "ubuntu", "ubuntu1", "debian", "fedora", "kali",
    "nomorals", "nomorals123", "codebeast", "codebeast123", "devon",
    "devon123", "bot", "bot123", "api", "api123", "apikey", "token",
    "token123", "secretkey", "private", "private123", "public",
    "localhost", "server", "server1", "database", "db", "dbadmin",
    "postgres", "mysql", "mongo", "redis", "nginx", "apache",
    "wordpress", "wp", "admin@site", "support", "help", "info",
    "sales", "billing", "office", "company", "business", "startup",
    "launch", "deploy", "release", "prod", "production", "staging",
    "sandbox", "debug", "release123", "build", "build123", "main",
    "master1", "develop", "feature", "hotfix", "patch", "update",
    "upgrade", "backup", "restore", "archive", "snapshot", "mirror",
    "proxy", "gateway", "router", "switch", "firewall", "gateway1",
    "cloud", "cloud1", "serverroom", "datacenter", "rack", "cluster",
    "node", "node1", "instance", "container", "docker", "k8s", "kub",
    "pipeline", "workflow", "cron", "daemon", "service", "job", "task",
)

#: De-duplicated, stable base wordlist (the actual corpus root).
BUILTIN_WORDS: tuple[str, ...] = tuple(dict.fromkeys(
    w.strip().lower() for w in _BUILTIN_RAW if w.strip()))


def corpus_stats(db: Any = None) -> dict[str, Any]:
    """Size/shape of the corpus (for help + tests).

    With a ``db`` the learned layer (wave 77) is counted: ``learned_words``
    = words cracked or taught since install, ``effective_base_words`` =
    what the cracker actually consumes (builtins + learned, deduped)."""
    t0 = time.perf_counter()
    expanded = set()
    for w in BUILTIN_WORDS:
        expanded.update(apply_rules(w))
    out: dict[str, Any] = {
        "base_words": len(BUILTIN_WORDS),
        "rules": len(RULE_ORDER),
        "expanded_unique": len(expanded),
        "build_seconds": round(time.perf_counter() - t0, 3),
        "years": [YEARS[0], YEARS[-1]],
    }
    if db is not None:
        learned = learned_words(db)
        out["learned_words"] = len(learned)
        out["effective_base_words"] = len(base_words(db))
    return out


def export_wordlist(path: str, *, max_per_word: int = 0,
                    stop: Any = None) -> int:
    """Write the fully-expanded corpus to ``path`` (one candidate/line).

    Returns the number of lines written."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for cand in rule_stream(BUILTIN_WORDS, max_per_word=max_per_word,
                                stop=stop):
            fh.write(cand + "\n")
            n += 1
    return n


def builtin_wordlist_path() -> str | None:
    """Materialize the base wordlist (not expanded) to a stable path and
    return it, so the Engine can treat the corpus like any file
    wordlist."""
    import os
    import tempfile

    d = os.path.join(tempfile.gettempdir(), "nomorals")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "builtin_wordlist.txt")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            for w in BUILTIN_WORDS:
                fh.write(w + "\n")
    return p


# ── the learning loop (wave 77) ──────────────────────────────────────────────
# The corpus is no longer frozen: every plaintext the crackers break is
# written back into a persistent learned layer (``corpus_words``, migration
# 28) and merged into ``base_words(db)`` — the same list the Engine's
# phases 1/2 and the Markov trainer consume.  A password cracked once is
# never searched for twice.

#: A learned word is at least this long — single/double chars belong to
#: brute force, not the wordlist, and "learning" them is just noise.
_MIN_LEARNED_LEN = 3


def learn_word(db: Any, word: str, source: str = "manual") -> bool:
    """Teach the corpus one new word (idempotent, case-insensitive).

    ``source`` records how it got here: ``'cracked'`` (auto-learned from a
    successful hash crack) or ``'manual'`` (an explicit ``nm crack
    --corpus-learn`` / tool call) or any descriptive tag.

    Returns True when the word is NEW to the store, False when it was
    already known (or rejected: empty / shorter than 3 chars — the word
    list is for words; brute force owns the short space).

    Words that carry punctuation also teach their alphanumeric core
    (``"Sunshine2019!"`` → also ``sunshine2019``): the rule set only
    ADDS punctuation, it never strips it, so the core is the form the
    candidates actually take."""
    if db is None:
        return False
    w = (word or "").strip().lower()
    if len(w) < _MIN_LEARNED_LEN:
        return False
    targets = [w]
    core = re.sub(r"[^a-z0-9]+", "", w)
    if core and core != w and len(core) >= _MIN_LEARNED_LEN:
        targets.append(core)
    added = False
    for t in targets:
        cursor = db.execute(
            "INSERT OR IGNORE INTO corpus_words (word, source, ts) "
            "VALUES (?,?,?)", (t, (source or "")[:64], time.time()))
        # rowcount=1 → this call ADDED it; 0 → OR IGNORE fired (known)
        if getattr(cursor, "rowcount", 0) > 0:
            added = True
    return added


def learn_phrase(db: Any, plaintext: str,
                 source: str = "cracked") -> list[str]:
    """Learn everything worth learning from a cracked plaintext.

    Adds the full plaintext (when word-shaped, ≤64 chars) plus every
    embedded run of word characters ≥3 long, so "Sunshine2019!" teaches
    both ``sunshine2019`` (the exact cracked form — instant match next
    time) and ``sunshine`` / ``sunshine2019`` parts.  Returns the words
    actually added (deduped, in insertion order)."""
    if db is None or not plaintext:
        return []
    p = plaintext.strip()
    candidates: list[str] = []
    if 2 < len(p) <= 64:
        candidates.append(p)
    for part in re.split(r"[^a-z0-9]+", p.lower()):
        if len(part) >= _MIN_LEARNED_LEN:
            candidates.append(part)
    added: list[str] = []
    for cand in dict.fromkeys(c.lower() for c in candidates):
        if learn_word(db, cand, source):
            added.append(cand)
    return added


def learned_words(db: Any, limit: int = 100_000) -> list[str]:
    """All learned words, most recent last (stable, lowercased)."""
    if db is None:
        return []
    rows = db.query(
        "SELECT word FROM corpus_words ORDER BY ts DESC, word LIMIT ?",
        (max(0, int(limit)),))
    return [r["word"] for r in rows]


def forget_word(db: Any, word: str) -> bool:
    """Remove a word from the learned layer. True if it was there.

    (The alphanumeric core is NOT touched — only the exact word, so
    forgetting a crack never erases the normalization of it.)"""
    if db is None:
        return False
    w = (word or "").strip().lower()
    cursor = db.execute("DELETE FROM corpus_words WHERE word=?", (w,))
    return bool(getattr(cursor, "rowcount", 0) > 0)


def base_words(db: Any = None) -> list[str]:
    """The EFFECTIVE base wordlist: builtins + learned, deduped, stable.

    Builtins first (their curated order), learned most-recent last.  This
    is the single merge point the crackers consume — hashcrack Engine
    phases 1 and 2 and the Markov trainer — so a word cracked or taught
    once is available to every future attack in every process."""
    words: list[str] = []
    seen: set[str] = set()
    for w in BUILTIN_WORDS:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            words.append(lw)
    for w in learned_words(db):
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words
