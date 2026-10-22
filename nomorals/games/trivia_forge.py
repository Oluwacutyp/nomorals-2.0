"""TriviaForge: dynamic, personalized trivia question generation.

The old trivia bank was 143 static Q&A pairs — the same questions every
game, forever. This module replaces that with:

* **LLM generation** — fresh questions on any topic, at any difficulty,
  via the ``suggest`` bridge (same pattern as GameMaster). The model
  gets the topic, difficulty, and anti-repeat list; it returns JSON.
* **Template forge fallback** — seeded composable questions from
  structured facts when no model is available. Never stalls.
* **Interest-profile topics** — question topics are sampled from the
  player's interest profile (arena activity), so trivia feels personal.
* **Persisted anti-repeat** — served question hashes live in KVStore;
  nothing repeats within the window (default 50).

**M3 — the verified Library bank.** LLM-invented questions are never
checked against anything, so the forge also draws from a *bank* of
questions generated from passages Devon actually holds (WisdomKeeper
digests, keeper material, BookForge holdings, research claims) and
admitted only through fail-closed mechanical gates: the evidence
quote must sit near-verbatim in the source passage, the answer must
appear in the evidence, and no distractor may appear in the evidence
or equal the answer. Unverifiable candidates are discarded, never
served — and the discard is counted, so bank stats stay honest.
Maths enters through both doors (owner's have-it-all correction):
procedural :class:`~nomorals.games.math_class.MathClass` problems
(verified by construction, worked solution attached) *and* brain-posed
word problems whose claimed answers are cross-verified by an
independent computation before banking. ``deal()`` mixes the sources:
library/bank questions take at least half a dealt set when the bank
has fresh stock, the generative forge serves the rest, and the
template forge remains the last-resort filler. A background refiller
(:meth:`TriviaBank.tick`, wired as a stage of the wisdom organ's
tick and topped up opportunistically at deal time) keeps the bank
stocked. Banked questions carry provenance — the ladder shows
"📚 From her library: …" after the answer.

Usage::

    forge = TriviaForge(suggest=model_suggest_fn, db=db)
    questions = forge.deal(count=8, difficulty="normal",
                           profile=interest_profile(db))
    # → [(question, answer), ...]
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import random
import re
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = ["TriviaForge", "TRIVIA_TOPICS", "ANTI_REPEAT_DEFAULT",
           "TriviaLadder", "LADDER_PRIZES", "LADDER_DIFFICULTIES",
           "LADDER_SAFE_HAVENS", "LIFELINES", "build_options",
           "TriviaBank", "LibraryQuestionSource",
           "BANK_TARGET", "BANK_CAP", "LIBRARY_SHARE",
           "verify_library_candidate", "verify_llm_maths",
           "safe_eval_arith", "maths_distractors", "source_tag",
           "render_bank_status"]

#: question topic pools, weighted by generality. The interest profile
#: re-weights these per player.
TRIVIA_TOPICS: dict[str, list[str]] = {
    "general": ["science", "history", "geography", "sports", "arts",
                "technology", "nature", "food"],
    "nigeria": ["nigerian history", "nollywood", "afrobeats",
                "nigerian geography", "nigerian politics"],
    "tech": ["programming", "ai", "gadgets", "internet culture",
             "video games"],
    "pop": ["movies", "music", "celebrities", "tv shows", "memes"],
    "brain": ["riddles", "logic puzzles", "wordplay", "math trivia"],
}

ANTI_REPEAT_DEFAULT = 50

_JSON_RE = re.compile(r"\[.*\]", re.S)


def _hash_q(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


# ── M3: the verified bank — gates, maths verification, provenance ─────
#
# Everything in this section is mechanical and fail-closed: a
# candidate that cannot be checked is not "probably fine", it is
# discarded, and the discard is counted in the bank stats.  The gates
# prove *grounding* (the answer really sits in a passage she holds,
# or a computation really produces it) — not that a question is
# elegant.  That bias is deliberate (design §6.4).

#: the refiller's stock goal, and the hard cap above it.
BANK_TARGET = 200
BANK_CAP = 260
#: opportunistic deal-time top-up fires below this fresh-stock level.
BANK_LOW_WATER = 24
#: share of a dealt set the bank takes when it has fresh stock.
LIBRARY_SHARE = 0.5
#: persisted ring of banked/served question hashes (restart dedupe).
BANK_HASH_RING = 500

_BANK_KV = "trivia_forge.bank"
_BANK_STATS_KV = "trivia_forge.bank_stats"

_WORD_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_text(s: Any) -> str:
    """Lowercase, punctuation-stripped, whitespace-collapsed text —
    the comparison space for every gate below."""
    return " ".join(_WORD_RE.findall(str(s or "").lower()))


def _norm_answer(s: Any) -> str:
    """Answer-space normalization: like :func:`_norm_text` but a
    leading minus survives — for numeric answers the sign is part
    of the answer ("-4" is a classic distractor FOR "4", not a
    duplicate of it)."""
    t = _norm_text(s)
    raw = str(s or "").strip()
    if t and raw.startswith(("-", "−", "–")):
        return "-" + t
    return t


def _token_containment(needle: str, haystack: str) -> float:
    """Fraction of ``needle``'s tokens present (as a multiset) in
    ``haystack`` — the near-verbatim tolerance for evidence quotes
    whose whitespace/punctuation drifted in generation."""
    n_tokens = needle.split()
    if not n_tokens:
        return 0.0
    pool: dict[str, int] = {}
    for tok in haystack.split():
        pool[tok] = pool.get(tok, 0) + 1
    hit = 0
    for tok in n_tokens:
        if pool.get(tok, 0) > 0:
            pool[tok] -= 1
            hit += 1
    return hit / len(n_tokens)


def verify_library_candidate(passage: str,
                             cand: dict[str, Any]) -> tuple[bool, str]:
    """The design's fail-closed admission gates for a library question.

    (a) the evidence quote must be a near-verbatim substring of the
        source passage (normalized containment; token containment
        ≥ 0.92 as the near-verbatim tolerance);
    (b) the normalized answer must appear in the evidence quote;
    (c) exactly 3 usable distractors: all distinct, none equal to the
        answer, none containing the answer, none contained in the
        evidence quote (a distractor the evidence also supports
        would make the question ambiguous).
    Returns (ok, reason) — reason is "" when ok.
    """
    q = str(cand.get("question") or "").strip()
    a = str(cand.get("answer") or "").strip()
    ev = str(cand.get("evidence_quote") or "").strip()
    if not q or not a:
        return False, "missing question/answer"
    if not ev:
        return False, "missing evidence"
    n_pass, n_ev, n_ans = _norm_text(passage), _norm_text(ev), \
        _norm_text(a)
    if len(n_ev) < 12:
        return False, "evidence too short"
    if n_ev not in n_pass and _token_containment(n_ev, n_pass) < 0.92:
        return False, "evidence not in source"
    if n_ans not in n_ev:
        return False, "answer not in evidence"
    ds = cand.get("distractors")
    if not isinstance(ds, list) or len(ds) < 3:
        return False, "needs 3 distractors"
    picked: list[str] = []
    seen: set[str] = set()
    na_ans = _norm_answer(a)
    for d in ds[:3]:
        nd = _norm_text(d)
        if not nd:
            return False, "empty distractor"
        if _norm_answer(d) == na_ans or na_ans in _norm_answer(d):
            return False, "distractor equals/contains answer"
        if nd in seen:
            return False, "duplicate distractor"
        if nd in n_ev:
            return False, "distractor appears in evidence"
        seen.add(nd)
        picked.append(str(d).strip())
    cand["distractors"] = picked
    return True, ""


_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def safe_eval_arith(expr: str) -> float | None:
    """Evaluate a pure arithmetic expression with stdlib ``ast``.

    Only numeric literals, ``+ - * / // % **`` and unary signs are
    allowed — no names, calls, or attributes, so this is an
    independent computation path, never code execution.  Returns
    None for anything else, for division by zero, and for results
    that are non-finite or absurdly large.
    """
    try:
        tree = ast.parse(str(expr or ""), mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or \
                    not isinstance(node.value, (int, float)):
                raise ValueError("non-numeric literal")
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and \
                isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent too large")
            v = _BINOPS[type(node.op)](left, right)
            if not math.isfinite(v) or abs(v) > 1e15:
                raise ValueError("result out of range")
            return v
        raise ValueError("disallowed node")

    try:
        return ev(tree)
    except Exception:  # noqa: BLE001 - any failure == not computable
        return None


def _expr_literals(expr: str) -> list[float]:
    try:
        tree = ast.parse(str(expr or ""), mode="eval")
    except (SyntaxError, ValueError):
        return []
    out: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and \
                isinstance(node.value, (int, float)) and \
                not isinstance(node.value, bool):
            out.append(float(node.value))
    return out


def verify_llm_maths(question: str, answer: Any,
                     expression: str) -> tuple[bool, str, float | None]:
    """Cross-verify a brain-posed maths problem by computation.

    The claimed answer is believed only when an independent path
    reproduces it: the stated ``expression`` must (1) evaluate under
    :func:`safe_eval_arith`, (2) use only numbers that literally
    appear in the question text (the computation is grounded in the
    posed problem, not invented beside it), and (3) evaluate to the
    claimed answer within tolerance.  Returns (ok, reason, computed).
    """
    from .math_class import parse_answer

    if not str(question or "").strip():
        return False, "missing question", None
    if not str(expression or "").strip():
        return False, "missing expression", None
    claimed = parse_answer(answer)
    if claimed is None or isinstance(claimed, complex):
        return False, "answer not numeric", None
    computed = safe_eval_arith(expression)
    if computed is None:
        return False, "expression not computable", None
    q_nums = []
    for raw in _NUM_RE.findall(str(question)):
        try:
            q_nums.append(float(raw))
        except ValueError:
            continue
    for lit in _expr_literals(expression):
        if not any(abs(lit - qn) < 1e-9 for qn in q_nums):
            return False, "expression number not in question", None
    tol = max(1e-6, abs(computed) * 1e-6)
    if abs(computed - float(claimed)) > tol:
        return False, "answer mismatch", computed
    return True, "", computed


def _fmt_num(value: float, like: str = "") -> str:
    """Format ``value`` in the style of ``like`` (int vs decimals)."""
    if "." in str(like or ""):
        places = len(str(like).split(".", 1)[1])
        return f"{value:.{min(places, 4)}f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def maths_distractors(value: float, answer_text: str,
                      rng: random.Random) -> list[str]:
    """Programmatic near-miss distractors for a numeric answer:
    off-by-one/two, sign flip, digit transposition, ±10% — the
    MathClass-style wrong answers a student actually gives."""
    ans = str(answer_text or "").strip()
    pct_hi, pct_lo = value * 1.1, value * 0.9
    if float(value).is_integer():
        pct_hi, pct_lo = round(pct_hi), round(pct_lo)
    cands: list[float] = [value + 1, value - 1, value + 2, value - 2,
                          -value, pct_hi, pct_lo,
                          value + 10, value - 10]
    if float(value).is_integer() and abs(value) >= 10:
        s = str(int(abs(value)))
        if len(s) >= 2:
            t = s[:-2] + s[-1] + s[-2]
            try:
                cands.append(float(t) * (1 if value >= 0 else -1))
            except ValueError:
                pass
    rng.shuffle(cands)
    out: list[str] = []
    for c in cands:
        txt = _fmt_num(c, ans)
        if txt != ans and txt not in out and abs(c - value) > 1e-12:
            out.append(txt)
        if len(out) == 3:
            break
    while len(out) < 3:  # pragma: no cover - arithmetic always yields 3
        out.append(_fmt_num(value + 100 + len(out), ans))
    return out


def source_tag(entry: dict[str, Any]) -> str:
    """The provenance line a served bank question carries in play.

    Library questions trace to a work she holds ("📚 From her
    library: Meditations — WisdomKeeper"); maths questions carry
    their worked solution instead (rendered on a miss), so they
    return "" here.
    """
    src = entry.get("source") or {}
    if entry.get("kind") == "library" or \
            src.get("type") in ("wisdom", "book", "research"):
        return f"📚 From her library: {src.get('label') or 'her library'}"
    return ""


def render_bank_status(stats: dict[str, Any]) -> str:
    """The bank/status view: stock by source + honest admission math."""
    lines = [f"📚 trivia bank — {stats.get('stock', 0)} verified "
             f"questions banked (target {stats.get('target', BANK_TARGET)})"]
    by_kind = stats.get("by_kind") or {}
    if by_kind:
        lines.append("  stock: " + " · ".join(
            f"{k} {v}" for k, v in sorted(by_kind.items())))
    by_source = stats.get("by_source") or {}
    if by_source:
        lines.append("  sources: " + " · ".join(
            f"{k} {v}" for k, v in sorted(by_source.items())))
    cand = int(stats.get("candidates", 0))
    adm = int(stats.get("admitted", 0))
    dis = int(stats.get("discarded", 0))
    if cand:
        rate = round(100.0 * adm / cand, 1) if cand else 0.0
        lines.append(f"  admission: {adm}/{cand} ({rate}%) — "
                     f"{dis} discarded, never served")
        reasons = stats.get("discard_reasons") or {}
        if reasons:
            lines.append("  discards: " + " · ".join(
                f"{k} ×{v}" for k, v in
                sorted(reasons.items(), key=lambda kv: -kv[1])[:4]))
    last = float(stats.get("last_tick", 0) or 0)
    if last:
        lines.append(f"  refills: {int(stats.get('ticks', 0))} "
                     f"(last {time.strftime('%Y-%m-%d %H:%M', time.gmtime(last))} UTC)")
    return "\n".join(lines)


class TriviaForge:
    """Deal fresh trivia questions. LLM-first, template fallback."""

    def __init__(
        self,
        suggest: Callable[[str], str] | None = None,
        db: Any = None,
        seed: int | None = None,
        anti_repeat: int = ANTI_REPEAT_DEFAULT,
        context: Any = None,
    ) -> None:
        self._suggest = suggest
        self._kv = KVStore(db) if db is not None else None
        #: the runtime context (engine-passed) — the M3 bank reads the
        #: Library and the Teachings ledger through it.  Without one
        #: the forge is exactly the pre-M3 forge: generative + template.
        self._context = context
        self.rng = random.Random(seed)
        self.anti_repeat = anti_repeat
        self._lock = threading.RLock()
        self._seen: list[str] = []
        self._load_seen()
        self._bank_obj: TriviaBank | None = None
        self._ledger_obj: Any = None
        self._last_autotick = 0.0

    # ── anti-repeat ──────────────────────────────────────────────────
    def _load_seen(self) -> None:
        if self._kv is None:
            return
        try:
            data = self._kv.get("trivia_forge.seen") or {}
            self._seen = list(data.get("hashes", []))[-self.anti_repeat:]
        except Exception:  # noqa: BLE001
            self._seen = []

    def _save_seen(self) -> None:
        if self._kv is None:
            return
        try:
            self._kv.set("trivia_forge.seen",
                         {"hashes": self._seen[-self.anti_repeat:],
                          "updated": time.time()})
        except Exception:  # noqa: BLE001
            pass

    def _is_fresh(self, question: str) -> bool:
        return _hash_q(question) not in self._seen

    def _mark_seen(self, question: str) -> None:
        with self._lock:
            self._seen.append(_hash_q(question))
            self._seen = self._seen[-self.anti_repeat:]
            self._save_seen()

    # ── topic sampling ───────────────────────────────────────────────
    def sample_topics(self, count: int,
                      profile: dict[str, float] | None = None) -> list[str]:
        """Pick topics, weighted by the player's interest profile."""
        all_topics = [t for sub in TRIVIA_TOPICS.values() for t in sub]
        if not profile:
            return [self.rng.choice(all_topics) for _ in range(count)]
        # map profile categories onto trivia topics loosely
        weights: list[float] = []
        for t in all_topics:
            w = 1.0
            tl = t.lower()
            for cat, score in profile.items():
                if cat.lower() in tl or tl in cat.lower():
                    w += float(score)
            weights.append(w)
        return self.rng.choices(all_topics, weights=weights, k=count)

    # ── LLM generation ───────────────────────────────────────────────
    def _ask_model(self, topic: str, difficulty: str,
                   count: int) -> list[tuple[str, str]]:
        if self._suggest is None:
            return []
        prompt = (
            f"Generate {count} {difficulty} trivia questions about {topic}. "
            "Return ONLY a JSON array of [question, answer] pairs, e.g. "
            '[["What is the capital of France?", "Paris"]]. '
            "Short answers (1-3 words). No duplicates. No explanations."
        )
        try:
            raw = (self._suggest(prompt) or "").strip()
        except Exception:  # noqa: BLE001
            return []
        m = _JSON_RE.search(raw)
        if not m:
            return []
        try:
            pairs = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
        out = []
        batch_hashes: set[str] = set()
        for p in pairs:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                q, a = str(p[0]).strip(), str(p[1]).strip()
                h = _hash_q(q)
                if q and a and h not in batch_hashes and self._is_fresh(q):
                    batch_hashes.add(h)
                    out.append((q, a))
        return out

    # ── template forge (offline fallback) ────────────────────────────
    #: (template, answer_fn_key) — the answer is computed from the seed data
    _TEMPLATES: tuple[tuple[str, str], ...] = (
        ("What is {n1} + {n2}?", "add"),
        ("What is {n1} × {n2}?", "mul"),
        ("How many days are in {n3} weeks?", "weeks"),
        ("What is the {ord} planet from the sun?", "planet"),
        ("How many sides does a {shape} have?", "sides"),
        ("What year is {n4} years after 2000?", "year"),
        ("Spell the word '{word}' backwards.", "reverse"),
        ("How many letters are in the word '{word}'?", "letters"),
    )

    _PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter",
                "Saturn", "Uranus", "Neptune")
    _ORDS = ("first", "second", "third", "fourth", "fifth",
             "sixth", "seventh", "eighth")
    _SHAPES = {"triangle": 3, "square": 4, "pentagon": 5,
               "hexagon": 6, "octagon": 8, "decagon": 10}
    _WORDS = ("python", "rocket", "garden", "thunder", "piano",
              "bridge", "candle", "forest", "anchor", "comet")

    def _forge_one(self, difficulty: str) -> tuple[str, str]:
        tpl, kind = self.rng.choice(self._TEMPLATES)
        if kind == "add":
            n1 = self.rng.randint(11, 99)
            n2 = self.rng.randint(11, 99)
            return tpl.format(n1=n1, n2=n2), str(n1 + n2)
        if kind == "mul":
            hi = 12 if difficulty in ("easy", "normal") else 20
            n1 = self.rng.randint(3, hi)
            n2 = self.rng.randint(3, hi)
            return tpl.format(n1=n1, n2=n2), str(n1 * n2)
        if kind == "weeks":
            n3 = self.rng.randint(2, 12)
            return tpl.format(n3=n3), str(n3 * 7)
        if kind == "planet":
            i = self.rng.randrange(8)
            return tpl.format(ord=self._ORDS[i]), self._PLANETS[i]
        if kind == "sides":
            shape = self.rng.choice(list(self._SHAPES))
            return tpl.format(shape=shape), str(self._SHAPES[shape])
        if kind == "year":
            n4 = self.rng.randint(1, 30)
            return tpl.format(n4=n4), str(2000 + n4)
        if kind == "reverse":
            word = self.rng.choice(self._WORDS)
            return tpl.format(word=word), word[::-1]
        if kind == "letters":
            word = self.rng.choice(self._WORDS)
            return tpl.format(word=word), str(len(word))
        return "What is 2 + 2?", "4"

    def _forge(self, count: int, difficulty: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        batch_hashes: set[str] = set()
        attempts = 0
        while len(out) < count and attempts < count * 10:
            attempts += 1
            q, a = self._forge_one(difficulty)
            h = _hash_q(q)
            if h not in batch_hashes and self._is_fresh(q):
                batch_hashes.add(h)
                out.append((q, a))
        return out

    def _interest_profile(self) -> dict[str, float] | None:
        """Build the player's interest profile from arena activity."""
        if self._kv is None:
            return None
        try:
            db = self._kv._db  # internal but same package family
            from ..agents.arena.activity import interest_profile
            return interest_profile(db)
        except Exception:  # noqa: BLE001
            return None

    # ── M3: the bank, the ledger, the mix ────────────────────────────
    @property
    def bank(self) -> "TriviaBank":
        """The verified question bank (lazy, one per forge)."""
        if self._bank_obj is None:
            self._bank_obj = TriviaBank(self)
        return self._bank_obj

    def _ledger(self) -> Any:
        """The Teachings ledger for this forge's context (or None)."""
        if self._context is None:
            return None
        if self._ledger_obj is None:
            try:
                from ..agents.teachings import TeachingsLedger
                self._ledger_obj = TeachingsLedger(self._context)
            except Exception:  # noqa: BLE001 - ledger is an enhancement
                return None
        return self._ledger_obj

    def _maybe_autotick(self) -> None:
        """Opportunistic bank top-up at deal time (context-bound,
        brain-bound forges only, ≤ once per 10 min, fail-soft): the
        wisdom-tick stage does the heavy refilling; this keeps a
        games-only host's bank from running dry between ticks."""
        if self._context is None or self._suggest is None:
            return
        now = time.time()
        if now - self._last_autotick < 600:
            return
        self._last_autotick = now
        try:
            if self.bank.fresh_count(set(self._seen)) < BANK_LOW_WATER:
                self.bank.tick(max_passages=3, max_maths=6,
                               max_maths_llm=2)
        except Exception:  # noqa: BLE001 - a dry bank degrades the mix
            _log.debug("trivia bank autotick failed", exc_info=True)

    def _blocked_themes(self, entries: list[dict[str, Any]]) -> set[str]:
        """Themes currently inside the Teachings ledger's 48h
        cross-surface cooldown — trivia respects them at deal time
        when a ledger is attached, so a theme motivation served this
        morning is not re-taught as trivia tonight."""
        ledger = self._ledger()
        if ledger is None:
            return set()
        blocked: set[str] = set()
        for theme in {str(e.get("theme") or "") for e in entries}:
            if theme:
                try:
                    if ledger.theme_in_cooldown(theme):
                        blocked.add(theme)
                except Exception:  # noqa: BLE001
                    pass
        return blocked

    def _draw_bank(self, count: int, difficulty: str,
                   used: set[str]) -> list[dict[str, Any]]:
        """Draw up to ``count`` detailed records from the bank.

        Drawing is serving: each drawn entry is counted as served in
        the bank, and its underlying teaching is marked served in
        the Teachings ledger (surface="trivia") so motivation and
        the pulses inherit the cooldown.
        """
        if count <= 0:
            return []
        source = LibraryQuestionSource(self.bank)
        stock = source.stock()
        if not stock:
            return []
        blocked = self._blocked_themes(stock)
        entries = source.draw(count, difficulty=difficulty,
                              seen=set(self._seen) | used,
                              blocked_themes=blocked)
        out: list[dict[str, Any]] = []
        for entry in entries:
            self.bank.note_served(entry)
            self._mark_teaching_served(entry)
            out.append({
                "q": entry["q"], "answer": entry["answer"],
                "distractors": list(entry.get("distractors") or []),
                "kind": entry.get("kind", "library"),
                "source": entry.get("source"),
                "tag": source_tag(entry),
                "solution": entry.get("solution", ""),
                "theme": entry.get("theme", ""),
                "difficulty": entry.get("difficulty", difficulty),
            })
        return out

    def _mark_teaching_served(self, entry: dict[str, Any]) -> None:
        ledger = self._ledger()
        teach = entry.get("teach")
        if ledger is None or not teach:
            return
        try:
            ledger.mark_served(teach, surface="trivia",
                               body=entry.get("q", ""), kind="teaching")
        except Exception:  # noqa: BLE001 - marking must never sink a game
            _log.debug("trivia: teachings mark failed", exc_info=True)

    # ── public API ───────────────────────────────────────────────────
    def deal(self, count: int = 8, difficulty: str = "normal",
             profile: dict[str, float] | None = None) -> list[tuple[str, str]]:
        """Deal ``count`` fresh questions as (question, answer) pairs.

        The M3 mix: the verified bank takes at least half the set
        when it has fresh stock, the generative forge serves the
        rest, the template forge fills last.  With no bank stock this
        is exactly the pre-M3 deal.
        """
        return [(r["q"], r["answer"]) for r in
                self.deal_detailed(count, difficulty, profile)]

    def deal_detailed(self, count: int = 8, difficulty: str = "normal",
                      profile: dict[str, float] | None = None
                      ) -> list[dict[str, Any]]:
        """Deal with provenance: records carry kind/source/tag/
        distractors/solution alongside q+answer."""
        if profile is None:
            profile = self._interest_profile()
        self._maybe_autotick()
        out: list[dict[str, Any]] = []
        used: set[str] = set()

        def _take(records: list[dict[str, Any]]) -> None:
            for r in records:
                h = _hash_q(r["q"])
                if h in used or not self._is_fresh(r["q"]):
                    continue
                used.add(h)
                out.append(r)

        # 1) the bank's preferred share (≥ LIBRARY_SHARE when stocked)
        stock = self.bank.fresh_count(set(self._seen))
        if stock > 0:
            share = min(stock, max(1, math.ceil(count * LIBRARY_SHARE)))
            _take(self._draw_bank(share, difficulty, used))
        # 2) the generative forge, as before
        if len(out) < count and self._suggest is not None:
            topics = self.sample_topics(count - len(out), profile)
            for topic in dict.fromkeys(topics):  # dedup, keep order
                need = count - len(out)
                if need <= 0:
                    break
                got = self._ask_model(topic, difficulty, need)
                _take([{"q": q, "answer": a, "distractors": [],
                        "kind": "generative", "source": None, "tag": "",
                        "solution": "", "theme": topic,
                        "difficulty": difficulty} for q, a in got])
        # 3) more bank before the template forge (design order)
        if len(out) < count:
            _take(self._draw_bank(count - len(out), difficulty, used))
        # 4) template forge — the never-stall filler, as before
        if len(out) < count:
            got = self._forge(count - len(out), difficulty)
            _take([{"q": q, "answer": a, "distractors": [],
                    "kind": "template", "source": None, "tag": "",
                    "solution": "", "theme": "",
                    "difficulty": difficulty} for q, a in got])
        # mark everything seen
        for r in out[:count]:
            self._mark_seen(r["q"])
        return out[:count]

    def deal_ladder(self, count: int = 8, difficulty: str = "normal",
                    profile: dict[str, float] | None = None
                    ) -> list[tuple[str, str, list[str], str, str]]:
        """Ladder-ready questions: (question, answer, options, tag,
        solution) — banked questions bring their verified distractors
        and provenance tag; generative ones get forged options."""
        out = []
        for r in self.deal_detailed(count, difficulty, profile):
            opts = build_options(r["answer"], r.get("distractors") or [],
                                 rng=self.rng)
            out.append((r["q"], r["answer"], opts, r.get("tag", ""),
                        r.get("solution", "")))
        return out

    # ── bank surfaces ────────────────────────────────────────────────
    def bank_status(self) -> dict[str, Any]:
        """Bank stock by kind/source + honest admission counters."""
        return self.bank.stats()

    def render_bank_status(self) -> str:
        return render_bank_status(self.bank_status())

    def refill_bank(self, **kw: Any) -> dict[str, Any]:
        """Run one refiller tick (the background maintenance pass)."""
        return self.bank.tick(**kw)

    def purge_bank(self, question: str = "") -> dict[str, Any]:
        """Purge one question (by text) or the whole bank — the
        owner's eyeball-and-purge surface (design §6.4)."""
        return self.bank.purge(question)


# ── the verified bank (M3) ───────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)

_MATH_TIERS: tuple[tuple[str, str], ...] = (
    ("easy", "easy"), ("medium", "normal"),
    ("hard", "hard"), ("expert", "expert"),
)


class TriviaBank:
    """The verified question bank + its background refiller.

    Storage follows the forge's own idiom: two KV blobs next to the
    anti-repeat hashes (``trivia_forge.bank`` — items + served-hash
    ring + expiry count; ``trivia_forge.bank_stats`` — the honest
    counters).  With no KV the bank lives on the forge instance.

    The refiller (:meth:`tick`) follows the wisdom organ's tick
    pattern: staged, fail-soft, budget-bounded, and it returns a
    report of exactly what it generated / admitted / discarded —
    nothing is banked silently and nothing unverifiable is banked
    at all.
    """

    def __init__(self, forge: "TriviaForge") -> None:
        self._forge = forge
        self._math_obj: Any = None

    # ── storage ──
    def _load_bank(self) -> tuple[list[dict[str, Any]], list[str], int]:
        kv = self._forge._kv
        if kv is not None:
            try:
                data = kv.get(_BANK_KV) or {}
            except Exception:  # noqa: BLE001
                data = {}
            return (list(data.get("items", [])),
                    list(data.get("ring", [])),
                    int(data.get("expired", 0)))
        blob = getattr(self._forge, "_bank_mem", None) or {}
        return (list(blob.get("items", [])), list(blob.get("ring", [])),
                int(blob.get("expired", 0)))

    def _save_bank(self, items: list[dict[str, Any]], ring: list[str],
                   expired: int) -> None:
        blob = {"items": items[-BANK_CAP:], "ring": ring[-BANK_HASH_RING:],
                "expired": expired}
        kv = self._forge._kv
        if kv is not None:
            try:
                kv.set(_BANK_KV, blob)
            except Exception:  # noqa: BLE001
                pass
        else:
            self._forge._bank_mem = blob  # type: ignore[attr-defined]

    def _load_stats(self) -> dict[str, Any]:
        kv = self._forge._kv
        data: dict[str, Any] = {}
        if kv is not None:
            try:
                data = kv.get(_BANK_STATS_KV) or {}
            except Exception:  # noqa: BLE001
                data = {}
        else:
            data = getattr(self._forge, "_bank_mem_stats", None) or {}
        stats = {"candidates": 0, "admitted": 0, "discarded": 0,
                 "reasons": {}, "ticks": 0, "last_tick": 0.0}
        stats.update({k: data[k] for k in stats if k in data})
        return stats

    def _save_stats(self, stats: dict[str, Any]) -> None:
        kv = self._forge._kv
        if kv is not None:
            try:
                kv.set(_BANK_STATS_KV, stats)
            except Exception:  # noqa: BLE001
                pass
        else:
            self._forge._bank_mem_stats = stats  # type: ignore[attr-defined]

    # ── stock views ──
    def items(self) -> list[dict[str, Any]]:
        return self._load_bank()[0]

    def stock_count(self) -> int:
        return len(self._load_bank()[0])

    def fresh_count(self, seen: set[str]) -> int:
        return sum(1 for e in self._load_bank()[0]
                   if _hash_q(e["q"]) not in seen)

    def stats(self) -> dict[str, Any]:
        items, _ring, expired = self._load_bank()
        st = self._load_stats()
        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for e in items:
            kind = str(e.get("kind") or "?")
            by_kind[kind] = by_kind.get(kind, 0) + 1
            stype = str((e.get("source") or {}).get("type") or "?")
            by_source[stype] = by_source.get(stype, 0) + 1
        return {"stock": len(items), "target": BANK_TARGET,
                "by_kind": by_kind, "by_source": by_source,
                "candidates": int(st["candidates"]),
                "admitted": int(st["admitted"]),
                "discarded": int(st["discarded"]),
                "discard_reasons": dict(st["reasons"]),
                "ticks": int(st["ticks"]),
                "last_tick": float(st["last_tick"]),
                "expired": expired}

    def purge(self, question: str = "") -> dict[str, Any]:
        items, ring, expired = self._load_bank()
        if question:
            h = _hash_q(question)
            kept = [e for e in items if _hash_q(e["q"]) != h]
        else:
            kept = []
        purged = len(items) - len(kept)
        self._save_bank(kept, ring, expired)
        return {"purged": purged, "stock": len(kept)}

    # ── admission + drawing ──
    def _admit(self, entry: dict[str, Any]) -> tuple[bool, str]:
        """Bank one verified entry. (False, reason) when the shape is
        unusable or the question is already banked/served — the
        caller counts the discard; nothing is repaired silently."""
        q = str(entry.get("q") or "").strip()
        a = str(entry.get("answer") or "").strip()
        ds = [str(d).strip() for d in (entry.get("distractors") or [])]
        if not q or not a or len(ds) < 3:
            return False, "bad shape"
        n_ans = _norm_answer(a)
        n_ds = [_norm_answer(d) for d in ds[:3]]
        if len(set(n_ds)) < 3 or n_ans in n_ds:
            return False, "bad distractors"
        items, ring, expired = self._load_bank()
        h = _hash_q(q)
        if h in {_hash_q(e["q"]) for e in items} or h in ring:
            return False, "duplicate"
        while len(items) >= BANK_CAP:
            items.pop(0)
            expired += 1
        entry = dict(entry)
        entry["q"], entry["answer"] = q, a
        entry["distractors"] = ds[:3]
        entry.setdefault("verified_at", time.time())
        entry["serves"] = 0
        items.append(entry)
        self._save_bank(items, ring, expired)
        return True, ""

    def draw(self, count: int, *, difficulty: str = "",
             seen: set[str] | frozenset[str] = frozenset(),
             blocked_themes: set[str] | frozenset[str] = frozenset()
             ) -> list[dict[str, Any]]:
        """Fresh entries for a deal: question hash outside the
        anti-repeat window, theme outside the ledger cooldown.
        Requested difficulty is preferred, never a hard filter —
        a stocked bank always outranks an empty slot."""
        if count <= 0:
            return []
        fresh = [e for e in self._load_bank()[0]
                 if _hash_q(e["q"]) not in seen
                 and str(e.get("theme") or "") not in blocked_themes]
        self._forge.rng.shuffle(fresh)
        if difficulty:
            pref = [e for e in fresh
                    if e.get("difficulty") == difficulty]
            rest = [e for e in fresh
                    if e.get("difficulty") != difficulty]
            fresh = pref + rest
        return fresh[:count]

    def note_served(self, entry: dict[str, Any]) -> None:
        items, ring, expired = self._load_bank()
        h = _hash_q(str(entry.get("q") or ""))
        for e in items:
            if _hash_q(e["q"]) == h:
                e["serves"] = int(e.get("serves", 0)) + 1
                e["last_served"] = time.time()
                break
        ring.append(h)
        self._save_bank(items, ring, expired)

    # ── the refiller ──
    def tick(self, *, max_passages: int = 6, max_maths: int = 8,
             max_maths_llm: int = 4,
             now: float | None = None) -> dict[str, Any]:
        """One maintenance pass: top the bank up toward BANK_TARGET.

        Stages (each fail-soft, wisdom-organ style): library
        questions from held passages → procedural maths → brain-posed
        maths.  Every candidate is counted — generated, admitted, or
        discarded with its reason — in the persisted stats.
        """
        now = time.time() if now is None else now
        stats = self._load_stats()
        stats["ticks"] = int(stats["ticks"]) + 1
        stats["last_tick"] = now
        report: dict[str, Any] = {
            "generated": 0, "admitted": 0, "discarded": 0,
            "by_kind": {}, "notes": [], "errors": []}

        def _count(ok: bool, reason: str, kind: str) -> None:
            stats["candidates"] = int(stats["candidates"]) + 1
            report["generated"] += 1
            if ok:
                stats["admitted"] = int(stats["admitted"]) + 1
                report["admitted"] += 1
                report["by_kind"][kind] = \
                    report["by_kind"].get(kind, 0) + 1
            else:
                stats["discarded"] = int(stats["discarded"]) + 1
                report["discarded"] += 1
                reasons = stats["reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1

        if self.stock_count() >= BANK_TARGET:
            report["notes"].append("bank at target — nothing to do")
        else:
            stages = (
                ("library", lambda: self._tick_library(
                    report, _count, max_passages=max_passages, now=now)),
                ("maths", lambda: self._tick_maths(
                    report, _count, max_items=max_maths, now=now)),
                ("maths_llm", lambda: self._tick_maths_llm(
                    report, _count, max_items=max_maths_llm, now=now)),
            )
            for stage, run in stages:
                try:
                    run()
                except Exception as exc:  # noqa: BLE001
                    report["errors"].append(f"{stage}: {exc}")
        self._save_stats(stats)
        report["stock"] = self.stock_count()
        return report

    def _tick_library(self, report: dict[str, Any], count: Any, *,
                      max_passages: int, now: float) -> None:
        forge = self._forge
        context = forge._context
        if context is None:
            report["notes"].append("library: no context — skipped")
            return
        suggest = forge._suggest
        if suggest is None:
            report["notes"].append("library: no brain — skipped")
            return
        from ..agents.teachings import (
            BASE_THEMES, ITEM_COOLDOWN_S, TeachingsLedger,
            item_key as teach_key, library_candidates)

        ledger = forge._ledger() or TeachingsLedger(context)
        try:
            themes = ledger.eligible_themes(now=now)[:4]
        except Exception:  # noqa: BLE001
            themes = []
        if not themes:
            themes = list(BASE_THEMES[:4])
        items = library_candidates(context, themes)
        items += self._research_items(context)
        blocked: set[str] = set()
        try:
            for it in ledger.served_items(since=now - ITEM_COOLDOWN_S):
                if it.get("_key"):
                    blocked.add(str(it["_key"]))
        except Exception:  # noqa: BLE001
            pass
        passages: list[tuple[dict[str, Any], str, str]] = []
        seen_keys: set[str] = set()
        for it in items:
            src = it.get("source") or {}
            passage = str(src.get("passage") or "").strip()
            if len(passage) < 120:
                continue
            key = teach_key(str(src.get("type") or ""),
                            str(src.get("ref") or ""), passage)
            if key in blocked or key in seen_keys:
                continue
            seen_keys.add(key)
            passages.append((it, key, passage))
        forge.rng.shuffle(passages)
        difficulties = ("easy", "normal", "hard")
        made = 0
        for idx, (it, key, passage) in enumerate(passages):
            if made >= max_passages:
                break
            if self.stock_count() >= BANK_TARGET:
                break
            made += 1
            difficulty = difficulties[idx % len(difficulties)]
            src = it.get("source") or {}
            prompt = (
                "Using ONLY the passage below, write ONE "
                f"{difficulty} trivia question whose answer is stated "
                "in the passage. Return ONLY a JSON object: "
                '{"question": "...", "answer": "...", '
                '"distractors": ["wrong", "wrong", "wrong"], '
                '"evidence_quote": "an exact sentence copied from '
                'the passage that states the answer"}. Short answer '
                "(1-4 words). The distractors must be plausible but "
                "must NOT appear in the evidence quote.\n\n"
                f"Passage (from {src.get('label') or 'her library'}):\n"
                f"{passage[:1200]}")
            try:
                raw = (suggest(prompt) or "").strip()
            except Exception:  # noqa: BLE001
                raw = ""
            m = _JSON_OBJ_RE.search(raw)
            cand = None
            if m:
                try:
                    cand = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    cand = None
            if not isinstance(cand, dict):
                count(False, "unparseable", "library")
                continue
            ok, reason = verify_library_candidate(passage, cand)
            if not ok:
                count(False, reason, "library")
                continue
            entry = {
                "q": str(cand["question"]).strip(),
                "answer": str(cand["answer"]).strip(),
                "distractors": [str(d).strip()
                                for d in cand["distractors"]],
                "source": {"type": str(src.get("type") or ""),
                           "ref": str(src.get("ref") or ""),
                           "label": str(src.get("label") or "")},
                "evidence": str(cand["evidence_quote"]).strip(),
                "theme": str(it.get("theme") or ""),
                "difficulty": difficulty,
                "kind": "library",
                "verified_at": now,
                "teach": {
                    "id": it.get("id"), "kind": it.get("kind") or
                    "teaching", "title": it.get("title"),
                    "theme": it.get("theme"),
                    "tradition": it.get("tradition"),
                    "source": {"type": str(src.get("type") or ""),
                               "ref": str(src.get("ref") or ""),
                               "label": str(src.get("label") or ""),
                               "passage": str(src.get("passage") or "")},
                    "_key": key},
            }
            ok2, reason2 = self._admit(entry)
            count(ok2, reason2, "library")

    def _research_items(self, context: Any) -> list[dict[str, Any]]:
        """Passage candidates from promoted research claims (the
        knowledge graph's claim nodes) — the third Library picker."""
        out: list[dict[str, Any]] = []
        try:
            from ..agents.kg import KnowledgeGraph

            kg = KnowledgeGraph(getattr(context, "db", None))
            nodes = kg.top(ntype="claim", n=6) or []
        except Exception:  # noqa: BLE001 - research is one picker of 3
            return out
        for node in nodes:
            try:
                props = node.get("properties") or {}
                text = ""
                for k in ("text", "claim", "claim_text", "summary",
                          "body"):
                    v = props.get(k)
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
                if not text:
                    label = str(node.get("label") or "")
                    if label and not label.startswith("claim:"):
                        text = label
                if len(text) < 60:
                    continue
                urls = props.get("source_urls") or \
                    props.get("sources") or []
                ref = str(urls[0]) if isinstance(urls, list) and urls \
                    else str(node.get("id") or "research")
                domain = str(props.get("domain") or "research")
                out.append({
                    "id": f"research_{node.get('id')}",
                    "kind": "teaching", "title": domain,
                    "theme": domain, "tradition": "",
                    "source": {"type": "research", "ref": ref,
                               "label": f"{domain} — her research",
                               "passage": text[:1200]}})
            except Exception:  # noqa: BLE001
                continue
        return out

    def _math_class(self) -> Any:
        if self._math_obj is None:
            from .math_class import MathClass
            self._math_obj = MathClass()
        return self._math_obj

    def _tick_maths(self, report: dict[str, Any], count: Any, *,
                    max_items: int, now: float) -> None:
        """Procedural maths: MathClass problems, verified by
        construction *and* re-checked through MathClass's own
        answer checker before banking."""
        if max_items <= 0:
            return
        from .math_class import check_answer

        mc = self._math_class()
        rng = self._forge.rng
        for i in range(max_items):
            if self.stock_count() >= BANK_TARGET:
                break
            tier, difficulty = _MATH_TIERS[i % len(_MATH_TIERS)]
            try:
                prob = mc.generate("trivia-bank", tier=tier)
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"maths generate: {exc}")
                break
            value = prob.answer_value
            if isinstance(value, complex):
                count(False, "complex answer", "maths")
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                count(False, "non-numeric answer", "maths")
                continue
            if not check_answer(prob, prob.answer_text):
                count(False, "maths self-check failed", "maths")
                continue
            entry = {
                "q": prob.question, "answer": prob.answer_text,
                "distractors": maths_distractors(
                    fval, prob.answer_text, rng),
                "source": {"type": "maths",
                           "ref": f"{tier}/{prob.topic}",
                           "label": f"MathClass {tier} — {prob.topic}"},
                "evidence": prob.solution, "solution": prob.solution,
                "theme": "maths", "difficulty": difficulty,
                "kind": "maths", "verified_at": now, "teach": None}
            ok, reason = self._admit(entry)
            count(ok, reason, "maths")

    def _tick_maths_llm(self, report: dict[str, Any], count: Any, *,
                        max_items: int, now: float) -> None:
        """Brain-posed maths: believed only after the independent
        computation reproduces the claimed answer (owner correction:
        maths uses BOTH paths, the LLM path cross-verified)."""
        suggest = self._forge._suggest
        if suggest is None:
            report["notes"].append("maths_llm: no brain — skipped")
            return
        rng = self._forge.rng
        difficulties = ("easy", "normal", "hard")
        for i in range(max_items):
            if self.stock_count() >= BANK_TARGET:
                break
            difficulty = difficulties[i % len(difficulties)]
            prompt = (
                "Invent ONE " + difficulty + " maths WORD problem — "
                "a short real-world story with numbers — whose answer "
                "is a single number. Also give the arithmetic "
                "expression that computes the answer from the numbers "
                "in the problem, using only + - * / and parentheses. "
                "Every number in the expression must appear in the "
                "question. Return ONLY a JSON object: "
                '{"question": "...", "answer": "...", '
                '"expression": "..."}')
            try:
                raw = (suggest(prompt) or "").strip()
            except Exception:  # noqa: BLE001
                raw = ""
            m = _JSON_OBJ_RE.search(raw)
            cand = None
            if m:
                try:
                    cand = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    cand = None
            if not isinstance(cand, dict):
                count(False, "unparseable", "maths_llm")
                continue
            ok, reason, computed = verify_llm_maths(
                cand.get("question"), cand.get("answer"),
                cand.get("expression"))
            if not ok or computed is None:
                count(False, reason, "maths_llm")
                continue
            answer_text = str(cand["answer"]).strip()
            entry = {
                "q": str(cand["question"]).strip(),
                "answer": answer_text,
                "distractors": maths_distractors(
                    computed, answer_text, rng),
                "source": {"type": "maths_llm", "ref": "brain",
                           "label": "Brain-posed — computation-checked"},
                "evidence": f"{cand['expression']} = "
                            f"{_fmt_num(computed, answer_text)}",
                "solution": f"{cand['expression']} = "
                            f"{_fmt_num(computed, answer_text)}",
                "theme": "maths", "difficulty": difficulty,
                "kind": "maths_llm", "verified_at": now, "teach": None}
            ok2, reason2 = self._admit(entry)
            count(ok2, reason2, "maths_llm")


class LibraryQuestionSource:
    """The design's named source inside the forge: verified Library
    questions, drawn from the bank the refiller maintains.  A thin
    facade so the forge's deal speaks "source", not "storage"."""

    def __init__(self, bank: TriviaBank) -> None:
        self._bank = bank

    def stock(self) -> list[dict[str, Any]]:
        return self._bank.items()

    def available(self, seen: set[str]) -> int:
        return self._bank.fresh_count(seen)

    def draw(self, count: int, *, difficulty: str = "",
             seen: set[str] | frozenset[str] = frozenset(),
             blocked_themes: set[str] | frozenset[str] = frozenset()
             ) -> list[dict[str, Any]]:
        return self._bank.draw(count, difficulty=difficulty, seen=seen,
                               blocked_themes=blocked_themes)


# ── TriviaLadder: Millionaire-style match structure ──────────────────────────
#
# Dealt questions are raw material; a *match* needs stakes. The ladder
# climbs difficulty (and prize) each round: answer to bank the round's
# coins, miss and fall to the last safe haven. Three lifelines —
# 50:50, skip, audience — one use each per match. Answer streaks
# multiply the prize. Wrong answers never end the night empty-handed:
# safe havens guarantee a floor.

LADDER_PRIZES: tuple[int, ...] = (10, 25, 50, 100, 200, 350, 600, 1000)
LADDER_DIFFICULTIES: tuple[str, ...] = (
    "easy", "easy", "normal", "normal", "normal", "hard", "hard", "expert",
)
#: rounds that are safe havens — falling later still banks this prize.
LADDER_SAFE_HAVENS: tuple[int, ...] = (2, 5)

LIFELINES: tuple[tuple[str, str], ...] = (
    ("fifty", "5️⃣0️⃣ 50:50 — remove two wrong options"),
    ("skip", "⏭️ skip — swap this question for a fresh one"),
    ("audience", "🗳️ audience — the crowd votes (usually right)"),
)


class TriviaLadder:
    """One ladder match: state, lifelines, streaks, prizes.

    ``questions`` is ``[(question, answer, options)]`` — options a list
    of 4 with the answer included (the forge deals Q&A; the ladder
    builds options via the model or distractor forge). ``ask()`` renders
    the current question card; ``answer(text)`` scores it; lifelines via
    :meth:`use_lifeline`.
    """

    def __init__(self, questions: list[tuple],
                 seed: int | None = None) -> None:
        """``questions`` entries are (question, answer, options) —
        optionally extended with a 4th element, the provenance tag
        shown after the answer (M3: "📚 From her library: …"), and a
        5th, a worked solution shown on a miss (M3 maths)."""
        self.rng = random.Random(seed)
        self.questions = [(q, a, list(o)) for q, a, o, *_ in questions]
        self._tags = [(q[3] if len(q) > 3 else "") or ""
                      for q in questions]
        self._solutions = [(q[4] if len(q) > 4 else "") or ""
                           for q in questions]
        self.round = 0
        self.streak = 0
        self.banked = 0
        self.lifelines = {slug: True for slug, _ in LIFELINES}
        self.over = False
        self._fifty: list[str] | None = None
        self._log: list[str] = []

    # ── state ────────────────────────────────────────────────────────────
    @property
    def prize(self) -> int:
        idx = min(self.round, len(LADDER_PRIZES) - 1)
        return LADDER_PRIZES[idx]

    @property
    def difficulty(self) -> str:
        idx = min(self.round, len(LADDER_DIFFICULTIES) - 1)
        return LADDER_DIFFICULTIES[idx]

    def current(self) -> tuple[str, str, list[str]] | None:
        if self.over or self.round >= len(self.questions):
            return None
        return self.questions[self.round]

    def ask(self) -> str:
        """The question card."""
        cur = self.current()
        if cur is None:
            return self._final()
        q, _a, options = cur
        opts = list(options)
        if self._fifty is not None:
            opts = [o for o in opts if o in self._fifty]
        self.rng.shuffle(opts)
        letters = "ABCD"
        lines = [f"❓ round {self.round + 1}/{len(self.questions)} — "
                 f"**{self.prize}c** ({self.difficulty})"]
        if self.streak >= 2:
            lines[0] += f" · 🔥 streak ×{self.streak}"
        lines.append(q)
        for i, o in enumerate(opts[:4]):
            lines.append(f"  {letters[i]}. {o}")
        avail = [slug for slug, _desc in LIFELINES
                 if self.lifelines[slug]]
        if avail:
            lines.append("lifelines: " + " ".join(
                f"/lifeline {s}" for s in avail))
        haven = max([h for h in LADDER_SAFE_HAVENS if h <= self.round],
                    default=None)
        if haven is not None:
            lines.append(f"🛟 safe haven: {LADDER_PRIZES[haven]}c banked")
        return "\n".join(lines)

    # ── lifelines ────────────────────────────────────────────────────────
    def use_lifeline(self, slug: str) -> str:
        slug = (slug or "").strip().lower()
        if self.over:
            return "the match is over."
        if slug not in self.lifelines:
            return f"unknown lifeline. try: {', '.join(self.lifelines)}"
        if not self.lifelines[slug]:
            return "that lifeline is spent."
        cur = self.current()
        if cur is None:
            return "no active question."
        _q, answer, options = cur
        self.lifelines[slug] = False
        if slug == "fifty":
            wrong = [o for o in options if o != answer]
            keep = self.rng.sample(wrong, min(2, len(wrong)))
            self._fifty = [answer] + [o for o in wrong if o not in keep]
            return "5️⃣0️⃣ two wrong answers removed."
        if slug == "skip":
            self._fifty = None
            self._log.append(f"round {self.round + 1} skipped")
            self.round += 1
            if self.round >= len(self.questions):
                self.over = True
                return "⏭️ skipped — and that was the last question.\n" \
                    + self._final()
            return "⏭️ skipped — fresh question:\n" + self.ask()
        # audience: weighted vote, usually (not always) right
        weights = []
        for o in options:
            weights.append(55 if o == answer else 15)
        total = sum(weights)
        pick = self.rng.choices(options,
                                weights=[w / total for w in weights])[0]
        conf = self.rng.randint(52, 88)
        return (f"🗳️ the audience votes **{pick}** ({conf}% sure). "
                f"trust them?")

    # ── answering ────────────────────────────────────────────────────────
    def answer(self, text: str) -> tuple[bool, str]:
        """Score an answer. Returns (match_over, message)."""
        cur = self.current()
        if cur is None or self.over:
            return True, self._final()
        _q, answer, _options = cur
        guess = (text or "").strip().lower()
        # accept letter or full text
        letters = "abcd"
        if len(guess) == 1 and guess in letters:
            opts = [o for o in _options
                    if self._fifty is None or o in self._fifty]
            guess = opts[letters.index(guess)].lower() \
                if letters.index(guess) < len(opts) else guess
        correct = guess == answer.strip().lower()
        self._fifty = None
        # M3 provenance: the tag rides AFTER the answer is revealed,
        # never on the question card; a worked solution teaches on a
        # miss (the MathClass rule: a wrong answer still teaches).
        tag = self._tags[self.round] if self.round < len(self._tags) \
            else ""
        sol = self._solutions[self.round] \
            if self.round < len(self._solutions) else ""
        tag_line = f"\n{tag}" if tag else ""
        if correct:
            self.streak += 1
            mult = 1 + 0.25 * (self.streak - 1)
            won = int(round(self.prize * mult))
            self.banked += won
            self._log.append(f"round {self.round + 1}: +{won}c")
            self.round += 1
            if self.round >= len(self.questions):
                self.over = True
                return True, (f"✅ correct! +{won}c{tag_line}\n\n"
                              f"{self._final()}")
            return False, (f"✅ correct! +{won}c (banked {self.banked}c)"
                           f"{tag_line}\n" + self.ask())
        # miss: fall to the last safe haven — banked keeps only what
        # was actually won up to the haven floor
        haven = max([h for h in LADDER_SAFE_HAVENS if h <= self.round],
                    default=None)
        self.banked = LADDER_PRIZES[haven] if haven is not None else 0
        self.over = True
        self.streak = 0
        sol_line = f"\n🧮 {sol}" if sol else ""
        return True, (f"❌ wrong — the answer was **{answer}**."
                      f"{tag_line}{sol_line}\n"
                      f"🛟 you fall to the safe haven: {self.banked}c banked.\n\n"
                      + self._final())

    def walk_away(self) -> str:
        """Bank current winnings and end the match."""
        if self.over:
            return self._final()
        self.over = True
        return (f"🚶 you walk away with **{self.banked}c** — "
                f"smart money.\n\n" + self._final())

    def _final(self) -> str:
        self.over = True
        lines = [f"🏁 ladder complete — banked **{self.banked}c** "
                 f"in {len(self._log)} rounds"]
        if self._log:
            lines.append("  " + " · ".join(self._log[-4:]))
        spent = [s for s, v in self.lifelines.items() if not v]
        if spent:
            lines.append(f"  lifelines used: {', '.join(spent)}")
        return "\n".join(lines)


def build_options(answer: str, distractors: list[str],
                  rng: random.Random | None = None) -> list[str]:
    """4 options with the answer shuffled in (deduped, padded)."""
    rng = rng or random.Random()
    opts = [answer]
    for d in distractors:
        if d and d != answer and d not in opts:
            opts.append(d)
        if len(opts) == 4:
            break
    while len(opts) < 4:
        opts.append(f"none of these ({len(opts)})")
    rng.shuffle(opts)
    return opts
