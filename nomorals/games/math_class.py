"""Math Class: procedurally generated maths, by tier, for the group.

Six tiers, each with its own topic pool. Every problem is generated
from a seed — never pulled from a static bank — so the supply is
effectively endless and no two players get the same stream. Every
problem carries a verified answer *and* a worked solution: the solution
is shown after the attempt, so a wrong answer still teaches.

Formats: quick-fire (60s, one problem), timed set (10 in 5 min),
daily challenge (same problem for everyone, one day), head-to-head
duel (same problems, first correct wins), group tournament, streak
mode (keep going until you miss).

Anti-cheat, enforced here, not promised:
  * per-player seeds — your stream is yours; copying a friend's
    answer key does not help;
  * never repeat the same problem for the same player (seen hashes);
  * cooldown after a failed attempt, per tier;
  * capped attempts per tier per day;
  * decaying rewards on repeat farming (see reward_ladder).

Adaptive: win 3 in a row and the class moves you up a tier; lose 2 in
a row and it moves you down. You can always *attempt* a higher tier —
the adaptive tier is a recommendation, never a ceiling.

Answers are checked by value with a per-problem tolerance, so
``1/2``, ``0.5`` and ``0.50`` all count for one half. No answer is
ever sent to the player before they attempt the problem — handlers
must use :meth:`MathClass.public_problem`, which strips it.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "TIERS", "TIER_TOPICS", "FORMAT_CONFIG", "ATTEMPT_CAPS",
    "COOLDOWN_SECONDS", "Problem", "MathClass", "Duel",
    "check_answer", "parse_answer", "daily_seed",
]

#: Tier order, easiest first.
TIERS: tuple[str, ...] = (
    "easy", "medium", "hard", "expert", "master", "impossible",
)

#: Topics per tier — the syllabus. Generators live in
#: ``_GENERATORS`` below, keyed by (tier, topic).
TIER_TOPICS: dict[str, tuple[str, ...]] = {
    "easy": ("arithmetic", "times_tables", "fractions", "percentages",
             "missing_number"),
    "medium": ("algebra", "geometry", "ratios", "word_problems",
               "percent_change", "speed"),
    "hard": ("quadratics", "trigonometry", "logarithms", "sequences",
             "simultaneous", "exponents"),
    "expert": ("calculus", "combinatorics", "number_theory", "vectors",
               "probability", "trig_equations"),
    "master": ("olympiad", "diophantine", "primes", "differential_equations",
               "complex", "modular"),
    "impossible": ("chained_recurrence", "modular_sum", "divisor_hunt",
                   "three_equations", "digit_sum", "factorial_divisibility",
                   "collatz", "last_two_digits"),
}

#: Format → (problems, seconds for the whole format, blurb).
FORMAT_CONFIG: dict[str, dict[str, Any]] = {
    "quickfire": {"problems": 1, "seconds": 60,
                  "blurb": "One problem, 60 seconds. Fast fingers, faster brain."},
    "timed_set": {"problems": 10, "seconds": 300,
                  "blurb": "10 problems in 5 minutes. Pace yourself."},
    "daily": {"problems": 1, "seconds": 600,
              "blurb": "Today's challenge — same problem for everyone."},
    "duel": {"problems": 5, "seconds": 300,
             "blurb": "5 problems, head to head. First correct wins each round."},
    "tournament": {"problems": 8, "seconds": 600,
                   "blurb": "8 problems, group tournament. Top of the board takes the pool."},
    "streak": {"problems": 0, "seconds": 0,
               "blurb": "Endless — keep going until you miss. One miss ends the run."},
}

#: Max attempts per tier per player per day (UTC day). Impossible is
#: deliberately tight: 8 shots a day, no brute-force farming.
ATTEMPT_CAPS: dict[str, int] = {
    "easy": 100, "medium": 60, "hard": 40,
    "expert": 25, "master": 15, "impossible": 8,
}

#: Cooldown (seconds) after a failed attempt, per tier. Fail fast,
#: wait, think, retry — no hammering the same tier.
COOLDOWN_SECONDS: dict[str, int] = {
    "easy": 10, "medium": 20, "hard": 45,
    "expert": 90, "master": 180, "impossible": 300,
}

#: Win streak that promotes / loss streak that demotes (adaptive tier).
PROMOTE_AFTER_WINS = 3
DEMOTE_AFTER_LOSSES = 2


# ── answer parsing / checking ──────────────────────────────────────────────

def parse_answer(text: Any) -> float | complex | None:
    """Parse a submitted answer: int, decimal, fraction ``a/b``, or
    complex ``a+bi``. Returns None when unparsable."""
    if text is None:
        return None
    s = str(text).strip().lower().replace(" ", "").replace("−", "-").replace("–", "-")
    if not s:
        return None
    if "i" in s:  # complex: "3+4i", "-2i", "5i"
        try:
            return complex(s.replace("i", "j"))
        except ValueError:
            return None
    try:
        if "/" in s:
            return float(Fraction(s))
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        return None


def check_answer(problem: "Problem", submitted: Any) -> bool:
    """True when ``submitted`` matches the problem's verified answer
    within its tolerance."""
    got = parse_answer(submitted)
    if got is None:
        return False
    want = problem.answer_value
    tol = problem.tolerance
    if isinstance(want, complex) or isinstance(got, complex):
        try:
            w, g = complex(want), complex(got)
        except (TypeError, ValueError):
            return False
        return abs(w.real - g.real) <= tol and abs(w.imag - g.imag) <= tol
    try:
        return abs(float(want) - float(got)) <= tol
    except (TypeError, ValueError):
        return False


# ── problem ────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    """One generated problem. ``answer_value``/``answer_text`` are
    server-side only — never render them to a player pre-attempt."""

    problem_id: str
    seed: int
    tier: str
    topic: str
    format: str
    question: str
    answer_value: Any
    answer_text: str
    solution: str                    # worked solution, shown AFTER attempt
    solution_steps: list[str] = field(default_factory=list)
    tolerance: float = 1e-9
    time_limit_s: int = 60
    created_at: float = 0.0

    def public(self) -> dict[str, Any]:
        """Player-facing view: question + metadata, NO answer."""
        return {"problem_id": self.problem_id, "tier": self.tier,
                "topic": self.topic, "format": self.format,
                "question": self.question,
                "time_limit_s": self.time_limit_s,
                "created_at": self.created_at}

    def reveal(self) -> dict[str, Any]:
        """Post-attempt view: public fields + answer + worked solution."""
        out = self.public()
        out.update({"answer": self.answer_text,
                    "solution": self.solution,
                    "solution_steps": list(self.solution_steps)})
        return out

    def to_dict(self) -> dict[str, Any]:
        return self.reveal()


# ── generators ─────────────────────────────────────────────────────────────
# Each generator takes a Random and returns
# (question, answer_value, answer_text, solution, steps, tolerance).

Gen = Callable[[random.Random], tuple[str, Any, str, str, list[str], float]]


def _frac_str(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def _g_easy_arithmetic(rng: random.Random) -> tuple:
    a, b = rng.randint(12, 240), rng.randint(7, 180)
    op = rng.choice(["+", "-", "×"])
    if op == "+":
        q, ans = f"{a} + {b} = ?", a + b
        steps = [f"Add: {a} + {b} = {ans}."]
    elif op == "-":
        a, b = max(a, b), min(a, b)
        q, ans = f"{a} − {b} = ?", a - b
        steps = [f"Subtract: {a} − {b} = {ans}."]
    else:
        a, b = rng.randint(6, 48), rng.randint(3, 19)
        q, ans = f"{a} × {b} = ?", a * b
        steps = [f"Multiply: {a} × {b} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_easy_times(rng: random.Random) -> tuple:
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    ans = a * b
    return (f"{a} × {b} = ?", ans, str(ans), f"Answer: {ans}",
            [f"Times table: {a} × {b} = {ans}."], 1e-9)


def _g_easy_fractions(rng: random.Random) -> tuple:
    d1 = rng.randint(2, 12)
    d2 = rng.randint(2, 12)
    n1, n2 = rng.randint(1, d1 - 1), rng.randint(1, d2 - 1)
    ans = Fraction(n1, d1) + Fraction(n2, d2)
    q = f"{n1}/{d1} + {n2}/{d2} = ?  (give a fraction)"
    steps = [f"Common denominator {d1 * d2}: "
             f"{n1 * d2}/{d1 * d2} + {n2 * d1}/{d1 * d2} "
             f"= {n1 * d2 + n2 * d1}/{d1 * d2} = {_frac_str(ans)}."]
    return q, float(ans), _frac_str(ans), f"Answer: {_frac_str(ans)}", steps, 1e-9


def _g_easy_percent(rng: random.Random) -> tuple:
    p = rng.choice([5, 10, 20, 25, 40, 50, 60, 75, 80])
    n = rng.randint(2, 30) * 20
    ans = p * n // 100
    return (f"What is {p}% of {n}?", ans, str(ans), f"Answer: {ans}",
            [f"{p}% = {p}/100, so {p}/100 × {n} = {ans}."], 1e-9)


def _g_easy_missing(rng: random.Random) -> tuple:
    a, b = rng.randint(8, 90), rng.randint(8, 90)
    ans = a + b
    return (f"? + {a} = {ans}  — what is ? ", b, str(b), f"Answer: {b}",
            [f"? = {ans} − {a} = {b}."], 1e-9)


def _g_med_algebra(rng: random.Random) -> tuple:
    a = rng.randint(2, 9)
    x = rng.randint(2, 15)
    b = rng.randint(1, 20)
    c = a * x + b
    return (f"Solve for x: {a}x + {b} = {c}", x, str(x), f"Answer: x = {x}",
            [f"{a}x = {c} − {b} = {c - b}", f"x = {c - b}/{a} = {x}."], 1e-9)


def _g_med_geometry(rng: random.Random) -> tuple:
    shape = rng.choice(["rectangle", "triangle"])
    if shape == "rectangle":
        w, h = rng.randint(3, 30), rng.randint(3, 30)
        ans = w * h
        return (f"Area of a rectangle {w} by {h}?", ans, str(ans),
                f"Answer: {ans}", [f"Area = {w} × {h} = {ans}."], 1e-9)
    base, height = rng.randint(4, 30), rng.randint(2, 20)
    if (base * height) % 2:
        height += 1
    ans = base * height // 2
    return (f"Area of a triangle with base {base} and height {height}?",
            ans, str(ans), f"Answer: {ans}",
            [f"Area = ½ × {base} × {height} = {ans}."], 1e-9)


def _g_med_ratios(rng: random.Random) -> tuple:
    m, n = rng.randint(2, 7), rng.randint(2, 7)
    if m == n:
        n += 1
    unit = rng.randint(3, 25)
    total = (m + n) * unit
    larger = max(m, n) * unit
    return (f"Share {total} in the ratio {m}:{n}. What is the LARGER share?",
            larger, str(larger), f"Answer: {larger}",
            [f"{m} + {n} = {m + n} parts; one part = {total}/{m + n} = {unit}.",
             f"Larger share = {max(m, n)} × {unit} = {larger}."], 1e-9)


def _g_med_word(rng: random.Random) -> tuple:
    pens, pen_price = rng.randint(2, 9), rng.randint(3, 15)
    books, book_price = rng.randint(2, 8), rng.randint(5, 25)
    ans = pens * pen_price + books * book_price
    q = (f"Ada buys {pens} pens at {pen_price} coins each and {books} "
         f"notebooks at {book_price} coins each. What does she pay in total?")
    return (q, ans, str(ans), f"Answer: {ans}",
            [f"Pens: {pens} × {pen_price} = {pens * pen_price}.",
             f"Notebooks: {books} × {book_price} = {books * book_price}.",
             f"Total: {pens * pen_price} + {books * book_price} = {ans}."], 1e-9)


def _g_med_percent_change(rng: random.Random) -> tuple:
    price = rng.randint(4, 60) * 5
    pct = rng.choice([10, 20, 25, 30, 50])
    up = rng.random() < 0.5
    if up:
        ans = price * (100 + pct) // 100
        q = f"A price of {price} rises by {pct}%. What is the new price?"
        steps = [f"Increase = {pct}% of {price} = {price * pct // 100}.",
                 f"New price = {price} + {price * pct // 100} = {ans}."]
    else:
        ans = price * (100 - pct) // 100
        q = f"A price of {price} falls by {pct}%. What is the new price?"
        steps = [f"Decrease = {pct}% of {price} = {price * pct // 100}.",
                 f"New price = {price} − {price * pct // 100} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_med_speed(rng: random.Random) -> tuple:
    speed = rng.randint(4, 30) * 5
    hours = rng.randint(2, 9)
    ans = speed * hours
    return (f"A car travels at {speed} km/h for {hours} hours. How far does it go (km)?",
            ans, str(ans), f"Answer: {ans}",
            [f"Distance = speed × time = {speed} × {hours} = {ans}."], 1e-9)


def _g_hard_quadratic(rng: random.Random) -> tuple:
    r1 = rng.randint(-9, 9)
    r2 = rng.randint(-9, 9)
    if r1 == r2:
        r2 += 2
    b = -(r1 + r2)
    c = r1 * r2
    ans = max(r1, r2)
    q = f"Find the LARGER root of x² {b:+d}x {c:+d} = 0  (x² means x squared)"
    steps = [f"Factor: (x − {r1})(x − {r2}) = x² + {b}x + {c}.",
             f"Roots are x = {r1} and x = {r2}; larger is {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_hard_trig(rng: random.Random) -> tuple:
    fn, deg, val = rng.choice([
        ("sin", 0, 0.0), ("sin", 30, 0.5), ("sin", 45, math.sqrt(2) / 2),
        ("sin", 60, math.sqrt(3) / 2), ("sin", 90, 1.0),
        ("cos", 0, 1.0), ("cos", 30, math.sqrt(3) / 2),
        ("cos", 45, math.sqrt(2) / 2), ("cos", 60, 0.5), ("cos", 90, 0.0),
        ("tan", 0, 0.0), ("tan", 30, 1 / math.sqrt(3)),
        ("tan", 45, 1.0), ("tan", 60, math.sqrt(3)),
    ])
    ans_text = f"{val:.4f}".rstrip("0").rstrip(".") or "0"
    return (f"What is {fn}({deg}°)? Give a decimal to 3+ places.",
            val, ans_text, f"Answer: ≈ {ans_text}",
            [f"{fn}({deg}°) is a standard special angle: ≈ {ans_text}."], 2e-3)


def _g_hard_log(rng: random.Random) -> tuple:
    base = rng.choice([2, 3, 5, 10])
    k = rng.randint(2, 6)
    n = base ** k
    return (f"What is log base {base} of {n}? (log_{base}({n}))", k, str(k),
            f"Answer: {k}",
            [f"{base}^{k} = {n}, so log_{base}({n}) = {k}."], 1e-9)


def _g_hard_sequence(rng: random.Random) -> tuple:
    kind = rng.choice(["arith", "geo"])
    if kind == "arith":
        a1, d = rng.randint(2, 20), rng.randint(2, 9)
        n = rng.randint(6, 15)
        ans = a1 + (n - 1) * d
        q = f"Arithmetic sequence: first term {a1}, common difference {d}. What is term {n}?"
        steps = [f"a_n = a₁ + (n−1)d = {a1} + {n - 1}×{d} = {ans}."]
    else:
        a1, r = rng.randint(2, 5), rng.randint(2, 3)
        n = rng.randint(4, 7)
        ans = a1 * r ** (n - 1)
        q = f"Geometric sequence: first term {a1}, common ratio {r}. What is term {n}?"
        steps = [f"a_n = a₁·r^(n−1) = {a1} × {r}^{n - 1} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_hard_simultaneous(rng: random.Random) -> tuple:
    x, y = rng.randint(-8, 12), rng.randint(-8, 12)
    a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
    a2, b2 = rng.randint(1, 4), rng.randint(-3, 3)
    if a1 * b2 == a2 * b1:
        b2 += 1
    c1, c2 = a1 * x + b1 * y, a2 * x + b2 * y
    ans = x + y
    q = (f"Solve the system: {a1}x {b1:+d}y = {c1} and {a2}x {b2:+d}y = {c2}. "
         f"What is x + y?")
    steps = [f"Solving the two equations gives x = {x}, y = {y}.",
             f"x + y = {x} + {y} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_hard_exponent(rng: random.Random) -> tuple:
    a = rng.randint(2, 6)
    b = rng.randint(2, 5)
    c = rng.randint(2, 4)
    ans = a + b * c
    q = f"Simplify 2^{a} × (2^{b})^{c} as a single power of 2: 2^? — give the exponent."
    steps = [f"(2^{b})^{c} = 2^{b * c}.",
             f"2^{a} × 2^{b * c} = 2^{a + b * c} = 2^{ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_exp_calculus(rng: random.Random) -> tuple:
    a = rng.randint(1, 4)
    b = rng.randint(-5, 5)
    c = rng.randint(-6, 6)
    x0 = rng.randint(-3, 4)
    ans = 3 * a * x0 * x0 + 2 * b * x0 + c
    d = rng.randint(1, 9)
    q = (f"Let f(x) = {a}x³ {b:+d}x² {c:+d}x {d:+d}. "
         f"What is f'({x0})? (f' means the derivative)")
    steps = [f"f'(x) = {3 * a}x² {2 * b:+d}x {c:+d}.",
             f"f'({x0}) = {3 * a}×{x0}² {2 * b:+d}×{x0} {c:+d} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_exp_combinatorics(rng: random.Random) -> tuple:
    kind = rng.choice(["C", "P"])
    n = rng.randint(6, 12)
    k = rng.randint(2, min(5, n - 1))
    if kind == "C":
        ans = math.comb(n, k)
        q = f"How many ways to choose {k} items from {n}? (C({n},{k}))"
        steps = [f"C({n},{k}) = {n}! / ({k}!·{n - k}!) = {ans}."]
    else:
        ans = math.perm(n, k)
        q = f"How many ways to arrange {k} items chosen from {n} in order? (P({n},{k}))"
        steps = [f"P({n},{k}) = {n}! / ({n - k})! = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_exp_number_theory(rng: random.Random) -> tuple:
    kind = rng.choice(["gcd", "lcm", "mod"])
    if kind == "gcd":
        g = rng.randint(2, 12)
        a, b = g * rng.randint(2, 9), g * rng.randint(2, 9)
        ans = math.gcd(a, b)
        q, steps = f"What is gcd({a}, {b})?", [f"gcd({a}, {b}) = {ans} (Euclid's algorithm)."]
    elif kind == "lcm":
        a, b = rng.randint(4, 18), rng.randint(4, 18)
        ans = math.lcm(a, b)
        q, steps = f"What is lcm({a}, {b})?", [f"lcm({a}, {b}) = {a}×{b} / gcd = {ans}."]
    else:
        base, exp, mod = rng.randint(3, 9), rng.randint(3, 6), rng.randint(7, 50)
        ans = pow(base, exp, mod)
        q = f"What is {base}^{exp} mod {mod}? (remainder on division by {mod})"
        steps = [f"{base}^{exp} = {base ** exp}; {base ** exp} mod {mod} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_exp_vectors(rng: random.Random) -> tuple:
    a = [rng.randint(-6, 6) for _ in range(3)]
    b = [rng.randint(-6, 6) for _ in range(3)]
    ans = sum(x * y for x, y in zip(a, b))
    q = f"Dot product of ({a[0]}, {a[1]}, {a[2]}) and ({b[0]}, {b[1]}, {b[2]})?"
    steps = [f"{a[0]}×{b[0]} + {a[1]}×{b[1]} + {a[2]}×{b[2]} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_exp_probability(rng: random.Random) -> tuple:
    red = rng.randint(2, 8)
    blue = rng.randint(2, 8)
    total = red + blue
    ans = Fraction(red, total)
    q = (f"A bag has {red} red and {blue} blue balls. One ball is drawn at "
         f"random. P(red) = ? (give a fraction)")
    return (q, float(ans), _frac_str(ans), f"Answer: {_frac_str(ans)}",
            [f"P(red) = red / total = {red}/{total} = {_frac_str(ans)}."], 1e-9)


def _g_exp_trig_eq(rng: random.Random) -> tuple:
    deg, val = rng.choice([(30, "1/2"), (45, "√2/2"), (60, "√3/2"), (0, "0"), (90, "1")])
    ans = deg
    return (f"sin θ = {val} with 0° ≤ θ ≤ 90°. What is θ (degrees)?",
            ans, str(ans), f"Answer: {ans}°",
            [f"sin({deg}°) = {val}, so θ = {deg}°."], 1e-9)


def _g_master_olympiad(rng: random.Random) -> tuple:
    n = rng.randint(8, 20)
    ans = n * (n + 1) * (2 * n + 1) // 6
    q = f"What is 1² + 2² + … + {n}²? (sum of squares up to {n})"
    steps = [f"Sum = n(n+1)(2n+1)/6 = {n}×{n + 1}×{2 * n + 1}/6 = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_master_diophantine(rng: random.Random) -> tuple:
    a, b = rng.randint(3, 9), rng.randint(3, 9)
    x0, y0 = rng.randint(1, 6), rng.randint(1, 6)
    c = a * x0 + b * y0
    # The TRUE smallest positive x with a positive-integer y: brute force.
    smallest = None
    for x in range(1, c):
        rem = c - a * x
        if rem > 0 and rem % b == 0:
            smallest = x
            break
    assert smallest is not None  # x0 itself is a witness
    y_sol = (c - a * smallest) // b
    q = (f"Find the SMALLEST positive integer x such that {a}x + {b}y = {c} "
         f"has a positive integer solution y. What is x?")
    steps = [f"Need {c} − {a}x divisible by {b} with remainder > 0.",
             f"Smallest such x is {smallest}, giving y = {y_sol}: "
             f"{a}×{smallest} + {b}×{y_sol} = {c}."]
    return q, smallest, str(smallest), f"Answer: {smallest}", steps, 1e-9


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def _g_master_primes(rng: random.Random) -> tuple:
    lo = rng.randint(10, 60)
    hi = lo + rng.randint(15, 30)
    ans = sum(1 for n in range(lo, hi + 1) if _is_prime(n))
    q = f"How many primes are there between {lo} and {hi} inclusive?"
    return (q, ans, str(ans), f"Answer: {ans}",
            [f"Counting primes from {lo} to {hi} gives {ans}."], 1e-9)


def _g_master_diff_eq(rng: random.Random) -> tuple:
    c = rng.randint(1, 9)
    t = rng.randint(2, 4)
    ans = t * t + c  # y' = 2x, y(0) = c → y(t) = t² + c
    q = (f"y satisfies dy/dx = 2x and y(0) = {c}. What is y({t})?")
    steps = [f"Integrate: y = x² + C; y(0) = {c} gives C = {c}.",
             f"y({t}) = {t}² + {c} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_master_complex(rng: random.Random) -> tuple:
    a, b = rng.randint(-5, 5), rng.randint(-5, 5)
    c, d = rng.randint(-5, 5), rng.randint(-5, 5)
    part = rng.choice(["real", "imag"])
    real, imag = a * c - b * d, a * d + b * c
    ans = real if part == "real" else imag
    q = (f"({a}{b:+d}i) × ({c}{d:+d}i) — what is its {part} part?")
    steps = [f"({a}{b:+d}i)({c}{d:+d}i) = {real}{imag:+d}i.",
             f"The {part} part is {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_master_modular(rng: random.Random) -> tuple:
    base = rng.randint(3, 12)
    exp = rng.randint(10, 40)
    mod = rng.randint(20, 99)
    ans = pow(base, exp, mod)
    q = f"What is {base}^{exp} mod {mod}? (remainder on division by {mod})"
    steps = [f"By repeated squaring / modular reduction, "
             f"{base}^{exp} mod {mod} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_recurrence(rng: random.Random) -> tuple:
    p = rng.randint(2, 3)
    q_ = rng.randint(1, 5)
    a0 = rng.randint(1, 4)
    seq = [a0]
    for _ in range(6):
        seq.append(p * seq[-1] + q_)
    ans = seq[5] + seq[6]
    q = (f"Sequence: a₀ = {a0}, and aₙ₊₁ = {p}·aₙ + {q_}. "
         f"What is a₅ + a₆?")
    steps = [f"Iterate: " + ", ".join(f"a{i} = {v}" for i, v in enumerate(seq)) + ".",
             f"a₅ + a₆ = {seq[5]} + {seq[6]} = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_modular_sum(rng: random.Random) -> tuple:
    a, b = rng.randint(2, 9), rng.randint(50, 400)
    c, d = rng.randint(2, 9), rng.randint(50, 400)
    ans = (pow(a, b, 100) + pow(c, d, 100)) % 100
    q = (f"Find the remainder when {a}^{b} + {c}^{d} is divided by 100. "
         f"(i.e. its last two digits as a number 0–99)")
    steps = [f"{a}^{b} mod 100 = {pow(a, b, 100)}; {c}^{d} mod 100 = {pow(c, d, 100)}.",
             f"Sum mod 100 = ({pow(a, b, 100)} + {pow(c, d, 100)}) mod 100 = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_divisor_hunt(rng: random.Random) -> tuple:
    ea, eb, ec = rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 3)
    n_val = (2 ** ea) * (3 ** eb) * (5 ** ec)
    mult = rng.choice([6, 10, 12, 15, 30])
    divisors = [d for d in range(1, n_val + 1)
                if n_val % d == 0 and d % mult == 0]
    ans = len(divisors)
    q = (f"Let N = 2^{ea} × 3^{eb} × 5^{ec} = {n_val}. How many positive "
         f"divisors of N are multiples of {mult}?")
    steps = [f"N = {n_val}. Counting its divisors that are multiples of {mult} gives {ans}.",
             f"(Hint used: a divisor 2^a·3^b·5^c is a multiple of {mult} iff "
             f"its exponents cover {mult}'s prime factors.)"]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_three_eq(rng: random.Random) -> tuple:
    x, y, z = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    s1 = x + y + z
    s2 = x + 2 * y + 3 * z
    s3 = 2 * x - y + z
    ans = x + y + z
    q = (f"Solve: x + y + z = {s1}; x + 2y + 3z = {s2}; 2x − y + z = {s3}. "
         f"What is x + y + z? (x, y, z are positive integers)")
    steps = [f"Solving the three equations gives x = {x}, y = {y}, z = {z}.",
             f"x + y + z = {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_digit_sum(rng: random.Random) -> tuple:
    n = rng.randint(60, 120)
    ans = sum(int(ch) for ch in str(2 ** n))
    q = f"What is the sum of the decimal digits of 2^{n}?"
    steps = [f"2^{n} has {len(str(2 ** n))} digits; adding them all gives {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_factorial(rng: random.Random) -> tuple:
    k = rng.randint(3, 7)
    n = 1
    while True:
        # exponent of 5 in n! — the count of trailing zeros
        e, m = 0, n
        while m:
            m //= 5
            e += m
        if e >= k:
            break
        n += 1
    ans = n
    q = f"What is the SMALLEST n such that n! is divisible by 10^{k}?"
    steps = [f"n! divisible by 10^{k} needs at least {k} factors of 5 "
             f"(Legendre's formula for the exponent of 5 in n!).",
             f"The smallest n with that property is {ans}."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_collatz(rng: random.Random) -> tuple:
    n = rng.randint(27, 120)
    steps_count, m = 0, n
    while m != 1:
        m = m // 2 if m % 2 == 0 else 3 * m + 1
        steps_count += 1
    ans = steps_count
    q = (f"Collatz: if n is even halve it, if odd take 3n+1, repeat until 1. "
         f"Starting from {n}, how many steps to reach 1?")
    steps = [f"Iterating the Collatz rule from {n} reaches 1 after {ans} steps."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


def _g_imp_last_two(rng: random.Random) -> tuple:
    a = rng.randint(3, 19)
    b = rng.randint(200, 2026)
    ans = pow(a, b, 100)
    q = f"What are the last two digits of {a}^{b}? (give as a number 0–99)"
    steps = [f"Last two digits = {a}^{b} mod 100 = {ans} "
             f"(cyclicity / Euler's theorem does the heavy lifting)."]
    return q, ans, str(ans), f"Answer: {ans}", steps, 1e-9


_GENERATORS: dict[tuple[str, str], Gen] = {
    ("easy", "arithmetic"): _g_easy_arithmetic,
    ("easy", "times_tables"): _g_easy_times,
    ("easy", "fractions"): _g_easy_fractions,
    ("easy", "percentages"): _g_easy_percent,
    ("easy", "missing_number"): _g_easy_missing,
    ("medium", "algebra"): _g_med_algebra,
    ("medium", "geometry"): _g_med_geometry,
    ("medium", "ratios"): _g_med_ratios,
    ("medium", "word_problems"): _g_med_word,
    ("medium", "percent_change"): _g_med_percent_change,
    ("medium", "speed"): _g_med_speed,
    ("hard", "quadratics"): _g_hard_quadratic,
    ("hard", "trigonometry"): _g_hard_trig,
    ("hard", "logarithms"): _g_hard_log,
    ("hard", "sequences"): _g_hard_sequence,
    ("hard", "simultaneous"): _g_hard_simultaneous,
    ("hard", "exponents"): _g_hard_exponent,
    ("expert", "calculus"): _g_exp_calculus,
    ("expert", "combinatorics"): _g_exp_combinatorics,
    ("expert", "number_theory"): _g_exp_number_theory,
    ("expert", "vectors"): _g_exp_vectors,
    ("expert", "probability"): _g_exp_probability,
    ("expert", "trig_equations"): _g_exp_trig_eq,
    ("master", "olympiad"): _g_master_olympiad,
    ("master", "diophantine"): _g_master_diophantine,
    ("master", "primes"): _g_master_primes,
    ("master", "differential_equations"): _g_master_diff_eq,
    ("master", "complex"): _g_master_complex,
    ("master", "modular"): _g_master_modular,
    ("impossible", "chained_recurrence"): _g_imp_recurrence,
    ("impossible", "modular_sum"): _g_imp_modular_sum,
    ("impossible", "divisor_hunt"): _g_imp_divisor_hunt,
    ("impossible", "three_equations"): _g_imp_three_eq,
    ("impossible", "digit_sum"): _g_imp_digit_sum,
    ("impossible", "factorial_divisibility"): _g_imp_factorial,
    ("impossible", "collatz"): _g_imp_collatz,
    ("impossible", "last_two_digits"): _g_imp_last_two,
}


def daily_seed(date_iso: str) -> int:
    """Stable seed for a calendar day — the daily challenge is the same
    for every player on that day."""
    return int(hashlib.sha256(f"math-daily:{date_iso}".encode()).hexdigest()[:12], 16)


# ── engine: per-player streams, adaptive tier, anti-cheat, formats ────────

@dataclass
class Duel:
    """Head-to-head duel: both players answer the SAME seeded problems;
    first correct answer per round wins the round."""

    duel_id: str
    players: list[str]
    problems: list[Problem] = field(default_factory=list)
    round_winners: list[str | None] = field(default_factory=list)
    answers: dict[str, dict[int, Any]] = field(default_factory=dict)
    status: str = "open"  # open | finished
    winner: str | None = None
    created_at: float = 0.0

    def current_round(self) -> int:
        return len(self.round_winners)

    def score(self, player: str) -> int:
        return sum(1 for w in self.round_winners if w == player)


class MathClass:
    """The class itself: generation, adaptive tiers, anti-cheat, formats.

    State is in-memory by default; pass ``store`` (a dict-like) to share
    it across instances. All time is injectable via ``now`` params so
    tests can drive cooldowns/caps deterministically.
    """

    def __init__(self, store: dict[str, Any] | None = None) -> None:
        self.store: dict[str, Any] = store if store is not None else {}
        self.store.setdefault("seen", {})        # player -> {hash: True}
        self.store.setdefault("players", {})     # player -> adaptive state
        self.store.setdefault("attempts", {})    # (player, tier, day) -> count
        self.store.setdefault("cooldowns", {})   # (player, tier) -> ts
        self.store.setdefault("active", {})      # problem_id -> Problem
        self.store.setdefault("scores", {})      # player -> score dict
        self.store.setdefault("duels", {})       # duel_id -> Duel
        self.store.setdefault("counters", {})    # (player, tier) -> int
        self.store.setdefault("streaks", {})     # player -> current run streak
        self.store.setdefault("daily_best", {})  # (player, date) -> best

    # ── adaptive tier ──
    def _player_state(self, player: str) -> dict[str, Any]:
        players = self.store["players"]
        state = players.get(player)
        if state is None:
            state = {"tier": "medium", "win_run": 0, "loss_run": 0,
                     "wins": 0, "losses": 0}
            players[player] = state
        return state

    def adaptive_tier(self, player: str) -> str:
        return self._player_state(player)["tier"]

    def record_result(self, player: str, tier: str, correct: bool,
                      now: float | None = None) -> dict[str, Any]:
        """Feed a graded result into the adaptive ladder.

        Win 3 in a row → tier up. Lose 2 in a row → tier down.
        Returns the new state (tier, win_run, loss_run, moved).
        """
        state = self._player_state(player)
        idx = TIERS.index(tier) if tier in TIERS else TIERS.index(state["tier"])
        moved = ""
        if correct:
            state["win_run"] += 1
            state["loss_run"] = 0
            state["wins"] += 1
            if state["win_run"] >= PROMOTE_AFTER_WINS and idx < len(TIERS) - 1:
                state["tier"] = TIERS[idx + 1]
                state["win_run"] = 0
                moved = "up"
        else:
            state["loss_run"] += 1
            state["win_run"] = 0
            state["losses"] += 1
            if state["loss_run"] >= DEMOTE_AFTER_LOSSES and idx > 0:
                state["tier"] = TIERS[idx - 1]
                state["loss_run"] = 0
                moved = "down"
        return {"player": player, "tier": state["tier"], "moved": moved,
                "win_run": state["win_run"], "loss_run": state["loss_run"],
                "wins": state["wins"], "losses": state["losses"]}

    # ── anti-cheat gates ──
    @staticmethod
    def _day_key(now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def attempt_allowed(self, player: str, tier: str,
                        now: float | None = None) -> tuple[bool, str, float]:
        """(allowed, reason, wait_seconds). Cooldown + daily cap."""
        now = now if now is not None else time.time()
        tier = tier if tier in TIERS else "medium"
        cd_until = float(self.store["cooldowns"].get((player, tier), 0.0))
        if now < cd_until:
            return False, "cooldown", cd_until - now
        used = int(self.store["attempts"].get(
            (player, tier, self._day_key(now)), 0))
        if used >= ATTEMPT_CAPS[tier]:
            return False, "daily_cap", 0.0
        return True, "", 0.0

    def _note_attempt(self, player: str, tier: str, correct: bool,
                      now: float) -> None:
        key = (player, tier, self._day_key(now))
        self.store["attempts"][key] = int(self.store["attempts"].get(key, 0)) + 1
        if not correct:
            self.store["cooldowns"][(player, tier)] = (
                now + COOLDOWN_SECONDS[tier])

    # ── generation ──
    def _next_seed(self, player: str, tier: str) -> int:
        key = (player, tier)
        n = int(self.store["counters"].get(key, 0)) + 1
        self.store["counters"][key] = n
        raw = hashlib.sha256(f"math:{player}:{tier}:{n}".encode()).hexdigest()
        return int(raw[:12], 16)

    @staticmethod
    def _problem_hash(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]

    def generate(self, player: str, tier: str = "",
                 format: str = "quickfire",
                 topic: str = "", now: float | None = None) -> Problem:
        """Generate a fresh problem for ``player`` at ``tier``.

        Empty ``tier`` means the player's adaptive tier. Any player may
        request any tier — the adaptive tier never blocks an attempt.
        The same problem is never served twice to the same player
        (seen hashes); after 200 collisions the salt advances so
        generation always terminates.
        """
        now = now if now is not None else time.time()
        tier = tier.strip().lower() if tier else self.adaptive_tier(player)
        if tier not in TIERS:
            tier = "medium"
        fmt = format if format in FORMAT_CONFIG else "quickfire"
        topics = TIER_TOPICS[tier]
        topic = topic if topic in topics else ""
        seen = self.store["seen"].setdefault(player, {})
        for _ in range(200):
            seed = self._next_seed(player, tier)
            rng = random.Random(seed)
            pick = topic or rng.choice(topics)
            gen = _GENERATORS[(tier, pick)]
            question, ans, ans_text, sol, steps, tol = gen(rng)
            h = self._problem_hash(question)
            if h in seen:
                continue
            seen[h] = True
            break
        else:  # pragma: no cover — only if the space is exhausted
            seed = self._next_seed(player, tier)
            rng = random.Random(seed)
            pick = topic or rng.choice(topics)
            question, ans, ans_text, sol, steps, tol = _GENERATORS[(tier, pick)](rng)
        problem = Problem(
            problem_id=f"math_{seed:x}_{int(now)}",
            seed=seed, tier=tier, topic=pick, format=fmt,
            question=question, answer_value=ans, answer_text=ans_text,
            solution=sol, solution_steps=steps, tolerance=tol,
            time_limit_s=FORMAT_CONFIG[fmt]["seconds"] or 60,
            created_at=now)
        self.store["active"][problem.problem_id] = problem
        return problem

    def public_problem(self, problem: Problem) -> dict[str, Any]:
        """Player-facing view — question only, never the answer."""
        return problem.public()

    def daily_challenge(self, date_iso: str = "") -> Problem:
        """The day's challenge: same problem for every player."""
        date_iso = date_iso or time.strftime("%Y-%m-%d", time.gmtime())
        # Tier rotates through the week so every day feels different.
        tier = TIERS[time.strptime(date_iso, "%Y-%m-%d").tm_yday % len(TIERS)]
        seed = daily_seed(date_iso)
        rng = random.Random(seed)
        pick = rng.choice(TIER_TOPICS[tier])
        question, ans, ans_text, sol, steps, tol = _GENERATORS[(tier, pick)](rng)
        return Problem(
            problem_id=f"math_daily_{date_iso}", seed=seed, tier=tier,
            topic=pick, format="daily", question=question,
            answer_value=ans, answer_text=ans_text, solution=sol,
            solution_steps=steps, tolerance=tol,
            time_limit_s=FORMAT_CONFIG["daily"]["seconds"],
            created_at=time.time())

    # ── grading ──
    def submit(self, player: str, problem_id: str, submitted: Any,
               now: float | None = None) -> dict[str, Any]:
        """Grade an attempt. Wrong answers earn nothing; the worked
        solution is returned either way so the attempt teaches."""
        now = now if now is not None else time.time()
        problem = self.store["active"].get(problem_id)
        if problem is None:
            return {"ok": False, "error": "unknown_problem",
                    "message": "That problem is not active. Ask for a new one."}
        allowed, reason, wait = self.attempt_allowed(player, problem.tier, now)
        if not allowed:
            return {"ok": False, "error": reason, "wait_s": round(wait, 1),
                    "message": ("Cool down first — think it through."
                                if reason == "cooldown"
                                else "Daily attempt cap for this tier reached.")}
        correct = check_answer(problem, submitted)
        self._note_attempt(player, problem.tier, correct, now)
        adaptive = self.record_result(player, problem.tier, correct, now)
        scores = self.store["scores"].setdefault(
            player, {"wins": 0, "losses": 0, "xp": 0, "coins": 0,
                     "best_streak": 0})
        if correct:
            scores["wins"] += 1
            self.store["streaks"][player] = self.store["streaks"].get(player, 0) + 1
            scores["best_streak"] = max(scores["best_streak"],
                                        self.store["streaks"][player])
        else:
            scores["losses"] += 1
            self.store["streaks"][player] = 0
        out: dict[str, Any] = {
            "ok": True, "correct": correct, "tier": problem.tier,
            "problem_id": problem_id, "adaptive": adaptive,
            "streak": self.store["streaks"].get(player, 0),
        }
        out.update(problem.reveal())
        out.pop("problem_id", None)
        out["problem_id"] = problem_id
        return out

    # ── formats ──
    def timed_set(self, player: str, tier: str = "",
                  now: float | None = None) -> dict[str, Any]:
        """10 problems in 5 minutes. Returns the set (public views)."""
        now = now if now is not None else time.time()
        tier = tier or self.adaptive_tier(player)
        problems = [self.generate(player, tier, "timed_set", now=now)
                    for _ in range(FORMAT_CONFIG["timed_set"]["problems"])]
        return {"player": player, "tier": tier, "format": "timed_set",
                "seconds": FORMAT_CONFIG["timed_set"]["seconds"],
                "problems": [p.public() for p in problems]}

    def grade_timed_set(self, player: str, answers: dict[str, Any],
                        now: float | None = None) -> dict[str, Any]:
        """Grade a whole timed set at once. ``answers`` maps
        problem_id → submitted answer."""
        now = now if now is not None else time.time()
        correct = 0
        details = []
        for pid, submitted in answers.items():
            res = self.submit(player, pid, submitted, now=now)
            if res.get("ok") and res.get("correct"):
                correct += 1
            details.append(res)
        return {"player": player, "correct": correct,
                "total": len(answers), "details": details}

    def streak_start(self, player: str, tier: str = "",
                     now: float | None = None) -> dict[str, Any]:
        """Start (or continue) a streak run: one problem at a time."""
        problem = self.generate(player, tier or self.adaptive_tier(player),
                                "streak", now=now)
        return {"player": player, "format": "streak",
                "streak": self.store["streaks"].get(player, 0),
                "problem": problem.public()}

    # ── duels ──
    def create_duel(self, player_a: str, player_b: str,
                    tier: str = "", now: float | None = None) -> Duel:
        """A duel: both players get the SAME seeded problems, so the
        contest is fair — first correct answer per round wins it."""
        now = now if now is not None else time.time()
        tier = tier or self.adaptive_tier(player_a)
        seed_base = int(hashlib.sha256(
            f"duel:{player_a}:{player_b}:{int(now)}".encode()).hexdigest()[:12], 16)
        problems = []
        for i in range(FORMAT_CONFIG["duel"]["problems"]):
            rng = random.Random(seed_base + i)
            pick = rng.choice(TIER_TOPICS[tier if tier in TIERS else "medium"])
            question, ans, ans_text, sol, steps, tol = _GENERATORS[
                (tier if tier in TIERS else "medium", pick)](rng)
            problems.append(Problem(
                problem_id=f"duel_{seed_base:x}_{i}", seed=seed_base + i,
                tier=tier, topic=pick, format="duel", question=question,
                answer_value=ans, answer_text=ans_text, solution=sol,
                solution_steps=steps, tolerance=tol,
                time_limit_s=FORMAT_CONFIG["duel"]["seconds"],
                created_at=now))
            self.store["active"][problems[-1].problem_id] = problems[-1]
        duel = Duel(duel_id=f"duel_{seed_base:x}",
                    players=[player_a, player_b], problems=problems,
                    answers={player_a: {}, player_b: {}},
                    created_at=now)
        self.store["duels"][duel.duel_id] = duel
        return duel

    def duel_answer(self, duel_id: str, player: str, round_idx: int,
                    submitted: Any, now: float | None = None) -> dict[str, Any]:
        """Answer one duel round. First correct answer wins the round;
        a later correct answer does not steal it."""
        duel = self.store["duels"].get(duel_id)
        if duel is None or duel.status != "open":
            return {"ok": False, "error": "no_duel"}
        if player not in duel.players:
            return {"ok": False, "error": "not_in_duel"}
        if not (0 <= round_idx < len(duel.problems)):
            return {"ok": False, "error": "bad_round"}
        if round_idx != len(duel.round_winners):
            return {"ok": False, "error": "round_not_current",
                    "current_round": len(duel.round_winners),
                    "winner": (duel.round_winners[round_idx]
                               if round_idx < len(duel.round_winners) else None)}
        correct = check_answer(duel.problems[round_idx], submitted)
        duel.answers[player][round_idx] = submitted
        if correct:
            duel.round_winners.append(player)
            self.record_result(player, duel.problems[round_idx].tier, True)
        else:
            # A wrong answer doesn't close the round — the opponent
            # can still win it. Only a correct answer closes a round.
            return {"ok": True, "correct": False, "round": round_idx,
                    "message": "Not quite — your opponent can still take this round."}
        if len(duel.round_winners) >= len(duel.problems):
            duel.status = "finished"
            sa = duel.score(duel.players[0])
            sb = duel.score(duel.players[1])
            duel.winner = (duel.players[0] if sa > sb
                           else duel.players[1] if sb > sa else None)
        return {"ok": True, "correct": True, "round": round_idx,
                "tier": duel.problems[round_idx].tier,
                "round_winner": player, "duel_winner": duel.winner,
                "score": {p: duel.score(p) for p in duel.players},
                "status": duel.status}

    # ── leaderboard ──
    def leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Top solvers by wins, then best streak, then fewest losses."""
        rows = [{"player": p,
                 "wins": s.get("wins", 0), "losses": s.get("losses", 0),
                 "best_streak": s.get("best_streak", 0),
                 "coins": s.get("coins", 0), "xp": s.get("xp", 0)}
                for p, s in self.store["scores"].items()]
        rows.sort(key=lambda r: (-r["wins"], -r["best_streak"], r["losses"]))
        return rows[:max(1, int(limit))]
