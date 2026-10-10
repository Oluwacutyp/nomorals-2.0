"""Photo meal logging with 2-question completion.

Fixes the exact NIH failure mode: trusting photos alone undercounts calories
by ~33%. The fix is product, not model: ask at most 2 questions about what the
photo CAN'T show — hidden fats, portion scale, drinks — and never about what's
clearly visible.

Estimates are always ranges ("approximately 600-800 cal"), never false
precision. This is TRACKING ONLY — no nutrition advice ("you should eat X")
ever appears in any output of this module.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "NIGERIAN_FOODS",
    "FoodEntry",
    "MealItem",
    "MealDraft",
    "MealLog",
    "DayCard",
    "NutritionStore",
    "log_meal",
    "follow_up_questions",
    "complete_meal",
    "parse_meal_answers",
    "arm_meal",
    "arm_meal_flow",
    "pending_meal",
    "consume_meal",
    "disarm_meal_flow",
    "meal_flow_armed",
    "meal_intent_in_text",
    "format_draft_message",
    "daily_totals",
    "log_water",
    "MACRO_TARGETS",
]

_log = get_logger(__name__)


# ── Nigerian food database ──────────────────────────────────────────────────
# Approximate calories per typical serving. These are ESTIMATES from standard
# nutrition references for Nigerian dishes — always presented as ranges.
# Serving sizes are "typical plate/bowl as served", not lab portions.

@dataclass(frozen=True)
class FoodEntry:
    name: str
    cal_low: int
    cal_high: int
    serving: str
    oily: bool = False  # hidden fats likely — triggers the oil question
    aliases: tuple[str, ...] = ()
    # macro estimates per serving (protein, carbs, fat grams) — from
    # standard nutrition references, always presented as ranges.
    protein_g: tuple[float, float] = (0.0, 0.0)
    carbs_g: tuple[float, float] = (0.0, 0.0)
    fat_g: tuple[float, float] = (0.0, 0.0)


NIGERIAN_FOODS: dict[str, FoodEntry] = {}

def _food(name: str, low: int, high: int, serving: str, *,
          oily: bool = False, aliases: tuple[str, ...] = (),
          p: tuple[float, float] = (0.0, 0.0),
          c: tuple[float, float] = (0.0, 0.0),
          f: tuple[float, float] = (0.0, 0.0)) -> None:
    NIGERIAN_FOODS[name] = FoodEntry(name, low, high, serving,
                                     oily=oily, aliases=aliases,
                                     protein_g=p, carbs_g=c, fat_g=f)

_food("jollof rice", 450, 700, "per plate", oily=True,
      aliases=("jollof", "party jollof"),
      p=(12, 18), c=(70, 95), f=(15, 25))
_food("fried rice", 450, 700, "per plate", oily=True,
      p=(12, 18), c=(65, 90), f=(18, 28))
_food("white rice", 350, 550, "per plate",
      aliases=("plain rice", "boiled rice"),
      p=(7, 10), c=(75, 100), f=(2, 4))
_food("ofada rice", 400, 650, "per plate", oily=True,
      aliases=("ofada",), p=(10, 15), c=(70, 95), f=(12, 20))
_food("pounded yam", 350, 550, "per portion (2 wraps)",
      aliases=("iyan",), p=(5, 8), c=(80, 110), f=(2, 5))
_food("fufu", 300, 500, "per portion", aliases=("akpu",),
      p=(4, 7), c=(70, 95), f=(2, 4))
_food("amala", 300, 500, "per portion", p=(5, 8), c=(65, 90), f=(3, 6))
_food("eba", 350, 550, "per portion", aliases=("garri",),
      p=(4, 6), c=(85, 115), f=(2, 4))
_food("egusi soup", 300, 550, "per bowl", oily=True, aliases=("egusi",),
      p=(15, 25), c=(8, 15), f=(25, 40))
_food("efo riro", 250, 450, "per serving", oily=True,
      p=(12, 20), c=(8, 14), f=(18, 30))
_food("okro soup", 200, 350, "per serving", aliases=("okra soup", "okro"),
      p=(8, 14), c=(10, 18), f=(10, 18))
_food("oha soup", 300, 500, "per serving", oily=True, aliases=("oha",),
      p=(14, 22), c=(10, 16), f=(22, 35))
_food("bitter leaf soup", 300, 500, "per serving", oily=True,
      aliases=("ofe onugbu",), p=(12, 20), c=(10, 16), f=(20, 32))
_food("pepper soup", 150, 300, "per bowl", aliases=("point and kill",),
      p=(20, 30), c=(3, 6), f=(8, 15))
_food("beans", 300, 500, "per plate", aliases=("beans porridge", "ewa",
      "honey beans"), p=(15, 22), c=(45, 65), f=(8, 15))
_food("moi moi", 200, 350, "per wrap", aliases=("moin moin", "olele"),
      p=(12, 18), c=(20, 30), f=(8, 14))
_food("dodo", 200, 350, "per serving", oily=True,
      aliases=("fried plantain", "plantain"), p=(2, 4), c=(45, 65),
      f=(10, 18))
_food("boli", 200, 300, "per serving", aliases=("roasted plantain", "bole"),
      p=(2, 3), c=(50, 70), f=(1, 3))
_food("akara", 200, 350, "per serving (4-5 balls)", oily=True,
      aliases=("bean cake",), p=(8, 12), c=(20, 30), f=(12, 20))
_food("puff puff", 200, 350, "per serving (4-5 pieces)", oily=True,
      p=(3, 5), c=(35, 50), f=(10, 16))
_food("suya", 250, 450, "per serving", p=(25, 35), c=(3, 6), f=(15, 25))
_food("nkwobi", 400, 600, "per plate", oily=True,
      p=(20, 30), c=(5, 10), f=(30, 45))
_food("ewedu", 100, 200, "per serving", p=(5, 8), c=(10, 18), f=(4, 8))
_food("gbegiri", 150, 250, "per serving", p=(8, 12), c=(18, 28), f=(5, 10))
_food("zobo", 80, 150, "per glass", aliases=("zoborodo", "hibiscus drink"),
      p=(0, 1), c=(20, 35), f=(0, 0))
_food("chapman", 150, 250, "per glass", p=(0, 1), c=(35, 60), f=(0, 0))
_food("kunu", 150, 250, "per glass", aliases=("kunu aya", "tigernut drink"),
      p=(3, 5), c=(30, 45), f=(8, 14))
_food("agege bread", 250, 400, "per portion",
      aliases=("bread", "agege"), p=(8, 12), c=(50, 70), f=(5, 10))
_food("roasted corn", 150, 250, "per cob", aliases=("corn", "roasted maize"),
      p=(4, 6), c=(30, 45), f=(2, 4))
_food("meat", 200, 400, "per serving",
      aliases=("beef", "chicken", "turkey", "goat meat", "assorted meat",
               "chicken pieces", "turkey pieces"),
      p=(25, 35), c=(0, 2), f=(12, 25))
_food("fish", 150, 300, "per serving",
      aliases=("tilapia", "croaker", "mackerel", "titus", "fried fish"),
      p=(20, 30), c=(0, 2), f=(8, 18))

#: Drinks are tracked separately for the drink follow-up.
_DRINK_NAMES = {"zobo", "chapman", "kunu", "soda", "coke", "fanta",
                "sprite", "malt", "water", "juice", "beer"}


# ── data types ──────────────────────────────────────────────────────────────

@dataclass
class MealItem:
    name: str
    portion_hint: str = ""
    cal_low: int = 300
    cal_high: int = 500
    known: bool = True   # False = not in DB, honest generic estimate
    oily: bool = False
    protein_g: tuple[float, float] = (0.0, 0.0)
    carbs_g: tuple[float, float] = (0.0, 0.0)
    fat_g: tuple[float, float] = (0.0, 0.0)

    @property
    def macro_text(self) -> str:
        p = f"{self.protein_g[0]:.0f}–{self.protein_g[1]:.0f}g protein"
        c = f"{self.carbs_g[0]:.0f}–{self.carbs_g[1]:.0f}g carbs"
        f = f"{self.fat_g[0]:.0f}–{self.fat_g[1]:.0f}g fat"
        return f"{p}, {c}, {f}"


@dataclass
class MealDraft:
    photo_path: str
    items: list[MealItem] = field(default_factory=list)
    raw_description: str = ""
    error: str = ""      # set when vision failed — honest, no fake items

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.items)


@dataclass
class MealLog:
    items: list[MealItem]
    answers: dict[str, str]
    cal_low: int
    cal_high: int
    cost_kobo: int | None = None
    timeline_event_id: str = ""
    expense_txn_id: str = ""

    @property
    def range_text(self) -> str:
        return f"approximately {self.cal_low}-{self.cal_high} cal"

    @property
    def macro_totals(self) -> dict[str, tuple[float, float]]:
        p = [0.0, 0.0]
        c = [0.0, 0.0]
        f = [0.0, 0.0]
        for i in self.items:
            p[0] += i.protein_g[0]; p[1] += i.protein_g[1]
            c[0] += i.carbs_g[0]; c[1] += i.carbs_g[1]
            f[0] += i.fat_g[0]; f[1] += i.fat_g[1]
        return {"protein": (round(p[0], 1), round(p[1], 1)),
                "carbs": (round(c[0], 1), round(c[1], 1)),
                "fat": (round(f[0], 1), round(f[1], 1))}

    def summary(self) -> str:
        names = ", ".join(i.name for i in self.items) or "meal"
        parts = [f"🍽️ logged: {names} — {self.range_text}."]
        mt = self.macro_totals
        parts.append(
            f"~{mt['protein'][0]:.0f}–{mt['protein'][1]:.0f}g protein, "
            f"{mt['carbs'][0]:.0f}–{mt['carbs'][1]:.0f}g carbs, "
            f"{mt['fat'][0]:.0f}–{mt['fat'][1]:.0f}g fat (estimates).")
        if self.cost_kobo:
            from ..finance.ledger import format_naira
            parts.append(f"cost {format_naira(self.cost_kobo)}.")
        parts.append("tracking only — not nutrition advice.")
        return " ".join(parts)


# ── food identification ─────────────────────────────────────────────────────

_SEER_PROMPT = (
    "List each distinct food or drink visible in this photo. "
    "One per line, exactly like this: name | portion you can see "
    "(e.g. 'jollof rice | one plate', 'dodo | a few pieces'). "
    "Only list what you can actually see. "
    "If nothing edible is visible, reply with exactly: NONE"
)

def _match_food(name: str) -> FoodEntry | None:
    """Fuzzy-match a food name against the database."""
    norm = name.strip().lower()
    if not norm:
        return None
    if norm in NIGERIAN_FOODS:
        return NIGERIAN_FOODS[norm]
    for entry in NIGERIAN_FOODS.values():
        if norm == entry.name:
            return entry
        if norm in entry.aliases:
            return entry
    # substring fallback: "plate of jollof rice" -> jollof rice
    for entry in NIGERIAN_FOODS.values():
        if entry.name in norm or norm in entry.name:
            return entry
        for alias in entry.aliases:
            if alias in norm:
                return entry
    return None


def _parse_seer_lines(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line or line.upper() == "NONE":
            continue
        if "|" in line:
            name, _, portion = line.partition("|")
        else:
            name, portion = line, ""
        name = name.strip()
        if name:
            out.append((name, portion.strip()))
    return out


def log_meal(photo_path: str | Any, *, seer: Any = None) -> MealDraft:
    """Identify foods in a meal photo. Never raises; failures are honest."""
    path = str(photo_path)
    draft = MealDraft(photo_path=path)
    try:
        if seer is None:
            from ..vision.seer import get_seer
            seer = get_seer()
        description = seer.see(path, _SEER_PROMPT)
        draft.raw_description = description or ""
        if not description or description.strip().upper() == "NONE":
            draft.error = "no food visible in the photo"
            return draft
        for name, portion in _parse_seer_lines(description):
            entry = _match_food(name)
            if entry is not None:
                draft.items.append(MealItem(
                    name=entry.name, portion_hint=portion,
                    cal_low=entry.cal_low, cal_high=entry.cal_high,
                    known=True, oily=entry.oily,
                    protein_g=entry.protein_g, carbs_g=entry.carbs_g,
                    fat_g=entry.fat_g))
            else:
                # Honest generic estimate for unknown foods — marked unknown.
                draft.items.append(MealItem(
                    name=name, portion_hint=portion,
                    cal_low=250, cal_high=450, known=False,
                    protein_g=(8, 15), carbs_g=(30, 50),
                    fat_g=(8, 15)))
        if not draft.items:
            draft.error = "couldn't identify any food in the photo"
    except Exception as exc:  # noqa: BLE001 — vision failures are honest
        draft.error = f"vision unavailable: {exc}"
        _log.debug("log_meal vision failed", exc_info=True)
    return draft


# ── the 2 questions ─────────────────────────────────────────────────────────
# Only what the photo CAN'T answer. Never about what's clearly visible.

_Q_OIL = ("was this cooked with much oil, or fried in oil? "
          "(invisible fats are the #1 reason photo estimates come in low)")
_Q_PORTION = "roughly what size portion — small, medium, or large?"
_Q_DRINK = "anything to drink with that? (water, zobo, soda, malt…)"


def follow_up_questions(draft: MealDraft) -> list[str]:
    """At most 2 questions, only about what the photo can't show."""
    if not draft.ok:
        return []
    questions: list[str] = []
    if any(i.oily for i in draft.items):
        questions.append(_Q_OIL)
    # Portion scale is almost never certain from a photo alone.
    questions.append(_Q_PORTION)
    if len(questions) < 2:
        questions.append(_Q_DRINK)
    return questions[:2]


def format_draft_message(draft: MealDraft) -> str:
    """The chat message: what I see + the follow-up questions."""
    if not draft.ok:
        return (f"couldn't log this one — {draft.error}. "
                "try a clearer photo of the food.")
    names = ", ".join(
        f"{i.name}" + (f" ({i.portion_hint})" if i.portion_hint else "")
        for i in draft.items)
    low = sum(i.cal_low for i in draft.items)
    high = sum(i.cal_high for i in draft.items)
    lines = [f"i see: {names}.",
             f"photo estimate: approximately {low}-{high} cal — "
             "but photos alone undercount by about a third, so two quick "
             "questions:"]
    for n, q in enumerate(follow_up_questions(draft), 1):
        lines.append(f"{n}. {q}")
    return "\n".join(lines)


# ── answer parsing + completion ─────────────────────────────────────────────

_YES = re.compile(r"\b(yes|yeah|yep|plenty|a lot|much|fried)\b", re.I)
_NO = re.compile(r"\b(no|not|none|small amount|little)\b", re.I)
_PORTION = re.compile(r"\b(small|medium|large|big|huge|tiny)\b", re.I)


def parse_meal_answers(text: str, questions: list[str]) -> dict[str, str]:
    """Parse free-text answers to the follow-up questions. Best effort."""
    text = (text or "").strip()
    answers: dict[str, str] = {}
    low = text.lower()
    for q in questions:
        if q == _Q_OIL:
            if _YES.search(low) and not _NO.search(low):
                answers["oil"] = "yes"
            elif _NO.search(low):
                answers["oil"] = "no"
            else:
                answers["oil"] = "unknown"
        elif q == _Q_PORTION:
            m = _PORTION.search(low)
            answers["portion"] = m.group(1).lower() if m else "medium"
        elif q == _Q_DRINK:
            if _NO.search(low) or "water" in low and "only" in low:
                answers["drink"] = "none"
            else:
                # keep the raw text; complete_meal matches drink names
                answers["drink"] = text if text and not _NO.search(low) \
                    else "none"
    return answers


def _apply_adjustments(draft: MealDraft,
                       answers: dict[str, str]) -> tuple[int, int,
                                                         list[MealItem]]:
    items = [MealItem(**{**i.__dict__}) for i in draft.items]
    low = sum(i.cal_low for i in items)
    high = sum(i.cal_high for i in items)

    if answers.get("oil") == "yes":
        # Invisible cooking fats: the NIH 33% lives here.
        low += 100
        high += 150
        for i in items:
            lo, hi = i.fat_g
            i.fat_g = (round(lo + 11, 1), round(hi + 17, 1))
    portion = (answers.get("portion") or "medium").lower()
    if portion in ("large", "big", "huge"):
        low, high = int(low * 1.25), int(high * 1.25)
    elif portion in ("small", "tiny"):
        low, high = int(low * 0.75), int(high * 0.75)

    drink = (answers.get("drink") or "none").strip().lower()
    if drink and drink != "none":
        for dname in _DRINK_NAMES:
            if dname in drink:
                entry = NIGERIAN_FOODS.get(dname)
                if entry:
                    items.append(MealItem(
                        name=entry.name, portion_hint="one serving",
                        cal_low=entry.cal_low, cal_high=entry.cal_high,
                        known=True))
                    low += entry.cal_low
                    high += entry.cal_high
                break
    return low, high, items


def complete_meal(draft: MealDraft, answers: dict[str, str], *,
                  cost_kobo: int | None = None,
                  timeline: Any = None,
                  ledger: Any = None,
                  store: NutritionStore | None = None,
                  source: str = "chat") -> MealLog:
    """Apply answers, finalize the range, log to timeline (+ledger).

    Never raises on logging failures — the MealLog is always returned with
    what was recorded.
    """
    low, high, items = _apply_adjustments(draft, answers)
    log = MealLog(items=items, answers=dict(answers),
                  cal_low=low, cal_high=high, cost_kobo=cost_kobo)
    names = ", ".join(i.name for i in items) or "meal"

    # 1. Health timeline (tracking only — never interprets).
    try:
        tl = timeline
        if tl is None:
            from .timeline import HealthTimeline
            tl = HealthTimeline()
        ev = tl.log(
            "note",
            f"meal: {names} — approximately {low}-{high} cal",
            value=f"{low}-{high}", unit="cal", source=source,
            items=[i.name for i in items], answers=dict(answers))
        log.timeline_event_id = ev.id
    except Exception:  # noqa: BLE001
        _log.debug("meal timeline log failed", exc_info=True)

    # 2. Nutrition store (daily totals, repeats, water).
    try:
        (store or NutritionStore()).save_meal(log)
    except Exception:  # noqa: BLE001
        _log.debug("meal store save failed", exc_info=True)

    # 3. Optional expense pairing: "that jollof cost ₦2,500".
    if cost_kobo and cost_kobo > 0 and ledger is not None:
        try:
            txn = ledger.log(int(cost_kobo), category="food",
                             note=f"meal: {names}", source="meal")
            log.expense_txn_id = getattr(txn, "id", "")
        except Exception:  # noqa: BLE001
            _log.debug("meal expense log failed", exc_info=True)
    return log


# ── persistence: meals + water ────────────────────────────────────────────

#: Daily targets (general adult defaults; tracking reference, not advice).
MACRO_TARGETS = {
    "calories": (2000, 2500),
    "protein_g": (70, 120),
    "water_ml": 2500,
}

_DEFAULT_NUTRITION_DB = None  # resolved lazily (keeps import side-effect free)


def _nutrition_db_path() -> str:
    import os
    base = os.environ.get("NOMORALS_HOME", os.path.expanduser("~/.nomorals"))
    return os.path.join(base, "health", "nutrition.db")


class NutritionStore:
    """SQLite store: completed meals + water. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        import os
        import sqlite3
        self._db = None
        try:
            p = db_path or _nutrition_db_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS meals (
                       id TEXT PRIMARY KEY, day TEXT NOT NULL,
                       items_json TEXT NOT NULL,
                       cal_low INTEGER, cal_high INTEGER,
                       protein_lo REAL, protein_hi REAL,
                       carbs_lo REAL, carbs_hi REAL,
                       fat_lo REAL, fat_hi REAL,
                       created_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS water (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       day TEXT NOT NULL, ml INTEGER NOT NULL,
                       created_at REAL)""")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_meals_day ON meals(day)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_water_day ON water(day)")
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("nutrition store unavailable", exc_info=True)
            self._db = None

    def save_meal(self, log: MealLog, day: str = "") -> str:
        """Persist a completed meal. Returns the meal id ("" on failure)."""
        import json as _json
        import uuid as _uuid
        try:
            if self._db is None:
                return ""
            day = day or time.strftime("%Y-%m-%d")
            mt = log.macro_totals
            mid = "meal_" + _uuid.uuid4().hex[:8]
            self._db.execute(
                """INSERT INTO meals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mid, day,
                 _json.dumps([i.__dict__ for i in log.items]),
                 log.cal_low, log.cal_high,
                 mt["protein"][0], mt["protein"][1],
                 mt["carbs"][0], mt["carbs"][1],
                 mt["fat"][0], mt["fat"][1], time.time()))
            self._db.commit()
            return mid
        except Exception:  # noqa: BLE001
            _log.debug("save_meal failed", exc_info=True)
            return ""

    def meals_on(self, day: str) -> list[dict[str, Any]]:
        """Meals logged on a YYYY-MM-DD day. Never raises."""
        import json as _json
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM meals WHERE day = ? ORDER BY created_at",
                (day,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["items"] = _json.loads(d["items_json"] or "[]")
                except Exception:  # noqa: BLE001
                    d["items"] = []
                out.append(d)
            return out
        except Exception:  # noqa: BLE001
            return []

    def recent_meals(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            import json as _json
            rows = self._db.execute(
                "SELECT * FROM meals ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["items"] = _json.loads(d["items_json"] or "[]")
                except Exception:  # noqa: BLE001
                    d["items"] = []
                out.append(d)
            return out
        except Exception:  # noqa: BLE001
            return []

    def log_water(self, ml: int, day: str = "") -> int:
        """Log water intake (ml). Returns today's total ml."""
        try:
            if self._db is None:
                return 0
            day = day or time.strftime("%Y-%m-%d")
            self._db.execute(
                "INSERT INTO water (day, ml, created_at) VALUES (?,?,?)",
                (day, max(0, int(ml)), time.time()))
            self._db.commit()
            return self.water_on(day)
        except Exception:  # noqa: BLE001
            return 0

    def water_on(self, day: str) -> int:
        try:
            if self._db is None:
                return 0
            row = self._db.execute(
                "SELECT SUM(ml) AS total FROM water WHERE day = ?",
                (day,)).fetchone()
            return int(row["total"] or 0)
        except Exception:  # noqa: BLE001
            return 0


@dataclass
class DayCard:
    """One day's nutrition: meals + water vs targets. Tracking only."""
    day: str
    meals: int = 0
    cal_low: int = 0
    cal_high: int = 0
    protein_lo: float = 0.0
    protein_hi: float = 0.0
    carbs_lo: float = 0.0
    carbs_hi: float = 0.0
    fat_lo: float = 0.0
    fat_hi: float = 0.0
    water_ml: int = 0

    def format(self) -> str:
        try:
            t = MACRO_TARGETS
            lines = [f"🍽️ **today** — {self.meals} meal(s) logged:"]
            lines.append(f"• calories: ~{self.cal_low}–{self.cal_high} "
                         f"(ref {t['calories'][0]}–{t['calories'][1]})")
            lines.append(f"• protein: ~{self.protein_lo:.0f}–"
                         f"{self.protein_hi:.0f}g "
                         f"(ref {t['protein_g'][0]}–{t['protein_g'][1]}g)")
            lines.append(f"• carbs: ~{self.carbs_lo:.0f}–"
                         f"{self.carbs_hi:.0f}g · fat: ~{self.fat_lo:.0f}–"
                         f"{self.fat_hi:.0f}g")
            wtarget = t["water_ml"]
            wbar = "█" * min(10, int(self.water_ml / wtarget * 10)) + \
                "░" * max(0, 10 - min(10, int(self.water_ml / wtarget * 10)))
            lines.append(f"• 💧 water: {wbar} {self.water_ml}/{wtarget}ml")
            lines.append("_tracking only — ranges are estimates, not "
                         "nutrition advice._")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "couldn't build today's card."


def daily_totals(store: NutritionStore | None = None,
                 day: str = "") -> DayCard:
    """Today's (or a day's) logged nutrition vs targets. Never raises."""
    store = store or NutritionStore()
    day = day or time.strftime("%Y-%m-%d")
    card = DayCard(day=day)
    try:
        meals = store.meals_on(day)
        card.meals = len(meals)
        for m in meals:
            card.cal_low += int(m.get("cal_low") or 0)
            card.cal_high += int(m.get("cal_high") or 0)
            card.protein_lo += float(m.get("protein_lo") or 0)
            card.protein_hi += float(m.get("protein_hi") or 0)
            card.carbs_lo += float(m.get("carbs_lo") or 0)
            card.carbs_hi += float(m.get("carbs_hi") or 0)
            card.fat_lo += float(m.get("fat_lo") or 0)
            card.fat_hi += float(m.get("fat_hi") or 0)
        card.water_ml = store.water_on(day)
        return card
    except Exception:  # noqa: BLE001
        return card


def log_water(ml: int, store: NutritionStore | None = None,
              day: str = "") -> str:
    """Log water; returns a one-line confirmation. Never raises."""
    try:
        store = store or NutritionStore()
        total = store.log_water(ml, day=day or time.strftime("%Y-%m-%d"))
        target = MACRO_TARGETS["water_ml"]
        return (f"💧 +{ml}ml water — {total}/{target}ml today.")
    except Exception:  # noqa: BLE001
        return "couldn't log water right now."


def repeat_meal(store: NutritionStore | None = None,
                index: int = 0) -> MealDraft:
    """'Same as last time': rebuild a MealDraft from a recent meal.

    The highest-frequency real-world action — no photo needed.
    Never raises.
    """
    store = store or NutritionStore()
    draft = MealDraft(photo_path="repeat")
    try:
        recent = store.recent_meals(limit=10)
        if not recent or index >= len(recent):
            draft.error = "no recent meals to repeat yet"
            return draft
        for item in recent[index].get("items") or []:
            draft.items.append(MealItem(
                name=item.get("name", "meal"),
                portion_hint=item.get("portion_hint", ""),
                cal_low=int(item.get("cal_low") or 300),
                cal_high=int(item.get("cal_high") or 500),
                known=bool(item.get("known", True)),
                oily=bool(item.get("oily", False)),
                protein_g=tuple(item.get("protein_g") or (0, 0)),
                carbs_g=tuple(item.get("carbs_g") or (0, 0)),
                fat_g=tuple(item.get("fat_g") or (0, 0))))
        draft.raw_description = "repeated from a previous log"
        return draft
    except Exception:  # noqa: BLE001
        draft.error = "couldn't repeat that meal"
        return draft


# ── chat glue ───────────────────────────────────────────────────────────────

_MEAL_INTENT = re.compile(
    r"\b(meal|food|ate|eating|lunch|dinner|breakfast|calories?|"
    r"log (this |my )?food|what i (ate|eat))\b", re.I)

#: chat_key -> MealDraft awaiting answers
_pending: dict[str, MealDraft] = {}

#: chat_keys with an armed "/health meal" flow (next photo starts logging)
_armed: set[str] = set()


def meal_intent_in_text(text: str) -> bool:
    return bool(_MEAL_INTENT.search(text or ""))


def arm_meal(chat_key: str, draft: MealDraft) -> None:
    _pending[chat_key] = draft


def pending_meal(chat_key: str) -> MealDraft | None:
    return _pending.get(chat_key)


def consume_meal(chat_key: str) -> MealDraft | None:
    return _pending.pop(chat_key, None)


def arm_meal_flow(chat_key: str) -> None:
    """Arm the meal flow: the next photo in this chat starts logging."""
    _armed.add(chat_key)


def meal_flow_armed(chat_key: str) -> bool:
    return chat_key in _armed


def disarm_meal_flow(chat_key: str) -> None:
    _armed.discard(chat_key)
