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
    "log_meal",
    "follow_up_questions",
    "complete_meal",
    "parse_meal_answers",
    "arm_meal",
    "pending_meal",
    "consume_meal",
    "meal_intent_in_text",
    "format_draft_message",
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


NIGERIAN_FOODS: dict[str, FoodEntry] = {}

def _food(name: str, low: int, high: int, serving: str, *,
          oily: bool = False, aliases: tuple[str, ...] = ()) -> None:
    NIGERIAN_FOODS[name] = FoodEntry(name, low, high, serving,
                                    oily=oily, aliases=aliases)

_food("jollof rice", 450, 700, "per plate", oily=True,
      aliases=("jollof", "party jollof"))
_food("fried rice", 450, 700, "per plate", oily=True)
_food("white rice", 350, 550, "per plate",
      aliases=("plain rice", "boiled rice"))
_food("ofada rice", 400, 650, "per plate", oily=True,
      aliases=("ofada",))
_food("pounded yam", 350, 550, "per portion (2 wraps)", aliases=("iyan",))
_food("fufu", 300, 500, "per portion", aliases=("akpu",))
_food("amala", 300, 500, "per portion")
_food("eba", 350, 550, "per portion", aliases=("garri",))
_food("egusi soup", 300, 550, "per bowl", oily=True, aliases=("egusi",))
_food("efo riro", 250, 450, "per serving", oily=True)
_food("okro soup", 200, 350, "per serving", aliases=("okra soup", "okro"))
_food("oha soup", 300, 500, "per serving", oily=True, aliases=("oha",))
_food("bitter leaf soup", 300, 500, "per serving", oily=True,
      aliases=("ofe onugbu",))
_food("pepper soup", 150, 300, "per bowl", aliases=("point and kill",))
_food("beans", 300, 500, "per plate", aliases=("beans porridge", "ewa",
      "honey beans"))
_food("moi moi", 200, 350, "per wrap", aliases=("moin moin", "olele"))
_food("dodo", 200, 350, "per serving", oily=True,
      aliases=("fried plantain", "plantain"))
_food("boli", 200, 300, "per serving", aliases=("roasted plantain", "bole"))
_food("akara", 200, 350, "per serving (4-5 balls)", oily=True,
      aliases=("bean cake",))
_food("puff puff", 200, 350, "per serving (4-5 pieces)", oily=True)
_food("suya", 250, 450, "per serving")
_food("nkwobi", 400, 600, "per plate", oily=True)
_food("ewedu", 100, 200, "per serving")
_food("gbegiri", 150, 250, "per serving")
_food("zobo", 80, 150, "per glass", aliases=("zoborodo", "hibiscus drink"))
_food("chapman", 150, 250, "per glass")
_food("kunu", 150, 250, "per glass", aliases=("kunu aya", "tigernut drink"))
_food("agege bread", 250, 400, "per portion",
      aliases=("bread", "agege"))
_food("roasted corn", 150, 250, "per cob", aliases=("corn", "roasted maize"))
_food("meat", 200, 400, "per serving",
      aliases=("beef", "chicken", "turkey", "goat meat", "assorted meat",
               "chicken pieces", "turkey pieces"))
_food("fish", 150, 300, "per serving",
      aliases=("tilapia", "croaker", "mackerel", "titus", "fried fish"))

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

    def summary(self) -> str:
        names = ", ".join(i.name for i in self.items) or "meal"
        parts = [f"🍽️ logged: {names} — {self.range_text}."]
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
                    known=True, oily=entry.oily))
            else:
                # Honest generic estimate for unknown foods — marked unknown.
                draft.items.append(MealItem(
                    name=name, portion_hint=portion,
                    cal_low=250, cal_high=450, known=False))
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

    # 2. Optional expense pairing: "that jollof cost ₦2,500".
    if cost_kobo and cost_kobo > 0 and ledger is not None:
        try:
            txn = ledger.log(int(cost_kobo), category="food",
                             note=f"meal: {names}", source="meal")
            log.expense_txn_id = getattr(txn, "id", "")
        except Exception:  # noqa: BLE001
            _log.debug("meal expense log failed", exc_info=True)
    return log


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
