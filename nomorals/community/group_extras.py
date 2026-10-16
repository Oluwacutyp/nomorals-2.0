"""Group extras: language, voice, polls, logistics, health, broadcast.

Everything here is *data + injected callables*, following the community
isolation contract (see ``nomorals/community/__init__.py``):

* **Emoji vocabulary** — per-context palettes as plain data
  (trading alert ≠ party ≠ scam ≠ devotional ≠ announcement), plus
  **emoji as semantics**: stable machine-readable markers
  (``[ALERT]`` etc.) that survive rephrasing, so downstream code can
  classify a message without guessing from vibes.
* **Register packs** — group-admin / formal-report / hype-man / market
  analyst voices as data (greeting, sign-off, style notes, emoji set),
  applied by :func:`render_in_register`.
* **Voice notes both ways** — transcript → answer via an injected
  ``responder``; text → voice via an injected ``tts``. No TTS/STT is
  faked: without the injected callable the function raises an honest
  ``ValueError`` naming what is missing.
* **Translation** — in-group Yoruba → English (and any pair) via an
  injected ``translator`` callable. Same honesty rule: no fake
  translation is ever returned.
* **Polls with consequence** — a closed poll can execute an injected
  ``consequence`` callable (post the result, schedule the meeting,
  ...). No consequence → the result is just reported.
* **Meeting logistics, birthdays, /explain, health report, quiet mode**
  (via ``autonomy.interruption.assess_worth`` — low-worth messages are
  held during quiet hours), **emergency broadcast** (one command, all
  groups, all platforms, priority), and the **onboarding drip**
  (day-1 / day-3 / day-7 messages).

Only stdlib + ``nomorals.core`` + ``nomorals.autonomy.interruption``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "EMOJI_VOCAB", "SEMANTIC_MARKERS", "REGISTER_PACKS",
    "emoji_palette", "mark_semantics", "render_in_register",
    "handle_voice_note", "text_to_voice", "translate_in_group",
    "Poll", "Meeting", "BirthdayStore", "explain", "group_health_report",
    "should_send_quiet", "EmergencyBroadcast", "OnboardingDrip",
    "ONBOARDING_DRIP", "GLOSSARY",
]

# ───────────────────────── emoji vocabulary (data) ──────────────────────────

EMOJI_VOCAB: dict[str, dict[str, list[str]]] = {
    "trading_alert": {
        "palette": ["📈", "📉", "🚨", "💰", "🎯", "⚠️", "🟢", "🔴"],
        "open": ["📈", "🎯"], "close": ["💰", "✅"], "danger": ["🚨", "⚠️", "🔴"],
    },
    "party": {
        "palette": ["🎉", "🥳", "🎊", "🍾", "💃", "🕺", "🔥", "🎶"],
        "open": ["🎉", "🥳"], "close": ["🎊", "🔥"], "danger": ["😅"],
    },
    "scam": {
        "palette": ["🚨", "⚠️", "🛑", "🔍", "❌", "📛"],
        "open": ["🚨", "⚠️"], "close": ["🛑", "❌"], "danger": ["🚨", "🛑", "📛"],
    },
    "devotional": {
        "palette": ["🙏", "✨", "🕊️", "📖", "🌅", "💛", "🕯️"],
        "open": ["🙏", "🌅"], "close": ["✨", "🕊️"], "danger": ["🙏"],
    },
    "announcement": {
        "palette": ["📢", "📌", "📣", "ℹ️", "✅", "📅"],
        "open": ["📢", "📌"], "close": ["✅", "📌"], "danger": ["📢"],
    },
}

#: Emoji as semantics: stable markers that classify a message regardless
#: of wording. Rendered as ``[MARKER]`` prefixes; downstream code can
#: match on the marker instead of re-inferring intent.
SEMANTIC_MARKERS: dict[str, str] = {
    "trading_alert": "[ALERT:TRADE]",
    "party": "[VIBE:PARTY]",
    "scam": "[WARN:SCAM]",
    "devotional": "[NOTE:DEVOTIONAL]",
    "announcement": "[INFO:ANNOUNCE]",
    "emergency": "[PRIORITY:EMERGENCY]",
    "poll": "[POLL]",
    "meeting": "[MEETING]",
    "birthday": "[BIRTHDAY]",
}


def emoji_palette(context: str, slot: str = "palette") -> list[str]:
    """Emoji palette for a context + slot (palette/open/close/danger).

    Unknown context → announcement palette (neutral), never an error.
    """
    vocab = EMOJI_VOCAB.get(context, EMOJI_VOCAB["announcement"])
    return list(vocab.get(slot, vocab["palette"]))


def mark_semantics(text: str, context: str) -> str:
    """Prefix a message with its stable semantic marker + lead emoji."""
    marker = SEMANTIC_MARKERS.get(context, "[INFO]")
    lead = emoji_palette(context, "open")[:1]
    prefix = f"{marker} {' '.join(lead)} " if lead else f"{marker} "
    return prefix + text if not text.startswith("[") else text


# ───────────────────────── register packs (data) ────────────────────────────

REGISTER_PACKS: dict[str, dict[str, Any]] = {
    "group_admin": {
        "greeting": "Hello everyone,",
        "signoff": "— Admin",
        "style": "short, clear, actionable; decisions first, reasons second",
        "emoji": EMOJI_VOCAB["announcement"]["palette"],
        "formality": 0.7,
    },
    "formal_report": {
        "greeting": "Dear members,",
        "signoff": "Respectfully submitted.",
        "style": "full sentences, numbers with units, no slang, no hype",
        "emoji": [],
        "formality": 1.0,
    },
    "hype_man": {
        "greeting": "YOOO family!! 🔥",
        "signoff": "LET'S GOOO 🚀",
        "style": "high energy, caps for emphasis, short punchy lines",
        "emoji": EMOJI_VOCAB["party"]["palette"],
        "formality": 0.1,
    },
    "market_analyst": {
        "greeting": "Market update —",
        "signoff": "Not financial advice. Trade your plan.",
        "style": "levels first, bias stated, invalidation stated, no guarantees",
        "emoji": EMOJI_VOCAB["trading_alert"]["palette"],
        "formality": 0.6,
    },
}


def render_in_register(text: str, register: str) -> str:
    """Wrap text in a register pack (greeting + sign-off + emoji frame).

    Unknown register → ``group_admin``. This is formatting only: it
    never rewrites the claim, so facts survive the register change.
    """
    pack = REGISTER_PACKS.get(register, REGISTER_PACKS["group_admin"])
    emoji = pack["emoji"][:2]
    head = f"{pack['greeting']} {' '.join(emoji)}" if emoji else pack["greeting"]
    return f"{head}\n{text}\n{pack['signoff']}"


# ───────────────────────── voice notes both ways ────────────────────────────

def handle_voice_note(transcript: str,
                      responder: Callable[[str], str] | None = None) -> str:
    """Transcript → answer. ``responder`` is the injected answerer.

    Without a responder there is no answer to give — raising is honest;
    returning a canned reply would fake understanding.
    """
    if responder is None:
        raise ValueError("handle_voice_note needs an injected responder callable")
    if not transcript.strip():
        raise ValueError("empty voice-note transcript")
    return responder(transcript)


def text_to_voice(text: str, tts: Callable[[str], Any] | None = None) -> Any:
    """Text → voice via injected ``tts`` (returns audio path / bytes).

    Without a TTS callable, raises — never returns fake audio bytes.
    """
    if tts is None:
        raise ValueError("text_to_voice needs an injected tts callable")
    if not text.strip():
        raise ValueError("cannot synthesise empty text")
    return tts(text)


def translate_in_group(text: str,
                       translator: Callable[[str, str, str], str] | None = None,
                       *, src: str = "yo", dst: str = "en") -> str:
    """Translate via injected ``translator(text, src, dst)``.

    Default pair is Yoruba (``yo``) → English (``en``) for in-group use.
    Without a translator, raises — a dictionary-guess translation would
    be wrong in exactly the cases that matter.
    """
    if translator is None:
        raise ValueError("translate_in_group needs an injected translator callable")
    return translator(text, src, dst)


# ───────────────────────── polls with consequence ───────────────────────────

@dataclass
class Poll:
    """A poll whose close can trigger a real consequence."""

    id: str
    group_id: str
    question: str
    options: list[str]
    votes: dict[str, str] = field(default_factory=dict)   # voter -> option
    closed: bool = False
    consequence_result: Any = None

    def vote(self, voter: str, option: str) -> tuple[bool, str]:
        if self.closed:
            return False, "poll is closed"
        if option not in self.options:
            return False, f"{option!r} is not an option"
        self.votes[voter] = option
        return True, f"{voter} voted {option}"

    def tally(self) -> dict[str, int]:
        counts = {opt: 0 for opt in self.options}
        for opt in self.votes.values():
            counts[opt] = counts.get(opt, 0) + 1
        return counts

    def winner(self) -> str | None:
        if not self.votes:
            return None
        tally = self.tally()
        return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def close(self, consequence: Callable[[dict[str, Any]], Any] | None = None,
              *, min_votes: int = 1) -> dict[str, Any]:
        """Close the poll. If ``min_votes`` is met and a consequence is
        given, the consequence runs with the result — e.g. posting the
        outcome or scheduling the winning meeting time."""
        self.closed = True
        result = {"poll_id": self.id, "question": self.question,
                  "tally": self.tally(), "winner": self.winner(),
                  "total_votes": len(self.votes)}
        if consequence is not None and len(self.votes) >= min_votes:
            try:
                self.consequence_result = consequence(result)
                result["consequence"] = self.consequence_result
            except Exception as exc:  # noqa: BLE001 — consequence is caller code
                _log.warning("poll consequence failed: %s", exc)
                result["consequence_error"] = str(exc)
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────── meeting logistics ────────────────────────────────

@dataclass
class Meeting:
    """A group meeting: proposed times, RSVPs, reminder."""

    id: str
    group_id: str
    title: str
    proposed_times: list[str] = field(default_factory=list)
    rsvps: dict[str, str] = field(default_factory=dict)   # member -> yes/no/maybe
    chosen_time: str | None = None
    location: str = ""

    def propose(self, times: list[str]) -> None:
        self.proposed_times = list(times)

    def rsvp(self, member: str, answer: str) -> tuple[bool, str]:
        answer = answer.strip().lower()
        if answer not in ("yes", "no", "maybe"):
            return False, "answer must be yes / no / maybe"
        self.rsvps[member] = answer
        return True, f"{member}: {answer}"

    def confirm(self, chosen: str) -> tuple[bool, str]:
        if chosen not in self.proposed_times:
            return False, f"{chosen!r} was not proposed"
        self.chosen_time = chosen
        return True, f"confirmed: {self.title} at {chosen}"

    def reminder_text(self) -> str:
        yes = sorted(m for m, a in self.rsvps.items() if a == "yes")
        return (f"{SEMANTIC_MARKERS['meeting']} reminder: {self.title}\n"
                f"when: {self.chosen_time or 'TBD'}\n"
                f"where: {self.location or 'TBD'}\n"
                f"going ({len(yes)}): {', '.join(yes) or 'nobody yet'}")


# ───────────────────────── birthdays ────────────────────────────────────────

class BirthdayStore:
    """Per-group birthdays. Group-scoped JSON under ~/.devon/community/."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else \
            Path.home() / ".devon" / "community" / "birthdays"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, group_id: str) -> Path:
        safe = "".join(c for c in group_id if c.isalnum() or c in "-_") or "group"
        return self.root / f"{safe}.json"

    def _load(self, group_id: str) -> dict[str, str]:
        path = self._path(group_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def set_birthday(self, group_id: str, member: str, month_day: str) -> None:
        """``month_day`` is ``MM-DD``."""
        parts = month_day.strip().split("-")
        if len(parts) != 2 or not (1 <= int(parts[0]) <= 12 and 1 <= int(parts[1]) <= 31):
            raise ValueError(f"bad month_day {month_day!r} (expected MM-DD)")
        data = self._load(group_id)
        data[member] = month_day.strip()
        self._path(group_id).write_text(json.dumps(data, indent=1), encoding="utf-8")

    def upcoming(self, group_id: str, *, within_days: int = 7,
                 today: str = "") -> list[dict[str, str]]:
        """Birthdays within ``within_days`` of ``today`` (MM-DD)."""
        import datetime as _dt
        if today:
            cur = _dt.datetime.strptime(today, "%m-%d").date().replace(year=2000)
        else:
            now = _dt.datetime.now()
            cur = now.date().replace(year=2000)
        out = []
        for member, md in self._load(group_id).items():
            try:
                day = _dt.datetime.strptime(md, "%m-%d").date().replace(year=2000)
            except ValueError:
                continue
            delta = (day - cur).days % 366
            if delta <= within_days:
                out.append({"member": member, "birthday": md, "in_days": delta})
        return sorted(out, key=lambda r: (r["in_days"], r["member"]))

    def birthday_message(self, member: str) -> str:
        return (f"{SEMANTIC_MARKERS['birthday']} 🎂 Happy birthday, {member}! "
                f"Wishing you a great year — from the whole group 🎉")


# ───────────────────────── /explain ─────────────────────────────────────────

GLOSSARY: dict[str, str] = {
    "quasimodo": "Quasimodo (QM): a reversal pattern — price sweeps a prior high/low, then fails back through it. The failed sweep is the signal.",
    "bos": "Break of structure (BOS): price breaks a prior swing high/low, signalling the trend may be continuing.",
    "fvg": "Fair value gap (FVG): a gap between candles where price moved too fast to trade — often revisited as an entry zone.",
    "liquidity": "Liquidity: clusters of stop orders sitting above highs / below lows. Price often runs to them before reversing.",
    "killzone": "Killzone: a session window (e.g. London / New York open) where institutional flow concentrates and setups are most reliable.",
    "ajo": "Ajo: rotating savings circle — members contribute a fixed amount on a schedule and take turns receiving the pot.",
}


def explain(topic: str,
            explainer: Callable[[str], str] | None = None) -> str:
    """Explain a topic: injected explainer wins, else the group glossary.

    Unknown topic + no explainer → an honest 'not in the glossary'
    answer, never a hallucinated definition.
    """
    topic = topic.strip()
    if explainer is not None:
        return explainer(topic)
    hit = GLOSSARY.get(topic.strip().lower())
    if hit:
        return hit
    return (f"No glossary entry for {topic!r}. "
            f"Available: {', '.join(sorted(GLOSSARY))}.")


# ───────────────────────── health report + quiet mode ───────────────────────

def group_health_report(stats: dict[str, Any]) -> dict[str, Any]:
    """Health score (0..100) + plain-language report from group stats.

    ``stats``: messages_7d, active_members, total_members, polls_7d,
    events_7d, reports_7d (scam/spam reports). All optional, default 0.
    """
    msgs = float(stats.get("messages_7d", 0))
    active = float(stats.get("active_members", 0))
    total = float(stats.get("total_members", 0)) or 1.0
    polls = float(stats.get("polls_7d", 0))
    events = float(stats.get("events_7d", 0))
    reports = float(stats.get("reports_7d", 0))
    activity = min(40.0, msgs / 5.0)                 # 200 msgs/wk → full
    participation = min(30.0, 30.0 * active / total)
    engagement = min(20.0, 10.0 * polls + 10.0 * events)
    safety = max(0.0, 10.0 - 2.0 * reports)
    score = round(activity + participation + engagement + safety, 1)
    verdict = ("thriving" if score >= 75 else "healthy" if score >= 50
               else "quiet" if score >= 30 else "at risk")
    report = (f"📊 group health: {score}/100 — {verdict}\n"
              f"  messages (7d): {msgs:.0f} · active: {active:.0f}/{total:.0f}\n"
              f"  polls: {polls:.0f} · events: {events:.0f} · reports: {reports:.0f}")
    return {"score": score, "verdict": verdict, "report": report,
            "breakdown": {"activity": activity, "participation": participation,
                          "engagement": engagement, "safety": safety}}


def should_send_quiet(message: dict[str, Any],
                      context: dict[str, Any] | None = None,
                      assess_fn: Any | None = None) -> dict[str, Any]:
    """Quiet-mode gate: should this message go out now?

    Uses injected ``assess_fn`` when supplied (same shape as
    ``autonomy.interruption.assess_worth``); otherwise a local
    heuristic: safety / owner-direct always interrupt, else defer.
    Layering: this L4 module must not import L5 interruption.
    """
    if assess_fn is not None:
        result = assess_fn(message, context)
        decision = result.get("decision", "defer")
        return {"send_now": decision == "interrupt",
                "decision": decision, "score": result.get("score"),
                "reasons": result.get("reasons", [])}
    prio = str(message.get("priority", "")).lower()
    if prio in ("safety", "owner-direct", "urgent"):
        return {"send_now": True, "decision": "interrupt",
                "score": 1.0, "reasons": ["local heuristic: priority"]}
    return {"send_now": False, "decision": "defer", "score": 0.0,
            "reasons": ["local heuristic: quiet mode"]}


# ───────────────────────── emergency broadcast ──────────────────────────────

@dataclass
class EmergencyBroadcast:
    """One command → all groups, all platforms, priority.

    ``groups``: list of ``{"group_id", "platform", "target"}``.
    ``senders``: platform → ``callable(target, text) -> bool``.
    The emergency marker + priority prefix is always applied so the
    message cannot be mistaken for a routine post.
    """

    groups: list[dict[str, str]] = field(default_factory=list)

    def broadcast(self, text: str,
                  senders: dict[str, Callable[[str, str], bool]] | None = None,
                  ) -> list[dict[str, Any]]:
        prefixed = (f"{SEMANTIC_MARKERS['emergency']} 🚨 EMERGENCY BROADCAST\n{text}")
        senders = senders or {}
        results: list[dict[str, Any]] = []
        for group in self.groups:
            platform, target = group.get("platform", ""), group.get("target", "")
            sender = senders.get(platform)
            sent, error = False, ""
            if sender is None:
                error = f"no sender for {platform}"
            else:
                try:
                    sent = bool(sender(target, prefixed))
                except Exception as exc:  # noqa: BLE001 — sender is caller code
                    error = str(exc)
            results.append({"group_id": group.get("group_id", ""),
                            "platform": platform, "target": target,
                            "sent": sent, "error": error})
        return results


# ───────────────────────── onboarding drip ──────────────────────────────────

ONBOARDING_DRIP: dict[int, str] = {
    1: ("👋 Welcome! Day 1: introduce yourself — name, what you do, "
        "and one thing you want from this group."),
    3: ("📖 Day 3: have you read the pinned rules? Drop a ✅ if yes, "
        "or ask anything — no silly questions here."),
    7: ("🎉 Day 7: you've been here a week! Share one win, one question, "
        "and join this week's event if you haven't yet."),
}


class OnboardingDrip:
    """Day-1 / day-3 / day-7 onboarding messages for a new member."""

    def __init__(self, messages: dict[int, str] | None = None) -> None:
        self.messages = dict(messages or ONBOARDING_DRIP)

    def due_for(self, joined_ts: float, now: float | None = None,
                already_sent: list[int] | None = None) -> list[dict[str, Any]]:
        """Drip steps due for a member who joined at ``joined_ts``."""
        now = now if now is not None else time.time()
        sent = set(already_sent or [])
        age_days = (now - joined_ts) / 86400.0
        return [{"day": day, "text": text}
                for day, text in sorted(self.messages.items())
                if day <= age_days + 1e-9 and day not in sent]

    def render(self, day: int, member: str) -> str:
        return f"Hi {member}! {self.messages.get(day, '')}"
