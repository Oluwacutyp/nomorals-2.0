"""Core Mind — the always-on natural-language layer of the Agent OS (wave 87).

The Agent OS layering this module completes:

    Core Mind (here)                = the mind. Understands a natural-language
                                      goal, decides which organ does the work,
                                      keeps continuity across turns/sessions.
      ├── Orchestrator              = the nervous system (multi-part goals,
      │                               supervision, parallelism)
      ├── ResearchSwarm             = research organ (parallel, conflict-aware)
      ├── CodingAgent / Devon       = builder organs (draft → run → fix)
      ├── BrowserSession            = eyes and hands on the web (multi-step)
      ├── media download            = the downloader organ
      ├── DirectivesAgent           = the durable work queue ("missions")
      ├── GameEngine (nomorals.games) = the games organ — it owns rules,
      │                               turns, scoring, match state and economy;
      │                               the Core Mind only decides WHEN to route
      │                               into it and preserves continuity
      └── PartnerBrain              = the voice (everything that is not a goal)

Trigger rules (wave 87 spec — implemented structurally, not by politeness):

* Natural-language dispatch happens ONLY in the owner's DMs.  For every
  other chat this module returns ``None`` — no launch path exists, so a
  false trigger is impossible there.  Commands (``/hangman``, ``/game``,
  ``/research`` …) remain the manual override in every chat.
* The word "game" alone never starts anything.  A game needs a known game
  name AND a play-intent verb (or an explicit command).  "this game is
  boring", "football game", "I played a game last night" → conversation.
* Ambiguous owner intent → one short clarification question, persisted so
  it survives a restart — never a forced launch.
* Fast path (wave F1 stream 2): trivial chat — greetings, time/date,
  simple chitchat, thanks, farewells, bare acknowledgements — gets a
  deterministic reply from ``fast_path()`` with ZERO model calls and
  zero heavy-organ activations.  The patterns are whole-message
  anchored, so explicit invocations always fall through to the organs;
  ambiguity fails OPEN toward the heavy path, never toward cheapness.
* Every decision carries a one-line ``why`` (provenance) so routing is
  inspectable: ``/mind status``, ``/mind <goal>``, ``nm goal``.

Profile-aware: dispatch sizes follow the runtime tune (fewer research
workers on a phone, download caps from the media organ, browser step caps
already profile-tuned inside the browser organ).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..storage import router_telemetry

_log = logging.getLogger("nomorals.coremind")

__all__ = ["Intent", "CoreMind", "GAME_ALIASES", "game_names",
           "MODEL_CHECK_TIMEOUT_S", "CODING_JOB_TIMEOUT_S",
           "MAX_INFLIGHT_DEFAULT", "INFLIGHT_ACQUIRE_TIMEOUT_S"]

# ── fast-path vs heavy-path budgets (explicit, not magic numbers) ───────────
#: INTERACTIVE budget: max wall time the chat thread waits for the model in
#: ``_model_check``. On expiry the deterministic intent stands and the
#: timeout is logged + counted (``status()``) — never silently swallowed.
MODEL_CHECK_TIMEOUT_S = 8.0

#: BACKGROUND budget: the builder organ's per-job wall clock. Heavy work
#: always runs off the chat thread (``_send_async``), so a long budget here
#: never blocks a reply.
CODING_JOB_TIMEOUT_S = 300.0

#: BACKGROUND concurrency: max jobs ``_send_async`` may run at once.
#: Configurable via ``mind.max_inflight`` (env NM_MIND_MAX_INFLIGHT);
#: ``CoreMind`` clamps non-positive values to this default.
MAX_INFLIGHT_DEFAULT = 8

#: How long a new send waits for an in-flight slot before it is shed
#: (``mind.inflight_acquire_timeout_s``, env NM_MIND_INFLIGHT_TIMEOUT).
INFLIGHT_ACQUIRE_TIMEOUT_S = 30.0

# ── the game catalogue (the games organ owns the rules — here is the map) ──

#: engine game name -> every alias a human might say (longest-match wins).
#: Deliberately singular and unambiguous: "game", "games", "gaming",
#: "football", "video" are NOT here, so casual talk can never launch.
GAME_ALIASES: dict[str, str] = {
    "word chain": "wordchain",
    "wordchain": "wordchain",
    "hangman": "hangman",
    "number guess": "numberguess",
    "numberguess": "numberguess",
    "two truths": "two_truths",
    "two_truths": "two_truths",
    "wyrr": "wyrr",
    "spy": "spy",
    "auction": "auction",
    "trivia royale": "trivia",
    "trivia": "trivia",
    "mafia": "mafia",
    "king of the hill": "king",
    "king": "king",
    "story chain": "story",
    "story": "story",
    "rpg adventure": "rpg",
    "rpg": "rpg",
    "dungeon": "rpg",
    "shop game": "shop",
    "shop": "shop",
    "quiz duel": "duel",
    "duel": "duel",
    "the case": "case",
    "investigation": "case",
    "case": "case",
    "world": "world",
    "battle arena": "arena",
    "arena": "arena",
    "escape room": "escape",
    "escape": "escape",
    "political": "political",
    # classics now live in the multiplayer engine (20q, rps, digits)
    "20 questions": "20q",
    "20q": "20q",
    "rock paper scissors": "rps",
    "rps": "rps",
}

#: names the /<name> chat commands can start (``arena`` is already taken
#: by the self-improvement arena control command — it stays /game arena).
COMMAND_STARTABLE = [
    "wordchain", "hangman", "numberguess", "two_truths", "wyrr", "spy",
    "auction", "trivia", "mafia", "king", "story", "rpg", "shop", "duel",
    "case", "world", "escape", "political",
]


def game_names() -> list[str]:
    return sorted({v for v in GAME_ALIASES.values() if v not in ("20q", "rps")})


_ALIAS_SORTED = sorted(GAME_ALIASES, key=len, reverse=True)
_ALIAS_RE = {
    alias: re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
    for alias in _ALIAS_SORTED
}

# ── fast-path responder (wave F1 stream 2): trivial chat, zero model ─────────
# Simple chat NEVER wakes the router model, the swarms, research loops, or
# any heavy organ: these whole-message-anchored patterns match ONLY the
# entire message, so an explicit organ invocation ("research what time the
# market opens", "build me a clock app", "hi, download the report") can
# never fast-path — it falls through to understand() → the heavy path.
# Anything ambiguous returns None and also falls through: fail-open toward
# capability, never toward cheapness.  The one deliberate exception is
# "how's it going" / "how's things": _RE_STATUS claims those as a system
# status intent, so they stay OUT of the chitchat pattern — the fast path
# must never preempt an existing organ intent.
_RE_FP_GREET = re.compile(
    r"(hi+|hello+|hey+|yo+|hiya+|howdy|greetings?|"
    r"good\s*(morning|afternoon|evening|day)|sup|what'?s\s+up)"
    r"\s*[!.,\u2026]*", re.I)
_RE_FP_TIME = re.compile(
    r"(what('?s| is) the time|what time is it(\s+now)?|current time|"
    r"tell me the time|the time(\s+please)?|got the time)"
    r"\s*\??\s*[!.,]*", re.I)
_RE_FP_DATE = re.compile(
    r"(what('?s| is) (today'?s )?date|what('?s| is) the date(\s+today)?|"
    r"what day is it|which day is it|today'?s date|current date)"
    r"\s*\??\s*[!.,]*", re.I)
_RE_FP_HOWRU = re.compile(
    r"(how (are|r) (you|u|ya)(\s+doing)?|how'?re (you|u|ya)|"
    r"how do you feel)\s*\??\s*[!.,]*", re.I)
_RE_FP_THANKS = re.compile(
    r"(thanks?|thank\s+you(\s+(very|so)\s+much)?|thx|ty(vm)?|"
    r"much\s+appreciated|appreciated)\s*[!.,]*", re.I)
_RE_FP_BYE = re.compile(
    r"(bye+|good\s*bye|good\s*night|see\s+(you|ya|u)"
    r"(\s+(later|tomorrow|soon))?|later[sz]?|ttyl|cya|farewell)"
    r"\s*[!.,]*", re.I)
_RE_FP_ACK = re.compile(
    r"(ok(ay)?|k|cool|nice|sweet|great|awesome|perfect|sure|yep|yeah|"
    r"lol|lmao|haha+|roger|got\s+it|understood|alright|all\s+right|"
    r"\U0001f44d|\U0001f64f|\U0001f602|\u2764\ufe0f?|\u2665)"
    r"\s*[!.,\u2026]*", re.I)

_FP_GREETINGS = ("hey!", "hey — what's on your mind?", "hi there.")
_FP_HOWRU = ("doing great — you?", "running smooth. what's up?",
             "all good here — what are we doing?")
_FP_THANKS = ("anytime.", "you got it.", "of course.")
_FP_BYE = ("later!", "see you soon.", "bye for now.")
_FP_ACK = ("\U0001f44d", "got it.", "cool.")

_fp_lock = threading.Lock()
_fp_idx = 0


def _rotate(options: tuple[str, ...]) -> str:
    """Deterministic variety: cycle the canned replies, thread-safe."""
    global _fp_idx
    with _fp_lock:
        _fp_idx += 1
        return options[_fp_idx % len(options)]


def _fp_time() -> str:
    now = datetime.now().astimezone()
    hm = now.strftime("%I:%M %p").lstrip("0")
    tz = now.strftime("%Z") or "local time"
    return f"it's {hm} {tz}."


def _fp_date() -> str:
    now = datetime.now().astimezone()
    return f"today is {now.strftime('%A, %B')} {now.day}, {now.year}."


def fast_path(text: str) -> tuple[str, str] | None:
    """Zero-model fast-path reply for trivial chat.

    Returns ``(reply, why)`` when the WHOLE message is a greeting, a
    time/date question, simple chitchat, thanks, a farewell, or a bare
    acknowledgement — no router model, no swarms, no research loops, no
    heavy organs, no brain call.  Returns ``None`` for everything else:
    the caller falls through to the normal heavy path (fail-open toward
    capability).
    """
    t = text.strip()
    if _RE_FP_GREET.fullmatch(t):
        return _rotate(_FP_GREETINGS), "fast-path: greeting (no model)"
    if _RE_FP_TIME.fullmatch(t):
        return _fp_time(), "fast-path: time (no model)"
    if _RE_FP_DATE.fullmatch(t):
        return _fp_date(), "fast-path: date (no model)"
    if _RE_FP_HOWRU.fullmatch(t):
        return _rotate(_FP_HOWRU), "fast-path: chitchat (no model)"
    if _RE_FP_THANKS.fullmatch(t):
        return _rotate(_FP_THANKS), "fast-path: thanks (no model)"
    if _RE_FP_BYE.fullmatch(t):
        return _rotate(_FP_BYE), "fast-path: farewell (no model)"
    if _RE_FP_ACK.fullmatch(t):
        return _rotate(_FP_ACK), "fast-path: acknowledgement (no model)"
    return None


# ── intent verbs (word-boundary, case-insensitive) ──────────────────────────

_RE_PLAY = re.compile(r"\b(let'?s|play|start|begin|open|kick off|fire up)\b", re.I)
_RE_CONTINUE = re.compile(
    r"\b(continue|resume|pick up|back to|return to|keep (on|playing|going)|"
    r"carry on)\b", re.I)
_RE_BOARD = re.compile(r"\b(leaderboard|leaderboards|board|ranks?|standings|scores?)\b", re.I)
_RE_ECONOMY = re.compile(r"\b(balance|coins?|points?|inventory|buy)\b", re.I)
_RE_SHOPPLAY = re.compile(r"\b(shop|shop game)\b", re.I)
_RE_STATUS = re.compile(
    r"^\s*(status|what are you doing|whats? (you|it) doing|how'?s it going|"
    r"what'?s active|what is active|what'?s running|what have you been up to|"
    r"where (are|is) we (at|on))\s*[?!.]?\s*$", re.I)
_RE_RESEARCH = re.compile(
    r"\b(research|investigate|look into|dig into|find out about|find out on|"
    r"study)\b", re.I)
_RE_BUILD = re.compile(r"\b(build|create|make|write|code|develop)\b", re.I)
_RE_BUILD_NOUN = re.compile(
    r"\b(app|application|website|web app|site|webpage|bot|script|tool|api|"
    r"scraper|crawler|dashboard|extension|plugin|program|page|cli|game|"
    r"game engine|bot net|webapp)\b", re.I)
_RE_BROWSE = re.compile(r"\b(open|visit|browse|go to|check out|check|look at|see)\b", re.I)
_RE_URL = re.compile(r"\bhttps?://[^\s)\"']+", re.I)
_RE_DOWNLOAD = re.compile(
    r"\b(download|get me|fetch|grab|pull|save (me|to disk))\b", re.I)
_RE_DOWNLOAD_NOUN = re.compile(
    r"\b(video|image|photo|picture|pdf|file|dataset|song|audio|track|episode|"
    r"podcast|archive|zip)\b", re.I)
_RE_MISSION_ADD = re.compile(
    r"^(mission|task|directive)\s*:|\b(set|add|create|queue|make) (a |me )?"
    r"(new )?(mission|task|directive)\b\s*[:\-–]?", re.I)
_RE_MISSION_RUN = re.compile(
    r"\b(continue|resume|run|execute|work on|keep working on)\b.{0,24}"
    r"\b(mission|task|directive)s?\b", re.I)
_RE_MISSION_LIST = re.compile(
    r"\b(mission|task|directive)s?\b.{0,24}\b(status|list|pending|queued|"
    r"waiting|progress|how'?s)\b|\b(status|list|progress)\b.{0,24}"
    r"\b(mission|task|directive)s?\b|\bhow'?s (the|my) (mission|task|directive)s?\b", re.I)
_RE_CANCEL = re.compile(r"^\s*(no|nope|never ?mind|stop|cancel|scrub|drop it|forget it)\b", re.I)

_MULTI_JOINERS = re.compile(r"\b(and|then|after that|also|while you'?re at it|plus)\b", re.I)


def _clean_topic(text: str, verb: re.Pattern) -> str:
    """The payload after the intent verb, stripped of filler."""
    m = verb.search(text)
    if not m:
        return text.strip()
    tail = text[m.end():].strip()
    tail = re.sub(r"^(me |us |for me|about|on|the topic of|regarding|:|-|—)\s*", "", tail,
                  flags=re.I)
    tail = re.sub(r"\b(for me|please|can you|could you)$", "", tail, flags=re.I).strip()
    return tail


@dataclass(frozen=True)
class Intent:
    """One Core Mind decision.  ``why`` is the inspectable provenance."""

    kind: str                      # chat|research|build|browse|download|mission|game|status|multi|fastchat
    confidence: float              # 0..1
    target: str = ""               # topic / url / goal payload
    action: str = ""               # game: start|resume|board|economy|ask
                                    # mission: add|run|list
    route: str = ""                # organ: research_swarm|coding|browser|media|
                                    # directives|games|orchestrator|brain
    why: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "confidence": round(self.confidence, 2),
                "target": self.target, "action": self.action,
                "route": self.route, "why": self.why}


# ── intent detection (pure, deterministic — the reflex layer) ───────────────

def _find_game_alias(text: str) -> str | None:
    """Longest alias present as a whole word; None otherwise."""
    for alias in _ALIAS_SORTED:
        if _ALIAS_RE[alias].search(text):
            return GAME_ALIASES[alias]
    return None


def _game_intent(text: str, live_game: str | None) -> Intent | None:
    """Game trigger policy.

    A launch needs a known game name AND a play verb (or an explicit
    command, which never reaches here).  The generic word "game" is not
    an alias, so "game", "this game is boring", "football game", "I played
    a game last night" all fall through to conversation.
    """
    name = _find_game_alias(text)
    # hypotheticals ("we should play some games") are conversation, not a
    # launch — strip them before the play-verb test
    imperative = re.sub(r"\b(?:should|would|could|might)\s+(?:to\s+)?"
                        r"(?:play|start|begin|open)\b", "", text, flags=re.I)
    has_play = bool(_RE_PLAY.search(imperative))
    has_continue = bool(_RE_CONTINUE.search(text))
    has_board = bool(_RE_BOARD.search(text))
    has_econ = bool(_RE_ECONOMY.search(text))
    has_shop = bool(_RE_SHOPPLAY.search(text))

    # viewing intents are safe anywhere in an owner DM — they never launch
    if has_board:
        return Intent("game", 0.9, target=name or "", action="board", route="games",
                      why=f"leaderboard intent" + (f" for {name}" if name else ""))
    if has_econ and not has_play:
        action = "economy"
        target = ""
        m = re.search(r"\bbuy\s+(.+)", text, re.I)
        if m and (live_game or name == "shop"):
            target = f"buy {m.group(1).strip()}"
        elif name == "shop" and has_play:
            pass  # falls through to start below
        else:
            return Intent("game", 0.85, target=target, action=action, route="games",
                          why="economy intent (balance/coins/buy)")
    if name:
        if has_play:
            if live_game == name:
                return Intent("game", 0.9, target=name, action="resume", route="games",
                              why=f"{name} is already live — resuming it")
            return Intent("game", 0.9, target=name, action="start", route="games",
                          why=f"play verb + game name “{name}”")
        if has_continue:
            if live_game == name:
                return Intent("game", 0.85, target=name, action="resume", route="games",
                              why=f"continuing live {name}")
            return Intent("game", 0.8, target=name, action="start", route="games",
                          why=f"“continue the {name}” — no live room, starting it")
        if has_econ:
            return Intent("game", 0.8, target=name, action="economy", route="games",
                          why=f"economy intent on {name}")
        if live_game == name:
            return Intent("game", 0.7, target=name, action="resume", route="games",
                          why=f"“{name}” is live in this chat — treating as a move")
        return Intent("game", 0.55, target=name, action="ask", route="games",
                      why=f"“{name}” without a play verb — confirming before launch")
    # no known game name
    if has_play:
        if live_game:
            return Intent("game", 0.75, target=live_game, action="resume", route="games",
                          why=f"play verb with {live_game} live — resuming it")
        return Intent("game", 0.6, target="", action="ask", route="games",
                      why="“let's play” with no game named — asking which one")
    if has_continue and live_game:
        return Intent("game", 0.8, target=live_game, action="resume", route="games",
                      why=f"continue verb with {live_game} live")
    return None


def _research_intent(text: str) -> Intent | None:
    m = _RE_RESEARCH.search(text)
    if not m:
        return None
    topic = _clean_topic(text, _RE_RESEARCH)
    if not topic or len(topic) < 3:
        return Intent("research", 0.5, target="", action="ask", route="research_swarm",
                      why="research verb with no topic — asking what to research")
    return Intent("research", 0.85, target=topic, route="research_swarm",
                  why=f"research verb + topic “{topic[:40]}”")


def _build_intent(text: str) -> Intent | None:
    m = _RE_BUILD.search(text)
    if not m:
        return None
    # "let's play a game" is a game, not a build — the play verb wins
    if _RE_PLAY.search(text) and not _RE_BUILD_NOUN.search(text):
        return None
    noun = _RE_BUILD_NOUN.search(text)
    task = _clean_topic(text, _RE_BUILD)
    if not task or len(task) < 4:
        task = text.strip()
    if noun:
        return Intent("build", 0.85, target=task, route="coding",
                      why=f"build verb + artifact “{noun.group(0)}”")
    if len(task) >= 8:
        return Intent("build", 0.6, target=task, route="coding",
                      why="build verb, no artifact noun — model check next")
    return None


def _url_intent(text: str) -> tuple[Intent | None, str]:
    m = _RE_URL.search(text)
    url = m.group(0) if m else ""
    if not url:
        return None, ""
    if _RE_DOWNLOAD.search(text):
        return Intent("download", 0.85, target=url, route="media",
                      why=f"download verb + URL"), url
    if _RE_BROWSE.search(text) or re.search(
            r"\b(screenshot|summary|summarize|read|what'?s on|what does)\b", text, re.I):
        return Intent("browse", 0.85, target=url, route="browser",
                      why="browse intent + URL"), url
    return Intent("browse", 0.6, target=url, route="browser",
                  why="bare URL — assuming browse"), url


def _browse_intent(text: str) -> Intent | None:
    site = re.search(
        r"\b(?:hacker news|hn|news? ycombinator|youtube|reddit|x\.com|twitter|"
        r"github|stack ?overflow|medium\.com|wikipedia|docs?\.py)\b", text, re.I)
    if site and _RE_BROWSE.search(text):
        return Intent("browse", 0.7, target=site.group(0), route="browser",
                      why=f"browse verb + site “{site.group(0)}”")
    return None


def _download_intent(text: str) -> Intent | None:
    if not _RE_DOWNLOAD.search(text):
        return None
    noun = _RE_DOWNLOAD_NOUN.search(text)
    target = _clean_topic(text, _RE_DOWNLOAD)
    if not target or not noun:
        return None
    conf = 0.7 if noun else 0.5
    return Intent("download", conf, target=target, route="media",
                  why=f"download verb + “{noun.group(0) if noun else 'file'}”")


def _mission_intent(text: str) -> Intent | None:
    m = _RE_MISSION_ADD.search(text)
    if m:
        target = _clean_topic(text, _RE_MISSION_ADD)
        if target:
            return Intent("mission", 0.85, target=target, action="add",
                          route="directives", why="mission verb with a goal")
        return Intent("mission", 0.5, action="ask", route="directives",
                      why="“mission:” with no goal — asking what")
    m = _RE_MISSION_RUN.search(text)
    if m:
        target = _clean_topic(text, _RE_MISSION_RUN)
        return Intent("mission", 0.85, target=target, action="run",
                      route="directives", why="run/continue a queued mission")
    if _RE_MISSION_LIST.search(text):
        return Intent("mission", 0.85, action="list", route="directives",
                      why="mission status/list intent")
    return None


def _status_intent(text: str) -> Intent | None:
    if _RE_STATUS.match(text):
        return Intent("status", 0.8, route="mind", why="system status intent")
    return None


def understand(text: str, *, live_game: str | None = None) -> list[Intent]:
    """Deterministic intent pass.  Returns every candidate, best first."""
    cands: list[Intent] = []
    for fn in (_status_intent, _mission_intent, _game_intent,
               _research_intent, _build_intent):
        if fn is _game_intent:
            it = fn(text, live_game)
        else:
            it = fn(text)
        if it is not None and it.kind != "chat":
            cands.append(it)
    url_it, _url = _url_intent(text)
    if url_it is not None:
        cands.append(url_it)
    else:
        it = _browse_intent(text) or _download_intent(text)
        if it is not None:
            cands.append(it)
    cands = [c for c in cands if c.kind != "chat"]
    cands.sort(key=lambda c: c.confidence, reverse=True)
    return cands


# ── the mind ─────────────────────────────────────────────────────────────────

class CoreMind:
    """The always-on mind: understand → clarify → route → remember.

    ``runtime`` is the Social Operator (PartnerRuntime) when we live in
    chat; it is ``None`` in the CLI / tests, where dispatch runs inline.
    """

    PENDING_TTL = 86400.0  # a clarification older than a day is stale

    def __init__(self, context: Any, *, runtime: Any = None) -> None:
        self.context = context
        self.runtime = runtime
        try:
            self.state_dir = Path(context.settings.resolve("data/coremind"))
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / "state.json"
        except Exception:  # noqa: BLE001 - state is best-effort
            self.state_dir = None
            self.state_file = None
        self._state = self._load_state()
        self._jobs: list[dict[str, Any]] = self._state.get("jobs", [])[-40:]
        self._lock = threading.Lock()
        self._router_calls = 0  # observability: how often the model was consulted
        self._router_timeouts = 0  # how often _model_check hit its deadline
        self._model_check_note = ""  # last model-check degradation, surfaced in status()
        # ── bounded background sends ──────────────────────────────────
        # _send_async spawns one thread per heavy job; the semaphore caps
        # how many may run at once (mind.max_inflight, default 8). When
        # the bound is hit, new sends wait up to inflight_acquire_timeout_s
        # and are then SHED: logged, counted, and the job is marked failed
        # with an explicit note — never silently dropped, and the chat
        # thread never blocks on the bound.
        mind_cfg = getattr(getattr(self.context, "settings", None), "mind", None)
        try:
            max_inflight = int(getattr(mind_cfg, "max_inflight", 0) or 0)
        except (TypeError, ValueError):
            max_inflight = 0
        self._max_inflight = max_inflight if max_inflight > 0 else MAX_INFLIGHT_DEFAULT
        try:
            self._inflight_timeout = float(
                getattr(mind_cfg, "inflight_acquire_timeout_s", 0) or 0)
        except (TypeError, ValueError):
            self._inflight_timeout = 0.0
        if self._inflight_timeout <= 0:
            self._inflight_timeout = INFLIGHT_ACQUIRE_TIMEOUT_S
        self._inflight = threading.Semaphore(self._max_inflight)
        self._inflight_now = 0  # gauge: sends currently holding a slot
        self._send_started = 0  # sends that took a slot
        self._send_shed = 0  # sends rejected by the bound

    # ── continuity (state file + memory) ────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            if self.state_file and self.state_file.exists():
                return json.loads(self.state_file.read_text())
        except Exception:  # noqa: BLE001
            _log.warning("coremind state unreadable — starting fresh")
        return {"pending": {}, "jobs": [], "last_objective": None}

    def _save_state(self) -> None:
        if self.state_file is None:
            return
        try:
            payload = {"pending": self._state.get("pending", {}),
                       "jobs": self._jobs[-40:],
                       "last_objective": self._state.get("last_objective")}
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(self.state_file)
        except Exception:  # noqa: BLE001
            _log.exception("coremind state write failed")

    def _remember_objective(self, intent: Intent, job_id: str) -> str:
        """Long-term continuity: the goal lives in memory, not just state."""
        rid = ""
        try:
            memory = getattr(self.context, "memory", None)
            if memory is not None:
                rid = memory.remember(
                    f"objective: {intent.target or intent.kind} (routed to {intent.route or 'brain'})",
                    kind="objective", source="coremind", agent="coremind",
                    metadata={"route": intent.route, "job": job_id,
                              "status": "active"},
                    tags=["objective", intent.kind],
                )
        except Exception:  # noqa: BLE001
            _log.exception("objective memory write failed")
        return rid

    def _set_pending(self, chat_key: str, intent: Intent, question: str) -> None:
        with self._lock:
            self._state.setdefault("pending", {})[chat_key] = {
                "kind": intent.kind, "action": intent.action,
                "target": intent.target, "question": question,
                "created": time.time(),
            }
            self._save_state()

    def _get_pending(self, chat_key: str) -> dict[str, Any] | None:
        with self._lock:
            p = self._state.get("pending", {}).get(chat_key)
            if p and time.time() - float(p.get("created", 0)) > self.PENDING_TTL:
                del self._state["pending"][chat_key]
                self._save_state()
                return None
            return p

    def _clear_pending(self, chat_key: str) -> None:
        with self._lock:
            if self._state.get("pending", {}).pop(chat_key, None) is not None:
                self._save_state()

    # ── job registry (inspectable progress + recovery) ──────────────────────
    def _new_job(self, intent: Intent) -> str:
        job_id = new_short_id(length=8)
        self._jobs.append({"id": job_id, "kind": intent.kind,
                           "target": intent.target[:200], "route": intent.route,
                           "status": "active", "created": time.time(),
                           "mem": self._remember_objective(intent, job_id),
                           "note": ""})
        self._state["last_objective"] = {"text": intent.target or intent.kind,
                                         "route": intent.route,
                                         "job": job_id, "created": time.time()}
        self._save_state()
        return job_id

    def _job_done(self, job_id: str, ok: bool, note: str = "") -> None:
        with self._lock:
            for job in self._jobs:
                if job.get("id") == job_id:
                    job["status"] = "done" if ok else "failed"
                    job["note"] = note[:300]
                    job["finished"] = time.time()
                    if job.get("mem"):
                        try:
                            getattr(self.context, "memory", None) and \
                                self.context.memory.update(
                                    job["mem"],
                                    metadata={**({"route": job["route"]}),
                                              "job": job_id,
                                              "status": "done" if ok else "failed"})
                        except Exception:  # noqa: BLE001
                            pass
                    break
            self._save_state()

    # ── understanding ───────────────────────────────────────────────────────
    def _tune(self) -> Any:
        try:
            return self.context.extras.get("tune")
        except Exception:  # noqa: BLE001
            return None

    def _research_workers(self) -> int:
        """Profile-aware fan-out: a phone runs fewer researchers."""
        tune = self._tune()
        if tune is not None:
            env = getattr(getattr(tune, "profile", None), "kind", "")
            vcpu = getattr(tune, "vcpu_target", 4)
            if env in ("termux", "mobile", "embedded"):
                return 2
            if vcpu >= 8:
                return 4
            return 3
        return 3

    def _model_check(self, text: str, best: Intent) -> Intent | None:
        """The reasoning layer: one strict classification call, only when
        the deterministic pass is unsure (0.5..0.8) — never on plain chat.

        The chat thread waits at most MODEL_CHECK_TIMEOUT_S for the router;
        past that the deterministic intent stands. A timeout or a router
        failure is logged and counted (``status()`` shows it) — the model
        layer must never silently kill the mind, and must never stall the
        chat either.
        """
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        prompt = (
            "You are the intent router for an agent OS. Classify the user "
            "message. Reply with JSON only: "
            '{"kind": "chat|research|build|browse|download|mission|game|status", '
            '"target": "...", "confidence": 0.0, "why": "max 12 words"}. '
            f"Kinds: research (web investigation), build (create software), "
            f"browse (open/read a page), download (fetch a file), "
            f"mission (queue durable work), game (start/continue one of: "
            f"{', '.join(game_names())}), status (system state), chat "
            f"(everything else). Message: {text[:400]}"
        )
        # the router has no per-call deadline (providers default to
        # 120s x 3 retries), so bound it here: run it on a daemon thread
        # and take only what arrives within the interactive budget.
        box: dict[str, Any] = {"resp": None, "exc": None}

        def _call() -> None:
            try:
                box["resp"] = router.complete(prompt)
            except Exception as exc:  # noqa: BLE001
                box["exc"] = exc

        self._router_calls += 1
        db = getattr(self.context, "db", None)
        if db is not None:
            router_telemetry.record_model_check(db)
        worker = threading.Thread(target=_call, name="mind-model-check",
                                  daemon=True)
        worker.start()
        worker.join(timeout=MODEL_CHECK_TIMEOUT_S)
        if worker.is_alive():
            self._router_timeouts += 1
            if db is not None:
                router_telemetry.record_model_check(db, timed_out=True)
            self._model_check_note = (
                f"model check timed out after {MODEL_CHECK_TIMEOUT_S:.0f}s "
                f"— kept deterministic “{best.kind}”")
            _log.warning("coremind model check timed out after %.0fs on %r — "
                         "staying with deterministic %s", MODEL_CHECK_TIMEOUT_S,
                         text[:60], best.kind)
            return None
        if box["exc"] is not None:
            self._model_check_note = (
                f"model check failed: {str(box['exc'])[:100]} "
                f"— kept deterministic “{best.kind}”")
            _log.warning("coremind model check failed (%s) — staying with "
                         "deterministic %s", box["exc"], best.kind)
            return None
        resp = box["resp"]
        try:
            if not getattr(resp, "ok", False):
                return None
            raw = (getattr(resp, "text", "") or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            data = json.loads(raw)
            kind = str(data.get("kind", "")).lower()
            conf = float(data.get("confidence", 0))
            if kind in ("chat", "") or conf < max(0.75, best.confidence):
                return None
            return Intent(kind, min(conf, 0.95),
                          target=str(data.get("target", ""))[:300],
                          route={"research": "research_swarm", "build": "coding",
                                 "browse": "browser", "download": "media",
                                 "mission": "directives", "game": "games",
                                 "status": "mind"}.get(kind, "brain"),
                          why=f"model: {str(data.get('why', ''))[:80]}")
        except Exception:  # noqa: BLE001 - the model layer must never kill the mind
            _log.debug("coremind model check failed", exc_info=True)
            return None

    def decide(self, text: str, *, live_game: str | None = None,
               allow_model: bool = True) -> Intent:
        """The decision, pure and inspectable.  ``chat`` = just talk.

        Every decision is also counted in the persisted router telemetry
        (per-route counts via ``nomorals.storage.router_telemetry``) so
        ``nm mind`` shows live route behavior across processes. The write
        is best-effort — it never changes or delays the decision.
        """
        intent = self._decide(text, live_game=live_game, allow_model=allow_model)
        self._record_route(intent)
        return intent

    def _record_route(self, intent: Intent) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        router_telemetry.record_route(db, intent.route or intent.kind)

    def record_plan_error(self, error: str, route: str = "") -> None:
        """Persist the latest plan failure (with timestamp) for ``nm mind``.

        Called by dispatch paths whose agent surfaced a ``plan_error``
        (a model-made plan degraded to a template/heuristic). Never raises.
        """
        db = getattr(self.context, "db", None)
        if db is None:
            return
        router_telemetry.record_plan_error(db, error, route=route)

    def _decide(self, text: str, *, live_game: str | None = None,
                allow_model: bool = True) -> Intent:
        """The decision, pure and inspectable.  ``chat`` = just talk."""
        cands = understand(text, live_game=live_game)
        if not cands:
            return Intent("chat", 1.0, route="brain", why="no goal signal")
        best = cands[0]
        # two strong distinct signals → a multi-part goal for the orchestrator
        strong = {c.kind for c in cands if c.confidence >= 0.8}
        if len(strong) >= 2 and _MULTI_JOINERS.search(text):
            return Intent("multi", 0.8, target=text.strip()[:400], route="orchestrator",
                          why=f"{len(strong)} intent classes: {', '.join(sorted(strong))}")
        if 0.5 <= best.confidence < 0.8 and allow_model and best.kind not in ("game",):
            model = self._model_check(text, best)
            if model is not None:
                return model
        return best

    # ── clarification (owner DMs only) ──────────────────────────────────────
    def _pending_resolves(self, pending: dict[str, Any], text: str) -> Intent | None:
        """Does the next message answer the open question?"""
        t = text.strip().lower().strip(".!? ")
        if not t or _RE_CANCEL.match(text):
            return None
        if len(t) > 120:
            return None  # a new sentence, not an answer
        if pending.get("kind") == "game":
            name = _find_game_alias(text)
            if name:
                return Intent("game", 0.9, target=name, action="start",
                              route="games", why=f"answered clarification with {name}")
            if re.fullmatch(r"(yes|yep|sure|ok|okay|go|start it)", t):
                return None  # ambiguous even so — stay quiet, keep pending
            return None
        if pending.get("kind") in ("research", "build", "download", "browse", "mission"):
            if pending.get("kind") == "mission":
                return Intent("mission", 0.85, target=t, action="add",
                              route="directives", why="answered mission clarification")
            kind = pending["kind"]
            route = {"research": "research_swarm", "build": "coding",
                     "download": "media", "browse": "browser"}[kind]
            return Intent(kind, 0.85, target=t, route=route,
                          why="answered clarification")
        return None

    # ── the entry point (Social Operator calls this per message) ───────────
    def handle(self, text: str, *, message: Any, chat_key: str) -> str | None:
        """Route one inbound message.  Returns a reply to send, or None to
        fall through to the normal conversation flow.

        STRUCTURAL GATE: anything that is not the owner's own DM returns
        None immediately — in non-owner chats, commands are the only
        trigger, and no launch path exists for natural language.
        """
        if not text or not text.strip():
            return None
        if not self._is_owner_dm(message, chat_key):
            return None
        text = text.strip()

        # an open clarification? this message may be the answer
        pending = self._get_pending(chat_key)
        if pending is not None:
            if _RE_CANCEL.match(text):
                self._clear_pending(chat_key)
                return "ok — scrapped. what's next?"
            resolved = self._pending_resolves(pending, text)
            if resolved is not None:
                self._clear_pending(chat_key)
                return self._dispatch(resolved, chat_key, message) or None
            # not an answer — clear the stale question and fall through
            self._clear_pending(chat_key)

        live_game = self._live_game(chat_key)
        # wave F1 stream 2: the fast path sits ABOVE the heavy path.
        # Trivial chat (greetings, time/date, chitchat, thanks, farewells,
        # bare acks) gets a deterministic reply right here — no router
        # model call, no swarms, no research loops, no heavy organs, and
        # no brain call either.  The patterns are whole-message anchored,
        # so explicit organ invocations and multi-intent inputs can never
        # match; anything ambiguous returns None and falls through below
        # (fail-open toward capability, never toward cheapness).
        fast = fast_path(text)
        if fast is not None:
            reply, why = fast
            self._record_route(Intent("fastchat", 1.0, route="fastchat",
                                     why=why))
            _log.info("core mind fast path in %s: %s", chat_key, why)
            return reply

        intent = self.decide(text, live_game=live_game, allow_model=True)
        if intent.kind == "chat":
            return None
        if intent.action == "ask":
            question = self._question_for(intent)
            self._set_pending(chat_key, intent, question)
            return question
        return self._dispatch(intent, chat_key, message) or None

    def _is_owner_dm(self, message: Any, chat_key: str) -> bool:
        """The structural gate: owner's own DMs (or the console) only.

        ``_is_operator`` is exactly "console or a configured owner chat" —
        group chats and other people's DMs never match, so no launch path
        exists for them.  In tests/CLI (no runtime) we accept a plain DM.
        """
        if chat_key.endswith(":console"):
            return True
        if self.runtime is not None:
            try:
                return bool(self.runtime._is_operator(message))
            except Exception:  # noqa: BLE001
                return False
        return self._dm_kind(message)

    def _dm_kind(self, message: Any) -> bool:
        try:
            from ..social.chat import ChatKind

            return message.chat.kind == ChatKind.DM
        except Exception:  # noqa: BLE001
            kind = str(getattr(getattr(message, "chat", None), "kind", "")).lower()
            return kind in ("dm", "kind.dm", "chatkind.dm", "private", "direct")

    def _live_game(self, chat_key: str) -> str | None:
        if self.runtime is None:
            return None
        try:
            room = self.runtime._game_engine().live(chat_key)
            return getattr(room, "game", None)
        except Exception:  # noqa: BLE001
            return None

    def _question_for(self, intent: Intent) -> str:
        if intent.kind == "game":
            names = ", ".join(["hangman", "mafia", "wordchain", "rpg", "trivia", "spy"])
            return (f"which one — {names}? (or /game list for all 19, "
                    f"and I'll start it)")
        if intent.kind == "research":
            return "research what exactly? give me the topic and I'll fan out."
        if intent.kind == "build":
            return "build what? one line on what it should do and I'll start."
        if intent.kind in ("download", "browse"):
            return "give me the URL (or the exact file name) and I'll get it."
        if intent.kind == "mission":
            return "what should the mission actually accomplish?"
        return "can you be more specific?"

    # ── dispatch (the hands) ────────────────────────────────────────────────
    def _dispatch(self, intent: Intent, chat_key: str, message: Any) -> str | None:
        job_id = self._new_job(intent)
        fn = {
            "game": self._dispatch_game,
            "research": self._dispatch_research,
            "build": self._dispatch_build,
            "browse": self._dispatch_browse,
            "download": self._dispatch_download,
            "mission": self._dispatch_mission,
            "status": self._dispatch_status,
            "multi": self._dispatch_multi,
        }.get(intent.kind)
        if fn is None:
            self._job_done(job_id, True, "no route — treated as chat")
            return None
        try:
            reply = fn(intent, job_id, chat_key, message)
        except Exception as exc:  # noqa: BLE001 - a dispatch failure must not kill the chat
            _log.exception("coremind dispatch failed for %s", intent.kind)
            self._job_done(job_id, False, str(exc)[:200])
            return f"that route just failed: {str(exc)[:160]} — /mind status shows the detail."
        if reply is None:
            self._job_done(job_id, False, "route returned nothing")
        else:
            self._job_done(job_id, not reply.startswith("❌"))
        return reply

    def _route_line(self, intent: Intent) -> str:
        return f"(route: {intent.route or 'brain'} — {intent.why})"

    def _send_async(self, chat_key: str, job: Callable[[], str | None],
                    job_id: str, started_note: str, kind: str = "job") -> str:
        """Heavy organs run off the chat thread and report back here.

        BOUNDED: at most ``mind.max_inflight`` jobs run at once (default
        8, env NM_MIND_MAX_INFLIGHT). A send that cannot take a slot
        within ``mind.inflight_acquire_timeout_s`` (default 30s, env
        NM_MIND_INFLIGHT_TIMEOUT) is SHED — a warning is logged, the
        ``shed`` counter rises (visible in ``status()`` and ``nm mind``),
        and the job is marked failed with an explicit "shed" note so the
        owner sees it instead of silence. The chat thread never blocks on
        the bound: ``Semaphore.acquire`` is given the timeout, not an
        unbounded wait.
        """
        acquired = self._inflight.acquire(timeout=self._inflight_timeout)
        if not acquired:
            note = (f"shed: {self._max_inflight} background jobs already "
                    f"in flight (no slot in {self._inflight_timeout:.0f}s)")
            _log.warning("coremind shed %s %s — %s", kind, job_id, note)
            with self._lock:
                self._send_shed += 1
            self._job_done(job_id, False, note)
            return started_note + f"\n⚠️ {note}"

        def _run() -> None:
            note = ""
            ok = False
            try:
                note = job() or ""
                ok = True
            except Exception as exc:  # noqa: BLE001
                _log.exception("coremind job %s failed", job_id)
                note = str(exc)[:300]
            finally:
                self._inflight.release()
                with self._lock:
                    self._inflight_now = max(0, self._inflight_now - 1)
            self._job_done(job_id, ok, note)
            if self.runtime is not None:
                try:
                    ref = self.runtime._ref_from_key(chat_key)
                    text = (note if (ok and note)
                            else f"✅ done: {kind} {job_id}" if ok
                            else f"❌ that job failed: {note}")
                    self.runtime._send_long(ref.platform, ref, text)
                except Exception:  # noqa: BLE001
                    _log.exception("coremind notify failed")
        with self._lock:
            self._send_started += 1
            self._inflight_now += 1
        threading.Thread(target=_run, name=f"mind-{kind}-{job_id}",
                         daemon=True).start()
        return started_note

    def _dispatch_research(self, intent: Intent, job_id: str, chat_key: str,
                           message: Any) -> str:
        workers = self._research_workers()
        tune = self._tune()
        env = getattr(tune, "environment", "auto") if tune else "auto"

        def job() -> str:
            from .research_swarm import ResearchSwarm

            report = ResearchSwarm(self.context, workers=workers).run(
                intent.target, save_memory=True)
            text = report.to_text()
            # Wave E: close the lexicon acquisition loop — scored findings
            # feed candidate terms into the partner lexicon categories
            # (scored, versioned, reloaded). Best-effort: a loop failure
            # is a log line, never a broken research reply.
            try:
                from ..partner.lexicon_acquire import feed_partner_lexicon

                loop = feed_partner_lexicon(
                    self.context.db, report.findings,
                    source="research_swarm")
                if loop.get("added"):
                    _log.info(
                        "lexicon loop: %d terms from research %r "
                        "(partner lexicon v%s)",
                        len(loop["added"]), intent.target[:60],
                        loop.get("version"))
            except Exception:  # noqa: BLE001
                _log.debug("lexicon acquisition loop failed", exc_info=True)
            return (f"🔎 research done — “{intent.target[:80]}”\n\n{text[:4000]}")

        return self._send_async(
            chat_key, job, job_id,
            f"on it — researching “{intent.target[:80]}” with {workers} "
            f"researchers (profile: {env}). findings land here.\n{self._route_line(intent)}",
            kind="research")

    def _dispatch_build(self, intent: Intent, job_id: str, chat_key: str,
                        message: Any) -> str:
        def job() -> str:
            from .coding import CodingAgent

            started = time.time()
            result = CodingAgent(self.context).run(intent.target,
                                                   max_iterations=5,
                                                   timeout=CODING_JOB_TIMEOUT_S)
            elapsed = time.time() - started
            if result.ok:
                files = ", ".join(getattr(result, "files", []) or []) or "output"
                return (f"✅ built “{intent.target[:70]}” in {result.iterations} "
                        f"iteration(s), {elapsed:.0f}s — {files}\n"
                        f"run output:\n{str(getattr(result, 'output', ''))[:800]}")
            return (f"❌ the builder gave up after {result.iterations} "
                    f"iteration(s): {str(getattr(result, 'error', ''))[:400]}")

        return self._send_async(
            chat_key, job, job_id,
            f"building — “{intent.target[:80]}”. I'll report back here when "
            f"it runs.\n{self._route_line(intent)}",
            kind="build")

    def _browser_session_dir(self) -> str:
        try:
            return self.context.settings.resolve("data/browser/sessions")
        except Exception:  # noqa: BLE001
            return ""

    def _dispatch_browse(self, intent: Intent, job_id: str, chat_key: str,
                         message: Any) -> str:
        url = intent.target if intent.target.startswith("http") else ""
        if not url:
            # a site name: open its front door
            url = {"hn": "https://news.ycombinator.com",
                   "hacker news": "https://news.ycombinator.com"}.get(
                       intent.target.lower(),
                       f"https://{intent.target if '.' in intent.target else intent.target + '.com'}")

        def job() -> str:
            from ..tools.browser import BrowserSession

            session = BrowserSession(name=f"mind-{job_id}",
                                     session_dir=self._browser_session_dir())
            report = session.task(steps=[
                {"act": "open", "url": url},
                {"act": "extract", "kind": "markdown"},
            ])
            md = ""
            for step in report.get("steps", []):
                if step.get("act") == "extract" and step.get("ok"):
                    md = str(step.get("data", ""))
            if not report.get("ok"):
                return f"❌ browse failed: {str(report.get('error', report))[:300]}"
            return f" from {url}:\n\n{md[:3500]}"

        return self._send_async(
            chat_key, job, job_id,
            f"opening {url[:80]} — I'll post what's there.\n{self._route_line(intent)}",
            kind="browse")

    def _dispatch_download(self, intent: Intent, job_id: str, chat_key: str,
                           message: Any) -> str:
        tune = self._tune()
        max_mb = getattr(tune, "max_download_mb", 0) or 0

        def job() -> str:
            from ..tools import media

            dest = self.context.settings.resolve("data/downloads")
            result = media.download(intent.target, dest, max_mb=max_mb)
            size = int(result.get("bytes", 0))
            return (f"💾 saved {result.get('title') or intent.target[:60]} — "
                    f"{size / 1048576:.1f} MB at {result.get('path')}")

        return self._send_async(
            chat_key, job, job_id,
            f"downloading {intent.target[:80]} (profile cap: "
            f"{max_mb:.0f} MB) — I'll confirm when it lands.\n{self._route_line(intent)}",
            kind="download")

    def _directives(self):
        from .directives import DirectivesAgent

        notifier = None
        if self.runtime is not None:
            try:
                from .notifier import Notifier

                notifier = Notifier(self.context, self.runtime.gateway)
            except Exception:  # noqa: BLE001
                notifier = None
        return DirectivesAgent(self.context, notifier=notifier)

    def _dispatch_mission(self, intent: Intent, job_id: str, chat_key: str,
                          message: Any) -> str:
        agent = self._directives()
        if intent.action == "add":
            result = agent.add(intent.target)
            if not result.get("ok"):
                return f"❌ could not queue it: {result.get('error')}"
            return (f"📋 mission queued {result['id'][:8]}: “{intent.target[:80]}” — "
                    f"/task run to execute now.\n{self._route_line(intent)}")
        if intent.action == "run":
            # executing a mission can take minutes (downloads, research) —
            # it is heavy work, so it runs off the chat thread like every
            # other organ. add/list stay inline: they are fast DB reads.
            def job() -> str:
                result = agent.run(intent.target or "")
                if not result.get("ok"):
                    return f"❌ mission: {result.get('error')}"
                return (f"✅ mission {str(result.get('id', ''))[:8]} done:\n"
                        f"{str(result.get('result', ''))[:1500]}")

            return self._send_async(
                chat_key, job, job_id,
                f"running the mission — “{(intent.target or 'next queued')[:80]}”. "
                f"I'll post the result here.\n{self._route_line(intent)}",
                kind="mission")
        rows = agent.list(8)
        if not rows:
            return "no queued missions — give me one: “mission: …”."
        lines = [f"queued missions ({len(rows)}):"]
        for row in rows:
            lines.append(f"  [{row.get('status', '?')}] {row.get('id', '')[:8]} "
                         f"{str(row.get('text', ''))[:70]}")
        return "\n".join(lines)

    def _dispatch_status(self, intent: Intent, job_id: str, chat_key: str,
                         message: Any) -> str:
        return self.status()

    def _dispatch_multi(self, intent: Intent, job_id: str, chat_key: str,
                        message: Any) -> str:
        """Two-plus strong intents → the nervous system runs them as a plan."""

        def job() -> str:
            from .orchestrator import Orchestrator

            orch = Orchestrator(self.context)
            result = orch.run(intent.target, reflect=False)
            steps = result.steps if hasattr(result, "steps") else []
            lines = [f"orchestrator: {getattr(result, 'status', 'done')} "
                     f"({len(steps)} steps)"]
            for step in steps[-8:]:
                name = getattr(step, "name", getattr(step, "kind", "?"))
                ok = getattr(step, "ok", True)
                lines.append(f"  [{'✓' if ok else '✗'}] {name}")
            return "\n".join(lines)[:2500]

        return self._send_async(
            chat_key, job, job_id,
            f"multi-part goal — the orchestrator is planning it:\n"
            f"“{intent.target[:120]}”\n{self._route_line(intent)}",
            kind="multi")

    def _dispatch_game(self, intent: Intent, job_id: str, chat_key: str,
                       message: Any) -> str:
        if self.runtime is None:
            return ("games live in the chat (run `nm chat`), not the console — "
                    "or just use /game there.")
        tail = {"start": intent.target, "resume": intent.target,
                "board": f"leaderboard {intent.target}".strip(),
                "economy": intent.target or "balance"}.get(intent.action, intent.target)
        try:
            player = self.runtime._game_player(message)
        except Exception:  # noqa: BLE001
            player = None
        return self.runtime._control_game(tail, chat_key, player=player, kind="dm") or \
            "the games organ had nothing to say."

    # ── the inspectable surface ─────────────────────────────────────────────
    def status(self) -> str:
        tune = self._tune()
        env = getattr(getattr(tune, "profile", None), "kind", "auto") if tune else "auto"
        active = [j for j in self._jobs if j.get("status") == "active"]
        done = [j for j in self._jobs if j.get("status") == "done"]
        failed = [j for j in self._jobs if j.get("status") == "failed"]
        lines = ["🧠 core mind"]
        last = self._state.get("last_objective")
        if last:
            age = max(0, time.time() - float(last.get("created", 0))) / 60
            lines.append(f"  last objective: “{last.get('text', '')[:70]}” "
                         f"→ {last.get('route')} ({age:.0f}m ago)")
        pend = self._state.get("pending", {})
        for chat_key, p in list(pend.items())[-3:]:
            lines.append(f"  pending question in {chat_key}: “{p.get('question', '')[:60]}”")
        lines.append(f"  jobs: {len(active)} active · {len(done)} done · "
                     f"{len(failed)} failed")
        for j in self._jobs[-4:]:
            lines.append(f"    [{j.get('status')}] {j.get('kind')} "
                         f"{str(j.get('target', ''))[:50]} — {str(j.get('note', ''))[:40]}")
        if self.runtime is not None:
            try:
                rooms = self.runtime._game_engine().rooms()
                if rooms:
                    names = ", ".join(f"{r.get('game')} in {r.get('chat_key')}"
                                      for r in rooms[:4])
                    lines.append(f"  live games: {names}")
            except Exception:  # noqa: BLE001
                pass
            try:
                directives = self._directives().list(5)
                queued = [d for d in directives if d.get("status") in ("queued", "pending")]
                if queued:
                    lines.append(f"  queued missions: {len(queued)} (/task list)")
            except Exception:  # noqa: BLE001
                pass
        lines.append(f"  model consulted {self._router_calls}× this session "
                     f"(timed out {self._router_timeouts}×) · profile: {env}")
        lines.append(f"  background jobs: {self._inflight_now}/{self._max_inflight} "
                     f"in flight · started {self._send_started} · "
                     f"shed {self._send_shed} (bound: NM_MIND_MAX_INFLIGHT)")
        if self._model_check_note:
            lines.append(f"  last model check: {self._model_check_note}")
        return "\n".join(lines)

    def clear(self, chat_key: str = "") -> int:
        with self._lock:
            if chat_key:
                n = 1 if self._state.get("pending", {}).pop(chat_key, None) else 0
            else:
                n = len(self._state.get("pending", {}))
                self._state["pending"] = {}
            self._save_state()
            return n

    def pending(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.get("pending", {}))


# ── console / CLI entry ──────────────────────────────────────────────────────

def run_goal(context: Any, goal: str, *, json_out: bool = False) -> str:
    """`nm goal <goal>` — the mind from the terminal.

    The console counts as the owner's DM; dispatch runs inline (no chat
    thread) so the result lands on stdout.
    """
    mind = CoreMind(context)
    intent = mind.decide(goal, allow_model=True)
    if intent.kind == "chat":
        line = f"no goal signal in “{goal[:60]}” — say what you want done " \
               "(research/build/browse/download/mission/game)."
        return json.dumps({"intent": intent.to_dict(), "reply": line}) if json_out else line
    if intent.action == "ask":
        line = f"need one more detail: {mind._question_for(intent)}"
        return json.dumps({"intent": intent.to_dict(), "reply": line}) if json_out else line
    if intent.kind == "game":
        line = "games live in the chat — start one there with “let's play …” or /game."
        return json.dumps({"intent": intent.to_dict(), "reply": line}) if json_out else line

    job_id = mind._new_job(intent)
    try:
        fn = {"research": mind._dispatch_research, "build": mind._dispatch_build,
              "browse": mind._dispatch_browse, "download": mind._dispatch_download,
              "mission": mind._dispatch_mission, "status": mind._dispatch_status,
              "multi": mind._dispatch_multi}[intent.kind]
        # run the inner work synchronously: swap the async wrapper
        reply = _dispatch_inline(mind, fn, intent, job_id)
    except Exception as exc:  # noqa: BLE001
        mind._job_done(job_id, False, str(exc)[:200])
        reply = f"❌ {exc}"
    out = f"{reply}\n{mind._route_line(intent)}"
    return json.dumps({"intent": intent.to_dict(), "reply": reply}) if json_out else out


def _dispatch_inline(mind: CoreMind, fn: Callable, intent: Intent, job_id: str) -> str:
    """Execute a dispatch job's inner work on the calling thread."""
    if intent.kind == "mission":
        return fn(intent, job_id, "console", None)
    # heavy organs: replicate the job() closure synchronously
    if intent.kind == "research":
        from .research_swarm import ResearchSwarm

        workers = mind._research_workers()
        report = ResearchSwarm(mind.context, workers=workers).run(
            intent.target, save_memory=True)
        mind._job_done(job_id, True)
        return f"🔎 research done — “{intent.target[:80]}”\n\n{report.to_text()[:4000]}"
    if intent.kind == "build":
        from .coding import CodingAgent

        started = time.time()
        result = CodingAgent(mind.context).run(intent.target, max_iterations=5,
                                               timeout=CODING_JOB_TIMEOUT_S)
        elapsed = time.time() - started
        ok = bool(result.ok)
        mind._job_done(job_id, ok)
        if ok:
            files = ", ".join(getattr(result, "files", []) or []) or "output"
            return (f"✅ built “{intent.target[:70]}” in {result.iterations} "
                    f"iteration(s), {elapsed:.0f}s — {files}\n"
                    f"{str(getattr(result, 'output', ''))[:800]}")
        return f"❌ gave up: {str(getattr(result, 'error', ''))[:400]}"
    if intent.kind == "browse":
        from ..tools.browser import BrowserSession

        url = intent.target if intent.target.startswith("http") else \
            f"https://{intent.target}"
        session = BrowserSession(name=f"mind-{job_id}",
                                 session_dir=mind._browser_session_dir())
        report = session.task(steps=[{"act": "open", "url": url},
                                      {"act": "extract", "kind": "markdown"}])
        md = ""
        for step in report.get("steps", []):
            if step.get("act") == "extract" and step.get("ok"):
                md = str(step.get("data", ""))
        mind._job_done(job_id, bool(report.get("ok")))
        return f"🌐 from {url}:\n\n{md[:3500]}" if report.get("ok") else \
            f"❌ {str(report.get('error', report))[:300]}"
    if intent.kind == "download":
        from ..tools import media

        tune = mind._tune()
        max_mb = getattr(tune, "max_download_mb", 0) or 0
        result = media.download(intent.target,
                                mind.context.settings.resolve("data/downloads"),
                                max_mb=max_mb)
        mind._job_done(job_id, True)
        return f"💾 saved {result.get('title') or intent.target[:60]} at {result.get('path')}"
    if intent.kind == "multi":
        from .orchestrator import Orchestrator

        result = Orchestrator(mind.context).run(intent.target, reflect=False)
        mind._job_done(job_id, True)
        steps = getattr(result, "steps", []) or []
        lines = [f"orchestrator: {getattr(result, 'status', 'done')} ({len(steps)} steps)"]
        for step in steps[-8:]:
            name = getattr(step, "name", getattr(step, "kind", "?"))
            ok = getattr(step, "ok", True)
            lines.append(f"  [{'✓' if ok else '✗'}] {name}")
        return "\n".join(lines)[:2500]
    if intent.kind == "status":
        mind._job_done(job_id, True)
        return mind.status()
    return "no route"
