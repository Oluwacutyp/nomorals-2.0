"""Experience packs (Living Chat P4, 2026-11-08).

Phase 4 of the Living Chat program: the journeys of doc 2 §6 —
music deck, research cockpit, learning cards, group boards and
the rest — shipped as ONE pack system, not 25 one-offs.

A pack is a DECLARATIVE definition (:class:`PackDef`): its
identity, the manifest surfaces it composes, its style family,
its entry points. Packs compose the machinery the earlier
phases built — the surface manifest, the rich action schema,
the operation store, living cards, owner style pins and the
procedural creation pipeline. A pack that wraps an existing
Devon feature points its home card at that feature's real
command; it never reimplements it. A pack with behaviour of
its own (a travel board, a decision room, a branching story)
keeps that behaviour as pure state transitions here, over a
durable JSON store (the PinStore pattern — ``packs.json``
under the Devon home; ``storage/migrations.py`` is never
touched).

Lifecycle: every registered pack is installed and enabled by
default. Disabling a pack persists; a disabled pack's surfaces
stop resolving — opening it answers unavailable in words and
publishes nothing, and its verbs refuse the same way.

Style: a pack's home card is a living card whose family IS the
pack's id, so an owner style pin targets the pack through the
existing P3 seam (``/stylepin set music-deck theme neon``) and
the card names the pin in force, like every other family.

Sessions: stateful packs keep a session per scope — the owner
(``owner:<sender>``) for personal packs, the chat
(``chat:<chat_key>``) for shared rooms (the decision room).
Owner-scoped state is what makes the cross-device handoff
pack true: open the pack on another chat or device and the
same session is there.
"""

from __future__ import annotations

import html
import json
import os
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_schema import Action

__all__ = [
    "PackDef", "PACKS", "PACK_IDS", "pack_by_id",
    "PackStore", "get_pack_store", "set_pack_store",
    "HomeView", "VerbOutcome", "build_home", "apply_verb",
    "initial_session",
    "scope_key", "listing_lines", "pack_inline_articles",
    "render_companion_html",
    "ADVENTURE_NODES", "ADVENTURE_START", "LEARNING_DECK",
]


# ── the definition ──────────────────────────────────────────────


@dataclass(frozen=True)
class PackDef:
    """One experience pack, declared — not coded per-pack."""

    id: str
    title: str
    tagline: str
    icon: str
    kind: str                      # wrapped | native | hybrid
    entry: str = ""                # the wrapped feature's command
    entry_label: str = ""          # its button label
    #: The typography surface class the pack's home card renders
    #: under. A pack home is PRESENTED output (a deck, a cockpit,
    #: a board), so the default is a presented class — this is
    #: what carries the stamp; before the field existed, pack
    #: families resolved to living_cards' chat default and pack
    #: homes shipped unstamped (owner report, 2026-11-11).
    surface_class: str = "menu"
    surfaces: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ()    # native verbs (pack act …)
    session_scope: str = "owner"   # owner | chat
    traits: dict[str, str] = field(default_factory=dict)
    needs: tuple[str, ...] = ()    # declared, unbuilt capabilities

    @property
    def family(self) -> str:
        """The style-pin scope / living-card family: the id."""
        return self.id


def _p(**kw: Any) -> PackDef:
    return PackDef(**kw)


PACKS: tuple[PackDef, ...] = (
    _p(id="command-center", title="Live command center",
       tagline="ongoing jobs, live cards and one-tap resume for "
               "this chat",
       icon="🛎️", kind="wrapped", entry="status",
       entry_label="Full status",
       surfaces=("live-session", "startup", "restart", "packs")),
    _p(id="artifact-workbench", title="Artifact workbench",
       tagline="preview, revise, compare and export the files "
               "Devon creates",
       icon="🧰", kind="wrapped", entry="create",
       entry_label="Create something",
       surfaces=("creation", "packs")),
    _p(id="research-cockpit", title="Research cockpit",
       tagline="claims, evidence, disagreements and source "
               "freshness for a question",
       icon="🔬", kind="wrapped", entry="research",
       entry_label="Research a question",
       surfaces=("adaptive-output", "packs")),
    _p(id="music-deck", title="Music deck",
       tagline="cover, queue and real controls for what's playing",
       icon="🎛️", kind="wrapped", entry="play",
       entry_label="Play something",
       surfaces=("play-picklist", "downloads", "packs"),
       traits={"accent": "violet"}),
    _p(id="creative-director", title="Creative director",
       tagline="moodboards, style, aspect and pacing choices for "
               "image and video work",
       icon="🎬", kind="wrapped", entry="director",
       entry_label="Open the director",
       surfaces=("creation", "style-pins", "packs")),
    _p(id="sticker-atelier", title="Sticker atelier",
       tagline="design, preview, animate and organize original "
               "sticker packs",
       icon="🏷️", kind="wrapped", entry="sticker",
       entry_label="Make a sticker",
       surfaces=("expression", "creation", "packs")),
    _p(id="learning-cards", title="Interactive learning cards",
       tagline="reveal, hint, quiz and spaced review, with "
               "progress that persists",
       icon="🃏", kind="hybrid", entry="wisdom",
       entry_label="Wisdom keeper",
       surfaces=("live-session", "packs"),
       verbs=("reveal", "hint", "known", "next")),
    _p(id="game-boards", title="Group game boards",
       tagline="turn indicator, move choices, scores, rematch "
               "and tournament navigation",
       icon="🎲", kind="wrapped", entry="game",
       entry_label="Games",
       surfaces=("games", "group-boards", "packs")),
    _p(id="book-studio", title="Worldbuilding & book studio",
       tagline="character cards, plot branches, chapter "
               "navigation and continuity checks",
       icon="📚", kind="wrapped", entry="book",
       entry_label="Open the studio",
       surfaces=("creation", "packs")),
    _p(id="smart-digest", title="Smart digest",
       tagline="step through the day's items — read, save, "
               "summarize, investigate",
       icon="🗞️", kind="hybrid", entry="news",
       entry_label="Latest news",
       surfaces=("adaptive-output", "packs"),
       verbs=("next", "prev")),
    _p(id="memory-shelf", title="Conversation memory shelf",
       tagline="save a useful fact, correct it, link its source, "
               "see where it was shelved",
       icon="🧠", kind="native",
       surfaces=("adaptive-output", "packs"),
       verbs=("shelve", "list", "forget")),
    _p(id="calendar-host", title="Calendar & event host",
       tagline="time-zone-aware proposals, RSVP, reschedule, "
               "agenda and recaps",
       icon="📅", kind="wrapped", entry="schedule",
       entry_label="My schedule",
       surfaces=("adaptive-output", "packs")),
    _p(id="travel-board", title="Travel & planning board",
       tagline="itinerary legs you add, drop and rearrange — "
               "the board persists",
       icon="🧳", kind="native",
       surfaces=("live-session", "packs"),
       verbs=("add", "drop", "clear")),
    _p(id="media-vault", title="Personal media vault",
       tagline="search the media Devon has saved for you, by "
               "kind and tag",
       icon="🗄️", kind="native",
       surfaces=("expression", "packs"),
       verbs=("find",)),
    _p(id="decision-room", title="Collaborative decision room",
       tagline="options, votes and a decision record the room "
               "can stand on",
       icon="🗳️", kind="native", session_scope="chat",
       surfaces=("live-session", "packs"),
       verbs=("propose", "vote", "close")),
    _p(id="diagnostic-panel", title="Live diagnostic panel",
       tagline="real bridge and account state, the last failure "
               "stage, and repair choices",
       icon="🩺", kind="wrapped", entry="doctor",
       entry_label="Run diagnostics",
       surfaces=("startup", "packs")),
    _p(id="adventure", title="Choose-your-own adventure",
       tagline="a persistent branching story — your choices "
               "are remembered",
       icon="🗺️", kind="native",
       surfaces=("games", "live-session", "packs"),
       verbs=("choose", "restart")),
    _p(id="celebration-engine", title="Celebration engine",
       tagline="personalized cards and artwork for milestones, "
               "birthdays and launches — rendered by her own "
               "studio, never a generation model",
       icon="🎉", kind="hybrid", entry="create",
       entry_label="Create",
       surfaces=("expression", "creation", "packs"),
       verbs=("celebrate",)),
    _p(id="workflow-buttons", title="Reusable workflow buttons",
       tagline="save a command that worked as a named shortcut "
               "and replay it in one tap",
       icon="⏺️", kind="native",
       surfaces=("button-fallback", "packs"),
       verbs=("save", "run", "list", "drop")),
    _p(id="inline-creation", title="Inline creation",
       tagline="search and insert prepared pack entries from "
               "any chat's inline mode",
       icon="⌨️", kind="native",
       surfaces=("inline-query", "packs"),
       verbs=("results",)),
    _p(id="companion-views", title="Richer companion views",
       tagline="any pack rendered as a full designed page you "
               "can open anywhere",
       icon="🖼️", kind="native",
       surfaces=("live-session", "packs"),
       verbs=("view",)),
    _p(id="cross-device-handoff", title="Cross-device handoff",
       tagline="begin in one chat, continue in another — pack "
               "state follows the owner, not the device",
       icon="🔀", kind="native",
       surfaces=("live-session", "button-fallback", "packs"),
       verbs=("note",)),
    _p(id="language-expression", title="Language-aware expression",
       tagline="reactions and captions with the original text "
               "always retained",
       icon="🗣️", kind="native",
       surfaces=("expression", "packs"),
       verbs=("say",),
       needs=("per-locale caption catalogues beyond the "
              "originals",)),
    _p(id="action-bar", title="Personal action bar",
       tagline="your most-used pack actions, learned from real "
               "usage — pins lead",
       icon="⭐", kind="native",
       surfaces=("button-fallback", "menu", "packs"),
       verbs=("pin", "unpin")),
    _p(id="surprise-me", title="Surprise me",
       tagline="a useful tool, a fresh style, a library item or "
               "a mini-creation — labeled for what it is",
       icon="🎲", kind="native",
       surfaces=("expression", "creation", "packs"),
       verbs=("roll",)),
)

PACK_IDS: tuple[str, ...] = tuple(p.id for p in PACKS)

# Every pack's home card renders as a living card of the pack's
# family; register each pack's declared surface class with the
# living-card renderer so a pack home carries its presented
# class (and the stamp that class carries) BY CONSTRUCTION —
# not by each home builder remembering to ask. (Deferred import
# inside the call: living_cards does not import packs, and packs
# stays importable without the card machinery at module top.)
def _register_pack_surface_classes() -> None:
    from .living_cards import register_family_class

    for pack in PACKS:
        register_family_class(pack.family, pack.surface_class)


_register_pack_surface_classes()
_BY_ID = {p.id: p for p in PACKS}


def pack_by_id(pack_id: str) -> PackDef:
    """The definition for ``pack_id`` (KeyError if absent)."""
    return _BY_ID[str(pack_id or "")]


# ── the durable store ───────────────────────────────────────────


class PackStore:
    """Pack lifecycle + sessions + usage, as one JSON file.

    ``path=None`` keeps the store in memory only. Reads never
    raise: a corrupt file reads as a fresh store.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load(self._path) \
            if self._path is not None else {}

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt reads as fresh
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1,
                                      sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:  # noqa: BLE001 - a failed save keeps memory
            pass

    # ── lifecycle ────────────────────────────────────────────

    def is_enabled(self, pack_id: str) -> bool:
        with self._lock:
            states = self._data.get("lifecycle") or {}
            return bool(states.get(str(pack_id), True))

    def set_enabled(self, pack_id: str, enabled: bool) -> None:
        with self._lock:
            states = dict(self._data.get("lifecycle") or {})
            states[str(pack_id)] = bool(enabled)
            self._data["lifecycle"] = states
            self._save()

    # ── sessions ─────────────────────────────────────────────

    def session(self, pack_id: str, scope: str) -> dict[str, Any]:
        """One pack session (a copy — save it back to persist)."""
        with self._lock:
            sessions = self._data.get("sessions") or {}
            raw = sessions.get(f"{pack_id}\x1f{scope}") or {}
            return json.loads(json.dumps(raw)) \
                if isinstance(raw, dict) else {}

    def save_session(self, pack_id: str, scope: str,
                     session: dict[str, Any]) -> None:
        with self._lock:
            sessions = dict(self._data.get("sessions") or {})
            sessions[f"{pack_id}\x1f{scope}"] = dict(session or {})
            self._data["sessions"] = sessions
            self._save()

    # ── usage + pins (the action bar's memory) ───────────────

    def record_use(self, pack_id: str) -> None:
        with self._lock:
            usage = dict(self._data.get("usage") or {})
            usage[str(pack_id)] = int(usage.get(str(pack_id), 0)) + 1
            self._data["usage"] = usage
            self._save()

    def usage_counts(self) -> dict[str, int]:
        with self._lock:
            return {str(k): int(v) for k, v in
                    (self._data.get("usage") or {}).items()}

    def pinned(self) -> list[str]:
        with self._lock:
            return [str(x) for x in (self._data.get("pinned") or [])]

    def set_pinned(self, pack_id: str, pinned: bool) -> None:
        with self._lock:
            pins = [str(x) for x in (self._data.get("pinned") or [])]
            if pinned and pack_id not in pins:
                pins.append(str(pack_id))
            if not pinned:
                pins = [x for x in pins if x != str(pack_id)]
            self._data["pinned"] = pins
            self._save()

    # ── workflows (the workflow-buttons pack's shortcuts) ────

    def workflows(self) -> dict[str, str]:
        with self._lock:
            return {str(k): str(v) for k, v in
                    (self._data.get("workflows") or {}).items()}

    def set_workflow(self, name: str, command: str) -> None:
        with self._lock:
            flows = dict(self._data.get("workflows") or {})
            flows[str(name)] = str(command)
            self._data["workflows"] = flows
            self._save()

    def drop_workflow(self, name: str) -> bool:
        with self._lock:
            flows = dict(self._data.get("workflows") or {})
            if str(name) not in flows:
                return False
            del flows[str(name)]
            self._data["workflows"] = flows
            self._save()
            return True


def _default_path() -> Path:
    home = os.environ.get("NM_HOME", "") or str(Path.home() / ".nomorals")
    return Path(home).expanduser() / "packs.json"


_STORE: PackStore | None = None
_STORE_LOCK = threading.Lock()


def get_pack_store() -> PackStore:
    """The process-wide pack store."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = PackStore(path=_default_path())
        return _STORE


def set_pack_store(store: PackStore | None) -> PackStore | None:
    """Replace the process-wide store; returns the previous one."""
    global _STORE
    with _STORE_LOCK:
        previous = _STORE
        _STORE = store
        return previous


def scope_key(pack: PackDef, chat_key: str, owner_key: str) -> str:
    """The session scope for one pack in one conversation."""
    if pack.session_scope == "chat":
        return f"chat:{chat_key}"
    return f"owner:{owner_key}" if owner_key else f"chat:{chat_key}"


# ── home cards ──────────────────────────────────────────────────


@dataclass
class HomeView:
    """One pack home, as content: the runtime wraps it in a
    living card of the pack's family."""

    title: str
    body: str = ""
    rows: tuple[tuple[str, str], ...] = ()
    actions: tuple[Action, ...] = ()


@dataclass
class VerbOutcome:
    """The result of one native verb: a reply in words, the
    session to persist, and optionally an artifact to deliver
    or a command for the runtime to dispatch."""

    reply: str = ""
    session: dict[str, Any] | None = None
    artifact: tuple[str, str] | None = None   # (path, media kind)
    dispatch: str = ""                        # a control command


def _act(label: str, data: str, emoji: str = "") -> Action:
    return Action(label=label, intent="callback", data=data,
                  emoji=emoji)


def _verb_act(pack: PackDef, label: str, verb: str,
              args: str = "") -> Action:
    data = f"pack act {pack.id} {verb}"
    if args:
        data += f" {args}"
    return _act(label, data)


def _packs_act() -> Action:
    return _act("All packs", "pack", "🧩")


def _wrapped_home(pack: PackDef, session: dict, facts: dict,
                  extra_body: str = "",
                  extra_actions: tuple[Action, ...] = ()) -> HomeView:
    body = pack.tagline + "."
    if extra_body:
        body += "\n" + extra_body
    actions: list[Action] = []
    if pack.entry:
        actions.append(_act(pack.entry_label or pack.entry,
                            pack.entry, pack.icon))
    actions.extend(extra_actions)
    actions.append(_packs_act())
    return HomeView(title=f"{pack.icon} {pack.title}", body=body,
                    actions=tuple(actions))


def _command_center_home(pack, session, facts):
    live = facts.get("live_cards") or []
    lines = []
    for family, title, status in live[:8]:
        lines.append(f"• {title} ({family})"
                     + (f" — {status}" if status else ""))
    extra = ("live cards in this chat:\n" + "\n".join(lines)) \
        if lines else "no live cards in this chat right now."
    return _wrapped_home(pack, session, facts, extra_body=extra,
                         extra_actions=(
                             _act("Diagnostics", "doctor", "🩺"),
                             _act("Platforms", "platforms", "📡")))


def _music_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Pause", "play pause", "⏸️"),
        _act("Music home", "music", "🎵")))


def _atelier_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Emoji art", "emojiart", "🎨"),
        _act("Custom emoji", "customemoji list", "😀")))


def _director_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Create", "create", "✨"),
        _act("Style pins", "stylepin", "📌")))


def _workbench_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Style pins", "stylepin", "📌"),
        _act("Emoji art", "emojiart", "🎨")))


def _research_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Deep search", "searchdeep", "🔎"),
        _act("Brief", "brief", "📄")))


def _games_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Leaderboards", "boards", "🏆"),
        _act("My stats", "game stats", "📊")))


def _books_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Novel reader", "novel", "📖"),
        _act("My books", "books", "📚")))


def _calendar_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Scheduled jobs", "schedule list", "🗓️"),))


def _diagnostics_home(pack, session, facts):
    return _wrapped_home(pack, session, facts, extra_actions=(
        _act("Status", "status", "📊"),
        _act("Platforms", "platforms", "📡")))


def _celebration_home(pack, session, facts):
    return _wrapped_home(
        pack, session, facts,
        extra_body="or send: /pack act celebration-engine "
                   "celebrate <the occasion> and a studio-rendered "
                   "celebration sticker is made, inspected and "
                   "delivered.",
        extra_actions=(_act("Stickers", "sticker", "🏷️"),))


# ── native pack data ────────────────────────────────────────────

#: The learning deck: a real starter set. Progress (index,
#: revealed, known) lives in the pack session.
LEARNING_DECK: tuple[dict[str, str], ...] = (
    {"question": "What does a database index speed up?",
     "answer": "lookups — reads trade a little write cost for much "
          "faster queries",
     "hint": "think of the index at the back of a book"},
    {"question": "In rhetoric, what is anaphora?",
     "answer": "repetition of a word or phrase at the start of "
          "successive clauses",
     "hint": "a drumbeat at the head of each line"},
    {"question": "What is the half-life of carbon-14, roughly?",
     "answer": "about 5,730 years",
     "hint": "between five and six thousand years"},
    {"question": "What does 'amortize' mean in algorithms?",
     "answer": "to spread a cost across many operations so the "
          "average cost per operation stays low",
     "hint": "one expensive step, paid for by many cheap ones"},
    {"question": "Which ocean is the largest by surface area?",
     "answer": "the Pacific Ocean",
     "hint": "it borders Asia, Australia and the Americas"},
    {"question": "What is a monad, in one honest sentence?",
     "answer": "a design pattern for chaining computations that "
          "carry context, the way a pipeline carries values",
     "hint": "context-riding composition, not a burrito"},
)

#: The adventure graph: node -> text, choices, ending. The
#: first-choice path descends shore → lantern → lamp → dawn
#: without revisiting a node (the P4 test walks it).
ADVENTURE_START = "shore"
ADVENTURE_NODES: dict[str, dict[str, Any]] = {
    "shore": {
        "text": "You stand on a black-sand shore at dusk. A "
                "lighthouse blinks far north; drums sound from "
                "the village behind you.",
        "choices": (("Take the lantern path", "lantern"),
                    ("Follow the drums", "drums")),
        "ending": False},
    "lantern": {
        "text": "The lantern path climbs in switchbacks. The "
                "lighthouse door stands open, breathing warm air.",
        "choices": (("Enter the lighthouse", "lamp"),
                    ("Rest on the steps", "rest")),
        "ending": False},
    "drums": {
        "text": "The village circle is mid-dance. An elder "
                "beckons you in, smiling like she knows you.",
        "choices": (("Join the circle", "circle"),),
        "ending": False},
    "circle": {
        "text": "You dance until the fire is embers and the "
                "elder presses a carved token into your palm: "
                "'For the road back.'",
        "choices": (("Dance till moonrise", "moon"),),
        "ending": False},
    "moon": {
        "text": "Moonrise gilds the bay. You are fed, sung to "
                "and entirely unwilling to leave. Some journeys "
                "end by arriving. — THE END",
        "choices": (),
        "ending": True},
    "lamp": {
        "text": "Inside, the great lamp turns on its brass "
                "spine. The keeper's log lies open beside it.",
        "choices": (("Light the great lamp", "dawn"),
                    ("Read the keeper's log", "log")),
        "ending": False},
    "rest": {
        "text": "You rest until the stars wheel once, then "
                "climb with steadier legs.",
        "choices": (("Climb to the lamp room", "lamp"),),
        "ending": False},
    "log": {
        "text": "The last entry is in your own handwriting: "
                "'When you read this, take the map.' A chart of "
                "a hidden cove is folded inside.",
        "choices": (("Follow the map", "cove"),),
        "ending": False},
    "cove": {
        "text": "The cove holds a boat, a stocked chest and a "
                "note: 'She's yours. The sea is waiting.' "
                "— THE END",
        "choices": (),
        "ending": True},
    "dawn": {
        "text": "The lamp catches, and its beam sweeps the "
                "whole bay gold. Below, three ships answer with "
                "their horns — you have guided them home without "
                "leaving the tower. — THE END",
        "choices": (),
        "ending": True},
}


def _learning_home(pack, session, facts):
    idx = int(session.get("index", 0)) % len(LEARNING_DECK)
    card = LEARNING_DECK[idx]
    known = int(session.get("known", 0))
    body = (f"card {idx + 1} of {len(LEARNING_DECK)} · "
            f"known so far: {known}\n{card['question']}")
    if session.get("revealed"):
        body += f"\nanswer: {card['answer']}"
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_verb_act(pack, "Reveal answer", "reveal"),
                 _verb_act(pack, "Hint", "hint"),
                 _verb_act(pack, "I knew it", "known"),
                 _verb_act(pack, "Next card", "next"),
                 _act("Wisdom keeper", "wisdom", "📿")))


def _digest_home(pack, session, facts):
    items = facts.get("digest_items") or []
    idx = int(session.get("index", 0))
    if items:
        idx = max(0, min(idx, len(items) - 1))
        body = (f"item {idx + 1} of {len(items)}\n{items[idx]}")
    else:
        body = ("the digest has no items right now — the news "
                "action below pulls the latest.")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_verb_act(pack, "Next item", "next"),
                 _verb_act(pack, "Previous", "prev"),
                 _act("Latest news", "news", "📰"),
                 _act("Pulse", "pulse", "💓")))


def _shelf_home(pack, session, facts):
    facts_list = session.get("facts") or []
    if facts_list:
        lines = [f"{i}. {f.get('text', '')} "
                 f"(source: {f.get('source', '')})"
                 for i, f in enumerate(facts_list, 1)]
        body = "on your shelf:\n" + "\n".join(lines)
    else:
        body = ("your shelf is empty. Shelve one with: /pack act "
                "memory-shelf shelve <the fact> | <its source>")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_act("Refresh shelf", f"pack open {pack.id}", "🔄"),
                 _packs_act()))


def _travel_home(pack, session, facts):
    legs = session.get("legs") or []
    if legs:
        lines = [f"{i}. {leg.get('label', '')}"
                 + (f" — {leg['detail']}" if leg.get("detail")
                    else "")
                 for i, leg in enumerate(legs, 1)]
        body = "your itinerary:\n" + "\n".join(lines)
    else:
        body = ("no legs yet. Add one with: /pack act "
                "travel-board add <place> | <detail>")
    body += ("\nthis board is yours — it resumes on any chat "
             "or device you open it from.")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_act("Refresh board", f"pack open {pack.id}", "🔄"),
                 _verb_act(pack, "Clear board", "clear"),
                 _packs_act()))


def _vault_home(pack, session, facts):
    items = facts.get("library_items") or []
    if items:
        counts: dict[str, int] = {}
        for item in items:
            kind = str(item.get("kind", "media") or "media")
            counts[kind] = counts.get(kind, 0) + 1
        kinds = ", ".join(f"{k} ×{n}" for k, n in
                          sorted(counts.items()))
        body = (f"{len(items)} saved item(s): {kinds}.\n"
                f"search with: /pack act media-vault find <tag>")
    else:
        body = ("the vault is empty — media Devon saves, steals "
                "or makes for you lands here.")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_act("Refresh vault", f"pack open {pack.id}", "🔄"),
                 _packs_act()))


def _decision_home(pack, session, facts):
    proposal = session.get("proposal")
    record = session.get("record")
    if proposal:
        tally: dict[str, int] = {}
        for idx in (proposal.get("votes") or {}).values():
            label = proposal["options"][int(idx)]
            tally[label] = tally.get(label, 0) + 1
        lines = [f"open question: {proposal['question']}"]
        for i, opt in enumerate(proposal["options"], 1):
            lines.append(f"{i}. {opt} — {tally.get(opt, 0)} vote(s)")
        lines.append("vote with: /pack act decision-room vote <n>")
        body = "\n".join(lines)
        actions = (_verb_act(pack, "Close & record", "close"),
                   _act("Refresh room", f"pack open {pack.id}", "🔄"))
    elif record:
        body = (f"last decision: {record.get('question', '')} → "
                f"{record.get('winner', '')} "
                f"({record.get('tally', {})})\npropose the next "
                f"with: /pack act decision-room propose "
                f"<question> | <option> | <option>")
        actions = (_act("Refresh room", f"pack open {pack.id}", "🔄"),)
    else:
        body = ("no open proposal. Start one with: /pack act "
                "decision-room propose <question> | <option> | "
                "<option>")
        actions = (_act("Refresh room", f"pack open {pack.id}", "🔄"),)
    return HomeView(title=f"{pack.icon} {pack.title}", body=body,
                    actions=actions + (_packs_act(),))


def _adventure_home(pack, session, facts):
    node_id = str(session.get("node") or ADVENTURE_START)
    node = ADVENTURE_NODES.get(node_id) or ADVENTURE_NODES[
        ADVENTURE_START]
    actions: list[Action] = []
    for i, (label, _nxt) in enumerate(node["choices"], 1):
        actions.append(_verb_act(pack, label, "choose", str(i)))
    if node["ending"]:
        actions.append(_verb_act(pack, "Start again", "restart"))
    actions.append(_packs_act())
    return HomeView(title=f"{pack.icon} {pack.title}",
                    body=node["text"], actions=tuple(actions))


def _workflows_home(pack, session, facts):
    store = facts.get("store")
    flows = store.workflows() if store is not None else {}
    if flows:
        lines = [f"• {name} → /{cmd}" for name, cmd in
                 sorted(flows.items())]
        body = "your shortcuts:\n" + "\n".join(lines)
    else:
        body = ("no shortcuts yet. Save one with: /pack act "
                "workflow-buttons save <name> = <command>")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_act("Refresh", f"pack open {pack.id}", "🔄"),
                 _packs_act()))


def _inline_home(pack, session, facts):
    body = ("in any Telegram chat, type the bot's name then "
            "'pack' and a word — pack entries appear as inline "
            "results; tapping one posts /pack open <id> there. "
            "Here, 'Pack articles' shows exactly what a query "
            "returns.")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_verb_act(pack, "Pack articles", "results"),
                 _act("Refresh", f"pack open {pack.id}", "🔄")))


def _companion_home(pack, session, facts):
    body = ("any pack, rendered as a full designed page (HTML "
            "artifact) you can open anywhere. Try one below, or: "
            "/pack act companion-views view <pack id>")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_verb_act(pack, "View: music deck", "view",
                           "music-deck"),
                 _verb_act(pack, "View: adventure", "view",
                           "adventure"),
                 _packs_act()))


def _handoff_home(pack, session, facts):
    note = str(session.get("note") or "")
    body = ("pack state is keyed to YOU, not to a chat or a "
            "device: the travel board's legs, the adventure's "
            "place, this note — they resume wherever you open "
            "the pack.")
    if note:
        body += f"\nyour carried note: {note}"
    else:
        body += ("\nleave a note with: /pack act "
                 "cross-device-handoff note <text> — then open "
                 "this pack from another chat and read it back.")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_act("Refresh", f"pack open {pack.id}", "🔄"),
                 _packs_act()))


def _language_home(pack, session, facts):
    body = ("reactions carry their original caption text — "
            "never silently replaced. Ask for one: /pack act "
            "language-expression say <emotion>, or tap:")
    return HomeView(
        title=f"{pack.icon} {pack.title}", body=body,
        actions=(_verb_act(pack, "Hug", "say", "hug"),
                 _verb_act(pack, "Wave", "say", "wave"),
                 _verb_act(pack, "Dance", "say", "dance")))


def _action_bar_home(pack, session, facts):
    store = facts.get("store")
    usage = store.usage_counts() if store is not None else {}
    pins = store.pinned() if store is not None else []
    ranked: list[str] = []
    for pid in pins:
        if pid in _BY_ID and pid not in ranked:
            ranked.append(pid)
    for pid, _n in sorted(usage.items(), key=lambda kv: -kv[1]):
        if pid in _BY_ID and pid not in ranked:
            ranked.append(pid)
    for pid in ("music-deck", "adventure", "research-cockpit"):
        if pid not in ranked:
            ranked.append(pid)
    ranked = [pid for pid in ranked if pid != pack.id][:6]
    actions = tuple(
        _act(_BY_ID[pid].title, f"pack open {pid}",
             _BY_ID[pid].icon) for pid in ranked)
    lines = []
    for pid in ranked:
        mark = "📌 " if pid in pins else ""
        used = usage.get(pid, 0)
        lines.append(f"{mark}{_BY_ID[pid].title}"
                     + (f" — opened ×{used}" if used else ""))
    body = ("your bar, learned from real usage (pins lead):\n"
            + "\n".join(lines)
            + "\npin one with: /pack act action-bar pin <pack id>")
    return HomeView(title=f"{pack.icon} {pack.title}", body=body,
                    actions=actions)


def _surprise_home(pack, session, facts):
    return HomeView(
        title=f"{pack.icon} {pack.title}",
        body=("one roll picks — and LABELS — a kind of surprise: "
              "another pack, a library item, or a mini-creation "
              "brief."),
        actions=(_verb_act(pack, "Roll a surprise", "roll"),
                 _packs_act()))


_HOME_BUILDERS = {
    "command-center": _command_center_home,
    "artifact-workbench": _workbench_home,
    "research-cockpit": _research_home,
    "music-deck": _music_home,
    "creative-director": _director_home,
    "sticker-atelier": _atelier_home,
    "learning-cards": _learning_home,
    "game-boards": _games_home,
    "book-studio": _books_home,
    "smart-digest": _digest_home,
    "memory-shelf": _shelf_home,
    "calendar-host": _calendar_home,
    "travel-board": _travel_home,
    "media-vault": _vault_home,
    "decision-room": _decision_home,
    "diagnostic-panel": _diagnostics_home,
    "adventure": _adventure_home,
    "celebration-engine": _celebration_home,
    "workflow-buttons": _workflows_home,
    "inline-creation": _inline_home,
    "companion-views": _companion_home,
    "cross-device-handoff": _handoff_home,
    "language-expression": _language_home,
    "action-bar": _action_bar_home,
    "surprise-me": _surprise_home,
}


def initial_session(pack_id: str) -> dict[str, Any]:
    """The session a stateful pack starts from. The runtime
    persists it at first open (and verbs seed from it), so a
    pack's state is durable and inspectable from its first
    card — never an implicit default living only in a builder.
    """
    if pack_id == "adventure":
        return {"node": ADVENTURE_START, "finished": False}
    if pack_id == "learning-cards":
        return {"index": 0, "revealed": False, "known": 0}
    if pack_id == "travel-board":
        return {"legs": []}
    if pack_id == "memory-shelf":
        return {"facts": []}
    if pack_id == "smart-digest":
        return {"index": 0}
    return {}


def build_home(pack: PackDef, *, session: dict | None = None,
               facts: dict | None = None) -> HomeView:
    """The pack's home view for one session + fact snapshot."""
    builder = _HOME_BUILDERS.get(pack.id) or (
        lambda p, s, f: _wrapped_home(p, s, f))
    return builder(pack, dict(session or {}), dict(facts or {}))


# ── native verbs ────────────────────────────────────────────────


def _v_travel(pack, verb, args, session, facts):
    legs = list(session.get("legs") or [])
    if verb == "add":
        label, _, detail = str(args or "").partition("|")
        label = label.strip()
        if not label:
            return VerbOutcome(
                "a leg needs a place: add <place> | <detail>",
                session)
        legs.append({"label": label, "detail": detail.strip()})
        session["legs"] = legs
        return VerbOutcome(
            f"added leg {len(legs)}: {label} — the board now "
            f"holds {len(legs)} leg(s).", session)
    if verb == "drop":
        try:
            idx = int(str(args).strip()) - 1
        except ValueError:
            return VerbOutcome("drop takes a leg number.", session)
        if not 0 <= idx < len(legs):
            return VerbOutcome(
                f"there is no leg {idx + 1} on the board.",
                session)
        gone = legs.pop(idx)
        session["legs"] = legs
        return VerbOutcome(
            f"dropped {gone.get('label', '')} — {len(legs)} "
            f"leg(s) remain.", session)
    if verb == "clear":
        session["legs"] = []
        return VerbOutcome("the board is cleared.", session)
    return None


def _v_decision(pack, verb, args, session, facts):
    if verb == "propose":
        parts = [p.strip() for p in str(args or "").split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 3:
            return VerbOutcome(
                "a proposal reads: propose <question> | <option> "
                "| <option>", session)
        session["proposal"] = {
            "question": parts[0], "options": parts[1:],
            "votes": {}, "closed": False}
        return VerbOutcome(
            f"proposal open: {parts[0]} — options: "
            + ", ".join(f"{i}. {o}" for i, o in
                        enumerate(parts[1:], 1))
            + ". Vote with: /pack act decision-room vote <n>",
            session)
    proposal = session.get("proposal")
    if verb == "vote":
        if not proposal:
            return VerbOutcome("no proposal is open in this "
                               "room.", session)
        try:
            idx = int(str(args).strip()) - 1
        except ValueError:
            return VerbOutcome("vote takes an option number.",
                               session)
        if not 0 <= idx < len(proposal["options"]):
            return VerbOutcome(
                f"option {idx + 1} doesn't exist here.", session)
        votes = dict(proposal.get("votes") or {})
        votes[str(facts.get("actor") or "someone")] = idx
        proposal["votes"] = votes
        session["proposal"] = proposal
        return VerbOutcome(
            f"vote recorded for "
            f"{proposal['options'][idx]}.", session)
    if verb == "close":
        if not proposal:
            return VerbOutcome("nothing is open to close.",
                               session)
        tally: dict[str, int] = {}
        for idx in (proposal.get("votes") or {}).values():
            label = proposal["options"][int(idx)]
            tally[label] = tally.get(label, 0) + 1
        winner = ""
        best = -1
        for opt in proposal["options"]:
            if tally.get(opt, 0) > best:
                best = tally.get(opt, 0)
                winner = opt
        record = {"question": proposal["question"],
                  "winner": winner, "tally": tally}
        session["record"] = record
        session["proposal"] = None
        tally_line = ", ".join(f"{k}: {v}" for k, v in
                               sorted(tally.items())) or "no votes"
        return VerbOutcome(
            f"decision recorded: {winner} wins "
            f"({tally_line}) — '{proposal['question']}' is "
            f"settled.", session)
    return None


def _v_adventure(pack, verb, args, session, facts):
    node_id = str(session.get("node") or ADVENTURE_START)
    node = ADVENTURE_NODES.get(node_id) or ADVENTURE_NODES[
        ADVENTURE_START]
    if verb == "restart":
        session["node"] = ADVENTURE_START
        session["finished"] = False
        return VerbOutcome(
            "the story restarts.\n"
            + ADVENTURE_NODES[ADVENTURE_START]["text"], session)
    if verb == "choose":
        if node["ending"]:
            return VerbOutcome(
                "this story has ended — restart to walk it "
                "again.", session)
        try:
            idx = int(str(args).strip()) - 1
        except ValueError:
            return VerbOutcome("choose takes a choice number.",
                               session)
        if not 0 <= idx < len(node["choices"]):
            return VerbOutcome(
                f"choice {idx + 1} isn't offered here.", session)
        nxt = node["choices"][idx][1]
        session["node"] = nxt
        target = ADVENTURE_NODES[nxt]
        session["finished"] = bool(target["ending"])
        return VerbOutcome(target["text"], session)
    return None


def _v_learning(pack, verb, args, session, facts):
    idx = int(session.get("index", 0)) % len(LEARNING_DECK)
    card = LEARNING_DECK[idx]
    if verb == "reveal":
        session["revealed"] = True
        return VerbOutcome(f"answer: {card['answer']}", session)
    if verb == "hint":
        return VerbOutcome(f"hint: {card['hint']}", session)
    if verb in ("known", "next"):
        if verb == "known":
            session["known"] = int(session.get("known", 0)) + 1
        session["index"] = (idx + 1) % len(LEARNING_DECK)
        session["revealed"] = False
        nxt = LEARNING_DECK[session["index"]]
        return VerbOutcome(
            f"card {session['index'] + 1} of {len(LEARNING_DECK)}"
            f" · known so far: {session.get('known', 0)}\n"
            f"{nxt['question']}", session)
    return None


def _v_workflows(pack, verb, args, session, facts):
    store = facts.get("store")
    if store is None:
        return VerbOutcome("the workflow store is unavailable "
                           "here.", session)
    if verb == "save":
        name, _, command = str(args or "").partition("=")
        name = name.strip().lower()
        command = " ".join(command.split()).strip()
        if not name or not command:
            return VerbOutcome(
                "a shortcut reads: save <name> = <command>",
                session)
        from .control import parse_control
        parsed = parse_control("/" + command)
        if parsed is None:
            return VerbOutcome(
                f"can't save that: /{command} isn't a runnable "
                f"command here.", session)
        if parsed.kind == "pack":
            return VerbOutcome(
                f"a workflow can't replay {command!r} — pack "
                f"commands would loop back into the pack "
                f"system.", session)
        store.set_workflow(name, command)
        return VerbOutcome(
            f"saved '{name}' → /{command}. Run it with: /pack "
            f"act workflow-buttons run {name}", session)
    if verb == "run":
        name = str(args or "").strip().lower()
        command = store.workflows().get(name)
        if not command:
            return VerbOutcome(
                f"no workflow named {name!r} — save one first.",
                session)
        return VerbOutcome(session=session, dispatch=command)
    if verb == "list":
        flows = store.workflows()
        if not flows:
            return VerbOutcome("no workflows saved yet.", session)
        return VerbOutcome(
            "workflows:\n" + "\n".join(
                f"• {n} → /{c}" for n, c in sorted(flows.items())),
            session)
    if verb == "drop":
        name = str(args or "").strip().lower()
        if store.drop_workflow(name):
            return VerbOutcome(f"dropped '{name}'.", session)
        return VerbOutcome(f"no workflow named {name!r}.", session)
    return None


def _v_shelf(pack, verb, args, session, facts):
    facts_list = list(session.get("facts") or [])
    if verb == "shelve":
        text, _, source = str(args or "").partition("|")
        text = text.strip()
        if not text:
            return VerbOutcome(
                "shelve reads: shelve <the fact> | <its source>",
                session)
        facts_list.append({
            "text": text,
            "source": source.strip() or "this chat",
            "shelved_in": str(facts.get("chat_key") or "")})
        session["facts"] = facts_list
        return VerbOutcome(
            f"shelved: {text} (source: "
            f"{facts_list[-1]['source']}) — {len(facts_list)} "
            f"fact(s) on your shelf.", session)
    if verb == "list":
        if not facts_list:
            return VerbOutcome("your shelf is empty.", session)
        return VerbOutcome(
            "on your shelf:\n" + "\n".join(
                f"{i}. {f.get('text', '')} "
                f"(source: {f.get('source', '')}, shelved in "
                f"{f.get('shelved_in', '')})"
                for i, f in enumerate(facts_list, 1)), session)
    if verb == "forget":
        try:
            idx = int(str(args).strip()) - 1
        except ValueError:
            return VerbOutcome("forget takes a shelf number.",
                               session)
        if not 0 <= idx < len(facts_list):
            return VerbOutcome(f"there is no fact {idx + 1} on "
                               f"the shelf.", session)
        gone = facts_list.pop(idx)
        session["facts"] = facts_list
        return VerbOutcome(
            f"forgot: {gone.get('text', '')}", session)
    return None


def _v_vault(pack, verb, args, session, facts):
    if verb == "find":
        query = str(args or "").strip().lower()
        if not query:
            return VerbOutcome("find takes a tag or kind.",
                               session)
        hits = []
        for item in facts.get("library_items") or []:
            tags = [str(t).lower() for t in item.get("tags") or []]
            hay = " ".join(tags + [str(item.get("kind", "")),
                                   str(item.get("path", ""))]).lower()
            if query in hay:
                hits.append(item)
        if not hits:
            return VerbOutcome(
                f"no saved media matches {query!r} yet — the "
                f"vault grows as media is saved, stolen or made.",
                session)
        lines = []
        for item in hits[:10]:
            tags = ", ".join(str(t) for t in item.get("tags") or [])
            lines.append(f"• {item.get('kind', 'media')} — {tags}"
                         f" — {item.get('path', '')}")
        return VerbOutcome(
            f"{len(hits)} item(s) match {query!r}:\n"
            + "\n".join(lines), session)
    return None


def _v_digest(pack, verb, args, session, facts):
    items = facts.get("digest_items") or []
    if not items:
        return VerbOutcome(
            "the digest has no items right now — /news pulls "
            "the latest.", session)
    idx = int(session.get("index", 0))
    idx = idx + 1 if verb == "next" else idx - 1
    idx = max(0, min(idx, len(items) - 1))
    session["index"] = idx
    return VerbOutcome(f"item {idx + 1} of {len(items)}\n"
                       f"{items[idx]}", session)


def _v_action_bar(pack, verb, args, session, facts):
    store = facts.get("store")
    target = str(args or "").strip().lower()
    if target not in _BY_ID:
        return VerbOutcome(
            f"{target!r} isn't a pack — pins name pack ids.",
            session)
    if verb == "pin":
        store.set_pinned(target, True)
        return VerbOutcome(
            f"pinned {_BY_ID[target].title} — it leads your "
            f"action bar now.", session)
    if verb == "unpin":
        store.set_pinned(target, False)
        return VerbOutcome(
            f"unpinned {_BY_ID[target].title}.", session)
    return None


_CREATION_BRIEFS = (
    "a gold heart", "a silver moon", "a red rocket",
    "a green leaf badge", "a blue wave",
)


def _v_surprise(pack, verb, args, session, facts):
    if verb != "roll":
        return None
    seed = facts.get("seed")
    rng = random.Random(seed) if seed is not None else random.Random()
    kinds = ["pack", "creation"]
    items = facts.get("library_items") or []
    if items:
        kinds.append("library")
    kind = rng.choice(kinds)
    if kind == "pack":
        others = [p for p in PACKS if p.id != pack.id]
        chosen = rng.choice(others)
        return VerbOutcome(
            f"🎲 surprise — a pack: {chosen.icon} {chosen.title}"
            f" — {chosen.tagline}. Open it: /pack open "
            f"{chosen.id}", session)
    if kind == "library":
        item = rng.choice(items)
        tags = ", ".join(str(t) for t in item.get("tags") or [])
        return VerbOutcome(
            f"🎲 surprise — a library item: "
            f"{item.get('kind', 'media')} ({tags}) — "
            f"{item.get('path', '')}. Browse the vault: /pack "
            f"open media-vault", session)
    brief = rng.choice(_CREATION_BRIEFS)
    return VerbOutcome(
        f"🎲 surprise — a mini-creation brief: '{brief}'. Make "
        f"it: /create {brief}", session)


def _v_companion(pack, verb, args, session, facts):
    if verb != "view":
        return None
    target_id = str(args or "").strip().lower() or "music-deck"
    if target_id not in _BY_ID:
        return VerbOutcome(f"{target_id!r} isn't a pack.",
                           session)
    out_dir = facts.get("out_dir")
    if not out_dir:
        return VerbOutcome("nowhere to render the page to here.",
                           session)
    target = _BY_ID[target_id]
    home = build_home(target, session={}, facts=facts)
    page = render_companion_html(target, home)
    path = Path(out_dir) / f"{target_id}.html"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - named, not hidden
        assert isinstance(exc, Exception)
        return VerbOutcome(f"the page didn't render: {exc}",
                           session)
    return VerbOutcome(
        f"companion view for {target.title} rendered: "
        f"{path.name} — a full designed page for the pack, "
        f"saved with its artifacts.",
        session, artifact=(str(path), "document"))


def _v_celebration(pack, verb, args, session, facts):
    if verb != "celebrate":
        return None
    occasion = str(args or "").strip() or "today"
    out_dir = facts.get("out_dir")
    if not out_dir:
        return VerbOutcome("nowhere to render the celebration "
                           "to here.", session)
    from .creation import author_studio_spec, create_procedural
    spec = author_studio_spec("gold star with sparkles",
                              form="sticker")
    result = create_procedural(spec, Path(out_dir),
                               db=facts.get("db"))
    if not result.ok or result.path is None:
        return VerbOutcome(
            f"the celebration for {occasion} didn't render: "
            f"{result.describe()}", session)
    return VerbOutcome(
        f"🎉 a celebration sticker for {occasion} — authored "
        f"as a studio description, rendered by my own engines, "
        f"inspected and verified ({result.path.name}).",
        session, artifact=(str(result.path), "sticker"))


def _v_language(pack, verb, args, session, facts):
    if verb != "say":
        return None
    from .expression import REACTION_GIFS
    emotion = str(args or "").strip().lower()
    entry = REACTION_GIFS.get(emotion)
    if entry is None:
        known = ", ".join(sorted(REACTION_GIFS)[:12])
        return VerbOutcome(
            f"I don't carry a {emotion!r} reaction — the "
            f"library's emotions include: {known}…", session)
    caption = str(entry.get("plain") or entry.get("obj") or "")
    emoji = str(entry.get("emoji") or "")
    return VerbOutcome(
        f"{emoji} {emotion} — the original caption: "
        f"“{caption}” — originals are always kept, never "
        f"silently replaced. Send it with /{emotion}.", session)


def _v_inline(pack, verb, args, session, facts):
    if verb != "results":
        return None
    articles = pack_inline_articles(str(args or ""))
    if not articles:
        return VerbOutcome("no pack matches that query.",
                           session)
    lines = [f"• {a['title']} — posts "
             f"{a['input_message_content']['message_text']}"
             for a in articles]
    return VerbOutcome(
        "inline results for that query:\n" + "\n".join(lines),
        session)


def _v_handoff(pack, verb, args, session, facts):
    if verb != "note":
        return None
    text = str(args or "").strip()
    if not text:
        current = str(session.get("note") or "")
        return VerbOutcome(
            f"your carried note: {current}" if current else
            "no note carried yet — note <text> leaves one.",
            session)
    session["note"] = text
    return VerbOutcome(
        "noted — it will be here on any chat or device you "
        "open this pack from.", session)


_VERB_HANDLERS = {
    "travel-board": _v_travel,
    "decision-room": _v_decision,
    "adventure": _v_adventure,
    "learning-cards": _v_learning,
    "workflow-buttons": _v_workflows,
    "memory-shelf": _v_shelf,
    "media-vault": _v_vault,
    "smart-digest": _v_digest,
    "action-bar": _v_action_bar,
    "surprise-me": _v_surprise,
    "companion-views": _v_companion,
    "celebration-engine": _v_celebration,
    "language-expression": _v_language,
    "inline-creation": _v_inline,
    "cross-device-handoff": _v_handoff,
}


def apply_verb(pack: PackDef, verb: str, args: str, *,
               session: dict | None = None,
               facts: dict | None = None) -> VerbOutcome:
    """Run one native verb as a pure state transition.

    Unknown verbs answer with the pack's real verb list — an
    honest usage line, never a silent no-op.
    """
    session = {**initial_session(pack.id), **dict(session or {})}
    facts = dict(facts or {})
    handler = _VERB_HANDLERS.get(pack.id)
    if handler is not None:
        outcome = handler(pack, str(verb or "").lower(),
                          str(args or ""), session, facts)
        if outcome is not None:
            return outcome
    verbs = ", ".join(pack.verbs) or "none"
    return VerbOutcome(
        f"the {pack.title} pack doesn't take "
        f"{verb!r} — its verbs are: {verbs}.",
        session)


# ── listing, inline articles, companion pages ───────────────────


def listing_lines(store: PackStore) -> list[str]:
    """The /pack listing: every pack, with its honest state."""
    lines = ["🧩 experience packs — installed:"]
    for pack in PACKS:
        state = "on" if store.is_enabled(pack.id) else "off"
        lines.append(f"{pack.icon} {pack.title} ({pack.id}) — "
                     f"{state} — {pack.tagline}")
    lines.append("open: /pack open <id> · switch: /pack on|off "
                 "<id> · act: /pack act <id> <verb> …")
    return lines


def pack_inline_articles(query: str) -> list[dict[str, Any]]:
    """Inline-query articles for pack entries (adapter shape,
    mirroring tgbot_buttons' article dicts)."""
    q = str(query or "").strip().lower()
    matches = [p for p in PACKS
               if not q or q in p.id or q in p.title.lower()]
    out: list[dict[str, Any]] = []
    for pack in matches[:10]:
        out.append({
            "type": "article",
            "id": f"pk_{pack.id}",
            "title": f"{pack.icon} {pack.title}",
            "description": pack.tagline,
            "input_message_content": {
                "message_text": f"/pack open {pack.id}"},
        })
    return out


def render_companion_html(pack: PackDef, home: HomeView) -> str:
    """A pack home as a standalone designed page (the
    companion-views pack's artifact)."""
    accent = html.escape(pack.traits.get("accent", "slate"))
    actions = "".join(
        f"<li>{html.escape(a.effective_fallback_label)} — "
        f"<code>/{html.escape(a.data or a.url)}</code></li>"
        for a in home.actions)
    body = "<br>".join(html.escape(line)
                       for line in home.body.split("\n"))
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head>"
        "<meta charset=\"utf-8\">"
        f"<title>{html.escape(pack.title)} — Devon pack</title>"
        "<style>body{font-family:system-ui,sans-serif;"
        "margin:2rem auto;max-width:40rem;padding:0 1rem}"
        f"h1{{color:{accent}}}</style></head><body>"
        f"<h1>{html.escape(pack.icon)} "
        f"{html.escape(pack.title)}</h1>"
        f"<p><em>{html.escape(pack.tagline)}</em></p>"
        f"<p>{body}</p>"
        f"<h2>Actions</h2><ul>{actions}</ul>"
        "<footer>Powered by peacethefirst</footer>"
        "</body></html>")
