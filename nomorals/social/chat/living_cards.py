"""Living cards (Living Chat P2, 2026-11-08).

Phase 1 gave chat surfaces an inventory (the surface manifest),
a rich action schema, and operation targeting. Phase 2 gives the
cards themselves a life:

* **Identity + revision.** A :class:`Card` is a durable object:
  a stable id, a family, the chat/owner/platform it belongs to,
  its content, its actions, and a revision that increments on
  every content change. The record lives in the runtime-created
  ``living_cards`` table (the Batch B pattern — a table the
  owning module creates at first use; ``storage/migrations.py``
  is never touched), so a card survives the process that posted
  it.
* **Edit in place, honestly.** A revision UPDATES the card's one
  live message: through the adapter's edit verb where the
  platform supports text edits (the Telegram adapters and both
  WhatsApp bridges — :mod:`.platform_limits` is the evidence),
  and by an honest superseding send where it does not (Discord,
  SMS, web, local today). The mechanism is recorded per platform
  in :data:`CARD_UPDATE_FORMS` and in the surface manifest — an
  update is never silently a second message on an edit surface,
  and never pretends to be an edit on a send-only one.
* **Revision fencing.** A card's buttons carry a compact stamp —
  ``cardtap <card_id> <revision> <index>`` — instead of the bare
  command. Resolving a tap (:func:`resolve_card_tap`) checks the
  DURABLE card: the tap executes only at the card's current
  revision; a tap cut at a superseded revision answers STALE
  with the current card state ("this card has moved on — here's
  the current one") and never yields the stale command. The
  fencing reads the durable record, so the first tap after a
  restart works exactly like the last tap before it.
* **Renderers.** One renderer per card family in
  :data:`RENDERERS`, producing every platform form from the same
  card model: Telegram inline keyboards, WhatsApp interactive
  forms, and the numbered fallback (a peer path — the same
  commands, registered as an operation in the sender's slot).
  Styling rides the existing typography theme system; button
  fields ride only what :data:`.action_schema.FIELD_SUPPORT`
  records as honored — declared style / custom-emoji icons are
  named dropped at resolution, never emitted into a pair.
* **Journeys.** The startup readiness card (doc 4: real facts,
  ready / degraded / recovery-required, evidence on every row)
  and the restart journey card (preflight → swap → restart →
  confirmed) are families here. The restart card's final state
  can only be written by :func:`finalize_restart_card` — the
  new process, reading the restart-truth receipt
  (``restart_confirmed`` / ``restart_unconfirmed``) — never by
  the dying process's side of the journey.

This module never raises on the store/host paths: a card failure
degrades to the plain send path, which is always available.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ...core.progress_events import render_progress_text
from .action_schema import Action, resolve_action
from .platform_limits import for_platform
from .typography import Card as ThemeCard
from .typography import card_text

_log = logging.getLogger(__name__)

__all__ = [
    "FAMILY_GENERIC", "FAMILY_PROGRESS", "FAMILY_READINESS",
    "FAMILY_RESTART", "FAMILY_CREATION", "FAMILY_GAME",
    "CARD_FAMILIES",
    "CARD_UPDATE_FORMS",
    "Card", "CardStore", "CardHost", "RenderedCard", "RENDERERS",
    "render_card", "update_form",
    "TapStatus", "TapOutcome", "resolve_card_tap",
    "STALE_CARD_MESSAGE",
    "progress_card", "apply_progress_snapshot",
    "readiness_verdict", "build_readiness_card",
    "RESTART_PHASES", "build_restart_card", "advance_restart_card",
    "finalize_restart_card", "finalize_plain_restart_card",
    "restart_phase_lines",
]

FAMILY_GENERIC = "generic"
FAMILY_PROGRESS = "progress"
FAMILY_READINESS = "readiness"
FAMILY_RESTART = "restart"
FAMILY_CREATION = "creation"
#: The live game table (games hardening, doc 8 P4): the games
#: organ (nomorals/games/table.py) owns the card's content and
#: registers its rich renderer over the default below.
FAMILY_GAME = "game"

#: Card family → typography surface class (the theme system owns
#: presets and stamps; families pick a class, never a look).
CARD_FAMILIES: dict[str, str] = {
    FAMILY_GENERIC: "chat",
    FAMILY_PROGRESS: "chat",
    FAMILY_READINESS: "briefing",
    FAMILY_RESTART: "announcement",
    FAMILY_CREATION: "chat",
    FAMILY_GAME: "game",
}

#: How a card update reaches each platform: "edit" (the recorded
#: message is edited in place) or "supersede" (a fresh send
#: replaces it and the record moves). Derived from — and pinned
#: by test against — platform_limits' text-edit evidence.
CARD_UPDATE_FORMS: dict[str, str] = {}


def update_form(platform: str) -> str:
    """The update mechanism for one platform: edit | supersede."""
    plat = str(platform or "")
    form = CARD_UPDATE_FORMS.get(plat)
    if form is None:
        limits = for_platform(plat)
        form = "edit" if limits is not None \
            and limits.supports_text_edit else "supersede"
        CARD_UPDATE_FORMS[plat] = form
    return form


# ── the card model ──────────────────────────────────────────────


@dataclass
class Card:
    """One living card: identity, content, actions, revision."""

    card_id: str
    family: str
    chat_key: str = ""
    owner_key: str = ""
    platform: str = ""
    chat_id: str = ""
    title: str = ""
    body: str = ""
    rows: tuple[tuple[str, str], ...] = ()
    status: str = ""
    actions: tuple[Action, ...] = ()
    revision: int = 1
    message_id: str = ""
    delivered: bool = False
    terminal: bool = False
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id, "family": self.family,
            "chat_key": self.chat_key, "owner_key": self.owner_key,
            "platform": self.platform, "chat_id": self.chat_id,
            "title": self.title, "body": self.body,
            "rows": [list(r) for r in self.rows],
            "status": self.status,
            "actions": [{
                "label": a.label, "intent": a.intent, "data": a.data,
                "url": a.url, "emoji": a.emoji, "style": a.style,
                "icon_custom_emoji_id": a.icon_custom_emoji_id,
                "fallback_label": a.fallback_label,
                "target": a.target, "operation_id": a.operation_id,
                "revision": a.revision, "id": a.id,
            } for a in self.actions],
            "revision": self.revision,
            "message_id": self.message_id,
            "delivered": self.delivered, "terminal": self.terminal,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Card:
        actions = []
        for raw in data.get("actions") or []:
            try:
                actions.append(Action(
                    label=str(raw.get("label", "")),
                    intent=str(raw.get("intent", "callback")),
                    data=str(raw.get("data", "")),
                    url=str(raw.get("url", "")),
                    emoji=str(raw.get("emoji", "")),
                    style=str(raw.get("style", "")),
                    icon_custom_emoji_id=str(
                        raw.get("icon_custom_emoji_id", "")),
                    fallback_label=str(raw.get("fallback_label", "")),
                    target=str(raw.get("target", "")),
                    operation_id=str(raw.get("operation_id", "")),
                    revision=int(raw.get("revision", 0) or 0),
                    id=str(raw.get("id", ""))))
            except (TypeError, ValueError):
                continue  # a card never dies for one bad action row
        return cls(
            card_id=str(data.get("card_id", "")),
            family=str(data.get("family", FAMILY_GENERIC)),
            chat_key=str(data.get("chat_key", "")),
            owner_key=str(data.get("owner_key", "")),
            platform=str(data.get("platform", "")),
            chat_id=str(data.get("chat_id", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            rows=tuple((str(k), str(v))
                       for k, v in (data.get("rows") or [])),
            status=str(data.get("status", "")),
            actions=tuple(actions),
            revision=int(data.get("revision", 1) or 1),
            message_id=str(data.get("message_id", "")),
            delivered=bool(data.get("delivered", False)),
            terminal=bool(data.get("terminal", False)),
            state=dict(data.get("state") or {}),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0))


def new_card_id() -> str:
    """A card identity in the operation store's id shape."""
    return secrets.token_hex(4)


# ── the durable store ───────────────────────────────────────────

_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS living_cards ("
    "card_id TEXT PRIMARY KEY, family TEXT NOT NULL DEFAULT '', "
    "chat_key TEXT NOT NULL DEFAULT '', "
    "owner_key TEXT NOT NULL DEFAULT '', "
    "revision INTEGER NOT NULL DEFAULT 1, "
    "terminal INTEGER NOT NULL DEFAULT 0, "
    "payload TEXT NOT NULL DEFAULT '{}', "
    "updated_at REAL NOT NULL DEFAULT 0)")


class CardStore:
    """Durable card records (SQLite via the context db), with an
    in-memory fallback for db-less contexts (CLI, tests).

    ``save(card, expect_revision=N)`` is a compare-and-swap: the
    write lands only when the stored revision is N, so two
    writers can never silently overwrite each other's revision —
    the loser learns its copy was stale.
    """

    def __init__(self, db: Any = None) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._mem: dict[str, Card] = {}
        self._table_ready = False
        if db is not None:
            try:
                db.execute(_TABLE_SQL)
                self._table_ready = True
            except Exception as exc:  # noqa: BLE001 - degrade to memory
                _log.debug("living_cards table unavailable: %s", exc)
                self._db = None

    # ── writes ──────────────────────────────────────────────────

    def save(self, card: Card, *,
             expect_revision: int | None = None) -> bool:
        """Persist ``card``. With ``expect_revision``, the write
        lands only if the stored revision matches (a fresh card
        expects None stored). Returns whether the write landed."""
        if not card.card_id:
            return False
        card.updated_at = time.time()
        if not card.created_at:
            card.created_at = card.updated_at
        with self._lock:
            current = self._read(card.card_id)
            if expect_revision is not None:
                stored = current.revision if current is not None \
                    else None
                if stored != expect_revision:
                    return False
            if self._db is not None and self._table_ready:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO living_cards "
                        "(card_id, family, chat_key, owner_key, "
                        "revision, terminal, payload, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (card.card_id, card.family, card.chat_key,
                         card.owner_key, card.revision,
                         1 if card.terminal else 0,
                         json.dumps(card.to_dict()), card.updated_at))
                except Exception as exc:  # noqa: BLE001
                    _log.debug("living card save failed: %s", exc)
                    return False
            else:
                self._mem[card.card_id] = Card.from_dict(card.to_dict())
            return True

    # ── reads ───────────────────────────────────────────────────

    def _read(self, card_id: str) -> Card | None:
        if self._db is not None and self._table_ready:
            try:
                row = self._db.query_one(
                    "SELECT payload FROM living_cards "
                    "WHERE card_id = ?", (card_id,))
            except Exception as exc:  # noqa: BLE001
                _log.debug("living card read failed: %s", exc)
                return None
            if not row:
                return None
            try:
                return Card.from_dict(json.loads(row["payload"]))
            except (TypeError, ValueError):
                return None
        card = self._mem.get(card_id)
        return Card.from_dict(card.to_dict()) if card else None

    def get(self, card_id: str) -> Card | None:
        with self._lock:
            return self._read(str(card_id or ""))

    def _query(self, where: str, params: tuple,
               limit: int) -> list[Card]:
        if self._db is not None and self._table_ready:
            try:
                rows = self._db.query(
                    "SELECT payload FROM living_cards WHERE "
                    + where + " ORDER BY updated_at DESC LIMIT ?",
                    (*params, int(limit)))
            except Exception as exc:  # noqa: BLE001
                _log.debug("living card query failed: %s", exc)
                return []
            out = []
            for row in rows or []:
                try:
                    out.append(Card.from_dict(
                        json.loads(row["payload"])))
                except (TypeError, ValueError):
                    continue
            return out
        # The in-memory path filters in Python (same predicates
        # the SQL arms express).
        out = []
        for card in sorted(self._mem.values(),
                           key=lambda c: -c.updated_at):
            if where.startswith("chat_key = ? AND terminal = 0"):
                if card.chat_key != params[0] or card.terminal:
                    continue
            elif where.startswith("family = ?"):
                if card.family != params[0]:
                    continue
                if len(params) > 1 and card.chat_key != params[1]:
                    continue
            out.append(Card.from_dict(card.to_dict()))
            if len(out) >= limit:
                break
        return out

    def live_for_chat(self, chat_key: str,
                      limit: int = 50) -> list[Card]:
        """The chat's non-terminal cards, newest first."""
        with self._lock:
            return self._query("chat_key = ? AND terminal = 0",
                               (str(chat_key or ""),), limit)

    def latest(self, family: str, chat_key: str = "") -> Card | None:
        """The newest card of a family (optionally in one chat),
        terminal or not — the restart journey's finalizer reads
        its card this way after the process that opened it died."""
        with self._lock:
            if chat_key:
                rows = self._query("family = ? AND chat_key = ?",
                                   (str(family), str(chat_key)), 1)
            else:
                rows = self._query("family = ?", (str(family),), 1)
            return rows[0] if rows else None


# ── rendering ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RenderedCard:
    """One card rendered for one platform.

    ``form``: inline (Telegram keyboard) / interactive (WhatsApp
    quick-reply or list) / numbered (the numbered fallback —
    ``keyboard`` is empty and the actions ride as numbered lines
    in ``text``) / text (no actions to render).
    """

    text: str
    keyboard: list[list[tuple[str, str]]] = field(
        default_factory=list)
    form: str = "text"


def _theme_text(card: Card, *, phase_lines: list[str] | None = None,
                ) -> str:
    """The card's text form under the typography theme system."""
    theme = ThemeCard(CARD_FAMILIES.get(card.family, "chat"),
                      title=card.title)
    if card.status:
        theme.line(f"status: {card.status}")
    body = phase_lines if phase_lines is not None else (
        card.body.split("\n") if card.body else [])
    for line in body:
        if str(line).strip():
            theme.line(str(line))
    if card.rows:
        theme.stats(list(card.rows))
    return card_text(theme)


def _render_progress(card: Card) -> str:
    snap = card.state.get("snapshot")
    if isinstance(snap, dict) and snap:
        return render_progress_text(snap, title=card.title)
    return _theme_text(card)


def _render_restart(card: Card) -> str:
    return _theme_text(card, phase_lines=restart_phase_lines(card))


#: One renderer per card family — the manifest records these.
#: The game family's rich renderer registers itself from
#: nomorals/games/table.py when the games organ loads; this
#: default keeps the family renderable before that.
RENDERERS: dict[str, Callable[[Card], str]] = {
    FAMILY_GENERIC: lambda card: _theme_text(card),
    FAMILY_PROGRESS: _render_progress,
    FAMILY_READINESS: lambda card: _theme_text(card),
    FAMILY_RESTART: _render_restart,
    FAMILY_CREATION: lambda card: _theme_text(card),
    FAMILY_GAME: lambda card: _theme_text(card),
}


def _pin_line(card: Card) -> str:
    """The rendered attribution of an owner style pin in force
    for this card's family (Living Chat P3): a card under a pin
    NAMES it, so a pinned look is never mistaken for a default.
    "" when no pin applies. Never raises."""
    try:
        from ...core.style_pins import effective_pin, get_pin_store
        pin = effective_pin(get_pin_store(), card.family)
    except Exception as exc:  # noqa: BLE001 - pins never break a card
        _log.debug("card pin lookup failed: %s", exc)
        return ""
    if pin is None:
        return ""
    return f"style pin: {pin.label} ({pin.describe()})"


def render_card(card: Card, platform: str) -> RenderedCard:
    """Render one card for one platform: the family renderer's
    text plus the platform form of its actions.

    Button surfaces get fenced callback data
    (``cardtap <id> <revision> <index>``) — the command itself
    stays on the durable card and is only released by
    :func:`resolve_card_tap` at the current revision. Buttonless
    surfaces get the numbered fallback: the actions ride the
    text as numbered lines and the host registers them as an
    operation in the sender's slot.
    """
    plat = str(platform or card.platform or "")
    renderer = RENDERERS.get(card.family, RENDERERS[FAMILY_GENERIC])
    text = renderer(card)
    pin_line = _pin_line(card)
    if pin_line:
        text = f"{text}\n{pin_line}" if text else pin_line
    if not card.actions:
        return RenderedCard(text=text, form="text")
    probe = resolve_action(card.actions[0], plat)
    if probe.form in ("inline", "interactive"):
        rows: list[list[tuple[str, str]]] = []
        row: list[tuple[str, str]] = []
        for index, action in enumerate(card.actions, 1):
            resolved = resolve_action(action, plat)
            if resolved.form not in ("inline", "interactive"):
                # A text-line action (URL/copy on WhatsApp) rides
                # the text, not the keyboard.
                label = action.effective_fallback_label
                value = action.url if action.intent == "url" \
                    else action.data
                text = f"{text}\n{label}: {value}"
                continue
            label = action.label
            if action.emoji and action.emoji not in label:
                label = f"{action.emoji} {label}"
            row.append((label,
                        f"cardtap {card.card_id} {card.revision} "
                        f"{index}"))
            if len(row) == 2:  # the house grid: two per row
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return RenderedCard(text=text, keyboard=rows,
                            form=probe.form)
    # The numbered fallback is a peer path, not a degraded one.
    lines = [text] if text else []
    for index, action in enumerate(card.actions, 1):
        lines.append(
            f"{index} — {action.effective_fallback_label}")
    lines.append("reply with the number to run it")
    return RenderedCard(text="\n".join(lines), form="numbered")


# ── revision fencing ────────────────────────────────────────────

#: The honest answer to a tap on a superseded card.
STALE_CARD_MESSAGE = (
    "that card has moved on — here's the current one:")


class TapStatus(Enum):
    """The outcome of resolving a card tap (mirrors PickStatus)."""

    OK = "ok"                #: current revision; command released
    STALE = "stale"          #: the card has been revised since
    MISSING = "missing"      #: no such card (here, for this owner)
    OUT_OF_RANGE = "out_of_range"  #: live card, index outside it


@dataclass
class TapOutcome:
    status: TapStatus
    command: str = ""
    card: Card | None = None


def resolve_card_tap(store: CardStore, card_id: str, revision: int,
                     index: int, *, chat_key: str = "",
                     owner_key: str = "") -> TapOutcome:
    """Resolve one fenced card tap against the DURABLE card.

    The tap executes only when it names the card's current
    revision, in the card's own chat, for the card's owner. A
    superseded revision answers STALE with the current card
    attached (the caller shows its current state); the stale
    command is never released. An out-of-range index leaves the
    card untouched — a typo must not burn it.
    """
    card = store.get(str(card_id or ""))
    if card is None:
        return TapOutcome(TapStatus.MISSING)
    if chat_key and card.chat_key and card.chat_key != chat_key:
        return TapOutcome(TapStatus.MISSING)
    if card.owner_key and owner_key != card.owner_key:
        return TapOutcome(TapStatus.MISSING)
    if int(revision) != card.revision:
        return TapOutcome(TapStatus.STALE, card=card)
    n = int(index)
    if n < 1 or n > len(card.actions):
        return TapOutcome(TapStatus.OUT_OF_RANGE, card=card)
    action = card.actions[n - 1]
    command = action.url if action.intent == "url" else action.data
    return TapOutcome(TapStatus.OK, command=command, card=card)


# ── the host: publish / revise / deliver ────────────────────────


class CardHost:
    """Sends and updates cards through injected transport seams.

    ``send(platform, chat, text, buttons) -> (ok, message_id)``
    and ``edit(chat, message_id, text, buttons) -> bool`` are the
    caller's real gateway/adapter verbs (the runtime wires its
    gateway send and the live-session edit verb). The host owns
    the revision bookkeeping and the edit-vs-supersede decision;
    it never raises.

    ``register_picks(chat_key, commands, owner_key, operation_id)``
    (Living Chat P3, closing P2 remainder #1): whenever a
    delivery lands in the NUMBERED form, the card's actions are
    registered into the operation store AT SEND TIME, under the
    card's own identity — so a bare-number reply resolves
    against THIS card's operation, whichever producer built the
    card, not only producers that register picks themselves.
    A failed send registers nothing: numbers nobody saw must
    not execute.
    """

    def __init__(self, store: CardStore, *,
                 send: Callable[..., tuple[bool, str]],
                 edit: Callable[..., bool],
                 register_picks: Callable[..., Any] | None = None,
                 ) -> None:
        self.store = store
        self._send = send
        self._edit = edit
        self._register_picks = register_picks

    def _register_numbered(self, card: Card,
                           rendered: RenderedCard) -> None:
        """Register a numbered delivery's actions under the
        card's identity. Guarded: registration is a targeting
        aid, never a reason to fail a delivered card."""
        if self._register_picks is None \
                or rendered.form != "numbered" \
                or not card.actions or not card.chat_key:
            return
        commands = [
            a.url if a.intent == "url" else a.data
            for a in card.actions]
        commands = [c for c in commands if str(c).strip()]
        if not commands:
            return
        try:
            self._register_picks(card.chat_key, commands,
                                 card.owner_key, card.card_id)
        except Exception as exc:  # noqa: BLE001 - degrade, never die
            _log.debug("card pick registration failed for %s: %s",
                       card.card_id, exc)

    def publish(self, card: Card, chat: Any) -> Card:
        """First delivery: send the render, record the message id
        and the delivery verdict on the durable card. An
        undeliverable card is still persisted — the retained
        record the startup journey promises."""
        rendered = render_card(card, card.platform)
        try:
            ok, message_id = self._send(
                card.platform, chat, rendered.text,
                rendered.keyboard or None)
        except Exception as exc:  # noqa: BLE001 - degrade, never die
            _log.debug("card publish failed for %s: %s",
                       card.card_id, exc)
            ok, message_id = False, ""
        if ok:
            card.delivered = True
            card.message_id = str(message_id or "")
            self._register_numbered(card, rendered)
        self.store.save(card)
        return card

    def revise(self, card: Card, chat: Any, **changes: Any) -> Card:
        """Apply content changes as a new revision and deliver
        the update: an in-place edit on edit surfaces (falling
        back to a fresh send when the edit is refused), a
        superseding send on send-only surfaces."""
        fresh = self.store.get(card.card_id) or card
        old_revision = fresh.revision
        for key, value in changes.items():
            if key in ("title", "body", "rows", "status", "actions",
                       "state", "terminal"):
                setattr(fresh, key, value)
        fresh.revision = old_revision + 1
        if not self.store.save(fresh, expect_revision=old_revision):
            # Lost a race: re-read and re-apply once rather than
            # overwriting the winning revision.
            winner = self.store.get(card.card_id)
            if winner is not None:
                for key, value in changes.items():
                    if key in ("title", "body", "rows", "status",
                               "actions", "state", "terminal"):
                        setattr(winner, key, value)
                winner.revision += 1
                self.store.save(winner,
                                expect_revision=winner.revision - 1)
                fresh = winner
        self._deliver_now(fresh, chat)
        return fresh

    def deliver(self, card: Card, chat: Any) -> Card:
        """(Re)deliver the card's current render without a
        revision change — the pending-startup delivery and the
        stale-tap re-render ride this."""
        self._deliver_now(card, chat)
        return card

    def _deliver_now(self, card: Card, chat: Any) -> None:
        rendered = render_card(card, card.platform)
        delivered = False
        if update_form(card.platform) == "edit" and card.message_id:
            try:
                delivered = bool(self._edit(
                    chat, card.message_id, rendered.text,
                    rendered.keyboard or None))
            except Exception as exc:  # noqa: BLE001
                _log.debug("card edit failed for %s: %s",
                           card.card_id, exc)
                delivered = False
        if not delivered:
            try:
                ok, message_id = self._send(
                    card.platform, chat, rendered.text,
                    rendered.keyboard or None)
            except Exception as exc:  # noqa: BLE001
                _log.debug("card send failed for %s: %s",
                           card.card_id, exc)
                ok, message_id = False, ""
            if ok:
                card.delivered = True
                if message_id:
                    card.message_id = str(message_id)
                delivered = True
        if delivered:
            self._register_numbered(card, rendered)
            self.store.save(card)


# ── the progress tenant ─────────────────────────────────────────


def progress_card(tracker: Any, *, chat_key: str, owner_key: str = "",
                  platform: str = "", chat_id: str = "") -> Card:
    """Open the living card for one measured operation: the
    progress stream (core.progress_events) is the tenant, the
    card is its durable, revisable surface record."""
    card = Card(
        card_id=str(tracker.op_id), family=FAMILY_PROGRESS,
        chat_key=str(chat_key or ""), owner_key=str(owner_key or ""),
        platform=str(platform or ""), chat_id=str(chat_id or ""),
        title=str(getattr(tracker, "title", "") or ""))
    apply_progress_snapshot(card, tracker.snapshot())
    return card


def apply_progress_snapshot(card: Card, snapshot: dict[str, Any],
                            ) -> Card:
    """Fold one measured snapshot into the card: the render
    reads it verbatim — no percentage or state is ever invented
    at the card layer that the stream did not measure."""
    card.state = {**card.state, "snapshot": dict(snapshot or {})}
    terminal = snapshot.get("terminal") if snapshot else None
    if terminal:
        card.terminal = True
        card.status = str(terminal)
    elif snapshot:
        card.status = str(snapshot.get("stage") or card.status)
    return card


# ── the startup journey (doc 4) ─────────────────────────────────

#: Update-recovery actions that mean Devon is booting on a state
#: that needs the owner's eyes — mirrors the failure set the
#: runtime's own recovery report treats as failed.
_RECOVERY_REQUIRED_ACTIONS = frozenset({
    "rollback_failed", "pending_restore_failed",
    "tree_restore_failed", "restart_unconfirmed"})


def _agency_degraded(agency: Any) -> tuple[bool, str]:
    """(degraded, evidence) from a composition readiness map:
    values read "ready…", "unavailable: …" or "delegated: …" —
    only an unavailable system degrades the verdict; delegated
    systems are named design, not failures."""
    if not isinstance(agency, dict) or not agency:
        return False, ""
    down = sorted(name for name, state in agency.items()
                  if str(state).startswith("unavailable"))
    ready = sum(1 for state in agency.values()
                if str(state).startswith("ready"))
    evidence = f"{ready}/{len(agency)} systems ready"
    if down:
        evidence += f" — down: {', '.join(down)}"
    return bool(down), evidence


def readiness_verdict(facts: dict[str, Any]) -> str:
    """ready | degraded | recovery-required, from real facts.

    recovery-required: an update recovery failed or a restart
    was never confirmed — Devon may be running a state pairing
    nobody vouched for. degraded: something the owner uses is
    down (a platform, the brain, the scheduler, storage, an
    agency system). Everything else is ready — including a
    recovery that ACTED and succeeded, which is information,
    not a verdict against this boot.
    """
    recovery = facts.get("recovery")
    if isinstance(recovery, dict) and str(
            recovery.get("action", "")) in _RECOVERY_REQUIRED_ACTIONS:
        return "recovery-required"
    if facts.get("storage") is False:
        return "degraded"
    for _name, state, _detail in facts.get("platforms") or []:
        if str(state) != "up":
            return "degraded"
    brain = facts.get("brain")
    if brain and str(brain[0]) not in ("ready", "up", "ok"):
        return "degraded"
    scheduler = facts.get("scheduler")
    if scheduler and str(scheduler[0]) not in (
            "running", "ready", "up"):
        return "degraded"
    degraded, _evidence = _agency_degraded(facts.get("agency"))
    if degraded:
        return "degraded"
    return "ready"


def build_readiness_card(facts: dict[str, Any], *,
                         chat_key: str = "", owner_key: str = "",
                         platform: str = "", chat_id: str = "",
                         card_id: str = "") -> Card:
    """The startup journey's one surface: what is up, what is
    degraded, what is down — every row carrying its evidence.
    The composition readiness map is one tenant (the agency row),
    never the whole report."""
    verdict = readiness_verdict(facts)
    rows: list[tuple[str, str]] = []
    rows.append(("storage", "ready" if facts.get("storage")
                 else "unavailable"))
    recovery = facts.get("recovery")
    if isinstance(recovery, dict) and (
            recovery.get("recovered") or recovery.get("action")):
        rows.append(("update recovery",
                     str(recovery.get("action") or "acted")))
    for name, state, detail in facts.get("platforms") or []:
        rows.append((f"platform {name}",
                     str(state) + (f" — {detail}" if detail else "")))
    brain = facts.get("brain")
    if brain:
        rows.append(("brain", str(brain[0])
                     + (f" — {brain[1]}" if len(brain) > 1 and brain[1]
                        else "")))
    scheduler = facts.get("scheduler")
    if scheduler:
        rows.append(("scheduler", str(scheduler[0])
                     + (f" — {scheduler[1]}"
                        if len(scheduler) > 1 and scheduler[1]
                        else "")))
    agency = facts.get("agency")
    if isinstance(agency, dict) and agency:
        _down, evidence = _agency_degraded(agency)
        rows.append(("agency", evidence))
    missions = facts.get("missions")
    if isinstance(missions, dict) and (
            missions.get("resumed") or missions.get("failed")):
        rows.append(("missions",
                     f"resumed {len(missions.get('resumed') or [])}, "
                     f"failed {len(missions.get('failed') or [])}"))
    headline = {
        "ready": "everything Devon needs is up",
        "degraded": "up, with something degraded — the rows name it",
        "recovery-required":
            "up, but an update recovery needs your eyes — "
            "the rows name it",
    }[verdict]
    return Card(
        card_id=card_id or new_card_id(), family=FAMILY_READINESS,
        chat_key=str(chat_key or ""), owner_key=str(owner_key or ""),
        platform=str(platform or ""), chat_id=str(chat_id or ""),
        title="Devon is up", body=headline, rows=tuple(rows),
        status=verdict, state={"verdict": verdict})


# ── the restart journey (doc 4) ─────────────────────────────────

#: The journey's phases, in order. "confirmed" is writable ONLY
#: by finalize_restart_card (the new process); the dying side's
#: advance stops at "restart".
RESTART_PHASES = ("preflight", "swap", "restart", "confirmed")

_PHASE_MARKS = {"done": "✓", "current": "▸", "pending": "·",
                "attention": "⚠"}


def _restart_state(card: Card) -> dict[str, Any]:
    state = card.state.get("restart")
    if not isinstance(state, dict):
        state = {"phases": {name: "pending" for name in RESTART_PHASES},
                 "kind": "", "to_sha": "", "detail": ""}
        state["phases"]["preflight"] = "current"
        card.state = {**card.state, "restart": state}
    return state


def restart_phase_lines(card: Card) -> list[str]:
    """The journey as lines: each phase with its honest mark."""
    state = _restart_state(card)
    phases = state.get("phases") or {}
    lines = []
    for name in RESTART_PHASES:
        mark_state = str(phases.get(name, "pending"))
        mark = _PHASE_MARKS.get(mark_state, "·")
        suffix = "" if mark_state in ("done", "pending") else \
            f" — {mark_state}"
        lines.append(f"{mark} {name}{suffix}")
    detail = str(state.get("detail") or "")
    if detail:
        lines.append(detail)
    return lines


def build_restart_card(*, chat_key: str = "", owner_key: str = "",
                       platform: str = "", chat_id: str = "",
                       kind: str = "update",
                       card_id: str = "") -> Card:
    """Open the restart journey card (the dying process's side):
    preflight is current, nothing is claimed beyond it."""
    card = Card(
        card_id=card_id or new_card_id(), family=FAMILY_RESTART,
        chat_key=str(chat_key or ""), owner_key=str(owner_key or ""),
        platform=str(platform or ""), chat_id=str(chat_id or ""),
        title="Devon restart",
        body=("a restart is a journey, not a reply: each phase "
              "lands on this card as it happens, and the final "
              "state is written by the process that comes up — "
              "never declared by the one going down."),
        status="working")
    state = _restart_state(card)
    state["kind"] = str(kind or "update")
    card.state = {**card.state, "restart": state}
    return card


def advance_restart_card(card: Card, report: Any = None, *,
                         phase: str = "",
                         detail: str = "") -> Card:
    """Advance the journey on the DYING process's side.

    With an update report: preflight is done; the swap is done
    when the report says the update landed (attention when it
    did not); the journey then waits at restart. Naming
    ``phase="restart"`` records the handoff itself. This side
    NEVER marks confirmed and NEVER closes the card — that
    verdict belongs to the process that comes up.
    """
    state = _restart_state(card)
    phases = dict(state.get("phases") or {})
    if report is not None:
        phases["preflight"] = "done"
        if getattr(report, "ok", False):
            phases["swap"] = "done"
        else:
            phases["swap"] = "attention"
            card.status = "attention"
        to_sha = str(getattr(report, "to_sha", "") or "")
        if to_sha:
            state["to_sha"] = to_sha
        # The restart phase is NOT advanced here: a pull's
        # journey has no restart in it. The handoff is recorded
        # only when a restart is actually requested, via
        # phase="restart" below.
    if phase == "restart":
        phases["restart"] = "current"
        state["detail"] = (
            "handoff recorded — the old process is going down; "
            "the new process confirms what it came up on.")
    if detail:
        state["detail"] = str(detail)
    state["phases"] = phases
    card.state = {**card.state, "restart": state}
    return card


def finalize_plain_restart_card(card: Card, *, boot_id: str = "",
                                ) -> Card:
    """Close a PLAIN restart's journey from the new process.

    A plain /restart claims no version change, so there is no
    restart-truth receipt for it; the only fact the new process
    can witness is its own existence. The confirmation written
    here claims exactly that — a new process came up — and
    nothing about versions.
    """
    state = _restart_state(card)
    phases = dict(state.get("phases") or {})
    phases["preflight"] = "done"
    phases["swap"] = "done"
    phases["restart"] = "done"
    phases["confirmed"] = "done"
    state["phases"] = phases
    state["detail"] = ""
    card.state = {**card.state, "restart": state}
    card.status = "confirmed"
    card.body = (
        "restart confirmed by the process that came up"
        + (f" (boot {str(boot_id)[:12]})" if boot_id else "")
        + ": a plain restart claimed no version change, and "
          "this process — the one now running — is its witness.")
    card.terminal = True
    return card


def finalize_restart_card(card: Card, receipt: dict[str, Any],
                          ) -> Card:
    """Close the journey from the NEW process, on the
    restart-truth receipt's evidence alone.

    ``restart_confirmed`` (the new process verified it is
    running the landed version) completes every phase.
    ``restart_unconfirmed`` leaves the confirmed phase at
    attention and says so in plain words — the card never
    upgrades an unverified restart into a success.
    """
    state = _restart_state(card)
    phases = dict(state.get("phases") or {})
    action = str(receipt.get("action", ""))
    running = str(receipt.get("running_sha") or "?")[:12]
    landed = str(receipt.get("to_sha")
                 or state.get("to_sha") or "?")[:12]
    phases["preflight"] = "done"
    phases["swap"] = "done"
    phases["restart"] = "done"
    if action == "restart_confirmed":
        phases["confirmed"] = "done"
        card.status = "confirmed"
        card.body = (
            f"restart confirmed by the process that came up: "
            f"this process is running {running}, the version the "
            f"update landed. The confirmation is the new "
            f"process's own witness, not the old one's promise.")
    elif action == "restart_unconfirmed":
        phases["confirmed"] = "attention"
        card.status = "attention"
        card.body = (
            f"the restart is NOT confirmed: the process that "
            f"came up is running {running}, not the version the "
            f"update landed ({landed}). The new code may not be "
            f"running — check the install before trusting it.")
    else:
        phases["confirmed"] = "attention"
        card.status = "attention"
        card.body = (
            f"the restart journey closed on an unexpected "
            f"receipt ({action or 'none'}) — treat the restart "
            f"as unconfirmed and check the install.")
    state["phases"] = phases
    state["detail"] = ""
    card.state = {**card.state, "restart": state}
    card.terminal = True
    return card
