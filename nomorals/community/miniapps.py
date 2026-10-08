"""Group mini-apps: chat-native interactive widgets for group chats.

Three working templates — ``poll``, ``expenses``, ``rsvp`` — with full
state logic. The surface is chat text: every render ends with the exact
``/miniapp`` command that drives the next action.

State is group-scoped JSON (``~/.devon/community/miniapps/``), keyed by
platform + chat id (+ thread when present). There is no web hosting:
:func:`web_url` returns ``None`` and says so plainly — chat-native is
the surface. Nothing here is a placeholder: every action validates,
mutates, persists, and replies.

Isolation: this module imports stdlib + ``nomorals.core.ids`` only.
No memory, no accounts, no vaults, no connectors — see the package
docstring.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.miniapps")

KINDS = ("poll", "expenses", "rsvp")

_DEFAULT_DIR = Path.home() / ".devon" / "community" / "miniapps"


def _sanitize_group_key(group_key: str) -> str:
    """Make a group key safe as a filename (platform:chat_id → platform_chat_id)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", group_key).strip("_") or "group"


def web_url(app: "MiniApp") -> None:
    """Where this mini-app is hosted on the web.

    Returns ``None`` — always. Mini-apps are chat-native; there is no
    web hosting wired for them (no public URL, no artifact publish).
    The chat render (:func:`render_miniapp`) is the complete interface.
    """
    _ = app
    return None


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class MiniApp:
    """One mini-app instance living in one group chat."""

    id: str
    kind: str  # "poll" | "expenses" | "rsvp"
    group_key: str  # "platform:chat_id" (+ ":thread_id" when threaded)
    title: str
    state: dict[str, Any] = field(default_factory=dict)
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniApp":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            group_key=str(data.get("group_key", "")),
            title=str(data.get("title", "")),
            state=dict(data.get("state") or {}),
            created_ts=float(data.get("created_ts") or 0.0),
        )


class MiniAppStore:
    """Group-scoped JSON persistence. Files only — no owner DB tables."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DIR

    def _path(self, group_key: str) -> Path:
        return self.data_dir / f"{_sanitize_group_key(group_key)}.json"

    def load(self, group_key: str) -> list[MiniApp]:
        """All mini-apps for a group. Malformed file → empty list, never raises."""
        path = self._path(group_key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        apps: list[MiniApp] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                app = MiniApp.from_dict(item)
                if app.id and app.kind in KINDS:
                    apps.append(app)
            except (TypeError, ValueError, AttributeError):
                continue
        return apps

    def save(self, group_key: str, apps: list[MiniApp]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(group_key).with_suffix(".tmp")
            tmp.write_text(
                json.dumps([a.to_dict() for a in apps], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._path(group_key))
        except OSError as exc:
            _log.warning("miniapp save failed for %s: %s", group_key, exc)

    def get(self, group_key: str, app_id: str) -> MiniApp | None:
        for app in self.load(group_key):
            if app.id == app_id or app.id.startswith(app_id):
                return app
        return None

    def put(self, app: MiniApp) -> None:
        apps = [a for a in self.load(app.group_key) if a.id != app.id]
        apps.append(app)
        self.save(app.group_key, apps)

    def remove(self, group_key: str, app_id: str) -> bool:
        apps = self.load(group_key)
        kept = [a for a in apps if not (a.id == app_id or a.id.startswith(app_id))]
        if len(kept) == len(apps):
            return False
        self.save(group_key, kept)
        return True


# ── creation ───────────────────────────────────────────────────────────────


def create_miniapp(kind: str, group_key: str, title: str, **params: Any) -> MiniApp:
    """Create a mini-app. Raises ValueError on bad kind/params (caller-facing)."""
    kind = (kind or "").lower()
    if kind not in KINDS:
        raise ValueError(f"unknown mini-app kind {kind!r} (want one of {', '.join(KINDS)})")
    title = (title or "").strip() or kind.title()
    now = time.time()
    if kind == "poll":
        options = [str(o).strip() for o in params.get("options", []) if str(o).strip()]
        if len(options) < 2:
            raise ValueError("a poll needs at least 2 options")
        state: dict[str, Any] = {
            "options": options,
            "votes": {},  # user_id -> option index
            "names": {},  # user_id -> display name
            "closed": False,
        }
    elif kind == "expenses":
        state = {"expenses": [], "settlements": []}
    else:  # rsvp
        state = {
            "date": str(params.get("date", "")).strip(),
            "responses": {},  # user_id -> "yes" | "no" | "maybe"
            "names": {},
            "seen": {},  # user_id -> display name (everyone who interacted)
        }
    return MiniApp(
        id=new_short_id(prefix=kind[:4] + "_", length=8),
        kind=kind,
        group_key=group_key,
        title=title,
        state=state,
        created_ts=now,
    )


# ── money helpers (integer cents — no float drift) ──────────────────────────


def _parse_cents(raw: str) -> int | None:
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d{1,2})?)\s*", str(raw or ""))
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", ".")) * 100))


def _fmt_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def _balances(state: dict[str, Any]) -> dict[str, int]:
    """Net balance per participant id: positive = is owed money."""
    bal: dict[str, int] = {}
    for exp in state.get("expenses", []):
        amount = int(exp.get("amount_cents", 0))
        split = exp.get("split_among") or []
        if not split or amount <= 0:
            continue
        share, rem = divmod(amount, len(split))
        payer = str(exp.get("payer", ""))
        for i, uid in enumerate(split):
            uid = str(uid)
            bal[uid] = bal.get(uid, 0) - (share + (1 if i < rem else 0))
        bal[payer] = bal.get(payer, 0) + amount
    for s in state.get("settlements", []):
        amt = int(s.get("amount_cents", 0))
        bal[str(s.get("from", ""))] = bal.get(str(s.get("from", "")), 0) + amt
        bal[str(s.get("to", ""))] = bal.get(str(s.get("to", "")), 0) - amt
    return {k: v for k, v in bal.items() if v != 0}


def simplify_debts(balances: dict[str, int]) -> list[tuple[str, str, int]]:
    """Greedy who-owes-whom: minimal transfers settling every balance."""
    debtors = sorted(((u, -b) for u, b in balances.items() if b < 0), key=lambda x: -x[1])
    creditors = sorted(((u, b) for u, b in balances.items() if b > 0), key=lambda x: -x[1])
    transfers: list[tuple[str, str, int]] = []
    i = j = 0
    debtors = [list(d) for d in debtors]
    creditors = [list(c) for c in creditors]
    while i < len(debtors) and j < len(creditors):
        du, da = debtors[i]
        cu, ca = creditors[j]
        amt = min(da, ca)
        transfers.append((du, cu, amt))
        da -= amt
        ca -= amt
        if da == 0:
            i += 1
        else:
            debtors[i][1] = da
        if ca == 0:
            j += 1
        else:
            creditors[j][1] = ca
    return transfers


# ── rendering (chat-native UI) ─────────────────────────────────────────────


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.0f}%" if total else "0%"


def render_miniapp(app: MiniApp) -> str:
    """Render a mini-app as chat text with the exact next-action commands."""
    if app.kind == "poll":
        return _render_poll(app)
    if app.kind == "expenses":
        return _render_expenses(app)
    return _render_rsvp(app)


def _render_poll(app: MiniApp) -> str:
    st = app.state
    options: list[str] = st.get("options", [])
    votes: dict[str, int] = st.get("votes", {})
    names: dict[str, str] = st.get("names", {})
    counts = [0] * len(options)
    for idx in votes.values():
        if 0 <= int(idx) < len(counts):
            counts[int(idx)] += 1
    total = sum(counts)
    status = "🔒 closed" if st.get("closed") else "🟢 open"
    lines = [f"📊 *{app.title}* [{status}]", ""]
    for i, opt in enumerate(options):
        bar = "█" * round(10 * counts[i] / total) if total else ""
        voters = ", ".join(names[uid] for uid, v in votes.items() if int(v) == i and uid in names)
        lines.append(f"{i + 1}. {opt} — {counts[i]} vote(s) ({_pct(counts[i], total)}) {bar}")
        if voters:
            lines.append(f"    ↳ {voters}")
    lines += ["", f"id: `{app.id}`"]
    if not st.get("closed"):
        lines.append(f"Vote: /miniapp vote {app.id} <number>   •   Close: /miniapp close {app.id}")
    return "\n".join(lines)


def _render_expenses(app: MiniApp) -> str:
    st = app.state
    names: dict[str, str] = {}
    for exp in st.get("expenses", []):
        names.update({str(k): str(v) for k, v in (exp.get("names") or {}).items()})
    lines = [f"💸 *{app.title}*", ""]
    if not st.get("expenses"):
        lines.append("No expenses yet.")
    for exp in st.get("expenses", [])[-10:]:
        who = exp.get("payer_name") or str(exp.get("payer", "?"))
        lines.append(f"• {who} paid {_fmt_money(exp.get('amount_cents', 0))} — {exp.get('what', '')}")
    bal = _balances(st)
    if bal:
        lines += ["", "*Balances:*"]
        for uid, b in sorted(bal.items(), key=lambda x: -x[1]):
            nm = names.get(uid, uid)
            lines.append(f"  {nm}: {'is owed ' if b > 0 else 'owes '}{_fmt_money(abs(b))}")
        lines += ["", "*Settle up:*"]
        for frm, to, amt in simplify_debts(bal):
            lines.append(f"  {names.get(frm, frm)} → {names.get(to, to)}: {_fmt_money(amt)}")
    lines += [
        "",
        f"id: `{app.id}`",
        f"Add: /miniapp expense {app.id} <amount> <what> [for name1,name2]",
        f"Record a payment: /miniapp settle {app.id} <from> <to> <amount>",
    ]
    return "\n".join(lines)


def _render_rsvp(app: MiniApp) -> str:
    st = app.state
    responses: dict[str, str] = st.get("responses", {})
    names: dict[str, str] = st.get("names", {})
    yes = [names.get(u, u) for u, r in responses.items() if r == "yes"]
    no = [names.get(u, u) for u, r in responses.items() if r == "no"]
    maybe = [names.get(u, u) for u, r in responses.items() if r == "maybe"]
    date = st.get("date") or "date TBD"
    lines = [f"📅 *{app.title}* — {date}", "",
             f"✅ Yes ({len(yes)}): {', '.join(yes) or '—'}",
             f"❔ Maybe ({len(maybe)}): {', '.join(maybe) or '—'}",
             f"❌ No ({len(no)}): {', '.join(no) or '—'}",
             "", f"id: `{app.id}`",
             f"RSVP: /miniapp rsvp {app.id} yes|no|maybe",
             f"Who hasn't answered: /miniapp nudge {app.id}"]
    return "\n".join(lines)


# ── actions ────────────────────────────────────────────────────────────────


def apply_action(
    app: MiniApp, user_id: str, user_name: str, action: str, args: list[str]
) -> tuple[MiniApp, str]:
    """Apply one action. Never raises on bad input — returns an error reply.

    Returns (possibly updated app, reply text). The caller persists the app.
    """
    user_id = (user_id or "").strip() or "anon"
    user_name = (user_name or "").strip() or user_id
    action = (action or "").lower()
    try:
        if app.kind == "poll":
            return _poll_action(app, user_id, user_name, action, args)
        if app.kind == "expenses":
            return _expense_action(app, user_id, user_name, action, args)
        if app.kind == "rsvp":
            return _rsvp_action(app, user_id, user_name, action, args)
        return app, f"Unknown mini-app kind {app.kind!r}."
    except Exception as exc:  # noqa: BLE001 — action layer never raises
        _log.debug("miniapp action %s failed: %s", action, exc)
        return app, f"Couldn't do that: {exc}"


def _poll_action(app, user_id, user_name, action, args):
    st = app.state
    if action == "vote":
        if st.get("closed"):
            return app, "This poll is closed — no more votes."
        if not args:
            return app, f"Usage: /miniapp vote {app.id} <option number>"
        try:
            idx = int(args[0]) - 1
        except ValueError:
            return app, f"Pick a number 1–{len(st['options'])}."
        if not 0 <= idx < len(st["options"]):
            return app, f"Pick a number 1–{len(st['options'])}."
        prev = st["votes"].get(user_id)
        st["votes"][user_id] = idx
        st["names"][user_id] = user_name
        note = f" (changed from option {int(prev) + 1})" if prev is not None and int(prev) != idx else ""
        return app, f"🗳️ {user_name} voted for “{st['options'][idx]}”{note}."
    if action == "close":
        if st.get("closed"):
            return app, "Already closed."
        st["closed"] = True
        return app, f"🔒 Poll closed by {user_name}.\n\n" + _render_poll(app)
    return app, f"Poll actions: vote, close. Try /miniapp vote {app.id} <number>"


def _expense_action(app, user_id, user_name, action, args):
    st = app.state
    if action == "expense":
        if len(args) < 2:
            return app, f"Usage: /miniapp expense {app.id} <amount> <what> [for name1,name2]"
        cents = _parse_cents(args[0])
        if cents is None or cents <= 0:
            return app, f"Couldn't parse amount {args[0]!r} — try like 12.50."
        rest = " ".join(args[1:])
        what, _, for_part = rest.partition(" for ")
        what = what.strip() or "expense"
        if for_part.strip():
            split_names = [n.strip() for n in for_part.split(",") if n.strip()]
        else:
            split_names = []
        # Resolve names → ids among known participants; unknown names become
        # their own participant id (display-name keyed — group-local only).
        # The payer's own name resolves to the payer (no double-counting).
        known: dict[str, str] = {user_name.lower(): user_id}
        for exp in st["expenses"]:
            for uid, nm in (exp.get("names") or {}).items():
                known.setdefault(str(nm).lower(), str(uid))
        split_ids = [user_id]
        for nm in split_names:
            uid = known.get(nm.lower(), f"name:{nm.lower()}")
            if uid not in split_ids:
                split_ids.append(uid)
        names = {user_id: user_name}
        for nm in split_names:
            names[f"name:{nm.lower()}"] = nm
        exp = {
            "id": new_short_id("exp_", 8),
            "payer": user_id,
            "payer_name": user_name,
            "amount_cents": cents,
            "what": what,
            "split_among": split_ids,
            "names": names,
            "ts": time.time(),
        }
        st["expenses"].append(exp)
        return app, f"💸 Recorded: {user_name} paid {_fmt_money(cents)} for “{what}”."
    if action == "settle":
        if len(args) < 3:
            return app, f"Usage: /miniapp settle {app.id} <from> <to> <amount>"
        frm, to = args[0], args[1]
        cents = _parse_cents(args[2])
        if cents is None or cents <= 0:
            return app, f"Couldn't parse amount {args[2]!r}."
        st["settlements"].append(
            {"from": frm, "to": to, "amount_cents": cents, "ts": time.time()}
        )
        return app, f"✅ Recorded payment: {frm} → {to} {_fmt_money(cents)}."
    return app, f"Expense actions: expense, settle. Try /miniapp expense {app.id} 12.50 dinner"


def _rsvp_action(app, user_id, user_name, action, args):
    st = app.state
    st.setdefault("seen", {})[user_id] = user_name
    if action in ("yes", "no", "maybe"):
        st["responses"][user_id] = action
        st["names"][user_id] = user_name
        return app, f"📅 {user_name} → {action.upper()} for “{app.title}”."
    if action == "nudge":
        responded = set(st["responses"])
        quiet = [nm for uid, nm in st["seen"].items() if uid not in responded]
        if not quiet:
            return app, "Everyone who's seen this has answered. 🎉"
        return app, "Still waiting on: " + ", ".join(quiet)
    return app, f"RSVP actions: yes, no, maybe, nudge. Try /miniapp rsvp {app.id} yes"


# ── chat command ───────────────────────────────────────────────────────────


def _group_key(chat: Any) -> str:
    base = f"{chat.platform}:{chat.chat_id}"
    thread = getattr(chat, "thread_id", "") or ""
    return f"{base}:{thread}" if thread else base


def control_miniapp(
    tail: str,
    context: Any = None,
    chat: Any = None,
    sender_id: str = "",
    sender: str = "",
) -> str:
    """``/miniapp`` — group mini-apps. Group chats only; DMs are refused.

    ``chat`` is a ChatRef (its ``kind`` decides group vs DM); ``sender_id``
    / ``sender`` identify the acting participant. Never raises.
    """
    # Compare the kind as a plain string ("group") so this module needs no
    # imports beyond stdlib + core — ChatKind.GROUP is exactly "group".
    if chat is None or getattr(chat, "kind", "dm") != "group":
        return ("Mini-apps live in group chats — they track the whole group's "
                "polls, expenses and RSVPs. Try this in a group. 👥")
    data_dir = getattr(getattr(context, "settings", None), "community_dir", None)
    store = MiniAppStore(data_dir=data_dir) if data_dir else MiniAppStore()
    gkey = _group_key(chat)
    user_id = (sender_id or "").strip() or (sender or "").strip() or "anon"
    user_name = (sender or "").strip() or user_id

    words = (tail or "").split()
    sub = words[0].lower() if words else "help"
    rest = words[1:]

    if sub == "help" or sub == "":
        return _HELP
    if sub == "list":
        apps = store.load(gkey)
        if not apps:
            return ("No mini-apps in this group yet.\n" + _NEW_HINT)
        lines = ["Mini-apps in this group:"]
        for a in apps:
            lines.append(f"• {a.kind} “{a.title}” — /miniapp show {a.id}")
        return "\n".join(lines)
    if sub == "new":
        if len(rest) < 1:
            return "Usage: /miniapp new <poll|expenses|rsvp> <title> [options…]\n" + _NEW_HINT
        kind = rest[0].lower()
        # Title may be quoted: new poll "Best day?" Mon Tue Wed
        m = re.match(r'\s*\S+\s+"([^"]+)"\s*(.*)$', " " + " ".join(rest))
        if m:
            title, opt_str = m.group(1), m.group(2)
        else:
            title, opt_str = " ".join(rest[1:]), ""
        options = opt_str.split()
        try:
            app = create_miniapp(kind, gkey, title,
                                 options=options,
                                 date=" ".join(rest[1:]) if kind == "rsvp" else "")
        except ValueError as exc:
            return f"Couldn't create that: {exc}\n" + _NEW_HINT
        store.put(app)
        return f"Created! 🎉\n\n{render_miniapp(app)}"
    if sub == "show":
        if not rest:
            return "Usage: /miniapp show <id>"
        app = store.get(gkey, rest[0])
        return render_miniapp(app) if app else f"No mini-app {rest[0]!r} in this group."
    if sub == "rm":
        if not rest:
            return "Usage: /miniapp rm <id>"
        return ("🗑️ Removed." if store.remove(gkey, rest[0])
                else f"No mini-app {rest[0]!r} in this group.")
    # actions: vote|close → poll · expense|settle → expenses · rsvp|nudge → rsvp
    action_aliases = {
        "vote": "vote", "close": "close",
        "expense": "expense", "settle": "settle",
        "rsvp": "rsvp", "nudge": "nudge",
        "yes": "yes", "no": "no", "maybe": "maybe",
    }
    if sub in action_aliases:
        if not rest:
            return f"Usage: /miniapp {sub} <id> [args…]"
        app = store.get(gkey, rest[0])
        if app is None:
            return f"No mini-app {rest[0]!r} in this group."
        action = action_aliases[sub]
        if sub in ("yes", "no", "maybe"):
            action = sub  # /miniapp yes <id>
            app_id, aargs = rest[0], []
            app = store.get(gkey, app_id)
        else:
            aargs = rest[1:]
        if app is None:
            return f"No mini-app {rest[0]!r} in this group."
        updated, reply = apply_action(app, user_id, user_name, action, aargs)
        store.put(updated)
        return reply
    return f"Unknown /miniapp subcommand {sub!r}.\n\n{_HELP}"


_NEW_HINT = ("Create one: /miniapp new poll \"Question?\" opt1 opt2 opt3\n"
             "           /miniapp new expenses \"Trip fund\"\n"
             "           /miniapp new rsvp \"Game night\" 2026-10-20")

_HELP = ("/miniapp — group mini-apps (polls, shared expenses, RSVPs)\n"
         "/miniapp new <poll|expenses|rsvp> <title> [options…]\n"
         "/miniapp list   •   /miniapp show <id>   •   /miniapp rm <id>\n"
         "/miniapp vote <id> <number>   •   /miniapp close <id>\n"
         "/miniapp expense <id> <amount> <what> [for name1,name2]\n"
         "/miniapp settle <id> <from> <to> <amount>\n"
         "/miniapp rsvp <id> yes|no|maybe   •   /miniapp nudge <id>")


# ── Panels (Hark pattern): persistent, connector-backed mini-apps ──────────
#
# A Panel is a user-described mini-app wired to LIVE data:
#   "track my marathon training" → Panel(data_source="strava")
# Panels persist (not one-shot), refresh from their connector on a cadence,
# and answer questions in chat: "how's my marathon training going?"
#
# Data fetching goes through an injectable ``fetcher`` seam
# (``fetcher(connector_id) -> dict``) so tests and the chat layer can supply
# mock or real connectors. The default fetcher is honest: it reports that
# no live connector is wired rather than fabricating data.


_PANEL_DIR = Path.home() / ".devon" / "community" / "panels"

#: refresh cadences (seconds)
_PANEL_CADENCES = {"hourly": 3600, "daily": 86400, "weekly": 604800, "manual": 0}


@dataclass
class Panel:
    """A persistent, connector-backed mini-app.

    ``data_source`` names a connector id (e.g. ``"strava"``, ``"mono"``).
    ``data`` is the last fetched snapshot; ``last_refresh`` its timestamp.
    """

    id: str
    name: str
    user_key: str  # owner-scoped: "owner" or a community member id
    data_source: str
    description: str = ""  # the user's own words: "track my marathon training"
    query_template: str = ""  # optional hint for how to read the data
    refresh_cadence: str = "daily"  # hourly | daily | weekly | manual
    last_refresh: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Panel":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            user_key=str(data.get("user_key", "owner")),
            data_source=str(data.get("data_source", "")),
            description=str(data.get("description", "")),
            query_template=str(data.get("query_template", "")),
            refresh_cadence=str(data.get("refresh_cadence", "daily")),
            last_refresh=float(data.get("last_refresh") or 0.0),
            data=dict(data.get("data") or {}),
            created_ts=float(data.get("created_ts") or 0.0),
        )

    def needs_refresh(self) -> bool:
        """True when the cadence says it's time for fresh data."""
        try:
            cadence = _PANEL_CADENCES.get(self.refresh_cadence, 86400)
            if not cadence:
                return False  # manual
            return (time.time() - self.last_refresh) >= cadence
        except Exception:  # noqa: BLE001 — never raises
            return False

    def age_str(self) -> str:
        """Human age of the current snapshot."""
        try:
            if not self.last_refresh:
                return "never refreshed"
            age = time.time() - self.last_refresh
            if age < 90:
                return "just now"
            if age < 3600:
                return f"{int(age // 60)}m ago"
            if age < 86400:
                return f"{int(age // 3600)}h ago"
            return f"{int(age // 86400)}d ago"
        except Exception:  # noqa: BLE001
            return "unknown"


class PanelStore:
    """Owner-scoped JSON persistence. Files only — never raises."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _PANEL_DIR

    def _path(self, user_key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_key).strip("_") or "owner"
        return self.data_dir / f"{safe}.json"

    def load(self, user_key: str) -> list[Panel]:
        """All panels for a user. Malformed file → empty list, never raises."""
        try:
            raw = json.loads(self._path(user_key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        panels: list[Panel] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                panel = Panel.from_dict(item)
                if panel.id and panel.name:
                    panels.append(panel)
            except (TypeError, ValueError, AttributeError):
                continue
        return panels

    def save(self, user_key: str, panels: list[Panel]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(user_key).with_suffix(".tmp")
            tmp.write_text(
                json.dumps([p.to_dict() for p in panels], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._path(user_key))
        except OSError as exc:
            _log.warning("panel save failed for %s: %s", user_key, exc)

    def get(self, user_key: str, panel_id: str) -> Panel | None:
        for panel in self.load(user_key):
            if panel.id == panel_id or panel.id.startswith(panel_id):
                return panel
        return None

    def find_by_name(self, user_key: str, name: str) -> Panel | None:
        needle = (name or "").strip().lower()
        for panel in self.load(user_key):
            if panel.name.lower() == needle or needle in panel.name.lower():
                return panel
        return None

    def put(self, panel: Panel) -> None:
        panels = [p for p in self.load(panel.user_key) if p.id != panel.id]
        panels.append(panel)
        self.save(panel.user_key, panels)

    def remove(self, user_key: str, panel_id: str) -> bool:
        panels = self.load(user_key)
        kept = [p for p in panels if not (p.id == panel_id or p.id.startswith(panel_id))]
        if len(kept) == len(panels):
            return False
        self.save(user_key, kept)
        return True


# ── creation ───────────────────────────────────────────────────────────────


def create_panel(
    name: str,
    user_key: str,
    data_source: str,
    description: str = "",
    *,
    refresh_cadence: str = "daily",
    query_template: str = "",
    store: PanelStore | None = None,
) -> Panel:
    """Create a persistent panel. Never raises — returns the panel."""
    try:
        name = (name or "").strip()[:80] or "Untitled panel"
        user_key = (user_key or "owner").strip() or "owner"
        data_source = (data_source or "").strip().lower() or "manual"
        cadence = refresh_cadence if refresh_cadence in _PANEL_CADENCES else "daily"
        panel = Panel(
            id="panel_" + new_short_id(),
            name=name,
            user_key=user_key,
            data_source=data_source,
            description=(description or "").strip()[:300],
            query_template=(query_template or "").strip()[:300],
            refresh_cadence=cadence,
            created_ts=time.time(),
        )
        (store or PanelStore()).put(panel)
        return panel
    except Exception:  # noqa: BLE001 — never raises
        return Panel(
            id="panel_" + new_short_id(),
            name="Untitled panel",
            user_key="owner",
            data_source="manual",
            created_ts=time.time(),
        )


# ── live refresh ───────────────────────────────────────────────────────────


def _default_fetcher(connector_id: str) -> dict[str, Any]:
    """Honest default: no live connector wired — say so, don't fabricate."""
    return {
        "_error": (
            f"no live connector wired for {connector_id!r} — "
            "wire a fetcher or connect the account first"
        )
    }


def refresh_panel(
    panel_id: str,
    user_key: str = "owner",
    *,
    fetcher: Any = None,
    store: PanelStore | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pull live data for a panel. Returns {"ok", "panel"|"reason"}.

    ``fetcher`` is ``fetcher(connector_id) -> dict`` — inject a mock in
    tests or the real connector adapter in production. Never raises.
    """
    try:
        st = store or PanelStore()
        panel = st.get(user_key, panel_id)
        if panel is None:
            return {"ok": False, "reason": f"no panel {panel_id!r}"}
        if not force and not panel.needs_refresh() and panel.data:
            return {"ok": True, "panel": panel, "cached": True}
        fetch = fetcher or _default_fetcher
        try:
            data = fetch(panel.data_source)
        except Exception as exc:  # noqa: BLE001 — fetcher failure is data
            data = {"_error": f"fetch failed: {exc}"}
        if not isinstance(data, dict):
            data = {"_error": "fetcher returned non-dict", "value": str(data)[:200]}
        panel.data = data
        panel.last_refresh = time.time()
        st.put(panel)
        return {"ok": True, "panel": panel, "cached": False}
    except Exception as exc:  # noqa: BLE001 — never raises
        return {"ok": False, "reason": f"refresh failed: {exc}"}


# ── Q&A over panel data ────────────────────────────────────────────────────


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to dotted key paths for keyword matching."""
    flat: dict[str, Any] = {}
    try:
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten(value, path))
            elif isinstance(value, (list, tuple)):
                flat[path] = value
                for i, item in enumerate(value[:10]):
                    if isinstance(item, dict):
                        flat.update(_flatten(item, f"{path}[{i}]"))
                    else:
                        flat[f"{path}[{i}]"] = item
            else:
                flat[path] = value
    except Exception:  # noqa: BLE001
        pass
    return flat


def _fmt_value(value: Any) -> str:
    try:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, (list, tuple)):
            return f"{len(value)} items"
        return str(value)[:120]
    except Exception:  # noqa: BLE001
        return "?"


_SUMMARY_WORDS = {"summary", "overview", "how", "going", "status", "report", "all", "everything"}


def ask_panel(
    panel_id: str,
    question: str,
    user_key: str = "owner",
    *,
    store: PanelStore | None = None,
) -> dict[str, Any]:
    """Answer a question from a panel's data. Returns {"ok", "answer"|"reason"}.

    Keyword-matches the question against flattened data keys; falls back
    to a summary of the snapshot. Never raises.
    """
    try:
        st = store or PanelStore()
        panel = st.get(user_key, panel_id)
        if panel is None:
            return {"ok": False, "reason": f"no panel {panel_id!r}"}
        question = (question or "").strip()
        if not question:
            return {"ok": False, "reason": "ask me something about the panel"}
        data = panel.data or {}
        if data.get("_error"):
            return {
                "ok": True,
                "answer": (
                    f"⚠️ {panel.name}: {data['_error']}\n"
                    f"Refresh with `/panel refresh {panel.id}` once it's wired up."
                ),
            }
        if not data:
            return {
                "ok": True,
                "answer": (
                    f"📊 {panel.name} has no data yet — "
                    f"run `/panel refresh {panel.id}` first."
                ),
            }
        flat = _flatten(data)
        words = set(re.findall(r"[a-z]{3,}", question.lower()))
        # summary-style questions → overview of the snapshot
        if words & _SUMMARY_WORDS and len(words - _SUMMARY_WORDS) < 3:
            lines = [f"📊 {panel.name} — updated {panel.age_str()}"]
            for path, value in list(flat.items())[:12]:
                if isinstance(value, (dict, list)):
                    continue
                label = path.replace("_", " ").replace(".", " › ")
                lines.append(f"• {label}: {_fmt_value(value)}")
            if panel.query_template:
                lines.append(f"\n_{panel.query_template}_")
            return {"ok": True, "answer": "\n".join(lines)}
        # keyword match: score each key path by word overlap
        scored: list[tuple[int, str, Any]] = []
        for path, value in flat.items():
            if isinstance(value, (dict, list)) or value is None:
                continue
            key_words = set(re.findall(r"[a-z]{3,}", path.lower().replace("_", " ")))
            overlap = len(words & key_words)
            if overlap:
                scored.append((overlap, path, value))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            keys = ", ".join(sorted(set(
                re.findall(r"[a-z]{3,}", " ".join(flat.keys()).lower())
            ))[:15])
            return {
                "ok": True,
                "answer": (
                    f"🤔 Nothing in {panel.name} matched {question!r}.\n"
                    f"I track: {keys or '—'}.\n"
                    f"Try `/panel ask {panel.id} summary`."
                ),
            }
        lines = [f"📊 {panel.name} — {_fmt_value(v)}" for _, path, v in scored[:1]]
        for _, path, value in scored[1:6]:
            label = path.replace("_", " ").replace(".", " › ")
            lines.append(f"• {label}: {_fmt_value(value)}")
        lines.append(f"\n_updated {panel.age_str()}_")
        return {"ok": True, "answer": "\n".join(lines)}
    except Exception as exc:  # noqa: BLE001 — never raises
        return {"ok": False, "reason": f"ask failed: {exc}"}


def render_panel(panel: Panel) -> str:
    """Chat render of a panel."""
    try:
        lines = [
            f"📊 {panel.name}",
            f"  source: {panel.data_source} · refresh: {panel.refresh_cadence} "
            f"· updated {panel.age_str()}",
        ]
        if panel.description:
            lines.append(f"  _{panel.description[:120]}_")
        data = panel.data or {}
        if data.get("_error"):
            lines.append(f"  ⚠️ {str(data['_error'])[:120]}")
        else:
            shown = 0
            for path, value in _flatten(data).items():
                if isinstance(value, (dict, list)) or value is None or shown >= 6:
                    continue
                label = path.replace("_", " ").replace(".", " › ")
                lines.append(f"  • {label}: {_fmt_value(value)}")
                shown += 1
            if not shown:
                lines.append("  _(no data yet — /panel refresh)_")
        lines.append(f"\n`/panel ask {panel.id} <question>` · `/panel refresh {panel.id}`")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"📊 {panel.name}"


# ── chat ───────────────────────────────────────────────────────────────────


_PANEL_HELP = (
    "/panel — persistent live-data panels (Hark pattern)\n"
    "/panel new \"<name>\" <connector> [hourly|daily|weekly|manual] — create\n"
    "/panel list   •   /panel show <id>   •   /panel rm <id>\n"
    "/panel refresh <id>   •   /panel ask <id> <question>"
)


def control_panel(
    text: str,
    user_id: str = "owner",
    user_name: str = "",
    *,
    store: PanelStore | None = None,
    fetcher: Any = None,
) -> str:
    """Chat entry point for panels. Never raises."""
    try:
        st = store or PanelStore()
        parts = (text or "").strip().split(None, 2)
        sub = parts[1].lower() if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        if sub in ("new", "create"):
            # /panel new "Marathon training" strava daily
            m = re.match(r'"([^"]+)"\s+(\S+)(?:\s+(hourly|daily|weekly|manual))?', rest)
            if not m:
                m2 = re.match(r"(\S+)\s+(\S+)(?:\s+(hourly|daily|weekly|manual))?", rest)
                name, source, cadence = (m2.groups() if m2 else (None, None, None))
            else:
                name, source, cadence = m.groups()
            if not name or not source:
                return "Usage: /panel new \"<name>\" <connector> [hourly|daily|weekly|manual]"
            panel = create_panel(
                name, user_id, source,
                refresh_cadence=cadence or "daily", store=st,
            )
            return f"✅ Panel created:\n\n{render_panel(panel)}"

        if sub == "list":
            panels = st.load(user_id)
            if not panels:
                return "No panels yet. Create one: /panel new \"<name>\" <connector>"
            lines = ["📊 Your panels:"]
            for p in panels:
                lines.append(f"• {p.name} ({p.data_source}, {p.refresh_cadence}) — `{p.id}`")
            return "\n".join(lines)

        if sub in ("show", "refresh", "rm", "delete", "ask"):
            pid = rest.split(None, 1)[0] if rest else ""
            if not pid:
                return f"Usage: /panel {sub} <id> [question]"
            if sub == "show":
                panel = st.get(user_id, pid)
                return render_panel(panel) if panel else f"No panel {pid!r}."
            if sub in ("rm", "delete"):
                ok = st.remove(user_id, pid)
                return "🗑️ Panel deleted." if ok else f"No panel {pid!r}."
            if sub == "refresh":
                result = refresh_panel(pid, user_id, fetcher=fetcher, store=st, force=True)
                if not result["ok"]:
                    return f"⚠️ {result['reason']}"
                panel = result["panel"]
                note = " (cached)" if result.get("cached") else ""
                return f"🔄 Refreshed{note}:\n\n{render_panel(panel)}"
            # ask
            question = rest.split(None, 1)[1] if len(rest.split(None, 1)) > 1 else ""
            # resolve by name too: /panel ask marathon how's it going?
            panel = st.get(user_id, pid) or st.find_by_name(user_id, pid)
            if panel is None:
                return f"No panel {pid!r}."
            if not question:
                return f"Usage: /panel ask {pid} <question>"
            result = ask_panel(panel.id, question, user_id, store=st)
            return result.get("answer", result.get("reason", "?"))

        return f"Unknown /panel subcommand {sub!r}.\n\n{_PANEL_HELP}"
    except Exception as exc:  # noqa: BLE001 — never raises
        return f"⚠️ panel error: {exc}"
