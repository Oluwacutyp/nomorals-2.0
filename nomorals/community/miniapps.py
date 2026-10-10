"""Group mini-apps: chat-native interactive widgets for group chats.

Working templates — ``poll``, ``expenses``, ``rsvp``, ``quiz`` — with
full state logic. The surface is chat text: every render ends with the
exact ``/miniapp`` command that drives the next action.

Telegram-Polls-2.0 / Splitwise-grade upgrades (mined 2026-10-10):
* **Quiz mode** (Telegram Polls 2.0): ``quiz`` kind — one correct answer,
  per-user attempts, points, streaks, and a medal scoreboard.
* **Multiple-answer polls** (Telegram Polls 2.0): ``multi`` polls take
  approval ballots; renders show approval bars.
* **Poll deadlines**: ``closes_at`` + ``due_polls()``; auto-status in render.
* **Poll comments** (Doodle's comments section): one-level discussion
  thread under the results.
* **Split strategies** (Splitwise strategy pattern): expenses split
  ``equal`` (default), ``exact:<c1,c2,…>`` (cents, sum-validated — the
  classic "vanishing remainder" bug is rejected loudly), or
  ``percent:<p1,p2,…>`` (must sum to 100, rounding drift fixed on the
  largest share).
* **Edit/delete expenses** (expense-splitter): balances recompute from
  the ledger; nothing is a special case.
* **Categories + summaries**: ``cat:<name>`` tags; ``summary`` renders
  per-category, per-month, and per-member contribution tables.
* **Pairwise vs smart settlement** (real Splitwise shows both): render
  shows raw who-owes-whom edges AND the simplified transfers.
* **RSVP capacity + waitlist + guests**: mirrors events.py.
* **Panels** (Hark pattern, unchanged core): ``due_refreshes()`` for the
  host scheduler and ``digest()`` for one-line-per-panel briefings.

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

KINDS = ("poll", "expenses", "rsvp", "quiz")

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
    kind: str  # "poll" | "expenses" | "rsvp" | "quiz"
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
            "votes": {},  # user_id -> option index (single-choice)
            "ballots": {},  # user_id -> [option indexes] (multi/approval)
            "names": {},  # user_id -> display name
            "closed": False,
            "multi": bool(params.get("multi", False)),
            "closes_at": float(params.get("closes_at") or 0.0),
            "comments": [],  # [{user, text, ts}]
        }
    elif kind == "expenses":
        state = {"expenses": [], "settlements": [],
                 "currency": str(params.get("currency", "") or "")}
    elif kind == "quiz":
        options = [str(o).strip() for o in params.get("options", []) if str(o).strip()]
        if len(options) < 2:
            raise ValueError("a quiz needs at least 2 options")
        correct = params.get("correct", 0)
        try:
            correct = int(correct)
        except (TypeError, ValueError):
            correct = 0
        if not 0 <= correct < len(options):
            raise ValueError(f"correct must be a number 1–{len(options)}")
        state = {
            "options": options,
            "correct": correct,
            "votes": {},  # user_id -> option index (their answer)
            "names": {},
            "attempts": {},  # user_id -> tries
            "scores": {},  # user_id -> points
            "streaks": {},  # user_id -> current correct streak
            "closed": False,
        }
    else:  # rsvp
        state = {
            "date": str(params.get("date", "")).strip(),
            "responses": {},  # user_id -> "yes" | "no" | "maybe"
            "names": {},
            "seen": {},  # user_id -> display name (everyone who interacted)
            "capacity": max(0, int(params.get("capacity") or 0)),
            "waitlist": [],  # FIFO user_ids
            "guests": {},  # user_id -> +N
        }
    return MiniApp(
        id=new_short_id(prefix=kind[:4] + "_", length=8),
        kind=kind,
        group_key=group_key,
        title=title,
        state=state,
        created_ts=now,
    )


def due_polls(store: MiniAppStore, group_key: str,
              now: float | None = None) -> list[MiniApp]:
    """Polls past their deadline but not closed — the host should close them."""
    now = now if now is not None else time.time()
    return [a for a in store.load(group_key)
            if a.kind == "poll" and not a.state.get("closed")
            and a.state.get("closes_at") and now >= a.state["closes_at"]]


# ── money helpers (integer cents — no float drift) ──────────────────────────


def _parse_cents(raw: str) -> int | None:
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d{1,2})?)\s*", str(raw or ""))
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", ".")) * 100))


def _fmt_money(cents: int, symbol: str = "") -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}{symbol}{cents // 100}.{cents % 100:02d}"


def _split_shares(amount: int, split_ids: list[str],
                  mode: str, params: list[str]) -> dict[str, int] | str:
    """Per-person shares in cents, or an error string. Strategy pattern."""
    if mode in ("", "equal"):
        share, rem = divmod(amount, len(split_ids))
        return {uid: share + (1 if i < rem else 0)
                for i, uid in enumerate(split_ids)}
    if mode == "exact":
        try:
            parts = [int(round(float(p.replace(",", ".")) * 100)) for p in params]
        except ValueError:
            return "exact split needs amounts like split:exact:30,20,50"
        if len(parts) != len(split_ids):
            return (f"exact split needs {len(split_ids)} amounts "
                    f"(one per person), got {len(parts)}")
        if sum(parts) != amount:
            return (f"exact amounts add up to {_fmt_money(sum(parts))} but the "
                    f"expense is {_fmt_money(amount)} — nothing may vanish")
        return dict(zip(split_ids, parts))
    if mode == "percent":
        try:
            pcts = [float(p) for p in params]
        except ValueError:
            return "percent split needs numbers like split:percent:50,30,20"
        if len(pcts) != len(split_ids):
            return (f"percent split needs {len(split_ids)} values, "
                    f"got {len(pcts)}")
        if abs(sum(pcts) - 100.0) > 0.01:
            return f"percentages must add up to 100 (got {sum(pcts):g})"
        shares = [int(round(amount * p / 100.0)) for p in pcts]
        drift = amount - sum(shares)
        if drift:
            # fix rounding drift on the largest share (deterministic rule)
            i = max(range(len(shares)), key=lambda k: shares[k])
            shares[i] += drift
        return dict(zip(split_ids, shares))
    return f"unknown split mode {mode!r} — use equal, exact:<…>, or percent:<…>"


def _balances(state: dict[str, Any]) -> dict[str, int]:
    """Net balance per participant id: positive = is owed money."""
    bal: dict[str, int] = {}
    for exp in state.get("expenses", []):
        amount = int(exp.get("amount_cents", 0))
        if amount <= 0:
            continue
        shares = exp.get("shares")
        if shares:
            split = [(str(uid), int(c)) for uid, c in shares.items()]
        else:
            split_among = exp.get("split_among") or []
            if not split_among:
                continue
            share, rem = divmod(amount, len(split_among))
            split = [(str(uid), share + (1 if i < rem else 0))
                     for i, uid in enumerate(split_among)]
        payer = str(exp.get("payer", ""))
        for uid, cents in split:
            bal[uid] = bal.get(uid, 0) - cents
        bal[payer] = bal.get(payer, 0) + amount
    for s in state.get("settlements", []):
        amt = int(s.get("amount_cents", 0))
        bal[str(s.get("from", ""))] = bal.get(str(s.get("from", "")), 0) + amt
        bal[str(s.get("to", ""))] = bal.get(str(s.get("to", "")), 0) - amt
    return {k: v for k, v in bal.items() if v != 0}


def _pairwise(state: dict[str, Any]) -> dict[tuple[str, str], int]:
    """Raw who-owes-whom edges, before simplification."""
    edges: dict[tuple[str, str], int] = {}
    for exp in state.get("expenses", []):
        amount = int(exp.get("amount_cents", 0))
        if amount <= 0:
            continue
        shares = exp.get("shares")
        if shares:
            split = [(str(uid), int(c)) for uid, c in shares.items()]
        else:
            split_among = exp.get("split_among") or []
            if not split_among:
                continue
            share, rem = divmod(amount, len(split_among))
            split = [(str(uid), share + (1 if i < rem else 0))
                     for i, uid in enumerate(split_among)]
        payer = str(exp.get("payer", ""))
        for uid, cents in split:
            if uid == payer:
                continue
            key = (uid, payer)
            edges[key] = edges.get(key, 0) + cents
    for s in state.get("settlements", []):
        # a settlement cancels the edge it pays down
        key = (str(s.get("to", "")), str(s.get("from", "")))
        edges[key] = edges.get(key, 0) - int(s.get("amount_cents", 0))
    return {k: v for k, v in edges.items() if v > 0}


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


def _bar(n: int, total: int, width: int = 10) -> str:
    return "█" * round(width * n / total) if total else ""


def render_miniapp(app: MiniApp) -> str:
    """Render a mini-app as chat text with the exact next-action commands."""
    if app.kind == "poll":
        return _render_poll(app)
    if app.kind == "expenses":
        return _render_expenses(app)
    if app.kind == "quiz":
        return _render_quiz(app)
    return _render_rsvp(app)


def _render_poll(app: MiniApp) -> str:
    st = app.state
    options: list[str] = st.get("options", [])
    votes: dict[str, int] = st.get("votes", {})
    ballots: dict[str, list[int]] = st.get("ballots", {})
    names: dict[str, str] = st.get("names", {})
    multi = bool(st.get("multi"))
    counts = [0] * len(options)
    if multi:
        for bl in ballots.values():
            for idx in set(int(i) for i in bl if 0 <= int(i) < len(counts)):
                counts[idx] += 1
    else:
        for idx in votes.values():
            if 0 <= int(idx) < len(counts):
                counts[int(idx)] += 1
    total = sum(counts)
    status = "🔒 closed" if st.get("closed") else "🟢 open"
    if multi:
        status += " · multi-answer"
    if st.get("closes_at") and not st.get("closed"):
        left = st["closes_at"] - time.time()
        status += (f" · closes in {int(left // 3600)}h{int(left % 3600 // 60)}m"
                   if left > 0 else " · deadline passed")
    lines = [f"📊 *{app.title}* [{status}]", ""]
    for i, opt in enumerate(options):
        voters = ", ".join(names[uid] for uid, v in votes.items()
                           if int(v) == i and uid in names)
        if multi:
            voters = ", ".join(names[uid] for uid, bl in ballots.items()
                               if i in [int(x) for x in bl] and uid in names)
        lines.append(f"{i + 1}. {opt} — {counts[i]} vote(s) ({_pct(counts[i], total)}) "
                     f"{_bar(counts[i], total)}")
        if voters:
            lines.append(f"    ↳ {voters}")
    lines += ["", f"id: `{app.id}`"]
    comments = st.get("comments", [])
    if comments:
        lines += ["", "💬 *Comments:*"]
        for c in comments[-4:]:
            lines.append(f"  {c.get('user', '?')}: {str(c.get('text', ''))[:120]}")
    if not st.get("closed"):
        vote_hint = (f"Vote (pick any): /miniapp vote {app.id} <numbers…>"
                     if multi else f"Vote: /miniapp vote {app.id} <number>")
        lines.append(vote_hint + f"   •   Close: /miniapp close {app.id}")
        lines.append(f"Comment: /miniapp comment {app.id} <text>")
    return "\n".join(lines)


def _expense_names(st: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for exp in st.get("expenses", []):
        names.update({str(k): str(v) for k, v in (exp.get("names") or {}).items()})
    return names


def _render_expenses(app: MiniApp) -> str:
    st = app.state
    sym = str(st.get("currency", "") or "")
    names = _expense_names(st)
    lines = [f"💸 *{app.title}*", ""]
    if not st.get("expenses"):
        lines.append("No expenses yet.")
    for exp in st.get("expenses", [])[-10:]:
        who = exp.get("payer_name") or str(exp.get("payer", "?"))
        cat = f" [{exp.get('category')}]" if exp.get("category") else ""
        mode = exp.get("split_mode", "equal")
        mode_s = "" if mode == "equal" else f" ({mode})"
        lines.append(f"• {who} paid {_fmt_money(exp.get('amount_cents', 0), sym)}"
                     f" — {exp.get('what', '')}{cat}{mode_s} `{exp.get('id', '')}`")
    bal = _balances(st)
    if bal:
        lines += ["", "*Balances:*"]
        for uid, b in sorted(bal.items(), key=lambda x: -x[1]):
            nm = names.get(uid, uid)
            lines.append(f"  {nm}: {'is owed ' if b > 0 else 'owes '}"
                         f"{_fmt_money(abs(b), sym)}")
        lines += ["", "*Settle up (fewest transfers):*"]
        for frm, to, amt in simplify_debts(bal):
            lines.append(f"  {names.get(frm, frm)} → {names.get(to, to)}: "
                         f"{_fmt_money(amt, sym)}")
        edges = _pairwise(st)
        if edges:
            lines += ["", "*Raw debts:*"]
            for (frm, to), amt in sorted(edges.items(), key=lambda kv: -kv[1])[:6]:
                lines.append(f"  {names.get(frm, frm)} owes {names.get(to, to)}: "
                             f"{_fmt_money(amt, sym)}")
    lines += [
        "",
        f"id: `{app.id}`",
        f"Add: /miniapp expense {app.id} <amount> <what> [for name1,name2] "
        "[split:equal|exact:a,b|percent:p,q] [cat:food]",
        f"Edit: /miniapp edit {app.id} <exp_id> <amount> <what>  •  "
        f"Delete: /miniapp del {app.id} <exp_id>",
        f"Summary: /miniapp summary {app.id}  •  "
        f"Payment: /miniapp settle {app.id} <from> <to> <amount>",
    ]
    return "\n".join(lines)


def _render_quiz(app: MiniApp) -> str:
    st = app.state
    options: list[str] = st.get("options", [])
    names: dict[str, str] = st.get("names", {})
    scores: dict[str, int] = st.get("scores", {})
    attempts: dict[str, int] = st.get("attempts", {})
    status = "🔒 closed" if st.get("closed") else "🟢 open"
    lines = [f"🧠 *{app.title}* [{status}]", ""]
    for i, opt in enumerate(options):
        mark = ""
        if st.get("closed") and i == st.get("correct"):
            mark = " ✅"
        lines.append(f"{i + 1}. {opt}{mark}")
    if scores:
        lines += ["", "🏆 *Scoreboard:*"]
        medals = ["🥇", "🥈", "🥉"]
        for rank, (uid, pts) in enumerate(
                sorted(scores.items(), key=lambda kv: -kv[1])[:10]):
            medal = medals[rank] if rank < 3 else f"{rank + 1}."
            streak = st.get("streaks", {}).get(uid, 0)
            fire = f" 🔥{streak}" if streak >= 2 else ""
            lines.append(f"  {medal} {names.get(uid, uid)} — {pts} pt"
                         f"{'s' if pts != 1 else ''}{fire} "
                         f"({attempts.get(uid, 0)} tries)")
    lines += ["", f"id: `{app.id}`"]
    if not st.get("closed"):
        lines.append(f"Answer: /miniapp answer {app.id} <number>   •   "
                     f"Close: /miniapp close {app.id}")
    return "\n".join(lines)


def _render_rsvp(app: MiniApp) -> str:
    st = app.state
    responses: dict[str, str] = st.get("responses", {})
    names: dict[str, str] = st.get("names", {})
    guests: dict[str, int] = st.get("guests", {})
    yes = [names.get(u, u) + (f" +{guests[u]}" if guests.get(u) else "")
           for u, r in responses.items() if r == "yes"]
    no = [names.get(u, u) for u, r in responses.items() if r == "no"]
    maybe = [names.get(u, u) for u, r in responses.items() if r == "maybe"]
    wl = [names.get(u, u) for u in st.get("waitlist", [])]
    date = st.get("date") or "date TBD"
    cap = ""
    if st.get("capacity"):
        used = len(yes)
        left = max(0, st["capacity"] - used)
        bar = _bar(used, st["capacity"], 8)
        cap = f"\n🎟️ {bar} {used}/{st['capacity']} seats" + \
              (f" · {left} left" if left else " · FULL")
    lines = [f"📅 *{app.title}* — {date}{cap}", "",
             f"✅ Yes ({len(yes)}): {', '.join(yes) or '—'}",
             f"❔ Maybe ({len(maybe)}): {', '.join(maybe) or '—'}",
             f"❌ No ({len(no)}): {', '.join(no) or '—'}"]
    if wl:
        lines.append(f"⏳ Waitlist ({len(wl)}): {', '.join(wl)}")
    lines += ["", f"id: `{app.id}`",
              f"RSVP: /miniapp rsvp {app.id} yes|no|maybe [+N guests]",
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
        if app.kind == "quiz":
            return _quiz_action(app, user_id, user_name, action, args)
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
        idxs = []
        for tok in args:
            try:
                i = int(tok) - 1
            except ValueError:
                return app, f"Pick number(s) 1–{len(st['options'])}."
            if not 0 <= i < len(st["options"]):
                return app, f"Pick number(s) 1–{len(st['options'])}."
            idxs.append(i)
        st["names"][user_id] = user_name
        if st.get("multi"):
            prev = st["ballots"].get(user_id, [])
            st["ballots"][user_id] = sorted(set(idxs))
            return app, (f"🗳️ {user_name} approves "
                         f"{', '.join('“' + st['options'][i] + '”' for i in sorted(set(idxs)))}.")
        idx = idxs[0]
        prev = st["votes"].get(user_id)
        st["votes"][user_id] = idx
        note = f" (changed from option {int(prev) + 1})" if prev is not None and int(prev) != idx else ""
        return app, f"🗳️ {user_name} voted for “{st['options'][idx]}”{note}."
    if action == "close":
        if st.get("closed"):
            return app, "Already closed."
        st["closed"] = True
        return app, f"🔒 Poll closed by {user_name}.\n\n" + _render_poll(app)
    if action == "comment":
        text = " ".join(args).strip()[:500]
        if not text:
            return app, f"Usage: /miniapp comment {app.id} <text>"
        st.setdefault("comments", []).append(
            {"user": user_name, "text": text, "ts": time.time()})
        return app, f"💬 Comment added by {user_name}."
    if action == "deadline":
        if not args:
            return app, f"Usage: /miniapp deadline {app.id} <hours>"
        try:
            hours = float(args[0])
        except ValueError:
            return app, "Hours must be a number."
        st["closes_at"] = time.time() + hours * 3600.0
        return app, f"⏰ Poll closes in {args[0]}h."
    return app, (f"Poll actions: vote, close, comment, deadline. "
                 f"Try /miniapp vote {app.id} <number>")


def _extract_markers(rest: str) -> tuple[str, dict[str, str]]:
    """Pull trailing `split:…` / `cat:…` markers out of expense text."""
    markers: dict[str, str] = {}
    for key in ("split", "cat"):
        m = re.search(rf"\s+{key}:(\S+)\s*$", rest)
        if m:
            markers[key] = m.group(1)
            rest = rest[:m.start()].rstrip()
    return rest, markers


def _expense_action(app, user_id, user_name, action, args):
    st = app.state
    sym = str(st.get("currency", "") or "")
    if action == "expense":
        if len(args) < 2:
            return app, f"Usage: /miniapp expense {app.id} <amount> <what> [for name1,name2]"
        cents = _parse_cents(args[0])
        if cents is None or cents <= 0:
            return app, f"Couldn't parse amount {args[0]!r} — try like 12.50."
        rest = " ".join(args[1:])
        what, _, for_part = rest.partition(" for ")
        for_part, for_markers = _extract_markers(for_part)
        what, what_markers = _extract_markers(what)
        markers = {**what_markers, **for_markers}
        what = what.strip() or "expense"
        category = markers.get("cat", "").strip()[:40]
        split_raw = markers.get("split", "equal")
        mode, _, split_params = split_raw.partition(":")
        mode = mode.lower()
        params = [p for p in split_params.split(",") if p] if split_params else []
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
        shares = _split_shares(cents, split_ids, mode, params)
        if isinstance(shares, str):
            return app, f"⚠️ {shares}"
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
            "shares": {uid: c for uid, c in shares.items()},
            "split_mode": mode,
            "category": category,
            "names": names,
            "ts": time.time(),
        }
        st["expenses"].append(exp)
        cat_s = f" [{category}]" if category else ""
        return app, (f"💸 Recorded: {user_name} paid {_fmt_money(cents, sym)} "
                     f"for “{what}”{cat_s} ({mode} split).")
    if action == "edit":
        if len(args) < 3:
            return app, f"Usage: /miniapp edit {app.id} <exp_id> <amount> <what>"
        exp_id = args[0]
        cents = _parse_cents(args[1])
        if cents is None or cents <= 0:
            return app, f"Couldn't parse amount {args[1]!r}."
        what = " ".join(args[2:]).strip()[:200] or "expense"
        for exp in st["expenses"]:
            if exp.get("id") == exp_id or str(exp.get("id", "")).startswith(exp_id):
                old = exp["amount_cents"]
                exp["amount_cents"] = cents
                exp["what"] = what
                # re-split with the same strategy on the new amount
                shares = _split_shares(cents, exp.get("split_among", [exp["payer"]]),
                                       exp.get("split_mode", "equal"), [])
                if isinstance(shares, dict):
                    exp["shares"] = shares
                return app, (f"✏️ Updated: {_fmt_money(old, sym)} → "
                             f"{_fmt_money(cents, sym)} for “{what}”. Balances recomputed.")
        return app, f"No expense {exp_id!r} here."
    if action == "del":
        if not args:
            return app, f"Usage: /miniapp del {app.id} <exp_id>"
        exp_id = args[0]
        before = len(st["expenses"])
        st["expenses"] = [e for e in st["expenses"]
                          if not (e.get("id") == exp_id
                                  or str(e.get("id", "")).startswith(exp_id))]
        if len(st["expenses"]) == before:
            return app, f"No expense {exp_id!r} here."
        return app, "🗑️ Expense deleted. Balances recomputed."
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
        return app, f"✅ Recorded payment: {frm} → {to} {_fmt_money(cents, sym)}."
    if action == "currency":
        sym_new = (args[0] if args else "").strip()[:4]
        st["currency"] = sym_new
        return app, f"💱 Currency symbol set to {sym_new or 'none'}."
    if action == "summary":
        return app, _expense_summary(st, sym)
    return app, (f"Expense actions: expense, edit, del, settle, currency, summary. "
                 f"Try /miniapp expense {app.id} 12.50 dinner")


def _expense_summary(st: dict[str, Any], sym: str) -> str:
    expenses = st.get("expenses", [])
    if not expenses:
        return "No expenses to summarize."
    names = _expense_names(st)
    total = sum(int(e.get("amount_cents", 0)) for e in expenses)
    by_cat: dict[str, int] = {}
    by_month: dict[str, int] = {}
    by_payer: dict[str, int] = {}
    for e in expenses:
        amt = int(e.get("amount_cents", 0))
        by_cat[e.get("category") or "uncategorized"] = \
            by_cat.get(e.get("category") or "uncategorized", 0) + amt
        by_month[time.strftime("%Y-%m", time.localtime(e.get("ts", 0)))] = \
            by_month.get(time.strftime("%Y-%m", time.localtime(e.get("ts", 0))), 0) + amt
        payer = names.get(str(e.get("payer", "")), str(e.get("payer", "?")))
        by_payer[payer] = by_payer.get(payer, 0) + amt
    lines = [f"📊 *Spending summary* — total {_fmt_money(total, sym)} "
             f"across {len(expenses)} expenses", "",
             "*By category:*"]
    for cat, amt in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {cat}: {_fmt_money(amt, sym)} {_bar(amt, total)}")
    lines.append("*By month:*")
    for mo, amt in sorted(by_month.items()):
        lines.append(f"  {mo}: {_fmt_money(amt, sym)}")
    lines.append("*Paid by:*")
    for who, amt in sorted(by_payer.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {who}: {_fmt_money(amt, sym)}")
    return "\n".join(lines)


def _quiz_action(app, user_id, user_name, action, args):
    st = app.state
    if action in ("vote", "answer"):
        if st.get("closed"):
            return app, "This quiz is closed."
        if not args:
            return app, f"Usage: /miniapp answer {app.id} <option number>"
        try:
            idx = int(args[0]) - 1
        except ValueError:
            return app, f"Pick a number 1–{len(st['options'])}."
        if not 0 <= idx < len(st["options"]):
            return app, f"Pick a number 1–{len(st['options'])}."
        st["names"][user_id] = user_name
        st["attempts"][user_id] = st.get("attempts", {}).get(user_id, 0) + 1
        tries = st["attempts"][user_id]
        correct = idx == st.get("correct")
        if correct:
            pts = 2 if tries == 1 else 1  # first-try bonus
            st["scores"][user_id] = st.get("scores", {}).get(user_id, 0) + pts
            st["streaks"][user_id] = st.get("streaks", {}).get(user_id, 0) + 1
            streak = st["streaks"][user_id]
            return app, (f"✅ Correct! +{pts} pt{'s' if pts != 1 else ''} "
                         f"{user_name}." + (f" 🔥 {streak} in a row!" if streak >= 2 else ""))
        st["streaks"][user_id] = 0
        left = " (answer revealed when the quiz closes)" if not st.get("closed") else ""
        return app, f"❌ Not quite, {user_name}.{left}"
    if action == "close":
        if st.get("closed"):
            return app, "Already closed."
        st["closed"] = True
        return app, f"🔒 Quiz closed by {user_name}.\n\n" + _render_quiz(app)
    if action == "scoreboard":
        return app, _render_quiz(app)
    return app, (f"Quiz actions: answer, scoreboard, close. "
                 f"Try /miniapp answer {app.id} <number>")


def _rsvp_action(app, user_id, user_name, action, args):
    st = app.state
    st.setdefault("seen", {})[user_id] = user_name
    cap = int(st.get("capacity") or 0)
    if action in ("yes", "no", "maybe"):
        guests = 0
        for tok in args:
            if tok.startswith("+") and tok[1:].isdigit():
                guests = min(20, int(tok[1:]))
        prev = st["responses"].get(user_id)
        if action == "yes" and cap:
            used = sum(1 for r in st["responses"].values() if r == "yes")
            if prev != "yes" and used >= cap:
                if user_id not in st.setdefault("waitlist", []):
                    st["waitlist"].append(user_id)
                st["names"][user_id] = user_name
                return app, (f"⏳ Full! {user_name} is #{len(st['waitlist'])} "
                             f"on the waitlist.")
            if user_id in st.get("waitlist", []):
                st["waitlist"].remove(user_id)
        if prev == "yes" and action != "yes":
            # a seat freed — promote the waitlist head
            wl = st.get("waitlist", [])
            if wl:
                nxt = wl.pop(0)
                st["responses"][nxt] = "yes"
                st["names"].setdefault(nxt, nxt)
        st["responses"][user_id] = action
        st["names"][user_id] = user_name
        st["guests"][user_id] = guests
        g = f" +{guests}" if guests else ""
        return app, f"📅 {user_name} → {action.upper()}{g} for “{app.title}”."
    if action == "capacity":
        if not args or not args[0].isdigit():
            return app, f"Usage: /miniapp capacity {app.id} <number> (0 = unlimited)"
        st["capacity"] = max(0, int(args[0]))
        return app, (f"🎟️ Capacity set to {args[0]}."
                     if st["capacity"] else "🎟️ Capacity removed (unlimited).")
    if action == "nudge":
        responded = set(st["responses"])
        quiet = [nm for uid, nm in st["seen"].items() if uid not in responded]
        wl = st.get("waitlist", [])
        bits = []
        if quiet:
            bits.append("Still waiting on: " + ", ".join(quiet))
        if wl:
            bits.append(f"⏳ Waitlist ({len(wl)}): " +
                        ", ".join(st.get("names", {}).get(u, u) for u in wl))
        return app, "\n".join(bits) if bits else "Everyone who's seen this has answered. 🎉"
    return app, f"RSVP actions: yes, no, maybe, capacity, nudge. Try /miniapp rsvp {app.id} yes"


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
        # surface due polls first
        due = due_polls(store, gkey)
        lines = ["Mini-apps in this group:"]
        for a in apps:
            tag = " ⏰ closes soon" if a in due else ""
            lines.append(f"• {a.kind} “{a.title}” — /miniapp show {a.id}{tag}")
        return "\n".join(lines)
    if sub == "new":
        if len(rest) < 1:
            return "Usage: /miniapp new <poll|expenses|rsvp|quiz> <title> [options…]\n" + _NEW_HINT
        kind = rest[0].lower()
        # Title may be quoted: new poll "Best day?" Mon Tue Wed
        m = re.match(r'\s*\S+\s+"([^"]+)"\s*(.*)$', " " + " ".join(rest))
        if m:
            title, opt_str = m.group(1), m.group(2)
        else:
            title, opt_str = " ".join(rest[1:]), ""
        tokens = opt_str.split()
        # trailing flags: multi, correct:<n>, cap:<n>, closes:<h>
        flags: dict[str, Any] = {}
        opts: list[str] = []
        for tok in tokens:
            low = tok.lower()
            if low == "multi":
                flags["multi"] = True
            elif low.startswith("correct:") and low[8:].isdigit():
                flags["correct"] = int(low[8:]) - 1
            elif low.startswith("cap:") and low[4:].isdigit():
                flags["capacity"] = int(low[4:])
            elif low.startswith("closes:"):
                try:
                    flags["closes_at"] = time.time() + float(low[7:]) * 3600.0
                except ValueError:
                    pass
            else:
                opts.append(tok)
        try:
            app = create_miniapp(kind, gkey, title, options=opts,
                                 date=" ".join(rest[1:]) if kind == "rsvp" else "",
                                 **flags)
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
    # actions: vote|close|comment|deadline → poll · expense|settle|edit|del|summary|currency
    # → expenses · answer|scoreboard → quiz · rsvp|nudge|capacity → rsvp
    action_aliases = {
        "vote": "vote", "close": "close", "comment": "comment",
        "deadline": "deadline",
        "expense": "expense", "settle": "settle", "edit": "edit",
        "del": "del", "summary": "summary", "currency": "currency",
        "answer": "answer", "scoreboard": "scoreboard",
        "rsvp": "rsvp", "nudge": "nudge", "capacity": "capacity",
        "yes": "yes", "no": "no", "maybe": "maybe",
    }
    if sub in action_aliases:
        if not rest:
            return f"Usage: /miniapp {sub} <id> [args…]"
        app = store.get(gkey, rest[0])
        if app is None:
            return f"No mini-app {rest[0]!r} in this group."
        action = action_aliases[sub]
        # /miniapp yes|no|maybe <id> [+N] and /miniapp <action> <id> [args…]
        aargs = rest[1:]
        updated, reply = apply_action(app, user_id, user_name, action, aargs)
        store.put(updated)
        return reply
    return f"Unknown /miniapp subcommand {sub!r}.\n\n{_HELP}"


_NEW_HINT = ("Create one: /miniapp new poll \"Question?\" opt1 opt2 opt3 [multi] [closes:24]\n"
             "           /miniapp new quiz \"Capital?\" Lagos Abuja Kano correct:1\n"
             "           /miniapp new expenses \"Trip fund\"\n"
             "           /miniapp new rsvp \"Game night\" 2026-10-20 [cap:20]")

_HELP = ("/miniapp — group mini-apps (polls, quizzes, shared expenses, RSVPs)\n"
         "/miniapp new <poll|expenses|rsvp|quiz> <title> [options…]\n"
         "/miniapp list   •   /miniapp show <id>   •   /miniapp rm <id>\n"
         "/miniapp vote <id> <number> [numbers…]   •   /miniapp close <id>\n"
         "/miniapp comment <id> <text>   •   /miniapp deadline <id> <hours>\n"
         "/miniapp answer <id> <number>   •   /miniapp scoreboard <id>\n"
         "/miniapp expense <id> <amount> <what> [for n1,n2] [split:…] [cat:…]\n"
         "/miniapp edit|del <id> <exp_id> …   •   /miniapp summary <id>\n"
         "/miniapp settle <id> <from> <to> <amount>\n"
         "/miniapp rsvp <id> yes|no|maybe [+N]   •   /miniapp nudge <id>   •   /miniapp capacity <id> <n>")


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


def due_refreshes(user_key: str = "owner",
                  store: PanelStore | None = None) -> list[Panel]:
    """Panels whose cadence says they're stale — the host scheduler calls
    this to know what to refresh without waking every panel."""
    try:
        st = store or PanelStore()
        return [p for p in st.load(user_key) if p.needs_refresh()]
    except Exception:  # noqa: BLE001 — never raises
        return []


def digest(user_key: str = "owner", *, store: PanelStore | None = None,
           fetcher: Any = None) -> str:
    """One-line-per-panel auto-briefing: name, freshness, headline numbers.

    The scheduler can post this as a morning briefing. Never raises.
    """
    try:
        st = store or PanelStore()
        panels = st.load(user_key)
        if not panels:
            return "No panels yet."
        lines = ["📊 **Panel digest:**"]
        for p in panels:
            if fetcher is not None and p.needs_refresh():
                refresh_panel(p.id, user_key, fetcher=fetcher, store=st)
                p = st.get(user_key, p.id) or p
            flat = _flatten(p.data or {})
            headline = ""
            for path, value in flat.items():
                if isinstance(value, (dict, list)) or value is None:
                    continue
                label = path.replace("_", " ").replace(".", " › ")
                headline = f"{label}: {_fmt_value(value)}"
                break
            if (p.data or {}).get("_error"):
                headline = f"⚠️ {str(p.data['_error'])[:80]}"
            elif not headline:
                headline = "no data yet"
            lines.append(f"• **{p.name}** ({p.age_str()}) — {headline}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — never raises
        return f"⚠️ digest failed: {exc}"


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
    "/panel refresh <id>   •   /panel ask <id> <question>   •   /panel digest"
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
                stale = " ⏰ stale" if p.needs_refresh() else ""
                lines.append(f"• {p.name} ({p.data_source}, {p.refresh_cadence}) — `{p.id}`{stale}")
            return "\n".join(lines)

        if sub == "digest":
            return digest(user_id, store=st, fetcher=fetcher)

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
