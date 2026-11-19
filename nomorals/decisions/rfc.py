"""RFC process — cross-module decisions, the RFC-0000 pattern.

Mined from storytold/craft-contrib (``rfcs/0000-decision-process.md``,
no licence — pattern only, no upstream file or text vendored or
copied). The whole upstream design fits in sixteen lines; this module
implements exactly what it says, and nothing it does not:

* Changes to the repo, and decisions that affect several apps, go
  through RFCs: numbered proposals with a short title, an author, a
  kind, and the modules they affect. The process itself changes the
  same way — a ``process``-kind RFC is how these rules change.
* Anyone can open an RFC or comment on one.
* A maintainer starts the comment period by setting its end date:
  7 days for adding or removing an app or changing this process,
  3 days for everything else. The end date follows from the kind —
  it is computed here, never negotiated.
* After the period, the RFC is accepted with approval from a
  maintainer other than the author, if no maintainer's objection is
  unresolved. An objection gives a reason and says what would
  resolve it; both are required, and both are recorded.
* An owner can decide an objection, or accept or close an RFC early,
  and says why — the reason is required and recorded.

What this module does NOT invent: there is no withdraw, no reopen,
and no author self-approval. A closed RFC returns only as a NEW RFC
that supersedes it (``supersede``) — that is the mined path back.
An accepted RFC writes a decision-journal entry, so the decision
lands in the owner's journal with its prediction slot still open:
the owner resolves it when the change actually lands.

Roles are names, not GitHub teams: the store carries an ``owner``
(default ``"owner"``) and a maintainer set (default ``{"owner"}``).
The owner manages maintainers with ``rfc maintainers add|remove``.
In production the acting user is the owner, so owner-early decisions
always work; the full period + approval path needs a second named
maintainer, which is the point — one voice cannot merge its own RFC.

Only the standard library is imported (same rule as the journal).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "KINDS",
    "PERIOD_DAYS",
    "STATUSES",
    "TERMINAL_STATUSES",
    "Objection",
    "Rfc",
    "RfcComment",
    "RfcError",
    "RfcStore",
    "RfcUsageError",
    "handle_rfc",
    "slugify",
]

#: Comment-period lengths in days, by RFC kind — the mined rule.
#: These change only through a ``process``-kind RFC.
PERIOD_DAYS: dict[str, int] = {
    "app_add": 7,
    "app_remove": 7,
    "process": 7,
    "other": 3,
}

KINDS: tuple[str, ...] = ("app_add", "app_remove", "process", "other")

STATUSES: tuple[str, ...] = ("draft", "in_comment", "accepted", "closed")

#: Terminal states never move again.
TERMINAL_STATUSES: tuple[str, ...] = ("accepted", "closed")

_DAY = 86400.0


class RfcError(ValueError):
    """Any refusal from the RFC process — always names what is wrong."""


class RfcUsageError(RfcError):
    """Bad command shape (unknown subverb, missing argument)."""


def slugify(title: str) -> str:
    """``"Adopt the new cache"`` → ``"adopt-the-new-cache"``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return slug or "rfc"


@dataclass
class Objection:
    """A maintainer's objection: reason + what would resolve it.

    Both fields are required at creation — an objection that does not
    say what would resolve it cannot be worked, so it cannot exist.
    """

    id: str
    by: str
    reason: str
    resolves: str
    resolved: bool = False
    resolved_by: str = ""
    resolved_note: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "by": self.by, "reason": self.reason,
            "resolves": self.resolves, "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_note": self.resolved_note,
            "created_at": self.created_at, "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Objection:
        return cls(
            id=str(row.get("id", "")),
            by=str(row.get("by", "")),
            reason=str(row.get("reason", "")),
            resolves=str(row.get("resolves", "")),
            resolved=bool(row.get("resolved", False)),
            resolved_by=str(row.get("resolved_by", "")),
            resolved_note=str(row.get("resolved_note", "")),
            created_at=float(row.get("created_at") or 0.0),
            resolved_at=row.get("resolved_at"),
        )


@dataclass
class RfcComment:
    """One comment on an RFC — anyone may comment, on a live RFC."""

    by: str
    text: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"by": self.by, "text": self.text,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> RfcComment:
        return cls(by=str(row.get("by", "")),
                   text=str(row.get("text", "")),
                   created_at=float(row.get("created_at") or 0.0))


@dataclass
class Rfc:
    """One RFC: the proposal, its period, its objections, its fate."""

    number: int
    title: str
    kind: str
    author: str
    modules: list[str] = field(default_factory=list)
    summary: str = ""
    body: str = ""
    status: str = "draft"
    created_at: float = field(default_factory=time.time)
    period_started_at: float | None = None
    period_ends_at: float | None = None
    period_started_by: str = ""
    approvals: list[dict[str, Any]] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    comments: list[RfcComment] = field(default_factory=list)
    supersedes: int | None = None
    superseded_by: int | None = None
    decision: dict[str, Any] | None = None

    # ── derived ──────────────────────────────────────────────────
    @property
    def slug(self) -> str:
        return slugify(self.title)

    @property
    def filename(self) -> str:
        """Canonical file name — the mined ``rfcs/NNNN-short-title.md``."""
        return f"rfcs/{self.number:04d}-{self.slug}.md"

    @property
    def period_days(self) -> int:
        return PERIOD_DAYS[self.kind]

    @property
    def unresolved_objections(self) -> list[Objection]:
        return [o for o in self.objections if not o.resolved]

    @property
    def non_author_approvals(self) -> list[dict[str, Any]]:
        return [a for a in self.approvals if a.get("by") != self.author]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "title": self.title, "kind": self.kind,
            "author": self.author, "modules": list(self.modules),
            "summary": self.summary, "body": self.body,
            "status": self.status, "created_at": self.created_at,
            "period_started_at": self.period_started_at,
            "period_ends_at": self.period_ends_at,
            "period_started_by": self.period_started_by,
            "approvals": list(self.approvals),
            "objections": [o.to_dict() for o in self.objections],
            "comments": [c.to_dict() for c in self.comments],
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Rfc:
        return cls(
            number=int(row.get("number") or 0),
            title=str(row.get("title", "")),
            kind=str(row.get("kind", "other")),
            author=str(row.get("author", "")),
            modules=list(row.get("modules") or []),
            summary=str(row.get("summary", "")),
            body=str(row.get("body", "")),
            status=str(row.get("status", "draft")),
            created_at=float(row.get("created_at") or 0.0),
            period_started_at=row.get("period_started_at"),
            period_ends_at=row.get("period_ends_at"),
            period_started_by=str(row.get("period_started_by", "")),
            approvals=list(row.get("approvals") or []),
            objections=[Objection.from_dict(o)
                        for o in (row.get("objections") or [])],
            comments=[RfcComment.from_dict(c)
                      for c in (row.get("comments") or [])],
            supersedes=row.get("supersedes"),
            superseded_by=row.get("superseded_by"),
            decision=row.get("decision"),
        )


class RfcStore:
    """The RFC record: numbered proposals, their periods, their fate.

    JSON-backed, atomic write (tmp + replace), same shape as the
    decision journal. A store with no path is memory-only.
    """

    def __init__(self, path: str | None = None, *,
                 maintainers: list[str] | None = None,
                 owner: str = "owner",
                 journal: Any = None) -> None:
        self.path = path
        self.owner = owner or "owner"
        self.maintainers: set[str] = set(maintainers or [self.owner])
        self.maintainers.add(self.owner)
        self.journal = journal
        self.rfcs: dict[int, Rfc] = {}
        self.next_number = 1
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        try:
            self.next_number = int(raw.get("next_number") or 1)
        except (TypeError, ValueError):
            self.next_number = 1
        if raw.get("owner"):
            self.owner = str(raw["owner"])
        for name in raw.get("maintainers") or []:
            self.maintainers.add(str(name))
        self.maintainers.add(self.owner)
        for row in raw.get("rfcs") or []:
            try:
                rfc = Rfc.from_dict(row)
            except (TypeError, ValueError):
                continue
            if rfc.number >= 1:
                self.rfcs[rfc.number] = rfc
                self.next_number = max(self.next_number, rfc.number + 1)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "owner": self.owner,
                    "maintainers": sorted(self.maintainers),
                    "next_number": self.next_number,
                    "rfcs": [r.to_dict() for r in self.rfcs.values()],
                }, fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            _log.debug("rfc._save failed", exc_info=True)

    # ── roles ────────────────────────────────────────────────────
    def is_maintainer(self, name: str) -> bool:
        return name in self.maintainers

    def is_owner(self, name: str) -> bool:
        return name == self.owner

    def _require_maintainer(self, name: str) -> None:
        if not self.is_maintainer(name):
            raise RfcError(
                f"{name!r} is not a maintainer — maintainers are "
                f"{', '.join(sorted(self.maintainers))}")

    # ── lookup ───────────────────────────────────────────────────
    def get(self, number: int) -> Rfc | None:
        return self.rfcs.get(int(number))

    def resolve_ref(self, ref: str) -> Rfc | None:
        """``7``, ``"0007"``, ``"rfc-7"`` — exact number only, no
        guessing: a prefix that matches nothing is None."""
        raw = (ref or "").strip().lower().removeprefix("rfc-")
        try:
            return self.rfcs.get(int(raw))
        except (TypeError, ValueError):
            return None

    def by_module(self, module: str) -> list[Rfc]:
        rows = [r for r in self.rfcs.values() if module in r.modules]
        return sorted(rows, key=lambda r: r.number)

    def by_status(self, status: str) -> list[Rfc]:
        rows = [r for r in self.rfcs.values() if r.status == status]
        return sorted(rows, key=lambda r: r.number)

    def open_rfcs(self) -> list[Rfc]:
        rows = [r for r in self.rfcs.values()
                if r.status in ("draft", "in_comment")]
        return sorted(rows, key=lambda r: r.number)

    # ── intake ───────────────────────────────────────────────────
    def open_rfc(self, *, title: str, kind: str, modules: list[str],
                 author: str, summary: str = "", body: str = "",
                 supersedes: int | None = None,
                 ts: float | None = None) -> Rfc:
        """Open an RFC. Anyone may open one; the required fields are
        named when missing, never silently defaulted."""
        title = (title or "").strip()
        if not title:
            raise RfcError("an RFC needs a title — "
                           "rfc open <title> | <kind> | <modules>")
        kind = (kind or "").strip().lower()
        if kind not in KINDS:
            raise RfcError(
                f"kind {kind!r} is not an RFC kind — "
                f"use one of {', '.join(KINDS)}")
        clean_modules = [m.strip() for m in (modules or []) if m.strip()]
        if not clean_modules:
            raise RfcError(
                "an RFC needs at least one affected module — "
                "rfc open <title> | <kind> | <module1; module2>")
        author = (author or "").strip()
        if not author:
            raise RfcError("an RFC needs an author")
        old = None
        if supersedes is not None:
            old = self.rfcs.get(int(supersedes))
            if old is None:
                raise RfcError(f"no RFC {supersedes} to supersede")
            if old.status != "closed":
                raise RfcError(
                    f"RFC {old.number:04d} is {old.status}, not closed — "
                    "only a closed RFC can be superseded")
        now = float(ts if ts is not None else time.time())
        number = self.next_number
        self.next_number += 1
        rfc = Rfc(number=number, title=title, kind=kind, author=author,
                  modules=clean_modules, summary=summary, body=body,
                  created_at=now,
                  supersedes=(old.number if old is not None else None))
        self.rfcs[number] = rfc
        if old is not None:
            old.superseded_by = number
        self._save()
        return rfc

    def supersede(self, number: int, *, title: str, kind: str,
                  modules: list[str], author: str, summary: str = "",
                  body: str = "", ts: float | None = None) -> Rfc:
        """The mined path back for a closed RFC: a NEW RFC that names
        the closed one. The old RFC stays closed, linked both ways.
        There is no reopen — a closed RFC never moves again."""
        old = self.rfcs.get(int(number))
        if old is None:
            raise RfcError(f"no RFC {number}")
        if old.status != "closed":
            raise RfcError(
                f"RFC {old.number:04d} is {old.status}, not closed — "
                "only a closed RFC can be superseded")
        return self.open_rfc(title=title, kind=kind, modules=modules,
                             author=author, summary=summary, body=body,
                             supersedes=old.number, ts=ts)

    # ── comment period ───────────────────────────────────────────
    def start_period(self, number: int, *, by: str,
                     ts: float | None = None) -> Rfc:
        """A maintainer starts the comment period. The end date
        follows from the kind — 7 days for app add/remove and
        process changes, 3 days for everything else."""
        rfc = self._live(number)
        self._require_maintainer(by)
        if rfc.status != "draft":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — the comment "
                "period starts only from draft")
        now = float(ts if ts is not None else time.time())
        rfc.period_started_at = now
        rfc.period_started_by = by
        rfc.period_ends_at = now + rfc.period_days * _DAY
        rfc.status = "in_comment"
        self._save()
        return rfc

    def comment(self, number: int, *, by: str, text: str,
                ts: float | None = None) -> Rfc:
        """Anyone may comment on a live RFC. Terminal RFCs take no
        more comments — the discussion is over."""
        rfc = self._live(number)
        if rfc.status in TERMINAL_STATUSES:
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — no more comments")
        text = (text or "").strip()
        if not text:
            raise RfcError("a comment needs text")
        now = float(ts if ts is not None else time.time())
        rfc.comments.append(RfcComment(by=by, text=text, created_at=now))
        self._save()
        return rfc

    # ── approval + objection ─────────────────────────────────────
    def approve(self, number: int, *, by: str,
                ts: float | None = None) -> Rfc:
        """A maintainer approves. The author cannot approve their own
        RFC, and approval counts only once the comment period is
        running — approving a draft is refused, not queued."""
        rfc = self._live(number)
        if rfc.status != "in_comment":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — approval counts "
                "only once the comment period is running")
        if by == rfc.author:
            raise RfcError(
                f"{by!r} wrote RFC {rfc.number:04d} — the author cannot "
                "approve their own RFC")
        self._require_maintainer(by)
        now = float(ts if ts is not None else time.time())
        if not any(a.get("by") == by for a in rfc.approvals):
            rfc.approvals.append({"by": by, "at": now})
        self._save()
        return rfc

    def object(self, number: int, *, by: str, reason: str, resolves: str,
               ts: float | None = None) -> Objection:
        """A maintainer objects. The objection must give a reason AND
        say what would resolve it — either missing is refused by name.
        """
        rfc = self._live(number)
        self._require_maintainer(by)
        if rfc.status != "in_comment":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — objections count "
                "only once the comment period is running")
        reason = (reason or "").strip()
        if not reason:
            raise RfcError(
                "an objection needs a reason — what is wrong with the RFC")
        resolves = (resolves or "").strip()
        if not resolves:
            raise RfcError(
                "an objection needs what would resolve it — "
                "say what change would lift the objection")
        now = float(ts if ts is not None else time.time())
        digest = f"{rfc.number}|{by}|{reason}|{now}"
        ob = Objection(
            id="ob-" + hashlib.sha1(digest.encode("utf-8")).hexdigest()[:10],
            by=by, reason=reason, resolves=resolves, created_at=now)
        rfc.objections.append(ob)
        self._save()
        return ob

    def resolve_objection(self, number: int, *, objection_id: str, by: str,
                          note: str = "", ts: float | None = None) -> Rfc:
        """Resolve an objection: the objector, or any maintainer.
        The resolution says how it was addressed — the note is
        required, so the record shows what changed."""
        rfc = self._live(number)
        if rfc.status != "in_comment":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — objections can "
                "only be resolved while the comment period is running")
        ob = next((o for o in rfc.objections if o.id == objection_id), None)
        if ob is None:
            raise RfcError(
                f"no objection {objection_id!r} on RFC {rfc.number:04d}")
        if ob.resolved:
            raise RfcError(f"objection {objection_id!r} is already resolved")
        if by != ob.by and not self.is_maintainer(by):
            raise RfcError(
                f"{by!r} cannot resolve this objection — only the "
                f"objector ({ob.by}) or a maintainer can")
        note = (note or "").strip()
        if not note:
            raise RfcError(
                "resolving an objection needs a note — say how it was "
                "addressed")
        now = float(ts if ts is not None else time.time())
        ob.resolved = True
        ob.resolved_by = by
        ob.resolved_note = note
        ob.resolved_at = now
        self._save()
        return rfc

    # ── decision ─────────────────────────────────────────────────
    def accept(self, number: int, *, by: str, reason: str = "",
               ts: float | None = None) -> Rfc:
        """Accept an RFC. The normal path needs the comment period
        over, an approval from a maintainer other than the author,
        and no unresolved objection — each missing piece is named.
        The owner may accept early, from draft or in comment, but
        must say why; the reason is recorded on the decision."""
        rfc = self._live(number)
        if rfc.status in TERMINAL_STATUSES:
            raise RfcError(f"RFC {rfc.number:04d} is already {rfc.status}")
        now = float(ts if ts is not None else time.time())
        reason = (reason or "").strip()
        if self.is_owner(by) and reason:
            return self._decide(rfc, "accepted", by, now,
                                method="owner-early", reason=reason)
        if rfc.status != "in_comment":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — acceptance needs "
                "a finished comment period, or an owner decision with "
                "a reason")
        if rfc.period_ends_at is not None and now < rfc.period_ends_at:
            raise RfcError(
                f"RFC {rfc.number:04d} is still in its comment period — "
                f"it ends at {rfc.period_ends_at:.0f} "
                f"({rfc.period_days} days for a {rfc.kind} RFC)")
        if not rfc.non_author_approvals:
            raise RfcError(
                f"RFC {rfc.number:04d} has no approval from a maintainer "
                f"other than its author ({rfc.author})")
        if rfc.unresolved_objections:
            names = ", ".join(o.by for o in rfc.unresolved_objections)
            raise RfcError(
                f"RFC {rfc.number:04d} has unresolved objections from "
                f"{names} — resolve them first")
        return self._decide(rfc, "accepted", by, now, method="period")

    def close(self, number: int, *, by: str, reason: str = "",
              ts: float | None = None) -> Rfc:
        """Close an RFC. After the period, any maintainer may close
        it with a reason. Before that, only the owner may close it
        early — also with a reason, recorded on the decision."""
        rfc = self._live(number)
        if rfc.status in TERMINAL_STATUSES:
            raise RfcError(f"RFC {rfc.number:04d} is already {rfc.status}")
        reason = (reason or "").strip()
        if not reason:
            raise RfcError(
                "closing an RFC needs a reason — say why it is closed")
        now = float(ts if ts is not None else time.time())
        if self.is_owner(by):
            return self._decide(rfc, "closed", by, now,
                                method="owner-early", reason=reason)
        if rfc.status != "in_comment":
            raise RfcError(
                f"RFC {rfc.number:04d} is {rfc.status} — only the owner "
                "can close it before its comment period")
        self._require_maintainer(by)
        if rfc.period_ends_at is not None and now < rfc.period_ends_at:
            raise RfcError(
                f"RFC {rfc.number:04d} is still in its comment period — "
                "only the owner can close it early")
        return self._decide(rfc, "closed", by, now,
                            method="period", reason=reason)

    def _decide(self, rfc: Rfc, status: str, by: str, now: float, *,
                method: str, reason: str = "") -> Rfc:
        rfc.status = status
        rfc.decision = {
            "status": status,
            "method": method,
            "decided_by": by,
            "decided_at": now,
            "reason": reason,
            "approvals": [a.get("by") for a in rfc.approvals],
            "modules": list(rfc.modules),
        }
        self._save()
        if status == "accepted" and self.journal is not None:
            context = rfc.summary or rfc.body or ""
            if rfc.modules:
                context = (context + f" [modules: {', '.join(rfc.modules)}]"
                           if context else
                           f"modules: {', '.join(rfc.modules)}")
            self.journal.record(
                decision=f"RFC {rfc.number:04d}: {rfc.title}",
                context=context, chosen="accepted",
                tags=["rfc", f"rfc-{rfc.number:04d}"])
        return rfc

    # ── maintainers ──────────────────────────────────────────────
    def add_maintainer(self, name: str, *, by: str) -> list[str]:
        """The owner names a maintainer. Only the owner can."""
        if not self.is_owner(by):
            raise RfcError(
                f"{by!r} is not the owner — only the owner names "
                "maintainers")
        name = (name or "").strip()
        if not name:
            raise RfcError("a maintainer needs a name")
        self.maintainers.add(name)
        self._save()
        return sorted(self.maintainers)

    def remove_maintainer(self, name: str, *, by: str) -> list[str]:
        """The owner removes a maintainer. The owner themself cannot
        be removed — the owner is always a maintainer."""
        if not self.is_owner(by):
            raise RfcError(
                f"{by!r} is not the owner — only the owner removes "
                "maintainers")
        if name == self.owner:
            raise RfcError("the owner cannot be removed as a maintainer")
        self.maintainers.discard(name)
        self._save()
        return sorted(self.maintainers)

    # ── helpers ──────────────────────────────────────────────────
    def _live(self, number: int) -> Rfc:
        rfc = self.rfcs.get(int(number))
        if rfc is None:
            raise RfcError(f"no RFC {number}")
        return rfc


def _render(rfc: Rfc) -> str:
    lines = [f"RFC {rfc.number:04d}: {rfc.title} [{rfc.status}]",
             f"  kind: {rfc.kind} · author: {rfc.author} · "
             f"modules: {', '.join(rfc.modules)}",
             f"  file: {rfc.filename}"]
    if rfc.summary:
        lines.append(f"  {rfc.summary}")
    if rfc.status == "in_comment" and rfc.period_ends_at is not None:
        lines.append(f"  comment period ends at {rfc.period_ends_at:.0f} "
                     f"(started by {rfc.period_started_by})")
    if rfc.approvals:
        lines.append("  approvals: " +
                     ", ".join(a.get("by", "?") for a in rfc.approvals))
    live = rfc.unresolved_objections
    if live:
        lines.append("  unresolved objections:")
        for ob in live:
            lines.append(f"    {ob.id} by {ob.by}: {ob.reason} "
                         f"(resolves: {ob.resolves})")
    if rfc.supersedes is not None:
        lines.append(f"  supersedes RFC {rfc.supersedes:04d}")
    if rfc.superseded_by is not None:
        lines.append(f"  superseded by RFC {rfc.superseded_by:04d}")
    if rfc.decision:
        d = rfc.decision
        lines.append(f"  decided: {d['status']} by {d['decided_by']} "
                     f"({d['method']})"
                     + (f" — {d['reason']}" if d.get("reason") else ""))
    return "\n".join(lines)


def handle_rfc(store: RfcStore, parts: list[str], *,
               actor: str = "owner") -> str:
    """One dispatcher for the RFC verbs, shared by chat and CLI.

    ``parts`` is the argument list after ``rfc``. Returns the reply
    text; raises :class:`RfcUsageError` for bad shapes and
    :class:`RfcError` for process refusals — both name the problem.
    """
    if not parts:
        raise RfcUsageError(
            "usage: rfc list · show <n> · open <title> | <kind> | "
            "<mod1; mod2> [| summary] · period <n> · comment <n> <text> · "
            "approve <n> · object <n> | <reason> | <resolves> · "
            "resolve <n> <objection-id> | <note> · accept <n> [| reason] · "
            "close <n> | <reason> · supersede <n> <title> | <kind> | "
            "<modules> · maintainers [add|remove <name>] · module <name>")
    verb = parts[0].lower()
    rest = parts[1:]

    if verb == "list":
        rows = store.open_rfcs()
        if not rows:
            return "no open RFCs — rfc open <title> | <kind> | <modules> starts one."
        lines = [f"open RFCs ({len(rows)}):"]
        for r in rows:
            lines.append(f"  RFC {r.number:04d} [{r.status}] {r.title} "
                         f"({r.kind}, {r.author})")
        return "\n".join(lines)

    if verb == "show":
        if not rest:
            raise RfcUsageError("usage: rfc show <n>")
        rfc = store.resolve_ref(rest[0])
        if rfc is None:
            raise RfcError(f"no RFC {rest[0]!r}")
        return _render(rfc)

    if verb == "open":
        fields = [f.strip() for f in " ".join(rest).split("|")]
        title = fields[0] if fields else ""
        kind = fields[1] if len(fields) > 1 else ""
        modules = ([m.strip() for m in fields[2].split(";") if m.strip()]
                   if len(fields) > 2 and fields[2] else [])
        summary = fields[3] if len(fields) > 3 else ""
        author = fields[4] if len(fields) > 4 and fields[4] else actor
        rfc = store.open_rfc(title=title, kind=kind, modules=modules,
                             author=author, summary=summary)
        return (f"opened RFC {rfc.number:04d}: {rfc.title} "
                f"({rfc.kind}, {rfc.period_days}-day comment period) — "
                f"a maintainer starts the period with "
                f"rfc period {rfc.number}")

    if verb == "period":
        if not rest:
            raise RfcUsageError("usage: rfc period <n>")
        rfc = store.resolve_ref(rest[0])
        if rfc is None:
            raise RfcError(f"no RFC {rest[0]!r}")
        store.start_period(rfc.number, by=actor)
        return (f"RFC {rfc.number:04d} comment period started — ends at "
                f"{rfc.period_ends_at:.0f} "
                f"({rfc.period_days} days for a {rfc.kind} RFC)")

    if verb == "comment":
        if len(rest) < 2:
            raise RfcUsageError("usage: rfc comment <n> <text>")
        rfc = store.resolve_ref(rest[0])
        if rfc is None:
            raise RfcError(f"no RFC {rest[0]!r}")
        store.comment(rfc.number, by=actor, text=" ".join(rest[1:]))
        return f"comment recorded on RFC {rfc.number:04d}"

    if verb == "approve":
        if not rest:
            raise RfcUsageError("usage: rfc approve <n>")
        rfc = store.resolve_ref(rest[0])
        if rfc is None:
            raise RfcError(f"no RFC {rest[0]!r}")
        store.approve(rfc.number, by=actor)
        return f"{actor} approved RFC {rfc.number:04d}"

    if verb == "object":
        fields = [f.strip() for f in " ".join(rest).split("|")]
        if len(fields) < 3 or not fields[0]:
            raise RfcUsageError(
                "usage: rfc object <n> | <reason> | <what would resolve it>")
        rfc = store.resolve_ref(fields[0])
        if rfc is None:
            raise RfcError(f"no RFC {fields[0]!r}")
        ob = store.object(rfc.number, by=actor, reason=fields[1],
                          resolves=fields[2])
        return (f"objection {ob.id} recorded on RFC {rfc.number:04d} — "
                "acceptance is blocked until it is resolved")

    if verb == "resolve":
        if len(rest) < 2:
            raise RfcUsageError(
                "usage: rfc resolve <n> <objection-id> | <note>")
        fields = [f.strip() for f in " ".join(rest).split("|")]
        head = fields[0].split()
        if len(head) < 2:
            raise RfcUsageError(
                "usage: rfc resolve <n> <objection-id> | <note>")
        rfc = store.resolve_ref(head[0])
        if rfc is None:
            raise RfcError(f"no RFC {head[0]!r}")
        note = fields[1] if len(fields) > 1 else ""
        store.resolve_objection(rfc.number, objection_id=head[1],
                                by=actor, note=note)
        return f"objection {head[1]} on RFC {rfc.number:04d} resolved"

    if verb == "accept":
        fields = [f.strip() for f in " ".join(rest).split("|")]
        if not fields or not fields[0]:
            raise RfcUsageError("usage: rfc accept <n> [| reason]")
        rfc = store.resolve_ref(fields[0])
        if rfc is None:
            raise RfcError(f"no RFC {fields[0]!r}")
        reason = fields[1] if len(fields) > 1 else ""
        store.accept(rfc.number, by=actor, reason=reason)
        return (f"RFC {rfc.number:04d} accepted: {rfc.title} — "
                "recorded in the decision journal")

    if verb == "close":
        fields = [f.strip() for f in " ".join(rest).split("|")]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            raise RfcUsageError("usage: rfc close <n> | <reason>")
        rfc = store.resolve_ref(fields[0])
        if rfc is None:
            raise RfcError(f"no RFC {fields[0]!r}")
        store.close(rfc.number, by=actor, reason=fields[1])
        return f"RFC {rfc.number:04d} closed: {fields[1]}"

    if verb == "supersede":
        fields = [f.strip() for f in " ".join(rest).split("|")]
        if len(fields) < 3 or not fields[0]:
            raise RfcUsageError(
                "usage: rfc supersede <n> <title> | <kind> | <modules> "
                "[| summary]")
        head = fields[0].split(None, 1)
        if len(head) < 2:
            raise RfcUsageError(
                "usage: rfc supersede <n> <title> | <kind> | <modules> "
                "[| summary]")
        rfc = store.resolve_ref(head[0])
        if rfc is None:
            raise RfcError(f"no RFC {head[0]!r}")
        kind = fields[1]
        modules = ([m.strip() for m in fields[2].split(";") if m.strip()]
                   if fields[2] else [])
        summary = fields[3] if len(fields) > 3 else ""
        author = fields[4] if len(fields) > 4 and fields[4] else actor
        new = store.supersede(rfc.number, title=head[1], kind=kind,
                              modules=modules, author=author,
                              summary=summary)
        return (f"RFC {rfc.number:04d} stays closed — superseded by new "
                f"RFC {new.number:04d}: {new.title}")

    if verb == "maintainers":
        if len(rest) >= 2 and rest[0].lower() == "add":
            names = store.add_maintainer(rest[1], by=actor)
            return "maintainers: " + ", ".join(names)
        if len(rest) >= 2 and rest[0].lower() == "remove":
            names = store.remove_maintainer(rest[1], by=actor)
            return "maintainers: " + ", ".join(names)
        if rest:
            raise RfcUsageError(
                "usage: rfc maintainers [add|remove <name>]")
        return "maintainers: " + ", ".join(sorted(store.maintainers))

    if verb == "module":
        if not rest:
            raise RfcUsageError("usage: rfc module <name>")
        rows = store.by_module(rest[0])
        if not rows:
            return f"no RFCs affect module {rest[0]!r}"
        lines = [f"RFCs affecting {rest[0]} ({len(rows)}):"]
        for r in rows:
            lines.append(f"  RFC {r.number:04d} [{r.status}] {r.title}")
        return "\n".join(lines)

    raise RfcUsageError(f"unknown rfc verb {verb!r} — "
                        "list · show · open · period · comment · approve · "
                        "object · resolve · accept · close · supersede · "
                        "maintainers · module")
