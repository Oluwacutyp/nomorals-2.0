"""Ticket relay store: the durable ticket↔issue link.

One JSON file under the home dir (``<home>/relay/tickets.json``,
the recruitment-store shape): each ticket is one link between a
Discord channel where the ticket was raised and the GitHub issue it
was filed as. One link per ticket, never two issues for one ticket —
the link is written *before* the issue is created, so a crash or a
refused create leaves a ``failed`` link with the reason instead of
a second filing on retry.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RelayError",
    "TicketLink",
    "TicketRelayStore",
    "ticket_relay_path",
]


class RelayError(ValueError):
    """The relay will not run as asked — bad ref, duplicate, bad repo."""


TICKET_STATES = ("open", "closed", "failed")


def ticket_relay_path(settings: Any = None) -> Path:
    """The relay store path, honoring the runtime settings home.

    Mirrors :func:`nomorals.recruitment.store.recruitment_path`: the
    settings home when one is in force, else ``~/.nomorals``.
    """
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "relay" / "tickets.json"
    return Path.home() / ".nomorals" / "relay" / "tickets.json"


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


@dataclass
class TicketLink:
    """One Discord ticket and the GitHub issue it was filed as."""

    ref: str
    repo: str
    issue_number: int = 0
    issue_url: str = ""
    title: str = ""
    reporter: str = ""
    reporter_id: str = ""
    origin_platform: str = "discord"
    origin_chat: str = ""
    origin_thread: str = ""
    state: str = "open"           # open | closed | failed
    last_issue_state: str = ""
    last_comment_count: int = 0
    seen_comment_ids: list[Any] = field(default_factory=list)
    pending_posts: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0
    last_synced_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "repo": self.repo,
            "issue_number": self.issue_number,
            "issue_url": self.issue_url,
            "title": self.title,
            "reporter": self.reporter,
            "reporter_id": self.reporter_id,
            "origin_platform": self.origin_platform,
            "origin_chat": self.origin_chat,
            "origin_thread": self.origin_thread,
            "state": self.state,
            "last_issue_state": self.last_issue_state,
            "last_comment_count": self.last_comment_count,
            "seen_comment_ids": list(self.seen_comment_ids),
            "pending_posts": list(self.pending_posts),
            "failures": list(self.failures),
            "created_at": self.created_at,
            "last_synced_at": self.last_synced_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TicketLink:
        state = str(data.get("state", "open") or "open")
        if state not in TICKET_STATES:
            state = "open"
        return cls(
            ref=_clean_str(data.get("ref")),
            repo=_clean_str(data.get("repo")),
            issue_number=int(data.get("issue_number", 0) or 0),
            issue_url=_clean_str(data.get("issue_url")),
            title=_clean_str(data.get("title")),
            reporter=_clean_str(data.get("reporter")),
            reporter_id=_clean_str(data.get("reporter_id")),
            origin_platform=_clean_str(data.get("origin_platform"))
            or "discord",
            origin_chat=_clean_str(data.get("origin_chat")),
            origin_thread=_clean_str(data.get("origin_thread")),
            state=state,
            last_issue_state=_clean_str(data.get("last_issue_state")),
            last_comment_count=int(data.get("last_comment_count", 0) or 0),
            seen_comment_ids=list(data.get("seen_comment_ids") or []),
            pending_posts=list(data.get("pending_posts") or []),
            failures=list(data.get("failures") or []),
            created_at=float(data.get("created_at", 0) or 0),
            last_synced_at=(float(data["last_synced_at"])
                            if data.get("last_synced_at") else None),
        )


class TicketRelayStore:
    """The ticket links, one JSON file, loaded whole and saved whole."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._tickets: list[TicketLink] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        items = raw.get("tickets") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            items = []
        self._tickets = []
        for item in items:
            if not isinstance(item, dict):
                continue
            link = TicketLink.from_dict(item)
            if link.ref:
                self._tickets.append(link)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tickets": [t.to_dict() for t in self._tickets]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.path)

    def mint_ref(self) -> str:
        """A fresh ticket ref, unique in this store."""
        self._load()
        taken = {t.ref for t in self._tickets}
        for _ in range(100):
            ref = "T-" + secrets.token_hex(4)
            if ref not in taken:
                return ref
        raise RelayError("could not mint a fresh ticket ref")

    def add(self, link: TicketLink) -> TicketLink:
        """Store a new link; a duplicate ref is refused, never merged."""
        self._load()
        if not link.ref:
            raise RelayError("a ticket link needs a ref")
        if any(t.ref == link.ref for t in self._tickets):
            raise RelayError(f"ticket {link.ref} already exists")
        if not link.created_at:
            link.created_at = time.time()
        self._tickets.append(link)
        self._save()
        return link

    def get(self, ref: str) -> TicketLink | None:
        """A link by ref, by unique prefix, or by '#<issue number>'."""
        self._load()
        ref = (ref or "").strip()
        if not ref:
            return None
        for link in self._tickets:
            if link.ref == ref:
                return link
        matches = [t for t in self._tickets
                   if t.ref.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if ref.startswith("#") and ref[1:].isdigit():
            number = int(ref[1:])
            same = [t for t in self._tickets
                    if t.issue_number == number]
            if len(same) == 1:
                return same[0]
        return None

    def find_by_issue(self, repo: str, number: int) -> TicketLink | None:
        """The link for one GitHub issue, or None when never relayed."""
        self._load()
        for link in self._tickets:
            if link.repo == repo and link.issue_number == number:
                return link
        return None

    def list_all(self, state: str | None = None) -> list[TicketLink]:
        """Every link, oldest first; ``state`` narrows to one state."""
        self._load()
        if state is None:
            return list(self._tickets)
        return [t for t in self._tickets if t.state == state]

    def update(self, link: TicketLink) -> None:
        """Persist a link's changed fields (matched by ref)."""
        self._load()
        for i, current in enumerate(self._tickets):
            if current.ref == link.ref:
                self._tickets[i] = link
                self._save()
                return
        raise RelayError(f"ticket {link.ref} is not on record")

    def record_failure(self, ref: str, reason: str) -> None:
        """Note a failed filing/sync on the link, durably.

        The link's state becomes ``failed`` so a later retry knows
        the last attempt did not land — a failure is never silent.
        """
        link = self.get(ref)
        if link is None:
            return
        link.failures.append({
            "at": time.time(), "reason": str(reason)[:500],
        })
        link.state = "failed"
        self.update(link)
