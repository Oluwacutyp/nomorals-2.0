"""The ticket relay engine: Discord tickets ↔ GitHub issues.

The engine owns the loop and nothing else. It takes its two seams
as arguments — a GitHub client with the issue surface
(``create_issue`` / ``comment_issue`` / ``get_issue`` /
``list_issue_comments`` / ``close_issue``, the
:class:`nomorals.connectors.github.GitHubConnector` shape) and a
``send(chat_key, text)`` callable for posting back to Discord (the
live chat gateway's ``send``, resolved by the caller) — so tests ride
fakes and the layering test stays green, the same discipline as
``social.cross_relay``.

The loop, in both directions:

* **Discord → GitHub** (``open_ticket`` / ``post_comment``): a
  ticket is filed as an issue whose body carries the submission
  marker; a Discord reply becomes an issue comment carrying the
  relay footer. The link is written *before* the issue is created,
  so a refused create leaves a ``failed`` link with the reason —
  never a fabricated issue link, never a second filing for the
  same ticket ref.
* **GitHub → Discord** (``sync_ticket`` / ``sync_all``): the issue
  is polled (upstream's own sync is polling too — its GitHub
  webhook is optional; Devon has no public webhook endpoint, so
  polling is the honest shape here). New comments and state changes
  are posted back to the ticket's origin channel, each exactly
  once: seen comment ids are recorded, and the relay's own
  comments (they carry the footer) are never echoed back. With no
  ``send`` callable the events are queued on the link as
  ``pending_posts`` and delivered by the next sync that has one.
* **Close** (``close_ticket``): closes the issue and notes it in
  the Discord channel; closing an already-closed ticket answers
  honestly instead of failing.

Every method returns ``{"ok": True, ...}`` or
``{"ok": False, "reason": ...}`` — a bad ticket, a refused create,
a dead issue never raises, never silently drops.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from . import content as _content
from .store import RelayError, TicketLink, TicketRelayStore

__all__ = ["RELAY_COMMENT_FOOTER", "RelayEngine"]

#: Footer on every comment the relay posts to GitHub from Discord.
#: The sync reads it back to tell its own echoes from human comments.
RELAY_COMMENT_FOOTER = "— relayed from Discord by Devon"

#: ``owner/name`` — the only repo shape GitHub accepts here.
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

#: How much of a GitHub comment body rides back to Discord.
_COMMENT_PREVIEW = 500


def _clean(value: Any) -> str:
    return str(value or "").strip()


class RelayEngine:
    """One relay: a GitHub client, a Discord send, and the loop."""

    def __init__(self, github: Any, *,
                 send: Callable[[str, str], Any] | None = None) -> None:
        self.github = github
        self.send = send

    # ── Discord → GitHub ──────────────────────────────────────

    def open_ticket(
        self, store: TicketRelayStore, *, repo: str, title: str,
        body: str = "", reporter: str = "", reporter_id: str = "",
        origin_platform: str = "discord", origin_chat: str = "",
        origin_thread: str = "", ticket_type: str | None = None,
        labels: list[str] | None = None,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """File a ticket as a GitHub issue; idempotent per ticket ref.

        ``ref`` is the ticket's own id when the caller already minted
        one (a retry replays the same ref and gets the same issue
        back); without one the store mints it. The link is written
        first, so a refused create is a ``failed`` link with the
        reason, never a fabricated URL.
        """
        repo = _clean(repo)
        title = _clean(title)
        if not repo or not _REPO_RE.match(repo):
            return {"ok": False, "reason": (
                "a ticket needs a repo like owner/name — "
                f"got {repo!r}" if repo else
                "a ticket needs a repo like owner/name")}
        if not title:
            return {"ok": False, "reason": "a ticket needs a title"}
        if ref:
            existing = store.get(ref)
            if existing is not None and existing.state == "open":
                return {"ok": True, "ref": existing.ref,
                        "issue_number": existing.issue_number,
                        "issue_url": existing.issue_url,
                        "already": True}
            if existing is not None and existing.state == "failed":
                ref = None  # a failed filing re-files under a fresh ref
        if ref is None:
            try:
                ref = store.mint_ref()
            except RelayError as exc:
                return {"ok": False, "reason": str(exc)}
        link = TicketLink(
            ref=ref, repo=repo, title=title,
            reporter=_clean(reporter), reporter_id=_clean(reporter_id),
            origin_platform=_clean(origin_platform) or "discord",
            origin_chat=_clean(origin_chat),
            origin_thread=_clean(origin_thread),
            state="open",
        )
        try:
            store.add(link)
        except RelayError as exc:
            return {"ok": False, "reason": str(exc)}
        app_name = repo.split("/", 1)[1]
        issue_title = _content.build_issue_title(
            app_name, title, type_prefix=ticket_type)
        issue_body = _content.build_issue_body(
            ref=ref, app_name=app_name, title=title,
            description=body,
            reporter_display_name=link.reporter or "Discord member",
            reporter_discord_id=link.reporter_id,
            submitted_at=datetime.now(UTC),
            channel_id=link.origin_chat.split(":", 1)[-1]
            if ":" in link.origin_chat else link.origin_chat,
            ticket_type=ticket_type,
        )
        try:
            issue = self.github.create_issue(
                repo, issue_title, body=issue_body,
                labels=list(labels or []))
        except Exception as exc:  # noqa: BLE001 - the client is opaque
            reason = str(exc)[:500]
            store.record_failure(ref, reason)
            return {"ok": False, "ref": ref, "reason": reason}
        link.issue_number = int(issue.get("number") or 0)
        link.issue_url = str(issue.get("url") or "")
        link.last_issue_state = str(issue.get("state") or "open")
        link.last_comment_count = int(issue.get("comments") or 0)
        store.update(link)
        return {"ok": True, "ref": ref,
                "issue_number": link.issue_number,
                "issue_url": link.issue_url}

    def post_comment(self, store: TicketRelayStore, ref: str, *,
                     author: str = "", text: str = "") -> dict[str, Any]:
        """Post a Discord reply to the ticket's GitHub issue."""
        link = store.get(ref)
        if link is None:
            return {"ok": False,
                    "reason": f"no ticket {ref} on record"}
        if link.state == "failed" or not link.issue_number:
            return {"ok": False,
                    "reason": f"ticket {ref} was never filed — "
                              "its issue create failed"}
        text = _clean(text)
        if not text:
            return {"ok": False, "reason": "a comment needs text"}
        who = _clean(author) or link.reporter or "Discord member"
        body = (
            f"{who} (via Discord):\n\n"
            f"{_content.neutralize_github_mentions(text)}\n\n"
            f"{RELAY_COMMENT_FOOTER}"
        )
        try:
            comment = self.github.comment_issue(
                link.repo, link.issue_number, body)
        except Exception as exc:  # noqa: BLE001 - the client is opaque
            reason = str(exc)[:500]
            store.record_failure(ref, reason)
            return {"ok": False, "reason": reason}
        link.last_comment_count += 1
        store.update(link)
        return {"ok": True, "ref": ref,
                "issue_number": link.issue_number,
                "comment_id": comment.get("id") if isinstance(
                    comment, dict) else None}

    # ── GitHub → Discord ──────────────────────────────────────

    def _deliver(self, link: TicketLink, text: str) -> bool:
        """Post one sync event to the ticket's origin channel.

        With no send callable the event queues on the link as a
        pending post for the next sync that has one; a failed send
        queues too. Returns True only when the text went out.
        """
        if self.send is None:
            link.pending_posts.append(text)
            return False
        try:
            result = self.send(link.origin_chat, text)
        except Exception:  # noqa: BLE001 - the gateway is opaque
            link.pending_posts.append(text)
            return False
        ok = True
        if isinstance(result, dict):
            ok = bool(result.get("ok", True))
        elif hasattr(result, "ok"):
            ok = bool(result.ok)
        if not ok:
            link.pending_posts.append(text)
            return False
        return True

    def sync_ticket(self, store: TicketRelayStore,
                    ref: str) -> dict[str, Any]:
        """Pull one ticket's issue state/comments back to Discord.

        New human comments and state changes are posted exactly
        once; a second sync posts nothing. The relay's own comments
        (footer) are never echoed back.
        """
        link = store.get(ref)
        if link is None:
            return {"ok": False, "ref": ref,
                    "reason": f"no ticket {ref} on record"}
        if link.state == "failed" or not link.issue_number:
            return {"ok": False, "ref": link.ref,
                    "reason": f"ticket {ref} was never filed — "
                              "its issue create failed"}
        try:
            issue = self.github.get_issue(link.repo, link.issue_number)
            comments = self.github.list_issue_comments(
                link.repo, link.issue_number)
        except Exception as exc:  # noqa: BLE001 - the client is opaque
            reason = str(exc)[:500]
            store.record_failure(ref, reason)
            return {"ok": False, "ref": link.ref, "reason": reason}
        posted = 0
        pending = list(link.pending_posts)
        link.pending_posts = []
        for text in pending:
            if self._deliver(link, text):
                posted += 1
        new_state = str(issue.get("state") or "")
        if new_state and new_state != link.last_issue_state:
            if self._deliver(
                    link,
                    f"Ticket {link.ref}: {link.repo}#{link.issue_number} "
                    f"is now {new_state}."):
                posted += 1
            link.last_issue_state = new_state
        seen = {str(c) for c in link.seen_comment_ids}
        for comment in comments or []:
            cid = str(comment.get("id", ""))
            body = str(comment.get("body", "") or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            if RELAY_COMMENT_FOOTER in body:
                continue
            author = (comment.get("user") or {}).get("login", "")
            preview = body.strip().replace("\n", " ")
            if len(preview) > _COMMENT_PREVIEW:
                preview = preview[:_COMMENT_PREVIEW] + "…"
            who = f" by {author}" if author else ""
            if self._deliver(
                    link,
                    f"New comment on {link.repo}#{link.issue_number}"
                    f"{who}: {preview}"):
                posted += 1
        link.seen_comment_ids = sorted(seen)
        link.last_comment_count = len(comments or [])
        link.last_issue_state = new_state or link.last_issue_state
        link.last_synced_at = time.time()
        store.update(link)
        return {"ok": True, "ref": link.ref, "posted": posted}

    def sync_all(self, store: TicketRelayStore) -> dict[str, Any]:
        """Sync every open ticket; one dead ticket never kills the rest."""
        results: list[dict[str, Any]] = []
        posted = 0
        for link in store.list_all(state="open"):
            try:
                one = self.sync_ticket(store, link.ref)
            except Exception as exc:  # noqa: BLE001 - never kill the run
                one = {"ok": False, "ref": link.ref, "reason": str(exc)}
            results.append(one)
            posted += int(one.get("posted", 0) or 0)
        return {"ok": True, "posted": posted, "results": results}

    def close_ticket(self, store: TicketRelayStore, ref: str, *,
                     by: str = "") -> dict[str, Any]:
        """Close the ticket's issue and note it in the Discord channel."""
        link = store.get(ref)
        if link is None:
            return {"ok": False,
                    "reason": f"no ticket {ref} on record"}
        if link.state == "closed":
            return {"ok": True, "ref": ref, "already": True,
                    "issue_number": link.issue_number}
        if link.state == "failed" or not link.issue_number:
            return {"ok": False,
                    "reason": f"ticket {ref} was never filed — "
                              "its issue create failed"}
        try:
            self.github.close_issue(link.repo, link.issue_number)
        except Exception as exc:  # noqa: BLE001 - the client is opaque
            reason = str(exc)[:500]
            store.record_failure(ref, reason)
            return {"ok": False, "reason": reason}
        link.state = "closed"
        link.last_issue_state = "closed"
        store.update(link)
        who = f" by {_clean(by)}" if _clean(by) else ""
        self._deliver(
            link,
            f"Ticket {link.ref} closed{who}: "
            f"{link.repo}#{link.issue_number}.")
        store.update(link)
        return {"ok": True, "ref": ref,
                "issue_number": link.issue_number}
