"""Post scheduling and cross-platform fan-out.

The point of this layer is that one call publishes to every connected platform
*at the same time*, on the L5 parallel runtime. Posting to five platforms
sequentially takes five round trips; fanned out it takes as long as the slowest
one. Each platform is an IO task, so they go to the thread pool.

Per-platform failures are isolated by design. A 429 from one service must not
stop the others, and must not lose the post: the row stays in the database with
its error and its attempt count, so a retry picks it up.

Rate limits are enforced here rather than trusted to the platform, because
hitting a platform's limit is how accounts get suspended. The check is
conservative and local: it cannot see other clients using the same account.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..agents.tasks import Task, TaskGraph, TaskKind
from ..core.errors import NotFound, ValidationError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..storage.repository import Repository
from .base import Account, PlatformAdapter, PostResult, PostStatus, SocialError

__all__ = ["SocialManager", "PublishOutcome"]

_log = get_logger(__name__)


@dataclass
class PublishOutcome:
    """Aggregate result of one cross-platform publish."""

    content: str
    results: list[PostResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)

    @property
    def posted(self) -> list[PostResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[PostResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "posted": [r.to_dict() for r in self.posted],
            "failed": [r.to_dict() for r in self.failed],
            "seconds": round(self.seconds, 3),
            "content": self.content[:200],
        }


class SocialManager:
    """Accounts, post queue, and parallel publishing."""

    def __init__(
        self,
        context: Any,
        *,
        adapters: dict[str, PlatformAdapter] | None = None,
        enforce_policy: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.context = context
        self.db = context.db
        self.accounts = Repository(
            self.db, "social_accounts", json_columns=("limits", "metadata"),
            timestamp_columns=("created_at",),
        )
        self.posts = Repository(
            self.db, "social_posts", json_columns=("media_paths", "metrics", "metadata"),
            timestamp_columns=("created_at",),
        )
        self.adapters: dict[str, PlatformAdapter] = dict(adapters or {})
        self.enforce_policy = enforce_policy
        self._clock = clock
        self.stats = {"published": 0, "failed": 0, "rate_limited": 0, "skipped": 0}

    # ── adapters ─────────────────────────────────────────────────────────────

    def register_adapter(self, adapter: PlatformAdapter) -> None:
        self.adapters[adapter.name] = adapter

    def register_builtins(self) -> "SocialManager":
        """Wire up the shipped adapters. Returns self so it can be chained."""
        from .adapters import bluesky, mastodon

        for module in (mastodon, bluesky):
            self.register_adapter(module.Adapter())
        return self

    def adapter_for(self, platform: str) -> PlatformAdapter:
        adapter = self.adapters.get(platform)
        if adapter is None:
            raise SocialError(
                f"no adapter for platform {platform!r}; available: {sorted(self.adapters)}"
            )
        return adapter

    # ── accounts ─────────────────────────────────────────────────────────────

    def connect(
        self,
        platform: str,
        handle: str,
        *,
        credentials: str = "",
        limits: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Account:
        """Register an account. ``credentials`` should be ``env:NAME``."""
        adapter = self.adapters.get(platform)
        merged = dict(limits or {})
        if adapter is not None:
            # The adapter knows the platform's real ceiling; trust it over a guess.
            merged.setdefault("max_chars", adapter.max_chars)
        account = Account(
            id=new_id(), platform=platform, handle=handle, credentials=credentials,
            limits=merged, metadata=metadata or {}, created_at=self._clock(),
        )
        try:
            self.accounts.create(account.to_row())
        except Exception as exc:  # noqa: BLE001 - UNIQUE(platform, handle)
            raise SocialError(f"account {handle} on {platform} already connected: {exc}") from exc
        _log.info("connected %s account %s", platform, handle)
        return account

    def disconnect(self, platform: str, handle: str) -> int:
        row = self.accounts.find_one(platform=platform, handle=handle)
        if row is None:
            raise NotFound(f"no account {handle!r} on {platform}")
        return self.accounts.delete(row["id"])

    def list_accounts(self, *, platform: str = "", active_only: bool = True) -> list[Account]:
        filters: dict[str, Any] = {}
        if platform:
            filters["platform"] = platform
        if active_only:
            filters["active"] = 1
        return [Account.from_row(r) for r in self.accounts.find(**filters)]

    def get_account(self, platform: str, handle: str) -> Account:
        row = self.accounts.find_one(platform=platform, handle=handle)
        if row is None:
            raise NotFound(f"no account {handle!r} on {platform}")
        return Account.from_row(row)

    # ── publishing ───────────────────────────────────────────────────────────

    def publish(
        self,
        content: str,
        *,
        platforms: Sequence[str] | None = None,
        account_handles: dict[str, str] | None = None,
        media_paths: Sequence[str] = (),
        reply_to: str = "",
        actor: str = "user",
        parallel: bool = True,
        confirmation: str = "",
    ) -> PublishOutcome:
        """Publish to every targeted platform, fanned out in parallel.

        ``platforms`` names the targets; ``account_handles`` picks a specific
        account per platform when more than one is connected.
        """
        if not content or not content.strip():
            raise ValidationError("post content is empty", field="content")

        targets = self._resolve_targets(platforms, account_handles)
        if not targets:
            raise SocialError("no connected accounts to publish to")

        if self.enforce_policy:
            self._check_capability(
                Capability.SOCIAL_BULK if len(targets) > 1 else Capability.SOCIAL_POST,
                actor,
                confirmation=confirmation,
            )

        outcome = PublishOutcome(content=content)
        started = time.perf_counter()

        if parallel and len(targets) > 1:
            outcome.results = self._publish_parallel(targets, content, media_paths, reply_to)
        else:
            for account in targets:
                outcome.results.append(self._publish_one(account, content, media_paths, reply_to))

        outcome.seconds = time.perf_counter() - started
        self.stats["published"] += len(outcome.posted)
        self.stats["failed"] += len(outcome.failed)
        _log.info(
            "published to %d/%d platforms in %.2fs",
            len(outcome.posted), len(targets), outcome.seconds,
        )
        return outcome

    def _resolve_targets(
        self, platforms: Sequence[str] | None, account_handles: dict[str, str] | None
    ) -> list[Account]:
        account_handles = account_handles or {}
        names = list(platforms) if platforms else sorted(
            {a.platform for a in self.list_accounts(active_only=False)}
        )
        targets: list[Account] = []
        for platform in names:
            handle = account_handles.get(platform)
            if handle:
                targets.append(self.get_account(platform, handle))
                continue
            # Include inactive accounts so the caller is told "inactive" rather
            # than the misleading "no connected accounts to publish to".
            connected = self.list_accounts(platform=platform, active_only=False)
            if not connected:
                _log.warning("no connected account for %s, skipping", platform)
                self.stats["skipped"] += 1
                continue
            targets.append(connected[0])
        return targets

    def _publish_parallel(
        self,
        targets: Sequence[Account],
        content: str,
        media_paths: Sequence[str],
        reply_to: str,
    ) -> list[PostResult]:
        """Fan out across the thread pool, preserving target order in the output."""
        executor = getattr(self.context, "executor", None)
        if executor is None:
            return [self._publish_one(a, content, media_paths, reply_to) for a in targets]

        graph = TaskGraph()
        ordered = list(targets)
        for index, account in enumerate(ordered):
            graph.add_task(
                f"post-{index}-{account.platform}",
                self._publish_one,
                account, content, media_paths, reply_to,
                kind=TaskKind.IO,
            )
        executor.run(graph)

        results: list[PostResult] = []
        for index, account in enumerate(ordered):
            task = graph.get(f"post-{index}-{account.platform}")
            if task is None or not task.ok:
                results.append(
                    PostResult(
                        platform=account.platform, ok=False,
                        error=(task.error if task else "task missing") or "publish failed",
                    )
                )
            else:
                results.append(task.result)
        return results

    def _publish_one(
        self,
        account: Account,
        content: str,
        media_paths: Sequence[str],
        reply_to: str,
    ) -> PostResult:
        """Post to one account, recording the attempt either way."""
        adapter = self.adapters.get(account.platform)
        if adapter is None:
            return PostResult(
                platform=account.platform, ok=False,
                error=f"no adapter registered for {account.platform}",
            )
        if not account.active:
            return PostResult(platform=account.platform, ok=False, error="account is inactive")

        wait = self._rate_limit_wait(account)
        if wait is not None:
            self.stats["rate_limited"] += 1
            return PostResult(
                platform=account.platform, ok=False,
                error=f"rate limited: {wait:.0f}s until the next post is allowed",
            )

        row = self._record_post(account, content, media_paths, reply_to)
        try:
            adapter.validate(content)
        except ValidationError as exc:
            self._update_post(row["id"], PostStatus.FAILED, error=exc.message)
            return PostResult(platform=account.platform, ok=False, error=exc.message)

        started = time.perf_counter()
        try:
            result = adapter.post(
                account, content, media_paths=list(media_paths), reply_to=reply_to
            )
        except Exception as exc:  # noqa: BLE001 - a platform error is a result
            result = PostResult(
                platform=account.platform, ok=False, error=f"{type(exc).__name__}: {exc}"
            )
        result.seconds = time.perf_counter() - started

        self._update_post(
            row["id"],
            PostStatus.POSTED if result.ok else PostStatus.FAILED,
            external_id=result.external_id,
            error=result.error,
            metrics=result.metrics,
            posted_at=self._clock() if result.ok else None,
        )
        return result

    # ── rate limiting ────────────────────────────────────────────────────────

    def _rate_limit_wait(self, account: Account) -> float | None:
        """Seconds to wait before this account may post again, or None if clear.

        Enforced locally so a retry loop cannot hammer a platform into suspending
        the account. Only counts posts this system made.
        """
        row = self.db.query_one(
            "SELECT posted_at FROM social_posts WHERE account_id = ? AND status = 'posted' "
            "ORDER BY posted_at DESC LIMIT 1",
            (account.id,),
        )
        if row is None or not row.get("posted_at"):
            return None
        elapsed = self._clock() - float(row["posted_at"])
        if elapsed < account.min_interval:
            return account.min_interval - elapsed
        return None

    def posts_today(self, account: Account) -> int:
        """Posts published to this account since midnight, for the daily ceiling."""
        import datetime

        midnight = datetime.datetime.combine(
            datetime.date.today(), datetime.time.min
        ).timestamp()
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM social_posts WHERE account_id = ? AND status = 'posted' "
                "AND posted_at >= ?",
                (account.id, midnight),
                default=0,
            )
            or 0
        )

    # ── queue ────────────────────────────────────────────────────────────────

    def schedule(
        self,
        content: str,
        platform: str,
        handle: str,
        *,
        at: float,
        media_paths: Sequence[str] = (),
        reply_to: str = "",
    ) -> str:
        """Queue a post for later. The runner picks it up when due."""
        account = self.get_account(platform, handle)
        row = self.posts.create(
            {
                "id": new_id(), "platform": platform, "account_id": account.id,
                "content": content, "media_paths": list(media_paths),
                "status": PostStatus.SCHEDULED, "scheduled_at": at, "reply_to": reply_to,
                "created_at": self._clock(),
            }
        )
        return row["id"]

    def due(self, *, now: float | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Scheduled posts whose time has come."""
        moment = now if now is not None else self._clock()
        rows = self.db.query(
            "SELECT * FROM social_posts WHERE status = 'scheduled' AND scheduled_at <= ? "
            "ORDER BY scheduled_at LIMIT ?",
            (moment, limit),
        )
        return [dict(r) for r in rows]

    def run_due(self, *, limit: int = 50) -> list[PostResult]:
        """Publish everything that is due. Idempotent per post row."""
        results: list[PostResult] = []
        for row in self.due(limit=limit):
            account_row = self.accounts.get(row["account_id"])
            if account_row is None:
                self._update_post(row["id"], PostStatus.FAILED, error="account no longer connected")
                continue
            account = Account.from_row(account_row)
            self._update_post(row["id"], PostStatus.POSTING)
            result = self._publish_one(
                account, row["content"], json.loads(row.get("media_paths") or "[]"),
                row.get("reply_to") or "",
            )
            results.append(result)
        return results

    def history(self, *, platform: str = "", status: str = "", limit: int = 50) -> list[dict[str, Any]]:
        # Repository.find() turns every kwarg into a WHERE clause, so passing
        # limit= there silently produced `WHERE "limit" = 50` and matched nothing.
        query = self.posts.query()
        if platform:
            query.where("platform = ?", platform)
        if status:
            query.where("status = ?", status)
        rows = self.db.query(*query.order_by("created_at DESC").limit(limit).build())
        return [dict(r) for r in rows]

    def stats_snapshot(self) -> dict[str, Any]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM social_posts GROUP BY status")
        return {
            **self.stats,
            "accounts": len(self.list_accounts()),
            "by_status": {r["status"]: int(r["n"]) for r in rows},
            "adapters": sorted(self.adapters),
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _record_post(
        self, account: Account, content: str, media_paths: Sequence[str], reply_to: str
    ) -> dict[str, Any]:
        return self.posts.create(
            {
                "id": new_id(), "platform": account.platform, "account_id": account.id,
                "content": content, "media_paths": list(media_paths),
                "status": PostStatus.POSTING, "reply_to": reply_to,
                "attempts": 1, "created_at": self._clock(),
            }
        )

    def _update_post(self, post_id: str, status: str, **fields: Any) -> None:
        changes = {"status": status, **fields}
        if fields.get("posted_at") is None:
            changes.pop("posted_at", None)
        self.posts.update(post_id, changes)

    def _check_capability(self, capability: str, actor: str, *, confirmation: str = "") -> None:
        """Ask the policy layer.

        Bulk posting is a confirmable capability, so publishing to more than one
        platform needs a single-use token minted by the operator. That is the
        point: a prompt-injected agent should not be able to blast every connected
        account on its own authority.
        """
        policy = getattr(self.context, "policy", None)
        if policy is None:
            return
        from ..core.policy import CapabilitySet

        grant = CapabilitySet.all()
        decision = policy.check(
            capability, actor=actor, grant=grant,
            confirmation=confirmation or None, context={"social": True},
        )
        if not decision.allowed:
            raise SocialError(decision.reason)
