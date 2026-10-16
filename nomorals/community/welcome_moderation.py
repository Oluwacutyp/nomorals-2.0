"""Welcome / goodbye flows + group moderation + captcha-gated joins.

Three pieces that make a group feel run instead of abandoned:

**Welcome / goodbye** — a join is a pipeline, not a ping:
``rules → intro → role``. The new member sees the rules first,
then an intro prompt (who are you, what do you bring), and only then
gets their starting role. Leaving gets a goodbye, not silence.
Templates use ``{member}`` / ``{group}`` / ``{rules}`` placeholders.

**Moderation** — deterministic, appealable, cross-group:
* flood: > N messages inside a sliding window;
* links: any URL whose domain is not on the per-group allowlist;
* repeated: the same normalised text K times in a row / window;
* cross-group bans: a :class:`BanList` shared across groups, so a
  spammer banned in one group does not get a clean slate in the next.
Every action is logged; every ban carries an :class:`Appeal` path —
a banned member can file an appeal, a moderator reviews it, and an
approved appeal lifts the ban. No silent permanent bans.

**Captcha gate on join** — joins pass through a pluggable
:class:`CaptchaChecker` (protocol: ``check(user_id, answer) -> bool``).
The real checker can be the captcha pipeline; tests inject a fake.
Identity boundary (hard, enforced in code): trial / community
identities are **self-generated only — never the owner's real email
or phone**. :const:`ENFORCE_SELF_GENERATED_IDENTITIES` is ``True``
and cannot be switched off at runtime; :func:`validate_identity`
rejects any identity that carries owner-real contact details or
claims to be the owner.

**Roster intelligence** — from join/leave/message events, the roster
classifies members as ``active`` (messaged in the last 7 days),
``silent`` (joined > 14 days ago, never messaged), ``new`` (joined
in the last 7 days), ``left``, or ``answering`` (replied to a
question). No owner data: members are platform ids / display names
that arrived with the chat message itself.

Isolation: stdlib + ``nomorals.core`` only. No memory, accounts,
vaults, or connectors — the community isolation test is the gate.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "ENFORCE_SELF_GENERATED_IDENTITIES",
    "WelcomeConfig",
    "CaptchaChecker",
    "CaptchaGate",
    "ModerationConfig",
    "ModerationAction",
    "Ban",
    "Appeal",
    "BanList",
    "Moderator",
    "Roster",
    "render_welcome",
    "render_goodbye",
    "welcome_flow",
    "validate_identity",
    "IdentityError",
]


#: Hard identity boundary — self-generated identities only, never the
#: owner's real email/phone. Deliberately a module constant, not a
#: setting: there is no supported way to turn this off.
ENFORCE_SELF_GENERATED_IDENTITIES = True

_ACTIVE_WINDOW_S = 7 * 86400
_SILENT_AFTER_S = 14 * 86400


class IdentityError(ValueError):
    """Raised when an identity violates the self-generated-only boundary."""


# ── welcome / goodbye ──────────────────────────────────────────────────

@dataclass
class WelcomeConfig:
    """Per-group welcome configuration."""

    group_name: str = ""
    rules: list[str] = field(default_factory=list)
    intro_prompt: str = (
        "👋 Welcome {member} to {group}! Introduce yourself: "
        "who are you, and what do you bring?"
    )
    welcome_template: str = (
        "Welcome {member} to {group}! 🎉\n{rules}\n{intro}"
    )
    goodbye_template: str = "👋 {member} left {group}. Thanks for stopping by!"
    default_role: str = "member"
    # role granted after the member completes the intro step
    role_after_intro: str = "member"
    dm_welcome: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WelcomeConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def _rules_block(rules: list[str]) -> str:
    # No default rules. Rules are owner-configured only — empty means
    # no rules, not a fallback boilerplate set.
    if not rules:
        return ""
    return "📜 Rules:\n" + "\n".join(f"  {i}. {r}" for i, r in enumerate(rules, 1))


def render_welcome(member: str, config: WelcomeConfig) -> str:
    intro = config.intro_prompt.format(member=member, group=config.group_name)
    return config.welcome_template.format(
        member=member, group=config.group_name,
        rules=_rules_block(config.rules), intro=intro,
    )


def render_goodbye(member: str, config: WelcomeConfig) -> str:
    return config.goodbye_template.format(member=member, group=config.group_name)


def welcome_flow(member: str, config: WelcomeConfig) -> list[dict[str, Any]]:
    """Join pipeline: rules → intro → role. Ordered steps, each with text."""
    intro = config.intro_prompt.format(member=member, group=config.group_name)
    return [
        {
            "step": "rules",
            "order": 1,
            "text": _rules_block(config.rules),
            "requires_ack": True,
        },
        {"step": "intro", "order": 2, "text": intro, "requires_reply": True},
        {
            "step": "role",
            "order": 3,
            "text": f"You are now a {config.default_role} in {config.group_name}.",
            "role": config.default_role,
        },
    ]


# ── identity boundary ──────────────────────────────────────────────────

_OWNER_REAL_KEYS = (
    "owner_real_email", "owner_real_phone", "owner_email", "owner_phone",
    "real_email", "real_phone", "owner_contact",
)


def validate_identity(identity: dict[str, Any] | None) -> dict[str, Any]:
    """Enforce self-generated-identities-only.

    Rejects any identity that carries the owner's real contact
    details, claims to *be* the owner, or asks to disable the
    boundary. Returns the identity unchanged on success.
    """
    if not ENFORCE_SELF_GENERATED_IDENTITIES:
        # Unreachable by construction — the constant is not a setting.
        raise IdentityError("identity boundary disabled (unsupported)")
    ident = dict(identity or {})
    if ident.get("is_owner") or ident.get("claims_owner"):
        raise IdentityError("community identity must not claim to be the owner")
    if ident.get("use_owner_real_contact") or ident.get("disable_identity_check"):
        raise IdentityError(
            "owner real contact is never used for community identities"
        )
    for key in _OWNER_REAL_KEYS:
        if ident.get(key):
            raise IdentityError(
                f"identity carries owner-real contact ({key}) — "
                "use a self-generated identity instead"
            )
    return ident


# ── captcha gate ───────────────────────────────────────────────────────

@runtime_checkable
class CaptchaChecker(Protocol):
    """Pluggable captcha checker: ``check(user_id, answer) -> bool``."""

    def check(self, user_id: str, answer: str) -> bool: ...


class _AlwaysFailChecker:
    def check(self, user_id: str, answer: str) -> bool:
        return False


class CaptchaGate:
    """Join gate: pending → captcha verified → admitted.

    ``checker`` is injected (real captcha pipeline in production, a
    stub in tests). Attempts are capped; a member who fails too often
    stays pending until a moderator intervenes.
    """

    def __init__(
        self,
        checker: CaptchaChecker | None = None,
        *,
        max_attempts: int = 3,
        enabled: bool = True,
    ) -> None:
        self.checker = checker or _AlwaysFailChecker()
        self.max_attempts = max_attempts
        self.enabled = enabled
        self._pending: dict[str, dict[str, Any]] = {}
        self._admitted: set[str] = set()

    def request_join(
        self, user_id: str, *, identity: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Start a join. Identity boundary enforced before any captcha."""
        validate_identity(identity)
        if not self.enabled:
            self._admitted.add(user_id)
            return {"user_id": user_id, "state": "admitted", "captcha_required": False}
        self._pending[user_id] = {"attempts": 0, "created": time.time()}
        return {"user_id": user_id, "state": "pending_captcha", "captcha_required": True}

    def verify(self, user_id: str, answer: str) -> dict[str, Any]:
        entry = self._pending.get(user_id)
        if entry is None:
            if user_id in self._admitted:
                return {"user_id": user_id, "state": "admitted", "captcha_required": False}
            return {"user_id": user_id, "state": "unknown", "captcha_required": True}
        entry["attempts"] += 1
        if entry["attempts"] > self.max_attempts:
            return {
                "user_id": user_id, "state": "blocked",
                "reason": "too many captcha attempts — ask a moderator",
            }
        if self.checker.check(user_id, answer):
            del self._pending[user_id]
            self._admitted.add(user_id)
            return {"user_id": user_id, "state": "admitted", "captcha_required": False}
        return {
            "user_id": user_id, "state": "pending_captcha",
            "attempts": entry["attempts"],
        }

    def is_admitted(self, user_id: str) -> bool:
        return user_id in self._admitted

    def pending(self) -> list[str]:
        return sorted(self._pending)


# ── moderation ─────────────────────────────────────────────────────────

@dataclass
class ModerationConfig:
    flood_window_s: float = 30.0
    flood_max: int = 8  # > N messages inside the window → flood
    link_allowlist: list[str] = field(default_factory=list)  # domains
    repeat_threshold: int = 3  # same text this many times → repeated
    repeat_window_s: float = 300.0
    warn_before_mute: bool = True


@dataclass
class ModerationAction:
    kind: str  # "allow" | "warn" | "delete" | "mute" | "ban"
    reason: str = ""
    rule: str = ""
    user_id: str = ""
    group_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ban:
    id: str = ""
    user_id: str = ""
    group_id: str = ""  # "" = global (cross-group)
    scope: str = "group"  # "group" | "global"
    reason: str = ""
    banned_by: str = ""
    created_at: float = 0.0
    lifted_at: float | None = None
    lifted_by: str = ""
    appeal_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def active(self) -> bool:
        return self.lifted_at is None


@dataclass
class Appeal:
    id: str = ""
    ban_id: str = ""
    user_id: str = ""
    group_id: str = ""
    reason: str = ""
    created_at: float = 0.0
    status: str = "open"  # "open" | "approved" | "rejected"
    reviewed_by: str = ""
    review_note: str = ""
    reviewed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_URL_RE = re.compile(r"https?://([A-Za-z0-9.-]+)(?::\d+)?(?:[/?#]|$)", re.I)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class BanList:
    """Cross-group ban list. One list, shared across groups.

    A ``global`` ban applies everywhere; a ``group`` ban applies to
    one group. Appeals live here too so the appeal path is never
    lost when a moderator rotates.
    """

    def __init__(self) -> None:
        self._bans: dict[str, Ban] = {}
        self._appeals: dict[str, Appeal] = {}
        self._seq = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:06d}"

    def ban(
        self, user_id: str, *, reason: str, group_id: str = "",
        scope: str = "group", banned_by: str = "",
    ) -> Ban:
        if scope not in ("group", "global"):
            raise ValueError(f"scope must be group|global, got {scope!r}")
        ban = Ban(
            id=self._next_id("ban"), user_id=user_id,
            group_id="" if scope == "global" else group_id,
            scope=scope, reason=reason, banned_by=banned_by,
            created_at=time.time(),
        )
        self._bans[ban.id] = ban
        return ban

    def is_banned(self, user_id: str, group_id: str = "") -> Ban | None:
        for ban in self._bans.values():
            if not ban.active or ban.user_id != user_id:
                continue
            if ban.scope == "global":
                return ban
            if group_id and ban.group_id == group_id:
                return ban
        return None

    def lift(self, ban_id: str, *, by: str = "") -> Ban | None:
        ban = self._bans.get(ban_id)
        if ban and ban.active:
            ban.lifted_at = time.time()
            ban.lifted_by = by
        return ban

    def bans_for(self, user_id: str) -> list[Ban]:
        return [b for b in self._bans.values() if b.user_id == user_id]

    # ── appeals ────────────────────────────────────────────────────

    def file_appeal(self, user_id: str, ban_id: str, reason: str) -> Appeal:
        ban = self._bans.get(ban_id)
        if ban is None or ban.user_id != user_id:
            raise ValueError("no such ban for this user — cannot appeal")
        if not ban.active:
            raise ValueError("ban already lifted — nothing to appeal")
        appeal = Appeal(
            id=self._next_id("appeal"), ban_id=ban_id, user_id=user_id,
            group_id=ban.group_id, reason=reason, created_at=time.time(),
        )
        self._appeals[appeal.id] = appeal
        ban.appeal_id = appeal.id
        return appeal

    def review_appeal(
        self, appeal_id: str, *, approve: bool, reviewer: str = "", note: str = ""
    ) -> Appeal:
        appeal = self._appeals.get(appeal_id)
        if appeal is None:
            raise ValueError(f"unknown appeal {appeal_id!r}")
        if appeal.status != "open":
            raise ValueError(f"appeal {appeal_id!r} already {appeal.status}")
        appeal.status = "approved" if approve else "rejected"
        appeal.reviewed_by = reviewer
        appeal.review_note = note
        appeal.reviewed_at = time.time()
        if approve:
            self.lift(appeal.ban_id, by=reviewer)
        return appeal

    def appeals(self, *, status: str | None = None) -> list[Appeal]:
        rows = list(self._appeals.values())
        if status:
            rows = [a for a in rows if a.status == status]
        return sorted(rows, key=lambda a: a.created_at)

    def get_appeal(self, appeal_id: str) -> Appeal | None:
        return self._appeals.get(appeal_id)


class Moderator:
    """Deterministic per-group moderation over a shared BanList."""

    def __init__(
        self,
        ban_list: BanList | None = None,
        config: ModerationConfig | None = None,
    ) -> None:
        self.bans = ban_list or BanList()
        self.config = config or ModerationConfig()
        self._recent: dict[tuple[str, str], deque] = defaultdict(deque)
        self._recent_text: dict[tuple[str, str], deque] = defaultdict(deque)
        self._warned: set[tuple[str, str]] = set()
        self.log: list[dict[str, Any]] = []

    def _log(self, action: ModerationAction) -> None:
        self.log.append({**action.to_dict(), "ts": time.time()})

    def check_message(
        self, group_id: str, user_id: str, text: str, ts: float | None = None,
    ) -> list[ModerationAction]:
        """Evaluate one message. Returns actions (empty = allow)."""
        now = ts if ts is not None else time.time()
        cfg = self.config
        actions: list[ModerationAction] = []
        banned = self.bans.is_banned(user_id, group_id)
        if banned:
            act = ModerationAction(
                kind="ban", reason=f"already banned: {banned.reason}",
                rule="banned", user_id=user_id, group_id=group_id,
            )
            actions.append(act)
            self._log(act)
            return actions
        # Flood: sliding window per (group, user).
        key = (group_id, user_id)
        window = self._recent[key]
        window.append(now)
        while window and window[0] < now - cfg.flood_window_s:
            window.popleft()
        if len(window) > cfg.flood_max:
            act = ModerationAction(
                kind="mute", reason=f"flood: {len(window)} msgs in {cfg.flood_window_s:.0f}s",
                rule="flood", user_id=user_id, group_id=group_id,
            )
            actions.append(act)
            self._log(act)
            return actions
        # Links: any URL whose domain is not allowlisted.
        for domain in _URL_RE.findall(text or ""):
            domain = domain.lower().lstrip("www.")
            allowed = any(
                domain == allow.lower().lstrip("www.")
                or domain.endswith("." + allow.lower().lstrip("www."))
                for allow in cfg.link_allowlist
            )
            if cfg.link_allowlist and not allowed:
                act = ModerationAction(
                    kind="delete", reason=f"link to non-allowlisted domain {domain}",
                    rule="link_allowlist", user_id=user_id, group_id=group_id,
                )
                actions.append(act)
                self._log(act)
                return actions
        # Repeated: same normalised text threshold times in window.
        norm = _normalise(text)
        if norm:
            bucket = self._recent_text[key]
            bucket.append((now, norm))
            while bucket and bucket[0][0] < now - cfg.repeat_window_s:
                bucket.popleft()
            same = sum(1 for _, t in bucket if t == norm)
            if same >= cfg.repeat_threshold:
                if cfg.warn_before_mute and key not in self._warned:
                    self._warned.add(key)
                    act = ModerationAction(
                        kind="warn", reason=f"repeated message ×{same}",
                        rule="repeated", user_id=user_id, group_id=group_id,
                    )
                else:
                    act = ModerationAction(
                        kind="delete", reason=f"repeated message ×{same}",
                        rule="repeated", user_id=user_id, group_id=group_id,
                    )
                actions.append(act)
                self._log(act)
                return actions
        return actions

    def ban_member(
        self, group_id: str, user_id: str, *, reason: str,
        scope: str = "group", banned_by: str = "",
    ) -> Ban:
        ban = self.bans.ban(
            user_id, reason=reason, group_id=group_id, scope=scope, banned_by=banned_by,
        )
        self._log(ModerationAction(
            kind="ban", reason=reason, rule="manual",
            user_id=user_id, group_id=group_id,
        ))
        return ban

    def appeal(self, user_id: str, ban_id: str, reason: str) -> Appeal:
        """Appeal path: every ban can be appealed, always."""
        return self.bans.file_appeal(user_id, ban_id, reason)


# ── roster intelligence ────────────────────────────────────────────────

class Roster:
    """Member buckets from join/leave/message events. Deterministic."""

    def __init__(self) -> None:
        self._joined: dict[str, float] = {}
        self._left: dict[str, float] = {}
        self._last_msg: dict[str, float] = {}
        self._msg_count: dict[str, int] = defaultdict(int)
        self._replies_to_questions: dict[str, int] = defaultdict(int)
        self._names: dict[str, str] = {}

    def record_join(self, user_id: str, *, name: str = "", ts: float | None = None) -> None:
        self._joined[user_id] = ts if ts is not None else time.time()
        self._left.pop(user_id, None)
        if name:
            self._names[user_id] = name

    def record_leave(self, user_id: str, *, ts: float | None = None) -> None:
        self._left[user_id] = ts if ts is not None else time.time()

    def record_message(
        self, user_id: str, *, ts: float | None = None,
        name: str = "", reply_to_question: bool = False,
    ) -> None:
        now = ts if ts is not None else time.time()
        self._last_msg[user_id] = now
        self._msg_count[user_id] += 1
        if reply_to_question:
            self._replies_to_questions[user_id] += 1
        if name:
            self._names[user_id] = name
        self._joined.setdefault(user_id, now)

    def buckets(self, now: float | None = None) -> dict[str, list[str]]:
        now = now if now is not None else time.time()
        out: dict[str, list[str]] = {
            "active": [], "silent": [], "new": [], "left": [], "answering": [],
        }
        for uid in sorted(self._joined):
            if uid in self._left:
                out["left"].append(uid)
                continue
            last = self._last_msg.get(uid)
            joined = self._joined[uid]
            if last is not None and now - last <= _ACTIVE_WINDOW_S:
                out["active"].append(uid)
            elif now - joined <= _ACTIVE_WINDOW_S:
                out["new"].append(uid)
            elif now - joined > _SILENT_AFTER_S and last is None:
                out["silent"].append(uid)
            elif last is not None and now - last > _ACTIVE_WINDOW_S:
                out["silent"].append(uid)
            else:
                out["new"].append(uid)
            if self._replies_to_questions.get(uid):
                out["answering"].append(uid)
        return out

    def report(self, now: float | None = None) -> dict[str, Any]:
        b = self.buckets(now)
        top_answerers = sorted(
            self._replies_to_questions.items(), key=lambda kv: (-kv[1], kv[0])
        )[:5]
        return {
            "counts": {k: len(v) for k, v in b.items()},
            "buckets": b,
            "names": {uid: self._names.get(uid, uid) for uid in b["active"]},
            "top_answerers": [
                {"user_id": uid, "answers": n} for uid, n in top_answerers
            ],
            "total_members": len(self._joined) - len(b["left"]),
        }
