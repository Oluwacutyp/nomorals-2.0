"""Deletion runbooks — AccountKiller (#28) as a maintained registry.

The exit side of the account estate. ``accounts/`` runs creation
(vault, temp mail/SMS, trial flows); nothing ran leaving. This organ
is the registry idiom pointed at leaving: a service -> deletion
runbook map, seeded from public directories and grown by discovery.

It is a registry, NOT a deleter. Devon guides and assists; the
account holder's own hand finishes at the genuine walls (login,
CAPTCHA, final confirm) — the same standing account doctrine as her
signup flows. Every step that needs the owner's hand is marked
``owner_hand`` and the walk stops there.

``DeletionGuide`` carries ``checked_at`` (ISO date). A guide older
than ``STALE_AFTER_DAYS`` is labelled stale, never served as current:
deletion flows churn, and a stale guide that pretends to be fresh is
worse than no guide. ``difficulty`` is one of easy / medium / hard /
impossible — "impossible" is a real rating in the source data and is
never softened.

Store: ``<home>/data/deletion_guides.json`` —
``{"guides": {slug: guide}, "walks": {slug: index},
"active": slug | null}``. ``active`` is the walk the owner is
currently on, so a bare "done" / "next" knows which runbook to
advance. Missing store seeds from :data:`SEEDS`; a corrupt store
is quarantined aside and re-seeded, never allowed to eat the
registry.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "DIFFICULTIES",
    "STALE_AFTER_DAYS",
    "DeletionGuide",
    "DeletionStep",
    "DeletionStore",
    "SEEDS",
    "is_stale",
    "parse_delete_request",
    "render_guide",
    "render_walk_step",
    "slugify_service",
    "walk_advance",
    "walk_current",
    "walk_reset",
]

DIFFICULTIES = ("easy", "medium", "hard", "impossible")
STALE_AFTER_DAYS = 180
_STORE_NAME = "deletion_guides.json"
_TODAY = "2026-11-18"


def slugify_service(name: str) -> str:
    """Canonical key for a service name: lowercased, spaces to dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    return slug.strip("-")


@dataclass(frozen=True)
class DeletionStep:
    """One step of a deletion runbook.

    ``owner_hand`` marks a step Devon cannot do: signing in, passing
    a CAPTCHA / 2FA check, or pressing the final confirm. The walk
    stops there and waits for the owner to say ``done``.
    """

    text: str
    owner_hand: bool = False


@dataclass
class DeletionGuide:
    """A deletion runbook for one service.

    ``grace_days`` is the window to change your mind after starting
    deletion (0 = no window: once it is gone, it is gone).
    ``export_first`` means the owner's data should be downloaded
    before anything is deleted — losing photos, mail or playlists is
    the irreversible part of leaving.
    """

    service: str
    aliases: tuple[str, ...] = ()
    delete_url: str = ""
    steps: tuple[DeletionStep, ...] = ()
    difficulty: str = "medium"
    grace_days: int = 0
    export_first: bool = True
    checked_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(
                f"bad difficulty {self.difficulty!r} for "
                f"{self.service!r}: use one of {DIFFICULTIES}"
            )
        if self.grace_days < 0:
            raise ValueError(
                f"bad grace_days {self.grace_days} for {self.service!r}"
            )
        if not self.steps:
            raise ValueError(f"guide for {self.service!r} has no steps")

    @property
    def slug(self) -> str:
        return slugify_service(self.service)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "aliases": list(self.aliases),
            "delete_url": self.delete_url,
            "steps": [
                {"text": step.text, "owner_hand": step.owner_hand}
                for step in self.steps
            ],
            "difficulty": self.difficulty,
            "grace_days": self.grace_days,
            "export_first": self.export_first,
            "checked_at": self.checked_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> DeletionGuide:
        steps = tuple(
            DeletionStep(
                text=str(step.get("text", "")).strip(),
                owner_hand=bool(step.get("owner_hand", False)),
            )
            for step in row.get("steps") or []
            if str(step.get("text", "")).strip()
        )
        return cls(
            service=str(row.get("service", "")).strip(),
            aliases=tuple(
                str(alias).strip() for alias in row.get("aliases") or []
            ),
            delete_url=str(row.get("delete_url", "")).strip(),
            steps=steps,
            difficulty=str(row.get("difficulty", "medium")).strip().lower(),
            grace_days=int(row.get("grace_days", 0)),
            export_first=bool(row.get("export_first", True)),
            checked_at=str(row.get("checked_at", "")).strip(),
            notes=str(row.get("notes", "")).strip(),
        )


def _step(text: str, hand: bool = False) -> DeletionStep:
    return DeletionStep(text=text, owner_hand=hand)


def _guide(
    service: str,
    steps: list[DeletionStep],
    *,
    aliases: tuple[str, ...] = (),
    delete_url: str = "",
    difficulty: str = "medium",
    grace_days: int = 0,
    export_first: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    return DeletionGuide(
        service=service,
        aliases=aliases,
        delete_url=delete_url,
        steps=tuple(steps),
        difficulty=difficulty,
        grace_days=grace_days,
        export_first=export_first,
        checked_at=_TODAY,
        notes=notes,
    ).to_dict()


# Seeds — checked 2026-11-18 against the services' own help pages and
# public deletion directories. Grace windows are the "change your mind"
# window after starting deletion; 0 means no window. A seed is a
# starting point: ``checked_at`` ages it out of currency after
# STALE_AFTER_DAYS unless a later discovery refreshes it.
SEEDS: list[dict[str, Any]] = [
    _guide(
        "google",
        [
            _step("Download your data first: takeout.google.com — "
                  "mail, Drive, Photos and YouTube go with the account."),
            _step("Open myaccount.google.com/delete-services-or-account "
                  "and sign in.", hand=True),
            _step("Choose Delete your Google Account (or delete a "
                  "single service instead)."),
            _step("Read what goes away with it — Gmail, YouTube, "
                  "Drive, Photos."),
            _step("Confirm with your password and complete any 2-step "
                  "check.", hand=True),
            _step("Click Delete account to finish.", hand=True),
        ],
        aliases=("gmail",),
        delete_url="https://myaccount.google.com/delete-services-or-account",
        difficulty="easy",
        grace_days=20,
        notes="Google may let you recover a recently deleted account "
        "for a short time — not guaranteed, so treat deletion as final.",
    ),
    _guide(
        "facebook",
        [
            _step("Download your information first: Settings → Your "
                  "information → Download your information."),
            _step("Go to Settings & privacy → Settings → Accounts "
                  "Center → Personal details → Account ownership and "
                  "control → Deactivation or deletion."),
            _step("Pick your Facebook profile, choose Delete account, "
                  "then Continue."),
            _step("Enter your password to confirm.", hand=True),
            _step("Click Delete account. Changed your mind? Log in "
                  "within 30 days and click Cancel deletion — simply "
                  "logging in does not cancel it on Facebook.",
                  hand=True),
        ],
        aliases=("fb", "meta"),
        delete_url="https://www.facebook.com/help/224562897555674",
        difficulty="medium",
        grace_days=30,
        notes="The old facebook.com/help/delete_account link no longer "
        "works — deletion moved into Accounts Center.",
    ),
    _guide(
        "instagram",
        [
            _step("Download your data first: Settings → Accounts "
                  "Centre → Your information and permissions → "
                  "Download your information."),
            _step("Open instagram.com/accounts/remove/request/"
                  "permanent/ in a browser — the app has no deletion "
                  "page."),
            _step("Pick a reason for leaving, re-enter your password.",
                  hand=True),
            _step("Click Delete. The account hides at once and is "
                  "gone after 30 days — do not log in during that "
                  "window.", hand=True),
        ],
        aliases=("ig",),
        delete_url="https://www.instagram.com/accounts/remove/request/permanent/",
        difficulty="medium",
        grace_days=30,
    ),
    _guide(
        "x",
        [
            _step("Download your archive first: Settings → Your "
                  "account → Download an archive of your data."),
            _step("Open x.com/settings/deactivate and sign in.",
                  hand=True),
            _step("Read what deactivation removes, enter your password, "
                  "click Deactivate.", hand=True),
            _step("Stay logged out for 30 days — logging back in "
                  "cancels the deletion."),
        ],
        aliases=("twitter",),
        delete_url="https://x.com/settings/deactivate",
        difficulty="easy",
        grace_days=30,
    ),
    _guide(
        "linkedin",
        [
            _step("Download your data first: Me → Settings & Privacy "
                  "→ Data privacy → Get a copy of your data."),
            _step("Cancel Premium first — a paid account cannot be "
                  "closed."),
            _step("Click Me → Settings & Privacy → Account preferences "
                  "→ Account management → Close account."),
            _step("Pick a reason, click Next, enter your password.",
                  hand=True),
            _step("Click Close account.", hand=True),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=14,
        notes="Reactivation window is 14 days per LinkedIn Help (older "
        "guides say 20). There is no stable direct deletion URL — "
        "the flow lives in Settings & Privacy.",
    ),
    _guide(
        "snapchat",
        [
            _step("Download your data first: Settings → My Data → "
                  "Submit Request."),
            _step("Open accounts.snapchat.com/accounts/delete_account "
                  "and sign in.", hand=True),
            _step("Confirm the deletion.", hand=True),
            _step("The account deactivates for 30 days, then is gone "
                  "for good. Do not log in during those 30 days."),
        ],
        aliases=("snap",),
        delete_url="https://accounts.snapchat.com/accounts/delete_account",
        difficulty="easy",
        grace_days=30,
    ),
    _guide(
        "tiktok",
        [
            _step("Download your data first: Profile → Menu → Settings "
                  "and privacy → Account → Download your data."),
            _step("In the app: Profile → Menu → Settings and privacy "
                  "→ Manage account → Delete account."),
            _step("Confirm with your password or verification code.",
                  hand=True),
            _step("The account deactivates for 30 days, then is "
                  "permanently deleted. Do not log in during that "
                  "window."),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=30,
        notes="Deletion is in-app only — there is no web deletion page.",
    ),
    _guide(
        "whatsapp",
        [
            _step("Back up your chats first: Settings → Chats → Chat "
                  "backup — deletion does not keep them."),
            _step("In the app: Settings → Account → Delete my account."),
            _step("Enter your phone number and tap Delete my account.",
                  hand=True),
            _step("Deletion is immediate; copies in backups can take "
                  "up to 90 days to disappear."),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=0,
        notes="Deletion is in-app only. Groups you created pass to "
        "another admin; your messages in other people's chats stay.",
    ),
    _guide(
        "telegram",
        [
            _step("Export your data first: desktop app → Settings → "
                  "Advanced → Export Telegram data."),
            _step("Open telegram.org/deactivate and sign in with your "
                  "phone number.", hand=True),
            _step("Enter the confirmation code Telegram sends you in "
                  "the app.", hand=True),
            _step("Click Delete My Account. There is no grace window — "
                  "it is gone immediately.", hand=True),
        ],
        delete_url="https://telegram.org/deactivate",
        difficulty="easy",
        grace_days=0,
    ),
    _guide(
        "discord",
        [
            _step("Download your data first: Settings → Privacy & "
                  "Safety → Request all of my Data."),
            _step("In the app: User Settings → My Account → Delete "
                  "Account."),
            _step("Enter your password (and 2FA code if set) and "
                  "confirm.", hand=True),
            _step("The account enters a pending state; log back in "
                  "within 14 days to cancel the deletion."),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=14,
        notes="Deletion is in-app only. Server ownership passes to "
        "another member or the server is deleted with the account.",
    ),
    _guide(
        "reddit",
        [
            _step("Download your data first: User Settings → Privacy & "
                  "Security → Request your data."),
            _step("Open old.reddit.com/prefs/deactivate/ and sign in.",
                  hand=True),
            _step("Enter your username and password, tick the "
                  "confirmation box, click Deactivate.", hand=True),
            _step("Deactivation is immediate and permanent."),
        ],
        delete_url="https://old.reddit.com/prefs/deactivate/",
        difficulty="easy",
        grace_days=0,
    ),
    _guide(
        "pinterest",
        [
            _step("Download your data first: Settings → Privacy and "
                  "data → Download your data."),
            _step("Go to Settings → Account management → Delete "
                  "account (in the app: Delete your data and "
                  "account)."),
            _step("Choose a reason, click Send email."),
            _step("Open the email Pinterest sends and confirm the "
                  "deletion.", hand=True),
            _step("The profile deactivates at once and is permanently "
                  "deleted after 7 days. Log in within 7 days to "
                  "reactivate."),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=7,
        notes="Deletion lives in Settings — there is no direct "
        "deletion URL.",
    ),
    _guide(
        "spotify",
        [
            _step("Cancel Premium first — a paid account cannot be "
                  "closed. Cancel at spotify.com/account/subscription."),
            _step("Save your playlists: export them or copy the links "
                  "— closing wipes them."),
            _step("Open spotify.com/account/close/ in a desktop browser "
                  "and sign in.", hand=True),
            _step("Click Close my account and delete my data, then "
                  "confirm in the email Spotify sends.", hand=True),
            _step("The account enters a 7-day reactivation window, "
                  "then is gone for good."),
        ],
        delete_url="https://www.spotify.com/account/close/",
        difficulty="medium",
        grace_days=7,
    ),
    _guide(
        "netflix",
        [
            _step("Request your information first: "
                  "netflix.com/account/getmyinfo."),
            _step("Cancel the membership first: Account → Cancel "
                  "Membership (or netflix.com/cancelplan). Cancelling "
                  "alone does not delete the account."),
            _step("If you pay through Apple, Google Play or a carrier, "
                  "cancel there — Netflix cannot stop that billing."),
            _step("After the billing period ends, open "
                  "netflix.com/account/privacy → Privacy and data "
                  "settings → Submit a request to delete your "
                  "account."),
            _step("Submit the deletion request and confirm.",
                  hand=True),
            _step("Netflix keeps the profile ~10 months after "
                  "cancelling so you can restart; a deletion request "
                  "asks for removal within ~30 days."),
        ],
        delete_url="https://www.netflix.com/account/privacy",
        difficulty="medium",
        grace_days=300,
        notes="Grace here is Netflix's ~10-month restart window after "
        "cancelling; the deletion request itself is the ask to remove "
        "the account sooner.",
    ),
    _guide(
        "steam",
        [
            _step("Export your data first: help.steampowered.com → My "
                  "Account → Data Related to Your Steam Account → "
                  "download what you want to keep."),
            _step("Sign in at help.steampowered.com, open My Account → "
                  "Data Related to Your Steam Account → Delete My "
                  "Steam Account.", hand=True),
            _step("Read the warning — every game, DLC, item and your "
                  "Wallet balance are destroyed, with no refunds and "
                  "no transfers."),
            _step("Fill in the deletion request form (first email used, "
                  "payment info, phone number) and send it.",
                  hand=True),
            _step("Steam Support reviews the request by hand; accounts "
                  "with recent purchases wait ~30 days before they are "
                  "eligible. The request starts a 30-day grace period "
                  "once accepted."),
        ],
        delete_url="https://help.steampowered.com/",
        difficulty="hard",
        grace_days=30,
        notes="Deletion is a manual Support review, not a button — "
        "expect back-and-forth, and the 30-day grace runs from "
        "acceptance, not from the request.",
    ),
    _guide(
        "github",
        [
            _step("Back up your repositories first — deletion removes "
                  "them, including forks others depend on."),
            _step("Open github.com/settings/admin and sign in.",
                  hand=True),
            _step("Scroll to Delete your account, click it, type your "
                  "username or password to confirm.", hand=True),
            _step("Deletion is immediate and permanent."),
        ],
        delete_url="https://github.com/settings/admin",
        difficulty="easy",
        grace_days=0,
    ),
    _guide(
        "amazon",
        [
            _step("Download your data first: Account → Privacy → "
                  "Request Your Information."),
            _step("Open amazon.com/privacy/data-deletion and sign in.",
                  hand=True),
            _step("Choose the account-closure option and follow the "
                  "verification steps.", hand=True),
            _step("Closing the account also closes Prime, Kindle "
                  "books, and any order history tied to it."),
        ],
        delete_url="https://www.amazon.com/privacy/data-deletion",
        difficulty="medium",
        grace_days=0,
    ),
    _guide(
        "apple",
        [
            _step("Download a copy of your data first: privacy.apple."
                  "com → Obtain a copy of your data."),
            _step("Sign in at privacy.apple.com and choose Delete your "
                  "account.", hand=True),
            _step("Apple verifies the request — this can take up to 7 "
                  "days. Your account is deactivated while "
                  "verification runs."),
            _step("You get a unique access code: keep it. It is the "
                  "only way to cancel the deletion or contact Apple "
                  "about it.", hand=True),
        ],
        aliases=("icloud",),
        delete_url="https://privacy.apple.com/",
        difficulty="medium",
        grace_days=7,
    ),
    _guide(
        "microsoft",
        [
            _step("Download your data first: account.microsoft.com → "
                  "Privacy → Download your data."),
            _step("Cancel any subscriptions (Microsoft 365, Xbox) — a "
                  "paid account cannot be closed."),
            _step("Open account.live.com/closeaccount.aspx, sign in, "
                  "and confirm you have read what closing removes.",
                  hand=True),
            _step("Mark the account for closure. You have 60 days to "
                  "sign back in and reopen it; after that it is gone.",
                  hand=True),
        ],
        aliases=("outlook", "hotmail", "skype"),
        delete_url="https://account.live.com/closeaccount.aspx",
        difficulty="medium",
        grace_days=60,
    ),
    _guide(
        "paypal",
        [
            _step("Withdraw your balance first — an account with money "
                  "in it cannot be closed."),
            _step("Download statements and tax records: Settings → "
                  "Statements."),
            _step("Resolve open disputes and remove linked banks and "
                  "cards."),
            _step("Go to Settings → Account options → Close your "
                  "account, then Close Account.", hand=True),
            _step("Re-enter your password to confirm. Closure is "
                  "permanent and cannot be reopened.", hand=True),
        ],
        delete_url="",
        difficulty="medium",
        grace_days=0,
        notes="There is no direct deletion URL — the flow lives in "
        "Settings → Account options.",
    ),
    _guide(
        "archive.org",
        [
            _step("There is no self-service deletion for archive.org "
                  "accounts — the only route is email."),
            _step("Email info@archive.org from the address on the "
                  "account, asking for the account to be deleted, and "
                  "include the username.", hand=True),
            _step("Archive staff reply when the account is removed. "
                  "Keep the thread until they confirm."),
        ],
        aliases=("internet archive",),
        delete_url="",
        difficulty="impossible",
        grace_days=0,
        export_first=True,
        notes="Rated impossible because no self-service deletion exists "
        "at all — deletion depends on staff action, so there is no "
        "guaranteed path.",
    ),
]


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def is_stale(guide: DeletionGuide, *, today: date | None = None) -> bool:
    """True when a guide is unusable as current: never checked, or its
    ``checked_at`` is older than ``STALE_AFTER_DAYS``."""
    checked = _parse_date(guide.checked_at)
    if checked is None:
        return True
    today = today or date.today()
    return (today - checked).days > STALE_AFTER_DAYS


def _grace_line(guide: DeletionGuide) -> str:
    if guide.grace_days <= 0:
        return ("No grace window — once it is gone, it is gone. "
                "Be sure before the final confirm.")
    if guide.grace_days == 1:
        return "You have 1 day to change your mind after starting."
    return (f"You have {guide.grace_days} days to change your mind "
            "after starting.")


def render_guide(guide: DeletionGuide, *, today: date | None = None) -> str:
    """Render a guide completely: difficulty and grace window up front,
    export-first warning, every step, the delete URL.

    A stale guide is labelled stale at the top and its steps are marked
    unverified — it is never served as current.
    """
    stale = is_stale(guide, today=today)
    service = guide.service
    lines = [f"How to delete your {service} account — {guide.difficulty}."]
    if stale:
        checked = guide.checked_at or "never"
        lines.append(
            f"STALE — last checked {checked}. Deletion flows change; "
            "verify each step before you follow it."
        )
    lines.append(_grace_line(guide))
    if guide.export_first:
        lines.append(
            "Export your data first — download what you want to keep "
            "before you delete anything. Deletion does not keep it."
        )
    if guide.delete_url:
        lines.append(f"Delete link: {guide.delete_url}")
    else:
        lines.append(
            "No direct delete link — the flow lives in the service's "
            "settings or app; the steps below take you there."
        )
    for index, step in enumerate(guide.steps, start=1):
        mark = " [YOUR HAND]" if step.owner_hand else ""
        unverified = " (unverified)" if stale else ""
        lines.append(f"{index}. {step.text}{mark}{unverified}")
    if guide.notes:
        lines.append(f"Note: {guide.notes}")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class DeletionStore:
    """Durable deletion-runbook registry: ``<home>/data/deletion_guides.json``.

    ``home`` may be empty (tests / in-memory use): the store then
    holds seeds only and never touches disk.
    """

    def __init__(self, home: Any) -> None:
        home_str = str(home or "").strip()
        self._path = (
            Path(home_str) / "data" / _STORE_NAME if home_str else None
        )
        self._guides: dict[str, DeletionGuide] = {}
        self._walks: dict[str, int] = {}
        self._active: str | None = None
        self._load()

    @property
    def path(self) -> str | None:
        return str(self._path) if self._path is not None else None

    def _seed(self) -> None:
        for row in SEEDS:
            try:
                guide = DeletionGuide.from_dict(row)
            except ValueError:
                continue
            self._guides[guide.slug] = guide

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            self._seed()
            return
        raw = _read_json(self._path)
        rows = raw.get("guides")
        if not isinstance(rows, dict) or not rows:
            self._quarantine("empty or unreadable")
            self._seed()
            return
        loaded = 0
        for row in rows.values():
            if not isinstance(row, dict):
                continue
            try:
                guide = DeletionGuide.from_dict(row)
            except (ValueError, TypeError, KeyError):
                continue
            self._guides[guide.slug] = guide
            loaded += 1
        if not loaded:
            self._quarantine("no valid guides")
            self._seed()
            return
        walks = raw.get("walks")
        if isinstance(walks, dict):
            for key, index in walks.items():
                try:
                    self._walks[str(key)] = max(0, int(index))
                except (TypeError, ValueError):
                    continue
        active = raw.get("active")
        if isinstance(active, str) and slugify_service(active) in self._guides:
            self._active = slugify_service(active)

    def _quarantine(self, reason: str) -> None:
        if self._path is None:
            return
        stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        target = self._path.with_suffix(f".corrupt-{stamp}.json")
        with contextlib.suppress(OSError):
            self._path.replace(target)

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "guides": {
                slug: guide.to_dict()
                for slug, guide in self._guides.items()
            },
            "walks": dict(self._walks),
            "active": self._active,
        }
        with contextlib.suppress(Exception):
            _atomic_json_write(self._path, payload)

    # -- lookup ------------------------------------------------------
    def known_names(self) -> set[str]:
        names: set[str] = set()
        for guide in self._guides.values():
            names.add(guide.service.lower())
            names.update(alias.lower() for alias in guide.aliases)
        return names

    def get(self, service: str) -> DeletionGuide | None:
        key = slugify_service(service)
        if key in self._guides:
            return self._guides[key]
        lowered = (service or "").strip().lower()
        for guide in self._guides.values():
            if guide.service.lower() == lowered:
                return guide
            if lowered in (alias.lower() for alias in guide.aliases):
                return guide
        return None

    def list_all(self) -> list[DeletionGuide]:
        return sorted(self._guides.values(), key=lambda g: g.service)

    def search(self, query: str) -> list[DeletionGuide]:
        needle = (query or "").strip().lower()
        if not needle:
            return self.list_all()
        hits: list[DeletionGuide] = []
        for guide in self.list_all():
            haystack = " ".join(
                [guide.service, *guide.aliases, guide.notes]
            ).lower()
            if needle in haystack:
                hits.append(guide)
        return hits

    # -- growth by discovery -----------------------------------------
    def add(self, guide: DeletionGuide) -> DeletionGuide:
        if not guide.checked_at:
            guide.checked_at = date.today().isoformat()
        self._guides[guide.slug] = guide
        self._save()
        return guide

    def refresh(self, service: str, *, today: date | None = None) -> bool:
        """Mark a guide freshly checked (its steps were re-verified).
        Returns False when the service is unknown."""
        guide = self.get(service)
        if guide is None:
            return False
        guide.checked_at = (today or date.today()).isoformat()
        self._save()
        return True

    # -- walk state ---------------------------------------------------
    def walk_position(self, service: str) -> int:
        guide = self.get(service)
        if guide is None:
            return 0
        return self._walks.get(guide.slug, 0)

    def set_walk_position(self, service: str, index: int) -> None:
        guide = self.get(service)
        if guide is None:
            return
        position = max(0, min(index, len(guide.steps)))
        self._walks[guide.slug] = position
        self._active = guide.slug if position < len(guide.steps) else None
        self._save()

    def reset_walk(self, service: str) -> None:
        guide = self.get(service)
        if guide is None:
            return
        self._walks.pop(guide.slug, None)
        self._active = guide.slug
        self._save()

    def active_walk(self) -> DeletionGuide | None:
        """The guide the owner is currently walking, or None."""
        if self._active is None:
            return None
        guide = self._guides.get(self._active)
        if guide is None:
            return None
        if self.walk_position(guide.service) >= len(guide.steps):
            return None
        return guide


_GUIDE_PATTERNS = (
    # "how do i delete my facebook account?" / "how to delete my x"
    re.compile(
        r"^(?:how\s+(?:do|can)\s+i|how\s+to)\s+delete\s+(?:my\s+)?"
        r"(?P<service>.+?)(?:\s+account)?\s*\??$",
        re.IGNORECASE,
    ),
    # "delete my facebook account" / "delete facebook"
    re.compile(
        r"^delete\s+(?:my\s+)?(?P<service>.+?)"
        r"(?:\s+account)?\s*\??$",
        re.IGNORECASE,
    ),
    # "walk me through deleting my facebook account"
    re.compile(
        r"^(?:walk\s+me\s+through|talk\s+me\s+through)\s+"
        r"(?:deleting|closing)\s+(?:my\s+)?(?P<service>.+?)"
        r"(?:\s+account)?\s*\??$",
        re.IGNORECASE,
    ),
    # "close my facebook" / "close my facebook account"
    re.compile(
        r"^close\s+(?:my\s+)?(?P<service>.+?)(?:\s+account)?\s*\??$",
        re.IGNORECASE,
    ),
)


def parse_delete_request(
    text: str, known_names: set[str] | None = None,
) -> dict[str, Any] | None:
    """Parse an NL deletion request.

    Returns ``{"kind": "guide" | "walk", "service": str,
    "has_account_word": bool}`` or ``None`` when the text is not a
    deletion request at all. ``known_names`` (service names and
    aliases, lowercased) sharpens the match: a bare "close my X"
    only fires when X is known — "close my laptop" is not a
    deletion request. A request that names an account
    ("how do I delete my myspace account?") always fires, and the
    caller answers honestly when the service has no guide.
    """
    cleaned = (text or "").strip()
    if not cleaned or cleaned.startswith("/"):
        return None
    lowered = cleaned.lower()
    for index, pattern in enumerate(_GUIDE_PATTERNS):
        match = pattern.match(cleaned)
        if not match:
            continue
        service = match.group("service").strip()
        if not service or len(service) > 80:
            return None
        if service.lower() in {"account", "it", "this", "that"}:
            # "delete my account" names no service — not a request.
            return None
        has_account_word = "account" in lowered
        if (known_names is not None
                and service.lower() not in known_names
                and not has_account_word):
            # "close my laptop": not a known service, no account
            # named — not a deletion request.
            return None
        # "close my laptop" / "delete my homework" are not requests.
        kind = "walk" if index >= 2 else "guide"
        return {
            "kind": kind,
            "service": service,
            "has_account_word": has_account_word,
        }
    return None


def walk_current(store: DeletionStore, guide: DeletionGuide) -> str:
    """Render the current walk step for ``guide`` (or the done line)."""
    position = store.walk_position(guide.service)
    return render_walk_step(guide, position)


def render_walk_step(guide: DeletionGuide, position: int) -> str:
    total = len(guide.steps)
    if position >= total:
        return (
            f"That's the whole {guide.service} runbook — done. "
            f"Difficulty was {guide.difficulty}; "
            f"{_grace_line(guide)}"
        )
    step = guide.steps[position]
    lines = [
        f"{guide.service} — step {position + 1} of {total}: {step.text}"
    ]
    if step.owner_hand:
        lines.append(
            "This one is yours — I can't sign in, pass a check, or "
            "press the final button for you. Say \"done\" when you've "
            "done it and I'll take you to the next step."
        )
    else:
        lines.append("Say \"done\" when you've done it, or \"next\" to "
                     "move on.")
    return "\n".join(lines)


def walk_advance(store: DeletionStore, guide: DeletionGuide) -> str:
    """Mark the current walk step done and render the next one."""
    position = store.walk_position(guide.service)
    store.set_walk_position(guide.service, position + 1)
    return render_walk_step(guide, position + 1)


def walk_reset(store: DeletionStore, guide: DeletionGuide) -> str:
    """Restart the walk from step one."""
    store.reset_walk(guide.service)
    return render_walk_step(guide, 0)
