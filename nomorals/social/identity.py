"""Cross-platform identity: one person, every surface.

The problem: the same human is "Mama" on WhatsApp, @mama_vee on Telegram,
and mama.vee on Discord — three strangers to the system. Memories split,
relationships split, context splits. She talks to the same person three
times without knowing it.

The fix: identity resolution. Link platform identities into one person
record, mined from actual signals — never guessed:

* **Phone number** — the strongest link. A WhatsApp JID
  (2348012345678@s.whatsapp.net) IS a phone number; a Telegram contact
  sharing the same number is the same person. Deterministic.
* **Username** — @handle on Telegram matching a Discord username, or an
  explicit owner-confirmed link. Strong but not deterministic alone.
* **Display name** — weak signal, used only to *suggest* links, never to
  merge automatically.
* **Owner confirmation** — the owner saying "that's the same person" is
  authoritative and overrides everything.

Design rules:
- Merges are explicit and auditable: every link records WHY (signal +
  confidence). Nothing merges silently on a weak signal.
- The owner is always identity #1 — recognized on every platform via the
  existing owner JID/ID allowlists.
- Splits are supported: a wrong merge can be undone without data loss.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Confidence tiers. Only HIGH links merge automatically.
CONF_HIGH = "high"      # phone-number match, owner confirmation
CONF_MEDIUM = "medium"  # username match across platforms
CONF_LOW = "low"        # display-name similarity (suggest only, never merge)


@dataclass
class PlatformIdentity:
    """One face of a person on one platform."""

    platform: str
    platform_id: str  # JID, numeric Telegram ID, Discord user ID…
    display_name: str = ""
    username: str = ""
    phone: str = ""  # digits only, when known

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "platform_id": self.platform_id,
            "display_name": self.display_name,
            "username": self.username,
            "phone": self.phone,
        }


@dataclass
class Person:
    """One human across every surface."""

    person_id: str
    display_name: str = ""
    identities: list[PlatformIdentity] = field(default_factory=list)
    #: person_id -> {"signal": …, "confidence": …, "ts": …}
    links: dict[str, dict[str, Any]] = field(default_factory=dict)
    is_owner: bool = False
    notes: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "identities": [i.to_dict() for i in self.identities],
            "is_owner": self.is_owner,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def has_platform(self, platform: str, platform_id: str) -> bool:
        return any(
            i.platform == platform and i.platform_id == str(platform_id)
            for i in self.identities
        )


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def _phone_from_jid(jid: str) -> str:
    """2348012345678@s.whatsapp.net -> 2348012345678."""
    local = (jid or "").split("@")[0]
    digits = _digits(local)
    return digits if len(digits) >= 7 else ""


class IdentityStore:
    """Person records with cross-platform linking. JSON-backed."""

    def __init__(self, path: str = "data/social_identities.json") -> None:
        self.path = path
        self._people: dict[str, Person] = {}
        self._by_platform: dict[tuple[str, str], str] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for pid, prow in (raw.get("people") or {}).items():
            try:
                person = Person(
                    person_id=pid,
                    display_name=prow.get("display_name", ""),
                    is_owner=bool(prow.get("is_owner")),
                    notes=prow.get("notes", ""),
                    created_at=prow.get("created_at", time.time()),
                )
                for irow in prow.get("identities") or []:
                    ident = PlatformIdentity(
                        platform=irow.get("platform", ""),
                        platform_id=str(irow.get("platform_id", "")),
                        display_name=irow.get("display_name", ""),
                        username=irow.get("username", ""),
                        phone=_digits(irow.get("phone", "")),
                    )
                    person.identities.append(ident)
                    self._by_platform[(ident.platform, ident.platform_id)] = pid
                self._people[pid] = person
            except Exception:  # noqa: BLE001 — one bad row, not all
                _log.warning("identity store: skipping bad row %s", pid)

    def _save(self) -> None:
        import os

        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"people": {p: q.to_dict() for p, q in self._people.items()}},
                    fh,
                    ensure_ascii=False,
                )
            os.replace(tmp, self.path)
        except OSError as exc:
            _log.warning("identity store: save failed: %s", exc)

    # ── lookup ───────────────────────────────────────────────────
    def find(self, platform: str, platform_id: str) -> Person | None:
        pid = self._by_platform.get((platform, str(platform_id)))
        return self._people.get(pid) if pid else None

    def find_by_phone(self, phone: str) -> Person | None:
        digits = _digits(phone)
        if not digits:
            return None
        for person in self._people.values():
            for ident in person.identities:
                if ident.phone and ident.phone == digits:
                    return person
                if ident.platform == "whatsapp" and _phone_from_jid(
                    ident.platform_id
                ) == digits:
                    return person
        return None

    def owner(self) -> Person | None:
        for person in self._people.values():
            if person.is_owner:
                return person
        return None

    # ── registration ─────────────────────────────────────────────
    def register(
        self,
        platform: str,
        platform_id: str,
        *,
        display_name: str = "",
        username: str = "",
        phone: str = "",
        is_owner: bool = False,
    ) -> Person:
        """Register (or fetch) a platform identity. Auto-links on HIGH
        confidence signals only — phone match or owner flag."""
        platform_id = str(platform_id)
        existing = self.find(platform, platform_id)
        if existing:
            return existing

        # HIGH-confidence auto-link: same phone number.
        digits = _digits(phone) or _phone_from_jid(platform_id)
        if digits and not is_owner:
            linked = self.find_by_phone(digits)
            if linked:
                linked.identities.append(
                    PlatformIdentity(
                        platform=platform,
                        platform_id=platform_id,
                        display_name=display_name,
                        username=username,
                        phone=digits,
                    )
                )
                self._by_platform[(platform, platform_id)] = linked.person_id
                self._save()
                _log.info(
                    "identity: auto-linked %s:%s to %s (phone match)",
                    platform,
                    platform_id,
                    linked.person_id,
                )
                return linked

        # New person.
        import random as _random
        pid = f"p{int(time.time() * 1000)}{_random.randint(1000, 9999)}"
        person = Person(
            person_id=pid,
            display_name=display_name or username or platform_id,
            is_owner=is_owner,
        )
        person.identities.append(
            PlatformIdentity(
                platform=platform,
                platform_id=platform_id,
                display_name=display_name,
                username=username,
                phone=digits,
            )
        )
        self._people[pid] = person
        self._by_platform[(platform, platform_id)] = pid
        self._save()
        return person

    def link(
        self,
        person_id: str,
        platform: str,
        platform_id: str,
        *,
        signal: str,
        confidence: str = CONF_HIGH,
        display_name: str = "",
        username: str = "",
        phone: str = "",
    ) -> bool:
        """Explicitly link a platform identity to a person (owner-confirmed
        or username match). Records WHY for audit."""
        person = self._people.get(person_id)
        if person is None:
            return False
        if confidence not in (CONF_HIGH, CONF_MEDIUM):
            _log.warning(
                "identity: refusing %s-confidence link for %s", confidence, person_id
            )
            return False
        platform_id = str(platform_id)
        if not person.has_platform(platform, platform_id):
            person.identities.append(
                PlatformIdentity(
                    platform=platform,
                    platform_id=platform_id,
                    display_name=display_name,
                    username=username,
                    phone=_digits(phone),
                )
            )
            self._by_platform[(platform, platform_id)] = person_id
        person.links[f"{platform}:{platform_id}"] = {
            "signal": signal,
            "confidence": confidence,
            "ts": time.time(),
        }
        self._save()
        return True

    def suggest_links(self, platform: str, platform_id: str) -> list[dict[str, Any]]:
        """LOW-confidence suggestions (display-name similarity). Never merge
        automatically — returned for the owner to confirm."""
        target = self.find(platform, platform_id)
        if target is None:
            return []
        suggestions: list[dict[str, Any]] = []
        names = {
            i.display_name.lower().strip()
            for i in target.identities
            if i.display_name
        } | {target.display_name.lower().strip()}
        names.discard("")
        for person in self._people.values():
            if person.person_id == target.person_id:
                continue
            for ident in person.identities:
                cand = (ident.display_name or "").lower().strip()
                if cand and cand in names:
                    suggestions.append(
                        {
                            "person_id": person.person_id,
                            "display_name": person.display_name,
                            "matched_name": cand,
                            "confidence": CONF_LOW,
                            "signal": "display-name similarity (confirm before linking)",
                        }
                    )
                    break
        return suggestions

    def unlink(self, person_id: str, platform: str, platform_id: str) -> bool:
        """Split a wrong merge. The identity becomes its own person."""
        person = self._people.get(person_id)
        if person is None:
            return False
        platform_id = str(platform_id)
        ident = next(
            (
                i
                for i in person.identities
                if i.platform == platform and i.platform_id == platform_id
            ),
            None,
        )
        if ident is None:
            return False
        person.identities.remove(ident)
        self._by_platform.pop((platform, platform_id), None)
        person.links.pop(f"{platform}:{platform_id}", None)
        # The split identity becomes its own person record.
        import random as _random
        new_pid = f"p{int(time.time() * 1000)}{_random.randint(1000, 9999)}"
        new_person = Person(
            person_id=new_pid, display_name=ident.display_name or platform_id
        )
        new_person.identities.append(ident)
        self._people[new_pid] = new_person
        self._by_platform[(platform, platform_id)] = new_pid
        self._save()
        return True
