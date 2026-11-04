"""Cross-group reputation — the notorious-user watchlist
(DESIGN §32).

ONE ledger across every group, channel and platform Devon
serves.  When an admin in one group confirms that a person is a
scammer, a fraudster or a raider, the next group they join hears
about it — factually, through the notice layer — without the
record ever deciding admission on its own: §28 acceptance still
judges the join on the group's own criteria, exactly as if the
record did not exist.

Identity is stitched the §29 way: a record binds the RESOLVED
person — platform ids plus the username history — so changing a
handle does not shed the record, and the same person on another
platform is the same person here (a shared username is the
bridge; a bare display name never is).

Creation discipline: a record exists ONLY because a human admin
confirmed a flag in a source group.  Reports and defense sensors
may PROPOSE flags into a source group's queue; an unconfirmed
proposal expires and brands nobody.  Every record is attributed
(who confirmed, where, when) and journaled (created, warned-in
groups, dismissed-locally, cleared, retracted, aged); the source
group can RETRACT, which clears the record everywhere — past
warnings are not un-sent, the record simply stops warning.

Aging: categories carry a stated retention.  A spam-ring flag is
not a lifetime brand; a scam flag lasts years.  Expired records
stop warning and are marked aged in the book, not deleted — the
history stays readable to admins.

Persistence follows the house rule: one JSON file beside the
group store, atomic writes (tmp + replace), and a corrupt file
is quarantined aside with ReputationStoreError raised — these
records are evidence, never silently reset.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

# Category → severity (orders multiple records, worst first) and
# retention (how long a live record keeps warning).  Stated per
# category in the design: severe dealing harms brand for years;
# spam-ring membership is transient by nature.
CATEGORIES: dict[str, dict[str, Any]] = {
    "scam": {"severity": 5, "retention_days": 730,
             "label": "scam"},
    "fraud": {"severity": 5, "retention_days": 730,
              "label": "fraud"},
    "raid": {"severity": 4, "retention_days": 180,
             "label": "raid"},
    "harassment": {"severity": 3, "retention_days": 365,
                   "label": "harassment"},
    "spam-ring": {"severity": 2, "retention_days": 90,
                  "label": "spam ring"},
    "other": {"severity": 1, "retention_days": 180,
              "label": "other"},
}

# Per-group warning modes (§32): what a group does when a flagged
# person is admitted — a public notice in-group, an admins-only
# trail (console + /glog), a restriction until an admin clears it,
# or nothing at all.
WARNING_MODES = ("public", "admins", "restricted", "off")
REPUTATION_DEFAULTS: dict[str, Any] = {"warning": "public"}

# Unconfirmed flag proposals die after this long; they never
# brand anyone while they wait (DESIGN §32: "a report can
# PROPOSE; a human admin confirms").
PROPOSAL_TTL_S = 3 * 86400.0

EVENT_RING = 100


class ReputationStoreError(RuntimeError):
    """The reputation book file is unreadable; it was quarantined
    aside rather than reset (records are evidence)."""


def normalize_person(person: dict[str, Any] | None) -> dict[str, Any]:
    """The canonical person snapshot: platform ids, username
    history (lowercased, chronological), display-name history."""
    person = person or {}
    ids: dict[str, str] = {}
    for platform, user_id in (person.get("ids") or {}).items():
        if platform and user_id:
            ids[str(platform)] = str(user_id)
    usernames: list[str] = []
    for name in person.get("usernames") or []:
        clean = str(name or "").strip().lstrip("@").lower()
        if clean and clean not in usernames:
            usernames.append(clean)
    names: list[str] = []
    for name in person.get("names") or []:
        clean = str(name or "").strip()
        if clean and clean not in names:
            names.append(clean)
    return {"ids": ids, "usernames": usernames, "names": names}


def merge_person(base: dict[str, Any],
                 extra: dict[str, Any]) -> dict[str, Any]:
    """Union two snapshots of the same person (ids win per
    platform; histories append)."""
    out = normalize_person(base)
    add = normalize_person(extra)
    for platform, user_id in add["ids"].items():
        out["ids"].setdefault(platform, user_id)
    for name in add["usernames"]:
        if name not in out["usernames"]:
            out["usernames"].append(name)
    for name in add["names"]:
        if name not in out["names"]:
            out["names"].append(name)
    return out


def person_matches(record_person: dict[str, Any],
                   person: dict[str, Any]) -> bool:
    """Identity by resolved ids, bridged by username history —
    never by display name alone (names collide; ids and handles
    are the §29 stitch)."""
    known = normalize_person(record_person)
    probe = normalize_person(person)
    for platform, user_id in probe["ids"].items():
        if known["ids"].get(platform) == user_id:
            return True
    known_names = set(known["usernames"])
    return any(name in known_names for name in probe["usernames"])


class ReputationBook:
    """The global reputation ledger: records + their event
    journals, persisted beside the group store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"seq": 0, "records": {}}
        self.load()

    # ── persistence ───────────────────────────────────────────

    def load(self) -> None:
        if not self.path.exists():
            self.data = {"seq": 0, "records": {}}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("records"), dict):
                raise ValueError("reputation book shape is wrong")
            self.data = raw
            self.data.setdefault("seq", 0)
        except (OSError, ValueError) as exc:
            stamp = int(time.time())
            aside = self.path.with_suffix(f".corrupt-{stamp}")
            try:
                self.path.replace(aside)
            except OSError:
                aside = self.path
            raise ReputationStoreError(
                f"reputation book at {self.path} is unreadable "
                f"({exc}); quarantined to {aside}, NOT reset — "
                f"restore or adjudicate it by hand") from exc

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ── records ───────────────────────────────────────────────

    def _event(self, record: dict[str, Any], kind: str, *,
               now: float, **details: Any) -> None:
        events = record.setdefault("events", [])
        events.append({"kind": kind, "ts": now, **details})
        del events[:-EVENT_RING]

    def add_record(self, *, person: dict[str, Any], category: str,
                   note: str, source_group: str,
                   source_platform: str, source_title: str = "",
                   confirmed_by: str, confirmed_by_label: str = "",
                   now: float | None = None) -> dict[str, Any]:
        """Create a LIVE record.  Only an admin-confirmed flag
        reaches this — proposals live in the source group's state
        until a human decides them."""
        now = now if now is not None else time.time()
        spec = CATEGORIES[category]
        self.data["seq"] = int(self.data.get("seq") or 0) + 1
        record = {
            "id": f"rep-{self.data['seq']:04d}",
            "person": normalize_person(person),
            "category": category,
            "severity": int(spec["severity"]),
            "note": str(note or "")[:400],
            "source_group": source_group,
            "source_platform": source_platform,
            "source_title": source_title,
            "created_ts": now,
            "expires_ts": now
            + float(spec["retention_days"]) * 86400.0,
            "confirmed_by": str(confirmed_by),
            "confirmed_by_label": confirmed_by_label,
            "status": "live",
            "events": [],
        }
        self._event(record, "created", now=now,
                    group=source_group, by=str(confirmed_by))
        self.data["records"][record["id"]] = record
        self.save()
        return copy.deepcopy(record)

    def get(self, record_id: str) -> dict[str, Any] | None:
        record = self.data["records"].get(record_id)
        return copy.deepcopy(record) if record else None

    def _apply_aging(self, record: dict[str, Any],
                     now: float) -> bool:
        """A live record past its retention stops warning and is
        marked aged — the history stays in the book."""
        if record.get("status") == "live" \
                and now > float(record.get("expires_ts") or 0):
            record["status"] = "aged"
            self._event(record, "aged", now=now)
            return True
        return False

    def _matching(self, person: dict[str, Any], *,
                  now: float) -> list[dict[str, Any]]:
        """Every record naming this person, aging applied, worst
        first (severity, then newest) — the one join warning
        names the worst, never a pile."""
        probe = normalize_person(person)
        changed = False
        hits: list[dict[str, Any]] = []
        for record in self.data["records"].values():
            if self._apply_aging(record, now):
                changed = True
            if person_matches(record.get("person") or {}, probe):
                hits.append(record)
        if changed:
            self.save()
        hits.sort(key=lambda r: (-(int(r.get("severity") or 0)),
                                 -float(r.get("created_ts") or 0)))
        return hits

    def live_for(self, person: dict[str, Any], *,
                 now: float | None = None) -> list[dict[str, Any]]:
        """Live records for a person, worst first."""
        now = now if now is not None else time.time()
        return [copy.deepcopy(r) for r in
                self._matching(person, now=now)
                if r.get("status") == "live"]

    def all_for(self, person: dict[str, Any], *,
                now: float | None = None) -> list[dict[str, Any]]:
        """Every record for a person — live, retracted and aged —
        for the admin /rep view."""
        now = now if now is not None else time.time()
        return [copy.deepcopy(r) for r in
                self._matching(person, now=now)]

    def stitch(self, record_id: str, person: dict[str, Any], *,
               now: float | None = None) -> dict[str, Any] | None:
        """Fold newly-resolved identity into a record (a second
        platform's id, a newer handle) so the stitch survives in
        the book itself, not just in one lookup."""
        record = self.data["records"].get(record_id)
        if record is None:
            return None
        before = normalize_person(record.get("person") or {})
        merged = merge_person(before, person)
        if merged != before:
            record["person"] = merged
            self._event(record, "stitched",
                        now=now if now is not None else time.time(),
                        ids=sorted(merged["ids"]))
            self.save()
        return copy.deepcopy(record)

    def note_event(self, record_id: str, kind: str, *,
                   now: float | None = None,
                   **details: Any) -> bool:
        record = self.data["records"].get(record_id)
        if record is None:
            return False
        self._event(record, kind,
                    now=now if now is not None else time.time(),
                    **details)
        self.save()
        return True

    def retract(self, record_id: str, *, by: str,
                by_label: str = "",
                now: float | None = None) -> dict[str, Any] | None:
        """The source group's retraction: the record clears
        everywhere — it stops warning from this moment."""
        record = self.data["records"].get(record_id)
        if record is None or record.get("status") != "live":
            return None
        now = now if now is not None else time.time()
        record["status"] = "retracted"
        self._event(record, "retracted", now=now, by=str(by),
                    by_label=by_label)
        self.save()
        return copy.deepcopy(record)

    def stats(self, *, now: float | None = None) -> dict[str, int]:
        now = now if now is not None else time.time()
        changed = False
        live = 0
        for record in self.data["records"].values():
            if self._apply_aging(record, now):
                changed = True
            if record.get("status") == "live":
                live += 1
        if changed:
            self.save()
        return {"live": live,
                "total": len(self.data["records"])}
