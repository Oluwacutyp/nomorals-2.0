"""Antibody store — immune memory.

In-memory store with optional JSON persistence (atomic tmp + os.replace,
like ``commitments/ledger.py``). Second exposure to a known pathogen is
caught at innate speed via a compiled-regex cache.

Rules enforced here, not by convention:

* **Reason required.** ``add`` without a reason raises :class:`ValueError`.
* **Versioning.** Re-adding an antibody with an existing id bumps its
  version and keeps the superseded copy in history.
* **Memory transfer.** :meth:`AntibodyStore.export_bundle` /
  :meth:`AntibodyStore.import_bundle` move immune memory across tissues
  (phone immunizes workstation): version + reason are preserved, and
  on conflict the HIGHER version wins.
* **Lifecycle (doc 20 P3).** A stored antibody is ``probation``,
  ``active`` or ``retired``. An antibody created by LEARNING enters
  on probation (:meth:`AntibodyStore.add_learned`): it may match
  and RECORD — every match lands on its record as evidence — but
  it drives no effect response; the response seam reads active
  only. Promotion is a versioned transition, earned two ways: the
  record's own evidence (:func:`promotion_policy` — a threshold of
  recorded matches with zero recorded false positives over a
  minimum observation window, evaluated lazily at match/read time)
  or an explicit :meth:`AntibodyStore.promote` with a reason (the
  owner's verb). A recorded false positive on an active antibody
  demotes it back to probation — versioned, reason on the record.
  Built-in detection (the innate catalog in ``pathogens.py``) is
  not store memory and carries no lifecycle. Records written
  before P3 keep the standing they carry: the upgrade neither
  demotes nor promotes anything by itself.

The store is duck-typed: it accepts antibody dataclasses (e.g.
:class:`nomorals.immune.thymus.Antibody`), dicts, or any object with
the antibody fields as attributes. It imports nothing from its
siblings, so the package stays stdlib-only.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "AntibodyStore",
    "STATUS_ACTIVE",
    "STATUS_PROBATION",
    "STATUS_RETIRED",
    "PROMOTION_MIN_MATCHES",
    "PROMOTION_MIN_WINDOW_S",
    "promotion_policy",
]

#: Lifecycle states of a stored record. ``proposed`` exists only on
#: thymus drafts before acceptance; a record in the store is in one
#: of these three.
STATUS_PROBATION = "probation"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"

#: Promotion policy defaults (doc 20 P3). Five recorded matches:
#: more than one scan burst can produce casually, few enough that a
#: real recurring pathogen promotes within days of live scans. A
#: one-hour minimum window: the evidence must span real time, not
#: one pasted blob matched repeatedly in a single minute.
PROMOTION_MIN_MATCHES = 5
PROMOTION_MIN_WINDOW_S = 3600.0

#: Env overrides, following the immune package's ``NM_IMMUNE_*``
#: knob convention (cf. wiring.STORE_ENV_VAR).
PROMOTION_ENV_MIN_MATCHES = "NM_IMMUNE_PROMOTION_MIN_MATCHES"
PROMOTION_ENV_WINDOW_S = "NM_IMMUNE_PROMOTION_WINDOW_S"


def promotion_policy() -> tuple[int, float]:
    """The promotion evidence policy: ``(min_matches, min_window_s)``.

    A probation antibody promotes when its CURRENT probation stint
    shows at least ``min_matches`` recorded matches spanning at
    least ``min_window_s``, with zero recorded false positives in
    the window — the last clause holds by construction: a false
    positive restarts the stint (``probation_since`` moves to the
    false positive, the match count resets), so the window being
    measured always begins after the last one. Malformed or
    non-positive env overrides fall back to the defaults — a typo
    in the environment must never silently weaken or break the
    gate.
    """
    min_matches = PROMOTION_MIN_MATCHES
    raw = os.environ.get(PROMOTION_ENV_MIN_MATCHES, "")
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            min_matches = parsed
    min_window = PROMOTION_MIN_WINDOW_S
    raw = os.environ.get(PROMOTION_ENV_WINDOW_S, "")
    if raw:
        try:
            parsed_f = float(raw)
        except ValueError:
            parsed_f = -1.0
        if parsed_f >= 0:
            min_window = parsed_f
    return min_matches, min_window


_ANTIBODY_FIELDS = (
    "id",
    "pattern",
    "pathogen_class",
    "reason",
    "created_ts",
    "version",
    "status",
    "retire_reason",
    "origin",
    "match_count",
    "false_positive_count",
    "first_match_ts",
    "last_match_ts",
    "last_false_positive_ts",
    "last_false_positive_reason",
    "probation_since",
    "probation_matches",
    "probation_false_positives",
    "status_reason",
    "status_ts",
    "status_evidence",
)

#: Record fields an update inherits from the stored record when the
#: update does not mention them: a definition edit is not a
#: lifecycle decision and must not reset standing or evidence.
_INHERITED_ON_UPDATE = (
    "status",
    "origin",
    "match_count",
    "false_positive_count",
    "first_match_ts",
    "last_match_ts",
    "last_false_positive_ts",
    "last_false_positive_reason",
    "probation_since",
    "probation_matches",
    "probation_false_positives",
    "status_reason",
    "status_ts",
    "status_evidence",
)


def _normalize_lifecycle(data: dict[str, Any]) -> None:
    """Fill the lifecycle fields a record does not carry.

    Only defaults are filled — a value already on the record is its
    recorded standing and is never rewritten here. A record with NO
    status has made no lifecycle claim, so it enters on probation
    (unproven until evidence or the owner promotes it). Producers
    that mean a standing say so explicitly: the thymus's ``active``
    on the owner-directed path, :meth:`AntibodyStore.add_learned`'s
    ``probation`` on the learning path.
    """
    data.setdefault("status", STATUS_PROBATION)
    data.setdefault("origin", "")
    data.setdefault("retire_reason", None)
    for key in ("match_count", "false_positive_count",
                "probation_matches", "probation_false_positives"):
        try:
            data[key] = int(data.get(key) or 0)
        except (TypeError, ValueError):
            data[key] = 0
    data.setdefault("first_match_ts", None)
    data.setdefault("last_match_ts", None)
    data.setdefault("last_false_positive_ts", None)
    data.setdefault("last_false_positive_reason", "")
    data.setdefault("status_reason", "")
    data.setdefault("status_ts", None)
    data.setdefault("status_evidence", None)
    if data.get("status") == STATUS_PROBATION:
        if not data.get("probation_since"):
            data["probation_since"] = data.get("created_ts") or time.time()
    else:
        data.setdefault("probation_since", None)
    if data.get("status") == STATUS_RETIRED and not data.get("status_reason"):
        data["status_reason"] = str(data.get("retire_reason") or "")


@lru_cache(maxsize=512)
def _compile(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def _to_dict(antibody: Any) -> Dict[str, Any]:
    if isinstance(antibody, dict):
        return copy.deepcopy(antibody)
    if is_dataclass(antibody):
        return copy.deepcopy(asdict(antibody))
    data: Dict[str, Any] = {}
    for field_name in _ANTIBODY_FIELDS:
        if hasattr(antibody, field_name):
            data[field_name] = copy.deepcopy(getattr(antibody, field_name))
    if not data:
        raise ValueError(f"cannot interpret {antibody!r} as an antibody")
    return data


def _require_reason(data: Dict[str, Any]) -> None:
    reason = data.get("reason")
    if not reason or not str(reason).strip():
        raise ValueError("antibody requires a reason — add without reason raises")


class AntibodyStore:
    """In-memory antibody store with optional atomic JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        #: One lock serialises every mutation (and match recording):
        #: since doc 20 P3 the production store is ONE shared
        #: instance (immune/wiring.shared_antibody_store) read and
        #: written from chat and sampler threads alike.
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        if self.path is not None and self.path.exists():
            self.load()

    # ── persistence ──────────────────────────────────────────────

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {
                "format": "immune.antibodies.v1",
                "saved_ts": time.time(),
                "antibodies": list(self._store.values()),
                "history": self._history,
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self._store = {
            str(ab["id"]): dict(ab) for ab in payload.get("antibodies", [])
        }
        # Records from a pre-P3 file keep their recorded standing;
        # normalization only fills lifecycle fields they never had.
        for record in self._store.values():
            _normalize_lifecycle(record)
        self._history = {
            str(k): [dict(entry) for entry in v]
            for k, v in payload.get("history", {}).items()
        }

    # ── CRUD (no delete: immune memory is append-only) ───────────

    def add(self, antibody: Any, *, origin: str = "") -> Dict[str, Any]:
        """Add an antibody (normally via thymus accept). Re-adding an
        existing id bumps the version and keeps history. ``origin``
        stamps a NEW record's provenance (``"learned"``, ``"owner"``,
        …); an update never rewrites the origin of creation, and an
        update that does not mention lifecycle fields inherits the
        stored record's standing and evidence (see
        ``_INHERITED_ON_UPDATE``)."""
        with self._lock:
            data = _to_dict(antibody)
            _require_reason(data)
            ab_id = str(data.get("id") or uuid.uuid4().hex)
            data["id"] = ab_id
            existing = self._store.get(ab_id)
            if existing is not None:
                for key in _INHERITED_ON_UPDATE:
                    if key not in data and existing.get(key) is not None:
                        data[key] = copy.deepcopy(existing[key])
                self._history.setdefault(ab_id, []).append(
                    copy.deepcopy(existing))
                data["version"] = int(existing.get("version", 1)) + 1
            else:
                data["version"] = int(data.get("version") or 1)
                if origin:
                    data["origin"] = str(origin)
            data.setdefault("created_ts", time.time())
            _normalize_lifecycle(data)
            self._store[ab_id] = copy.deepcopy(data)
            self.save()
            return copy.deepcopy(data)

    def add_learned(self, antibody: Any, *, now: float | None = None) -> dict[str, Any]:
        """Add an antibody CREATED BY LEARNING (the scan/adaptive path).

        A learned antibody enters on PROBATION (doc 20 P3): it may
        match and record evidence, but it drives no effect response
        until it is promoted. If the id already exists in the store,
        the record keeps its recorded standing — re-learning never
        silently demotes an active antibody.
        """
        with self._lock:
            data = _to_dict(antibody)
            existing = self._store.get(str(data.get("id") or ""))
            if existing is None:
                data["status"] = STATUS_PROBATION
                if now is not None:
                    data["probation_since"] = now
                if not data.get("status_reason"):
                    data["status_reason"] = (
                        "entered on probation: created by learning, "
                        "not yet proven")
            return self.add(data, origin="learned")

    def update(self, antibody: Any) -> Dict[str, Any]:
        """Update an antibody: same as add — version bumps, history kept."""
        return self.add(antibody)

    def get(self, antibody_id: str) -> Optional[Dict[str, Any]]:
        found = self._store.get(str(antibody_id))
        return copy.deepcopy(found) if found is not None else None

    def all(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(ab) for ab in self._store.values()]

    def list_active(self) -> list[dict[str, Any]]:
        """The active population — the response seam's read.

        Promotion is evaluated lazily here (and in :meth:`match`):
        a probation antibody whose record now meets the promotion
        policy is transitioned before the list is read, so this is
        never a stale answer.
        """
        with self._lock:
            for ab_id, record in list(self._store.items()):
                if record.get("status") == STATUS_PROBATION:
                    self.evaluate_promotion(ab_id)
            return [ab for ab in self.all()
                    if ab.get("status") == STATUS_ACTIVE]

    def list_probation(self) -> list[dict[str, Any]]:
        """Probation records, as recorded (no lazy evaluation —
        callers that need post-evaluation standing read
        :meth:`list_active` first)."""
        return [ab for ab in self.all()
                if ab.get("status") == STATUS_PROBATION]

    def history(self, antibody_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._history.get(str(antibody_id), []))

    # ── lifecycle: evidence, promotion, demotion ─────────────────

    def _transition(self, record: dict[str, Any], to_status: str,
                    reason: str, now: float) -> dict[str, Any]:
        """Move ``record`` to ``to_status`` as a NEW VERSION.

        The superseded version stays in history; the new version
        carries the transition's reason and a snapshot of the
        evidence that justified it. Entering probation starts a
        fresh stint: match/false-positive stint counters reset and
        the observation window restarts at ``now``.
        """
        new = copy.deepcopy(record)
        since = record.get("probation_since")
        new["status"] = to_status
        new["status_reason"] = reason
        new["status_ts"] = now
        new["status_evidence"] = {
            "match_count": int(record.get("match_count") or 0),
            "false_positive_count": int(
                record.get("false_positive_count") or 0),
            "probation_matches": int(record.get("probation_matches") or 0),
            "probation_false_positives": int(
                record.get("probation_false_positives") or 0),
            "probation_window_s": (
                now - float(since)) if since else None,
        }
        if to_status == STATUS_PROBATION:
            new["probation_since"] = now
            new["probation_matches"] = 0
            new["probation_false_positives"] = 0
        return self.add(new)

    def evaluate_promotion(self, antibody_id: str, *,
                           now: float | None = None) -> dict[str, Any] | None:
        """Promote a probation antibody whose OWN record proves it.

        The evidence is the record's current probation stint:
        at least ``min_matches`` recorded matches, ZERO recorded
        false positives, and at least ``min_window_s`` elapsed
        (see :func:`promotion_policy`). Returns the (possibly
        transitioned) record, or None for an unknown id. Called
        lazily from :meth:`match` and :meth:`list_active` — there
        is no background promoter.
        """
        with self._lock:
            record = self._store.get(str(antibody_id))
            if record is None:
                return None
            if record.get("status") != STATUS_PROBATION:
                return copy.deepcopy(record)
            now = time.time() if now is None else now
            min_matches, min_window = promotion_policy()
            since = record.get("probation_since") \
                or record.get("created_ts") or now
            window = now - float(since)
            matches = int(record.get("probation_matches") or 0)
            if matches >= min_matches and window >= min_window:
                reason = (
                    f"promoted on its own evidence: {matches} recorded "
                    f"matches, 0 false positives over a {window:.0f}s "
                    f"probation window (policy: >={min_matches} matches, "
                    f">={min_window:.0f}s clean window)")
                return self._transition(record, STATUS_ACTIVE, reason, now)
            return copy.deepcopy(record)

    def promote(self, antibody_id: str, *, reason: str,
                now: float | None = None) -> dict[str, Any]:
        """Explicitly promote a probation antibody (the owner's verb).

        A named refusal, never a silent no-op: an unknown id, a
        missing reason, or an antibody that is not on probation
        each raise :class:`ValueError` saying which.
        """
        if not reason or not str(reason).strip():
            raise ValueError("promoting an antibody requires a reason")
        with self._lock:
            record = self._store.get(str(antibody_id))
            if record is None:
                raise ValueError(
                    f"no antibody {antibody_id!r} to promote")
            status = record.get("status")
            if status != STATUS_PROBATION:
                raise ValueError(
                    f"antibody {antibody_id} is {status or 'stateless'}, "
                    "not on probation — only a probation antibody can "
                    "be promoted")
            now = time.time() if now is None else now
            return self._transition(
                record, STATUS_ACTIVE, str(reason).strip(), now)

    def record_match(self, antibody_id: str, *,
                     now: float | None = None) -> dict[str, Any] | None:
        """Record one match on the antibody's record.

        Counters and timestamps update in place (a match is
        evidence, not a definition change, so no version bump);
        the store persists. A probation antibody is then evaluated
        for promotion — this match may be the one that proves it.
        Returns the record after any transition, or None for an
        unknown id.
        """
        with self._lock:
            record = self._store.get(str(antibody_id))
            if record is None:
                return None
            now = time.time() if now is None else now
            _normalize_lifecycle(record)
            record["match_count"] = int(record.get("match_count") or 0) + 1
            if not record.get("first_match_ts"):
                record["first_match_ts"] = now
            record["last_match_ts"] = now
            if record.get("status") == STATUS_PROBATION:
                record["probation_matches"] = int(
                    record.get("probation_matches") or 0) + 1
            self.save()
            if record.get("status") == STATUS_PROBATION:
                return self.evaluate_promotion(str(antibody_id), now=now)
            return copy.deepcopy(record)

    def record_false_positive(self, antibody_id: str, *, reason: str,
                              now: float | None = None,
                              demote: bool = True) -> dict[str, Any] | None:
        """Record a false-positive outcome on the antibody's record.

        The verdict always lands (counter, timestamp, reason). Its
        lifecycle consequence depends on the standing: on PROBATION
        it breaks the clean window — the stint restarts, so the
        antibody must re-earn promotion over a clean window. On an
        ACTIVE antibody it DEMOTES back to probation (a versioned
        transition, reason on the record) unless ``demote`` is
        False — the chat retire path passes False because the
        owner's retirement in the same act supersedes demotion.
        Returns the record, or None for an unknown id.
        """
        if not reason or not str(reason).strip():
            raise ValueError(
                "recording a false positive requires a reason")
        with self._lock:
            record = self._store.get(str(antibody_id))
            if record is None:
                return None
            now = time.time() if now is None else now
            _normalize_lifecycle(record)
            record["false_positive_count"] = int(
                record.get("false_positive_count") or 0) + 1
            record["last_false_positive_ts"] = now
            record["last_false_positive_reason"] = str(reason).strip()
            status = record.get("status")
            if status == STATUS_PROBATION:
                # The clean window is broken: the stint restarts
                # here. The false positive itself stays on the
                # lifetime record (count, timestamp, reason above);
                # the restarted stint begins clean, so its counters
                # read zero — the antibody must re-earn promotion
                # over a fresh, clean, full-length window.
                record["probation_matches"] = 0
                record["probation_false_positives"] = 0
                record["probation_since"] = now
                self.save()
                return copy.deepcopy(record)
            if status == STATUS_ACTIVE and demote:
                self.save()
                return self._transition(
                    record, STATUS_PROBATION,
                    f"demoted: false positive recorded — "
                    f"{str(reason).strip()}", now)
            self.save()
            return copy.deepcopy(record)

    # ── second exposure: innate-speed matching ───────────────────

    def match(self, text: str, *,
              now: float | None = None) -> list[dict[str, Any]]:
        """Return the antibodies whose pattern matches ``text``.

        Matching is COMPLETE over the non-retired population —
        probation and active alike — because a probation antibody's
        matches are exactly the evidence it promotes on. Every hit
        is RECORDED on its record (:meth:`record_match`) and lazy
        promotion is evaluated before the hits return, so each
        hit's ``status`` is its standing after this exposure.

        The RESPONSE seam reads active only: a caller that would
        act on a match must take the hits with
        ``status == "active"`` — a probation hit is an observation
        to report, never a driver of an effect.
        """
        with self._lock:
            haystack = text or ""
            hit_ids: list[str] = []
            for antibody in self.all():
                if antibody.get("status") not in (
                        STATUS_PROBATION, STATUS_ACTIVE):
                    continue
                pattern = str(antibody.get("pattern") or "")
                if not pattern:
                    continue
                try:
                    compiled = _compile(pattern)
                except re.error:
                    continue
                if compiled.search(haystack):
                    hit_ids.append(str(antibody.get("id")))
            hits: list[dict[str, Any]] = []
            for ab_id in hit_ids:
                recorded = self.record_match(ab_id, now=now)
                if recorded is not None:
                    hits.append(recorded)
            return hits

    # ── immune memory transfer ───────────────────────────────────

    def export_bundle(self) -> Dict[str, Any]:
        """Export immune memory for another tissue. Version + reason
        travel with every entry."""
        return {
            "format": "immune.antibodies.v1",
            "exported_ts": time.time(),
            "antibodies": self.all(),
            "history": copy.deepcopy(self._history),
        }

    def import_bundle(
        self, bundle: dict[str, Any], *, overwrite: bool = False
    ) -> dict[str, int]:
        """Import an immune-memory bundle. Conflicts keep the HIGHER
        version; reason and version are preserved, never reset."""
        with self._lock:
            return self._import_bundle_locked(bundle, overwrite=overwrite)

    def _import_bundle_locked(
        self, bundle: Dict[str, Any], *, overwrite: bool = False
    ) -> Dict[str, int]:
        imported = 0
        kept_existing = 0
        skipped = 0
        for incoming in bundle.get("antibodies", []):
            data = _to_dict(incoming)
            try:
                _require_reason(data)
            except ValueError:
                skipped += 1
                continue
            ab_id = str(data.get("id") or uuid.uuid4().hex)
            data["id"] = ab_id
            existing = self._store.get(ab_id)
            if existing is not None and not overwrite:
                if int(existing.get("version", 1)) >= int(data.get("version", 1)):
                    kept_existing += 1
                    continue
                self._history.setdefault(ab_id, []).append(copy.deepcopy(existing))
            elif existing is not None:
                self._history.setdefault(ab_id, []).append(copy.deepcopy(existing))
            # The record's lifecycle travels with it (another
            # tissue's probation is probation here); normalization
            # only fills fields the bundle never carried.
            data.setdefault("created_ts", time.time())
            _normalize_lifecycle(data)
            self._store[ab_id] = copy.deepcopy(data)
            imported += 1
        for ab_id, entries in bundle.get("history", {}).items():
            merged = self._history.setdefault(str(ab_id), [])
            seen = {json.dumps(e, sort_keys=True, default=str) for e in merged}
            for entry in entries:
                key = json.dumps(entry, sort_keys=True, default=str)
                if key not in seen:
                    merged.append(copy.deepcopy(entry))
                    seen.add(key)
        self.save()
        return {"imported": imported, "kept_existing": kept_existing, "skipped": skipped}
