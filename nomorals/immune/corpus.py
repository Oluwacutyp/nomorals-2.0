"""Self-corpus — the thymus's notion of "self", grown from evidence.

Before doc 20 P4, the corpus negative selection tested against was
eight hand-written strings (the seeds below), rebuilt in memory on
every chat-state creation: Devon's idea of her own normal activity
was eight sentences someone typed once, while her real self —
verified self-heal fixes in the incident journal, owner-directed
operations her own police reported on — sat in her records unused.
A corpus that small makes the gate brittle in both directions:
self-patterns outside the eight get flagged, and a "pass" against
eight strings proves almost nothing.

What lives here (Layer 0, stdlib only, like the rest of the immune
package — sources are read as plain data or injected, never imported):

* :data:`SEED_TEXTS` — the eight shipped seeds. They stay: the
  floor of the corpus, provenance ``seed``, corpus version 0.
* :class:`CorpusStore` — the corpus as VERSIONED, PERSISTED data.
  Exemplars harvested from Devon's own records are appended by
  :meth:`CorpusStore.harvest`; each admission bumps the corpus
  version (monotonic — a harvest that adds nothing changes no
  version), and every harvested exemplar carries the version that
  admitted it, so any prior version stays readable via
  :meth:`CorpusStore.corpus_at`. Persistence mirrors the antibody
  store: one JSON file in the settings home's ``data/`` dir
  (``immune/wiring.py`` resolves the path), atomic tmp+replace
  writes. A missing file is a seed-only corpus; a corrupt one
  degrades to seed-only with the reason recorded on the store
  (:attr:`CorpusStore.degraded_reason`) and logged — the immune
  surface never crashes over its own memory file.

Harvest sources (each exemplar CITES its source record; a candidate
whose source cannot be cited is never admitted):

* ``verified_fix`` — the incident journal's verified-fix table
  (``core/incidents.py``, read here read-only via sqlite3 from the
  journal file): a fix Devon's own healing verified is self by
  evidence. The exemplar text is the fix's recorded detail.
* ``owner_receipt`` — police receipts on the calling surface that
  recorded owner-directed / owner work as report-only
  (``action == "report_only"`` with ``details["owner_directed"]``):
  the operation shapes the owner directed and the immune system
  itself classified as owner work. The exemplar text is the
  receipt's recorded target (falling back to its reason).

Deliberately NOT harvested: active antibodies' patterns (those are
non-self detectors — the opposite direction), the command catalog
and repo file shapes (code, not records of the owner's operations),
and case-file texts (owner-authored evidence about incidents, not
operation shapes).

**Honesty bounds.** Harvested exemplars are SELF evidence only:
they never become antibodies, never promote anything, and never
leave the immune organs. A source that is missing or empty
harvests zero and the report SAYS so per source — never an error,
never a fabricated exemplar. Negative selection consumes seeds and
harvested exemplars alike (provenance does not change an
exemplar's vote), and a rejection cites the matched exemplars by
id, provenance kind and source ref
(:func:`nomorals.immune.thymus.exemplar_citation`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .thymus import SelfCorpus, SelfExemplar

__all__ = [
    "CORPUS_FORMAT",
    "PROVENANCE_SEED",
    "PROVENANCE_VERIFIED_FIX",
    "PROVENANCE_OWNER_RECEIPT",
    "PROVENANCES",
    "SEED_TEXTS",
    "CorpusStore",
    "normalize_text",
    "seed_exemplars",
]

_log = logging.getLogger(__name__)

#: On-disk corpus format version.
CORPUS_FORMAT = "immune.selfcorpus.v1"

#: Provenance kinds. Every exemplar in the corpus carries one.
PROVENANCE_SEED = "seed"
PROVENANCE_VERIFIED_FIX = "verified_fix"
PROVENANCE_OWNER_RECEIPT = "owner_receipt"
PROVENANCES = (
    PROVENANCE_SEED,
    PROVENANCE_VERIFIED_FIX,
    PROVENANCE_OWNER_RECEIPT,
)

#: The shipped seed corpus — normal owner activity, hand-written
#: once and kept verbatim: the floor the grown corpus builds on
#: (previously the whole corpus, defined in ``immune/chat.py``).
SEED_TEXTS = [
    "owner sends a message to a friend",
    "owner plays music",
    "owner searches the web",
    "owner reads email",
    "owner checks the calendar",
    "owner sets a reminder",
    "owner starts a game",
    "owner opens a document",
]

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """The dedupe normal form: whitespace-collapsed, case-folded.

    Two exemplars whose text differs only in case or spacing are
    the same self-pattern; harvest admits at most one of them.
    """
    return _WHITESPACE.sub(" ", str(text or "").strip()).casefold()


def seed_exemplars() -> list[SelfExemplar]:
    """The seed population: deterministic ids, provenance ``seed``."""
    return [
        SelfExemplar(
            id=f"seed-{index + 1}",
            text=text,
            provenance=PROVENANCE_SEED,
            source_ref="seed",
            corpus_version=0,
        )
        for index, text in enumerate(SEED_TEXTS)
    ]


def _field(record: Any, name: str, default: Any = None) -> Any:
    """Duck-typed field read (dict or object) — receipts arrive as
    the police's dataclass on the chat surface, but harvest accepts
    any record carrying the fields."""
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _exemplar_to_dict(exemplar: SelfExemplar) -> dict[str, Any]:
    return {
        "id": exemplar.id,
        "text": exemplar.text,
        "tool_call": exemplar.tool_call,
        "capability": exemplar.capability,
        "provenance": exemplar.provenance,
        "source_ref": exemplar.source_ref,
        "harvested_ts": exemplar.harvested_ts,
        "corpus_version": exemplar.corpus_version,
    }


def _exemplar_from_dict(entry: dict[str, Any]) -> SelfExemplar:
    harvested_ts = entry.get("harvested_ts")
    try:
        harvested_ts = float(harvested_ts) if harvested_ts is not None \
            else None
    except (TypeError, ValueError):
        harvested_ts = None
    try:
        corpus_version = int(entry.get("corpus_version") or 0)
    except (TypeError, ValueError):
        corpus_version = 0
    return SelfExemplar(
        id=str(entry.get("id") or uuid.uuid4().hex),
        text=str(entry.get("text") or ""),
        tool_call=entry.get("tool_call"),
        capability=str(entry.get("capability") or ""),
        provenance=str(entry.get("provenance") or ""),
        source_ref=str(entry.get("source_ref") or ""),
        harvested_ts=harvested_ts,
        corpus_version=corpus_version,
    )


# ── harvest sources ───────────────────────────────────────────────

def _verified_fix_candidates(
    journal_path: Any,
) -> tuple[int, list[tuple[str, str]], str]:
    """Read the incident journal's verified fixes, read-only.

    Returns ``(found, candidates, note)`` — candidates are
    ``(text, source_ref)`` pairs. A missing/unreadable journal is
    an honest zero with the reason as the note, never an exception:
    the journal is another organ's store, and its absence means
    "nothing verified yet", not "corpus broken".
    """
    if journal_path is None:
        return 0, [], "no incident journal path resolved — source skipped"
    path = Path(journal_path)
    if not path.exists():
        return 0, [], f"incident journal not found at {path} — nothing to harvest"
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT signature, strategy, detail FROM verified_fixes"
                " ORDER BY first_seen, signature"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return 0, [], f"incident journal unreadable ({exc}) — nothing harvested from it"
    candidates: list[tuple[str, str]] = []
    for signature, strategy, detail in rows:
        text = str(detail or "").strip()
        if not text and strategy:
            text = f"verified self-heal fix via {strategy}"
        if not text:
            continue
        candidates.append((text, f"verified_fix:{signature}"))
    return len(rows), candidates, ""


def _receipt_candidates(
    receipts: Iterable[Any] | None,
) -> tuple[int, list[tuple[str, str]], str]:
    """Owner-directed operation shapes from police receipts.

    Qualifies: ``action == "report_only"`` AND the receipt's details
    record ``owner_directed`` — the receipts where the police itself
    classified the work as the owner's and reported instead of
    acting. The exemplar text is the receipt's target (the owner
    work's shape), falling back to its reason.
    """
    if receipts is None:
        return 0, [], "no receipts supplied — source skipped"
    found = 0
    candidates: list[tuple[str, str]] = []
    for receipt in receipts:
        if _field(receipt, "action") != "report_only":
            continue
        details = _field(receipt, "details")
        if not isinstance(details, dict) \
                or details.get("owner_directed") is not True:
            continue
        found += 1
        receipt_id = str(_field(receipt, "id") or "")
        text = str(_field(receipt, "target") or "").strip() \
            or str(_field(receipt, "reason") or "").strip()
        if not text or not receipt_id:
            continue
        candidates.append((text, f"police_receipt:{receipt_id}"))
    note = "" if found else "no owner-directed receipts recorded on this surface yet"
    return found, candidates, note


# ── the store ─────────────────────────────────────────────────────

class CorpusStore:
    """The persisted, versioned self-corpus.

    One shared instance per file in production (see
    ``immune/wiring.shared_corpus_store``): chat states for
    different chats all test candidates against the same grown
    corpus. ``path=None`` is an in-memory store (seeds + session
    harvests, nothing persisted) — the chat surface's fallback when
    the shared store cannot load, and the shape tests use.
    """

    def __init__(self, path: Any = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._version = 0
        self._harvested: list[SelfExemplar] = []
        #: Why the on-disk corpus could not be read, when it could
        #: not. Empty when the file loaded (or never existed): a
        #: degraded corpus is seed-only AND says so, never a crash
        #: and never a silent thinning of "self".
        self.degraded_reason = ""
        self._corpus = SelfCorpus(seed_exemplars())
        if self.path is not None and self.path.exists():
            self._load()

    # ── reads ─────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        """The current corpus version (0 = seeds only). Monotonic:
        it moves only when a harvest actually admits an exemplar."""
        return self._version

    def corpus(self) -> SelfCorpus:
        """The live corpus — the SAME object on every call.

        A :class:`~nomorals.immune.thymus.Thymus` holding it sees
        harvests as they land: admission refreshes the exemplar
        list in place, so no consumer needs re-wiring to test
        against the grown corpus.
        """
        return self._corpus

    def exemplars(self) -> list[SelfExemplar]:
        """Every exemplar, seeds first, then harvested in admission
        order."""
        with self._lock:
            return list(self._corpus.exemplars)

    def counts_by_provenance(self) -> dict[str, int]:
        """Exemplar counts per provenance kind (every known kind is
        present, zero when none) — the corpus's size is a reported
        fact, never a silent constant."""
        counts: dict[str, int] = {kind: 0 for kind in PROVENANCES}
        with self._lock:
            for exemplar in self._corpus.exemplars:
                kind = exemplar.provenance or "unprovenanced"
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def corpus_at(self, version: int) -> SelfCorpus:
        """The corpus as of ``version``: seeds plus the harvested
        exemplars admitted at or before it.

        Prior versions stay readable without snapshotting: every
        harvested exemplar carries the version that admitted it,
        so history is a filter, not an archive of copies.
        """
        with self._lock:
            return SelfCorpus([
                exemplar for exemplar in self._corpus.exemplars
                if exemplar.corpus_version <= version
            ])

    # ── persistence ───────────────────────────────────────────────

    def _refresh(self) -> None:
        self._corpus.exemplars[:] = seed_exemplars() + list(self._harvested)

    def _load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict) \
                    or data.get("format") != CORPUS_FORMAT:
                raise ValueError("not a self-corpus file")
            version = int(data.get("version") or 0)
            harvested = [
                _exemplar_from_dict(entry)
                for entry in (data.get("exemplars") or [])
                if isinstance(entry, dict)
            ]
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            self.degraded_reason = (
                f"corpus file unreadable ({exc}) — seed-only corpus in use")
            _log.warning(
                "immune self-corpus %s unreadable — degrading to "
                "seed-only: %s", self.path, exc)
            self._version = 0
            self._harvested = []
            self._refresh()
            return
        self._version = version
        self._harvested = harvested
        self._refresh()

    def save(self) -> None:
        """Persist the harvested population + version, atomically
        (same-dir tmp + ``os.replace``, the baseline store's
        discipline). Seeds are code, not data — they are never
        written; the file carries only what harvest admitted."""
        if self.path is None:
            return
        with self._lock:
            payload = {
                "format": CORPUS_FORMAT,
                "version": self._version,
                "saved_ts": time.time(),
                "exemplars": [
                    _exemplar_to_dict(e) for e in self._harvested],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(
                f"{self.path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8")
            os.replace(tmp, self.path)

    # ── harvest ───────────────────────────────────────────────────

    def harvest(
        self,
        *,
        journal_path: Any = None,
        receipts: Iterable[Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Grow the corpus from Devon's own records — one rung.

        Reads the verified fixes from the incident journal at
        ``journal_path`` (read-only; a missing journal is an honest
        zero) and the owner-directed receipts among ``receipts``
        (the calling surface's recorded police receipts — in-memory
        there, so only what that surface has recorded is
        reachable), dedupes against the existing corpus by source
        ref and by normalized text, and admits what is new as ONE
        new corpus version. Idempotent: a second harvest over the
        same sources admits zero and the version does not move.

        Returns the report the surface renders: per-source
        ``found`` / ``added`` / ``note`` plus the version movement.
        Harvested exemplars are self evidence only — this method
        touches no antibody store and promotes nothing.
        """
        now = time.time() if now is None else now
        with self._lock:
            version_before = self._version
            known_refs = {
                e.source_ref for e in self._harvested if e.source_ref}
            known_texts = {
                normalize_text(e.text)
                for e in self._corpus.exemplars if normalize_text(e.text)}
            sources: dict[str, dict[str, Any]] = {}
            pending: list[SelfExemplar] = []

            vf_found, vf_candidates, vf_note = _verified_fix_candidates(
                journal_path)
            rc_found, rc_candidates, rc_note = _receipt_candidates(
                receipts)
            for kind, found, candidates, note in (
                (PROVENANCE_VERIFIED_FIX, vf_found, vf_candidates, vf_note),
                (PROVENANCE_OWNER_RECEIPT, rc_found, rc_candidates, rc_note),
            ):
                added = 0
                for text, source_ref in candidates:
                    if source_ref in known_refs:
                        continue
                    normalized = normalize_text(text)
                    if not normalized or normalized in known_texts:
                        continue
                    pending.append(SelfExemplar(
                        text=text,
                        provenance=kind,
                        source_ref=source_ref,
                        harvested_ts=now,
                        corpus_version=version_before + 1,
                    ))
                    known_refs.add(source_ref)
                    known_texts.add(normalized)
                    added += 1
                sources[kind] = {
                    "found": found, "added": added, "note": note}

            if pending:
                self._version = version_before + 1
                self._harvested.extend(pending)
                self._refresh()
                self.save()
            return {
                "version_before": version_before,
                "version_after": self._version,
                "added": len(pending),
                "total": len(self._corpus.exemplars),
                "sources": sources,
                "degraded_reason": self.degraded_reason,
            }
