"""Store-integrity guard — unreadable is not empty.

The data-loss family pattern (shipped precedent: AA-73, the wisdom
blob index). A small persistent store whose on-disk state exists but
cannot be read must not treat that state as empty: the next
load-then-rewrite mutation would silently overwrite the only copy of
the owner's data with a near-empty image.

The repair, shared by every store in the family:

1. Distinguish *missing* (legitimately empty / new) from *unreadable*
   (exists but unparseable, wrong-shaped, or the read itself crashed).
2. On unreadable: log at ERROR, record the failure in the ledger, and
   quarantine the bytes (files rename to ``<name>.corrupt``; KV values
   are copied to ``<key>.corrupt`` and left in place so every later
   load keeps seeing the corruption instead of overwriting it).
3. The owning store refuses to rewrite while its last load was
   unreadable — mutations surface :class:`StoreUnreadableError` (or
   the store's own honest equivalent) instead of succeeding over the
   loss. Reads still proceed from the empty image, loudly.

Result convention for the loaders: ``(value, None)`` on success,
``(None, None)`` when the state is simply missing, ``(None, exc)``
when unreadable. A ``ValueError`` exc means the stored bytes are
wrong-shaped (durable — retrying the read will not heal it); any
other exception means the read itself failed (possibly transient —
the stored bytes were left untouched).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .failures import record_failure as _core_record_failure

__all__ = [
    "CORRUPT_SUFFIX",
    "StoreUnreadableError",
    "note_unreadable",
    "quarantine_file",
    "read_json_file",
    "read_kv_json",
]

#: Suffix appended to a store's file name / KV key to quarantine its
#: unreadable bytes without destroying them.
CORRUPT_SUFFIX = ".corrupt"

#: A failures-ledger record function: ``record(site, exc)``.
RecordFn = Callable[[str, BaseException], None]


class StoreUnreadableError(RuntimeError):
    """A mutation was refused because the store's persisted state
    exists but could not be read. The original bytes were quarantined
    (see ``<name>.corrupt``); rewriting from the empty in-memory image
    would have destroyed them."""


def note_unreadable(site: str, exc: BaseException, *, logger: Any,
                    record: RecordFn | None = None) -> None:
    """Surface an unreadable store: ERROR log + failure ledger.

    ``record`` defaults to the core ledger; packages with their own
    ledger (books, agents, community, …) pass theirs so the entry
    lands in the partition their tooling reads.
    """
    logger.error("%s: persisted state unreadable (%s: %s) — corruption, "
                 "not emptiness; refusing to treat it as empty",
                 site, type(exc).__name__, exc)
    (record or _core_record_failure)(site, exc)


def quarantine_file(path: Path, *, site: str, exc: BaseException,
                    logger: Any,
                    record: RecordFn | None = None) -> Path | None:
    """Log + ledger the corruption and rename ``path`` aside to
    ``<name>.corrupt``, preserving the bytes for recovery. Returns the
    quarantine path, or None when the rename itself failed (the
    failure is still logged + ledgered either way)."""
    note_unreadable(site, exc, logger=logger, record=record)
    target = path.with_name(path.name + CORRUPT_SUFFIX)
    try:
        path.replace(target)
    except OSError as qexc:
        logger.warning("could not quarantine corrupt store %s: %s",
                       path, qexc)
        return None
    return target


def read_json_file(path: Path, *, site: str, logger: Any,
                   expect: type = dict,
                   record: RecordFn | None = None) -> tuple[Any, Any]:
    """Read a whole-file JSON store with the family guard.

    ``(value, None)`` on success · ``(None, None)`` when the file does
    not exist (legitimately empty) · ``(None, exc)`` when the file
    exists but is unreadable or its root is not an ``expect`` — the
    bytes are quarantined in that case.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        quarantine_file(path, site=site, exc=exc, logger=logger,
                        record=record)
        return None, exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        quarantine_file(path, site=site, exc=exc, logger=logger,
                        record=record)
        return None, exc
    if not isinstance(data, expect):
        exc = ValueError(
            f"{path.name}: root is {type(data).__name__}, "
            f"expected {expect.__name__}")
        quarantine_file(path, site=site, exc=exc, logger=logger,
                        record=record)
        return None, exc
    return data, None


def read_kv_json(store: Any, key: str, *, site: str, logger: Any,
                 expect: type = dict,
                 record: RecordFn | None = None) -> tuple[Any, Any]:
    """KV analogue of :func:`read_json_file`, over the store's
    ``get``/``set_raw`` pair (the contract the family's KV consumers
    already use — no wider store API is assumed).

    ``(value, None)`` on success · ``(None, None)`` when the key is
    missing · ``(None, exc)`` when unreadable. A ``get`` crash leaves
    the stored bytes untouched (the failure may be transient); a
    wrong-shaped stored value is copied to ``<key>.corrupt`` and left
    in place, so later loads keep detecting it instead of any writer
    overwriting it.
    """
    try:
        value = store.get(key)
    except Exception as exc:  # noqa: BLE001 — a crashed read is the trigger
        note_unreadable(site, exc, logger=logger, record=record)
        return None, exc
    if value is None:
        return None, None
    if isinstance(value, expect):
        return value, None
    exc = ValueError(
        f"kv {key!r}: stored value is {type(value).__name__}, "
        f"expected {expect.__name__}")
    note_unreadable(site, exc, logger=logger, record=record)
    try:
        raw = value if isinstance(value, str) else json.dumps(value)
        store.set_raw(key + CORRUPT_SUFFIX, raw, "json")
    except Exception:  # noqa: BLE001 — the copy is best-effort; refusal stands
        pass
    return None, exc
