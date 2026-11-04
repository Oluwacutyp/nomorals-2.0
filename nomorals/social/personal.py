"""Personal surfaces (§15 slices 2–3): status saving, view-once
preservation, and DM antidelete/antiedit for the owner's OWN paired
accounts.

These are as-him surfaces.  The §33 system-output gate does not
apply — nothing here ever posts anywhere.  The organ has no send
capability at all: it observes what the owner's accounts receive,
keeps what the owner told it to keep, and answers the owner in his
DM (via ``/pair saved`` / ``/pair recovered``).  A counterpart never
sees any of it; nothing is reposted into a conversation, ever.

Three surfaces, each behind a PER-ACCOUNT toggle in the pairing
registry (all default OFF — the owner enables them account by
account):

- **Status saving** — WhatsApp statuses / Telegram stories the
  account can see land in a per-account store (media copied, text
  journaled) with poster + timestamp.  A saver, not a viewer.
- **View-once preservation** — view-once / self-destruct media that
  ARRIVES at the account is copied into the same owner-only store
  before it vanishes.
- **DM antidelete / antiedit** — personal DM messages are cached in
  a bounded ring per conversation; when the counterpart deletes or
  edits, the original is written to the recovery journal for the
  owner to read.  (Group antidelete evidence is the group defense
  organ's job — this is the DM/personal surface only, and it does
  not touch group chats at all.)

Store discipline, matching the registry's: the DM cache is a JSON
file written atomically; an unreadable cache raises (quarantined
aside), never a silent reset — cached originals are evidence.
Saved media is bounded per account by count AND bytes; pruning
takes the oldest first and the journal keeps the fact of the prune.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .pairing import (PERSONAL_TOGGLES, AccountRecord, PairingRegistry,
                      resolve_account)

#: Bounds (constructor-overridable; the values are the shipped caps).
DM_CACHE_PER_CHAT = 100        # messages kept per conversation
DM_CACHE_CHATS = 200           # conversations kept per account
SAVED_PER_ACCOUNT = 500        # saved status/view-once items
SAVED_MAX_BYTES = 512 * 1024 * 1024
JOURNAL_TAIL = 500             # entries returned by one listing


class PersonalStoreError(RuntimeError):
    """A personal-surface store exists but cannot be read."""


def _slug(account_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (account_id or "").lower()).strip("_")


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, tail: int = JOURNAL_TAIL) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-max(tail * 4, tail):]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out[-tail:]


class PersonalVault:
    """The owner-only store: saved media, journals, the DM cache."""

    def __init__(self, root: str | Path, *,
                 saved_per_account: int = SAVED_PER_ACCOUNT,
                 saved_max_bytes: int = SAVED_MAX_BYTES,
                 dm_per_chat: int = DM_CACHE_PER_CHAT,
                 dm_chats: int = DM_CACHE_CHATS) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.saved_per_account = int(saved_per_account)
        self.saved_max_bytes = int(saved_max_bytes)
        self.dm_per_chat = int(dm_per_chat)
        self.dm_chats = int(dm_chats)
        self._lock = threading.RLock()
        self._cache_path = self.root / "dm_cache.json"
        self._cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._load_cache()

    # ── paths ───────────────────────────────────────────────────────
    def account_dir(self, account_id: str) -> Path:
        d = self.root / _slug(account_id) / "saved"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def recovery_path(self) -> Path:
        return self.root / "recoveries.jsonl"

    # ── saved media (statuses + view-once) ──────────────────────────
    def save_media(self, account_id: str, kind: str, src_path: str, *,
                   platform: str, counterpart_id: str = "",
                   poster_name: str = "", media_kind: str = "",
                   source_id: str = "", ts: float = 0.0) -> dict[str, Any]:
        """Copy one media file into the owner's store + journal it.

        Raises FileNotFoundError when the source is gone — a save
        that did not happen must never journal as one."""
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"media source gone: {src_path}")
        ts = float(ts or time.time())
        dest_dir = self.account_dir(account_id)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))
        safe_src = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.name) or "media"
        dest = dest_dir / f"{stamp}-{kind}-{safe_src}"
        n = 1
        while dest.exists():
            n += 1
            dest = dest_dir / f"{stamp}-{kind}-{n}-{safe_src}"
        shutil.copyfile(src, dest)
        entry = {
            "ts": ts, "kind": kind, "account": account_id,
            "platform": platform, "counterpart_id": counterpart_id,
            "poster_name": poster_name, "media_kind": media_kind,
            "source_id": source_id, "path": str(dest),
            "bytes": dest.stat().st_size,
        }
        with self._lock:
            _append_jsonl(self.journal_path, entry)
            self._prune_account(account_id)
        return entry

    def save_text(self, account_id: str, kind: str, text: str, *,
                  platform: str, counterpart_id: str = "",
                  poster_name: str = "", source_id: str = "",
                  ts: float = 0.0) -> dict[str, Any]:
        """A media-less item (text status): the journal IS the store."""
        entry = {
            "ts": float(ts or time.time()), "kind": kind,
            "account": account_id, "platform": platform,
            "counterpart_id": counterpart_id,
            "poster_name": poster_name, "media_kind": "text",
            "source_id": source_id, "text": text, "path": "",
            "bytes": 0,
        }
        with self._lock:
            _append_jsonl(self.journal_path, entry)
        return entry

    def _prune_account(self, account_id: str) -> None:
        """Enforce the per-account count + byte caps, oldest first.

        Only rows whose file still exists count: the journal is
        append-only, so a pruned item's original entry stays in it
        forever — counting ghosts would make every later pass
        delete real files to satisfy a cap already met."""
        rows = [r for r in _read_jsonl(self.journal_path, tail=100000)
                if r.get("account") == account_id and r.get("path")
                and Path(str(r["path"])).exists()]
        rows.sort(key=lambda r: float(r.get("ts") or 0))
        total = sum(int(r.get("bytes") or 0) for r in rows)
        while rows and (len(rows) > self.saved_per_account
                        or total > self.saved_max_bytes):
            victim = rows.pop(0)
            total -= int(victim.get("bytes") or 0)
            try:
                Path(str(victim["path"])).unlink(missing_ok=True)
            except OSError:
                pass
            _append_jsonl(self.journal_path, {
                "ts": time.time(), "kind": "pruned",
                "account": account_id,
                "platform": victim.get("platform", ""),
                "counterpart_id": victim.get("counterpart_id", ""),
                "poster_name": victim.get("poster_name", ""),
                "media_kind": victim.get("media_kind", ""),
                "source_id": victim.get("source_id", ""),
                "path": "", "bytes": 0,
                "note": f"cap prune of {victim.get('kind')} item "
                        f"from {victim.get('ts')}"})

    def journal(self, account_id: str = "",
                kind: str = "") -> list[dict[str, Any]]:
        rows = _read_jsonl(self.journal_path)
        if account_id:
            rows = [r for r in rows if r.get("account") == account_id]
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        return rows

    def saved_entries(self, account_id: str = "") -> list[dict[str, Any]]:
        return [r for r in self.journal(account_id)
                if r.get("kind") in ("status", "story", "view_once")]

    # ── recoveries ──────────────────────────────────────────────────
    def add_recovery(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            _append_jsonl(self.recovery_path, entry)
        return entry

    def recoveries(self, account_id: str = "") -> list[dict[str, Any]]:
        rows = _read_jsonl(self.recovery_path)
        if account_id:
            rows = [r for r in rows if r.get("account") == account_id]
        return rows

    # ── DM message cache (bounded rings) ────────────────────────────
    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(
                encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("dm cache is not an object")
            self._cache = data
        except Exception as exc:  # noqa: BLE001 - quarantine, no reset
            aside = self._cache_path.with_suffix(
                f".corrupt-{int(time.time())}.json")
            try:
                os.replace(self._cache_path, aside)
            except OSError:
                aside = self._cache_path  # type: ignore[assignment]
            raise PersonalStoreError(
                f"DM message cache unreadable ({exc}); original "
                f"moved to {aside} — cached originals were not "
                "silently forgotten") from exc

    def _save_cache(self) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        os.replace(tmp, self._cache_path)

    def note_message(self, account_id: str, chat_key: str,
                     entry: dict[str, Any]) -> None:
        with self._lock:
            acct = self._cache.setdefault(account_id, {})
            ring = acct.setdefault(chat_key, [])
            mid = str(entry.get("message_id") or "")
            if mid:
                ring[:] = [r for r in ring
                           if str(r.get("message_id") or "") != mid]
            ring.append(dict(entry))
            if len(ring) > self.dm_per_chat:
                del ring[:len(ring) - self.dm_per_chat]
            if len(acct) > self.dm_chats:
                # evict the conversation whose newest entry is oldest
                def _newest(k: str) -> float:
                    rows = acct.get(k) or []
                    return float(rows[-1].get("ts") or 0) if rows else 0
                for k in sorted(acct, key=_newest)[:len(acct)
                                                   - self.dm_chats]:
                    del acct[k]
            self._save_cache()

    def get_message(self, account_id: str, chat_key: str,
                    message_id: str) -> dict[str, Any] | None:
        with self._lock:
            ring = (self._cache.get(account_id) or {}).get(
                chat_key) or []
            for row in ring:
                if str(row.get("message_id") or "") == str(message_id):
                    return dict(row)
            return None

    def drop_message(self, account_id: str, chat_key: str,
                     message_id: str) -> None:
        with self._lock:
            ring = (self._cache.get(account_id) or {}).get(chat_key)
            if not ring:
                return
            ring[:] = [r for r in ring
                       if str(r.get("message_id") or "")
                       != str(message_id)]
            self._save_cache()


class PersonalSurfaces:
    """The observation service the runtime feeds and /pair reads.

    Every method is gated on the account's own toggles and returns
    what it DID (entries written), so callers and tests can see the
    truth.  Nothing here sends a message anywhere — recovery is
    pull-only, in the owner's DM, by his command.
    """

    def __init__(self, registry: PairingRegistry, vault: PersonalVault,
                 *, story_fetcher: Callable[[AccountRecord],
                                            list[dict[str, Any]]] | None
                 = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.registry = registry
        self.vault = vault
        self.story_fetcher = story_fetcher
        self._clock = clock
        self._seen_status: set[tuple[str, str]] = set()
        self._seen_stories: set[tuple[str, str]] = set()
        for row in vault.journal():
            sid = str(row.get("source_id") or "")
            if sid and row.get("kind") == "status":
                self._seen_status.add((str(row.get("account") or ""),
                                       sid))
            if sid and row.get("kind") == "story":
                self._seen_stories.add((str(row.get("account") or ""),
                                        sid))

    # ── account resolution ──────────────────────────────────────────
    def account_for(self, platform: str,
                    account_id: str = "") -> AccountRecord | None:
        return resolve_account(self.registry, platform, account_id)

    def ensure_primary(self, platform: str) -> AccountRecord:
        """The gateway's live session for a platform, as a PERSISTED
        registry record — an owner act (he runs /pair toggle/use on
        it), never an observation side effect.  Paired accounts win;
        when the registry has none for the platform, the live
        session is recorded as ``<platform>:primary`` (status
        linked — it demonstrably is) so its toggles have somewhere
        honest to live."""
        existing = self.registry.accounts_for(platform)
        if existing:
            return resolve_account(self.registry, platform) \
                or existing[0]
        rec = AccountRecord(
            id=f"{platform}:primary", platform=platform,
            identity="primary", label="live session (gateway)",
            status="linked", created_at=self._clock())
        return self.registry.upsert(rec)

    def resolve_event_account(self, platform: str,
                              account_id: str = "") -> AccountRecord:
        """The account an inbound event belongs to.  Resolution
        never writes: with no registry account for the platform the
        event resolves to an unpersisted ``<platform>:primary``
        stand-in whose toggles are all off — observation of a
        platform the owner never paired stays a no-op, and the
        registry only grows by his acts."""
        rec = self.account_for(platform, account_id)
        if rec is not None:
            return rec
        return AccountRecord(
            id=f"{platform}:primary", platform=platform,
            identity="primary", label="live session (gateway)",
            status="linked")

    # ── message observation ─────────────────────────────────────────
    def observe_message(self, platform: str, *, chat_key: str,
                        counterpart_id: str = "",
                        sender_id: str = "", sender_name: str = "",
                        message_id: str = "", text: str = "",
                        media: list[dict[str, Any]] | None = None,
                        view_once: bool = False, ttl_seconds: int = 0,
                        incoming: bool = True,
                        account_id: str = "",
                        ts: float = 0.0) -> list[dict[str, Any]]:
        """One DM message the account saw.  Caches it when the
        antidelete toggle is on; preserves its media when the
        view-once toggle is on and the message is view-once /
        self-destruct.  Returns the vault entries written."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        media = media or []
        ts = float(ts or self._clock())
        if rec.toggle("antidelete") and message_id:
            self.vault.note_message(rec.id, chat_key, {
                "message_id": message_id, "sender_id": sender_id,
                "sender_name": sender_name, "text": text,
                "media": [str(m.get("path") or "") for m in media],
                "incoming": bool(incoming), "ts": ts})
        ephemeral = bool(view_once) or int(ttl_seconds or 0) > 0
        if ephemeral and media and rec.toggle("view_once"):
            for m in media:
                path = str(m.get("path") or "")
                if not path:
                    continue
                try:
                    out.append(self.vault.save_media(
                        rec.id, "view_once", path, platform=platform,
                        counterpart_id=counterpart_id or sender_id,
                        poster_name=sender_name,
                        media_kind=str(m.get("kind") or ""),
                        source_id=message_id, ts=ts))
                except FileNotFoundError:
                    _append_jsonl(self.vault.journal_path, {
                        "ts": ts, "kind": "view_once_missed",
                        "account": rec.id, "platform": platform,
                        "counterpart_id": counterpart_id or sender_id,
                        "poster_name": sender_name,
                        "media_kind": str(m.get("kind") or ""),
                        "source_id": message_id, "path": "",
                        "bytes": 0,
                        "note": "media was gone before it could be "
                                "preserved"})
        return out

    def observe_delete(self, platform: str, *, chat_key: str,
                       counterpart_id: str = "",
                       deleted_ids: list[str] | tuple[str, ...] = (),
                       account_id: str = "",
                       ts: float = 0.0) -> list[dict[str, Any]]:
        """Counterpart deleted message(s) in a DM.  Each original we
        hold is written to the recovery journal; each id we never
        cached is journaled as unrecovered — the owner sees both,
        the counterpart sees nothing."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        if not rec.toggle("antidelete"):
            return out
        ts = float(ts or self._clock())
        for mid in deleted_ids:
            cached = self.vault.get_message(rec.id, chat_key, str(mid))
            entry = {
                "ts": ts, "kind": "delete", "account": rec.id,
                "platform": platform, "chat_key": chat_key,
                "counterpart_id": counterpart_id
                or (str(cached.get("sender_id") or "") if cached
                    else ""),
                "message_id": str(mid),
                "recovered": cached is not None,
                "original_text": (str(cached.get("text") or "")
                                  if cached else ""),
                "original_media": (list(cached.get("media") or [])
                                   if cached else []),
                "original_ts": (float(cached.get("ts") or 0)
                                if cached else 0.0),
            }
            out.append(self.vault.add_recovery(entry))
            if cached is not None:
                self.vault.drop_message(rec.id, chat_key, str(mid))
        return out

    def observe_edit(self, platform: str, *, chat_key: str,
                     counterpart_id: str = "", message_id: str = "",
                     new_text: str = "", sender_id: str = "",
                     sender_name: str = "",
                     account_id: str = "",
                     ts: float = 0.0) -> dict[str, Any] | None:
        """Counterpart edited a DM message.  When the cached
        original differs, the pair (original, new) is journaled and
        the cache moves to the new text — a second edit compares
        against the newest, never a ghost."""
        rec = self.resolve_event_account(platform, account_id)
        if not rec.toggle("antidelete") or not message_id:
            return None
        ts = float(ts or self._clock())
        cached = self.vault.get_message(rec.id, chat_key,
                                        str(message_id))
        if cached is None:
            # Nothing to recover against; cache the edit as the
            # earliest known state so a later delete still lands.
            self.vault.note_message(rec.id, chat_key, {
                "message_id": message_id, "sender_id": sender_id,
                "sender_name": sender_name, "text": new_text,
                "media": [], "incoming": True, "ts": ts})
            return None
        original = str(cached.get("text") or "")
        if original == new_text:
            return None
        entry = {
            "ts": ts, "kind": "edit", "account": rec.id,
            "platform": platform, "chat_key": chat_key,
            "counterpart_id": counterpart_id
            or str(cached.get("sender_id") or ""),
            "message_id": str(message_id), "recovered": True,
            "original_text": original, "new_text": new_text,
            "original_ts": float(cached.get("ts") or 0),
        }
        self.vault.add_recovery(entry)
        cached["text"] = new_text
        cached["ts"] = ts
        self.vault.note_message(rec.id, chat_key, cached)
        return entry

    # ── statuses / stories ──────────────────────────────────────────
    def observe_status_post(self, platform: str, *, poster_id: str,
                            poster_name: str = "", text: str = "",
                            media: list[dict[str, Any]] | None = None,
                            source_id: str = "",
                            account_id: str = "",
                            ts: float = 0.0) -> list[dict[str, Any]]:
        """One status post the account can see (WhatsApp
        status@broadcast feed).  Saved when the account's
        status-saving toggle is on; the same post never saves
        twice."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        if not rec.toggle("status_saving"):
            return out
        ts = float(ts or self._clock())
        dedupe_key = (rec.id, str(source_id or ""))
        if source_id and dedupe_key in self._seen_status:
            return out
        media = media or []
        if media:
            for m in media:
                path = str(m.get("path") or "")
                if not path:
                    continue
                try:
                    out.append(self.vault.save_media(
                        rec.id, "status", path, platform=platform,
                        counterpart_id=poster_id,
                        poster_name=poster_name,
                        media_kind=str(m.get("kind") or ""),
                        source_id=source_id, ts=ts))
                except FileNotFoundError:
                    continue
        elif text:
            out.append(self.vault.save_text(
                rec.id, "status", text, platform=platform,
                counterpart_id=poster_id, poster_name=poster_name,
                source_id=source_id, ts=ts))
        if source_id and out:
            self._seen_status.add(dedupe_key)
        return out

    def cycle_stories(self, platform: str = "telegram",
                      account_id: str = "") -> list[dict[str, Any]]:
        """One story sweep: ask the fetcher for the account's
        currently-visible stories and save the new ones.  No
        fetcher (adapter exposes none) → nothing happens and the
        caller can say exactly that."""
        if self.story_fetcher is None:
            return []
        rec = self.resolve_event_account(platform, account_id)
        if not rec.toggle("status_saving"):
            return []
        out: list[dict[str, Any]] = []
        for story in self.story_fetcher(rec) or []:
            sid = str(story.get("story_id") or "")
            dedupe_key = (rec.id, sid)
            if sid and dedupe_key in self._seen_stories:
                continue
            path = str(story.get("path") or "")
            ts = float(story.get("ts") or self._clock())
            entry = None
            if path:
                try:
                    entry = self.vault.save_media(
                        rec.id, "story", path, platform=platform,
                        counterpart_id=str(story.get("poster_id")
                                           or ""),
                        poster_name=str(story.get("poster_name")
                                        or ""),
                        media_kind=str(story.get("kind") or ""),
                        source_id=sid, ts=ts)
                except FileNotFoundError:
                    entry = None
            if entry is not None:
                out.append(entry)
                if sid:
                    self._seen_stories.add(dedupe_key)
        return out

    # ── toggles ─────────────────────────────────────────────────────
    def set_toggle(self, account_id: str, name: str,
                   on: bool) -> AccountRecord:
        if name not in PERSONAL_TOGGLES:
            raise ValueError(
                f"unknown personal toggle {name!r} — known: "
                f"{', '.join(PERSONAL_TOGGLES)}")
        return self.registry.set_toggles(account_id, **{name: on})
