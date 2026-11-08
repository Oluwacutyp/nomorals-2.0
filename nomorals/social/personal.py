"""Personal surfaces (§15 slices 2–3): status saving, view-once
preservation, and DM antidelete/antiedit for the owner's OWN paired
accounts.

These are as-him surfaces.  The §33 system-output gate does not
apply — nothing here ever posts anywhere.  The organ has no send
capability at all: it observes what the owner's accounts receive,
keeps what the owner told it to keep, and answers the owner in his
DM (via ``/pair saved`` / ``/pair recovered``).  A counterpart never
sees any of it; nothing is reposted into a conversation, ever.

Five surfaces, each behind a PER-ACCOUNT toggle in the pairing
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
- **Always online** (``always_online``) — while the account's
  surface is live, its presence is asserted "available" through
  the platform's real verb (bridge presence on WhatsApp, Telethon
  account status on Telegram) and re-asserted on the housekeeping
  beat.  An owner stop order defeats it absolutely; a surface
  that is down asserts nothing.  Episodes (online asserted /
  released) are journaled once per transition.
- **Status viewing** (``status_viewing``) — statuses/stories the
  account can see are marked VIEWED as the owner (WhatsApp read
  receipts on status@broadcast, Telegram ReadStories) — viewing
  only: no replies, no reactions, each post viewed at most once
  per process.  Independent of saving: a man can have his
  statuses read without Devon keeping copies, and vice versa.
- **Status posting** — not a toggle and not this organ's act:
  the owner's /poststatus command posts through the adapters'
  own verbs (bridge status_send / Telethon SendStory) and the
  landed post is journaled here (``status_posted``), so an act
  taken as him is never a silent one.

The acting half (presence + viewing) rides injected callables —
the same absence-honesty as ``story_fetcher``: no callable wired
(the adapter exposes no verb) means the surface reports exactly
that, and nothing is ever faked as done.

Store discipline, matching the registry's: the DM cache is a JSON
file written atomically; an unreadable cache raises (quarantined
aside), never a silent reset — cached originals are evidence.
Saved media is bounded per account by count AND bytes; pruning
takes the oldest first and the journal keeps the fact of the prune.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .pairing import PERSONAL_TOGGLES, AccountRecord, PairingRegistry, resolve_account

#: Bounds (constructor-overridable; the values are the shipped caps).
DM_CACHE_PER_CHAT = 100        # messages kept per conversation
DM_CACHE_CHATS = 200           # conversations kept per account
SAVED_PER_ACCOUNT = 500        # saved status/view-once items
SAVED_MAX_BYTES = 512 * 1024 * 1024
JOURNAL_TAIL = 500             # entries returned by one listing
#: Mass-tag DM flags (wave F): at most one delivered flag per
#: (account, chat) per hour — a tagall-heavy room journals one
#: flag an hour with the suppressed count carried into it, not
#: a DM per message.
MASS_TAG_FLAG_COOLDOWN = 3600.0


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
                if r.get("kind") in ("status", "story", "view_once",
                                     "channel_post")]

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
    pull-only, in the owner's DM, by his command.  The presence
    pack's acting seams (presence assertion, status viewing) post
    nothing either: they flip the account's own presence and read
    receipts, through the injected callables, only where the
    owner's toggles say so.
    """

    def __init__(self, registry: PairingRegistry, vault: PersonalVault,
                 *, story_fetcher: Callable[[AccountRecord],
                                            list[dict[str, Any]]] | None
                 = None,
                 presence_setter: Callable[[str, AccountRecord, bool],
                                           bool] | None = None,
                 status_viewer: Callable[[AccountRecord,
                                          dict[str, Any]], bool] | None
                 = None,
                 story_viewer: Callable[[AccountRecord, str, int],
                                        bool] | None = None,
                 status_reactor: Callable[[AccountRecord,
                                           dict[str, Any]], bool] | None
                 = None,
                 channel_fetcher: Callable[[AccountRecord],
                                           list[dict[str, Any]]] | None
                 = None,
                 channel_viewer: Callable[[AccountRecord,
                                           dict[str, Any]], bool] | None
                 = None,
                 call_rejector: Callable[[AccountRecord,
                                          dict[str, Any]], bool] | None
                 = None,
                 status_react_pool: Callable[[AccountRecord],
                                             list[str]] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.registry = registry
        self.vault = vault
        self.story_fetcher = story_fetcher
        #: The acting seams (presence pack).  Each returns True only
        #: when the platform verb actually landed; None = the
        #: surface has no such verb wired, which the state surfaces
        #: report as absence, never as success.
        self.presence_setter = presence_setter
        self.status_viewer = status_viewer
        self.story_viewer = story_viewer
        #: The chat & privacy wave's acting seams, same contract:
        #: status_reactor places one reaction on a contact's status
        #: post as the owner; call_rejector rejects one incoming
        #: WhatsApp call as the owner.  Both are ACTIONS on his
        #: account, so their observe paths additionally honor the
        #: owner's sticky stop — a stopped platform never acts.
        self.status_reactor = status_reactor
        self.call_rejector = call_rejector
        #: The channel-stories seams (2026-11-07), same contract
        #: as the story seams: the fetcher returns the followed
        #: channels' recent posts (WhatsApp newsletters), the
        #: viewer marks one post viewed as the owner.  None =
        #: the surface has no channel feed wired, reported as
        #: absence, never as success.
        self.channel_fetcher = channel_fetcher
        self.channel_viewer = channel_viewer
        #: The rotation's custom half: her /customemoji set's
        #: unicode faces for this account, supplied by the
        #: runtime (the expression library lives a layer away;
        #: this surface stays transport- and library-free).
        #: None → the pool is the account's own + the built-in.
        self.status_react_pool = status_react_pool
        self._clock = clock
        self._seen_status: set[tuple[str, str]] = set()
        self._missed_status: set[tuple[str, str]] = set()
        self._seen_channel_posts: set[tuple[str, str]] = set()
        self._missed_channel_posts: set[tuple[str, str]] = set()
        self._viewed_channel_posts: set[tuple[str, str]] = set()
        self._view_failed_channel_posts: set[tuple[str, str]] = set()
        self._seen_stories: set[tuple[str, str]] = set()
        self._viewed_status: set[tuple[str, str]] = set()
        self._viewed_stories: set[tuple[str, str]] = set()
        self._reacted_status: set[tuple[str, str]] = set()
        # Refused-verb journaling guards (truth pass, 2026-11-07):
        # a bridge verdict of not-ok used to vanish — the post
        # simply "stayed unviewed" and no record anywhere said the
        # verb had been tried and refused, which is how a dead
        # status surface read as "audited intact".  One journal
        # row per post per process keeps the refusal visible in
        # the vault without a row per redelivery.
        self._view_failed_status: set[tuple[str, str]] = set()
        self._react_failed_status: set[tuple[str, str]] = set()
        self._story_view_failed: set[tuple[str, str, int]] = set()
        # Status-fix guards (2026-11-07): views WhatsApp
        # downgraded to read-self (receipts off — landed on the
        # wire, invisible to the poster) journal apart from
        # refusals, and EVERY observed post leaves one
        # status_view_trace row: the resolved record, the toggle
        # value read from it, the exact values handed to the
        # viewer seam and the bridge's raw verdict — the live
        # chain's evidence, once per post per process like the
        # rest of the family.
        self._view_ineffective_status: set[tuple[str, str]] = set()
        self._view_traced_status: set[tuple[str, str]] = set()
        #: Mass-tag DM flags: (account, chat_key) → last delivered
        #: flag ts, and the count suppressed since (carried into
        #: the next delivered flag so he sees the volume, once).
        self._mass_tag_last: dict[tuple[str, str], float] = {}
        self._mass_tag_suppressed: dict[tuple[str, str], int] = {}
        #: account_id -> {"asserted": bool, "state": str} — the
        #: presence truth the last cycle_presence computed.
        self._presence: dict[str, dict[str, Any]] = {}
        for row in vault.journal():
            sid = str(row.get("source_id") or "")
            if sid and row.get("kind") == "status":
                self._seen_status.add((str(row.get("account") or ""),
                                       sid))
            if sid and row.get("kind") == "status_missed":
                self._missed_status.add(
                    (str(row.get("account") or ""), sid))
            if sid and row.get("kind") == "story":
                self._seen_stories.add((str(row.get("account") or ""),
                                        sid))
            if sid and row.get("kind") == "channel_post":
                self._seen_channel_posts.add(
                    (str(row.get("account") or ""), sid))
            if sid and row.get("kind") == "channel_post_missed":
                self._missed_channel_posts.add(
                    (str(row.get("account") or ""), sid))

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
                        media_missing: bool = False,
                        incoming: bool = True,
                        account_id: str = "",
                        ts: float = 0.0) -> list[dict[str, Any]]:
        """One DM message the account saw.  Caches it when the
        antidelete toggle is on; preserves its media when the
        view-once toggle is on and the message is view-once /
        self-destruct.  ``media_missing`` is the adapter saying
        the ephemeral media could not be downloaded in time —
        the miss is journaled, exactly like a vanished file.
        Returns the vault entries written."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        media = media or []
        ts = float(ts or self._clock())
        # The cache feeds BOTH recovery halves: delete recovery is
        # antidelete's, edit recovery is antiedit's — either toggle
        # on means originals are worth holding.  (Delete recovery
        # itself still gates on antidelete alone, in observe_delete.)
        if (rec.toggle("antidelete") or rec.toggle("antiedit")) \
                and message_id:
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
        if ephemeral and media_missing and not media \
                and rec.toggle("view_once"):
            # The adapter saw the ephemeral message but its media
            # never made it to disk (expired file reference, a
            # failed download) — the same honest miss as a vanished
            # file, journaled instead of silently dropped.
            _append_jsonl(self.vault.journal_path, {
                "ts": ts, "kind": "view_once_missed",
                "account": rec.id, "platform": platform,
                "counterpart_id": counterpart_id or sender_id,
                "poster_name": sender_name,
                "media_kind": "", "source_id": message_id,
                "path": "", "bytes": 0,
                "note": "media could not be downloaded before it "
                        "expired"})
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
        against the newest, never a ghost.  Gated on antidelete OR
        antiedit: edit recovery historically rode the antidelete
        toggle, and antiedit (wave F) is its own switch over the
        same organ — either one on keeps it alive, both off is
        silence, exactly like the cache gate in observe_message."""
        rec = self.resolve_event_account(platform, account_id)
        if not (rec.toggle("antidelete") or rec.toggle("antiedit")) \
                or not message_id:
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
                            media_missing: bool = False,
                            account_id: str = "",
                            ts: float = 0.0) -> list[dict[str, Any]]:
        """One status post the account can see (WhatsApp
        status@broadcast feed).  Saved when the account's
        status-saving toggle is on; the same post never saves
        twice.  Marked viewed as the owner when the status-viewing
        toggle is on — independently of saving, at most once per
        post per process; every observed post also leaves one
        ``status_view_trace`` row (the resolved record, the
        toggle read from it, the exact values handed to the
        viewer seam, the bridge's raw verdict), and a view the
        platform downgraded to read-self journals
        ``status_view_ineffective`` instead of counting as
        viewed.  Reacted to as the owner when the
        status-react toggle is on (wave F): one reaction per
        post, at most once per post per process, journaled —
        the face drawn from the account's rotation pool (her
        custom set ∪ his additions ∪ the built-in pool), or the
        pinned emoji when rotation is off — and never while an
        owner stop order
        stands for the platform, because a reaction is an ACTION
        on his account, not an observation.  ``media_missing`` is
        the adapter saying the post carried media its download
        failed on: when saving is on, the miss is journaled
        (``status_missed``, once per post) instead of the post
        vanishing from the record entirely."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        ts = float(ts or self._clock())
        # The exact item handed to the viewer seam for this post
        # (kept so the trace below can journal what the chain
        # carried, including the seam's verdict annotation).
        view_item: dict[str, Any] | None = None
        if source_id and rec.toggle("status_viewing") \
                and self.status_viewer is not None:
            vkey = (rec.id, str(source_id))
            if vkey not in self._viewed_status:
                view_item = {
                    "poster_id": poster_id,
                    "poster_name": poster_name,
                    "source_id": str(source_id), "ts": ts}
                try:
                    viewed = bool(self.status_viewer(rec, view_item))
                except Exception:  # noqa: BLE001 - the wired seam
                    viewed = False     # records its own failures
                if viewed:
                    self._viewed_status.add(vkey)
                elif str(view_item.get("receipt_mode") or "") \
                        == "read-self":
                    # The verb LANDED but WhatsApp downgraded
                    # the receipt to read-self — the account's
                    # read receipts are OFF, so the poster is
                    # never notified (status mining, 2026-11-07).
                    # Not a landed view (the post stays unviewed
                    # for dedupe; a redelivery retries) and not
                    # a bridge refusal either: journal the fact,
                    # once per post per process, so a SETTING can
                    # never again read as either "broken" or
                    # "live".  (The mode gates the branch: a
                    # retried read-self must never fall through
                    # to the refusal journal below.)
                    if vkey not in self._view_ineffective_status:
                        self._view_ineffective_status.add(vkey)
                        _append_jsonl(self.vault.journal_path, {
                            "ts": ts,
                            "kind": "status_view_ineffective",
                            "account": rec.id, "platform": platform,
                            "counterpart_id": poster_id,
                            "poster_name": poster_name,
                            "media_kind": "",
                            "source_id": str(source_id),
                            "path": "", "bytes": 0,
                            "note": "status view fired but cannot "
                                    "register — WhatsApp read "
                                    "receipts are OFF on this "
                                    "account, so the receipt went "
                                    "out as read-self and the "
                                    "poster is not notified"})
                elif vkey not in self._view_failed_status:
                    # A refused view is a FACT about the surface,
                    # not a miss: journal it (once per post per
                    # process) so "off" and "broken" can never
                    # read the same again.  The post itself stays
                    # unviewed and a redelivery retries the verb.
                    self._view_failed_status.add(vkey)
                    _append_jsonl(self.vault.journal_path, {
                        "ts": ts, "kind": "status_view_failed",
                        "account": rec.id, "platform": platform,
                        "counterpart_id": poster_id,
                        "poster_name": poster_name,
                        "media_kind": "", "source_id": str(source_id),
                        "path": "", "bytes": 0,
                        "note": "status_view verb did not land — "
                                "the bridge refused it or is "
                                "unreachable; the post stays "
                                "unviewed"})
        if source_id:
            # The live trace (status fix, 2026-11-07): one row
            # per observed post per process, on EVERY outcome —
            # success included — carrying the resolved record,
            # the status_viewing value read from THAT record,
            # the exact poster/source/ts the seam was handed and
            # the bridge's raw verdict once known.  The chain
            # has been read three times and found canonically
            # shaped while dying live; this row is the evidence
            # that names the break point on the real account
            # instead of a fourth reading.
            tkey = (rec.id, str(source_id))
            if tkey not in self._view_traced_status:
                self._view_traced_status.add(tkey)
                _append_jsonl(self.vault.journal_path, {
                    "ts": ts, "kind": "status_view_trace",
                    "account": rec.id, "platform": platform,
                    "counterpart_id": poster_id,
                    "poster_name": poster_name,
                    "media_kind": "", "source_id": str(source_id),
                    "path": "", "bytes": 0,
                    "status_viewing_toggle": bool(
                        rec.toggle("status_viewing")),
                    "seam_poster_id": poster_id,
                    "seam_source_id": str(source_id),
                    "seam_ts": ts,
                    "view_attempted": view_item is not None,
                    "bridge_verdict": (
                        view_item.get("bridge_verdict")
                        if view_item is not None else None),
                    "note": "status view trace — the exact "
                            "values the view chain carried for "
                            "this post"})
        if source_id and rec.toggle("status_react") \
                and self.status_reactor is not None \
                and not self.registry.owner_stopped(platform):
            rkey = (rec.id, str(source_id))
            if rkey not in self._reacted_status:
                # The rotation (expression completion): each
                # status draws the NEXT face from the pool —
                # her custom set ∪ the account's additions ∪
                # the built-in pool — never the same face twice
                # in a row; rotation off means the pinned emoji.
                # The pick is a peek: cursor + last commit onto
                # the record only when the reaction lands.
                from .pairing import status_react_pick

                custom: list[str] = []
                if self.status_react_pool is not None:
                    try:
                        custom = list(self.status_react_pool(rec)
                                      or [])
                    except Exception:  # noqa: BLE001 - pool is garnish
                        custom = []
                emoji, new_cursor = status_react_pick(rec, custom)
                try:
                    reacted = bool(self.status_reactor(rec, {
                        "poster_id": poster_id,
                        "poster_name": poster_name,
                        "source_id": str(source_id),
                        "emoji": emoji, "ts": ts}))
                except Exception:  # noqa: BLE001 - the wired seam
                    reacted = False    # records its own failures
                if not reacted and rkey not in \
                        self._react_failed_status:
                    # Same law as the view: a refused reaction is
                    # journaled once per post per process instead
                    # of vanishing into "stays unreacted".
                    self._react_failed_status.add(rkey)
                    _append_jsonl(self.vault.journal_path, {
                        "ts": ts, "kind": "status_react_failed",
                        "account": rec.id, "platform": platform,
                        "counterpart_id": poster_id,
                        "poster_name": poster_name,
                        "media_kind": "", "source_id": str(source_id),
                        "path": "", "bytes": 0,
                        "note": "status_react verb did not land — "
                                "the bridge refused it or is "
                                "unreachable; no reaction was "
                                "placed"})
                if reacted:
                    self._reacted_status.add(rkey)
                    if rec.status_react_rotate:
                        rec.status_react_cursor = new_cursor
                        rec.status_react_last = emoji
                        # the walk state is garnish — a failed
                        # persist never undoes a landed reaction
                        with contextlib.suppress(Exception):
                            self.registry.upsert(rec)
                    # (the journal below is the reaction's record)
                    _append_jsonl(self.vault.journal_path, {
                        "ts": ts, "kind": "status_react",
                        "account": rec.id, "platform": platform,
                        "counterpart_id": poster_id,
                        "poster_name": poster_name,
                        "media_kind": "", "source_id": str(source_id),
                        "path": "", "bytes": 0,
                        "note": f"reacted {emoji}"})
        if media_missing and rec.toggle("status_saving"):
            mkey = (rec.id, str(source_id or ""))
            if not (source_id and mkey in self._missed_status):
                if source_id:
                    self._missed_status.add(mkey)
                entry = {
                    "ts": ts, "kind": "status_missed",
                    "account": rec.id, "platform": platform,
                    "counterpart_id": poster_id,
                    "poster_name": poster_name, "media_kind": "",
                    "source_id": str(source_id or ""), "path": "",
                    "bytes": 0,
                    "note": "status media could not be downloaded "
                            "before it expired"}
                _append_jsonl(self.vault.journal_path, entry)
                out.append(entry)
        if not rec.toggle("status_saving"):
            return out
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

    def note_status_posted(self, platform: str, *,
                           account_id: str = "", text: str = "",
                           media_kind: str = "",
                           source_id: str = "",
                           ts: float = 0.0) -> dict[str, Any]:
        """Journal one status the OWNER posted through Devon
        (/poststatus): his own acts on his account are part of
        the personal record like everything the surface does —
        an unjournaled act would be a silent one.  Returns the
        journal entry written."""
        rec = self.resolve_event_account(platform, account_id)
        entry = {
            "ts": float(ts or self._clock()),
            "kind": "status_posted", "account": rec.id,
            "platform": platform, "counterpart_id": "",
            "poster_name": "", "media_kind": media_kind,
            "source_id": str(source_id or ""), "path": "",
            "bytes": 0,
            "text": str(text or "")[:200],
            "note": "posted as the owner via /poststatus"}
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    # ── channel posts (channel stories build, 2026-11-07) ──────

    def observe_channel_post(self, platform: str, *, channel_id: str,
                             channel_name: str = "", text: str = "",
                             media: list[dict[str, Any]] | None = None,
                             source_id: str = "",
                             media_missing: bool = False,
                             account_id: str = "",
                             ts: float = 0.0) -> list[dict[str, Any]]:
        """One post from a channel the account follows (WhatsApp
        newsletters — a WhatsApp channel's "stories" ARE its
        posts; there is no separate story object on the
        platform).  The status family's posture, mirrored
        exactly: saved when the account's status-saving toggle
        is on (the same post never saves twice, across restarts
        too — the seen set is journal-seeded), marked viewed as
        the owner when the status-viewing toggle is on (at most
        once per post per process), and a media post whose bytes
        could not be fetched is journaled
        (``channel_post_missed``, once) instead of vanishing.
        A refused view verb journals ``channel_view_failed``
        once per post per process, with the seam's own failure
        record carrying the bridge's reason — "off" and
        "broken" never read the same (truth-pass law)."""
        rec = self.resolve_event_account(platform, account_id)
        out: list[dict[str, Any]] = []
        ts = float(ts or self._clock())
        if source_id and rec.toggle("status_viewing") \
                and self.channel_viewer is not None:
            vkey = (rec.id, str(source_id))
            if vkey not in self._viewed_channel_posts:
                try:
                    viewed = bool(self.channel_viewer(rec, {
                        "poster_id": channel_id,
                        "poster_name": channel_name,
                        "source_id": str(source_id), "ts": ts}))
                except Exception:  # noqa: BLE001 - the wired seam
                    viewed = False     # records its own failures
                if viewed:
                    self._viewed_channel_posts.add(vkey)
                elif vkey not in self._view_failed_channel_posts:
                    self._view_failed_channel_posts.add(vkey)
                    _append_jsonl(self.vault.journal_path, {
                        "ts": ts, "kind": "channel_view_failed",
                        "account": rec.id, "platform": platform,
                        "counterpart_id": channel_id,
                        "poster_name": channel_name,
                        "media_kind": "", "source_id": str(source_id),
                        "path": "", "bytes": 0,
                        "note": "channel_view verb did not land — "
                                "the bridge refused it or is "
                                "unreachable; the post stays "
                                "unviewed"})
        if media_missing and rec.toggle("status_saving"):
            mkey = (rec.id, str(source_id or ""))
            if not (source_id and mkey in self._missed_channel_posts):
                if source_id:
                    self._missed_channel_posts.add(mkey)
                entry = {
                    "ts": ts, "kind": "channel_post_missed",
                    "account": rec.id, "platform": platform,
                    "counterpart_id": channel_id,
                    "poster_name": channel_name, "media_kind": "",
                    "source_id": str(source_id or ""), "path": "",
                    "bytes": 0,
                    "note": "channel post media could not be "
                            "downloaded"}
                _append_jsonl(self.vault.journal_path, entry)
                out.append(entry)
        if not rec.toggle("status_saving"):
            return out
        dedupe_key = (rec.id, str(source_id or ""))
        if source_id and dedupe_key in self._seen_channel_posts:
            return out
        media = media or []
        if media:
            for m in media:
                path = str(m.get("path") or "")
                if not path:
                    continue
                try:
                    out.append(self.vault.save_media(
                        rec.id, "channel_post", path,
                        platform=platform, counterpart_id=channel_id,
                        poster_name=channel_name,
                        media_kind=str(m.get("kind") or ""),
                        source_id=source_id, ts=ts))
                except FileNotFoundError:
                    continue
        elif text:
            out.append(self.vault.save_text(
                rec.id, "channel_post", text, platform=platform,
                counterpart_id=channel_id,
                poster_name=channel_name, source_id=source_id,
                ts=ts))
        if source_id and out:
            self._seen_channel_posts.add(dedupe_key)
        return out

    def cycle_channel_posts(self, platform: str = "whatsapp",
                            account_id: str = ""
                            ) -> list[dict[str, Any]]:
        """One channel sweep: ask the fetcher for the followed
        channels' recent posts and observe each (save + view
        per the account's toggles — see observe_channel_post).
        The fetcher runs only when at least one of the two
        toggles could act; no fetcher wired (the surface has no
        channel feed) → nothing happens and the caller can say
        exactly that."""
        if self.channel_fetcher is None:
            return []
        rec = self.resolve_event_account(platform, account_id)
        if not (rec.toggle("status_saving")
                or (rec.toggle("status_viewing")
                    and self.channel_viewer is not None)):
            return []
        out: list[dict[str, Any]] = []
        try:
            posts = self.channel_fetcher(rec) or []
        except Exception:  # noqa: BLE001 - the wired seam
            return []                  # records its own failures
        for post in posts:
            media: list[dict[str, Any]] = []
            path = str(post.get("path") or "")
            if path:
                media.append({"path": path,
                              "kind": str(post.get("kind") or "")})
            out.extend(self.observe_channel_post(
                platform,
                channel_id=str(post.get("poster_id") or ""),
                channel_name=str(post.get("poster_name") or ""),
                text=str(post.get("text") or ""),
                media=media,
                source_id=str(post.get("story_id") or ""),
                media_missing=bool(post.get("media_missing")),
                account_id=rec.id,
                ts=float(post.get("ts") or 0.0)))
        return out

    def note_channel_posted(self, platform: str, *,
                            account_id: str = "", channel_id: str = "",
                            channel_name: str = "", text: str = "",
                            media_kind: str = "",
                            source_id: str = "",
                            ts: float = 0.0) -> dict[str, Any]:
        """Journal one channel post the OWNER published through
        Devon (/chanstory): his own acts on his account are
        part of the personal record like everything the surface
        does — an unjournaled act would be a silent one.
        Returns the journal entry written."""
        rec = self.resolve_event_account(platform, account_id)
        entry = {
            "ts": float(ts or self._clock()),
            "kind": "channel_posted", "account": rec.id,
            "platform": platform, "counterpart_id": channel_id,
            "poster_name": channel_name, "media_kind": media_kind,
            "source_id": str(source_id or ""), "path": "",
            "bytes": 0, "text": str(text or "")[:200],
            "note": "posted as the owner via /chanstory"}
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    def note_profile_changed(self, platform: str, *,
                             account_id: str = "", field: str = "",
                             value: str = "",
                             ts: float = 0.0) -> dict[str, Any]:
        """Journal one own-identity change on a paired account
        (parity G3): the profile name, about text or photo, or
        one privacy category — the acts /myprofile and /privacy
        perform AS the owner.  The change itself is the record:
        kind ``profile_changed``, ``field`` naming what moved
        (``name`` / ``about`` / ``photo`` / ``privacy:<cat>``)
        and the text carrying the new value (a photo change
        carries its picture id, or "removed")."""
        rec = self.resolve_event_account(platform, account_id)
        entry = {
            "ts": float(ts or self._clock()),
            "kind": "profile_changed", "account": rec.id,
            "platform": platform, "counterpart_id": rec.id,
            "poster_name": "", "media_kind": "",
            "source_id": "", "path": "", "bytes": 0,
            "text": str(value or "")[:200],
            "field": str(field or ""),
            "note": "identity change as the owner via /myprofile"}
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    def note_poll_posted(self, platform: str, *,
                         account_id: str = "",
                         counterpart_id: str = "",
                         source_id: str = "", question: str = "",
                         options: Any = (),
                         ts: float = 0.0) -> dict[str, Any]:
        """Journal one native poll posted AS the owner (parity
        G3, /makepoll): kind ``poll_posted``, the chat in
        counterpart_id, the poll's own message id in source_id
        (the handle /pollres reads back), the question as the
        text and the option list beside it."""
        rec = self.resolve_event_account(platform, account_id)
        entry = {
            "ts": float(ts or self._clock()),
            "kind": "poll_posted", "account": rec.id,
            "platform": platform,
            "counterpart_id": str(counterpart_id or ""),
            "poster_name": "", "media_kind": "",
            "source_id": str(source_id or ""), "path": "",
            "bytes": 0, "text": str(question or "")[:200],
            "options": [str(o) for o in (options or ())],
            "note": "poll posted as the owner via /makepoll"}
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    def note_message_acted(self, platform: str, *,
                           kind: str,
                           account_id: str = "",
                           chat_id: str = "",
                           source_id: str = "",
                           dest_id: str = "",
                           note: str = "",
                           ts: float = 0.0) -> dict[str, Any]:
        """Journal one message-level act the OWNER took through
        Devon on a personal surface (/react, /forward — parity
        G1): his own acts on his account are part of the
        personal record like everything the surface does — an
        unjournaled act would be a silent one.  Returns the
        journal entry written."""
        rec = self.resolve_event_account(platform, account_id)
        entry = {
            "ts": float(ts or self._clock()),
            "kind": str(kind or ""), "account": rec.id,
            "platform": platform, "counterpart_id": chat_id,
            "poster_name": "", "media_kind": "",
            "source_id": str(source_id or ""), "path": "",
            "bytes": 0, "text": "",
            "note": str(note or "")[:200]}
        if dest_id:
            entry["dest_id"] = str(dest_id)
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    def cycle_stories(self, platform: str = "telegram",
                      account_id: str = "") -> list[dict[str, Any]]:
        """One story sweep: ask the fetcher for the account's
        currently-visible stories, save the new ones (saving
        toggle), and mark the unseen ones viewed as the owner
        (viewing toggle — ReadStories per peer at its newest
        story id).  The two toggles are independent.  No fetcher
        (adapter exposes none) → nothing happens and the caller
        can say exactly that."""
        if self.story_fetcher is None:
            return []
        rec = self.resolve_event_account(platform, account_id)
        saving = rec.toggle("status_saving")
        viewing = rec.toggle("status_viewing") \
            and self.story_viewer is not None
        if not saving and not viewing:
            return []
        out: list[dict[str, Any]] = []
        stories = self.story_fetcher(rec) or []
        if saving:
            for story in stories:
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
        if viewing:
            # ReadStories marks everything up to a peer's newest
            # story id as read — one call per peer per sweep, for
            # peers with stories this process has not viewed yet.
            newest: dict[str, tuple[int, list[str]]] = {}
            for story in stories:
                sid = str(story.get("story_id") or "")
                poster = str(story.get("poster_id") or "")
                if not sid or not poster:
                    continue
                if (rec.id, sid) in self._viewed_stories:
                    continue
                try:
                    sid_int = int(sid)
                except ValueError:
                    continue
                cur = newest.get(poster)
                if cur is None or sid_int > cur[0]:
                    newest[poster] = (sid_int,
                                      (cur[1] if cur else []) + [sid])
                elif cur is not None:
                    cur[1].append(sid)
            for poster, (max_id, sids) in newest.items():
                try:
                    viewed = bool(self.story_viewer(rec, poster,
                                                    max_id))
                except Exception:  # noqa: BLE001 - the wired seam
                    viewed = False     # records its own failures
                if viewed:
                    for sid in sids:
                        self._viewed_stories.add((rec.id, sid))
                else:
                    # A refused story view journals like a refused
                    # status view (truth pass): once per peer per
                    # newest-id per process — visible, not a silent
                    # "stays unviewed" on every sweep.
                    fkey = (rec.id, poster, max_id)
                    if fkey not in self._story_view_failed:
                        self._story_view_failed.add(fkey)
                        _append_jsonl(self.vault.journal_path, {
                            "ts": float(self._clock()),
                            "kind": "story_view_failed",
                            "account": rec.id, "platform": platform,
                            "counterpart_id": poster,
                            "poster_name": "",
                            "media_kind": "",
                            "source_id": str(max_id),
                            "path": "", "bytes": 0,
                            "note": "story view verb did not land "
                                    "— the surface refused it or "
                                    "is unreachable; the stories "
                                    "stay unviewed"})
        return out

    # ── calls & mass tags (chat & privacy wave) ─────────────────
    def observe_call(self, platform: str, *, from_id: str = "",
                     from_name: str = "", call_id: str = "",
                     account_id: str = "",
                     ts: float = 0.0) -> dict[str, Any] | None:
        """One incoming call offer on the account (WhatsApp — the
        only surface whose bridges forward call events).  When
        the anticall toggle is on the call is rejected as the
        owner through the wired seam and the outcome is journaled
        either way — a rejection is an ACTION on his account, so
        an owner stop order standing for the platform suppresses
        it completely (no seam call, no journal: the surface is
        stopped, not performing).  Returns the journal entry when
        a rejection was attempted, else None."""
        rec = self.resolve_event_account(platform, account_id)
        if not rec.toggle("anticall"):
            return None
        if self.registry.owner_stopped(platform):
            return None
        ts = float(ts or self._clock())
        rejected = False
        if self.call_rejector is not None and call_id:
            try:
                rejected = bool(self.call_rejector(rec, {
                    "from_id": from_id, "from_name": from_name,
                    "call_id": str(call_id), "ts": ts}))
            except Exception:  # noqa: BLE001 - the wired seam
                rejected = False       # records its own failures
        entry = {
            "ts": ts, "kind": "call_offer", "account": rec.id,
            "platform": platform, "counterpart_id": from_id,
            "poster_name": from_name, "media_kind": "",
            "source_id": str(call_id), "path": "", "bytes": 0,
            "note": ("rejected (anticall on)" if rejected else
                     "anticall on, but the rejection did not land "
                     "— no reject verb wired or it refused"),
        }
        _append_jsonl(self.vault.journal_path, entry)
        return entry

    def observe_mass_tag(self, platform: str, *, chat_key: str,
                         sender_id: str = "", sender_name: str = "",
                         mention_count: int = 0,
                         account_id: str = "",
                         ts: float = 0.0) -> dict[str, Any] | None:
        """One group message that mass-tagged the owner (the
        caller counted the mention payload and confirmed his own
        mention is in it).  When the antitag toggle is on, the
        detection is journaled and — at most once per chat per
        cooldown — an entry marked ``deliver`` comes back for the
        runtime to DM him; repeats inside the cooldown only bump
        the suppressed count the next delivered flag carries.
        A standing owner stop suppresses delivery (the flag is a
        send as him-adjacent surface traffic) but the journal
        still records the detection."""
        rec = self.resolve_event_account(platform, account_id)
        if not rec.toggle("antitag"):
            return None
        ts = float(ts or self._clock())
        key = (rec.id, str(chat_key))
        last = self._mass_tag_last.get(key)
        stopped = self.registry.owner_stopped(platform)
        deliver = not stopped and (
            last is None
            or (ts - float(last)) >= MASS_TAG_FLAG_COOLDOWN)
        suppressed = int(self._mass_tag_suppressed.get(key) or 0)
        entry = {
            "ts": ts, "kind": "mass_tag", "account": rec.id,
            "platform": platform, "chat_key": str(chat_key),
            "counterpart_id": sender_id,
            "poster_name": sender_name,
            "mention_count": int(mention_count),
            "deliver": bool(deliver),
            "suppressed_before": (0 if deliver else suppressed),
            "note": ("owner mass-tagged in a group message"
                     + (" — flag suppressed by owner stop"
                        if stopped and not deliver else "")),
        }
        _append_jsonl(self.vault.journal_path, entry)
        if deliver:
            self._mass_tag_last[key] = ts
            self._mass_tag_suppressed[key] = 0
            entry["suppressed_before"] = suppressed
            return entry
        self._mass_tag_suppressed[key] = suppressed + 1
        return None

    def set_status_react_emoji(self, account_id: str,
                               emoji: str) -> AccountRecord:
        """Set one account's status-react emoji (the /statusreact
        emoji surface).  Unknown account / empty emoji → ValueError
        with the plain reason; the caller turns it into its line."""
        emoji = str(emoji or "").strip()
        if not emoji:
            raise ValueError("the emoji is empty")
        rec = self.registry.get(str(account_id or ""))
        if rec is None:
            raise ValueError(f"no paired account {account_id!r}")
        rec.status_react_emoji = emoji
        return self.registry.upsert(rec)

    def set_status_react_rotation(self, account_id: str, *,
                                  rotate: bool | None = None,
                                  add: Any = (),
                                  remove: Any = ()
                                  ) -> AccountRecord:
        """The rotation controls behind /statusreact: turn
        rotation on/off, and manage the account's own pool
        additions (her custom set and the built-in pool are
        always members — this list is only the owner's extras).
        Same contract as set_status_react_emoji: unknown
        account → ValueError with the plain reason."""
        rec = self.registry.get(str(account_id or ""))
        if rec is None:
            raise ValueError(f"no paired account {account_id!r}")
        pool = [str(e).strip() for e in (rec.status_react_pool
                                         or []) if str(e).strip()]
        for emoji in add or ():
            emoji = str(emoji or "").strip()
            if emoji and emoji not in pool:
                pool.append(emoji)
        for emoji in remove or ():
            emoji = str(emoji or "").strip()
            pool = [e for e in pool if e != emoji]
        rec.status_react_pool = pool
        if rotate is not None:
            rec.status_react_rotate = bool(rotate)
        return self.registry.upsert(rec)

    # ── presence (always online) ────────────────────────────────
    def presence_state(self, account_id: str) -> str:
        """The presence truth for one account as of the last
        cycle ("" when no cycle has considered it)."""
        row = self._presence.get(account_id) or {}
        return str(row.get("state") or "")

    def _journal_presence(self, rec: AccountRecord, online: bool,
                          reason: str) -> None:
        _append_jsonl(self.vault.journal_path, {
            "ts": self._clock(), "kind": "presence",
            "account": rec.id, "platform": rec.platform,
            "counterpart_id": "", "poster_name": "",
            "media_kind": "", "source_id": "", "path": "",
            "bytes": 0,
            "note": ("online asserted" if online
                     else f"online released — {reason}")})

    def cycle_presence(self) -> dict[str, str]:
        """One presence beat: for every registered account, hold
        its presence online iff the owner asked for exactly that
        (always_online on, account linked, NO owner stop order
        standing for the platform, and the wired verb lands).
        Asserting is a keepalive — the verb is re-sent every beat
        while wanted — but journal episodes record TRANSITIONS
        only, once each.  A stopped or down surface asserts
        nothing and resurrects nothing.  Returns account_id →
        plain state text (also kept for presence_state())."""
        states: dict[str, str] = {}
        for rec in self.registry.all():
            cur = self._presence.get(rec.id) or {
                "asserted": False, "state": "off"}
            asserted = bool(cur.get("asserted"))
            toggle_on = rec.toggle("always_online")
            stopped = self.registry.owner_stopped(rec.platform)
            linked = rec.status == "linked"
            if not toggle_on:
                reason, want = "toggle off", False
            elif stopped:
                reason, want = "stopped by the owner", False
            elif not linked:
                reason, want = f"account {rec.status}", False
            else:
                reason, want = "", True
            if self.presence_setter is None:
                state = ("on — but no presence verb is wired for "
                         "this surface" if toggle_on else "off")
                self._presence[rec.id] = {
                    "asserted": False, "state": state}
                states[rec.id] = state
                continue
            if want:
                try:
                    landed = bool(self.presence_setter(
                        rec.platform, rec, True))
                except Exception:  # noqa: BLE001 - the wired seam
                    landed = False     # records its own failures
                if landed:
                    if not asserted:
                        self._journal_presence(rec, True, "")
                    self._presence[rec.id] = {
                        "asserted": True, "state": "online"}
                    states[rec.id] = "online"
                else:
                    self._presence[rec.id] = {
                        "asserted": False,
                        "state": "wanted, but the surface is not "
                                 "live to assert it"}
                    states[rec.id] = self._presence[rec.id]["state"]
                continue
            if asserted:
                try:
                    self.presence_setter(rec.platform, rec, False)
                except Exception:  # noqa: BLE001 - releasing is
                    pass               # best-effort by contract
                self._journal_presence(rec, False, reason)
            self._presence[rec.id] = {
                "asserted": False, "state": f"off — {reason}"}
            states[rec.id] = self._presence[rec.id]["state"]
        return states

    # ── toggles ─────────────────────────────────────────────────────
    def set_toggle(self, account_id: str, name: str,
                   on: bool) -> AccountRecord:
        if name not in PERSONAL_TOGGLES:
            raise ValueError(
                f"unknown personal toggle {name!r} — known: "
                f"{', '.join(PERSONAL_TOGGLES)}")
        return self.registry.set_toggles(account_id, **{name: on})
