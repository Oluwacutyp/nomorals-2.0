"""Per-chat context profiles: mined, incremental, never hardcoded.

Every chat gets a lightweight profile — who talks here, what it's about,
how it feels — built from the actual message history and refreshed as new
messages arrive. The brain receives it as prompt context so replies are
grounded in the chat's reality instead of guessing from the last few lines.

Design notes:
- Topics are mined with TF scoring over the chat's own recent window.
  No topic lists, no keyword dictionaries, no intent patterns.
- Updates are incremental and off-thread (mirrors ``_maybe_extract``):
  participant counts bump every turn; the expensive topic pass runs at
  most once per N new messages.
- Everything is best-effort: a missing profile degrades to no extra
  context, never to a broken reply.
"""

from __future__ import annotations

import json
import math
import re
import time
from contextlib import suppress
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# Tiny closed-class stoplist for topic mining. This is linguistic
# plumbing (the / it / and class), not a topic dictionary — the topics
# themselves always come from the chat's own words.
_STOPWORDS = frozenset(
    "i me my we our you your he him his she her it its they them their "
    "this that these those am is are was were be been being have has had "
    "do does did will would could should may might must can shall a an "
    "the and but or nor for so yet as at by in of on to up out off over "
    "under with without about into through during before after above "
    "below between no not only own same than too very just don now here "
    "there when where which who whom what why how all any both each few "
    "more most other some such lol lmao omg btw tbh idk gonna wanna "
    "yeah yes no ok okay hey hi hello thanks thank please".split()
)

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'\-]{2,}")

_PROFILE_UPDATE_EVERY = 10  # recompute topics at most every N new messages
_TOPIC_WINDOW = 120  # messages back to mine topics from
_MAX_TOPICS = 8
_MAX_PARTICIPANTS = 12


def _ensure_table(db: Any) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS chat_profiles (
            chat_key TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL DEFAULT '{}',
            message_count INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT 0
        )"""
    )


def _mine_topics(texts: list[str]) -> list[str]:
    """TF-ranked topic words from the chat's own recent messages."""
    tf: dict[str, int] = {}
    doc_count: dict[str, int] = {}
    n_docs = 0
    for text in texts:
        n_docs += 1
        seen: set[str] = set()
        for m in _WORD.finditer((text or "").lower()):
            w = m.group(0).strip("'-")
            if len(w) < 3 or w in _STOPWORDS:
                continue
            tf[w] = tf.get(w, 0) + 1
            if w not in seen:
                seen.add(w)
                doc_count[w] = doc_count.get(w, 0) + 1
    scored: list[tuple[float, str]] = []
    for w, count in tf.items():
        if count < 2:
            continue  # said once = noise, not a topic
        # TF-IDF-lite: frequent overall, but spread across messages =
        # a real thread of conversation, not one person's tic.
        idf = math.log(1 + n_docs / max(1, doc_count.get(w, 1)))
        spread = doc_count.get(w, 1) / max(1, n_docs)
        scored.append((count * (0.5 + idf) * (0.5 + spread), w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:_MAX_TOPICS]]


# ── refresh-failure record (AA-68) ──────────────────────────────
# A failed refresh used to be invisible: refresh_profile returned
# the stored profile as though it were current, and the brain built
# prompt context from it as fresh.  Failures are now recorded
# against the db handle (attribute first, module map as fallback)
# so both the refresh return value AND get_profile can mark the
# stored copy as what it is: stale, with its age and the error.
_FAILURE_ATTR = "_chat_profile_refresh_failures"
_FALLBACK_FAILURES: dict[int, dict[str, dict[str, Any]]] = {}

#: Marker keys a stale profile carries; never persisted — a
#: successful refresh strips them from the rebuilt profile.
STALE_KEYS = ("stale", "refresh_error", "profile_age_s", "updated_at")


def _failure_store(db: Any) -> dict[str, dict[str, Any]]:
    try:
        store = getattr(db, _FAILURE_ATTR, None)
        if store is None:
            store = {}
            setattr(db, _FAILURE_ATTR, store)
        return store
    except Exception:  # noqa: BLE001 - exotic db handles
        return _FALLBACK_FAILURES.setdefault(id(db), {})


def _note_refresh_failure(db: Any, chat_key: str, exc: Exception) -> None:
    # best-effort: the log + failure ledger are the durable trail
    with suppress(Exception):
        _failure_store(db)[chat_key] = {
            "failed_at": time.time(), "error": str(exc)}


def _clear_refresh_failure(db: Any, chat_key: str) -> None:
    with suppress(Exception):
        _failure_store(db).pop(chat_key, None)


def _stale_markers(db: Any, chat_key: str,
                   updated_at: float) -> dict[str, Any] | None:
    """The stale markers for a stored profile, when a refresh has
    failed since that copy was written (AA-68); else None."""
    try:
        failure = _failure_store(db).get(chat_key)
    except Exception:  # noqa: BLE001
        failure = None
    if not failure:
        return None
    if updated_at and failure["failed_at"] < updated_at:
        return None  # a later refresh already succeeded past it
    markers: dict[str, Any] = {"stale": True,
                               "refresh_error": failure["error"]}
    if updated_at:
        markers["updated_at"] = updated_at
        markers["profile_age_s"] = round(time.time() - updated_at, 1)
    return markers


def get_profile(db: Any, chat_key: str) -> dict[str, Any]:
    """Load the profile; returns {} when none exists yet.

    When the last refresh for this chat failed after the stored
    copy was written, the copy is returned WITH stale markers
    (AA-68) — consumers can tell a fallback from a fresh profile.
    """
    try:
        _ensure_table(db)
        row = db.query_one(
            "SELECT profile_json, updated_at FROM chat_profiles "
            "WHERE chat_key = ?",
            (chat_key,),
        )
        if row and row.get("profile_json"):
            profile = json.loads(row["profile_json"])
            if profile:
                markers = _stale_markers(
                    db, chat_key, float(row.get("updated_at") or 0.0))
                if markers:
                    profile.update(markers)
            return profile
    except Exception:  # noqa: BLE001 - no profile beats a broken chat
        pass
    return {}


def _vibe_label(valence: float) -> str:
    if valence >= 0.35:
        return "warm"
    if valence >= 0.12:
        return "easy"
    if valence <= -0.35:
        return "tense"
    if valence <= -0.12:
        return "heavy"
    return "neutral"


def _activity_label(msgs_per_day: float) -> str:
    if msgs_per_day < 2:
        return "quiet lately"
    if msgs_per_day < 12:
        return "steady"
    return "buzzing"


def build_context_lines(profile: dict[str, Any]) -> list[str]:
    """Render the profile as prompt context lines for the brain."""
    if not profile:
        return []
    lines: list[str] = []
    parts: list[str] = []
    participants = profile.get("participants") or {}
    if participants:
        top = sorted(participants.items(), key=lambda kv: kv[1],
                     reverse=True)[:5]
        names = ", ".join(f"{n} ({c})" for n, c in top)
        parts.append(f"people here: {names}")
    topics = profile.get("topics") or []
    if topics:
        parts.append(f"recent threads: {', '.join(topics)}")
    purpose = profile.get("purpose") or ""
    if purpose:
        parts.append(purpose)
    vibe = profile.get("vibe_label") or ""
    if vibe:
        parts.append(f"the vibe here lately: {vibe}")
    activity = profile.get("activity_label") or ""
    if activity:
        parts.append(f"pace: {activity}")
    owner_note = profile.get("owner_note") or ""
    if owner_note:
        parts.append(owner_note)
    if parts:
        lines.append("chat context — " + " · ".join(parts))
    if profile.get("stale"):
        # AA-68: honest degradation — the brain is told the context
        # it just read is a fallback copy, never presented as fresh.
        age = profile.get("profile_age_s")
        age_txt = (f" (last good copy {int(age)}s old)"
                   if isinstance(age, (int, float)) and age >= 0
                   else "")
        lines.append(
            "chat profile note: this chat's saved profile is STALE"
            f"{age_txt} — its last refresh failed, so the context "
            "above may be out of date.")
    return lines


def _chat_purpose(chat_key: str, participants: dict[str, int],
                  total: int) -> str:
    kind = "group" if (chat_key.count(":") >= 2) else "chat"
    if len(participants) <= 2:
        return f"a {kind} mostly between you and one other person"
    return f"a {kind} with {len(participants)} voices ({total} messages seen)"


def refresh_profile(db: Any, chat_key: str, *, force: bool = False) -> dict[str, Any]:
    """Incrementally refresh the profile from message history.

    Participant counts update every call; the topic pass runs when at
    least ``_PROFILE_UPDATE_EVERY`` new messages arrived since the last
    one (or on ``force``).
    """
    try:
        _ensure_table(db)
        row = db.query_one(
            "SELECT profile_json, message_count FROM chat_profiles WHERE chat_key = ?",
            (chat_key,),
        )
        old = json.loads(row["profile_json"]) if row and row.get("profile_json") else {}
        old_count = int(row["message_count"]) if row else 0

        # Participant counts from the whole history (cheap aggregate).
        participants: dict[str, int] = {}
        total = 0
        for r in db.query(
            "SELECT name, COUNT(*) AS n FROM messages "
            "WHERE conversation_id = ? AND role = 'user' "
            "GROUP BY name ORDER BY n DESC LIMIT ?",
            (chat_key, _MAX_PARTICIPANTS),
        ):
            name = (r.get("name") or "").strip() or "someone"
            participants[name] = int(r.get("n") or 0)
            total += int(r.get("n") or 0)

        new_messages = total - old_count
        profile = dict(old)
        # AA-68: a rebuilt profile is fresh by construction — stale
        # markers (if a marked copy was ever persisted) never ride
        # into the new stored profile.
        for _key in STALE_KEYS:
            profile.pop(_key, None)
        profile["participants"] = participants
        profile["purpose"] = _chat_purpose(chat_key, participants, total)
        if "first_seen" not in profile:
            profile["first_seen"] = time.time()

        # Owner presence note: the owner's own messages are stored under
        # the __owner__ marker (see brain._persist_inbound).
        owner_msgs = participants.get("__owner__", 0)
        if owner_msgs:
            profile["owner_note"] = "the owner is active in this chat"
        elif "owner_note" in profile:
            del profile["owner_note"]

        if force or new_messages >= _PROFILE_UPDATE_EVERY or not old.get("topics"):
            texts: list[str] = []
            try:
                for r in db.query(
                    "SELECT content FROM messages "
                    "WHERE conversation_id = ? AND role = 'user' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (chat_key, _TOPIC_WINDOW),
                ):
                    c = (r.get("content") or "").strip()
                    if c and len(c) < 2000:
                        texts.append(c)
            except Exception:  # noqa: BLE001
                pass
            if texts:
                profile["topics"] = _mine_topics(texts)
                # Vibe: mean valence of the recent window, via the affect
                # scorer. Best-effort — a failed import leaves the old vibe.
                try:
                    from .affect import AffectScorer
                    scorer = AffectScorer()
                    valences = [scorer.score(t).valence for t in texts[:40]]
                    if valences:
                        mean_v = sum(valences) / len(valences)
                        profile["vibe"] = round(mean_v, 3)
                        profile["vibe_label"] = _vibe_label(mean_v)
                except Exception:  # noqa: BLE001 - vibe is garnish, not load-bearing
                    pass

        # Activity rhythm: messages per day since first seen.
        try:
            first_seen = float(profile.get("first_seen") or time.time())
            days = max(1.0 / 24.0, (time.time() - first_seen) / 86400.0)
            mpd = total / days
            profile["msgs_per_day"] = round(mpd, 2)
            profile["activity_label"] = _activity_label(mpd)
        except Exception:  # noqa: BLE001
            pass

        profile["message_count"] = total
        db.execute(
            "INSERT INTO chat_profiles (chat_key, profile_json, message_count, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chat_key) DO UPDATE SET profile_json = excluded.profile_json, "
            "message_count = excluded.message_count, updated_at = excluded.updated_at",
            (chat_key, json.dumps(profile), total, time.time()),
        )
        _clear_refresh_failure(db, chat_key)
        return profile
    except Exception as exc:  # noqa: BLE001
        # AA-68: the fallback is still the stored profile (a missing
        # profile degrades to no context, never a broken reply) —
        # but the failure is now LOGGED, recorded in the failure
        # ledger, and the returned copy carries stale markers so no
        # consumer mistakes it for a fresh refresh.
        _log.warning("chat profile refresh failed for %s: %s",
                     chat_key, exc)
        try:
            from ..storage.failures import record_failure
            record_failure("partner.chat_profile.refresh", exc)
        except Exception:  # noqa: BLE001 - ledger must not break it
            pass
        _note_refresh_failure(db, chat_key, exc)
        return get_profile(db, chat_key)
