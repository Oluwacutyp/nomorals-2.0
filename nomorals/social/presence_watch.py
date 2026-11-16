"""Presence watch — alert the owner when a named contact posts a status,
comes online, or joins a channel (census §6 superhuman #4).

Pure data organ (L4, stdlib only): a durable watch list plus the
matching and episode-dedupe rules.  It knows nothing about the
runtime, the gateway, or the notifier — the runtime's hooks call
``check``/``note_offline`` from the three places the signals already
arrive, and deliver whatever alert comes back through the notifier.

Signal matrix — what each platform can actually observe today:
* WhatsApp status — live: the status_post service branch calls
  observe_status_post once per post.
* Telegram status — the story sweep: the housekeeping beat drives
  cycle_stories("telegram"), which returns the newly saved stories;
  a story watch fires when status saving is on (the toggle that
  preserves stories at all).
* Telegram online — live: Telethon UserUpdate events carry the
  transition; the adapter forwards it in the service message meta.
* WhatsApp online — NOT observable: neither bridge forwards contact
  presence (both only assert the account's own presence, §15
  always-online).  A WhatsApp online watch is refused at add time
  with the wall named, never silently stored.
* Join (both) — live: the service == "join" handler records every
  admitted arrival; the watch fires only for admitted arrivals,
  mirroring the reputation hook's gate.

One alert per episode, always: a status post replays under its own
source id, an online transition under the last-seen state, a join
under its group + second.  The store is one durable JSON file under
the home's data/ directory, written atomically; a corrupt file is
quarantined and the store starts empty rather than taking intake
down with it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

SIGNALS = ("status", "online", "join")

#: What each platform can actually observe today (see module docstring).
PLATFORM_SIGNALS: dict[str, tuple[str, ...]] = {
    "whatsapp": ("status", "join"),
    "telegram": ("status", "online", "join"),
}

PLATFORM_LABELS = {"whatsapp": "WhatsApp", "telegram": "Telegram"}

#: The WhatsApp online wall, in the owner's words where it matters:
#: no inbound presence event reaches Devon at all.
WA_ONLINE_WALL = (
    "WhatsApp doesn't forward contact presence — both bridge engines "
    "only assert your own presence (§15 always-online), so a WhatsApp "
    "online watch can never fire. A bridge event forward would be "
    "needed first."
)

STORE_VERSION = 1

_DIGITS = re.compile(r"\D+")
_SIGNAL_WORDS = {"status", "online", "join"}


def platform_label(platform: str) -> str:
    """Display name for a platform key; the key itself when unknown."""
    return PLATFORM_LABELS.get(str(platform or "").lower(),
                               str(platform or ""))


def supports(platform: str, signal: str) -> bool:
    """Can this signal ever fire on this platform?"""
    return str(signal or "").lower() in PLATFORM_SIGNALS.get(
        str(platform or "").lower(), ())


def unsupported_reason(platform: str, signal: str) -> str:
    """Why this signal can never fire here ("" when it can)."""
    platform = str(platform or "").lower()
    signal = str(signal or "").lower()
    if supports(platform, signal):
        return ""
    if platform == "whatsapp" and signal == "online":
        return WA_ONLINE_WALL
    if platform not in PLATFORM_SIGNALS:
        return (f"unknown platform {platform!r} — whatsapp or telegram")
    return (f"{signal} is not observable on "
            f"{platform_label(platform)}")


def _digits(value: str) -> str:
    return _DIGITS.sub("", str(value or ""))


def _norm_name(value: str) -> str:
    return str(value or "").strip().casefold()


def _contact_key(platform: str, contact: str) -> str:
    """Dedupe key for a watch: platform + normalized contact."""
    contact = str(contact or "").strip()
    digits = _digits(contact)
    if len(digits) >= 7:
        return f"{platform}|{digits}"
    return f"{platform}|{_norm_name(contact).lstrip('@')}"


def _matches(contact: str, user_id: str, username: str,
             name: str) -> bool:
    """Does an observed identity match a watch's contact string?

    Digits match a JID/LID/user-id by digit tail (a phone number
    entered bare matches ``<digits>@s.whatsapp.net`` and
    ``<digits>@lid``); otherwise the username (leading @ stripped,
    case-insensitive) or the display name (case-insensitive exact)
    must match.  A name never matches by substring — a watch for
    "Ada" must not fire on "Adaeze".
    """
    contact = str(contact or "").strip()
    if not contact:
        return False
    digits = _digits(contact)
    if len(digits) >= 7:
        observed = _digits(user_id)
        return bool(observed) and (
            observed == digits
            or observed.endswith(digits)
            or digits.endswith(observed))
    if _norm_name(username).lstrip("@") == _norm_name(contact).lstrip("@"):
        return True
    return _norm_name(name) == _norm_name(contact)


def render_alert(alert: dict[str, Any]) -> tuple[str, str]:
    """(title, body) for one alert — the subject is always named."""
    label = str(alert.get("label") or alert.get("contact") or "someone")
    platform = platform_label(str(alert.get("platform") or ""))
    signal = str(alert.get("signal") or "")
    if signal == "status":
        return (f"{label} posted a status",
                f"{label} posted a status on {platform}.")
    if signal == "online":
        return (f"{label} is online",
                f"{label} came online on {platform}.")
    group = str(alert.get("group") or "")
    if group:
        return (f"{label} joined {group}",
                f"{label} joined {group} on {platform}.")
    return (f"{label} joined a channel",
            f"{label} joined a channel on {platform}.")


class PresenceWatchStore:
    """Durable watch list + episode ledger (one JSON file)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        #: Quarantined file name when the store was corrupt on load.
        self.corrupted: str | None = None
        self._data: dict[str, Any] | None = None

    # ── persistence ──────────────────────────────────────────────

    def _fresh(self) -> dict[str, Any]:
        return {"version": STORE_VERSION, "watches": [],
                "last_alert": {}, "last_online": {}}

    def load(self) -> dict[str, Any]:
        """Current data; missing file → fresh, corrupt → quarantined."""
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = self._fresh()
            return self._data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = None
        if (not isinstance(raw, dict)
                or not isinstance(raw.get("watches"), list)):
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            dest = self.path.with_name(
                f"{self.path.name}.corrupt-{stamp}")
            try:
                self.path.replace(dest)
            except OSError:
                dest = self.path
            self.corrupted = str(dest)
            self._data = self._fresh()
            return self._data
        data = self._fresh()
        data.update({k: raw[k] for k in ("watches", "last_alert",
                                         "last_online") if k in raw})
        # A half-written watches list must not take the store down:
        # keep only well-formed watch rows.
        data["watches"] = [
            w for w in (data["watches"] or [])
            if isinstance(w, dict)
            and isinstance(w.get("platform"), str)
            and isinstance(w.get("contact"), str)
            and isinstance(w.get("signals"), list)]
        if not isinstance(data["last_alert"], dict):
            data["last_alert"] = {}
        if not isinstance(data["last_online"], dict):
            data["last_online"] = {}
        self._data = data
        return self._data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True)
                       + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._data = data

    # ── watches ──────────────────────────────────────────────────

    def list_watches(self) -> list[dict[str, Any]]:
        """All watches, in add order."""
        return [dict(w) for w in self.load().get("watches") or []]

    def add_watch(self, platform: str, contact: str,
                  signals: list[str] | tuple[str, ...] | None = None
                  ) -> tuple[dict[str, Any] | None,
                             list[tuple[str, str]]]:
        """Add (or extend) a watch.  Returns (watch, refused).

        ``refused`` lists (signal, reason) for every signal that
        could not be added: unknown words, and signals the platform
        can never observe (WhatsApp online).  When every requested
        signal is refused, no watch is stored and watch is None.
        With no signals given, the watch covers every signal the
        platform supports — never an unobservable one.
        """
        platform = str(platform or "").lower()
        contact = str(contact or "").strip()
        if not contact:
            return None, [("", "give me a contact to watch")]
        if platform not in PLATFORM_SIGNALS:
            return None, [("", f"unknown platform {platform!r} — "
                               "whatsapp or telegram")]
        wanted = [str(s).lower() for s in (signals or [])]
        refused: list[tuple[str, str]] = []
        kept: list[str] = []
        for signal in wanted:
            if signal not in _SIGNAL_WORDS:
                refused.append((signal, "unknown signal — "
                                "status, online, or join"))
            elif not supports(platform, signal):
                refused.append((signal,
                                unsupported_reason(platform, signal)))
            elif signal not in kept:
                kept.append(signal)
        if not wanted:
            kept = list(PLATFORM_SIGNALS[platform])
        if not kept:
            return None, refused
        data = self.load()
        key = _contact_key(platform, contact)
        for watch in data["watches"]:
            if _contact_key(str(watch.get("platform") or ""),
                            str(watch.get("contact") or "")) == key:
                merged = list(watch.get("signals") or [])
                for signal in kept:
                    if signal not in merged:
                        merged.append(signal)
                watch["signals"] = merged
                self._save(data)
                return dict(watch), refused
        watch = {"platform": platform, "contact": contact,
                 "signals": kept, "added_ts": time.time()}
        data["watches"].append(watch)
        self._save(data)
        return dict(watch), refused

    def remove_watch(self, platform: str, contact: str,
                     signals: list[str] | tuple[str, ...] | None = None
                     ) -> bool:
        """Remove signals from a watch, or the whole watch.  True
        when anything was removed."""
        platform = str(platform or "").lower()
        key = _contact_key(platform, contact)
        data = self.load()
        changed = False
        for watch in list(data["watches"]):
            if _contact_key(str(watch.get("platform") or ""),
                            str(watch.get("contact") or "")) != key:
                continue
            if not signals:
                data["watches"].remove(watch)
                changed = True
                continue
            kept = [s for s in (watch.get("signals") or [])
                    if str(s).lower() not in
                    {str(s).lower() for s in signals}]
            if len(kept) != len(watch.get("signals") or []):
                changed = True
            if kept:
                watch["signals"] = kept
            else:
                data["watches"].remove(watch)
        if changed:
            self._save(data)
        return changed

    def match_watches(self, platform: str, signal: str, user_id: str,
                      username: str = "", name: str = ""
                      ) -> list[dict[str, Any]]:
        """Watches on this platform + signal whose contact matches
        this observed identity."""
        platform = str(platform or "").lower()
        signal = str(signal or "").lower()
        return [dict(w) for w in self.load().get("watches") or []
                if str(w.get("platform") or "").lower() == platform
                and signal in [str(s).lower()
                               for s in (w.get("signals") or [])]
                and _matches(str(w.get("contact") or ""), user_id,
                             username, name)]

    # ── events ───────────────────────────────────────────────────

    def check(self, platform: str, signal: str, *, user_id: str,
              username: str = "", name: str = "", episode_id: str = "",
              group: str = "", ts: float | None = None
              ) -> dict[str, Any] | None:
        """One observed event: the alert for the first matching
        watch, or None (no watch, or this episode already alerted).

        ``episode_id`` is what makes an episode: a status post's
        source id (the same post replayed never re-alerts), a join's
        group + second.  An online event alerts only on the
        offline→online transition — the last-seen state is the
        episode, so a re-fired update while still online stays
        silent.
        """
        platform = str(platform or "").lower()
        signal = str(signal or "").lower()
        user_id = str(user_id or "")
        if not user_id:
            return None
        hits = self.match_watches(platform, signal, user_id,
                                  username, name)
        if not hits:
            return None
        data = self.load()
        now = float(ts if ts is not None else time.time())
        if signal == "online":
            state_key = f"{platform}|{user_id}"
            if data["last_online"].get(state_key) is True:
                return None
            data["last_online"][state_key] = True
            self._save(data)
        else:
            episode_key = (f"{platform}|{signal}|{user_id}|"
                           f"{episode_id}")
            if episode_key in data["last_alert"]:
                return None
            data["last_alert"][episode_key] = now
            self._save(data)
        watch = hits[0]
        label = (str(name or "").strip()
                 or (f"@{username.lstrip('@')}" if username else "")
                 or str(watch.get("contact") or ""))
        return {"platform": platform, "signal": signal,
                "contact": str(watch.get("contact") or ""),
                "label": label, "user_id": user_id,
                "username": str(username or ""),
                "name": str(name or ""), "group": str(group or ""),
                "episode_id": str(episode_id or ""), "ts": now}

    def note_offline(self, platform: str, *, user_id: str,
                     username: str = "", name: str = "") -> bool:
        """Record that a watched contact went offline, so the next
        online transition alerts again.  True when a matching
        online watch exists (and the state was recorded)."""
        platform = str(platform or "").lower()
        user_id = str(user_id or "")
        if not user_id:
            return False
        if not self.match_watches(platform, "online", user_id,
                                  username, name):
            return False
        data = self.load()
        data["last_online"][f"{platform}|{user_id}"] = False
        self._save(data)
        return True
