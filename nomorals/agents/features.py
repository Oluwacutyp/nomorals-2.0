"""Feature flags: every major capability of the bot, on/off from a chat.

The owner asked for "all features that can be turned on or off with a
command". A flag is a row in ``kv_store`` (``feature.<key>`` = ``"on"`` /
``"off"``) — no migration needed, survives restarts, visible in ``/features``.

Defaults: everything the bot does today stays ON, EXCEPT the arena loop
(which is a new heavy subsystem — power-gated AND flag-gated).
"""

from __future__ import annotations

from typing import Any

from ..storage.kv import KVStore

__all__ = ["FEATURES", "FeatureRegistry", "feature_enabled"]

#: key -> (default_on, human description)
FEATURES: dict[str, tuple[bool, str]] = {
    "arena": (False, "self-improvement loop: research random tech topics, digest, "
                     "and (power) build candidate features for your review"),
    "group_posts": (True, "autonomous posts in groups she participates in"),
    "proactive_dm": (True, "spontaneous DMs to you"),
    "vision": (True, "process inbound media (photos, videos, documents)"),
    "search": (True, "web research: /search, /searchdeep, /searchleads + the web_research tool"),
    "research": (False, "always-on research & suggestions (lifestyle / tech / cyber domains)"),
    "news": (False, "news sub-agent: fetch feeds, summarize, deliver digests"),
    "games": (True, "the social game engine: /game — 41 games across DM, group "
                    "and channel, with a shared economy, items and leaderboards"),
    "notifier": (True, "deliver alerts (arena builds, research, news, tasks) to your chats"),
    "voice": (True, "voice-note ping-pong: transcribe inbound voice notes, reply with voice"),
}


class FeatureRegistry:
    """Read/write feature flags against the bot's database."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def _key(self, name: str) -> str:
        return f"feature.{name}"

    def get(self, name: str) -> bool:
        if name not in FEATURES:
            return False
        default_on, _ = FEATURES[name]
        try:
            raw = KVStore(self.db).get_raw(self._key(name))
            if raw is not None:
                return raw.lower() == "on"
        except Exception:  # noqa: BLE001 - flag read must never break the bot
            pass
        return default_on

    def set(self, name: str, on: bool, actor: str = "") -> bool:
        if name not in FEATURES:
            return False
        try:
            KVStore(self.db).set_raw(self._key(name), "on" if on else "off")
        except Exception:  # noqa: BLE001
            return False
        return True

    def list(self) -> list[dict[str, Any]]:
        out = []
        for name, (_default, description) in FEATURES.items():
            out.append({"name": name, "on": self.get(name), "description": description})
        return out


def feature_enabled(context: Any, name: str) -> bool:
    """Convenience: is feature ``name`` on, given a context with a db?

    A context without a database is treated as "default" (so offline unit
    tests of the subsystems keep working).
    """
    db = getattr(context, "db", None)
    if db is None:
        return FEATURES.get(name, (False, ""))[0]
    return FeatureRegistry(db).get(name)
