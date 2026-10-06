"""Real capability backends for running plugins.

:mod:`.loader` defines the *gated* surface (:class:`PluginCapabilities`);
this module wires it to real implementations so a plugin granted
``artifacts.write`` / ``storage.kv`` / ``network.fetch`` / ``llm.chat``
can actually use them (e.g. via ``nm plugin run``). Every accessor still
checks the granted permission set first — wiring never widens what the
manifest allows.

``llm.chat`` is wired to a real model path: the agent context's own router
when one is available (the user's configured, settings-driven chain with
its broker and learning hook), else the env-based provider chain from
:mod:`nomorals.llm.defaults` (free/local first) with a
:class:`~nomorals.llm.broker.ModelBroker` attached. The fallback chain
builds lazily on the first ``chat()`` call — plugin runs that never touch
the model pay nothing. When neither exists, the chatter stays ``None`` and
the loader reports its honest ``llm.chat: no model wired`` error.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..llm.base import Message
from ..llm.broker import ModelBroker
from ..llm.defaults import build_chain, specs_from_env, sync_broker_cards
from .errors import PluginError
from .loader import PluginCapabilities

__all__ = [
    "PluginKV",
    "make_fetcher",
    "wire_capabilities",
]

_log = get_logger(__name__)

PLUGIN_KV_DDL = """
CREATE TABLE IF NOT EXISTS plugin_kv (
    plugin     TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (plugin, key)
);
"""

# A single plugin fetch shouldn't be able to eat the disk or hang the run.
FETCH_MAX_BYTES = 10 * 1024 * 1024


class PluginKV:
    """A plugin's private key-value store, namespaced per plugin.

    Backed by the main :class:`Database` (same pattern as the plugin
    registry itself). Values must be JSON-serializable; ``get`` returns
    the stored object, or ``default`` when the key is missing.
    """

    def __init__(self, db: Any, plugin_name: str) -> None:
        self._db = db
        self._plugin = plugin_name
        db.executescript(PLUGIN_KV_DDL)

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.execute(
            "SELECT value FROM plugin_kv WHERE plugin=? AND key=?",
            (self._plugin, str(key))).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return default

    def set(self, key: str, value: Any) -> None:
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise PluginError(
                f"kv value for {key!r} is not JSON-serializable: {exc}"
            ) from exc
        now = time.time()
        self._db.execute(
            "INSERT INTO plugin_kv (plugin, key, value, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(plugin, key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (self._plugin, str(key), encoded, now))

    def delete(self, key: str) -> bool:
        cur = self._db.execute(
            "DELETE FROM plugin_kv WHERE plugin=? AND key=?",
            (self._plugin, str(key)))
        return cur.rowcount > 0

    def keys(self) -> list[str]:
        rows = self._db.execute(
            "SELECT key FROM plugin_kv WHERE plugin=? ORDER BY key",
            (self._plugin,)).fetchall()
        return [r["key"] for r in rows]

    def clear(self) -> int:
        cur = self._db.execute(
            "DELETE FROM plugin_kv WHERE plugin=?", (self._plugin,))
        return cur.rowcount


def make_fetcher(*, timeout: int = 30,
                 max_bytes: int = FETCH_MAX_BYTES):
    """Build the ``fetch(url, timeout=...)`` callable for capabilities.

    Plain urllib, Devon user-agent, hard size cap — a plugin can't
    download the internet into memory.
    """
    def _fetch(url: str, *, timeout: int = timeout) -> bytes:
        if not str(url).startswith(("http://", "https://")):
            raise PluginError(f"fetch refuses non-http(s) url: {url!r}")
        req = urllib.request.Request(
            str(url), headers={"User-Agent": "DevonPlugin/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = 0
                chunks: list[bytes] = []
                for chunk in iter(lambda: resp.read(65536), b""):
                    total += len(chunk)
                    if total > max_bytes:
                        raise PluginError(
                            f"fetch of {url!r} exceeds {max_bytes} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
        except PluginError:
            raise
        except Exception as exc:
            raise PluginError(f"fetch of {url!r} failed: {exc}") from exc
    return _fetch


_CHAT_ROLES = ("system", "user", "assistant", "tool")


def _to_messages(raw: Any) -> list[Message]:
    """Validate a plugin's ``[{role, content}, ...]`` into :class:`Message`s.

    Fail-fast: a malformed call surfaces as a clear :class:`PluginError`
    here, not as a confusing provider rejection three layers down.
    """
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise PluginError(
            "llm.chat: messages must be a list of {role, content} dicts, "
            f"got {type(raw).__name__}")
    if not raw:
        raise PluginError("llm.chat: messages must not be empty")
    messages: list[Message] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PluginError(
                f"llm.chat: message {i} must be a dict, "
                f"got {type(item).__name__}")
        role = str(item.get("role", "")).strip().lower()
        if role not in _CHAT_ROLES:
            raise PluginError(
                f"llm.chat: message {i} has invalid role "
                f"{item.get('role')!r}; want one of {_CHAT_ROLES}")
        content = item.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        messages.append(Message(role=role, content=content))
    return messages


def _context_router(context: Any) -> Any | None:
    """The agent's own LLM router, when the context carries one.

    Duck-typed on purpose: any object with a ``chat(messages)`` method
    works. Returns None for exotic contexts (tests, bare Database) so the
    caller can fall back to the env-based chain.
    """
    router = getattr(context, "router", None)
    if router is not None and callable(getattr(router, "chat", None)):
        return router
    return None


def _make_chatter(specs: list[Any], context: Any = None) -> Any:
    """Build the lazy ``chatter(messages) -> str`` for ``llm.chat``.

    Prefers the agent context's own router when one is available — that's
    the user's configured, settings-driven chain (with its broker and
    learning hook), not a parallel one. Otherwise the env-based provider
    chain (free/local first) builds lazily on the *first* chat call, so
    plugin runs that never touch the model import no provider modules and
    open no connections. Thread-safe: concurrent first calls build the
    fallback chain exactly once.

    Failures raise :class:`PluginError` with the router's own diagnosis
    (which providers failed, who served) — never a bare empty string.
    """
    lock = threading.Lock()
    state: dict[str, Any] = {"router": None}

    def _chatter(messages: list[dict[str, str]]) -> str:
        # Validate first: malformed input fails fast without importing a
        # single provider module.
        msgs = _to_messages(messages)
        router = state["router"]
        if router is None:
            router = _context_router(context)
            if router is None:
                with lock:
                    router = state["router"]
                    if router is None:
                        router = build_chain(specs)
                        broker = ModelBroker()
                        sync_broker_cards(broker, router)
                        # Also register lifecycle-managed models (local GGUFs etc.)
                        try:
                            from ..cmdline.commands.models import _broker_for as _lifecycle_broker
                            from ..llm.lifecycle import ModelLifecycle
                            lc = ModelLifecycle(getattr(context, "db", None))
                            for model in lc.list():
                                from ..cmdline.commands.models import _card_for
                                try:
                                    broker.register(_card_for(model))
                                except Exception:
                                    pass
                            primary = lc.primary
                            if primary and broker.card(primary) is not None:
                                broker.promote(primary)
                        except Exception:
                            pass
                        router.set_broker(broker)
                        state["router"] = router
                        _log.info(
                            "plugin llm.chat chain ready: %s",
                            router.providers(),
                        )
            if router is not None:
                state["router"] = router
        try:
            resp = router.chat(msgs)
        except Exception as exc:  # noqa: BLE001 - contract is PluginError
            raise PluginError(f"llm.chat failed: {exc}") from exc
        if not resp.ok:
            note = f" ({resp.fallback_note})" if resp.fallback_note else ""
            raise PluginError(
                f"llm.chat failed: {resp.error or 'unknown error'}{note}")
        return resp.text

    return _chatter


def wire_capabilities(context: Any, plugin: Any,
                      workspace_base: str | Path | None = None
                      ) -> PluginCapabilities:
    """Build :class:`PluginCapabilities` with real backends.

    ``plugin`` is an :class:`InstalledPlugin` (name, manifest, ...).
    ``workspace_base`` defaults to the context's workspace dir (same
    layout ``nm plugin`` and ``nm datasci`` use).
    """
    from ..storage.artifacts import ArtifactStore
    from ..storage.blob import BlobStore

    settings = getattr(context, "settings", None)
    root = workspace_base or (
        getattr(settings, "workspace_dir", None) if settings else None)
    base = Path(root) if root else Path.cwd() / "workspace"
    db = context.db
    # The artifact/blob tables come from storage migrations; ensure the
    # schema is present (idempotent) so wiring works on a fresh Database
    # too, not just a fully-booted agent context.
    migrate = getattr(db, "migrate", None)
    if callable(migrate):
        migrate()
    artifacts = ArtifactStore(db, BlobStore(db, str(base / "blobs")))
    kv = PluginKV(db, plugin.name)
    # llm.chat: prefer the agent context's own router (the user's
    # configured chain) when available; otherwise wire a lazy env-based
    # chain (free/local first) with a capability broker. No specs and no
    # context router → chatter stays None and the loader reports the
    # honest "llm.chat: no model wired" error.
    specs = specs_from_env()
    if _context_router(context) is None and not specs:
        chatter = None
    else:
        chatter = _make_chatter(specs, context)
    return PluginCapabilities(
        granted=frozenset(plugin.manifest.permissions),
        artifacts=artifacts,
        kv=kv,
        fetcher=make_fetcher(),
        chatter=chatter,
    )
