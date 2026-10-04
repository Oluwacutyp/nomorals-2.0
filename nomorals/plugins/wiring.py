"""Real capability backends for running plugins.

:mod:`.loader` defines the *gated* surface (:class:`PluginCapabilities`);
this module wires it to real implementations so a plugin granted
``artifacts.write`` / ``storage.kv`` / ``network.fetch`` can actually use
them (e.g. via ``nm plugin run``). Every accessor still checks the granted
permission set first — wiring never widens what the manifest allows.

``llm.chat`` stays unwired here: the CLI has no provider chain to hand
over, so a plugin that calls it gets the loader's honest "no model wired"
error instead of a half-working stub.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
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
    return PluginCapabilities(
        granted=frozenset(plugin.manifest.permissions),
        artifacts=artifacts,
        kv=kv,
        fetcher=make_fetcher(),
    )
