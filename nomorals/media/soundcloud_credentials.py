"""SoundCloud credential mechanism — client_id resolution + persistence.

The audit (§N3) found no cookie / client_id mechanism in
``nomorals/media/*``: the connector discovered its client_id fresh
from ``soundcloud.com``'s JS bundles on every process start (memory
only) and retried only on 401.  This module is the mechanism, in one
place, with an honest account of where the id comes from:

* The client_id is SoundCloud's own PUBLIC id — embedded in the
  JavaScript bundles its web player loads.  It is discovered, never
  invented and never hardcoded: the connector scrapes it from the
  bundles (see :mod:`nomorals.connectors.soundcloud`), exactly the
  way the web player and yt-dlp's SoundCloud extractor obtain one.
  A discovered id can rot — SoundCloud revokes or rate-limits ids —
  so a 401/403 refreshes it (the connector invalidates and
  rediscovers once) and the fresh id overwrites the stored one.
* Resolution precedence — env pin, vault, live discovery:
  ``SOUNDCLOUD_CLIENT_ID`` wins when set (the owner pinning a
  known-good id); otherwise the id this module persisted in the
  vault after a previous discovery; otherwise a live discovery
  through the connector (which itself falls back to the browser
  session's homepage when the plain HTTP fetch fails).  A live
  discovery is persisted to the vault (best-effort) so the next
  process does not re-scrape.
* The value is a credential in storage only: it lives in the vault
  (``VaultCookieStore``, encrypted at rest — the same store the
  Cloudflare-clearance and site-session flows use), never in code,
  never in a log line (only the first 6 characters are ever logged,
  by the connector), never in an error message.
* Sessions are separate from the id: the owner's signed-in session
  is his ``oauth_token`` cookie (see the SoundCloud site descriptor
  and :mod:`nomorals.media.site_auth`) — it is resolved at download
  time and handed to the connector in memory, never stored here.

No paid service, no new API: discovery is a read of SoundCloud's
own public pages.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from typing import Any

from ..core.logging_setup import get_logger
from .downloader_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "SOUNDCLOUD_CLIENT_ID_ENV",
    "VAULT_KEY",
    "valid_client_id",
    "load_discovered_client_id",
    "store_discovered_client_id",
    "resolve_client_id",
    "vault_store_for",
]

#: Env var override — the owner's pin for a known-good client_id.
#: The connector reads the same variable; this module checks it
#: first so a pin never loses to a stored discovery.
SOUNDCLOUD_CLIENT_ID_ENV = "SOUNDCLOUD_CLIENT_ID"

#: Vault key for a discovered client_id.  The key splits (see
#: ``accounts.cf_cookie_store._split_key``) into service
#: ``soundcloud_client_id``, username ``discovered`` — the entry is
#: ``{"client_id": ..., "discovered_at": ...}`` in the credential's
#: encrypted password field.
VAULT_KEY = "soundcloud_client_id:discovered"

#: A client_id is 32 alphanumeric characters.
_CLIENT_ID_SHAPE = re.compile(r"^[a-zA-Z0-9]{32}$")


def valid_client_id(value: Any) -> bool:
    """True when *value* has the shape of a SoundCloud client_id."""
    try:
        return bool(_CLIENT_ID_SHAPE.match(str(value or "").strip()))
    except Exception as e:  # noqa: BLE001 - validation never raises
        record_failure("soundcloud_credentials.valid_client_id", e)
        return False


def load_discovered_client_id(store: Any) -> str:
    """The vault-persisted discovered id, or "".

    A malformed or missing entry reads as absent — a stored value
    that is not a client_id must never be handed to the API.
    Never raises; never logs the value."""
    try:
        if store is None:
            return ""
        entry = store.get(VAULT_KEY)
        if not isinstance(entry, dict):
            return ""
        candidate = str(entry.get("client_id") or "").strip()
        return candidate if valid_client_id(candidate) else ""
    except Exception as e:  # noqa: BLE001 - a failed read is absent
        record_failure("soundcloud_credentials.load", e)
        return ""


def store_discovered_client_id(store: Any, client_id: str, *,
                               now: float | None = None) -> bool:
    """Persist a freshly discovered client_id in the vault.

    A malformed id is refused (False) — a bad value must never
    overwrite a working stored one.  A store that cannot open
    (locked vault, no database) is False, never a raise: the
    download proceeds with the in-memory id either way.  Never
    logs the value."""
    try:
        if store is None or not valid_client_id(client_id):
            return False
        store.set(VAULT_KEY, {
            "client_id": str(client_id).strip(),
            "discovered_at": float(
                time.time() if now is None else now),
        })
        return True
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        record_failure("soundcloud_credentials.store", e)
        return False


def resolve_client_id(*, connector: Any = None,
                      vault_store: Any = None,
                      env: Mapping[str, str] | None = None
                      ) -> tuple[str, str]:
    """Resolve a working SoundCloud client_id.

    Returns ``(client_id, source)`` with ``source`` one of
    ``"env"`` (the owner's pin), ``"vault"`` (a previously
    discovered id, persisted), or ``"discovered"`` (a live
    discovery through the connector just now — persisted to the
    vault on the way out, best-effort).  ``("", "unavailable")``
    when nothing resolved: the caller falls back to the
    connector's own resolution instead of failing the download
    here — this function never makes a download worse.  The
    connector is duck-typed (``client_id()``); None means no live
    discovery is possible.  Never raises; never logs the value."""
    environ = env if env is not None else os.environ
    try:
        pinned = str(environ.get(SOUNDCLOUD_CLIENT_ID_ENV, "")
                     or "").strip()
        if valid_client_id(pinned):
            return pinned, "env"
        stored = load_discovered_client_id(vault_store)
        if stored:
            return stored, "vault"
        if connector is None:
            return "", "unavailable"
        try:
            discovered = str(connector.client_id() or "").strip()
        except Exception as exc:  # noqa: BLE001 - discovery failed
            _log.info("soundcloud: live client_id discovery failed: "
                      "%s", type(exc).__name__)
            return "", "unavailable"
        if not valid_client_id(discovered):
            return "", "unavailable"
        if vault_store is not None:
            store_discovered_client_id(vault_store, discovered)
        _log.info("soundcloud: client_id resolved via live discovery "
                  "(%s…)", discovered[:6])
        return discovered, "discovered"
    except Exception as e:  # noqa: BLE001 - resolution never raises
        record_failure("soundcloud_credentials.resolve", e)
        return "", "unavailable"


def vault_store_for(db: Any, context: Any = None, *,
                    actor: str = "soundcloud-client-id") -> Any | None:
    """The vault-backed get/set store for a discovered client_id,
    or None when there is no database to open a vault over.

    The same :class:`accounts.cf_cookie_store.VaultCookieStore`
    the cookie flows use (lazy open — constructing it is
    side-effect free; a locked vault reads as unavailable, never
    a crash).  None keeps the caller on in-memory resolution."""
    if db is None:
        return None
    try:
        from ..accounts.cf_cookie_store import VaultCookieStore
        return VaultCookieStore(db, context=context, actor=actor)
    except Exception as e:  # noqa: BLE001 - no store is no store
        record_failure("soundcloud_credentials.vault_store", e)
        return None
