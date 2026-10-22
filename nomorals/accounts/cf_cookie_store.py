"""Production vault adapter for the download kit's cookie store.

The download kit (``nomorals/media/download_kit.py``) models the
owner-cookie flow for Cloudflare clearance: ``store_owner_cookie``
and ``get_or_request_cookie`` speak to a duck-typed store —
``get(key)`` / ``set(key, entry)`` with keys of the form
``cloudflare_cookies:{domain}`` and entries of the form
``{"value", "expires_at", "domain"}``. Pass #79 gave that flow its
first consumers (the browser solver and the downloader gate) but no
production backing: the store existed only where a caller injected
one, and no caller did — the cookie rung was decorative in the real
deployment.

This module is the backing. It lives in the accounts layer because
that layer owns every vault door (Pass #74 converged all accounts-
vault opens onto ``open_vault``: owner seal → power-mode key →
passphrase), and a clearance cookie is a session credential — it
belongs inside the encrypted vault, never in a plaintext jar beside
it.

Semantics, deliberately thin:

* The kit keeps sole authority over expiry: the adapter returns
  the stored entry exactly as written (even when its
  ``expires_at`` has passed) and the kit's
  ``get_or_request_cookie`` decides "have" vs "expired" — there is
  no parallel expiry logic here, and ``CredentialVault.get``
  (which never filters on expiry) is the read path for the same
  reason.
* The cookie value is encrypted at rest: the whole entry is a JSON
  payload in a credential's encrypted password field, under
  ``service=<key namespace>, username=<key remainder>``.
* A locked vault is a reported state, not a crash: ``get``
  returns None (the kit reads that as "no cookie" and asks),
  ``set`` raises :class:`CookieStoreUnavailable` (the kit's
  ``store_owner_cookie`` turns that into a plain False), and
  :attr:`VaultCookieStore.unavailable_reason` carries the vault's
  own words — ``open_vault`` names every source it checked — for
  the consumers that surface why the rung is down.
* The vault opens LAZILY (first get/set), so constructing the
  adapter is side-effect free — the same posture as the
  downloader's other lazy kit accessors (Pass #76). A successful
  open is cached; a failed one is retried on the next call, so an
  owner who unlocks the vault mid-process arms the rung without a
  restart.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.errors import NotFound
from ..core.logging_setup import get_logger
from .vault import open_vault

_log = get_logger(__name__)

__all__ = [
    "CookieStoreUnavailable",
    "VaultCookieStore",
    "default_cf_cookie_store",
]


class CookieStoreUnavailable(Exception):
    """The vault behind the cookie store could not be opened."""


def _split_key(key: str) -> tuple[str, str]:
    """Kit key → (vault service, vault username).

    ``cloudflare_cookies:example.com`` → service ``cloudflare_cookies``,
    username ``example.com``. A key without a namespace is stored
    under a generic service so the adapter stays total over the
    kit's key space."""
    head, sep, tail = str(key).partition(":")
    if sep and head and tail:
        return head, tail
    return "kit_store", str(key)


class VaultCookieStore:
    """The kit's get/set cookie store, backed by the accounts vault."""

    def __init__(self, db: Any, context: Any = None, *,
                 vault: Any = None, actor: str = "cloudflare-cookies"
                 ) -> None:
        self._db = db
        self._context = context
        self._vault = vault          # injectable (tests / ready vaults)
        self._actor = actor
        self._reason = ""

    # -- availability ------------------------------------------------------

    @property
    def unavailable_reason(self) -> str:
        """Why the store is unavailable ("" when it is, or when no
        open has been attempted yet). Set by the last failed open or
        read; the consumers surface it verbatim."""
        return self._reason

    def available(self) -> bool:
        """True when the vault opens in the calling context."""
        return self._open() is not None

    def _open(self) -> Any:
        if self._vault is not None:
            return self._vault
        if self._db is None:
            self._reason = ("no database is attached to this "
                            "context — the credential vault cannot "
                            "be opened")
            return None
        try:
            self._vault = open_vault(self._db, context=self._context,
                                     actor=self._actor)
            self._reason = ""
        except Exception as exc:  # noqa: BLE001 - locked is a state,
            # not a crash; open_vault's message names every source
            # it checked (seal / power key / passphrase)
            self._reason = str(exc) or exc.__class__.__name__
            _log.debug("cf cookie store: vault unavailable: %s",
                       self._reason)
            return None
        return self._vault

    # -- the kit's duck-typed contract --------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        """The stored entry for *key*, or None (absent, corrupt, or
        vault unavailable). Never raises — the kit treats a raise
        and a None identically, and None keeps the reason here
        instead of in an exception nobody reads."""
        vault = self._open()
        if vault is None:
            return None
        service, username = _split_key(key)
        try:
            cred = vault.get(service, username, mark_used=False)
        except NotFound:
            return None
        except Exception as exc:  # noqa: BLE001 - a failed read is
            # "no cookie" to the kit, with the reason kept here
            self._reason = f"vault read failed: {exc}"
            _log.debug("cf cookie store: read failed: %s", exc)
            return None
        try:
            entry = json.loads(cred.password or "")
        except (TypeError, ValueError):
            return None
        return entry if isinstance(entry, dict) else None

    def set(self, key: str, entry: dict[str, Any]) -> None:
        """Store *entry* under *key*, encrypted at rest.

        Raises :class:`CookieStoreUnavailable` when the vault
        cannot be opened — the kit's ``store_owner_cookie`` catches
        exceptions and reports False, so a silent no-op here would
        read as a successful store there. Other vault errors
        propagate for the same reason."""
        vault = self._open()
        if vault is None:
            raise CookieStoreUnavailable(
                self._reason or "credential vault unavailable")
        service, username = _split_key(key)
        vault.store(service, username, json.dumps(entry),
                    credential_type="cf_clearance",
                    tags=["cloudflare-cookie"])


def default_cf_cookie_store(context: Any,
                            actor: str = "cloudflare-cookies"
                            ) -> VaultCookieStore | None:
    """The production default store for *context*, or None.

    None when the context carries no database (flows built without
    a runtime context — the gate/solver then behave exactly as an
    unwired store did: detect/ask only). The adapter itself opens
    the vault lazily, so returning one costs nothing here."""
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        return None
    return VaultCookieStore(db, context=context, actor=actor)
