"""The authenticated download rung (B5) — a site's session, resolved
at run time, never in code.

B4 declared the requirement: a :class:`SiteDescriptor` with
``auth="account"`` / ``"vault"`` names a site whose downloads need
the owner's signed-in account, and a descriptor with
``auth_domains`` / ``auth_cookie_names`` names the shape of that
session (the cookie domains it lives under, the cookie names that
prove a signed-in session — an anonymous ``cf_clearance`` is not a
login).  B4 declared; it did not enforce.  This module enforces:

* :class:`SiteAuthResolver` resolves a descriptor's session at run
  time, in this order — the unified cookie estate first (a live
  session the owner already earned there, via a browser login or a
  ``/cookies import``, is the session; no vault touch), then the
  accounts vault (a stored session, upserted into the estate so the
  next run finds it there).  A credential is fetched once per run
  and never persisted to disk outside the vault / estate.
* The vault entry lives under the kit's own cookie-store adapter
  (:class:`nomorals.accounts.cf_cookie_store.VaultCookieStore`,
  namespace ``site_auth_cookies``) — the same encrypted-at-rest
  shape the Cloudflare clearance flow already uses, so there is one
  vault door for session credentials, not a second.  The entry is
  ``{"cookies": [...], "expires_at": ...}``; the kit keeps sole
  authority over expiry here too (the resolver reads
  ``expires_at`` and each cookie's own expiry, and never invents a
  live session from a stale one).
* The owner's OWN account only — no shared, pooled, or trial
  identities; no bulk-account anything.  A session is the owner's
  session, provided by the owner (``/cookies import`` with a
  cookies.txt / JSON export from their own browser, or a stored
  vault entry); nothing here logs in, scrapes a login, or fabricates
  a session.

Value discipline (the solved-session rule, extended): a credential
VALUE appears nowhere — not in a log line, not in a journal entry,
not in an error message, not in a recipe, not in a descriptor.
Log lines carry the site, the status, the source, and the cookie
COUNT.  Error paths name the credential (``auth_token``) and the
exact owner action, never the value.

The engine (:mod:`nomorals.media.downloader`) engages this rung
two ways:

1. Up-front gate — a descriptor with ``auth != "none"`` resolves
   before the chain; a missing / expired / unavailable session
   fails the download immediately as found-but-blocked, class
   ``login``, with the owner action — the chain never runs
   anonymously for a site that declared it needs an account.
2. Login-block retry — a descriptor with ``auth == "none"`` whose
   connector reports a classified ``login`` block (the site itself
   says this content needs a signed-in account — SoundCloud's
   ``BLOCK_LOGIN``) resolves the session and retries the direct
   stage once, authenticated through the estate's cookies file.
   No session, no retry: the failure stands as found-but-blocked
   ``login`` and the remaining anonymous rungs are not burned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger
from .downloader_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "SiteAuth",
    "SiteAuthResolver",
    "store_site_session",
    "vault_key",
]

#: Vault key namespace for owner-provided site sessions.  Mirrors
#: the kit's ``cloudflare_cookies`` namespace
#: (:mod:`nomorals.media.download_kit`): the key is
#: ``site_auth_cookies:<site>`` and the entry is
#: ``{"cookies": [...], "expires_at": ...}`` — the whole entry a
#: JSON payload in one credential's encrypted password field.
_AUTH_NS = "site_auth_cookies"

#: The source stamp a resolved session carries into the estate.
_ESTATE_SOURCE = "site-auth"

#: Default session lifetime for a stored session (7 days — a
#: browser session's usual honest horizon; the entry's own
#: ``expires_at`` wins over this whenever the owner flow stored
#: a real one).
_DEFAULT_TTL_S = 7 * 24 * 3600


def vault_key(site: str) -> str:
    """The vault-store key for *site*'s session."""
    return f"{_AUTH_NS}:{str(site or '').strip().lower()}"


def owner_action_for(site: str, status: str, reason: str = "") -> str:
    """The exact owner action for a session that did not resolve.

    Names the site and the command; never a value, never a guess
    at the credential itself."""
    name = str(site or "this site")
    if status == "expired":
        return (
            f"your stored {name} session expired — run /cookies "
            "import again with a fresh export from your own "
            "browser (your own account).")
    if status == "unavailable":
        detail = f" ({reason})" if reason else ""
        return (
            "the credential vault could not be opened"
            f"{detail} — unlock it and retry.")
    return (
        f"add your {name} session: run /cookies import with a "
        "cookies.txt or JSON export from your own browser (your "
        "own account) — I use only your session, never anyone "
        "else's.")


@dataclass(frozen=True)
class SiteAuth:
    """The outcome of resolving one site's session.

    ``status``: ``ok`` (a live session — ``cookies`` holds it, in
    memory only), ``missing`` (nothing stored), ``expired`` (a
    stored session went stale), ``unavailable`` (the vault could
    not be opened — ``reason`` carries the vault's own words).
    ``source``: ``estate`` (already live there) or ``vault``
    (resolved from the vault, upserted into the estate) — ""
    when no session resolved.  ``owner_action`` is the exact
    owner sentence when ``status`` is not ``ok`` ("" when ok).
    ``cookies`` are the session's cookie records in the estate's
    canonical shape — values live here in memory only, never
    logged, never written outside the vault / estate.
    """

    site: str
    status: str = "ok"
    source: str = ""
    cookies: tuple[dict[str, Any], ...] = ()
    owner_action: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class SiteAuthResolver:
    """Resolve a site descriptor's session at run time.

    ``vault_store`` is the duck-typed get/set store (the kit's
    contract: ``get(key) -> dict | None``, ``set(key, entry)``) —
    an injected store wins (tests hand in a dict-backed fake); the
    production default is the accounts-vault adapter, built lazily
    so constructing a resolver is side-effect free and the vault
    opens only when a session is actually needed.  ``cookie_estate``
    is the unified estate (A6) — injected wins, otherwise the home
    estate over ``db`` / the context's database.  A resolver with
    no database has no vault and no estate: every non-``"none"``
    descriptor resolves ``unavailable`` with the reason named,
    never a fabricated session.
    """

    def __init__(self, *, vault_store: Any = None,
                 cookie_estate: Any = None, db: Any = None,
                 context: Any = None,
                 actor: str = "downloader-site-auth") -> None:
        self._vault_store_injected = vault_store
        self._vault_store_cache: Any = None
        self._estate_injected = cookie_estate
        self._estate_cache: Any = None
        self._db = db
        self._context = context
        self._actor = actor

    # -- lazy production defaults ------------------------------------

    def _vault_store(self) -> Any | None:
        """The vault-backed get/set store, or None when there is
        no database to open a vault over."""
        if self._vault_store_cache is not None:
            return self._vault_store_cache
        if self._vault_store_injected is not None:
            self._vault_store_cache = self._vault_store_injected
            return self._vault_store_cache
        db = self._db
        if db is None and self._context is not None:
            db = getattr(self._context, "db", None)
        if db is None:
            return None
        try:
            from ..accounts.cf_cookie_store import VaultCookieStore
            self._vault_store_cache = VaultCookieStore(
                db, context=self._context, actor=self._actor)
        except Exception as e:  # a store that cannot be built is
            # an unavailable vault, never a crash at resolve time
            record_failure("site_auth.vault_store", e)
            self._vault_store_cache = None
        return self._vault_store_cache

    def _estate(self) -> Any | None:
        """The unified cookie estate.  The context's database wins
        when one is attached; otherwise the home estate (resolved
        through load_settings(), honouring $NM_HOME) — the same
        estate the download chain's cookies file is exported
        from, so a session resolved here is the session the
        direct stage sends.  None only when no estate can be
        opened at all."""
        if self._estate_cache is not None:
            return self._estate_cache
        if self._estate_injected is not None:
            self._estate_cache = self._estate_injected
            return self._estate_cache
        db = self._db
        if db is None and self._context is not None:
            db = getattr(self._context, "db", None)
        try:
            from ..storage.cookie_estate import home_estate
            self._estate_cache = (
                home_estate(db=db) if db is not None
                else home_estate())
        except Exception as e:  # an estate that cannot be opened
            # reads as "no estate session", never a crash
            record_failure("site_auth.estate", e)
            self._estate_cache = None
        return self._estate_cache

    # -- resolution ---------------------------------------------------

    def resolve(self, descriptor: Any, *,
                now: float | None = None, require: bool = False
                ) -> SiteAuth:
        """Resolve *descriptor*'s session.  Never raises — every
        failure mode is a status, and the log carries the site,
        the status, the source, and the cookie count, never a
        value.

        ``require`` means the caller already knows a session is
        needed — a classified login block said so.  An
        ``auth == "none"`` descriptor then resolves the same way
        instead of answering ok without a session: an anonymous
        retry after a login block is never acceptable, and the
        retry must fail LOGIN when no session resolves."""
        site = str(getattr(descriptor, "site", "") or "")
        auth = str(getattr(descriptor, "auth", "none") or "none")
        if auth == "none" and not require:
            return SiteAuth(site=site, status="ok")
        now = time.time() if now is None else float(now)
        domains = tuple(getattr(descriptor, "auth_domains", ()) or ())
        names = tuple(getattr(descriptor, "auth_cookie_names", ())
                      or ())
        if not domains:
            # A descriptor that declares auth but names no session
            # domain is a definition bug, not an owner action —
            # say so, and resolve nothing.
            reason = (f"site descriptor for {site!r} declares auth "
                      f"{auth!r} but names no auth_domains")
            _log.warning("site auth site=%s status=unavailable "
                         "reason=descriptor", site)
            return SiteAuth(site=site, status="unavailable",
                            reason=reason,
                            owner_action=owner_action_for(
                                site, "unavailable", reason))
        try:
            # 1. The estate first: a live session the owner already
            # earned there is the session — no vault touch.
            estate = self._estate()
            if estate is not None:
                session = self._estate_session(estate, domains,
                                               names, now=now)
                if session:
                    _log.info("site auth site=%s status=ok "
                              "source=estate cookies=%d",
                              site, len(session))
                    return SiteAuth(site=site, status="ok",
                                    source="estate",
                                    cookies=tuple(session))
            # 2. The vault: a stored session, upserted into the
            # estate so the next run finds it there.
            store = self._vault_store()
            if store is None:
                reason = ("no database is attached to this context "
                          "— the credential vault cannot be opened")
                _log.info("site auth site=%s status=unavailable "
                          "reason=no-vault", site)
                return SiteAuth(site=site, status="unavailable",
                                reason=reason,
                                owner_action=owner_action_for(
                                    site, "unavailable", reason))
            entry = self._vault_entry(store, site)
            if entry is None:
                if getattr(store, "unavailable_reason", ""):
                    reason = str(store.unavailable_reason)
                    _log.info("site auth site=%s status=unavailable "
                              "reason=vault-locked", site)
                    return SiteAuth(site=site, status="unavailable",
                                    reason=reason,
                                    owner_action=owner_action_for(
                                        site, "unavailable", reason))
                _log.info("site auth site=%s status=missing", site)
                return SiteAuth(site=site, status="missing",
                                owner_action=owner_action_for(
                                    site, "missing"))
            cookies, expired = _live_cookies(entry, now=now)
            if expired or not cookies:
                _log.info("site auth site=%s status=expired", site)
                return SiteAuth(site=site, status="expired",
                                owner_action=owner_action_for(
                                    site, "expired"))
            if estate is not None:
                try:
                    estate.upsert(list(cookies), source=_ESTATE_SOURCE)
                except Exception as e:  # the session still counts
                    # — the estate is persistence, not the proof
                    record_failure("site_auth.estate_upsert", e)
            _log.info("site auth site=%s status=ok source=vault "
                      "cookies=%d", site, len(cookies))
            return SiteAuth(site=site, status="ok", source="vault",
                            cookies=tuple(cookies))
        except Exception as e:  # noqa: BLE001 — resolve never raises
            record_failure("site_auth.resolve", e)
            reason = type(e).__name__
            return SiteAuth(site=site, status="unavailable",
                            reason=reason,
                            owner_action=owner_action_for(
                                site, "unavailable", reason))

    # -- helpers -------------------------------------------------------

    def _vault_entry(self, store: Any, site: str) -> Any | None:
        """The stored session entry for *site*, or None.  A store
        read failure is None here — the unavailable check above
        (``unavailable_reason``) is what separates a locked vault
        from an absent session."""
        try:
            entry = store.get(vault_key(site))
        except Exception as e:  # noqa: BLE001 — read failure reads
            record_failure("site_auth.vault_read", e)  # as absent
            return None
        return entry if isinstance(entry, dict) else None

    @staticmethod
    def _estate_session(estate: Any, domains: tuple[str, ...],
                        names: tuple[str, ...],
                        *, now: float) -> list[dict[str, Any]]:
        """Live session cookies in the estate for *domains* — only
        the named session cookies when the descriptor names them
        (an anonymous clearance cookie is not a login)."""
        out: list[dict[str, Any]] = []
        try:
            for domain in domains:
                for rec in estate.cookies_for(domain):
                    if not isinstance(rec, dict):
                        continue
                    if names and str(rec.get("name") or "") \
                            not in names:
                        continue
                    if 0 < float(rec.get("expires") or 0) <= now:
                        continue
                    out.append(rec)
        except Exception as e:  # noqa: BLE001 — an estate read
            record_failure("site_auth.estate_read", e)  # failure is
            return []                                    # "no session"
        return out


def _live_cookies(entry: dict[str, Any], *,
                  now: float) -> tuple[list[dict[str, Any]], bool]:
    """(live cookies, entry-expired?) from a stored session entry.

    The entry's ``expires_at`` is the session's horizon; a cookie
    whose own expiry passed is dropped even when the entry is
    live.  ``expired`` is True when the entry itself went stale —
    the caller reports that, not a bare missing session."""
    raw = entry.get("cookies")
    if not isinstance(raw, list):
        return [], False
    expires_at = entry.get("expires_at")
    try:
        expired = bool(expires_at is not None
                       and float(expires_at) <= now)
    except (TypeError, ValueError):
        expired = False
    if expired:
        return [], True
    live: list[dict[str, Any]] = []
    try:
        from ..storage.cookie_estate import normalize_record
    except Exception:  # noqa: BLE001 — without the normalizer
        normalize_record = None  # every row reads as unusable
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        norm = normalize_record(rec) if normalize_record else None
        if norm is None:
            continue
        if 0 < float(norm.get("expires") or 0) <= now:
            continue
        live.append(norm)
    return live, False


def store_site_session(store: Any, site: str,
                       cookies: list[dict[str, Any]],
                       *, ttl_s: float = _DEFAULT_TTL_S,
                       now: float | None = None) -> bool:
    """Store *site*'s session cookies in a vault-backed store.

    The owner flow (``/cookies import`` today, a vault flow
    tomorrow) calls this — the engine only ever READS.  Returns
    False on any problem (empty session, unusable rows, store
    failure); never raises, never logs a value.  ``store`` is the
    kit's duck-typed get/set store."""
    try:
        now = time.time() if now is None else float(now)
        from ..storage.cookie_estate import normalize_record
        live = [n for n in (normalize_record(c)
                            for c in (cookies or []))
                if n is not None]
        if not live or not str(site or "").strip():
            return False
        store.set(vault_key(site),
                  {"cookies": live,
                   "expires_at": now + float(ttl_s)})
        return True
    except Exception as e:  # noqa: BLE001 — a failed store is False
        record_failure("store_site_session", e)
        return False
