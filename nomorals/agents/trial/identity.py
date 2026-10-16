"""Trial identities, bound one-per-service in the TrialVault.

A trial identity is a fully synthetic persona — generated names,
a disposable-domain email, a synthetic-shaped phone, a random
password — minted for exactly one service and stored sealed in
:mod:`nomorals.agents.trial.vault` (the TrialVault, kept separate
from ``accounts/vault.py`` by design).

Three structural guarantees, all in code, not policy:

* **One per service.** :meth:`TrialIdentityBinder.bind` returns the
  already-bound identity when one exists; it never mints a second
  identity for the same service. There is no batch API — ``bind``
  takes one service, so a farm cannot be expressed through this
  module.
* **Reuse across services is structurally impossible.** An identity
  is stored under its own service's TrialVault platform key, and
  :meth:`TrialIdentityBinder.identity_for` reads only that key and
  verifies the stored record names the requested service. There is
  no method that lists identities, copies one to another service, or
  returns an identity for a service it was not minted for.
* **Synthetic-only generation.** :func:`generate_trial_identity`
  takes a service name (plus an optional RNG for tests) and nothing
  else. It has no parameter, import, or read path for a person's
  actual details — the values come from built-in synthetic pools and
  the secrets module. If a caller wants a person's actual details
  used, this module cannot do it: there is no input for them.

The email domains are the disposable providers already wired into
:mod:`nomorals.accounts.temp_mail` (1secmail, guerrillamail,
mail.tm) — the identity's mailbox is a throwaway address from one of
those providers, fetched at drive time, never a personal inbox.
"""

from __future__ import annotations

import json
import random
import secrets
import string
import time
from typing import Any

from ...core.logging_setup import get_logger
from .vault import TrialVault

__all__ = ["TrialIdentityBinder", "generate_trial_identity",
           "SYNTHETIC_DOMAINS"]

_log = get_logger(__name__)

#: disposable-domain emails only — same providers as temp_mail.
SYNTHETIC_DOMAINS: tuple[str, ...] = (
    "1secmail.com",
    "guerrillamail.com",
    "mail.tm",
)

_FIRST_NAMES: tuple[str, ...] = (
    "Adaeze", "Chiamaka", "Tunde", "Funke", "Emeka", "Zainab",
    "Kunle", "Ngozi", "Ibrahim", "Yetunde", "Olumide", "Amina",
)

_LAST_NAMES: tuple[str, ...] = (
    "Okafor", "Adeyemi", "Balogun", "Eze", "Okonkwo",
    "Musa", "Adebayo", "Nwosu", "Bello", "Olawale",
)

_PHONE_PREFIXES: tuple[str, ...] = (
    "803", "806", "810", "813", "814", "816", "903", "905",
)


def _pick(rng: random.Random | None, seq: tuple[str, ...]) -> str:
    if rng is not None:
        return rng.choice(seq)
    return secrets.choice(seq)


def _password(rng: random.Random | None, length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    if rng is not None:
        return "".join(rng.choice(alphabet) for _ in range(length))
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_trial_identity(service: str, *,
                            rng: random.Random | None = None) -> dict[str, Any]:
    """Mint one synthetic identity for ``service``.

    Takes only the service name. Every field is generated from the
    synthetic pools above — names, a disposable-domain email, a
    synthetic-shaped +234 phone, a random password. Deterministic
    when ``rng`` is seeded (tests); secrets-based otherwise.
    """
    service = str(service or "").strip().lower()
    if not service:
        raise ValueError("generate_trial_identity needs a service name")
    first = _pick(rng, _FIRST_NAMES)
    last = _pick(rng, _LAST_NAMES)
    digits = (rng.randint(10, 99) if rng is not None
              else secrets.randbelow(90) + 10)
    username = f"{first.lower()}{last.lower()}{digits}"
    domain = _pick(rng, SYNTHETIC_DOMAINS)
    prefix = _pick(rng, _PHONE_PREFIXES)
    if rng is not None:
        mid = f"{rng.randint(0, 9999):04d}"
        tail = f"{rng.randint(0, 9999):04d}"
    else:
        mid = f"{secrets.randbelow(10000):04d}"
        tail = f"{secrets.randbelow(10000):04d}"
    return {
        "service": service,
        "first_name": first,
        "last_name": last,
        "username": username,
        "email": f"{username}@{domain}",
        "phone": f"+234 {prefix} {mid} {tail}",
        "password": _password(rng),
        "disposable": True,
        "created_at": time.time(),
    }


class TrialIdentityBinder:
    """Bind / fetch trial identities, one per service, in a TrialVault.

    Storage shape: one TrialVault entry per service, keyed by the
    service name. The login is an opaque ``trial-<service>-<hex>``
    handle; the full identity JSON is the sealed secret, so the
    email/phone/password never sit in plaintext in the vault file.
    """

    def __init__(self, home: str | Any) -> None:
        self.vault = TrialVault(home)

    def bind(self, service: str, *,
             rng: random.Random | None = None) -> dict[str, Any]:
        """Return the service's identity — existing, or newly minted.

        Never mints a second identity for a service that already has
        one: the bound identity is the service's identity, full stop.
        """
        service = str(service or "").strip().lower()
        if not service:
            raise ValueError("bind needs a service name")
        try:
            return self.identity_for(service)
        except KeyError:
            pass
        identity = generate_trial_identity(service, rng=rng)
        handle = f"trial-{service}-{secrets.token_hex(4)}"
        self.vault.store(service, handle, json.dumps(identity),
                          note="trial identity — synthetic, disposable")
        _log.info("bound trial identity for %s", service)
        return identity

    def identity_for(self, service: str) -> dict[str, Any]:
        """The identity bound to ``service`` — and to no other service.

        Raises :class:`KeyError` when nothing is bound, and
        :class:`ValueError` if the stored record names a different
        service (tampered vault) — cross-service reuse is refused,
        not coerced.
        """
        service = str(service or "").strip().lower()
        if not service:
            raise ValueError("identity_for needs a service name")
        entry = self.vault.get(service)
        if entry is None or not entry.get("secret"):
            raise KeyError(f"no trial identity bound for {service!r}")
        try:
            identity = json.loads(entry["secret"])
        except ValueError as exc:
            raise ValueError(
                f"trial identity for {service!r} is unreadable") from exc
        if not isinstance(identity, dict) or \
                identity.get("service") != service:
            raise ValueError(
                f"trial identity stored under {service!r} names a "
                "different service — cross-service reuse refused")
        return identity

    def has(self, service: str) -> bool:
        """True when ``service`` already has a bound identity."""
        try:
            self.identity_for(service)
            return True
        except (KeyError, ValueError):
            return False
