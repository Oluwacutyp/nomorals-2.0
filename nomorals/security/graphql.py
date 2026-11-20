"""GraphQL: introspection, then the abuse introspection enables.

Introspection first because it is free and decisive — a disabled endpoint
is reported as such instead of guessed at. When it answers: sensitive
fields, mutations, and depth/alias amplification.

Alias amplification is the real one: N aliases in one document defeat a
per-request rate limit, which is the most common GraphQL control.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.http import HttpClient
from ..core.policy import Capability
from ..core.logging_setup import get_logger
from .scope import Governor

_log = get_logger(__name__)
__all__ = ["INTROSPECTION", "graphql_probe", "amplify", "register"]

INTROSPECTION = """
query IntrospectionQuery { __schema { queryType { name }
  mutationType { name } types { name kind
    fields { name args { name } type { name kind ofType { name kind } } } } } }
"""

_SENSITIVE = ("password", "hash", "token", "secret", "key", "email",
              "phone", "ssn", "dob", "address", "card", "iban", "internal",
              "role", "admin", "deleted", "ip", "salt")


@dataclass
class GqlResult:
    url: str
    reachable: bool = False
    introspection: bool = False
    types: int = 0
    mutations: list[str] = field(default_factory=list)
    sensitive: list[str] = field(default_factory=list)
    aliases_ok: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "reachable": self.reachable,
                "introspection": self.introspection, "types": self.types,
                "mutations": self.mutations[:20],
                "sensitive": self.sensitive[:30],
                "aliases_ok": self.aliases_ok, "notes": self.notes}


def graphql_probe(url: str, *, governor: Governor,
                  client: Optional[HttpClient] = None,
                  timeout: float = 20.0) -> GqlResult:
    client = client or HttpClient(timeout=timeout)
    result = GqlResult(url=url)
    try:
        governor.spend(url)
        resp = client.post(url, data=json.dumps({"query": INTROSPECTION}),
                           headers={"Content-Type": "application/json"})
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"unreachable: {type(exc).__name__}")
        return result
    result.reachable = resp.ok
    if not resp.ok:
        result.notes.append(f"http {resp.status}")
        return result
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        result.notes.append("not JSON — probably not GraphQL")
        return result
    schema = ((data.get("data") or {}).get("__schema"))
    if not schema:
        result.notes.append("introspection disabled")
        return result
    result.introspection = True
    types = schema.get("types") or []
    result.types = len(types)
    if (schema.get("mutationType") or {}).get("name"):
        result.notes.append("mutations exposed")
    for t in types:
        for f in t.get("fields") or []:
            name = str(f.get("name") or "")
            if any(s in name.lower() for s in _SENSITIVE):
                result.sensitive.append(f"{t.get('name')}.{name}")
    return result


def amplify(url: str, query_type: str = "query", field_name: str = "__typename",
            count: int = 50, *, governor: Governor,
            client: Optional[HttpClient] = None,
            timeout: float = 30.0) -> dict[str, Any]:
    """One document, `count` aliases. Proves per-request limits only."""
    client = client or HttpClient(timeout=timeout)
    body = "{ " + " ".join(f"a{i}:{field_name}" for i in range(count)) + " }"
    started = time.monotonic()
    governor.spend(url, cost=count)
    try:
        resp = client.post(url, data=json.dumps({"query": body}),
                           headers={"Content-Type": "application/json"})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)[:100]}
    elapsed = time.monotonic() - started
    try:
        data = json.loads(resp.text or "{}")
    except ValueError:
        return {"ok": False, "reason": f"http {resp.status}"}
    answered = len((data.get("data") or {}))
    return {"ok": answered >= count // 2, "answered": answered,
            "requested": count, "seconds": round(elapsed, 2),
            "note": "one request, N fields — defeats a per-request limit"
                    if answered else "aliases rejected"}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register("graphql_probe",
                       description="Introspect a GraphQL endpoint and list "
                                   "sensitive fields and mutations.",
                       capability=Capability.NET_OUT,
                       parameters={"url": "str", "scope": "str"})
    def graphql_probe_tool(url: str, scope: str) -> dict[str, Any]:
        from .scope import load_scope
        gov = Governor(load_scope(scope), max_total=10)
        return graphql_probe(url, governor=gov).as_dict()

    @registry.register("graphql_amplify",
                       description="Alias-amplification test against a GraphQL "
                                   "endpoint.",
                       capability=Capability.NET_OUT,
                       parameters={"url": "str", "scope": "str",
                                   "count": "int (optional, 50)"})
    def graphql_amplify_tool(url: str, scope: str,
                             *, count: str = "50") -> dict[str, Any]:
        from .scope import load_scope
        gov = Governor(load_scope(scope), max_total=10_000)
        return amplify(url, count=int(count or 50), governor=gov)
