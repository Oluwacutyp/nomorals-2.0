"""OSINT god tier: persistent identity correlation + campaign automation.

The collection layer (tools/osint.py, tools/osint_people.py) answers
single-source questions.  This module is what makes OSINT *OSINT*:

1. **Entity extraction** — real regex extraction from raw text: emails,
   phones (E.164 + local forms), domains, IPs, @handles, and capitalized
   name sequences.  Every entity is normalized so the same identity
   extracted as ``Oluwaseun.adebayo@gmail.com`` and
   ``OLUWASEUN ADEBAYO @GMAIL.COM`` is one node.

2. **Persistent identity graph** (kv_store) — nodes with aliases,
   per-source provenance and confidence; two edge classes:
   * **alias** — deterministic merge (same identity, different spellings;
     manual ``merge`` or name-variant heuristics marked as such)
   * **associated** — co-occurrence evidence with weight + source, so
     clusters carry a *confidence* instead of a boolean.

3. **Union-find clustering** — identity clusters ranked by internal
   edge weight and source diversity; ``node(v)`` gives the full
   neighborhood (depth 2) with per-edge provenance.

4. **Campaign runner** — the automation: hand it a seed (phone / email /
   username / domain / IP), it walks the web of the investigation in
   dependency order (worklist BFS): each investigated entity's output is
   ingested into the graph, every newly discovered entity joins the
   worklist, until exhaustion or budget.  Source functions are
   injectable — production wires the real tools, tests wire fakes.

5. **Timeline** — every graph event carries a timestamp; the dossier
   includes the ordered trail of how the web was built.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = [
    "Entity",
    "EntityExtractor",
    "IdentityGraph",
    "CampaignRunner",
    "name_variants",
    "register",
]

_GRAPH_KEY = "osint.graph"
_MAX_NODES = 5000
_MAX_EDGES = 20000


# ── entity extraction ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entity:
    kind: str    # email | phone | domain | ip | username | name | location
    value: str
    line: int = 0


_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:(?<=\D)|^)(\+?\d{1,3}[\s.\-]?)?\(?\d{2,4}\)?[\s.\-]?\d{3,4}"
    r"(?:[\s.\-]?\d{3,4}){1,2}(?=\D|$)")
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE)
_USERNAME_RE = re.compile(
    r"(?:^|[\s>\"'(,])@([a-zA-Z0-9_\-]{2,60})(?=\s|[,.;:!?\"')]|$)")
_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,3})\b")
_LOCATION_STOP = {
    "The", "This", "That", "With", "From", "After", "Before", "Please",
    "Hello", "Dear", "Subject", "Received", "Sent", "Date", "From", "To",
    "Cc", "Bcc", "Mailing", "List", "University", "City", "House",
}
#: common Nigerian state/city tokens for location tagging
_NG_LOCATIONS = {
    "lagos", "abeokuta", "ibadan", "iloje", "akure", "osogbo", "oyo",
    "abuja", "kaduna", "kano", "jos", "port", "harcourt", "enugu",
    "onitsha", "owerrri", "benin", "warri", "uglandoji", "calabar",
    "yola", "sokoto", "bauchi", "makurdi", "abakaliki", "ibom",
    "adebayo", "osun", "oye",  # state names seen in addresses
}


def _norm_email(v: str) -> str:
    return v.strip().lower()


def _norm_phone(v: str) -> str:
    digits = re.sub(r"\D", "", v)
    if digits.startswith("0") and len(digits) == 11:      # Nigerian local 0xxx
        digits = "234" + digits[1:]
    elif digits.startswith("234"):
        pass
    elif len(digits) == 10:
        digits = "234" + digits
    return digits


def _norm_domain(v: str) -> str:
    return v.strip().lower().rstrip(".")


def _norm_name(v: str) -> str:
    v = unicodedata.normalize("NFKD", v)
    v = "".join(c for c in v if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", v).strip().lower()


def _norm_username(v: str) -> str:
    return v.strip().lower()


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "email": _norm_email,
    "phone": _norm_phone,
    "domain": _norm_domain,
    "ip": lambda v: v.strip(),
    "username": _norm_username,
    "name": _norm_name,
    "location": lambda v: v.strip().lower(),
}


class EntityExtractor:
    """Extracts normalized entities from raw text, line by line."""

    def extract(self, text: str) -> list[Entity]:
        out: list[Entity] = []
        seen: set[tuple[str, str]] = set()

        def add(kind: str, raw: str, line: int) -> None:
            norm = _NORMALIZERS[kind](raw)
            if not norm or (kind, norm) in seen:
                return
            if kind == "name" and len(norm.split()) < 2:
                return
            seen.add((kind, norm))
            out.append(Entity(kind=kind, value=norm, line=line))

        for i, line in enumerate(text.splitlines(), 1):
            for m in _EMAIL_RE.finditer(line):
                add("email", m.group(0), i)
            for m in _IP_RE.finditer(line):
                add("ip", m.group(0), i)
            # phones: skip matches that are substrings of emails/domains
            taken = {m.span() for m in _EMAIL_RE.finditer(line)}
            for m in _PHONE_RE.finditer(line):
                if any(not (m.end() <= s or m.start() >= e)
                       for s, e in taken):
                    continue
                digits = re.sub(r"\D", "", m.group(0))
                if 9 <= len(digits) <= 15 and not m.group(0).isdigit():
                    pass
                if 9 <= len(digits) <= 15:
                    add("phone", m.group(0), i)
            for m in _DOMAIN_RE.finditer(line):
                dom = m.group(0)
                if _IP_RE.fullmatch(dom):
                    continue  # it's an address, not a host
                if "@" in line[max(0, m.start() - 60):m.start() + len(dom)]:
                    continue  # part of an email
                tld = dom.rsplit(".", 1)[-1].lower()
                if tld in {"png", "jpg", "jpeg", "gif", "webp", "mp4",
                           "mp3", "pdf", "txt", "csv", "json", "html",
                           "zip", "wav", "m4a", "py", "js", "css"}:
                    continue  # a file, not a host
                add("domain", dom, i)
            for m in _USERNAME_RE.finditer(line):
                add("username", m.group(1), i)
            for m in _NAME_RE.finditer(line):
                words = m.group(1)
                if words.split()[0].capitalize() in _LOCATION_STOP:
                    continue
                add("name", words, i)
            low = line.lower()
            for tok in _NG_LOCATIONS:
                if re.search(rf"\b{re.escape(tok)}\b", low):
                    add("location", tok, i)
                    break
        return out


def name_variants(name: str) -> list[str]:
    """Normalized variants of a person's name (for alias matching).

    Deterministic, documented heuristics: full, no-spaces, initials,
    last-first swap.  These generate *proposed* associations — never
    automatic identity claims."""
    norm = _norm_name(name)
    words = norm.split()
    if not words:
        return []
    out = {norm}
    out.add("".join(words))
    if len(words) >= 2:
        initials = " ".join(w[0] for w in words) + "."
        out.add(initials)
        out.add(" ".join(w[0] for w in words))
        out.add(f"{words[-1]} {words[0]}")
        if len(words) > 2:
            out.add(f"{words[0]} {words[-1]}")
    return sorted(v for v in out if v)


# ── identity graph ───────────────────────────────────────────────────────────


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra if ra > rb else rb] = rb if ra > rb else ra


class IdentityGraph:
    """Persistent (kv_store) identity correlation graph."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        db = getattr(self.context, "db", None)
        if db is None:
            return {"nodes": {}, "edges": [], "events": []}
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = ?",
                               (_GRAPH_KEY,))
            if row:
                data = json.loads(row["value"])
                data.setdefault("nodes", {})
                data.setdefault("edges", [])
                data.setdefault("events", [])
                return data
        except Exception:  # noqa: BLE001
            pass
        return {"nodes": {}, "edges": [], "events": []}

    def save(self) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO kv_store (key, value, kind, updated_at) "
                    "VALUES (?, ?, 'json', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at",
                    (_GRAPH_KEY,
                     json.dumps(self.data, default=str), time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not persist identity graph: %s", exc)

    # -- mutation -------------------------------------------------------------
    @staticmethod
    def _key(kind: str, value: str) -> str:
        return f"{kind}:{value}"

    def _upsert_node(self, kind: str, value: str, source: str,
                     ts: float) -> None:
        nodes = self.data["nodes"]
        key = self._key(kind, value)
        if len(nodes) >= _MAX_NODES:
            return
        node = nodes.get(key)
        if node is None:
            nodes[key] = {
                "kind": kind, "value": value, "aliases": [], "sources": {},
                "confidence": 1.0 if kind in {"email", "phone", "ip"} else 0.6,
                "first_seen": ts, "last_seen": ts,
            }
        node = nodes[key]
        node["last_seen"] = ts
        node["sources"][source] = node["sources"].get(source, ts)

    def ingest(self, text: str, source: str,
               ts: float | None = None) -> list[Entity]:
        """Extract entities from raw text and record co-occurrence edges."""
        ts = float(ts or time.time())
        entities = EntityExtractor().extract(text)
        for e in entities:
            self._upsert_node(e.kind, e.value, source, ts)
        # co-occurrence: entities seen in the same document
        keys = [self._key(e.kind, e.value) for e in entities]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i] != keys[j]:
                    self._associate(keys[i], keys[j], source, 1.0,
                                    f"co-occurrence in {source}", ts)
        self._event(ts, f"ingest:{source}",
                    f"{len(entities)} entities from {source}")
        self.save()
        return entities

    def _event(self, ts: float, kind: str, detail: str) -> None:
        events = self.data["events"]
        events.append({"ts": ts, "kind": kind, "detail": detail[:300]})
        if len(events) > 2000:
            del events[:len(events) - 2000]

    def _associate(self, a: str, b: str, source: str, weight: float,
                   note: str, ts: float) -> None:
        edges = self.data["edges"]
        if len(edges) >= _MAX_EDGES:
            return
        for edge in edges:
            if edge["class"] == "associated" and {edge["a"], edge["b"]} == {a, b}:
                edge["weight"] = min(1.0, float(edge["weight"]) + weight * 0.5)
                edge["sources"].setdefault(source, ts)
                edge["ts"] = ts
                return
        edges.append({
            "class": "associated", "a": a, "b": b, "weight": weight,
            "sources": {source: ts}, "ts": ts, "note": note[:160],
        })

    def merge(self, a: str, b: str, source: str = "manual",
              ts: float | None = None) -> dict[str, Any]:
        """Deterministic alias merge: b becomes an alias of a (the more
        common/shorter value becomes representative when possible)."""
        ts = float(ts or time.time())
        nodes = self.data["nodes"]
        ka = self._node_key(a)
        kb = self._node_key(b)
        if ka is None or kb is None:
            missing = a if ka is None else b
            raise ToolError(f"entity not in graph: {missing}")
        if ka == kb:
            return {"merged": False, "note": "already the same node"}
        # representative: prefer emails/phones over names for stability
        rank = {"email": 0, "phone": 0, "ip": 1, "domain": 1,
                "username": 2, "location": 3, "name": 4}
        na, nb = nodes[ka], nodes[kb]
        if rank.get(nb["kind"], 5) < rank.get(na["kind"], 5):
            ka, kb, na, nb = kb, ka, nb, na
        rep_value, alias_value = na["value"], nb["value"]
        na["aliases"].append(kb)
        for src, t in nb["sources"].items():
            na["sources"].setdefault(src, t)
        na["confidence"] = max(na["confidence"], nb["confidence"])
        self.data["edges"].append({
            "class": "alias", "a": ka, "b": kb, "weight": 1.0,
            "sources": {source: ts}, "ts": ts,
            "note": "alias merge" if source == "manual" else source[:160],
        })
        del nodes[kb]
        # rehome edges that referenced the absorbed node (either direction);
        # alias edges keep their tombstone ref for provenance but are skipped
        # by clustering (see clusters()).
        for edge in self.data["edges"]:
            if edge.get("class") == "associated":
                if edge.get("a") == kb:
                    edge["a"] = ka
                if edge.get("b") == kb:
                    edge["b"] = ka
        self._event(ts, "merge", f"{kb} → {ka}")
        self.save()
        return {"merged": True, "representative": rep_value,
                "alias": alias_value}

    def _node_key(self, ref: str) -> str | None:
        """Resolve 'kind:value' or a bare value against the node map."""
        nodes = self.data["nodes"]
        if ":" in ref:
            kind, value = ref.split(":", 1)
            key = self._key(kind, value)
            if key in nodes:
                return key
            return None
        # bare ref: exact value match (any kind, preferring strong kinds)
        matches = []
        for key, node in nodes.items():
            if node.get("value", "") == ref:
                matches.append(key)
            elif node.get("kind") in {"name", "location"} and \
                    _norm_name(node.get("value", "")) == _norm_name(ref):
                matches.append(key)
        if matches:
            rank = {"email": 0, "phone": 0, "ip": 1, "domain": 1,
                    "username": 2, "location": 3, "name": 4}
            matches.sort(key=lambda k: (rank.get(nodes[k]["kind"], 5), k))
            return matches[0]
        # alias lookup
        for key, node in nodes.items():
            if ref in node.get("aliases", []):
                return key
        return None

    # -- queries -----------------------------------------------------------------
    def clusters(self, *, min_size: int = 2) -> list[dict[str, Any]]:
        """Union-find over alias + associated edges, ranked by confidence."""
        nodes = self.data["nodes"]
        if not nodes:
            return []
        uf = _UnionFind(list(nodes))
        for edge in self.data["edges"]:
            a, b = edge.get("a"), edge.get("b")
            if a == b:
                continue  # self-loop (alias tombstone provenance)
            if a in uf.parent and b in uf.parent:
                uf.union(a, b)
        groups: dict[str, list[str]] = {}
        for node in nodes:
            groups.setdefault(uf.find(node), []).append(node)
        out = []
        for members in groups.values():
            if len(members) < min_size:
                continue
            member_set = set(members)
            internal = [e for e in self.data["edges"]
                        if e.get("a") in member_set and e.get("b") in member_set]
            weight = sum(float(e.get("weight", 0)) for e in internal)
            sources = set()
            for e in internal:
                sources.update((e.get("sources") or {}).keys())
            confidence = min(1.0, 0.5 + 0.1 * len(sources)
                             + 0.05 * min(len(internal), 10))
            entities = []
            for m in members:
                node = nodes[m]
                entities.append({
                    "kind": node["kind"], "value": node["value"],
                    "aliases": list(node.get("aliases", [])),
                    "sources": sorted(node.get("sources", {}).keys())[:8],
                })
            entities.sort(key=lambda e: (e["kind"], e["value"]))
            out.append({
                "size": len(members),
                "confidence": round(confidence, 2),
                "evidence_edges": len(internal),
                "evidence_sources": len(sources),
                "entities": entities,
            })
        out.sort(key=lambda c: (-c["size"], -c["confidence"]))
        return out

    def node(self, ref: str, *, depth: int = 2) -> dict[str, Any]:
        key = self._node_key(ref)
        if key is None:
            raise ToolError(f"entity not in graph: {ref}")
        nodes = self.data["nodes"]
        node = nodes[key]
        neighbors: list[dict[str, Any]] = []
        frontier = {key}
        for _ in range(max(1, min(int(depth), 3))):
            nxt: set[str] = set()
            for edge in self.data["edges"]:
                for a, b in ((edge.get("a"), edge.get("b")),
                             (edge.get("b"), edge.get("a"))):
                    if a in frontier and b in nodes and b not in frontier:
                        neighbors.append({
                            "kind": nodes[b]["kind"],
                            "value": nodes[b]["value"],
                            "edge": edge.get("class", "associated"),
                            "weight": round(float(edge.get("weight", 0)), 2),
                            "sources": sorted(
                                (edge.get("sources") or {}).keys())[:6],
                            "note": edge.get("note", ""),
                        })
                        nxt.add(b)
                        frontier.add(b)
            frontier = nxt
        seen: set[str] = set()
        unique = []
        for n in neighbors:
            ident = (n["kind"], n["value"])
            if ident not in seen:
                seen.add(ident)
                unique.append(n)
        return {
            "kind": node["kind"], "value": node["value"],
            "aliases": list(node.get("aliases", [])),
            "confidence": node.get("confidence"),
            "sources": node.get("sources", {}),
            "first_seen": node.get("first_seen"),
            "last_seen": node.get("last_seen"),
            "links": unique[:60],
        }

    def timeline(self, *, limit: int = 50) -> list[dict[str, Any]]:
        events = sorted(self.data["events"], key=lambda e: e.get("ts", 0))
        return events[-max(1, int(limit)):]

    def stats(self) -> dict[str, Any]:
        nodes = self.data["nodes"]
        by_kind: dict[str, int] = {}
        for n in nodes.values():
            by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
        return {
            "nodes": len(nodes),
            "edges": len(self.data["edges"]),
            "by_kind": by_kind,
            "clusters": len(self.clusters()),
            "events": len(self.data["events"]),
        }

    def clear(self) -> None:
        self.data = {"nodes": {}, "edges": [], "events": []}
        self.save()

    # ── decoder findings consumer ─────────────────────────────────────────
    #: JWT claims → identity kind (value must be a string/number)
    _JWT_PERSON_CLAIMS = {
        "email": "email", "mail": "email", "email_address": "email",
        "name": "name", "full_name": "name", "given_name": "name",
        "preferred_username": "username", "username": "username",
        "user_name": "username", "handle": "username",
        "screen_name": "username",
    }

    @staticmethod
    def _domain_of(v: str) -> str:
        v = str(v or "").strip()
        if not v:
            return ""
        if "://" in v:
            v = v.split("://", 1)[1]
        v = v.split("/", 1)[0].split(":", 1)[0]
        v = v.lstrip("@").strip().lower().rstrip(".")
        return v if "." in v else ""

    def ingest_decoder_findings(self, report: Any,
                                source: str = "decoder",
                                ts: float | None = None) -> dict[str, Any]:
        """Consume a Universal Decoder report (DecodeReport or dict form)
        and map its cookies / JWTs onto person + domain entities.

        * JWT claims (top-level, or nested in a cookie's value) → person
          nodes (email / name / username), associated with the issuer's
          domain
        * cookie domains → domain nodes
        * best decode output + raw tokens → the normal entity extractor

        Every person found is edge-linked to every domain found (weight
        0.9), so a cluster = "this person ↔ these services" survives into
        the ranking layer.
        """
        ts = float(ts or time.time())
        src = str(source or "decoder").strip() or "decoder"
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except (ValueError, TypeError):
                report = {}
        if not isinstance(report, dict):
            report = getattr(report, "__dict__", {}) or {}

        best = report.get("best")
        if not isinstance(best, dict):
            best = getattr(best, "__dict__", {}) or {}
        jwt = report.get("jwt")
        if not isinstance(jwt, dict):
            jwt = {}
        cookies = report.get("cookies") or []
        if not isinstance(cookies, list):
            cookies = []
        tokens = report.get("tokens") or []
        if not isinstance(tokens, list):
            tokens = []

        person_keys: list[str] = []
        domain_keys: list[str] = []

        def add_person(kind: str, value: Any) -> None:
            value = str(value or "").strip()
            if len(value) < 2:
                return
            norm = _NORMALIZERS.get(kind, lambda v: v)(value)
            if not norm:
                return
            key = self._key(kind, norm)
            self._upsert_node(kind, norm, src, ts)
            if key not in person_keys:
                person_keys.append(key)

        def add_domain(value: Any) -> None:
            dom = self._domain_of(str(value or ""))
            if not dom:
                return
            norm = _norm_domain(dom)
            key = self._key("domain", norm)
            self._upsert_node("domain", norm, src, ts)
            if key not in domain_keys:
                domain_keys.append(key)

        def consume_jwt(payload: Any, issuer_hint: str = "") -> None:
            if not isinstance(payload, dict):
                return
            # the decoder wraps JWTs as {"header", "payload": {claims}, …}
            inner = payload.get("payload")
            if isinstance(inner, dict):
                payload = {k: v for k, v in payload.items()
                           if k not in ("header", "payload", "signed",
                                        "warnings")}
                payload.update(inner)
            add_domain(str(issuer_hint or payload.get("iss") or ""))
            for claim, val in payload.items():
                claim_l = str(claim).lower()
                if isinstance(val, (list, tuple)):
                    vals = [str(v) for v in val
                            if isinstance(v, (str, int, float))]
                elif isinstance(val, (str, int, float)):
                    vals = [str(val)]
                else:
                    continue
                kind = self._JWT_PERSON_CLAIMS.get(claim_l)
                for v in vals:
                    if kind:
                        add_person(kind, v)
                    if claim_l in ("iss", "host", "website", "uri",
                                   "origin", "aud"):
                        add_domain(v)
                    if claim_l in ("email", "mail", "email_address"):
                        add_domain(v.split("@")[-1])

        consume_jwt(jwt)
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            cdom = str(cookie.get("domain") or "")
            flag = str(cookie.get("flag") or "")
            if flag.lower().startswith("domain="):
                cdom = cdom or flag.split("=", 1)[1]
            if cdom:
                add_domain(cdom)
            consume_jwt(cookie.get("jwt"), cdom)

        blob_parts: list[str] = []
        out = best.get("output")
        if isinstance(out, str):
            blob_parts.append(out)
        elif isinstance(out, (dict, list)):
            try:
                blob_parts.append(json.dumps(out))
            except (TypeError, ValueError):  # noqa: E103 - unserializable output skipped; other blob parts still used
                pass
        for t in tokens[:25]:
            if isinstance(t, dict):
                t = str(t.get("token") or t.get("value") or "")
            if isinstance(t, str) and 3 <= len(t) <= 500:
                blob_parts.append(t)
        text = "\n".join(p for p in blob_parts if p)
        if text:
            for e in EntityExtractor().extract(text):
                self._upsert_node(e.kind, e.value, src, ts)
                key = self._key(e.kind, e.value)
                if e.kind in ("email", "name", "username"):
                    if key not in person_keys:
                        person_keys.append(key)
                elif e.kind == "domain":
                    if key not in domain_keys:
                        domain_keys.append(key)

        for p in person_keys:
            for d in domain_keys:
                self._associate(p, d, src, 0.9,
                                f"decoder findings via {src}", ts)
        self._event(ts, f"decoder:{src}",
                    f"{len(person_keys)} person + {len(domain_keys)} "
                    f"domain entities from decoder findings")
        self.save()
        return {"source": src, "persons": person_keys,
                "domains": domain_keys,
                "persons_found": len(person_keys),
                "domains_found": len(domain_keys)}


# ── campaign runner ──────────────────────────────────────────────────────────

SourceFn = Callable[[str], str]  # (entity value) -> raw investigation output


class CampaignRunner:
    """Walks the investigation web: seed → investigate → ingest → expand.

    ``sources`` maps entity kind → callable(value) → raw text/JSON.
    Production wires the real OSINT tools; tests inject fakes, which is
    how the whole BFS is verified hermetically.
    """

    def __init__(self, graph: IdentityGraph,
                 sources: dict[str, SourceFn], *,
                 max_steps: int = 40,
                 max_entities: int = 300,
                 time_budget: float = 300.0) -> None:
        self.graph = graph
        self.sources = sources
        self.max_steps = max(1, int(max_steps))
        self.max_entities = max(1, int(max_entities))
        self.time_budget = float(time_budget)

    def run(self, seeds: list[tuple[str, str]],
            *, source_label: str = "campaign") -> dict[str, Any]:
        started = time.time()
        worklist: list[tuple[str, str]] = list(seeds)
        done: set[tuple[str, str]] = set()
        steps = 0
        findings: list[dict[str, Any]] = []
        while worklist:
            if steps >= self.max_steps or len(done) >= self.max_entities \
                    or time.time() - started > self.time_budget:
                break
            kind, value = worklist.pop(0)
            if (kind, value) in done or kind not in self.sources:
                continue
            done.add((kind, value))
            steps += 1
            label = f"{source_label}:{kind}:{value}"
            try:
                output = self.sources[kind](value)
            except Exception as exc:  # noqa: BLE001 - one dead source, on
                findings.append({"kind": kind, "value": value,
                                 "error": str(exc)[:160]})
                continue
            new_entities = self.graph.ingest(str(output or ""), label)
            summary = _flatten_output(output)
            findings.append({"kind": kind, "value": value, "summary": summary})
            for e in new_entities:
                if (e.kind, e.value) not in done \
                        and e.kind in self.sources \
                        and (e.kind, e.value) not in worklist:
                    # don't re-walk the entity we just came from
                    if e.value == value:
                        continue
                    worklist.append((e.kind, e.value))
        return {
            "steps": steps,
            "entities_investigated": len(done),
            "worklist_left": len(worklist),
            "findings": findings,
            "graph": self.graph.stats(),
            "clusters": self.graph.clusters(),
            "timeline": self.graph.timeline(limit=30),
            "seconds": round(time.time() - started, 1),
            "stopped": "budget" if (steps >= self.max_steps
                                    or len(done) >= self.max_entities
                                    or time.time() - started > self.time_budget)
                       else "exhausted",
        }


def _flatten_output(output: Any) -> dict[str, Any]:
    """A compact, JSON-safe summary of a source's output."""
    if isinstance(output, dict):
        return {k: v for k, v in list(output.items())[:12]
                if isinstance(v, (str, int, float, bool, list))}
    if isinstance(output, str):
        return {"chars": len(output), "head": output[:200]}
    return {"value": str(output)[:200]}


def _production_sources(context: Any) -> dict[str, SourceFn]:
    """Wire the real OSINT tools as campaign sources."""

    def _call(tool: str, **params: str) -> str:
        outcome = context.tools.call(tool, **params)
        if not outcome.ok:
            return json.dumps({"error": str(
                getattr(outcome.error, "message", outcome.error))[:200]})
        return json.dumps(outcome.value, default=str)

    return {
        "email": lambda v: _call("email_investigate", email=v),
        "phone": lambda v: _call("phone_investigate", phone=v),
        "username": lambda v: _call("username_check", username=v),
        "domain": lambda v: _call("osint_domain", domain=v),
        "ip": lambda v: _call("osint_ip", ip=v),
    }


def _parse_seeds(raw: str) -> list[tuple[str, str]]:
    """'phone:+234803..., email:x@y.com, domain:z.ng' or bare values with
    auto-detection."""
    seeds: list[tuple[str, str]] = []
    for part in re.split(r"[,;]\s*|\n", raw or ""):
        part = part.strip()
        if not part:
            continue
        if ":" in part and part.split(":", 1)[0].lower() in \
                {"email", "phone", "username", "domain", "ip"}:
            kind, value = part.split(":", 1)
            seeds.append((kind.lower(), _NORMALIZERS[kind.lower()](value)))
            continue
        extracted = EntityExtractor().extract(part)
        if extracted:
            for e in extracted[:3]:
                seeds.append((e.kind, e.value))
        elif _norm_phone(part) and re.fullmatch(r"\d{9,15}", _norm_phone(part)):
            seeds.append(("phone", _norm_phone(part)))
    seen: set[tuple[str, str]] = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "osint_graph",
        description=(
            "Persistent identity correlation graph. ingest: extract entities "
            "from raw text into the graph. ingest_decoder: feed a Universal "
            "Decoder report (JSON) — cookies/JWTs become person + domain "
            "entities, auto-linked. clusters: ranked identity clusters with "
            "confidence. node <ref>: full neighborhood + provenance. "
            "merge <a> <b>: deterministic alias merge. timeline / stats / "
            "clear. This is where investigation results compound."
        ),
        capability=Capability.DB_WRITE,
        parameters={
            "action": "str — ingest | ingest_decoder | clusters | node | merge | timeline | stats | clear",
            "text": "str (ingest) — raw text/reports to mine for entities",
            "report": "str (ingest_decoder) — decoder report JSON (from the decoder tool)",
            "source": "str (ingest, optional) — where this text came from",
            "node": "str (node) — entity or kind:value",
            "a": "str (merge) — first entity",
            "b": "str (merge) — second entity",
        },
    )
    def osint_graph(*, action: str = "stats", text: str = "", source: str = "",
                    node: str = "", a: str = "", b: str = "",
                    report: str = "") -> dict[str, Any]:
        graph = IdentityGraph(context)
        action = (action or "stats").strip().lower()
        if action == "ingest":
            entities = graph.ingest(text or "", source or "manual")
            return {"ingested": len(entities),
                    "entities": [e.__dict__ for e in entities[:40]],
                    "stats": graph.stats()}
        if action == "ingest_decoder":
            if not report:
                raise ToolError(
                    "ingest_decoder needs report (the decoder tool's "
                    "report JSON)")
            out = graph.ingest_decoder_findings(report,
                                                source or "decoder")
            return {"ingest_decoder": out, "stats": graph.stats()}
        if action == "clusters":
            return {"clusters": graph.clusters()}
        if action == "node":
            if not node:
                raise ToolError("node action needs a node ref")
            return graph.node(node)
        if action == "merge":
            if not a or not b:
                raise ToolError("merge needs both a and b")
            return graph.merge(a, b)
        if action == "timeline":
            return {"timeline": graph.timeline()}
        if action == "clear":
            graph.clear()
            return {"cleared": True}
        return graph.stats()

    @registry.register(
        "osint_campaign",
        description=(
            "Automated investigation walk: given a seed (phone/email/username/"
            "domain/IP, 'kind:value' or auto-detected), runs the full OSINT "
            "toolkit over the discovered web of the target in dependency "
            "order — every result is ingested into the identity graph and "
            "every newly discovered entity joins the worklist. Returns the "
            "dossier: findings, ranked clusters with confidence, timeline."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "seeds": "str — one or more seeds: 'phone:+234…' | 'email:x@y' | comma list | auto-detected",
            "max_steps": "int (optional, 40) — investigations to run",
            "max_entities": "int (optional, 300)",
            "time_budget": "float (optional, 300) — seconds",
        },
    )
    def osint_campaign(*, seeds: str = "", max_steps: str = "",
                       max_entities: str = "",
                       time_budget: str = "") -> dict[str, Any]:
        parsed = _parse_seeds(seeds)
        if not parsed:
            raise ToolError("campaign needs at least one seed")
        graph = IdentityGraph(context)
        try:
            steps = max(1, min(int(max_steps or 40), 200))
        except ValueError:
            steps = 40
        try:
            max_ent = max(1, min(int(max_entities or 300), 2000))
        except ValueError:
            max_ent = 300
        try:
            budget = max(10.0, min(float(time_budget or 300), 1800))
        except ValueError:
            budget = 300.0
        runner = CampaignRunner(
            graph, _production_sources(context),
            max_steps=steps, max_entities=max_ent, time_budget=budget)
        return runner.run(parsed, source_label="osint_campaign")
