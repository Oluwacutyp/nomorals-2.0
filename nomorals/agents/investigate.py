"""InvestigateAgent — one entry point for ANY artifact.

Hand it a hash digest, a JWT, a cookie header, a URL, a base64 blob, or a
file path, and it runs the full pipeline in a single pass:

    classify → decode (Universal Decoder) → crack (known-hash chain →
    bounded live attack) → OSINT (cookies/JWTs → person + domain entities)
    → knowledge graph (every finding curated)

Every stage is fail-soft and the whole run is summarized in one report the
main AI or a human can read: what it was, what decoded, what cracked, whose
identity it touches, and where each finding now lives (graph node, report
id, learned hash).

    from nomorals.agents.investigate import InvestigateAgent
    rep = InvestigateAgent(context).run("<artifact>")

Registered as the ``investigate`` tool and ``nm investigate``.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.policy import Capability

__all__ = ["InvestigateAgent", "classify_artifact", "register"]

_DIGEST_RE = re.compile(
    r"^[a-f0-9]{32}$|^[a-f0-9]{40}$|^[a-f0-9]{64}$|^[a-f0-9]{96}$|"
    r"^[a-f0-9]{128}$|^[a-z0-9+/\-]{27,32}={0,2}$|^\$2[aby]\$\d{2,2}\$"
    r"^[a-zA-Z0-9./]{53}$|^\$argon2(?:[di])?\$")
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")
_URL_RE = re.compile(r"^https?://\S+", re.I)
_COOKIE_RE = re.compile(r"[\w.\-]+=.*;\s*[\w.\-]+=.*|[\w.\-]+=.*;\s*Domain=")


def classify_artifact(artifact: str) -> str:
    """What is this thing?  digest | jwt | url | cookie | text."""
    a = (artifact or "").strip()
    if _DIGEST_RE.match(a):
        return "digest"
    if _JWT_RE.search(a):
        return "jwt"
    if _URL_RE.match(a):
        return "url"
    if _COOKIE_RE.search(a) and "=" in a:
        return "cookie"
    return "text"


class InvestigateAgent:
    """The single-pass artifact investigator (decode → crack → OSINT → KG)."""

    role = "investigate"

    def __init__(self, context: Any, name: str = "investigate") -> None:
        self.context = context
        self.name = name

    def run(self, artifact: str, *, file: bool = False,
            algo: str = "", charset: str = "",
            min_len: int = 1, max_len: int = 6,
            max_candidates: int = 300_000,
            source: str = "") -> dict[str, Any]:
        db = getattr(self.context, "db", None)
        from ..core.decoder import (analyze, identify_hash,
                                    known_hash_lookup,
                                    known_hash_lookup_chained,
                                    learn_hash, save_report)

        art = (artifact or "").strip()
        if file:
            from ..tools.filesystem import safe_path

            p = safe_path(self.context, art, must_exist=True)
            raw = p.read_bytes()[:2_000_000]
            art = raw.decode("utf-8", "ignore")
            file_note = str(p)
        else:
            file_note = ""
        kind = "file" if file else classify_artifact(art)
        if not art:
            return {"ok": False, "error": "nothing to investigate"}

        src = source or f"investigate:{kind}"
        out: dict[str, Any] = {
            "ok": True, "kind": kind, "artifact_head": art[:120],
            "file": file_note, "source": src,
            "steps": [], "cracked": {}, "osint": None,
            "kg": None, "report_id": None,
        }

        # 1 ── decode ───────────────────────────────────────────────────────
        report = analyze(art, max_depth=3)
        best = report.best or {}
        chain = (best.get("chain") or []) if isinstance(best, dict) else []
        out["decode"] = {
            "best": chain,
            "confidence": best.get("confidence") if isinstance(best, dict)
            else None,
            "note": best.get("note") if isinstance(best, dict) else "",
            "hits": len(report.hits or []),
            "cookies": len(report.cookies or []),
            "jwt": bool(report.jwt),
            "tokens": len(report.tokens or []),
        }
        out["steps"].append(f"decoded ({','.join(map(str, chain)) or 'no hit'})")

        # 2 ── crack: known-hash chain first, then bounded live attack ─────
        cracked: dict[str, str] = {}
        hinfo = report.hash
        digest = ""
        if hinfo and hinfo.get("digest"):
            digest = str(hinfo["digest"])
        elif re.fullmatch(r"[a-f0-9]{32,128}", art):
            digest = art
        if digest:
            hit = known_hash_lookup_chained(db, digest) if db is not None \
                else None
            if hit is None:
                hit = known_hash_lookup_chained(None, digest)
            if hit is not None:
                cracked[digest] = hit["plaintext"]
                out["crack"] = {"via": "known-hash-chain",
                                "known_source": (
                                    "built-in" if
                                    known_hash_lookup(digest) else
                                    "learned")}
            else:
                cands = identify_hash(digest)
                if cands:
                    try:
                        from ..tools.decoder import auto_attack

                        res = auto_attack(
                            digest, algo=algo, charset=charset, db=db,
                            min_len=min_len, max_len=max_len,
                            max_candidates=max_candidates)
                        if res.get("cracked"):
                            cracked[digest] = res["plaintext"]
                            out["crack"] = {
                                "via": res.get("via"),
                                "known_source": res.get("known_source"),
                                "tested": res.get("tested"),
                                "learned": bool(res.get("learned")),
                            }
                        else:
                            out["crack"] = {
                                "via": "live-attack",
                                "cracked": False,
                                "candidates": cands[:4],
                                "note": res.get("note", ""),
                            }
                    except Exception as exc:  # noqa: BLE001
                        out["crack"] = {"via": "live-attack",
                                        "cracked": False,
                                        "error": str(exc)[:200]}
            for d, plain in cracked.items():
                if db is not None:
                    learn_hash(db, d, plain,
                               algorithm=out.get("crack", {}).get("algorithm",
                                                                 ""),
                               source=src)
            out["cracked"] = cracked
            out["steps"].append(
                f"cracked → {cracked.get(digest)!r}" if cracked
                else "not cracked (known chain + bounded live attack)")
        else:
            # not a digest — still learn nothing, but record the branch
            out["steps"].append("no hash digest in artifact (skip crack)")

        # 3 ── OSINT: cookies/JWTs → person + domain entities ───────────────
        try:
            from ..agents.osint_graph import IdentityGraph

            stats = IdentityGraph(self.context).ingest_decoder_findings(
                report, source=src)
            out["osint"] = stats
            out["steps"].append(
                f"OSINT: {stats['persons_found']} person(s), "
                f"{stats['domains_found']} domain(s) → identity graph")
        except Exception as exc:  # noqa: BLE001
            out["osint"] = {"error": str(exc)[:200]}

        # 4 ── knowledge graph: curate the decoded content ──────────────────
        try:
            from ..agents.kg import KnowledgeGraph

            blob_parts = []
            out_best = (best.get("output")
                        if isinstance(best, dict) else None)
            if isinstance(out_best, str):
                blob_parts.append(out_best)
            elif isinstance(out_best, dict):
                import json as _json
                blob_parts.append(_json.dumps(out_best)[:20_000])
            for tok in (report.tokens or [])[:20]:
                t = tok.get("token") if isinstance(tok, dict) else tok
                if isinstance(t, str):
                    blob_parts.append(t)
            if cracked:
                blob_parts.append("digest cracked, plaintext: "
                                  + " | ".join(cracked.values()))
            blob = "\n".join(p for p in blob_parts if p)
            if blob.strip() and db is not None:
                kg_res = KnowledgeGraph(db).curate_from_text(
                    blob[:200_000], source=src, max_items=40)
                out["kg"] = {"added_nodes": kg_res.get("added_nodes", 0),
                             "added_links": kg_res.get("added_links", 0)}
                out["steps"].append(
                    f"KG: +{kg_res.get('added_nodes', 0)} nodes, "
                    f"+{kg_res.get('added_links', 0)} links")
        except Exception as exc:  # noqa: BLE001
            out["kg"] = {"error": str(exc)[:200]}

        # 5 ── archive the report ───────────────────────────────────────────
        rid = save_report(db, report, source=src, kind=kind,
                          input_text=art[:500])
        if rid:
            out["report_id"] = rid
        return out


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "investigate",
        description=(
            "One-pass artifact investigator: hand it a hash digest, JWT, "
            "cookie header, URL, base64 blob, or file path — it decodes "
            "(Universal Decoder), cracks any digest (known-hash chain "
            "first, then bounded live attack with the bundled corpus), "
            "feeds cookies/JWTs into the OSINT identity graph "
            "(person + domain entities), and curates everything into the "
            "knowledge graph. Returns the full report: what it was, what "
            "decoded, what cracked, whose identities it touches, and the "
            "archived report id."
        ),
        capability=Capability.FS_READ,
        parameters={
            "artifact": "str — the hash/JWT/cookie/URL/blob to investigate",
            "file": "bool (false) — treat artifact as a workspace file path",
            "algo": "str — crack algorithm hint (md5|sha1|sha256|sha512|ntlm)",
            "charset": "str — brute charset override (e.g. 0123456789)",
            "min_len": "int (1)", "max_len": "int (6)",
            "max_candidates": "int (300000) — live-attack budget",
        },
    )
    def investigate(artifact: str = "", *, file: bool = False,
                    algo: str = "", charset: str = "",
                    min_len: int = 1, max_len: int = 6,
                    max_candidates: int = 300_000) -> dict[str, Any]:
        return InvestigateAgent(context).run(
            artifact, file=file, algo=algo, charset=charset,
            min_len=min_len, max_len=max_len,
            max_candidates=max_candidates, source="investigate-tool")
