"""DecoderAgent — the Universal Decoder as a living sub-agent.

Wraps ``nomorals.core.decoder`` with agent orchestration:

* **run** — decode a string or workspace file, and when the winning decode
  is a BINARY (base64→PNG, hexdump→ELF, …) quarantine it in the artifact
  store (D1): inert, provenance-recorded, unverified — the store's
  managed copy is the only copy, and it leaves the store only through
  an explicit owner action (a surface resolving the artifact ref).
* **explain** — a plain-language interpretation of the report.  Model-
  sourced when a live model is answering, rule-based otherwise (never
  invents what the engine didn't find).
* **decoders** — list the engine's registered decoders.

Registered as the ``decoder_agent`` tool; the raw ``decoder`` tool stays
available for one-shot use.  Fully offline — the data never leaves the
machine.

    from nomorals.agents.decoder import DecoderAgent
    agent = DecoderAgent(context=context)
    result = agent.run({"path": "downloads/mystery", "save": True})
    result.output["saved_to"]   # artifact:// URI when a binary was quarantined
    result.output["artifact"]   # the quarantine record (name/kind/hash)
    result.output["limits"]     # the decoder's stated limits (D1)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..llm.brain import brain_for
from ..core.decoder import (DECODERS, analyze, identify_hash,
                            known_hash_lookup)
from ..core.errors import ToolError
from ..core.policy import Capability
from .base import Agent, AgentResult

__all__ = ["DecoderAgent", "register"]

_log = logging.getLogger(__name__)

_MAGIC_EXT = {
    "png": ".png", "jpeg": ".jpg", "gif": ".gif", "pdf": ".pdf",
    "gzip": ".gz", "zip": ".zip", "elf": ".elf", "mp3": ".mp3",
    "mp3-id3": ".mp3", "riff/wav": ".wav", "ogg": ".ogg", "flac": ".flac",
    "webp": ".webp", "bmp": ".bmp", "ole2/doc": ".doc", "sqlite": ".sqlite",
    "bzip2": ".bz2", "xz": ".xz", "7zip": ".7z", "zlib": ".zlib",
    "zstd": ".zst", "pe/dos-executable": ".bin", "script-shebang": ".sh",
    "webm/mkv": ".webm",
}


def _artifact_store_for(context: Any) -> Any | None:
    """The WS1 artifact store over the context (the D1 quarantine home).

    An explicitly attached store wins; otherwise the store is built
    over the context's db exactly as the composition root and the
    partner runtime build it (db + a BlobStore in ``blobs`` beside
    the db file).  None when the context carries no database — the
    caller then persists nothing, and says so on the record.
    """
    for attr in ("artifact_store", "artifacts"):
        store = getattr(context, attr, None)
        if store is not None and hasattr(store, "put"):
            return store
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        from pathlib import Path

        from ..storage.artifacts import ArtifactStore
        from ..storage.blob import BlobStore

        db_path = getattr(db, "path", None)
        blob_dir = (str(Path(db_path).parent / "blobs") if db_path
                    else "data/blobs")
        return ArtifactStore(db, BlobStore(db, blob_dir))
    except Exception as exc:  # noqa: BLE001 — quarantine is best-effort
        _log.debug("decoder artifact store unavailable: %s", exc)
        return None


def _quarantine_record(art: Any, *, name: str,
                       magic: str) -> dict[str, Any]:
    """The record of one quarantined decode: name + kind + hash/size
    facts for surfaces, the artifact id/uri for the decode RECORD's
    linkage (chat surfaces show name/kind, never the raw id)."""
    meta = art.metadata or {}
    return {
        "id": art.id,
        "uri": art.uri,
        "name": meta.get("filename") or name,
        "kind": meta.get("kind") or "decoded-binary",
        "quarantine": meta.get("quarantine") or "decoded-untrusted",
        "magic": magic or meta.get("magic") or "",
        "size": art.size,
        "sha256": art.content_hash,
        "verification_state": art.provenance.verification_state,
    }


def _explain_rules(report: Any) -> str:
    """Rule-based explanation (no model needed)."""
    from ..tools.decoder import _summarize

    lines = [_summarize(report)]
    if report.best is None:
        lines.append("No decoder produced a confident result — the input is "
                     "most likely already plain text or unknown encoding. "
                     "Entropy "
                     f"{report.forensics['entropy']} bits/char"
                     + (" (high = random/binary)"
                        if report.forensics["entropy"] > 5.5 else
                        " (low = structured text)") + ".")
    return " ".join(lines)


class DecoderAgent(Agent):
    """Decode + interpret + persist: the universal decoder, agentic."""

    role = "decoder"
    required_capabilities = (Capability.FS_READ, Capability.FS_WRITE)

    def work(self, task_input: Any) -> Any:
        spec = self._normalize(task_input)
        data: Any = spec.get("data")
        path: str | None = spec.get("path")
        save: bool = bool(spec.get("save", True))

        if path:
            from ..tools.filesystem import safe_path

            target = safe_path(self.context, path, must_exist=True)
            raw = target.read_bytes()[:2_000_000]
            report = analyze(raw, max_depth=spec.get("max_depth", 3))
            report_dict = report.to_dict()
            report_dict["source"] = str(target)
        elif data:
            raw = data
            report = analyze(data, max_depth=spec.get("max_depth", 3))
            report_dict = report.to_dict()
        else:
            raise ToolError("decoder_agent needs data or path")

        artifact = None
        saved_to = None
        best = report.best
        if best and save and isinstance(best["output"], (bytes, bytearray)):
            artifact = self._quarantine_bytes(bytes(best["output"]))
            if artifact is not None:
                saved_to = artifact["uri"]
                report_dict["saved_to"] = saved_to
                report_dict["artifact"] = artifact
            else:
                # No store reachable: the bytes are persisted NOWHERE
                # (never a loose file) and the record says so.
                report_dict["storage"] = (
                    "unavailable — no artifact store is attached, so "
                    "the decoded binary was not persisted anywhere")

        explanation = self._explain(report, spec.get("explain", True))
        self._curate(report_dict, explanation, raw)
        from ..tools.decoder import decoder_limits

        return {"report": report_dict, "saved_to": saved_to,
                "artifact": artifact, "limits": decoder_limits(),
                "explanation": explanation}

    def _curate(self, report: dict[str, Any], explanation: str,
                raw: str | bytes) -> None:
        """Persist findings into the knowledge graph (never fails).

        The *input* is where the interesting artifacts live (digests,
        URLs, tokens, cookie jars, JWTs), so it is curated alongside the
        decode result.
        """
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            from .kg import KnowledgeGraph

            src_text = (raw if isinstance(raw, str)
                        else bytes(raw).decode("utf-8", "ignore"))
            text = "\n".join([
                explanation,
                src_text[:200_000],
                json.dumps({k: v for k, v in report.items()
                            if k in ("best", "hits", "forensics", "hash")},
                           default=str)[:20000],
            ])
            KnowledgeGraph(db).curate_from_text(
                text, source=f"decode:{str(report.get('target', 'stdin'))[:80]}")
        except Exception as exc:  # noqa: BLE001 — memory, not dependency
            _log.debug("decoder kg curation failed: %s", exc)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(task_input: Any) -> dict[str, Any]:
        if isinstance(task_input, str):
            s = task_input.strip()
            if s.startswith("file:"):
                return {"path": s[len("file:"):].strip()}
            return {"data": s}
        if isinstance(task_input, dict):
            return dict(task_input)
        raise ToolError("task_input must be a string or a dict")

    def _quarantine_bytes(self, raw: bytes) -> dict[str, Any] | None:
        """Quarantine decoded binary bytes in the artifact store (D1).

        The bytes land as an inert artifact — provenance-recorded as
        decoder-produced, marked decoded-untrusted, verification
        state ``unverified`` (nothing has scanned them; quarantine
        is storage + provenance, not a safety claim) — and the
        store's managed copy is the ONLY copy: no loose workspace
        file is written, and the bytes leave only when a surface
        explicitly resolves the artifact (an owner action).
        Identical bytes already quarantined by the decoder reuse
        their row (the store's content-hash dedup read).  Returns
        the artifact record, or None when no store is reachable —
        the decode itself never fails for want of a quarantine.
        """
        store = _artifact_store_for(self.context)
        if store is None:
            return None
        magic = ""
        try:
            from ..core.decoder import identify_magic

            magic = identify_magic(raw)
        except Exception:  # noqa: BLE001
            magic = ""
        ext = _MAGIC_EXT.get(magic, ".bin")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"decoded-{stamp}{ext}"
        try:
            import hashlib

            from ..storage.artifacts import Provenance

            sha = hashlib.sha256(raw).hexdigest()
            for art in store.find_by_content_hash(sha):
                if (art.metadata or {}).get("quarantine") == \
                        "decoded-untrusted":
                    return _quarantine_record(art, name=name, magic=magic)
            art = store.put(
                raw, type="file", creator="decoder",
                mime="application/octet-stream",
                metadata={
                    "filename": name,
                    "kind": "decoded-binary",
                    "quarantine": "decoded-untrusted",
                    "inert": True,
                    "release": "owner-action-only",
                    "magic": magic,
                    "sha256": sha,
                },
                provenance=Provenance(
                    source_type="tool", source_id="decoder",
                    source_timestamp=time.time(),
                    verification_state="unverified"),
            )
            return _quarantine_record(art, name=name, magic=magic)
        except Exception as exc:  # noqa: BLE001 — decode must not fail
            _log.debug("decoder quarantine failed: %s", exc)
            return None

    def _explain(self, report: Any, want: bool) -> str:
        if not want:
            return ""
        router = getattr(self.context, "router", None)
        if router is not None and self.context is not None:
            try:
                from ..llm.base import Message

                digest = {
                    "best": report.best,
                    "hash": report.hash,
                    "tokens": report.tokens[:8],
                    "cookies": [c for c in report.cookies if "name" in c][:8],
                    "forensics": {k: report.forensics[k]
                                  for k in ("bytes", "entropy", "charset",
                                            "magic")},
                }
                prompt = (
                    "You are a precise security/decoding analyst. Explain in "
                    "2-4 plain sentences what this decode report found, what "
                    "the data most likely IS, and any security-relevant "
                    "flags. Only state what the report supports.\n"
                    + json.dumps(digest, default=str)[:6000]
                )
                resp = brain_for(self.context).chat([Message(role="user", content=prompt)], task_kind="judge")
                text = str(getattr(resp, "text", "") or "").strip()
                if text:
                    # A canned fallback ("no real model here") names
                    # nothing the report found — it is not an
                    # explanation. Only trust model text that names at
                    # least one decoder from the winning chain; else
                    # use the grounded rule-based explanation.
                    chain = (report.best or {}).get("chain") or []
                    named = [c for c in chain if c != "text"]
                    if not named or any(c.lower() in text.lower()
                                        for c in named):
                        return text
            except Exception as exc:  # noqa: BLE001 — fall back to rules
                _log.debug("decoder explain model failed: %s", exc)
        return _explain_rules(report)


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "decoder_agent",
        description=(
            "Universal Decoder agent: analyze/decode a string or workspace "
            "file (all encodings, hashes, cookies, JWTs, tokens, file "
            "forensics), QUARANTINE binary decodes in the artifact store "
            "(inert, provenance-recorded, unverified), and return "
            "a plain-language explanation. task: string, 'file:<path>', or "
            "JSON {data|path, save, explain, max_depth}. "
            "action=run (default) | explain | decoders | hash."
        ),
        capability=Capability.FS_READ,
    )
    def decoder_agent(task: str = "", *, action: str = "run",
                      data: str = "", path: str = "",
                      save: bool = True, explain: bool = True) -> dict[str, Any]:
        if action == "decoders":
            return {"decoders": [
                {"name": d.name, "description": d.description}
                for d in DECODERS
            ], "count": len(DECODERS)}
        if action == "hash":
            digest = (data or task or "").strip()
            if not digest:
                raise ToolError("action=hash needs the digest string")
            candidates = identify_hash(digest)
            return {"digest": digest, "candidates": candidates,
                    "known": known_hash_lookup(digest) if candidates else None}

        spec: dict[str, Any] = {"save": save, "explain": explain}
        if data:
            spec["data"] = data
        elif path:
            spec["path"] = path
        elif task:
            spec = DecoderAgent._normalize(task)
            spec.setdefault("save", save)
            spec["explain"] = explain
        else:
            raise ToolError("decoder_agent needs task, data, or path")

        agent = DecoderAgent(context=context,
                             name=f"decoder-{int(time.time() * 1000) % 10**6}")
        result: AgentResult = agent.run(spec)
        if not result.ok:
            raise ToolError(result.error or "decoder_agent failed")
        return result.output

    @registry.register(
        "cookie_analyze",
        description=(
            "Cookie analysis & handling (the CookieLab): parse a Cookie / "
            "Set-Cookie header, classify each cookie (session/auth/csrf/"
            "tracking/preference/jwt/encoded), fingerprint the platforms "
            "and services it implies, decode opaque values (URL, base64, "
            "JSON, JWT, hex, gzip blobs), and flag security posture "
            "(HttpOnly/Secure/SameSite). action=report (default) | "
            "ingest (report + feed entities into the knowledge graph)."
        ),
        capability="memory.write",
        parameters={
            "text": "str — the Cookie / Set-Cookie header(s), or "
                    "'file:<path>' to read them from a file",
            "action": "str — report|ingest",
            "source": "str — label for ingest (default 'cookies')",
        },
    )
    def cookie_analyze(
        text: str = "", *, action: str = "report", source: str = "cookies",
    ) -> dict[str, Any]:
        from ..core.cookies import CookieLab

        raw = (text or "").strip()
        if not raw:
            raise ToolError("cookie_analyze needs the cookie header text")
        if raw.startswith("file:"):
            from pathlib import Path
            p = Path(raw[5:])
            if not p.is_file():
                raise ToolError(f"no such file: {p}")
            raw = p.read_text(encoding="utf-8", errors="replace")
        lab = CookieLab()
        if (action or "report").strip().lower() == "ingest":
            from .kg import KnowledgeGraph
            return {"ok": True, "report": lab.report(raw),
                    "ingested": lab.ingest(context, raw, source=source,
                                           graph=KnowledgeGraph(context.db))}
        return {"ok": True, "report": lab.report(raw)}
