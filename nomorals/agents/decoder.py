"""DecoderAgent — the Universal Decoder as a living sub-agent.

Wraps ``nomorals.core.decoder`` with agent orchestration:

* **run** — decode a string or workspace file, and when the winning decode
  is a BINARY (base64→PNG, hexdump→ELF, …) quarantine it in the artifact
  store (D1): inert, provenance-recorded, unverified — the store's
  managed copy is the only copy, and it leaves the store only through
  an explicit owner action (a surface resolving the artifact ref).
  File inputs stream (never truncated); oversized decodes spool to
  the decoder's staging area and are quarantined from disk.  Each
  run also sweeps the quarantine: decoded-untrusted residents past
  the size budget are evicted oldest-first (their records stay,
  marked evicted with the reason; kept/referenced artifacts are
  exempt).
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
from ..core.decoder import (DECODERS, StagedOutput, analyze, analyze_path,
                            decompression_run, identify_hash,
                            known_hash_lookup, quarantine_budget_bytes)
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


def _quarantine_metadata(name: str, magic: str,
                         sha: str) -> dict[str, Any]:
    """The D1 quarantine marking every decoded binary carries."""
    return {
        "filename": name,
        "kind": "decoded-binary",
        "quarantine": "decoded-untrusted",
        "inert": True,
        "release": "owner-action-only",
        "magic": magic,
        "sha256": sha,
    }


def _live_quarantine_row(store: Any, sha: str) -> Any | None:
    """The live (not evicted) decoded-untrusted row for ``sha`` —
    the store's content-hash dedup read, minus reclaimed rows: an
    evicted row's bytes are gone, so it must never be handed back
    as if the decode were still held."""
    for art in store.find_by_content_hash(sha):
        meta = art.metadata or {}
        if meta.get("quarantine") == "decoded-untrusted" and \
                not meta.get("evicted"):
            return art
    return None


def sweep_quarantine(store: Any, *,
                     budget: int | None = None) -> dict[str, Any]:
    """Reclaim quarantine space: evict decoded-untrusted residents
    beyond the size budget, oldest first.

    An eviction is journaled ON the artifact record (an ``evicted``
    metadata entry naming the time, the size, the budget and the
    reason) and the blob bytes are released back to the store —
    the record remains as the audit trail; nothing silently
    disappears.  Exempt, always: artifacts an owner action has
    kept — alias targets (a named pointer is a deliberate keep),
    artifacts other artifacts derive from / supersede / contradict
    (lineage is kept work), mission/task-linked artifacts, and any
    record carrying a ``released_at`` or ``keep`` mark.  (Release
    itself is an ACTION in the shipped system — a surface resolves
    the artifact and exports its bytes; no release state is
    recorded on the record, so these shipped keep-signals are what
    the sweep honors.)  Artifacts with any other quarantine value,
    or none, are never touched.  Fail-soft: a sweep problem is
    logged and reported in the result, never raised into a decode.
    """
    if budget is None:
        budget = quarantine_budget_bytes()
    result: dict[str, Any] = {
        "budget_bytes": budget, "resident_bytes": 0,
        "resident_bytes_after": 0, "evicted": [], "bytes_reclaimed": 0,
        "error": None,
    }
    try:
        residents = []
        for art in store.search(
                creator="decoder",
                metadata={"quarantine": "decoded-untrusted"},
                limit=100_000):
            if not (art.metadata or {}).get("evicted"):
                residents.append(art)
        total = sum(a.size for a in residents)
        result["resident_bytes"] = total
        result["resident_bytes_after"] = total
        if total <= budget:
            return result
        exempt = set(store.aliases().values())
        for art in store.search(limit=100_000):
            prov = art.provenance
            exempt.update(prov.derived_from)
            exempt.update(prov.supersedes)
            exempt.update(prov.contradicts)
        for art in sorted(residents, key=lambda a: a.created_at):
            if total <= budget:
                break
            meta = art.metadata or {}
            if art.id in exempt or art.mission_id or art.task_id \
                    or meta.get("released_at") or meta.get("keep"):
                continue
            record = {
                "artifact_id": art.id,
                "name": meta.get("filename") or "",
                "size": art.size,
                "evicted_at": time.time(),
                "reason": "quarantine-budget",
                "budget_bytes": budget,
                "resident_bytes_before": result["resident_bytes"],
            }
            try:
                store.tag(art.id, "evicted", record)
                if hasattr(store.blobs, "release"):
                    store.blobs.release(art.content_hash,
                                        delete_at_zero=True)
            except Exception as exc:  # noqa: BLE001 — fail-soft
                _log.debug("quarantine eviction failed for %s: %s",
                           art.id, exc)
                continue
            total -= art.size
            result["bytes_reclaimed"] += art.size
            result["evicted"].append(record)
        result["resident_bytes_after"] = total
    except Exception as exc:  # noqa: BLE001 — the sweep never breaks a decode
        _log.debug("quarantine sweep failed: %s", exc)
        result["error"] = str(exc)
    return result


def _read_sample(target: Any, limit: int = 200_001) -> bytes:
    """A bounded head sample of a file (curation context only —
    analysis itself streams the whole file via analyze_path)."""
    buf = bytearray()
    with open(target, "rb") as fh:
        while len(buf) < limit:
            chunk = fh.read(min(1 << 20, limit - len(buf)))
            if not chunk:
                break
            buf += chunk
    return bytes(buf)


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

        store = _artifact_store_for(self.context) if save else None
        sweep_before = sweep_quarantine(store) if store is not None \
            else None

        artifact = None
        saved_to = None
        # One decompression run spans the analysis AND the
        # quarantine copy: staged decodes live in the staging area
        # until the store has taken its copy, and the run's close
        # deletes the staging either way (success, refusal, error).
        with decompression_run():
            if path:
                from ..tools.filesystem import safe_path

                target = safe_path(self.context, path, must_exist=True)
                report = analyze_path(
                    target, max_depth=spec.get("max_depth", 3))
                report_dict = report.to_dict()
                report_dict["source"] = str(target)
                raw: Any = _read_sample(target)
            elif data:
                raw = data
                report = analyze(data, max_depth=spec.get("max_depth", 3))
                report_dict = report.to_dict()
            else:
                raise ToolError("decoder_agent needs data or path")

            best = report.best
            if best and save:
                out = best["output"]
                if isinstance(out, (bytes, bytearray)):
                    artifact = self._quarantine_bytes(bytes(out),
                                                      store=store)
                elif isinstance(out, StagedOutput):
                    artifact = self._quarantine_staged(out, store=store)
                if artifact is not None:
                    saved_to = artifact["uri"]
                    report_dict["saved_to"] = saved_to
                    report_dict["artifact"] = artifact
                elif isinstance(out, (bytes, bytearray, StagedOutput)):
                    # No store reachable: the bytes are persisted
                    # NOWHERE (never a loose file) and the record
                    # says so.
                    report_dict["storage"] = (
                        "unavailable — no artifact store is attached, so "
                        "the decoded binary was not persisted anywhere")

            explanation = self._explain(report, spec.get("explain", True))
            self._curate(report_dict, explanation, raw)

        sweep_after = sweep_quarantine(store) if (
            store is not None and artifact is not None) else None
        from ..tools.decoder import decoder_limits

        out_dict: dict[str, Any] = {
            "report": report_dict, "saved_to": saved_to,
            "artifact": artifact, "limits": decoder_limits(),
            "explanation": explanation}
        evicted = [rec for sweep in (sweep_before, sweep_after)
                   if sweep for rec in sweep["evicted"]]
        if evicted:
            out_dict["quarantine_sweep"] = {
                "evicted": evicted,
                "bytes_reclaimed": sum(
                    rec["size"] for rec in evicted),
                "budget_bytes": quarantine_budget_bytes(),
            }
        return out_dict

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

    def _quarantine_bytes(self, raw: bytes,
                          store: Any = None) -> dict[str, Any] | None:
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
        if store is None:
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
            existing = _live_quarantine_row(store, sha)
            if existing is not None:
                return _quarantine_record(existing, name=name, magic=magic)
            art = store.put(
                raw, type="file", creator="decoder",
                mime="application/octet-stream",
                metadata=_quarantine_metadata(name, magic, sha),
                provenance=Provenance(
                    source_type="tool", source_id="decoder",
                    source_timestamp=time.time(),
                    verification_state="unverified"),
            )
            return _quarantine_record(art, name=name, magic=magic)
        except Exception as exc:  # noqa: BLE001 — decode must not fail
            _log.debug("decoder quarantine failed: %s", exc)
            return None

    def _quarantine_staged(self, staged: StagedOutput,
                           store: Any = None) -> dict[str, Any] | None:
        """Quarantine a staged decode (too large to hold in RAM).

        The store takes its copy by STREAMING the staging file
        (``ArtifactStore.put_file`` over the blob store's streaming
        put) — the bytes never materialize in memory on this side
        either.  Same D1 marking, same content-hash dedup as the
        RAM-held path; the staging file itself is deleted by the
        decompression run when it closes, so the store's copy is
        again the only copy.
        """
        if store is None:
            store = _artifact_store_for(self.context)
        if store is None:
            return None
        magic = staged.magic
        ext = _MAGIC_EXT.get(magic, ".bin")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"decoded-{stamp}{ext}"
        try:
            from ..storage.artifacts import Provenance

            existing = _live_quarantine_row(store, staged.sha256)
            if existing is not None:
                return _quarantine_record(existing, name=name, magic=magic)
            metadata = _quarantine_metadata(name, magic, staged.sha256)
            metadata["staged"] = "streamed"
            art = store.put_file(
                staged.path, type="file", creator="decoder",
                mime="application/octet-stream",
                metadata=metadata,
                provenance=Provenance(
                    source_type="tool", source_id="decoder",
                    source_timestamp=time.time(),
                    verification_state="unverified"),
            )
            return _quarantine_record(art, name=name, magic=magic)
        except Exception as exc:  # noqa: BLE001 — decode must not fail
            _log.debug("decoder staged quarantine failed: %s", exc)
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
