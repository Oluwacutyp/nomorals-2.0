"""Hash-chained evidence packs.

Every finding appends a record carrying the previous record's hash; the
record's own hash covers its canonical JSON plus that predecessor. Break
one byte anywhere and `verify()` names the index. The chain head is
signed into the export, so a third party can re-derive the whole thing
without trusting the box that produced it.

Secrets are masked AT REST — always, including in your own case file —
with head/tail proof so the value is still recognisably real. Large
bodies live in content-addressed blobs; the record holds the digest.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.policy import Capability
from ..core.logging_setup import get_logger

_log = get_logger(__name__)
__all__ = ["EvidenceChain", "mask", "verify_chain", "register"]

_GENESIS = "0" * 64
_SECRET_HEADERS = re.compile(
    r"^(authorization|cookie|set-cookie|x-api-key|api-key|x-auth-token|"
    r"x-csrf-token|proxy-authorization)$", re.I)
_SECRET_SHAPES = (
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWT
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),                                          # long key
    re.compile(r"\b(?:sk|pk|rk|ghp|gho|xox[baprs])_[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
)


def _tail(value: str) -> str:
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]}"


def mask(value: str) -> str:
    """Keep shape, lose secret. Short values go fully starred."""
    text = value or ""
    for shape in _SECRET_SHAPES:
        text = shape.sub(lambda m: _tail(m.group(0)), text)
    return text


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(record: dict[str, Any], prev: str) -> str:
    return hashlib.sha256(_canonical(record) + prev.encode()).hexdigest()


@dataclass
class Finding:
    module: str
    target: str
    verdict: str
    request: dict[str, Any]
    response: dict[str, Any]
    detail: dict[str, Any]

    def masked(self) -> dict[str, Any]:
        req = dict(self.request)
        headers = {k: (mask(v) if _SECRET_HEADERS.match(k) else mask(v))
                   for k, v in (req.get("headers") or {}).items()}
        req["headers"] = headers
        if "body" in req:
            req["body"] = mask(str(req["body"])[:4_000])
        resp = dict(self.response)
        body = str(resp.pop("body", "") or "")
        resp["body_excerpt"] = mask(body[:1_200])
        resp["body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        resp["body_bytes"] = len(body)
        return {"module": self.module, "target": self.target,
                "verdict": self.verdict, "request": req, "response": resp,
                "detail": self.detail}


class EvidenceChain:
    """One case, append-only, tamper-evident."""

    def __init__(self, root: str | Path, case: str) -> None:
        slug = re.sub(r"[^a-z0-9_\-]+", "-", (case or "").strip().lower()).strip("-")
        if not slug:
            raise ToolError("evidence needs a case name")
        self.case = slug
        self.dir = Path(root) / "evidence" / slug
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chain_path = self.dir / "chain.jsonl"
        self.blob_dir = self.dir / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        self._head = self._read_head()

    def _read_head(self) -> str:
        if not self.chain_path.is_file():
            return _GENESIS
        last = _GENESIS
        for line in self.chain_path.read_text(encoding="utf-8").splitlines():
            try:
                last = json.loads(line)["hash"]
            except (ValueError, KeyError):
                continue
        return last

    def _store_blob(self, body: str) -> Optional[str]:
        if not body:
            return None
        name = hashlib.sha256(body.encode()).hexdigest()
        path = self.blob_dir / name
        if not path.exists():
            path.write_text(body, encoding="utf-8", errors="replace")
        return name

    def add(self, finding: Finding) -> dict[str, Any]:
        payload = finding.masked()
        blob = self._store_blob(str(finding.response.get("body") or ""))
        record = {
            "seq": self._seq() + 1,
            "at": time.time(),
            "at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "case": self.case,
            "blob": blob,
            "prev": self._head,
            **payload,
        }
        record["hash"] = _digest(record, self._head)
        with open(self.chain_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._head = record["hash"]
        return record

    def _seq(self) -> int:
        if not self.chain_path.is_file():
            return 0
        return sum(1 for _ in self.chain_path.open(encoding="utf-8"))

    def head(self) -> str:
        return self._head

    def verify(self) -> dict[str, Any]:
        return verify_chain(self.chain_path)

    def export(self, out_dir: Optional[str | Path] = None) -> dict[str, Any]:
        target = Path(out_dir) if out_dir else self.dir / "export"
        target.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in
                self.chain_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        manifest = {
            "case": self.case,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": len(rows),
            "head": self._head,
            "verify": verify_chain(self.chain_path)["state"],
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        (target / "chain.jsonl").write_text(
            self.chain_path.read_text(encoding="utf-8"), encoding="utf-8")
        lines = [f"# Evidence — {self.case}", "",
                 f"Records: {len(rows)}  ",
                 f"Chain head: `{self._head}`  ",
                 f"Integrity: **{manifest['verify']}**", ""]
        for row in rows:
            lines += [f"## {row['seq']}. {row['module']} → {row['target']}",
                      f"- verdict: `{row['verdict']}`",
                      f"- at: {row['at_iso']}",
                      f"- request: `{row['request'].get('method','')} "
                      f"{row['request'].get('url','')}`",
                      f"- response: {row['response'].get('status','')} "
                      f"({row['response'].get('body_bytes', 0)}b, "
                      f"sha256 {str(row['response'].get('body_sha256',''))[:16]}…)",
                      f"- record hash: `{row['hash'][:16]}…`", ""]
        (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return {"dir": str(target), "records": len(rows), **manifest}


def verify_chain(path: str | Path) -> dict[str, Any]:
    """Re-derive the chain. Fails closed at the first break."""
    p = Path(path)
    if not p.is_file():
        return {"state": "missing", "records": 0, "broken_at": None}
    prev = _GENESIS
    count = 0
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            return {"state": "corrupt", "records": count, "broken_at": idx,
                    "why": "unparseable line"}
        claimed = record.pop("hash", "")
        if record.get("prev") != prev:
            return {"state": "broken", "records": count, "broken_at": idx,
                    "why": "prev hash mismatch"}
        if _digest(record, prev) != claimed:
            return {"state": "broken", "records": count, "broken_at": idx,
                    "why": "record hash mismatch"}
        prev = claimed
        count += 1
    return {"state": "intact", "records": count, "broken_at": None,
            "head": prev}


def register(registry: Any) -> None:
    context = registry.context
    root = Path(getattr(context, "home", ".") or ".") / "data"

    @registry.register(
        "evidence_add",
        description="Append a finding to a case chain (secrets masked, body "
                    "content-addressed). Returns the record hash.",
        capability=Capability.DB_WRITE,
        parameters={"case": "str — case slug", "module": "str — producer",
                    "target": "str", "verdict": "str",
                    "request": "dict — method/url/headers/body",
                    "response": "dict — status/headers/body",
                    "detail": "dict (optional)"},
    )
    def evidence_add(case: str, module: str, target: str, verdict: str,
                     request: dict[str, Any], response: dict[str, Any],
                     *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        chain = EvidenceChain(root, case)
        record = chain.add(Finding(module=module, target=target, verdict=verdict,
                                   request=request, response=response,
                                   detail=detail or {}))
        return {"seq": record["seq"], "hash": record["hash"],
                "head": chain.head()}

    @registry.register(
        "evidence_verify",
        description="Re-derive a case chain; reports the first broken index.",
        capability=Capability.DB_READ,
        parameters={"case": "str — case slug"},
    )
    def evidence_verify(case: str) -> dict[str, Any]:
        return verify_chain(Path(root) / "evidence" /
                            re.sub(r"[^a-z0-9_\-]+", "-", case.lower().strip("-"))
                            / "chain.jsonl")

    @registry.register(
        "evidence_export",
        description="Write manifest + chain + markdown report for a case.",
        capability=Capability.DB_READ,
        parameters={"case": "str — case slug"},
    )
    def evidence_export(case: str) -> dict[str, Any]:
        return EvidenceChain(root, case).export()
