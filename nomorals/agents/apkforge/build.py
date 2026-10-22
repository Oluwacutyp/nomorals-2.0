"""ApkForge builder — apktool rebuild → zipalign → vault-key sign →
verify, with a build card that always tells the truth (design §3.4).

The card names: the source APK sha, the edit-journal hash the build
was made from (so any shipped APK traces to its exact edit set),
the signing alias + cert fingerprint, the schemes apksigner
actually applied (from its own verify output), and the standing
re-sign facts.  A build that fails names its step — decode, apktool
build, zipalign, sign, verify — and produces no card.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import signing as signing_mod
from .edit import ensure_decoded, list_edits
from .project import Project, sha256_file
from .toolchain import ToolchainError, run_tool

__all__ = ["BuildError", "build_project", "render_build_card",
           "RESIGN_FACTS"]

RESIGN_FACTS = signing_mod.RESIGN_FACTS


class BuildError(RuntimeError):
    """A designed build failure — names the failing step."""


def _journal_hash(proj: Project) -> str:
    from .project import JOURNAL

    path = proj.dir / JOURNAL
    return sha256_file(path) if path.exists() else ""


def build_project(context: Any, root: Path, proj: Project, *,
                  alias: str = "") -> dict[str, Any]:
    """Decode (if needed) → build → align → sign → verify → card."""
    try:
        ddir = ensure_decoded(root, proj)
    except Exception as exc:  # EditError carries the step name
        raise BuildError(f"decode step: {exc}") from exc

    builds_dir = proj.dir / "builds"
    builds_dir.mkdir(exist_ok=True)
    seq = int(proj.record.get("build_seq") or 0) + 1
    ident = proj.record
    stem = (f"{(ident.get('package') or proj.id).replace('.', '_')}"
            f"-{ident.get('version_name') or 'v'}"
            f"-build{seq}")
    unsigned = builds_dir / f"{stem}-unsigned.apk"
    signed = builds_dir / f"{stem}-signed.apk"

    try:
        res = run_tool("", "apktool",
                       ["b", str(ddir), "-o", str(unsigned)],
                       timeout=900, root=root)
    except ToolchainError as exc:
        raise BuildError(f"apktool build step: {exc}") from exc
    if res["rc"] != 0 or not unsigned.exists():
        tail = (res["stdout"] + res["stderr"]).strip().splitlines()
        raise BuildError(
            "apktool build failed: "
            + (tail[-1][:300] if tail else f"rc={res['rc']}"))

    try:
        sig = signing_mod.sign_apk(context, root, unsigned, signed,
                                   alias=alias)
    except signing_mod.SigningError as exc:
        raise BuildError(f"sign step: {exc}") from exc

    edits = [e for e in list_edits(proj) if not e["undone"]]
    card: dict[str, Any] = {
        "project": proj.id,
        "package": ident.get("package", ""),
        "version_name": ident.get("version_name", ""),
        "version_code": ident.get("version_code"),
        "build_seq": seq,
        "built_at": time.time(),
        "output": str(signed),
        "output_sha256": sha256_file(signed),
        "size": signed.stat().st_size,
        "source_sha256": proj.sha256,
        "edit_journal_sha256": _journal_hash(proj),
        "live_edit_ids": [e["edit_id"] for e in edits],
        "signing": {
            "alias": sig["alias"],
            "debug_key": sig["debug"],
            "cert_sha256": sig["cert_sha256"],
            "schemes": sig["verify"]["schemes"],
            "signer_dn": sig["verify"].get("signer_dn", ""),
        },
        "verify": sig["verify"],
        "resign_facts": RESIGN_FACTS,
        "museum_card": {
            "id": f"{proj.id}-build{seq}",
            "title": f"{proj.title} build {seq} (signed)",
            "kind": "apk-build",
            "shelf": "apk-builds",
            "path": str(signed),
            "hash": sha256_file(signed),
            "size": signed.stat().st_size,
            "provenance": {"type": "build", "ref": proj.id,
                           "tool": "apkforge"},
            "tags": ["apk", "build", "signed",
                     ident.get("package", "")],
        },
    }
    card_path = builds_dir / f"{stem}.json"
    card_path.write_text(json.dumps(card, indent=1, sort_keys=True),
                         encoding="utf-8")
    card["card_path"] = str(card_path)
    proj.record["build_seq"] = seq
    proj.save()
    proj.journal("build",
                 f"build {seq}: {signed.name} — signed with "
                 f"{sig['alias']} ({sig['cert_sha256'][:16]}…), "
                 f"edits live: {card['live_edit_ids']}",
                 build_seq=seq, output=str(signed),
                 output_sha256=card["output_sha256"],
                 edit_journal_sha256=card["edit_journal_sha256"])
    return card


def render_build_card(card: dict[str, Any]) -> str:
    sig = card.get("signing") or {}
    schemes = ", ".join(f"{k}={'yes' if v else 'no'}"
                        for k, v in (sig.get("schemes") or {}).items())
    lines = [
        f"🪓 **built + signed** — {card.get('package')} "
        f"v{card.get('version_name')} (build {card.get('build_seq')})",
        f"file: {card.get('output')}",
        f"sha256: {str(card.get('output_sha256'))[:16]}… · "
        f"{card.get('size', 0):,} bytes",
        f"signed by: alias {sig.get('alias')} "
        f"({'DEBUG key — shared public identity, install testing only'
          if sig.get('debug_key') else 'Devon class keystore, vault-held'})",
        f"cert sha256: {str(sig.get('cert_sha256'))[:24]}…",
        f"schemes verified: {schemes or '—'}",
        f"built from source {str(card.get('source_sha256'))[:16]}… "
        f"+ edit journal {str(card.get('edit_journal_sha256'))[:16]}… "
        f"(live edits: {card.get('live_edit_ids')})",
        "re-sign facts:",
    ]
    lines += [f"  • {fact}" for fact in card.get("resign_facts") or []]
    return "\n".join(lines)
