"""The Ghidra headless rung (Pass 2) — Devon's deep provider.

REA bridges into a Ghidra the human installed. Devon provisions
her own (see :mod:`.provision`: pinned Ghidra 12.2 + pinned
Temurin JDK 25, sha256-verified, probe-executed) and drives it
headless: ``analyzeHeadless`` imports the target into an ephemeral
project, runs Devon's in-engine export script
(``ghidra_scripts/DevonGhidraExport.java`` — the Java rung, because
Ghidra's extension API is Java), and deletes the project. The
script's JSON — program identity, the function table with call
edges, imports, defined strings, bounded decompiler pseudocode —
is folded here into Evidence records with Ghidra named as the
authority and the engine's limits stated on every record.

The target is never executed: headless static analysis only.
Dynamic execution is the gated Pass 4 rung and is not built here.

REA's stated limits, and where this rung stands on each:

* *No source recovery* — matched, and said on the record: the
  decompiler emits reconstructed pseudocode, not the original
  source; names and types the engine inferred are inference.
* *Ephemeral project databases* — matched: the project is deleted
  after export; what persists is the Evidence, in Devon's store.
* *Read-only against the target* — matched and stronger: the
  target file is opened by the importer only; nothing here writes
  to it, and analysis writes go to the scratch project only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import provision
from .failures import record_failure

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "GhidraError", "analyze",
    "fold_evidence", "LIMITATIONS",
]

PROVIDER_ID = "devon.ghidra-headless"
PROVIDER_NAME = "Ghidra headless (self-provisioned by Devon)"

#: Analysis budget for one target, seconds. Bounded by design: a
#: binary that cannot be analyzed inside it yields partial
#: evidence + a residual unknown, not an unbounded run.
ANALYSIS_TIMEOUT_S = 900.0

#: The limits carried on every Evidence record this rung emits.
LIMITATIONS = (
    "decompiler output is reconstructed pseudocode, not the "
    "original source — control flow is the engine's analysis, "
    "and identifier names/types it inferred are its inference",
    "the Ghidra project database is ephemeral: it is deleted "
    "after export, so only these Evidence records persist",
    "analysis is bounded (per-file timeout, function/string/"
    "decompile caps in the export script); a capped or timed-out "
    "analysis says so on the record and raises a residual unknown",
    "static analysis only — the target was never executed",
)

_SCRIPT = Path(__file__).resolve().parent / "ghidra_scripts" / \
    "DevonGhidraExport.java"


class GhidraError(RuntimeError):
    """The rung failed; the message names where (launch, analysis,
    export) rather than swallowing the engine's own words."""


Runner = Callable[..., tuple[int, str]]


def _run(argv: list[str], *, env: dict[str, str] | None = None,
         timeout: float = ANALYSIS_TIMEOUT_S) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def script_dir(root: Path | None = None) -> Path:
    """Stage the export script under the tools root (Ghidra's
    -scriptPath points at a directory; the repo copy is the source
    of truth, re-staged whenever it changes)."""
    dest_dir = provision.re_root(root) / "scripts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _SCRIPT.name
    try:
        if not dest.is_file() or dest.read_bytes() != \
                _SCRIPT.read_bytes():
            shutil.copyfile(_SCRIPT, dest)
    except OSError as exc:
        record_failure("reverse.ghidra.stage_script", exc)
        raise GhidraError(f"could not stage the export script: {exc}") \
            from exc
    return dest_dir


def analyze(target: str | Path, *, root: Path | None = None,
            runner: Runner | None = None,
            timeout: float = ANALYSIS_TIMEOUT_S) -> dict[str, Any]:
    """Run the headless rung over one target → the export dict.

    Raises :class:`GhidraError` (with the engine's tail attached)
    when the rung is not provisioned, the launch fails, or the
    export does not materialize. Never returns a partial dict
    dressed as success.
    """
    state = provision.ghidra_state(root, runner=runner)
    if not state["available"]:
        raise GhidraError("ghidra rung unavailable: " + state["detail"])
    jdk = provision.jdk_state(root, runner=runner)
    home = Path(state["path"])
    script = "analyzeHeadless.bat" if os.name == "nt" \
        else "analyzeHeadless"
    env = dict(os.environ)
    jdk_home = jdk["path"]
    if jdk["source"] == "system":
        jdk_home = str(Path(jdk_home).parent.parent)
    env["JAVA_HOME"] = jdk_home
    target_path = Path(target).expanduser()
    if not target_path.is_file():
        raise GhidraError(f"no readable file at {target!r}")
    run = runner or _run
    scripts = script_dir(root)
    work = Path(tempfile.mkdtemp(prefix="devon-re-ghidra-",
                                 dir=str(provision.re_root(root))))
    out_json = work / "export.json"
    proj_dir = work / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        str(home / "support" / script), str(proj_dir), "DevonRE",
        "-import", str(target_path),
        "-scriptPath", str(scripts),
        "-postScript", "DevonGhidraExport.java", str(out_json),
        "-deleteProject",
        "-analysisTimeoutPerFile", "600",
    ]
    try:
        rc, output = run(argv, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GhidraError(
            f"ghidra analysis exceeded the {timeout:.0f}s bound "
            f"and was stopped") from exc
    except OSError as exc:
        raise GhidraError(f"ghidra launch failed: {exc}") from exc
    if not out_json.is_file():
        tail = " · ".join((output or "").strip().splitlines()[-4:])
        raise GhidraError(
            f"ghidra produced no export (exit {rc}): {tail}")
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GhidraError(f"ghidra export was not readable JSON: "
                          f"{exc}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not isinstance(data, dict) or "program" not in data:
        raise GhidraError("ghidra export lacked the program block")
    data["_engine"] = {"rc": rc, "version": state["version"]}
    return data


def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, subject_size: int,
                  subject_path: str, export: dict[str, Any],
                  link_to: str = "") -> dict[str, Any]:
    """Fold one Ghidra export into Evidence records on the store.

    Returns {"evidence_ids": [...], "unknowns": [...]} — the
    records are linked to the inspection's identify record when
    ``link_to`` names it, and each record carries the rung's
    limitations verbatim.
    """
    version = str((export.get("_engine") or {}).get("version")
                  or provision.DESCRIPTORS["ghidra"]["version"])
    program = export.get("program") or {}
    functions = export.get("functions") or []
    imports = export.get("imports") or []
    strings = export.get("strings") or []
    common = {
        "subject_sha256": subject_sha256,
        "subject_name": subject_name,
        "subject_size": subject_size,
        "provider_id": PROVIDER_ID,
        "provider_name": PROVIDER_NAME,
        "provider_version": version,
        "locations": [{"type": "artifact-path", "value": subject_path}],
    }
    links = [link_to] if link_to else []
    evidence_ids: list[str] = []

    identity = store.record(
        predicate_type="native.program", operation="ghidra.analyze",
        parameters={"engine": "ghidra-headless"},
        result={
            "format": program.get("executable_format"),
            "language": program.get("language"),
            "compiler": program.get("compiler"),
            "image_base": program.get("image_base"),
            "functions_total": export.get("functions_total",
                                          len(functions)),
            "decompiled_count": export.get("decompiled_count", 0),
            "decompiler_available":
                bool(export.get("decompiler_available")),
        },
        limitations=list(LIMITATIONS), links=links, **common)
    evidence_ids.append(identity.evidence_id)
    anchor = identity.evidence_id

    fn_rows = [{
        "name": f.get("name"), "entry": f.get("entry"),
        "body_size": f.get("body_size"),
        "signature": f.get("signature"),
        "is_thunk": bool(f.get("is_thunk")),
        "is_external": bool(f.get("is_external")),
        "calls": list(f.get("calls") or []),
    } for f in functions]
    fn_rec = store.record(
        predicate_type="native.functions", operation="ghidra.analyze",
        parameters={"engine": "ghidra-headless"},
        result={"functions": fn_rows,
                "count": len(fn_rows),
                "truncated": bool(export.get("functions_truncated"))},
        limitations=list(LIMITATIONS), links=[anchor], **common)
    evidence_ids.append(fn_rec.evidence_id)

    snippets = [{
        "function": f.get("name"), "entry": f.get("entry"),
        "code": f.get("decompile"),
    } for f in functions if f.get("decompile")]
    fail_names = [str(f.get("name")) for f in functions
                  if f.get("decompile_error")]
    if snippets or fail_names:
        decomp = store.record(
            predicate_type="native.decompile",
            operation="ghidra.decompile",
            parameters={"engine": "ghidra-headless"},
            result={"snippets": snippets,
                    "decompiled": len(snippets),
                    "failed": fail_names},
            confidence="derived",
            limitations=list(LIMITATIONS), links=[anchor], **common)
        evidence_ids.append(decomp.evidence_id)

    if imports:
        imp = store.record(
            predicate_type="native.imports",
            operation="ghidra.analyze",
            parameters={"engine": "ghidra-headless"},
            result={"imports": imports, "count": len(imports)},
            limitations=list(LIMITATIONS), links=[anchor], **common)
        evidence_ids.append(imp.evidence_id)

    if strings:
        strs = store.record(
            predicate_type="native.strings",
            operation="ghidra.analyze",
            parameters={"engine": "ghidra-headless"},
            result={"strings": strings, "count": len(strings),
                    "truncated": bool(export.get("strings_truncated"))},
            limitations=list(LIMITATIONS), links=[anchor], **common)
        evidence_ids.append(strs.evidence_id)

    unknowns = []
    if fail_names:
        unknowns.append(store.open_unknown(
            "what do these functions do — the decompiler failed "
            "on them in this run: " + ", ".join(fail_names[:8]),
            subject_sha256=subject_sha256, raised_by=anchor,
            supporting=[anchor],
            required_authority="shipped-artifact",
            required_confidence="derived"))
    if export.get("functions_truncated"):
        unknowns.append(store.open_unknown(
            "the function census of this binary is incomplete — "
            "the export hit its function cap; what lies beyond it "
            "is unestablished",
            subject_sha256=subject_sha256, raised_by=anchor,
            supporting=[anchor],
            required_authority="shipped-artifact",
            required_confidence="derived"))
    if not export.get("decompiler_available"):
        unknowns.append(store.open_unknown(
            "no decompiler output exists for this binary — the "
            "Ghidra decompiler did not initialize in this run, so "
            "function behavior is unestablished beyond the call "
            "graph",
            subject_sha256=subject_sha256, raised_by=anchor,
            supporting=[anchor],
            required_authority="shipped-artifact",
            required_confidence="derived"))
    return {"evidence_ids": evidence_ids,
            "unknowns": [u.to_dict() for u in unknowns]}
