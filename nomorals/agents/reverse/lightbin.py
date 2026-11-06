"""The light native rung (Pass 2) — binutils one-shot triage.

Where the deep rung is Ghidra, the light rung is the supervised
one-shot shape REA uses for its offline tools: ``readelf`` probed
on the machine and run, bounded, over one ELF target — header,
dynamic dependencies, the dynamic symbol view, sections — folded
into Evidence. It answers in a second what Ghidra answers in a
minute, at metadata depth: no code is disassembled and nothing is
executed.

Provisioning posture, stated plainly: binutils is probed where
present (it ships with every toolchain host). There is no honest
single-artifact publisher channel for binutils across Devon's
profiles the way there is for the JDK and Ghidra, so this rung
does not fake one — where readelf is absent the rung reports
unavailable with that reason, and the deep rung (which Devon
*does* provision) carries the work.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "probe", "triage",
    "parse_readelf", "fold_evidence", "LIMITATIONS",
]

PROVIDER_ID = "devon.binutils-triage"
PROVIDER_NAME = "binutils readelf one-shot triage (probed)"

RUN_TIMEOUT_S = 60.0

LIMITATIONS = (
    "header and dynamic-linking metadata only — no instruction "
    "was disassembled and no function behavior is established",
    "the symbol view is the dynamic symbol table (what the "
    "loader resolves), not the full symbol table of an unstripped "
    "binary",
)

Runner = Callable[..., tuple[int, str]]


def _run(argv: list[str], *, timeout: float = RUN_TIMEOUT_S,
         env: dict[str, str] | None = None) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def probe(runner: Runner | None = None) -> dict[str, Any]:
    """Is the light rung usable here? Probe-executed, like every
    other row of the matrix: the binary must exist AND answer."""
    path = shutil.which("readelf")
    if not path:
        return {"available": False, "path": "", "version": "",
                "detail": "readelf (binutils) is not on this "
                          "machine's PATH — the light rung probes "
                          "binutils where present and does not "
                          "provision it (no single-artifact "
                          "publisher channel); the Ghidra rung, "
                          "which Devon provisions, covers depth"}
    run = runner or _run
    try:
        _rc, out = run([path, "--version"], timeout=10.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "path": path, "version": "",
                "detail": f"readelf at {path} did not answer its "
                          f"version probe: {exc}"}
    version = (out or "").splitlines()[0].strip() if out else ""
    if "readelf" not in (out or ""):
        return {"available": False, "path": path, "version": "",
                "detail": f"readelf at {path} answered its version "
                          f"probe with something else: {version}"}
    return {"available": True, "path": path, "version": version,
            "detail": f"binutils probed at {path} ({version})"}


_HEADER_KEYS = {
    "Class": "elf_class", "Data": "data_encoding",
    "OS/ABI": "os_abi", "Type": "object_type",
    "Machine": "machine", "Entry point address": "entry_point",
}


def parse_readelf(header: str, dynamic: str, dynsyms: str,
                  sections: str) -> dict[str, Any]:
    """Parse the four readelf views into one facts dict. Pure —
    the shapes pinned here are readelf's real output shapes."""
    facts: dict[str, Any] = {}
    for line in (header or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        mapped = _HEADER_KEYS.get(key.strip())
        if mapped:
            facts[mapped] = value.strip()
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]",
                        dynamic or "")
    facts["needed_libraries"] = needed
    facts["bind_now"] = "(FLAGS)" in (dynamic or "") and \
        "BIND_NOW" in (dynamic or "")
    defined_funcs: list[str] = []
    imported: list[str] = []
    total = 0
    for line in (dynsyms or "").splitlines():
        parts = line.split()
        # Num: Value Size Type Bind Vis Ndx Name
        if len(parts) < 8 or not parts[0].rstrip(":").isdigit():
            continue
        total += 1
        sym_type, ndx, name = parts[3], parts[6], parts[7]
        name = name.split("@")[0]
        if ndx == "UND":
            imported.append(name)
        elif sym_type == "FUNC":
            defined_funcs.append(name)
    facts["dynamic_symbols"] = total
    facts["defined_functions"] = defined_funcs[:64]
    facts["defined_function_count"] = len(defined_funcs)
    facts["imported_symbols"] = imported[:64]
    facts["imported_symbol_count"] = len(imported)
    section_names = re.findall(r"\]\s+(\S+)\s+(?:PROGBITS|NOBITS|"
                               r"SYMTAB|STRTAB|DYNAMIC|NOTE|RELA|"
                               r"INIT_ARRAY|FINI_ARRAY|VERSYM|"
                               r"VERNEED|GNU_HASH|HASH|PLT|DYNSYM)",
                               sections or "")
    facts["sections"] = section_names
    return facts


def triage(target: str | Path, *,
           runner: Runner | None = None) -> dict[str, Any]:
    """Run the light rung over one ELF file → facts dict.

    Raises RuntimeError with the tool's own tail when readelf
    refuses the file (a non-ELF target is the common case) — the
    caller decides what an unavailable light rung means.
    """
    state = probe(runner)
    if not state["available"]:
        raise RuntimeError("light rung unavailable: "
                           + state["detail"])
    readelf = state["path"]
    run = runner or _run
    target = str(target)
    views: dict[str, str] = {}
    for label, args in (
            ("header", ["-h", "-W"]),
            ("dynamic", ["-d", "-W"]),
            ("dynsyms", ["-sW", "--dyn-syms"]),
            ("sections", ["-SW"])):
        try:
            rc, out = run([readelf, *args, target],
                          timeout=RUN_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"readelf {label} run failed: {exc}") \
                from exc
        if rc != 0 and label == "header":
            tail = " · ".join((out or "").strip().splitlines()[-2:])
            raise RuntimeError(f"readelf refused the target: {tail}")
        views[label] = out or ""
    facts = parse_readelf(views["header"], views["dynamic"],
                          views["dynsyms"], views["sections"])
    facts["probe_version"] = state["version"]
    return facts


def fold_evidence(store: Any, *, subject_sha256: str,
                  subject_name: str, subject_size: int,
                  subject_path: str, facts: dict[str, Any],
                  link_to: str = "") -> dict[str, Any]:
    """Fold one triage into an Evidence record."""
    rec = store.record(
        predicate_type="elf.triage", operation="readelf.triage",
        provider_id=PROVIDER_ID, provider_name=PROVIDER_NAME,
        provider_version=str(facts.pop("probe_version", "")),
        parameters={"tool": "readelf"},
        result=facts,
        subject_sha256=subject_sha256, subject_name=subject_name,
        subject_size=subject_size,
        limitations=list(LIMITATIONS),
        locations=[{"type": "artifact-path", "value": subject_path}],
        links=[link_to] if link_to else [])
    return {"evidence_ids": [rec.evidence_id], "unknowns": []}
