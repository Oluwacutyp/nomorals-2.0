"""Indd-Utils: InDesign (.indd) → IDML conversion with measured
fidelity as a brain tool (Indd-Utils port of storytold/indd-utils).

A thin surface over :mod:`nomorals.documents.indd` — the organ owns
the binary container, the database, the object cursor, the document
model, the IDML writer and the fidelity differ; the tool only routes
arguments and shapes the reply. Devon already had a documents engine
with a measured-fidelity corpus; this tool adds what it lacked: open
a real .indd, convert it to a real .idml package, and measure the
coverage of the result against a reference package. Pure local work,
no upload. Conversion writes .idml files, so FS_WRITE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..documents.indd import convert_file as _convert_file
from ..documents.indd import measure_fidelity as _measure_fidelity
from ..documents.indd import read_document as _read_document
from ..documents.indd import read_header as _read_header
from ..documents.indd.attribution import verify_attribution
from ..documents.indd.errors import InddError
from ..documents.indd.parity import run_parity

__all__ = ["register"]

_ACTIONS = ("convert", "inspect", "fidelity", "parity", "attribution",
            "status")


def _source_path(sources: list[str] | None) -> Path:
    if not sources:
        raise ValueError("inddutils needs a file path "
                         "(sources=[<path>])")
    return Path(str(sources[0])).expanduser()


def register(registry: Any) -> None:
    """Attach the inddutils family to a registry."""
    @registry.register(
        "inddutils",
        description=(
            "InDesign (.indd) to IDML conversion with measured fidelity "
            "— open a real .indd (InDesign 3.0–20.3, big- or "
            "little-endian), convert it to a real .idml ZIP package "
            "(designmap, fonts, styles, preferences, backing story, "
            "tags, spreads, stories, graphic), and diff a package "
            "against a reference with value coverage % as the headline. "
            "Indd-Utils actions: convert=<convert sources=[<indd>] to "
            "an .idml package, output=<path> to write it, params=<{name: "
            "<document name>}>>; inspect=<read a file's header and "
            "document summary, sources=[<indd>]>; fidelity=<diff "
            "sources=[<reference.idml>, <package.idml>] and report "
            "coverage>; parity=<run the machine-checked parity TOML>; "
            "attribution=<verify the SHA-256 attribution DB>; status=<"
            "organ state>. action=<convert|inspect|fidelity|parity|"
            "attribution|status>; sources=<list of paths>; output=<path "
            "to write>; params=<dict>."
        ),
        capability=Capability.FS_WRITE,
    )
    def inddutils_tool(action: str = "inspect",
                       sources: list[str] | None = None,
                       output: str = "", kind: str = "",
                       node: str = "", nodes: str | list[str] | None = None,
                       value: str = "", x: float = 0.0, y: float = 0.0,
                       command: str = "",
                       params: dict | None = None) -> dict[str, Any]:
        opts = dict(params or {})
        if action == "status":
            fixtures = Path(__file__).resolve().parent.parent.parent \
                / "tests" / "fixtures" / "inddutils"
            present = [p.name for p in sorted(fixtures.glob("*.indd"))] \
                if fixtures.is_dir() else []
            return {"ok": True, "organ": "nomorals.documents.indd",
                    "upstream": "storytold/indd-utils @ 136f973",
                    "fixtures": present}
        if action == "parity":
            return run_parity()
        if action == "attribution":
            return verify_attribution()
        if action == "inspect":
            path = _source_path(sources)
            try:
                header = _read_header(path.read_bytes())
                doc = _read_document(path.read_bytes())
            except InddError as exc:
                return {"ok": False, "error": str(exc),
                        "kind": exc.kind}
            return {"ok": True, "path": str(path),
                    "byte_order": header.byte_order.value,
                    "version": f"{header.version.major}."
                               f"{header.version.minor}",
                    "kind": header.kind.decode("ascii", "replace"),
                    "spreads": len(doc.spreads),
                    "master_spreads": len(doc.master_spreads),
                    "pages": sum(len(s.pages) for s in doc.spreads),
                    "stories": len(doc.stories),
                    "colors": len(doc.colors),
                    "fonts": len(doc.fonts),
                    "languages": len(doc.languages),
                    "styles": len(doc.styles),
                    "warnings": doc.warnings}
        if action == "convert":
            path = _source_path(sources)
            name = str(opts.get("name") or path.name)
            try:
                if output:
                    from ..documents.indd import convert_into
                    warnings = convert_into(path.read_bytes(), name,
                                            output)
                    return {"ok": True, "output": output,
                            "bytes": Path(output).stat().st_size,
                            "warnings": warnings}
                conv = _convert_file(path)
            except InddError as exc:
                return {"ok": False, "error": str(exc),
                        "kind": exc.kind}
            return {"ok": True, "name": name,
                    "bytes": len(conv.idml),
                    "warnings": conv.warnings,
                    "previews": len(conv.previews)}
        if action == "fidelity":
            if not sources or len(sources) < 2:
                raise ValueError(
                    "inddutils fidelity needs two packages "
                    "(sources=[<reference.idml>, <package.idml>])")
            reference = Path(str(sources[0])).expanduser()
            package = Path(str(sources[1])).expanduser()
            result = _measure_fidelity(reference.read_bytes(),
                                       package.read_bytes())
            return {"ok": True, "headline": result.headline,
                    "coverage": result.coverage,
                    "total": result.total,
                    "reproduced": result.reproduced,
                    "missing": result.missing,
                    "extra": result.extra}
        raise ValueError(f"unknown inddutils action {action!r}; "
                         f"actions: {', '.join(_ACTIONS)}")
