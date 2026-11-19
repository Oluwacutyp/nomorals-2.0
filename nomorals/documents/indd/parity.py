"""Parity runner — the machine-checked parity TOML.

Each [[case]] names a real fixture (tests/fixtures/inddutils/), the
operations applied to it, and the expected outcomes — and the runner
EXECUTES them against the real engine and reports pass/fail per case
with measured values. Parity is measured, never asserted: a case
whose expectation is wrong fails loudly, a missing fixture fails
(never a silent skip), and an operation that raises unexpectedly fails
with the exception named.

Schema (version 1):

    [schema]
    version = 1

    [[case]]
    id = "bluesquare-inspect"
    fixture = "BlueSquare.indd"
    ops = ["inspect"]
    [case.expect]
    version = "4.0"
    spreads = 1
    stories = 0

Ops: inspect, convert, fidelity, refuse, refuse_truncated.
Expect keys: version, spreads, master_spreads, pages, stories,
colors, fonts, languages, styles, sections, warnings, parts,
story_parts, text_contains (list, checked against the joined story
text of the converted package), coverage, error_kind.
"""

from __future__ import annotations

import io
import tomllib
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .convert import convert, read_document
from .errors import InddError
from .fidelity import measure_fidelity

__all__ = ["DEFAULT_PARITY_PATH", "load_parity", "run_parity"]

DEFAULT_PARITY_PATH = Path(__file__).resolve().parent.parent \
    / "inddutils-parity.toml"


def load_parity(path: str | Path | None = None) -> dict[str, Any]:
    """Load a parity TOML file (stdlib tomllib)."""
    path = Path(path) if path is not None else DEFAULT_PARITY_PATH
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise InddError(f"cannot read parity TOML {path}: {exc}",
                        kind="corrupt") from exc
    except tomllib.TOMLDecodeError as exc:
        raise InddError(f"bad parity TOML {path}: {exc}",
                        kind="corrupt") from exc
    if not isinstance(data.get("schema"), dict) \
            or data["schema"].get("version") != 1:
        raise InddError(f"parity TOML {path} is not schema version 1",
                        kind="unsupported")
    if not isinstance(data.get("case"), list):
        raise InddError(f"parity TOML {path} has no [[case]] rows",
                        kind="corrupt")
    return data


def _story_texts(idml: bytes) -> str:
    """The joined text of every Story part, in document order."""
    with zipfile.ZipFile(io.BytesIO(idml)) as zf:
        parts = {name: zf.read(name) for name in zf.namelist()}
    designmap = ET.fromstring(parts["designmap.xml"])
    order = (designmap.get("StoryList") or "").split()
    texts = []
    for uid in order:
        name = f"Stories/Story_{uid}.xml"
        if name not in parts:
            continue
        root = ET.fromstring(parts[name])
        texts.append("".join(
            node.text or "" for node in root.iter("Content")))
    return "".join(texts)


def _run_case(case: dict[str, Any], fixtures_dir: Path
              ) -> dict[str, Any]:
    """Execute one [[case]]; returns {id, status, measured, expected,
    detail}. A failure is a reported row, never a raised exception."""
    case_id = str(case.get("id", "?"))
    expected = case.get("expect", {})
    measured: dict[str, Any] = {}
    try:
        fixture = case.get("fixture", "")
        fixture_path = fixtures_dir / str(fixture)
        if not fixture_path.is_file():
            return {"id": case_id, "status": "fail",
                    "measured": {"fixture_missing": str(fixture)},
                    "expected": expected,
                    "detail": f"fixture {fixture!r} is not on disk — "
                              f"a missing fixture is a failure, never "
                              f"a silent skip"}
        data = fixture_path.read_bytes()
        ops = case.get("ops", ["inspect"])
        for op in ops:
            if op == "inspect":
                doc = read_document(data)
                measured["version"] = (
                    f"{doc.version.major}.{doc.version.minor}")
                measured["spreads"] = len(doc.spreads)
                measured["master_spreads"] = len(doc.master_spreads)
                measured["pages"] = sum(
                    len(s.pages)
                    for s in list(doc.spreads) + list(doc.master_spreads))
                measured["stories"] = len(doc.stories)
                measured["colors"] = len(doc.colors)
                measured["fonts"] = len(doc.fonts)
                measured["languages"] = len(doc.languages)
                measured["styles"] = len(doc.styles)
                measured["sections"] = len(doc.sections)
                measured["warnings"] = len(doc.warnings)
            elif op == "convert":
                conv = convert(data, str(fixture))
                with zipfile.ZipFile(io.BytesIO(conv.idml)) as zf:
                    names = zf.namelist()
                measured["parts"] = len(names)
                measured["story_parts"] = sum(
                    1 for n in names if n.startswith("Stories/"))
                measured["warnings"] = len(conv.warnings)
                text = _story_texts(conv.idml)
                for want in expected.get("text_contains", []):
                    measured.setdefault("text_contains", {})[want] = \
                        want in text
            elif op == "fidelity":
                conv = convert(data, str(fixture))
                result = measure_fidelity(conv.idml, conv.idml)
                measured["coverage"] = result.coverage
                measured["total"] = result.total
            elif op == "refuse":
                try:
                    convert(b"not an indd file at all, far too short",
                            "x.indd")
                except InddError as exc:
                    measured["error_kind"] = exc.kind
                else:
                    measured["error_kind"] = None
            elif op == "refuse_truncated":
                try:
                    convert(data[:100], "cut.indd")
                except InddError as exc:
                    measured["error_kind"] = exc.kind
                else:
                    measured["error_kind"] = None
            else:
                raise InddError(f"unknown parity op {op!r}",
                                kind="unsupported")
    except InddError as exc:
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"raised InddError kind {exc.kind!r}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the runner reports, never raises
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": f"unexpected {type(exc).__name__}: {exc}"}
    failures: list[str] = []
    for key in ("version", "spreads", "master_spreads", "pages",
                "stories", "colors", "fonts", "languages", "styles",
                "sections", "warnings", "parts", "story_parts",
                "coverage", "total", "error_kind"):
        if key in expected and measured.get(key) != expected[key]:
            failures.append(f"{key} {measured.get(key)!r} != "
                            f"{expected[key]!r}")
    for want, found in measured.get("text_contains", {}).items():
        if not found:
            failures.append(f"text {want!r} missing from the stories")
    if failures:
        return {"id": case_id, "status": "fail",
                "measured": measured, "expected": expected,
                "detail": "; ".join(failures)}
    return {"id": case_id, "status": "pass",
            "measured": measured, "expected": expected,
            "detail": "all expectations measured, not asserted"}


def run_parity(path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None
               ) -> dict[str, Any]:
    """Run every [[case]] in a parity TOML; returns {passed, failed,
    skipped, cases}. ``fixtures_dir`` defaults to the test fixtures
    directory (``tests/fixtures/inddutils`` relative to the repo
    root)."""
    data = load_parity(path)
    if fixtures_dir is None:
        fixtures_dir = (Path(__file__).resolve().parent.parent.parent
                        .parent / "tests" / "fixtures" / "inddutils")
    fixtures = Path(fixtures_dir)
    cases = [_run_case(case, fixtures) for case in data["case"]]
    passed = [c for c in cases if c["status"] == "pass"]
    failed = [c for c in cases if c["status"] != "pass"]
    return {"passed": len(passed), "failed": failed,
            "skipped": [], "cases": cases}
