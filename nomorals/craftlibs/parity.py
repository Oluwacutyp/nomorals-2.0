"""The shared parity runner (Craft-Libs).

``nomorals/craftlibs-parity.toml`` lists real fixtures +
operations + expected outcomes; :func:`run_parity` runs every
case and fails on any mismatch. Missing fixture = fail, never
skip. The organs' own parity modules delegate their loop here
so every organ's parity means the same thing.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

__all__ = ["load_toml", "run_cases", "run_parity"]

_DEFAULT_PATH = (Path(__file__).resolve().parents[1]
                 / "craftlibs-parity.toml")


def load_toml(path: str | Path) -> list[dict[str, Any]]:
    """The cases in a parity TOML, in file order."""
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    return list(data.get("cases", []))


def run_cases(cases, run_case=None) -> dict[str, Any]:
    """Run a list of case dicts. ``run_case`` runs one case and
    raises on mismatch; the default runs the craftlibs
    operations. One case's failure never stops the rest."""
    runner = run_case or _run_case
    passed_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        case_id = str(case.get("id", "<unnamed>"))
        try:
            runner(case)
        except Exception as exc:  # noqa: BLE001 — one case's failure
            failures.append({"id": case_id,
                             "reason": f"{type(exc).__name__}: {exc}"})
        else:
            passed_ids.append(case_id)
    return {"passed": len(passed_ids),
            "failed": [entry["id"] for entry in failures],
            "passed_ids": passed_ids,
            "failures": failures,
            "total": len(cases),
            "skipped": []}


def run_parity(source=None, run_case=None) -> dict[str, Any]:
    """Run the parity cases: the default TOML when ``source`` is
    ``None``, a TOML path, or a list of case dicts. A missing
    TOML fails, never skips."""
    if isinstance(source, (list, tuple)):
        return run_cases(list(source), run_case)
    toml_path = Path(source) if source is not None else _DEFAULT_PATH
    if not toml_path.exists():
        return {"passed": 0, "failed": ["<toml>"], "passed_ids": [],
                "failures": [{"id": "<toml>",
                              "reason": f"missing {toml_path}"}],
                "total": 0, "skipped": []}
    return run_cases(load_toml(toml_path), run_case)


def _close(got: float, want: float, tol: float = 1e-6) -> bool:
    return abs(got - want) <= tol


def _run_case(case: dict[str, Any]) -> None:
    op = str(case.get("operation", ""))
    if op == "noise_point":
        from . import noise as shared_noise
        function = getattr(shared_noise, str(case["function"]))
        point = tuple(float(v) for v in case["point"])
        seed = int(case["seed"])
        if str(case["function"]) == "point_fbm":
            base = getattr(shared_noise, str(case["base"]))
            got = function(point, int(case["octaves"]), seed, base)
        elif str(case["function"]) == "point_cellular":
            got = function(point, seed)[int(case.get("component", 0))]
        else:
            got = function(point, seed)
        if not _close(float(got), float(case["expected"]),
                      float(case.get("tol", 1e-6))):
            raise AssertionError(
                f"{case['function']}{point} = {got} "
                f"want {case['expected']}")
    elif op == "noise_grid":
        from . import noise as shared_noise
        function = getattr(shared_noise, str(case["function"]))
        args = list(case["args"])
        result = function(*args)
        index = [int(i) for i in case.get("index", [0, 0])]
        got = result
        for part in index:
            got = got[part]
        if not _close(float(got), float(case["expected"]),
                      float(case.get("tol", 1e-6))):
            raise AssertionError(
                f"{case['function']}{args} at {index} = {got} "
                f"want {case['expected']}")
    elif op == "codec":
        import json

        from . import codec as shared_codec
        document = case["document"]
        raw = json.dumps(document, sort_keys=True).encode()
        package = shared_codec.write_package(
            {"document.json": raw,
             "meta.json": b'{"format_version": 1}'})
        back = shared_codec.read_package(package)
        if back["document.json"] != raw:
            raise AssertionError("codec round trip changed the document")
        if shared_codec.write_package(back) != package:
            raise AssertionError("codec round trip is not byte-identical")
    elif op == "pdf_safe_url":
        from . import pdf as shared_pdf
        got = shared_pdf.safe_url(str(case["url"]))
        if got != bool(case["safe"]):
            raise AssertionError(
                f"safe_url({case['url']!r}) = {got} "
                f"want {case['safe']}")
    elif op == "pdf_round_trip":
        from . import pdf as shared_pdf
        meta = shared_pdf.Metadata(
            creator="craftlibs", title=str(case.get("title", "proof")))
        data = shared_pdf.write_pdf(str(case["text"]), meta)
        if not data.startswith(b"%PDF-"):
            raise AssertionError("written bytes are not a PDF")
        info = shared_pdf.read_pdf(data)
        if info["pages"] != int(case.get("pages", 1)):
            raise AssertionError(f"pages = {info['pages']}")
        if str(case.get("contains", "")) not in info["text"]:
            raise AssertionError("written text did not read back")
    elif op == "i18n_format":
        from . import i18n as shared_i18n
        fmt = shared_i18n.Format.from_tag(str(case["locale"]))
        got = fmt.number(float(case["value"]), int(case["decimals"]))
        if got != str(case["expected"]):
            raise AssertionError(
                f"format({case['locale']}, {case['value']}) = {got!r} "
                f"want {case['expected']!r}")
    elif op == "i18n_parse":
        from . import i18n as shared_i18n
        got = shared_i18n.parse_number(str(case["text"]),
                                       str(case["locale"]))
        if not math.isclose(got, float(case["expected"]),
                            rel_tol=0, abs_tol=1e-9):
            raise AssertionError(
                f"parse({case['text']!r}, {case['locale']}) = {got} "
                f"want {case['expected']}")
    elif op == "ai_suggest":
        from .ai import local as shared_local
        prompt = str(case["prompt"])
        if str(case.get("kind", "motion")) == "easing":
            got = shared_local.suggest_easing(prompt)
        else:
            got = shared_local.suggest_motion(prompt)
        if got != str(case["expected"]):
            raise AssertionError(
                f"suggest({prompt!r}) = {got!r} "
                f"want {case['expected']!r}")
    else:
        raise AssertionError(f"unknown parity operation: {op!r}")
