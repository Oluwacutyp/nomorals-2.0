"""Ruff lint-debt ratchet (Echelon Wave 8).

The repo carries a large standing ruff debt (measured in the thousands),
so a hard ``ruff check`` gate would be red on day one and train everyone
to ignore CI.  The ratchet is the honest middle shape, in the
``tests/test_layering.py`` idiom: the current findings are recorded per
rule code in ``ruff-ratchet.json`` at the repo root, and the gate fails
when

* any rule code's live count rises ABOVE its recorded ceiling (new debt),
* a rule code appears that the ratchet does not record at all (new debt
  class), or
* the recorded ceiling is HIGHER than the live count (debt was paid down
  but the ratchet was not tightened — the record must only ever shrink,
  exactly like the layering test's grandfathered-violation list).

Counts are only comparable for one ruff toolchain: rule sets grow
between ruff releases, so the ratchet records the ruff version it was
measured with and CI pins that version.  A run under a different ruff
version reports ``toolchain_drift`` instead of a verdict — re-measure
with the pinned toolchain rather than arguing with the numbers.

When ruff is absent the check returns ``ok=False`` with
``ruff_installed=False`` — an explicit "cannot measure", never a silent
pass (the ``tools/lint.py`` contract).

Paying debt down: fix findings, then run
``python3 -m nomorals.tools.ruff_ratchet --update`` and commit the
tightened ``ruff-ratchet.json`` with the fixes.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RATCHET_FILENAME",
    "DEFAULT_PATHS",
    "ratchet_path",
    "ruff_available",
    "ruff_version",
    "count_findings",
    "load_ratchet",
    "check_ratchet",
    "update_ratchet",
    "register",
    "main",
]

#: Ratchet data file, at the repo root beside ``pyproject.toml``.
RATCHET_FILENAME = "ruff-ratchet.json"

#: The paths CI lints — the ratchet measures exactly what CI gates.
DEFAULT_PATHS: tuple[str, ...] = ("nomorals/", "tests/")


def repo_root(start: Path | None = None) -> Path:
    """Repo root for a checkout (the parent of the ``nomorals`` package)."""
    here = (start or Path(__file__)).resolve()
    return here.parents[2]


def ratchet_path(repo: str | Path | None = None) -> Path:
    root = Path(repo).resolve() if repo else repo_root()
    return root / RATCHET_FILENAME


def ruff_available() -> bool:
    """True when ruff is importable or on PATH (mirrors tools/lint.py)."""
    if importlib.util.find_spec("ruff") is not None:
        return True
    return shutil.which("ruff") is not None


def _ruff_cmd() -> list[str]:
    if shutil.which("ruff"):
        return ["ruff"]
    return [sys.executable, "-m", "ruff"]


def ruff_version() -> str:
    """The running ruff's version string ('' when unavailable)."""
    if not ruff_available():
        return ""
    try:
        proc = subprocess.run(
            [*_ruff_cmd(), "--version"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    out = (proc.stdout or "").strip()
    # "ruff 0.17.3" → "0.17.3"
    return out.split()[-1] if out else ""


def count_findings(
    repo: str | Path | None = None,
    paths: tuple[str, ...] | list[str] = DEFAULT_PATHS,
) -> dict[str, Any]:
    """Run ruff over ``paths`` and count findings per rule code.

    Returns comparable data: ``{ruff_installed, ruff_version, total,
    by_code}``.  ``by_code`` keys are ruff rule codes (``"F401"``);
    a syntax failure ruff reports without a code is keyed
    ``"invalid-syntax"`` (ruff ≥0.17 emits that code itself).
    """
    root = Path(repo).resolve() if repo else repo_root()
    if not ruff_available():
        return {"ruff_installed": False, "ruff_version": "",
                "total": 0, "by_code": {}}
    proc = subprocess.run(
        [*_ruff_cmd(), "check", *paths, "--output-format", "json"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900,
    )
    try:
        findings = json.loads(proc.stdout or "[]")
    except ValueError:
        _log.error("ruff produced unparseable output (rc=%s): %s",
                   proc.returncode, (proc.stderr or "")[:400])
        raise
    by_code: dict[str, int] = {}
    for item in findings:
        code = item.get("code") or "invalid-syntax"
        by_code[code] = by_code.get(code, 0) + 1
    return {
        "ruff_installed": True,
        "ruff_version": ruff_version(),
        "total": sum(by_code.values()),
        "by_code": dict(sorted(by_code.items())),
    }


def load_ratchet(repo: str | Path | None = None) -> dict[str, Any]:
    """The recorded ratchet ({} when the file is absent)."""
    path = ratchet_path(repo)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check_ratchet(
    repo: str | Path | None = None,
    paths: tuple[str, ...] | list[str] = DEFAULT_PATHS,
) -> dict[str, Any]:
    """Compare live ruff counts against the recorded ratchet.

    The verdict is data: ``ok`` is True only when ruff is installed,
    the toolchain matches the recorded one, no code overruns its
    ceiling, no unrecorded code appears, and no ceiling is stale
    (recorded above live — debt paid but ratchet not tightened).
    """
    recorded = load_ratchet(repo)
    live = count_findings(repo, paths)
    result: dict[str, Any] = {
        "ok": False,
        "ruff_installed": live["ruff_installed"],
        "ruff_version": live["ruff_version"],
        "recorded_ruff_version": str(recorded.get("ruff_version", "")),
        "total": live["total"],
        "recorded_total": int(recorded.get("total", 0)),
        "by_code": live["by_code"],
        "recorded_by_code": recorded.get("by_code", {}),
        "overruns": [],
        "unrecorded": [],
        "stale_ceilings": [],
        "toolchain_drift": False,
        "ratchet_file": str(ratchet_path(repo)),
    }
    if not live["ruff_installed"]:
        result["detail"] = "ruff is not installed — the ratchet cannot be measured"
        return result
    if not recorded:
        result["detail"] = f"no ratchet recorded at {result['ratchet_file']}"
        return result
    if live["ruff_version"] != result["recorded_ruff_version"]:
        result["toolchain_drift"] = True
        result["detail"] = (
            f"ruff {live['ruff_version']} != the ratchet's recorded "
            f"{result['recorded_ruff_version']} — counts are not comparable "
            "across ruff releases; re-measure with the pinned toolchain")
        return result
    recorded_by: dict[str, int] = {
        str(k): int(v) for k, v in result["recorded_by_code"].items()}
    for code, count in live["by_code"].items():
        if code not in recorded_by:
            result["unrecorded"].append(
                {"code": code, "recorded": 0, "actual": count})
        elif count > recorded_by[code]:
            result["overruns"].append(
                {"code": code, "recorded": recorded_by[code],
                 "actual": count})
    for code, ceiling in recorded_by.items():
        actual = live["by_code"].get(code, 0)
        if ceiling > actual:
            result["stale_ceilings"].append(
                {"code": code, "recorded": ceiling, "actual": actual})
    result["ok"] = not (result["overruns"] or result["unrecorded"]
                        or result["stale_ceilings"])
    if result["ok"]:
        result["detail"] = (
            f"ruff debt at the recorded ceiling: {live['total']} findings "
            f"across {len(live['by_code'])} rule codes")
    else:
        bits = []
        if result["overruns"]:
            bits.append("overruns: " + ", ".join(
                f"{o['code']} {o['recorded']}→{o['actual']}"
                for o in result["overruns"]))
        if result["unrecorded"]:
            bits.append("unrecorded codes: " + ", ".join(
                f"{o['code']} ({o['actual']})" for o in result["unrecorded"]))
        if result["stale_ceilings"]:
            bits.append("stale ceilings (tighten with --update): "
                        + ", ".join(
                            f"{o['code']} {o['recorded']}→{o['actual']}"
                            for o in result["stale_ceilings"]))
        result["detail"] = "; ".join(bits)
    return result


def update_ratchet(
    repo: str | Path | None = None,
    paths: tuple[str, ...] | list[str] = DEFAULT_PATHS,
) -> dict[str, Any]:
    """Re-record the ratchet at the live counts (the paydown action).

    Refuses under toolchain drift against an existing ratchet: the new
    record would silently bless a different rule set.
    """
    live = count_findings(repo, paths)
    if not live["ruff_installed"]:
        return {"ok": False, "detail": "ruff is not installed"}
    existing = load_ratchet(repo)
    if existing and str(existing.get("ruff_version", "")) \
            != live["ruff_version"]:
        return {"ok": False,
                "detail": ("existing ratchet was measured with ruff "
                           f"{existing.get('ruff_version')}; running ruff "
                           f"is {live['ruff_version']} — update the pin "
                           "deliberately, then re-record")}
    payload = {
        "ruff_version": live["ruff_version"],
        "paths": list(paths),
        "total": live["total"],
        "by_code": live["by_code"],
        "note": ("Lint-debt ratchet (Echelon Wave 8): per-code ceilings "
                 "may only decrease. Re-recorded by "
                 "`python3 -m nomorals.tools.ruff_ratchet --update`."),
    }
    path = ratchet_path(repo)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return {"ok": True, "ratchet_file": str(path), **payload}


def register(registry: Any) -> None:
    """Attach the ratchet check to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "ruff_ratchet",
        description=("Ruff lint-debt ratchet: compares live ruff findings "
                     "per rule code against ruff-ratchet.json. Returns "
                     "{ok, total, recorded_total, overruns, unrecorded, "
                     "stale_ceilings, toolchain_drift, ruff_installed}. "
                     "ok=False with ruff_installed=False when ruff is "
                     "absent (never a silent pass)."),
        capability=Capability.EXEC_SHELL,
    )
    def _ruff_ratchet(repo: str | None = None) -> dict[str, Any]:
        return check_ratchet(repo=repo)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="nomorals.tools.ruff_ratchet",
        description="Ruff lint-debt ratchet gate.")
    parser.add_argument("--repo", default=None,
                        help="repo root (default: this checkout)")
    parser.add_argument("--update", action="store_true",
                        help="re-record the ratchet at the live counts")
    args = parser.parse_args(argv)
    if args.update:
        result = update_ratchet(repo=args.repo)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 2
    result = check_ratchet(repo=args.repo)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ruff_installed"] or result["toolchain_drift"]:
        return 2
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
