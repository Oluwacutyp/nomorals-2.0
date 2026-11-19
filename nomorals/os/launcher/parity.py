"""Machine-checked parity: the launcher's promises, run for real.

Each case in ``launcher-parity.toml`` names a fixture package and the
outcome the pipeline must produce for it. A missing fixture is a
failed case, never a skip — the TOML is the contract and the runner
is the proof.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from .catalog import Catalog, Component, Source, default_catalog
from .errors import InstallError, TrustError, VerificationError
from .package import read_package, verify_package
from .pipeline import LauncherPipeline
from .version import Version

__all__ = ["run_parity"]

_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent.parent / \
    "tests" / "fixtures" / "craftlauncher"


def _toml_path() -> Path:
    return Path(__file__).resolve().parent.parent / "launcher-parity.toml"


def _installed_tree(root: Path, component: str, version: str) -> None:
    """An installed component tree at ``version`` (the parity baseline)."""
    comp = root / component
    comp.mkdir(parents=True, exist_ok=True)
    (comp / "__init__.py").write_text(
        f'"""FakeCraft component."""\n\n__version__ = "{version}"\n',
        encoding="utf-8")
    (comp / "engine.py").write_text(
        f"VERSION = \"{version}\"\n\n\ndef run():\n"
        f"    return \"fakecraft-{version}\"\n",
        encoding="utf-8")
    (comp / ".launcher-version").write_text(
        json.dumps({"component": component, "version": version,
                    "source": "devon-official"}, indent=2),
        encoding="utf-8")


def _catalog_for(fixture_sha: str, *, revoked: bool = False) -> Catalog:
    """A catalog that trusts the fixture's hash (and revokes it if asked)."""
    catalog = default_catalog()
    catalog.components.append(
        Component(id="fakecraft", name="FakeCraft",
                  source="devon-official"))
    if revoked:
        catalog.revoke(fixture_sha, reason="parity revocation")
    return catalog


def _run_case(case: dict[str, Any], fixtures_dir: Path,
              work: Path) -> tuple[bool, str]:
    kind = str(case.get("kind", ""))
    fixture_name = str(case.get("fixture", ""))
    fixture = fixtures_dir / fixture_name if fixture_name else None
    if fixture_name and (fixture is None or not fixture.is_file()):
        return False, f"missing fixture: {fixture_name}"
    expect = case.get("expect", {})

    if kind == "version_compare":
        got = Version.parse(str(case["a"])) < Version.parse(str(case["b"]))
        return (got == bool(expect.get("less")), "version comparison")
    if kind == "version_stable":
        from .version import is_stable

        got = is_stable(Version.parse(str(case["version"])))
        return (got == bool(expect.get("stable")), "stability")
    if kind == "trust":
        catalog = Catalog(
            components=[Component(id="fakecraft", name="FakeCraft",
                                  source="community-x", community=True)],
            sources=[Source(id="community-x", kind="community",
                            repo="someone/x",
                            allowed_components=("fakecraft",))])
        verdict = catalog.check_trust(
            "fakecraft", "community-x", package_sha256="0" * 64)
        return (verdict.ok == bool(expect.get("ok")),
                f"trust: {verdict.reason}")

    # Package + pipeline cases run against a scratch install root.
    assert fixture is not None
    import hashlib

    fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    catalog = _catalog_for(
        fixture_sha, revoked=bool(case.get("revoked")))
    if kind == "package_verify":
        pkg = read_package(fixture)
        verdict = verify_package(pkg, catalog=catalog, devon_version="1.0.0")
        if bool(expect.get("ok")):
            return (verdict.ok, "; ".join(verdict.failures))
        needle = str(expect.get("failure_contains", ""))
        return (not verdict.ok and any(needle in f for f in verdict.failures),
                "; ".join(verdict.failures))
    if kind in ("install", "install_refused", "install_rollback"):
        root = work / "root"
        home = work / "home"
        _installed_tree(root, "fakecraft", "1.0.0")
        health_checks = None
        if kind == "install_rollback":
            def _failing() -> dict[str, Any]:
                return {"ok": False, "detail": "parity health failure"}

            health_checks = [_failing]
        pipe = LauncherPipeline(root=root, home=home, catalog=catalog,
                                health_checks=health_checks,
                                devon_version="1.0.0")
        if kind == "install":
            result = pipe.install(fixture)
            installed = pipe.installed_version("fakecraft")
            return (result.ok and installed == expect.get("version"),
                    f"installed={installed} ok={result.ok}")
        if kind == "install_refused":
            error_name = str(expect.get("error", "VerificationError"))
            try:
                pipe.install(fixture)
            except (VerificationError, TrustError, InstallError) as exc:
                installed = pipe.installed_version("fakecraft")
                return (type(exc).__name__ == error_name
                        and installed == "1.0.0",
                        f"raised {type(exc).__name__}, installed={installed}")
            return False, "install did not refuse"
        result = pipe.install(fixture)
        installed = pipe.installed_version("fakecraft")
        return (result.ok is False and result.rolled_back
                and installed == "1.0.0",
                f"ok={result.ok} rolled_back={result.rolled_back} "
                f"installed={installed}")
    return False, f"unknown parity kind: {kind}"


def run_parity(toml_path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None) -> dict[str, Any]:
    """Run every case in the parity TOML. Missing fixture = failed case."""
    path = Path(toml_path) if toml_path is not None else _toml_path()
    fixtures = Path(fixtures_dir) if fixtures_dir is not None else _FIXTURE_DIR
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    cases = data.get("case", [])
    passed: list[str] = []
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="launcher-parity-") as tmp:
        work = Path(tmp)
        for index, case in enumerate(cases):
            case_id = str(case.get("id", f"case-{index}"))
            try:
                ok, detail = _run_case(case, fixtures, work / case_id)
            except Exception as exc:  # noqa: BLE001 - a raising case fails
                ok, detail = False, f"raised {type(exc).__name__}: {exc}"
            if ok:
                passed.append(case_id)
            else:
                failed.append(f"{case_id}: {detail}")
    return {"passed": len(passed), "failed": failed, "total": len(cases),
            "passed_ids": passed}
