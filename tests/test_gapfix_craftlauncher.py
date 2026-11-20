"""Craft-Launcher port (census §3A.19): the self-update path — a
verify-then-install pipeline over a catalog with trust rules.

Failing-first: every test here names a real behaviour of the port.
Devon already has transactional ``nm update`` (git pull + snapshot +
rollback + health gates in nomorals/os/update.py); what it does NOT
have — the gap this port fills — is a package-level verify-before-
install and a catalog of update sources with trust rules. That is what
these tests pin down, following the upstream ArtCraft Launcher
(crates/core catalog/release/version, crates/native install/source/
pins/self_update, craft_update's verify-then-install shape).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.os.launcher.attribution import verify_attribution
from nomorals.os.launcher.catalog import (
    Catalog,
    Component,
    Source,
    VersionTrack,
    choose_version,
    default_catalog,
)
from nomorals.os.launcher.errors import (
    InstallError,
    TrustError,
    VerificationError,
)
from nomorals.os.launcher.journal import Journal
from nomorals.os.launcher.package import (
    build_package,
    read_package,
    verify_package,
)
from nomorals.os.launcher.parity import run_parity
from nomorals.os.launcher.pipeline import LauncherPipeline
from nomorals.os.launcher.version import Version, is_newer, is_stable
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "craftlauncher"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_component(root: Path, version: str = "1.0.0",
                    component: str = "fakecraft") -> Path:
    """A fake installed component: a package dir with real files.

    The component dir follows the pipeline's layout: a dotted id
    becomes nested dirs (nomorals.media.cad -> nomorals/media/cad).
    """
    comp = root / Path(*component.split("."))
    comp.mkdir(parents=True, exist_ok=True)
    (comp / "__init__.py").write_text(
        f'"""FakeCraft component."""\n\n__version__ = "{version}"\n', encoding="utf-8"
    )
    (comp / "engine.py").write_text(
        f"VERSION = \"{version}\"\n\n\ndef run():\n    return \"fakecraft-{version}\"\n",
        encoding="utf-8",
    )
    (comp / ".launcher-version").write_text(
        json.dumps({"component": component, "version": version,
                    "source": "devon-official"}, indent=2),
        encoding="utf-8",
    )
    return comp


def _component_files(root: Path, version: str) -> dict[str, bytes]:
    return {
        "__init__.py": (
            f'"""FakeCraft component."""\n\n__version__ = "{version}"\n'
        ).encode(),
        "engine.py": (
            f"VERSION = \"{version}\"\n\n\ndef run():\n"
            f"    return \"fakecraft-{version}\"\n"
        ).encode(),
    }


def _tree_snapshot(root: Path) -> dict[str, str]:
    snap = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snap[str(path.relative_to(root))] = _sha(path.read_bytes())
    return snap


def _registry() -> ToolRegistry:
    registry = ToolRegistry(SimpleNamespace())
    registry.register_builtins()
    return registry


# ---------------------------------------------------------------------------
# Version (upstream crates/core/src/version.rs)
# ---------------------------------------------------------------------------


def test_version_parse_compare() -> None:
    assert Version.parse("1.2.3") < Version.parse("1.10.0")
    assert Version.parse("2.0.0") > Version.parse("1.99.99")
    assert Version.parse("1.2.3") == Version.parse("1.2.3")
    assert Version.parse("1.2.3-beta") < Version.parse("1.2.3")
    assert is_newer("1.10.0", "1.2.3") is True
    assert is_newer("1.2.3", "1.10.0") is False


def test_version_stable() -> None:
    assert is_stable(Version.parse("2.0.0")) is True
    assert is_stable(Version.parse("2.0.0-beta")) is False
    assert is_stable(Version.parse("2.0.0-nightly.20261119")) is False


def test_version_invalid_refused() -> None:
    with pytest.raises(ValueError):
        Version.parse("banana")
    with pytest.raises(ValueError):
        Version.parse("1.2")
    with pytest.raises(ValueError):
        Version.parse("")


# ---------------------------------------------------------------------------
# Catalog + trust rules (upstream crates/core/src/catalog.rs,
# crates/native/src/source.rs, crates/native/src/pins.rs)
# ---------------------------------------------------------------------------


def test_catalog_seeds_real_devon_components() -> None:
    catalog = default_catalog()
    ids = {c.id for c in catalog.components}
    # The eight craft organs the launcher knows how to update, plus the
    # core runtime and the OS kernel/supervisor — real Devon components.
    for wanted in (
        "nomorals.media.cad", "nomorals.media.material",
        "nomorals.media.textures", "nomorals.media.animation",
        "nomorals.media.design", "nomorals.media.vector",
        "nomorals.media.sound", "nomorals.media.layout",
        "nomorals.core", "nomorals.os.kernel",
    ):
        assert wanted in ids, wanted
    assert catalog.source("devon-official").kind == "official"
    assert catalog.source("devon-official").repo == (
        "Oluwacutyp/nomorals-2.0")


def test_catalog_official_allows_all_components() -> None:
    catalog = default_catalog()
    for component in catalog.components:
        verdict = catalog.check_trust(
            component.id, "devon-official", package_sha256="a" * 64)
        assert verdict.ok, (component.id, verdict.reason)


def test_catalog_unknown_source_refused() -> None:
    catalog = default_catalog()
    verdict = catalog.check_trust(
        "nomorals.media.cad", "some-random-fork", package_sha256="a" * 64)
    assert verdict.ok is False
    assert "unknown source" in verdict.reason


def test_catalog_community_limited_to_its_components() -> None:
    catalog = Catalog(
        components=[Component(id="nomorals.media.cad", name="CADCraft",
                              source="devon-official"),
                    Component(id="solidcraft", name="SolidCraft",
                              source="community-solid", community=True)],
        sources=[Source(id="devon-official", kind="official",
                        repo="Oluwacutyp/nomorals-2.0"),
                 Source(id="community-solid", kind="community",
                        repo="someone/solidcraft",
                        allowed_components=("solidcraft",))],
    )
    assert catalog.check_trust(
        "solidcraft", "community-solid", package_sha256="b" * 64).ok is False
    # A community package with NO pinned hash is refused outright: the
    # hash pin IS the trust for a community source.
    catalog.pin("solidcraft", "b" * 64)
    assert catalog.check_trust(
        "solidcraft", "community-solid", package_sha256="b" * 64).ok is True
    assert catalog.check_trust(
        "nomorals.media.cad", "community-solid",
        package_sha256="b" * 64).ok is False


def test_catalog_revoked_hash_refused() -> None:
    catalog = default_catalog()
    catalog.revoke("c" * 64, reason="compromised build")
    verdict = catalog.check_trust(
        "nomorals.media.cad", "devon-official", package_sha256="c" * 64)
    assert verdict.ok is False
    assert "revoked" in verdict.reason


def test_choose_version_tracks() -> None:
    releases = [
        {"tag": "v1.0.0", "version": "1.0.0", "prerelease": False,
         "assets": [{"name": "a.zip", "sha256": "x" * 64}]},
        {"tag": "v2.0.0", "version": "2.0.0", "prerelease": False,
         "assets": [{"name": "a.zip", "sha256": "y" * 64}]},
        {"tag": "v2.1.0-nightly", "version": "2.1.0-nightly.20261119",
         "prerelease": True,
         "assets": [{"name": "a.zip", "sha256": "z" * 64}]},
    ]
    assert choose_version(releases, VersionTrack.STABLE)["tag"] == "v2.0.0"
    assert choose_version(
        releases, VersionTrack.NIGHTLY)["tag"] == "v2.1.0-nightly"
    assert choose_version(
        releases, VersionTrack.pinned("v1.0.0"))["tag"] == "v1.0.0"
    assert choose_version(releases, VersionTrack.pinned("v9.9.9")) is None


# ---------------------------------------------------------------------------
# Package build / read / verify (upstream crates/native/src/install.rs
# download() + published_checksum, crates/native/src/manifest.rs)
# ---------------------------------------------------------------------------


def test_package_build_verify_ok(tmp_path: Path) -> None:
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pkg = read_package(pkg_path)
    assert pkg.component == "fakecraft"
    assert str(pkg.version) == "2.0.0"
    verdict = verify_package(pkg, catalog=_catalog_with_fake(),
                             devon_version="1.0.0")
    assert verdict.ok, verdict.failures
    assert verdict.package_sha256 == _sha(pkg_path.read_bytes())


def test_package_tampered_file_refused(tmp_path: Path) -> None:
    good = build_package(
        tmp_path / "good.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    bad = tmp_path / "tampered.zip"
    shutil.copy(good, bad)
    with zipfile.ZipFile(bad, "a") as zf:
        zf.writestr("engine.py", "VERSION = \"2.0.0\"\n\n\ndef run():\n    return \"pwned\"\n")
    pkg = read_package(bad)
    verdict = verify_package(pkg, catalog=default_catalog(),
                             devon_version="1.0.0")
    assert verdict.ok is False
    assert any("engine.py" in f for f in verdict.failures)


def test_package_missing_manifest_refused(tmp_path: Path) -> None:
    path = tmp_path / "nomanifest.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("engine.py", "x = 1\n")
    with pytest.raises(VerificationError, match="manifest"):
        read_package(path)


def test_package_bad_manifest_refused(tmp_path: Path) -> None:
    path = tmp_path / "badmanifest.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "nope"}))
    with pytest.raises(VerificationError, match="manifest"):
        read_package(path)


def test_package_incompatible_refused(tmp_path: Path) -> None:
    pkg_path = build_package(
        tmp_path / "incompat.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="99.0.0")
    pkg = read_package(pkg_path)
    verdict = verify_package(pkg, catalog=default_catalog(),
                             devon_version="1.0.0")
    assert verdict.ok is False
    assert any("99.0.0" in f for f in verdict.failures)


def test_package_extra_file_refused(tmp_path: Path) -> None:
    pkg_path = build_package(
        tmp_path / "extra.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    with zipfile.ZipFile(pkg_path, "a") as zf:
        zf.writestr("surprise.sh", "#!/bin/sh\nrm -rf /\n")
    pkg = read_package(pkg_path)
    verdict = verify_package(pkg, catalog=default_catalog(),
                             devon_version="1.0.0")
    assert verdict.ok is False
    assert any("surprise.sh" in f for f in verdict.failures)


def test_package_path_escape_refused(tmp_path: Path) -> None:
    with pytest.raises(VerificationError, match="outside the component"):
        build_package(
            tmp_path / "escape.zip", component="fakecraft", version="2.0.0",
            source="devon-official",
            files={"../evil.py": b"x = 1\n"}, devon_min="0.1.0")


def test_package_corrupt_zip_refused(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"this is not a zip at all")
    with pytest.raises(VerificationError, match="zip"):
        read_package(path)


def test_package_attestation_pin_checked(tmp_path: Path) -> None:
    pkg_path = build_package(
        tmp_path / "attested.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0", attestation="attest-123")
    catalog = _catalog_with_fake()
    catalog.pin_attestation("fakecraft", "devon-official", "attest-123")
    pkg = read_package(pkg_path)
    assert verify_package(pkg, catalog=catalog,
                          devon_version="1.0.0").ok is True
    catalog.pin_attestation("fakecraft", "devon-official", "attest-OTHER")
    verdict = verify_package(pkg, catalog=catalog, devon_version="1.0.0")
    assert verdict.ok is False
    assert any("attestation" in f for f in verdict.failures)


# ---------------------------------------------------------------------------
# Pipeline: verify-then-install, atomic swap, health-gated rollback
# (upstream crates/native/src/install.rs swap_into_place,
# crates/native/src/self_update.rs, craft_update self_update)
# ---------------------------------------------------------------------------


def _catalog_with_fake() -> Catalog:
    """The default catalog plus the scratch fakecraft component."""
    catalog = default_catalog()
    catalog.components.append(
        Component(id="fakecraft", name="FakeCraft",
                  source="devon-official"))
    return catalog


def _pipeline(tmp_path: Path, **kwargs) -> LauncherPipeline:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    catalog = default_catalog()
    # The scratch component the pipeline tests install; the default
    # catalog seeds only Devon's real components.
    catalog.components.append(
        Component(id="fakecraft", name="FakeCraft",
                  source="devon-official"))
    return LauncherPipeline(root=tmp_path / "root", home=home,
                            catalog=catalog, **kwargs)


def test_pipeline_install_verified(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    result = pipe.install(pkg_path)
    assert result.ok is True
    assert result.installed_version == "2.0.0"
    assert result.rolled_back is False
    assert (root / "fakecraft" / "engine.py").read_text(
        encoding="utf-8").count("2.0.0") == 2
    version_file = json.loads((root / "fakecraft" / ".launcher-version")
                              .read_text(encoding="utf-8"))
    assert version_file["version"] == "2.0.0"
    assert version_file["package_sha256"] == _sha(pkg_path.read_bytes())
    assert result.health_before["ok"] is True
    assert result.health_after["ok"] is True


def test_pipeline_install_unverified_refused_leaves_tree(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    before = _tree_snapshot(root)
    good = build_package(
        tmp_path / "good.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    bad = tmp_path / "tampered.zip"
    shutil.copy(good, bad)
    with zipfile.ZipFile(bad, "a") as zf:
        zf.writestr("engine.py", "VERSION = \"2.0.0\"\n\n\ndef run():\n    return \"pwned\"\n")
    pipe = _pipeline(tmp_path)
    with pytest.raises(VerificationError):
        pipe.install(bad)
    assert _tree_snapshot(root) == before
    journal = Journal(pipe.home / "launcher-journal.jsonl")
    assert journal.entries()[-1]["event"] == "refused"
    assert journal.entries()[-1]["verification"]["ok"] is False


def test_pipeline_install_health_failure_rolls_back(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    before = _tree_snapshot(root)
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")

    def _failing_health() -> dict:
        return {"ok": False, "detail": "component does not import"}

    pipe = _pipeline(tmp_path, health_checks=[_failing_health])
    result = pipe.install(pkg_path)
    assert result.ok is False
    assert result.rolled_back is True
    assert _tree_snapshot(root) == before
    journal = Journal(pipe.home / "launcher-journal.jsonl")
    last = journal.entries()[-1]
    assert last["event"] == "rolled_back"
    assert last["health_after"]["ok"] is False


def test_pipeline_install_untrusted_source_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    before = _tree_snapshot(root)
    pkg_path = build_package(
        tmp_path / "untrusted.zip", component="fakecraft", version="2.0.0",
        source="some-random-fork", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    with pytest.raises(TrustError, match="unknown source"):
        pipe.install(pkg_path)
    assert _tree_snapshot(root) == before


def test_pipeline_downgrade_refused_by_default(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "2.0.0")
    pkg_path = build_package(
        tmp_path / "older.zip", component="fakecraft", version="1.0.0",
        source="devon-official", files=_component_files(tmp_path, "1.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    with pytest.raises(InstallError, match="downgrade"):
        pipe.install(pkg_path)
    result = pipe.install(pkg_path, allow_downgrade=True)
    assert result.ok is True
    assert result.installed_version == "1.0.0"


def test_pipeline_restart_intent_for_restart_component(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pkg_path = build_package(
        tmp_path / "restart.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0", restart_required=True)
    pipe = _pipeline(tmp_path)
    result = pipe.install(pkg_path)
    assert result.ok is True
    assert result.restart_required is True
    intent = json.loads((pipe.home / "update-restart-intent.json")
                        .read_text(encoding="utf-8"))
    assert intent["component"] == "fakecraft"
    assert intent["version"] == "2.0.0"


def test_pipeline_rollback_restores_after_swap_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    before = _tree_snapshot(root)
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    # Fault the pipeline's rename step (the same seam upstream's
    # PRETEND_VAR serves for self-update): the swap fails before
    # anything moves, and the old tree must come back byte-identical.
    import nomorals.os.launcher.pipeline as pipeline_mod

    def _failing_rename(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(pipeline_mod, "_rename", _failing_rename)
    result = pipe.install(pkg_path)
    assert result.ok is False
    assert _tree_snapshot(root) == before


def test_pipeline_rollback_action_restores_previous(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    assert pipe.install(pkg_path).ok is True
    assert pipe.installed_version("fakecraft") == "2.0.0"
    rolled = pipe.rollback("fakecraft")
    assert rolled["ok"] is True
    assert rolled["version"] == "1.0.0"
    assert pipe.installed_version("fakecraft") == "1.0.0"
    assert "fakecraft-1.0.0" in (root / "fakecraft" / "engine.py").read_text(
        encoding="utf-8")


def test_pipeline_journal_records_every_attempt(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pkg_path = build_package(
        tmp_path / "valid.zip", component="fakecraft", version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    pipe = _pipeline(tmp_path)
    pipe.install(pkg_path)
    entries = Journal(pipe.home / "launcher-journal.jsonl").entries()
    assert [e["event"] for e in entries] == ["verified", "installed"]
    installed = entries[-1]
    assert installed["component"] == "fakecraft"
    assert installed["version"] == "2.0.0"
    assert installed["health_before"]["ok"] is True
    assert installed["health_after"]["ok"] is True
    assert installed["package_sha256"] == _sha(pkg_path.read_bytes())


def test_pipeline_check_lists_available(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pipe = _pipeline(tmp_path)
    releases = [{"tag": "v2.0.0", "version": "2.0.0", "prerelease": False,
                 "assets": [{"name": "fakecraft-2.0.0.zip",
                             "sha256": "d" * 64}]}]
    available = pipe.check("fakecraft", releases)
    assert available["current"] == "1.0.0"
    assert available["latest"] == "2.0.0"
    assert available["update_available"] is True


def test_pipeline_status_reports_installed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    pipe = _pipeline(tmp_path)
    status = pipe.status("fakecraft")
    assert status["installed"] is True
    assert status["version"] == "1.0.0"
    assert pipe.status("no-such-component")["installed"] is False


# ---------------------------------------------------------------------------
# Tool surface + registry pin
# ---------------------------------------------------------------------------


def test_tool_registered_and_counts_778() -> None:
    registry = _registry()
    spec = registry.get("craftlauncher")
    assert spec is not None
    assert "verify" in spec.description
    # 730 after the anicraft port (lane 110), 731 after the
    # craft-anything port (lane 111), 732 after the craft-launcher
    # port (lane 112) added the craftlauncher tool, 733 after the
    # craft-libs port (lane 113) added the craftlibs tool, 734 after
    # the craft-fonts port (lane 114) added the craftfonts tool.
    # 735 after the indd-utils port (lane 117) added the inddutils tool.
    # 748 after the auth-plugin wiring added 5 tools (session_forge, session_verify, payload_png_bomb, payload_svg_bomb, payload_keylogger) on top of the bughunter port (lane 121). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall.
    assert len(registry._tools) == 784


def _call(registry, **kwargs):
    # registry.call returns an Outcome, not the dict — unwrap .value
    # the same way the cadcraft test does.
    outcome = registry.call("craftlauncher", **kwargs)
    if hasattr(outcome, "value"):
        return outcome.value
    return outcome


def test_tool_catalog_and_verify_actions(tmp_path: Path) -> None:
    registry = _registry()
    catalog = _call(registry, action="catalog")
    assert "nomorals.media.cad" in str(catalog)

    # Verify against a real catalog component: the trust rules refuse
    # packages for components the catalog does not know.
    pkg_path = build_package(
        tmp_path / "valid.zip", component="nomorals.media.cad",
        version="2.0.0", source="devon-official",
        files=_component_files(tmp_path, "2.0.0"), devon_min="0.1.0")
    verdict = _call(registry, action="verify", sources=[str(pkg_path)])
    assert verdict["ok"] is True
    assert verdict["component"] == "nomorals.media.cad"


# ---------------------------------------------------------------------------
# Parity + attribution
# ---------------------------------------------------------------------------


def test_parity_runs_all_cases() -> None:
    result = run_parity()
    assert result["passed"] == result["total"]
    assert result["failed"] == []


def test_attribution_verifies() -> None:
    result = verify_attribution()
    assert result["failed"] == 0, result
    assert result["checked"] > 0


def test_fixtures_present() -> None:
    # A missing fixture is a failure, never a silent skip.
    for name in ("valid.zip", "tampered.zip", "incompatible.zip",
                 "untrusted.zip", "revoked.zip", "corrupt.zip",
                 "invalid-manifest.zip"):
        assert (FIXTURES / name).is_file(), name


# ---------------------------------------------------------------------------
# CLI extension: nm update --package
# ---------------------------------------------------------------------------


def test_cli_package_install(tmp_path: Path) -> None:
    import argparse

    from nomorals.cmdline.commands.update import _cmd_update

    # The CLI uses the real default catalog, so the package must be
    # for a real catalog component.
    root = tmp_path / "root"
    _fake_component(root, "1.0.0", component="nomorals.media.cad")
    pkg_path = build_package(
        tmp_path / "valid.zip", component="nomorals.media.cad",
        version="2.0.0",
        source="devon-official", files=_component_files(tmp_path, "2.0.0"),
        devon_min="0.1.0")
    args = argparse.Namespace(
        package=str(pkg_path), root=str(root), home=str(tmp_path / "home"),
        yes=True, check_only=False, no_pull=False, repo="", config=None,
        json=False)
    rc = _cmd_update(args, settings=None)
    assert rc == 0
    assert json.loads((root / "nomorals" / "media" / "cad"
                       / ".launcher-version")
                      .read_text(encoding="utf-8"))["version"] == "2.0.0"


def test_cli_package_refused_leaves_tree(tmp_path: Path) -> None:
    import argparse

    from nomorals.cmdline.commands.update import _cmd_update

    root = tmp_path / "root"
    _fake_component(root, "1.0.0")
    before = _tree_snapshot(root)
    args = argparse.Namespace(
        package=str(FIXTURES / "tampered.zip"), root=str(root),
        home=str(tmp_path / "home"), yes=True, check_only=False,
        no_pull=False, repo="", config=None, json=False)
    rc = _cmd_update(args, settings=None)
    assert rc == 1
    assert _tree_snapshot(root) == before
