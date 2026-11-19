"""Craft-Launcher: verify-then-install updates through a trusted
catalog (Craft-Launcher port of storytold/craft-launcher — the
ArtCraft Launcher's self-update path in Rust).

A thin surface over :mod:`nomorals.os.launcher` — the organ owns the
catalog, the package format, the verify-then-install pipeline and
the journal; the tool only routes arguments and shapes the reply.
Devon already had transactional ``nm update`` (git pull + snapshot +
rollback + health gates); this tool adds what it lacked: package
verification before install and a catalog of trusted sources with
trust rules. Pure local work, stdlib only, no upload, no retention.
Writes files, so FS_WRITE.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.errors import http_status_of
from ..core.http import HttpClient
from ..core.policy import Capability
from ..os.launcher import (
    LauncherPipeline,
    default_catalog,
    read_package,
    run_parity,
    verify_attribution,
    verify_package,
)
from ..os.launcher.errors import (
    InstallError,
    TrustError,
    VerificationError,
)
from ..os.launcher.screenshots import (
    SCREENSHOTS_BASE,
    SCREENSHOTS_REV,
    SHOTS,
    NetworkFailure,
    ScreenshotError,
    caption_for,
    fetch_all,
    prepare_picture,
    read_manifest,
    screenshot_url,
    terms_for,
)

__all__ = ["register"]

_ACTIONS = ("catalog", "check", "verify", "install", "rollback", "status",
            "journal", "trust", "pins", "parity", "attribution",
            "shots", "fetch", "manifest", "terms", "prepare")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _home(params: dict[str, Any]) -> Path:
    override = params.get("home") or os.environ.get("NM_HOME", "")
    if override:
        return Path(str(override)).expanduser()
    return Path.home() / ".nomorals"


def _root(params: dict[str, Any]) -> Path:
    override = params.get("root") or ""
    if override:
        return Path(str(override)).expanduser()
    return _repo_root()


def _pipeline(params: dict[str, Any]) -> LauncherPipeline:
    return LauncherPipeline(root=_root(params), home=_home(params),
                            catalog=default_catalog())


def _parse_params(params: dict | None) -> dict[str, Any]:
    return dict(params or {})


def _is_loopback(url: str) -> bool:
    """True when ``url`` points at loopback (a local mirror)."""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1")


def _http_fetcher(client: HttpClient) -> Callable[[str], bytes | None]:
    """An HttpClient-backed fetcher for the screenshot pipeline.

    A 404 is "this picture isn't published" (None); anything else
    that goes wrong is the network's failure.
    """
    def fetch(url: str) -> bytes | None:
        try:
            resp = client.get(url)
        except Exception as exc:  # noqa: BLE001 - mapped below
            if http_status_of(exc) == 404:
                return None
            raise NetworkFailure(
                f"network failure fetching {url}: {exc}") from exc
        if resp.status == 404:
            return None
        if not resp.ok:
            raise NetworkFailure(f"HTTP {resp.status} fetching {url}")
        return resp.body

    return fetch


def _screenshot_client(base_url: str) -> HttpClient:
    """The client for picture fetching. A loopback base URL (a local
    mirror, the upstream loopback-only rule) gets loopback access;
    the real base goes through the normal SSRF guard."""
    loopback = _is_loopback(base_url)
    return HttpClient(allow_private_ips=loopback, force_direct=loopback)


def register(registry: Any) -> None:
    """Attach the craftlauncher family to a registry."""
    @registry.register(
        "craftlauncher",
        description=(
            "Verify-then-install updates through a trusted catalog, "
            "local and uncapped — check what a component could move "
            "to, verify a package (manifest hashes, compat, trust "
            "rules, attestation) before it installs, install it "
            "atomically with health-gated rollback, roll back to the "
            "retained previous copy, and read the update journal. "
            "Craft-Launcher actions: catalog=<list the components and "
            "sources the launcher trusts>; check=<newest release a "
            "component could move to, value=<component id>, params=<"
            "{releases: [...]}>>; verify=<verify a package without "
            "installing it, sources=[package.zip]>; install=<verify "
            "then install a package, sources=[package.zip], params=<"
            "{allow_downgrade: bool, root: <dir>, home: <dir>}>>; "
            "rollback=<restore the retained previous copy, value=<"
            "component id>>; status=<what is installed, value=<"
            "component id>>; journal=<read the update journal>; "
            "trust=<check one (component, source, hash) against the "
            "trust rules, params=<{component, source, sha256}>>; "
            "pins=<list the pinned hashes and revocations>; parity=<"
            "run the machine-checked parity TOML>; attribution=<verify "
            "the SHA-256 attribution DB>; shots=<list the details-page "
            "pictures (app, key, Devon component, terms), value=<app> "
            "to filter one app>; fetch=<download an app's details-page "
            "pictures from the pinned tag into the cache, value=<app>, "
            "params=<{rev, base_url (loopback mirror only), home}>>; "
            "manifest=<what pictures were fetched, with their "
            "attribution rows>; terms=<the per-picture licence rows, "
            "value=<app/key>>; prepare=<validate a picture against the "
            "publishing rules (JPEG, 1200x750, under 2 MB, metadata "
            "stripped) and write the cleaned file, sources=[<jpeg>], "
            "params=<{app, key, resize: bool}>, output=<where to write "
            "it>>. action=<catalog|check|verify|"
            "install|rollback|status|journal|trust|pins|parity|"
            "attribution|shots|fetch|manifest|terms|prepare>; sources=<list of package paths>; output=<"
            "path to write>; value=<component id or text>; params=<dict>."
        ),
        capability=Capability.FS_WRITE,
    )
    def craftlauncher_tool(action: str = "catalog",
                           sources: list[str] | None = None,
                           output: str = "", kind: str = "",
                           node: str = "", nodes: str | list[str] | None = None,
                           value: str = "", x: float = 0.0, y: float = 0.0,
                           command: str = "",
                           params: dict | None = None) -> dict[str, Any]:
        opts = _parse_params(params)
        if action == "catalog":
            catalog = default_catalog()
            return {"components": [
                {"id": c.id, "name": c.name, "source": c.source,
                 "community": c.community,
                 "restart_required": c.restart_required}
                for c in catalog.components],
                "sources": [
                    {"id": s.id, "kind": s.kind, "repo": s.repo,
                     "allowed_components": list(s.allowed_components)}
                    for s in catalog.sources]}
        if action == "parity":
            return run_parity()
        if action == "attribution":
            return verify_attribution()
        if action == "shots":
            app = value or str(opts.get("app", ""))
            apps = [app] if app else sorted(SHOTS)
            out: dict[str, Any] = {"rev": SCREENSHOTS_REV, "apps": {},
                                    "total": 0}
            for name in apps:
                if name not in SHOTS:
                    return {"ok": False, "error": f"unknown app: {name}"}
                out["apps"][name] = [
                    {"key": shot.key, "caption": caption_for(shot.key),
                     "devon_component": shot.devon_component,
                     "url": screenshot_url(name, shot.key)}
                    for shot in SHOTS[name]]
                out["total"] += len(SHOTS[name])
            return out
        if action == "manifest":
            return read_manifest(_home(opts))
        if action == "terms":
            target = value or str(opts.get("app", ""))
            if "/" in target:
                app_name, key = target.split("/", 1)
                row = terms_for(app_name, key)
                if row is None:
                    return {"ok": False,
                            "error": f"no attribution row for {target}"}
                return row
            if target:
                if target not in SHOTS:
                    return {"ok": False, "error": f"unknown app: {target}"}
                return {"app": target,
                        "pictures": [terms_for(target, shot.key)
                                     for shot in SHOTS[target]]}
            return {"apps": {
                name: [terms_for(name, shot.key) for shot in shots]
                for name, shots in sorted(SHOTS.items())}}
        if action == "fetch":
            app = value or str(opts.get("app", ""))
            if not app:
                return {"ok": False,
                        "error": "craftlauncher fetch needs an app "
                                 "(value=<app>)"}
            base_url = str(opts.get("base_url", SCREENSHOTS_BASE))
            if base_url != SCREENSHOTS_BASE and not _is_loopback(base_url):
                return {"ok": False,
                        "error": "a custom base_url must be a loopback "
                                 "mirror (127.0.0.1 / localhost)"}
            try:
                return fetch_all(
                    _home(opts), app,
                    rev=str(opts.get("rev", SCREENSHOTS_REV)),
                    base_url=base_url,
                    fetcher=_http_fetcher(_screenshot_client(base_url)))
            except ScreenshotError as exc:
                return {"ok": False, "error": str(exc)}
        if action == "prepare":
            if not sources:
                return {"ok": False,
                        "error": "craftlauncher prepare needs a picture "
                                 "path (sources=[<jpeg>])"}
            app = str(opts.get("app", ""))
            key = str(opts.get("key", ""))
            if not app or not key:
                return {"ok": False,
                        "error": "craftlauncher prepare needs params "
                                 "{app, key}"}
            src = Path(sources[0]).expanduser()
            try:
                data = src.read_bytes()
            except OSError as exc:
                return {"ok": False, "error": f"cannot read {src}: {exc}"}
            try:
                result = prepare_picture(
                    data, app=app, key=key,
                    resize=bool(opts.get("resize", False)))
            except ScreenshotError as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            cleaned = bytes(result.pop("data"))
            dest = Path(output).expanduser() if output else (
                src.with_name(f"{src.stem}.1200x750{src.suffix or '.jpg'}"))
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(cleaned)
            except OSError as exc:
                return {"ok": False,
                        "error": f"cannot write {dest}: {exc}"}
            result["output"] = str(dest)
            return result
        pipe = _pipeline(opts)
        if action == "journal":
            return {"entries": pipe.journal.entries()}
        if action == "pins":
            catalog = default_catalog()
            return {"pins": {k: sorted(v)
                             for k, v in catalog.pins.items()},
                    "attestation_pins": {
                        f"{c}@{s}": v
                        for (c, s), v in catalog.attestation_pins.items()},
                    "revoked": catalog.revoked}
        if action == "trust":
            catalog = default_catalog()
            verdict = catalog.check_trust(
                str(opts.get("component", value)),
                str(opts.get("source", "")),
                str(opts.get("sha256", "")))
            return {"ok": verdict.ok, "reason": verdict.reason}
        if action == "status":
            return pipe.status(value or str(opts.get("component", "")))
        if action == "check":
            return pipe.check(value or str(opts.get("component", "")),
                              list(opts.get("releases", [])))
        if action == "verify":
            if not sources:
                raise ValueError("craftlauncher verify needs a package path")
            pkg = read_package(sources[0])
            verdict = verify_package(
                pkg, catalog=default_catalog(),
                devon_version=pipe.devon_version)
            return {"ok": verdict.ok, "failures": verdict.failures,
                    "package_sha256": verdict.package_sha256,
                    "component": pkg.component,
                    "version": str(pkg.version)}
        if action == "install":
            if not sources:
                raise ValueError("craftlauncher install needs a package path")
            try:
                result = pipe.install(
                    sources[0],
                    allow_downgrade=bool(opts.get("allow_downgrade", False)))
            except (VerificationError, TrustError, InstallError) as exc:
                return {"ok": False, "error": str(exc),
                        "error_kind": type(exc).__name__}
            return {"ok": result.ok,
                    "installed_version": result.installed_version,
                    "rolled_back": result.rolled_back,
                    "restart_required": result.restart_required,
                    "health_before": result.health_before,
                    "health_after": result.health_after,
                    "error": result.error}
        if action == "rollback":
            return pipe.rollback(value or str(opts.get("component", "")))
        raise ValueError(f"unknown craftlauncher action: {action}")
