"""Craft-Launcher: Devon's self-update path (census §3A.19).

A verify-then-install pipeline over a catalog with trust rules,
ported from storytold/craft-launcher (the ArtCraft Launcher). The
organ owns the catalog, the package format, the pipeline and the
journal; the ``craftlauncher`` brain tool is the surface, and
``nm update --package`` is the CLI path. Devon already had
transactional ``nm update`` (git pull + snapshot + rollback + health
gates in :mod:`nomorals.os.update`); this organ adds what it lacked:
package-level verification before install and a catalog of trusted
sources.
"""

from __future__ import annotations

from .attribution import verify_attribution
from .catalog import Catalog, Component, Source, VersionTrack, choose_version, default_catalog
from .errors import InstallError, LauncherError, TrustError, VerificationError
from .journal import Journal
from .package import build_package, read_package, verify_package
from .parity import run_parity
from .pipeline import LauncherPipeline, PipelineResult
from .screenshots import (
    SCREENSHOTS_REV,
    SHOTS,
    ScreenshotError,
    check_use,
    fetch_all,
    prepare_picture,
    read_manifest,
    screenshot_url,
    terms_for,
)
from .version import Version, is_newer, is_stable

__all__ = [
    "Catalog",
    "Component",
    "InstallError",
    "Journal",
    "LauncherError",
    "LauncherPipeline",
    "PipelineResult",
    "SCREENSHOTS_REV",
    "SHOTS",
    "ScreenshotError",
    "Source",
    "TrustError",
    "VerificationError",
    "Version",
    "VersionTrack",
    "build_package",
    "check_use",
    "choose_version",
    "default_catalog",
    "fetch_all",
    "is_newer",
    "is_stable",
    "prepare_picture",
    "read_manifest",
    "read_package",
    "run_parity",
    "screenshot_url",
    "terms_for",
    "verify_attribution",
    "verify_package",
]
