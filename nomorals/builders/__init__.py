"""Project scaffolding and lifecycle: templates, run, smoke, install, export.

Top-level API::

    from nomorals.builders import scaffold, build_and_verify

    result = scaffold("webapp", "myapp", "/tmp/demo")   # render the template
    report = build_and_verify("webapp", "myapp", "/tmp/demo")  # full lifecycle
    print(report.summary())

    # Ship it: zip the project and send the archive to a chat.
    delivery = build_zip_and_deliver(
        "rest_api", "myapi", "/tmp/demo",
        platform="telegram", chat="@owner", gateway=gw)

The ten-stack :class:`AppBuilder` generator lives in
:mod:`nomorals.builders.app_builder` and is re-exported here for
backward compatibility.
"""

from __future__ import annotations

from .app_builder import STACKS, AppBuilder, register
from .deliver import (
    DeliverResult, DeliveryReport, ZipResult,
    build_zip_and_deliver, deliver_project, zip_project,
)
from .export import ExportResult, VerifyResult, export_project, verify_export
from .install import InstallResult, install_deps, parse_requirements
from .run import RunConfig, ServeError, ServeHandle, find_free_port, run_config, serve
from .scaffold import KINDS, ScaffoldResult, scaffold, template_dir
from .smoke import Check, SmokeResult, smoke_test
from .verify import BuildReport, BuildStep, build_and_verify, run_project_tests

__all__ = [
    # top-level API
    "scaffold", "build_and_verify",
    "ScaffoldResult", "BuildReport", "BuildStep",
    # lifecycle pieces
    "KINDS", "template_dir",
    "RunConfig", "run_config", "serve", "ServeHandle", "ServeError", "find_free_port",
    "Check", "SmokeResult", "smoke_test",
    "InstallResult", "install_deps", "parse_requirements",
    "ExportResult", "VerifyResult", "export_project", "verify_export",
    "ZipResult", "DeliverResult", "DeliveryReport",
    "zip_project", "deliver_project", "build_zip_and_deliver",
    "run_project_tests",
    # legacy ten-stack generator (moved from nomorals/builders.py)
    "AppBuilder", "STACKS", "register",
]
