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

from .app_builder import STACKS, THEMES, AppBuilder, register
from .deliver import (
    DeliverResult, DeliveryReport, ZipResult,
    build_zip_and_deliver, deliver_project, zip_project,
)
from .export import (ExportResult, VerifyResult, export_project,
                     verify_export, verify_reproducible, source_date_epoch)
from .install import (InstallResult, ensure_venv, generate_lock,
                      install_deps, parse_requirements, resolve_installer,
                      verify_lock)
from .run import (HttpProbe, ProbeResult, RunConfig, ServeError, ServeHandle,
                  TcpProbe, find_free_port, run_config, run_probe, serve,
                  stop_all)
from .scaffold import (KINDS, ScaffoldResult, TemplateInfo, describe,
                       list_templates, scaffold, template_dir)
from .smoke import Check, HttpExpectation, SmokeResult, smoke_test, tcp_probe
from .style import (Theme, banner, paint, render_kv, render_steps,
                    resolve_theme, rule, spinner_frames, status_glyph,
                    theme_names)
from .verify import (STEP_NAMES, BuildReport, BuildStep, build_and_verify,
                     run_project_tests, validate_sources)

__all__ = [
    # top-level API
    "scaffold", "build_and_verify",
    "ScaffoldResult", "BuildReport", "BuildStep",
    # template catalog
    "TemplateInfo", "list_templates", "describe",
    # lifecycle pieces
    "KINDS", "template_dir",
    "RunConfig", "run_config", "serve", "ServeHandle", "ServeError",
    "find_free_port", "stop_all",
    "HttpProbe", "TcpProbe", "ProbeResult", "run_probe",
    "Check", "HttpExpectation", "SmokeResult", "smoke_test", "tcp_probe",
    "InstallResult", "install_deps", "parse_requirements",
    "resolve_installer", "ensure_venv", "generate_lock", "verify_lock",
    "ExportResult", "VerifyResult", "export_project", "verify_export",
    "verify_reproducible", "source_date_epoch",
    "ZipResult", "DeliverResult", "DeliveryReport",
    "zip_project", "deliver_project", "build_zip_and_deliver",
    "run_project_tests", "validate_sources", "STEP_NAMES",
    # presentation
    "Theme", "theme_names", "resolve_theme", "paint", "banner", "rule",
    "render_kv", "render_steps", "spinner_frames", "status_glyph",
    # legacy ten-stack generator (moved from nomorals/builders.py)
    "AppBuilder", "STACKS", "THEMES", "register",
]
