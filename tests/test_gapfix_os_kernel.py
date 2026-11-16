"""Gapfix R1-14: the OS control plane (kernel/services/health) is retired.

Audit gap 14 found ``nomorals/os/kernel.py`` (``OSKernel``),
``nomorals/os/services.py`` (``ServiceRegistry``) and
``nomorals/os/health.py`` (``HealthMonitor``) with zero production
importers: the live process never started them.  The decision, proven
from the live code before the deletion, is RETIRE — not wire — because:

* the kernel's ``start_all()`` starts nothing by default (every
  built-in descriptor has ``autostart=False``), so a bare wire would
  have managed zero services — a Potemkin control plane;
* the registry's default factories would build DUPLICATE live
  services (a second ``LLMRouter``, a second ``WhatsAppAdapter``)
  if anyone ever looked them up;
* the kernel's ``arm_signals()`` would overwrite the SIGTERM-as-
  interrupt handlers the three entry points already own
  (``cmdline/dispatch.py``, ``agents/partner/runtime.py``,
  ``api/server.py``), breaking their deliberate teardown paths;
* the bus does not need the kernel: ``EventBus._dispatch`` delivers
  inline when no dispatcher thread is running, so every publisher
  already works;
* readiness/health already exist (``/health``, ``/ready``, the
  update health gate) with real consumers; the kernel's checks are
  import-only probes with no consumer.

These pins assert the retirement holds: the modules are gone, nothing
in production references them, the entry points still own shutdown,
and the bus still delivers without a dispatcher.
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED = (
    "nomorals.os.kernel",
    "nomorals.os.services",
    "nomorals.os.health",
)


def test_retired_modules_are_gone() -> None:
    """The three control-plane modules no longer import."""
    for dotted in RETIRED:
        with pytest.raises(ImportError):
            importlib.import_module(dotted)
    for rel in ("nomorals/os/kernel.py", "nomorals/os/services.py",
                "nomorals/os/health.py"):
        assert not (REPO_ROOT / rel).exists(), rel


def test_no_production_reference_to_retired_modules() -> None:
    """No production file imports (or dynamically loads) them.

    Relative imports resolve against the importing file's package
    (``from .health import ...`` inside ``nomorals/os/kernel.py`` is
    ``nomorals.os.health``).
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "nomorals").rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
        pkg = ".".join(path.relative_to(REPO_ROOT).parts[:-1])
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                    mods.append(f"{base}.{node.module}" if node.module else base)
                elif node.module:
                    mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            for mod in mods:
                if mod in RETIRED or mod.startswith(
                        tuple(d + "." for d in RETIRED)):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {mod}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        for dotted in RETIRED:
            if f'"{dotted}"' in text or f"'{dotted}'" in text:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: string {dotted!r}")
    assert offenders == []


def test_entry_points_still_own_shutdown() -> None:
    """The three entry points keep their SIGTERM-as-interrupt wiring."""
    for rel in ("nomorals/cmdline/dispatch.py",
                "nomorals/agents/partner/runtime.py",
                "nomorals/api/server.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "install_sigterm_as_interrupt" in text, rel


def test_bus_delivers_without_a_started_dispatcher() -> None:
    """Async subscribers still fire inline when the bus never starts.

    This is the behavior the retirement relies on: the kernel's
    ``bus.start()`` was never needed for delivery.
    """
    from nomorals.core.events import EventBus

    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("gapfix.os.*", lambda e: seen.append(e.topic))
    bus.publish("gapfix.os.probe", {"n": 1})
    assert seen == ["gapfix.os.probe"]
    assert bus._dispatcher is None or not bus._dispatcher.is_alive()


def test_retirement_is_recorded_in_package_docstring() -> None:
    """``nomorals/os/__init__.py`` records why the cluster went."""
    text = (REPO_ROOT / "nomorals" / "os" / "__init__.py").read_text(
        encoding="utf-8")
    lowered = text.lower()
    assert "retired" in lowered
    assert "oskernel" in lowered
    assert "audit gap r1-14" in lowered
