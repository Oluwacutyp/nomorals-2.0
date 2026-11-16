"""nomorals.os — the Devon OS control plane (Wave H2).

This package is the operating-system shell around Devon's organs: kernel
lifecycle, the service registry, health monitoring, first-class sessions,
and first-class projects.  It is layer 6 in the layering map: it may import
layers 1-6, and it is imported only by layer-7 entry points (cli, api,
builders).  Lower layers must NEVER import ``nomorals.os`` — integration
with them happens via callbacks, events on ``core.events.global_bus``, and
L7 wiring, never via upward imports.

Sibling H2 workers own the other os modules (artifact graph, mission state
machine, resource manager, timeline); this ``__init__`` stays minimal with
NO eager imports so those modules can land without touching this file and
without import cycles.

RETIRED 2026-11-16 (audit gap R1-14): the control-plane cluster —
``kernel.py`` (``OSKernel``), ``services.py`` (``ServiceRegistry``),
``health.py`` (``HealthMonitor``) — was deleted, not wired.  It had zero
production importers, and wiring it would have been a Potemkin fix:
``start_all()`` starts nothing by default (every built-in descriptor
has ``autostart=False``); the registry's default factories would build
duplicate live services (a second ``LLMRouter``, a second
``WhatsAppAdapter``); ``arm_signals()`` would overwrite the SIGTERM-as-
interrupt handlers the entry points already own; the bus delivers
inline without a started dispatcher; and readiness/health already
exist (``/health``, ``/ready``, the update health gate) with real
consumers, while the kernel's checks were import-only probes.  The
live process lifecycle stays where it is: the L7 entry points
(``nm`` dispatch, ``PartnerRuntime.run``, ``api.serve``) own signals
and teardown via ``core/shutdown`` + ``finally`` blocks.  Pinned by
``tests/test_gapfix_os_kernel.py``.
"""
