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

WIRED 2026-11-16 (audit gap R1-14, reversed): the control-plane
cluster — ``kernel.py`` (``OSKernel``), ``services.py``
(``ServiceRegistry``), ``health.py`` (``HealthMonitor``) — was
deleted once as "unfit" and RESTORED by owner order, because the
capability was wanted and only the implementation was broken.  One
``OSKernel`` is constructed per process in the live entry point
(``scripts/run_chat_bot.py::main``), AFTER ``PartnerRuntime.start()``,
and ADOPTS the live instances — scheduler, gateway adapters, router,
database — via ``ServiceRegistry.adopt()``; the registry never builds
a second ``LLMRouter`` or ``WhatsAppAdapter`` (the unadopted built-in
descriptors are unregistered at boot).  The kernel supervises: its
thread watches liveness and restarts a dead service through the
service's own stop/start path, with exponential backoff and a 5-
attempt budget, journaling the death in the failure ledger and the
restart in the beacon (``nm status`` / ``/health`` show it).  Signals
stay with the entry points — ``arm_signals()`` chains the previous
handler, it never replaces one — and the bus dispatcher stays off
(inline delivery is the proven behavior).  Health delegates to the
real probes ``/health`` and ``/ready`` already use (``SELECT 1``,
``providers()``, scheduler tick freshness, adapter ``health()``).
Pinned by ``tests/test_gapfix_os_control_plane.py``.
"""
