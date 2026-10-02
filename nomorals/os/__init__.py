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
"""
