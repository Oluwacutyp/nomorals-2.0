"""Devon's own computer: a persistent desktop she drives herself.

The organ in one paragraph: desktop *sessions* live on the host
Devon runs on (sessions.py) — a session owns a display, survives
between tasks, and journals its life. A per-platform *driver*
speaks one JSON-lines protocol (protocol.py) to Devon's own
desktop bridge (bridge/desktopbridge — raw X11 on Linux, raw
Win32 on Windows; no third-party automation stack anywhere). The
*control loop* (loop.py) generalises the browser agent's
perceive → decide → act → verify cycle to the whole desktop.
provision.py self-provisions whatever the host is missing (the
bridge binary, a virtual display). service.py is the facade the
tool and chat surfaces call, with profile gating and the
personal-surface control gate (sticky owner stop, owner takeover,
degraded brain = observe-only).

Layer 4: a service organ beside browser/. It depends downward on
core/ (config, failure ledgers, profile) only; the brain is
injected by the wiring layers above (tool registration, partner
runtime), never imported from agents/.
"""

from __future__ import annotations

__all__ = ["service", "sessions", "loop", "protocol", "provision"]
