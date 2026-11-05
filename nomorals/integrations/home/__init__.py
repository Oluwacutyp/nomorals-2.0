"""Native home & hardware backends (Echelon Wave 6).

Home Assistant is no longer the only wired smart-home backend:
:class:`WemoBackend` (Belkin WeMo, native UPnP/SOAP — the ``wemo.rs``
pattern) and :class:`RazberryBackend` (Z-Wave via a RaZberry/Z-Way
controller's HTTP JSON API — the ``razberry.rs`` pattern) join it under
one contract (:class:`HomeBackend`): capability data, read-only
discovery that degrades to named errors, and a single gated control
seam.

**Gate posture (owner rule, warn-first):** discovery and reads are
open; control of any device is OFF until the owner enables it per
device through :class:`ControlGate` (surfaced on
``SmartHomeIntegration`` as ``enable_device_control`` /
``disable_device_control``).  A control call on a non-enabled device
raises :class:`ControlNotEnabledError` before any transport use.

The emulator doubles in :mod:`.emulators` are the test oracles: strict
protocol servers with assertion logs, implemented independently of
the backends.
"""

from __future__ import annotations

from .contract import BackendCapabilities, HomeBackend, HomeDevice, clamp_int
from .errors import ControlNotEnabledError, HomeDeviceNotFoundError
from .gate import ControlGate
from .wemo import UrllibWemoTransport, WemoBackend
from .zwave import RazberryBackend, UrllibZwaveTransport

__all__ = [
    "BackendCapabilities",
    "ControlGate",
    "ControlNotEnabledError",
    "HomeBackend",
    "HomeDevice",
    "HomeDeviceNotFoundError",
    "RazberryBackend",
    "UrllibWemoTransport",
    "UrllibZwaveTransport",
    "WemoBackend",
    "clamp_int",
]
