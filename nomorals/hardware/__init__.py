"""Hardware organ — Devon's physical devices.

Today: the Azure Kinect depth sensor (``nomorals.hardware.kinect``).
The organ is a leaf service organ (L4): it reaches only downward
to the standard library and its own submodules, and nothing in
the layer stack imports it except through the probe.
"""

from __future__ import annotations

__all__: list[str] = []
