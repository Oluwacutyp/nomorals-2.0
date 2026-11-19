"""Crash test plugin — the quarantine proof.

Aborts its process on the first block. It must never be loaded
in-process: the host loads it quarantined, the worker dies, the
host names the failure and the engine keeps running.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CrashPlugin"]


class CrashPlugin:
    """A deliberately lethal plugin: aborts the process on first use."""

    def info(self) -> dict:
        return {"id": "crash", "name": "Crash", "params": {}}

    def process(self, samples: np.ndarray,
                sample_rate: int) -> np.ndarray:
        import os
        os.abort()
        return samples  # pragma: no cover — abort never returns
