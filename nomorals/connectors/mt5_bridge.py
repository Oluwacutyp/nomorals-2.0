"""MT5 bridge adapter — the second execution rung (shape only).

Devon's trading system executes on TWO rungs (owner's have-it-all
ruling): the Exness REST connector (``nomorals/connectors/exness.py``,
the working rung) and an MT5 bridge to a MetaTrader 5 terminal —
GoldenSwing's home turf. No MT5 terminal exists on the machines
Devon runs on today (EC2 runs the REST path; the terminal lives
with GoldenSwing's own stack, a separate product whose code is NOT
merged in), so this adapter is the honest shape of the rung:

- :func:`probe` reports whether a terminal is reachable — an
  ``MT5_TERMINAL_PATH`` setting, or an importable ``MetaTrader5``
  package. Absent both, the rung is unavailable and says why.
- :class:`MT5BridgeConnector` exposes the desk's connector protocol
  (snapshot / account details / candles / instrument conditions /
  open / close). Every method raises :class:`MT5Unavailable` with
  the probe's reason. There are NO simulated fills here and no
  fallback to another rung hidden inside: unavailability is the
  answer, and the execution engine journals it as a refusal.

When a terminal genuinely appears (the owner installs one and
points the setting at it), this module is where the real bridge
lands — the protocol below is the contract it must satisfy.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["MT5Unavailable", "MT5BridgeConnector", "probe"]


class MT5Unavailable(Exception):
    """The MT5 rung is not available on this machine — stated, never
    papered over with a simulated fill."""


def probe() -> dict[str, Any]:
    """Is an MT5 terminal reachable from this runtime?

    Checks, in order: the ``MT5_TERMINAL_PATH`` environment setting
    (a path that must exist), then an importable ``MetaTrader5``
    package (the official terminal binding). Returns
    ``{"available": bool, "reason": str, ...}``.
    """
    path = os.environ.get("MT5_TERMINAL_PATH", "").strip()
    if path:
        if os.path.exists(path):
            return {"available": True, "via": "MT5_TERMINAL_PATH",
                    "terminal_path": path,
                    "reason": "terminal path configured and present"}
        return {"available": False, "via": "MT5_TERMINAL_PATH",
                "terminal_path": path,
                "reason": (f"MT5_TERMINAL_PATH is set to {path!r} but "
                           f"nothing exists there")}
    try:
        import MetaTrader5  # noqa: F401 - presence probe only
    except ImportError:
        return {"available": False, "via": None,
                "reason": ("no MT5 terminal on this machine: "
                           "MT5_TERMINAL_PATH is unset and the "
                           "MetaTrader5 package is not installed — "
                           "the Exness REST rung is the working "
                           "execution home")}
    return {"available": True, "via": "MetaTrader5 package",
            "reason": "MetaTrader5 package importable"}


class MT5BridgeConnector:
    """Desk-protocol connector for the MT5 rung.

    Construction never raises (the execution engine must be able to
    hold the rung and report it); every protocol method raises
    :class:`MT5Unavailable` until a real terminal bridge lands.
    """

    def __init__(self) -> None:
        self._probe = probe()

    def status(self) -> dict[str, Any]:
        """The rung's honest availability report."""
        return {"rung": "mt5_bridge", **self._probe}

    def _unavailable(self) -> MT5Unavailable:
        return MT5Unavailable(
            f"MT5 bridge unavailable: {self._probe.get('reason')}")

    # ── the desk's connector protocol (all unavailable for now) ──

    def get_snapshot(self) -> dict[str, Any]:
        raise self._unavailable()

    def get_account_details(self) -> dict[str, Any]:
        raise self._unavailable()

    def get_candles(self, instrument: str, timeframe: str = "H1",
                    count: int = 200) -> list[dict[str, Any]]:
        raise self._unavailable()

    def get_instrument_condition(self, instrument: str) -> dict[str, Any]:
        raise self._unavailable()

    def open_position(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise self._unavailable()

    def close_position(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise self._unavailable()
