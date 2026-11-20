"""Serial robot hardware — the device seam.

Pattern from echelon/robot (AGPL-3.0, pattern only, no code taken):
the original's ``RCSerializer`` talks to a Robotics Connection
Serializer over one serial object — ``fw()``, ``battery()``,
``mogo(m1, m2)``, ``blink()``, ``stop()`` — and checks ACK/NACK
on every reply. Devon keeps the same seam shape, but the seam is
an interface: any backend that speaks those verbs attaches, and
with nothing attached the daemon refuses by name instead of
pretending a robot is there.

Backends:
* :func:`open_serial` — a real serial device via pyserial. Missing
  pyserial or a missing device path is ``no_serial``, named.
* :class:`LoopbackBackend` — an in-memory device for tests and
  dry runs. It records every command and answers canned firmware
  / battery strings. It is a test double, never a production
  claim: a daemon on a loopback backend reports
  ``backend: loopback`` in its status.
"""

from __future__ import annotations

import contextlib

from .errors import RobotError


class RobotBackend:
    """The verbs a robot device must speak. Subclass and attach."""

    kind = "backend"

    def fw(self) -> str:
        """Firmware version string."""
        raise NotImplementedError

    def battery(self) -> float:
        """Battery percent (0..100)."""
        raise NotImplementedError

    def mogo(self, m1: int, m2: int) -> bool:
        """Drive the motors (-100..100 each). True when the device
        acknowledged the command."""
        raise NotImplementedError

    def blink(self, l1: int, l2: int) -> bool:
        """Set the two LEDs (0/1 each)."""
        raise NotImplementedError

    def stop(self) -> bool:
        """Stop the motors."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the device."""


class LoopbackBackend(RobotBackend):
    """An in-memory robot for tests and dry runs. Records every
    command it receives; answers canned firmware/battery. Never
    used as a production claim — a daemon on this backend reports
    ``backend: loopback``."""

    kind = "loopback"

    def __init__(self, firmware: str = "loopback-1.0",
                 battery: float = 100.0) -> None:
        self.firmware = firmware
        self._battery = float(battery)
        self.commands: list[tuple] = []
        self.closed = False

    def fw(self) -> str:
        return self.firmware

    def battery(self) -> float:
        return self._battery

    def mogo(self, m1: int, m2: int) -> bool:
        self.commands.append(("mogo", int(m1), int(m2)))
        return True

    def blink(self, l1: int, l2: int) -> bool:
        self.commands.append(("blink", 1 if l1 else 0, 1 if l2 else 0))
        return True

    def stop(self) -> bool:
        self.commands.append(("stop",))
        return True

    def close(self) -> None:
        self.closed = True


class SerialBackend(RobotBackend):
    """A real serial device, spoken through pyserial with the
    original's line protocol: one command line, one ACK/NACK reply
    line. The protocol is the obvious one (``MOGO <m1> <m2>``,
    ``BLINK <l1> <l2>``, ``STOP``, ``FW``, ``BATTERY``) — no device
    was available to verify a vendor dialect against, so the
    backend refuses loudly on a NACK or a timeout instead of
    guessing."""

    kind = "serial"

    def __init__(self, port: str, baud: int = 9600,
                 timeout: float = 1.0) -> None:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RobotError(
                "no_serial",
                "pyserial is not installed — a serial robot needs it "
                "(pip install pyserial), or run with a loopback backend") from exc
        try:
            self._ser = serial.Serial(port, baud, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — port missing, denied, …
            raise RobotError(
                "no_serial",
                f"cannot open serial device {port!r}: {exc}") from exc
        self.port = port

    def _command(self, line: str) -> str:
        self._ser.write((line + "\n").encode("ascii"))
        reply = self._ser.readline().decode("ascii", "replace").strip()
        if not reply.startswith("ACK"):
            raise RobotError(
                "no_serial",
                f"device did not acknowledge {line!r} (got {reply!r})")
        return reply

    def fw(self) -> str:
        return self._command("FW").removeprefix("ACK ").strip() or "unknown"

    def battery(self) -> float:
        reply = self._command("BATTERY").removeprefix("ACK ").strip()
        try:
            return float(reply)
        except ValueError as exc:
            raise RobotError(
                "no_serial", f"device battery unreadable: {reply!r}") from exc

    def mogo(self, m1: int, m2: int) -> bool:
        self._command(f"MOGO {int(m1)} {int(m2)}")
        return True

    def blink(self, l1: int, l2: int) -> bool:
        self._command(f"BLINK {1 if l1 else 0} {1 if l2 else 0}")
        return True

    def stop(self) -> bool:
        self._command("STOP")
        return True

    def close(self) -> None:
        with contextlib.suppress(Exception):  # noqa: BLE001 — close is best-effort
            self._ser.close()


def open_serial(port: str, baud: int = 9600) -> SerialBackend:
    """Attach a real serial robot. Refuses ``no_serial`` when
    pyserial is missing or the port cannot be opened."""
    return SerialBackend(port, baud=baud)
