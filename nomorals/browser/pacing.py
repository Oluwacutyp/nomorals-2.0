"""Human-like pacing for browser actions (layer 4).

An explicit, owner-controlled setting: a fixed delay plus optional random
jitter applied *between* mutating browser actions (navigate, fill, click,
submit, select, check, upload). This is a politeness / rate-limit-avoidance
control the owner configures — it is not deception machinery and does not
touch fingerprints, headers, or any anti-bot control.

Usage::

    from nomorals.browser.pacing import Pacing

    pacing = Pacing(enabled=True, delay_ms=400, jitter_ms=300)
    pacing.pause("fill")          # sleeps ~0.4-0.7s, returns seconds slept

The :class:`BrowserService` owns one ``Pacing`` shared by every tab it
opens; ``set_pacing`` / the ``pacing`` daemon op / ``nm browse pacing``
configure it live. ``NOMORALS_BROWSER_PACING`` env var pins it at startup
as ``delay_ms,jitter_ms`` (e.g. ``400,300``) — either part may be 0.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from ..core.logging_setup import get_logger

__all__ = ["Pacing", "pacing_from_env"]

_log = get_logger(__name__)

#: Env var pinning startup pacing as "delay_ms,jitter_ms".
PACING_ENV = "NOMORALS_BROWSER_PACING"


@dataclass
class Pacing:
    """Delay + jitter applied between browser actions.

    ``enabled`` off (the default) means zero behavioral change — every
    pause is a no-op and costs nothing. ``delay_ms`` is the fixed pause;
    ``jitter_ms`` adds ``uniform(0, jitter_ms)`` on top so the cadence is
    not metronomic.
    """

    enabled: bool = False
    delay_ms: int = 0
    jitter_ms: int = 0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.delay_ms = max(0, int(self.delay_ms or 0))
        self.jitter_ms = max(0, int(self.jitter_ms or 0))
        if (self.delay_ms or self.jitter_ms) and not self.enabled:
            # A configured-but-disabled pacing is a silent lie about what
            # will happen; fail loud at configuration time.
            raise ValueError(
                "Pacing has delay/jitter configured but enabled=False — "
                "set enabled=True or clear the delays")

    @classmethod
    def disabled(cls) -> "Pacing":
        """The default: no pauses, actions run back-to-back."""
        return cls(enabled=False)

    @classmethod
    def human(cls, delay_ms: int = 400, jitter_ms: int = 300) -> "Pacing":
        """A human-ish cadence preset: ~delay_ms plus up to jitter_ms."""
        return cls(enabled=True, delay_ms=delay_ms, jitter_ms=jitter_ms)

    @classmethod
    def from_profile(cls) -> "Pacing":
        """Suggested preset for the current resource profile.

        This is a *suggestion*, not an applied default — the service
        starts :meth:`disabled` unless the owner configures pacing. Slow
        / metered profiles (termux) suggest a gentler cadence; fast
        profiles suggest a light one.
        """
        try:
            from ..core.profiles import get_profile_kind
            kind = get_profile_kind()
        except Exception:  # noqa: BLE001 - profiles are optional
            kind = "workstation"
        if kind == "termux":
            return cls.human(delay_ms=800, jitter_ms=500)
        if kind == "laptop":
            return cls.human(delay_ms=300, jitter_ms=200)
        return cls.human(delay_ms=150, jitter_ms=150)

    def pause(self, action: str = "") -> float:
        """Sleep the configured cadence before an action.

        Returns the seconds actually slept (0.0 when disabled). Never
        raises — a pacing failure must not break the action it paces.
        """
        if not self.enabled:
            return 0.0
        wait_ms = self.delay_ms
        if self.jitter_ms:
            try:
                wait_ms += random.uniform(0, self.jitter_ms)
            except Exception:  # noqa: BLE001 - jitter is cosmetic
                pass
        seconds = wait_ms / 1000.0
        if seconds <= 0:
            return 0.0
        try:
            _log.debug("pacing: %.0fms before %s", wait_ms, action or "action")
            time.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 - pacing never breaks work
            _log.debug("pacing sleep interrupted: %r", exc)
            return 0.0
        return seconds

    def describe(self) -> dict[str, object]:
        """JSON-safe view for CLI/API output."""
        return {
            "enabled": self.enabled,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "max_pause_ms": self.delay_ms + self.jitter_ms,
        }


def pacing_from_env() -> Pacing | None:
    """Pacing pinned by ``NOMORALS_BROWSER_PACING`` ("delay_ms,jitter_ms"),
    or None when the env var is unset/blank. Raises ValueError on garbage
    — a misconfigured env var must fail loud, not silently pace at 0."""
    raw = (os.environ.get(PACING_ENV, "") or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"{PACING_ENV} must be 'delay_ms,jitter_ms' (e.g. '400,300'), "
            f"got {raw!r}")
    delay_ms, jitter_ms = int(parts[0]), int(parts[1])
    if not delay_ms and not jitter_ms:
        return Pacing.disabled()
    return Pacing(enabled=True, delay_ms=delay_ms, jitter_ms=jitter_ms)
