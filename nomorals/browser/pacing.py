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

Per-action overrides let delicate actions (submit, navigate) pace slower
than cheap ones (hover, wait) without slowing the whole session down.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

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
    not metronomic. ``per_action`` maps an action name (e.g. "submit") to
    a ``(delay_ms, jitter_ms)`` pair that overrides the global cadence
    for that action only.
    """

    enabled: bool = False
    delay_ms: int = 0
    jitter_ms: int = 0
    per_action: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.delay_ms = max(0, int(self.delay_ms or 0))
        self.jitter_ms = max(0, int(self.jitter_ms or 0))
        cleaned: dict[str, tuple[int, int]] = {}
        for action, pair in (self.per_action or {}).items():
            try:
                delay, jitter = int(pair[0]), int(pair[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"per_action[{action!r}] must be a "
                    f"(delay_ms, jitter_ms) pair: {exc}") from exc
            cleaned[str(action).strip().lower()] = (max(0, delay),
                                                    max(0, jitter))
        self.per_action = cleaned
        #: observable usage: actions paced and seconds actually slept.
        self.pauses = 0
        self.slept_s = 0.0
        if (self.delay_ms or self.jitter_ms or self.per_action) \
                and not self.enabled:
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
    def careful(cls) -> "Pacing":
        """A slower, more deliberate cadence for aggressive sites.

        Longer pauses overall, and state-changing actions (submit,
        navigate, upload) pace slower still — the behavioral-mimicry
        shape: cheap actions stay quick, expensive ones take their time.
        """
        return cls(
            enabled=True,
            delay_ms=700,
            jitter_ms=500,
            per_action={
                "submit": (1400, 800),
                "navigate": (1200, 800),
                "upload": (1400, 800),
                "r_submit": (1400, 800),
                "r_upload": (1400, 800),
            },
        )

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

    def _cadence_for(self, action: str) -> tuple[int, int]:
        """(delay_ms, jitter_ms) for an action: per-action override wins,
        then the global cadence."""
        override = self.per_action.get((action or "").strip().lower())
        if override is not None:
            return override
        return self.delay_ms, self.jitter_ms

    def pause(self, action: str = "") -> float:
        """Sleep the configured cadence before an action.

        Returns the seconds actually slept (0.0 when disabled). Never
        raises — a pacing failure must not break the action it paces.
        """
        if not self.enabled:
            return 0.0
        delay_ms, jitter_ms = self._cadence_for(action)
        wait_ms = float(delay_ms)
        if jitter_ms:
            try:
                wait_ms += random.uniform(0, jitter_ms)
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
        self.pauses += 1
        self.slept_s += seconds
        return seconds

    def read(self, chars: int = 0, *, wpm: int = 220) -> float:
        """Pause as if reading ``chars`` characters of page content.

        Reading time is proportional to content — a human does not skim
        a 5,000-word article in the same beat as a login form. Uses the
        global cadence as the floor so ``read()`` never paces faster than
        a plain ``pause()``. No-op when disabled.
        """
        if not self.enabled:
            return 0.0
        chars = max(0, int(chars or 0))
        words = chars / 5.0
        reading_s = (words / max(60, int(wpm or 220))) * 60.0
        floor_s = self.delay_ms / 1000.0
        seconds = max(reading_s, floor_s)
        if seconds <= 0:
            return 0.0
        try:
            time.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 - pacing never breaks work
            _log.debug("pacing read sleep interrupted: %r", exc)
            return 0.0
        self.pauses += 1
        self.slept_s += seconds
        return seconds

    def stats(self) -> dict[str, object]:
        """Observable pacing usage: pauses taken and seconds slept."""
        return {
            "enabled": self.enabled,
            "pauses": self.pauses,
            "slept_s": round(self.slept_s, 2),
            "per_action": {k: list(v) for k, v in self.per_action.items()},
        }

    def describe(self) -> dict[str, object]:
        """JSON-safe view for CLI/API output."""
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "max_pause_ms": self.delay_ms + self.jitter_ms,
            "per_action": {k: list(v) for k, v in self.per_action.items()},
        }
        payload.update(self.stats())
        return payload


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
