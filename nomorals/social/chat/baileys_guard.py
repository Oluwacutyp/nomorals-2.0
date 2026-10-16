"""Baileys safety: pacing, warm-up, first-warning stop, encrypted backup.

A personal WhatsApp number over Baileys is a *borrowed* identity:
WhatsApp bans numbers that behave like bulk senders, and a ban is
usually permanent for that number. This module is the guard rail:

* **Warm-up.** A fresh session starts at a low daily cap and ramps
  (``WARMUP_SCHEDULE``: day 1 → 20 sends, then ×2/day up to the steady
  cap). A brand-new session never cold-blasts.
* **Pacing.** ``BaileysPacer`` spaces sends with jitter so traffic looks
  human (never a machine-gun burst). ``never_cold_bulk``: the pacer
  refuses a burst (``>burst_max`` sends inside ``burst_window_s``).
* **Detect-and-stop on first warning.** :func:`is_warning` matches
  WhatsApp's warning signals (ban/temporary-ban/restriction/warning
  strings from the bridge). The first warning latches
  ``WarningDetector.stopped`` — the sender must stop, not retry. There
  is no "try once more": the second attempt after a warning is what
  converts a warning into a ban.
* **Encrypted session backup.** :func:`backup_session_encrypted` seals
  the Baileys auth dir with Fernet (PBKDF2-derived key from a
  passphrase + random salt). A plaintext session backup is a credential
  leak; this one is not.

Invariant: after the first warning is noted, ``may_send()`` is False
forever (until a fresh detector is constructed) — no code path can
talk itself past a warning.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable

__all__ = [
    "WARMUP_SCHEDULE",
    "STEADY_DAILY_CAP",
    "BaileysPacer",
    "WarningDetector",
    "is_warning",
    "backup_session_encrypted",
    "restore_session_encrypted",
]

#: Warm-up: day index (0-based) → max sends that day. Ramps ×2 to steady.
WARMUP_SCHEDULE: tuple[int, ...] = (20, 40, 80, 160)
#: Steady-state daily cap once warmed up.
STEADY_DAILY_CAP = 300
#: Burst guard: at most this many sends inside the window.
BURST_MAX = 5
BURST_WINDOW_S = 60.0

_WARNING_RES = (
    re.compile(r"\b(banned|ban\b|temporarily banned|blocked|restricted|restriction)\b", re.I),
    re.compile(r"\b(warning|warned|violat|spam|bulk|automated behaviour|automated behavior)\b", re.I),
    re.compile(r"\b(401|403|419|515)\b.*\b(auth|session|connect)", re.I),
)


def is_warning(text: str) -> bool:
    """True iff this bridge/log text carries a WhatsApp warning signal."""
    t = text or ""
    return any(rx.search(t) for rx in _WARNING_RES)


class WarningDetector:
    """Latching first-warning stop. Once tripped, stays tripped."""

    def __init__(self) -> None:
        self.stopped = False
        self.first_warning: str = ""
        self.first_warning_at: float = 0.0

    def note(self, text: str) -> bool:
        """Record bridge/log text. Returns True iff this tripped the stop."""
        if self.stopped:
            return False
        if is_warning(text):
            self.stopped = True
            self.first_warning = (text or "")[:500]
            self.first_warning_at = time.time()
            return True
        return False

    def may_send(self) -> bool:
        """False forever after the first warning — no talking past it."""
        return not self.stopped


class BaileysPacer:
    """Warm-up caps + human-like pacing + burst refusal for Baileys."""

    def __init__(self, *, session_age_days: int = 0,
                 clock: Callable[[], float] | None = None,
                 rng: random.Random | None = None) -> None:
        self.session_age_days = max(0, int(session_age_days))
        self._clock = clock or time.time
        self._rng = rng or random.Random()
        self._sent_today = 0
        self._day = int(self._clock() // 86400)
        self._recent: list[float] = []

    def daily_cap(self) -> int:
        """Warm-up cap for this session age (ramps to steady)."""
        if self.session_age_days < len(WARMUP_SCHEDULE):
            return WARMUP_SCHEDULE[self.session_age_days]
        return STEADY_DAILY_CAP

    def _roll_day(self) -> None:
        day = int(self._clock() // 86400)
        if day != self._day:
            self._day = day
            self._sent_today = 0

    def may_send(self) -> tuple[bool, str]:
        """``(ok, reason)`` — False when the daily cap or burst guard trips."""
        self._roll_day()
        if self._sent_today >= self.daily_cap():
            return (False, f"daily cap reached ({self.daily_cap()} sends, warm-up day {self.session_age_days})")
        now = self._clock()
        self._recent = [t for t in self._recent if now - t < BURST_WINDOW_S]
        if len(self._recent) >= BURST_MAX:
            return (False, f"burst guard: {BURST_MAX} sends in {BURST_WINDOW_S:.0f}s — pacing, not blasting")
        return (True, "")

    def next_delay_s(self, *, base: float = 8.0) -> float:
        """Human-like inter-send delay: base ±50% jitter, min 2s."""
        jitter = self._rng.uniform(0.5, 1.5)
        return max(2.0, base * jitter)

    def record_sent(self) -> None:
        self._roll_day()
        self._sent_today += 1
        self._recent.append(self._clock())

    @property
    def sent_today(self) -> int:
        self._roll_day()
        return self._sent_today


def _fernet_from_password(password: str, salt: bytes):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8"))))


def backup_session_encrypted(session_dir: str | os.PathLike[str],
                             dest: str | os.PathLike[str],
                             password: str) -> Path:
    """Seal a Baileys session dir (tar + Fernet) to ``dest``.

    Format: ``{"salt": b64, "data": b64(fernet(tar.gz))}`` as JSON.
    The plaintext session never touches disk outside the source dir.
    Raises ValueError for a missing source dir or empty password.
    """
    src = Path(session_dir)
    if not src.is_dir():
        raise ValueError(f"session dir not found: {session_dir}")
    if not password:
        raise ValueError("backup needs a non-empty passphrase")
    salt = os.urandom(16)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "session.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(src, arcname="session")
        blob = tar_path.read_bytes()
    token = _fernet_from_password(password, salt).encrypt(blob)
    payload = {"salt": base64.b64encode(salt).decode(),
               "data": base64.b64encode(token).decode()}
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(out, 0o600)
    return out


def restore_session_encrypted(backup: str | os.PathLike[str],
                              dest_dir: str | os.PathLike[str],
                              password: str) -> Path:
    """Restore an encrypted session backup into ``dest_dir``. Returns it.

    Raises ValueError on a wrong passphrase / corrupt backup (Fernet
    authentication failure is surfaced, never silently accepted).
    """
    if not password:
        raise ValueError("restore needs the backup passphrase")
    payload = json.loads(Path(backup).read_text(encoding="utf-8"))
    salt = base64.b64decode(payload["salt"])
    token = base64.b64decode(payload["data"])
    try:
        blob = _fernet_from_password(password, salt).decrypt(token)
    except Exception as exc:  # noqa: BLE001 - auth failure: wrong pass or tampered
        raise ValueError("cannot decrypt session backup — wrong passphrase or corrupt file") from exc
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "session.tar.gz"
        tar_path.write_bytes(blob)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        extracted = Path(tmp) / "session"
        for item in extracted.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return dest
