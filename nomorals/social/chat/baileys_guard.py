"""Baileys safety: pacing, first-warning stop, encrypted backup.

A personal WhatsApp number over Baileys is a *borrowed* identity:
WhatsApp bans numbers that behave like bulk senders, and a ban is
usually permanent for that number. This module is the guard rail:

* **No daily cap.** There is no daily send limit and no warm-up
  ramp (owner order 2026-11-25: the daily cap is gone). Volume is
  not throttled by the day — the burst guard and the warning latch
  below are the protection.
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
from typing import Any, Callable

__all__ = [
    "BaileysPacer",
    "WarningDetector",
    "is_warning",
    "backup_session_encrypted",
    "restore_session_encrypted",
    "session_backup_path",
    "newest_session_backup",
    "backup_status",
    "seal_whatsapp_sessions",
]

#: Burst guard: at most this many sends inside the window.
BURST_MAX = 5
BURST_WINDOW_S = 60.0

#: Account-level warning signals ONLY.  A latched warning stops every
#: send on the number until the owner clears it, so a pattern must mean
#: "WhatsApp is acting against THIS account" and never match routine
#: per-chat errors: a contact who blocked us, a group in
#: admins-only/"restricted" mode, a ``515`` restart-required after
#: pairing, a ``401`` logged-out.  Baileys' ``DisconnectReason.forbidden``
#: (403) is the one status code that really is a ban signal.
_WARNING_RES = (
    re.compile(r"\b(temporarily\s+banned|permanently\s+banned"
               r"|account\s+(?:has\s+been\s+|is\s+|was\s+)?"
               r"(?:banned|blocked|restricted|suspended)"
               r"|(?:number|phone)\s+(?:has\s+been\s+|is\s+|was\s+)?"
               r"(?:banned|blocked|suspended)"
               r"|you\s+(?:are|were|have\s+been)\s+(?:banned|suspended))\b",
               re.I),
    re.compile(r"\b(?:automated\s+behaviou?r|unusual\s+activity"
               r"|bulk\s+(?:messag\w+|behaviou?r|sending)"
               r"|violat\w+\s+(?:our\s+|the\s+|whatsapp'?s?\s+)?terms"
               r"|spam\s+(?:detected|reports?|activity|warning))\b", re.I),
    re.compile(r"\bstate:\s*(?:banned|blocked|restricted|suspended|forbidden)\b",
               re.I),
    re.compile(r"\b403\b.*\b(forbidden|auth|session|connect)", re.I),
    re.compile(r"\bwarning\b.*\b(account|whatsapp)\b|"
               r"\b(account|whatsapp)\b.*\bwarning\b", re.I),
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

    def state_dict(self) -> dict:
        """Serialisable detector state (for the adapter's durable store)."""
        return {
            "stopped": self.stopped,
            "first_warning": self.first_warning,
            "first_warning_at": self.first_warning_at,
        }

    @classmethod
    def from_state(cls, state: dict) -> WarningDetector:
        """Rebuild a detector from :meth:`state_dict` output."""
        detector = cls()
        if isinstance(state, dict):
            detector.stopped = bool(state.get("stopped", False))
            detector.first_warning = str(state.get("first_warning") or "")
            detector.first_warning_at = float(
                state.get("first_warning_at") or 0.0)
        return detector

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
    """Human-like pacing + burst refusal for Baileys.

    No daily cap: the owner removed it (2026-11-25) — no daily
    limit, no warm-up ramp. The pacer only refuses a machine-gun
    burst and spaces sends with jitter; the first-warning latch
    (``WarningDetector``) is the ban protection.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None,
                 rng: random.Random | None = None) -> None:
        self._clock = clock or time.time
        self._rng = rng or random.Random()
        self._recent: list[float] = []

    def may_send(self) -> tuple[bool, str]:
        """``(ok, reason)`` — False when the burst guard trips."""
        now = self._clock()
        self._recent = [t for t in self._recent if now - t < BURST_WINDOW_S]
        if len(self._recent) >= BURST_MAX:
            return (False, f"burst guard: {BURST_MAX} sends in {BURST_WINDOW_S:.0f}s — pacing, not blasting")
        return (True, "")

    def next_delay_s(self, *, base: float = 8.0) -> float:
        """Human-like inter-send delay: base ±50% jitter, min 2s."""
        jitter = self._rng.uniform(0.5, 1.5)
        return max(2.0, base * jitter)

    def state_dict(self) -> dict:
        """Serialisable pacer state (for the adapter's durable store).

        Only the burst window survives. A legacy state file may
        still carry ``session_age_days`` / ``sent_today`` / ``day``
        — keys written by older builds. They are never written
        again and are ignored on load, never re-imposed as a cap.
        """
        return {
            "recent": list(self._recent[-BURST_MAX:]),
        }

    @classmethod
    def from_state(cls, state: dict, *, clock=None, rng=None) -> BaileysPacer:
        """Rebuild a pacer from :meth:`state_dict` output.

        Missing or malformed fields fall back to a fresh pacer — a
        corrupt state file must never break the adapter's startup.
        Legacy daily fields (``session_age_days``, ``sent_today``,
        ``day``) are accepted and IGNORED: they belonged to the
        removed daily cap and must never limit a send again.
        """
        if not isinstance(state, dict):
            state = {}
        pacer = cls(clock=clock, rng=rng)
        try:
            recent = state.get("recent") or []
            pacer._recent = [float(t) for t in recent
                             if isinstance(t, (int, float))]
        except (TypeError, ValueError):
            pass
        return pacer

    def record_sent(self) -> None:
        self._recent.append(self._clock())


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

#: Sealed-copy filename suffix: the sealed copy sits beside the live
#: session dir (``wa-<digits>-session-backup.json`` next to
#: ``wa-<digits>/``), the same sibling convention the bridge's
#: ``-media`` dir already uses. It never sits INSIDE the creds dir:
#: sealing would tar its own previous copy.
SESSION_BACKUP_SUFFIX = "-session-backup.json"


def session_backup_path(creds_dir: str | os.PathLike[str]) -> Path:
    """Default sealed-copy location for one WhatsApp account.

    The location derives from the account's live session dir — the
    context the copy belongs to — so one account can never seal over
    another's copy, and a sealed copy is always found next to the
    session it seals. Under the home dir via the workspace's
    ``whatsapp_sessions`` root, never in the repo, never in chat.
    """
    creds = Path(creds_dir)
    return creds.parent / (creds.name + SESSION_BACKUP_SUFFIX)


def newest_session_backup(creds_dir: str | os.PathLike[str]) -> Path | None:
    """The newest sealed copy for one account, or None.

    The canonical file is the one the daily run and ``/pair session
    backup`` write; dated siblings (``-session-backup-*.json``) are
    honoured too, newest by mtime, so a hand-kept copy is never
    ignored in favour of an older canonical one.
    """
    creds = Path(creds_dir)
    pattern = creds.name + "-session-backup*.json"
    candidates = [p for p in creds.parent.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def backup_status(records: Any) -> list[dict[str, Any]]:
    """Sealed-copy state, one row per linked WhatsApp account.

    ``records`` are pairing records (duck-typed: ``id`` /
    ``platform`` / ``identity`` / ``creds_dir``) — the live registry
    in chat, the registry snapshot in the backup run. Never raises
    for a missing file: absence is a state, not an error.
    """
    rows: list[dict[str, Any]] = []
    for rec in records or []:
        if str(getattr(rec, "platform", "")).lower() != "whatsapp":
            continue
        creds = Path(str(getattr(rec, "creds_dir", "") or ""))
        sealed = session_backup_path(creds)
        row: dict[str, Any] = {
            "account": str(getattr(rec, "id", "")),
            "identity": str(getattr(rec, "identity", "")),
            "creds_dir": str(creds),
            "creds_present": creds.is_dir(),
            "sealed_path": str(sealed),
            "sealed": sealed.is_file(),
            "sealed_mtime": None,
            "sealed_bytes": None,
        }
        if sealed.is_file():
            try:
                st = sealed.stat()
                row["sealed_mtime"] = st.st_mtime
                row["sealed_bytes"] = st.st_size
            except OSError:
                pass
        rows.append(row)
    return rows


def seal_whatsapp_sessions(records: Any,
                           passphrase: str) -> dict[str, Any]:
    """Seal every linked WhatsApp session dir that exists on disk.

    Returns ``{"written": [...], "skipped": [...]}`` — written rows
    carry ``account`` / ``identity`` / ``path`` / ``bytes``; skipped
    rows carry ``account`` / ``reason``. An account whose auth dir
    is absent is skipped, never an error: Devon runs without
    WhatsApp on some profiles, and one missing session must not
    fail the backup (or the verb) for the others. The passphrase
    itself is never returned, never logged, never in a message.
    """
    if not passphrase:
        raise ValueError("sealing needs a non-empty passphrase")
    written: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for rec in records or []:
        if str(getattr(rec, "platform", "")).lower() != "whatsapp":
            continue
        account = str(getattr(rec, "id", ""))
        creds = Path(str(getattr(rec, "creds_dir", "") or ""))
        if not creds.is_dir():
            skipped.append({
                "account": account,
                "reason": ("no Baileys auth dir on disk "
                           f"({creds}) — nothing to seal")})
            continue
        dest = session_backup_path(creds)
        out = backup_session_encrypted(creds, dest, passphrase)
        written.append({
            "account": account,
            "identity": str(getattr(rec, "identity", "")),
            "path": str(out),
            "bytes": out.stat().st_size,
        })
    return {"written": written, "skipped": skipped}
