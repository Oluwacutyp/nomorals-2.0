"""Pairing-with-code system (group build §15): the account registry
and the WhatsApp bridge supervisor.

Devon's pool of personal accounts — the reach behind the group and
channel tools — is built by pairing accounts in, one explicit act at
a time:

- ``/pair whatsapp <number>`` registers the number, spawns a
  dedicated bridge process for it (own port, own creds dir) in
  pairing-code mode, and returns the 8-character code WhatsApp
  issues for that number. The owner types it into WhatsApp →
  Linked Devices → Link with phone number instead; the account is
  then a live Devon session.
- Every account lives in ONE registry (WhatsApp numbers today,
  Telegram userbots as they pair) with registered vs connected
  counts — the header the §14 mass tools and the social status
  surface read.
- Supervision is per account: start / stop / restart from chat,
  health probed over the bridge's own protocol.

Store discipline: the registry is a JSON file written atomically;
an unreadable registry RAISES (quarantined aside) — it is never
silently reset to empty, because an empty registry would orphan
live sessions and forget which numbers are whose.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_ACCOUNT_ID_RE = re.compile(r"[^a-z0-9]+")
_CODE_LINE_RE = re.compile(r"PAIRING CODE for \+(\d+):\s*(\S+)")


class PairingStoreError(RuntimeError):
    """The registry file exists but cannot be read — never a reset."""


def account_id_for(platform: str, identity: str) -> str:
    slug = _ACCOUNT_ID_RE.sub("-", identity.strip().lower()).strip("-")
    return f"{platform}:{slug}"


#: Personal-surface toggles every account carries (§15 slices 2–3).
#: All default OFF — the owner enables each surface per account;
#: Devon never starts saving a man's statuses or shadowing his DMs
#: because a default said so.
PERSONAL_TOGGLES = ("status_saving", "view_once", "antidelete")


def default_toggles() -> dict[str, bool]:
    return {name: False for name in PERSONAL_TOGGLES}


@dataclass
class AccountRecord:
    id: str
    platform: str            # "whatsapp" | "telegram"
    identity: str            # WA: international digits; TG: user id/phone
    label: str = ""
    creds_dir: str = ""      # WA: this account's bridge creds directory
    port: int = 0            # WA: this account's bridge port
    status: str = "registered"   # registered|pairing|linked|down|stopped
    created_at: float = 0.0
    last_seen: float = 0.0
    # ── slices 2–3 ──
    session_path: str = ""   # TG: Telethon session file stem (no suffix)
    remote_id: str = ""      # TG: numeric user id once linked
    primary: bool = False    # routing: the platform's named default
    toggles: dict[str, bool] = field(default_factory=default_toggles)

    def toggle(self, name: str) -> bool:
        return bool((self.toggles or {}).get(name, False))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccountRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class PairingRegistry:
    """The durable account registry (one JSON file, atomic writes)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._accounts: dict[str, AccountRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            rows = data.get("accounts") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise ValueError("registry has no accounts list")
            for row in rows:
                rec = AccountRecord.from_dict(row)
                self._accounts[rec.id] = rec
        except Exception as exc:  # noqa: BLE001 - quarantine, never reset
            aside = self.path.with_suffix(
                f".corrupt-{int(time.time())}.json")
            try:
                os.replace(self.path, aside)
            except OSError:
                aside = self.path  # type: ignore[assignment]
            raise PairingStoreError(
                f"paired-account registry unreadable ({exc}); "
                f"original moved to {aside} — no accounts were "
                "forgotten silently") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accounts": [r.to_dict() for r in
                                sorted(self._accounts.values(),
                                       key=lambda r: r.created_at)]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def upsert(self, rec: AccountRecord) -> AccountRecord:
        with self._lock:
            self._accounts[rec.id] = rec
            self._save()
            return rec

    def get(self, account_id: str) -> AccountRecord | None:
        with self._lock:
            return self._accounts.get(account_id)

    def remove(self, account_id: str) -> bool:
        with self._lock:
            if account_id not in self._accounts:
                return False
            del self._accounts[account_id]
            self._save()
            return True

    def all(self) -> list[AccountRecord]:
        with self._lock:
            return sorted(self._accounts.values(),
                          key=lambda r: r.created_at)

    def set_status(self, account_id: str, status: str) -> None:
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                return
            rec.status = status
            rec.last_seen = time.time()
            self._save()

    def accounts_for(self, platform: str) -> list[AccountRecord]:
        with self._lock:
            return [r for r in self.all() if r.platform == platform]

    def set_toggles(self, account_id: str, **kw: bool) -> AccountRecord:
        """Flip personal-surface toggles on one account.  Unknown
        toggle names raise — a typo must never read as success."""
        unknown = set(kw) - set(PERSONAL_TOGGLES)
        if unknown:
            raise ValueError(
                f"unknown personal toggle(s): {', '.join(sorted(unknown))} "
                f"— known: {', '.join(PERSONAL_TOGGLES)}")
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                raise ValueError(f"no paired account {account_id}")
            toggles = default_toggles()
            toggles.update(rec.toggles or {})
            toggles.update({k: bool(v) for k, v in kw.items()})
            rec.toggles = toggles
            self._save()
            return rec

    def set_primary(self, account_id: str) -> AccountRecord:
        """Name one account its platform's routing default.  Exactly
        one primary per platform — setting a new one clears the old."""
        with self._lock:
            rec = self._accounts.get(account_id)
            if rec is None:
                raise ValueError(f"no paired account {account_id}")
            for other in self._accounts.values():
                if other.platform == rec.platform:
                    other.primary = other.id == rec.id
            self._save()
            return rec


def resolve_account(registry: PairingRegistry, platform: str,
                    account_id: str = "") -> AccountRecord | None:
    """The routing primitive (§15 slice 3): which account speaks for
    a platform.  An explicit id wins (and must belong to the
    platform); otherwise the account the owner named primary;
    otherwise the earliest-created LINKED account (first-connected);
    otherwise the earliest-registered account; otherwise None — the
    platform has no paired account and callers must say so, never
    silently borrow another platform's session."""
    rows = registry.accounts_for(platform)
    if not rows:
        return None
    if account_id:
        for rec in rows:
            if rec.id == account_id:
                return rec
        return None
    for rec in rows:
        if rec.primary:
            return rec
    for rec in rows:
        if rec.status == "linked":
            return rec
    return rows[0]


@dataclass
class _Live:
    proc: Any = None
    code: str = ""
    code_at: float = 0.0
    reader: threading.Thread | None = field(default=None, repr=False)


class BridgeSupervisor:
    """One bridge process per WhatsApp account, supervised.

    Spawning is injectable (``spawn``) so tests drive the whole
    lifecycle without node; production uses subprocess.Popen with
    the bridge's stdout drained on a reader thread (the pairing
    code arrives on stdout and on the bridge protocol alike).
    """

    def __init__(self, registry: PairingRegistry, *,
                 bridge_path: str | Path,
                 sessions_root: str | Path,
                 base_port: int = 8787,
                 node: str = "node",
                 spawn: Callable[..., Any] | None = None) -> None:
        self.registry = registry
        self.bridge_path = Path(bridge_path)
        self.sessions_root = Path(sessions_root)
        self.base_port = int(base_port)
        self.node = node
        self._spawn = spawn or self._default_spawn
        self._live: dict[str, _Live] = {}
        self._lock = threading.RLock()

    # ── process plumbing ────────────────────────────────────────────
    @staticmethod
    def _default_spawn(argv: list[str], env: dict[str, str]) -> Any:
        return subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _alloc_port(self) -> int:
        used = {r.port for r in self.registry.all() if r.port}
        port = self.base_port
        while port in used:
            port += 1
        return port

    def _launch(self, rec: AccountRecord, *, pairing_number: str = "") -> _Live:
        creds = Path(rec.creds_dir)
        creds.mkdir(parents=True, exist_ok=True)
        media = creds.parent / (creds.name + "-media")
        argv = [self.node, str(self.bridge_path),
                "--port", str(rec.port),
                "--creds", str(creds),
                "--media", str(media)]
        env = dict(os.environ)
        if pairing_number:
            env["WA_PAIRING_CODE"] = "1"
            env["WA_PAIRING_NUMBER"] = pairing_number
        proc = self._spawn(argv, env)
        live = _Live(proc=proc)
        # register BEFORE the reader starts: a fast bridge can emit
        # its pairing code immediately, and a code that arrives before
        # the live entry exists would be dropped on the floor.
        with self._lock:
            self._live[rec.id] = live
        reader = threading.Thread(
            target=self._drain, args=(rec.id, proc),
            name=f"pair-reader-{rec.id}", daemon=True)
        live.reader = reader
        reader.start()
        return live

    def _drain(self, account_id: str, proc: Any) -> None:
        stream = getattr(proc, "stdout", None)
        if stream is None:
            return
        try:
            for line in stream:
                m = _CODE_LINE_RE.search(line or "")
                if m:
                    with self._lock:
                        live = self._live.get(account_id)
                        if live is not None:
                            live.code = m.group(2)
                            live.code_at = time.time()
                    self.registry.set_status(account_id, "pairing")
                if "connected as" in (line or ""):
                    self.registry.set_status(account_id, "linked")
        except Exception:  # noqa: BLE001 - a dead stream just ends it
            return

    def _alive(self, account_id: str) -> bool:
        with self._lock:
            live = self._live.get(account_id)
        if live is None or live.proc is None:
            return False
        try:
            return live.proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    def is_running(self, account_id: str) -> bool:
        """Public view of the process table (supervision reads it)."""
        return self._alive(account_id)

    # ── pairing ─────────────────────────────────────────────────────
    def pair_whatsapp(self, number: str, label: str = "") -> AccountRecord:
        digits = re.sub(r"\D", "", number or "")
        if len(digits) < 7:
            raise ValueError(
                "that doesn't look like a phone number — give the "
                "number in international format, e.g. 2348012345678")
        rec_id = account_id_for("whatsapp", digits)
        rec = self.registry.get(rec_id)
        if rec is None:
            creds = self.sessions_root / f"wa-{digits}"
            rec = AccountRecord(
                id=rec_id, platform="whatsapp", identity=digits,
                label=(label or "").strip(), creds_dir=str(creds),
                port=self._alloc_port(), status="pairing",
                created_at=time.time())
            self.registry.upsert(rec)
        else:
            rec.status = "pairing"
            if label:
                rec.label = label.strip()
            self.registry.upsert(rec)
        if self._alive(rec.id):
            self.stop(rec.id)
        self._launch(rec, pairing_number=digits)
        return rec

    def pairing_code(self, account_id: str) -> tuple[str, float] | None:
        """(code, age_seconds) once the bridge has issued one."""
        with self._lock:
            live = self._live.get(account_id)
            if live is None or not live.code:
                return None
            return live.code, max(0.0, time.time() - live.code_at)

    # ── lifecycle ───────────────────────────────────────────────────
    def start(self, account_id: str) -> AccountRecord:
        rec = self.registry.get(account_id)
        if rec is None:
            raise ValueError(f"no paired account {account_id}")
        if rec.platform != "whatsapp":
            raise ValueError(
                f"{account_id} is a {rec.platform} account — its "
                "session is supervised by its own adapter, not the "
                "WhatsApp bridge")
        if not self._alive(account_id):
            self._launch(rec)
        return rec

    def stop(self, account_id: str) -> bool:
        with self._lock:
            live = self._live.pop(account_id, None)
        if live is None or live.proc is None:
            return False
        try:
            live.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.registry.set_status(account_id, "stopped")
        return True

    def remove(self, account_id: str) -> bool:
        self.stop(account_id)
        return self.registry.remove(account_id)

    # ── health ──────────────────────────────────────────────────────
    def probe(self, rec: AccountRecord, timeout: float = 2.0) -> str:
        """Live session state over the bridge protocol: the bridge
        answers a status request with its connection state."""
        if rec.platform != "whatsapp" or not rec.port:
            return rec.status
        try:
            with socket.create_connection(("127.0.0.1", rec.port),
                                          timeout=timeout) as sock:
                sock.sendall(b'{"id":"probe","cmd":"status"}\n')
                sock.settimeout(timeout)
                buf = b""
                while b"\n" not in buf and len(buf) < 65536:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            for line in buf.decode("utf-8", "replace").splitlines():
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if obj.get("type") == "status":
                    return "linked" if obj.get("state") == "open" \
                        else "down"
                if obj.get("id") == "probe" and "state" in obj:
                    return "linked" if obj.get("state") == "open" \
                        else "down"
            return "down"
        except OSError:
            return "down"

    def status_all(self, *, probe: bool = True) -> list[dict[str, Any]]:
        out = []
        for rec in self.registry.all():
            row = rec.to_dict()
            row["process"] = "running" if self._alive(rec.id) else "—"
            if probe and rec.platform == "whatsapp" \
                    and row["process"] == "running":
                row["status"] = self.probe(rec)
            out.append(row)
        return out

    def counts(self, *, probe: bool = True) -> dict[str, int]:
        rows = self.status_all(probe=probe)
        return {
            "registered": len(rows),
            "connected": sum(1 for r in rows
                             if r.get("status") == "linked"),
            # dropped: known to the registry, believed linked once,
            # currently unreachable — the supervision sweep's quarry.
            "dropped": sum(1 for r in rows
                           if r.get("status") == "down"),
        }


class PoolSupervision:
    """The supervision sweep (§15): one pass over the registry that
    flips account states to what is TRUE right now.

    - WhatsApp: an account believed linked whose bridge process is
      gone gets probed over the bridge protocol (a bridge Devon did
      not spawn — e.g. surviving a Devon restart — still answers).
      Probe says down → the record flips to ``down`` (dropped) and
      the bridge is relaunched through the supervisor's own start —
      the bridge's internal reconnect policy then owns the session;
      the sweep never writes "linked" itself, the next probe earns
      it back.  Relaunches are throttled per account so a bridge
      that keeps dying is not fork-bombed.
    - Telegram: a linked account is probed through the pairer's
      session prober (throttled — an MTProto authorization check is
      a network round-trip).  Unauthorized → ``down``.  A down
      Telegram account is NEVER auto-relaunched: re-login needs a
      fresh code from the owner, and pretending otherwise would be
      a fake "connected".
    - Pairing flows that outlived their TTL are expired by the
      pairer as part of the same pass.

    ``pairer`` is duck-typed (probe / expire_flows) so this module
    never imports the Telethon layer.
    """

    def __init__(self, registry: PairingRegistry,
                 supervisor: "BridgeSupervisor | None" = None,
                 pairer: Any = None, *,
                 clock: Callable[[], float] = time.time,
                 tg_probe_interval: float = 300.0,
                 relaunch_interval: float = 300.0) -> None:
        self.registry = registry
        self.supervisor = supervisor
        self.pairer = pairer
        self._clock = clock
        self.tg_probe_interval = float(tg_probe_interval)
        self.relaunch_interval = float(relaunch_interval)
        self._tg_probed_at: dict[str, float] = {}
        self._relaunched_at: dict[str, float] = {}

    def sweep(self) -> list[dict[str, Any]]:
        """Run one pass; returns the list of state changes made."""
        changes: list[dict[str, Any]] = []
        if self.pairer is not None:
            try:
                for account_id in self.pairer.expire_flows():
                    changes.append({"account": account_id,
                                    "change": "pairing flow expired"})
            except Exception:  # noqa: BLE001 - expiry is best-effort
                pass
        for rec in self.registry.all():
            if rec.platform == "whatsapp" and self.supervisor is not None:
                change = self._sweep_whatsapp(rec)
            elif rec.platform == "telegram" and self.pairer is not None:
                change = self._sweep_telegram(rec)
            else:
                change = None
            if change:
                changes.append(change)
        return changes

    def _sweep_whatsapp(self, rec: AccountRecord) -> dict[str, Any] | None:
        sup = self.supervisor
        assert sup is not None
        if rec.status == "stopped":
            return None
        if sup.is_running(rec.id):
            return None  # the bridge's own policy owns a live process
        probed = sup.probe(rec)
        if probed == "linked":
            if rec.status != "linked":
                self.registry.set_status(rec.id, "linked")
                return {"account": rec.id,
                        "change": "probe found the bridge linked"}
            return None
        change = None
        if rec.status != "down":
            self.registry.set_status(rec.id, "down")
            change = {"account": rec.id,
                      "change": f"{rec.status} → down (bridge gone)"}
        now = self._clock()
        last = self._relaunched_at.get(rec.id, 0.0)
        if now - last >= self.relaunch_interval:
            self._relaunched_at[rec.id] = now
            try:
                sup.start(rec.id)
                change = {"account": rec.id,
                          "change": "dropped — bridge relaunched; "
                                    "its reconnect policy owns the "
                                    "session until a probe earns "
                                    "'linked' back"}
            except Exception as exc:  # noqa: BLE001
                change = {"account": rec.id,
                          "change": f"relaunch failed: {exc}"}
        return change

    def _sweep_telegram(self, rec: AccountRecord) -> dict[str, Any] | None:
        if rec.status != "linked":
            return None
        now = self._clock()
        last = self._tg_probed_at.get(rec.id, 0.0)
        if now - last < self.tg_probe_interval:
            return None
        self._tg_probed_at[rec.id] = now
        try:
            state = self.pairer.probe(rec)
        except Exception:  # noqa: BLE001 - a failed probe is not a
            return None                 # verdict; state stays as-is
        if state != "linked" and rec.status != "down":
            self.registry.set_status(rec.id, "down")
            return {"account": rec.id,
                    "change": "linked → down (session probe: "
                              "not authorized)"}
        return None


def default_supervisor(workspace_dir: str | Path) -> BridgeSupervisor:
    """The production supervisor rooted at the runtime workspace:
    registry under data/social/, sessions under data/whatsapp_sessions/,
    bridge script at the repo's bridge/whatsapp-bridge.mjs."""
    root = Path(workspace_dir)
    registry = PairingRegistry(root / "data" / "social"
                               / "paired_accounts.json")
    repo_root = Path(__file__).resolve().parents[2]
    return BridgeSupervisor(
        registry,
        bridge_path=repo_root / "bridge" / "whatsapp-bridge.mjs",
        sessions_root=root / "data" / "whatsapp_sessions")
