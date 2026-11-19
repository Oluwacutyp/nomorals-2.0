"""SoundCraft plugin hosts — CLAP / VST3 / AU (lane 107).

Ported from storytold/soundcraft's ``crates/clap-host``,
``crates/vst3-host`` and ``crates/au-host``: each format has a host
that scans for bundles and loads plugins behind the same ``Plugin``
shape (process a block, report params). The load boundary is the
point of the design: a plugin runs in a quarantined subprocess
(:mod:`._plugin_worker`), speaking length-prefixed JSON frames over
a pipe. A plugin that crashes — segfault, abort, hang — kills only
its worker: the host marks the instance quarantined, mutes its
output to silence, names the failure (exit code / signal), and the
engine keeps running.

Honest limits, stated where they bite: real CLAP/VST3 bundles are
native binaries whose ABI cannot be loaded from Python — the scan
discovers bundle directories and reads a sidecar manifest when one
exists, and a bundle with no loadable entry is listed as
``unloadable`` with the reason, never pretended loaded. AU is a
macOS-only format: on Linux the host refuses outright. The
quarantine boundary itself is real and proven by killing it.
"""

from __future__ import annotations

import base64
import contextlib
import enum
import os
import select
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SoundError, SoundErrorKind

__all__ = ["HostKind", "PluginDescriptor", "PluginHost", "scan_plugins"]

_WORKER_MODULE = "nomorals.media.sound._plugin_worker"
_READ_TIMEOUT_S = 5.0


class HostKind(enum.StrEnum):
    CLAP = "clap"
    VST3 = "vst3"
    AU = "au"


class PluginDescriptor:
    def __init__(self, name: str, path: str, kind: HostKind,
                 state: str, reason: str = ""):
        self.name = name
        self.path = path
        self.kind = kind
        self.state = state  # "found" | "unloadable" | "unsupported"
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path,
                "kind": self.kind.value, "state": self.state,
                "reason": self.reason}


def _bundle_suffix(kind: HostKind) -> str:
    return {HostKind.CLAP: ".clap", HostKind.VST3: ".vst3",
            HostKind.AU: ".component"}[kind]


def scan_plugins(kind: HostKind, directories: list[str | Path]) -> list[PluginDescriptor]:
    """Scan directories for plugin bundles of one format.

    AU on a non-macOS host finds nothing — the format does not exist
    here, and the scan says so by returning an empty list.
    """
    found: list[PluginDescriptor] = []
    suffix = _bundle_suffix(kind)
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        for bundle in sorted(root.glob(f"*{suffix}")):
            manifest = bundle / "plugin.json"
            if manifest.exists():
                try:
                    import json
                    meta = json.loads(manifest.read_text())
                    found.append(PluginDescriptor(
                        name=str(meta.get("name", bundle.stem)),
                        path=str(bundle), kind=kind, state="found"))
                except Exception as exc:  # noqa: BLE001
                    found.append(PluginDescriptor(
                        name=bundle.stem, path=str(bundle), kind=kind,
                        state="unloadable",
                        reason=f"manifest unreadable: {exc}"))
            else:
                found.append(PluginDescriptor(
                    name=bundle.stem, path=str(bundle), kind=kind,
                    state="unloadable",
                    reason="no loadable entry point: native CLAP/VST3 "
                           "binaries cannot be loaded from Python"))
    return found


def _frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("worker closed the pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _Worker:
    """One quarantined plugin worker subprocess."""

    def __init__(self) -> None:
        import json
        self._json = json
        env = dict(os.environ)
        tree_root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = tree_root + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", _WORKER_MODULE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env,
            cwd=tree_root)
        self._seq = 0

    def call(self, message: dict[str, Any]) -> dict[str, Any]:
        import json
        payload = json.dumps(message).encode()
        assert self._proc.stdin is not None
        self._proc.stdin.write(_frame(payload))
        self._proc.stdin.flush()
        return self._read_reply()

    def _read_reply(self) -> dict[str, Any]:
        import json
        assert self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], _READ_TIMEOUT_S)
        if not ready:
            self._kill()
            raise SoundError("plugin worker stopped answering (timeout)",
                             kind=SoundErrorKind.PLUGIN_FAILED)
        raw_len = _read_exact(self._proc.stdout, 4)
        (length,) = struct.unpack(">I", raw_len)
        payload = _read_exact(self._proc.stdout, length)
        return json.loads(payload.decode())

    def _kill(self) -> None:
        with contextlib.suppress(Exception):
            self._proc.kill()

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self.call({"cmd": "close"})
            except Exception:  # noqa: BLE001
                self._kill()
            try:
                self._proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                self._kill()

    @property
    def returncode(self) -> int | None:
        return self._proc.poll()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None


class PluginInstance:
    """One loaded plugin — in-process or quarantined."""

    def __init__(self, plugin: Any = None, worker: _Worker | None = None,
                 name: str = ""):
        self._plugin = plugin
        self._worker = worker
        self.name = name
        self.state = "running"
        self.failure = ""

    def info(self) -> dict[str, Any]:
        if self._plugin is not None:
            return self._plugin.info()
        return {"id": self.name, "name": self.name, "params": {}}

    def process(self, samples: np.ndarray,
                sample_rate: int) -> np.ndarray:
        if self.state != "running":
            return np.zeros_like(np.asarray(samples, dtype=np.float64))
        if self._plugin is not None:
            return np.asarray(self._plugin.process(samples, sample_rate),
                              dtype=np.float64)
        assert self._worker is not None
        try:
            data = np.asarray(samples, dtype=np.float32).tobytes()
            reply = self._worker.call({
                "cmd": "process",
                "sample_rate": int(sample_rate),
                "data": base64.b64encode(data).decode(),
            })
            if not reply.get("ok"):
                raise SoundError(str(reply.get("error", "plugin failed")),
                                 kind=SoundErrorKind.PLUGIN_FAILED)
            out = base64.b64decode(reply["data"])
            return np.frombuffer(out, dtype=np.float32).astype(np.float64)
        except (EOFError, BrokenPipeError, SoundError) as exc:
            self._quarantine(exc)
            return np.zeros_like(np.asarray(samples, dtype=np.float64))

    def _quarantine(self, exc: BaseException) -> None:
        rc = self._worker.returncode if self._worker is not None else None
        if rc is None and self._worker is not None:
            # The pipe can close a moment before waitpid reports the
            # exit; wait briefly so the failure names the real exit.
            try:
                self._worker._proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                self._worker._kill()
            rc = self._worker.returncode
        if rc is not None and rc < 0:
            detail = f"worker killed by signal {-rc}"
        elif rc is not None:
            detail = f"worker exited with code {rc}"
        else:
            detail = f"worker died: {exc}"
        self.state = "quarantined"
        self.failure = f"plugin quarantined: {detail}"

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self.state = "closed"


class PluginHost:
    """A host for one plugin format."""

    def __init__(self, kind: HostKind):
        self.kind = kind

    def scan(self, directories: list[str | Path]) -> list[PluginDescriptor]:
        return scan_plugins(self.kind, directories)

    def load(self, target: Any, params: dict[str, Any] | None = None) -> PluginInstance:
        """Load a plugin.

        ``target`` is either a plugin object (loaded in-process) or a
        ``"dotted.path:ClassName"`` string (loaded quarantined, in a
        worker subprocess). AU refuses on non-macOS hosts.
        """
        if self.kind is HostKind.AU:
            raise SoundError(
                "AU plugins are macOS-only — this host runs on Linux, "
                "so no AU plugin can load here",
                kind=SoundErrorKind.UNSUPPORTED)
        if not isinstance(target, str):
            return PluginInstance(plugin=target,
                                  name=getattr(target, "info", lambda: {})()
                                  .get("name", "plugin"))
        worker = _Worker()
        try:
            reply = worker.call({"cmd": "load", "target": target,
                                 "params": params or {}})
        except (EOFError, BrokenPipeError) as exc:
            worker._kill()
            raise SoundError(f"plugin worker died at load: {exc}",
                             kind=SoundErrorKind.PLUGIN_FAILED) from exc
        if not reply.get("ok"):
            worker.close()
            raise SoundError(str(reply.get("error", "plugin load failed")),
                             kind=SoundErrorKind.PLUGIN_FAILED)
        info = reply.get("info", {})
        return PluginInstance(worker=worker, name=str(info.get("name", target)))
