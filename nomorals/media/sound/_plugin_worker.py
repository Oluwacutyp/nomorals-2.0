"""Quarantined plugin worker — runs one plugin out of process.

Started by :class:`nomorals.media.sound.hosts.PluginHost` as
``python -m nomorals.media.sound._plugin_worker``. Speaks
length-prefixed JSON frames on stdin/stdout (4-byte big-endian
length + payload). Audio rides base64 inside the JSON — simple and
unambiguous on a pipe, fast enough for block-sized plugin work.

Commands: ``load`` (dotted path + params), ``process`` (base64
float32 block -> base64 float32 block), ``close``. A plugin that
aborts or segfaults kills this process and nothing else — the host
sees EOF and quarantines the instance.
"""

from __future__ import annotations

import base64
import importlib
import json
import struct
import sys

import numpy as np


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("stdin closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _reply(out, message: dict) -> None:
    out.write(_frame(json.dumps(message).encode()))
    out.flush()


def main() -> int:
    stream_in = sys.stdin.buffer
    stream_out = sys.stdout.buffer
    plugin = None
    while True:
        try:
            raw_len = _read_exact(stream_in, 4)
        except EOFError:
            return 0
        (length,) = struct.unpack(">I", raw_len)
        try:
            message = json.loads(_read_exact(stream_in, length).decode())
        except EOFError:
            return 0
        cmd = message.get("cmd")
        try:
            if cmd == "close":
                return 0
            if cmd == "load":
                target = str(message["target"])
                module_name, _, class_name = target.partition(":")
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                plugin = cls(**(message.get("params") or {}))
                info = plugin.info() if hasattr(plugin, "info") else {}
                _reply(stream_out, {"ok": True, "info": info})
            elif cmd == "process":
                if plugin is None:
                    _reply(stream_out, {"ok": False,
                                        "error": "no plugin loaded"})
                    continue
                data = base64.b64decode(message["data"])
                block = np.frombuffer(data, dtype=np.float32)
                out = np.asarray(
                    plugin.process(block, int(message["sample_rate"])),
                    dtype=np.float32)
                _reply(stream_out, {
                    "ok": True,
                    "data": base64.b64encode(out.tobytes()).decode(),
                })
            else:
                _reply(stream_out, {"ok": False,
                                    "error": f"unknown command {cmd!r}"})
        except Exception as exc:  # noqa: BLE001 — report, stay alive
            try:
                _reply(stream_out, {"ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
