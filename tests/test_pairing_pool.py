"""Pairing-with-code pool (group build §15, first slice).

The registry (durable, never silently reset), the bridge supervisor
(one bridge process per WhatsApp account — pairing spawn carries the
number, the issued code is captured from the bridge's own output,
ports and creds dirs are per-account), the protocol probe, and the
WhatsApp adapter capturing real pairing codes from the bridge.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from nomorals.social.pairing import (
    BridgeSupervisor,
    PairingRegistry,
    PairingStoreError,
    account_id_for,
)


class FakeProc:
    def __init__(self, lines=()):
        self.stdout = iter(list(lines))
        self.terminated = False
        self._poll = None

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0


def make_supervisor(tmp_path, lines=()):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn)
    return sup, registry, spawns


def wait_code(sup, account_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = sup.pairing_code(account_id)
        if got:
            return got
        time.sleep(0.02)
    return None


def test_registry_round_trip_and_counts(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678", "main line")
    assert rec.id == "whatsapp:2348012345678"
    assert rec.port == 8787
    again = PairingRegistry(tmp_path / "paired.json")
    assert [r.id for r in again.all()] == ["whatsapp:2348012345678"]
    assert again.get(rec.id).label == "main line"


def test_registry_corrupt_file_raises_and_quarantines(tmp_path):
    path = tmp_path / "paired.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PairingStoreError):
        PairingRegistry(path)
    assert not path.exists()  # moved aside, not overwritten
    assert list(tmp_path.glob("paired.corrupt-*.json"))


def test_pairing_spawn_carries_number_and_captures_code(tmp_path):
    lines = ["2026 [bridge] PAIRING CODE for +2348012345678: "
             "WXYZ-1234 (type into WhatsApp > Linked Devices)"]
    sup, registry, spawns = make_supervisor(tmp_path, lines)
    rec = sup.pair_whatsapp("+234 801 234 5678")
    argv, env, _proc = spawns[0]
    assert env["WA_PAIRING_CODE"] == "1"
    assert env["WA_PAIRING_NUMBER"] == "2348012345678"
    assert "--creds" in argv and "wa-2348012345678" in " ".join(argv)
    got = wait_code(sup, rec.id)
    assert got is not None and got[0] == "WXYZ-1234"
    assert registry.get(rec.id).status == "pairing"
    # a second account takes the next port
    rec2 = sup.pair_whatsapp("2349098765432")
    assert rec2.port == 8788
    assert sup.counts(probe=False)["registered"] == 2


def test_bad_number_rejected():
    from nomorals.social.pairing import PairingRegistry as PR
    import tempfile
    from pathlib import Path

    registry = PR(Path(tempfile.mkdtemp()) / "p.json")
    sup = BridgeSupervisor(registry, bridge_path="/x.mjs",
                           sessions_root="/tmp/s",
                           spawn=lambda a, e: FakeProc())
    with pytest.raises(ValueError):
        sup.pair_whatsapp("123")


def test_stop_and_start_lifecycle(tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    assert sup.stop(rec.id) is True
    assert spawns[0][2].terminated is True
    assert registry.get(rec.id).status == "stopped"
    sup.start(rec.id)
    assert len(spawns) == 2
    # a plain start is a reconnect — no pairing env
    assert "WA_PAIRING_NUMBER" not in spawns[1][1]
    assert sup.remove(rec.id) is True
    assert registry.all() == []


def test_probe_reads_bridge_status(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    # nothing listening → down
    assert sup.probe(rec, timeout=0.5) == "down"
    # a stand-in bridge answering the status request
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", rec.port))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        conn.recv(4096)
        conn.sendall(b'{"type":"status","state":"open",'
                     b'"user":"2348012345678@s.whatsapp.net"}\n')
        conn.close()
        server.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    assert sup.probe(rec, timeout=2.0) == "linked"
    t.join(timeout=2.0)


def test_adapter_captures_pairing_code():
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_enabled=False)
    adapter._dispatch({"type": "pairing_code", "data": "ABCD-EFGH"},
                      handler=lambda m: None)
    assert adapter.last_pairing_code == "ABCD-EFGH"
    health = adapter.health()
    assert health["pairing_code_pending"] is True
    assert health["pairing_code_age_s"] is not None


def test_account_id_slug():
    assert account_id_for("whatsapp", "234 801") == "whatsapp:234-801"
