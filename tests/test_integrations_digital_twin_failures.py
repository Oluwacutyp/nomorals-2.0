"""Tail-74 pins for nomorals/integrations/digital_twin.py —
0 converts / 6 exclusions (88, 160, 298, 315, 486, 818).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, then the failure half asserting the pinned
degradation. The file contains no exception handlers, so it
adds zero census sites.

Module posture: the twin never breaks streams — ingest()
never raises and records state-write failures itself via the
outer _log.debug guard (not a census site). NOT-recording is
pinned as caplog silence on the module logger
(nomorals.integrations.digital_twin) for the silent sites.

Verdict notes (full grounds in the tail-74 SURVEY.md):
- 88: WAL pragma failure -> bare pass; "WAL is a nicety" is
  the code's own contract — journal mode degrades, the twin
  constructs and ingests.
- 160: metadata enrichment is best-effort by the file's own
  comment; the state write already landed (its failure is the
  outer guard's, recorded) and the callee register_entity
  self-records its own failures.
- 298 / 315: corrupt attributes JSON in a stored row -> the
  StateEvent survives with state+ts intact and attributes {};
  no in-module consumer reads .attributes at all.
- 486: VACUUM failure -> bare pass; the delete committed and
  the removed count returns honestly — VACUUM is "a nicety".
- 818: mqtt_client import failure -> locally-derived entity
  id fallback (resolution ladder); ingestion continues.
  Silent rung — observability sketch shelved in the SURVEY.
"""

import logging
import sqlite3
import sys
import time
from unittest import mock

import nomorals.integrations.digital_twin as T
from nomorals.integrations.digital_twin import HomeTwin

TWIN_LOG = "nomorals.integrations.digital_twin"


def _twin(tmp_path, name="twin.db"):
    return HomeTwin(db_path=str(tmp_path / name))


def _silent(caplog):
    return [r for r in caplog.records if r.name == TWIN_LOG]


def test_88_wal_pragma_failure_still_constructs_and_ingests(
        tmp_path, caplog):
    control = _twin(tmp_path, "control.db")
    assert control.ingest("light.lamp", "on") is True
    real_connect = sqlite3.connect

    class PragmaShy:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **k):
            if "journal_mode" in str(sql):
                raise sqlite3.OperationalError("wal unavailable")
            return self._real.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with mock.patch.object(
            T.sqlite3, "connect",
            lambda *a, **k: PragmaShy(real_connect(*a, **k))):
        with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
            caplog.clear()
            twin = _twin(tmp_path, "shy.db")
            assert twin.ingest("light.lamp", "on") is True
    assert twin.current_state() == {"light.lamp": "on"}
    assert _silent(caplog) == []


def test_160_enrichment_failure_keeps_the_recorded_state(
        tmp_path, caplog):
    twin = _twin(tmp_path)
    assert twin.ingest("light.lamp", "on",
                       attributes={"device_class": "light"}) is True
    assert twin.entity_meta("light.lamp")["device_class"] == "light"
    with mock.patch.object(HomeTwin, "register_entity",
                           side_effect=RuntimeError("meta gone")):
        with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
            caplog.clear()
            assert twin.ingest("light.lamp", "off") is True
    assert twin.current_state() == {"light.lamp": "off"}
    assert _silent(caplog) == []


def _corrupt_row(twin, entity_id, state):
    with twin._lock:
        twin._db.execute(
            "INSERT INTO states (entity_id, state, attributes, ts)"
            " VALUES (?, ?, ?, ?)",
            (entity_id, state, "{corrupt-json", time.time()))
        twin._db.commit()


def test_298_query_corrupt_attributes_degrade_to_empty(tmp_path, caplog):
    twin = _twin(tmp_path)
    twin.ingest("sensor.door", "closed",
                attributes={"battery": 90})
    healthy = twin.query("sensor.door", 0)[0]
    assert healthy.attributes == {"battery": 90}  # attempt proven
    _corrupt_row(twin, "sensor.window", "open")
    with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
        caplog.clear()
        ev = twin.query("sensor.window", 0)[0]
    assert ev.state == "open"
    assert ev.attributes == {}
    assert _silent(caplog) == []


def test_315_changes_since_corrupt_attributes_degrade_to_empty(
        tmp_path, caplog):
    twin = _twin(tmp_path)
    twin.ingest("sensor.door", "closed",
                attributes={"battery": 90})
    healthy = [e for e in twin.changes_since(0)
               if e.entity_id == "sensor.door"][0]
    assert healthy.attributes == {"battery": 90}  # attempt proven
    _corrupt_row(twin, "sensor.window", "open")
    with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
        caplog.clear()
        ev = [e for e in twin.changes_since(0)
              if e.entity_id == "sensor.window"][0]
    assert ev.state == "open"
    assert ev.attributes == {}
    assert _silent(caplog) == []


def test_486_vacuum_failure_keeps_the_prune_result(tmp_path, caplog):
    twin = _twin(tmp_path)
    old = time.time() - 200 * T.DAY_S
    twin.ingest("sensor.old", "on", ts=old)
    assert twin.prune(keep_days=90) == 1  # control: vacuum works

    twin2 = _twin(tmp_path, "twin2.db")
    twin2.ingest("sensor.old", "on", ts=old)

    class VacuumShy:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a, **k):
            if str(sql).strip().upper() == "VACUUM":
                raise sqlite3.OperationalError("vacuum blocked")
            return self._real.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_db = twin2._db
    twin2._db = VacuumShy(real_db)
    with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
        caplog.clear()
        removed = twin2.prune(keep_days=90)
    twin2._db = real_db
    assert removed == 1
    assert twin2.current_state() == {}
    # the only record is the designed prune info — the VACUUM
    # pass itself records nothing
    msgs = [r.getMessage() for r in caplog.records
            if r.name == TWIN_LOG]
    assert msgs == ["twin pruned 1 rows older than 90 days"]


def test_818_mqtt_import_failure_falls_back_to_local_entity_id(
        tmp_path, caplog):
    twin = _twin(tmp_path)
    captured = {}

    class FakeBridge:
        def subscribe(self, topic, cb):
            captured["cb"] = cb

    T.attach_twin_to_mqtt(FakeBridge(), twin)
    captured["cb"]("zigbee2mqtt/Kitchen Plug", {"state": "OFF"})
    assert "zigbee2mqtt.kitchen_plug" in twin.entity_ids()

    captured.clear()
    with mock.patch.dict(
            sys.modules,
            {"nomorals.integrations.mqtt_client": None}):
        with caplog.at_level(logging.DEBUG, logger=TWIN_LOG):
            caplog.clear()
            T.attach_twin_to_mqtt(FakeBridge(), twin)
            captured["cb"]("zigbee2mqtt/Living Room Lamp",
                           {"state": "ON"})
    assert "zigbee2mqtt.living room lamp" in twin.entity_ids()
    assert "zigbee2mqtt.living_room_lamp" not in twin.entity_ids()
    assert _silent(caplog) == []
