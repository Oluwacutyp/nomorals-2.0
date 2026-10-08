"""Tests for the home digital twin (#64). All offline."""

import os
import time

import pytest

from nomorals.integrations.digital_twin import (
    HomeTwin, MIN_RHYTHM_DAYS,
)

DAY = 86400


@pytest.fixture()
def twin(tmp_path):
    return HomeTwin(db_path=str(tmp_path / "twin.db"))


def _seed(twin, base, days=3):
    """3 days of normal: kitchen light on 18-23h, front door used daytime."""
    for d in range(days):
        day = base - (days - 1 - d) * DAY
        for h in (18, 19, 20, 21, 22):
            twin.ingest("light.kitchen", "on", ts=day + h * 3600)
            twin.ingest("light.kitchen", "off", ts=day + h * 3600 + 1800)
        # door opens/closes during the day, never at night
        for h in (8, 12, 17):
            twin.ingest("binary_sensor.front_door", "on",
                        ts=day + h * 3600)
            twin.ingest("binary_sensor.front_door", "off",
                        ts=day + h * 3600 + 300)


# ── ingestion ────────────────────────────────────────────────────

def test_ingest_and_query(twin):
    now = time.time()
    assert twin.ingest("light.kitchen", "on", ts=now)
    assert twin.ingest("light.kitchen", "on", ts=now + 1) is False  # dedup
    assert twin.ingest("light.kitchen", "off", ts=now + 2)
    events = twin.query("light.kitchen", now - 60)
    assert [e.state for e in events] == ["on", "off"]


def test_ingest_never_raises(twin):
    assert twin.ingest("", "on") is False
    assert twin.ingest(None, "on") is False  # type: ignore[arg-type]


def test_current_state(twin):
    now = time.time()
    twin.ingest("light.kitchen", "on", ts=now)
    twin.ingest("light.kitchen", "off", ts=now + 1)
    twin.ingest("lock.front", "locked", ts=now)
    current = twin.current_state()
    assert current["light.kitchen"] == "off"
    assert current["lock.front"] == "locked"


def test_changes_since(twin):
    now = time.time()
    twin.ingest("light.kitchen", "on", ts=now - 3600)
    twin.ingest("light.bedroom", "on", ts=now - 1800)
    changes = twin.changes_since(now - 7200)
    assert len(changes) == 2
    assert twin.what_changed(now - 7200).startswith("2 change(s)")


def test_community_refused(tmp_path):
    with pytest.raises(PermissionError):
        HomeTwin(db_path=str(tmp_path / "t.db"), community=True)


# ── derived signals ──────────────────────────────────────────────

def test_occupancy_motion(twin):
    now = time.time()
    twin.ingest("binary_sensor.hall_motion", "on", ts=now - 300)
    occ = twin.occupancy(now=now)
    assert occ["likely_home"] is True
    assert "motion" in occ["basis"]


def test_occupancy_quiet(twin):
    now = time.time()
    twin.ingest("light.kitchen", "on", ts=now - 7200)
    occ = twin.occupancy(now=now)
    assert occ["likely_home"] is False


def test_energy_baseline(twin):
    now = time.time()
    for i in range(12):
        twin.ingest("sensor.fridge_power", str(95 + i), ts=now - i * 3600)
    base = twin.energy_baseline("sensor.fridge_power", now=now)
    assert base is not None
    assert base["n"] == 12
    assert 95 < base["mean"] < 110


def test_energy_baseline_thin_data(twin):
    now = time.time()
    twin.ingest("sensor.fridge_power", "100", ts=now)
    assert twin.energy_baseline("sensor.fridge_power", now=now) is None


def test_rhythms_need_history(twin):
    base = time.time()
    _seed(twin, base, days=3)
    assert twin.rhythms(now=base) == []  # < 7 days → honest empty


def test_rhythms_detected(twin):
    base = time.time()
    _seed(twin, base, days=10)
    rhythms = twin.rhythms(now=base)
    kitchen = [r for r in rhythms if r.entity_id == "light.kitchen"]
    assert kitchen, "kitchen rhythm should be learned"
    # 5 contiguous "on" hours seeded — timezone-agnostic check
    import re
    m = re.search(r"(\d{2}):00–(\d{2}):00", kitchen[0].description)
    assert m and int(m.group(2)) - int(m.group(1)) == 5
    assert kitchen[0].confidence >= 0.7


# ── anomalies ────────────────────────────────────────────────────

def test_power_spike(twin):
    now = time.time()
    for i in range(24):
        twin.ingest("sensor.fridge_power", str(98 + (i % 5)),
                    ts=now - (i + 1) * 3600)
    twin.ingest("sensor.fridge_power", "900", ts=now)
    anomalies = twin.anomalies(now=now)
    spikes = [a for a in anomalies if a.kind == "power_spike"]
    assert spikes, "900 vs baseline ~100 should spike"
    assert "sensor.fridge_power" in spikes[0].message


def test_no_spike_when_stable(twin):
    now = time.time()
    for i in range(24):
        twin.ingest("sensor.fridge_power", str(100 + (i % 3)),
                    ts=now - (i + 1) * 3600)
    twin.ingest("sensor.fridge_power", "101", ts=now)
    anomalies = twin.anomalies(now=now)
    assert not [a for a in anomalies if a.kind == "power_spike"]


def test_unexpected_offline(twin):
    now = time.time()
    twin.ingest("sensor.temp", "22", ts=now - 3600)
    twin.ingest("sensor.temp", "unavailable", ts=now - 600)
    anomalies = twin.anomalies(now=now)
    off = [a for a in anomalies if a.kind == "unexpected_offline"]
    assert off, "recently-ok sensor going unavailable should flag"


def test_unusual_door_open(twin):
    # Deterministic: build history where the door is only ever "on"
    # at local hours 9-17, then open it at local hour 2.
    base = time.time()
    DAY = 86400
    lt = time.localtime(base)
    # anchor: 10 days ago at local midnight
    anchor = base - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec) - 9 * DAY
    for d in range(10):
        day = anchor + d * DAY
        for h in (9, 13, 17):  # local hours — daytime only
            twin.ingest("binary_sensor.front_door", "on", ts=day + h * 3600)
            twin.ingest("binary_sensor.front_door", "off",
                        ts=day + h * 3600 + 300)
    assert twin.history_days() >= MIN_RHYTHM_DAYS
    odd_ts = anchor + 9 * DAY + 2 * 3600  # local 2am, last day
    assert time.localtime(odd_ts).tm_hour == 2
    twin.ingest("binary_sensor.front_door", "on", ts=odd_ts)
    anomalies = twin.anomalies(now=odd_ts + 60)
    unusual = [a for a in anomalies if a.kind == "unusual_open"]
    assert unusual, "door open at 2am should be unusual"


# ── summary ──────────────────────────────────────────────────────

def test_summary_empty(twin):
    s = twin.summary()
    assert "no home data yet" in s


def test_summary_format(twin):
    now = time.time()
    twin.ingest("light.kitchen", "on", ts=now)
    twin.ingest("lock.front", "locked", ts=now)
    s = twin.summary(now=now)
    assert "your home right now" in s
    assert "kitchen: on" in s
    assert "front: locked" in s
    assert "presence:" in s


def test_summary_includes_anomalies(twin):
    now = time.time()
    for i in range(24):
        twin.ingest("sensor.fridge_power", str(98 + (i % 5)),
                    ts=now - (i + 1) * 3600)
    twin.ingest("sensor.fridge_power", "900", ts=now)
    s = twin.summary(now=now)
    assert "unusual" in s
