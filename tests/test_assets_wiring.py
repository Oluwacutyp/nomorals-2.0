"""Vehicle / asset maintenance, wired into production (E-6).

MORE_TO_ADD gap 7: the domain (nomorals/assets/maintenance.py)
existed and was unit-pinned (tests/test_more_gaps710.py) but nothing
in production reached it — no durable store, no chat command, no CLI.
Every test here drives the PRODUCTION wiring: the durable AssetLedger
and the /assets control — never the dormant organ in isolation.

Teeth: on the pristine base each test fails for the missing wiring
(AssetLedger takes no path, no _control_assets, no "assets" command
kind).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from nomorals.assets.maintenance import Asset, AssetLedger, ServiceRecord
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    help_text,
)


def _reach_host(tmp_path):
    """Production control host on a throwaway home (decisions/focus harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _add_car(host, name="Camry", kind="vehicle", interval_km=5000,
             interval_days=180, odometer_km=14000):
    out = host._control_assets(
        f"add {name} | {kind} | {interval_km} | {interval_days} | {odometer_km}")
    assert out.startswith("recorded: "), out
    return out.split(":")[1].strip().split()[0]


# ── durable store ───────────────────────────────────────────────────


def test_ledger_persists_across_instances(tmp_path):
    path = str(tmp_path / "assets.json")
    ledger = AssetLedger(path)
    car = ledger.add_asset(Asset(
        name="Camry", kind="vehicle", interval_km=5000,
        last_service_km=10000, odometer_km=14000))
    ledger.log_service(ServiceRecord(
        asset_id=car.id, service_date=date(2026, 11, 1),
        description="oil change", cost_minor=4500000,
        odometer_km=15000, vendor="AutoServe"))

    fresh = AssetLedger(path)
    got = fresh.get_asset(car.id)
    assert got.name == "Camry" and got.kind == "vehicle"
    assert got.odometer_km == 15000 and got.last_service_km == 15000
    assert got.last_service_date == date(2026, 11, 1)
    assert len(fresh.records) == 1
    assert fresh.records[0].description == "oil change"
    assert fresh.records[0].cost_minor == 4500000
    assert fresh.total_cost(car.id) == 4500000


def test_ledger_tolerant_load(tmp_path):
    missing = AssetLedger(str(tmp_path / "nope.json"))
    assert missing.assets == {} and missing.records == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert AssetLedger(str(corrupt)).assets == {}

    mixed = tmp_path / "mixed.json"
    good = Asset(name="Gen", kind="generator")
    mixed.write_text(json.dumps({
        "assets": [good.to_dict(), {"name": 123, "kind": "vehicle"}],
        "records": [{"asset_id": good.id, "service_date": "2026-01-01",
                     "description": "oil", "cost_minor": 1000},
                    {"asset_id": good.id, "service_date": "junk",
                     "description": "bad", "cost_minor": 5}],
    }), encoding="utf-8")
    loaded = AssetLedger(str(mixed))
    assert list(loaded.assets) == [good.id]
    assert len(loaded.records) == 1 and loaded.records[0].cost_minor == 1000


def test_asset_and_record_dict_round_trip():
    asset = Asset(name="Camry", kind="vehicle", interval_km=5000,
                  purchase_date=date(2024, 3, 1),
                  last_service_date=date(2026, 11, 1))
    assert Asset.from_dict(asset.to_dict()) == asset
    record = ServiceRecord(asset_id=asset.id, service_date=date(2026, 11, 1),
                           description="oil change", cost_minor=4500000,
                           odometer_km=15000, vendor="AutoServe",
                           receipt_ref="RC-1")
    assert ServiceRecord.from_dict(record.to_dict()) == record
    bare = Asset(name="Laptop", kind="laptop")
    assert Asset.from_dict(bare.to_dict()) == bare


# ── /assets control ─────────────────────────────────────────────────


def test_assets_add_list_show(tmp_path):
    host = _reach_host(tmp_path)
    asset_id = _add_car(host)
    out = host._control_assets("list")
    assert "Camry" in out and asset_id in out and "vehicle" in out
    shown = host._control_assets(f"show {asset_id}")
    assert "Camry" in shown and "no services logged" in shown


def test_assets_add_rejects_bad_kind_and_interval(tmp_path):
    host = _reach_host(tmp_path)
    assert "unknown asset kind" in host._control_assets("add Boat | boat")
    assert "interval_km" in host._control_assets(
        "add Camry | vehicle | -5")
    assert "usage:" in host._control_assets("add")


def test_assets_service_costs_and_history(tmp_path):
    host = _reach_host(tmp_path)
    asset_id = _add_car(host)
    out = host._control_assets(
        f"service {asset_id} oil change | 45000 | 15000 | AutoServe")
    assert "logged" in out and "₦45,000.00" in out
    costs = host._control_assets("costs")
    assert "Camry" in costs and "₦45,000.00" in costs
    shown = host._control_assets(f"show {asset_id}")
    assert "oil change" in shown and "AutoServe" in shown
    assert "15,000 km" in shown


def test_assets_service_backdate_and_due(tmp_path):
    host = _reach_host(tmp_path)
    asset_id = _add_car(host, interval_days=90)
    past = (date.today() - timedelta(days=100)).isoformat()
    host._control_assets(
        f"service {asset_id} first service | 0 | 14000 | | {past}")
    due = host._control_assets("due")
    assert "Camry" in due and "OVERDUE" in due and "time" in due
    assert "no services due" not in due
    # Overdue never hides behind the window: due 0 still lists it.
    assert "Camry" in host._control_assets("due 0")


def test_assets_service_rejects_bad_input(tmp_path):
    host = _reach_host(tmp_path)
    asset_id = _add_car(host)
    assert "no asset matching" in host._control_assets(
        "service nope oil change")
    assert "cost" in host._control_assets(
        f"service {asset_id} oil change | twelve")
    assert "usage:" in host._control_assets("service")
    host._control_assets(f"service {asset_id} oil change | 100 | 15000")
    assert "backwards" in host._control_assets(
        f"service {asset_id} second service | 100 | 14000")


def test_assets_unknown_id_and_empty_list(tmp_path):
    host = _reach_host(tmp_path)
    assert "no assets" in host._control_assets("list")
    assert "no asset matching" in host._control_assets("show nope")
    assert "no services due" in host._control_assets("due")


# ── registry ────────────────────────────────────────────────────────


def test_assets_registered():
    assert "assets" in CONTROL_COMMANDS
    slots = [title for title, kinds in LIST_GROUPS if "assets" in kinds]
    assert slots == ["memory & thinking"]
    assert "assets" in LIST_ONELINERS and "/assets" in LIST_ONELINERS["assets"]
    assert "/assets" in help_text()
