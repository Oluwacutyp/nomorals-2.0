"""Wave 6: the owner's per-device control gate.

The posture under test (warn-first standing rule): control is OFF by
default for every device; only an explicit owner action records an
enable; a corrupt gate file raises instead of guessing.
"""
from __future__ import annotations

import pytest

from nomorals.core.errors import StorageError
from nomorals.integrations.home import ControlGate
from nomorals.integrations.home.errors import ControlNotEnabledError


class TestGateDefaults:
    def test_unrecorded_device_is_disabled(self):
        gate = ControlGate()
        assert gate.is_enabled("wemo", "uuid:X") is False

    def test_require_enabled_raises_named_refusal(self):
        gate = ControlGate()
        with pytest.raises(ControlNotEnabledError) as ctx:
            gate.require_enabled("zwave", "zwave-3", action="turn_on")
        err = ctx.value
        assert err.code == "home.control_not_enabled"
        assert err.details["backend"] == "zwave"
        assert err.details["device_id"] == "zwave-3"
        assert err.details["action"] == "turn_on"
        assert "how_to_enable" in err.details

    def test_enable_then_require_passes(self):
        gate = ControlGate()
        gate.set_enabled("wemo", "uuid:X", True, by="owner")
        assert gate.is_enabled("wemo", "uuid:X") is True
        gate.require_enabled("wemo", "uuid:X")  # no raise

    def test_disable_revokes(self):
        gate = ControlGate()
        gate.set_enabled("wemo", "uuid:X", True, by="owner")
        gate.set_enabled("wemo", "uuid:X", False, by="owner")
        assert gate.is_enabled("wemo", "uuid:X") is False
        with pytest.raises(ControlNotEnabledError):
            gate.require_enabled("wemo", "uuid:X")

    def test_enable_is_per_device_not_per_backend(self):
        gate = ControlGate()
        gate.set_enabled("wemo", "uuid:A", True, by="owner")
        assert gate.is_enabled("wemo", "uuid:B") is False
        assert gate.is_enabled("zwave", "uuid:A") is False

    def test_set_enabled_records_who_and_when(self):
        gate = ControlGate()
        record = gate.set_enabled("wemo", "uuid:X", True, by="owner",
                                  note="kitchen plug ok")
        assert record["updated_by"] == "owner"
        assert record["updated_at"] > 0
        assert record["note"] == "kitchen plug ok"
        assert record["enabled"] is True


class TestGatePersistence:
    def test_records_survive_reload(self, tmp_path):
        path = tmp_path / "gate.json"
        gate = ControlGate(path)
        gate.set_enabled("zwave", "zwave-2", True, by="owner")
        gate.set_enabled("wemo", "uuid:Y", False, by="owner")
        reloaded = ControlGate(path)
        assert reloaded.is_enabled("zwave", "zwave-2") is True
        assert reloaded.is_enabled("wemo", "uuid:Y") is False

    def test_corrupt_gate_file_raises_not_guesses(self, tmp_path):
        path = tmp_path / "gate.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StorageError):
            ControlGate(path)

    def test_wrong_shape_gate_file_raises(self, tmp_path):
        path = tmp_path / "gate.json"
        path.write_text('["enabled"]', encoding="utf-8")
        with pytest.raises(StorageError):
            ControlGate(path)

    def test_status_lists_records_as_data(self, tmp_path):
        gate = ControlGate(tmp_path / "gate.json")
        gate.set_enabled("wemo", "uuid:B", True, by="owner")
        gate.set_enabled("wemo", "uuid:A", False, by="owner")
        status = gate.status()
        assert [r["device_id"] for r in status] == ["uuid:A", "uuid:B"]
        assert status[1]["enabled"] is True

    def test_status_for_marks_unrecorded_disabled(self):
        gate = ControlGate()
        gate.set_enabled("wemo", "uuid:A", True, by="owner")
        rows = gate.status_for("wemo", ["uuid:A", "uuid:ZZ"])
        assert rows[0]["control_enabled"] is True
        assert rows[1]["control_enabled"] is False
        assert rows[1]["record"] is None
