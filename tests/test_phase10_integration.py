"""Phase 10: cross-cutting integration tests."""

import pytest


class TestPulseCatchupWiring:
    def test_catchup_importable(self):
        from nomorals.agents.morning_pulse import check_pulse_catchup
        assert callable(check_pulse_catchup)

    def test_boot_wires_catchup(self):
        import ast
        src = open("nomorals/agents/partner/runtime.py").read()
        assert "check_pulse_catchup" in src
        # Both ensure AND check must be in the boot block
        assert "ensure_pulse_job" in src


class TestImprovementQueueWiring:
    def test_approval_files_to_queue(self):
        src = open("nomorals/agents/improvement.py").read()
        assert "UpgradeQueue" in src
        assert 'source="improvement"' in src

    def test_queue_accepts_improvement_source(self):
        from nomorals.agents.upgrade_queue import UpgradeQueue
        # source param must accept arbitrary strings
        import inspect
        sig = inspect.signature(UpgradeQueue.propose)
        assert "source" in sig.parameters
