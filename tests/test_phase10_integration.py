"""Phase 10: cross-cutting integration tests."""

import pytest


class TestPulseCatchupWiring:
    def test_catchup_importable(self):
        from nomorals.agents.morning_pulse import check_pulse_catchup
        assert callable(check_pulse_catchup)

    def test_boot_wires_catchup(self):
        src = open("nomorals/agents/partner/runtime.py").read()
        # Boot wires the multi-slot successor: 443b784a (afternoon +
        # evening briefing slots) generalized the direct phase-10
        # check_pulse_catchup boot call into check_all_catchup, which
        # walks every slot at boot.
        assert "check_all_catchup" in src
        # Both ensure AND check must be in the boot block
        assert "ensure_pulse_jobs" in src
        # The chain must still REACH the original pulse catch-up: the
        # all-slot walk delegates per slot, and the morning slot is
        # the existing check_pulse_catchup path.
        pulse = open("nomorals/agents/morning_pulse.py").read()
        assert "check_slot_catchup(context, slot)" in pulse
        assert "return check_pulse_catchup(context)" in pulse


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
