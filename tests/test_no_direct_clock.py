"""No direct wall-clock reads in tests (T2.5) — ratchet, not a ban.

Tests must take "now" from :mod:`tests.frozen_clock` and pass it
explicitly (``now=`` / ``ts=``). A test that reads the wall clock
directly passes at noon and fails at 3am (the money-guard "unusual
time" signal fires 23:00–05:00 local).

The ratchet: ``BASELINE`` maps each test file to its count of direct
clock reads (``time.time()`` / ``datetime.now()`` / ``datetime.today()``
calls, AST-counted). The test fails when any file's count GROWS or a
file not in the baseline gains a read. Decreases are allowed — convert
a file fully (count → 0) and remove it from the baseline.

Grandfathered reads stay in the baseline with a reason. The known
ones: live-clock tests where production reads the real clock
internally (idle monitor, weakness scan, coordinator tick budgets)
cannot take a frozen "now" — see the LIVE-CLOCK comments in
``test_autonomy_p9.py``.

``perf_counter`` / ``monotonic`` are NOT counted: duration measurement
is legitimate. Bare ``time()`` (``from time import time``) is not
counted either — the rule names the dotted forms; grep for it if you
suspect evasion.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: file -> direct-clock-read count, measured 2026-10-14 (post-T2.5).
#: Shrink it by converting files to tests.frozen_clock; never grow it.
BASELINE: dict[str, int] = {
    "tests/test_accounts.py": 24,
    "tests/test_accounts_r20.py": 7,
    "tests/test_accounts_sweep.py": 5,
    "tests/test_agents_sweep.py": 4,
    "tests/test_arena_ship.py": 1,
    "tests/test_arena_topics.py": 1,
    "tests/test_audit_r15.py": 2,
    "tests/test_audit_r16.py": 2,
    "tests/test_audit_r17.py": 4,
    "tests/test_audit_r18.py": 6,
    "tests/test_audit_r23.py": 4,
    "tests/test_autonomy.py": 3,
    "tests/test_autonomy_ledger.py": 3,
    "tests/test_autonomy_p9.py": 4,
    "tests/test_autonomy_strategies.py": 6,
    "tests/test_autonomy_sweep.py": 10,
    "tests/test_books_sweep.py": 4,
    "tests/test_briefing.py": 5,
    "tests/test_builders_sweep.py": 5,
    "tests/test_challenges.py": 4,
    "tests/test_character_rebuild.py": 2,
    "tests/test_characters_sweep.py": 7,
    "tests/test_chat_context.py": 9,
    "tests/test_cli_command_center.py": 6,
    "tests/test_cognition_sweep.py": 4,
    "tests/test_cognition_trajectories.py": 2,
    "tests/test_comfy_backend.py": 2,
    "tests/test_community_groups.py": 9,
    "tests/test_community_sweep.py": 14,
    "tests/test_competitor.py": 1,
    "tests/test_complexity_routing.py": 2,
    "tests/test_concurrency_r9.py": 2,
    "tests/test_connectors_sweep.py": 5,
    "tests/test_console_sweep.py": 1,
    "tests/test_content_pipeline.py": 4,
    "tests/test_contentops_pipeline.py": 2,
    "tests/test_contentops_publish.py": 1,
    "tests/test_continuous_tz_power.py": 3,
    "tests/test_core.py": 1,
    "tests/test_core_sweep.py": 2,
    "tests/test_cost_display.py": 7,
    "tests/test_deep_research.py": 2,
    "tests/test_digital_twin.py": 15,
    "tests/test_dj_live.py": 1,
    "tests/test_drift.py": 25,
    "tests/test_email_triage.py": 4,
    "tests/test_error_godtier.py": 2,
    "tests/test_finance_ledger.py": 1,
    "tests/test_finance_phase2.py": 1,
    "tests/test_finance_sweep.py": 3,
    "tests/test_game_identity_username.py": 1,
    "tests/test_games_ai.py": 2,
    "tests/test_games_dm.py": 1,
    "tests/test_games_f2.py": 3,
    "tests/test_games_lifecycle.py": 6,
    "tests/test_games_mastery.py": 6,
    "tests/test_games_matchmaking.py": 1,
    "tests/test_games_quit_hygiene.py": 3,
    "tests/test_games_seasons.py": 2,
    "tests/test_games_sweep.py": 1,
    "tests/test_gateway_inbound.py": 1,
    "tests/test_gifting.py": 1,
    "tests/test_goals_sweep.py": 18,
    "tests/test_goals_tracker.py": 7,
    "tests/test_group_travel.py": 2,
    "tests/test_health_sweep.py": 17,
    "tests/test_health_timeline.py": 3,
    "tests/test_hub_sweep.py": 4,
    "tests/test_id_prefix_resolution.py": 2,
    "tests/test_idempotency.py": 2,
    "tests/test_inbox.py": 6,
    "tests/test_integrations_sweep.py": 5,
    "tests/test_jumia_connector.py": 1,
    "tests/test_learn_sweep.py": 2,
    "tests/test_legal_sweep.py": 2,
    "tests/test_marketing_sweep.py": 6,
    "tests/test_matching_sweep.py": 1,
    "tests/test_memory_deep.py": 3,
    "tests/test_mesh.py": 3,
    "tests/test_mesh_sweep.py": 2,
    "tests/test_mission_replay.py": 2,
    "tests/test_missions_g3.py": 8,
    "tests/test_missions_p9.py": 1,
    "tests/test_missions_sweep.py": 4,
    "tests/test_missions_wiring.py": 4,
    "tests/test_monitor_validation.py": 2,
    "tests/test_mood_patterns.py": 3,
    "tests/test_mqtt.py": 2,
    "tests/test_orchestration_p9.py": 2,
    "tests/test_os_sweep.py": 9,
    "tests/test_os_timeline.py": 2,
    "tests/test_partner_runtime.py": 5,
    "tests/test_partner_social_f2.py": 3,
    "tests/test_partner_sweep.py": 3,
    "tests/test_persona.py": 2,
    "tests/test_phone_reliability.py": 10,
    "tests/test_planning_p9.py": 11,
    "tests/test_planning_sweep.py": 8,
    "tests/test_portfolio.py": 9,
    "tests/test_power_sweep.py": 7,
    "tests/test_presence.py": 5,
    "tests/test_proactive_delivery.py": 3,
    "tests/test_property_sweep.py": 2,
    "tests/test_proxylab_stream1.py": 9,
    "tests/test_r10_startup_shutdown.py": 8,
    "tests/test_repetition.py": 2,
    "tests/test_representation.py": 1,
    "tests/test_research_loop.py": 4,
    "tests/test_research_loop_g2.py": 2,
    "tests/test_research_pipeline.py": 3,
    "tests/test_resources_full.py": 4,
    "tests/test_rooms.py": 1,
    "tests/test_saved_search.py": 2,
    "tests/test_scheduler_deep.py": 13,
    "tests/test_scheduler_delivery.py": 9,
    "tests/test_scheduler_resources.py": 15,
    "tests/test_scheduler_section_b.py": 36,
    "tests/test_scheduler_sweep.py": 27,
    "tests/test_scheduler_upgrades.py": 9,
    "tests/test_search_jobs.py": 4,
    "tests/test_self_improvement_v2.py": 7,
    "tests/test_shutdown_drain.py": 5,
    "tests/test_skills_sweep.py": 3,
    "tests/test_slugify.py": 1,
    "tests/test_snapshots.py": 2,
    "tests/test_social.py": 5,
    "tests/test_social_adapters_deep.py": 4,
    "tests/test_social_sweep.py": 1,
    "tests/test_storage.py": 2,
    "tests/test_storage_sweep.py": 5,
    "tests/test_stream.py": 2,
    "tests/test_stream_sweep.py": 10,
    "tests/test_subagents_coremind_lifecycle.py": 6,
    "tests/test_swarm_specialists.py": 3,
    "tests/test_sync.py": 3,
    "tests/test_sync_sweep.py": 11,
    "tests/test_tools_sweep.py": 2,
    "tests/test_trading.py": 3,
    "tests/test_trading_risk.py": 3,
    "tests/test_training_backends.py": 2,
    "tests/test_travel_sweep.py": 1,
    "tests/test_trial_p9.py": 3,
    "tests/test_triggers.py": 5,
    "tests/test_triggers_sweep.py": 10,
    "tests/test_universal_storage_backup_offsite.py": 1,
    "tests/test_vision_seer.py": 2,
    "tests/test_wave85_games.py": 1,
    "tests/test_wave_d_stream_e.py": 3,
    "tests/test_wave_e_gaps.py": 7,
    "tests/test_wave_f1_honesty.py": 1,
    "tests/test_whatsapp.py": 2,
    "tests/test_wisdom_ingestor.py": 1,
    "tests/test_workspace_sweep.py": 2,
}


def _count_direct_clock_reads(path: Path) -> int:
    """AST count of time.time() / datetime.now() / datetime.today() calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue
        if fn.attr == "time" and isinstance(fn.value, ast.Name) \
                and fn.value.id == "time":
            count += 1
        elif fn.attr in ("now", "today"):
            value = fn.value
            if isinstance(value, ast.Name) and value.id == "datetime":
                count += 1
            elif isinstance(value, ast.Attribute) \
                    and value.attr == "datetime":
                count += 1
    return count


def test_no_direct_clock_reads_grow():
    tests_dir = Path(__file__).parent
    failures: list[str] = []
    for path in sorted(tests_dir.rglob("*.py")):
        rel = str(path.relative_to(tests_dir.parent))
        count = _count_direct_clock_reads(path)
        if rel not in BASELINE:
            if count > 0:
                failures.append(
                    f"{rel}: {count} direct clock read(s), not in the "
                    "baseline — use tests.frozen_clock instead, or add "
                    "a grandfathered entry with a reason")
        elif count > BASELINE[rel]:
            failures.append(
                f"{rel}: {count} direct clock reads, above baseline "
                f"{BASELINE[rel]} — use tests.frozen_clock instead")
    assert not failures, "\n".join(failures)


def test_frozen_clock_fixture_is_clock_free():
    """The fixture itself must not read the wall clock (else it lies)."""
    assert _count_direct_clock_reads(
        Path(__file__).parent / "frozen_clock.py") == 0
