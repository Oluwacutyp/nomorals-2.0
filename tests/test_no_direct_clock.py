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
    # Grandfathered (batch H18): 3 reads stay on the real clock.
    # (1) The pulse markers — check_pulse_catchup (morning_pulse.py)
    # suppresses only when last_ts >= due on the real clock, and the
    # pulse is enabled by default, so start()'s check_all_catchup
    # would fire the catch-up pulse into the adapter at boot if the
    # markers were frozen. (2) The same-pid live claim —
    # MissionRunner._claim_is_live (runner.py) returns
    # age <= _stale_heartbeat_after() (900s) for our own pid, so a
    # frozen claim reads dead and the mission would resume instead
    # of being skipped. Foreign-pid claims are frozen (the verdict
    # is purely _process_alive), as are the dead-pid stale seeds,
    # the briefing seed, the milestone seed, and the wait loops
    # (monotonic deadlines).
    "tests/test_boot_resume.py": 3,
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
    # Grandfathered (batch H39): 1 read stays on the real clock.
    # s07a's fresh heartbeat (test_d15p6_acceptance_battery.py:290)
    # must be stamped on the real clock — mission_liveness
    # (missions/mission.py:111) computes the heartbeat age against
    # time.time() with no now seam, and MissionsAdapter.discover
    # calls it without passing its own `now`
    # (outcome_steward.py:249), so a frozen MIDDAY heartbeat reads
    # ~32 days stale and the healthy RUNNING mission opens an
    # "orphaned" case instead of opening nothing. Threading a now
    # seam through mission_liveness would be a production API change
    # in a hygiene lane for a non-regression (H21 precedent). The
    # file's other 5 reads are frozen to MIDDAY_TS: the mission
    # created/updated seeds are only compared as ages against
    # thresholds they clear by orders of magnitude (PENDING_STALE_S
    # 86400s; the reap limit is 5s of silence; UNSEEN_RESULT_S 600s),
    # and the dead-heartbeat seeds only need to read "older than the
    # limit" on any clock.
    "tests/test_d15p6_acceptance_battery.py": 1,
    "tests/test_deep_research.py": 2,
    "tests/test_digital_twin.py": 15,
    "tests/test_dj_live.py": 1,
    "tests/test_drift.py": 25,
    # Grandfathered (batch H14): the two failover sweeps must fire
    # on deferrals registered at the real clock — register()
    # stamps created_at/deadline from time.time()
    # (peer_deferral.py) and expire() fires only when deadline
    # <= now, so a frozen now (~32 days before the deadline)
    # never expires them. The file's other reads (wait loops,
    # seeds, empty-pending sweeps, retry stamp) are frozen.
    "tests/test_dupfix4_group.py": 2,
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
    # Grandfathered (batch H41): 3 reads stay on the real clock.
    # The drift briefing provider ignores its `since` arg and calls
    # monitor.check() with no now (drift.py:933), and the
    # __health_drift_tick__ tool calls run_health_drift_tick_once
    # with no now (tools/autonomy.py:1071-1072) — both window the
    # last 3 days against time.time() (drift.py:367-368), so a frozen
    # MIDDAY seed sits ~35 days stale and falls outside the window:
    # check() returns None for "not enough data" (drift.py:389).
    # The drift-provider test would fail outright; the
    # stable-provider and tool tests would pass vacuously (proven:
    # frozen seeds + real-now check() -> None, while
    # check(now=MIDDAY_TS) -> ('act', ['sleep_debt'])). Threading
    # now seams through the provider and the tool would be a
    # production API change in a hygiene lane for a non-regression
    # (H21 precedent). The file's other 7 reads are frozen to
    # MIDDAY_TS: the tick tests pass `now` explicitly through
    # run_health_drift_tick_once into monitor.check(now=...), and
    # the never-raises test's instant is never read (construction
    # booms first).
    "tests/test_gapfix_health_drift.py": 3,
    # Grandfathered (batch H40): 8 reads stay on the real clock — the
    # run seeds in PickTopicsTests (3 tests) and TickTests stamp a run
    # "10 seconds ago" that pick_topics judges on its own real clock
    # (research_loop.py:303 takes now = time.time() with no now seam,
    # and topic_ages compares started_at against it). A frozen MIDDAY
    # seed reads ~35 days old on the real clock, so "just covered"
    # (eligibility at the floor, ranked last) becomes "fully eligible"
    # and the curve ordering the tests pin breaks. Threading a now
    # seam through pick_topics/tick would be a production API change
    # in a hygiene lane for a non-regression (H21 precedent). The
    # file's other 16 reads are frozen to MIDDAY_TS: gap/watch seeds
    # (harvest ignores ts except ORDER BY), kg-stale/proposal seeds
    # (the harvesters take an explicit now, passed as MIDDAY_TS),
    # the ages-test now (topic_ages takes now=), and the ledger seed
    # (first_seen_at is tiebreak data only).
    "tests/test_gapfix_topic_engine.py": 8,
    "tests/test_gateway_inbound.py": 1,
    "tests/test_gifting.py": 1,
    "tests/test_goals_sweep.py": 18,
    "tests/test_goals_tracker.py": 7,
    # Grandfathered (batch H8): the feed/claim test opens a drop
    # with an explicit now and claims it through the runtime
    # control, which stamps the real clock — claim_sysdrop
    # refuses a drop whose expires (open + 900s) has passed, so
    # the open must stay on the real clock.
    "tests/test_group_conflict.py": 1,
    # Grandfathered (batch H14): the coincident-window sweeps,
    # the role-stagger expire() calls and the both-unknown
    # sweeps are proofs about real-clock-relative windows —
    # the deferrals register at the real clock (register()
    # stamps created_at/deadline from time.time(),
    # peer_deferral.py) and expire() fires only when deadline
    # <= now, so a frozen now (~32 days before the deadlines)
    # expires nothing and the failover claim never runs. The
    # file's other reads (seeds, the empty-pending sweep) are
    # frozen.
    "tests/test_groupdup2_followups.py": 7,
    # Grandfathered (batch H28): the live failover sweep is a
    # proof about a real-clock-relative window — the deferral
    # registers at the real clock (register() stamps
    # created_at/deadline from time.time(), peer_deferral.py)
    # and expire() fires only when deadline <= now, so a frozen
    # now (~32 days before the deadline) expires nothing and
    # the failover claim never runs. The file's other reads
    # (pulse/briefing seeds, the empty-pending sweeps, the
    # bridge ts-ms) are frozen.
    "tests/test_groupdup_fix.py": 1,
    # Grandfathered (batch H8): the daily-streak test's first
    # claim goes through the runtime control (real clock) and
    # the second must be the consecutive real day; the rob test's
    # jail is read back through jail_state() and _control_work
    # with no now (real clock), so a frozen robbery time would
    # read as already expired.
    "tests/test_group_economy_system.py": 2,
    "tests/test_group_travel.py": 2,
    # Grandfathered (batch H9): the dashboard seed stages a flag
    # proposal whose proposed_ts must be near-real — the dashboard
    # reads flag_proposals() on the real clock and prunes anything
    # older than REPUTATION_PROPOSAL_TTL_S (3 days), so a frozen
    # stamp reads as expired and "flags 1" vanishes (proved:
    # freezing NOW drops the flag from the waiting line).
    "tests/test_groups_command.py": 1,
    "tests/test_health_sweep.py": 17,
    "tests/test_health_timeline.py": 3,
    # Grandfathered (batch H8): the blacklisted-deposit test's
    # bust goes through heist_go at the real clock because the
    # blacklist (48h from the bust) is enforced through the
    # runtime control (_control_deposit) on the real clock — a
    # frozen bust-now would read as expired before the deposit.
    "tests/test_heist_consequences.py": 1,
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
    # Grandfathered (batch H13): the trial-flow seed's `now` must
    # stay on the real clock — _load_assist_runs TTL-prunes
    # terminal runs with started < now - 7d (flow.py:915-955), and
    # the frozen MIDDAY instant (2026-10-14) is 32 days old, so a
    # frozen seed's stored row is deleted on reload and the raw
    # read-back finds nothing (proven by a round-1 freeze that
    # broke exactly that assertion on both seam rungs). The file's
    # other 8 reads are converted: monotonic mongod wait loops,
    # frozen monitor tick nows (last_ts starts at 0), frozen
    # failure-window starts (rows carry real-now ts, so the frozen
    # starts sit earlier and the same rows pass).
    "tests/test_mongo_p3_conformance.py": 1,
    # Grandfathered (batch H13): six reads whose seeds production
    # compares against the real clock, and the frozen MIDDAY instant
    # (2026-10-14) sits on the wrong side of the comparison. The
    # shopping _scenario seeds (2 reads) feed compute_steal_score's
    # 30-day and get_steals' 7-day windows; the timeline _noon
    # helper (1 read) feeds summary()/vitals_trend()'s 30-day
    # windows; the congestion queue_depth probe (1 read) reads a
    # hold whose expires_at acquire() stamped on the real clock;
    # the trip poll deadline (1 read) is judged by is_closed() and
    # polls_closing_soon() against real now; the campaign send_at
    # (1 read) is judged by process() against real now. In every
    # case a frozen seed changes the production outcome, and the
    # real-clock comparison is correct behavior, not a seam to add.
    # The file's other 11 reads are converted: monotonic wait
    # loops, frozen Gmail watch expiry, frozen challenge _noon
    # (all its production reads take an explicit now), frozen
    # _rates now (the window filter has no upper bound, so the
    # real-now-stamped corpus counts the same), and frozen past
    # seeds (closed poll, stale claim) that are past on both clocks.
    "tests/test_mongo_p5.py": 6,
    "tests/test_monitor_validation.py": 2,
    "tests/test_mood_patterns.py": 3,
    "tests/test_mqtt.py": 2,
    # Grandfathered (batch H29): _future_ts's datetime.now(LAGOS)
    # stays on the real clock — the one-shots it arms go through
    # Scheduler.add, which refuses any "at" instant <= time.time()
    # on the real clock (scheduler.py:990) with no now seam, so a
    # frozen date would arm in the past and the re-arm / pending-shot
    # assertions would fail (proved: freezing to MIDDAY arms
    # 2026-10-22, already past). Threading a now seam through
    # Scheduler.add would be a production API change in a hygiene
    # lane for a non-regression, and a fixed far-future anchor would
    # break the helper's own documented contract — a rolling future
    # date, never a fixed calendar date (the 2026-10-23 pins rotted
    # exactly that way). The file's other 6 reads are frozen: the
    # item-dict created_at (never read on the candidate path —
    # _store_candidates stores fetched_at=now, mark_served builds
    # its own row with created_at=now), the backstop test's `now`
    # (only feeds _pending_shot's next_run > now - 3600 check, and
    # the armed shot lands ~7 days ahead), and the wait-loop
    # deadlines (monotonic).
    "tests/test_motivation_m1.py": 1,
    "tests/test_orchestration_p9.py": 2,
    "tests/test_os_sweep.py": 9,
    "tests/test_os_timeline.py": 2,
    # Grandfathered (batch H21): 7 reads stay on the real clock —
    # production computes price freshness against the real clock
    # with no now seam on the driven paths, and these tests pin
    # exact ages / fresh verdicts that only hold relative to real
    # now. (1) The fresh quote (:218) pins freshness=="fresh",
    # quote_age_s < 5 and quote_ts ≈ now; quote() calls
    # _record_quote_result with no now (market_data.py:787), so a
    # frozen provider ts would read stale. (2) The fresh OHLCV
    # (:313) pins freshness=="fresh"; _attempt_chain compares the
    # newest bar against time.time() directly (:1869) and
    # get_ohlcv takes no now. (3) The stale quote (:328) pins
    # quote_age_s ≈ 3600 (abs=5) — the age itself is asserted.
    # (4–6) The shared _fresh_router payloads (:453/:492/:499)
    # feed the health loop AND the scheduler-job test, which
    # drives the real price_sources tool through sched.run_now
    # with the pinned payload {"action": "health"} — no now
    # reaches verify_price_sources there (its own now seam is
    # unreachable from the job path), so frozen payloads would
    # read stale and verified_live would be 0. (7) The persist
    # test (:540) asserts ok==1 after rebind; a frozen closeTime
    # would record "stale" instead. The file's other 4 reads are
    # frozen: the day-one chain test (only source + call count
    # asserted; stale still serves), the 30d-old bars (no age
    # pin; ~62d still past the 10800s bar-stale bound), the
    # gold-api stale spot (no age pin; ~32d still past the 300s
    # cadence), and the demoted-provider test (only source +
    # source_demoted asserted; the flag is health-driven).
    "tests/test_pass91_price_sources_open.py": 7,
    # Grandfathered (batch H12): the composed "+2h" test asserts
    # the parsed landing sits in a real-clock window — production
    # parses relative offsets against the real clock
    # (parse_schedule_args takes time.time() from the /sendlater
    # handler), so a frozen "before" would sit ~32 days before
    # the real landing and the window would never hold. The
    # file's other reads are converted: the _future helper
    # anchors on a fixed far-future instant (no upper bound in
    # the parser or send_scheduled), past seeds freeze to
    # MIDDAY_TS minus seconds, _op ids ride a counter and its
    # ts freezes, and the absolute-stamp test uses a literal
    # 2030 date.
    "tests/test_parity14_tg_lifecycle.py": 2,
    # Grandfathered (batch H10): the pulse-marker ts must be stamped
    # on the real clock — check_pulse_catchup suppresses the catch-up
    # only when last_ts >= today's real-clock due time, so a frozen
    # ts reads as "not delivered today" and the briefing fires anyway;
    # and the AutonomyTest last_active seed must sit 8h before the
    # real now — the brain compares it to the real clock and the
    # mood's silence scale saturates at 24h, so a frozen stamp reads
    # as a month of silence and changes the headspace the seed builds.
    # The file's other reads (wait loops, briefing generated_at) are
    # converted: monotonic deadlines, frozen generated_at.
    # Grandfathered (batch H25): the pulse-marker seed in
    # _make_runtime_context stays on the real clock — start() runs
    # check_all_catchup on boot (runtime.py:3259-3261) and
    # check_slot_catchup suppresses only when last_ts >= due on the
    # real clock, with pulse_enabled defaulting True, so a frozen
    # MIDDAY marker would fire the catch-up pulse into the adapter
    # and break the exact adapter.sent assertions. Same shape as
    # H10/H18/H23. The file's other 7 reads are frozen: wait-loop
    # deadlines (monotonic), briefing generated_at (presence by
    # date), message ts (no driven handler reads message.ts — the
    # only read is the stale_replay log line, gated on adapter-
    # stamped meta these directly-constructed messages lack).
    "tests/test_partner_deliberation_guard.py": 1,
    # Grandfathered (batch H23): the pulse-marker seed in
    # _make_runtime_context — start() runs check_all_catchup on
    # boot (runtime.py:3259) and check_slot_catchup suppresses
    # only when last_ts >= due on the real clock, with the pulse
    # enabled by default, so a frozen marker would fire the
    # catch-up pulse into the adapter. The file's other 8 reads
    # are frozen: bridge payload ts (stale_replay stamp only,
    # never asserted), wait-loop deadlines (monotonic), briefing
    # generated_at (presence by date), _group_msg ts (no driven
    # handler reads message.ts).
    "tests/test_partner_photo_receipt.py": 1,
    "tests/test_partner_runtime.py": 2,
    "tests/test_partner_social_f2.py": 3,
    "tests/test_partner_sweep.py": 3,
    "tests/test_persona.py": 2,
    "tests/test_phone_reliability.py": 10,
    "tests/test_planning_p9.py": 11,
    "tests/test_planning_sweep.py": 8,
    "tests/test_portfolio.py": 9,
    "tests/test_power_sweep.py": 7,
    # Grandfathered (batch H10): the pulse-marker ts must be stamped
    # on the real clock — check_pulse_catchup suppresses the catch-up
    # only when last_ts >= today's real-clock due time, so a frozen
    # ts reads as "not delivered today" and the briefing fires anyway.
    # The file's other reads (wait loops, briefing generated_at) are
    # converted: monotonic deadlines, frozen generated_at.
    "tests/test_presence.py": 1,
    # Grandfathered (batch H30): the pulse-marker ts must be stamped
    # on the real clock — check_slot_catchup suppresses a slot only
    # when its marker's ts >= today's slot time on the real clock
    # (morning_pulse.py:1899–1923), and start() runs check_all_catchup
    # (runtime.py:3259–3261), so a frozen MIDDAY marker fires the
    # catch-up on start() — proven: with the marker frozen,
    # test_dm_reaction fires "pulse/midnight: catch-up firing" and
    # spends ~10.6s on a real pulse delivery attempt. The seed's job
    # is to pin the pulse quiet; freezing it defeats that job and
    # makes the test's runtime depend on the real hour (which slots
    # fire). Same shape as H10/H18/H23/H25. The file's other 6 reads
    # are converted: monotonic wait-loop deadlines, frozen _msg /
    # inline message ts (the runtime's only message.ts read on the
    # driven path is the stale_replay log line, gated on
    # adapter-stamped meta these directly-constructed messages lack),
    # and frozen briefing generated_at (latest_briefing queries by
    # date; generated_at only orders within the date).
    "tests/test_presentation_ack.py": 1,
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
    "tests/test_swarm_specialists.py": 3,
    "tests/test_sync.py": 3,
    "tests/test_sync_sweep.py": 11,
    # Grandfathered (batch H9): the stale-session-token test seats
    # a console session whose "at" must be fresh — the console
    # re-checks time.time() - at > _CONSOLE_TTL_S (1800s) on the
    # real clock, so a frozen "at" reads as expired and the link
    # lands on the console home instead of the "don't admin"
    # refusal. The file's other read (NOW for joins/dates) is
    # frozen; only this one stays real.
    "tests/test_telegram_communities.py": 1,
    "tests/test_tools_sweep.py": 2,
    "tests/test_trading.py": 3,
    "tests/test_trading_risk.py": 3,
    "tests/test_training_backends.py": 2,
    "tests/test_travel_sweep.py": 1,
    "tests/test_trial_p9.py": 3,
    "tests/test_triggers.py": 5,
    "tests/test_triggers_sweep.py": 10,
    "tests/test_universal_storage_backup_offsite.py": 1,
    # Grandfathered (batch H33): 2 reads stay on the real clock.
    # They are the `started` duration anchor handed to
    # UpdateManager._rollback (os/update.py:2114), which records
    # report.seconds = time.time() - started (os/update.py:2175)
    # with no now seam — freezing started to MIDDAY_TS would record
    # a ~32-day duration in the report. The test's claim is the
    # quarantine guard (never quarantine the pre-update sha), not
    # timing; threading a now seam through _rollback/_run would be
    # a production API change in a hygiene lane for a non-regression
    # (H21 precedent). The file's other 4 reads are frozen to
    # MIDDAY_TS: the journaled "ts" seeds are carried data —
    # _quarantine_is_stale reads only the journal's sha fields,
    # _journal_runs groups by run_id in file order, and the only ts
    # consumer in the organ (_pending_restore_superseded_by)
    # compares ts only on "complete" events, while these seeds are
    # "step" / "rolled_back".
    "tests/test_updater_hardening.py": 2,
    "tests/test_vision_seer.py": 2,
    "tests/test_wave85_games.py": 1,
    "tests/test_wave_d_stream_e.py": 3,
    "tests/test_wave_e_gaps.py": 7,
    "tests/test_wave_f1_honesty.py": 1,
    # Grandfathered (batch H10): the FlapBridge's delivered ts must
    # be stamped on the real clock — the adapter's history reconcile
    # (_history_find_own) compares it against the real dispatch time
    # with a 5s floor, so a frozen ts reads as "sent too long ago"
    # and the flap tests' reconciliation never finds the message.
    # The file's other reads (wait loops) are converted to monotonic
    # deadlines.
    "tests/test_whatsapp.py": 1,
    # Grandfathered (batch H9): the economy-page awards must be
    # stamped on the real clock — economy_health/issuance_state
    # read the real clock over a 7-day window, so frozen awards
    # fall outside it and the throttled-policy test reads
    # throttled False (proved: freezing _fund's now flips the
    # assertion). Same family as H8's economy grandfathering.
    "tests/test_wp_economy_pages.py": 2,
    "tests/test_wisdom_ingestor.py": 1,
    "tests/test_workspace_sweep.py": 2,
    # Grandfathered (batch H17): 4 claim seeds stay on the real
    # clock — the INSTALLED feed reads it. make_claims_feed's _feed
    # calls claims_as_findings with no now (feed.py), so the
    # pipeline's weekly_content run stamps freshness against
    # time.time(); a frozen MIDDAY seed (~32 days old on the real
    # clock) reads stale and the claim never feeds. The feed
    # callable takes only the niche name, so there is no now seam
    # to thread a frozen instant through, and hand-rolling the feed
    # in the test would stop proving the production factory feeds.
    # The file's other 10 reads (explicit-now findings, niche
    # discovery/adoption seeds, artifact citations, outcome NOW)
    # are frozen.
    "tests/test_ws4p5_ws3_contracts.py": 4,
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
