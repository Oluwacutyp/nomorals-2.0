"""GROUPS Tier 0 tests: pulse timezone anchor, runtime envelope,
formatting conformance, profile-visible line.

Invariants under test:
* pulses fire at the intended *local* time across DST (Denver moves,
  Lagos does not);
* a rendered message never leaks another platform's markup;
* chunking never splits mid-markup or mid-emoji;
* runtime workers respect the Termux envelope (never cpu//2).
"""

from __future__ import annotations

import re
from datetime import timedelta

import pytest

from nomorals.agents.runtime_profile import (
    RuntimeProfileManager,
    process_workers,
    thread_workers,
    workers_for,
)
from nomorals.core.profile import EnvironmentProfile
from nomorals.core.profile_visible import (
    profile_summary,
    profile_visible_line,
)
from nomorals.pulses.timezone import (
    PulseAnchor,
    confirm_routine,
    drift_report,
    fire_at_local,
)
from nomorals.social.format_conformance import (
    check_message,
    chunk,
    convert,
)


def _denver():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/Denver")


def _lagos():
    from zoneinfo import ZoneInfo

    return ZoneInfo("Africa/Lagos")


def _termux_profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        kind="termux", min_vcpus=1, target_vcpus=2, max_vcpus=3,
        cpu=8, memory_mb=4096, detail="test termux")


def _pc_profile() -> EnvironmentProfile:
    return EnvironmentProfile(
        kind="pc", min_vcpus=2, target_vcpus=4, max_vcpus=8,
        cpu=8, memory_mb=16384, detail="test pc")


# ── pulse timezone ───────────────────────────────────────────────────

def test_fire_at_local_denver_dst_transition():
    # 23:00 Denver wall time stays 23:00 across the fall-back, while
    # the UTC instant moves by one hour (MDT UTC-6 -> MST UTC-7).
    before = fire_at_local(23, "America/Denver", on_date="2026-10-31")
    after = fire_at_local(23, "America/Denver", on_date="2026-11-02")
    assert before.astimezone(_denver()).hour == 23
    assert after.astimezone(_denver()).hour == 23
    assert before.astimezone(_denver()).utcoffset() == timedelta(hours=-6)
    assert after.astimezone(_denver()).utcoffset() == timedelta(hours=-7)
    assert (after - before).total_seconds() == 2 * 86400 + 3600


def test_fire_at_local_lagos_no_dst():
    # Lagos (WAT, UTC+1, no DST) fires at the same UTC instant offset
    # on both sides of Denver's transition.
    before = fire_at_local(6, "Africa/Lagos", on_date="2026-10-31")
    after = fire_at_local(6, "Africa/Lagos", on_date="2026-11-02")
    assert before.astimezone(_lagos()).utcoffset() == timedelta(hours=1)
    assert after.astimezone(_lagos()).utcoffset() == timedelta(hours=1)
    assert (after - before).total_seconds() == 2 * 86400


def test_fire_at_local_spring_gap_shifts_forward():
    # 2026-03-08 02:30 does not exist in Denver (spring forward).
    dt = fire_at_local(2, "America/Denver", minute=30,
                       on_date="2026-03-08")
    local = dt.astimezone(_denver())
    assert (local.hour, local.minute) == (3, 0)
    # Round-trips: a real instant, not a phantom wall time.
    assert local.utcoffset() == timedelta(hours=-6)


def test_fire_at_local_fall_fold_first_occurrence():
    # 2026-11-01 01:30 happens twice in Denver; take the first (MDT).
    dt = fire_at_local(1, "America/Denver", minute=30,
                       on_date="2026-11-01")
    assert dt.astimezone(_denver()).utcoffset() == timedelta(hours=-6)


def test_confirm_routine_flags_evening_0400():
    anchor = PulseAnchor("Africa/Lagos")
    routine = confirm_routine(anchor, on_date="2026-10-20")
    legacy = routine["legacy_denver_anchored"]
    assert legacy["evening"]["owner_hour"] == 4
    assert legacy["morning"]["owner_hour"] == 6
    assert routine["needs_confirmation"] is True
    evening_flags = [f for f in routine["flags"]
                     if f["slot"] == "evening"]
    assert len(evening_flags) == 1
    assert evening_flags[0]["severity"] == "needs_confirmation"
    assert evening_flags[0]["status"] == "unconfirmed"
    assert "04:00" in evening_flags[0]["reason"]


def test_drift_report_evening_shifts_across_dst():
    report = drift_report(PulseAnchor("Africa/Lagos"))
    evening = report["slots"]["evening"]
    assert evening["before"].endswith("04:00 MDT") or "04:00" in evening["before"]
    # Denver falls back MDT->MST, so the same 21:00 Denver wall time
    # fires 1h later in UTC -> 05:00 Lagos (was 04:00).
    assert "05:00" in evening["after"]
    assert evening["shift_minutes"] == 60


def test_confirm_routine_proposed_schedule_sane():
    routine = confirm_routine(PulseAnchor("Africa/Lagos"),
                              on_date="2026-10-20")
    proposed = routine["proposed_owner_anchored"]
    assert proposed["evening"]["owner_hour"] == 19
    assert proposed["morning"]["owner_hour"] == 6
    assert proposed["afternoon"]["owner_hour"] == 13
    assert all(row["owner_hour"] >= 5 for row in proposed.values())


# ── formatting conformance ───────────────────────────────────────────

def test_check_message_html_into_whatsapp():
    violations = check_message("whatsapp", "Hello <b>world</b>")
    assert any(v.startswith("html-in-whatsapp") for v in violations)


def test_check_message_markdown_into_telegram():
    violations = check_message("telegram", "Hello **world**")
    assert any(v.startswith("markdown-in-telegram") for v in violations)
    violations = check_message("telegram", "see [docs](https://x.test)")
    assert any(v.startswith("markdown-in-telegram") for v in violations)


def test_check_message_clean_passes():
    assert check_message("telegram", "<b>hi</b> and <code>x</code>") == []
    assert check_message("whatsapp", "*hi* and _there_") == []
    assert check_message("discord", "**hi** and `code`") == []
    assert check_message("sms", "plain text only") == []


def test_check_message_unclosed_tag():
    violations = check_message("telegram", "<b>oops, never closed")
    assert any("unclosed-html-tag" in v for v in violations)
    violations = check_message("telegram", "<div>nope</div>")
    assert any("unsupported-html-tag" in v for v in violations)


def test_chunk_never_splits_mid_markup():
    parts = []
    for i in range(30):
        parts.append(f"Section {i} <b>bold claim number {i}</b> "
                     f"with trailing words 🎉 and more text. ")
    text = "".join(parts)
    chunks = chunk(text, "telegram", max_len=120)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    for c in chunks:
        # No chunk cuts inside a tag: every '<' has its '>'.
        assert c.count("<") == c.count(">")
        # Balanced <b> per chunk.
        assert c.count("<b>") == c.count("</b>")
        # No chunk-level markup violations.
        assert not [v for v in check_message("telegram", c)
                    if "unclosed" in v or "misnested" in v
                    or "stray" in v]
    # Emoji survive whole.
    assert "".join(chunks).count("🎉") == text.count("🎉")


def test_chunk_never_splits_mid_emoji():
    family = "\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466"
    text = " ".join([f"word{i} {family} tail{i}" for i in range(40)])
    chunks = chunk(text, "telegram", max_len=80)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert "".join(chunks).count(family) == text.count(family)
    for c in chunks:
        assert not c.endswith("\u200d")  # never cut after a joiner
        assert not c.startswith("\ufe0f")  # never cut before a VS16
        assert not c.endswith("\ufe0f")


def test_chunk_whatsapp_markdown_balanced():
    parts = []
    for i in range(30):
        parts.append(f"Item {i} *bold point {i}* plus filler words. ")
    text = "".join(parts)
    chunks = chunk(text, "whatsapp", max_len=120)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    for c in chunks:
        assert not [v for v in check_message("whatsapp", c)
                    if "unbalanced" in v]


def test_convert_html_to_whatsapp():
    converted = convert("<b>hi</b> there", "whatsapp")
    assert converted == "*hi* there"
    assert check_message("whatsapp", converted) == []
    converted = convert("*hi* there", "telegram")
    assert converted == "<b>hi</b> there"
    assert check_message("telegram", converted) == []


# ── runtime envelope ─────────────────────────────────────────────────

def test_runtime_workers_respect_termux_envelope():
    prof = _termux_profile()
    assert workers_for(prof, shed="none") == 2  # envelope target, not 8//2
    assert 1 <= workers_for(prof, shed="none") <= 3
    assert workers_for(prof, shed="heavy") == 1  # heavy shed -> min
    assert workers_for(prof, shed="medium") <= 2
    assert process_workers(prof) == 0  # fork unreliable on Termux
    assert thread_workers(prof) <= 4  # Termux platform cap
    assert workers_for(prof, shed="none", load=0.0) == 1
    assert workers_for(prof, shed="none", load=1.0) == 3


def test_runtime_re_detect_reports_change():
    mgr = RuntimeProfileManager(profile=_pc_profile())
    res = mgr.re_detect(profile=_termux_profile())
    assert res["changed"] is True
    assert res["changes"]["kind"] == {"old": "pc", "new": "termux"}
    assert mgr.profile.kind == "termux"
    again = mgr.re_detect(profile=_termux_profile())
    assert again["changed"] is False


def test_profile_visible_line():
    summary = profile_summary(profile=_termux_profile(), shed="heavy")
    assert summary["kind"] == "termux"
    assert summary["vcpu"]["min"] == 1
    assert summary["vcpu"]["target"] == 2
    assert summary["vcpu"]["max"] == 3
    assert summary["workers"] == 1  # heavy shed -> envelope min
    assert summary["power_shed"] == "heavy"
    assert summary["free_mb"] >= 0
    assert "line" in summary
    line = summary["line"]
    assert line.startswith("termux 1 vcpu ")
    assert line.endswith("heavy shed")
    assert re.search(r"\d+\.\dGB free", line)
    # No-arg path works on the real machine.
    assert isinstance(profile_visible_line(), str)
    assert "vcpu" in profile_visible_line()
    assert "shed" in profile_visible_line()
