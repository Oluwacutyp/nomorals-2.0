"""Tail-106 pins: nomorals/pulses/timezone.py census sites
(116, 180, 234, 237).

All four sites are held as EXCLUSIONS. The module computes *when*
pulses fire; its printed contracts are "Never raises" and
degradation-with-disclosure:

116 — ``_try_zoneinfo`` is a probe: the ZoneInfo construction IS
the check, and None is its honest negative. The degradation is
disclosed in-band: ``resolve_owner_tz`` reports the fallback in
its returned source string (``...+fixed-offset``) together with a
``FixedOffset`` tzinfo, so no caller can mistake the approximation
for a resolved IANA zone. Pinned both ways.

180 — ``_coerce_date``: an unparseable date string falls through
to ``datetime.now(tz).date()`` — byte-identical to the module's
own legitimate None path ("on_date defaults to today in tz" is
``fire_at_local``'s documented primary contract). Degradation to
the module's own default, not an invented date. FLAGGED-ACCEPTED:
the except carries no noqa, so this site stays in the flagged
scan as an accepted remainder.

234 — ``fire_at_local``'s outer never-raises guard: on garbage
input it returns midnight UTC of the requested date, exactly the
fallback its docstring prints ("Never raises — on garbage input
returns midnight UTC of the requested date"). Every type coercion
sits inside the guard, so the reachable triggers are garbage
inputs; the DST gap/fold math before the guard is total for real
tzinfo objects (round-trip, fold=1, minute-walk). Pinned with a
garbage hour plus working controls (Lagos offset, Denver MDT,
spring-forward gap walk).

237 — the inner guard inside 234's fallback: if ``_coerce_date``
itself raises while computing the fallback day, the day becomes
today UTC and the function still returns midnight UTC — it never
raises. Reachability note: ``_coerce_date`` returns/raises nothing
for any input class it accepts (date -> itself, str -> internally
guarded, anything else -> now()), so this rung exists only
against a pathological datetime failure. Pinned by forcing
``_coerce_date`` to raise and proving the double failure still
lands on the documented midnight-UTC shape.

The module has no logger and no failure ledger: these guards
record nothing, and the pins assert pure return behavior.
"""
from __future__ import annotations

from datetime import date, datetime, time as _time, timezone
from unittest import mock

from nomorals.pulses import timezone as tzmod


# ── 116: zoneinfo probe — None is the honest negative ────────────

def test_116_unresolvable_zone_returns_none():
    assert tzmod._try_zoneinfo("Bogus/Zone") is None


def test_116_resolvable_zone_returns_tzinfo():
    assert tzmod._try_zoneinfo("Africa/Lagos") is not None


def test_116_fallback_is_disclosed_in_source_string():
    name, tz, source = tzmod.resolve_owner_tz(
        {"owner_timezone": "Bogus/Zone"})
    assert name == "Bogus/Zone"
    assert isinstance(tz, tzmod.FixedOffset)
    assert source.endswith("+fixed-offset")  # degradation disclosed


def test_116_resolved_zone_reports_no_fallback():
    _name, tz, source = tzmod.resolve_owner_tz(
        {"owner_timezone": "Africa/Lagos"})
    assert not isinstance(tz, tzmod.FixedOffset)
    assert "+fixed-offset" not in source


# ── 180: garbage date string -> today (the None-path default) ────

def test_180_garbage_date_string_becomes_today():
    today = datetime.now(timezone.utc).date()
    assert tzmod._coerce_date("not-a-date", timezone.utc) == today
    assert tzmod._coerce_date("2026-13-45", timezone.utc) == today


def test_180_none_takes_the_same_legitimate_default():
    today = datetime.now(timezone.utc).date()
    assert tzmod._coerce_date(None, timezone.utc) == today


def test_180_valid_inputs_parse():
    assert tzmod._coerce_date("2026-11-02", timezone.utc) == date(2026, 11, 2)
    assert tzmod._coerce_date(date(2026, 1, 30), timezone.utc) == date(2026, 1, 30)


# ── 234: garbage input -> midnight UTC of the requested date ─────

def test_234_garbage_hour_returns_midnight_utc_of_requested_date():
    got = tzmod.fire_at_local("x", "Africa/Lagos", on_date=date(2026, 11, 2))
    assert got == datetime.combine(date(2026, 11, 2), _time(0, 0),
                                   tzinfo=timezone.utc)


def test_234_control_lagos_wall_time():
    got = tzmod.fire_at_local(6, "Africa/Lagos", on_date=date(2026, 11, 2))
    assert got == datetime(2026, 11, 2, 5, 0, tzinfo=timezone.utc)  # WAT = UTC+1


def test_234_control_denver_mdt_wall_time():
    got = tzmod.fire_at_local(21, "America/Denver", on_date=date(2026, 10, 20))
    assert got == datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)  # MDT = UTC-6


def test_234_control_spring_forward_gap_walks_to_first_real_instant():
    # 2026-03-08 02:30 does not exist in Denver (02:00 MST -> 03:00 MDT)
    got = tzmod.fire_at_local(2, "America/Denver", minute=30,
                              on_date=date(2026, 3, 8))
    assert got == datetime(2026, 3, 8, 9, 0, tzinfo=timezone.utc)  # 03:00 MDT


# ── 237: double failure still never raises ───────────────────────

def test_237_coerce_failure_inside_fallback_returns_midnight_utc_today():
    def _boom(*args, **kwargs):
        raise RuntimeError("coerce boom")

    today = datetime.now(timezone.utc).date()
    with mock.patch.object(tzmod, "_coerce_date", _boom):
        got = tzmod.fire_at_local(6, "Africa/Lagos", on_date=date(2026, 11, 2))
    assert got == datetime.combine(today, _time(0, 0), tzinfo=timezone.utc)
