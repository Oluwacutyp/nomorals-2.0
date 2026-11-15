"""Account-age estimation (member-intel dossier): a Telegram
account's registration era from its ID range.

Telegram exposes NO creation-date field through the Bot API or
MTProto — the only honest route is ID-range interpolation
against public reference sets (AgedSocials / tdage /
CreationDate, MIT).  The estimate is ALWAYS labelled an
estimate: a window between two reference months, never a
single exact date, and IDs newer than the table are
unplaceable, never guessed.
"""
from __future__ import annotations

from nomorals.social.chat.account_age import (
    REFERENCE_POINTS,
    estimate_account_age,
    render_account_age,
)


def test_reference_table_is_monotonic():
    ids = [point[0] for point in REFERENCE_POINTS]
    assert ids == sorted(ids)
    months = [point[1] for point in REFERENCE_POINTS]
    assert months == sorted(months)
    assert len(REFERENCE_POINTS) >= 50


def test_window_between_two_points():
    # 50,000,000 sits between May 2014 (46,145,305) and
    # Sep 2014 (54,845,238).
    est = estimate_account_age(50_000_000)
    assert est is not None
    assert est["low"] == (2014, 5)
    assert est["high"] == (2014, 9)
    assert est["unplaceable"] is False


def test_exact_reference_point_is_its_own_month():
    est = estimate_account_age(2_768_409)
    assert est is not None
    assert est["low"] == (2013, 11)
    assert est["high"] == (2013, 11)


def test_below_first_point_is_open_ended_early():
    est = estimate_account_age(1_000_000)
    assert est is not None
    assert est["low"] is None
    assert est["high"] == (2013, 11)
    assert est["unplaceable"] is False


def test_above_last_point_is_unplaceable():
    est = estimate_account_age(9_000_000_000)
    assert est is not None
    assert est["unplaceable"] is True
    assert est["high"] is None


def test_string_ids_parse():
    est = estimate_account_age("50000000")
    assert est is not None
    assert est["low"] == (2014, 5)


def test_unparseable_id_is_none():
    assert estimate_account_age("not-an-id") is None
    assert estimate_account_age(None) is None
    assert estimate_account_age(0) is None
    assert estimate_account_age(-5) is None


def test_windows_never_go_backwards():
    prev_high = None
    for probe in (3_000_000, 50_000_000, 200_000_000,
                  1_000_000_000, 6_000_000_000, 8_000_000_000):
        est = estimate_account_age(probe)
        assert est is not None and not est["unplaceable"]
        if prev_high is not None:
            assert est["low"] is None or est["low"] >= prev_high
        prev_high = est["high"]


def test_render_always_labels_the_estimate():
    rendered = render_account_age(50_000_000)
    assert "estimate" in rendered.lower()
    assert "May 2014" in rendered
    assert "Sep 2014" in rendered
    # never a bare exact date presented as fact
    assert "registered on" not in rendered.lower()


def test_render_unplaceable_says_so():
    rendered = render_account_age(9_000_000_000)
    assert "unknown" in rendered.lower()
    assert "estimate" in rendered.lower() or "newer" in rendered.lower()


def test_render_unparseable_says_unknown():
    rendered = render_account_age("nope")
    assert "unknown" in rendered.lower()
