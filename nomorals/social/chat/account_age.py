"""Account-age estimation from a Telegram ID (member-intel
dossier).

Telegram exposes NO creation-date field through the Bot API
or MTProto — there is no ``created_at`` on the ``User`` or
``UserFull`` TL objects, and no public lookup of registration
dates by ID.  The only honest route is ID-range interpolation:
Telegram assigns IDs roughly in registration order, so an ID
that falls between two reference points was probably
registered between those two months.

Reference set: the AgedSocials table (59 points, Nov 2013 –
Nov 2025), built from the MIT-licensed open-source tdage data
to Oct 2021 and CreationDate after it.  Rows marked adjusted
in the source carry the latest date forward so a higher ID
never maps to an earlier month.  Reused here with attribution;
the table is data, not code.

Rules the dossier obeys:
- an estimate is ALWAYS labelled an estimate, with its basis
  named — never a single exact date presented as fact;
- IDs above the last reference point (8,355,000,000, Nov 2025)
  are UNPLACEABLE, never guessed;
- IDs below the first point are "Nov 2013 or earlier";
- a low ID says nothing about restrictions, history, or who
  controls the account — and the dossier never implies it does.
"""
from __future__ import annotations

from typing import Any

# (user_id, (year, month)) — the earliest account observed
# with an ID at or above this value, per the reference set.
REFERENCE_POINTS: list[tuple[int, tuple[int, int]]] = [
    (2_768_409, (2013, 11)),
    (7_679_610, (2013, 12)),
    (11_538_514, (2014, 1)),
    (15_835_244, (2014, 2)),
    (23_646_077, (2014, 2)),
    (38_015_510, (2014, 3)),
    (44_634_663, (2014, 5)),
    (46_145_305, (2014, 5)),
    (54_845_238, (2014, 9)),
    (63_263_518, (2014, 10)),
    (101_260_938, (2015, 3)),
    (101_323_197, (2015, 3)),
    (103_151_531, (2015, 6)),
    (103_258_382, (2015, 6)),
    (109_393_468, (2015, 8)),
    (111_220_210, (2015, 8)),
    (112_594_714, (2015, 8)),
    (116_812_045, (2015, 8)),
    (122_600_695, (2015, 8)),
    (124_872_445, (2015, 8)),
    (125_828_524, (2015, 10)),
    (130_029_930, (2015, 10)),
    (133_909_606, (2015, 10)),
    (143_445_125, (2015, 12)),
    (148_670_295, (2016, 1)),
    (152_079_341, (2016, 1)),
    (157_242_073, (2016, 1)),
    (171_295_414, (2016, 3)),
    (181_783_990, (2016, 4)),
    (222_021_233, (2016, 6)),
    (225_034_354, (2016, 6)),
    (278_941_742, (2016, 9)),
    (285_253_072, (2016, 10)),
    (294_851_037, (2016, 11)),
    (297_621_225, (2016, 12)),
    (328_594_461, (2016, 12)),
    (337_808_429, (2017, 2)),
    (341_546_272, (2017, 2)),
    (352_940_995, (2017, 2)),
    (369_669_043, (2017, 3)),
    (400_169_472, (2017, 7)),
    (805_158_066, (2019, 7)),
    (1_974_255_900, (2021, 10)),
    (5_027_000_000, (2021, 12)),
    (5_170_000_000, (2022, 2)),
    (5_351_000_000, (2022, 5)),
    (5_469_000_000, (2022, 8)),
    (5_720_000_000, (2022, 11)),
    (5_870_000_000, (2023, 2)),
    (6_001_000_000, (2023, 5)),
    (6_523_000_000, (2023, 8)),
    (6_647_000_000, (2023, 11)),
    (6_718_000_000, (2024, 2)),
    (7_095_000_000, (2024, 5)),
    (7_358_000_000, (2024, 8)),
    (7_665_000_000, (2024, 11)),
    (7_908_000_000, (2025, 3)),
    (8_190_000_000, (2025, 8)),
    (8_355_000_000, (2025, 11)),
]

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
           "Aug", "Sep", "Oct", "Nov", "Dec")


def _month_label(month: tuple[int, int]) -> str:
    year, mon = month
    return f"{_MONTHS[mon - 1]} {year}"


def estimate_account_age(user_id: Any) -> dict[str, Any] | None:
    """The registration window for a Telegram ID, or None when
    the ID cannot be parsed at all.

    Returns ``{"low": (year, month) | None, "high": (year, month)
    | None, "unplaceable": bool}``: ``low`` is None when the ID
    predates the earliest reference point ("Nov 2013 or
    earlier"); ``high`` is None and ``unplaceable`` True when
    the ID is newer than the latest reference point.
    """
    try:
        uid = int(str(user_id).strip())
    except (TypeError, ValueError):
        return None
    if uid <= 0:
        return None
    points = REFERENCE_POINTS
    if uid < points[0][0]:
        return {"low": None, "high": points[0][1],
                "unplaceable": False}
    if uid > points[-1][0]:
        return {"low": points[-1][1], "high": None,
                "unplaceable": True}
    low = points[0][1]
    for prev_id, prev_month in points:
        if uid == prev_id:
            return {"low": prev_month, "high": prev_month,
                    "unplaceable": False}
        if uid < prev_id:
            return {"low": low, "high": prev_month,
                    "unplaceable": False}
        low = prev_month
    return {"low": low, "high": low, "unplaceable": False}


def render_account_age(user_id: Any) -> str:
    """One honest line for the dossier.  Always says estimate;
    never prints a made-up exact date as fact."""
    est = estimate_account_age(user_id)
    if est is None:
        return "account age unknown — no usable Telegram ID"
    if est["unplaceable"]:
        return ("account age unknown — Telegram ID is newer than "
                "the reference table (Nov 2025); estimate not "
                "possible")
    low = est["low"]
    high = est["high"]
    if low is None:
        return (f"probably registered {_month_label(high)} or "
                "earlier (estimate from Telegram ID range)")
    if low == high:
        return (f"probably registered around {_month_label(low)} "
                "(estimate from Telegram ID range)")
    return (f"probably registered between {_month_label(low)} and "
            f"{_month_label(high)} (estimate from Telegram ID "
            "range)")
