"""Tail-35 convert guard for nomorals/travel/whitelabel.py site 293.

Site 293 (client_spend_report) used to return "no spend data for <id>" when
the CostTracker could not even be constructed — a data-absence claim for a
client that may well HAVE spend. The genuine zero-spend report is
"Devon spent ₦0 on WhatsApp this week (0 messages).", never "no spend data",
so the failure string was not the documented empty value (tail-10/14 escape
does not apply). The convert returns an honest unavailability line instead.

This guard FAILS on the old code (old string contains "no spend data") and
PASSES on the new code. The file itself contains no exception handlers, so
it adds zero census sites.
"""

from pathlib import Path

from nomorals.travel import whitelabel as W


def test_293_spend_failure_is_unavailable_not_no_data(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    report = W.client_spend_report(
        "clientX", db_path=str(tmp_path / "blocker" / "cost.db"))
    assert "spend data unavailable" in report
    assert "clientX" in report
    assert "no spend data" not in report


def test_293_spend_healthy_zero_report_unchanged(tmp_path):
    report = W.client_spend_report(
        "clientX", db_path=str(tmp_path / "fresh-cost.db"))
    assert "Devon spent" in report and "₦0" in report
    assert report != "💸 spend data unavailable for clientX."
