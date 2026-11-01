"""Tail-82 failure pins — nomorals/accounts/temp_sms.py.

All five sites are EXCLUSIONS, probed live first
(nm-tail82-verify/probes.py, 22/22 behavior confirmations
across both batch modules). SevenSimProvider is documented
BEST-EFFORT / probe-gated: every parse is tolerant and no
failure ever reaches a caller as an exception.

- 278: probe() returns False when the provider is unusable —
  the base-class contract ("a failed probe means 'skip me'
  in the cascade"); grab_number_cascade records the skip in
  its notes and moves to the next provider. The guard is
  belt-and-braces over list_numbers' own internal guards,
  and the failure is debug-logged.
- 297: list_numbers returns [] when the homepage fetch
  fails — [] is the no-numbers value every consumer handles
  honestly (probe -> False; grab_number -> {"status":
  "failed"}; cascade -> "no numbers listed" note + next
  provider).
- 305: a failing listing page is skipped (continue) — the
  page ladder keeps the homepage's own numbers.
- 322: the per-row link extraction tolerates IndexError and
  leaves link "" — the number row is still appended. With
  the shipped patterns group(1) always exists (probe-proven
  over both patterns), so the guard is defensive; "" is the
  no-link value, and get_messages honestly returns [] for a
  non-http inbox_id.
- 344: get_messages returns [] when the inbox fetch fails —
  [] is the empty-inbox value, the normal state while
  wait_for_code polls; the poll keeps waiting and times out
  honestly with "". No success is ever claimed from it.
"""
from unittest import mock

from nomorals.accounts import temp_sms as ts


def test_278_probe_false_when_provider_unusable():
    prov = ts.SevenSimProvider()
    with mock.patch.object(ts, "_fetch", side_effect=RuntimeError("down")):
        assert prov.probe() is False
    with mock.patch.object(ts.SevenSimProvider, "list_numbers",
                           side_effect=RuntimeError("boom")):
        assert prov.probe() is False
    listing = '<a href="/number/1">+1 530 403 1584</a>'
    with mock.patch.object(ts, "_fetch", return_value=listing):
        assert prov.probe() is True


def test_297_list_numbers_empty_when_home_fetch_fails():
    prov = ts.SevenSimProvider()
    with mock.patch.object(ts, "_fetch", side_effect=RuntimeError("down")):
        assert prov.list_numbers() == []


def test_305_listing_page_failure_keeps_home_numbers():
    prov = ts.SevenSimProvider()
    home = ('<a href="/numbers/us">numbers</a> '
            '<a href="/number/9">+1 202 555 0134</a>')

    def flaky(url, timeout=20):
        if url == ts.SevenSimProvider._BASE + "/":
            return home
        raise RuntimeError("page down")

    with mock.patch.object(ts, "_fetch", side_effect=flaky):
        numbers = prov.list_numbers()
    assert numbers
    assert numbers[0].number == "+12025550134"
    assert numbers[0].inbox_id == "https://7sim.net/number/9"


def test_322_linkless_number_row_still_listed():
    prov = ts.SevenSimProvider()
    page = "<p>Call me</p> +442071234567</a>"
    with mock.patch.object(ts, "_fetch", return_value=page):
        numbers = prov.list_numbers()
    assert len(numbers) == 1
    assert numbers[0].number == "+442071234567"
    # the row's inbox_id is unusable (never http) and get_messages
    # honestly yields [] for it — the row degrades, never crashes
    assert not numbers[0].inbox_id.startswith("http")
    assert prov.get_messages(numbers[0]) == []


def test_344_get_messages_empty_when_inbox_fetch_fails():
    prov = ts.SevenSimProvider()
    num = ts.TempNumber(number="+12025550134",
                        inbox_id="https://7sim.net/number/9")
    with mock.patch.object(ts, "_fetch", side_effect=RuntimeError("down")):
        assert prov.get_messages(num) == []
    inbox = "From: 123456 Your code is 482913 now 5 minutes ago"
    with mock.patch.object(ts, "_fetch", return_value=inbox):
        msgs = prov.get_messages(num)
    assert len(msgs) == 1
    assert msgs[0].code == "482913"
