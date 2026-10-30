"""Tail-61 pins for nomorals/social/render.py —
0 converts / 7 exclusions (151, 210, 237, 266, 296, 354, 374).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the double's own get() call log. The
file contains no exception handlers, so it adds zero census
sites.

render.py calls no failures-ledger record function anywhere
(grep-proven), so NOT-recording is pinned against the social
package ledger (nomorals/social/failures.py): recent_failures()
snapshot before/after every trigger (tail-50/51 precedent).

Verdict notes (full grounds in the tail-61 SURVEY.md): the
module's documented contract is "Never raises — worst case
returns a plain-text fallback"; every renderer is a pure
function over a caller-supplied dict, so each fallback is a
value derived from — and literally true of — the input itself.
- 151: card-build failure -> the input's own subject string.
- 210: roster failure -> len(members), literally true.
- 237: subgroup-build failure -> the bare group card (itself
  guarded at 151); the subgroups section is dropped, never
  asserted absent.
- 266: channel failure -> the input's own name.
- 296: list failure -> len(groups), literally true; the
  happy-path empty renders a DIFFERENT string ("no groups
  found"), so failure never masquerades as absence.
- 354: metrics failure -> field lines re-derived from the
  same dict; "no metrics" only when no readable numeric field
  exists (weakest in file on the absence-flavored string).
- 374: float() failure -> str(val), the input itself.
"""

from nomorals.social import failures as soc_fail
from nomorals.social import render as R


def _snap():
    return list(soc_fail.recent_failures(200))


class GetLog(dict):
    """dict whose get() logs keys and can be poisoned per key."""

    def __init__(self, *a, poison=(), **kw):
        super().__init__(*a, **kw)
        self.log = []
        self.poison = set(poison)

    def get(self, key, default=None):
        self.log.append(key)
        if key in self.poison:
            raise RuntimeError(f"get {key} boom")
        return super().get(key, default)


def test_151_group_card_failure_yields_subject():
    before = _snap()
    info = GetLog({"subject": "Chess Club", "size": "not-a-number",
                   "admins": []})
    out = R.render_group_card(info)
    assert "size" in info.log  # the card build was actually attempted
    assert out == "Chess Club"
    good = R.render_group_card({"subject": "Chess Club", "size": 5,
                                "admins": []})
    assert "5 members" in good  # control
    assert _snap() == before


def test_210_members_failure_yields_true_count():
    before = _snap()
    bad = GetLog({"name": "Ada", "role": "admin"}, poison=("role",))
    out = R.render_members([bad, {"name": "Bob", "role": ""}])
    assert "role" in bad.log  # the role read was actually attempted
    assert out == "2 members"
    good = R.render_members([{"name": "Ada", "role": "admin"},
                             {"name": "Bob", "role": ""}])
    assert "Members (2)" in good and "Ada" in good  # control
    assert _snap() == before


def test_237_community_failure_yields_bare_card():
    before = _snap()
    info = {"subject": "Chess Club", "size": 5, "admins": []}
    bad_sub = GetLog({"subject": "Sub A", "size": 3}, poison=("subject",))
    out = R.render_community(info, [bad_sub])
    assert "subject" in bad_sub.log  # the subgroup read was attempted
    assert out == R.render_group_card(info)
    assert "Subgroups" not in out  # dropped, never asserted absent
    good = R.render_community(info, [{"subject": "Sub A", "size": 3}])
    assert "Subgroups (1)" in good  # control
    assert _snap() == before


def test_266_channel_failure_yields_name():
    before = _snap()
    info = GetLog({"name": "News Chan", "followers": 10},
                  poison=("description",))
    out = R.render_channel(info)
    assert "description" in info.log  # the desc read was attempted
    assert out == "News Chan"
    good = R.render_channel({"name": "News Chan", "followers": 10})
    assert "10 followers" in good  # control
    assert _snap() == before


def test_296_group_list_failure_yields_true_count():
    before = _snap()
    bad = GetLog({"subject": "G1", "size": 3}, poison=("subject",))
    out = R.render_group_list([bad])
    assert "subject" in bad.log  # the row build was attempted
    assert out == "1 groups"
    good = R.render_group_list([{"subject": "G1", "size": 3}])
    assert "Groups (1)" in good and "G1" in good  # control
    assert _snap() == before


def test_354_metrics_failure_rederives_fields():
    before = _snap()
    metrics = GetLog({"likes": 5, "quotes": 9}, poison=("quotes",))
    out = R.render_metrics_card(metrics)
    assert "quotes" in metrics.log  # the main build attempted the read
    assert "• likes: 5" in out  # fallback re-derives the real fields
    only = GetLog({"quotes": 9}, poison=("quotes",))
    # the one value sits behind a raising get: no READABLE metric
    assert R.render_metrics_card(only) == \
        "📊 Post performance: no metrics"
    good = R.render_metrics_card({"likes": 5, "quotes": 9})
    assert "total engagement" in good and "Quotes" in good  # control
    assert _snap() == before


class _BadFloat:
    def __init__(self):
        self.calls = 0

    def __float__(self):
        self.calls += 1
        raise TypeError("float boom")

    def __str__(self):
        return "BADFLOAT"


def test_374_fmt_num_failure_yields_str_of_value():
    before = _snap()
    bf = _BadFloat()
    out = R._fmt_num(bf)
    assert bf.calls == 1  # the conversion was actually attempted
    assert out == "BADFLOAT"
    assert R._fmt_num(2_500_000) == "2.5M"  # control
    assert _snap() == before
