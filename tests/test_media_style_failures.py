"""Tail-76 pins — nomorals/media/style.py silent-failure sites.

Module contract (its own docstring): "``Theme.render_*`` helpers
never raise — worst case they return the plain-text fallback."
Every pin asserts the CURRENT behaviour at one census site:
the guarded render degrades to its documented fallback and
never raises.  No site records anything — the module has no
logger and no failure store; the fallback IS the surface.
"""
from nomorals.media.style import THEMES

TH = THEMES["ninja"]


def test_banner_failure_returns_plain_title():
    # site: Theme.banner except -> str(title)
    assert "╔" in TH.banner("Hello")  # control: boxed success
    assert TH.banner("Hello", width="bad") == "Hello"


def test_section_failure_returns_plain_title():
    # site: Theme.section except -> str(title)
    assert TH.section("Mix").startswith("\n── Mix ")
    assert TH.section(123) == "123"


def test_kv_failure_returns_empty():
    # site: Theme.kv except -> ""
    assert "a" in TH.kv([("a", 1)])  # control
    # a 3-tuple pair cannot unpack -> guarded failure -> ""
    assert TH.kv([("a", 1, 2)]) == ""


def test_bar_failure_returns_empty():
    # site: Theme.bar except -> ""
    # control: a successful 0.0 bar is a full-width empty-glyph
    # bar, NOT "" — the failure value is distinguishable.
    assert TH.bar(0.0) == "░" * 20
    assert TH.bar("not-a-float") == ""


def test_sparkbar_failure_returns_empty():
    # site: Theme.sparkbar except -> ""
    assert TH.sparkbar([0.5, 1.0]) != ""  # control
    assert TH.sparkbar(["x"]) == ""


class _FlakyStops:
    """First iteration raises, second yields — drives the try
    into its fallback with content still available."""

    def __init__(self):
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return iter(["8A", "9A"])


def test_journey_map_failure_returns_plain_join():
    # site: Theme.journey_map except -> " → ".join(...)
    assert TH.journey_map(["8A", "9A"]) == "  8A → 9A"  # control
    # fallback: same content, no theme indent
    assert TH.journey_map(_FlakyStops()) == "8A → 9A"
