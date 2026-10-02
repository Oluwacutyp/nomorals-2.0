"""Test taxonomy for the nomorals-2.0 suite.

The default suite must be fully offline and never hang. Every test file
belongs to exactly one base tier; ``slow`` and ``chaos`` are modifiers that
can stack on top of a base tier.

Base tiers
----------
``unit``
    Pure offline tests. Everything external is stubbed, mocked, or canned:
    no sockets, no subprocesses, no real network, no credentials, no live
    LLM. These always run.
``integration``
    Still offline and credential-free, but exercises real local machinery:
    localhost TCP servers/sockets (ephemeral ports), subprocesses (git,
    ffmpeg, ``sys.executable -c`` probes), threads with real sleeps. Heavier
    and slightly flakier than ``unit``, but safe in the default suite.
``live``
    Needs the real world: external network, credentials/API keys, a live
    LLM, or a real service session (e.g. the WhatsApp bridge with a linked
    phone). Never runs by default — gate with :data:`requires_integration`.

Modifiers
---------
``slow``
    Takes seconds-to-minutes by design: reconnect backoffs, retry loops,
    fault-injection supervisors, long sleeps.
``chaos``
    Adversarial by design: fault injection, synthetic crashes, Byzantine
    inputs. Expects failures and asserts the recovery path.

Integration gating
------------------
Live tests opt out of the default suite with::

    from tests.taxonomy import requires_integration

    @requires_integration
    class LiveBridgeTests(unittest.TestCase):
        ...

``requires_integration`` is ``unittest.skipUnless`` on
``NM_RUN_INTEGRATION=1``. Set the env var to run the live tier::

    NM_RUN_INTEGRATION=1 python -m unittest discover -s tests

File -> tier map (surveyed 2026-10-02; ``unit`` is the default, only
deviations are listed — see :data:`TIER_TABLE` and :func:`tier_of`)
--------------------------------------------------------------------
unit (explicitly verified offline, fully mocked/stubbed, no sockets):
    test_captcha.py, test_web_godtier.py
integration (localhost sockets on ephemeral ports — offline, no external
network; explicitly verified 2026-10-02):
    test_connectors_new.py (WebhookTests; Telegram adapter tests are pure
    unit), test_whatsapp.py (FakeBridge; the mocked QRDATA dispatch only
    prints a fake QR banner, it never connects anywhere)
integration (spawns subprocesses — git, ffmpeg, python -c probes):
    test_arena_ship.py, test_builders.py, test_builders_f2.py,
    test_coding_mission.py, test_edit_loop.py, test_inbox.py,
    test_market_data.py, test_media_edit.py, test_media_honesty.py,
    test_media_pipeline_g3.py, test_missions.py, test_phase_a.py,
    test_phase_b.py, test_phase_c.py, test_phase_d.py,
    test_self_improvement_v2.py, test_studio.py, test_tools.py,
    test_trading.py, test_watchers.py, test_wave69.py, test_wave71_decoder.py
chaos:
    test_fault_injection.py
slow (modifier, non-exhaustive):
    test_fault_injection.py, test_games_lifecycle.py,
    test_mission_control.py, test_whatsapp.py (ReconnectTests),
    test_wave_e_gaps.py
live (known live-LLM/network dependent; NOT gated yet — currently fail
without a live LLM, tracked separately):
    test_deep_research.py, test_phase_b.py, test_search_and_model.py,
    test_wave85_core.py, test_wave85_games.py
"""

from __future__ import annotations

import os
import unittest

__all__ = [
    "INTEGRATION_ENV_VAR",
    "integration_enabled",
    "unit",
    "integration",
    "live",
    "slow",
    "chaos",
    "requires_integration",
    "TIER_TABLE",
    "tier_of",
]

#: Env var that opts the suite into the ``live`` tier.
INTEGRATION_ENV_VAR = "NM_RUN_INTEGRATION"


def integration_enabled() -> bool:
    """True when the live/integration tier is explicitly opted in."""
    return os.environ.get(INTEGRATION_ENV_VAR) == "1"


def _tier_tag(tier: str):
    """Build a decorator that tags a test class/method with a tier.

    Works on classes and functions; safe under ``unittest`` discovery
    (it only sets an attribute). Tiers stack: applying ``@slow`` after
    ``@integration`` records both.
    """

    def decorator(obj):
        tiers = getattr(obj, "_nm_test_tier", ())
        if tier not in tiers:
            obj._nm_test_tier = (*tiers, tier)
        return obj

    decorator.__name__ = tier
    decorator.__doc__ = f"Tag a test as the {tier!r} tier (see tests.taxonomy)."
    return decorator


unit = _tier_tag("unit")
integration = _tier_tag("integration")
live = _tier_tag("live")
slow = _tier_tag("slow")
chaos = _tier_tag("chaos")

#: Skip decorator for live tests: they only run with NM_RUN_INTEGRATION=1.
requires_integration = unittest.skipUnless(
    integration_enabled(),
    f"live test — set {INTEGRATION_ENV_VAR}=1 to run",
)

#: Module name (without ``.py``) -> base tier. Anything not listed here is
#: ``unit``. Modifiers (``slow``/``chaos``) are documented in the table
#: above, not in this map.
TIER_TABLE: dict[str, str] = {
    # explicitly verified offline 2026-10-02
    "test_captcha": "unit",  # urlopen stubbed; unpatched path raises pre-HTTP
    "test_web_godtier": "unit",  # HttpClient mocked in setUp
    # localhost sockets on ephemeral ports — offline, no external network
    "test_connectors_new": "integration",  # WebhookTests; rest is pure unit
    "test_whatsapp": "integration",  # FakeBridge; mocked QR dispatch is fake
    # localhost sockets / subprocesses — offline, heavier
    "test_arena_ship": "integration",
    "test_builders": "integration",
    "test_builders_f2": "integration",
    "test_coding_mission": "integration",
    "test_edit_loop": "integration",
    "test_inbox": "integration",
    "test_market_data": "integration",
    "test_media_edit": "integration",
    "test_media_honesty": "integration",
    "test_media_pipeline_g3": "integration",
    "test_missions": "integration",
    "test_phase_a": "integration",
    "test_phase_b": "integration",
    "test_phase_c": "integration",
    "test_phase_d": "integration",
    "test_self_improvement_v2": "integration",
    "test_studio": "integration",
    "test_tools": "integration",
    "test_trading": "integration",
    "test_watchers": "integration",
    "test_wave69": "integration",
    "test_wave71_decoder": "integration",
    # adversarial by design
    "test_fault_injection": "chaos",
    # live tier: known live-LLM/network dependent (not gated yet)
    "test_deep_research": "live",
    "test_search_and_model": "live",
    "test_wave85_core": "live",
    "test_wave85_games": "live",
}


def tier_of(module_name: str) -> str:
    """Return the base tier for a test module (``unit`` when unlisted).

    Accepts ``"test_foo"`` or ``"test_foo.py"`` (or a dotted path — the
    last component is used).
    """
    name = module_name
    if name.endswith(".py"):
        name = name[:-3]
    base = name.rsplit(".", 1)[-1]
    return TIER_TABLE.get(base, "unit")
