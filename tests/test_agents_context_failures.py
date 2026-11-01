"""Tail-83 pins for nomorals/agents/context.py — 5 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving
the except path was taken (trigger == fallback AND control !=
fallback). The file itself contains no exception handlers, so it adds
zero census sites.

context.py calls no failures-ledger record function anywhere
(grep-proven), so NOT-recording is pinned with the snapshot method:
agency recent_failures() before/after each trigger must be unchanged
(snapshot stability across full build_router/build_context runs was
verified live before pinning).

Verdict notes (full grounds in SURVEY.md):
- 87: _progression_grants is additive-only and documented
  "Never raises"; failure yields set() — FEWER capabilities, never
  more (fail-closed). Policy double-guards the same callable
  (core/policy.py, debug-logged denial). The base grant from role
  config is never touched by this function.
- 312: side-chat wiring is an optional subsystem's boot rung ("side
  chats are optional"); boot completes without it and the side_chat
  tool then fails loudly at call time ("SideChatManager not
  initialized") instead of pretending to work.
- 650: the CARD_HINTS owner-marking rung is supplementary — 
  ModelCard.from_provider already derives owner from the model id;
  a hints failure leaves cards exactly as the primary rung made them.
  owner is a routing preference, not a capability.
- 684: a per-model lifecycle registration failure skips that model;
  the promote step re-checks broker.card(primary), so a skipped
  primary is never falsely promoted. The whole block's failure is
  warning-logged by its outer guard. A skipped model is visible as
  broker absence, not as a false claim; the missing per-model log
  line is a shelved observability sketch (SURVEY shelf 2).
- 714: the DebugHub hook is console telemetry only; its failure
  removes LLM call traces and nothing else ("must never break boot
  or routing").
"""

from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents import failures as ag_fail
from nomorals.agents.context import AgentContext, build_context, build_router
from nomorals.core.config import Settings
from nomorals.core.events import EventBus
from nomorals.games import achievements as ach


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def test_87_progression_grants_failure_is_fail_closed():
    ctx = AgentContext(settings=SimpleNamespace(), db=SimpleNamespace(),
                       bus=SimpleNamespace(), actor="player1")
    before = _snap()
    with patch.object(ach, "progression_capabilities",
                      side_effect=RuntimeError("achievements down")):
        assert ctx._progression_grants() == set()
    assert _snap() == before
    with patch.object(ach, "progression_capabilities",
                      return_value={"fly", "swim"}):
        assert ctx._progression_grants() == {"fly", "swim"}
    ctx2 = AgentContext(settings=SimpleNamespace(), db=None,
                        bus=SimpleNamespace(), actor="player1")
    assert ctx2._progression_grants() == set()


def test_312_side_chat_wiring_failure_boot_completes(tmp_path):
    calls = []

    class StubDB:
        def migrate(self):
            pass

    class StubSCM:
        def __init__(self, db):
            self.db = db

    def raising_init(mgr):
        raise RuntimeError("side chats wiring down")

    # SideChatManager is stubbed too, so the ONLY failure source in
    # the wiring block is init_side_chats itself.
    before = _snap()
    with patch("nomorals.social.chat.side_chats.SideChatManager",
               StubSCM), \
            patch("nomorals.tools.side_chats.init_side_chats",
                  raising_init):
        ctx = build_context(Settings(home=tmp_path / "h1"), db=StubDB(),
                            with_executor=False, with_router=False,
                            with_memory=False, with_tools=True)
    assert ctx is not None and ctx.tools is not None
    assert _snap() == before
    with patch("nomorals.social.chat.side_chats.SideChatManager",
               StubSCM), \
            patch("nomorals.tools.side_chats.init_side_chats",
                  side_effect=lambda mgr: calls.append(mgr)):
        ctx2 = build_context(Settings(home=tmp_path / "h2"), db=StubDB(),
                             with_executor=False, with_router=False,
                             with_memory=False, with_tools=True)
    assert len(calls) == 1 and isinstance(calls[0], StubSCM)
    assert ctx2.tools is not None


def test_650_card_hints_failure_leaves_card_unmarked(tmp_path):
    from nomorals.llm import defaults as defaults_mod
    from nomorals.llm.broker import ModelBroker
    from nomorals.llm.capabilities import ModelCard

    card = ModelCard(id="m1", provider="provx")

    class RaisingHints(dict):
        def get(self, *a, **k):
            raise RuntimeError("hints broken")

    before = _snap()
    with patch.object(ModelBroker, "cards", lambda self: [card]), \
            patch.object(defaults_mod, "CARD_HINTS", RaisingHints()):
        router = build_router(Settings(home=tmp_path / "h1"), EventBus())
    assert router is not None and router.broker is not None
    assert card.owner is False
    assert _snap() == before
    card2 = ModelCard(id="m1", provider="provx")
    with patch.object(ModelBroker, "cards", lambda self: [card2]), \
            patch.object(defaults_mod, "CARD_HINTS",
                         {"provx": {"owner": True}}):
        build_router(Settings(home=tmp_path / "h2"), EventBus())
    assert card2.owner is True


def test_684_lifecycle_model_failure_skips_model_only(tmp_path):
    from nomorals.llm import lifecycle as lc_mod
    from nomorals.llm.capabilities import ModelCard

    class FakeLC:
        def __init__(self, db):
            self.provisioner = SimpleNamespace(_managers={})

        @property
        def primary(self):
            return "m1"

        def list(self, status=""):
            return [SimpleNamespace(id="m1", provider="other",
                                    status="ok")]

    before = _snap()
    with patch.object(lc_mod, "ModelLifecycle", FakeLC), \
            patch.object(lc_mod, "card_for_model",
                         side_effect=RuntimeError("card build down")):
        router = build_router(Settings(home=tmp_path / "l1"), EventBus())
    assert router is not None and router.broker is not None
    assert router.broker.card("m1") is None
    assert _snap() == before
    with patch.object(lc_mod, "ModelLifecycle", FakeLC), \
            patch.object(lc_mod, "card_for_model",
                         return_value=ModelCard(id="m1",
                                                provider="other")):
        router2 = build_router(Settings(home=tmp_path / "l2"), EventBus())
    assert router2.broker.card("m1") is not None


def test_714_debug_hub_failure_never_breaks_boot(tmp_path):
    from nomorals.console import debug as debug_mod

    calls = []

    def raising_install(router):
        raise RuntimeError("debug hub down")

    before = _snap()
    with patch.object(debug_mod.DebugHub, "install_llm_hook",
                      staticmethod(raising_install)):
        router = build_router(Settings(home=tmp_path / "d1"), EventBus())
    assert router is not None
    assert _snap() == before
    with patch.object(debug_mod.DebugHub, "install_llm_hook",
                      staticmethod(lambda r: calls.append(r))):
        router2 = build_router(Settings(home=tmp_path / "d2"), EventBus())
    assert calls == [router2]
