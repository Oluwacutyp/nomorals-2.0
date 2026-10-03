"""Wave H3 parity: the partner_runtime facade must expose the exact same
objects as the new nomorals.agents.partner subpackage, and the split
PartnerRuntime must behave like the pre-split god class.

* facade identity: every public name importable from
  ``nomorals.agents.partner_runtime`` is ``is``-identical to the object in
  ``nomorals.agents.partner``.
* MRO: the 12 concern mixins compose in a defined order, no method is
  defined twice (no shadowing), and representative methods resolve to the
  mixin that owns them.
* smoke: construct PartnerRuntime through the facade exactly the way the
  existing partner tests do, and exercise the control dispatcher plus a few
  helpers end to end.
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

import nomorals.agents.partner as partner_pkg
import nomorals.agents.partner_runtime as facade
from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings

PUBLIC_NAMES = [
    "PartnerBrain",
    "PartnerRuntime",
    "PresenceOutcome",
    "_key_set",
    "_direct_approve",
    "_direct_deny",
]

EXPECTED_MRO = [
    "PartnerRuntime",
    "RuntimeLiveMixin",
    "RuntimeMetaMixin",
    "RuntimeMissionMixin",
    "RuntimeBuildMixin",
    "RuntimeScheduleMixin",
    "RuntimeVoiceMixin",
    "RuntimeMemoryMixin",
    "RuntimeGamesMixin",
    "RuntimeSystemMixin",
    "RuntimeMediaMixin",
    "RuntimeIntelMixin",
    "RuntimeSearchMixin",
    "RuntimeWisdomMixin",
    "object",
]

# one representative method per mixin module, proving the method lives where
# the split says it does and resolves through the composed class unchanged.
MIXIN_PROBES = {
    "nomorals.agents.partner.runtime_search": ("RuntimeSearchMixin", "_control_search"),
    "nomorals.agents.partner.runtime_intel": ("RuntimeIntelMixin", "_control_decode"),
    "nomorals.agents.partner.runtime_media": ("RuntimeMediaMixin", "_control_music"),
    "nomorals.agents.partner.runtime_system": ("RuntimeSystemMixin", "_control_exec"),
    "nomorals.agents.partner.runtime_games": ("RuntimeGamesMixin", "_control_game"),
    "nomorals.agents.partner.runtime_memory": ("RuntimeMemoryMixin", "_control_remember"),
    "nomorals.agents.partner.runtime_voice": ("RuntimeVoiceMixin", "_control_tts"),
    "nomorals.agents.partner.runtime_schedule": ("RuntimeScheduleMixin", "_control_schedule"),
    "nomorals.agents.partner.runtime_build": ("RuntimeBuildMixin", "_control_code"),
    "nomorals.agents.partner.runtime_mission": ("RuntimeMissionMixin", "_control_mission"),
    "nomorals.agents.partner.runtime_meta": ("RuntimeMetaMixin", "_control_status"),
    "nomorals.agents.partner.runtime_live": ("RuntimeLiveMixin", "_control_weather"),
    "nomorals.agents.partner.runtime_wisdom": ("RuntimeWisdomMixin", "_control_wisdom"),
}


class FacadeIdentityTest(unittest.TestCase):
    def test_public_names_importable_from_facade(self) -> None:
        for name in PUBLIC_NAMES:
            self.assertTrue(hasattr(facade, name), f"facade missing {name}")

    def test_facade_objects_are_package_objects(self) -> None:
        for name in PUBLIC_NAMES:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(facade, name),
                    getattr(partner_pkg, name),
                    f"{name} is not identical across facade and package",
                )

    def test_all_matches_package_namespace(self) -> None:
        self.assertEqual(facade.__all__, ["PartnerBrain", "PartnerRuntime", "PresenceOutcome"])
        self.assertEqual(partner_pkg.__all__, facade.__all__)

    def test_classes_come_from_new_modules(self) -> None:
        self.assertEqual(PartnerBrain.__module__, "nomorals.agents.partner.brain")
        self.assertEqual(PartnerRuntime.__module__, "nomorals.agents.partner.runtime")
        self.assertEqual(
            facade.PresenceOutcome.__module__, "nomorals.agents.partner.outcome")


class MroTest(unittest.TestCase):
    def test_mro_order_is_defined(self) -> None:
        self.assertEqual([c.__name__ for c in PartnerRuntime.__mro__], EXPECTED_MRO)

    def test_no_method_defined_twice(self) -> None:
        owners: dict[str, str] = {}
        clashes: list[str] = []
        for klass in PartnerRuntime.__mro__:
            if klass is object or klass is PartnerRuntime:
                continue
            for name in klass.__dict__:
                if name.startswith("__"):
                    continue
                if name in owners:
                    clashes.append(f"{name}: {owners[name]} vs {klass.__name__}")
                owners[name] = klass.__name__
        self.assertEqual(clashes, [], "shadowed methods across mixins")

    def test_init_defined_once_on_runtime(self) -> None:
        self.assertIn("__init__", PartnerRuntime.__dict__)
        for klass in PartnerRuntime.__mro__[1:-1]:
            self.assertNotIn("__init__", klass.__dict__, klass.__name__)

    def test_mixin_methods_resolve_to_mixins(self) -> None:
        import importlib

        for mod_name, (mixin_name, method) in MIXIN_PROBES.items():
            with self.subTest(method=method):
                mod = importlib.import_module(mod_name)
                mixin = getattr(mod, mixin_name)
                self.assertIs(
                    getattr(PartnerRuntime, method),
                    getattr(mixin, method),
                    f"{method} does not resolve to {mixin_name}",
                )

    def test_staticmethods_survived_the_split(self) -> None:
        for name in ("_ref_from_key", "_game_chunks", "_format_search_report",
                     "_status_note", "_fake_message"):
            with self.subTest(name=name):
                self.assertIsInstance(
                    inspect.getattr_static(PartnerRuntime, name), staticmethod)


class _StubGateway:
    adapters: dict = {}

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass


class RuntimeSmokeTest(unittest.TestCase):
    """Build a runtime through the facade and run representative behavior."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-parity-")
        settings = load_settings(
            overrides={
                "home": self.tmp.name,
                "partner.platforms": "local",
                "chat.local_enabled": "true",
            }
        )
        self.context = build_context(settings, with_executor=False, with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("smoke test must not call the model")

        self.context.router = _Router()
        # through the facade, like every existing caller does
        self.runtime = facade.PartnerRuntime(self.context, gateway=_StubGateway())

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_brain_wired_cross_module(self) -> None:
        self.assertIsInstance(self.runtime.brain, facade.PartnerBrain)
        self.assertIs(self.runtime.brain.__class__, partner_pkg.PartnerBrain)

    def test_help_dispatches(self) -> None:
        out = self.runtime.handle_control("/help", "local:console")
        self.assertTrue(out.strip(), "expected /help output")

    def test_proposals_empty_initially(self) -> None:
        self.assertEqual(self.runtime.proposals(), [])

    def test_key_set_helper(self) -> None:
        self.assertEqual(facade._key_set("local:console"), {"local:console"})
        self.assertEqual(facade._key_set(""), set())

    def test_static_helpers(self) -> None:
        self.assertEqual(
            PartnerRuntime._game_chunks("abcdef", limit=2), ["ab", "cd", "ef"])
        self.assertEqual(
            PartnerRuntime._ref_from_key("telegram:123").chat_id, "123")

    def test_status_shape(self) -> None:
        status = self.runtime.status()
        self.assertIsInstance(status, dict)
        self.assertTrue(status, "status() should report something")


if __name__ == "__main__":
    unittest.main()
