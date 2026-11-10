"""Doc 18 P6 / G7 — ToolSpec cancellation + lease fields and the
grouped brain listing.

Pins: (a) declarations land on the spec, in schema() (the doc §3 row
export) and in describe(), verbatim; (b) honesty of the defaults —
an undeclared tool reads cancellation "" / "undeclared" and
lease_needs None, and None (never declared) stays distinguishable
from () (declared none) end to end; (c) malformed declarations are
refused typed at registration (ValidationError), never coerced;
(d) grouped_listing puts every tool in exactly one of the doc's
four state groups, shows gated tools instead of hiding them, and
keeps the flat line shape; (e) the flat prompt_listing is
byte-unchanged (parity for its existing consumers); (f) the shipped
builtin catalog classifies whole: every tool has a real handler and
exactly one group.
"""

from __future__ import annotations

import unittest

from nomorals.core.errors import ValidationError
from nomorals.core.policy import CapabilitySet
from nomorals.tools.registry import (
    CANCELLATION_MECHANISMS,
    LISTING_GROUPS,
    ToolHealth,
    ToolRegistry,
    ToolSpec,
)


def _fn(x: str = "") -> str:
    return x


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("plain", _fn, description="plain tool")
    reg.register("leased", _fn, description="needs the gpu",
                 cancellation="cooperative", lease_needs=["gpu"])
    reg.register("declared_none", _fn, description="no leases at all",
                 lease_needs=[])
    reg.register("gated", _fn, description="admin only",
                 capability="admin.danger", cancellation="none")
    reg.register("sick", _fn, description="failing tool")
    reg._health["sick"] = ToolHealth(
        name="sick", status="down", consecutive_failures=3)
    reg.register("tired", _fn, description="degraded tool")
    reg._health["tired"] = ToolHealth(
        name="tired", status="degraded", consecutive_failures=1)
    return reg


class SpecFieldTests(unittest.TestCase):
    def test_declared_fields_round_trip(self):
        spec = _registry().get("leased")
        self.assertEqual(spec.cancellation, "cooperative")
        self.assertEqual(spec.lease_needs, ("gpu",))
        schema = spec.schema()
        self.assertEqual(schema["cancellation"], "cooperative")
        self.assertEqual(schema["lease_needs"], ["gpu"])
        described = _registry().describe("leased")
        self.assertEqual(described["cancellation"], "cooperative")
        self.assertEqual(described["lease_needs"], ["gpu"])

    def test_undeclared_is_honest_not_guessed(self):
        reg = _registry()
        spec = reg.get("plain")
        self.assertEqual(spec.cancellation, "")
        self.assertIsNone(spec.lease_needs)
        self.assertEqual(spec.schema()["cancellation"], "")
        self.assertIsNone(spec.schema()["lease_needs"])
        described = reg.describe("plain")
        self.assertEqual(described["cancellation"], "undeclared")
        self.assertIsNone(described["lease_needs"])

    def test_declared_none_differs_from_never_declared(self):
        spec = _registry().get("declared_none")
        self.assertEqual(spec.lease_needs, ())
        self.assertEqual(spec.schema()["lease_needs"], [])
        self.assertIsNotNone(spec.schema()["lease_needs"])

    def test_every_mechanism_accepted(self):
        for mechanism in CANCELLATION_MECHANISMS:
            spec = ToolSpec(name="t", fn=_fn, cancellation=mechanism)
            self.assertEqual(spec.cancellation, mechanism)

    def test_malformed_cancellation_refused_typed(self):
        with self.assertRaises(ValidationError):
            ToolSpec(name="t", fn=_fn, cancellation="whenever")
        reg = ToolRegistry()
        with self.assertRaises(ValidationError):
            reg.register("bad", _fn, cancellation="maybe")

    def test_malformed_lease_needs_refused_typed(self):
        # A bare string would otherwise iterate into single letters.
        with self.assertRaises(ValidationError):
            ToolSpec(name="t", fn=_fn, lease_needs="gpu")
        with self.assertRaises(ValidationError):
            ToolSpec(name="t", fn=_fn, lease_needs=42)
        with self.assertRaises(ValidationError):
            ToolSpec(name="t", fn=_fn, lease_needs=["gpu", "  "])

    def test_lease_needs_list_normalizes_to_tuple(self):
        spec = ToolSpec(name="t", fn=_fn, lease_needs=[" gpu ", "file"])
        self.assertEqual(spec.lease_needs, ("gpu", "file"))


class GroupedListingTests(unittest.TestCase):
    def test_groups_and_exactly_one_home(self):
        listing = _registry().grouped_listing(
            capabilities=CapabilitySet.none())
        for group in LISTING_GROUPS:
            self.assertIn(f"## {group} (", listing)
        lines = [ln for ln in listing.splitlines() if ln.startswith("- ")]
        names = [ln[2:].split("(", 1)[0] for ln in lines]
        self.assertEqual(sorted(names), ["declared_none", "gated",
                                         "leased", "plain", "sick",
                                         "tired"])

    def test_group_membership_and_line_facts(self):
        listing = _registry().grouped_listing(
            capabilities=CapabilitySet.none())
        sections = {}
        current = None
        for line in listing.splitlines():
            if line.startswith("## "):
                current = line[3:].rsplit(" (", 1)[0]
                sections[current] = []
            elif line.startswith("- ") and current:
                sections[current].append(line)
        self.assertIn("- plain(", "\n".join(sections["available now"]))
        self.assertIn("- declared_none(",
                      "\n".join(sections["available now"]))
        tired = "\n".join(sections["available now"])
        self.assertIn("[health: degraded]", tired)
        leased = "\n".join(sections["requires resource/lease"])
        self.assertIn("[leases: gpu]", leased)
        gated = "\n".join(sections["rejected unless authority"])
        self.assertIn("- gated(", gated)
        self.assertIn("[cancellation: none]", gated)
        self.assertIn("- sick(", "\n".join(sections["known down"]))

    def test_granted_capability_leaves_the_rejected_group(self):
        listing = _registry().grouped_listing(
            capabilities=CapabilitySet.of("admin.danger"))
        sections = listing.split("## rejected unless authority (0)")
        self.assertEqual(len(sections), 2)
        self.assertIn("- gated(", listing.split("## available now")[1])

    def test_no_capability_set_means_no_rejections(self):
        listing = _registry().grouped_listing()
        self.assertIn("## rejected unless authority (0)", listing)
        self.assertIn("- gated(", listing)

    def test_flat_listing_byte_parity(self):
        reg = _registry()
        flat = reg.prompt_listing()
        self.assertEqual(
            flat,
            "- plain(x=''): plain tool\n"
            "- leased(x=''): needs the gpu\n"
            "- declared_none(x=''): no leases at all\n"
            "- gated(x=''): admin only\n"
            "- sick(x=''): failing tool\n"
            "- tired(x=''): degraded tool")


class BuiltinCatalogTests(unittest.TestCase):
    def test_builtin_catalog_classifies_whole(self):
        reg = ToolRegistry().register_builtins()
        names = [spec.name for spec in reg._tools.values()
                 if not spec.deprecated]
        self.assertGreater(len(names), 50)
        for spec in reg._tools.values():
            self.assertTrue(callable(spec.fn), spec.name)
        listing = reg.grouped_listing(
            capabilities=CapabilitySet.none())
        seen = [ln[2:].split("(", 1)[0]
                for ln in listing.splitlines() if ln.startswith("- ")]
        self.assertEqual(sorted(seen), sorted(names))


if __name__ == "__main__":
    unittest.main()
