"""Wave F3/help — help/catalog sync regression test.

Fails if help drifts from reality again:

CLI side (``nm``)
  * every top-level ``add_parser`` subcommand appears in ``nm help cli``
    output (the overview is generated from the parser — this pins that).
  * every ``CLI_ALIASES`` entry names a real command and its aliases are
    visible in the overview; no alias collides with a canonical command
    name or with another alias.

Chat side (``/commands`` in chat)
  * every ``CONTROL_COMMANDS`` catalog entry has a handler path: either a
    branch in ``PartnerRuntime.handle_control`` or membership in
    ``GAME_COMMANDS`` (intercepted by the any-chat game path into
    ``_control_game``).
  * every ``GAME_COMMANDS`` entry is in the catalog — otherwise
    ``parse_control`` returns None and the any-chat game trigger can never
    fire for it.
  * every ``handle_control`` branch kind is in the catalog, unless it is an
    explicitly listed known orphan (with a reason).
  * every ``self._control_*`` call inside ``handle_control`` resolves to a
    method actually defined on the runtime (catches dispatch typos).

The chat-side checks parse source instead of importing the runtime so the
test stays fast and hermetic.
"""

from __future__ import annotations

import argparse
import ast
import re
import unittest
from pathlib import Path

from nomorals.cli import CLI_ALIASES, _cli_overview, _cli_subparsers

REPO = Path(__file__).resolve().parent.parent
CONTROL_SRC = REPO / "nomorals" / "social" / "chat" / "control.py"
RUNTIME_SRC = REPO / "nomorals" / "agents" / "partner" / "runtime.py"

#: handle_control branch kinds that are intentionally NOT chat catalog
#: entries.  Key → why it is fine.
KNOWN_ORPHANS: dict[str, str] = {
    "error": "parse_control pseudo-kind for argument-validation failures; "
             "never typed as a command, so it has no catalog entry.",
}


def _canonical_commands() -> dict[str, argparse.ArgumentParser]:
    sub = _cli_subparsers()
    assert sub is not None
    out: dict[str, argparse.ArgumentParser] = {}
    for name, target in sub.choices.items():
        out.setdefault(target.prog.split()[-1], target)
    return out


def _chat_namespaces() -> dict[str, set[str]]:
    """CONTROL_COMMANDS keys + GAME_COMMANDS, via AST (no heavy imports)."""
    tree = ast.parse(CONTROL_SRC.read_text())
    ns: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "CONTROL_COMMANDS":
            ns["catalog"] = set(ast.literal_eval(node.value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GAME_COMMANDS":
                    ns["games"] = set(ast.literal_eval(node.value))
    assert "catalog" in ns and "games" in ns, "could not parse control.py"
    return ns


def _handle_control_source() -> str:
    """The source text of PartnerRuntime.handle_control."""
    src = RUNTIME_SRC.read_text()
    start = src.index("    def handle_control(")
    rest = src[start:]
    # next method at the same indent level ends the body
    nxt = re.search(r"\n    def ", rest[1:])
    return rest[: nxt.start() + 1] if nxt else rest


def _handled_kinds(body: str) -> set[str]:
    kinds = set(re.findall(r'if kind == "([^"]+)"', body))
    for m in re.finditer(r"if kind in \{([^}]+)\}", body):
        kinds.update(re.findall(r'"([^"]+)"', m.group(1)))
    return kinds


class CliHelpSyncTests(unittest.TestCase):
    def test_every_subcommand_appears_in_help_cli(self):
        page = _cli_overview()
        missing = [name for name in _canonical_commands()
                   if f"  {name}" not in page]
        self.assertEqual(missing, [],
                         f"subcommands missing from `nm help cli`: {missing}")

    def test_every_alias_entry_is_consumed_and_visible(self):
        page = _cli_overview()
        canonicals = set(_canonical_commands())
        for canonical, aliases in CLI_ALIASES.items():
            self.assertIn(canonical, canonicals,
                          f"CLI_ALIASES key {canonical!r} is not a command")
            self.assertIn(f"  {canonical}", page)
            for alias in aliases:
                self.assertIn(alias, page,
                              f"alias {alias!r} of {canonical!r} not shown "
                              f"in `nm help cli`")

    def test_no_alias_collisions(self):
        canonicals = set(_canonical_commands())
        seen: dict[str, str] = {}
        for canonical, aliases in CLI_ALIASES.items():
            for alias in aliases:
                self.assertNotIn(alias, canonicals,
                                 f"alias {alias!r} collides with a command")
                self.assertNotIn(alias, seen,
                                 f"alias {alias!r} claimed by both "
                                 f"{seen.get(alias)!r} and {canonical!r}")
                seen[alias] = canonical


class ChatCatalogSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = _chat_namespaces()
        cls.catalog: set[str] = ns["catalog"]
        cls.games: set[str] = ns["games"]
        cls.body: str = _handle_control_source()
        cls.handled: set[str] = _handled_kinds(cls.body)
        # Wave H3: PartnerRuntime is a mixin composite — its _control_*
        # handlers live across the partner/ package, so scan every module.
        cls.defined_handlers: set[str] = set()
        for _py in sorted(RUNTIME_SRC.parent.glob("*.py")):
            cls.defined_handlers.update(
                re.findall(r"def (_control_[a-z0-9_]+)\(", _py.read_text()))
        cls.called_handlers: set[str] = set(
            re.findall(r"self\.(_control_[a-z0-9_]+)\(", cls.body))

    def test_every_catalog_command_has_a_handler_path(self):
        orphans = sorted(
            k for k in self.catalog
            if k not in self.handled and k not in self.games
            and k not in KNOWN_ORPHANS
        )
        self.assertEqual(
            orphans, [],
            f"catalog commands with no handler path (add a handle_control "
            f"branch, route via GAME_COMMANDS, or list in KNOWN_ORPHANS): "
            f"{orphans}")

    def test_every_game_command_is_parseable(self):
        # parse_control returns None for kinds outside CONTROL_COMMANDS, so a
        # game in GAME_COMMANDS but not in the catalog can never trigger —
        # not in the owner's chat, not in any chat.
        missing = sorted(g for g in self.games if g not in self.catalog)
        self.assertEqual(missing, [],
                         f"GAME_COMMANDS entries missing from CONTROL_COMMANDS "
                         f"(their /command never parses): {missing}")

    def test_every_handled_kind_is_in_the_catalog(self):
        orphans = sorted(k for k in self.handled
                         if k not in self.catalog and k not in KNOWN_ORPHANS)
        self.assertEqual(
            orphans, [],
            f"handle_control branches with no CONTROL_COMMANDS entry "
            f"(add the catalog entry, or list in KNOWN_ORPHANS with a "
            f"reason): {orphans}")

    def test_known_orphans_are_still_accurate(self):
        # keep the escape hatch honest: every listed orphan must still exist
        # as a handled kind, and the dict must stay small and documented.
        for kind, reason in KNOWN_ORPHANS.items():
            self.assertIn(kind, self.handled,
                          f"known orphan {kind!r} no longer handled — remove "
                          f"it from KNOWN_ORPHANS")
            self.assertTrue(reason.strip(),
                            f"known orphan {kind!r} needs a reason")

    def test_every_called_control_method_is_defined(self):
        missing = sorted(m for m in self.called_handlers
                         if m not in self.defined_handlers)
        self.assertEqual(missing, [],
                         f"handle_control calls undefined methods: {missing}")


if __name__ == "__main__":
    unittest.main()
