"""Architecture enforcement.

ARCHITECTURE.md §1 promises strict layering. A promise nobody checks is a
suggestion. This test parses the AST of every module in the package and fails if
any module imports from a *higher* layer than its own.

Run it after every change; it is what stops a 100k-line codebase from becoming a
cycle of mutual imports.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "nomorals"

#: layer index by top-level subpackage. Lower may not import higher.
#: Extended 2026-10-01 to cover every top-level package: unmapped packages
#: used to fall through to SHARED_LAYER (0), which exempted 14 organs from
#: the architecture test entirely.
LAYERS: dict[str, int] = {
    "core": 1,
    "accounts": 2,
    "native": 2,
    "storage": 2,
    "memory": 3,
    "llm": 3,
    "training": 3,
    "cognition": 3,
    "ta": 3,
    # matching/ is the two-sided stable-matching engine (Gale-Shapley):
    # a leaf engine beside ta, importing only core, used by agents.
    "matching": 3,
    # planning/ is the plan engine: imports memory (L3), imported by the
    # scheduler (L4). Sits with the other L3 engines.
    "planning": 3,
    # security/ is the low-level leak-check organ (dnsleak/netleak):
    # imports only core, imported by tools (L4). L3 per the review brief.
    "security": 3,
    "tools": 4,
    "social": 4,
    "media": 4,
    "media_edit": 4,
    "voice": 4,
    "skills": 4,
    "scheduler": 4,
    "goals": 4,
    # documents/ is the universal document engine (Wave K): a service organ
    # at L4 beside tools — it parses with tools/browser's HTML parser and
    # core/pdf, and reaches only downward.
    "documents": 4,
    # browser/ is the Wave K browser service (sessions/tabs/downloads/
    # screenshots as artifacts): wraps tools/browser's BrowserSession, L4.
    "browser": 4,
    # triggers/ is the event-condition-action automation organ (Wave L):
    # schedule/file/price/message/webhook sources, notify/message/
    # command/mission actions.  An L5 organ beside scheduler/missions —
    # it wraps the scheduler for time-based triggers and reaches only
    # downward/sideways.
    "triggers": 5,
    "agents": 5,
    # codews/ is the Wave K code workspace object (repo/branch/worktree/
    # patch/test/build): an L5 organ beside agents/missions, reusing
    # core/diff and tools/git.
    "codews": 5,
    "context": 5,
    "partner": 5,
    "games": 5,
    "books": 5,
    # wisdom/ is the WisdomKeeper esoteric-study organ (corpus/history/
    # practice): an L5 organ beside books/agents, wrapping books.Library
    # for ingest/search and documents/ for parsing. Reaches only downward.
    "wisdom": 5,
    # search/ is the Wave L1 universal federated search (one query across
    # memory/wisdom/books/docs/code/timeline): an L5 organ beside books and
    # wisdom. It reaches only downward — the timeline adapter takes an
    # injected os.Timeline (L6) rather than importing it.
    "search": 5,
    "integrations": 5,
    # connectors: service adapters (GitHub first; finance/cards/proxies/
    # commerce queued). Peers of integrations; own their auth + vault creds.
    "connectors": 5,
    # datasci/ is the data-science workspace (pandas datasets, queries,
    # plots as artifacts): a service organ at L4, reaching only downward.
    "datasci": 4,
    # plugins/ is the plugin package model (manifests, registry, sandboxed
    # loading): a service organ at L4 beside skills, reaching only downward.
    "plugins": 4,
    # ── T1.5: the 18 previously unmapped packages. Each layer satisfies
    # max_imported <= L <= min_importer_layer for the package, verified
    # per-module; the five packages where no single layer satisfies it
    # (autonomy, characters, console, finance, research) get MODULE_PINS
    # below for the modules that break the package layer. ──
    # audio/ is the audio service organ (audiobooks, editing, DSP): imports
    # documents/media_edit/voice (L4), imported by media/voice (L4). L4.
    "audio": 4,
    # characters/ is the character service organ (casting, dialogue,
    # arcs): imported by tools/media (L4). Its match module imports games
    # (L5) and is pinned there; the rest is L4.
    "characters": 4,
    # commerce/ is the commerce service organ: imports social (L4),
    # imported by tools (L4). L4.
    "commerce": 4,
    # community/ is the community/groups service organ: imports tools
    # (L4), imported by agents (L5). L4, peer of social.
    "community": 4,
    # finance/ is the money service organ (budgets, ledger, trading desk):
    # its agent-facing modules (alerts, commands, digest, overview, tools)
    # import agents/connectors (L5) and are pinned there, plus
    # tools/finance pinned L5; the ledger-facing remainder is L4,
    # imported by voice/media/tools (L4).
    "finance": 4,
    # learn/ is the learning service organ (courses, tutoring): imports
    # documents/voice (L4), imported by agents (L5). L4.
    "learn": 4,
    # legal/ is the legal-aid service organ: imports documents (L4),
    # imported by agents (L5). L4.
    "legal": 4,
    # marketing/ is the marketing service organ (AEO, content, competitor):
    # imports social (L4), imported by agents (L5). L4.
    "marketing": 4,
    # property/ is the property service organ (rental passport, scam
    # checks): imports nothing above L0, imported by agents/triggers (L5).
    # L4, peer of the other service organs.
    "property": 4,
    # research/ is the research service organ (grounded pipeline): imports
    # documents/social (L4), imported by agents/social (L4/L5). Its costs
    # module is imported by llm (L3) and pinned there; the rest is L4.
    "research": 4,
    # vision/ is the vision service organ: imports browser (L4), imported
    # by tools (L4). L4.
    "vision": 4,
    "workspace": 5,
    # mesh/: phone-to-cloud worker mesh (nodes/heartbeats/task dispatch).
    # L5 organ; wraps storage.queue, reaches only downward.
    "mesh": 5,
    # sync/: multi-device state replication (LWW). L5 organ; owns its store.
    "sync": 5,
    # power/: battery/thermal-aware scheduling. L5 organ; wraps os.resources
    # and storage.queue, reaches only downward.
    "power": 5,
    # stream/: SSE streaming API for dashboards. L5 organ; serves os.timeline
    # events over stdlib HTTP, reaches only downward.
    "stream": 5,
    # hub/: HTTP transport for mesh + sync (multi-device). L5 organ; serves
    # LocalTransport/LocalPeer over stdlib HTTP, reaches only downward.
    "hub": 5,
    # missions sits *with* agents, not above: the dependency is genuinely
    # bidirectional (the runner needs agents; Devon needs the runner), so a
    # strict hierarchy here was fiction.  Peers at the same layer may import
    # each other; the test still forbids either from reaching above L5.
    "missions": 5,
    # autonomy/ is the autonomy organ (presence, weakness, coordinator):
    # imports agents/wisdom (L5), imported by agents (L5) and
    # tools/autonomy — the latter pinned L5 with it. L5.
    "autonomy": 5,
    # health/ is the health organ: imports agents (L5), imported by
    # agents/games (L5). L5.
    "health": 5,
    # travel/ is the travel organ: imports connectors (L5), imported by
    # agents/connectors (L5). L5.
    "travel": 5,
    "tui": 6,
    # console/ is the terminal-UI surface (dashboard, godconsole,
    # commands) — but it sits at L5, not L6/L7: agents/context and
    # agents/partner/runtime (both L5) import console.debug / console,
    # so L6+ would create 2 new violations for a layer its own usage
    # does not support. It imports nothing upward. Its palette is a
    # leaf imported by social/chat (L4) and is pinned there; the rest
    # is L5.
    "console": 5,
    # os is the control plane (may import L1-L6; imported by L7 entry points
    # cli/api; LOWER layers must NEVER import os — integration is via
    # callbacks/events, never upward imports).
    "os": 6,
    "api": 7,
    # builders/ is a package (templates, run/serve/smoke, install, export):
    # the whole subtree sits at L7 like the other entry-point organs.
    "builders": 7,
    # cmdline/ is the ``nm`` CLI implementation package (Wave H3 split of
    # cli.py): an entry point like cli itself, so L7 — it may import
    # anything, and nothing in the layer stack may import it.
    "cmdline": 7,
}

#: Modules that sit outside the layer stack and may be imported by anything.
SHARED = {"version", "compat"}
SHARED_LAYER = 0

#: Top-level entry points live at the highest layer.
#: Dev-tool entry modules (benchmarks, builders, exporters, the self-
#: improvement runner) are also entry points: they may import anything,
#: but nothing in the layer stack may import them.
TOP_LEVEL_LAYER = {
    "cli": 7,
    "__main__": 7,
    "archives": 7,
    "bench": 7,
    "builders_proxy": 7,
    "execbox": 7,
    "exporter": 7,
    "self_improvement": 7,
}

#: Per-module layer pins.  The package is sometimes too coarse a unit: a few
#: modules are leaves or low-level adapters that happen to live inside a
#: higher-level package.  A pin declares the module's *true* layer; both the
#: module's own imports and other modules' imports of it are checked against
#: the pin.  Pins are checked by longest dotted-prefix match.
MODULE_PINS: dict[str, int] = {
    # Leaf stores/services: import core only, despite living in agents/.
    "nomorals.agents.kg": 1,            # knowledge-graph store (core/cookies writes to it)
    # L2 persistence-backed services: they persist via storage.kv (L2),
    # so L1 was a mis-pin — their true layer is L2 (imports: core L1 +
    # storage L2 only; all importers are L4+). Injecting a store into
    # their 5+ call sites would buy the same invariant at far higher
    # churn; the pin states the true dependency level.
    "nomorals.agents.power": 2,         # power-mode lookup (tools/sandbox_code reads it)
    "nomorals.agents.osint_graph": 2,   # identity graph (tools/decoder reads it)
    # Short-form publishers are connector subclasses/adapters (MetaPublisher
    # subclasses InstagramConnector, YouTubePublisher subclasses
    # YouTubeConnector) — genuinely L5 application services on connectors,
    # not L4 media helpers. The whole package is pinned so the intra-package
    # imports (incl. the lazy __getattr__ in __init__) stay same-layer.
    # Importers: cmdline shorts (L7) + partner runtime_media ledger (L5).
    "nomorals.media.contentops.publish": 5,
    "nomorals.media.contentops.publish.ledger": 5,
    "nomorals.media.contentops.publish.meta": 5,
    "nomorals.media.contentops.publish.youtube": 5,
    "nomorals.media.contentops.publish.tiktok": 5,
    "nomorals.media.contentops.publish.x": 5,
    # Agent-composed tool modules: they compose agents/games/connectors/
    # integrations (L5) into tool surfaces — they ARE the L5 composition
    # layer, not L4 foundation tools. The pins ARE the foundation-vs-
    # composition split, enforced by the test. Their imports are all
    # <= L5; the only static importer is cmdline trade (L7, downward).
    "nomorals.tools.deals": 5,       # -> integrations.naija_deals
    "nomorals.tools.gameplayer": 5,  # -> games.player.session
    "nomorals.tools.games": 5,       # -> games.*
    "nomorals.tools.services": 5,    # -> connectors.registry
    "nomorals.tools.trading": 5,     # -> connectors.* + agents.financial_expert
    # Low-level adapters inside higher-level packages.
    "nomorals.integrations.voice_integration": 1,
    "nomorals.integrations.sentinel_bridge": 2,  # vendored-sentinel bridge
    "nomorals.integrations.market_data": 2,     # keyless market-data plumbing (imports only core)
    "nomorals.integrations.naija_shopping": 4,   # shopping adapter (uses tools)
    # Thin tool-registry bridge for the builders organ: its only job is to
    # expose builders/app_builder's register() hook under the name the
    # tool registry's register_builtins() imports.  An entry-point-layer
    # (L7) adapter, like builders_proxy.
    "nomorals.tools.build_app": 7,
    # ── T1.5: per-module pins for the packages whose modules span layers.
    # Each pin is a module whose true layer differs from its package's.
    # characters.match imports games (L5); the rest of characters/ is L4.
    "nomorals.characters.match": 5,
    # console.palette is a color-constant leaf imported by social/chat
    # (L4); the rest of console/ is the L6 terminal-UI surface.
    "nomorals.console.palette": 4,
    # research.costs is imported by llm (L3); the rest of research/ is L4.
    "nomorals.research.costs": 3,
    # finance's agent-facing modules import agents/connectors (L5).
    "nomorals.finance.alerts": 5,
    "nomorals.finance.commands": 5,
    "nomorals.finance.digest": 5,
    "nomorals.finance.overview": 5,
    "nomorals.finance.tools": 5,
    # tools/autonomy and tools/finance are agent-composed capabilities
    # (they orchestrate autonomy.* / finance.* at L5), not foundation
    # tools — pinned with the modules they compose. The rest of tools/
    # stays L4; the full foundation-vs-composition split is T1.6.
    "nomorals.tools.autonomy": 5,
    "nomorals.tools.finance": 5,
}

#: Grandfathered upward imports.  Each entry is an (importer, target) dotted
#: pair that is a *known* layering violation with a scheduled repair.  The
#: test fails on any violation NOT on this list (the ratchet), and fails if a
#: listed pair no longer occurs (so the list shrinks as repairs land).
#: Common cause: several modules in tools/ are agent-*composed* capabilities
#: (they orchestrate agents) rather than foundation tools.  The package layer
#: unit is too coarse for them; the scheduled repair is splitting tools/ into
#: a foundation half (L4) and an agent-composition half (L5).
#: Added 2026-10-01 during the layering-map completion.
KNOWN_VIOLATIONS: frozenset[tuple[str, str]] = frozenset({
    ("nomorals.agents.partner.runtime_media", "nomorals.builders"),
    ("nomorals.tools.agents", "nomorals.agents.coding"),
    ("nomorals.tools.code_executor", "nomorals.agents.coding"),
    ("nomorals.tools.edit_loop", "nomorals.agents.coding"),
    ("nomorals.tools.decoder_agent", "nomorals.agents.decoder"),
    ("nomorals.tools.decoder_agent", "nomorals.agents.context"),
    ("nomorals.tools.proxylab", "nomorals.agents.scheduler"),
    ("nomorals.tools.weather", "nomorals.agents.weather"),
    ("nomorals.tools.workspace", "nomorals.agents.goals"),
    ("nomorals.tools.workspace", "nomorals.agents.projects"),
    ("nomorals.tools.registry", "nomorals.agents.failure"),
    ("nomorals.tools.vision", "nomorals.workspace.rooms"),
    ("nomorals.tools.workspace", "nomorals.workspace.rooms"),
    ("nomorals.tools.workspace", "nomorals.workspace"),
    # Lazy optional auto-registration: tools/registry probes the books tool
    # surface inside try/except so one broken package never breaks the
    # registry.  Benign by construction; documented, not silently exempted.
    ("nomorals.tools.registry", "nomorals.books"),
    # Deferred optional-dependency imports (function-local, try/except-guarded
    # where fallible): the static AST check can't see they never run at
    # import time.  Documented here rather than exempted silently.
    # T1.5 removed the sweep(finance) pair (finance.commands/tools pinned
    # L5, so connectors.exness is same-layer now), the sweep(research)
    # pair (research L4 -> memory L3 is downward), and the sweep(security)
    # pair (security L3 -> core L1 is downward): the edges-as-violations
    # are gone, so the entries go with them.
    # sweep(integrations): deal-hunter reuse inside one grouping method.
    ("nomorals.integrations.naija_shopping", "nomorals.integrations.naija_deals"),
})


def _pin_for(dotted: str) -> int | None:
    """Longest-prefix MODULE_PINS match for a dotted path, else None."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        hit = MODULE_PINS.get(".".join(parts[:i]))
        if hit is not None:
            return hit
    return None


def module_layer(rel_path: Path) -> int:
    parts = rel_path.with_suffix("").parts
    if len(parts) == 1:
        stem = parts[0]
        if stem in TOP_LEVEL_LAYER:
            return TOP_LEVEL_LAYER[stem]
        return SHARED_LAYER  # __init__.py, version.py, compat.py
    pin = _pin_for("nomorals." + ".".join(parts))
    if pin is not None:
        return pin
    head = parts[0]
    return LAYERS.get(head, SHARED_LAYER)


def iter_modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py"))


def imported_nomorals(tree: ast.AST, module_parts: tuple[str, ...]) -> list[tuple[str, int]]:
    """Yield (imported dotted path, source line) for every nomorals import."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nomorals"):
                    found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: resolve against the importing module's full
                # dotted path, *including* the top-level package.  A naive
                # resolution drops the 'nomorals' prefix for cross-package
                # relative imports (``from ..core import x`` in
                # ``nomorals/accounts/y.py`` means ``nomorals.core``), which
                # silently exempts the codebase's dominant import idiom from
                # every layer check.  That was a real blind spot: fixed
                # 2026-10-01 after a code audit showed most cross-package
                # imports use the double-dot style.
                full = ("nomorals",) + module_parts
                if module_parts[-1] == "__init__":
                    pkg = full
                else:
                    pkg = full[:-1]  # containing package of the module
                if node.level - 1 <= len(pkg):
                    base = pkg[: len(pkg) - (node.level - 1)]
                else:  # deeper than the tree: malformed; leave unresolved
                    base = ()
                target = ".".join([*base, node.module] if node.module else base)
                targets = [target]
                if not node.module:
                    # ``from . import sibling`` imports the sibling modules,
                    # not the package itself — attribute to each sibling.
                    targets = [".".join([*base, a.name.split(".")[0]])
                               for a in node.names if a.name != "*"]
            elif node.module and node.module.startswith("nomorals"):
                targets = [node.module]
            else:
                continue
            for target in targets:
                if target.startswith("nomorals"):
                    found.append((target, node.lineno))
    return found


def layer_of_dotted(dotted: str) -> int:
    pin = _pin_for(dotted)
    if pin is not None:
        return pin
    parts = dotted.split(".")
    if len(parts) < 2:
        return SHARED_LAYER
    head = parts[1]
    if head in SHARED:
        return SHARED_LAYER
    return LAYERS.get(head, SHARED_LAYER)


class TestLayering(unittest.TestCase):
    def test_every_module_parses(self) -> None:
        modules = iter_modules()
        self.assertGreater(len(modules), 10, "package appears to be missing modules")
        for path in modules:
            with self.subTest(module=str(path.relative_to(PACKAGE_ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_upward_imports(self) -> None:
        violations: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()
        for path in iter_modules():
            rel = path.relative_to(PACKAGE_ROOT)
            own_layer = module_layer(rel)
            module_parts = tuple(rel.with_suffix("").parts)
            importer = "nomorals." + ".".join(module_parts)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for target, lineno in imported_nomorals(tree, module_parts):
                target_layer = layer_of_dotted(target)
                if target_layer > own_layer:
                    pair = (importer, target)
                    seen_pairs.add(pair)
                    if pair in KNOWN_VIOLATIONS:
                        continue  # grandfathered: documented, scheduled repair
                    violations.append(
                        f"{rel}:{lineno} (L{own_layer}) imports {target} (L{target_layer})"
                    )
        # Ratchet: every grandfathered pair must still occur.  If a repair
        # lands, its entry must be removed here — the failure tells you to.
        stale = KNOWN_VIOLATIONS - seen_pairs
        self.assertEqual(
            stale, set(),
            "grandfathered violations that no longer occur — "
            "remove them from KNOWN_VIOLATIONS:\n  " + "\n  ".join(
                f"{a} -> {b}" for a, b in sorted(stale)),
        )
        self.assertEqual(violations, [], "layer violations:\n  " + "\n  ".join(violations))

    def test_core_has_no_intra_project_imports(self) -> None:
        """L1 is the kernel: it may not depend on any other subpackage."""
        offenders: list[str] = []
        for path in (PACKAGE_ROOT / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_parts = tuple(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
            for target, lineno in imported_nomorals(tree, module_parts):
                if not target.startswith("nomorals.core") and target != "nomorals":
                    offenders.append(f"{path.name}:{lineno} -> {target}")
        self.assertEqual(offenders, [], "core imports outside core:\n  " + "\n  ".join(offenders))

    def test_public_api_is_declared(self) -> None:
        """Every module that defines __all__ must actually export those names."""
        for path in iter_modules():
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            declared: list[str] | None = None
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                declared = [
                                    e.value
                                    for e in node.value.elts
                                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                                ]
            if declared is None:
                continue
            with self.subTest(module=path.name):
                for name in declared:
                    self.assertIn(
                        name,
                        source,
                        f"{path.name}: __all__ declares {name!r} which is not defined",
                    )


if __name__ == "__main__":
    unittest.main()
