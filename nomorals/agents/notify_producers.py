"""Producer census for the owner Notifier (Doc 15, Pass 1).

Every call site that can reach :meth:`Notifier.publish` — directly, via
``publish_doc``, or via the module-level ``notify`` / ``notify_doc``
helpers — is enumerated in :data:`PRODUCER_SITES`, classified by
producer class with its *current* AlertHints posture:

* ``hinted`` — the call passes ``alert_hints=`` explicitly, so the
  alert-policy decision runs for it today.
* ``forwarding`` — the call forwards ``**kwargs``; hints reach the
  policy only when an upstream caller supplied them (the notifier's
  own plumbing and duck-typed forwarders).
* ``unhinted`` — the call passes no hints.  In Pass 1 these
  *bypassed* the policy entirely (the decision ran only ``if
  alert_hints is not None``); since Pass 2 the bypass is closed —
  the mandatory decision runs on the ReportCandidate the adapter
  builds from the call (see :mod:`.report_candidate` and
  :meth:`.alert_policy.AlertPolicy.decide_candidate`), and each one
  leaves a candidate + decision record in the candidate journal
  with its missing evidence marked.  The posture recorded here
  still describes the call's *hints* shape only, exactly as the
  AST derives it.

Attribute calls named ``publish``/``publish_doc`` whose receiver is
NOT the Notifier (the event bus, living-card hosts, platform
publishers, the MQTT client, …) are enumerated in
:data:`EXCLUDED_SITES` with the reason — the census is complete over
the syntactic population, so a *new* publish-shaped call site anywhere
in the package fails the inventory test until a human classifies it
here.  That pin is the point: the producer set can never again drift
silently the way it did between the doc's baseline and the WS3/WS4
ships.

Postures in the table are not trusted prose: the test re-derives each
site's posture from the AST (an explicit ``alert_hints=`` keyword, a
``**kwargs`` forward, or neither) and fails on any mismatch.

Updating this file: add the new call site, then add one row —
``PRODUCER_SITES`` with its class (and the posture the AST shows) or
``EXCLUDED_SITES`` with its reason — in the same change.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PUBLISH_METHODS", "HELPER_METHODS",
    "POSTURE_HINTED", "POSTURE_UNHINTED", "POSTURE_FORWARDING",
    "PRODUCER_CLASSES", "EXCLUSION_REASONS",
    "ScannedSite", "ProducerSite", "ExcludedSite",
    "PRODUCER_SITES", "EXCLUDED_SITES",
    "scan_publish_sites", "derive_posture",
    "unregistered_sites", "stale_sites", "validate_inventory",
    "producer_class_for_module", "MODULE_CLASSES",
]

#: attribute methods that reach the Notifier's decision path
PUBLISH_METHODS = ("publish", "publish_doc")
#: module-level helpers in :mod:`.notifier` that funnel into publish
HELPER_METHODS = ("notify", "notify_doc")

POSTURE_HINTED = "hinted"
POSTURE_UNHINTED = "unhinted"
POSTURE_FORWARDING = "forwarding"

#: the producer-class vocabulary (doc §2's domain owners, plus the
#: notifier's own plumbing as ``internal``)
PRODUCER_CLASSES = (
    "mission", "job", "social", "trading", "finance", "research",
    "briefing", "partner", "system", "media", "internal",
)

#: why an attribute publish-shaped call is NOT a Notifier producer
EXCLUSION_REASONS = (
    "event-bus",           # core.events.EventBus.publish — internal bus
    "card-host",           # living-card host publish (cards, not alerts)
    "platform-publisher",  # SocialManager / video publishers post content
    "mqtt",                # MQTT client publish
    "git",                 # git publisher (evolution)
    "engine-hop",          # engine-internal publish forwarding to a
                           # producer site inventoried at its notifier hop
    "scoreboard",          # games scoreboard bookkeeping
)


@dataclass(frozen=True)
class ScannedSite:
    """One publish-shaped call found by the AST scan."""

    path: str        # repo-relative, e.g. "nomorals/agents/monitor.py"
    qualname: str    # enclosing class/function chain
    index: int       # n-th publish-family call inside that qualname
    method: str      # publish | publish_doc | notify | notify_doc
    hints: bool      # call passes alert_hints= explicitly
    star: bool       # call forwards **kwargs
    line: int = 0    # source line at scan time (debugging aid only)

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.path, self.qualname, self.index, self.method)


@dataclass(frozen=True)
class ProducerSite:
    path: str
    qualname: str
    index: int
    method: str
    producer_class: str
    posture: str
    note: str = ""

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.path, self.qualname, self.index, self.method)


@dataclass(frozen=True)
class ExcludedSite:
    path: str
    qualname: str
    index: int
    method: str
    reason: str
    note: str = ""

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.path, self.qualname, self.index, self.method)


def derive_posture(site: ScannedSite) -> str:
    """The hints posture the AST proves for one scanned call site."""
    if site.hints:
        return POSTURE_HINTED
    if site.star:
        return POSTURE_FORWARDING
    return POSTURE_UNHINTED


# ── the scanner ──────────────────────────────────────────────────────


def _is_notifier_module(module: str) -> bool:
    return module == "notifier" or module.endswith(".notifier")


def _direct_bindings(stmts: list[ast.stmt]) -> dict[str, str]:
    """Names bound directly in one scope (no descent into nested
    defs): name → the ORIGINAL helper method (``"notify"`` /
    ``"notify_doc"``) when the binding is one of the notifier's
    helpers imported from a ``*notifier`` module (aliases resolve to
    the imported name), ``""`` for any other binding (parameters are
    added by the caller).

    Scope-awareness is what keeps a *parameter* or callback that
    happens to be called ``notify`` (arena run_cycle, browser jobs)
    from ever scanning as a helper site, while the dominant
    function-local ``from .notifier import notify`` pattern binds
    exactly inside its own function.
    """
    bindings: dict[str, str] = {}

    class _BindWalk(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            helper = _is_notifier_module(node.module or "")
            for alias in node.names:
                bindings[alias.asname or alias.name] = (
                    alias.name if helper and alias.name in HELPER_METHODS
                    else "")

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                bindings[(alias.asname or alias.name).split(".")[0]] = ""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            bindings[node.name] = ""  # the def binds; body is a new scope

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            bindings[node.name] = ""

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass  # a lambda's bindings live in the lambda

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        bindings[sub.id] = ""
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                bindings[node.target.id] = ""
            self.generic_visit(node)

    walk = _BindWalk()
    for stmt in stmts:
        walk.visit(stmt)
    return bindings


class _SiteVisitor(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.stack: list[str] = []
        self.scopes: list[dict[str, str]] = []
        self.sites: list[ScannedSite] = []
        self._counts: dict[tuple[str, str], int] = {}
        #: receiver calls of an attribute publish site (``notify()
        #: .publish(...)``) — the attribute site is the producer; the
        #: inner helper call is its receiver, not a second site.
        self._skip: set[int] = set()

    def _qualname(self) -> str:
        return ".".join(self.stack)

    def _helper_method(self, name: str) -> str:
        """The original helper method ``name`` is bound to in the
        nearest enclosing scope, or ``""`` when it is not the
        notifier's helper there."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return ""

    def _add(self, method: str, node: ast.Call) -> None:
        key = (self.relpath, self._qualname())
        index = self._counts.get(key, 0)
        self._counts[key] = index + 1
        self.sites.append(ScannedSite(
            path=self.relpath, qualname=self._qualname(), index=index,
            method=method,
            hints=any(k.arg == "alert_hints" for k in node.keywords),
            star=any(k.arg is None for k in node.keywords),
            line=int(getattr(node, "lineno", 0) or 0)))

    def visit_Module(self, node: ast.Module) -> None:
        self.scopes.append(_direct_bindings(node.body))
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.scopes.append(_direct_bindings(node.body))
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        scope: dict[str, str] = {}
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            scope[arg.arg] = ""
        if args.vararg is not None:
            scope[args.vararg.arg] = ""
        if args.kwarg is not None:
            scope[args.kwarg.arg] = ""
        scope.update(_direct_bindings(node.body))
        self.stack.append(node.name)
        self.scopes.append(scope)
        for stmt in node.body:
            self.visit(stmt)
        self.scopes.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        scope: dict[str, str] = {}
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            scope[arg.arg] = ""
        if args.vararg is not None:
            scope[args.vararg.arg] = ""
        if args.kwarg is not None:
            scope[args.kwarg.arg] = ""
        self.scopes.append(scope)
        self.visit(node.body)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in PUBLISH_METHODS:
            self._add(func.attr, node)
            if isinstance(func.value, ast.Call):
                self._skip.add(id(func.value))
        elif isinstance(func, ast.Name) and id(node) not in self._skip:
            method = self._helper_method(func.id)
            if method:
                self._add(method, node)
        self.generic_visit(node)


def scan_publish_sites(root: str | Path) -> list[ScannedSite]:
    """Every publish-shaped call site under ``<root>/nomorals``.

    Deterministic order (path, then source order).  Files that fail
    to parse are skipped here — a syntax error is a build failure
    long before it is an inventory concern.
    """
    package = Path(root) / "nomorals"
    sites: list[ScannedSite] = []
    if not package.is_dir():
        return sites
    for path in sorted(package.rglob("*.py")):
        relpath = path.relative_to(Path(root)).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        visitor = _SiteVisitor(relpath)
        visitor.visit(tree)
        sites.extend(visitor.sites)
    return sites


# ── the tables ───────────────────────────────────────────────────────
# Generated from the scan on current main (Doc 15 Pass 1) and pinned
# by tests/test_d15p1_report_candidates.py.  Posture is re-derived
# from the AST by the test; class/reason are the human census.

PRODUCER_SITES: tuple[ProducerSite, ...] = (
    ProducerSite("nomorals/agents/arena/core.py", "Arena._choked_review_push._wrapped", 0, "notify", "system", "hinted"),
    ProducerSite("nomorals/agents/betdesk/signals.py", "SignalEngine.publish", 0, "publish", "trading", "unhinted"),
    ProducerSite("nomorals/agents/cognition.py", "CognitiveLoop._tick_train", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/directives.py", "DirectivesAgent._notify", 0, "publish", "partner", "unhinted"),
    ProducerSite("nomorals/agents/monitor.py", "MonitorAgent._note_check", 0, "publish", "system", "unhinted"),
    ProducerSite("nomorals/agents/monitor.py", "MonitorAgent._note_error", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/monitor.py", "MonitorAgent._note_error", 1, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/morning_briefing.py", "_deliver", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/morning_briefing.py", "_deliver_fallback", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/morning_pulse.py", "_deliver_text", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/morning_pulse.py", "_deliver_stories", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/morning_pulse.py", "_deliver_stories", 1, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/morning_pulse.py", "_deliver_slot_text", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/motivation.py", "run_motivation", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/news.py", "NewsAgent._score_and_tier", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/news.py", "NewsAgent.brief", 0, "publish", "briefing", "unhinted"),
    ProducerSite("nomorals/agents/notifier.py", "Notifier.publish_doc", 0, "publish", "internal", "forwarding"),
    ProducerSite("nomorals/agents/notifier.py", "notify", 0, "publish", "internal", "forwarding"),
    ProducerSite("nomorals/agents/notifier.py", "notify_doc", 0, "publish", "internal", "forwarding"),
    ProducerSite("nomorals/agents/ops_alerts.py", "ops_alerts", 0, "notify", "system", "hinted"),
    ProducerSite("nomorals/agents/partner/runtime.py", "PartnerRuntime._report_update_recovery", 0, "publish", "partner", "hinted"),
    ProducerSite("nomorals/agents/partner/runtime.py", "PartnerRuntime._report_boot_mission_resume", 0, "publish", "partner", "hinted"),
    ProducerSite("nomorals/agents/partner/runtime_reach.py", "RuntimeReachMixin._takeover_notifier._notify", 0, "publish", "partner", "hinted"),
    ProducerSite("nomorals/agents/pumpdesk/calls.py", "CallEngine.publish", 0, "publish", "trading", "unhinted"),
    ProducerSite("nomorals/agents/pumpdesk/desk.py", "PumpDesk._health_report", 0, "publish", "trading", "hinted"),
    ProducerSite("nomorals/agents/pumpdesk/desk.py", "PumpDesk._health_report", 1, "publish", "trading", "hinted"),
    ProducerSite("nomorals/agents/research_digest.py", "ResearchPipeline.run", 0, "notify", "research", "hinted"),
    ProducerSite("nomorals/agents/research_loop.py", "ResearchLoop._notify", 0, "publish", "research", "hinted"),
    ProducerSite("nomorals/agents/researcher.py", "ResearchAgent.run_cycle", 0, "publish", "research", "hinted"),
    ProducerSite("nomorals/agents/scheduler.py", "Scheduler._execute", 0, "publish", "job", "unhinted"),
    ProducerSite("nomorals/agents/scheduler.py", "Scheduler._execute", 1, "publish", "job", "hinted"),
    ProducerSite("nomorals/agents/scheduler.py", "Scheduler._execute", 2, "publish", "job", "unhinted"),
    ProducerSite("nomorals/agents/scheduler.py", "Scheduler._run_message", 0, "publish", "job", "unhinted"),
    ProducerSite("nomorals/agents/search/jobs.py", "_deliver", 0, "notify", "research", "unhinted"),
    ProducerSite("nomorals/agents/trial/flow.py", "TrialFlow.recover_interrupted_runs", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/trial/flow.py", "TrialFlow.recover_interrupted_watches._report_watch_recovery", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/trial/flow.py", "TrialFlow.recover_interrupted_watches._resumed", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/trial/flow.py", "TrialFlow._notify_owner", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/upgrade_queue.py", "UpgradeQueue.propose", 0, "notify", "partner", "hinted"),
    ProducerSite("nomorals/agents/upgrade_queue.py", "UpgradeQueue._merge_duplicate", 0, "notify", "partner", "hinted"),
    ProducerSite("nomorals/agents/upgrade_queue.py", "UpgradeQueue.approve", 0, "notify", "partner", "hinted"),
    ProducerSite("nomorals/agents/upgrade_queue.py", "UpgradeQueue.deny", 0, "notify", "partner", "hinted"),
    ProducerSite("nomorals/agents/upgrade_queue.py", "UpgradeQueue.record_implemented", 0, "notify", "partner", "hinted"),
    ProducerSite("nomorals/agents/watchers.py", "AlertEngine._send", 0, "publish", "system", "unhinted"),
    ProducerSite("nomorals/agents/watchers.py", "AlertEngine.handle", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/watchers.py", "AlertEngine.handle", 1, "publish", "system", "hinted"),
    ProducerSite("nomorals/agents/watchers.py", "AlertEngine.flush_digest", 0, "publish", "system", "unhinted"),
    ProducerSite("nomorals/backup/report.py", "_publish", 0, "publish", "system", "forwarding"),
    ProducerSite("nomorals/backup/report.py", "_publish", 1, "publish", "system", "forwarding"),
    ProducerSite("nomorals/books/reading_engine.py", "StoryReadingEngine._announce", 0, "publish", "media", "unhinted"),
    ProducerSite("nomorals/connectors/checkpoints.py", "request_human_action", 0, "notify", "system", "hinted"),
    ProducerSite("nomorals/finance/alerts.py", "ensure_alerts_job._run._notify", 0, "publish", "finance", "unhinted"),
    ProducerSite("nomorals/finance/digest.py", "compose_weekly_digest", 0, "publish", "finance", "unhinted"),
    ProducerSite("nomorals/finance/tools.py", "register.finance_alert._notify", 0, "publish", "finance", "unhinted"),
    ProducerSite("nomorals/missions/progress.py", "MissionMilestones.on_started", 0, "publish", "mission", "unhinted"),
    ProducerSite("nomorals/missions/progress.py", "MissionMilestones._on_step", 0, "publish", "mission", "unhinted"),
    ProducerSite("nomorals/missions/progress.py", "MissionMilestones.on_stalled", 0, "publish", "mission", "hinted"),
    ProducerSite("nomorals/missions/progress.py", "MissionMilestones.on_terminal", 0, "publish", "mission", "hinted"),
    ProducerSite("nomorals/missions/runner.py", "MissionRunner._escalate", 0, "publish", "mission", "hinted"),
    ProducerSite("nomorals/missions/runner.py", "MissionRunner._escalate", 1, "publish", "mission", "unhinted"),
    ProducerSite("nomorals/tools/autonomy.py", "register._immune_sampler_tool", 0, "publish", "system", "hinted"),
    ProducerSite("nomorals/triggers/actions.py", "default_notify", 0, "notify", "system", "unhinted"),
    ProducerSite("nomorals/wisdom/chat_session.py", "ChatPracticeSession._send", 0, "publish", "system", "unhinted"),
    ProducerSite("nomorals/workspace/inbox.py", "Inbox._notify", 0, "publish", "system", "forwarding"),
    ProducerSite("nomorals/workspace/inbox.py", "Inbox._notify", 1, "publish", "system", "forwarding"),
)

EXCLUDED_SITES: tuple[ExcludedSite, ...] = (
    ExcludedSite("nomorals/agents/autonomy.py", "AutonomyAgent._emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/agents/betdesk/desk.py", "BettingDesk.run_signals", 0, "publish", "engine-hop"),
    ExcludedSite("nomorals/agents/cognition.py", "CognitiveLoop.tick", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/agents/evolution.py", "register.evolve_git", 0, "publish", "git"),
    ExcludedSite("nomorals/agents/morning_pulse.py", "_emit_bus", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/agents/partner/runtime.py", "PartnerRuntime._publish_startup_card", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime.py", "PartnerRuntime._open_restart_card", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime_build.py", "RuntimeBuildMixin._control_evolve", 0, "publish", "git"),
    ExcludedSite("nomorals/agents/partner/runtime_expression.py", "RuntimeExpressionMixin._control_create", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime_expression.py", "RuntimeExpressionMixin._skillbook_card", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime_expression.py", "RuntimeExpressionMixin._delegation_card", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime_expression.py", "RuntimeExpressionMixin._pack_open", 0, "publish", "card-host"),
    ExcludedSite("nomorals/agents/partner/runtime_search.py", "RuntimeSearchMixin._control_novel", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/agents/pumpdesk/desk.py", "PumpDesk.watch_cycle", 0, "publish", "engine-hop"),
    ExcludedSite("nomorals/agents/scheduler.py", "Scheduler._emit_bus", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/agents/shorts_publisher.py", "_deliver_one", 0, "publish", "platform-publisher"),
    ExcludedSite("nomorals/autonomy/coordinator.py", "IdleCoordinator._maintenance_cycle", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/idle.py", "IdleMonitor._emit_stage", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/idle.py", "IdleMonitor.check_once", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/idle.py", "IdleMonitor.check_once", 1, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/presence.py", "surface", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/presence.py", "heartbeat", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/weakness.py", "investigate_weakness", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/weakness.py", "record_proposal", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/weakness.py", "approve_weakness", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/autonomy/weakness.py", "resolve_weakness", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/books/reader.py", "StoryReader.follow", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/books/reading_engine.py", "StoryReadingEngine._work", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/books/tools.py", "register.story_discovery_sweep", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/browser/daemon.py", "republish_events", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/browser/service.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/cmdline/commands/shorts.py", "_cmd_post", 0, "publish", "platform-publisher"),
    ExcludedSite("nomorals/codews/patch.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/codews/run.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/codews/workspace.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/connectors/base.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/core/events.py", "EventBus.emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/core/events.py", "EventBus._invoke", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/core/jobs.py", "_publish", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/datasci/workspace.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/documents/index.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/documents/parsers.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/games/table.py", "sync_table_card", 0, "publish", "card-host"),
    ExcludedSite("nomorals/hub/server.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/integrations/mqtt_client.py", "MQTTBridge.publish", 0, "publish", "mqtt"),
    ExcludedSite("nomorals/integrations/mqtt_client.py", "MQTTBridge.publish_discovery", 0, "publish", "mqtt"),
    ExcludedSite("nomorals/integrations/mqtt_client.py", "MQTTBridge.set_state", 0, "publish", "mqtt"),
    ExcludedSite("nomorals/integrations/mqtt_client.py", "MQTTBridge.request_state", 0, "publish", "mqtt"),
    ExcludedSite("nomorals/integrations/mqtt_client.py", "MQTTBridge._serve_once._on_connect", 0, "publish", "mqtt"),
    ExcludedSite("nomorals/mesh/node.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/mesh/tasks.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/missions/runner.py", "MissionRunner._emit_bus", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/os/mission_state.py", "_emit_transition", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/os/replay.py", "_emit_redrive", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/os/session.py", "SessionStore.create", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/os/session.py", "SessionStore.end", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/plugins/registry.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/plugins/wiring.py", "make_notifier._notify", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/plugins/wiring.py", "make_audit._audit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/power/telemetry.py", "emit_event", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/search/federated.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/social/chat/group_surface.py", "GroupSurface.render_board", 0, "publish", "scoreboard"),
    ExcludedSite("nomorals/social/drafts.py", "execute_post", 0, "publish", "platform-publisher"),
    ExcludedSite("nomorals/storage/artifacts.py", "_emit_created", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/stream/hub.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/stream/server.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/sync/engine.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/tools/registry.py", "ToolRegistry._emit_tool_call", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/triggers/engine.py", "_emit", 0, "publish", "event-bus"),
    ExcludedSite("nomorals/wisdom/practice.py", "_emit", 0, "publish", "event-bus"),
)


def _inventory_keys() -> set[tuple[str, str, int, str]]:
    return ({s.key for s in PRODUCER_SITES}
            | {s.key for s in EXCLUDED_SITES})


def unregistered_sites(root: str | Path) -> list[ScannedSite]:
    """Scanned call sites with no inventory row — the pin's teeth."""
    known = _inventory_keys()
    return [s for s in scan_publish_sites(root) if s.key not in known]


def stale_sites(root: str | Path) -> list[ProducerSite | ExcludedSite]:
    """Inventory rows whose call site no longer exists."""
    found = {s.key for s in scan_publish_sites(root)}
    return [s for s in (*PRODUCER_SITES, *EXCLUDED_SITES)
            if s.key not in found]


def validate_inventory(root: str | Path) -> list[str]:
    """All inventory problems over ``root`` (empty list = the census
    is exact): unregistered sites, stale rows, posture mismatches,
    unknown classes/reasons, one module claiming two classes."""
    problems: list[str] = []
    scanned_list = scan_publish_sites(root)
    scanned = {s.key: s for s in scanned_list}
    known = _inventory_keys()
    for site in scanned_list:
        if site.key not in known:
            problems.append(
                f"unregistered publish site: {site.path} "
                f"{site.qualname} #{site.index} .{site.method}")
    for entry in (*PRODUCER_SITES, *EXCLUDED_SITES):
        if entry.key not in scanned:
            problems.append(
                f"stale inventory row: {entry.path} {entry.qualname} "
                f"#{entry.index} .{entry.method}")
    for entry in PRODUCER_SITES:
        site = scanned.get(entry.key)
        if site is None:
            continue
        derived = derive_posture(site)
        if derived != entry.posture:
            problems.append(
                f"posture drift: {entry.path} {entry.qualname} "
                f"#{entry.index} table={entry.posture} ast={derived}")
        if entry.producer_class not in PRODUCER_CLASSES:
            problems.append(
                f"unknown producer class {entry.producer_class!r}: "
                f"{entry.path} {entry.qualname}")
    for entry in EXCLUDED_SITES:
        if entry.reason not in EXCLUSION_REASONS:
            problems.append(
                f"unknown exclusion reason {entry.reason!r}: "
                f"{entry.path} {entry.qualname}")
    by_module: dict[str, str] = {}
    for entry in PRODUCER_SITES:
        module = entry.path.replace("/", ".")[:-3]
        other = by_module.setdefault(module, entry.producer_class)
        if other != entry.producer_class:
            problems.append(
                f"module {module} claims two producer classes: "
                f"{other} and {entry.producer_class}")
    return problems


def _build_module_classes() -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in PRODUCER_SITES:
        module = entry.path.replace("/", ".")[:-3]
        out.setdefault(module, entry.producer_class)
    return out


#: dotted producer module -> its census class (first class wins; the
#: test pins that no module ever claims two).
MODULE_CLASSES: dict[str, str] = _build_module_classes()


def producer_class_for_module(dotted: str) -> str:
    """The census class for a producer module ('' when the module is
    not an inventoried producer — a fact the candidate records)."""
    return MODULE_CLASSES.get(str(dotted or ""), "")


def iter_producer_entries() -> Iterable[ProducerSite]:
    return PRODUCER_SITES


def summarize() -> dict[str, Any]:
    """Census counts, for reports and the coverage tooling."""
    by_class: dict[str, int] = {}
    by_posture: dict[str, int] = {}
    for entry in PRODUCER_SITES:
        by_class[entry.producer_class] = \
            by_class.get(entry.producer_class, 0) + 1
        by_posture[entry.posture] = by_posture.get(entry.posture, 0) + 1
    return {
        "producers": len(PRODUCER_SITES),
        "excluded": len(EXCLUDED_SITES),
        "by_class": by_class,
        "by_posture": by_posture,
    }
