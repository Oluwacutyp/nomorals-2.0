"""Agent lifecycle and budgets.

An :class:`Agent` is a capability-scoped unit of work with a budget. Budgets are
not a nicety: an autonomous system that spawns sub-agents recursively will
eventually spawn a loop, and the budget is what turns an unbounded token bill into
a bounded one with a clear error.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import BudgetExceeded, DeadlineExceeded
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import CapabilitySet

__all__ = ["Agent", "AgentResult", "AgentState"]

_log = get_logger(__name__)


class AgentState:
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Budget:
    """Resource ceilings for one agent, inherited downward by intersection."""

    wall_seconds: float = 3600.0
    tokens: int = 2_000_000
    children: int = 256
    cost_usd: float = 0.0  # 0 = unlimited
    started_at: float = field(default_factory=time.monotonic)

    spent_wall: float = 0.0
    spent_tokens: int = 0
    spawned_children: int = 0
    spent_cost: float = 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_wall(self) -> float:
        return max(0.0, self.wall_seconds - self.elapsed())

    @property
    def deadline(self) -> float:
        return self.started_at + self.wall_seconds

    def charge_tokens(self, count: int) -> None:
        self.spent_tokens += max(0, int(count))

    def charge_cost(self, amount: float) -> None:
        self.spent_cost += max(0.0, float(amount))

    def charge_child(self) -> None:
        self.spawned_children += 1

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if any ceiling has been passed."""
        if self.elapsed() > self.wall_seconds:
            raise DeadlineExceeded(
                f"wall budget {self.wall_seconds:.0f}s exceeded ({self.elapsed():.0f}s)",
                kind="wall",
            )
        if self.tokens and self.spent_tokens > self.tokens:
            raise BudgetExceeded(
                f"token budget {self.tokens} exceeded ({self.spent_tokens})", kind="tokens"
            )
        if self.children and self.spawned_children > self.children:
            raise BudgetExceeded(
                f"child budget {self.children} exceeded ({self.spawned_children})", kind="children"
            )
        if self.cost_usd and self.spent_cost > self.cost_usd:
            raise BudgetExceeded(
                f"cost budget ${self.cost_usd:.2f} exceeded (${self.spent_cost:.2f})", kind="cost"
            )

    def child_budget(self, *, fraction: float = 0.5) -> "Budget":
        """Derive a sub-budget for a child.

        Halving by default means a depth-N spawn chain consumes at most the
        parent's budget, no matter how deep it goes — the geometric series bounds
        it. That is what makes unbounded recursive spawning safe.
        """
        fraction = min(1.0, max(0.01, fraction))
        return Budget(
            wall_seconds=max(1.0, self.remaining_wall() * fraction),
            tokens=max(1, int(max(0, self.tokens - self.spent_tokens) * fraction)) if self.tokens else 0,
            children=max(1, int(self.children * fraction)) if self.children else 0,
            cost_usd=self.cost_usd * fraction if self.cost_usd else 0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_seconds": round(self.wall_seconds, 2),
            "elapsed": round(self.elapsed(), 2),
            "tokens": self.tokens,
            "spent_tokens": self.spent_tokens,
            "children": self.children,
            "spawned_children": self.spawned_children,
            "cost_usd": self.cost_usd,
            "spent_cost": round(self.spent_cost, 4),
        }


@dataclass
class AgentResult:
    """What an agent produced."""

    agent_id: str
    role: str
    output: Any = None
    ok: bool = True
    error: str = ""
    tokens: int = 0
    seconds: float = 0.0
    children: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "ok": self.ok,
            "error": self.error,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "children": self.children,
            "metadata": self.metadata,
        }


class Agent(abc.ABC):
    """Base class for every specialized agent.

    Subclasses implement :meth:`work`. The base class owns identity, capability
    scoping, budget enforcement, cancellation, and result accounting, so a role
    author writes only the domain logic.
    """

    role: str = "generic"
    #: Capabilities this role needs; the grant it receives is the intersection of
    #: this and its parent's grant. Privilege narrows down the tree.
    required_capabilities: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        name: str = "",
        context: Any = None,
        budget: Budget | None = None,
        parent: "Agent | None" = None,
    ) -> None:
        self.id = new_id()
        self.name = name or f"{self.role}-{self.id[-6:]}"
        self.context = context
        self.budget = budget or Budget()
        self.parent = parent
        self.state = AgentState.IDLE
        self.children: list["Agent"] = []
        self.created_at = time.time()
        self.cancel_event = threading.Event()
        self.result: AgentResult | None = None

        parent_grant = parent.capabilities if parent is not None else None
        own = CapabilitySet.of(*self.required_capabilities) if self.required_capabilities else CapabilitySet.all()
        self.capabilities = own if parent_grant is None else own.intersect(parent_grant)

    # ── subclass API ─────────────────────────────────────────────────────────
    @abc.abstractmethod
    def work(self, task_input: Any) -> Any:
        """Do the work. Raise on failure; return anything serializable on success."""

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self, task_input: Any = None) -> AgentResult:
        """Execute with budget enforcement, cancellation, and accounting."""
        started = time.perf_counter()
        self.state = AgentState.RUNNING
        _log.debug("agent %s (%s) started", self.name, self.role)
        try:
            self.budget.check()
            output = self.work(task_input)
            self.state = AgentState.DONE
            self.result = AgentResult(
                agent_id=self.id,
                role=self.role,
                output=output,
                ok=True,
                seconds=time.perf_counter() - started,
                tokens=self.budget.spent_tokens,
                children=len(self.children),
            )
        except (BudgetExceeded, DeadlineExceeded) as exc:
            self.state = AgentState.CANCELLED
            _log.warning("agent %s stopped by budget: %s", self.name, exc.message)
            self.result = AgentResult(
                agent_id=self.id, role=self.role, ok=False, error=exc.message,
                seconds=time.perf_counter() - started,
                # The supervisor decides whether to restart based on the failure
                # *kind*. Substring-matching the human-readable message misses
                # BudgetExceeded("out of tokens") entirely, so carry the type.
                metadata={"error_type": type(exc).__name__},
            )
        except Exception as exc:  # noqa: BLE001 - agent failures are results
            self.state = AgentState.FAILED
            _log.debug("agent %s failed: %s", self.name, exc)
            self.result = AgentResult(
                agent_id=self.id, role=self.role, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - started,
            )
        return self.result

    def __call__(self, task_input: Any = None) -> AgentResult:
        return self.run(task_input)

    # ── control ──────────────────────────────────────────────────────────────
    def cancel(self, reason: str = "cancelled") -> None:
        self.cancel_event.set()
        for child in self.children:
            child.cancel(reason)
        if self.state is AgentState.RUNNING or self.state is AgentState.IDLE:
            self.state = AgentState.CANCELLED

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        """Cooperative cancellation checkpoint for long-running work loops."""
        if self.cancel_event.is_set():
            from ..core.errors import TaskCancelled

            raise TaskCancelled(f"agent {self.name} cancelled")
        self.budget.check()

    # ── spawning ─────────────────────────────────────────────────────────────
    def spawn(self, agent: "Agent", *, budget_fraction: float = 0.5) -> "Agent":
        """Register a sub-agent, narrowing its budget and capabilities."""
        self.budget.charge_child()
        self.budget.check()
        agent.parent = self
        agent.capabilities = agent.capabilities.intersect(self.capabilities)
        agent.budget = self.budget.child_budget(fraction=budget_fraction)
        self.children.append(agent)
        return agent

    def lineage(self) -> list[str]:
        chain: list[str] = []
        node: Agent | None = self
        while node is not None:
            chain.append(f"{node.role}:{node.name}")
            node = node.parent
        return list(reversed(chain))

    def depth(self) -> int:
        return len(self.lineage()) - 1

    def descendants(self) -> list["Agent"]:
        out: list[Agent] = []
        for child in self.children:
            out.append(child)
            out.extend(child.descendants())
        return out

    # ── reporting ────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "capabilities": self.capabilities.as_list(),
            "budget": self.budget.to_dict(),
            "children": len(self.children),
            "lineage": self.lineage(),
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} {self.name} {self.state}>"
