"""Planning substrate: world graph, routing pipeline, estimates."""

from .graph import (
    WorldGraph,
    GraphNode,
    GraphEdge,
    control_graph,
    disruption_alerts,
)
from .route import (
    Stop,
    Leg,
    RouteResult,
    CostModel,
    GeoIndex,
    RouteSolver,
    RoutePlanner,
    control_route,
    cell_for,
    h3_available,
)
from .estimates import (
    Estimate,
    Segment,
    EstimateStore,
    MonteCarloResult,
    estimate,
    pert,
    pert_sample,
    pert_estimate,
    aggregate,
    monte_carlo,
    reference_class_check,
    calibration,
    record_actual,
    control_eta,
    eta_text,
)
from .congestion import (
    Resource,
    Hold,
    ContentionMonitor,
    hold,
    route_around,
    pre_fanout_check,
    format_status,
    control_congestion,
)

__all__ = [
    "WorldGraph",
    "GraphNode",
    "GraphEdge",
    "control_graph",
    "disruption_alerts",
    "Stop",
    "Leg",
    "RouteResult",
    "CostModel",
    "GeoIndex",
    "RouteSolver",
    "RoutePlanner",
    "control_route",
    "cell_for",
    "h3_available",
    "Estimate",
    "Segment",
    "EstimateStore",
    "MonteCarloResult",
    "estimate",
    "pert",
    "pert_sample",
    "pert_estimate",
    "aggregate",
    "monte_carlo",
    "reference_class_check",
    "calibration",
    "record_actual",
    "control_eta",
    "eta_text",
    "Resource",
    "Hold",
    "ContentionMonitor",
    "hold",
    "route_around",
    "pre_fanout_check",
    "format_status",
    "control_congestion",
]

from .graph_missions import (
    PlanNode,
    PlanEdge,
    PlanGraph,
    CompiledPlan,
    ExecutionResult,
    plan_authoring,
    compiled,
    execute_graph,
)

__all__ += [
    "PlanNode",
    "PlanEdge",
    "PlanGraph",
    "CompiledPlan",
    "ExecutionResult",
    "plan_authoring",
    "compiled",
    "execute_graph",
]

# Tier 3: planning -> commerce logistics (injected quotes; no commerce import).
from .commerce_logistics import (
    LogisticsDescriptor,
    LogisticsPlanStep,
    attach_logistics_step,
    estimate_cost,
    estimate_eta,
    plan_logistics_step,
)

__all__ += [
    "LogisticsDescriptor",
    "LogisticsPlanStep",
    "attach_logistics_step",
    "estimate_cost",
    "estimate_eta",
    "plan_logistics_step",
]
