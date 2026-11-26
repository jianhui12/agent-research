"""Parameterized synthetic DAG scenarios; original pilot remains reproducible."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

from .budget_manager import BudgetManager
from .graph import DependencyGraph
from .objective import ObjectiveConfig, TaskTarget
from .pilot import TOPOLOGIES
from .state import DAVState


FIXED_TOPOLOGIES = {
    **TOPOLOGIES,
    # One shared source, independent branches, three required outputs.
    "fanout": ((), (0,), (0,), (0,), (1,), (2,), (3,)),
    # Multiple inputs, shared paths and two required outputs (D6, D7).
    "multi_output": ((), (), (0, 1), (0,), (2, 3), (1, 2), (4, 5), (4,)),
}
SCENARIO_TOPOLOGIES = (*FIXED_TOPOLOGIES, "random_dag")


@dataclass(frozen=True)
class ScenarioConfig:
    intrinsic_error: float = 0.12
    propagation: float = 0.75
    sensitivity: float = 0.90
    specificity: float = 0.90
    repair_success: float = 0.90
    recompute_success: float = 0.95
    residual_error: float = 0.05
    replay_intrinsic: float | None = None
    verification_budget: float = 3
    recovery_budget: float = 4
    verify_cost_low: float = 1
    verify_cost_high: float = 2
    repair_cost: float = 1
    recompute_cost: float = 1
    error_loss: float = 20
    blocked_loss: float = 30
    lambda_v: float = 0.25
    lambda_r: float = 0.25
    recovery_mode: str = "acquire"
    # In derived mode, cheaper shared-source check vs expensive target check.
    target_verify_multiplier: float = 1
    random_nodes: int = 12
    random_fan_in: int = 3
    random_targets: int = 3

    def __post_init__(self):
        for name in ("intrinsic_error", "propagation", "sensitivity", "specificity",
                     "repair_success", "recompute_success", "residual_error", "replay_intrinsic"):
            x = getattr(self, name)
            if x is None and name == "replay_intrinsic":
                continue
            if not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("verification_budget", "recovery_budget", "repair_cost", "recompute_cost",
                     "error_loss", "blocked_loss", "lambda_v", "lambda_r",
                     "verify_cost_low", "verify_cost_high", "target_verify_multiplier"):
            x = getattr(self, name)
            if not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.verify_cost_low, self.verify_cost_high, self.target_verify_multiplier) <= 0:
            raise ValueError("Verification costs and target multiplier must be positive")
        if self.recovery_mode not in ("acquire", "derived"):
            raise ValueError("recovery_mode must be acquire or derived")
        if any(type(v) is not int for v in
               (self.random_nodes, self.random_fan_in, self.random_targets)):
            raise ValueError("Random DAG dimensions must be integers")
        if not (2 <= self.random_nodes and 1 <= self.random_targets < self.random_nodes
                and 1 <= self.random_fan_in < self.random_nodes):
            raise ValueError("Invalid random DAG dimensions")


PROFILES = {
    "legacy": ScenarioConfig(),
    "derived": ScenarioConfig(recovery_mode="derived"),
    "shared_source": ScenarioConfig(recovery_mode="derived", propagation=0.95,
                                    target_verify_multiplier=2,
                                    verification_budget=5, recovery_budget=10),
    "high_error": ScenarioConfig(intrinsic_error=0.30, recovery_mode="derived"),
    "weak_verifier": ScenarioConfig(sensitivity=0.70, specificity=0.75,
                                    recovery_mode="derived"),
    "low_propagation": ScenarioConfig(propagation=0.25, recovery_mode="derived"),
}


def random_parents(config: ScenarioConfig, seed: int):
    """Topologically ordered DAG, with designated output suffix as sinks.

    Every non-root has >=1 parent; all nodes can reach a required output.
    The final output suffix never serves as a parent. Parent fan-in is capped.
    A deterministic backbone over internal nodes ensures no orphaned work.
    """
    rng = random.Random(seed)
    n = config.random_nodes
    split = n - config.random_targets
    parents = [()]
    for i in range(1, n):
        limit = min(i, split)
        required = i - 1 if i < split else split - 1
        pool = [j for j in range(limit) if j != required]
        k = rng.randint(0, min(len(pool), config.random_fan_in - 1))
        parents.append(tuple(sorted([required, *rng.sample(pool, k)])))
    return tuple(parents)


def make_scenario(topology: str, config: ScenarioConfig, *, graph_seed: int = 0):
    if topology == "random_dag":
        parents = random_parents(config, graph_seed)
    elif topology in FIXED_TOPOLOGIES:
        parents = FIXED_TOPOLOGIES[topology]
    else:
        raise ValueError(f"Unknown topology: {topology}")
    internal = {j for sources in parents for j in sources}
    targets = tuple(i for i in range(len(parents)) if i not in internal)
    graph = DependencyGraph()
    for i, sources in enumerate(parents):
        node = f"D{i}"
        graph.add_logical_node(node)
        cost = config.verify_cost_low if i % 2 == 0 else config.verify_cost_high
        if i in targets:
            cost *= config.target_verify_multiplier
        graph.add_version(
            node, intrinsic_error_prob=config.intrinsic_error,
            verifier_sensitivity=config.sensitivity,
            verifier_specificity=config.specificity,
            repair_success_prob=config.repair_success,
            recompute_success_prob=config.recompute_success,
            repair_residual_error_prob=config.residual_error,
            recompute_intrinsic_error_prob=config.replay_intrinsic,
            verify_cost=cost, repair_cost=config.repair_cost,
            recompute_cost=config.recompute_cost,
            metadata={"repair_mode": "replay"}
                     if sources and config.recovery_mode == "derived" else {},
        )
    for i, sources in enumerate(parents):
        if sources:
            inputs = [f"D{j}@v1" for j in sources]
            output = f"D{i}@v1"
            graph.record_execution(
                f"init-{i}", "compute", inputs, [output],
                edge_metadata={(v, output): {"propagation_prob": config.propagation}
                               for v in inputs})
    graph.validate_graph()
    objective = ObjectiveConfig(
        tuple(TaskTarget(f"D{i}", config.error_loss, config.blocked_loss) for i in targets),
        lambda_v=config.lambda_v, lambda_r=config.lambda_r,
    )
    return DAVState(graph, BudgetManager(config.verification_budget, config.recovery_budget)), objective
