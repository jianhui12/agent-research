"""Reproducible paired pilot on small synthetic DAGs, not a full benchmark.

Run: python -m dav.pilot --episodes 10 --rollouts 64 --output results/pilot.csv
All policy families receive identical initial states and environment seeds.
No failed episodes are silently excluded.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
from statistics import fmean
from time import perf_counter

from .baselines import HEURISTICS, HeuristicPolicy
from .budget_manager import BudgetManager
from .cost_estimator import MonteCarloCostEstimator
from .graph import DependencyGraph
from .objective import ObjectiveConfig, TaskTarget
from .policy import AMCRPolicy, StopPolicy
from .simulator import SyntheticSimulator
from .state import DAVState


TOPOLOGIES = {
    "chain": ((), (0,), (1,), (2,), (3,), (4,)),
    "diamond": ((), (0,), (0,), (1, 2)),
    "layered": ((), (), (0, 1), (0,), (1, 2), (3, 4), (2, 5)),
}
POLICIES = ("no_verification", *HEURISTICS, "amcr")


def make_pilot_state(topology: str, verification_budget: float,
                     recovery_budget: float) -> tuple[DAVState, ObjectiveConfig]:
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}")
    parents = TOPOLOGIES[topology]
    graph = DependencyGraph()
    for i in range(len(parents)):
        node = f"D{i}"
        graph.add_logical_node(node)
        graph.add_version(
            node, intrinsic_error_prob=0.12, verifier_sensitivity=0.90,
            verifier_specificity=0.90, repair_success_prob=0.90,
            recompute_success_prob=0.95, repair_residual_error_prob=0.05,
            verify_cost=1.0 if i % 2 == 0 else 2.0,
            repair_cost=1.0, recompute_cost=1.0)
    for i, sources in enumerate(parents):
        if not sources:
            continue
        inputs = [f"D{j}@v1" for j in sources]
        output = f"D{i}@v1"
        graph.record_execution(
            f"init-{i}", "compute", inputs, [output],
            edge_metadata={(v, output): {"propagation_prob": 0.75} for v in inputs})
    # Explicit required outputs: the terminal nodes only.
    internal = {i for sources in parents for i in sources}
    targets = tuple(TaskTarget(f"D{i}", 20.0, 30.0)
                    for i in range(len(parents)) if i not in internal)
    graph.validate_graph()
    config = ObjectiveConfig(targets, lambda_v=0.25, lambda_r=0.25)
    return DAVState(graph, BudgetManager(verification_budget, recovery_budget)), config


def make_policy(name, config, *, seed, rollouts, posterior_samples):
    if name == "no_verification":
        return StopPolicy()
    if name == "amcr":
        return AMCRPolicy(MonteCarloCostEstimator(
            config, seed=seed, rollouts=rollouts, posterior_samples=posterior_samples,
        ))
    return HeuristicPolicy(name, seed=seed, posterior_samples=posterior_samples)


def run_pilot(*, topologies=("chain", "diamond"), policies=POLICIES,
              episodes=10, rollouts=64, posterior_samples=5000,
              verification_budget=3.0, recovery_budget=4.0, seed=0):
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be positive")
    if type(rollouts) is not int or rollouts < 2:
        raise ValueError("rollouts must be at least two")
    if type(posterior_samples) is not int or posterior_samples < 20:
        raise ValueError("posterior_samples must be at least 20")
    if len(set(topologies)) != len(topologies) or len(set(policies)) != len(policies):
        raise ValueError("Duplicate topologies or policies")
    if not topologies or not policies or any(p not in POLICIES for p in policies):
        raise ValueError("At least one topology and known policy are required")
    rng = random.Random(seed)
    # Draw environment and planner seeds independently before any policy runs.
    seeds = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(episodes)]
    rows = []
    for topology in topologies:
        state, config = make_pilot_state(topology, verification_budget, recovery_budget)
        simulator = SyntheticSimulator(config, initial_posterior_samples=posterior_samples)
        for episode, (environment_seed, planner_seed) in enumerate(seeds):
            expected_world = None
            for name in policies:
                policy = make_policy(name, config, seed=planner_seed, rollouts=rollouts,
                                     posterior_samples=posterior_samples)
                start = perf_counter()
                try:
                    result = simulator.run(state, policy, seed=environment_seed)
                except Exception as exc:
                    raise RuntimeError(
                        f"Pilot failed: topology={topology}, episode={episode}, "
                        f"policy={name}, environment_seed={environment_seed}") from exc
                elapsed = perf_counter() - start
                if expected_world is None:
                    expected_world = result.initial_truth
                elif expected_world != result.initial_truth:
                    raise AssertionError("Initial latent worlds differ across policies")
                o = result.objective
                rows.append({
                    "topology": topology, "episode": episode, "policy": name,
                    "environment_seed": environment_seed, "planner_seed": planner_seed,
                    "rollouts": rollouts, "posterior_samples": posterior_samples,
                    "verification_budget": verification_budget,
                    "recovery_budget": recovery_budget,
                    "total_cost": o.total, "error_loss": o.error_loss,
                    "blocked_loss": o.blocked_loss,
                    "verification_cost": o.verification_cost, "recovery_cost": o.recovery_cost,
                    "success": int(result.success), "completion_rate": result.completion_rate,
                    "correct_output_rate": result.correct_output_rate,
                    "steps": len(result.steps), "stop_reason": result.stop_reason,
                    "seconds": elapsed,
                })
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["topology"], row["policy"]), []).append(row)
    return [
        {"topology": topology, "policy": policy, "episodes": len(group),
         **{key: fmean(row[key] for row in group) for key in (
             "total_cost", "error_loss", "blocked_loss", "success", "completion_rate",
             "correct_output_rate", "verification_cost", "recovery_cost", "seconds")}}
        for (topology, policy), group in groups.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", choices=tuple(TOPOLOGIES),
                        default=["chain", "diamond"])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--posterior-samples", type=int, default=5000)
    parser.add_argument("--verification-budget", type=float, default=3)
    parser.add_argument("--recovery-budget", type=float, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/pilot.csv"))
    args = parser.parse_args()
    rows = run_pilot(
        topologies=tuple(args.topologies), policies=tuple(args.policies),
        episodes=args.episodes, rollouts=args.rollouts,
        posterior_samples=args.posterior_samples,
        verification_budget=args.verification_budget,
        recovery_budget=args.recovery_budget, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired episode-policy rows to {args.output}")
    for item in summarize(rows):
        print(f"{item['topology']:8} {item['policy']:24} "
              f"J={item['total_cost']:.3f} success={item['success']:.3f} "
              f"completion={item['completion_rate']:.3f} "
              f"V={item['verification_cost']:.2f} R={item['recovery_cost']:.2f} "
              f"sec={item['seconds']:.3f}")


if __name__ == "__main__":
    main()
