"""Paired parameterized benchmark with post-episode diagnostic traces.

Run python -m dav.benchmark --help. Legacy dav.pilot is unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import random
from statistics import fmean
from time import perf_counter

from .experiment_policies import BENCHMARK_POLICIES, make_benchmark_policy
from .scenarios import PROFILES, SCENARIO_TOPOLOGIES, ScenarioConfig, make_scenario
from .simulator import SyntheticSimulator


def _trace(result, row):
    """Offline only: truth is never passed to make_policy or decide."""
    steps = []
    for s in result.steps:
        recovery = s.recovery
        steps.append({
            "index": s.index,
            "version_id": s.decision.version_id,
            "observation": s.observation,
            "true_error": result.final_truth[s.decision.version_id],
            "budget_before": asdict(s.budget_before),
            "budget_after": asdict(s.budget_after),
            "decision": asdict(s.decision),
            "recovery": None if recovery is None else {
                "source": recovery.source_version_id,
                "replacements": recovery.replacements,
                "affected": sorted(recovery.affected),
                "blocked": sorted(recovery.blocked),
                "events": [asdict(e) for e in recovery.events],
                "spent": recovery.recovery_spent,
            },
        })
    return {
        **{k: row[k] for k in ("scenario_id", "topology", "episode", "policy",
                                "environment_seed", "planner_seed", "graph_seed")},
        "initial_truth": result.initial_truth,
        "final_truth": result.final_truth,
        "steps": steps,
        "stop_reason": result.stop_reason,
        "stop_decision": None if result.stop_decision is None else asdict(result.stop_decision),
        "targets": [asdict(t) for t in result.targets],
        "objective": asdict(result.objective),
    }


def _episode_row(result, state, config):
    targets = {t.logical_node_id for t in config.targets}
    initial_success = all(
        state.is_usable(state.current_version_id(t))
        and not result.initial_truth[state.current_version_id(t)]
        for t in targets
    )
    checks = [(s, result.final_truth[s.decision.version_id]) for s in result.steps]
    events = [e for s in result.steps if s.recovery is not None for e in s.recovery.events]
    o = result.objective
    return {
        "total_cost": o.total, "error_loss": o.error_loss, "blocked_loss": o.blocked_loss,
        "verification_cost": o.verification_cost, "recovery_cost": o.recovery_cost,
        "success": int(result.success), "initial_success": int(initial_success),
        "corrected": int(not initial_success and result.success),
        "degraded": int(initial_success and not result.success),
        "completion_rate": result.completion_rate,
        "correct_output_rate": result.correct_output_rate,
        "steps": len(checks),
        "first_selected": checks[0][0].decision.version_id if checks else "",
        "target_checks": sum(result.final_state.graph.versions[s.decision.version_id].node_id
                             in targets for s, _ in checks),
        "true_positive": sum(e and s.observation == "fail" for s, e in checks),
        "false_positive": sum(not e and s.observation == "fail" for s, e in checks),
        "true_negative": sum(not e and s.observation == "pass" for s, e in checks),
        "false_negative": sum(e and s.observation == "pass" for s, e in checks),
        "recovery_failed_events": sum(e.outcome == "failed" for e in events),
        "recovery_budget_exhausted_events": sum(e.outcome == "budget_exhausted" for e in events),
        "recovery_blocked_input_events": sum(e.outcome == "blocked_input" for e in events),
        "repair_attempts": sum(e.action == "repair" and e.outcome in ("succeeded", "failed")
                               for e in events),
        "recompute_attempts": sum(e.action == "recompute" and e.outcome in ("succeeded", "failed")
                                  for e in events),
        "stop_reason": result.stop_reason,
    }


def run_benchmark(*, scenarios, topologies=("chain", "diamond", "fanout"),
                  policies=BENCHMARK_POLICIES, episodes=10, rollouts=64,
                  posterior_samples=5000, seed=0, trace=False):
    """scenarios: mapping of unique names to immutable ScenarioConfig objects.

    Within each scenario/topology/episode the graph and environment are paired.
    Common seeds across scenarios/topologies also enable paired sensitivity
    analysis; those repeated seeds are NOT independent observations.
    """
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be positive")
    if type(rollouts) is not int or rollouts < 2:
        raise ValueError("rollouts must be at least 2")
    if type(posterior_samples) is not int or posterior_samples < 20:
        raise ValueError("posterior_samples must be at least 20")
    if (not scenarios or not topologies or not policies
            or len(set(topologies)) != len(topologies)
            or len(set(policies)) != len(policies)
            or any(t not in SCENARIO_TOPOLOGIES for t in topologies)
            or any(p not in BENCHMARK_POLICIES for p in policies)
            or any(not isinstance(c, ScenarioConfig) for c in scenarios.values())):
        raise ValueError("Invalid benchmark scenarios, topologies or policies")
    rng = random.Random(seed)
    graph_rng = random.Random(seed ^ 0x444156)
    seeds = [(rng.getrandbits(64), rng.getrandbits(64), graph_rng.getrandbits(64))
             for _ in range(episodes)]
    rows, traces = [], []
    for scenario_id, scenario in scenarios.items():
        parameters = asdict(scenario)
        parameters["target_error_loss"] = parameters.pop("error_loss")
        parameters["target_blocked_loss"] = parameters.pop("blocked_loss")
        for topology in topologies:
            for episode, (environment_seed, planner_seed, graph_seed) in enumerate(seeds):
                state, config = make_scenario(topology, scenario, graph_seed=graph_seed)
                sim = SyntheticSimulator(config, initial_posterior_samples=posterior_samples)
                expected_world = None
                for name in policies:
                    policy = make_benchmark_policy(
                        name, config, seed=planner_seed, rollouts=rollouts,
                        posterior_samples=posterior_samples)
                    start = perf_counter()
                    try:
                        result = sim.run(state, policy, seed=environment_seed)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Benchmark failed: scenario={scenario_id} topology={topology} "
                            f"episode={episode} policy={name} environment_seed={environment_seed} "
                            f"planner_seed={planner_seed} graph_seed={graph_seed}") from exc
                    elapsed = perf_counter() - start
                    if expected_world is None:
                        expected_world = result.initial_truth
                    elif expected_world != result.initial_truth:
                        raise AssertionError("Initial truth differs across paired policies")
                    row = {
                        "scenario_id": scenario_id, "topology": topology,
                        "episode": episode, "policy": name,
                        "environment_seed": environment_seed, "planner_seed": planner_seed,
                        "graph_seed": graph_seed, "rollouts": rollouts,
                        "posterior_samples": posterior_samples,
                        "nodes": len(state.graph.versions),
                        "edges": state.graph.graph.number_of_edges(),
                        "targets": len(config.targets),
                        **parameters, **_episode_row(result, state, config),
                        "seconds": elapsed,
                    }
                    rows.append(row)
                    if trace:
                        item = _trace(result, row)
                        item["initial_edges"] = sorted(state.graph.graph.edges())
                        traces.append(item)
    return rows, traces


def summarize_benchmark(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario_id"], row["topology"], row["policy"]), []).append(row)
    summary = []
    for (scenario, topology, policy), group in groups.items():
        n_wrong = sum(not r["initial_success"] for r in group)
        n_correct = len(group) - n_wrong
        checks = sum(r["steps"] for r in group)
        summary.append({
            "scenario_id": scenario, "topology": topology, "policy": policy,
            "episodes": len(group), "initial_wrong_count": n_wrong,
            "initial_correct_count": n_correct,
            **{key: fmean(r[key] for r in group) for key in (
                "total_cost", "error_loss", "blocked_loss", "success", "completion_rate",
                "correct_output_rate", "verification_cost", "recovery_cost", "seconds", "steps")},
            "correction_rate": sum(r["corrected"] for r in group) / n_wrong if n_wrong else None,
            "degradation_rate": sum(r["degraded"] for r in group) / n_correct if n_correct else None,
            "target_check_fraction": sum(r["target_checks"] for r in group) / checks if checks else None,
            "active_stop_rate": fmean(r["stop_reason"] == "no_positive_gain" for r in group),
            **{key: sum(r[key] for r in group) for key in (
                "true_positive", "false_positive", "true_negative", "false_negative",
                "recovery_failed_events", "recovery_budget_exhausted_events",
                "recovery_blocked_input_events")},
        })
    return summary


def _overrides(items):
    parsed = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or key not in ScenarioConfig.__dataclass_fields__ or key in parsed:
            raise ValueError(f"Invalid or duplicate scenario override: {item}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=["legacy"])
    parser.add_argument("--topologies", nargs="+", choices=SCENARIO_TOPOLOGIES,
                        default=["chain", "diamond", "fanout"])
    parser.add_argument("--policies", nargs="+", choices=BENCHMARK_POLICIES,
                        default=list(BENCHMARK_POLICIES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--posterior-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verification-budgets", nargs="+", type=float)
    parser.add_argument("--recovery-budgets", nargs="+", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--trace", action="store_true", help="Write offline JSONL episode traces")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    try:
        overrides = _overrides(args.set)
        scenarios = {}
        for name in args.profiles:
            base = replace(PROFILES[name], **overrides)
            for v, r in itertools.product(
                args.verification_budgets or [base.verification_budget],
                args.recovery_budgets or [base.recovery_budget]):
                key = f"{name}_v{v:g}_r{r:g}"
                if key in scenarios:
                    raise ValueError(f"Duplicate scenario: {key}")
                scenarios[key] = replace(base, verification_budget=v, recovery_budget=r)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    summary_path = args.output.with_suffix(".summary.csv")
    manifest_path = args.output.with_suffix(".manifest.json")
    trace_path = args.output.with_suffix(".trace.jsonl")
    paths = [args.output, summary_path, manifest_path] + ([trace_path] if args.trace else [])
    if len(set(paths)) != len(paths) or any(p.exists() for p in paths):
        parser.error("Output paths collide or already exist; choose a new --output filename")
    print(f"Running {len(scenarios)*len(args.topologies)*len(args.policies)*args.episodes} episode-policy records",
          flush=True)
    rows, traces = run_benchmark(
        scenarios=scenarios, topologies=tuple(args.topologies), policies=tuple(args.policies),
        episodes=args.episodes, rollouts=args.rollouts, posterior_samples=args.posterior_samples,
        seed=args.seed, trace=args.trace)
    summary = summarize_benchmark(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, records in ((args.output, rows), (summary_path, summary)):
        with path.open("x", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if args.trace:
        with trace_path.open("x", encoding="utf-8") as out:
            for item in traces:
                out.write(json.dumps(item, allow_nan=False) + "\n")
    with manifest_path.open("x", encoding="utf-8") as out:
        json.dump({
            "seed": args.seed, "episodes": args.episodes, "rollouts": args.rollouts,
            "posterior_samples": args.posterior_samples, "topologies": args.topologies,
            "policies": args.policies, "scenarios": {k: asdict(v) for k, v in scenarios.items()},
            "record_count": len(rows), "trace": args.trace,
        }, out, indent=2, allow_nan=False)
    print(f"Wrote {len(rows)} rows to {args.output}")
    for x in summary:
        print(f"{x['scenario_id']:23} {x['topology']:12} {x['policy']:24} "
              f"J={x['total_cost']:.3f} success={x['success']:.3f} "
              f"completion={x['completion_rate']:.3f} "
              f"V={x['verification_cost']:.2f} R={x['recovery_cost']:.2f}")


if __name__ == "__main__":
    main()
