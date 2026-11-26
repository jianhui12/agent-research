"""Closed-loop synthetic DAV evaluation with a hidden, persistent error world.

The simulator owns environment truth. A policy receives only a copy of
observable DAVState. A single episode begins with an already built graph;
online task construction and real external tools are outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .joint_sampler import sample_joint_posterior, sample_prior_world
from .latent_state import ScenarioNoise
from .objective import ObjectiveConfig, ObjectiveResult, evaluate_realized
from .policy import PolicyDecision, VerificationPolicy
from .recovery_engine import RecoveryOutcome
from .state import BudgetSnapshot, DAVState
from .transition import execute_verification


@dataclass(frozen=True)
class EpisodeStep:
    index: int
    decision: PolicyDecision
    observation: str
    recovery: RecoveryOutcome | None
    budget_before: BudgetSnapshot
    budget_after: BudgetSnapshot


@dataclass(frozen=True)
class TargetOutcome:
    logical_node_id: str
    version_id: str | None
    completed: bool
    correct: bool | None


@dataclass(frozen=True)
class EpisodeResult:
    objective: ObjectiveResult
    targets: tuple[TargetOutcome, ...]
    steps: tuple[EpisodeStep, ...]
    stop_reason: str
    stop_decision: PolicyDecision | None
    final_state: DAVState
    initial_truth: dict[str, bool]
    final_truth: dict[str, bool]

    @property
    def success(self) -> bool:
        return all(target.completed and target.correct is True for target in self.targets)

    @property
    def completion_rate(self) -> float:
        return sum(target.completed for target in self.targets) / len(self.targets)

    @property
    def correct_output_rate(self) -> float:
        # Denominator is all required outputs, not just completed ones.
        return sum(target.correct is True for target in self.targets) / len(self.targets)


class SyntheticSimulator:
    """Run an adaptive policy on one persistent synthetic latent world.

    Use the same environment seed with different policies for paired episodes.
    Policy randomness is configured separately by the caller.
    """

    def __init__(self, config: ObjectiveConfig, *, max_steps: int = 100,
                 initial_posterior_method: str = "auto",
                 initial_posterior_samples: int = 5000,
                 max_exact_events: int = 14, min_effective_samples: float = 20.0) -> None:
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer")
        self.config = config
        self.max_steps = max_steps
        self.initial_posterior_method = initial_posterior_method
        self.initial_posterior_samples = initial_posterior_samples
        self.max_exact_events = max_exact_events
        self.min_effective_samples = min_effective_samples

    def run(self, initial_state: DAVState, policy: VerificationPolicy, *,
            seed: int = 0) -> EpisodeResult:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        state = initial_state.fork()
        state.graph.validate_graph()
        for target in self.config.targets:
            state.current_version_id(target.logical_node_id)
        checkpoint = state.checkpoint()
        rng = random.Random(seed)
        truth_seed, draw_seed, noise_seed = (rng.getrandbits(64) for _ in range(3))
        if state.observations:
            # Resume an observational state by drawing a conditional world;
            # this is approximate if the initial posterior uses MC.
            posterior = sample_joint_posterior(
                state.graph, state.observations,
                method=self.initial_posterior_method,
                samples=self.initial_posterior_samples, seed=truth_seed,
                max_exact_events=self.max_exact_events,
                min_effective_samples=self.min_effective_samples,
            )
            truth = posterior.draw(1, seed=draw_seed)[0]
        else:
            truth = sample_prior_world(state.graph, seed=truth_seed)
        initial_truth = dict(truth)
        noise = ScenarioNoise(noise_seed)
        steps = []
        stop_reason = "max_steps"
        stop_decision = None
        for index in range(self.max_steps):
            # Never pass latent labels or environment RNG to the policy.
            # Forking also prevents accidental mutations of the environment.
            decision = policy.decide(state.fork())
            if decision.version_id is None:
                stop_reason = decision.reason
                stop_decision = decision
                break
            before = state.checkpoint()
            transition = execute_verification(state, decision.version_id, truth, noise)
            truth = transition.truth
            steps.append(EpisodeStep(
                index=index, decision=decision, observation=transition.observation,
                recovery=transition.recovery, budget_before=before,
                budget_after=state.checkpoint(),
            ))
        objective = evaluate_realized(state, self.config, truth, checkpoint=checkpoint)
        targets = []
        for target in self.config.targets:
            v = state.current_version_id(target.logical_node_id)
            completed = v is not None and state.is_usable(v)
            targets.append(TargetOutcome(target.logical_node_id, v, completed,
                                         not truth[v] if completed else None))
        state.graph.validate_graph()
        return EpisodeResult(objective, tuple(targets), tuple(steps), stop_reason,
                             stop_decision, state, initial_truth, dict(truth))
