"""Joint posterior tests: correlation, evidence, rare events, reproducibility."""
import math

import pytest

from dav.error_model import infer_error_risks
from dav.graph import DependencyGraph
from dav.joint_sampler import PosteriorSamplingError, sample_joint_posterior
from dav.models import NodeStatus


def diamond(p=0.4):
    graph = DependencyGraph()
    for node in ("A", "B", "C", "D"):
        graph.add_logical_node(node)
        graph.add_version(node, intrinsic_error_prob=p if node == "A" else 0)
    graph.record_execution("b", "compute", ["A@v1"], ["B@v1"])
    graph.record_execution("c", "compute", ["A@v1"], ["C@v1"])
    graph.record_execution("d", "compute", ["B@v1", "C@v1"], ["D@v1"])
    return graph


def test_exact_particles_keep_shared_ancestor_joint_structure():
    posterior = sample_joint_posterior(diamond(), method="exact")
    assert len(posterior.particles) == 2
    assert sum(p.weight for p in posterior.particles) == pytest.approx(1)
    assert posterior.marginal("D@v1") == pytest.approx(0.4)
    assert all(len(set(p.errors.values())) == 1 for p in posterior.particles)
    assert all(len(set(world.values())) == 1 for world in posterior.draw(100))


def test_exact_matches_existing_primitive_enumeration_with_stochastic_edges():
    graph = diamond()
    graph.graph.edges["A@v1", "B@v1"]["propagation_prob"] = 0.7
    graph.graph.edges["C@v1", "D@v1"]["propagation_prob"] = 0.3
    graph.versions["C@v1"].intrinsic_error_prob = 0.2
    d = graph.versions["D@v1"]
    d.verifier_sensitivity = 0.85
    d.verifier_specificity = 0.8
    obs = {"D@v1": "fail"}
    reference = infer_error_risks(graph, observations=obs)
    posterior = sample_joint_posterior(graph, obs, method="exact")
    for version_id, probability in reference.probabilities.items():
        assert posterior.marginal(version_id) == pytest.approx(probability)
    assert math.exp(posterior.log_evidence_probability) == pytest.approx(
        reference.evidence_probability)


def test_perfect_descendant_evidence_conditions_entire_world():
    graph = diamond()
    posterior = sample_joint_posterior(graph, {"D@v1": "fail"})
    assert len(posterior.particles) == 1
    assert all(posterior.particles[0].errors.values())
    assert math.exp(posterior.log_evidence_probability) == pytest.approx(0.4)


def test_mc_posterior_matches_exact_and_is_seed_reproducible():
    graph = diamond()
    graph.versions["D@v1"].verifier_sensitivity = 0.9
    graph.versions["D@v1"].verifier_specificity = 0.8
    obs = {"D@v1": "fail"}
    exact = sample_joint_posterior(graph, obs, method="exact")
    a = sample_joint_posterior(graph, obs, method="mc", samples=20000, seed=43)
    b = sample_joint_posterior(graph, obs, method="mc", samples=20000, seed=43)
    assert a.marginal("A@v1") == pytest.approx(exact.marginal("A@v1"), abs=0.02)
    assert a.draw(100, seed=7) == b.draw(100, seed=7)
    assert a.effective_sample_size > 1000
    assert a.proposal_count == 20000


def test_historical_evidence_still_informs_shared_ancestor():
    graph = diamond()
    graph.add_version("B", intrinsic_error_prob=0.15)
    graph.record_execution("repair-b", "repair", [], ["B@v2"])
    graph.supersede_version("B@v1", "B@v2")
    assert graph.versions["B@v1"].status == NodeStatus.INVALID
    posterior = sample_joint_posterior(graph, {"B@v1": "fail"})
    assert posterior.marginal("A@v1") == pytest.approx(1)
    assert posterior.marginal("C@v1") == pytest.approx(1)
    assert posterior.marginal("B@v2") == pytest.approx(0.15)


def test_exact_impossible_evidence_and_mc_degeneracy_are_distinct():
    with pytest.raises(PosteriorSamplingError, match="zero probability"):
        sample_joint_posterior(diamond(0), {"D@v1": "fail"}, method="exact")
    with pytest.raises(PosteriorSamplingError, match="no nonzero-weight"):
        sample_joint_posterior(diamond(0), {"D@v1": "fail"}, method="mc",
                               samples=100)
    with pytest.raises(PosteriorSamplingError, match="effective sample size"):
        sample_joint_posterior(diamond(), method="mc", samples=10,
                               min_effective_samples=20)


def test_auto_limit_switches_to_mc_and_exact_limit_is_enforced():
    graph = diamond()
    assert sample_joint_posterior(graph, max_exact_events=0,
                                  samples=100).method == "mc"
    with pytest.raises(ValueError, match="limit"):
        sample_joint_posterior(graph, method="exact", max_exact_events=0)


def test_worlds_are_independent_copies_and_particle_mapping_is_read_only():
    posterior = sample_joint_posterior(diamond())
    with pytest.raises(TypeError):
        posterior.particles[0].errors["A@v1"] = True
    worlds = posterior.draw(2, seed=1)
    worlds[0]["A@v1"] = not worlds[0]["A@v1"]
    assert posterior.draw(2, seed=1)[1] == worlds[1]


def test_log_weights_handle_rare_evidence_without_underflow():
    graph = DependencyGraph()
    observations = {}
    for i in range(30):
        node = f"N{i}"
        graph.add_logical_node(node)
        graph.add_version(node, verifier_specificity=1e-20)
        observations[node + "@v1"] = "pass"
    posterior = sample_joint_posterior(graph, observations, method="exact")
    assert posterior.log_evidence_probability == pytest.approx(30 * math.log(1e-20))
    assert len(posterior.particles) == 1
    assert posterior.particles[0].weight == 1
