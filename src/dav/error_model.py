"""Independent-edge marginal error propagation on artifact-version DAGs."""
from .graph import DependencyGraph


def propagate_error_beliefs(graph: DependencyGraph) -> dict[str, float]:
    """Compute version-level marginal risks under independent parent events.

    For each edge (parent -> child), the edge's propagation_prob is used.
    This closed-form product assumes independent parent error/propagation
    events; it is not valid for arbitrary correlated sources or shared causes.
    """
    beliefs: dict[str, float] = {}
    for version_id in graph.topological_sort():
        version = graph.versions[version_id]
        no_error = 1.0 - version.intrinsic_error_prob
        for parent_id in graph.graph.predecessors(version_id):
            edge_prob = graph.graph.edges[parent_id, version_id]["propagation_prob"]
            no_error *= 1.0 - beliefs[parent_id] * edge_prob
        beliefs[version_id] = 1.0 - no_error
    return beliefs
