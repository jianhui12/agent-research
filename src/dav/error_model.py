"""Independent-edge marginal error propagation prototype."""
from .graph import DependencyGraph


def propagate_error_beliefs(graph: DependencyGraph) -> dict[str, float]:
    """Compute marginals under independent parent error/edge events.

    Not valid for arbitrary correlated sources or shared latent causes.
    """
    beliefs: dict[str, float] = {}
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        no_error = 1.0 - node.intrinsic_error_prob
        for parent_id in graph.graph.predecessors(node_id):
            parent = graph.nodes[parent_id]
            no_error *= 1.0 - beliefs[parent_id] * parent.propagation_prob
        beliefs[node_id] = 1.0 - no_error
    return beliefs
