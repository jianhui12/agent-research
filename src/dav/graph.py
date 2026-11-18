"""Dependency DAG and conservative invalidation skeleton."""
import networkx as nx
from .models import Node, NodeStatus


class DependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)

    def add_dependency(self, parent_id: str, child_id: str) -> None:
        self.graph.add_edge(parent_id, child_id)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent_id, child_id)
            raise ValueError("Dependency graph must be a DAG")

    def topological_order(self, subset=None) -> list[str]:
        graph = self.graph
        if subset is not None:
            graph = self.graph.subgraph(set(subset)).copy()
        return list(nx.topological_sort(graph))

    def descendants(self, node_id: str) -> set[str]:
        return set(nx.descendants(self.graph, node_id))

    def invalidate_descendants(self, node_id: str) -> set[str]:
        """TODO: filter descendants by recorded parent-version lineage."""
        affected = self.descendants(node_id)
        for child_id in affected:
            if self.nodes[child_id].status != NodeStatus.BLOCKED:
                self.nodes[child_id].status = NodeStatus.STALE
        return affected
