import pytest

from dav.graph import DependencyGraph
from dav.models import NodeStatus


def build_diamond():
    graph = DependencyGraph()
    for node_id in ("D1", "D2", "D3", "D4"):
        graph.add_logical_node(node_id)
    for node_id in ("D1", "D2", "D3", "D4"):
        graph.add_version(node_id, value=f"{node_id}-v1")
    graph.record_execution(
        "exec-d2", "compute-d2", ["D1@v1"], ["D2@v1"],
        edge_metadata={("D1@v1", "D2@v1"): {"propagation_prob": 0.7, "input_role": "inventory"}},
    )
    graph.record_execution("exec-d3", "compute-d3", ["D1@v1"], ["D3@v1"])
    graph.record_execution(
        "exec-d4", "compute-d4", ["D2@v1", "D3@v1"], ["D4@v1"]
    )
    return graph


def test_shared_descendant_is_returned_once_and_topologically():
    graph = build_diamond()
    affected = graph.get_affected_nodes("D1@v1")
    assert affected == {"D2@v1", "D3@v1", "D4@v1"}
    order = graph.topological_sort(affected)
    assert order.index("D2@v1") < order.index("D4@v1")
    assert order.index("D3@v1") < order.index("D4@v1")


def test_superseding_old_version_invalidates_only_its_descendants():
    graph = build_diamond()
    graph.add_version("D1", value="corrected", status=NodeStatus.VERIFIED)
    affected = graph.supersede_version("D1@v1", "D1@v2")
    assert affected == {"D2@v1", "D3@v1", "D4@v1"}
    assert graph.get_current_version("D1").version_id == "D1@v2"
    assert graph.versions["D1@v1"].status == NodeStatus.INVALID
    assert graph.versions["D2@v1"].status == NodeStatus.STALE
    assert graph.versions["D1@v2"].status == NodeStatus.VERIFIED


def test_cycle_is_rejected_without_corrupting_graph():
    graph = DependencyGraph()
    graph.add_logical_node("A")
    graph.add_logical_node("B")
    graph.add_version("A")
    graph.add_version("B")
    graph.add_dependency("A@v1", "B@v1")
    with pytest.raises(ValueError):
        graph.add_dependency("B@v1", "A@v1")
    assert graph.graph.has_edge("A@v1", "B@v1")
    assert not graph.graph.has_edge("B@v1", "A@v1")


def test_execution_lineage_is_recorded_and_validated():
    graph = build_diamond()
    assert graph.versions["D4@v1"].execution_id == "exec-d4"
    assert graph.check_lineage() == []


def test_unknown_versions_are_rejected():
    graph = DependencyGraph()
    with pytest.raises(KeyError):
        graph.get_affected_nodes("missing@v1")
