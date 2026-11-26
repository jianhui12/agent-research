
from dav.graph import DependencyGraph
from dav.models import NodeStatus
from dav.recovery import build_recovery_plan


def test_diamond_repair_and_recompute():
    graph = DependencyGraph()

    # 1. 创建逻辑节点
    for name in ["D1", "D2", "D3", "D4"]:
        graph.add_logical_node(name)

    # 2. 创建初始版本
    for name in ["D1", "D2", "D3", "D4"]:
        graph.add_version(name, value=f"{name}-old")

    # 3. 记录原始执行关系
    graph.record_execution(
        "exec-d2", "compute-D2",
        ["D1@v1"], ["D2@v1"]
    )
    graph.record_execution(
        "exec-d3", "compute-D3",
        ["D1@v1"], ["D3@v1"]
    )
    graph.record_execution(
        "exec-d4", "compute-D4",
        ["D2@v1", "D3@v1"], ["D4@v1"]
    )

    # 4. 修复 D1，产生新版本
    graph.add_version(
        "D1",
        value="D1-corrected",
        status=NodeStatus.VERIFIED
    )
    graph.record_execution(
        "repair-D1", "repair",
        [], ["D1@v2"]
    )

    # 5. 生成恢复计划
    plan = build_recovery_plan(
        graph, "D1@v1", "D1@v2"
    )

    assert plan.affected_version_ids == {
        "D2@v1", "D3@v1", "D4@v1"
    }

    # 6. 模拟执行重计算
    recompute_count = []
    new_versions = {
        "D1@v1": "D1@v2"
    }

    for old_id in plan.recomputation_order:
        old_node = graph.versions[old_id]
        node_id = old_node.node_id

        # 找到旧版本的直接输入，并映射为最新版本
        old_inputs = list(graph.graph.predecessors(old_id))
        new_inputs = [
            new_versions.get(parent, parent)
            for parent in old_inputs
        ]

        # 创建新版本
        new_version = graph.add_version(
            node_id,
            value=f"{node_id}-recomputed",
            status=NodeStatus.VERIFIED
        )

        # 记录新执行
        execution_id = f"recompute-{node_id}"
        graph.record_execution(
            execution_id,
            f"recompute-{node_id}",
            new_inputs,
            [new_version.version_id]
        )

        new_versions[old_id] = new_version.version_id
        recompute_count.append(node_id)

    # 7. 检查重计算次数
    assert recompute_count.count("D2") == 1
    assert recompute_count.count("D3") == 1
    assert recompute_count.count("D4") == 1

    # 8. 检查拓扑顺序
    assert recompute_count.index("D2") < recompute_count.index("D4")
    assert recompute_count.index("D3") < recompute_count.index("D4")

    # 9. 检查 D4 是否消费了两个新版本
    d4_new = graph.get_current_version("D4")
    actual_inputs = set(graph.graph.predecessors(d4_new.version_id))

    assert actual_inputs == {
        new_versions["D2@v1"],
        new_versions["D3@v1"]
    }

    # 10. 检查旧版本保留，新版本有效
    assert graph.versions["D1@v1"].status == NodeStatus.INVALID
    assert graph.versions["D2@v1"].status == NodeStatus.STALE
    assert graph.versions["D3@v1"].status == NodeStatus.STALE
    assert graph.versions["D4@v1"].status == NodeStatus.STALE

    assert graph.get_current_version("D4").status == NodeStatus.VERIFIED

    # 11. 检查完整溯源
    assert graph.check_lineage() == []
