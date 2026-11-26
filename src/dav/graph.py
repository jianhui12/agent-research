"""Version-aware provenance DAG for DAV.

Graph vertices are ArtifactVersion IDs, not mutable logical-node IDs.
LogicalNode objects group versions; ExecutionRecord captures the exact
input/output versions of each computation.
"""
from __future__ import annotations

from typing import Any, Iterable
import networkx as nx

from .models import (
    ArtifactVersion,
    DependencyEdge,
    ExecutionRecord,
    LogicalNode,
    NodeStatus,
)


class DependencyGraph:
    """A version-level DAG with strict execution lineage."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.logical_nodes: dict[str, LogicalNode] = {}
        self.versions: dict[str, ArtifactVersion] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        # Compatibility alias for callers that previously accessed graph.nodes.
        # Values are ArtifactVersion instances keyed by version_id.
        self.nodes = self.versions

    def add_logical_node(
        self, node_id: str, name: str = "", node_type: str = "intermediate"
    ) -> LogicalNode:
        if node_id in self.logical_nodes:
            raise ValueError(f"Logical node already exists: {node_id}")
        node = LogicalNode(node_id=node_id, name=name or node_id, node_type=node_type)
        self.logical_nodes[node_id] = node
        return node

    def add_node(self, node: LogicalNode | str, **kwargs: Any) -> LogicalNode:
        """Register a logical node; retained as a convenient add_node API."""
        if isinstance(node, LogicalNode):
            if node.node_id in self.logical_nodes:
                raise ValueError(f"Logical node already exists: {node.node_id}")
            self.logical_nodes[node.node_id] = node
            return node
        return self.add_logical_node(node, **kwargs)

    def add_version(
        self,
        node_id: str,
        value: Any = None,
        *,
        status: NodeStatus = NodeStatus.UNVERIFIED,
        version_id: str | None = None,
        execution_id: str | None = None,
        make_current: bool = True,
        **metadata: Any,
    ) -> ArtifactVersion:
        if node_id not in self.logical_nodes:
            raise KeyError(f"Unknown logical node: {node_id}")
        logical = self.logical_nodes[node_id]
        version_index = len(logical.version_ids) + 1
        version_id = version_id or f"{node_id}@v{version_index}"
        if version_id in self.versions:
            raise ValueError(f"Artifact version already exists: {version_id}")

        version = ArtifactVersion(
            version_id=version_id,
            node_id=node_id,
            version_index=version_index,
            value=value,
            status=status,
            execution_id=execution_id,
            **metadata,
        )
        self.versions[version_id] = version
        self.graph.add_node(version_id)
        logical.version_ids.append(version_id)
        if make_current:
            logical.current_version_id = version_id
        return version

    def get_current_version(self, node_id: str) -> ArtifactVersion:
        logical = self.logical_nodes[node_id]
        if logical.current_version_id is None:
            raise LookupError(f"Logical node has no versions: {node_id}")
        return self.versions[logical.current_version_id]

    def add_dependency(
        self,
        source_version_id: str,
        target_version_id: str,
        propagation_prob: float = 1.0,
        input_role: str | None = None,
    ) -> DependencyEdge:
        self._require_version(source_version_id)
        self._require_version(target_version_id)
        if source_version_id == target_version_id:
            raise ValueError("A version cannot depend on itself")
        if not 0.0 <= propagation_prob <= 1.0:
            raise ValueError("propagation_prob must be in [0, 1]")
        if self.graph.has_edge(source_version_id, target_version_id):
            raise ValueError(
                f"Dependency already exists: {source_version_id} -> {target_version_id}"
            )

        self.graph.add_edge(
            source_version_id,
            target_version_id,
            propagation_prob=float(propagation_prob),
            input_role=input_role,
        )
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(source_version_id, target_version_id)
            raise ValueError("Version dependency graph must be a DAG")
        return DependencyEdge(
            source_version_id, target_version_id, float(propagation_prob), input_role
        )

    def record_execution(
        self,
        execution_id: str,
        operation: str,
        input_version_ids: Iterable[str],
        output_version_ids: Iterable[str],
        *,
        tool_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        execution_cost: float = 0.0,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
        edge_metadata: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> ExecutionRecord:
        """Record exact input/output versions and create provenance edges.

        Output versions must already exist and must not belong to another
        execution. The operation is atomic: on failure, newly-added edges and
        execution assignments are rolled back.
        """
        if execution_id in self.executions:
            raise ValueError(f"Execution already exists: {execution_id}")
        inputs, outputs = tuple(input_version_ids), tuple(output_version_ids)
        if not outputs:
            raise ValueError("An execution must produce at least one output version")
        for version_id in (*inputs, *outputs):
            self._require_version(version_id)
        for output_id in outputs:
            existing = self.versions[output_id].execution_id
            if existing is not None:
                raise ValueError(f"{output_id} already belongs to execution {existing}")
        if execution_cost < 0:
            raise ValueError("execution_cost must be non-negative")

        added_edges: list[tuple[str, str]] = []
        try:
            for source_id in inputs:
                for target_id in outputs:
                    attrs = (edge_metadata or {}).get((source_id, target_id), {})
                    self.add_dependency(
                        source_id,
                        target_id,
                        propagation_prob=attrs.get("propagation_prob", 1.0),
                        input_role=attrs.get("input_role"),
                    )
                    self.graph.edges[source_id, target_id]["execution_id"] = execution_id
                    added_edges.append((source_id, target_id))
            for output_id in outputs:
                self.versions[output_id].execution_id = execution_id
            record = ExecutionRecord(
                execution_id=execution_id,
                operation=operation,
                input_version_ids=inputs,
                output_version_ids=outputs,
                tool_name=tool_name,
                parameters=parameters or {},
                execution_cost=float(execution_cost),
                timestamp=timestamp,
                metadata=metadata or {},
            )
            self.executions[execution_id] = record
            return record
        except Exception:
            for source_id, target_id in reversed(added_edges):
                if self.graph.has_edge(source_id, target_id):
                    self.graph.remove_edge(source_id, target_id)
            for output_id in outputs:
                if self.versions[output_id].execution_id == execution_id:
                    self.versions[output_id].execution_id = None
            raise

    def get_direct_consumers(self, version_id: str) -> set[str]:
        self._require_version(version_id)
        return set(self.graph.successors(version_id))

    def get_affected_nodes(
        self, version_ids: str | Iterable[str], *, include_sources: bool = False
    ) -> set[str]:
        """Return version IDs transitively derived from the supplied versions."""
        roots = {version_ids} if isinstance(version_ids, str) else set(version_ids)
        for version_id in roots:
            self._require_version(version_id)
        affected: set[str] = set()
        for version_id in roots:
            affected.update(nx.descendants(self.graph, version_id))
        if include_sources:
            affected.update(roots)
        return affected

    def invalidate_versions(
        self,
        version_ids: str | Iterable[str],
        *,
        include_sources: bool = False,
    ) -> set[str]:
        """Mark derived versions STALE; optionally mark the source versions too."""
        affected = self.get_affected_nodes(
            version_ids, include_sources=include_sources
        )
        for version_id in affected:
            version = self.versions[version_id]
            if version.status != NodeStatus.BLOCKED:
                version.status = NodeStatus.STALE
        return affected

    def supersede_version(self, old_version_id: str, new_version_id: str) -> set[str]:
        """Make a newer version current and stale descendants of the old version.

        The old artifact remains in the graph for auditability. The new version
        must belong to the same logical node and have a strictly larger index.
        """
        old = self._require_version(old_version_id)
        new = self._require_version(new_version_id)
        if old.node_id != new.node_id:
            raise ValueError("Superseding versions must belong to the same logical node")
        if new.version_index <= old.version_index:
            raise ValueError("New version index must be greater than old version index")
        self.logical_nodes[old.node_id].current_version_id = new_version_id
        # The superseded source is retained for audit, but must no longer be
        # treated as a trusted current artifact.
        old.status = NodeStatus.INVALID
        return self.invalidate_versions(old_version_id)

    def mark_status(self, version_id: str, status: NodeStatus) -> None:
        self._require_version(version_id)
        self.versions[version_id].status = status

    def topological_sort(self, subset: Iterable[str] | None = None) -> list[str]:
        if subset is None:
            graph = self.graph
        else:
            selected = set(subset)
            unknown = selected.difference(self.versions)
            if unknown:
                raise KeyError(f"Unknown versions in subset: {sorted(unknown)}")
            graph = self.graph.subgraph(selected).copy()
        return list(nx.topological_sort(graph))

    def check_lineage(self) -> list[str]:
        """Return consistency errors; an empty list means lineage is consistent."""
        errors: list[str] = []
        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Version dependency graph contains a cycle")

        for version_id, version in self.versions.items():
            if version_id not in self.graph:
                errors.append(f"Version missing from graph: {version_id}")
            if version.node_id not in self.logical_nodes:
                errors.append(f"Unknown logical node for version {version_id}")
            elif version_id not in self.logical_nodes[version.node_id].version_ids:
                errors.append(f"Logical node does not register version {version_id}")
            if version.execution_id is not None:
                record = self.executions.get(version.execution_id)
                if record is None:
                    errors.append(f"Missing execution record {version.execution_id}")
                elif version_id not in record.output_version_ids:
                    errors.append(f"Execution does not list output {version_id}")

        for source_id, target_id, attrs in self.graph.edges(data=True):
            execution_id = attrs.get("execution_id")
            target_execution_id = self.versions[target_id].execution_id
            if execution_id is None:
                errors.append(f"Untracked dependency edge {source_id} -> {target_id}")
            elif execution_id != target_execution_id:
                errors.append(
                    f"Edge {source_id} -> {target_id} points to execution "
                    f"{execution_id}, expected {target_execution_id}"
                )

        for execution_id, record in self.executions.items():
            for version_id in record.input_version_ids:
                if version_id not in self.versions:
                    errors.append(f"{execution_id} references missing input {version_id}")
            for version_id in record.output_version_ids:
                if version_id not in self.versions:
                    errors.append(f"{execution_id} references missing output {version_id}")
                elif self.versions[version_id].execution_id != execution_id:
                    errors.append(f"Output {version_id} has inconsistent execution ID")
            for target_id in record.output_version_ids:
                actual_inputs = set(self.graph.predecessors(target_id))
                expected_inputs = set(record.input_version_ids)
                if actual_inputs != expected_inputs:
                    errors.append(
                        f"Input lineage mismatch for {target_id}: "
                        f"expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}"
                    )
                for source_id in record.input_version_ids:
                    if self.graph.has_edge(source_id, target_id):
                        edge_execution = self.graph.edges[source_id, target_id].get("execution_id")
                        if edge_execution != execution_id:
                            errors.append(
                                f"Provenance edge {source_id} -> {target_id} "
                                f"has wrong execution ID"
                            )
        return errors

    def validate_graph(self) -> None:
        errors = self.check_lineage()
        if errors:
            raise ValueError("Invalid provenance graph: " + "; ".join(errors))

    def _require_version(self, version_id: str) -> ArtifactVersion:
        try:
            return self.versions[version_id]
        except KeyError as exc:
            raise KeyError(f"Unknown artifact version: {version_id}") from exc
