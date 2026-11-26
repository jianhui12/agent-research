"""Typed data models for version-aware DAV provenance graphs."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    INVALID = "invalid"
    REPAIRING = "repairing"
    STALE = "stale"
    BLOCKED = "blocked"


@dataclass
class LogicalNode:
    """Stable identity for a logical datum; versions live in ArtifactVersion."""
    node_id: str
    name: str = ""
    node_type: str = "intermediate"
    current_version_id: str | None = None
    version_ids: list[str] = field(default_factory=list)


@dataclass
class ArtifactVersion:
    """Immutable-in-identity artifact version; status/beliefs may evolve."""
    version_id: str
    node_id: str
    version_index: int
    value: Any = None
    status: NodeStatus = NodeStatus.UNVERIFIED
    execution_id: str | None = None
    intrinsic_error_prob: float = 0.0
    posterior_error_prob: float = 0.0
    verifier_sensitivity: float = 1.0
    verifier_specificity: float = 1.0
    repair_success_prob: float = 1.0
    recompute_success_prob: float = 1.0
    verify_cost: float = 1.0
    repair_cost: float = 1.0
    recompute_cost: float = 1.0
    loss_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyEdge:
    """A version-to-version data dependency and its propagation metadata."""
    source_version_id: str
    target_version_id: str
    propagation_prob: float = 1.0
    input_role: str | None = None


@dataclass(frozen=True)
class ExecutionRecord:
    """Provenance record for one tool/model/computation execution."""
    execution_id: str
    operation: str
    input_version_ids: tuple[str, ...]
    output_version_ids: tuple[str, ...]
    tool_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    execution_cost: float = 0.0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
