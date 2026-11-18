"""Core typed models. Keep simulator truth separate from policy beliefs."""
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
class ParentVersion:
    node_id: str
    version: int


@dataclass
class Node:
    node_id: str
    value: Any = None
    parents: list[ParentVersion] = field(default_factory=list)
    version: int = 1
    status: NodeStatus = NodeStatus.UNVERIFIED
    intrinsic_error_prob: float = 0.0
    posterior_error_prob: float = 0.0
    propagation_prob: float = 1.0
    verifier_sensitivity: float = 1.0
    verifier_specificity: float = 1.0
    repair_success_prob: float = 1.0
    recompute_success_prob: float = 1.0
    verify_cost: float = 1.0
    repair_cost: float = 1.0
    recompute_cost: float = 1.0
    loss_weight: float = 1.0
