from __future__ import annotations

from dataclasses import dataclass
from src.graph.node import Node

@dataclass(frozen=True, slots=True)
class Path:
    """
    Represents the result of a routing algorithm.
    Attributes:
        nodes:
            Ordered list of Nodes.
        total_distance:
            Total route length in metres.
        total_travel_time:
            Total travel time in seconds.
        total_cost:
            Cost used by the planner.
            Initially equals travel time.
            Can later include RF delay penalties.
    """
    nodes: list[Node]
    total_distance: float
    total_travel_time: float
    total_cost: float

    @property
    def start(self) -> Node:
        return self.nodes[0]

    @property
    def destination(self) -> Node:
        return self.nodes[-1]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def is_empty(self) -> bool:
        return len(self.nodes) == 0