from __future__ import annotations

from dataclasses import dataclass

from src.graph.node import Node
from src.graph.edge import Edge


@dataclass(frozen=True, slots=True)
class Path:
    """
    Result of a routing algorithm.
    """
    nodes: list[Node]
    edges: list[Edge]
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
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def is_empty(self) -> bool:
        return not self.nodes