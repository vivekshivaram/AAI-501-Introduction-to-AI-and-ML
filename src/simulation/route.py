from src.graph.node import Node
from dataclasses import dataclass

@dataclass
class Route:
    nodes: list[Node]
    total_distance: float
    estimated_time: float
    route_cost: float
    arrival_times: list[float]
    current_index: int = 0
    
    def next_node(self, index: int) -> int | None:
        """
        Get the next node ID in the route at the given index.
        Returns None if index is out of bounds.
        """
        if index < 0 or index >= len(self.nodes):
            return None
        return self.nodes[index].id
