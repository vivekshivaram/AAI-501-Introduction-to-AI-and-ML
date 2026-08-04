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