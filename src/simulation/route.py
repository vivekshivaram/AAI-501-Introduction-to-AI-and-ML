@dataclass
class Route:
    nodes: list[int]
    total_distance: float
    estimated_time: float
    route_cost: float
    arrival_times: list[float]
    current_index: int = 0