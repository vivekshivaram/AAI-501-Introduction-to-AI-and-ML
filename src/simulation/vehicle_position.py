@dataclass
class VehiclePosition:
    current_node: int
    current_edge: tuple[int, int] | None = None
    edge_progress_meters: float = 0.0
    remaining_edge_distance: float = 0.0
    remaining_edge_time: float = 0.0
    route_index: int = 0