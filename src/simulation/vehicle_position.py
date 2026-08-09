from dataclasses import dataclass
from src.simulation.route import Route

@dataclass
class VehiclePosition:
    current_node: int
    current_edge: tuple[int, int] | None = None
    edge_progress_meters: float = 0.0
    remaining_edge_distance: float = 0.0
    remaining_edge_time: float = 0.0
    route_index: int = 0
    route: Route | None = None
    distance_on_edge: float = 0.0
    
    def has_route(self) -> bool:
        return self.route is not None and len(self.route.nodes) > 0
    
    def reached_destination(self) -> bool:
        """Check if vehicle has reached the final destination of its route."""
        if not self.has_route():
            return False
        # Vehicle reached destination if route_index >= total nodes in route
        return self.route_index >= len(self.route.nodes) - 1
