from src.simulation.route import Route
from dataclasses import dataclass, field

@dataclass
class Vehicle:
    vehicle_id: str
    capacity: float
    max_speed_kmh: float
    home_node: int
    current_node: int
    current_load: float = 0.0
    available: bool = True
    current_route: Route | None = None
    assigned_orders: list[str] = field(default_factory=list)
    vehicle_type: str | None = None