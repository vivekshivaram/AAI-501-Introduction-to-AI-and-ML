from dataclasses import dataclass, field
from datetime import datetime

from core.graph import Graph

from models.order import Order
from models.vehicle import Vehicle

from simulation.vehicle_position import VehiclePosition
from simulation.statistics import Statistics
from simulation.delay_map import DelayMap
from simulation.events import SimulationEvent


@dataclass
class SimulationContext:
    # Infrastructure
    graph: Graph
    current_time: datetime
    tick: int = 0

    # Vehicles
    vehicles: list[Vehicle] = field(default_factory=list)
    positions: dict[str, VehiclePosition] = field(default_factory=dict)

    # Orders
    pending_orders: list[Order] = field(default_factory=list)
    dispatched_orders: list[Order] = field(default_factory=list)
    delivered_orders: list[Order] = field(default_factory=list)
    rejected_orders: list[Order] = field(default_factory=list)

    # AI Outputs
    delay_map: DelayMap = field(default_factory=DelayMap)
    surge_multiplier: float = 1.0

    # Metrics
    statistics: Statistics = field(default_factory=Statistics)

    # Events
    events: list[SimulationEvent] = field(default_factory=list)