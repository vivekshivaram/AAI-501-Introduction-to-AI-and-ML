from dataclasses import dataclass
from datetime import datetime


@dataclass
class SimulationEvent:
    timestamp: datetime
    description: str

@dataclass
class VehicleMovedEvent(SimulationEvent):
    vehicle_id: str
    from_node: int
    to_node: int

@dataclass
class OrderAssignedEvent(SimulationEvent):
    order_id: str
    vehicle_id: str

@dataclass
class PickupCompletedEvent(SimulationEvent):
    order_id: str
    vehicle_id: str


@dataclass
class DeliveryCompletedEvent(SimulationEvent):
    order_id: str
    vehicle_id: str