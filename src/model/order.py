from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class OrderStatus(Enum):
    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    order_id: str
    pickup_node: int
    drop_node: int
    quantity: int
    weight: float
    priority: int = 1
    assigned_vehicle: Optional[str] = None
    predicted_delay: float = 0.0
    qc_status: str = "Intact"
    surge_multiplier: float = 1.0
    status: OrderStatus = OrderStatus.CREATED
    metadata: dict = field(default_factory=dict)

    def assign_vehicle(self, vehicle_id: str):
        self.assigned_vehicle = vehicle_id
        self.status = OrderStatus.ASSIGNED

    def mark_in_transit(self):
        self.status = OrderStatus.IN_TRANSIT

    def mark_delivered(self):
        self.status = OrderStatus.DELIVERED