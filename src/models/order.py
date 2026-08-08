from dataclasses import dataclass
import datetime

@dataclass
class Order:
    order_id: str
    pickup_node: int
    delivery_node: int
    package_image: str
    created_time: datetime
    deadline: datetime
    weight: float
    volume: float
    inspection_passed: bool | None = None
    predicted_delay: float = 0.0
    assigned_vehicle: str | None = None
    assigned_tick: int | None = None
    dispatch_tick: int | None = None  # When assigned to vehicle
    delivery_tick: int | None = None  # When delivered
    delivered: bool = False
    delivery_time_minutes: float | None = None  # Actual delivery time
    preferred_vehicle_type: str | None = None
