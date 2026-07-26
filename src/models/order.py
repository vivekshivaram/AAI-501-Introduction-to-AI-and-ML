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
    delivery_tick: int | None = None
    delivered: bool = False