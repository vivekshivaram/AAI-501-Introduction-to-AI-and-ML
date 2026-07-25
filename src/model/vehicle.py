from dataclasses import dataclass, field
from typing import List

@dataclass
class Vehicle:
    vehicle_id: str
    current_node: int
    capacity: int
    speed_kmph: float = 40
    current_load: int = 0
    active_orders: List[str] = field(default_factory=list)
    distance_travelled: float = 0
    available: bool = True

    def remaining_capacity(self):
        return self.capacity - self.current_load

    def assign_order(self, order_id: str, quantity: int):
        if self.remaining_capacity() < quantity:
            raise ValueError("Vehicle capacity exceeded.")
        self.current_load += quantity
        self.active_orders.append(order_id)

    def complete_order(self, order_id, quantity):
        self.current_load -= quantity
        self.active_orders.remove(order_id)