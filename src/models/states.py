from enum import Enum

"""
Example: Usage:

order = Order(
    order_id="ORD001",
    pickup_node=100,
    drop_node=200,
    quantity=4,
    weight=12.5
)

vehicle = Vehicle(
    vehicle_id="VAN001",
    current_node=100,
    capacity=30
)

vehicle.assign_order(order.order_id, order.quantity)
order.assign_vehicle(vehicle.vehicle_id)
print(vehicle)
print(order)
"""
class VehicleState(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    LOADING = "LOADING"
    UNLOADING = "UNLOADING"
    RETURNING = "RETURNING"

class SimulationState(Enum):
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"