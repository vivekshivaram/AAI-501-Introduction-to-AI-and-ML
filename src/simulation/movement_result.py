from dataclasses import dataclass
"""
movement = MovementEngine(graph)

results = movement.move_all(
    simulation_state.vehicles,
    simulation_state.positions,
)

for vehicle_id, result in results.items():

    if result.reached_destination:
        print(f"{vehicle_id} reached destination")

    elif result.moved:
        print(
            f"{vehicle_id} moved "
            f"{result.distance_travelled:.2f} m"
        )
"""

@dataclass
class MovementResult:
    """
    Result of advancing a vehicle by one simulation step.
    """
    moved: bool = False
    reached_node: bool = False
    reached_destination: bool = False
    previous_node: int | None = None
    current_node: int | None = None
    distance_travelled: float = 0.0
    travel_time: float = 0.0