from __future__ import annotations

from graph.graph import Graph

from models.vehicle import Vehicle

from simulation.vehicle_position import VehiclePosition
from simulation.movement_result import MovementResult


class MovementEngine:
    """
    Responsible for moving vehicles along assigned routes.

    This class knows nothing about:
        - Orders
        - Dispatching
        - Simulation state
    """

    def __init__(self, graph: Graph):

        self._graph = graph

    def move_vehicle(
        self,
        vehicle: Vehicle,
        position: VehiclePosition,
    ) -> MovementResult:

        result = MovementResult()

        if not position.has_route():
            return result

        next_node = position.route.next_node(position.route_index)

        if next_node is None:

            result.reached_destination = True

            return result

        edge = self._graph.get_edge(
            position.current_node,
            next_node,
        )

        result.previous_node = position.current_node

        position.current_edge = (
            position.current_node,
            next_node,
        )

        position.current_node = next_node

        position.route_index += 1

        position.distance_on_edge = 0.0

        result.current_node = next_node

        result.distance_travelled = edge.length

        result.travel_time = edge.travel_time

        result.moved = True

        result.reached_node = True

        if position.route.next_node(position.route_index) is None:

            result.reached_destination = True

        return result

    def move_all(
        self,
        vehicles: list[Vehicle],
        positions: dict[str, VehiclePosition],
    ) -> dict[str, MovementResult]:

        results = {}

        for vehicle in vehicles:

            if vehicle.vehicle_id not in positions:
                continue

            results[vehicle.vehicle_id] = self.move_vehicle(
                vehicle,
                positions[vehicle.vehicle_id],
            )

        return results