from __future__ import annotations
import random
from src.models.vehicle import Vehicle


class FleetGenerator:
    VEHICLE_CAPACITY = {
        "bicycle": 10.0,
        "scooter": 20.0,
        "motorcycle": 30.0,
        "van": 200.0,
    }

    VEHICLE_SPEED = {
        "bicycle": 15.0,
        "scooter": 30.0,
        "motorcycle": 45.0,
        "van": 35.0,
    }

    def generate(
        self,
        count: int,
        graph_nodes: list[int],
    ) -> list[Vehicle]:
        vehicles = []
        vehicle_types = list(
            self.VEHICLE_CAPACITY.keys()
        )

        for i in range(count):
            vehicle_type = random.choice(
                vehicle_types
            )
            node = random.choice(graph_nodes)
            vehicles.append(
                Vehicle(
                    vehicle_id=f"V{i+1:04d}",
                    capacity=self.VEHICLE_CAPACITY[
                        vehicle_type
                    ],
                    max_speed_kmh=self.VEHICLE_SPEED[
                        vehicle_type
                    ],
                    home_node=node,
                    current_node=node,
                    vehicle_type=vehicle_type
                )
            )

        return vehicles