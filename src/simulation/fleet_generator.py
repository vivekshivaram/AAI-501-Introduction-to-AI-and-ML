from __future__ import annotations
import random
from src.models.vehicle import Vehicle
from src.config import VEHICLE_TYPES


class FleetGenerator:
    VEHICLE_CAPACITY = {vtype: specs["capacity"] for vtype, specs in VEHICLE_TYPES.items()}
    VEHICLE_SPEED = {vtype: specs["speed_kmh"] for vtype, specs in VEHICLE_TYPES.items()}

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