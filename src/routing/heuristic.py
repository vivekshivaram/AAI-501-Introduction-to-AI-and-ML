from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from src.graph.graph import Graph
from src.graph.node import Node

from src.utils.geo_utils import haversine_distance

class Heuristic(ABC):
    """
    Base heuristic for A* search.
    """

    @abstractmethod
    def estimate(
        self,
        graph: Graph,
        current: Node,
        goal: Node,
    ) -> float:
        """
        Returns an admissible cost estimate.
        """

class DistanceHeuristic(Heuristic):
    def estimate(
        self,
        graph,
        current,
        goal,
    ) -> float:
        current
        goal
        
        return haversine_distance(current.latitude, current.longitude, goal.latitude, goal.longitude)

class TravelTimeHeuristic(Heuristic):
    def __init__(
        self,
        max_speed_kmh: float,
    ):
        self._speed = max_speed_kmh

    def estimate(
        self,
        graph,
        current,
        goal,
    ) -> float:
        metres = DistanceHeuristic().estimate(
            graph,
            current,
            goal,
        )
        metres_per_second = self._speed / 3.6
        return metres / metres_per_second