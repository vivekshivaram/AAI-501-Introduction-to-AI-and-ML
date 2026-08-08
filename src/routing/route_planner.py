from __future__ import annotations
from abc import ABC, abstractmethod
from src.routing.path import Path


class RoutePlanner(ABC):
    @abstractmethod
    def shortest_path(self, source: int, destination: int) -> Path:
        """Return the lowest-cost Path between two OSM node IDs."""