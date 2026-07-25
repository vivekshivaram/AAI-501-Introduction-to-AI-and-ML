from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: int
    latitude: float
    longitude: float