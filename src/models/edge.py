from dataclasses import dataclass

@dataclass(frozen=True)
class Edge:
    source: int
    destination: int
    length: float
    travel_time: float
    speed_kph: float