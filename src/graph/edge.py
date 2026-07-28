from dataclasses import dataclass

@dataclass(frozen=True)
class Edge:
    source: int
    destination: int
    length: float
    travel_time: float
    speed_kph: float
    
    @property
    def endpoints(self) -> tuple[int, int]:
        """Return the edge endpoints as a (source, destination) tuple."""
        return self.source, self.destination