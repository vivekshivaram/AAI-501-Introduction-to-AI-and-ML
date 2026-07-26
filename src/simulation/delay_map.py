from dataclasses import dataclass, field


@dataclass
class DelayMap:
    edge_penalties: dict[tuple[int, int], float] = field(default_factory=dict)
 
    def penalty(self, edge):
        return self.edge_penalties.get(edge, 0.0)

    def update(self, predictions):
        self.edge_penalties = predictions