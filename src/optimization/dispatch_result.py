from dataclasses import dataclass, field
from src.simulation.route import Route

@dataclass(slots=True)
class DispatchAssignment:
    vehicle_id: str
    order_id: str
    route: Route
    objective_cost: float


@dataclass(slots=True)
class DispatchResult:
    tick: int
    status: str
    assignments: list[DispatchAssignment] = field(default_factory=list)

    @property
    def assigned_count(self) -> int:
        return len(self.assignments)

    @property
    def success(self) -> bool:
        return self.status in ("Optimal", "Feasible")