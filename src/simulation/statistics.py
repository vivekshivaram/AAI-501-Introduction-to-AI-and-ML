from dataclasses import dataclass, field

@dataclass
class Statistics:
    sampled_orders: int = 0
    inspected_orders: int = 0
    rejected_orders: int = 0
    dispatched_orders: int = 0
    delivered_orders: int = 0
    total_distance: float = 0.0
    total_delay: float = 0.0
    average_reward: float = 0.0
    average_surge_multiplier: float = 1.0
    
    # Additional tracking
    total_delivery_time_minutes: float = 0.0
    surge_multipliers: list[float] = field(default_factory=list)
    
    def record_delivery(self, delivery_time_minutes: float, distance: float):
        """Record a completed delivery."""
        self.delivered_orders += 1
        self.total_delivery_time_minutes += delivery_time_minutes
        self.total_distance += distance
    
    def record_surge(self, surge: float):
        """Record surge multiplier for averaging."""
        self.surge_multipliers.append(surge)
    
    def get_avg_delivery_time(self) -> float:
        """Calculate average delivery time in minutes."""
        if self.delivered_orders == 0:
            return 0.0
        return self.total_delivery_time_minutes / self.delivered_orders
    
    def get_avg_surge(self) -> float:
        """Calculate average surge multiplier."""
        if not self.surge_multipliers:
            return 1.0
        return sum(self.surge_multipliers) / len(self.surge_multipliers)
    
    def get_delivery_rate(self) -> float:
        """Calculate delivery success rate (delivered / dispatched)."""
        if self.dispatched_orders == 0:
            return 0.0
        return self.delivered_orders / self.dispatched_orders
    
    def get_avg_distance_per_delivery(self) -> float:
        """Calculate average distance per delivery in km."""
        if self.delivered_orders == 0:
            return 0.0
        return (self.total_distance / 1000.0) / self.delivered_orders
