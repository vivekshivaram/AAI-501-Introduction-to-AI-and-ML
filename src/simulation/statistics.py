from dataclasses import dataclass

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