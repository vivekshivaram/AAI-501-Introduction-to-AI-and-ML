"""
dispatcher_config.py
Configuration settings for the CVRPTW Dispatcher.
This module centralizes all optimization parameters so that
different solver strategies and routing policies can be evaluated
without modifying dispatcher logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

SolverType = Literal[
    "pulp",
    "ortools",
]

SequencingStrategy = Literal[
    "nearest_neighbor",
    "greedy",
    "two_opt",
]

ObjectiveType = Literal[
    "travel_cost",
    "travel_time",
    "distance",
]


@dataclass(frozen=True, slots=True)
class DispatcherConfig:
    """
    Configuration for the CVRPTW dispatcher.
    Notes
    -----
    This object is intentionally immutable so that optimization
    runs remain reproducible.
    """

    # ------------------------------------------------------------------
    # Solver Configuration
    # ------------------------------------------------------------------

    solver: SolverType = "pulp"
    solver_time_limit: int = 60
    """
    Maximum solver execution time (seconds).
    """

    mip_gap: float = 0.01
    """
    Relative optimality gap.

    Example
    -------
    0.01 = accept solution within 1% of optimum.
    """

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    objective: ObjectiveType = "travel_cost"

    # ------------------------------------------------------------------
    # Route Sequencing
    # ------------------------------------------------------------------

    sequencing: SequencingStrategy = "nearest_neighbor"

    improve_routes: bool = True
    """
    Apply local optimization (e.g. 2-opt) after sequencing.
    """

    # ------------------------------------------------------------------
    # Vehicle Policies
    # ------------------------------------------------------------------

    allow_multiple_orders: bool = True

    require_return_to_depot: bool = False

    respect_vehicle_capacity: bool = True

    respect_vehicle_availability: bool = True

    # ------------------------------------------------------------------
    # Time Windows
    # ------------------------------------------------------------------

    enable_time_windows: bool = True

    allow_late_delivery: bool = True

    lateness_penalty: float = 500.0

    # ------------------------------------------------------------------
    # Cost Function Weights
    # ------------------------------------------------------------------

    travel_cost_weight: float = 1.0

    travel_time_weight: float = 1.0

    distance_weight: float = 0.0

    vehicle_usage_penalty: float = 25.0

    unassigned_order_penalty: float = 10_000.0

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    cache_shortest_paths: bool = True

    parallel_cost_matrix: bool = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    enable_logging: bool = True

    enable_statistics: bool = True

    save_solver_model: bool = False

    solver_model_filename: str = "cvrptw_model.lp"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:

        if self.solver_time_limit <= 0:
            raise ValueError(
                "solver_time_limit must be positive."
            )

        if not (0.0 <= self.mip_gap <= 1.0):
            raise ValueError(
                "mip_gap must be between 0 and 1."
            )

        if self.lateness_penalty < 0:
            raise ValueError(
                "lateness_penalty cannot be negative."
            )

        if self.vehicle_usage_penalty < 0:
            raise ValueError(
                "vehicle_usage_penalty cannot be negative."
            )

        if self.unassigned_order_penalty < 0:
            raise ValueError(
                "unassigned_order_penalty cannot be negative."
            )