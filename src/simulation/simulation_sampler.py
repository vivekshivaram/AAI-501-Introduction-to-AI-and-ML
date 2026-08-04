"""
simulation_sampler.py
Provides orders and fleet to the SimulationExecutor.
"""
from __future__ import annotations
import random
from collections import deque
from src.simulation.order_loader import OrderLoader
from src.simulation.fleet_generator import FleetGenerator
from src.models.order import Order
from src.models.vehicle import Vehicle
from src.graph.graph import Graph

class SimulationSampler:
    """
    Supplies synthetic simulation inputs.

    Responsibilities
    ----------------
    * Load all orders
    * Generate fleet once
    * Supply N orders every tick
    """

    def __init__(
        self,
        graph: Graph,
        order_loader: OrderLoader,
        fleet_generator: FleetGenerator,
        pickle_file: str,
        fleet_size: int = 1000,
        orders_per_tick: int = 5,
        shuffle_orders: bool = False,
    ):
        self.graph = graph
        if not self.graph.is_loaded():
            self.graph.load()
            
        self._orders = deque(
            order_loader.load(pickle_file)
        )
        if shuffle_orders:
            tmp = list(self._orders)
            random.shuffle(tmp)
            self._orders = deque(tmp)
        self._vehicles = fleet_generator.generate(
            count=fleet_size,
            graph_nodes=self.graph.get_node_ids(),
        )
        self._orders_per_tick = orders_per_tick


    @property
    def vehicles(self) -> list[Vehicle]:
        """
        Fleet remains constant throughout simulation.
        """
        return self._vehicles

    def has_more_orders(self) -> bool:
        return len(self._orders) > 0

    def next_orders(self) -> list[Order]:
        """
        Returns orders arriving during the current tick.
        """
        batch = []
        while ( self._orders and len(batch) < self._orders_per_tick):
            batch.append(self._orders.popleft())
            
        return batch

    def remaining_orders(self) -> int:
        return len(self._orders)