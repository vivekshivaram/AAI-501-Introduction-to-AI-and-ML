"""main.py - LogiSim-AI master simulation script."""
from __future__ import annotations
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.config import (ARTIFACTS_DIRECTORY, DATA_INTERIM_DIRECTORY, DATA_OUTPUTS_DIRECTORY,
    DEMAND_FORECAST_FILENAME, MAPPED_ORDERS_FILENAME, MAX_SIMULATION_STEPS, DEFAULT_VEHICLE_CAPACITY)
from src.graph.graph import Graph
from src.analytics.delay_model import DelayPredictor
from src.routing.heuristic import TravelTimeHeuristic
from src.optimization.astar_routing import AStarRouting
from src.optimization.cvrptw_dispatcher_milp import CVRPTWDispatcherMilp
from src.optimization.dispatcher_config import DispatcherConfig
from src.simulation.delay_map import DelayMap
from src.simulation.simulation_context import SimulationContext
from src.simulation.simulation_executor import SimulationExecutor
from src.simulation.movement_engine import MovementEngine
from src.simulation.vehicle_position import VehiclePosition
from src.models.order import Order
from src.models.vehicle import Vehicle
from src.ai.pricing_env import PricingEnv, load_demand_forecast
from src.ai.pricing_engine import PricingEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderSampler:
    def __init__(self, orders_path: Path, orders_per_tick: int = 5) -> None:
        self._df = pd.read_pickle(orders_path)
        self._orders_per_tick = orders_per_tick
        self._index = 0

    def sample(self, context: SimulationContext) -> list[Order]:
        rows = self._df.iloc[self._index : self._index + self._orders_per_tick]
        self._index = (self._index + self._orders_per_tick) % len(self._df)
        now = context.current_time
        orders: list[Order] = []
        for _, row in rows.iterrows():
            order = Order(
                order_id=str(row.get("Order_ID", f"ORD-{self._index}")),
                pickup_node=int(row["Store_Node"]),
                delivery_node=int(row["Drop_Node"]),
                package_image="",
                created_time=now,
                deadline=now + timedelta(hours=4),
                weight=random.uniform(1.0, 10.0),
                volume=random.uniform(0.01, 0.1),
            )
            orders.append(order)
            context.pending_orders.append(order)
        return orders


class PackageInspector:
    def __init__(self, engine: PricingEngine) -> None:
        self._engine = engine

    def inspect(self, orders: list[Order]) -> list[Order]:
        intact: list[Order] = []
        for order in orders:
            order.inspection_passed = True
            intact.append(order)
        return intact


class RLEnvironment:
    def __init__(self, env: PricingEnv) -> None:
        self._env = env
        self._env.reset()

    def update(self, context: SimulationContext):
        self._env._queue_length = len(context.pending_orders)
        return self._env.state_index()


class RLAgent:
    def __init__(self, engine: PricingEngine) -> None:
        self._engine = engine

    def act(self, state) -> float:
        q = int(state[0]) if hasattr(state, "__len__") else int(state)
        return self._engine.act(q)


class DispatcherAdapter:
    def __init__(self, milp: CVRPTWDispatcherMilp) -> None:
        self._milp = milp

    def dispatch(self, context: SimulationContext) -> None:
        result = self._milp.dispatch(vehicles=context.vehicles, orders=context.pending_orders, tick=context.tick)
        if result is None:
            return
        dispatched_ids = {a.order_id for a in result.assignments}
        context.pending_orders = [o for o in context.pending_orders if o.order_id not in dispatched_ids]
        logger.info(f"Tick {context.tick}: dispatched {result.assigned_count} orders")


class MovementAdapter:
    def __init__(self, engine: MovementEngine) -> None:
        self._engine = engine

    def move(self, context: SimulationContext) -> None:
        self._engine.move_all(list(context.vehicles), context.positions)


def _build_fleet(graph: Graph, n: int = 3):
    node_ids = list(graph.graph.nodes())[:n]
    vehicles, positions = [], {}
    for i, node_id in enumerate(node_ids):
        vid = f"VAN-{i+1:03d}"
        vehicles.append(Vehicle(vehicle_id=vid, capacity=float(DEFAULT_VEHICLE_CAPACITY),
            max_speed_kmh=40.0, home_node=node_id, current_node=node_id))
        positions[vid] = VehiclePosition(current_node=node_id)
    return vehicles, positions


def main() -> None:
    logger.info("=== LogiSim-AI starting ===")
    graph = Graph()
    graph.load()
    predictor = DelayPredictor()
    demand_forecast = load_demand_forecast()
    pricing_env = PricingEnv(demand_forecast=demand_forecast)
    pricing_engine = PricingEngine()
    heuristic = TravelTimeHeuristic(60.0)
    astar_routing = AStarRouting(graph, heuristic, DelayMap())
    milp_dispatcher = CVRPTWDispatcherMilp(astar_routing, DispatcherConfig())
    movement_engine = MovementEngine(graph)
    executor = SimulationExecutor(
        sampler=OrderSampler(DATA_INTERIM_DIRECTORY / MAPPED_ORDERS_FILENAME),
        inspector=PackageInspector(pricing_engine),
        predictor=predictor,
        dispatcher=DispatcherAdapter(milp_dispatcher),
        rl_environment=RLEnvironment(pricing_env),
        rl_agent=RLAgent(pricing_engine),
        movement_engine=MovementAdapter(movement_engine),
    )
    vehicles, positions = _build_fleet(graph)
    context = SimulationContext(graph=graph, current_time=datetime.now(), vehicles=vehicles, positions=positions)
    logger.info(f"Running {MAX_SIMULATION_STEPS} ticks with {len(vehicles)} vehicles")
    for tick in range(MAX_SIMULATION_STEPS):
        context.current_time += timedelta(minutes=1)
        executor.execute_tick(context)
        if tick % 100 == 0:
            logger.info(f"Tick {tick:4d} | pending={len(context.pending_orders)} | surge={context.surge_multiplier:.2f}")
    logger.info("=== Simulation complete ===")


if __name__ == "__main__":
    main()
