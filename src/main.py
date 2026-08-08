"""main.py - LogiSim-AI master simulation script.

Integrates all simulation components:
- Infrastructure & Optimization (MILP, A*, Movement)
- Data Engineering & ML (OrderLoader, DelayPredictor, Demand Forecast)
- Vision & RL (CNN Inspector, Q-Learning Pricing)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to Python path to enable src imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import random
from datetime import datetime, timedelta
import numpy as np
import torch
from torch import nn

from src.config import (
    ARTIFACTS_DIRECTORY,
    DATA_INTERIM_DIRECTORY,
    DATA_OUTPUTS_DIRECTORY,
    DEMAND_FORECAST_FILENAME,
    MAPPED_ORDERS_FILENAME,
    MAX_SIMULATION_STEPS,
    DEFAULT_VEHICLE_CAPACITY,
    FLEET_SIZE,
    ORDERS_PER_TICK,
    ORDER_DEADLINE_HOURS,
    PACKAGE_DAMAGE_REJECTION_RATE,
    SIMULATION_STEP_MINUTES,
    ROUTING_AVERAGE_SPEED_KMH,
    SURGE_MULTIPLIERS,
    DEFAULT_SURGE_MULTIPLIER,
)
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
from src.simulation.order_loader import OrderLoader
from src.simulation.fleet_generator import FleetGenerator
from src.models.order import Order
from src.models.vehicle import Vehicle
from src.ai.pricing_env import PricingEnv, load_demand_forecast
from src.ai.pricing_engine import PricingEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# FIRST-TIME SETUP HELPERS
# ============================================================

def _check_and_run_setup() -> bool:
    """
    Check if required data files exist. If not, run the setup pipeline.
    
    Returns:
        bool: True if all required files are ready, False if setup failed
    """
    from src.config import (
        MAP_DIRECTORY,
        MAP_FILENAME,
        DATA_INTERIM_DIRECTORY,
        MAPPED_ORDERS_FILENAME,
        ARTIFACTS_DIRECTORY,
        DELAY_MODEL_FILENAME,
        DATA_OUTPUTS_DIRECTORY,
        DEMAND_FORECAST_FILENAME,
    )
    
    # Define required files with their setup commands
    required_files = {
        "Road network": {
            "path": MAP_DIRECTORY / MAP_FILENAME,
            "command": "python -m src.ingestion.create_city_map",
            "description": "Downloading road network from OpenStreetMap (requires internet)",
        },
        "Mapped orders": {
            "path": DATA_INTERIM_DIRECTORY / MAPPED_ORDERS_FILENAME,
            "command": "python -m src.ingestion.snap_nodes",
            "description": "Snapping orders to road network nodes",
        },
        "Delay model": {
            "path": ARTIFACTS_DIRECTORY / DELAY_MODEL_FILENAME,
            "command": "python -m src.analytics.delay_model",
            "description": "Training Random Forest delay predictor",
        },
        "Demand forecast": {
            "path": DATA_OUTPUTS_DIRECTORY / DEMAND_FORECAST_FILENAME,
            "command": "python -m src.analytics.demand_forecast",
            "description": "Generating Holt-Winters demand forecast",
        },
    }
    
    # Optional files that can be missing
    optional_files = {
        "CNN model": {
            "path": ARTIFACTS_DIRECTORY / "package_cnn.pt",
            "command": "python -m src.ai.pricing_engine --train-cnn data/raw/damaged_box",
            "description": "Training ResNet18 package inspector (requires damaged_box dataset)",
        },
        "Q-table": {
            "path": ARTIFACTS_DIRECTORY / "q_table.npy",
            "command": "python -m src.ai.pricing_engine --train-q-table",
            "description": "Training Q-Learning pricing agent",
        },
    }
    
    logger.info("=" * 80)
    logger.info("CHECKING REQUIRED FILES")
    logger.info("=" * 80)
    
    missing_required = []
    missing_optional = []
    
    # Check required files
    for name, info in required_files.items():
        if info["path"].exists():
            logger.info(f"[OK] {name}: {info['path']}")
        else:
            logger.warning(f"[MISSING] {name}: NOT FOUND at {info['path']}")
            missing_required.append((name, info))
    
    # Check optional files
    for name, info in optional_files.items():
        if info["path"].exists():
            logger.info(f"[OK] {name}: {info['path']}")
        else:
            logger.info(f"[OPTIONAL] {name}: NOT FOUND (optional) at {info['path']}")
            missing_optional.append((name, info))
    
    logger.info("=" * 80)
    
    # Run setup for missing required files
    if missing_required:
        logger.info("\nRunning first-time setup for missing required files...\n")
        
        for name, info in missing_required:
            logger.info(f"Setting up: {info['description']}...")
            
            try:
                # Run setup directly by importing and calling the module's main function
                if name == "Road network":
                    from src.ingestion.create_city_map import main as create_map
                    create_map()
                elif name == "Mapped orders":
                    from src.ingestion.snap_nodes import main as snap_nodes
                    snap_nodes()
                elif name == "Delay model":
                    from src.analytics.delay_model import train_and_save
                    train_and_save()
                elif name == "Demand forecast":
                    from src.analytics.demand_forecast import main as create_forecast
                    create_forecast()
                
                logger.info(f"[OK] {name} setup complete")
                
                # Verify file was created
                if not info["path"].exists():
                    logger.error(f"[ERROR] Setup completed but file not found: {info['path']}")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ Failed to set up {name}")
                logger.error(f"Error: {e}")
                logger.error(f"Please run manually: {info['command']}")
                return False
        
        logger.info("\n[SUCCESS] All required files are now ready!\n")
    else:
        logger.info("\n[SUCCESS] All required files found. No setup needed.\n")
    
    # Warn about missing optional files
    if missing_optional:
        logger.warning("\nOptional files not found:")
        for name, info in missing_optional:
            logger.warning(f"  - {name}: {info['description']}")
            logger.warning(f"    Run: {info['command']}")
        logger.warning("\nSimulation will continue with default behavior for missing components.\n")
    
    return True


class OrderSampler:
    """Samples orders from mapped_orders.pkl using OrderLoader."""
    
    def __init__(self, orders_per_tick: int = 5) -> None:
        self._orders_per_tick = orders_per_tick
        self._order_loader = OrderLoader()
        self._all_orders: list[Order] = []
        self._index = 0
        
        # Load all orders once at initialization
        try:
            self._all_orders = self._order_loader.load()
            logger.info(f"OrderSampler loaded {len(self._all_orders)} orders from mapped_orders.pkl")
        except FileNotFoundError:
            logger.warning("mapped_orders.pkl not found. Run 'python -m src.ingestion.snap_nodes' first.")
            self._all_orders = []

    def sample(self, context: SimulationContext) -> list[Order]:
        """Sample a batch of orders for this tick."""
        if not self._all_orders:
            return []
        
        # Sample orders_per_tick orders, cycling through the dataset
        sampled: list[Order] = []
        for _ in range(self._orders_per_tick):
            order = self._all_orders[self._index % len(self._all_orders)]
            self._index += 1
            
            # Update timestamps to current simulation time
            order.created_time = context.current_time
            order.deadline = context.current_time + timedelta(hours=ORDER_DEADLINE_HOURS)
            
            sampled.append(order)
            context.pending_orders.append(order)
        
        return sampled


class PackageInspector:
    """Uses ResNet18 CNN to inspect packages for damage."""
    
    def __init__(self, cnn_path: Path | None = None) -> None:
        self._cnn_path = cnn_path or (ARTIFACTS_DIRECTORY / "package_cnn.pt")
        self._model: nn.Module | None = None
        self._device = self._select_device()
        
        # Try to load the CNN model
        try:
            self._load_model()
            logger.info(f"PackageInspector loaded CNN model from {self._cnn_path}")
        except FileNotFoundError:
            logger.warning(
                f"CNN model not found at {self._cnn_path}. "
                "Run 'python -m src.ai.pricing_engine --train-cnn data/raw/damaged_box' to train it. "
                "All packages will be marked as intact until model is available."
            )
    
    def _select_device(self) -> torch.device:
        """Select the best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _load_model(self) -> None:
        """Load the trained ResNet18 CNN model."""
        if not self._cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found: {self._cnn_path}")
        
        checkpoint = torch.load(self._cnn_path, map_location=self._device)
        
        # Load ResNet18 architecture
        from torchvision.models import resnet18
        self._model = resnet18(weights=None)
        
        # Determine the number of output classes from the checkpoint
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Check the final layer size
            if "fc.weight" in state_dict:
                num_classes = state_dict["fc.weight"].shape[0]
            else:
                num_classes = 2  # Default to binary
        else:
            num_classes = 2  # Default to binary
        
        # Replace final layer to match checkpoint
        in_features = self._model.fc.in_features
        self._model.fc = nn.Linear(in_features, num_classes)
        
        # Load trained weights
        if "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)
        
        self._model.to(self._device)
        self._model.eval()
        
        logger.info(f"PackageInspector loaded CNN with {num_classes} output classes")
    
    def inspect(self, orders: list[Order], context: 'SimulationContext') -> list[Order]:
        """
        Inspect packages using CNN model.
        
        For now, since we don't have actual package images in the simulation,
        we simulate rejections. In a real deployment, this would:
        1. Load package images from order.package_image path
        2. Preprocess images (resize to 224x224, normalize)
        3. Run inference with the CNN
        4. Set order.inspection_passed based on prediction
        """
        intact: list[Order] = []
        rejected: list[Order] = []
        
        for order in orders:
            # TODO: In production, this would load and classify actual images
            if self._model is None:
                # No model loaded, simulate inspection using configured rejection rate
                order.inspection_passed = random.random() > PACKAGE_DAMAGE_REJECTION_RATE
            else:
                # Model loaded: run actual CNN inference
                # For now, still using simulation until we have real images
                order.inspection_passed = random.random() > PACKAGE_DAMAGE_REJECTION_RATE
            
            if order.inspection_passed:
                intact.append(order)
            else:
                rejected.append(order)
                context.rejected_orders.append(order)
                logger.debug(f"Order {order.order_id} rejected: damaged package detected")
        
        logger.info(f"PackageInspector: {len(intact)}/{len(orders)} packages passed inspection, {len(rejected)} rejected")
        return intact


class RLEnvironment:
    """Simplified RL environment wrapper for pricing."""
    
    def __init__(self, env: PricingEnv) -> None:
        self._env = env
        self._env.reset()

    def update(self, context: SimulationContext) -> np.ndarray:
        """Update environment state based on current simulation context."""
        queue_length = len(context.pending_orders)
        # Get current tick's demand forecast value
        demand_index = context.tick % len(self._env.demand_forecast)
        demand_value = self._env.demand_forecast[demand_index]
        
        # Encode as state matrix
        state_matrix = self._env.encode_context(
            queue_length=queue_length,
            demand_value=demand_value
        )
        return state_matrix


class RLAgent:
    """Q-Learning agent for dynamic pricing (simplified for now)."""
    
    def __init__(self, q_table_path: Path | None = None) -> None:
        self._q_table_path = q_table_path or (ARTIFACTS_DIRECTORY / "q_table.npy")
        self._q_table: np.ndarray | None = None
        self._action_multipliers = SURGE_MULTIPLIERS
        
        # Try to load Q-table
        try:
            self._q_table = np.load(self._q_table_path)
            logger.info(f"RLAgent loaded Q-table from {self._q_table_path}")
        except FileNotFoundError:
            logger.warning(
                f"Q-table not found at {self._q_table_path}. "
                "Run 'python -m src.ai.pricing_engine --train-q-table' to train it. "
                "Using default pricing (1.0x) until Q-table is available."
            )

    def act(self, state: np.ndarray | int) -> float:
        """Select surge multiplier based on current state."""
        if self._q_table is None:
            # No Q-table loaded, use default pricing
            return DEFAULT_SURGE_MULTIPLIER
        
        # Convert state matrix to state index
        if isinstance(state, np.ndarray):
            state_index = int(np.argmax(state))
        else:
            state_index = int(state)
        
        # Select best action from Q-table
        action_index = int(np.argmax(self._q_table[state_index]))
        surge_multiplier = float(self._action_multipliers[action_index])
        
        return surge_multiplier


class DispatcherAdapter:
    """MILP-based CVRPTW dispatcher with A* routing."""
    
    def __init__(self, milp: CVRPTWDispatcherMilp) -> None:
        self._milp = milp

    def dispatch(self, context: SimulationContext) -> None:
        """Dispatch pending orders to vehicles using CVRPTW MILP solver."""
        if not context.pending_orders:
            return
        
        result = self._milp.dispatch(
            vehicles=context.vehicles,
            orders=context.pending_orders,
            tick=context.tick
        )
        
        if result is None:
            logger.debug(f"Tick {context.tick}: No feasible dispatch solution found")
            return
        
        # Move dispatched orders from pending to dispatched list
        dispatched_ids = {a.order_id for a in result.assignments}
        dispatched_orders = []
        remaining_pending = []
        
        for order in context.pending_orders:
            if order.order_id in dispatched_ids:
                # Mark order as dispatched
                order.dispatch_tick = context.tick
                dispatched_orders.append(order)
                context.dispatched_orders.append(order)
            else:
                remaining_pending.append(order)
        
        context.pending_orders = remaining_pending
        
        # Update vehicle positions with assigned routes
        for assignment in result.assignments:
            position = context.positions.get(assignment.vehicle_id)
            if position:
                position.route = assignment.route
                # Find the vehicle's current node (pickup) in the route
                # and start from the next node after it
                vehicle = next((v for v in context.vehicles if v.vehicle_id == assignment.vehicle_id), None)
                if vehicle:
                    # Sync position.current_node with vehicle.current_node (pickup node)
                    position.current_node = vehicle.current_node
                    
                    pickup_idx = None
                    for idx, node in enumerate(assignment.route.nodes):
                        if node.id == vehicle.current_node:
                            pickup_idx = idx
                            break
                    
                    if pickup_idx is not None and pickup_idx < len(assignment.route.nodes) - 1:
                        # Start from next node after pickup
                        position.route_index = pickup_idx + 1
                    else:
                        # Vehicle is already at destination or route is invalid
                        position.route_index = len(assignment.route.nodes) - 1
        
        logger.info(
            f"Tick {context.tick}: dispatched {len(dispatched_orders)} orders, "
            f"{len(context.pending_orders)} orders still pending, "
            f"{len(context.dispatched_orders)} in transit"
        )


class MovementAdapter:
    """Moves vehicles along their assigned routes."""
    
    def __init__(self, engine: MovementEngine) -> None:
        self._engine = engine

    def move(self, context: SimulationContext) -> dict:
        """Move all vehicles one step along their routes and return results."""
        return self._engine.move_all(list(context.vehicles), context.positions)


def _build_fleet(graph: Graph, n: int = 5) -> tuple[list[Vehicle], dict[str, VehiclePosition]]:
    """
    Build initial fleet using FleetGenerator.
    
    Args:
        graph: Road network graph
        n: Number of vehicles to create
        
    Returns:
        Tuple of (vehicles list, positions dict)
    """
    generator = FleetGenerator()
    node_ids = list(graph.graph.nodes())
    
    if not node_ids:
        logger.error("Graph has no nodes! Cannot create fleet.")
        return [], {}
    
    vehicles = generator.generate(count=n, graph_nodes=node_ids)
    
    # Initialize positions for each vehicle
    positions: dict[str, VehiclePosition] = {}
    for vehicle in vehicles:
        positions[vehicle.vehicle_id] = VehiclePosition(
            current_node=vehicle.current_node
        )
    
    logger.info(f"Created fleet of {len(vehicles)} vehicles")
    return vehicles, positions


def main() -> None:
    """
    Master simulation script - integrates all simulation components.
    
    Flow:
    0. Check and run first-time setup if needed
    1. Load infrastructure (graph, models)
    2. Initialize components:
       - Data & ML: OrderSampler, DelayPredictor, DemandForecast
       - Vision & RL: PackageInspector (CNN), RLAgent (Q-table)
       - Optimization: MILP Dispatcher, A* Router, MovementEngine
    3. Run simulation loop
    """
    logger.info("=" * 80)
    logger.info("LogiSim-AI - Multi-Agent Last-Mile Delivery Simulation")
    logger.info("=" * 80)
    
    # ============================================================
    # FIRST-TIME SETUP CHECK
    # ============================================================
    if not _check_and_run_setup():
        logger.error("Setup failed. Cannot continue.")
        return
    
    # ============================================================
    # Infrastructure - Load road network graph
    # ============================================================
    logger.info("\n[Infrastructure] Loading road network graph...")
    graph = Graph()
    graph.load()
    logger.info(f"Graph loaded: {len(graph.graph.nodes())} nodes, {len(graph.graph.edges())} edges")
    
    # ============================================================
    # Data & ML - Demand forecast and delay prediction
    # ============================================================
    logger.info("\n[Data & ML] Loading demand forecast and delay predictor...")
    demand_forecast = load_demand_forecast()
    logger.info(f"Demand forecast loaded: {len(demand_forecast)} hourly values")
    
    predictor = DelayPredictor()
    
    # ============================================================
    # Vision & RL - CNN and Q-learning
    # ============================================================
    logger.info("\n[Vision & RL] Initializing vision and RL components...")
    
    # Initialize pricing environment for RL
    pricing_env = PricingEnv(demand_forecast=demand_forecast)
    
    # Initialize CNN-based package inspector
    package_inspector = PackageInspector()
    
    # Initialize Q-learning agent for pricing
    rl_agent = RLAgent()
    
    # ============================================================
    # Optimization - A* routing and MILP dispatcher
    # ============================================================
    logger.info("\n[Optimization] Setting up routing and dispatch optimization...")
    
    # A* routing with travel time heuristic
    heuristic = TravelTimeHeuristic(ROUTING_AVERAGE_SPEED_KMH)
    astar_routing = AStarRouting(graph, heuristic, DelayMap())
    
    # MILP-based CVRPTW dispatcher
    dispatcher_config = DispatcherConfig()
    milp_dispatcher = CVRPTWDispatcherMilp(astar_routing, dispatcher_config)
    
    # Movement engine for vehicle navigation
    movement_engine = MovementEngine(graph)
    
    # ============================================================
    # ALL: Simulation Executor - Wires everything together
    # ============================================================
    logger.info("\n[Integration] Building simulation executor...")
    
    executor = SimulationExecutor(
        sampler=OrderSampler(orders_per_tick=ORDERS_PER_TICK),           # Data & ML
        inspector=package_inspector,                        # Vision & RL
        predictor=predictor,                                # Data & ML
        dispatcher=DispatcherAdapter(milp_dispatcher),      # Optimization
        rl_environment=RLEnvironment(pricing_env),          # Vision & RL
        rl_agent=rl_agent,                                  # Vision & RL
        movement_engine=MovementAdapter(movement_engine),   # Optimization
    )
    
    # ============================================================
    # Fleet initialization
    # ============================================================
    logger.info("\n[Optimization] Initializing vehicle fleet...")
    vehicles, positions = _build_fleet(graph, n=FLEET_SIZE)
    
    # ============================================================
    # Create simulation context
    # ============================================================
    context = SimulationContext(
        graph=graph,
        current_time=datetime.now(),
        vehicles=vehicles,
        positions=positions,
    )
    
    # ============================================================
    # RUN SIMULATION
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting simulation: {MAX_SIMULATION_STEPS} ticks, {len(vehicles)} vehicles")
    logger.info("=" * 80 + "\n")
    
    for tick in range(MAX_SIMULATION_STEPS):
        context.current_time += timedelta(minutes=SIMULATION_STEP_MINUTES)
        
        # Execute one simulation tick
        executor.execute_tick(context)
        
        # Log progress every 100 ticks
        if tick % 100 == 0:
            logger.info(
                f"Tick {tick:4d} | "
                f"Pending: {len(context.pending_orders):3d} | "
                f"Dispatched: {len(context.dispatched_orders):3d} | "
                f"Delivered: {len(context.delivered_orders):3d} | "
                f"Surge: {context.surge_multiplier:.2f}x"
            )
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 80)
    
    # Order counts
    total_sampled = context.statistics.sampled_orders or (
        len(context.pending_orders) + 
        len(context.dispatched_orders) + 
        len(context.delivered_orders) + 
        len(context.rejected_orders)
    )
    logger.info(f"Total Ticks: {MAX_SIMULATION_STEPS}")
    logger.info(f"Total Orders Sampled: {total_sampled}")
    logger.info(f"  - Rejected (Damaged): {len(context.rejected_orders)} ({100 * len(context.rejected_orders) / max(1, total_sampled):.1f}%)")
    logger.info(f"  - Delivered: {len(context.delivered_orders)} ({100 * len(context.delivered_orders) / max(1, total_sampled):.1f}%)")
    logger.info(f"  - In Transit: {len(context.dispatched_orders)} ({100 * len(context.dispatched_orders) / max(1, total_sampled):.1f}%)")
    logger.info(f"  - Still Pending: {len(context.pending_orders)} ({100 * len(context.pending_orders) / max(1, total_sampled):.1f}%)")
    logger.info("")
    
    # Performance metrics
    logger.info("Performance Metrics:")
    # Calculate success rate based on processed orders (excluding pending)
    processed_orders = len(context.delivered_orders) + len(context.rejected_orders) + len(context.dispatched_orders)
    delivery_rate = len(context.delivered_orders) / max(1, processed_orders)
    logger.info(f"  - Delivery Success Rate: {100 * delivery_rate:.1f}%")
    
    avg_delivery_time = context.statistics.get_avg_delivery_time()
    logger.info(f"  - Average Delivery Time: {avg_delivery_time:.1f} minutes")
    
    total_distance_km = context.statistics.total_distance / 1000.0
    logger.info(f"  - Total Distance Traveled: {total_distance_km:.1f} km")
    
    avg_distance = context.statistics.get_avg_distance_per_delivery()
    logger.info(f"  - Average Distance per Delivery: {avg_distance:.2f} km")
    logger.info("")
    
    # Fleet utilization
    logger.info("Fleet Utilization:")
    logger.info(f"  - Total Fleet Size: {len(vehicles)} vehicles")
    if len(vehicles) > 0:
        avg_deliveries_per_vehicle = len(context.delivered_orders) / len(vehicles)
        logger.info(f"  - Average Deliveries per Vehicle: {avg_deliveries_per_vehicle:.1f}")
    logger.info("")
    
    # Economic metrics
    logger.info("Economic Metrics:")
    avg_surge = context.statistics.get_avg_surge()
    logger.info(f"  - Average Surge Multiplier: {avg_surge:.2f}x")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
