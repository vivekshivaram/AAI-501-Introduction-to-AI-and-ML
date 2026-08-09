"""
experiment_runner.py - Systematic simulation experiments and analysis

Runs multiple simulations with varying parameters to analyze:
- Fleet size impact
- Vehicle capacity impact  
- Order volume impact
- Q-learning parameter sensitivity
- Vehicle speed variations

Generates comparative plots and statistical analysis.

Usage:
    python -m src.experiment_runner --all
    python -m src.experiment_runner --fleet-size
    python -m src.experiment_runner --capacity
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime, timedelta
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.config import (
    ARTIFACTS_DIRECTORY,
    DATA_OUTPUTS_DIRECTORY,
    MAX_SIMULATION_STEPS,
    FLEET_SIZE,
    ORDERS_PER_TICK,
    ROUTING_AVERAGE_SPEED_KMH,
    Q_LEARNING_ALPHA,
    Q_LEARNING_EPSILON,
    VEHICLE_TYPES,
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
from src.simulation.fleet_generator import FleetGenerator
from src.ai.pricing_env import PricingEnv, load_demand_forecast
from src.utils.logger import get_logger
from src.main import (
    OrderSampler,
    PackageInspector,
    RLAgent,
    DispatcherAdapter,
    RLEnvironment,
    MovementAdapter,
)

logger = get_logger(__name__)


class ExperimentResults:
    """Container for simulation experiment results."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.total_orders = 0
        self.delivered = 0
        self.rejected = 0
        self.in_transit = 0
        self.pending = 0
        self.delivery_rate = 0.0  # Delivered / (Delivered + Rejected) - excludes in-transit bias
        self.dispatch_efficiency = 0.0  # (Delivered + Rejected + In-Transit) / Total
        self.throughput = 0.0  # Deliveries per tick
        self.avg_delivery_time = 0.0
        self.total_distance_km = 0.0
        self.avg_distance_per_delivery = 0.0
        self.avg_surge = 0.0
        self.fleet_utilization = 0.0
        
    def to_dict(self) -> dict[str, Any]:
        return {
            'config': self.config,
            'metrics': {
                'total_orders': self.total_orders,
                'delivered': self.delivered,
                'rejected': self.rejected,
                'in_transit': self.in_transit,
                'pending': self.pending,
                'delivery_rate': self.delivery_rate,
                'dispatch_efficiency': self.dispatch_efficiency,
                'throughput': self.throughput,
                'avg_delivery_time': self.avg_delivery_time,
                'total_distance_km': self.total_distance_km,
                'avg_distance_per_delivery': self.avg_distance_per_delivery,
                'avg_surge': self.avg_surge,
                'fleet_utilization': self.fleet_utilization,
            }
        }


def run_single_simulation(
    fleet_size: int = FLEET_SIZE,
    orders_per_tick: int = ORDERS_PER_TICK,
    routing_speed: float = ROUTING_AVERAGE_SPEED_KMH,
    q_alpha: float = Q_LEARNING_ALPHA,
    q_epsilon: float = Q_LEARNING_EPSILON,
    vehicle_capacity_multiplier: float = 1.0,
) -> ExperimentResults:
    """
    Run a single simulation with specified parameters.
    
    Args:
        fleet_size: Number of vehicles
        orders_per_tick: Orders sampled per tick
        routing_speed: Average routing speed in km/h
        q_alpha: Q-learning learning rate
        q_epsilon: Q-learning exploration rate
        vehicle_capacity_multiplier: Multiplier for all vehicle capacities
        
    Returns:
        ExperimentResults with collected metrics
    """
    config = {
        'fleet_size': fleet_size,
        'orders_per_tick': orders_per_tick,
        'routing_speed': routing_speed,
        'q_alpha': q_alpha,
        'q_epsilon': q_epsilon,
        'vehicle_capacity_multiplier': vehicle_capacity_multiplier,
    }
    
    logger.info(f"Running simulation with config: {config}")
    
    # Load infrastructure
    graph = Graph()
    graph.load()
    
    demand_forecast = load_demand_forecast()
    predictor = DelayPredictor()
    
    # Initialize components
    pricing_env = PricingEnv(demand_forecast=demand_forecast)
    package_inspector = PackageInspector()
    rl_agent = RLAgent()
    
    heuristic = TravelTimeHeuristic(routing_speed)
    astar_routing = AStarRouting(graph, heuristic, DelayMap())
    
    dispatcher_config = DispatcherConfig()
    milp_dispatcher = CVRPTWDispatcherMilp(astar_routing, dispatcher_config)
    movement_engine = MovementEngine(graph)
    
    executor = SimulationExecutor(
        sampler=OrderSampler(orders_per_tick=orders_per_tick),
        inspector=package_inspector,
        predictor=predictor,
        dispatcher=DispatcherAdapter(milp_dispatcher),
        rl_environment=RLEnvironment(pricing_env),
        rl_agent=rl_agent,
        movement_engine=MovementAdapter(movement_engine),
    )
    
    # Build fleet with custom capacity
    generator = FleetGenerator()
    if vehicle_capacity_multiplier != 1.0:
        # Temporarily modify capacities
        original_capacities = generator.VEHICLE_CAPACITY.copy()
        generator.VEHICLE_CAPACITY = {
            vtype: cap * vehicle_capacity_multiplier 
            for vtype, cap in original_capacities.items()
        }
    
    node_ids = list(graph.graph.nodes())
    vehicles = generator.generate(count=fleet_size, graph_nodes=node_ids)
    
    # Restore original capacities if modified
    if vehicle_capacity_multiplier != 1.0:
        generator.VEHICLE_CAPACITY = original_capacities
    
    positions = {v.vehicle_id: VehiclePosition(current_node=v.current_node) for v in vehicles}
    
    # Create context
    context = SimulationContext(
        graph=graph,
        current_time=datetime.now(),
        vehicles=vehicles,
        positions=positions,
    )
    
    # Run simulation
    for tick in range(MAX_SIMULATION_STEPS):
        context.current_time += timedelta(minutes=1)
        executor.execute_tick(context)
    
    # Collect results
    results = ExperimentResults(config)
    results.total_orders = (
        len(context.pending_orders) + 
        len(context.dispatched_orders) + 
        len(context.delivered_orders) + 
        len(context.rejected_orders)
    )
    results.delivered = len(context.delivered_orders)
    results.rejected = len(context.rejected_orders)
    results.in_transit = len(context.dispatched_orders)
    results.pending = len(context.pending_orders)
    
    # IMPROVED METRIC: Delivery completion rate (excludes in-transit bias)
    # This better reflects actual completion efficiency
    completed = results.delivered + results.rejected
    results.delivery_rate = results.delivered / max(1, completed) if completed > 0 else 0.0
    
    # ADDITIONAL METRICS for better analysis:
    # - Dispatch efficiency: how many orders were processed vs total sampled
    total_sampled = results.total_orders
    dispatched_or_completed = results.delivered + results.rejected + results.in_transit
    results.dispatch_efficiency = dispatched_or_completed / max(1, total_sampled)
    
    # - Throughput: deliveries per tick
    results.throughput = results.delivered / MAX_SIMULATION_STEPS
    
    results.avg_delivery_time = context.statistics.get_avg_delivery_time()
    results.total_distance_km = context.statistics.total_distance / 1000.0
    results.avg_distance_per_delivery = context.statistics.get_avg_distance_per_delivery()
    results.avg_surge = context.statistics.get_avg_surge()
    results.fleet_utilization = results.delivered / max(1, fleet_size)
    
    logger.info(f"Simulation complete: {results.delivered}/{results.total_orders} delivered, "
                f"rate={results.delivery_rate:.2%}, surge={results.avg_surge:.2f}x")
    
    return results


def experiment_fleet_size() -> list[ExperimentResults]:
    """Vary fleet size and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: Fleet Size Variation")
    logger.info("="*80)
    
    fleet_sizes = [5, 10, 15, 20, 30, 50]
    results = []
    
    for size in fleet_sizes:
        result = run_single_simulation(fleet_size=size)
        results.append(result)
    
    return results


def experiment_order_volume() -> list[ExperimentResults]:
    """Vary order volume per tick and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: Order Volume Variation")
    logger.info("="*80)
    
    # More granular to identify aberrations in delivery rate
    order_volumes = [1, 2, 3, 5, 7, 10, 12, 15, 20]
    results = []
    
    for volume in order_volumes:
        result = run_single_simulation(orders_per_tick=volume)
        results.append(result)
    
    return results


def experiment_routing_speed() -> list[ExperimentResults]:
    """Vary routing speed and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 3: Routing Speed Variation")
    logger.info("="*80)
    
    speeds = [30.0, 45.0, 60.0, 80.0, 100.0]
    results = []
    
    for speed in speeds:
        result = run_single_simulation(routing_speed=speed)
        results.append(result)
    
    return results


def experiment_vehicle_capacity() -> list[ExperimentResults]:
    """Vary vehicle capacity and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 4: Vehicle Capacity Variation")
    logger.info("="*80)
    
    capacity_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    for multiplier in capacity_multipliers:
        result = run_single_simulation(vehicle_capacity_multiplier=multiplier)
        results.append(result)
    
    return results


def experiment_q_learning_alpha() -> list[ExperimentResults]:
    """Vary Q-learning alpha (learning rate) and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 5: Q-Learning Alpha Variation")
    logger.info("="*80)
    
    # More granular to capture learning rate sweet spot
    alphas = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = []
    
    for alpha in alphas:
        result = run_single_simulation(q_alpha=alpha)
        results.append(result)
    
    return results


def experiment_q_learning_epsilon() -> list[ExperimentResults]:
    """Vary Q-learning epsilon (exploration rate) and measure impact."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 6: Q-Learning Epsilon Variation")
    logger.info("="*80)
    
    # More granular around 0.20 to identify aberration cause
    epsilons = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]
    results = []
    
    for epsilon in epsilons:
        result = run_single_simulation(q_epsilon=epsilon)
        results.append(result)
    
    return results


def experiment_fleet_order_combination() -> list[ExperimentResults]:
    """Test fleet size under varying order volume stress (combined analysis)."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 7: Fleet-Order Volume Stress Test")
    logger.info("="*80)
    
    # Test small, medium, large fleets under low/medium/high order volume
    configurations = [
        # Small fleet (10 vehicles)
        (10, 1),   # Low load
        (10, 5),   # Medium load
        (10, 10),  # High load
        (10, 20),  # Extreme load
        # Medium fleet (20 vehicles)
        (20, 1),
        (20, 5),
        (20, 10),
        (20, 20),
        # Large fleet (50 vehicles)
        (50, 1),
        (50, 5),
        (50, 10),
        (50, 20),
    ]
    
    results = []
    for fleet_size, order_volume in configurations:
        logger.info(f"Testing fleet={fleet_size}, orders/tick={order_volume}")
        result = run_single_simulation(fleet_size=fleet_size, orders_per_tick=order_volume)
        results.append(result)
    
    return results


def experiment_speed_capacity_combination() -> list[ExperimentResults]:
    """Test routing speed under varying vehicle capacity (combined analysis)."""
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 8: Speed-Capacity Trade-off Analysis")
    logger.info("="*80)
    
    # Test slow/medium/fast vehicles with small/medium/large capacity
    configurations = [
        # Slow vehicles (30 km/h)
        (30.0, 0.5),   # Small capacity
        (30.0, 1.0),   # Normal capacity
        (30.0, 2.0),   # Large capacity
        # Medium vehicles (60 km/h)
        (60.0, 0.5),
        (60.0, 1.0),
        (60.0, 2.0),
        # Fast vehicles (100 km/h)
        (100.0, 0.5),
        (100.0, 1.0),
        (100.0, 2.0),
    ]
    
    results = []
    for speed, capacity in configurations:
        logger.info(f"Testing speed={speed} km/h, capacity={capacity}x")
        result = run_single_simulation(routing_speed=speed, vehicle_capacity_multiplier=capacity)
        results.append(result)
    
    return results


def plot_experiment_results(
    results: list[ExperimentResults],
    x_param: str,
    x_label: str,
    title: str,
    output_file: str,
):
    """
    Generate multi-panel plot for experiment results.
    
    Args:
        results: List of experiment results
        x_param: Parameter name to use as x-axis (from config dict)
        x_label: Label for x-axis
        title: Overall plot title
        output_file: Output filename for plot
    """
    x_values = [r.config[x_param] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Delivery Rate
    ax = axes[0, 0]
    delivery_rates = [r.delivery_rate * 100 for r in results]
    dispatch_efficiencies = [r.dispatch_efficiency * 100 for r in results]
    
    # Plot both metrics for comparison
    ax.plot(x_values, delivery_rates, marker='o', linewidth=2, markersize=8, color='#2ecc71', label='Completion Rate')
    ax.plot(x_values, dispatch_efficiencies, marker='s', linewidth=2, markersize=8, color='#3498db', label='Dispatch Efficiency', linestyle='--')
    
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Rate (%)', fontsize=11)
    ax.set_title('Delivery Completion vs Dispatch Efficiency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Completion = Delivered/(Delivered+Rejected)\nDispatch = Processed/Total', 
            transform=ax.transAxes, fontsize=7, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Average Delivery Time
    ax = axes[0, 1]
    delivery_times = [r.avg_delivery_time for r in results]
    ax.plot(x_values, delivery_times, marker='s', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Avg Delivery Time (minutes)', fontsize=11)
    ax.set_title('Average Delivery Time (Completed Only)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, 'Note: Only completed deliveries\nMay show selection bias', 
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    # Plot 3: Average Surge Multiplier
    ax = axes[0, 2]
    surges = [r.avg_surge for r in results]
    ax.plot(x_values, surges, marker='^', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Average Surge Multiplier', fontsize=11)
    ax.set_title('Dynamic Pricing - Surge Multiplier', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Fleet Utilization
    ax = axes[1, 0]
    utilization = [r.fleet_utilization for r in results]
    ax.plot(x_values, utilization, marker='D', linewidth=2, markersize=8, color='#9b59b6')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Deliveries per Vehicle', fontsize=11)
    ax.set_title('Fleet Utilization', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Total Distance
    ax = axes[1, 1]
    distances = [r.total_distance_km for r in results]
    ax.plot(x_values, distances, marker='p', linewidth=2, markersize=8, color='#f39c12')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Total Distance (km)', fontsize=11)
    ax.set_title('Total Distance Traveled', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Order Distribution (Stacked Bar)
    ax = axes[1, 2]
    delivered = [r.delivered for r in results]
    in_transit = [r.in_transit for r in results]
    pending = [r.pending for r in results]
    rejected = [r.rejected for r in results]
    
    width = 0.6
    x_pos = np.arange(len(x_values))
    
    ax.bar(x_pos, delivered, width, label='Delivered', color='#2ecc71')
    ax.bar(x_pos, in_transit, width, bottom=delivered, label='In Transit', color='#3498db')
    bottom = np.array(delivered) + np.array(in_transit)
    ax.bar(x_pos, pending, width, bottom=bottom, label='Pending', color='#95a5a6')
    bottom = bottom + np.array(pending)
    ax.bar(x_pos, rejected, width, bottom=bottom, label='Rejected', color='#e74c3c')
    
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Order Count', fontsize=11)
    ax.set_title('Order Distribution', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_values], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = DATA_OUTPUTS_DIRECTORY / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def plot_combined_fleet_order(results: list[ExperimentResults]):
    """Plot fleet-order stress test results as grouped bar chart."""
    # Group by fleet size
    fleet_10 = [r for r in results if r.config['fleet_size'] == 10]
    fleet_20 = [r for r in results if r.config['fleet_size'] == 20]
    fleet_50 = [r for r in results if r.config['fleet_size'] == 50]
    
    order_volumes = [1, 5, 10, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fleet-Order Volume Stress Test Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Completion Rate
    ax = axes[0, 0]
    x = np.arange(len(order_volumes))
    width = 0.25
    ax.bar(x - width, [r.delivery_rate*100 for r in fleet_10], width, label='10 vehicles', color='#3498db')
    ax.bar(x, [r.delivery_rate*100 for r in fleet_20], width, label='20 vehicles', color='#2ecc71')
    ax.bar(x + width, [r.delivery_rate*100 for r in fleet_50], width, label='50 vehicles', color='#9b59b6')
    ax.set_xlabel('Orders per Tick')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Completion Rate Under Load')
    ax.set_xticks(x)
    ax.set_xticklabels(order_volumes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Dispatch Efficiency
    ax = axes[0, 1]
    ax.bar(x - width, [r.dispatch_efficiency*100 for r in fleet_10], width, label='10 vehicles', color='#3498db')
    ax.bar(x, [r.dispatch_efficiency*100 for r in fleet_20], width, label='20 vehicles', color='#2ecc71')
    ax.bar(x + width, [r.dispatch_efficiency*100 for r in fleet_50], width, label='50 vehicles', color='#9b59b6')
    ax.set_xlabel('Orders per Tick')
    ax.set_ylabel('Dispatch Efficiency (%)')
    ax.set_title('Capacity Utilization Under Load')
    ax.set_xticks(x)
    ax.set_xticklabels(order_volumes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Throughput
    ax = axes[1, 0]
    ax.bar(x - width, [r.throughput for r in fleet_10], width, label='10 vehicles', color='#3498db')
    ax.bar(x, [r.throughput for r in fleet_20], width, label='20 vehicles', color='#2ecc71')
    ax.bar(x + width, [r.throughput for r in fleet_50], width, label='50 vehicles', color='#9b59b6')
    ax.set_xlabel('Orders per Tick')
    ax.set_ylabel('Deliveries per Tick')
    ax.set_title('Throughput Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(order_volumes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Fleet Utilization
    ax = axes[1, 1]
    ax.bar(x - width, [r.fleet_utilization for r in fleet_10], width, label='10 vehicles', color='#3498db')
    ax.bar(x, [r.fleet_utilization for r in fleet_20], width, label='20 vehicles', color='#2ecc71')
    ax.bar(x + width, [r.fleet_utilization for r in fleet_50], width, label='50 vehicles', color='#9b59b6')
    ax.set_xlabel('Orders per Tick')
    ax.set_ylabel('Deliveries per Vehicle')
    ax.set_title('Per-Vehicle Efficiency')
    ax.set_xticks(x)
    ax.set_xticklabels(order_volumes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = DATA_OUTPUTS_DIRECTORY / 'experiment_fleet_order_stress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def plot_combined_speed_capacity(results: list[ExperimentResults]):
    """Plot speed-capacity trade-off results as grouped bar chart."""
    # Group by speed
    speed_30 = [r for r in results if r.config['routing_speed'] == 30.0]
    speed_60 = [r for r in results if r.config['routing_speed'] == 60.0]
    speed_100 = [r for r in results if r.config['routing_speed'] == 100.0]
    
    capacities = [0.5, 1.0, 2.0]
    capacity_labels = ['0.5x', '1.0x', '2.0x']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Speed-Capacity Trade-off Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Completion Rate
    ax = axes[0, 0]
    x = np.arange(len(capacities))
    width = 0.25
    ax.bar(x - width, [r.delivery_rate*100 for r in speed_30], width, label='30 km/h', color='#e74c3c')
    ax.bar(x, [r.delivery_rate*100 for r in speed_60], width, label='60 km/h', color='#f39c12')
    ax.bar(x + width, [r.delivery_rate*100 for r in speed_100], width, label='100 km/h', color='#2ecc71')
    ax.set_xlabel('Vehicle Capacity')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Completion Rate by Speed & Capacity')
    ax.set_xticks(x)
    ax.set_xticklabels(capacity_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Delivery Time
    ax = axes[0, 1]
    ax.bar(x - width, [r.avg_delivery_time for r in speed_30], width, label='30 km/h', color='#e74c3c')
    ax.bar(x, [r.avg_delivery_time for r in speed_60], width, label='60 km/h', color='#f39c12')
    ax.bar(x + width, [r.avg_delivery_time for r in speed_100], width, label='100 km/h', color='#2ecc71')
    ax.set_xlabel('Vehicle Capacity')
    ax.set_ylabel('Avg Delivery Time (min)')
    ax.set_title('Delivery Time by Speed & Capacity')
    ax.set_xticks(x)
    ax.set_xticklabels(capacity_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Throughput
    ax = axes[1, 0]
    ax.bar(x - width, [r.throughput for r in speed_30], width, label='30 km/h', color='#e74c3c')
    ax.bar(x, [r.throughput for r in speed_60], width, label='60 km/h', color='#f39c12')
    ax.bar(x + width, [r.throughput for r in speed_100], width, label='100 km/h', color='#2ecc71')
    ax.set_xlabel('Vehicle Capacity')
    ax.set_ylabel('Deliveries per Tick')
    ax.set_title('Throughput Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(capacity_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Total Distance
    ax = axes[1, 1]
    ax.bar(x - width, [r.total_distance_km for r in speed_30], width, label='30 km/h', color='#e74c3c')
    ax.bar(x, [r.total_distance_km for r in speed_60], width, label='60 km/h', color='#f39c12')
    ax.bar(x + width, [r.total_distance_km for r in speed_100], width, label='100 km/h', color='#2ecc71')
    ax.set_xlabel('Vehicle Capacity')
    ax.set_ylabel('Total Distance (km)')
    ax.set_title('Distance Traveled')
    ax.set_xticks(x)
    ax.set_xticklabels(capacity_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = DATA_OUTPUTS_DIRECTORY / 'experiment_speed_capacity_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    plt.close()


def save_results_json(all_results: dict[str, list[ExperimentResults]], output_file: str):
    """Save all experiment results to JSON."""
    json_data = {
        experiment_name: [r.to_dict() for r in results]
        for experiment_name, results in all_results.items()
    }
    
    output_path = DATA_OUTPUTS_DIRECTORY / output_file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def main():
    """Run experiments based on command line arguments."""
    parser = argparse.ArgumentParser(description='Run simulation experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--fleet-size', action='store_true', help='Fleet size experiment')
    parser.add_argument('--order-volume', action='store_true', help='Order volume experiment')
    parser.add_argument('--routing-speed', action='store_true', help='Routing speed experiment')
    parser.add_argument('--capacity', action='store_true', help='Vehicle capacity experiment')
    parser.add_argument('--q-alpha', action='store_true', help='Q-learning alpha experiment')
    parser.add_argument('--q-epsilon', action='store_true', help='Q-learning epsilon experiment')
    parser.add_argument('--combined', action='store_true', help='Run combined/stress test experiments')
    parser.add_argument('--fleet-order', action='store_true', help='Fleet-order stress test experiment')
    parser.add_argument('--speed-capacity', action='store_true', help='Speed-capacity trade-off experiment')
    
    args = parser.parse_args()
    
    # If no specific experiment selected, run all basic experiments (not combined)
    if not any([args.fleet_size, args.order_volume, args.routing_speed, 
                args.capacity, args.q_alpha, args.q_epsilon, args.combined,
                args.fleet_order, args.speed_capacity]):
        args.all = True
    
    all_results = {}
    
    if args.all or args.fleet_size:
        results = experiment_fleet_size()
        all_results['fleet_size'] = results
        plot_experiment_results(
            results,
            x_param='fleet_size',
            x_label='Fleet Size (vehicles)',
            title='Impact of Fleet Size on Delivery Performance',
            output_file='experiment_fleet_size.png'
        )
    
    if args.all or args.order_volume:
        results = experiment_order_volume()
        all_results['order_volume'] = results
        plot_experiment_results(
            results,
            x_param='orders_per_tick',
            x_label='Orders per Tick',
            title='Impact of Order Volume on Delivery Performance',
            output_file='experiment_order_volume.png'
        )
    
    if args.all or args.routing_speed:
        results = experiment_routing_speed()
        all_results['routing_speed'] = results
        plot_experiment_results(
            results,
            x_param='routing_speed',
            x_label='Routing Speed (km/h)',
            title='Impact of Routing Speed on Delivery Performance',
            output_file='experiment_routing_speed.png'
        )
    
    if args.all or args.capacity:
        results = experiment_vehicle_capacity()
        all_results['vehicle_capacity'] = results
        plot_experiment_results(
            results,
            x_param='vehicle_capacity_multiplier',
            x_label='Vehicle Capacity Multiplier',
            title='Impact of Vehicle Capacity on Delivery Performance',
            output_file='experiment_vehicle_capacity.png'
        )
    
    if args.all or args.q_alpha:
        results = experiment_q_learning_alpha()
        all_results['q_learning_alpha'] = results
        plot_experiment_results(
            results,
            x_param='q_alpha',
            x_label='Q-Learning Alpha (Learning Rate)',
            title='Impact of Q-Learning Alpha on Pricing & Performance',
            output_file='experiment_q_alpha.png'
        )
    
    if args.all or args.q_epsilon:
        results = experiment_q_learning_epsilon()
        all_results['q_learning_epsilon'] = results
        plot_experiment_results(
            results,
            x_param='q_epsilon',
            x_label='Q-Learning Epsilon (Exploration Rate)',
            title='Impact of Q-Learning Epsilon on Pricing & Performance',
            output_file='experiment_q_epsilon.png'
        )
    
    # Combined/Advanced Experiments (only run if explicitly requested or with --combined flag)
    if args.combined or args.fleet_order:
        results = experiment_fleet_order_combination()
        all_results['fleet_order_stress'] = results
        plot_combined_fleet_order(results)
    
    if args.combined or args.speed_capacity:
        results = experiment_speed_capacity_combination()
        all_results['speed_capacity_tradeoff'] = results
        plot_combined_speed_capacity(results)
    
    # Save all results to JSON
    save_results_json(all_results, 'experiment_results.json')
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {DATA_OUTPUTS_DIRECTORY}")


if __name__ == "__main__":
    main()
