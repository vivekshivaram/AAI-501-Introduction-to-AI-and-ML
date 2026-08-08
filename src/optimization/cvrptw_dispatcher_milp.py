"""
cvrptw_dispatcher_milp.py
Simplified MILP-based CVRPTW dispatcher.
This implementation intentionally solves only the vehicle assignment
problem using Mixed Integer Linear Programming (PuLP).
Actual routing is delegated to the existing A* routing engine.
"""

from __future__ import annotations
from typing import Dict, Tuple
import pulp

from src.optimization.astar_routing import AStarRouting
from src.optimization.dispatcher_config import DispatcherConfig
from src.optimization.dispatch_result import (DispatchAssignment, DispatchResult)

from src.models.order import Order
from src.models.vehicle import Vehicle
from src.simulation.route import Route

from src.routing.path import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CVRPTWDispatcherMilp:
    """
    Simplified CVRPTW Dispatcher.
    Responsibilities
    ----------------
    * Compute travel costs using A*
    * Solve assignment using MILP
    * Construct routes
    * Update simulation objects
    """

    def __init__(self, router: AStarRouting, config: DispatcherConfig | None = None,) -> None:
        self._router = router
        self._config = config or DispatcherConfig()
        #
        # (vehicle_id, order_id) -> Path objects
        #
        self._paths: Dict[Tuple[str, str], Tuple[Path, Path],] = {}
        #
        # (vehicle_id, order_id) -> scalar objective cost
        #
        self._costs: Dict[Tuple[str, str], float,] = {}
    # ------------------------------------------------------------------

    def dispatch(self, vehicles: list[Vehicle], orders: list[Order], tick: int) -> DispatchResult:
        """
        Dispatch available vehicles.
        Parameters
        ----------
        vehicles
            Fleet
        orders
            Pending orders
        tick
            Current simulation tick
        """
        available_vehicles = [ v for v in vehicles if v.available ]
        pending_orders = [ o for o in orders if not o.delivered ]

        if not available_vehicles:
            logger.info("No vehicles available.")
            return
            
        if not pending_orders:
            logger.info("No pending orders.")
            return

        logger.info(f"Dispatcher started: {len(available_vehicles)} vehicles, {len(pending_orders)} orders")
        self._build_cost_matrix(available_vehicles, pending_orders)
        problem = pulp.LpProblem("VehicleAssignment", pulp.LpMinimize)
        assignment = self._create_variables(available_vehicles, pending_orders)
        self._build_objective(problem, assignment)
        self._assignment_constraints(problem, assignment, available_vehicles, pending_orders)
        self._capacity_constraints(problem, assignment, available_vehicles, pending_orders)
        (status, dispatch_assignments) = self._solve(problem, assignment, available_vehicles, pending_orders, tick)
        return DispatchResult(tick=tick, status=status, assignments=dispatch_assignments)

        
    def _build_cost_matrix(self, vehicles: list[Vehicle], orders: list[Order]) -> None:
        """
        Computes
        Vehicle
            ↓
        Pickup
            ↓
        Delivery
        for every feasible assignment.
        """
        self.reset()

        logger.info("Building cost matrix...")

        for vehicle in vehicles:
            for order in orders:
                # Skip if order weight exceeds vehicle capacity
                if order.weight > vehicle.capacity:
                    logger.debug(
                        f"Order {order.order_id} ({order.weight:.1f} kg) exceeds "
                        f"vehicle {vehicle.vehicle_id} capacity ({vehicle.capacity:.1f} kg)"
                    )
                    continue
                    
                path_to_pickup = self._router.shortest_path(vehicle.current_node, order.pickup_node)
                path_to_delivery = self._router.shortest_path(order.pickup_node, order.delivery_node)
                self._paths[(vehicle.vehicle_id, order.order_id)] = (path_to_pickup, path_to_delivery)
                travel_cost = (path_to_pickup.total_cost + path_to_delivery.total_cost)
                if (self._config.allow_late_delivery and order.predicted_delay > 0 ):
                    travel_cost += (order.predicted_delay * self._config.lateness_penalty)

                self._costs[(vehicle.vehicle_id,order.order_id)] = travel_cost

        logger.info(f"Built cost matrix: {len(self._costs)} feasible vehicle-order pairs")

    def _create_variables(self, vehicles: list[Vehicle], orders: list[Order]):
        """
        Create binary variables only for feasible vehicle-order pairs.
        A pair is feasible if it exists in the cost matrix (passed capacity check).
        """
        assignment = {}
        for vehicle in vehicles:
            for order in orders:
                key = (vehicle.vehicle_id, order.order_id)
                # Only create variables for feasible assignments
                if key in self._costs:
                    assignment[key] = pulp.LpVariable(
                        f"x_{vehicle.vehicle_id}_{order.order_id}", 
                        lowBound=0, 
                        upBound=1, 
                        cat="Binary"
                    )
        logger.info(f"Created {len(assignment)} feasible assignment variables")
        return assignment

    def _build_objective(self, problem, assignment) -> None:
        """
        Minimize total cost while encouraging assignments.
        The objective balances minimizing routing cost with maximizing number of assignments.
        """
        # Total cost = (routing cost + vehicle usage penalty) - (large bonus for making assignment)
        # The bonus ensures we prefer to assign orders even if it costs slightly more
        assignment_bonus = 10000.0  # Large value to encourage assignments
        problem += pulp.lpSum(
            assignment[key] * (self._costs[key] + self._config.vehicle_usage_penalty - assignment_bonus)
            for key in assignment
        )

    
    def _assignment_constraints(self, problem, assignment, vehicles, orders) -> None:
        """
        Every order may be assigned to at most one vehicle.
        Not all orders need to be assigned (when there are more orders than vehicles).
        Only consider feasible vehicle-order pairs (those with variables).
        """
        for order in orders:
            # Collect only feasible assignments for this order
            feasible_assignments = [
                assignment[(vehicle.vehicle_id, order.order_id)] 
                for vehicle in vehicles 
                if (vehicle.vehicle_id, order.order_id) in assignment
            ]
            if feasible_assignments:
                # Each order assigned to at most one vehicle
                problem += (pulp.lpSum(feasible_assignments) <= 1)

    
    def _capacity_constraints(self, problem, assignment, vehicles, orders) -> None:
        """
        Vehicle capacity constraint.
        Only consider feasible vehicle-order pairs (those with variables).
        """
        for vehicle in vehicles:
            # Collect only feasible assignments for this vehicle
            feasible_weight_assignments = [
                assignment[(vehicle.vehicle_id, order.order_id)] * order.weight
                for order in orders
                if (vehicle.vehicle_id, order.order_id) in assignment
            ]
            feasible_count_assignments = [
                assignment[(vehicle.vehicle_id, order.order_id)]
                for order in orders
                if (vehicle.vehicle_id, order.order_id) in assignment
            ]
            
            if feasible_weight_assignments:
                # Total weight of assigned orders must not exceed vehicle capacity
                problem += (pulp.lpSum(feasible_weight_assignments) <= vehicle.capacity)
            
            if feasible_count_assignments:
                # Limit to at most 1 order per vehicle for simplicity
                problem += (pulp.lpSum(feasible_count_assignments) <= 1)


    def _solve(self, problem, assignment, vehicles, orders, tick: int) -> tuple:
        """
        Solve the MILP and update the simulation.
        """
        logger.info("Solving MILP assignment problem...")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self._config.solver_time_limit)
        problem.solve(solver)
        status = pulp.LpStatus[problem.status]
        logger.info(f"Solver status : {status}")
        
        if status not in ("Optimal", "Feasible"):
            logger.warning("No feasible assignment found.")
            return (status, [])

        assignments = self._extract_assignments(assignment, vehicles, orders)

        if not assignments:
            logger.info("Solver returned zero assignments.")
            return (status, [])
            
        logger.info(f"Assigned {len(assignments)} orders.")
        return (status, self._build_dispatch_assignments(assignments, vehicles, orders, tick))
        #self._apply_assignments(assignments, vehicles, orders, tick)
        

    def _extract_assignments(self, assignment, vehicles, orders) -> list:
        """
        Extract selected vehicle-order pairs
        from the solved MILP.
        """
        vehicle_lookup = { vehicle.vehicle_id: vehicle for vehicle in vehicles }
        order_lookup = { order.order_id: order for order in orders }
        selected = []
        for (vehicle_id, order_id), variable in assignment.items():
            if pulp.value(variable) < 0.5:
                continue
            vehicle = vehicle_lookup[vehicle_id]
            order = order_lookup[order_id]
            selected.append((vehicle,order))
            logger.info(f"Assigned Vehicle {vehicle_id} -> Order {order_id}")
            
        return selected


    def _build_dispatch_assignments(self, assignments, vehicles, orders, tick: int) -> DispatchAssignment:
        dispatch_assignments = []
        for vehicle, order in assignments:
            route = self._build_route(vehicle, order)
            vehicle.current_route = route
            vehicle.available = False
            vehicle.current_load += order.weight # accommodating multiple orders if there is capacity
            vehicle.assigned_orders.append(order.order_id)
            #
            # Vehicle will start from the pickup node.
            # The simulation is expected to update this
            # as the vehicle traverses the route.
            #
            vehicle.current_node = order.pickup_node
            order.assigned_vehicle = vehicle.vehicle_id
            order.assigned_tick = tick
            route_hops = len(route.nodes) - 1 if route.nodes else 0
            logger.info(f"Vehicle {vehicle.vehicle_id} dispatched for Order {order.order_id} - Route: {route_hops} hops, {route.total_distance:.0f}m")
            dispatch_assignments.append(DispatchAssignment
                                            (
                                                vehicle_id=vehicle.vehicle_id,
                                                order_id=order.order_id,
                                                route=route,
                                                objective_cost=route.route_cost
                                            )
                                       )

        return dispatch_assignments
                                    

    def _apply_assignments(self, assignments, vehicles, orders, tick: int) -> None:
        """
        Apply solver assignments to simulation objects.
        """
        for vehicle, order in assignments:
            route = self._build_route(vehicle, order)
            vehicle.current_route = route
            vehicle.available = False
            vehicle.current_load = order.weight
            vehicle.assigned_orders.append(order.order_id)
            #
            # Vehicle will start from the pickup node.
            # The simulation is expected to update this
            # as the vehicle traverses the route.
            #
            vehicle.current_node = order.pickup_node
            order.assigned_vehicle = vehicle.vehicle_id
            order.assigned_tick = tick
            logger.info(f"Vehicle {vehicle.vehicle_id} dispatched for Order {order.order_id}")
            

    def _build_route(self, vehicle: Vehicle, order: Order) -> Route:
        """
        Build a simulation route from the cached A* paths.
        Route:
            Vehicle -> Pickup -> Delivery
        """
        path_to_pickup, path_to_delivery = self._paths[ (vehicle.vehicle_id, order.order_id) ]
        #
        # Merge node lists.
        #
        # Avoid duplicating the pickup node.
        #
        nodes = (path_to_pickup.nodes + path_to_delivery.nodes[1:])
        total_distance = path_to_pickup.total_distance + path_to_delivery.total_distance
        total_time = path_to_pickup.total_travel_time + path_to_delivery.total_travel_time

        total_cost = path_to_pickup.total_cost + path_to_delivery.total_cost
        #
        # Arrival times.
        #
        # For now, use cumulative travel time.
        # Can later be replaced with more
        # accurate edge-by-edge timestamps.
        #
        arrival_times = []
        elapsed = 0.0
        arrival_times.append(elapsed)
        for edge in path_to_pickup.edges:
            elapsed += edge.travel_time
            arrival_times.append(elapsed)
        #
        # Skip duplicate pickup node.
        #
        for edge in path_to_delivery.edges:
            elapsed += edge.travel_time
            arrival_times.append(elapsed)
        return Route(nodes=nodes, total_distance=total_distance, estimated_time=total_time, route_cost=total_cost, arrival_times=arrival_times)

    def reset(self) -> None:
        self._paths.clear()
        self._costs.clear()

    def assignment_cost(self, vehicle_id: str, order_id: str) -> float:
        """
        Return computed objective cost for an assignment.
        """
        return self._costs.get((vehicle_id, order_id), float("inf"))

    def assignment_path(self, vehicle_id: str, order_id: str):
        return self._paths.get((vehicle_id,order_id))