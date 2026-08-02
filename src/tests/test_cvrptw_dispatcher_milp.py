"""
tests/test_cvrptw_dispatcher_milp.py

Run:

    pytest -v -s tests/test_cvrptw_dispatcher_milp.py
"""

from datetime import datetime, timedelta

import pytest

from src.optimization.cvrptw_dispatcher_milp import CVRPTWDispatcherMilp
from src.optimization.dispatcher_config import DispatcherConfig

from src.models.order import Order
from src.models.vehicle import Vehicle

from src.routing.path import Path
from src.graph.node import Node
from src.graph.edge import Edge


# ---------------------------------------------------------------------
# Mock Router
# ---------------------------------------------------------------------

class MockRouter:
    """
    Deterministic replacement for AStarRouting.

    Always returns a simple two-node path.
    """

    def shortest_path(self, source: int, destination: int):

        start = Node(
            id=source,
            latitude=0.0,
            longitude=0.0,
        )

        end = Node(
            id=destination,
            latitude=0.0,
            longitude=0.0,
        )

        edge = Edge(
            source=source,
            destination=destination,
            length=100.0,
            travel_time=10.0,
            speed_kph=40.0,
        )

        return Path(
            nodes=[start, end],
            edges=[edge],
            total_distance=100.0,
            total_travel_time=10.0,
            total_cost=10.0,
        )


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def dispatcher():

    return CVRPTWDispatcherMilp(
        router=MockRouter(),
        config=DispatcherConfig(
            solver_time_limit=2,
        ),
    )


@pytest.fixture
def vehicle():

    return Vehicle(
        vehicle_id="V1",
        capacity=100.0,
        max_speed_kmh=50.0,
        home_node=1,
        current_node=1,
    )


@pytest.fixture
def order():

    return Order(
        order_id="O1",
        pickup_node=2,
        delivery_node=3,
        package_image="package.png",
        created_time=datetime.now(),
        deadline=datetime.now() + timedelta(hours=2),
        weight=20.0,
        volume=1.0,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_dispatch_single_order(
    dispatcher,
    vehicle,
    order,
):

    result = dispatcher.dispatch(
        [vehicle],
        [order],
        tick=10,
    )

    assert result.success
    assert result.status == "Optimal"
    assert result.tick == 10
    assert result.assigned_count == 1

    assignment = result.assignments[0]

    assert assignment.vehicle_id == "V1"
    assert assignment.order_id == "O1"
    assert assignment.route is not None
    assert assignment.objective_cost > 0


def test_build_cost_matrix(
    dispatcher,
    vehicle,
    order,
):

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    key = (
        vehicle.vehicle_id,
        order.order_id,
    )

    assert key in dispatcher._paths
    assert key in dispatcher._costs


def test_assignment_cost(
    dispatcher,
    vehicle,
    order,
):

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    cost = dispatcher.assignment_cost(
        vehicle.vehicle_id,
        order.order_id,
    )

    assert cost == pytest.approx(20.0)


def test_assignment_path(
    dispatcher,
    vehicle,
    order,
):

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    pickup_path, delivery_path = dispatcher.assignment_path(
        vehicle.vehicle_id,
        order.order_id,
    )

    assert pickup_path.total_cost == 10.0
    assert delivery_path.total_cost == 10.0


def test_build_route(
    dispatcher,
    vehicle,
    order,
):

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    route = dispatcher._build_route(
        vehicle,
        order,
    )

    assert route.total_distance == 200.0
    assert route.route_cost == 20.0
    assert route.estimated_time == 20.0

    #
    # Vehicle -> Pickup -> Delivery
    #
    assert len(route.nodes) == 3
    assert len(route.arrival_times) == 3


def test_capacity_constraint():

    dispatcher = CVRPTWDispatcherMilp(
        router=MockRouter(),
        config=DispatcherConfig(
            solver_time_limit=2,
        ),
    )

    vehicle = Vehicle(
        vehicle_id="V1",
        capacity=5.0,
        max_speed_kmh=50.0,
        home_node=1,
        current_node=1,
    )

    order = Order(
        order_id="O1",
        pickup_node=2,
        delivery_node=3,
        package_image="",
        created_time=datetime.now(),
        deadline=datetime.now(),
        weight=20.0,
        volume=1.0,
    )

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    #
    # No feasible pair should exist.
    #
    assert len(dispatcher._paths) == 0
    assert len(dispatcher._costs) == 0


def test_reset(
    dispatcher,
    vehicle,
    order,
):

    dispatcher._build_cost_matrix(
        [vehicle],
        [order],
    )

    dispatcher.reset()

    assert dispatcher._paths == {}
    assert dispatcher._costs == {}

def test_dispatch_multiple_vehicles_multiple_orders(dispatcher):
    vehicles = [

        Vehicle(
            vehicle_id="V1",
            capacity=100.0,
            max_speed_kmh=50.0,
            home_node=1,
            current_node=1,
        ),

        Vehicle(
            vehicle_id="V2",
            capacity=100.0,
            max_speed_kmh=50.0,
            home_node=5,
            current_node=5,
        ),
    ]

    orders = [

        Order(
            order_id="O1",
            pickup_node=2,
            delivery_node=3,
            package_image="",
            created_time=datetime.now(),
            deadline=datetime.now() + timedelta(hours=2),
            weight=10.0,
            volume=1.0,
        ),

        Order(
            order_id="O2",
            pickup_node=6,
            delivery_node=7,
            package_image="",
            created_time=datetime.now(),
            deadline=datetime.now() + timedelta(hours=2),
            weight=20.0,
            volume=1.0,
        ),
    ]

    result = dispatcher.dispatch(
        vehicles,
        orders,
        tick=25,
    )

    assert result.success
    assert result.status == "Optimal"

    #
    # Both orders should be assigned.
    #
    assert result.assigned_count == 2

    assigned_orders = {
        assignment.order_id
        for assignment in result.assignments
    }

    assigned_vehicles = {
        assignment.vehicle_id
        for assignment in result.assignments
    }

    assert assigned_orders == {"O1", "O2"}

    #
    # Each assignment should contain a route.
    #
    for assignment in result.assignments:

        assert assignment.route is not None
        assert assignment.objective_cost > 0

    #
    # Vehicle state updated.
    #
    used_vehicle_ids = {
        a.vehicle_id
        for a in result.assignments
    }
    for vehicle in vehicles:
        if vehicle.vehicle_id in used_vehicle_ids:
            assert vehicle.available is False
            assert vehicle.current_route is not None
        else:
            assert vehicle.available is True

    #
    # Order state updated.
    #
    for order in orders:

        assert order.assigned_vehicle is not None
        assert order.assigned_tick == 25

    #
    # One vehicle should not receive the same order twice.
    #
    assert len(assigned_orders) == len(result.assignments)

    #
    # One order should not be assigned to two vehicles.
    #
    #assert len(assigned_vehicles) == len(result.assignments)
    
""" For now restricting only one order per vehicle thus disabling this test
def test_multiple_orders_assigned_to_same_vehicle(dispatcher):

    #
    # Single vehicle with sufficient capacity.
    #
    vehicle = Vehicle(
        vehicle_id="V1",
        capacity=100.0,
        max_speed_kmh=50.0,
        home_node=1,
        current_node=1,
    )

    #
    # Two lightweight orders.
    #
    orders = [

        Order(
            order_id="O1",
            pickup_node=2,
            delivery_node=3,
            package_image="",
            created_time=datetime.now(),
            deadline=datetime.now() + timedelta(hours=2),
            weight=20.0,
            volume=1.0,
        ),

        Order(
            order_id="O2",
            pickup_node=4,
            delivery_node=5,
            package_image="",
            created_time=datetime.now(),
            deadline=datetime.now() + timedelta(hours=2),
            weight=30.0,
            volume=1.0,
        ),
    ]

    result = dispatcher.dispatch(
        [vehicle],
        orders,
        tick=100,
    )

    #
    # Dispatcher succeeded.
    #
    assert result.success
    assert result.status == "Optimal"

    #
    # Both orders should be assigned.
    #
    assert result.assigned_count == 2

    #
    # Both assignments should belong to the same vehicle.
    #
    assert {
        assignment.vehicle_id
        for assignment in result.assignments
    } == {"V1"}

    #
    # Orders are unique.
    #
    assert {
        assignment.order_id
        for assignment in result.assignments
    } == {"O1", "O2"}

    #
    # Vehicle state updated.
    #
    assert vehicle.available is False
    assert vehicle.current_load == 50.0
    assert set(vehicle.assigned_orders) == {"O1", "O2"}

    #
    # Capacity respected.
    #
    total_weight = sum(order.weight for order in orders)

    assert total_weight <= vehicle.capacity

    #
    # Every assignment has a route.
    #
    for assignment in result.assignments:

        assert assignment.route is not None
        assert assignment.objective_cost > 0

    #
    # Orders updated.
    #
    for order in orders:

        assert order.assigned_vehicle == "V1"
        assert order.assigned_tick == 100
"""