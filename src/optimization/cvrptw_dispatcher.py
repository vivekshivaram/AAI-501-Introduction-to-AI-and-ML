"""cvrptw_dispatcher.py"""
from collections import defaultdict
from src.models.order import Order
from src.models.vehicle import Vehicle
from src.optimization.astar_routing import AStarRouting
from src.simulation.route import Route
from src.graph.graph import Graph

class CVRPTWDispatcher:
    def __init__(self, graph: Graph, router:AStarRouting):
        self.graph=graph
        self.router=router

    def dispatch(self, vehicles:list[Vehicle], orders:list[Order], current_tick:int=0):
        assignments=defaultdict(list)
        available=[v for v in vehicles if v.available]
        for order in sorted(orders,key=lambda o:o.deadline):
            best=None
            best_cost=float("inf")
            for vehicle in available:
                if (vehicle.current_load+order.weight) > vehicle.capacity:
                    continue
                try:
                    p1=self.router.shortest_path(vehicle.home_node,order.pickup_node)
                    p2=self.router.shortest_path(order.pickup_node,order.delivery_node)
                except Exception:
                    continue
                c = p1.total_cost + p2.total_cost
                if c < best_cost:
                    best_cost=c
                    best=(vehicle,p1,p2)
            if best is None:
                continue
            vehicle,p1,p2=best
            assignments[vehicle.vehicle_id].append(order)
            vehicle.current_load+=order.weight
            vehicle.assigned_orders.append(order.order_id)
            order.assigned_vehicle=vehicle.vehicle_id
            order.assigned_tick=current_tick
        self._build_routes(assignments,vehicles)
        return assignments

    def _build_routes(self,assignments,vehicles):
        lookup={v.vehicle_id:v for v in vehicles}
        for vid,orders in assignments.items():
            v=lookup[vid]
            nodes=[v.home_node]
            dist=time=cost=0.0
            arrivals=[]
            current=v.home_node
            for order in orders:
                a=self.router.shortest_path(current,order.pickup_node)
                b=self.router.shortest_path(order.pickup_node,order.delivery_node)
                nodes.extend([n.id for n in a.nodes[1:]])
                nodes.extend([n.id for n in b.nodes[1:]])
                dist += a.total_distance + b.total_distance
                time += a.total_travel_time + b.total_travel_time
                cost += a.total_cost + b.total_cost
                arrivals.append(time)
                current=order.delivery_node
            v.current_route=Route(nodes=nodes,total_distance=dist,estimated_time=time,route_cost=cost,arrival_times=arrivals,current_index=0)
