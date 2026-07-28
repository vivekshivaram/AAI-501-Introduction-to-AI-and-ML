from __future__ import annotations

import heapq
from itertools import count

from src.graph.graph import Graph
from src.graph.node import Node
from src.graph.edge import Edge

from src.routing.path import Path
from src.routing.route_planner import RoutePlanner
from src.routing.heuristic import Heuristic
from src.routing.exceptions import (
    InvalidNodeError,
    RouteNotFoundError,
)

from src.simulation.delay_map import DelayMap

class AStarRouting(RoutePlanner):
    """
    A* shortest path planner.

    Edge cost =
        travel_time + delay penalty

    Heuristic =
        estimated remaining travel time.
    """

    def __init__(
        self,
        graph: Graph,
        heuristic: Heuristic,
        delay_map: DelayMap | None = None,
    ) -> None:

        self._graph = graph
        self._heuristic = heuristic
        self._delay_map = delay_map

    def shortest_path(
        self,
        source: int,
        destination: int,
    ) -> Path:

        if not self._graph.has_node(source):
            raise InvalidNodeError(source)

        if not self._graph.has_node(destination):
            raise InvalidNodeError(destination)

        return self._search(
            self._graph.get_node(source),
            self._graph.get_node(destination),
        )

    def _search(
        self,
        start: Node,
        goal: Node,
    ) -> Path:

        open_set: list[tuple[float, int, int]] = []

        counter = count()

        heapq.heappush(
            open_set,
            (
                0.0,
                next(counter),
                start.id,
            ),
        )

        #
        # node_id ->
        # (
        #     parent_node,
        #     edge_used
        # )
        #
        came_from: dict[
            int,
            tuple[int, Edge],
        ] = {}

        g_score = {
            start.id: 0.0,
        }

        closed: set[int] = set()

        while open_set:

            _, _, current_id = heapq.heappop(open_set)

            if current_id in closed:
                continue

            if current_id == goal.id:
                return self._build_path(
                    came_from,
                    start.id,
                    goal.id,
                    g_score[goal.id],
                )

            closed.add(current_id)

            current = self._graph.get_node(current_id)

            for edge in self._graph.outgoing_edges(current.id):

                neighbour = edge.destination

                if neighbour in closed:
                    continue

                cost = (
                    edge.travel_time
                    + self._edge_penalty(edge)
                )

                tentative = (
                    g_score[current.id]
                    + cost
                )

                if tentative >= g_score.get(
                    neighbour,
                    float("inf"),
                ):
                    continue

                came_from[neighbour] = (
                    current.id,
                    edge,
                )

                g_score[neighbour] = tentative

                heuristic = self._heuristic.estimate(
                    self._graph,
                    self._graph.get_node(neighbour),
                    goal,
                )

                heapq.heappush(
                    open_set,
                    (
                        tentative + heuristic,
                        next(counter),
                        neighbour,
                    ),
                )

        raise RouteNotFoundError(
            f"No route between {start.id} and {goal.id}"
        )

    def _build_path(
        self,
        came_from: dict[int, tuple[int, Edge]],
        source: int,
        destination: int,
        total_cost: float,
    ) -> Path:

        node_ids = [destination]
        edges: list[Edge] = []

        current = destination

        while current != source:

            parent, edge = came_from[current]

            edges.append(edge)

            node_ids.append(parent)

            current = parent

        node_ids.reverse()
        edges.reverse()

        nodes = [
            self._graph.get_node(node)
            for node in node_ids
        ]

        total_distance = sum(
            edge.length
            for edge in edges
        )

        total_travel_time = sum(
            edge.travel_time
            for edge in edges
        )

        return Path(
            nodes=nodes,
            edges=edges,
            total_distance=total_distance,
            total_travel_time=total_travel_time,
            total_cost=total_cost,
        )

    def _edge_penalty(
        self,
        edge: Edge,
    ) -> float:

        if self._delay_map is None:
            return 0.0

        return self._delay_map.penalty(edge)