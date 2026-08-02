"""
Offline GraphML Loader
Responsibilities

• Load local GraphML file, offline
• Validate graph
• Provide helper methods
• Nearest node lookup
• Shortest path wrapper
"""

from pathlib import Path

import networkx as nx
import osmnx as ox
from src.config import DEFAULT_SPEED_KMH
from src.routing.exceptions import InvalidNodeError

from src.config import (MAP_DIRECTORY, MAP_FILENAME)

from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.graph.node import Node
from src.graph.edge import Edge

from os import path
"""
Example simulation: 
graph = Graph()
graph.load()

vehicle.current_node = graph.nearest_node(vehicle.lat, vehicle.lon)

route = graph.shortest_path(vehicle.current_node, order.pickup_node)
distance = graph.path_length(route)
neighbors = graph.neighbors(vehicle.current_node)
"""
class Graph:
    def __init__(self):
        self.graph = None
        self._node_id_map = {}

    def load(self, graph_path:str = MAP_DIRECTORY / MAP_FILENAME):
        """
        Loads GraphML from disk.
        """
        if not path.exists(graph_path):
            raise FileNotFoundError(f"Map not found: {graph_path}")

        logger.info("Loading GraphML...")
        self.graph = ox.load_graphml(graph_path)
        logger.info("Graph loaded successfully.")
        logger.info(f"Nodes : {len(self.graph.nodes)}")
        logger.info(f"Edges : {len(self.graph.edges)}")
        ###
        for id in self.graph.nodes:
            self._node_id_map[id] = self._get_node(id)
        ###
        return self.graph
        
    def _ensure_loaded(self):
        if self.graph is None:
            raise RuntimeError("Graph has not been loaded.")

    def is_loaded(self) -> bool:
        return self.graph is not None

    def validate(self):
        """
        Basic graph validation.
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded.")

        if len(self.graph.nodes) == 0:
            raise RuntimeError( "Graph contains no nodes.")

        if len(self.graph.edges) == 0:
            raise RuntimeError("Graph contains no edges.")

        logger.info("Graph validation passed.")
        return True

    def node_count(self) -> int:
        self._ensure_loaded()
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        self._ensure_loaded()
        return self.graph.number_of_edges()

    def _get_node(self, node_id: int) -> Node:
        self._ensure_loaded()
        if not self.has_node(node_id):
            raise KeyError(f"Unknown node {node_id}")
            
        node = self.graph.nodes[node_id]
        return Node(
            id=node_id,
            latitude=node["y"],
            longitude=node["x"],
        )
        
    def get_node(self, node_id: int) -> Node:
        self._ensure_loaded()
        return self._node_id_map.get(node_id, None)

    def has_edge(self, source: int, destination: int):
        self._ensure_loaded()
        return self.graph.has_edge(source, destination)

    def has_node(self, node: int) -> bool:
        self._ensure_loaded()
        return self.graph.has_node(node)
        
    def get_edge(self, source: int, destination: int) -> Edge:
        self._ensure_loaded()
        if not self.has_edge(source, destination):
            raise KeyError(f"No edge from {source} to {destination}")

        edge = self.graph[source][destination][0]
        return Edge(
            source=source,
            destination=destination,
            length=edge.get("length", 0.0),
            travel_time=edge.get("travel_time", 0.0),
            speed_kph=edge.get("speed_kph", 0.0),
        )

    def nearest_node(self, latitude: float, longitude: float) -> Node:
        self._ensure_loaded()
        return self.get_node(node_id = ox.distance.nearest_nodes(self.graph, X=longitude,Y=latitude))

    def nearest_nodes(self, nodes: list[Node]) -> list[Node]:
        self._ensure_loaded()
        longitudes = [ node.longitude for node in nodes ]
        latitudes = [ node.latitude for node in nodes ]
        return_nodes = []
        for node_id in ox.distance.nearest_nodes(self.graph, X=longitudes,Y=latitudes):
            return_nodes.append(self.get_node(node_id))
        return return_nodes

    def neighbors(self, node: int) -> list[Node]:
        self._ensure_loaded()
        nodes = []
        for node_id in list(self.graph.neighbors(node)):
            nodes.append(self.get_node(node_id))
        return nodes

    def shortest_path(self, source: int, destination: int, weight: str = "travel_time") -> list[Node]:
        self._ensure_loaded()
        nodes = []
        for node_id in nx.shortest_path(self.graph, source=source, target=destination, weight=weight):
            nodes.append(self.get_node(node_id))
        return nodes

    def path_length(self, path: list[Node]) -> float:
        self._ensure_loaded()
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.get_edge(u.id, v.id)
            total += edge.length

        return total
 
    def travel_time(self, source: int, destination: int) -> float:
        path = self.shortest_path(source,destination, weight="travel_time")
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.get_edge(u.id, v.id)
            total += edge.travel_time

        return total
        
    def node_coordinates(self, node: int) -> tuple[float, float]:
        self._ensure_loaded()
        data = self.graph.nodes[node]
        return (data["y"], data["x"])

    def graph_statistics(self):
        self._ensure_loaded()
        stats = {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "strongly_connected": nx.is_strongly_connected(self.graph),
            "density": nx.density(self.graph),
        }
        return stats

    def export_graph_info(self):
        stats = self.graph_statistics()
        logger.info("-------------------------")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        logger.info("-------------------------")

    def get_node_ids(self) -> list[int]:
        self._ensure_loaded()
        return self._node_id_map.keys()

    def outgoing_edges(self, node_id: int) -> list[Edge]:
        """
        Returns all outgoing edges from a node.
        Args:
            node_id: Source node id.
        Returns: List of Edge objects.
        Raises:
            InvalidNodeError: If the node does not exist.
        """
        self._ensure_loaded()
        if not self.get_node(node_id):
            raise InvalidNodeError(
                f"Node {node_id} does not exist."
            )

        edges: list[Edge] = []

        # MultiDiGraph.adj[node] -> {neighbor: {key: attributes}}
        for destination, parallel_edges in self.graph.adj[node_id].items():
            #
            # There may be multiple parallel roads between
            # the same two intersections.
            #
            # Keep the fastest one.
            #
            best = None
            for attributes in parallel_edges.values():
                length = float(attributes.get("length", 0.0))
                speed = attributes.get("speed_kph")
                if speed is None:
                    speed = attributes.get("speed")
                #
                # Fall back to graph default if missing.
                #
                if speed is None:
                    speed = DEFAULT_SPEED_KMH

                speed = float(speed)    
                travel_time = attributes.get("travel_time")
                if travel_time is None:
                    travel_time = length / (speed / 3.6)
                    
                edge = Edge(
                    source=node_id,
                    destination=destination,
                    length=length,
                    travel_time=float(travel_time),
                    speed_kph=speed,
                )
    
                if best is None or edge.travel_time < best.travel_time:
                    best = edge
    
            if best is not None:
                edges.append(best)
    
        return edges