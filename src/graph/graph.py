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
        
    def get_node(self, node_id: int) -> Node:
        self._ensure_loaded()
        if not self.has_node(node_id):
            raise KeyError(f"Unknown node {node_id}")
            
        node = self.graph.nodes[node_id]
        return Node(
            id=node_id,
            latitude=node["y"],
            longitude=node["x"],
        )

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

    def coordinates(self, node_id: int,) -> tuple[float, float]:
        node = self.get_node(node_id)
        return node.latitude, node.longitude

    def nearest_node(self, latitude: float, longitude: float) -> int:
        self._ensure_loaded()
        return ox.distance.nearest_nodes(self.graph, X=longitude,Y=latitude)

    def neighbors(self, node_id: int) -> list[int]:
        self._ensure_loaded()
        return list(self.graph.neighbors(node_id))

    def shortest_path(self, source: int, destination: int, weight: str = "travel_time") -> list[int]:
        self._ensure_loaded()
        return nx.shortest_path(self.graph, source=source, target=destination, weight=weight)

    def path_length(self, path: list[int]) -> float:
        self._ensure_loaded()
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.get_edge(u, v)
            total += edge.length

        return total
 
    def travel_time(self, source: int, destination: int) -> float:
        path = self.shortest_path(source,destination, weight="travel_time")
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.get_edge(u, v)
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