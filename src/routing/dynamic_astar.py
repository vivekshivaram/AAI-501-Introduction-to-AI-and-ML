"""
dynamic_astar.py
Feature 4: Dynamic Edge-Weight Cost Fusion

Injects ML-predicted delay penalties into A* edge weights, enabling
context-aware routing that penalises high-friction segments.

Cost function:
    Weight_ij = Length_ij + (Predicted_Delay × alpha)

Integrates with the ``Graph`` wrapper class and the ``DelayPredictor``
trained model. Exposes a weight callable compatible with NetworkX A* and
Dijkstra, for direct consumption by the dispatcher.

Deliverable: src/routing/dynamic_astar.py  (this file)

Usage (from project root):
    from src.routing.dynamic_astar import DynamicAStar
    from src.analytics.delay_model import DelayPredictor
    from src.graph.graph import Graph

    graph = Graph()
    graph.load()
    predictor = DelayPredictor()

    router = DynamicAStar(graph, predictor)
    result = router.find_path(
        source_node=123456,
        target_node=789012,
        context={'weather': 'Stormy', 'vehicle': 'van', 'traffic': 'High'},
    )
    # result = {path, total_weight, n_edges, base_length_m, delay_penalty_m}

    # Direct NetworkX integration for the CVRPTW dispatcher:
    weight_fn = router.get_weight_function(context)
    path = nx.astar_path(G, source, target, weight=weight_fn)
"""

from __future__ import annotations

import networkx as nx

from src.analytics.delay_model import DelayPredictor
from src.config import DELAY_ALPHA
from src.graph.graph import Graph
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicAStar:
    """
    A* pathfinding with ML-predicted delay penalties fused into edge weights.

    Wraps the ``Graph`` class and uses the ``DelayPredictor`` model
    to compute context-sensitive path costs without network I/O.

    Parameters
    ----------
    graph : Graph
        Loaded Graph wrapper (``graph.load()`` must be called).
    predictor : DelayPredictor
        Loaded delay regression model for per-edge friction prediction.
    alpha : float, optional
        Penalty scale factor.  Default 50.0 means 1 minute of predicted delay
        adds 50 metres to the path cost — tunable by the dispatcher.
    """

    def __init__(
        self,
        graph: Graph,
        predictor: DelayPredictor,
        alpha: float = DELAY_ALPHA,
    ) -> None:
        if not graph.is_loaded():
            raise RuntimeError(
                "Graph must be loaded before passing to DynamicAStar. "
                "Call graph.load() first."
            )
        self._graph = graph
        self._predictor = predictor
        self._alpha = alpha
        logger.info(
            f"DynamicAStar initialised (alpha={alpha}, "
            f"nodes={graph.node_count():,}, edges={graph.edge_count():,})"
        )

    # ------------------------------------------------------------------ #
    # Internal weight computation
    # ------------------------------------------------------------------ #

    def _dynamic_weight(
        self,
        u: int,
        v: int,
        data: dict,
        context: dict | None = None,
    ) -> float:
        """
        Compute fused edge weight: ``Length_ij + Predicted_Delay × alpha``.

        Parameters
        ----------
        u, v : int
            Source and destination OSM node IDs.
        data : dict
            NetworkX edge attribute dictionary.
        context : dict, optional
            Routing context with optional keys ``weather``, ``vehicle``,
            ``traffic``.  Missing keys fall back to safe defaults.
        """
        base_length: float = data.get("length", 0.0)
        edge_km = base_length / 1000.0
        ctx = context or {}
        delay = self._predictor.predict_delay(
            distance_km=edge_km,
            weather=ctx.get("weather", "Sunny"),
            vehicle=ctx.get("vehicle", "van"),
            traffic=ctx.get("traffic", "Medium"),
        )
        return base_length + delay * self._alpha

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def find_path(
        self,
        source_node: int,
        target_node: int,
        context: dict | None = None,
    ) -> dict:
        """
        Find the A* shortest path using dynamic delay-fused edge weights.

        Parameters
        ----------
        source_node : int
            OSM node ID of the route origin.
        target_node : int
            OSM node ID of the route destination.
        context : dict, optional
            Routing context — keys: ``weather`` (str), ``vehicle`` (str),
            ``traffic`` (str).

        Returns
        -------
        dict
            ``path``            — list of OSM node IDs along the route
            ``total_weight``    — total fused cost (metres equivalent)
            ``n_edges``         — number of road segments traversed
            ``base_length_m``   — physical road distance in metres
            ``delay_penalty_m`` — additional delay cost in metres equivalent
        """
        raw_graph = self._graph.graph  # networkx.MultiDiGraph

        def weight_fn(u: int, v: int, data: dict) -> float:
            return self._dynamic_weight(u, v, data, context)

        try:
            path: list[int] = nx.astar_path(
                raw_graph, source_node, target_node, weight=weight_fn
            )
            total_dynamic = 0.0
            total_base = 0.0
            for i in range(len(path) - 1):
                edge_data = raw_graph[path[i]][path[i + 1]][0]
                base_len = edge_data.get("length", 0.0)
                total_base += base_len
                total_dynamic += weight_fn(path[i], path[i + 1], edge_data)
            return {
                "path": path,
                "total_weight": round(total_dynamic, 2),
                "n_edges": len(path) - 1,
                "base_length_m": round(total_base, 2),
                "delay_penalty_m": round(total_dynamic - total_base, 2),
            }
        except nx.NetworkXNoPath:
            return {
                "path": [],
                "total_weight": float("inf"),
                "n_edges": 0,
                "base_length_m": 0.0,
                "delay_penalty_m": 0.0,
            }

    def get_weight_function(self, context: dict | None = None):
        """
        Return a weight callable compatible with ``nx.astar_path`` and
        ``nx.dijkstra_path`` for direct use in the dispatcher.

        Parameters
        ----------
        context : dict, optional
            Routing context: ``{'weather': str, 'vehicle': str, 'traffic': str}``.

        Returns
        -------
        Callable[[int, int, dict], float]
            Edge weight function ``f(u, v, edge_data) -> cost``.

        Example
        -------
        >>> weight_fn = router.get_weight_function({'weather': 'Stormy', 'traffic': 'High'})
        >>> path = nx.astar_path(G, source, target, weight=weight_fn)
        """

        def weight_fn(u: int, v: int, data: dict) -> float:
            return self._dynamic_weight(u, v, data, context)

        return weight_fn
