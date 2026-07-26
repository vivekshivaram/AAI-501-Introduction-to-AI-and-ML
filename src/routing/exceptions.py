"""
Routing specific exceptions.
"""
class RoutingError(Exception):
    """Base routing exception."""

class RouteNotFoundError(RoutingError):
    """Raised when no route exists between two nodes."""

class InvalidNodeError(RoutingError):
    """Raised when a requested node does not exist."""