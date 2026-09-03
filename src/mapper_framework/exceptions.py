"""Custom exception hierarchy for Mapper framework."""

from typing import Any


class MapperError(Exception):
    """Base exception for all Mapper framework errors."""
    pass


class ConfigurationInvalidError(MapperError):
    """Raised when an invalid configuration or malformed input is supplied."""
    pass


class ClustererFailureError(MapperError):
    """Raised when a local clusterer fails unexpectedly or returns malformed output."""
    pass


class HomologyConsistencyError(MapperError):
    """Raised when an algebraic chain condition or homology bound is violated."""

    def __init__(self, message: str = "", partial_graph: Any = None, partial_nerve: Any = None):
        super().__init__(message)
        self.partial_graph = partial_graph
        self.partial_nerve = partial_nerve



class ResourceLimitError(MapperError):
    """Raised when computational resource guardrails (simplices/memory) are exceeded."""
    pass
