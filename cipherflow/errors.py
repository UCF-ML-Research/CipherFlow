"""Exceptions and warnings raised by cipherflow.

Every limit of the current version fails explicitly with one of these, instead
of producing a plan from a graph or cost data the planner cannot handle.
"""
from __future__ import annotations


class UnsupportedOperator(ValueError):
    """An operator has no backend method, no cost record, or no feasible domain."""


class UnsupportedGraph(ValueError):
    """The graph's structure is outside what the planner handles (see validate.check_graph)."""


class UnsupportedBackend(ValueError):
    """No MPC backend satisfies the requested backend, security model, hardware and profile."""


class UnsupportedModel(ValueError):
    """The PyTorch front end cannot map the model onto the operator set."""


class CostExtrapolationError(ValueError):
    """strict=True and some cost entries are not in the profile (they would be scaled)."""


class CostExtrapolationWarning(UserWarning):
    """Some cost entries were scaled from the nearest profiled shape."""
