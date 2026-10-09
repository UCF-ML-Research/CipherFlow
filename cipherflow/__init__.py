"""CipherFlow: deployment-aware planning of hybrid HE-MPC secure inference.

    import cipherflow as cf
    graph = cf.models.bert(num_blocks=12, seq=128)
    plan  = cf.compile(graph, cf.Deployment.of("cpu", "10Mbps/1ms"))   # mpc="auto", security="2pc"
    print(plan.explain()); plan.to_json("plan.json")
"""
from . import backends, errors, measure as _mb, models, profiles
from .calibration import Calibration
from .deploy import Deployment, Hardware, Network
from .errors import (CostExtrapolationError, CostExtrapolationWarning, UnsupportedBackend,
                     UnsupportedGraph, UnsupportedModel, UnsupportedOperator)
from .plan import Plan
from .planning import compile, match
from .profiles import ProfileNotFound
from .validate import check_graph

load_profile = profiles.load
microbench = _mb.microbench
BackendUnavailable = _mb.BackendUnavailable

__all__ = ["compile", "Deployment", "Hardware", "Network", "Plan", "models", "profiles",
           "backends", "load_profile", "microbench", "BackendUnavailable", "ProfileNotFound",
           "UnsupportedOperator", "UnsupportedGraph", "UnsupportedModel", "CostExtrapolationError",
           "CostExtrapolationWarning", "check_graph", "errors", "match", "UnsupportedBackend",
           "Calibration"]
__version__ = "0.4.0"


def __getattr__(name):                  # cipherflow.runtime needs PyTorch: import it on first use
    if name == "runtime":
        import importlib
        return importlib.import_module(".runtime", __name__)
    raise AttributeError(f"module 'cipherflow' has no attribute {name!r}")
