"""Graph structure check run before planning.

The planner handles a chain of operators in which residual connections form
two-branch regions: a fork feeds a main branch and a skip branch (each a chain,
possibly empty), and both meet at a join (the residual Add). A join may also be
the fork of the next region (pre-LN blocks). check_graph raises UnsupportedGraph
for anything else (several inputs or outputs, operators with more than two
inputs or consumers, branches that never rejoin, nested or overlapping
residuals), instead of letting the planner plan a different graph.
"""
from __future__ import annotations

from typing import Dict, List

from compiler.state_expanded.sese import find_sese_regions

from .errors import UnsupportedGraph

SUPPORTED = ("CipherFlow v0.1 plans a chain of operators with two-branch residual regions "
             "(fork -> main branch + skip branch -> Add)")


def _fail(msg: str) -> None:
    raise UnsupportedGraph(f"{msg}. {SUPPORTED}.")


def check_graph(graph) -> None:
    ids = [n.node_id for n in graph.nodes]
    if not ids:
        _fail("graph has no operators")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        _fail(f"duplicate node ids {dup[:5]}")
    known = set(ids)
    out_adj: Dict[str, List[str]] = {i: [] for i in ids}
    in_adj: Dict[str, List[str]] = {i: [] for i in ids}
    for e in graph.edges:
        if e.src not in known or e.dst not in known:
            _fail(f"edge {e.src} -> {e.dst} refers to a node not in the graph")
        if e.dst not in out_adj[e.src]:
            out_adj[e.src].append(e.dst)
            in_adj[e.dst].append(e.src)

    # acyclic
    indeg = {i: len(in_adj[i]) for i in ids}
    ready = [i for i in ids if indeg[i] == 0]
    seen = 0
    while ready:
        u = ready.pop()
        seen += 1
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    if seen != len(ids):
        _fail("graph has a cycle")

    sources = [i for i in ids if not in_adj[i]]
    sinks = [i for i in ids if not out_adj[i]]
    if len(sources) != 1:
        _fail(f"graph has {len(sources)} inputs {sources[:5]}; only single-input models are supported "
              "(e.g. encoder-decoder cross-attention is not)")
    if len(sinks) != 1:
        _fail(f"graph has {len(sinks)} outputs {sinks[:5]}; only single-output models are supported")
    for i in ids:
        if len(in_adj[i]) > 2:
            _fail(f"operator {i} has {len(in_adj[i])} inputs {in_adj[i][:5]}")
        if len(out_adj[i]) > 2:
            _fail(f"operator {i} feeds {len(out_adj[i])} operators {out_adj[i][:5]}")

    regions = find_sese_regions(graph)
    by_fork = {}
    owner: Dict[str, str] = {}
    for r in regions:
        if r.fork in by_fork:
            _fail(f"operator {r.fork} opens two residual regions")
        by_fork[r.fork] = r
        for branch in (r.main_path, r.skip_path):
            prev = r.fork
            for nid in branch:
                if nid in owner:
                    _fail(f"residual regions overlap at {nid} (nested residuals are not supported)")
                owner[nid] = r.region_id
                if in_adj[nid] != [prev] or len(out_adj[nid]) != 1:
                    _fail(f"branch of the residual region {r.fork} -> {r.join} is not a chain at {nid}")
                prev = nid
            if prev not in in_adj[r.join]:
                _fail(f"branch of the residual region {r.fork} -> {r.join} does not end at the join")
        if len(in_adj[r.join]) != 2:
            _fail(f"residual join {r.join} has {len(in_adj[r.join])} inputs")
    joins = {r.join for r in regions}
    for i in ids:
        if len(out_adj[i]) == 2 and i not in by_fork:
            _fail(f"operator {i} feeds two branches {out_adj[i]} that do not rejoin as a residual Add")
        if len(in_adj[i]) == 2 and i not in joins:
            _fail(f"operator {i} merges {in_adj[i]}, which is not the end of a residual region")

    # walk the chain: every operator must be on it or inside a region
    visited = set()
    cur = sources[0]
    while True:
        visited.add(cur)
        if cur in by_fork:
            r = by_fork[cur]
            visited.update(r.main_path)
            visited.update(r.skip_path)
            cur = r.join
            continue
        nxt = out_adj[cur]
        if not nxt:
            break
        cur = nxt[0]
    missing = [i for i in ids if i not in visited]
    if missing:
        _fail(f"operators {missing[:5]} are not on the chain from {sources[0]}")
