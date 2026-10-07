"""Graph structure check run before planning.

The planner handles a chain of operators in which branches form two-branch
regions: a fork feeds a main branch and a skip branch (each a chain, possibly
empty), and both meet at a join (a residual Add, or Attn*V receiving V from the
Q/K/V projections). A branch may contain whole regions of its own (nested
regions), and a join may also be the fork of the next region (pre-LN blocks).
check_graph raises UnsupportedGraph for anything else (several inputs or outputs,
operators with more than two inputs or consumers, branches that never rejoin,
regions that overlap without nesting), instead of letting the planner plan a
different graph.
"""
from __future__ import annotations

from typing import Dict, List

from compiler.state_expanded.sese import find_sese_regions

from .errors import UnsupportedGraph

SUPPORTED = ("CipherFlow plans a chain of operators with two-branch regions "
             "(fork -> main branch + skip branch -> join), possibly nested")


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
    for r in regions:
        if r.fork in by_fork:
            _fail(f"operator {r.fork} opens two regions")
        by_fork[r.fork] = r
    for r in regions:
        for branch in (r.main_path, r.skip_path):
            # a chain of operators and whole nested regions (fork ... join)
            prev, k = r.fork, 0
            while k < len(branch):
                nid = branch[k]
                if in_adj[nid] != [prev]:
                    _fail(f"branch of the region {r.fork} -> {r.join} is not a chain at {nid}")
                while nid in by_fork:                    # a nested region starts here
                    q = by_fork[nid]
                    if not (set(q.main_path) | set(q.skip_path) | {q.join}) <= set(branch):
                        _fail(f"regions {r.fork} -> {r.join} and {q.fork} -> {q.join} overlap without nesting")
                    k = branch.index(q.join)
                    nid = q.join
                if len(out_adj[nid]) != 1:
                    _fail(f"branch of the region {r.fork} -> {r.join} is not a chain at {nid}")
                prev, k = nid, k + 1
            if prev not in in_adj[r.join]:
                _fail(f"branch of the region {r.fork} -> {r.join} does not end at the join")
        if len(in_adj[r.join]) != 2:
            _fail(f"join {r.join} has {len(in_adj[r.join])} inputs")
    nested_in = {}
    for r in regions:
        for q in regions:
            if q is not r and q.fork in set(r.main_path) | set(r.skip_path):
                nested_in.setdefault(q.fork, []).append(r.fork)
    owner: Dict[str, str] = {}
    for r in regions:                                    # partial overlaps of non-nested regions
        for nid in r.main_path + r.skip_path:
            if nid in owner and owner[nid] not in nested_in.get(r.fork, []) \
                    and r.fork not in nested_in.get(owner[nid], []):
                _fail(f"regions overlap at {nid} without nesting")
            owner.setdefault(nid, r.fork)
    joins = {r.join for r in regions}
    for i in ids:
        if len(out_adj[i]) == 2 and i not in by_fork:
            _fail(f"operator {i} feeds two branches {out_adj[i]} that do not rejoin")
        if len(in_adj[i]) == 2 and i not in joins:
            _fail(f"operator {i} merges {in_adj[i]}, which is not the end of a region")

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
