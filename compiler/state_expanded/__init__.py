"""State-expanded hybrid HE-MPC compiler (paper §4.2).

Mirrors the paper line-by-line:

- ``bert_graph``     : builds the BERT-base operator DAG G=(V,E) with
                       residual branches.
- ``profiler``       : runs real HE primitives (CPU/GPU wallclock) and
                       real MPC primitives (comm bytes + rounds via SCI
                       bridges) to produce a hardware-aware latency
                       profile.
- ``cost_model``     : composes wallclock latency from the measured
                       profile under a given ``(bandwidth, RTT)`` setting.
- ``state_graph``    : state = (i, d, l), transitions = paper Table 1.
- ``sese``           : Single-Entry Single-Exit decomposition and local
                       transfer functions T_R(s_src, s_dst).
- ``planner``        : Dijkstra over the macro graph + backtracking to
                       materialise an end-to-end execution plan.
"""
