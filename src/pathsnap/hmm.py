"""HMM / Viterbi route reconstruction on top of the candidate classifier.

States  : directed arcs that are candidates of a fix (<= 40 per fix, from candidates.py)
Emission: w * log P(arc | features)  - the classifier probability of the arc being the true one
Transition between consecutive fixes (arc a -> arc b), all in log-space:
    a == b                        0                      (still on the same road)
    b reachable from a in h hops  -hop_pen * h           (moved on along the network)
    b is the reverse twin of a    -uturn_pen             (U-turn / opposite carriageway)
    otherwise                     -jump_pen              (topologically impossible, but kept finite so Viterbi never dies)
After a time gap (> gap_s) all transition penalties are multiplied by gap_scale (the vehicle may have driven far).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class HMMParams:
    emis_w: float = 1.0
    p_floor: float = 1e-4
    hop_pen: float = 1.0
    uturn_pen: float = 4.0
    jump_pen: float = 12.0
    max_hops: int = 4
    gap_s: float = 10.0
    gap_scale: float = 0.3

    def to_dict(self): return asdict(self)


class ArcGraph:
    """Directed arc graph: successors(a) = arcs that start where arc a ends. Built vectorised, BFS is lazy + cached."""
    def __init__(self, arcs: np.ndarray):
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.order = np.argsort(self.arcs[:, 0], kind="stable")
        n_nodes = int(self.arcs.max()) + 2
        self.start = np.searchsorted(self.arcs[self.order, 0], np.arange(n_nodes + 1))
        self._reach: dict[tuple[int, int], dict] = {}
        self._twin: dict[int, int] = {}

    def succ(self, a: int):
        b = self.arcs[a, 1]
        return self.order[self.start[b]:self.start[b + 1]]

    def twin(self, a: int) -> int:
        """reverse arc (v->u) of a=(u->v), or -1"""
        if a not in self._twin:
            u, v = self.arcs[a]
            s = self.order[self.start[v]:self.start[v + 1]]
            m = s[self.arcs[s, 1] == u]
            self._twin[a] = int(m[0]) if len(m) else -1
        return self._twin[a]

    def reach(self, a: int, H: int) -> dict:
        """{arc: min hops} for arcs reachable from a in 1..H directed transitions."""
        key = (a, H)
        r = self._reach.get(key)
        if r is None:
            r = {}; fr = [a]
            for h in range(1, H + 1):
                nxt = []
                for x in fr:
                    for y in self.succ(x).tolist():
                        if y not in r and y != a:
                            r[y] = h; nxt.append(y)
                fr = nxt
                if not fr: break
            self._reach[key] = r
        return r

    def shortest_path(self, a: int, b: int, max_hops: int = 15):
        """arc sequence a -> ... -> b (both included) by BFS, or None."""
        if a == b: return [a]
        prev = {a: -1}; fr = [a]
        for _ in range(max_hops):
            nxt = []
            for x in fr:
                for y in self.succ(x).tolist():
                    if y not in prev:
                        prev[y] = x
                        if y == b:
                            path = [b]
                            while prev[path[-1]] != -1: path.append(prev[path[-1]])
                            return path[::-1]
                        nxt.append(y)
            fr = nxt
            if not fr: break
        return None


def _trans(graph: ArcGraph, A: np.ndarray, B: np.ndarray, P: HMMParams, scale: float) -> np.ndarray:
    """(len(A) x len(B)) log transition matrix."""
    T = np.full((len(A), len(B)), -P.jump_pen * scale)
    bidx = {int(b): j for j, b in enumerate(B)}
    for i, a in enumerate(A.tolist()):
        for b, h in graph.reach(a, P.max_hops).items():
            j = bidx.get(b)
            if j is not None: T[i, j] = -P.hop_pen * h * scale
        tw = graph.twin(a)
        j = bidx.get(tw) if tw >= 0 else None
        if j is not None: T[i, j] = -P.uturn_pen * scale
        j = bidx.get(a)
        if j is not None: T[i, j] = 0.0
    return T


def viterbi(cand: dict, p: np.ndarray, t: np.ndarray, graph: ArcGraph, P: HMMParams = HMMParams()):
    """cand: {'obs': int array (sorted), 'arc': int array} candidate rows; p: classifier probabilities per row; t: time stamp per fix.
    Returns (arc_per_fix[T] (-1 = no candidate anywhere), prob_of_chosen[T])."""
    obs, arc = np.asarray(cand["obs"]), np.asarray(cand["arc"])
    T = len(t)
    order = np.argsort(obs, kind="stable"); obs, arc, p = obs[order], arc[order], np.asarray(p)[order]
    ids, first = np.unique(obs, return_index=True)
    bounds = np.r_[first, len(obs)]
    out = np.full(T, -1, dtype=np.int64); outp = np.zeros(T)
    if len(ids) == 0: return out, outp
    emis = P.emis_w * np.log(np.clip(p, P.p_floor, 1.0))
    delta = emis[bounds[0]:bounds[1]].copy(); back = []
    for k in range(1, len(ids)):
        A, B = arc[bounds[k - 1]:bounds[k]], arc[bounds[k]:bounds[k + 1]]
        dt = t[ids[k]] - t[ids[k - 1]]
        scale = P.gap_scale if dt > P.gap_s else 1.0
        S = delta[:, None] + _trans(graph, A, B, P, scale)
        bp = S.argmax(0); delta = S[bp, np.arange(len(B))] + emis[bounds[k]:bounds[k + 1]]
        delta -= delta.max()                       # numerical stability
        back.append(bp)
    j = int(delta.argmax())
    for k in range(len(ids) - 1, -1, -1):
        s = bounds[k] + j
        out[ids[k]] = arc[s]; outp[ids[k]] = p[s]
        if k > 0: j = int(back[k - 1][j])
    # fixes without any candidate inherit the neighbouring decoded arc
    have = out >= 0
    if (~have).any():
        idx = np.where(have, np.arange(T), -1); idx = np.maximum.accumulate(idx)
        first_ok = int(np.argmax(have)); idx = np.where(idx < 0, first_ok, idx)
        out = out[idx]
    return out, outp


def build_route(fix_arcs: np.ndarray, graph: ArcGraph, max_hops: int = 15):
    """Collapse consecutive duplicates and bridge gaps between non-adjacent arcs with the shortest directed path."""
    route = []
    for a in fix_arcs.tolist():
        if a < 0: continue
        if not route: route.append(a); continue
        if a == route[-1]: continue
        if a in graph.succ(route[-1]).tolist(): route.append(a); continue
        path = graph.shortest_path(route[-1], a, max_hops)
        route.extend(path[1:] if path else [a])
    return route
