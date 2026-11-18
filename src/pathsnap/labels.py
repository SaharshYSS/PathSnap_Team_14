"""Observation-level reference labels.

The dataset only supplies the reference *route* (ordered arc ids), not which arc each GPS
fix belongs to.  We derive per-observation labels with a monotone alignment (dynamic
programme): each fix is assigned a route position j_t, with j_t non-decreasing in t, minimising
the sum of truncated point-to-segment distances.  Labels are used ONLY for training/evaluation.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _align(px, py, ax, ay, bx, by, cap):
    T = px.shape[0]
    J = ax.shape[0]
    INF = 1e18
    dp = np.empty(J)
    bp = np.empty((T, J), dtype=np.int32)
    dist = np.empty((T, J), dtype=np.float32)
    # segment helpers
    ux = np.empty(J); uy = np.empty(J); L = np.empty(J)
    for j in range(J):
        dx = bx[j] - ax[j]; dy = by[j] - ay[j]
        l = (dx * dx + dy * dy) ** 0.5
        L[j] = l
        if l < 1e-9:
            ux[j] = 0.0; uy[j] = 0.0
        else:
            ux[j] = dx / l; uy[j] = dy / l
    for t in range(T):
        # prefix-min of previous dp with argmin
        if t > 0:
            best = INF; barg = 0
            pm = np.empty(J); pa = np.empty(J, dtype=np.int32)
            for j in range(J):
                if dp[j] < best:
                    best = dp[j]; barg = j
                pm[j] = best; pa[j] = barg
        for j in range(J):
            wx = px[t] - ax[j]; wy = py[t] - ay[j]
            s = wx * ux[j] + wy * uy[j]
            if s < 0.0:
                s = 0.0
            elif s > L[j]:
                s = L[j]
            d = ((wx - s * ux[j]) ** 2 + (wy - s * uy[j]) ** 2) ** 0.5
            dist[t, j] = d
            c = d if d < cap else cap
            if t == 0:
                dp[j] = c
                bp[t, j] = j
            else:
                dp[j] = c + pm[j]
                bp[t, j] = pa[j]
    # backtrack
    jb = 0; best = INF
    for j in range(J):
        if dp[j] < best:
            best = dp[j]; jb = j
    out = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        out[t] = jb
        jb = bp[t, jb]
    d_assigned = np.empty(T)
    for t in range(T):
        d_assigned[t] = dist[t, out[t]]
    return out, d_assigned


def align_gps_to_route(gx, gy, nx, ny, arcs, route, cap=60.0):
    """Returns (ref_arc_id[T], dist_to_ref_arc[T]) for every observation."""
    a = arcs[route, 0]; b = arcs[route, 1]
    pos, d = _align(gx.astype(np.float64), gy.astype(np.float64), nx[a], ny[a], nx[b], ny[b], float(cap))
    return route[pos], d, pos
