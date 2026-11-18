"""Evaluation metrics for candidate scoring + route-level quality."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, confusion_matrix


def threshold_scan(y, p, grid=np.linspace(0.05, 0.95, 37)):
    best = (0.5, -1)
    for th in grid:
        f = f1_score(y, p >= th, average="macro")
        if f > best[1]:
            best = (float(th), float(f))
    return best


def top1_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Highest-scoring candidate per fix. df needs obs, arc, trip, y, obs_valid."""
    d = df[["trip", "obs", "arc", "y", "obs_valid"]].reset_index(drop=True)
    d["p"] = p
    return d.loc[d.groupby(["trip", "obs"])["p"].idxmax()]


def candidate_metrics(y, p, thr):
    yh = p >= thr
    return dict(macro_f1=f1_score(y, yh, average="macro"), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
                cm=confusion_matrix(y, yh).tolist())


def _reach(next_of, a, b, hops=3):
    if a == b:
        return True
    fr = {a}
    for _ in range(hops):
        nx = set()
        for x in fr:
            nx.update(next_of.get(x, ()))
        if b in nx:
            return True
        fr = nx
        if not fr:
            return False
    return False


def build_next(arcs: np.ndarray):
    from collections import defaultdict
    by_from = defaultdict(list)
    for i, (a, b) in enumerate(arcs.tolist()):
        by_from[a].append(i)
    return {i: by_from.get(b, []) for i, (a, b) in enumerate(arcs.tolist())}


def route_continuity(pred_arcs, t, next_of, max_dt=5.0, hops=3):
    """Fraction of consecutive fixes (dt<=max_dt) whose predicted arcs are equal or reachable
    within <=hops directed arc transitions."""
    ok = tot = 0
    cache = {}
    for i in range(len(pred_arcs) - 1):
        if t[i + 1] - t[i] > max_dt:
            continue
        key = (pred_arcs[i], pred_arcs[i + 1])
        if key not in cache:
            cache[key] = _reach(next_of, key[0], key[1], hops)
        ok += cache[key]; tot += 1
    return ok, tot


def route_prf(pred_route, ref_route):
    """Arc-set overlap between predicted and reference route: (precision, recall, F1)."""
    P, R = set(pred_route), set(ref_route)
    tp = len(P & R)
    pr, rc = tp / max(len(P), 1), tp / max(len(R), 1)
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
