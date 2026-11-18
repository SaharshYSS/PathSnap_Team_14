"""Candidate road-segment generation + feature engineering.

For every GPS fix we collect all DIRECTED arcs lying within `radius` metres (exact
point-to-segment distance, found through a KD-tree over densified arc samples).  Each
(fix, arc) pair is a candidate row; a classifier scores P(arc is the true segment | features).

Nothing here touches the reference route => the identical function is used at inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import chain
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import make_projector, seg_proj, wrap180
from .io import Trip

DEFAULT_CFG = dict(radius=60.0, spacing=15.0, max_cand=40, hdg_w=2, min_move=3.0,
                   ctx_offsets=(-6, -3, -1, 1, 3, 6))

FEATURES = [
    # geometry of the fix relative to the arc
    "d", "lat_off", "frac", "proj_clamped", "arc_len", "dist_to_end", "dist_from_start",
    # kinematics / direction agreement
    "speed", "hdg_valid", "cos_dh", "abs_dh_deg", "motion_along",
    # relative to the other candidates of the same fix
    "n_cand", "rank_d", "d_minus_dmin", "d_ratio", "rank_d_dir", "twin_exists",
    # network topology
    "deg_from_out", "deg_to_out", "deg_from_in", "deg_to_in",
    # sequence context (neighbouring fixes measured against this same arc)
    "d_mean_w3", "d_max_w6", "d_mean_w6", "frac_near_w6", "prog_w3", "prog_w6", "prog_ratio_w3",
    # sampling
    "dt_prev", "dt_next",
]


@dataclass
class TripGeom:
    gx: np.ndarray
    gy: np.ndarray
    t: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    arcs: np.ndarray
    proj: object


def project_trip(trip: Trip) -> TripGeom:
    proj = make_projector(float(trip.lon.mean()), float(trip.lat.mean()))
    gx, gy = proj.transform(trip.lon, trip.lat)
    nx, ny = proj.transform(trip.nodes_lonlat[:, 0], trip.nodes_lonlat[:, 1])
    return TripGeom(np.asarray(gx), np.asarray(gy), trip.t.astype(float), np.asarray(nx), np.asarray(ny), trip.arcs, proj)


def _crop_arcs(g: TripGeom, margin: float):
    a, b = g.arcs[:, 0], g.arcs[:, 1]
    xmin, xmax = g.gx.min() - margin, g.gx.max() + margin
    ymin, ymax = g.gy.min() - margin, g.gy.max() + margin
    ax, ay, bx, by = g.nx[a], g.ny[a], g.nx[b], g.ny[b]
    keep = (np.maximum(ax, bx) >= xmin) & (np.minimum(ax, bx) <= xmax) & \
           (np.maximum(ay, by) >= ymin) & (np.minimum(ay, by) <= ymax)
    return np.nonzero(keep)[0]


def _heading_speed(g: TripGeom, w: int, min_move: float):
    T = len(g.gx)
    i = np.arange(T)
    i0 = np.clip(i - w, 0, T - 1)
    i1 = np.clip(i + w, 0, T - 1)
    dx, dy = g.gx[i1] - g.gx[i0], g.gy[i1] - g.gy[i0]
    dt = np.maximum(g.t[i1] - g.t[i0], 1e-6)
    disp = np.hypot(dx, dy)
    speed = disp / dt
    hdg = np.arctan2(dy, dx)
    valid = (disp >= min_move).astype(float)
    return hdg, speed, valid, dx, dy, dt


def generate_candidates(g: TripGeom, cfg: dict = DEFAULT_CFG) -> pd.DataFrame:
    """Return one row per (fix, arc) candidate with engineered features."""
    R, sp, K = cfg["radius"], cfg["spacing"], cfg["max_cand"]
    T = len(g.gx)
    keep = _crop_arcs(g, R + sp)
    a_n, b_n = g.arcs[keep, 0], g.arcs[keep, 1]
    ax, ay, bx, by = g.nx[a_n], g.ny[a_n], g.nx[b_n], g.ny[b_n]
    L = np.hypot(bx - ax, by - ay)
    cnt = np.ceil(L / sp).astype(np.int64) + 1
    cnt = np.maximum(cnt, 2)
    tot = int(cnt.sum())
    owner = np.repeat(np.arange(len(keep)), cnt)
    off = np.cumsum(cnt) - cnt
    k = np.arange(tot) - np.repeat(off, cnt)
    f = k / (np.repeat(cnt, 1)[owner] - 1)
    sx = ax[owner] + f * (bx[owner] - ax[owner])
    sy = ay[owner] + f * (by[owner] - ay[owner])
    tree = cKDTree(np.c_[sx, sy])
    nb = tree.query_ball_point(np.c_[g.gx, g.gy], r=R + sp / 2.0, workers=2, return_sorted=False)
    lens = np.fromiter((len(x) for x in nb), dtype=np.int64, count=T)
    if lens.sum() == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES)
    samp = np.fromiter(chain.from_iterable(nb), dtype=np.int64, count=int(lens.sum()))
    obs = np.repeat(np.arange(T), lens)
    loc = owner[samp]                       # index into cropped arc arrays
    key = np.unique(obs * len(keep) + loc)
    obs, loc = key // len(keep), key % len(keep)
    d, s, Lc, lat = seg_proj(g.gx[obs], g.gy[obs], ax[loc], ay[loc], bx[loc], by[loc])
    m = d <= R
    obs, loc, d, s, Lc, lat = obs[m], loc[m], d[m], s[m], Lc[m], lat[m]
    # keep the K nearest per fix
    order = np.lexsort((d, obs))
    obs, loc, d, s, Lc, lat = obs[order], loc[order], d[order], s[order], Lc[order], lat[order]
    first = np.r_[0, np.nonzero(np.diff(obs))[0] + 1]
    counts = np.diff(np.r_[first, len(obs)])
    rank = np.arange(len(obs)) - np.repeat(first, counts)
    m = rank < K
    obs, loc, d, s, Lc, lat, rank = obs[m], loc[m], d[m], s[m], Lc[m], lat[m], rank[m]
    arc_id = keep[loc]
    n_c = np.bincount(obs, minlength=T)[obs]

    # ---- direction agreement
    hdg, speed, hvalid, mdx, mdy, mdt = _heading_speed(g, cfg["hdg_w"], cfg["min_move"])
    abr = np.arctan2(by[loc] - ay[loc], bx[loc] - ax[loc])
    dh = wrap180(hdg[obs] - abr)
    hv = hvalid[obs]
    cos_dh = np.cos(dh) * hv
    abs_dh = np.abs(dh) * 180 / np.pi * hv + (1 - hv) * 90.0
    ux, uy = np.cos(abr), np.sin(abr)
    motion_along = (mdx[obs] * ux + mdy[obs] * uy) / mdt[obs]

    # ---- relative to other candidates of same fix
    dmin = np.minimum.reduceat(d, np.r_[0, np.nonzero(np.diff(obs))[0] + 1])
    ucount = np.bincount(obs, minlength=T)
    obs_first = np.r_[0, np.nonzero(np.diff(obs))[0] + 1]
    obs_ids = obs[obs_first]
    dmin_full = np.full(T, np.nan); dmin_full[obs_ids] = dmin
    dmin_row = dmin_full[obs]
    # rank among direction-compatible candidates (cos>0) – or plain rank if heading invalid
    comp = (cos_dh > 0) | (hv == 0)
    d_dir = np.where(comp, d, np.inf)
    df_tmp = pd.DataFrame({"obs": obs, "dd": d_dir})
    rank_dir = df_tmp.groupby("obs")["dd"].rank(method="first").to_numpy()
    rank_dir = np.where(comp, rank_dir, 99.0)

    # ---- topology
    out_deg = np.bincount(g.arcs[:, 0], minlength=len(g.nx))
    in_deg = np.bincount(g.arcs[:, 1], minlength=len(g.nx))
    an, bn = a_n[loc], b_n[loc]
    tw = pd.MultiIndex.from_arrays([g.arcs[:, 0], g.arcs[:, 1]])
    rev = pd.MultiIndex.from_arrays([bn, an])
    twin = rev.isin(tw).astype(float)

    # ---- sequence context: other fixes measured against the same arc
    A, B, C, D = ax[loc], ay[loc], bx[loc], by[loc]
    t_obs = g.t[obs]
    ctx = {}
    for kk in cfg["ctx_offsets"]:
        j = np.clip(obs + kk, 0, T - 1)
        dj, sj, _, _ = seg_proj(g.gx[j], g.gy[j], A, B, C, D)
        ok = (np.abs(g.t[j] - t_obs) <= 3 * abs(kk) + 2) & (j == obs + kk)
        ctx[kk] = (np.where(ok, dj, np.nan), np.where(ok, sj, np.nan))
    d0 = d
    dn = lambda ks: np.vstack([d0] + [ctx[k][0] for k in ks])
    w3 = dn((-3, -1, 1, 3)); w6 = dn((-6, -3, -1, 1, 3, 6))
    with np.errstate(all="ignore"):
        d_mean_w3 = np.nanmean(w3, 0)
        d_max_w6 = np.nanmax(w6, 0)
        d_mean_w6 = np.nanmean(w6, 0)
        frac_near_w6 = np.nanmean(np.where(np.isnan(w6), np.nan, (w6 <= 25).astype(float)), 0)
        prog3 = ctx[3][1] - ctx[-3][1]
        prog6 = ctx[6][1] - ctx[-6][1]
        # compare along-arc progress with straight-line displacement of the same fixes
        j3a, j3b = np.clip(obs - 3, 0, T - 1), np.clip(obs + 3, 0, T - 1)
        disp3 = np.hypot(g.gx[j3b] - g.gx[j3a], g.gy[j3b] - g.gy[j3a])
        prog_ratio3 = prog3 / np.maximum(disp3, 3.0)
    dt_prev = np.where(obs > 0, g.t[obs] - g.t[np.maximum(obs - 1, 0)], 0.0)
    dt_next = np.where(obs < T - 1, g.t[np.minimum(obs + 1, T - 1)] - g.t[obs], 0.0)

    df = pd.DataFrame({
        "obs": obs.astype(np.int32), "arc": arc_id.astype(np.int64),
        "d": d, "lat_off": lat, "frac": np.clip(s / np.maximum(Lc, 1e-6), 0, 1),
        "proj_clamped": ((s < 0) | (s > Lc)).astype(float), "arc_len": Lc,
        "dist_to_end": np.clip(Lc - s, 0, None), "dist_from_start": np.clip(s, 0, None),
        "speed": speed[obs], "hdg_valid": hv, "cos_dh": cos_dh, "abs_dh_deg": abs_dh,
        "motion_along": motion_along,
        "n_cand": n_c.astype(float), "rank_d": rank.astype(float),
        "d_minus_dmin": d - dmin_row, "d_ratio": d / (dmin_row + 1.0),
        "rank_d_dir": rank_dir, "twin_exists": twin,
        "deg_from_out": out_deg[an].astype(float), "deg_to_out": out_deg[bn].astype(float),
        "deg_from_in": in_deg[an].astype(float), "deg_to_in": in_deg[bn].astype(float),
        "d_mean_w3": d_mean_w3, "d_max_w6": d_max_w6, "d_mean_w6": d_mean_w6,
        "frac_near_w6": frac_near_w6, "prog_w3": prog3, "prog_w6": prog6, "prog_ratio_w3": prog_ratio3,
        "dt_prev": dt_prev, "dt_next": dt_next,
    })
    return df
