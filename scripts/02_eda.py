"""Step 2 - exploratory data analysis.  Partition used is stated in every figure title.
Outputs: results/tables/*.csv, results/figures/eda_*.png, maps/*.html, results/eda_data_quality.md"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from _common import *
from pathsnap.io import list_trips, load_trip
from pathsnap.candidates import project_trip
from pathsnap.viz import make_trip_map

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
obs = pd.read_parquet(f"{PROC}/obs_all.parquet")
rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(DATA_ROOT, part):
        tp = load_trip(DATA_ROOT, part, tid); g = project_trip(tp)
        N, A, T = len(g.nx), len(tp.arcs), len(tp.lon)
        ar = tp.arcs; rt = tp.route
        dt = np.diff(tp.t); step = np.hypot(np.diff(g.gx), np.diff(g.gy))
        L = np.hypot(g.nx[ar[:, 1]] - g.nx[ar[:, 0]], g.ny[ar[:, 1]] - g.ny[ar[:, 0]])
        adj = coo_matrix((np.ones(A), (ar[:, 0], ar[:, 1])), shape=(N, N))
        nw, lw = connected_components(adj, directed=False)
        ns, ls = connected_components(adj, directed=True, connection="strong")
        deg = np.bincount(ar[:, 0], minlength=N) + np.bincount(ar[:, 1], minlength=N)
        s = set(map(tuple, ar.tolist()))
        rev = np.mean([(b, a) in s for a, b in ar.tolist()])
        o = obs[(obs.partition == part) & (obs.trip == tid)]
        base = os.path.join(DATA_ROOT, part, tid, tid)
        rows.append(dict(
            partition=part, trip=tid, n_gps=T, n_nodes=N, n_arcs=A, n_route_arcs=len(rt),
            size_kb=sum(os.path.getsize(base + e) for e in (".track", ".nodes", ".arcs", ".route")) / 1024,
            nan_gps=int(np.isnan(np.c_[tp.lon, tp.lat, tp.t]).sum()), nan_net=int(np.isnan(tp.nodes_lonlat).sum()),
            lon_ok=bool(np.all(np.abs(tp.lon) <= 180)), lat_ok=bool(np.all(np.abs(tp.lat) <= 90)),
            t_monotonic=bool(np.all(dt > 0)), dt_eq_1_frac=float((dt == 1).mean()), dt_max=float(dt.max()), n_gaps_gt10s=int((dt > 10).sum()),
            speed_med_ms=float(np.median(step / dt)), speed_p99_ms=float(np.percentile(step / dt, 99)), max_jump_m=float(step.max()),
            gps_path_km=float(step.sum() / 1e3), route_km=float(L[rt].sum() / 1e3),
            extent_x_km=float(np.ptp(g.gx) / 1e3), extent_y_km=float(np.ptp(g.gy) / 1e3),
            arcs_bad_node=int(((ar < 0) | (ar >= N)).any(1).sum()), self_loops=int((ar[:, 0] == ar[:, 1]).sum()),
            dup_arcs=int(A - len(s)), frac_arcs_with_reverse=float(rev), isolated_nodes=int((deg == 0).sum()),
            weak_components=int(nw), largest_weak_frac=float(np.bincount(lw).max() / N), strong_components=int(ns),
            arc_len_med_m=float(np.median(L)), arc_len_max_m=float(L.max()),
            route_bad_id=int(((rt < 0) | (rt >= A)).sum()), route_connected_frac=float(np.mean(ar[rt[:-1], 1] == ar[rt[1:], 0])) if len(rt) > 1 else 1.0,
            route_repeats=int(len(rt) - len(set(rt.tolist()))),
            label_valid_frac=float(o.valid.mean()), d_ref_med=float(o.d_ref.median()), d_ref_p95=float(o.d_ref.quantile(.95)),
            cand_per_fix=float(o.n_cand.mean()), fixes_no_candidate=int((o.n_cand == 0).sum()),
        ))
inv = pd.DataFrame(rows); inv.to_csv(f"{RES}/tables/dataset_inventory.csv", index=False)
summ = inv.groupby("partition")[["n_gps", "n_nodes", "n_arcs", "n_route_arcs", "gps_path_km", "route_km", "dt_eq_1_frac", "label_valid_frac"]].agg(["sum", "median", "min", "max"])
summ.to_csv(f"{RES}/tables/dataset_summary_by_partition.csv")

# ---------------- figures
tr, st = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].hist(np.log10(tr.n_gps), 20, alpha=.7, label="train (80)"); ax[0, 0].hist(np.log10(st.n_gps), 10, alpha=.7, label="student_test (10)")
ax[0, 0].set(title="GPS fixes per trip", xlabel="log10(#fixes)"); ax[0, 0].legend()
ax[0, 1].scatter(tr.n_nodes, tr.n_arcs, s=14, label="train"); ax[0, 1].scatter(st.n_nodes, st.n_arcs, s=22, label="student_test")
ax[0, 1].set(xscale="log", yscale="log", title="Road-network size per trip", xlabel="#nodes", ylabel="#arcs"); ax[0, 1].legend()
ax[0, 2].hist(inv.dt_eq_1_frac, 25); ax[0, 2].set(title="Share of 1-second steps per trip (train+student_test)", xlabel="fraction dt==1s")
ax[1, 0].scatter(inv.gps_path_km, inv.route_km, s=16, c=np.where(inv.label_valid_frac < .95, "crimson", "steelblue"))
mx = max(inv.gps_path_km.max(), inv.route_km.max()); ax[1, 0].plot([0, mx], [0, mx], "k--", lw=.8)
ax[1, 0].set(title="GPS path length vs reference route length (red = <95% labelable)", xlabel="GPS path km", ylabel="route km", xscale="log", yscale="log")
ax[1, 1].hist(obs[obs.valid & (obs.partition == "train")].d_ref.clip(upper=60), 60, alpha=.7, label="train", density=True)
ax[1, 1].hist(obs[obs.valid & (obs.partition == "student_test")].d_ref.clip(upper=60), 60, alpha=.6, label="student_test", density=True)
ax[1, 1].set(title="GPS noise: distance fix -> aligned reference arc (m)", xlabel="metres"); ax[1, 1].legend()
ax[1, 2].hist(inv.weak_components, 30); ax[1, 2].set(title="Weakly-connected components per road graph", xlabel="#components", yscale="log")
fig.tight_layout(); fig.savefig(f"{RES}/figures/eda_overview.png"); plt.close(fig)

# candidate radius sensitivity (partition: train, labelable fixes only)
v = obs[obs.valid & (obs.partition == "train")]
Rs = np.arange(10, 71, 5); miss = [(v.d_ref > R).mean() for R in Rs]
pd.DataFrame({"radius_m": Rs, "candidate_miss_rate": miss}).to_csv(f"{RES}/tables/candidate_radius_sensitivity.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(Rs, miss, "o-"); ax[0].axvline(60, color="r", ls="--", label="chosen R=60 m"); ax[0].set(title="Candidate miss rate vs search radius (train, labelable fixes)", xlabel="radius (m)", ylabel="miss rate"); ax[0].legend()
ax[1].hist(obs[obs.partition == "train"].n_cand, 40); ax[1].set(title="Candidates per fix at R=60 m (train)", xlabel="#candidate arcs")
fig.tight_layout(); fig.savefig(f"{RES}/figures/eda_candidates.png"); plt.close(fig)

# feature separation (train, labelable fixes) - subsample for plotting
from pathsnap.candidates import FEATURES
df = pd.concat([pd.read_parquet(f) for f in sorted(__import__("glob").glob(f"{PROC}/train/*.parquet"))])
df = df[df.obs_valid]
df.groupby("y")[FEATURES].mean().T.to_csv(f"{RES}/tables/feature_means_by_class.csv")
miss_tab = df[FEATURES].isna().mean().rename("missing_frac"); miss_tab.to_csv(f"{RES}/tables/feature_missing_values.csv")
sm = df.sample(150000, random_state=SEED)
show = ["d", "cos_dh", "d_mean_w3", "prog_ratio_w3", "frac_near_w6", "rank_d", "arc_len", "motion_along"]
fig, ax = plt.subplots(2, 4, figsize=(17, 7))
for a, f in zip(ax.ravel(), show):
    lo, hi = sm[f].quantile([.01, .99]); b = np.linspace(lo, hi, 40)
    a.hist(sm.loc[sm.y == 0, f].dropna().clip(lo, hi), b, alpha=.6, density=True, label="not true arc"); a.hist(sm.loc[sm.y == 1, f].dropna().clip(lo, hi), b, alpha=.6, density=True, label="true arc"); a.set_title(f)
ax[0, 0].legend(); fig.suptitle("Feature distributions by class (partition: train, labelable fixes, 150k-row sample)"); fig.tight_layout(); fig.savefig(f"{RES}/figures/eda_feature_distributions.png"); plt.close(fig)
corr = sm[FEATURES + ["y"]].corr()
fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7); ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("Feature correlation (train sample)"); ax.grid(False); fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{RES}/figures/eda_feature_correlation.png"); plt.close(fig)

# trajectory plots on projected coordinates: a clean trip and the two anomalous ones
def plot_trip(ax, part, tid):
    tp = load_trip(DATA_ROOT, part, tid); g = project_trip(tp)
    a, b = tp.arcs[:, 0], tp.arcs[:, 1]
    xm, xM, ym, yM = g.gx.min() - 100, g.gx.max() + 100, g.gy.min() - 100, g.gy.max() + 100
    k = (np.maximum(g.nx[a], g.nx[b]) >= xm) & (np.minimum(g.nx[a], g.nx[b]) <= xM) & (np.maximum(g.ny[a], g.ny[b]) >= ym) & (np.minimum(g.ny[a], g.ny[b]) <= yM)
    for i in np.nonzero(k)[0][::max(1, k.sum() // 4000)]:
        ax.plot([g.nx[a[i]], g.nx[b[i]]], [g.ny[a[i]], g.ny[b[i]]], c="#bbbbbb", lw=.5)
    r = tp.route; 
    for i in r: ax.plot([g.nx[a[i]], g.nx[b[i]]], [g.ny[a[i]], g.ny[b[i]]], c="#1a9850", lw=2)
    ax.scatter(g.gx[::max(1, len(g.gx) // 800)], g.gy[::max(1, len(g.gx) // 800)], s=3, c="#2166ac", zorder=3)
    ax.set_title(f"{part} trip {tid}", fontsize=9); ax.set_aspect("equal"); ax.set_xlim(xm, xM); ax.set_ylim(ym, yM); ax.tick_params(labelsize=6)
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for a, (p, t) in zip(ax, [("train", "00000048"), ("train", "00000013"), ("train", "00000054"), ("student_test", "00000077")]): plot_trip(a, p, t)
fig.suptitle("GPS (blue), road network (grey), reference route (green) in local UTM metres; trip 00000054 shows a truncated reference route"); fig.tight_layout(); fig.savefig(f"{RES}/figures/eda_trajectories.png"); plt.close(fig)

# Folium maps (code in src/pathsnap/viz.py)
for part, tid in [("train", "00000048"), ("student_test", "00000077")]:
    tp = load_trip(DATA_ROOT, part, tid)
    m = make_trip_map(tp, title=f"{part} trip {tid}: GPS vs network vs reference"); m.save(f"{ROOT}/maps/eda_{part}_{tid}.html")
tp = load_trip(DATA_ROOT, "train", "00000048"); g = project_trip(tp)
cd = pd.read_parquet(f"{PROC}/train/00000048.parquet"); fx = 600
m = make_trip_map(tp, title="Candidate arcs of one fix (train 00000048)", candidate_fix=fx, candidate_arcs=cd[cd.obs == fx].arc.values)
m.save(f"{ROOT}/maps/eda_candidates_train_00000048.html")

# ---------------- data-quality record
tot = inv.groupby("partition")[["n_gps", "n_nodes", "n_arcs"]].sum()
bad = inv[inv.label_valid_frac < .95][["partition", "trip", "n_gps", "gps_path_km", "route_km", "label_valid_frac"]]
with open(f"{RES}/eda_data_quality.md", "w") as f:
    f.write("# Data-quality record (generated by scripts/02_eda.py)\n\n")
    f.write(f"- Partitions: train={len(tr)} trips, student_test={len(st)} trips; evaluator_test NOT provided/used.\n")
    f.write(f"- Totals:\n{tot.to_markdown()}\n\n")
    f.write(f"- Missing/invalid GPS values: {int(inv.nan_gps.sum())}; missing network values: {int(inv.nan_net.sum())}; out-of-range lon/lat trips: {int((~inv.lon_ok | ~inv.lat_ok).sum())}.\n")
    f.write(f"- Timestamps strictly increasing in all trips: {bool(inv.t_monotonic.all())}; trips with gaps >10 s: {int((inv.n_gaps_gt10s > 0).sum())}; largest gap {inv.dt_max.max():.0f} s.\n")
    f.write(f"- Arcs referencing non-existent nodes: {int(inv.arcs_bad_node.sum())}; self-loops: {int(inv.self_loops.sum())}; duplicate arcs: {int(inv.dup_arcs.sum())}; isolated nodes: {int(inv.isolated_nodes.sum())}.\n")
    f.write(f"- Reference routes: invalid arc ids {int(inv.route_bad_id.sum())}; consecutive arcs connected in {inv.route_connected_frac.min():.4f}..1.0 of transitions (min over trips); trips with repeated arcs {int((inv.route_repeats > 0).sum())}.\n")
    f.write(f"- Graphs are directed; ~{inv.frac_arcs_with_reverse.mean()*100:.0f}% of arcs have a reverse twin (two-way streets). Median weak-components per graph: {inv.weak_components.median():.0f}.\n")
    f.write(f"- No per-fix labels exist: labels were derived by monotone alignment of fixes to the reference route (src/pathsnap/labels.py).\n")
    f.write(f"- Fixes labelable (<=50 m from aligned reference arc): train {obs[obs.partition=='train'].valid.mean():.3f}, student_test {obs[obs.partition=='student_test'].valid.mean():.4f}.\n")
    f.write(f"- Trips whose reference route is incomplete (<95% labelable):\n{bad.to_markdown(index=False)}\n")
print("EDA done"); print(bad)
