"""Step 1 - preprocessing: project -> candidate generation -> features -> reference labels.
Writes data/processed/{partition}/{trip}.parquet (candidate rows) and obs_{partition}.parquet.
Reference labels (monotone alignment of GPS fixes to the reference route) are attached here and
are used for training / evaluation only."""
import time, json
import numpy as np, pandas as pd
from _common import *
from pathsnap.io import list_trips, load_trip
from pathsnap.candidates import project_trip, generate_candidates, DEFAULT_CFG
from pathsnap.labels import align_gps_to_route

LABEL_TOL = 50.0   # a fix farther than this from its aligned reference arc is flagged 'off-reference'
log = open(f"{RES}/logs/01_build_dataset.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

P("config", json.dumps({**DEFAULT_CFG, "label_tol": LABEL_TOL, "align_cap": 60.0}))
obs_rows = []
for part in ["train", "student_test"]:
    os.makedirs(f"{PROC}/{part}", exist_ok=True)
    for tid in list_trips(DATA_ROOT, part):
        t0 = time.time()
        tp = load_trip(DATA_ROOT, part, tid)
        g = project_trip(tp)
        df = generate_candidates(g, DEFAULT_CFG)
        ref, dref, pos = align_gps_to_route(g.gx, g.gy, g.nx, g.ny, g.arcs, tp.route)
        valid = dref <= LABEL_TOL
        df["y"] = (df["arc"].to_numpy() == ref[df["obs"].to_numpy()]).astype(np.int8)
        df["obs_valid"] = valid[df["obs"].to_numpy()]
        df["trip"] = tid
        df.to_parquet(f"{PROC}/{part}/{tid}.parquet", index=False)
        hit = df.groupby("obs")["y"].max().reindex(range(len(g.gx))).fillna(0).to_numpy()
        ncand = np.bincount(df["obs"].to_numpy(), minlength=len(g.gx))
        obs_rows.append(pd.DataFrame({"partition": part, "trip": tid, "obs": np.arange(len(g.gx)), "t": g.t,
                                      "ref_arc": ref, "d_ref": dref, "route_pos": pos, "valid": valid,
                                      "in_candidates": hit.astype(bool), "n_cand": ncand}))
        P(f"{part} {tid} T={len(g.gx)} rows={len(df)} valid={valid.mean():.3f} miss(valid)={1-hit[valid].mean():.4f} {time.time()-t0:.1f}s")
pd.concat(obs_rows).to_parquet(f"{PROC}/obs_all.parquet", index=False)
P("done")
