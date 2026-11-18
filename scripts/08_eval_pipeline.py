"""Step 8 - end-to-end evaluation of the deployed pipeline (raw trip -> candidates -> tuned HistGB -> HMM/Viterbi) on a partition
(default student_test; evaluator data is never touched).  Compares classifier-argmax vs Viterbi on the labelable fixes.
Usage: python scripts/08_eval_pipeline.py [partition=student_test] [max_trips=all]"""
import sys, json, time, numpy as np, pandas as pd
from _common import *
from pathsnap.io import list_trips, load_trip
from pathsnap.candidates import project_trip
from pathsnap.labels import align_gps_to_route
from pathsnap.matcher import Matcher
from pathsnap.metrics import build_next, _reach, route_prf
part = sys.argv[1] if len(sys.argv) > 1 else "student_test"; mx = int(sys.argv[2]) if len(sys.argv) > 2 else None
M = Matcher(f"{ARTI}/hgb_tuned.joblib", f"{ARTI}/hmm_params.json"); print("HMM params", M.params.to_dict(), flush=True)
rows = []
for tid in list_trips(DATA_ROOT, part)[:mx]:
    tp = load_trip(DATA_ROOT, part, tid); g = project_trip(tp)
    ref, dref, _ = align_gps_to_route(g.gx, g.gy, g.nx, g.ny, g.arcs, tp.route); ok = dref <= 50.0
    r = M.match(tp.lon, tp.lat, tp.t, tp.nodes_lonlat, tp.arcs, return_candidates=True)
    pred = np.array(r["fix_arcs"]); c, p = r["_cand"], r["_p"]
    top = c.assign(p=p).loc[lambda d: d.groupby("obs").p.idxmax()].set_index("obs").arc.reindex(range(len(tp.lon))).fillna(-1).astype(int).to_numpy()
    nxt = build_next(tp.arcs); cont = lambda a: np.mean([_reach(nxt, a[i], a[i + 1]) for i in range(len(a) - 1) if ok[i] and ok[i + 1] and tp.t[i + 1] - tp.t[i] <= 5] or [np.nan])
    pr, rc, f1 = route_prf(r["route"], tp.route.tolist())
    rows.append(dict(trip=tid, n_fix=len(tp.lon), n_valid=int(ok.sum()), top1_argmax=float((top[ok] == ref[ok]).mean()), top1_viterbi=float((pred[ok] == ref[ok]).mean()),
                     cont_argmax=cont(top), cont_viterbi=cont(pred), route_precision=pr, route_recall=rc, route_f1=f1, ms_total=r["timing_ms"]["total"], ms_per_fix=r["timing_ms"]["total"] / len(tp.lon)))
    print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{RES}/tables/pipeline_eval_{part}.csv", index=False)
w = D.n_valid.to_numpy()
S = dict(partition=part, n_trips=len(D), top1_argmax=float(np.average(D.top1_argmax, weights=w)), top1_viterbi=float(np.average(D.top1_viterbi, weights=w)),
         cont_argmax=float(D.cont_argmax.mean()), cont_viterbi=float(D.cont_viterbi.mean()), route_precision=float(D.route_precision.mean()), route_recall=float(D.route_recall.mean()),
         route_f1=float(D.route_f1.mean()), ms_per_fix=float(D.ms_total.sum() / D.n_fix.sum()), hmm_params=M.params.to_dict())
json.dump(S, open(f"{RES}/tables/pipeline_eval_{part}.json", "w"), indent=1); print(json.dumps(S, indent=1))
