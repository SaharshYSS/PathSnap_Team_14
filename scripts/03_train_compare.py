"""Step 3 - train & compare 6 model families (LogReg, GaussianNB, DecisionTree, HistGradBoost, LightGBM, MLP) as candidate scorers.
Protocol (no evaluator data involved):
  * fit / cross-validate on the 80 training trips with GroupKFold (groups = trips, 5 folds)
  * threshold for F1 chosen on out-of-fold predictions only
  * development comparison on the 10 student-test trips
Selection metric (declared before running): CV observation-level Top-1 accuracy; tie-breakers CV log-loss
(probability quality for HMM emissions), CV route continuity, inference time."""
import time, json, sys, glob, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb, lightgbm as lgb
import joblib
from _common import *
from pathsnap.candidates import FEATURES
from pathsnap.io import load_trip
from pathsnap.metrics import *
warnings.filterwarnings("ignore")

def load(part):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{PROC}/{part}/*.parquet"))], ignore_index=True)
    return d[d.obs_valid].reset_index(drop=True)
tr, te = load("train"), load("student_test")
print("train rows", len(tr), "pos rate", tr.y.mean(), "| test rows", len(te), flush=True)

# arc graphs + raw time stamps (evaluation only)
NEXT, TIME, ROUTE = {}, {}, {}
for part, d in (("train", tr), ("student_test", te)):
    for tid in d.trip.unique():
        tp = load_trip(DATA_ROOT, part, tid); NEXT[tid] = build_next(tp.arcs); TIME[tid] = tp.t; ROUTE[tid] = set(tp.route.tolist())

def sub_obs(df, frac, seed):
    if frac >= 1: return df
    rng = np.random.RandomState(seed)
    keys = df[["trip", "obs"]].drop_duplicates()
    keys = keys[rng.rand(len(keys)) < frac]
    return df.merge(keys, on=["trip", "obs"])

pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
N_JOBS = 2
MODELS = {
 "LogReg": dict(family="Linear / probabilistic", cfg=dict(C=1.0, penalty="l2", max_iter=300, solver="lbfgs"), frac=1.0, eval_frac=1.0,
   make=lambda: make_pipeline(*pre(), LogisticRegression(C=1.0, max_iter=300))),
 "GaussianNB": dict(family="Linear / probabilistic", cfg=dict(var_smoothing=1e-9), frac=1.0, eval_frac=1.0,
   make=lambda: make_pipeline(*pre(), GaussianNB())),
 "DecisionTree": dict(family="Tree-based", cfg=dict(max_depth=12, min_samples_leaf=20), frac=1.0, eval_frac=1.0,
   make=lambda: make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED))),
 "HistGradBoost": dict(family="Boosting (sklearn)", cfg=dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31), frac=1.0, eval_frac=1.0,
   make=lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
 "LightGBM": dict(family="Boosting (LightGBM)", cfg=dict(n_estimators=400, num_leaves=31, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8), frac=1.0, eval_frac=1.0,
   make=lambda: lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED, verbose=-1)),
 "MLP": dict(family="Neural network", cfg=dict(hidden=(64, 32), activation="relu", alpha=1e-4, lr=1e-3, epochs=30, train_obs_frac=0.3), frac=0.3, eval_frac=1.0,
   make=lambda: make_pipeline(*pre(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=30, early_stopping=False, random_state=SEED))),
}
only = sys.argv[1:] or list(MODELS)

def route_level(d, p):
    """obs-level top-1 metrics on labelable fixes + continuity + route membership."""
    t1 = top1_table(d, p)
    top1 = float(t1.y.mean())
    ok = tot = 0; inroute = 0; n = 0
    for tid, g in t1.groupby("trip"):
        g = g.sort_values("obs"); o, tt = g.obs.to_numpy(), TIME[tid][g.obs.to_numpy()]
        a = g.arc.to_numpy()
        pair_ok = (np.diff(o) == 1)
        # continuity only for adjacent labelable fixes
        for i in np.nonzero(pair_ok)[0]:
            if tt[i + 1] - tt[i] > 5: continue
            tot += 1; ok += _r(NEXT[tid], a[i], a[i + 1])
        inroute += sum(x in ROUTE[tid] for x in a); n += len(a)
    return dict(top1=top1, continuity=ok / max(tot, 1), route_membership=inroute / n, n_fix=len(t1))
from pathsnap.metrics import _reach as _r

results = json.load(open(f"{RES}/tables/_model_results.json")) if os.path.exists(f"{RES}/tables/_model_results.json") else {}
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(tr, groups=tr.trip))
for name in only:
    M = MODELS[name]; print(f"\n=== {name}", flush=True); t_all = time.time()
    oof = np.full(len(tr), np.nan); fold_rows = []; fit_t = []
    for k, (a, b) in enumerate(folds):
        dtr = sub_obs(tr.iloc[a], M["frac"], SEED + k); dva = sub_obs(tr.iloc[b], M["eval_frac"], SEED + 100 + k)
        t0 = time.time(); mdl = M["make"](); mdl.fit(dtr[FEATURES], dtr.y); fit_t.append(time.time() - t0)
        p = mdl.predict_proba(dva[FEATURES])[:, 1]; oof[dva.index.to_numpy()] = p
        cm = candidate_metrics(dva.y.values, p, 0.5); rl = route_level(dva, p)
        fold_rows.append(dict(fold=k, **{kk: v for kk, v in cm.items() if kk != "cm"}, **rl, fit_s=fit_t[-1]))
        print(f" fold{k}: top1={rl['top1']:.4f} macroF1@.5={cm['macro_f1']:.4f} logloss={cm['log_loss']:.4f} cont={rl['continuity']:.4f} fit={fit_t[-1]:.1f}s", flush=True)
    m = ~np.isnan(oof); dcv = tr[m]; pcv = oof[m]
    thr, _ = threshold_scan(dcv.y.values, pcv)
    cv_c = candidate_metrics(dcv.y.values, pcv, thr); cv_r = route_level(dcv, pcv)
    folds_df = pd.DataFrame(fold_rows)
    # ---- final refit on all 80 training trips -> student-test
    dfit = sub_obs(tr, M["frac"], SEED); t0 = time.time(); mdl = M["make"](); mdl.fit(dfit[FEATURES], dfit.y); fit_full = time.time() - t0
    t0 = time.time(); pte = mdl.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0
    te_c = candidate_metrics(te.y.values, pte, thr); te_c05 = candidate_metrics(te.y.values, pte, 0.5); te_r = route_level(te, pte)
    dtr_eval = sub_obs(tr, 0.1, SEED + 7); ptr = mdl.predict_proba(dtr_eval[FEATURES])[:, 1]
    tr_c = candidate_metrics(dtr_eval.y.values, ptr, thr); tr_r = route_level(dtr_eval, ptr)
    hist = None
    # epoch-wise histories (fold 0 train vs validation) only for models that expose them
    a, b = folds[0]; dtr0 = sub_obs(tr.iloc[a], M["frac"], SEED); dva0 = tr.iloc[b]
    if name == "XGBoost":
        h = M["make"](); h.fit(dtr0[FEATURES], dtr0.y, eval_set=[(dtr0[FEATURES], dtr0.y), (dva0[FEATURES], dva0.y)], verbose=False)
        r = h.evals_result(); hist = dict(train=r["validation_0"]["logloss"], val=r["validation_1"]["logloss"], metric="logloss")
    elif name == "LightGBM":
        h = M["make"](); h.fit(dtr0[FEATURES], dtr0.y, eval_set=[(dtr0[FEATURES], dtr0.y), (dva0[FEATURES], dva0.y)], eval_metric="binary_logloss")
        r = h.evals_result_; ks = list(r); hist = dict(train=r[ks[0]]["binary_logloss"], val=r[ks[1]]["binary_logloss"], metric="logloss")
    elif name == "MLP":
        h = M["make"](); h.fit(dtr0[FEATURES], dtr0.y); hist = dict(train=[float(x) for x in h[-1].loss_curve_], val=None, metric="train loss (log-loss)")
    elif name == "HistGradBoost":
        h = M["make"](); h.fit(dtr0[FEATURES], dtr0.y)  # sklearn HGB keeps no per-iteration eval history by default
    res = dict(model=name, family=M["family"], config=M["cfg"], seed=SEED, threshold=thr,
        cv=dict(macro_f1=cv_c["macro_f1"], macro_f1_fold_mean=float(folds_df.macro_f1.mean()), macro_f1_fold_std=float(folds_df.macro_f1.std()), top1_fold_std=float(folds_df.top1.std()),
                precision=cv_c["precision"], recall=cv_c["recall"], log_loss=cv_c["log_loss"], top1=cv_r["top1"], continuity=cv_r["continuity"], route_membership=cv_r["route_membership"], n_fix=cv_r["n_fix"]),
        student_test=dict(macro_f1=te_c["macro_f1"], macro_f1_at_0p5=te_c05["macro_f1"], precision=te_c["precision"], recall=te_c["recall"], log_loss=te_c["log_loss"], top1=te_r["top1"],
                continuity=te_r["continuity"], route_membership=te_r["route_membership"], cm=te_c["cm"], n_fix=te_r["n_fix"], n_rows=len(te)),
        train_fit=dict(macro_f1=tr_c["macro_f1"], top1=tr_r["top1"], log_loss=tr_c["log_loss"]),
        timing=dict(cv_fit_s_mean=float(np.mean(fit_t)), full_fit_s=fit_full, inference_s_student_test=inf_s, inference_us_per_candidate=inf_s / len(te) * 1e6, inference_ms_per_fix=inf_s / te.obs.nunique() * 1e3 if False else inf_s / len(te[["trip","obs"]].drop_duplicates()) * 1e3),
        history=hist, folds=fold_rows)
    results[name] = res
    json.dump(results, open(f"{RES}/tables/_model_results.json", "w"), indent=1)
    np.save(f"{PROC}/oof_{name}.npy", oof); 
    if name != "RandomForest" and name != "KNN": joblib.dump(mdl, f"{ARTI}/{name}.joblib") if (ARTI := ROOT + "/artifacts") else None
    print(f" CV: top1={cv_r['top1']:.4f} F1={cv_c['macro_f1']:.4f}(thr {thr:.2f}) ll={cv_c['log_loss']:.4f} cont={cv_r['continuity']:.4f}\n TEST: top1={te_r['top1']:.4f} F1={te_c['macro_f1']:.4f} ll={te_c['log_loss']:.4f} cont={te_r['continuity']:.4f} inf={inf_s:.1f}s | train top1={tr_r['top1']:.4f} | total {time.time()-t_all:.0f}s", flush=True)
