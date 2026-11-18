"""Step 5 - hyper-parameter tuning of the selected model: HistGradientBoosting (sklearn) with Optuna (TPE, seed 42).
Protocol (same as stage 1, no evaluator data):
  * search: 3-fold GroupKFold (groups = trips) on a 15% fix-level subsample of the 80 training trips
  * objective (declared up front, same as model selection): CV observation-level Top-1; log-loss logged for tie-breaks
  * final: refit best config on all 80 train trips, 5-fold OOF scores for ALL fixes (feeds the HMM tuning), student_test comparison vs the stage-1 config
Usage: python scripts/05_tune_hgb.py [n_trials=20] [timeout_min=30]"""
import sys, time, json, warnings
import numpy as np, pandas as pd, optuna, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from _common import *
from pathsnap.candidates import FEATURES
from pathsnap.metrics import top1_table, candidate_metrics, threshold_scan
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TIMEOUT = 60 * float(sys.argv[2]) if len(sys.argv) > 2 else 1800
SUB = 0.15
tr_all, te = load_candidates("train", valid_only=False), load_candidates("student_test")
tr = tr_all[tr_all.obs_valid].reset_index(drop=True)
for d in (tr, tr_all, te): d[FEATURES] = d[FEATURES].astype(np.float32)
print("train", len(tr), "test", len(te), flush=True)

def sub_obs(df, frac, seed):
    rng = np.random.RandomState(seed); keys = df[["trip", "obs"]].drop_duplicates()
    return df.merge(keys[rng.rand(len(keys)) < frac], on=["trip", "obs"])
top1 = lambda d, p: float(top1_table(d, p).y.mean())
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
BASE = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31)            # stage-1 HistGradBoost
mk = lambda **kw: HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **kw)

trs = sub_obs(tr, SUB, SEED).reset_index(drop=True)
folds = list(GroupKFold(n_splits=3).split(trs, groups=trs.trip)); print("search subsample rows", len(trs), flush=True)

def objective(trial):
    P = dict(max_iter=trial.suggest_int("max_iter", 100, 500, step=50), learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
             max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127, log=True), min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 400, log=True),
             l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 30, log=True), max_features=trial.suggest_float("max_features", 0.4, 1.0),
             max_bins=trial.suggest_categorical("max_bins", [63, 127, 255]))
    t1s, lls = [], []
    for k, (a, b) in enumerate(folds):
        m = mk(**P).fit(trs.loc[a, FEATURES], trs.y[a]); p = m.predict_proba(trs.loc[b, FEATURES])[:, 1]
        t1s.append(top1(trs.loc[b], p)); lls.append(ll(trs.y[b].to_numpy(), p)); trial.report(float(np.mean(t1s)), k)
        if trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr("logloss", float(np.mean(lls))); return float(np.mean(t1s))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1), study_name="hgb", storage=f"sqlite:///{RES}/tables/optuna_hgb.db", load_if_exists=True)
t0 = time.time(); b_t1, b_ll = [], []
for a, b in folds:
    p = mk(**BASE).fit(trs.loc[a, FEATURES], trs.y[a]).predict_proba(trs.loc[b, FEATURES])[:, 1]; b_t1.append(top1(trs.loc[b], p)); b_ll.append(ll(trs.y[b].to_numpy(), p))
print(f"baseline (stage-1 config) on search folds: top1={np.mean(b_t1):.4f} ll={np.mean(b_ll):.4f} ({time.time()-t0:.0f}s)", flush=True)
if not study.trials:
    study.enqueue_trial(dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1e-3, max_features=1.0, max_bins=255))
cb = lambda st, t: print(f" trial {t.number:2d} {t.state.name:8s} top1={t.value if t.value is not None else float('nan'):.4f} best={st.best_value:.4f} {t.params if t.state.name=='COMPLETE' else ''}", flush=True)
study.optimize(objective, n_trials=max(N_TRIALS - len(study.trials), 0), timeout=TIMEOUT, callbacks=[cb])
done = [t for t in study.trials if t.state.name == "COMPLETE"]
best = max(done, key=lambda t: (round(t.value, 4), -t.user_attrs["logloss"]))           # tie-break: log-loss
FIN = dict(best.params); print("BEST", best.number, best.value, FIN, flush=True)
pd.DataFrame([dict(trial=t.number, top1=t.value, logloss=t.user_attrs.get("logloss"), state=t.state.name, **t.params) for t in study.trials]).to_csv(f"{RES}/tables/tuning_hgb_trials.csv", index=False)

folds5 = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip)); oof = np.full(len(tr), np.nan); f_rows = []; sc_all = []
for k, (a, b) in enumerate(folds5):
    m = mk(**FIN).fit(tr.loc[a, FEATURES], tr.y[a]); p = m.predict_proba(tr.loc[b, FEATURES])[:, 1]; oof[b] = p
    held = tr_all[tr_all.trip.isin(set(tr.trip[b]))]          # ALL fixes (incl. off-reference) of held-out trips -> HMM tuning input
    sc_all.append(held[["trip", "obs", "arc", "y", "obs_valid"]].assign(p=m.predict_proba(held[FEATURES])[:, 1].astype(np.float32)))
    f_rows.append(dict(fold=k, top1=top1(tr.loc[b], p), macro_f1=candidate_metrics(tr.y[b].values, p, .5)["macro_f1"])); print(" fold", f_rows[-1], flush=True)
np.save(f"{PROC}/oof_HGB_tuned.npy", oof); pd.concat(sc_all).to_parquet(f"{PROC}/oof_scores.parquet", index=False)
thr, _ = threshold_scan(tr.y.values, oof); cv_c = candidate_metrics(tr.y.values, oof, thr)
t0 = time.time(); final = mk(**FIN).fit(tr[FEATURES], tr.y); fit_s = time.time() - t0
t0 = time.time(); pte = final.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0; te_c = candidate_metrics(te.y.values, pte, thr)
pb = mk(**BASE).fit(tr[FEATURES], tr.y).predict_proba(te[FEATURES])[:, 1]; bc = candidate_metrics(te.y.values, pb, .5)
out = dict(model="HistGradientBoosting", search=dict(n_trials=len(study.trials), n_complete=len(done), subsample=SUB, folds=3, best_trial=best.number, best_search_top1=best.value, baseline_search_top1=float(np.mean(b_t1)), baseline_search_logloss=float(np.mean(b_ll))),
    params=FIN, threshold=thr, seed=SEED, features=FEATURES,
    cv5=dict(top1=top1(tr, oof), top1_fold_std=float(np.std([r["top1"] for r in f_rows], ddof=1)), macro_f1=cv_c["macro_f1"], precision=cv_c["precision"], recall=cv_c["recall"], log_loss=cv_c["log_loss"], folds=f_rows),
    student_test=dict(top1=top1(te, pte), macro_f1=te_c["macro_f1"], precision=te_c["precision"], recall=te_c["recall"], log_loss=te_c["log_loss"],
                      baseline_top1=top1(te, pb), baseline_log_loss=bc["log_loss"], baseline_macro_f1_at_0p5=bc["macro_f1"]),
    timing=dict(full_fit_s=fit_s, inference_s=inf_s))
json.dump(out, open(f"{RES}/tables/tuning_hgb.json", "w"), indent=1)
joblib.dump(dict(model=final, features=FEATURES, threshold=thr, params=FIN), f"{ARTI}/hgb_tuned.joblib")
print(json.dumps({k: out[k] for k in ("cv5", "student_test", "timing")}, indent=1)[:1800])
