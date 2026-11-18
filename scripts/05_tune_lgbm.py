"""Step 5 - hyper-parameter tuning of the selected scorer (LightGBM) with Optuna (TPE, seed 42).
Protocol (same as stage 1, no evaluator data):
  * search: 3-fold GroupKFold (groups = trips) on a 15% fix-level subsample of the 80 training trips, early stopping on the validation fold
  * objective (declared up front, same as model selection): CV observation-level Top-1; log-loss is logged for tie-breaks
  * final: refit best config on all 80 train trips, 5-fold OOF predictions (feed the HMM tuning), student_test evaluation vs stage-1 baseline
Usage: python scripts/05_tune_lgbm.py [n_trials=20] [timeout_min=30]"""
import sys, time, json, warnings
import numpy as np, pandas as pd, optuna, joblib, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from _common import *
from pathsnap.candidates import FEATURES
from pathsnap.metrics import top1_table, candidate_metrics, threshold_scan
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TIMEOUT = 60 * float(sys.argv[2]) if len(sys.argv) > 2 else 1800
SUB = 0.15
N_JOBS = os.cpu_count() or 1
tr_all, te = load_candidates("train", valid_only=False), load_candidates("student_test")
tr = tr_all[tr_all.obs_valid].reset_index(drop=True)
tr[FEATURES] = tr[FEATURES].astype(np.float32); tr_all[FEATURES] = tr_all[FEATURES].astype(np.float32); te[FEATURES] = te[FEATURES].astype(np.float32)
print("train", len(tr), "test", len(te), flush=True)

def sub_obs(df, frac, seed):
    rng = np.random.RandomState(seed); keys = df[["trip", "obs"]].drop_duplicates()
    return df.merge(keys[rng.rand(len(keys)) < frac], on=["trip", "obs"])
top1 = lambda d, p: float(top1_table(d, p).y.mean())

BASE = dict(n_estimators=400, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)   # stage-1 LightGBM
mk = lambda **kw: lgb.LGBMClassifier(**{**dict(n_jobs=N_JOBS, random_state=SEED, verbose=-1, subsample_freq=1), **kw})

# ---------------- search
trs = sub_obs(tr, SUB, SEED).reset_index(drop=True)
folds = list(GroupKFold(n_splits=3).split(trs, groups=trs.trip))
print("search subsample rows", len(trs), flush=True)

def objective(trial):
    P = dict(num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
             learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
             min_child_samples=trial.suggest_int("min_child_samples", 20, 400, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
             min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]))
    t1s, lls, its = [], [], []
    for k, (a, b) in enumerate(folds):
        m = mk(n_estimators=600, **P)
        m.fit(trs.loc[a, FEATURES], trs.y[a], eval_set=[(trs.loc[b, FEATURES], trs.y[b])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(30, verbose=False)])
        p = m.predict_proba(trs.loc[b, FEATURES])[:, 1]
        t1s.append(top1(trs.loc[b], p)); its.append(m.best_iteration_ or 600)
        lls.append(float(-np.mean(trs.y[b] * np.log(np.clip(p, 1e-7, 1)) + (1 - trs.y[b]) * np.log(np.clip(1 - p, 1e-7, 1)))))
        trial.report(float(np.mean(t1s)), k)
        if trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr("logloss", float(np.mean(lls))); trial.set_user_attr("best_iter", int(np.mean(its)))
    return float(np.mean(t1s))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1), study_name="lgbm",
        storage=f"sqlite:///{RES}/tables/optuna_lgbm.db", load_if_exists=True)
# baseline (stage-1 config) on the same search folds, as trial 0 reference
t0 = time.time(); b_t1, b_ll = [], []
for a, b in folds:
    m = mk(**BASE).fit(trs.loc[a, FEATURES], trs.y[a]); p = m.predict_proba(trs.loc[b, FEATURES])[:, 1]
    b_t1.append(top1(trs.loc[b], p)); b_ll.append(float(-np.mean(trs.y[b] * np.log(np.clip(p, 1e-7, 1)) + (1 - trs.y[b]) * np.log(np.clip(1 - p, 1e-7, 1)))))
print(f"baseline on search folds: top1={np.mean(b_t1):.4f} ll={np.mean(b_ll):.4f} ({time.time()-t0:.0f}s)", flush=True)
study.enqueue_trial(dict(num_leaves=31, learning_rate=0.1, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3, min_split_gain=0.0, max_bin=255))
def cb(st, tr_):
    print(f" trial {tr_.number:2d} {tr_.state.name:8s} top1={tr_.value if tr_.value is not None else float('nan'):.4f} best={st.best_value:.4f} {tr_.params if tr_.state.name=='COMPLETE' else ''}", flush=True)
study.optimize(objective, n_trials=max(N_TRIALS - len(study.trials), 0), timeout=TIMEOUT, callbacks=[cb])
done = [t for t in study.trials if t.state.name == "COMPLETE"]
best = max(done, key=lambda t: (round(t.value, 4), -t.user_attrs["logloss"]))     # tie-break: log-loss
P = dict(best.params); n_est = int(max(100, best.user_attrs["best_iter"] * 1.15))
print("BEST", best.number, best.value, P, "n_estimators", n_est, flush=True)
pd.DataFrame([dict(trial=t.number, top1=t.value, logloss=t.user_attrs.get("logloss"), best_iter=t.user_attrs.get("best_iter"), state=t.state.name, **t.params) for t in study.trials]).to_csv(f"{RES}/tables/tuning_lgbm_trials.csv", index=False)

# ---------------- final: OOF (5 fold, all train) + full refit + student_test
FIN = {**P, "n_estimators": n_est}
folds5 = list(GroupKFold(n_splits=5).split(tr, groups=tr.trip)); oof = np.full(len(tr), np.nan); f_rows = []; sc_all = []
for k, (a, b) in enumerate(folds5):
    m = mk(**FIN).fit(tr.loc[a, FEATURES], tr.y[a]); p = m.predict_proba(tr.loc[b, FEATURES])[:, 1]; oof[b] = p
    held = tr_all[tr_all.trip.isin(set(tr.trip[b]))]          # score ALL fixes (incl. off-reference ones) of held-out trips -> HMM tuning input
    sc_all.append(held[["trip", "obs", "arc", "y", "obs_valid"]].assign(p=m.predict_proba(held[FEATURES])[:, 1].astype(np.float32)))
    f_rows.append(dict(fold=k, top1=top1(tr.loc[b], p), macro_f1=candidate_metrics(tr.y[b].values, p, .5)["macro_f1"]))
    print(" fold", f_rows[-1], flush=True)
np.save(f"{PROC}/oof_LightGBM_tuned.npy", oof); pd.concat(sc_all).to_parquet(f"{PROC}/oof_scores.parquet", index=False)
thr, _ = threshold_scan(tr.y.values, oof); cv_c = candidate_metrics(tr.y.values, oof, thr)
t0 = time.time(); final = mk(**FIN).fit(tr[FEATURES], tr.y); fit_s = time.time() - t0
t0 = time.time(); pte = final.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0
te_c = candidate_metrics(te.y.values, pte, thr)
base = mk(**BASE).fit(tr[FEATURES], tr.y); pb = base.predict_proba(te[FEATURES])[:, 1]
out = dict(search=dict(n_trials=len(study.trials), n_complete=len(done), subsample=SUB, folds=3, best_trial=best.number, best_search_top1=best.value, baseline_search_top1=float(np.mean(b_t1)), baseline_search_logloss=float(np.mean(b_ll))),
    params=FIN, threshold=thr, seed=SEED, features=FEATURES,
    cv5=dict(top1=top1(tr, oof), top1_fold_std=float(np.std([r["top1"] for r in f_rows], ddof=1)), macro_f1=cv_c["macro_f1"], precision=cv_c["precision"], recall=cv_c["recall"], log_loss=cv_c["log_loss"], folds=f_rows),
    student_test=dict(top1=top1(te, pte), macro_f1=te_c["macro_f1"], precision=te_c["precision"], recall=te_c["recall"], log_loss=te_c["log_loss"],
                      baseline_top1=top1(te, pb), baseline_log_loss=candidate_metrics(te.y.values, pb, .5)["log_loss"], baseline_macro_f1_at_0p5=candidate_metrics(te.y.values, pb, .5)["macro_f1"]),
    timing=dict(full_fit_s=fit_s, inference_s=inf_s))
json.dump(out, open(f"{RES}/tables/tuning_lgbm.json", "w"), indent=1)
joblib.dump(dict(model=final, features=FEATURES, threshold=thr, params=FIN), f"{ARTI}/lgbm_tuned.joblib")
print(json.dumps({k: out[k] for k in ("cv5", "student_test")}, indent=1)[:1500])
