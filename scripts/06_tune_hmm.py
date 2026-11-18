"""Step 6 - tune the HMM/Viterbi parameters (emission weight, hop / U-turn / jump penalties, hop limit, gap handling).
Inputs: out-of-fold classifier scores of the tuned HistGradientBoosting (05_tune_hgb.py -> data/processed/oof_scores.parquet) so that no trip is
scored by a model that saw it, plus the raw road graphs (PATHSNAP_DATA) for the transition model.
  * tuning trips : N_TUNE training trips (first MAXFIX fixes each);   objective = observation-level Top-1 after Viterbi
  * validation   : other training trips (never used in the search) - reported next to the argmax baseline
Usage: python scripts/06_tune_hmm.py [n_trials=30] [n_tune_trips=16] [n_val_trips=16] [maxfix=1500]"""
import sys, json, time, warnings
import numpy as np, pandas as pd, optuna
from _common import *
from pathsnap.io import load_trip
from pathsnap.hmm import ArcGraph, HMMParams, viterbi
from pathsnap.metrics import build_next, _reach
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_TUNE = int(sys.argv[2]) if len(sys.argv) > 2 else 16
N_VAL = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MAXFIX = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
PART = os.environ.get("PATHSNAP_PART", "train")

S = pd.read_parquet(f"{PROC}/oof_scores.parquet") if os.path.exists(f"{PROC}/oof_scores.parquet") else pd.read_parquet(f"{PREP}/oof_scores.parquet")
trips = sorted(S.trip.unique()); rng = np.random.RandomState(SEED); rng.shuffle(trips)
tune_t, val_t = trips[:N_TUNE], trips[N_TUNE:N_TUNE + N_VAL]
print("tune trips", tune_t, "\nval trips", val_t, flush=True)

def prep(tid):
    d = S[(S.trip == tid) & (S.obs < MAXFIX)].sort_values(["obs", "arc"]); tp = load_trip(DATA_ROOT, PART, tid)
    lab = d[d.y == 1].groupby("obs").arc.first()                       # reference arc per fix (valid fixes only)
    valid = d.groupby("obs").obs_valid.first(); lab = lab[lab.index.isin(valid[valid].index)]
    return dict(d=d, t=tp.t, graph=ArcGraph(tp.arcs), next=build_next(tp.arcs), lab=lab, n=min(len(tp.t), MAXFIX))
DATA = {tid: prep(tid) for tid in tune_t + val_t}

def evaluate(tids, P):
    ok = tot = cok = ctot = 0; ok_am = 0
    for tid in tids:
        D = DATA[tid]; d = D["d"]; cand = dict(obs=d.obs.to_numpy(), arc=d.arc.to_numpy()); p = d.p.to_numpy()
        arcs, _ = viterbi(cand, p, D["t"][:D["n"]], D["graph"], P)
        lab = D["lab"]; ok += int((arcs[lab.index.to_numpy()] == lab.to_numpy()).sum()); tot += len(lab)
        if P is not None:   # continuity of decoded sequence on labelled adjacent fixes (dt<=5 s)
            o = lab.index.to_numpy(); a = arcs[o]; tt = D["t"][o]
            for i in np.nonzero(np.diff(o) == 1)[0]:
                if tt[i + 1] - tt[i] <= 5: ctot += 1; cok += _reach(D["next"], a[i], a[i + 1])
    return ok / tot, cok / max(ctot, 1)

def argmax_baseline(tids):
    ok = tot = cok = ctot = 0
    for tid in tids:
        D = DATA[tid]; d = D["d"]; top = d.loc[d.groupby("obs").p.idxmax()].set_index("obs").arc
        lab = D["lab"]; ok += int((top.reindex(lab.index).to_numpy() == lab.to_numpy()).sum()); tot += len(lab)
        o = lab.index.to_numpy(); a = top.reindex(o).to_numpy(); tt = D["t"][o]
        for i in np.nonzero(np.diff(o) == 1)[0]:
            if tt[i + 1] - tt[i] <= 5: ctot += 1; cok += _reach(D["next"], a[i], a[i + 1])
    return ok / tot, cok / max(ctot, 1)

def objective(trial):
    P = HMMParams(emis_w=trial.suggest_float("emis_w", 0.3, 3.0, log=True), hop_pen=trial.suggest_float("hop_pen", 0.05, 3.0, log=True),
                  uturn_pen=trial.suggest_float("uturn_pen", 0.5, 15.0, log=True), jump_pen=trial.suggest_float("jump_pen", 3.0, 40.0, log=True),
                  max_hops=trial.suggest_int("max_hops", 2, 6), gap_s=trial.suggest_categorical("gap_s", [5.0, 10.0, 20.0]),
                  gap_scale=trial.suggest_float("gap_scale", 0.05, 1.0), p_floor=trial.suggest_categorical("p_floor", [1e-6, 1e-4, 1e-3]))
    acc, cont = evaluate(tune_t, P); trial.set_user_attr("continuity", cont); return acc

b_acc, b_cont = argmax_baseline(tune_t); print(f"argmax baseline (tune trips): top1={b_acc:.4f} cont={b_cont:.4f}", flush=True)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8), study_name="hmm",
                            storage=f"sqlite:///{RES}/tables/optuna_hmm.db", load_if_exists=True)
study.enqueue_trial(HMMParams().to_dict() | {"p_floor": 1e-4})
t0 = time.time()
study.optimize(objective, n_trials=max(N_TRIALS - len(study.trials), 0),
               callbacks=[lambda st, t: print(f" trial {t.number:2d} top1={t.value:.4f} cont={t.user_attrs['continuity']:.4f} best={st.best_value:.4f} ({time.time()-t0:.0f}s)", flush=True)])
best = study.best_trial; P = HMMParams(**{k: v for k, v in best.params.items()})
v_acc, v_cont = evaluate(val_t, P); vb_acc, vb_cont = argmax_baseline(val_t); d_acc, d_cont = evaluate(val_t, HMMParams())
out = dict(params=P.to_dict(), seed=SEED, n_trials=len(study.trials), tune_trips=tune_t, val_trips=val_t, maxfix=MAXFIX,
           tune=dict(argmax_top1=b_acc, argmax_cont=b_cont, viterbi_top1=best.value, viterbi_cont=best.user_attrs["continuity"]),
           validation=dict(argmax_top1=vb_acc, argmax_cont=vb_cont, viterbi_default_top1=d_acc, viterbi_default_cont=d_cont, viterbi_tuned_top1=v_acc, viterbi_tuned_cont=v_cont))
json.dump(out, open(f"{RES}/tables/tuning_hmm.json", "w"), indent=1); json.dump(out, open(f"{ARTI}/hmm_params.json", "w"), indent=1)
pd.DataFrame([dict(trial=t.number, top1=t.value, continuity=t.user_attrs.get("continuity"), **t.params) for t in study.trials]).to_csv(f"{RES}/tables/tuning_hmm_trials.csv", index=False)
print(json.dumps(out["validation"], indent=1), P)
