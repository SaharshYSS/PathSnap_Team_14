"""Step 10 - tuning / deployment figures + summary table for the stage-2 report (actual results only)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _common import *
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
L = json.load(open(f"{RES}/tables/tuning_hgb.json")); T = pd.read_csv(f"{RES}/tables/tuning_hgb_trials.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
c = T[T.state == "COMPLETE"].reset_index(drop=True); ax[0].plot(c.trial, c.top1, "o-", color="#2166ac", label="trial Top-1"); ax[0].plot(c.trial, c.top1.cummax(), "-", color="#d73027", label="best so far")
ax[0].axhline(L["search"]["baseline_search_top1"], ls="--", color="grey", label="stage-1 config"); ax[0].set_xlabel("trial"); ax[0].set_ylabel("3-fold CV Top-1 (search subsample)"); ax[0].set_title("HistGradientBoosting tuning history"); ax[0].legend(fontsize=8)
s = L["student_test"]; names = ["Top-1", "Macro-F1"]; b = [s["baseline_top1"], s["baseline_macro_f1_at_0p5"]]; t = [s["top1"], s["macro_f1"]]; x = np.arange(2)
ax[1].bar(x - .18, b, .36, label="stage-1 HistGB", color="#999"); ax[1].bar(x + .18, t, .36, label="tuned HistGB", color="#2166ac"); ax[1].set_xticks(x); ax[1].set_xticklabels(names)
ax[1].set_ylim(min(b + t) - .02, 1); ax[1].set_title("student_test (candidate scorer)"); ax[1].legend(fontsize=8)
for i, (u, v) in enumerate(zip(b, t)): ax[1].text(i - .18, u, f"{u:.4f}", ha="center", va="bottom", fontsize=7); ax[1].text(i + .18, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(f"{RES}/figures/tune_hgb.png"); plt.close(fig)
rows = [dict(Stage="Stage-1 HistGB (argmax)", Top1=s["baseline_top1"], LogLoss=s["baseline_log_loss"]), dict(Stage="Tuned HistGB (argmax)", Top1=s["top1"], LogLoss=s["log_loss"])]
import os
f = f"{RES}/tables/pipeline_eval_student_test.json"
if os.path.exists(f):
    P = json.load(open(f)); rows += [dict(Stage="Tuned HistGB + Viterbi (end-to-end)", Top1=P["top1_viterbi"], Continuity=P["cont_viterbi"], RouteF1=P["route_f1"], ms_per_fix=P["ms_per_fix"])]
    rows[1].update(Continuity=P["cont_argmax"])
pd.DataFrame(rows).round(4).to_csv(f"{RES}/tables/stage2_summary.csv", index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))
h = f"{RES}/tables/tuning_hmm_trials.csv"
if os.path.exists(h):
    H = pd.read_csv(h); V = json.load(open(f"{RES}/tables/tuning_hmm.json"))["validation"]; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(H.trial, H.top1, "o-", color="#2166ac"); ax[0].plot(H.trial, H.top1.cummax(), color="#d73027"); ax[0].set_title("HMM tuning history (Top-1 after Viterbi)"); ax[0].set_xlabel("trial")
    k = ["argmax_top1", "viterbi_default_top1", "viterbi_tuned_top1"]; ax[1].bar(range(3), [V[i] for i in k], color=["#999", "#92c5de", "#2166ac"]); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["argmax", "Viterbi\n(default)", "Viterbi\n(tuned)"])
    ax[1].set_ylim(min(V[i] for i in k) - .02, 1); ax[1].set_title("held-out training trips (OOF scores)")
    for i, kk in enumerate(k): ax[1].text(i, V[kk], f"{V[kk]:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{RES}/figures/tune_hmm.png"); plt.close(fig)
