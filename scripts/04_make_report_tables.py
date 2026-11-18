"""Step 4 - consolidate experiment results into the comparison table + figures (actual results only)."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open(f"{RES}/tables/_model_results.json"))
rows = []
for n, r in R.items():
    rows.append({"Model": n, "Algorithm family": r["family"], "Key configuration": json.dumps(r["config"]),
      "CV Macro F1 (cand.)": r["cv"]["macro_f1"], "CV Macro F1 fold std": r["cv"]["macro_f1_fold_std"],
      "CV Top-1": r["cv"]["top1"], "CV Top-1 fold std": r["cv"]["top1_fold_std"], "CV Log loss": r["cv"]["log_loss"],
      "CV Route continuity": r["cv"]["continuity"], "CV Precision": r["cv"]["precision"], "CV Recall": r["cv"]["recall"],
      "Threshold (OOF-tuned)": r["threshold"],
      "Test Top-1": r["student_test"]["top1"], "Test Macro F1 (cand.)": r["student_test"]["macro_f1"], "Test Macro F1 @0.5": r["student_test"]["macro_f1_at_0p5"],
      "Test Precision": r["student_test"]["precision"], "Test Recall": r["student_test"]["recall"],
      "Test Log loss": r["student_test"]["log_loss"], "Test Route continuity": r["student_test"]["continuity"],
      "Test Route membership": r["student_test"]["route_membership"], "Train Top-1": r["train_fit"]["top1"],
      "Inference s (test, 255k cand.)": r["timing"]["inference_s_student_test"], "Inference ms/fix": r["timing"]["inference_ms_per_fix"],
      "Full-fit time s": r["timing"]["full_fit_s"]})
T = pd.DataFrame(rows).sort_values("CV Top-1", ascending=False)
T.to_csv(f"{RES}/tables/model_comparison.csv", index=False)
short = T[["Model", "Algorithm family", "CV Macro F1 (cand.)", "CV Top-1", "CV Route continuity", "Test Macro F1 (cand.)", "Test Top-1", "Test Log loss", "Inference ms/fix"]].round(4)
open(f"{RES}/tables/model_comparison.md", "w").write(short.to_markdown(index=False))
print(short.to_string(index=False))
names = list(T.Model); x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(x, T["CV Macro F1 (cand.)"], yerr=T["CV Macro F1 fold std"], capsize=3, color="#2166ac")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_ylim(T["CV Macro F1 (cand.)"].min() - .03, 1); ax.set_title("5-fold GroupKFold CV Macro-F1 (candidate level, partition: train; bars = OOF F1, whiskers = fold-std of per-fold F1@0.5)"); ax.title.set_fontsize(8)
fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_cv_macro_f1.png"); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); w = .38
ax[0].bar(x - w/2, T["CV Top-1"], w, label="CV (train)"); ax[0].bar(x + w/2, T["Test Top-1"], w, label="student_test"); ax[0].set_ylim(T[["CV Top-1", "Test Top-1"]].min().min() - .03, 1)
ax[0].set_title("Observation-level Top-1 accuracy"); ax[0].legend()
ax[1].bar(x - w/2, T["CV Macro F1 (cand.)"], w, label="CV"); ax[1].bar(x + w/2, T["Test Macro F1 (cand.)"], w, label="student_test"); ax[1].set_ylim(T[["CV Macro F1 (cand.)", "Test Macro F1 (cand.)"]].min().min() - .03, 1)
ax[1].set_title("Candidate-level Macro-F1 (OOF-tuned threshold)"); ax[1].legend()
for a in ax: a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha="right")
fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_test_top1_f1.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(x - w/2, T["CV Log loss"], w, label="CV"); ax.bar(x + w/2, T["Test Log loss"], w, label="student_test"); ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_title("Log loss (lower is better; log scale)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_logloss.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(x - w/3*1.1, T["Train Top-1"], w/1.5, label="train (refit, 10% obs sample)"); ax.bar(x + w/3*1.1, T["CV Top-1"], w/1.5, label="CV (held-out trips)")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_ylim(T[["Train Top-1", "CV Top-1"]].min().min() - .03, 1); ax.set_title("Over-/under-fitting check: train vs held-out Top-1"); ax.legend(); fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_overfit_gap.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 5))
for _, r in T.iterrows(): ax.scatter(r["Inference ms/fix"], r["CV Top-1"], s=60); ax.annotate(r["Model"], (r["Inference ms/fix"], r["CV Top-1"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("inference ms per GPS fix (student_test)"); ax.set_ylabel("CV Top-1"); ax.set_title("Accuracy vs inference cost"); fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_accuracy_vs_time.png"); plt.close(fig)
n = len(names); cols = min(5, n); rws = int(np.ceil(n / cols))
fig, ax = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.1 * rws), squeeze=False)
for a in ax.ravel(): a.axis("off")
for a, nm in zip(ax.ravel(), names):
    cm = np.array(R[nm]["student_test"]["cm"]); a.axis("on"); a.imshow(cm, cmap="Blues"); a.grid(False)
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8, color="black" if cm[i, j] < cm.max() / 2 else "white")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"], fontsize=7); a.set_yticklabels(["true 0", "true 1"], fontsize=7); a.set_title(nm, fontsize=9)
fig.suptitle("Candidate-level confusion matrices (partition: student_test, OOF-tuned threshold)", fontsize=10); fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_confusion_matrices.png"); plt.close(fig)
hs = {k: v["history"] for k, v in R.items() if v.get("history")}
if hs:
    fig, ax = plt.subplots(1, len(hs), figsize=(5 * len(hs), 4), squeeze=False)
    for a, (k, h) in zip(ax[0], hs.items()):
        a.plot(h["train"], label="train"); 
        if h["val"]: a.plot(h["val"], label="validation (CV fold 0 held-out trips)")
        a.set_title(f"{k}: {h['metric']} per epoch/iteration"); a.set_xlabel("iteration"); a.legend()
    fig.tight_layout(); fig.savefig(f"{RES}/figures/cmp_training_curves.png"); plt.close(fig)
print("tables+figures written")
