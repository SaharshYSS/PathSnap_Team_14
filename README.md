# PathSnap: GPS Map-Matching & Route Reconstruction (Team 14)

**Course:** UE24CS352A Machine Learning, PES University (RR Campus)
**Stage 1:** dataset exploration, preprocessing, candidate generation, feature engineering, training/comparison of six models.
**Stage 2:** Bayesian hyperparameter optimisation, HMM + Viterbi route reconstruction, FastAPI backend (Laptop 1), evaluator UI (Laptop 2).

---

## Table of contents
1. [The problem](#1-the-problem)
2. [Pipeline overview](#2-pipeline-overview)
3. [Repository structure](#3-repository-structure)
4. [How to run](#4-how-to-run)
5. [The dataset](#5-the-dataset)
6. [Exploratory data analysis: what we found](#6-exploratory-data-analysis-what-we-found)
7. [Preprocessing](#7-preprocessing)
8. [Creating labels (reference alignment)](#8-creating-labels-reference-alignment)
9. [Candidate generation](#9-candidate-generation)
10. [Feature engineering (31 features)](#10-feature-engineering-31-features)
11. [Training and evaluation protocol](#11-training-and-evaluation-protocol)
12. [Metrics](#12-metrics)
13. [Models trained](#13-models-trained)
14. [Results](#14-results)
15. [Model selection: how we chose](#15-model-selection-how-we-chose)
16. [Data leakage safeguards](#16-data-leakage-safeguards)
17. [Bayesian hyperparameter optimisation](#17-bayesian-hyperparameter-optimisation)
18. [HMM + Viterbi route reconstruction](#18-hmm--viterbi-route-reconstruction)
19. [API, UI and two-laptop deployment](#19-api-ui-and-two-laptop-deployment)
20. [Limitations](#20-limitations)
21. [Team](#21-team)
22. [AI assistance disclosure](#22-ai-assistance-disclosure)

---

## 1. The problem

GPS readings are noisy. Because of tall buildings, signal reflections, dropouts and low sampling rates, a car driving down one road can appear to be on a parallel street or inside a building.

**Map matching** is the task of working out *which actual road segment the vehicle was on* for every GPS reading. PathSnap does this with a machine-learning model that scores nearby road segments. In the next stage, an HMM/Viterbi decoder will turn those scores into a continuous, drivable route.

## 2. Pipeline overview

```
Raw trip files (.track / .nodes / .arcs / .route)
   │
   ▼  io.py          read files → Trip object
   ▼  geo.py         project lon/lat (degrees) → UTM x/y (metres)
   ▼  candidates.py  for every GPS fix: all directed road arcs within 60 m (max 40)
   ▼  candidates.py  31 engineered features per (fix, candidate arc) pair
   ▼  labels.py      label = 1 if the candidate is the fix's reference arc (training/evaluation only)
   ▼  03_train_compare.py   6 classifiers score P(arc is correct | features)
   ▼  05_tune_hgb.py        Bayesian (Optuna TPE) tuning of the selected model, HistGradientBoosting
   ▼  hmm.py                HMM + Viterbi over candidate arcs → one arc per fix → connected route
   ▼  app/server.py         FastAPI on Laptop 1: trip id in → prediction + server-side scoring out
   ▼  ui/index.html         evaluator interface on Laptop 2: pick a trip → route map + metrics
```

## 3. Repository structure

```
PathSnap_Team_14/
├── README.md
├── requirements.txt
├── .gitignore
├── src/pathsnap/
│   ├── __init__.py
│   ├── io.py            # reads .track/.nodes/.arcs/.route into a Trip object
│   ├── geo.py           # UTM projection + point-to-segment geometry (seg_proj)
│   ├── labels.py        # monotone alignment of GPS fixes to the reference route (labels)
│   ├── candidates.py    # candidate-arc generation + 31 features
│   ├── metrics.py       # Top-1, macro F1, log loss, threshold scan, route continuity, route P/R/F1
│   ├── viz.py           # Folium map builder (GPS, network, reference, predicted, candidates)
│   ├── hmm.py           # ArcGraph, HMM transition model, Viterbi decoder, route builder
│   └── matcher.py       # end-to-end inference: raw trip → candidates → classifier → Viterbi
├── app/server.py        # FastAPI backend (Laptop 1)
├── ui/index.html        # evaluator interface (Laptop 2) - open in any browser
├── run_server.bat       # Windows one-click launcher for the API
├── artifacts/           # hgb_tuned.joblib (final model), lgbm_tuned.joblib, hmm_params.json
├── scripts/
│   ├── _common.py               # paths, output folders, SEED = 42
│   ├── 01_build_dataset.py      # preprocessing → data/processed/*.parquet
│   ├── 02_eda.py                # EDA tables, figures, Folium maps
│   ├── 03_train_compare.py      # trains + evaluates 6 models
│   ├── 04_make_report_tables.py # comparison table + comparison plots
│   ├── 05_tune_hgb.py           # Bayesian tuning of HistGradientBoosting (+ 05_tune_lgbm.py, extra)
│   ├── 06_tune_hmm.py           # tunes HMM penalties on out-of-fold scores (train trips only)
│   ├── 08_eval_pipeline.py      # end-to-end evaluation on student_test: argmax vs Viterbi
│   ├── 09_demo_client.py        # development-only raw-data client for POST /match
│   ├── 10_make_stage2_report.py # tuning / HMM figures + stage2_summary.csv
│   └── make_synthetic.py        # synthetic smoke-test data (not used for any result)
├── results/
│   ├── eda_data_quality.md      # data-quality record
│   ├── tables/                  # CSV/JSON results (see §14)
│   ├── figures/                 # eda_*.png, cmp_*.png
│   └── logs/                    # console logs of every run
└── maps/                        # interactive Folium HTML maps
```

**Not committed** (see `.gitignore`): the raw dataset, the evaluator data, and the processed parquet files (regenerated by `01_build_dataset.py` and `05_tune_hgb.py`). The final model weights in `artifacts/` **are** committed (2.8 MB).

## 4. How to run

**Requirements:** Python 3.10+.

```bash
pip install -r requirements.txt
```

Point the code at the dataset folder (the one containing `train/` and `student_test/`):

```bash
# Linux / macOS / Git Bash
export PATHSNAP_DATA="/path/to/Student folder"
```
```cmd
:: Windows CMD
set PATHSNAP_DATA=C:\path\to\Student folder
```

Then run the four steps in order:

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python scripts/01_build_dataset.py` | projection, candidates, features, labels | `data/processed/{train,student_test}/*.parquet`, `obs_all.parquet` |
| 2 | `python scripts/02_eda.py` | dataset inventory, plots, maps | `results/tables/`, `results/figures/eda_*.png`, `maps/*.html` |
| 3 | `python scripts/03_train_compare.py` | 5-fold GroupKFold CV + student-test evaluation of 6 models | `results/tables/_model_results.json` |
| 4 | `python scripts/04_make_report_tables.py` | comparison table + plots | `results/tables/model_comparison.*`, `results/figures/cmp_*.png` |

You can train a subset of models, e.g. `python scripts/03_train_compare.py LogReg MLP`.
All randomness uses `SEED = 42` (`scripts/_common.py`).

**Stage 2:**

| Step | Command | Output |
|---|---|---|
| 5 | `python scripts/05_tune_hgb.py 20 30` | `artifacts/hgb_tuned.joblib`, `results/tables/tuning_hgb*.{json,csv}`, `data/processed/oof_scores.parquet` |
| 6 | `python scripts/06_tune_hmm.py 30 16 16` | `artifacts/hmm_params.json`, `results/tables/tuning_hmm*.{json,csv}` |
| 8 | `python scripts/08_eval_pipeline.py` | `results/tables/pipeline_eval_student_test.{csv,json}` |
| 10 | `python scripts/10_make_stage2_report.py` | `results/figures/tune_hgb.png`, `tune_hmm.png`, `results/tables/stage2_summary.csv` |

`scikit-learn` is pinned to **1.9.1** because `hgb_tuned.joblib` was saved with that version; loading it with another version can fail.

**Deployment:** see [section 19](#19-api-ui-and-two-laptop-deployment).

---

## 5. The dataset

### 5.1 Partitions

| Partition | Trips | GPS fixes | Use |
|---|---|---|---|
| `train` | 80 | 208,417 | model fitting, cross-validation, hyperparameter search |
| `student_test` | 10 | 19,160 | development-stage comparison of models |
| `evaluator_test` | 10 | n/a | **not provided to us**, used only by evaluators |

Each trip is a folder named by an 8-digit ID (e.g. `00000048`). A trip never spans two partitions.

### 5.2 Raw files (no headers, whitespace-separated)

| File | Columns | Meaning |
|---|---|---|
| `.track` | `lon  lat  t` | GPS fixes. `t` is in seconds, nominally 1 Hz |
| `.nodes` | `lon  lat` | Road-network junction points. **Row index (0-based) = node ID** |
| `.arcs` | `from_node  to_node` | **Directed** road segments. **Row index = arc ID** |
| `.route` | `arc_id` | Reference route: the ordered list of arc IDs the vehicle drove |

**How the files connect:**

```
.route → arc_id → row of .arcs → (from_node, to_node) → rows of .nodes → coordinates
.track → GPS fixes: NOT linked row-by-row to any other file
```

There is **no per-GPS-fix label** in the data. `.track` row *i* does not correspond to `.route` row *i* (e.g. trip 00000048 has 1,240 fixes but 149 route arcs). Section 8 explains how we created per-fix labels.

Each trip ships **its own local road network**, so node and arc IDs are only meaningful within that trip.

### 5.3 Location
Coordinates are around lon ≈ −1.8, lat ≈ 53 (central England). This determines the projection we use (§7).

---

## 6. Exploratory data analysis: what we found

Generated by `scripts/02_eda.py`. Full per-trip table: `results/tables/dataset_inventory.csv`. Summary: `results/eda_data_quality.md`.

**Data completeness**
- **0 missing values** in GPS or network files. **0 out-of-range** coordinates.
- Timestamps are **strictly increasing** in every trip.

**Sampling and gaps**
- Most trips are close to 1 Hz (median share of 1-second steps: 0.99 in train).
- **42 trips have gaps longer than 10 s.** The largest is **153,064 s (about 42 hours) in trip 00000090**, so that trip contains a long recording break.
- Some trips sample much more sparsely (e.g. 00000022: only 53% of steps are 1 s).

**Trip size varies widely**
- GPS fixes per trip: **262 – 19,313** (median about 1,500).
- Road-network size per trip: **299 – 155,003 nodes** and **649 – 307,392 arcs**. Large networks are why we crop each network to the trip's bounding box and use a KD-tree.

**Road network**
- Graphs are **directed**. About **92% of arcs have a reverse twin** (two-way streets).
- **24 self-loops**, **1,903 duplicate arcs**, **0 isolated nodes**, **0 arcs pointing to non-existent nodes**.
- Median of 19 weakly-connected components per network. In most trips the largest component holds more than 90% of nodes.

**Reference routes are not perfect ground truth**
- Consecutive route arcs are connected in **98.1–100%** of transitions. A few route steps "jump."
- **5 training trips have incomplete reference routes**: the GPS trace goes far beyond the route.

| Trip | GPS path (km) | Route length (km) | Labelable fixes |
|---|---|---|---|
| 00000054 | 169.6 | 31.4 | **10.7%** |
| 00000033 | 47.9 | 6.4 | 41.0% |
| 00000002 | 55.9 | 8.0 | 88.0% |
| 00000066 | 83.3 | 73.0 | 92.0% |
| 00000043 | 63.8 | 59.3 | 92.8% |

**Figures:** `eda_overview.png` (trip sizes, sampling, network sizes), `eda_trajectories.png` (GPS traces), `eda_candidates.png`, `eda_feature_distributions.png`, `eda_feature_correlation.png`.
**Interactive maps:** `maps/eda_train_00000048.html`, `maps/eda_student_test_00000077.html`, `maps/eda_candidates_train_00000048.html`. Each has layers for raw GPS (blue), road network (grey), reference route (green) and candidate arcs of one fix (orange), plus a legend and layer control.

---

## 7. Preprocessing

**Why projection is needed.** Longitude and latitude are *angles*. At this latitude, 1° of longitude ≈ 67 km but 1° of latitude ≈ 111 km, so distance computed in degrees is distorted.

**What we do (`geo.py`).** Each trip is projected into its local **UTM zone** (computed from the trip's mean longitude, which gives zone 30N, **EPSG:32630**, for this data) with `pyproj`. All distances and angles afterwards are in **metres**.

**Core geometry (`seg_proj`).** For a GPS point P and a road segment A→B we compute:
- `d`: perpendicular distance from P to the segment (clamped to the segment's ends)
- `s`: how far along A→B the projection of P falls
- `L`: segment length
- `lat`: signed lateral offset (left of travel direction = positive)

**Coordinate order.** Files store `lon, lat`. Folium expects `[lat, lon]`. `viz.py` swaps them explicitly.

---

## 8. Creating labels (reference alignment)

**Problem.** The data gives only the route (an ordered list of arcs), not which arc each GPS fix belongs to.

**Our approach: monotone alignment by dynamic programming (`labels.py`).**
- Each fix *t* is assigned a position *j<sub>t</sub>* in the reference route.
- **Constraint:** positions never go backwards (*j<sub>t</sub> ≤ j<sub>t+1</sub>*), because a vehicle traverses its route in order.
- **Objective:** minimise the total point-to-segment distance between fixes and their assigned arcs, with each distance capped at 60 m so one outlier cannot distort the whole alignment.
- Solved exactly with a DP (prefix-minimum trick, O(T × J)), compiled with `numba`.

**Why monotone instead of "nearest route arc"?** When a route passes the same area twice (e.g. out-and-back trips), the nearest arc can belong to the wrong part of the journey. The ordering constraint prevents that.

**Unlabelable fixes.** If a fix is still **more than 50 m** from its aligned arc, the reference route does not cover it (incomplete reference). Such fixes get `obs_valid = False` and are **excluded from training and evaluation**.
- Labelable share: **train 88.9%**, **student_test 99.86%**.

These labels are used **only** for training and scoring. The prediction path (candidates → features → model) never reads the route.

---

## 9. Candidate generation

For each GPS fix, `candidates.py`:
1. Crops the trip's network to the GPS bounding box plus a margin.
2. Samples every arc every 15 m and builds a **KD-tree** over the samples for fast neighbour search.
3. Collects every **directed** arc within **60 m** (exact point-to-segment distance) and keeps the **40 nearest**.

**Why 60 m?** `results/tables/candidate_radius_sensitivity.csv` (labelable train fixes):

| Radius (m) | 10 | 20 | 30 | 40 | 45 | **50+** |
|---|---|---|---|---|---|---|
| Candidate miss rate | 18.7% | 3.6% | 1.4% | 0.49% | 0.24% | **0.0%** |

At 50 m the true arc is always among the candidates. 60 m adds a safety margin. A larger radius only adds more wrong candidates.

**Resulting dataset:** 2,150,636 labelable training rows and 255,743 student-test rows. Only **8.6% are positives** (class imbalance), which is why we use Macro F1 and log loss instead of plain accuracy.

---

## 10. Feature engineering (31 features)

Every (fix, candidate arc) pair is described by 31 numbers. The table shows the mean value for wrong arcs (y=0) vs correct arcs (y=1) on training data (`results/tables/feature_means_by_class.csv`). A large difference means the feature separates the classes well.

| Group | Features | What it captures | Example: y=0 → y=1 |
|---|---|---|---|
| **Geometry** | `d`, `lat_off`, `frac`, `proj_clamped`, `arc_len`, `dist_to_end`, `dist_from_start` | How close the fix is to the arc and whether it projects *onto* the segment or beyond its ends | `d`: 32.1 m → **6.5 m**; `proj_clamped`: 0.78 → **0.03** |
| **Direction & motion** | `speed`, `hdg_valid`, `cos_dh`, `abs_dh_deg`, `motion_along` | Whether the vehicle's heading (from fixes ±2 around it) agrees with the arc's direction. This separates the two directions of a two-way street | `abs_dh_deg`: 94° → **24°**; `motion_along`: −0.8 → **+11.2 m/s** |
| **Relative to other candidates** | `n_cand`, `rank_d`, `d_minus_dmin`, `d_ratio`, `rank_d_dir`, `twin_exists` | Whether this arc is the closest (direction-compatible) option for the fix | `rank_d`: 10.7 → **0.7**; `d_minus_dmin`: 26.9 → **0.8 m** |
| **Network topology** | `deg_from_out`, `deg_to_out`, `deg_from_in`, `deg_to_in` | Number of roads meeting at each end of the arc | ≈2.1 → ≈2.0 (weakest group) |
| **Sequence context** | `d_mean_w3`, `d_max_w6`, `d_mean_w6`, `frac_near_w6`, `prog_w3`, `prog_w6`, `prog_ratio_w3`, `dt_prev`, `dt_next` | Whether neighbouring fixes (±1, ±3, ±6) also lie on this arc and move *along* it | `frac_near_w6`: 0.33 → **0.82**; `prog_w3`: −5 m → **+70 m** |

**Why these features?** Distance alone cannot separate parallel roads or the two directions of the same street, since both are equally close. Heading agreement and neighbour consistency resolve those ties.

**Missing values.** Only `prog_w3`, `prog_ratio_w3` (2.3%) and `prog_w6` (3.2%) have NaNs, at trip edges and across time gaps where neighbours do not exist (`feature_missing_values.csv`). Tree/boosting models handle NaN natively. LogReg, GaussianNB and MLP use **median imputation + standard scaling inside an sklearn Pipeline**, so the imputer and scaler are fitted on training folds only.

**Note:** feature generation never uses the reference route, so the same code runs unchanged at inference time.

---

## 11. Training and evaluation protocol

| Item | Setting |
|---|---|
| Task | Binary classification of (fix, candidate arc) pairs → P(correct arc) |
| Training data | 80 train trips, labelable fixes only |
| Cross-validation | **5-fold GroupKFold, groups = trip ID**. All fixes of a trip stay in one fold |
| Decision threshold (for F1/precision/recall) | Scanned 0.05–0.95 on **out-of-fold** predictions only |
| Final fit | Refit on all 80 train trips, then evaluated once on the 10 student-test trips |
| Seed | 42 |
| Selection rule (declared before training) | CV observation-level Top-1; tie-breakers: CV log loss → route continuity → inference time |

**Why GroupKFold?** Consecutive GPS fixes from the same trip are almost identical. A random split would put fix 100 in training and fix 101 in validation, which inflates scores (leakage). Grouping by trip measures performance on **unseen trips**, which is the real task.

---

## 12. Metrics

| Metric | Level | Meaning |
|---|---|---|
| **Top-1 accuracy** (primary) | per GPS fix | Share of fixes whose highest-scoring candidate is the reference arc |
| Macro F1 | per candidate row | Mean of F1 for class 0 and class 1, at the OOF-tuned threshold. Fair under the 8.6% imbalance |
| Precision / Recall | per candidate row | For the positive class |
| Log loss | per candidate row | Penalises confident mistakes. Important because the HMM will use these probabilities as emission scores |
| Route continuity | per consecutive fix pair | Share of consecutive fixes (≤5 s apart) whose predicted arcs are the same or reachable within 3 directed arc hops |
| Route membership | per fix | Share of predicted arcs that belong to the trip's reference route |
| Candidate miss rate | per fix | Share of fixes whose true arc is not among the candidates (0% at 60 m) |
| Training / inference time | n/a | Full-fit seconds; inference milliseconds per GPS fix |

**Why Top-1 is primary:** the final system must output *one road per GPS fix*, so "is the top-ranked candidate correct?" matches the task directly. Candidate-level F1 is secondary because it depends on an arbitrary threshold.

---

## 13. Models trained

Six models covering four of the algorithm families listed in the project guidelines (linear/probabilistic, tree-based, boosting, neural networks), all trained with the same features, folds and seed. Two boosting implementations (scikit-learn HistGradientBoosting and LightGBM) were deliberately compared to check that boosting's performance was not specific to one library. They agreed within 0.001 CV Top-1. Configurations are recorded in `results/tables/_model_results.json`.

| Model | Family | Configuration | Preprocessing |
|---|---|---|---|
| Logistic Regression | Linear | C=1.0, L2, lbfgs, max_iter=300 | median impute + scale |
| Gaussian Naive Bayes | Probabilistic | var_smoothing=1e-9 | median impute + scale |
| Decision Tree | Tree | max_depth=12, min_samples_leaf=20 | median impute |
| HistGradientBoosting | Boosting (sklearn) | max_iter=300, learning_rate=0.1, max_leaf_nodes=31 | none (native NaN) |
| LightGBM | Boosting | n_estimators=400, num_leaves=31, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8 | none (native NaN) |
| MLP | Neural network | hidden=(64, 32), ReLU, alpha=1e-4, lr=1e-3, 30 epochs; trained on a random 30% of training fixes for compute reasons | median impute + scale |

---

## 14. Results

Source: `results/tables/model_comparison.csv` (all numbers produced by `03_train_compare.py`; console output in `results/logs/03_train_*.log`).

### 14.1 Cross-validation (80 train trips, 5-fold GroupKFold)

| Model | CV Top-1 | Top-1 fold std | CV Macro F1 | CV Log loss | CV Continuity | Train Top-1 | Train − CV gap |
|---|---|---|---|---|---|---|---|
| MLP | **0.8970** | 0.0246 | 0.9358 | 0.0608 | **0.9785** | 0.9304 | 0.033 |
| HistGradBoost | 0.8947 | 0.0247 | **0.9425** | **0.0518** | 0.9715 | 0.9261 | 0.031 |
| LightGBM | 0.8937 | 0.0256 | 0.9422 | 0.0599 | 0.9705 | 0.9286 | 0.035 |
| LogReg | 0.8927 | 0.0276 | 0.9314 | 0.0623 | 0.9751 | 0.8966 | 0.004 |
| DecisionTree | 0.8891 | 0.0287 | 0.9362 | 0.0654 | 0.9686 | 0.9274 | 0.038 |
| GaussianNB | 0.8618 | 0.0336 | 0.8964 | 0.3954 | 0.9650 | 0.8626 | 0.001 |

### 14.2 Student-test (10 trips, model refit on all 80 train trips)

| Model | Test Top-1 | Test Macro F1 | Precision | Recall | Log loss | Continuity | Inference (ms/fix) | Full-fit time (s) |
|---|---|---|---|---|---|---|---|---|
| LightGBM | **0.9002** | **0.9448** | 0.8962 | 0.8994 | 0.0447 | 0.9726 | 0.077 | 44.5 |
| HistGradBoost | 0.8991 | 0.9436 | 0.8950 | 0.8965 | **0.0401** | 0.9728 | **0.020** | 18.6 |
| MLP | 0.8983 | 0.9340 | 0.8486 | 0.9100 | 0.0459 | **0.9784** | 0.029 | 63.0 |
| DecisionTree | 0.8911 | 0.9357 | 0.9076 | 0.8553 | 0.0528 | 0.9710 | 0.006 | 85.3 |
| LogReg | 0.8896 | 0.9297 | 0.8593 | 0.8810 | 0.0513 | 0.9764 | 0.005 | 19.8 |
| GaussianNB | 0.8407 | 0.8861 | 0.7037 | 0.9046 | 0.4119 | 0.9718 | 0.010 | 10.2 |

Student-test confusion matrix, HistGradientBoosting (candidate level): TN = 234,597, FP = 2,013, FN = 1,981, TP = 17,152.

### 14.3 Figures
| File | Shows |
|---|---|
| `cmp_cv_macro_f1.png` | CV Macro F1 for all models |
| `cmp_test_top1_f1.png` | Student-test Top-1 and Macro F1 |
| `cmp_logloss.png` | Log loss comparison |
| `cmp_overfit_gap.png` | Train vs CV Top-1 (over/underfitting) |
| `cmp_accuracy_vs_time.png` | Accuracy vs inference time trade-off |
| `cmp_confusion_matrices.png` | Student-test confusion matrices |
| `cmp_training_curves.png` | Epoch-wise curves **only** for models that produce them: LightGBM (train vs validation log loss) and MLP (training loss only; early stopping was off, so no validation curve exists). No curves are shown for models without training histories |

### 14.4 What each model's results tell us

- **Logistic Regression.** Smallest train–CV gap (0.004), so **no overfitting**. But a linear boundary cannot learn interactions such as "small distance matters only if the heading also agrees," so it plateaus below the non-linear models (**mild underfitting**). It's the fastest model (0.005 ms/fix).
- **Gaussian Naive Bayes.** Clearly the weakest. NB assumes features are independent, but many of ours are strongly correlated (`d`, `d_mean_w3`, `d_mean_w6`, `rank_d` all measure closeness). It counts the same evidence several times and becomes **overconfident**: log loss is about 8× worse than the other models (0.40), and the OOF-optimal threshold was pushed to 0.95. **Underfits.**
- **Decision Tree.** **Largest train–CV gap (0.038), so it overfits** by memorising training trips. Leaf-based probabilities are coarse, which gives worse log loss than boosting.
- **HistGradientBoosting.** Best CV Macro F1 and best CV and test log loss (best-calibrated probabilities). Fastest of the strong models. Handles NaN natively. Train–CV gap 0.031 (moderate, normal for boosting).
- **LightGBM.** Best student-test Top-1 and Macro F1, but nearly identical to HistGradBoost in CV and about 4× slower at inference.
- **MLP.** Best CV Top-1 and best route continuity. Needs imputation and scaling, and was trained on 30% of fixes. Higher recall but lower precision than boosting.

**Fold pattern.** Every model scores about 0.87 on folds 0 and 4 and about 0.92 on folds 1 and 3. The difficulty of the trips in a fold (dense parallel roads, gaps) affects scores more than the choice of model. This is also why the fold standard deviation is around 0.025.

---

## 15. Model selection: how we chose

**Step 1. Apply the declared primary metric (CV Top-1).**
MLP 0.8970 > HistGradBoost 0.8947 > LightGBM 0.8937 > LogReg 0.8927 > DecisionTree 0.8891 > GaussianNB 0.8618.

**Step 2. Check whether the top differences are real.**
The top three differ by at most **0.0033** in CV Top-1, while the fold-to-fold standard deviation is **about 0.025**, roughly 8× larger. Fold by fold, MLP beats HistGradBoost in 3 folds and loses in 2, with per-fold differences of at most 0.006. **MLP, HistGradBoost and LightGBM are statistically tied on Top-1.**

**Step 3. Break the tie with the declared tie-breakers.**
- **CV log loss** (first tie-breaker): **HistGradBoost 0.0518** < LightGBM 0.0599 < MLP 0.0608. This matters most for the next stage, because the HMM uses classifier probabilities as emission probabilities. Better-calibrated probabilities mean better decoding.
- **Route continuity:** MLP is higher (0.9785 vs 0.9715). We accept this because the HMM/Viterbi stage is designed specifically to enforce connectivity.
- **Inference time:** HistGradBoost 0.020 ms/fix vs LightGBM 0.077 and MLP 0.029.

**Step 4. Sanity-check on student-test (not used for the decision).**
HistGradBoost 0.8991 Top-1 with the lowest test log loss (0.0401). This is consistent with CV, so there is no sign that the choice was a fluke of cross-validation.

**Decision: HistGradientBoostingClassifier** is carried forward to Bayesian hyperparameter optimisation and the HMM stage.
LightGBM and MLP remain close alternatives, and we will report them alongside the tuned model.

GaussianNB, Decision Tree and Logistic Regression are ruled out: GaussianNB because of poor calibration and accuracy, Decision Tree because of overfitting and coarse probabilities, and Logistic Regression because of its lower Top-1 ceiling.

---

## 16. Data leakage safeguards

- Evaluator-test trips are used **only** by the API at demonstration time (`/predict`). They were never used for training, tuning, threshold choice or model selection, and are not committed to the repository.
- Student-test trips are used **only** for comparison after training, never for fitting, threshold choice or model choice.
- Cross-validation is **grouped by trip**, so no trip appears in both training and validation.
- Imputers and scalers sit **inside** sklearn Pipelines, so they are fitted on training folds only.
- The F1 threshold is tuned on **out-of-fold** predictions only.
- Candidate generation and features **never read the reference route**. `load_trip(..., with_route=False)` loads a trip without it.
- **No hard-coded predictions** anywhere in the code.

---

## 17. Bayesian hyperparameter optimisation

Script: `scripts/05_tune_hgb.py`. Record: `results/tables/tuning_hgb_trials.csv` (every trial), `tuning_hgb.json` (summary), `results/logs/05_tune_hgb.log`, figure `results/figures/tune_hgb.png`.

**Method.** Optuna with the **TPE sampler** (a Bayesian method: it builds a probability model of which settings gave good scores and samples the next trial where improvement is most likely), seed 42, 6 random start-up trials, and a **median pruner** that stops a trial early when its first folds are worse than the median of earlier trials.

**Objective.** Mean **observation-level Top-1** over **3-fold GroupKFold (groups = trips)** on a **15% fix-level subsample** of the 80 training trips (321,370 candidate rows), the same metric used for model selection. Log loss was recorded for every trial.

**Search space**

| Hyperparameter | Range / distribution | Stage-1 value | Best value |
|---|---|---|---|
| `max_iter` | 100–500, step 50 | 300 | **400** |
| `learning_rate` | 0.03–0.3, log-uniform | 0.1 | **0.149** |
| `max_leaf_nodes` | 15–127, log-uniform | 31 | **61** |
| `min_samples_leaf` | 20–400, log-uniform | 20 | **158** |
| `l2_regularization` | 0.001–30, log-uniform | 0 | **0.018** |
| `max_features` | 0.4–1.0 (share of features per split) | 1.0 | **0.68** |
| `max_bins` | {63, 127, 255} | 255 | **127** |

**Trials.** 19 trials: **9 completed, 10 pruned**. Best trial: #7, search Top-1 **0.9021** (the stage-1 configuration scored 0.8927 on the same subsample folds). The best configuration was then refitted on all 80 training trips and evaluated with full 5-fold GroupKFold and on student_test.

**Tuned vs untuned** (same data partitions and metrics; untuned = the stage-1 HistGradientBoosting from §14):

| Metric | Untuned (stage 1) | Tuned | Change |
|---|---|---|---|
| CV Top-1 (5-fold GroupKFold) | 0.8947 | **0.9024** | +0.0077 |
| CV Macro F1 | 0.9425 | **0.9438** | +0.0013 |
| Student-test Top-1 | 0.8991 | **0.9052** | +0.0061 |
| Student-test Macro F1 | 0.9436 | **0.9468** | +0.0032 |
| Student-test log loss | 0.0401 | **0.0395** | −0.0006 |
| Inference time (ms per fix) | **0.020** | 0.209 | about 10× slower |
| Full-fit time (s) | **18.6** | 151.2 | |

The tuned model is better on every accuracy and probability metric. The price is inference speed: it uses 400 larger trees with early stopping off, so it is about 10× slower. At about 0.2 ms per fix, a 4,000-fix trip still takes under a second, which is acceptable for the API.

> `tuning_hgb.json` also lists a "baseline" of 0.8948 Top-1. That is the stage-1 settings refitted **with early stopping turned off**, to match the search setup. The stage-1 model in §14 had scikit-learn's default early stopping **on** and scored 0.8991. The table above compares against the real stage-1 model.

**Caveats.** The search used a 15% subsample and 3 folds to fit the time budget, and more than half of the trials were pruned. A longer search on more data could find better settings.

An extra LightGBM search (`05_tune_lgbm.py`, `tuning_lgbm*.{json,csv}`, `tune_lgbm.png`) is kept for comparison only. The deployed model is the tuned HistGradientBoosting.

---

## 18. HMM + Viterbi route reconstruction

Code: `src/pathsnap/hmm.py`, used by `src/pathsnap/matcher.py`.

**Why.** The classifier scores every fix independently, so two consecutive fixes can be assigned to roads that are not connected. The HMM adds the rule that a vehicle moves along the road network.

**Hidden states.** The candidate directed arcs of each fix (at most 40 per fix). The arc ids are kept unchanged, so each state maps directly to a road segment.

**Emission score.** `emis_w × log P(arc)`, using the tuned classifier's probability. Probabilities are floored at `p_floor` to avoid log(0).

**Transition score** between arc *a* (fix *t*) and arc *b* (fix *t+1*), in log space:

| Case | Score |
|---|---|
| same arc | 0 |
| *b* reachable from *a* in *h* ≤ `max_hops` directed hops | −`hop_pen` × *h* |
| *b* is the reverse twin of *a* (U-turn) | −`uturn_pen` |
| not reachable (disconnected) | −`jump_pen` (finite, so decoding never fails) |

**Large GPS gaps.** If consecutive fixes are more than `gap_s` seconds apart, all transition penalties are multiplied by `gap_scale` (< 1), because the vehicle may have travelled far during the gap.

**Missing candidates.** Fixes with no candidate inherit the arc decoded for the nearest earlier fix (or the first decoded fix, at the start of a trip).

**Disconnected consecutive arcs.** They are allowed at the `jump_pen` cost instead of being forbidden. When the final route is built, gaps between non-adjacent decoded arcs are bridged with the shortest directed path (BFS, up to 15 hops), so the returned route is connected wherever the network allows.

**Viterbi recurrence.** δ<sub>t</sub>(b) = max<sub>a</sub> [δ<sub>t−1</sub>(a) + trans(a, b)] + emis<sub>t</sub>(b), with back-pointers. The best final state is traced back to give one arc per fix. Scores are re-centred every step for numerical stability.

**Tuning** (`scripts/06_tune_hmm.py`). Optuna TPE over `emis_w`, `hop_pen`, `uturn_pen`, `jump_pen`, `max_hops`, `gap_s`, `gap_scale` and `p_floor`, maximising Top-1 after Viterbi on 16 training trips. The classifier probabilities used are **out-of-fold**, so no trip is scored by a model that was trained on it. The result is checked on 16 *other* training trips. Evaluator data is not used.

**Results: classifier-only vs classifier + HMM.**
<!-- TO FILL after running 06_tune_hmm.py and 08_eval_pipeline.py: copy the numbers from
     results/tables/tuning_hmm.json ("validation") and results/tables/pipeline_eval_student_test.json -->

| Metric (student_test, end-to-end) | Classifier only (argmax) | Classifier + HMM/Viterbi |
|---|---|---|
| Top-1 accuracy | _from pipeline_eval_student_test.json: top1_argmax_ | _top1_viterbi_ |
| Route continuity | _cont_argmax_ | _cont_viterbi_ |
| Route precision / recall / F1 | n/a | _route_precision / route_recall / route_f1_ |
| ms per fix | | _ms_per_fix_ |

---

## 19. API, UI and two-laptop deployment

| Laptop | Runs | Holds |
|---|---|---|
| **Laptop 1** | `app/server.py` (FastAPI) | tuned model, HMM parameters, road networks, **evaluator trips including their reference routes** |
| **Laptop 2** | `ui/index.html` in a browser | nothing except the UI file. No data and no ground truth |

**Endpoints**

| Method & path | Input | Output |
|---|---|---|
| `GET /trips` | n/a | the 10 evaluation trip ids found in `PATHSNAP_EVAL_DATA/evaluator_test` |
| `POST /predict` | `{"trip_id": "00000015"}` | predicted route (arc ids), arc per fix, confidence, **metrics** (Top-1 with HMM and classifier-only, route precision/recall/F1, continuity), timing, and a Folium map (GPS + road network + predicted route) |
| `GET /health`, `GET /info` | n/a | status, model and HMM settings |
| `POST /match`, `POST /match_files` | raw GPS + network | development endpoints; not used in the evaluation |

**How ground truth stays on the server.** `/predict` loads the trip **without** its `.route` file and runs the full prediction. Only afterwards does the server read `.route` (on Laptop 1) to compute the metrics. The reference route and labels are never included in the response, and the UI only displays numbers the API returns. Trip ids are checked against the list, so no other files can be requested. Every request is logged to Laptop 1's console with the trip id, client IP, metrics and timing.

**Run it**
```cmd
:: Laptop 1 - put the evaluator trips in %USERPROFILE%\evaluator_data\evaluator_test\ then:
run_server.bat
:: (or: set "PATHSNAP_EVAL_DATA=C:\path\to\evaluator_data"  &  python -m uvicorn app.server:app --host 0.0.0.0 --port 8000)

:: Laptop 2 - double-click ui\index.html, enter http://<Laptop-1-IP>:8000, press Connect, pick a trip.
```
Both laptops must be on the same network (a phone hotspot is the most reliable), and Windows Firewall on Laptop 1 must allow TCP port 8000. The map tiles need internet access.

**Measured latency** (development machine, tuned model, default HMM settings): 262-fix trip ≈ 1.2 s end to end, 3,958-fix trip ≈ 5.5 s, largest training trip (19,313 fixes) ≈ 5.4 s, largest network (307,392 arcs) ≈ 6.8 s. Most of this time is server-side scoring and map building, not prediction.

---

## 20. Limitations

- **Labels are derived, not given.** They depend on our alignment method and the 50 m tolerance. Reference routes are incomplete for 5 training trips, so labels inherit those limitations.
- **Excluded fixes.** 11.1% of training fixes (mostly from 5 trips) are unlabelable and excluded from training and scoring.
- **Tuning budget.** The HistGradientBoosting search used a 15% subsample, 3 folds and 19 trials (10 pruned).
- **Speed trade-off.** The tuned model is about 10× slower per fix than the stage-1 model.
- **MLP compute shortcut.** It was trained on 30% of training fixes, which may understate its potential.
- **Route F1 is set-based.** It compares the sets of arcs in the predicted and reference routes, not their order.

---

## 21. Team

| Member | Main responsibility (Stage 1) |
|---|---|
| Member 1 | Repository setup, data loading (`io.py`), projection and geometry (`geo.py`), shared config |
| Member 2 | Reference alignment labels (`labels.py`), candidate generation and features (`candidates.py`), `01_build_dataset.py` |
| Member 3 | EDA (`02_eda.py`), data-quality record, Folium maps (`viz.py`) |
| Member 4 | Metrics (`metrics.py`), model training and comparison (`03_train_compare.py`, `04_make_report_tables.py`) |

<!-- Add each member's Stage 2 responsibility (tuning, HMM, API, UI) here. -->

<!-- Replace "Member 1..4" with names and SRNs. -->

---

## 22. AI assistance disclosure

<!-- Edit this section so it accurately describes how your team used AI tools. -->
This project used an AI assistant (Claude, by Anthropic) to help write code and documentation. All results in this repository come from actually running the scripts on the provided data. The run logs are in `results/logs/`, and every number in this README can be traced to a file in `results/tables/`. Every team member has reviewed the code and results and can explain them.
