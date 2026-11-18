# PathSnap - GPS map matching (stage 1: preprocessing, features, model comparison)

Pipeline: GPS fix -> candidate directed arcs within 60 m -> classifier scores P(arc is true) -> (next stage) HMM/Viterbi.

## Run order
```
pip install -r requirements.txt
export PATHSNAP_DATA="<path>/Student folder"          # contains train/ and student_test/
python scripts/01_build_dataset.py                      # projection, candidates, features, derived labels -> data/processed
python scripts/02_eda.py                                # tables, figures, Folium maps (maps/*.html)
python scripts/03_train_compare.py                      # 6 models, 5-fold GroupKFold CV (groups = trips) + student_test
python scripts/04_make_report_tables.py                 # comparison table + figures
```
## Layout
src/pathsnap: io, geo (UTM projection), labels (monotone GPS->route alignment), candidates (candidate generation + features), metrics, viz (Folium).
results/tables: model_comparison.csv/.md, _model_results.json (all configs, seeds, per-fold results), dataset_inventory.csv.
results/figures: eda_*.png, cmp_*.png.  results/logs: run logs.  maps/: Folium HTML.

## Key decisions
- Labels: dataset has only the reference route, so each fix is aligned to it by a monotone DP; fixes >50 m from their aligned reference arc are "off-reference" and excluded (5 train trips have incomplete routes; 00000054 is only 11% labelable).
- Candidate radius 60 m: candidate miss rate 0.0% on labelable train fixes (30 m: 1.3%).
- Seed 42. Threshold for candidate F1 is chosen on out-of-fold predictions only. student_test is used for comparison only; evaluator data is never touched.
- Selection metric (declared up front): CV observation-level Top-1; tie-break CV log loss, continuity, inference time.
- Route continuity = share of consecutive fixes (dt<=5 s) whose predicted arcs are equal or reachable within 3 directed arc hops.
