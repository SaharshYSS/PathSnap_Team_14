import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA_ROOT = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(ROOT), "student", "Student folder"))
PROC = os.path.join(ROOT, "data", "processed")
# optional: the pre-built parquet bundle ("preprocessed data" folder) - used when data/processed does not exist
PREP = os.environ.get("PATHSNAP_PREP", "")
RES = os.path.join(ROOT, "results")
for d in (PROC, RES + "/figures", RES + "/tables", RES + "/logs", ROOT + "/maps", ROOT + "/artifacts"):
    os.makedirs(d, exist_ok=True)
SEED = 42

ARTI = os.path.join(ROOT, "artifacts")

def load_candidates(part, valid_only=True):
    """Candidate-feature rows for a partition ('train' | 'student_test').
    Reads data/processed/<part>/*.parquet (output of 01_build_dataset.py) or, if PATHSNAP_PREP is set,
    the bundled train_candidates_features_part*.parquet / student_test_candidates_features.parquet."""
    import glob, pandas as pd
    fs = sorted(glob.glob(f"{PROC}/{part}/*.parquet"))
    if not fs and PREP:
        fs = sorted(glob.glob(f"{PREP}/train_candidates_features_part*.parquet")) if part == "train" else [f"{PREP}/student_test_candidates_features.parquet"]
    if not fs:
        raise SystemExit(f"no candidate parquet for '{part}': run 01_build_dataset.py or set PATHSNAP_PREP")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d[d.obs_valid].reset_index(drop=True) if valid_only else d
