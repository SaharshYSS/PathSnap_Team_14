import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA_ROOT = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(ROOT), "student", "Student folder"))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
for d in (PROC, RES + "/figures", RES + "/tables", RES + "/logs", ROOT + "/maps", ROOT + "/artifacts"):
    os.makedirs(d, exist_ok=True)
SEED = 42
