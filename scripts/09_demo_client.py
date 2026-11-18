"""Step 9 - stand-in for the evaluator interface (Laptop 2): sends a trip to the API, scores the answer, writes a Folium map.
Usage: python scripts/09_demo_client.py <server_url=http://localhost:8000> [partition=student_test] [trip_id=first]"""
import sys, json, time, requests, numpy as np
from _common import *
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import route_prf
from pathsnap.viz import make_trip_map
url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
part = sys.argv[2] if len(sys.argv) > 2 else "student_test"; tid = sys.argv[3] if len(sys.argv) > 3 else list_trips(DATA_ROOT, part)[0]
print("health:", requests.get(f"{url}/health", timeout=10).json())
tp = load_trip(DATA_ROOT, part, tid)
body = dict(trip_id=tid, gps=np.c_[tp.lon, tp.lat, tp.t].tolist(), nodes=tp.nodes_lonlat.tolist(), arcs=tp.arcs.tolist())   # route is NOT sent
t0 = time.time(); r = requests.post(f"{url}/match", json=body, timeout=600); dt = time.time() - t0; r.raise_for_status(); r = r.json()
pr, rc, f1 = route_prf(r["route"], tp.route.tolist())
print(f"trip {tid}: {r['n_fixes']} fixes -> route of {len(r['route'])} arcs | round-trip {dt:.2f}s (server {r['timing_ms']}) | route P/R/F1 vs reference = {pr:.3f}/{rc:.3f}/{f1:.3f}")
m = make_trip_map(tp, pred_arcs=r["route"], title=f"{tid} (API)"); out = f"{ROOT}/maps/demo_{part}_{tid}.html"; m.save(out); print("map:", out)
