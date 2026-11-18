r"""PathSnap map-matching API (FastAPI) - runs on Laptop 1.

EVALUATOR FLOW (project brief, "Final Evaluation"):
    GET  /trips              -> the ten reserved evaluation trip ids (read from PATHSNAP_EVAL_DATA on THIS laptop)
    POST /predict {trip_id}  -> server loads the trip, runs candidates -> tuned HistGB -> HMM/Viterbi, scores the
                                prediction against the reference route stored on THIS laptop, and returns the
                                predicted route, metrics and a Folium map (GPS + predicted route only).
    The reference route / labels are read only AFTER prediction, only for scoring, and are never sent to the UI.
    Every request is logged to this console (the evaluator is shown these logs).

Start (Windows CMD):  set "PATHSNAP_EVAL_DATA=C:\path\to\evaluator_data"   (folder that contains evaluator_test\)
                      python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
Raw-data endpoints used during development (client sends GPS + network):
    PATHSNAP_MODEL=artifacts/hgb_tuned.joblib uvicorn app.server:app --host 0.0.0.0 --port 8000
The evaluator laptop then calls  http://<laptop-1-ip>:8000/match  (interactive docs at /docs).

POST /match        JSON  {trip_id?, gps | track, nodes, arcs}
                   gps   : [[lon,lat,t],...] | [{lon,lat,t},...] | {"lon":[..],"lat":[..],"t":[..]} | text "lon lat t" per line
                   nodes : [[lon,lat],...]   (row index = node id)         arcs: [[from,to],...] (row index = arc id)
POST /match_files  multipart form with the raw files  track / nodes / arcs  (same format as the dataset)
GET  /health, GET /info
Response: {trip_id, n_fixes, fix_arcs[T], fix_confidence[T], route[arc ids, connected], timing_ms}
"""
import os, sys, io, json, time, threading
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.matcher import Matcher

MODEL = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "artifacts", "hgb_tuned.joblib"))
HMMP = os.environ.get("PATHSNAP_HMM", os.path.join(ROOT, "artifacts", "hmm_params.json"))
matcher = Matcher(MODEL, HMMP)
STATS = dict(requests=0, fixes=0, started=time.time()); LOCK = threading.Lock()
app = FastAPI(title="PathSnap map matching", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_first = lambda d, ks: next((d[k] for k in ks if k in d and d[k] is not None), None)

def _table(x, ncol, names):
    """-> float ndarray (n, ncol) from list-of-lists | list-of-dicts | dict-of-arrays | text"""
    if isinstance(x, str):
        x = np.loadtxt(io.StringIO(x), ndmin=2); return np.asarray(x, dtype=float)[:, :ncol]
    if isinstance(x, dict):
        cols = [_first(x, n) for n in names]
        if any(c is None for c in cols[:ncol]): raise ValueError(f"expected keys like {names}")
        return np.column_stack([np.asarray(c, dtype=float) for c in cols if c is not None])
    if len(x) and isinstance(x[0], dict):
        rows = [[float(v) for v in (_first(r, n) for n in names[:ncol]) if v is not None] for r in x]
        return np.array(rows, dtype=float)
    return np.asarray(x, dtype=float).reshape(len(x), -1)

def parse_trip(p: dict):
    gps = _first(p, ["gps", "track", "points", "trajectory", "observations", "fixes", "coords", "coordinates"])
    nodes = _first(p, ["nodes", "network_nodes", "node_coords"]); arcs = _first(p, ["arcs", "edges", "network_arcs", "links"])
    if gps is None or nodes is None or arcs is None: raise HTTPException(422, "need gps (or track), nodes and arcs")
    try:
        G = _table(gps, 3, [["lon", "lng", "longitude", "x"], ["lat", "latitude", "y"], ["t", "time", "timestamp", "ts"]])
        N = _table(nodes, 2, [["lon", "lng", "longitude", "x"], ["lat", "latitude", "y"]])[:, :2]
        A = _table(arcs, 2, [["from", "from_node", "source", "u", "a"], ["to", "to_node", "target", "v", "b"]])[:, :2].astype(np.int64)
    except Exception as e: raise HTTPException(422, f"could not parse input: {e}")
    t = G[:, 2] if G.shape[1] >= 3 else np.arange(1, len(G) + 1, dtype=float)
    if len(G) == 0: raise HTTPException(422, "empty gps")
    if A.min() < 0 or A.max() >= len(N): raise HTTPException(422, "arcs refer to unknown node ids")
    return G[:, 0], G[:, 1], t, N, A

def run(payload: dict):
    lon, lat, t, N, A = parse_trip(payload)
    # a few evaluators send lat,lon order - catch it (India: lat<40, lon>60)
    if lon.mean() < 40 and lat.mean() > 60 and N[:, 0].mean() < 40: lon, lat, N = lat, lon, N[:, ::-1]
    r = matcher.match(lon, lat, t, N, A)
    with LOCK: STATS["requests"] += 1; STATS["fixes"] += len(lon)
    return dict(trip_id=payload.get("trip_id"), n_fixes=len(lon), fix_arcs=r["fix_arcs"], fix_confidence=r["fix_confidence"], route=r["route"], timing_ms=r["timing_ms"])

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": True, "uptime_s": round(time.time() - STATS["started"], 1)}

@app.get("/info")
def info():
    f = os.path.join(ROOT, "results", "tables", "tuning_hgb.json")
    m = json.load(open(f)) if os.path.exists(f) else {}
    return {"model": "HistGradientBoosting candidate scorer (tuned) + HMM/Viterbi", "n_features": len(matcher.features), "threshold": matcher.threshold,
            "hmm_params": matcher.params.to_dict(), "model_params": m.get("params"), "student_test": m.get("student_test"), "served": {k: STATS[k] for k in ("requests", "fixes")}}

@app.post("/match")
def match(payload: dict): return run(payload)

@app.post("/match_files")
def match_files(track: UploadFile = File(...), nodes: UploadFile = File(...), arcs: UploadFile = File(...), trip_id: str = Form(None)):
    rd = lambda u: u.file.read().decode()
    return run(dict(trip_id=trip_id, track=rd(track), nodes=rd(nodes), arcs=rd(arcs)))


# ----------------------------------------------------------------------------------------------------------------
# Evaluator flow: trip id in -> prediction + server-side scoring out
# ----------------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathsnap.io import list_trips, load_trip
from pathsnap.candidates import project_trip
from pathsnap.labels import align_gps_to_route
from pathsnap.metrics import route_prf, build_next, _reach
from pathsnap.viz import make_trip_map

EVAL_ROOT = os.environ.get("PATHSNAP_EVAL_DATA", os.path.join(os.path.dirname(ROOT), "evaluator_data"))
EVAL_PART = os.environ.get("PATHSNAP_EVAL_PART", "evaluator_test")
LABEL_TOL = 50.0   # same rule as training/evaluation: fixes > 50 m from their aligned reference arc are not scored

def log(msg):
    print(f"[PathSnap {datetime.now():%H:%M:%S}] {msg}", flush=True)

def eval_trips():
    try: return list_trips(EVAL_ROOT, EVAL_PART)
    except FileNotFoundError: return []

log(f"model={os.path.basename(MODEL)} | HMM params={'tuned' if os.path.exists(HMMP) else 'DEFAULT (artifacts/hmm_params.json not found)'}")
log(f"evaluation trips: {len(eval_trips())} found in {os.path.join(EVAL_ROOT, EVAL_PART)}")

@app.get("/trips")
def trips(request: Request):
    ids = eval_trips()
    log(f"GET /trips from {request.client.host if request.client else '?'} -> {len(ids)} trips")
    if not ids: raise HTTPException(500, f"no trips found in {os.path.join(EVAL_ROOT, EVAL_PART)} - set PATHSNAP_EVAL_DATA")
    return {"partition": EVAL_PART, "trips": ids}

@app.post("/predict")
def predict(payload: dict, request: Request):
    tid = str(payload.get("trip_id", "")).strip()
    if tid not in eval_trips(): raise HTTPException(404, f"unknown trip_id '{tid}'")
    who = request.client.host if request.client else "?"
    log(f"POST /predict trip={tid} from {who}")
    t0 = time.perf_counter()
    # 1) prediction - the trip is loaded WITHOUT its reference route
    tp = load_trip(EVAL_ROOT, EVAL_PART, tid, with_route=False)
    r = matcher.match(tp.lon, tp.lat, tp.t, tp.nodes_lonlat, tp.arcs, return_candidates=True)
    pred = np.asarray(r["fix_arcs"]); cand, p = r.pop("_cand"), r.pop("_p"); r.pop("_graph", None)
    T = len(tp.lon)
    top = np.full(T, -1, dtype=np.int64)
    if len(cand):
        best = cand.assign(p=p).loc[lambda d: d.groupby("obs").p.idxmax()]
        top[best.obs.to_numpy()] = best.arc.to_numpy()
    t_pred = time.perf_counter() - t0
    # 2) scoring - reference route read only now, server-side; never returned
    ref_route = np.atleast_1d(np.loadtxt(os.path.join(EVAL_ROOT, EVAL_PART, tid, tid + ".route"), dtype=np.int64))
    g = project_trip(tp)
    ref, dref, _ = align_gps_to_route(g.gx, g.gy, g.nx, g.ny, g.arcs, ref_route)
    ok = dref <= LABEL_TOL
    nxt = build_next(tp.arcs)
    def cont(a):
        v = [_reach(nxt, a[i], a[i + 1]) for i in range(T - 1) if ok[i] and ok[i + 1] and tp.t[i + 1] - tp.t[i] <= 5]
        return float(np.mean(v)) if v else None
    pr, rc, f1 = route_prf(r["route"], ref_route.tolist())
    metrics = dict(
        top1_hmm=float((pred[ok] == ref[ok]).mean()) if ok.any() else None,
        top1_classifier_only=float((top[ok] == ref[ok]).mean()) if ok.any() else None,
        route_precision=pr, route_recall=rc, route_f1=f1,
        continuity_hmm=cont(pred), continuity_classifier_only=cont(top),
        scored_fixes=int(ok.sum()), total_fixes=T)
    # 3) map: GPS + road network + predicted route (tp.route is None, so no reference layer)
    m = make_trip_map(tp, pred_arcs=r["route"], title=f"Trip {tid} - predicted route")
    html = m.get_root().render()
    total_ms = (time.perf_counter() - t0) * 1e3
    with LOCK: STATS["requests"] += 1; STATS["fixes"] += T
    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
    log(f"  trip={tid} fixes={T} scored={int(ok.sum())} | Top-1 HMM={fmt(metrics['top1_hmm'])} "
        f"classifier-only={fmt(metrics['top1_classifier_only'])} | route P/R/F1={pr:.3f}/{rc:.3f}/{f1:.3f} | "
        f"predict {t_pred*1e3:.0f} ms, total {total_ms:.0f} ms")
    return dict(trip_id=tid, n_fixes=T, metrics=metrics,
                predicted_route=r["route"], fix_arcs=r["fix_arcs"], fix_confidence=r["fix_confidence"],
                timing_ms=dict(r["timing_ms"], scoring_and_map=round(total_ms - t_pred * 1e3, 1), total=round(total_ms, 1)),
                map_html=html)
