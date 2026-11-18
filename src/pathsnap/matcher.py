"""End-to-end inference: raw trip (GPS + road network) -> candidates -> classifier -> HMM/Viterbi -> arcs + route.
Uses exactly the stage-1 functions (project_trip / generate_candidates) so training and serving features are identical."""
from __future__ import annotations
import json, time
import numpy as np, joblib
from .candidates import project_trip, generate_candidates, DEFAULT_CFG
from .io import Trip
from .hmm import ArcGraph, HMMParams, viterbi, build_route


class Matcher:
    def __init__(self, model_path: str, hmm_path: str | None = None):
        b = joblib.load(model_path)
        self.model, self.features, self.threshold = b["model"], b["features"], b["threshold"]
        self.params = HMMParams()
        if hmm_path:
            try: self.params = HMMParams(**json.load(open(hmm_path))["params"])
            except FileNotFoundError: pass

    def match(self, lon, lat, t, nodes_lonlat, arcs, params: HMMParams | None = None, return_candidates=False):
        tm = {}; t0 = time.perf_counter()
        lon, lat, t = (np.asarray(x, dtype=float) for x in (lon, lat, t))
        nodes_lonlat = np.asarray(nodes_lonlat, dtype=float).reshape(-1, 2); arcs = np.asarray(arcs, dtype=np.int64).reshape(-1, 2)
        if len(lon) == 0: return dict(fix_arcs=[], route=[], fix_confidence=[], timing_ms={})
        # the evaluator's time stamps must be increasing for dt features; keep order but never fail on ties
        g = project_trip(Trip("req", lon, lat, t, nodes_lonlat, arcs))
        cand = generate_candidates(g, DEFAULT_CFG); tm["candidates"] = time.perf_counter() - t0
        if len(cand) == 0:
            return dict(fix_arcs=[-1] * len(lon), route=[], fix_confidence=[0.0] * len(lon), timing_ms={k: v * 1e3 for k, v in tm.items()})
        t1 = time.perf_counter(); p = self.model.predict_proba(cand[self.features])[:, 1]; tm["classifier"] = time.perf_counter() - t1
        t1 = time.perf_counter(); graph = ArcGraph(arcs)
        fix_arcs, conf = viterbi(cand, p, g.t, graph, params or self.params)
        route = build_route(fix_arcs, graph); tm["viterbi"] = time.perf_counter() - t1; tm["total"] = time.perf_counter() - t0
        out = dict(fix_arcs=fix_arcs.tolist(), route=route, fix_confidence=np.round(conf, 4).tolist(), timing_ms={k: round(v * 1e3, 1) for k, v in tm.items()})
        if return_candidates: out["_cand"], out["_p"], out["_graph"] = cand, p, graph
        return out
