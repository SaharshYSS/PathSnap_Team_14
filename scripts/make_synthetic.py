"""Builds a small synthetic dataset in the RAW PathSnap format (.track/.nodes/.arcs/.route) to smoke-test the pipeline
(scripts 06/08, API, demo) without the real data.  NOT used for any reported result.
Usage: python scripts/make_synthetic.py <out_root> [n_train=6] [n_test=3]"""
import sys, os, numpy as np
from pyproj import Transformer
out = sys.argv[1]; n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 6; n_te = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.RandomState(0); N, SP = 40, 70.0
to_ll = Transformer.from_crs(32643, 4326, always_xy=True)
gx, gy = np.meshgrid(np.arange(N), np.arange(N)); gx, gy = gx.ravel() * SP + 770000, gy.ravel() * SP + 1435000
lon, lat = to_ll.transform(gx + rng.randn(len(gx)) * 6, gy + rng.randn(len(gy)) * 6)
nodes = np.c_[lon, lat]; nid = lambda i, j: j * N + i
arcs = []
for j in range(N):
    for i in range(N):
        for di, dj in ((1, 0), (0, 1)):
            if i + di < N and j + dj < N: arcs += [(nid(i, j), nid(i + di, j + dj)), (nid(i + di, j + dj), nid(i, j))]
arcs = np.array(arcs); out_of = {}
for k, (a, b) in enumerate(arcs): out_of.setdefault(a, []).append(k)
def make(part, tid):
    d = f"{out}/{part}/{tid}"; os.makedirs(d, exist_ok=True)
    a = rng.randint(len(arcs)); route = [a]
    while len(route) < 45:
        nxt = [k for k in out_of[arcs[route[-1], 1]] if arcs[k, 1] != arcs[route[-1], 0]] or out_of[arcs[route[-1], 1]]
        route.append(int(rng.choice(nxt)))
    pts = [];
    for k in route:
        p, q = nodes_xy[arcs[k, 0]], nodes_xy[arcs[k, 1]]; L = np.hypot(*(q - p)); n = max(int(L // 10), 1)
        pts += [p + (q - p) * s / n for s in range(n)]
    pts = np.array(pts) + rng.randn(len(pts), 2) * 7.0
    ll = np.c_[to_ll.transform(pts[:, 0], pts[:, 1])]; t = np.arange(1, len(pts) + 1, dtype=float)
    np.savetxt(f"{d}/{tid}.track", np.c_[ll, t], fmt="%.7f %.7f %.1f"); np.savetxt(f"{d}/{tid}.nodes", nodes, fmt="%.7f %.7f")
    np.savetxt(f"{d}/{tid}.arcs", arcs, fmt="%d"); np.savetxt(f"{d}/{tid}.route", np.array(route), fmt="%d")
nodes_xy = np.c_[gx, gy]
for i in range(n_tr): make("train", f"{i+1:08d}")
for i in range(n_te): make("student_test", f"{i+1:08d}")
print("written", out)
