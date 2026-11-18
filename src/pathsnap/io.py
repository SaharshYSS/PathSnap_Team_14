"""Raw-data loading. File formats (verified in scripts/01_eda.py):
  .track : lon lat t(seconds)     (no header)
  .nodes : lon lat                (row index = node_id)
  .arcs  : from_node to_node      (row index = arc_id, directed)
  .route : arc_id per row         (chronological reference route; may be absent)
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Trip:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    nodes_lonlat: np.ndarray      # (N,2)
    arcs: np.ndarray              # (A,2) int
    route: Optional[np.ndarray] = None   # reference arc ids (never used by prediction code)


def list_trips(root: str, partition: str):
    base = os.path.join(root, partition)
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def load_trip(root: str, partition: str, trip_id: str, with_route: bool = True) -> Trip:
    d = os.path.join(root, partition, trip_id)
    p = os.path.join(d, trip_id)
    tr = np.loadtxt(p + ".track", ndmin=2)
    nd = np.loadtxt(p + ".nodes", ndmin=2)
    ar = np.loadtxt(p + ".arcs", dtype=np.int64, ndmin=2)
    rt = None
    if with_route and os.path.exists(p + ".route"):
        rt = np.atleast_1d(np.loadtxt(p + ".route", dtype=np.int64))
    return Trip(trip_id, tr[:, 0], tr[:, 1], tr[:, 2], nd, ar, rt)
