"""Projection + point/segment geometry. Distances are in METRES in a local UTM zone
(lon/lat are angular, so Euclidean distance in degrees is not meaningful)."""
from __future__ import annotations
import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(lon_c: float, lat_c: float):
    return Transformer.from_crs(4326, utm_epsg(lon_c, lat_c), always_xy=True)


def seg_proj(px, py, ax, ay, bx, by):
    """Project points P onto segments AB (all arrays broadcastable, same length).
    Returns d (perp. distance to the *segment*), s (metres along AB from A, unclamped),
    L (segment length), lat (signed lateral offset, left of A->B positive)."""
    dx, dy = bx - ax, by - ay
    L = np.hypot(dx, dy)
    Ls = np.maximum(L, 1e-6)
    ux, uy = dx / Ls, dy / Ls
    wx, wy = px - ax, py - ay
    s = wx * ux + wy * uy
    lat = ux * wy - uy * wx
    sc = np.clip(s, 0.0, L)
    d = np.hypot(wx - sc * ux, wy - sc * uy)
    return d, s, L, lat


def wrap180(a):
    """Wrap an angle difference in radians to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
