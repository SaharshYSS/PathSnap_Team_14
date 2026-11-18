"""Folium map helpers (interactive HTML) used for EDA now and for predicted routes later."""
from __future__ import annotations
import numpy as np
import folium


def _crop_arcs_ll(trip, pad_deg=0.002, max_arcs=6000):
    lon, lat = trip.lon, trip.lat
    nd, ar = trip.nodes_lonlat, trip.arcs
    a, b = nd[ar[:, 0]], nd[ar[:, 1]]
    keep = (np.maximum(a[:, 0], b[:, 0]) >= lon.min() - pad_deg) & (np.minimum(a[:, 0], b[:, 0]) <= lon.max() + pad_deg) & \
           (np.maximum(a[:, 1], b[:, 1]) >= lat.min() - pad_deg) & (np.minimum(a[:, 1], b[:, 1]) <= lat.max() + pad_deg)
    idx = np.nonzero(keep)[0]
    if len(idx) > max_arcs:
        idx = idx[np.linspace(0, len(idx) - 1, max_arcs).astype(int)]
    return idx


def make_trip_map(trip, pred_arcs=None, title=None, gps_step=None, show_network=True, max_arcs=6000,
                  candidate_fix=None, candidate_arcs=None):
    """Layers: raw GPS (blue), road network (grey), reference route (green), predicted route (red),
    optional candidate arcs of one fix (orange). Layer control + legend included."""
    c = [float(np.mean(trip.lat)), float(np.mean(trip.lon))]
    m = folium.Map(location=c, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(trip.lat.min()), float(trip.lon.min())], [float(trip.lat.max()), float(trip.lon.max())]])
    nd, ar = trip.nodes_lonlat, trip.arcs
    if show_network:
        fg = folium.FeatureGroup(name="Road network (directed arcs)", show=True)
        for i in _crop_arcs_ll(trip, max_arcs=max_arcs):
            p, q = nd[ar[i, 0]], nd[ar[i, 1]]
            folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color="#8a8a8a", weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)

    def route_layer(arcs, name, color, weight, show=True):
        fg = folium.FeatureGroup(name=name, show=show)
        for a in arcs:
            p, q = nd[ar[a, 0]], nd[ar[a, 1]]
            folium.PolyLine([[p[1], p[0]], [q[1], q[0]]], color=color, weight=weight, opacity=0.9,
                            tooltip=f"arc {int(a)}").add_to(fg)
        fg.add_to(m)

    if trip.route is not None:
        route_layer(trip.route, "Reference route (green)", "#1a9850", 5)
    if pred_arcs is not None:
        route_layer(pred_arcs, "Predicted route (red)", "#d73027", 3)
    if candidate_fix is not None and candidate_arcs is not None:
        route_layer(candidate_arcs, f"Candidate arcs of fix {candidate_fix} (orange)", "#fd8d3c", 4)
    step = gps_step or max(1, len(trip.lon) // 1500)
    fg = folium.FeatureGroup(name="Raw GPS fixes (blue)", show=True)
    for i in range(0, len(trip.lon), step):
        folium.CircleMarker([trip.lat[i], trip.lon[i]], radius=2.5, color="#2166ac", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"fix {i} t={trip.t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    if candidate_fix is not None:
        folium.Marker([trip.lat[candidate_fix], trip.lon[candidate_fix]], tooltip=f"fix {candidate_fix}",
                      icon=folium.Icon(color="black", icon="crosshairs", prefix="fa")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 12px;
      border:1px solid #888;border-radius:4px;font:12px sans-serif">
      <b>{title or 'Trip ' + trip.trip_id}</b><br>
      <span style="color:#2166ac">&#9679;</span> raw GPS fixes<br>
      <span style="color:#8a8a8a">&#9473;</span> road network<br>
      <span style="color:#1a9850">&#9473;</span> reference route<br>
      <span style="color:#d73027">&#9473;</span> predicted route<br>
      <span style="color:#fd8d3c">&#9473;</span> candidates (one fix)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
