"""Plotly-Abbildungen der Genetic-Algorithm-Demo: Karte der Lieferroute, Kostenlandschaft der Standortwahl mit Population,
Fitness-/Diversitätsverlauf, Konvergenz-, Pareto- und Sweep-Abbildungen. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ga_constants as C

TOUR_COLOR = "#4c78a8"
BEST_COLOR = "#54a24b"
POP_COLOR = "#9ecae9"
STAU_COLOR = "#e45756"
REF_COLOR = "#7f7f7f"
CONTOUR_SCALE = "Blues_r"

LANDSCAPE_RES = 60


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def _line_trace(xy, edges, color, name, dash=None, width=2.5):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip")


def build_route(xy, tour, co2_factor_matrix=None, title=None):
    """Lieferroute; stark CO2-belastete Straßenabschnitte (oberhalb eines hohen Perzentils aller Abschnitte dieser Instanz,
    siehe `ga_constants.CO2_HIGHLIGHT_PERCENTILE`) rot hervorgehoben, Depot als Stern. Belastung ist eine Kanteneigenschaft, keine Knoteneigenschaft."""
    fig = go.Figure()
    edges = _tour_edges_list(tour)
    if co2_factor_matrix is not None:
        n = len(co2_factor_matrix)
        off_diag = co2_factor_matrix[np.triu_indices(n, 1)]
        threshold = float(np.percentile(off_diag, C.CO2_HIGHLIGHT_PERCENTILE))
        heavy = [(a, b) for a, b in edges if co2_factor_matrix[a, b] > threshold]
        normal = [(a, b) for a, b in edges if co2_factor_matrix[a, b] <= threshold]
        fig.add_trace(_line_trace(xy, normal, TOUR_COLOR, "Route"))
        if heavy:
            fig.add_trace(_line_trace(xy, heavy, STAU_COLOR, "stark CO2-belasteter Abschnitt", width=3.5))
    else:
        fig.add_trace(_line_trace(xy, edges, TOUR_COLOR, "Route"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def _landscape_grid(inst, res=LANDSCAPE_RES):
    xs = np.linspace(0.0, C.AREA, res)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    z = inst.cost(grid).reshape(res, res)
    return xs, z


def build_landscape(inst, population=None, best_xy=None, grid_xy=None, title=None):
    """Kostenlandschaft (Konturplot, dunkler = günstiger) mit optionaler Population, bestem Fund und wahrem Gitteroptimum."""
    xs, z = _landscape_grid(inst)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=xs, y=xs, z=z, colorscale=CONTOUR_SCALE, showscale=False, contours=dict(coloring="fill", showlines=False), hoverinfo="skip"))
    if population is not None:
        fig.add_trace(go.Scatter(x=population[:, 0], y=population[:, 1], mode="markers", marker=dict(size=5, color=POP_COLOR, line=dict(width=0.5, color=TOUR_COLOR)), name="Population"))
    if grid_xy is not None:
        fig.add_trace(go.Scatter(x=[grid_xy[0]], y=[grid_xy[1]], mode="markers", marker=dict(size=13, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="global günstigste Lage"))
    if best_xy is not None:
        fig.add_trace(go.Scatter(x=[best_xy[0]], y=[best_xy[1]], mode="markers", marker=dict(size=11, symbol="diamond", color=BEST_COLOR, line=dict(width=1, color="white")), name="bester Fund"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_fitness_curve(best_hist, mean_hist, reference=None):
    xs = list(range(len(best_hist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=mean_hist, mode="lines", line=dict(color=POP_COLOR, width=2), name="Mittelwert"))
    fig.add_trace(go.Scatter(x=xs, y=best_hist, mode="lines", line=dict(color=BEST_COLOR, width=2.5), name="bester Wert"))
    if reference is not None and np.isfinite(reference):
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Referenz", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig, 300)


def build_diversity_curve(div_hist):
    xs = list(range(len(div_hist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=div_hist, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), name="Diversität"))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Diversität")
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_convergence(rows):
    """Anteil im globalen Trichter und Diversität am Ende, je Populationsgröße."""
    xs = [r["pop"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Anteil im globalen Trichter", "Diversität am Ende"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=xs, y=[r["share_global"] for r in rows], marker_color=BEST_COLOR, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=[r["diversity_end"] for r in rows], marker_color=TOUR_COLOR, showlegend=False), row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="Populationsgröße")
    return _base(fig, 300)


def build_convergence_curves(curves):
    """Diversitätsverlauf einer kleinen gegen eine große Population."""
    fig = go.Figure()
    colors = {"klein": STAU_COLOR, "groß": TOUR_COLOR}
    for label, r in curves.items():
        xs = list(range(len(r.diversity_history)))
        fig.add_trace(go.Scatter(x=xs, y=r.diversity_history, mode="lines", line=dict(color=colors[label], width=2.5), name=f"Population {label}"))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Diversität")
    return _base(fig, 280)


def build_pareto(report):
    """Alle Touren (grau), Pareto-Front (Linie), vom GA gefundene Punkte je Gewicht (Farbe = Gewicht)."""
    fig = go.Figure()
    pts = report["all_points"]
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=4, color="#d3d3d3"), name="alle Touren", hoverinfo="skip"))
    front = report["front"]
    fig.add_trace(go.Scatter(x=front[:, 0], y=front[:, 1], mode="lines+markers", line=dict(color=REF_COLOR, width=1.5, dash="dot"), marker=dict(size=6, color=REF_COLOR), name="Pareto-Front"))
    found = report["found"]
    fig.add_trace(go.Scatter(x=found[:, 0], y=found[:, 1], mode="markers", marker=dict(size=10, color=report["weights"], colorscale="Viridis", showscale=True, colorbar=dict(title="Gewicht CO2", thickness=12)), name="GA (gewichtete Summe)"))
    fig.update_xaxes(title_text="Distanz (km)")
    fig.update_yaxes(title_text="CO2-Kosten")
    return _base(fig, 380)


def build_operator_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    mean = np.array([r["mean_best"] for r in rows])
    sd = np.array([r["sd_best"] for r in rows])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=mean + sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mean - sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(76,120,168,0.2)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=mean, mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Bester Wert")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abstand zur Referenz (%)", "Diversität am Ende"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["diversity_end"] for r in rows], mode="lines+markers", line=dict(color=BEST_COLOR, width=2.5), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text=param_label)
    return _base(fig, 300)
