"""Zwei Vehikel für die Genetic-Algorithm-Demo, beide seed-bestimmt:

- **Diskret** (`generate_perm`): ein Depot in der Mitte und n Kundenstopps in einem 100 x 100-km-Gebiet (wie hill-climbing-demo);
  zusätzlich ein CO2-Faktor je STRASSENABSCHNITT (Kante), unabhängig von der Distanz - manche Verbindungen sind stauanfälliger
  als andere, unabhängig vom Zielort. Das macht Distanz und CO2 zu echt unterschiedlichen Zielen (nicht nur eine Umgewichtung
  der Distanz): die kürzeste Tour ist nicht automatisch auch die CO2-günstigste.
- **Kontinuierlich** (`generate_real`): Standortwahl (x, y) im selben Gebiet; die Kosten sind mehrere Gauß-Mulden
  unterschiedlicher Tiefe und Breite - nur die tiefste ist die global günstigste Lage, die anderen sind lokale Minima."""

from dataclasses import dataclass

import numpy as np

import ga_constants as C


@dataclass(frozen=True)
class PermInstance:
    xy: np.ndarray                  # (n + 1, 2); Zeile 0 = Depot
    co2_factor_matrix: np.ndarray   # (n + 1, n + 1); symmetrisch, CO2-Faktor je Kante (unabhängig von der Distanz)
    n: int
    cluster_share: int
    seed: int

    @property
    def n_nodes(self):
        return self.n + 1


def generate_perm(n, cluster_share=0, seed=0):
    rng = np.random.default_rng(seed)
    n_grouped = int(round(n * cluster_share / 100))
    uniform = rng.random((n - n_grouped, 2)) * C.AREA
    centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
    which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
    grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
    depot = np.array([[C.AREA / 2, C.AREA / 2]])
    xy = np.vstack([depot, uniform, grouped])
    n_nodes = n + 1
    raw = rng.uniform(C.CO2_FACTOR_LO, C.CO2_FACTOR_HI, size=(n_nodes, n_nodes))
    co2_factor_matrix = (raw + raw.T) / 2.0
    np.fill_diagonal(co2_factor_matrix, 0.0)
    return PermInstance(xy, co2_factor_matrix, n, int(cluster_share), int(seed))


@dataclass(frozen=True)
class RealInstance:
    centres: np.ndarray        # (K, 2)
    amplitudes: np.ndarray     # (K,)
    sigmas: np.ndarray         # (K,)
    offset: float
    seed: int

    def cost(self, xy):
        """Kosten an Punkt(en) `xy` (..., 2) - niedriger ist besser. Beliebige führende Dimensionen (auch keine: ein einzelner Punkt gibt einen Skalar zurück)."""
        xy = np.asarray(xy, dtype=float)
        flat = xy.reshape(-1, 2)
        d2 = ((flat[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=-1)        # (M, K)
        wells = self.amplitudes[None, :] * np.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))
        result = self.offset - wells.sum(axis=-1)                                     # (M,)
        return result.reshape(xy.shape[:-1])


def generate_real(seed=0):
    rng = np.random.default_rng(seed)
    centres = C.WELL_MARGIN + rng.random((C.K_WELLS, 2)) * (C.AREA - 2 * C.WELL_MARGIN)
    amplitudes = rng.uniform(C.AMP_MIN, C.AMP_MAX, size=C.K_WELLS)
    sigmas = rng.uniform(C.SIGMA_MIN, C.SIGMA_MAX, size=C.K_WELLS)
    offset = float(amplitudes.sum())
    return RealInstance(centres, amplitudes, sigmas, offset, int(seed))


def grid_optimum(inst, step=C.GRID_STEP):
    """Bester Punkt eines feinen Gitters über das Gebiet - Referenz für "im globalen Trichter gefunden"."""
    xs = np.arange(0.0, C.AREA + step, step)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    costs = inst.cost(grid)
    k = int(np.argmin(costs))
    return grid[k], float(costs[k])
