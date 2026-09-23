"""Auswertung der Genetic-Algorithm-Demo: ein GA-Lauf gegen eine Referenz (Brute-Force bei kleinen diskreten Instanzen,
Gitter-Optimum bei der kontinuierlichen), Sweeps über die GA-Regler, und drei Experimente:
vorzeitige Konvergenz (kontinuierlich), gewichtete Summe verfehlt die Pareto-Front (diskret, Distanz vs. CO2),
Operator-Sensitivität (Crossover-/Mutationsrate)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import ga_algorithm as A
import ga_constants as C
import ga_scenario as S

BRUTE_FORCE_MAX_N = 9      # (n-1)! Touren; bei 9 sind das 40 320 - noch < 1 s


@dataclass(frozen=True)
class Settings:
    encoding: str = "perm"
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    co2_weight: float = C.DEFAULT_CO2_WEIGHT
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    cx: float = C.DEFAULT_CX
    mut: float = C.DEFAULT_MUT
    elitism: int = C.DEFAULT_ELITE
    k: int = C.DEFAULT_K
    ga_seed: int = C.DEFAULT_GA_SEED


@lru_cache(maxsize=256)
def perm_instance(n, cluster_share, seed):
    inst = S.generate_perm(n, cluster_share, seed)
    return inst, A.dist_matrix(inst.xy)


@lru_cache(maxsize=64)
def real_instance(seed):
    inst = S.generate_real(seed)
    grid_xy, grid_cost = S.grid_optimum(inst)
    return inst, grid_xy, grid_cost


def fitness_fn(settings):
    """(Fitnessfunktion, Knotenzahl/Dimension, Grenzen) für `settings`. Fitness ist ein Array je Populationszeile, niedriger ist besser."""
    if settings.encoding == "perm":
        inst, D = perm_instance(settings.n, settings.cluster_share, settings.seed)

        def fitness(pop):
            dist = A.tour_length_batch(pop, D)
            if settings.co2_weight <= 0:
                return dist
            co2 = A.tour_co2_batch(pop, D, inst.co2_factor_matrix)
            return (1.0 - settings.co2_weight) * dist + settings.co2_weight * co2
        return fitness, inst.n_nodes, (0.0, 1.0)
    inst, grid_xy, grid_cost = real_instance(settings.seed)

    def fitness(pop):
        return inst.cost(pop)
    return fitness, 2, (0.0, C.AREA)


def brute_force_perm(n_nodes, fitness):
    """Bestes von allen (n_nodes - 1)! Touren (Knoten 0 fest an erster Stelle) - nur für sehr kleine Instanzen."""
    best = None
    for perm in permutations(range(1, n_nodes)):
        tour = np.array((0,) + perm)
        f = float(fitness(tour[None, :])[0])
        if best is None or f < best:
            best = f
    return best


def run(settings, keep_history=False):
    fitness, dim_or_n, bounds = fitness_fn(settings)
    return A.run_ga(settings.encoding, fitness, dim_or_n, bounds, settings.pop, settings.gens, settings.cx, settings.mut, settings.elitism, settings.k, settings.ga_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    reference: float       # Brute-Force-/Gitter-Optimum, falls verfügbar, sonst NaN

    @property
    def gap(self):
        if self.reference is None or not np.isfinite(self.reference) or self.reference == 0:
            return float("nan")
        return 100.0 * (self.result.best_fitness - self.reference) / abs(self.reference)


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    if settings.encoding == "real":
        _, _, reference = real_instance(settings.seed)
    elif settings.n + 1 <= BRUTE_FORCE_MAX_N:
        fitness, dim_or_n, _ = fitness_fn(settings)
        reference = brute_force_perm(dim_or_n, fitness)
    else:
        reference = float("nan")
    return Analysis(settings, result, reference)


NEAR_GAP = 2.0      # Prozent über der Referenz, unterhalb dessen "nahezu optimal" gilt


def verdict(a):
    if not np.isfinite(a.gap):
        return "no_reference"
    return "near_optimal" if a.gap < NEAR_GAP else "gap"


# --- Sweeps (allgemein, wie hill-climbing-demo) --------------------------------------------------------------------------------------------


def run_config(base, seeds=None, ga_seeds=None, **changes):
    """Mittel über feste Vehikel-Seeds x GA-Seeds für `base` mit `changes`."""
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    ga_seeds = C.SWEEP_GA_SEEDS if ga_seeds is None else ga_seeds
    s0 = replace(base, **changes)
    runs = []
    for seed in seeds:
        for ga_seed in ga_seeds:
            a = analyse(replace(s0, seed=seed, ga_seed=ga_seed), keep_history=False)
            runs.append({"gap": a.gap, "best": a.result.best_fitness, "diversity_end": a.result.diversity_history[-1]})
    return {"gap": float(np.nanmean([r["gap"] for r in runs])), "best": float(np.mean([r["best"] for r in runs])), "diversity_end": float(np.mean([r["diversity_end"] for r in runs])), "n_runs": len(runs)}


def sweep(param, base=Settings(encoding="real"), values=None):
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


# --- Experiment 1: vorzeitige Konvergenz (kontinuierlich) ----------------------------------------------------------------------------------


def convergence_experiment(seed=C.DEFAULT_SEED, pop_sizes=None, seeds=None, gens=None):
    """Für jede Populationsgröße: Anteil der Läufe, die im globalen Trichter enden, und mittlere Diversität am Ende."""
    pop_sizes = C.CONVERGENCE_POP_SIZES if pop_sizes is None else pop_sizes
    seeds = C.CONVERGENCE_SEEDS if seeds is None else seeds
    gens = C.DEFAULT_GEN if gens is None else gens
    inst, grid_xy, grid_cost = real_instance(seed)
    rows = []
    for pop in pop_sizes:
        share_global, final_div = [], []
        for ga_seed in seeds:
            s = Settings(encoding="real", seed=seed, pop=pop, gens=gens, ga_seed=ga_seed)
            r = run(s, keep_history=False)
            dist_to_best = float(np.sqrt(((r.best_individual - grid_xy) ** 2).sum()))
            share_global.append(dist_to_best <= C.GLOBAL_TOL_KM)
            final_div.append(r.diversity_history[-1])
        rows.append({"pop": pop, "share_global": float(np.mean(share_global)), "diversity_end": float(np.mean(final_div))})
    return rows


def convergence_curves(seed=C.DEFAULT_SEED, small=None, large=None, gens=None, ga_seed=C.DEFAULT_GA_SEED):
    """Diversitäts- und Bestwert-Verlauf für eine kleine gegen eine große Population, ein GA-Seed je Größe."""
    small = C.CONVERGENCE_POP_SIZES[0] if small is None else small
    large = C.CONVERGENCE_POP_SIZES[-1] if large is None else large
    gens = C.DEFAULT_GEN if gens is None else gens
    out = {}
    for label, pop in (("klein", small), ("groß", large)):
        s = Settings(encoding="real", seed=seed, pop=pop, gens=gens, ga_seed=ga_seed)
        out[label] = run(s, keep_history=False)
    return out


# --- Experiment 2: gewichtete Summe verfehlt die Pareto-Front (diskret, Distanz vs. CO2) --------------------------------------------------


def pareto_front_bruteforce(n=None, cluster_share=0, seed=C.PARETO_VEHICLE_SEED):
    n = C.PARETO_N if n is None else n
    inst, D = perm_instance(n, cluster_share, seed)
    points = []
    for perm in permutations(range(1, n + 1)):
        tour = np.array((0,) + perm)[None, :]
        dist = float(A.tour_length_batch(tour, D)[0])
        co2 = float(A.tour_co2_batch(tour, D, inst.co2_factor_matrix)[0])
        points.append((dist, co2))
    points = np.array(points)
    order = np.argsort(points[:, 0])
    front, best_co2 = [], np.inf
    for idx in order:
        d, c = points[idx]
        if c < best_co2 - 1e-9:
            front.append((d, c))
            best_co2 = c
    return np.array(front), points


def pareto_experiment(n=None, weights=None, seeds=None, cluster_share=0, seed=C.PARETO_VEHICLE_SEED, pop=C.DEFAULT_POP, gens=C.DEFAULT_GEN):
    n = C.PARETO_N if n is None else n
    weights = C.PARETO_WEIGHTS if weights is None else weights
    seeds = C.PARETO_SEEDS if seeds is None else seeds
    front, all_points = pareto_front_bruteforce(n, cluster_share, seed)
    inst, D = perm_instance(n, cluster_share, seed)
    found = []
    for w in weights:
        best_point, best_fit = None, None
        for ga_seed in seeds:
            s = Settings(encoding="perm", n=n, cluster_share=cluster_share, co2_weight=w, seed=seed, pop=pop, gens=gens, ga_seed=ga_seed)
            r = run(s, keep_history=False)
            tour = r.best_individual[None, :]
            d = float(A.tour_length_batch(tour, D)[0])
            c = float(A.tour_co2_batch(tour, D, inst.co2_factor_matrix)[0])
            if best_fit is None or r.best_fitness < best_fit:
                best_fit, best_point = r.best_fitness, (d, c)
        found.append(best_point)
    found = np.array(found)
    reached = 0
    for fd, fc in front:
        if np.any((np.abs(found[:, 0] - fd) < 1e-6) & (np.abs(found[:, 1] - fc) < 1e-6)):
            reached += 1
    return {"front": front, "all_points": all_points, "found": found, "weights": np.array(weights), "front_size": len(front), "reached": reached}


# --- Experiment 3: Operator-Sensitivität (Crossover-/Mutationsrate) ------------------------------------------------------------------------


def operator_sweep(param, values, base=None, seeds=None):
    base = Settings(encoding="real") if base is None else base
    seeds = C.OPERATOR_SEEDS if seeds is None else seeds
    rows = []
    for v in values:
        s = replace(base, **{param: v})
        bests = []
        for ga_seed in seeds:
            r = run(replace(s, ga_seed=ga_seed), keep_history=False)
            bests.append(r.best_fitness)
        rows.append({"value": v, "mean_best": float(np.mean(bests)), "sd_best": float(np.std(bests))})
    return rows
