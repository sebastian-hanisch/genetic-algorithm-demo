"""Genetischer Algorithmus, numpy von Grund auf, für zwei Kodierungen:

- **Permutation** (diskrete Lieferroute, Knoten 0..N-1 als geschlossener Zyklus gelesen): Order Crossover (OX) + Tausch-Mutation.
- **Reellwertig** (kontinuierliche Standortwahl, (x, y) im Gebiet): BLX-α-Crossover + Gauß-Mutation.

Turnierselektion und Elitismus sind für beide Kodierungen gleich. Population und Individuen sind reine numpy-Arrays,
keine Wrapper-Klassen (Permutation: (pop, n_nodes) int; reellwertig: (pop, dim) float)."""

from dataclasses import dataclass, field

import numpy as np

import ga_constants as C

EPS = 1e-9
MUT_SIGMA = 5.0        # Streuung der Gauß-Mutation (reellwertig), in km - kein eigener Regler, siehe README


# --- Distanz / Tourlänge / CO2 (Vehikel "Lieferroute", wie hill-climbing-demo) ------------------------------------------------------------


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length(tour, D):
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def tour_length_batch(pop, D):
    """Tourlänge jeder Zeile von `pop` (pop, n_nodes)."""
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


def tour_co2_batch(pop, D, co2_factor_matrix):
    """CO2-Kosten jeder Zeile: Kante (i, j) kostet Distanz(i, j) * co2_factor_matrix[i, j] (unabhängig von der Distanz)."""
    nxt = np.roll(pop, -1, axis=1)
    return (D[pop, nxt] * co2_factor_matrix[pop, nxt]).sum(axis=1)


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    """Anteil der Kanten von `tour`, die auch in `reference` vorkommen."""
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


# --- Population: Erzeugen -------------------------------------------------------------------------------------------------------------------


def init_population_perm(pop_size, n_nodes, rng):
    return np.array([rng.permutation(n_nodes) for _ in range(pop_size)], dtype=np.int64)


def init_population_real(pop_size, dim, bounds, rng):
    lo, hi = bounds
    return rng.uniform(lo, hi, size=(pop_size, dim))


# --- Selektion (für beide Kodierungen gleich) ------------------------------------------------------------------------------------------------


def tournament_select(fitness, k, n_select, rng):
    """Indizes von `n_select` Eltern: je `k` zufällige Kandidaten antreten lassen, den mit der niedrigsten Fitness (= Kosten) wählen."""
    if n_select == 0:
        return np.empty(0, dtype=np.int64)
    n = len(fitness)
    contenders = rng.integers(0, n, size=(n_select, k))
    best = np.argmin(fitness[contenders], axis=1)
    return contenders[np.arange(n_select), best]


# --- Crossover --------------------------------------------------------------------------------------------------------------------------------


def order_crossover(p1, p2, rng):
    """Order Crossover (OX): ein Stück aus `p1` wird wörtlich übernommen, der Rest in der Reihenfolge von `p2` aufgefüllt."""
    n = len(p1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[i:j + 1] = p1[i:j + 1]
    taken = set(child[i:j + 1].tolist())
    fill = [g for g in p2.tolist() if g not in taken]
    pos = [k for k in range(n) if not (i <= k <= j)]
    for k, g in zip(pos, fill):
        child[k] = g
    return child


def blx_alpha_crossover(p1, p2, alpha, bounds, rng):
    """BLX-α: jedes Gen des Kindes gleichverteilt aus einem Intervall, das über die Elternwerte hinaus um α der Elterndifferenz erweitert ist."""
    lo, hi = bounds
    lo_g = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
    hi_g = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
    child = rng.uniform(lo_g, hi_g)
    return np.clip(child, lo, hi)


# --- Mutation ---------------------------------------------------------------------------------------------------------------------------------


def swap_mutation(ind, p_mut, rng):
    """Mit Wahrscheinlichkeit `p_mut` werden zwei zufällige Positionen vertauscht."""
    if rng.random() >= p_mut:
        return ind.copy()
    out = ind.copy()
    i, j = rng.integers(0, len(ind), size=2)
    out[i], out[j] = out[j], out[i]
    return out


def gaussian_mutation(ind, p_mut, sigma, bounds, rng):
    """Jedes Gen unabhängig: mit Wahrscheinlichkeit `p_mut` gaußsches Rauschen (Streuung `sigma`) addieren, auf die Grenzen kappen."""
    lo, hi = bounds
    mask = rng.random(len(ind)) < p_mut
    noise = rng.normal(0.0, sigma, size=len(ind))
    out = ind + np.where(mask, noise, 0.0)
    return np.clip(out, lo, hi)


# --- Diversität ---------------------------------------------------------------------------------------------------------------------------


def diversity_perm(pop, sample=40, rng=None):
    """1 - mittlerer Kantenanteil zwischen Paaren der Population (0 = alle Touren identisch, hoch = verschieden). Über eine Zufallsstichprobe von Paaren, wenn die Population groß ist."""
    n = len(pop)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return 0.0
    if rng is not None and len(pairs) > sample:
        idx = rng.choice(len(pairs), size=sample, replace=False)
        pairs = [pairs[k] for k in idx]
    shares = [edge_share(pop[i], pop[j]) for i, j in pairs]
    return float(1.0 - np.mean(shares))


def diversity_real(pop):
    """Mittlerer euklidischer Abstand vom Populationsschwerpunkt."""
    centre = pop.mean(axis=0)
    return float(np.mean(np.sqrt(((pop - centre) ** 2).sum(axis=1))))


# --- GA-Hauptschleife ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Generation:
    population: np.ndarray
    fitness: np.ndarray


@dataclass
class GAResult:
    best_individual: np.ndarray
    best_fitness: float
    best_history: np.ndarray       # (generations + 1,) - bester Wert je Generation (0 = Startpopulation)
    mean_history: np.ndarray
    diversity_history: np.ndarray
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_ga(encoding, fitness_fn, dim_or_n_nodes, bounds, pop_size, generations, cx_prob, mut_prob, elitism, tournament_k, seed, keep_history=False):
    """Ein GA-Lauf. `encoding` bestimmt Erzeuger/Operatoren; `fitness_fn(population) -> (pop_size,)`, niedriger ist besser.
    `dim_or_n_nodes`: Knotenzahl (perm) oder Dimension (real, hier immer 2). `bounds`: bei perm ungenutzt, sonst (lo, hi)."""
    if encoding not in C.ENCODINGS:
        raise ValueError(encoding)
    rng = np.random.default_rng(seed)
    elitism = min(elitism, pop_size)

    if encoding == "perm":
        pop = init_population_perm(pop_size, dim_or_n_nodes, rng)

        def crossover(a, b):
            return order_crossover(a, b, rng)

        def mutate(ind):
            return swap_mutation(ind, mut_prob, rng)

        def diversity(p):
            return diversity_perm(p, rng=rng)
    else:
        pop = init_population_real(pop_size, dim_or_n_nodes, bounds, rng)

        def crossover(a, b):
            return blx_alpha_crossover(a, b, C.BLX_ALPHA, bounds, rng)

        def mutate(ind):
            return gaussian_mutation(ind, mut_prob, MUT_SIGMA, bounds, rng)

        diversity = diversity_real

    fitness = fitness_fn(pop)
    best_hist, mean_hist, div_hist = [float(fitness.min())], [float(fitness.mean())], [diversity(pop)]
    gens = [Generation(pop.copy(), fitness.copy())] if keep_history else []
    best_idx = int(np.argmin(fitness))
    best_ind, best_fit = pop[best_idx].copy(), float(fitness[best_idx])

    for _ in range(generations):
        order = np.argsort(fitness)
        elite = pop[order[:elitism]]
        n_children = pop_size - elitism
        parents_a = tournament_select(fitness, tournament_k, n_children, rng)
        parents_b = tournament_select(fitness, tournament_k, n_children, rng)
        children = []
        for pa, pb in zip(parents_a, parents_b):
            child = crossover(pop[pa], pop[pb]) if rng.random() < cx_prob else pop[pa].copy()
            child = mutate(child)
            children.append(child)
        children_arr = np.array(children, dtype=pop.dtype) if children else np.empty((0,) + pop.shape[1:], dtype=pop.dtype)
        pop = np.concatenate([elite, children_arr], axis=0)
        fitness = fitness_fn(pop)
        best_hist.append(float(fitness.min()))
        mean_hist.append(float(fitness.mean()))
        div_hist.append(diversity(pop))
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_fit:
            best_ind, best_fit = pop[idx].copy(), float(fitness[idx])
        if keep_history:
            gens.append(Generation(pop.copy(), fitness.copy()))

    return GAResult(best_ind, best_fit, np.array(best_hist), np.array(mean_hist), np.array(div_hist), gens)
