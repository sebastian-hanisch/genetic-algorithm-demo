"""Kern gegen Handrechnung + Invarianten: OX bleibt immer eine gültige Permutation, BLX/Gauß bleiben in den Grenzen,
Turnierselektion bevorzugt niedrigere Fitness, der GA findet auf sehr kleinen Instanzen nachweislich das Optimum."""

from itertools import permutations

import numpy as np
import pytest

import ga_algorithm as A


class _FixedRNG:
    """Duck-typed RNG, die order_crossover feste Schnittpunkte liefert (für die Handrechnung)."""

    def __init__(self, cuts):
        self.cuts = cuts

    def integers(self, lo, hi, size=None):
        return np.array(self.cuts)


def test_order_crossover_matches_hand_calculation():
    # Klassisches OX-Lehrbuchbeispiel (Goldberg): Schnitt bei Positionen 3..6 (0-indexiert, inklusive)
    p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p2 = np.array([5, 4, 6, 9, 2, 1, 7, 8, 3])
    child = A.order_crossover(p1, p2, _FixedRNG([3, 6]))
    assert child.tolist() == [9, 2, 1, 4, 5, 6, 7, 8, 3]


@pytest.mark.parametrize("seed", range(30))
def test_order_crossover_always_produces_valid_permutation(seed):
    rng = np.random.default_rng(seed)
    n = 12
    p1, p2 = rng.permutation(n), rng.permutation(n)
    child = A.order_crossover(p1, p2, rng)
    assert sorted(child.tolist()) == list(range(n))


@pytest.mark.parametrize("seed", range(20))
def test_blx_alpha_crossover_stays_within_bounds(seed):
    rng = np.random.default_rng(seed)
    bounds = (0.0, 100.0)
    p1 = rng.uniform(*bounds, size=2)
    p2 = rng.uniform(*bounds, size=2)
    for _ in range(20):
        child = A.blx_alpha_crossover(p1, p2, 0.5, bounds, rng)
        assert np.all(child >= bounds[0]) and np.all(child <= bounds[1])


@pytest.mark.parametrize("seed", range(20))
def test_gaussian_mutation_stays_within_bounds(seed):
    rng = np.random.default_rng(seed)
    bounds = (0.0, 100.0)
    ind = rng.uniform(*bounds, size=2)
    for _ in range(20):
        out = A.gaussian_mutation(ind, 1.0, sigma=50.0, bounds=bounds, rng=rng)
        assert np.all(out >= bounds[0]) and np.all(out <= bounds[1])


def test_swap_mutation_probability_zero_never_changes():
    rng = np.random.default_rng(0)
    ind = np.arange(10)
    for _ in range(20):
        assert np.array_equal(A.swap_mutation(ind, 0.0, rng), ind)


def test_swap_mutation_probability_one_usually_changes():
    rng = np.random.default_rng(0)
    ind = np.arange(10)
    changed = sum(not np.array_equal(A.swap_mutation(ind, 1.0, rng), ind) for _ in range(50))
    assert changed > 40           # nur unverändert, wenn zufällig i == j gezogen wird (1 von 10 Fällen)


def test_tournament_select_prefers_lower_fitness():
    rng = np.random.default_rng(0)
    fitness = np.array([0.0, 10.0, 10.0, 10.0, 10.0])   # Index 0 ist klar am besten
    selected = A.tournament_select(fitness, k=4, n_select=200, rng=rng)
    assert (selected == 0).mean() > 0.5                  # gewinnt weit öfter als 1/5 (reiner Zufall)


def test_tournament_select_empty_selection():
    assert len(A.tournament_select(np.array([1.0, 2.0]), k=2, n_select=0, rng=np.random.default_rng(0))) == 0


def test_diversity_perm_zero_for_identical_population():
    pop = np.tile(np.arange(6), (5, 1))
    assert A.diversity_perm(pop) == pytest.approx(0.0)


def test_diversity_real_zero_for_identical_population():
    pop = np.tile(np.array([3.0, 4.0]), (5, 1))
    assert A.diversity_real(pop) == pytest.approx(0.0)


def test_tour_length_batch_matches_manual_computation():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = A.dist_matrix(xy)
    pop = np.array([[0, 1, 2]])
    # 0->1: 3, 1->2: 4, 2->0: 5 (3-4-5-Dreieck)
    assert A.tour_length_batch(pop, D)[0] == pytest.approx(12.0)


def test_tour_co2_batch_matches_manual_computation():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = A.dist_matrix(xy)
    # CO2-Faktor je Kante, unabhängig von der Distanz (symmetrisch, Diagonale ungenutzt)
    factor = np.array([[0.0, 2.0, 0.5], [2.0, 0.0, 1.5], [0.5, 1.5, 0.0]])
    pop = np.array([[0, 1, 2]])
    # Kante (0,1): 3 * 2.0 = 6.0; (1,2): 4 * 1.5 = 6.0; (2,0): 5 * 0.5 = 2.5
    assert A.tour_co2_batch(pop, D, factor)[0] == pytest.approx(14.5)


def test_best_fitness_equals_minimum_of_best_history():
    def fitness(pop):
        xy = pop
        return ((xy - np.array([50.0, 50.0])) ** 2).sum(axis=1)
    r = A.run_ga("real", fitness, 2, (0.0, 100.0), pop_size=20, generations=15, cx_prob=0.9, mut_prob=0.2, elitism=1, tournament_k=3, seed=1)
    assert r.best_fitness == pytest.approx(r.best_history.min())


def test_elitism_makes_generation_best_monotonic():
    def fitness(pop):
        return ((pop - np.array([50.0, 50.0])) ** 2).sum(axis=1)
    r = A.run_ga("real", fitness, 2, (0.0, 100.0), pop_size=20, generations=30, cx_prob=0.9, mut_prob=0.3, elitism=2, tournament_k=3, seed=2)
    assert np.all(np.diff(r.best_history) <= 1e-9)


def _brute_force_tsp(xy):
    D = A.dist_matrix(xy)
    n = len(xy)
    best = None
    for perm in permutations(range(1, n)):
        length = A.tour_length((0,) + perm, D)
        if best is None or length < best:
            best = length
    return best


def test_ga_perm_finds_brute_force_optimum_on_tiny_instance_in_most_seeds():
    # Stochastische Suche: wie bei jeder Metaheuristik-Kreuzprobe in diesem Portfolio wird die MEHRHEIT
    # der Läufe geprüft, nicht jeder einzelne Seed (siehe feedback_bootstrap_duplicate_rows_raise_tie_rate).
    rng = np.random.default_rng(900)
    xy = rng.random((6, 2)) * 100.0
    D = A.dist_matrix(xy)
    optimum = _brute_force_tsp(xy)

    def fitness(pop):
        return A.tour_length_batch(pop, D)
    hits = 0
    for seed in range(20):
        r = A.run_ga("perm", fitness, len(xy), (0, 0), pop_size=40, generations=80, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, seed=seed)
        hits += r.best_fitness == pytest.approx(optimum, rel=1e-6)
    assert hits >= 18


def test_ga_real_converges_near_single_well():
    inst_cost_centre = np.array([60.0, 40.0])

    def fitness(pop):
        return ((pop - inst_cost_centre) ** 2).sum(axis=1)
    r = A.run_ga("real", fitness, 2, (0.0, 100.0), pop_size=60, generations=150, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, seed=5)
    assert np.sqrt(((r.best_individual - inst_cost_centre) ** 2).sum()) < 1.0
