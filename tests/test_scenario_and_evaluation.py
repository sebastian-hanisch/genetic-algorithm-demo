"""Vehikel (Reproduzierbarkeit, Form) und Auswertung (Referenz, Urteil, Sweeps, Experimente) - schnelle Parameter über monkeypatch/Funktionsargumente."""

import numpy as np
import pytest

import ga_algorithm as A
import ga_constants as C
import ga_evaluation as E
import ga_scenario as S


def test_generate_perm_is_reproducible_and_shaped():
    a = S.generate_perm(20, cluster_share=30, seed=7)
    b = S.generate_perm(20, cluster_share=30, seed=7)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.co2_factor_matrix, b.co2_factor_matrix)
    assert a.xy.shape == (21, 2) and a.co2_factor_matrix.shape == (21, 21) and a.n_nodes == 21
    c = S.generate_perm(20, cluster_share=30, seed=8)
    assert not np.array_equal(a.xy, c.xy)


def test_generate_perm_co2_factor_matrix_is_symmetric_with_zero_diagonal_and_in_range():
    inst = S.generate_perm(50, cluster_share=0, seed=1)
    m = inst.co2_factor_matrix
    assert np.array_equal(m, m.T)
    assert np.all(np.diag(m) == 0.0)
    off_diag = m[~np.eye(len(m), dtype=bool)]
    assert off_diag.min() >= C.CO2_FACTOR_LO and off_diag.max() <= C.CO2_FACTOR_HI


def test_generate_real_is_reproducible_and_shaped():
    a = S.generate_real(seed=3)
    b = S.generate_real(seed=3)
    assert np.array_equal(a.centres, b.centres) and np.array_equal(a.amplitudes, b.amplitudes)
    assert a.centres.shape == (C.K_WELLS, 2) and a.amplitudes.shape == (C.K_WELLS,) and a.sigmas.shape == (C.K_WELLS,)


def test_real_cost_is_lower_at_well_centres_than_far_away():
    inst = S.generate_real(seed=3)
    far_point = np.array([C.AREA + 50.0, C.AREA + 50.0])       # weit außerhalb aller Mulden
    for k in range(C.K_WELLS):
        assert inst.cost(inst.centres[k]) < inst.cost(far_point)


def test_real_cost_is_vectorized_over_a_batch():
    inst = S.generate_real(seed=3)
    batch = np.vstack([inst.centres, np.array([[0.0, 0.0]])])
    costs = inst.cost(batch)
    assert costs.shape == (C.K_WELLS + 1,)
    assert np.array_equal(costs, np.array([inst.cost(p) for p in batch]))


def test_grid_optimum_matches_minimum_of_the_grid_and_lies_near_a_well():
    inst = S.generate_real(seed=3)
    grid_xy, grid_cost = S.grid_optimum(inst, step=2.0)
    xs = np.arange(0.0, C.AREA + 2.0, 2.0)
    gx, gy = np.meshgrid(xs, xs)
    all_costs = inst.cost(np.stack([gx.ravel(), gy.ravel()], axis=-1))
    assert grid_cost == pytest.approx(all_costs.min())
    nearest_well = min(np.sqrt(((inst.centres - grid_xy) ** 2).sum(axis=1)))
    assert nearest_well < 5.0     # das Gitteroptimum liegt nah an (mindestens) einer Mulde


def test_analyse_perm_has_finite_reference_up_to_nine_stops():
    s = E.Settings(encoding="perm", n=5, pop=20, gens=10, ga_seed=1)
    a = E.analyse(s, keep_history=False)
    assert np.isfinite(a.reference)
    fitness, dim, _ = E.fitness_fn(s)
    assert a.reference == pytest.approx(E.brute_force_perm(dim, fitness))


def test_analyse_perm_has_no_reference_above_nine_stops():
    s = E.Settings(encoding="perm", n=10, pop=10, gens=5, ga_seed=1)
    a = E.analyse(s, keep_history=False)
    assert not np.isfinite(a.reference) and not np.isfinite(a.gap)


def test_analyse_real_always_has_a_finite_reference():
    s = E.Settings(encoding="real", pop=10, gens=5, ga_seed=1)
    a = E.analyse(s, keep_history=False)
    assert np.isfinite(a.reference) and np.isfinite(a.gap)


def test_verdict_codes():
    s = E.Settings(encoding="real")

    class _Fake:
        def __init__(self, gap):
            self._gap = gap

        @property
        def gap(self):
            return self._gap
    assert E.verdict(_Fake(1.0)) == "near_optimal"
    assert E.verdict(_Fake(10.0)) == "gap"
    assert E.verdict(_Fake(float("nan"))) == "no_reference"


def test_run_config_and_sweep_smoke(monkeypatch):
    monkeypatch.setattr(C, "SWEEP_SEEDS", (1, 2))
    monkeypatch.setattr(C, "SWEEP_GA_SEEDS", (10, 11))
    rows = E.sweep("pop", base=E.Settings(encoding="real", gens=10), values=(10, 20))
    assert len(rows) == 2
    assert all(np.isfinite(r["gap"]) for r in rows)


def test_convergence_experiment_and_curves_smoke():
    rows = E.convergence_experiment(seed=1, pop_sizes=(5, 15), seeds=(1, 2, 3), gens=15)
    assert len(rows) == 2
    assert all(0.0 <= r["share_global"] <= 1.0 for r in rows)
    curves = E.convergence_curves(seed=1, small=5, large=15, gens=15, ga_seed=1)
    assert set(curves) == {"klein", "groß"}
    assert len(curves["klein"].diversity_history) == 16


def test_pareto_front_is_non_increasing_in_co2_as_distance_grows():
    front, all_points = E.pareto_front_bruteforce(n=6, seed=1)
    assert len(front) >= 1
    assert np.all(np.diff(front[:, 1]) < 0)   # per Konstruktion streng fallend (siehe pareto_front_bruteforce)
    assert len(all_points) == 720              # 6! Touren bei n=6 Stopps (7 Knoten, Depot fest an erster Stelle)


def test_pareto_experiment_smoke():
    report = E.pareto_experiment(n=6, weights=(0.0, 0.5, 1.0), seeds=(1, 2), pop=20, gens=20)
    assert report["front_size"] >= 1
    assert 0 <= report["reached"] <= report["front_size"]
    assert report["found"].shape == (3, 2)


def test_operator_sweep_smoke():
    rows = E.operator_sweep("mut", (0.0, 0.5), base=E.Settings(encoding="real", pop=15, gens=15), seeds=(1, 2))
    assert len(rows) == 2
    assert all(np.isfinite(r["mean_best"]) for r in rows)
