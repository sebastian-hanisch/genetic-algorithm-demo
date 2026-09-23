"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test."""

import numpy as np
import pytest

import ga_algorithm as A
import ga_constants as C
import ga_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(encoding=p["encoding"], n=p["n"], cluster_share=p["ballung"], co2_weight=p["co2_weight"], seed=p["seed"],
                   pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], elitism=p["elitism"], k=p["k"], ga_seed=p["ga_seed"])
    return E.analyse(s, keep_history=False)


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall (Voreinstellung)")
    r = a.result
    assert r.best_history[0] == pytest.approx(1434.4, abs=0.5)
    assert r.best_fitness == pytest.approx(733.0, abs=0.5)
    improve = 100.0 * (r.best_history[0] - r.best_fitness) / r.best_history[0]
    assert improve == pytest.approx(48.9, abs=0.2)
    assert r.diversity_history[0] == pytest.approx(0.93, abs=0.02)
    assert r.diversity_history[-1] == pytest.approx(0.06, abs=0.02)


def test_kleine_population_preset_claims():
    a = _preset_analysis("Kleine Population (vorzeitige Konvergenz)")
    r = a.result
    assert a.gap == pytest.approx(16.6, abs=0.3)
    improve = 100.0 * (r.best_history[0] - r.best_fitness) / r.best_history[0]
    assert improve == pytest.approx(1.5, abs=0.3)
    assert r.diversity_history[0] == pytest.approx(31.9, abs=0.5)
    assert r.diversity_history[-1] == pytest.approx(1.3, abs=0.2)


def test_grosse_population_preset_claims():
    a = _preset_analysis("Große Population")
    assert a.result.best_fitness == pytest.approx(614, abs=1)


def test_hohe_mutationsrate_preset_claims():
    a = _preset_analysis("Hohe Mutationsrate")
    assert a.result.best_fitness == pytest.approx(621, abs=1)
    assert a.result.diversity_history[-1] == pytest.approx(0.44, abs=0.02)


def test_kaum_mutation_preset_claims():
    a = _preset_analysis("Kaum Mutation")
    assert a.result.diversity_history[-1] == pytest.approx(0.01, abs=0.01)


def test_standortwahl_preset_claims():
    a = _preset_analysis("Standortwahl (kontinuierlich)")
    assert a.gap == pytest.approx(-0.02, abs=0.1)


def test_distanz_und_co2_preset_claims():
    p = C.PRESETS["Distanz und CO2 gewichtet"]
    inst, D = E.perm_instance(p["n"], p["ballung"], p["seed"])

    def tour_metrics(co2_weight):
        s = E.Settings(encoding="perm", n=p["n"], cluster_share=p["ballung"], co2_weight=co2_weight, seed=p["seed"],
                       pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], elitism=p["elitism"], k=p["k"], ga_seed=p["ga_seed"])
        r = E.run(s, keep_history=False)
        tour = r.best_individual[None, :]
        dist = float(A.tour_length_batch(tour, D)[0])
        co2 = float(A.tour_co2_batch(tour, D, inst.co2_factor_matrix)[0])
        return dist, co2

    d0, c0 = tour_metrics(0.0)
    d1, c1 = tour_metrics(p["co2_weight"])
    co2_reduction = 100.0 * (c0 - c1) / c0
    assert co2_reduction == pytest.approx(24, abs=1)
    assert abs(d1 - d0) / d0 < 0.05      # Distanz bleibt bei dieser Instanz nahezu gleich


def test_grosse_instanz_preset_claims():
    a = _preset_analysis("Große Instanz")
    r = a.result
    assert r.best_history[0] == pytest.approx(3817, abs=1)
    assert r.best_fitness == pytest.approx(1639, abs=1)
    improve = 100.0 * (r.best_history[0] - r.best_fitness) / r.best_history[0]
    assert improve == pytest.approx(57.1, abs=0.2)


def test_convergence_experiment_headline_claims():
    rows = {r["pop"]: r for r in E.convergence_experiment()}
    assert rows[10]["share_global"] == pytest.approx(0.55, abs=0.05)
    assert rows[100]["share_global"] == pytest.approx(0.95, abs=0.05)


def test_pareto_experiment_headline_claims():
    report = E.pareto_experiment()
    assert report["front_size"] == 8
    assert report["reached"] == 4


def test_operator_sensitivity_headline_claims():
    base = E.Settings(encoding="real", pop=C.OPERATOR_POP, gens=C.OPERATOR_GENS, elitism=C.DEFAULT_ELITE, k=C.DEFAULT_K)
    rows = {r["value"]: r["mean_best"] for r in E.operator_sweep("mut", C.OPERATOR_MUT_VALUES, base=base)}
    # Mutationsrate 0 ist klar am schlechtesten; jede positive Rate liegt deutlich darunter
    assert rows[0.0] > max(rows[v] for v in C.OPERATOR_MUT_VALUES if v > 0.0) + 1.0
    cx_rows = {r["value"]: r["mean_best"] for r in E.operator_sweep("cx", C.OPERATOR_CX_VALUES, base=base)}
    assert cx_rows[0.0] > cx_rows[1.0]


def test_pareto_front_size_is_not_a_fluke_of_the_exact_percentile_choice():
    # Grobe Gegenprobe zur Kernaussage "eine feste Gewichtung trifft nur einen Teil der Front":
    # bei n=8 Stopps hat die Brute-Force-Front unabhängig vom genauen CO2-Seed typischerweise mehrere Punkte.
    sizes = [len(E.pareto_front_bruteforce(n=8, seed=s)[0]) for s in range(5)]
    assert all(s >= 1 for s in sizes)
    assert np.mean(sizes) >= 2.0
