"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Kodierungs-Umschalter (versteckt/behält Regler), Generation-Slider,
Abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ga_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_metrics():
    at = _run()
    _ok(at)
    assert at.metric


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["encoding_select"] == p["encoding"] and at.session_state["pop_slider"] == p["pop"]
    assert at.metric


def test_encoding_toggle_hides_and_keeps_discrete_controls():
    at = _run(n_slider=45)
    assert any(s.key == "n_slider" for s in at.slider)
    at.selectbox(key="encoding_select").set_value("real").run()
    _ok(at)
    assert not any(s.key == "n_slider" for s in at.slider)
    assert not any(s.key == "ballung_slider" for s in at.slider)
    assert not any(s.key == "co2_weight_slider" for s in at.slider)
    at.selectbox(key="encoding_select").set_value("perm").run()
    _ok(at)
    assert next(s for s in at.slider if s.key == "n_slider").value == 45


@pytest.mark.parametrize("encoding", ["perm", "real"])
def test_generation_slider_runs_at_various_positions(encoding):
    at = _run(encoding_select=encoding, gens_slider=20)
    _ok(at)
    gen_slider = next(s for s in at.slider if s.key == "ga_gen")
    assert gen_slider.value == 20
    gen_slider.set_value(0).run()
    _ok(at)
    assert at.get("plotly_chart")
    gen_slider.set_value(10).run()
    _ok(at)
    assert at.get("plotly_chart")


def test_play_runs_without_duplicate_keys():
    at = _run(gens_slider=20, pop_slider=20)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_dice_buttons_change_the_seeds():
    at = _run()
    old_seed = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed
    old_ga_seed = at.session_state["ga_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuen GA-Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["ga_seed_input"] != old_ga_seed


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["pop"] = "17"
    at.query_params["enc"] = "nonsense"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["pop_slider"] == 20
    assert at.session_state["encoding_select"] == "perm"


@pytest.mark.parametrize("kw", [dict(encoding_select="real"), dict(n_slider=10, ballung_slider=100), dict(pop_slider=C.POP_MIN, gens_slider=C.GEN_MIN), dict(mut_slider=1.0, cx_slider=0.0), dict(co2_weight_slider=1.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_sweep_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "SWEEP_SEEDS", (1, 2))
    monkeypatch.setattr(C, "SWEEP_GA_SEEDS", (10,))
    at = _run(gens_slider=15, pop_slider=15)
    at.selectbox(key="sweep_select").set_value("gens").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_convergence_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "CONVERGENCE_POP_SIZES", (5, 10))
    monkeypatch.setattr(C, "CONVERGENCE_SEEDS", (1, 2))
    at = _run()
    next(b for b in at.button if b.key == "convergence_start").click().run()
    _ok(at)
    assert at.session_state["convergence_on"] and at.get("plotly_chart")


def test_pareto_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "PARETO_N", 6)
    monkeypatch.setattr(C, "PARETO_WEIGHTS", (0.0, 0.5, 1.0))
    monkeypatch.setattr(C, "PARETO_SEEDS", (1, 2))
    at = _run()
    next(b for b in at.button if b.key == "pareto_start").click().run()
    _ok(at)
    assert at.session_state["pareto_on"] and at.get("plotly_chart")


def test_operator_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "OPERATOR_SEEDS", (1, 2))
    at = _run(gens_slider=15, pop_slider=15)
    next(b for b in at.button if b.key == "operator_start").click().run()
    _ok(at)
    assert at.session_state["operator_on"] and at.get("plotly_chart")


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("vielfältig genug" in m.value for m in at.markdown)
