"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import ga_constants as C
import ga_evaluation as E
import ga_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert p["encoding"] in C.ENCODINGS
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert C.POP_MIN <= p["pop"] <= C.POP_MAX and C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.CX_MIN <= p["cx"] <= C.CX_MAX and C.MUT_MIN <= p["mut"] <= C.MUT_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    s = E.Settings(encoding=p["encoding"], n=p["n"], cluster_share=p["ballung"], co2_weight=p["co2_weight"], seed=p["seed"],
                   pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], elitism=p["elitism"], k=p["k"], ga_seed=p["ga_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"n_slider", "ballung_slider", "co2_weight_slider", "pop_slider", "gens_slider", "cx_slider", "mut_slider", "elitism_slider", "k_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_kept_widgets_are_exactly_the_ones_hidden_for_continuous_encoding():
    assert set(P.KEPT) == {"n_slider", "ballung_slider", "co2_weight_slider"}
