"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio, siehe hc_presets.py
in hill-climbing-demo); zusätzlich der Kodierungs-Umschalter (wie der Aufgaben-Umschalter in cart-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ga_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "encoding_select": SettingSpec("enc", _choice(C.ENCODINGS), "perm"),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "ballung_slider": SettingSpec("ballung", int, C.DEFAULT_BALLUNG, C.BALLUNG_MIN, C.BALLUNG_MAX),
    "co2_weight_slider": SettingSpec("co2w", float, C.DEFAULT_CO2_WEIGHT, C.CO2_WEIGHT_MIN, C.CO2_WEIGHT_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "pop_slider": SettingSpec("pop", int, C.DEFAULT_POP, C.POP_MIN, C.POP_MAX),
    "gens_slider": SettingSpec("gens", int, C.DEFAULT_GEN, C.GEN_MIN, C.GEN_MAX),
    "cx_slider": SettingSpec("cx", float, C.DEFAULT_CX, C.CX_MIN, C.CX_MAX),
    "mut_slider": SettingSpec("mut", float, C.DEFAULT_MUT, C.MUT_MIN, C.MUT_MAX),
    "elitism_slider": SettingSpec("elite", int, C.DEFAULT_ELITE, C.ELITE_MIN, C.ELITE_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "ga_seed_input": SettingSpec("gseed", int, C.DEFAULT_GA_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "encoding": "encoding_select", "n": "n_slider", "ballung": "ballung_slider", "co2_weight": "co2_weight_slider", "seed": "seed_input",
    "pop": "pop_slider", "gens": "gens_slider", "cx": "cx_slider", "mut": "mut_slider", "elitism": "elitism_slider", "k": "k_slider", "ga_seed": "ga_seed_input",
}
# Regler, die bei kontinuierlicher Kodierung ausgeblendet sind (sie betreffen nur die Lieferroute): Streamlit löscht ihren
# Zustand, sobald sie nicht gezeichnet werden - der zuletzt gewählte Wert bleibt hier erhalten
KEPT = {"n_slider": "_kept_n_slider", "ballung_slider": "_kept_ballung_slider", "co2_weight_slider": "_kept_co2_weight_slider"}
STEPS = {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP, "co2_weight_slider": C.CO2_WEIGHT_STEP, "pop_slider": C.POP_STEP,
         "gens_slider": C.GEN_STEP, "cx_slider": C.CX_STEP, "mut_slider": C.MUT_STEP, "elitism_slider": C.ELITE_STEP, "k_slider": C.K_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in KEPT and state_key not in st.session_state:       # ausblendbare Regler: siehe seed_widget
            st.session_state[state_key] = spec.default


def seed_widget(state_key):
    """Vor dem Zeichnen eines ausblendbaren Reglers: fehlt sein Zustand, kommt der zuletzt gewählte (oder der Standard-) Wert.
    Ein Wert, der in einem Lauf ohne den Regler in den Zustand des Reglers geschrieben wird, erscheint später als Mindestwert im Regler, während die App mit dem geschriebenen Wert rechnet."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def stash_kept_widget_state():
    """Permalink und Preset legen den Wert eines ausblendbaren Reglers nur in KEPT ab (der Regler holt ihn sich mit `seed_widget`, sobald er gezeichnet wird)."""
    for state_key, kept in KEPT.items():
        if state_key in st.session_state:
            st.session_state[kept] = st.session_state.pop(state_key)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = lo + round((st.session_state[key] - lo) / step) * step
            if isinstance(SETTING_SPECS[key].default, int):
                st.session_state[key] = int(st.session_state[key])
    stash_kept_widget_state()
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]
    stash_kept_widget_state()


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_ga_seed():
    st.session_state["ga_seed_input"] = random.randint(0, C.SEED_MAX)
