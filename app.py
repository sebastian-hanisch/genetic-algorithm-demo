"""Genetischer Algorithmus - eine Population statt einer einzelnen Lösung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren -
den Genetischen Algorithmus (GA) - an einem wachsenden Beispiel. Wurzelstück der Populations-Metaheuristiken-Linie der
"Konzepte"-Reihe: statt einer einzelnen Lösung schrittweise zu verbessern (wie die Trajektorien-Linie), hält der GA eine
ganze Population von Lösungen und verbessert sie über Selektion, Crossover und Mutation. Zwei Vehikel zeigen dieselbe
Idee auf zwei Kodierungen: eine Lieferroute (Permutation) und eine Standortwahl (reellwertig). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ga_algorithm as A
import ga_constants as C
from ga_evaluation import Settings, analyse, convergence_curves, convergence_experiment, operator_sweep, pareto_experiment, perm_instance, real_instance, sweep, verdict
from ga_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_ga_seed, randomize_seed, sync_query_params
from ga_visualization import build_convergence, build_convergence_curves, build_diversity_curve, build_fitness_curve, build_landscape, build_operator_sweep, build_pareto, build_route, build_sweep

st.set_page_config(page_title="Genetischer Algorithmus – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _convergence():
    return convergence_experiment()


@st.cache_data(show_spinner=False)
def _convergence_curves():
    return convergence_curves()


@st.cache_data(show_spinner=False)
def _pareto():
    return pareto_experiment()


@st.cache_data(show_spinner=False)
def _operator_sweep(param, base, values):
    return operator_sweep(param, values, base=base)


st.title("🧬 Genetischer Algorithmus – eine Population statt einer einzelnen Lösung")
st.markdown(
    """
Wie durchsucht man einen riesigen Lösungsraum, ohne in der ersten Sackgasse stecken zu bleiben? Der **Genetische Algorithmus (GA)** hält
statt einer einzelnen Lösung eine ganze **Population**: die besten Mitglieder werden zu Eltern (**Selektion**), ihre Lösungen gemischt
(**Crossover**) und leicht verändert (**Mutation**) - über viele **Generationen** verbessert sich die Population als Ganzes, auch wenn
einzelne Nachkommen schlechter sind als ihre Eltern. Diese Demo zeigt denselben GA auf zwei Aufgaben: eine **Lieferroute** (diskret,
Reihenfolge der Stopps) und eine **Standortwahl** (kontinuierlich, Lage eines Verteilzentrums).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - Wurzelstück der "
    "Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. Die direkten Nachfolger "
    "(NSGA-II, CMA-ES, Differential Evolution, Partikelschwarm-Optimierung, Ameisenalgorithmus) setzen an je einer eigenen Schwäche oder "
    "einem eigenen Kontrast zu dieser Wurzel an; mit dem Hill Climbing der Trajektorien-Linie ergäbe sich später ein Memetischer Algorithmus."
)

with st.expander("So funktioniert der Genetische Algorithmus", expanded=True):
    st.markdown(
        """
1. **Population.** Statt einer Lösung: mehrere hundert, zufällig erzeugt - bei der Lieferroute zufällige Reihenfolgen, bei der Standortwahl
   zufällige Punkte im Gebiet.
2. **Fitness.** Jedes Mitglied wird bewertet (Tourlänge bzw. Standortkosten) - niedriger ist besser.
3. **Selektion.** **Turnierselektion**: für jeden benötigten Elternteil treten `k` zufällige Mitglieder gegeneinander an, das beste gewinnt.
   Bessere Mitglieder werden dadurch öfter Eltern, ohne die schlechtesten ganz auszuschließen.
4. **Crossover.** Zwei Eltern erzeugen ein Kind, das Merkmale beider mischt: bei der Route ein zusammenhängendes Stück der einen Route,
   aufgefüllt in der Reihenfolge der anderen (**Order Crossover**); bei der Standortwahl ein Punkt zwischen (und etwas über) den beiden
   Elternpunkten (**BLX-α-Crossover**).
5. **Mutation.** Mit kleiner Wahrscheinlichkeit eine zufällige Änderung - bei der Route der Tausch zweier Stopps, bei der Standortwahl ein
   kleiner zufälliger Sprung - damit der GA nicht auf die anfängliche Population beschränkt bleibt.
6. **Elitismus.** Die besten Mitglieder überleben unverändert in die nächste Generation, damit ein guter Fund nicht durch Zufall verloren geht.
7. **Wiederholen**, bis die eingestellte Zahl an Generationen erreicht ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    encoding = st.selectbox(
        "Kodierung", C.ENCODINGS, key="encoding_select", format_func=lambda k: C.ENCODING_LABELS[k],
        help="Diskret: der GA sucht eine Reihenfolge der Lieferstopps (Permutation). Kontinuierlich: der GA sucht eine Lage (x, y) für ein "
             "Verteilzentrum in einer Kostenlandschaft mit mehreren lokalen Minima. Selektion und Elitismus sind identisch; Crossover und "
             "Mutation sind an die Kodierung angepasst (Order Crossover + Tausch gegen BLX-α + Gauß-Rauschen).",
    )
    if encoding == "perm":
        n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
        cluster_share = st.slider("Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
                                   help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.")
        co2_weight = st.slider("Gewicht CO2 (gewichtete Summe)", *bounds("co2_weight_slider"), key="co2_weight_slider", step=C.CO2_WEIGHT_STEP, format="%.2f",
                                help="0 = nur Distanz zählt. >0 = Distanz und CO2-Kosten werden gewichtet addiert (jeder Straßenabschnitt hat einen eigenen, von der Distanz "
                                     "unabhängigen CO2-Faktor - manche Verbindungen sind stauanfälliger als andere). Wie gut eine feste Gewichtung die Pareto-Front trifft, zeigt das Experiment weiter unten.")
        st.session_state["_kept_n_slider"] = n_stops
        st.session_state["_kept_ballung_slider"] = cluster_share
        st.session_state["_kept_co2_weight_slider"] = co2_weight
    else:
        n_stops = int(st.session_state.get("_kept_n_slider", C.DEFAULT_N))
        cluster_share = int(st.session_state.get("_kept_ballung_slider", C.DEFAULT_BALLUNG))
        co2_weight = float(st.session_state.get("_kept_co2_weight_slider", C.DEFAULT_CO2_WEIGHT))
        st.caption("Die Standortwahl hat eine feste Kostenlandschaft (mehrere Nachfrageschwerpunkte, per Seed bestimmt) - Stopps, Gruppen und CO2-Gewicht gelten nur für die Lieferroute.")
    st.markdown("**Genetischer Algorithmus**")
    pop_size = st.slider("Populationsgröße", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP, help="Zahl der Individuen je Generation. Kleine Populationen konvergieren schneller, aber oft zu einem lokalen statt globalen Optimum (siehe Experiment „Vorzeitige Konvergenz“).")
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP, help="Wie viele Runden aus Selektion, Crossover und Mutation gerechnet werden.")
    cx_prob = st.slider("Crossover-Rate", *bounds("cx_slider"), key="cx_slider", step=C.CX_STEP, format="%.2f", help="Wahrscheinlichkeit, dass ein Kind aus zwei Eltern gemischt wird (statt eine Kopie eines Elternteils zu sein, die dann noch mutieren kann).")
    mut_prob = st.slider("Mutationsrate", *bounds("mut_slider"), key="mut_slider", step=C.MUT_STEP, format="%.2f", help="Wahrscheinlichkeit einer zufälligen Änderung je Kind (Route) bzw. je Koordinate (Standort).")
    elitism = st.slider("Elitismus", *bounds("elitism_slider"), key="elitism_slider", step=C.ELITE_STEP, help="Wie viele der besten Mitglieder unverändert in die nächste Generation übernommen werden.")
    tournament_k = st.slider("Turniergröße k", *bounds("k_slider"), key="k_slider", step=C.K_STEP, help="Wie viele Kandidaten je Turnier antreten. Größeres k erhöht den Selektionsdruck (bessere Mitglieder gewinnen öfter, die Population verliert schneller an Vielfalt).")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1, help="Legt Stopps/Kostenlandschaft fest.")
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    ga_seed = st.number_input("Zufalls-Seed des GA-Laufs", *bounds("ga_seed_input"), key="ga_seed_input", step=1, help="Legt Startpopulation, Selektion, Crossover und Mutation fest.")
    st.button("🎲 Neuen GA-Lauf würfeln", width="stretch", on_click=randomize_ga_seed)

sync_query_params({
    "encoding_select": encoding, "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "co2_weight_slider": float(co2_weight), "seed_input": int(seed),
    "pop_slider": int(pop_size), "gens_slider": int(generations), "cx_slider": float(cx_prob), "mut_slider": float(mut_prob), "elitism_slider": int(elitism), "k_slider": int(tournament_k), "ga_seed_input": int(ga_seed),
})

settings = Settings(encoding, int(n_stops), int(cluster_share), float(co2_weight), int(seed), int(pop_size), int(generations), float(cx_prob), float(mut_prob), int(elitism), int(tournament_k), int(ga_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

if settings.encoding == "real":
    inst_real, grid_xy, grid_cost = real_instance(settings.seed)
else:
    inst_perm, D = perm_instance(settings.n, settings.cluster_share, settings.seed)

# --- Der GA in Aktion -------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Genetische Algorithmus in Aktion")
if "ga_gen" not in st.session_state or st.session_state.get("ga_gen_owner") != data_key:
    st.session_state["ga_gen"] = n_gens_run
    st.session_state["ga_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="ga_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    best_idx = int(np.argmin(gd.fitness))
    with view_slot.container():
        c1, c2 = st.columns(2)
        if settings.encoding == "perm":
            c1.markdown(f"**Generation {g} von {n_gens_run} – bester Wert dieser Generation: {gd.fitness[best_idx]:,.1f}**".replace(",", "."))
            c1.plotly_chart(build_route(inst_perm.xy, gd.population[best_idx], inst_perm.co2_factor_matrix), width="stretch", key=f"g_map_{g}")
        else:
            c1.markdown(f"**Generation {g} von {n_gens_run} – bester Wert dieser Generation: {gd.fitness[best_idx]:,.1f}**".replace(",", "."))
            c1.plotly_chart(build_landscape(inst_real, population=gd.population, best_xy=gd.population[best_idx], grid_xy=grid_xy), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Bester und mittlerer Wert je Generation**")
        c2.plotly_chart(build_fitness_curve(result.best_history[:g + 1], result.mean_history[:g + 1], reference=a.reference if np.isfinite(a.reference) else None), width="stretch", key=f"g_curve_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der GA gefunden hat")
code = verdict(a)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bester Wert", f"{result.best_fitness:,.1f}".replace(",", "."), delta=f"Start {result.best_history[0]:,.1f}".replace(",", "."), delta_color="off", help="Kosten der besten je gefundenen Lösung; im Delta der beste Wert der Startpopulation.")
if np.isfinite(a.gap):
    m2.metric("Abstand zur Referenz", f"{a.gap:.1f} %", help="Abstand zum Optimum: Brute-Force bei der Lieferroute (nur bis 9 Stopps möglich), Gitter-Suche bei der Standortwahl.")
else:
    m2.metric("Abstand zur Referenz", "keine (> 9 Stopps)", help="Eine Brute-Force-Referenz ist bei mehr als 9 Stopps nicht mehr in vertretbarer Zeit berechenbar (8! = 40 320 Touren, 30! ist astronomisch) - siehe die Exakte-Suche-Linie für Verfahren, die trotzdem eine Schranke berechnen.")
m3.metric("Diversität am Ende", f"{result.diversity_history[-1]:.2f}", delta=f"Start {result.diversity_history[0]:.2f}", delta_color="off", help="Wie unterschiedlich die Population am Ende noch ist (0 = alle identisch). Sinkt sie schon lange vor der letzten Generation auf nahe 0, war die Suche vorzeitig konvergiert.")
m4.metric("Generationen × Population", f"{settings.gens} × {settings.pop}", help="Gesamtzahl bewerteter Individuen: Generationen × Populationsgröße (plus die Startpopulation).")

if code == "near_optimal":
    st.success(f"✅ Nahe an der Referenz: nur {a.gap:.1f} % darüber.")
elif code == "gap":
    st.warning(f"⚠️ {a.gap:.1f} % über der Referenz – die Population ist noch nicht (oder nicht mehr, bei vorzeitiger Konvergenz) nah am Optimum.")
else:
    st.info("ℹ️ Keine Referenz bei dieser Stoppzahl – Genauigkeit vs. der wahren optimalen Route lässt sich hier nicht direkt messen.")

st.markdown("---")

# --- Sweeps -------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von den GA-Reglern ab?")
st.caption("Gemessen auf der kontinuierlichen Standortwahl (dort gibt es immer eine günstige Referenz), Mittel über 5 feste Vehikel-Seeds × 3 GA-Seeds.")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(encoding="real", pop=settings.pop, gens=settings.gens, cx=settings.cx, mut=settings.mut, elitism=settings.elitism, k=settings.k)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: vorzeitige Konvergenz --------------------------------------------------------------------------------------------------

st.subheader("🔬 Vorzeitige Konvergenz: verliert eine kleine Population zu früh an Vielfalt?")
if st.button("Populationsgrößen 10 bis 100 über je 20 GA-Seeds vergleichen (dauert etwa 20 Sekunden)", key="convergence_start"):
    st.session_state["convergence_on"] = True
if st.session_state.get("convergence_on"):
    with st.spinner("Rechne 4 Populationsgrößen × 20 GA-Seeds..."):
        rows_conv = _convergence()
        curves_conv = _convergence_curves()
    st.plotly_chart(build_convergence(rows_conv), width="stretch", key="convergence_chart")
    st.plotly_chart(build_convergence_curves(curves_conv), width="stretch", key="convergence_curves_chart")
    small, large = rows_conv[0], rows_conv[-1]
    q1, q2 = st.columns(2)
    q1.metric(f"Im globalen Trichter (Population {small['pop']})", f"{small['share_global']:.0%}", help="Anteil von 20 GA-Läufen, deren bester Fund höchstens 3 km vom global günstigsten Punkt entfernt liegt.")
    q2.metric(f"Im globalen Trichter (Population {large['pop']})", f"{large['share_global']:.0%}")
    st.caption("Auf der Standortwahl (mehrere unterschiedlich tiefe Trichter): eine kleine Population verliert ihre Vielfalt oft, bevor sie den tiefsten Trichter erreicht hat, und bleibt in einem flacheren lokalen Minimum hängen - "
               "eine größere Population erkundet mehr Trichter gleichzeitig, bevor sie sich festlegt. Das ist genau die Schwäche, an der CMA-ES und Differential Evolution (beide für kontinuierliche Landschaften) ansetzen.")

st.markdown("---")

# --- Experiment 2: gewichtete Summe verfehlt die Pareto-Front -----------------------------------------------------------------------------

st.subheader("🔬 Gewichtete Summe: trifft eine feste Gewichtung die Pareto-Front?")
if st.button(f"Pareto-Front (Brute-Force, {C.PARETO_N} Stopps) gegen gewichtete Summe rechnen (dauert etwa 20 Sekunden)", key="pareto_start"):
    st.session_state["pareto_on"] = True
if st.session_state.get("pareto_on"):
    with st.spinner("Rechne die Brute-Force-Pareto-Front und den GA für mehrere Gewichte..."):
        report = _pareto()
    st.plotly_chart(build_pareto(report), width="stretch", key="pareto_chart")
    p1, p2 = st.columns(2)
    p1.metric("Punkte auf der Pareto-Front", f"{report['front_size']}")
    p2.metric("Davon durch eine feste Gewichtung getroffen", f"{report['reached']} von {report['front_size']}", help=f"Getroffen über {len(C.PARETO_WEIGHTS)} verschiedene Gewichte von Distanz und CO2 (je {len(C.PARETO_SEEDS)} GA-Seeds, bester genommen).")
    st.caption("Eine gewichtete Summe kann nur Punkte auf dem **konvexen** Teil der Pareto-Front finden - bei einer kombinatorischen Front wie hier (Distanz gegen CO2 über alle Rundtouren) ist ein großer Teil der Front nicht konvex "
               "und bleibt für jede feste Gewichtung unerreichbar, egal wie fein man das Gewicht durchfährt. Genau diese Lücke motiviert **NSGA-II**: statt eine feste Gewichtung zu optimieren, sortiert es die Population direkt nach Dominanz.")

st.markdown("---")

# --- Experiment 3: Operator-Sensitivität ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Operator-Sensitivität: wie stark hängt das Ergebnis von Crossover- und Mutationsrate ab?")
st.caption(f"Eigenes, knapp bemessenes Budget (Population {C.OPERATOR_POP}, {C.OPERATOR_GENS} Generationen, unabhängig von der Seitenleiste) - erst wenn der GA nicht ohnehin genug Zeit zum Konvergieren hat, zeigt sich der Unterschied deutlich.")
if st.button("Crossover- und Mutationsrate durchfahren (dauert etwa 15 Sekunden)", key="operator_start"):
    st.session_state["operator_on"] = True
if st.session_state.get("operator_on"):
    base_op = Settings(encoding="real", pop=C.OPERATOR_POP, gens=C.OPERATOR_GENS, elitism=C.DEFAULT_ELITE, k=C.DEFAULT_K)
    with st.spinner(f"Rechne {len(C.OPERATOR_MUT_VALUES)} Mutations- und {len(C.OPERATOR_CX_VALUES)} Crossover-Raten × {len(C.OPERATOR_SEEDS)} GA-Seeds..."):
        rows_mut = _operator_sweep("mut", base_op, C.OPERATOR_MUT_VALUES)
        rows_cx = _operator_sweep("cx", base_op, C.OPERATOR_CX_VALUES)
    oc1, oc2 = st.columns(2)
    oc1.plotly_chart(build_operator_sweep(rows_mut, "Mutationsrate"), width="stretch", key="operator_mut_chart")
    oc2.plotly_chart(build_operator_sweep(rows_cx, "Crossover-Rate"), width="stretch", key="operator_cx_chart")
    st.caption("Mutationsrate 0 (keine Vielfalt außer der Startpopulation) ist klar am schlechtesten - ab einer kleinen positiven Rate spielt die genaue Höhe hier kaum noch eine Rolle, solange Elitismus die bisher beste Lösung schützt. "
               "Crossover-Rate 0 (jedes Kind ist nur eine mutierte Kopie eines Elternteils, nie eine Mischung) ist ebenfalls schlechter als jede positive Rate; mehr Crossover hilft hier leicht weiter, bis hin zu 100 %.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Population bleibt vielfältig genug** | Bei kleiner Population verliert sie ihre Vielfalt oft, bevor sie das globale Optimum erreicht hat - vorzeitige Konvergenz in ein flacheres lokales Minimum. | **CMA-ES**, **Differential Evolution** (beide passen die Suche an die kontinuierliche Landschaft an, statt blind zu mutieren) |
| **Ein einziges Ziel reicht** | Eine gewichtete Summe mehrerer Ziele erreicht nur den konvexen Teil der Pareto-Front, egal wie das Gewicht gewählt wird. | **NSGA-II** (sortiert direkt nach Pareto-Dominanz) |
| **Crossover und Mutation sind blinde Operatoren** | Feste Raten passen nicht zu jeder Landschaft; sie werden hier von Hand eingestellt, nicht aus dem Verlauf der Suche gelernt. | **CMA-ES** (passt die Suchverteilung selbst an) |
| **Positionsbasierte Kandidaten sind gleich teuer wie zufällige** | Der GA behandelt jedes Individuum unabhängig von seiner Position im Suchraum - keine gerichtete Bewegung wie bei einem Gradienten oder einer Geschwindigkeit. | **Partikelschwarm-Optimierung** (Geschwindigkeit statt Crossover), **Ameisenalgorithmus** (indirekte Kommunikation über Pheromone) |
"""
)
st.caption(
    "Die direkten Nachfolger der Populations-Linie (noch nicht gebaut): NSGA-II, CMA-ES, Differential Evolution, Partikelschwarm-Optimierung, Ameisenalgorithmus (→ Max-Min Ant System). Mit dem Hill Climbing der "
    "[Trajektorien-Metaheuristiken-Linie](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) ergäbe sich später ein Memetischer Algorithmus. Die Wurzel ist bewusst der bekannteste, allgemeinste Vertreter populationsbasierter Suche."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Population.** Eine Menge $P_t = \{x_1, \dots, x_{\mu}\}$ von $\mu$ Individuen in Generation $t$; bei der Lieferroute ist $x_i$ eine Permutation der $N$ Knoten, bei der Standortwahl $x_i \in [0, A]^2$.

**Fitness.** Lieferroute: $f(x) = \sum_k d_{\pi(k)\,\pi(k+1)}$ (Tourlänge), gewichtet mit CO2 als $f_w(x) = (1-w) \cdot f_{\text{dist}}(x) + w \cdot f_{\text{CO2}}(x)$. Standortwahl:
$f(x, y) = C - \sum_{j=1}^{K} A_j \exp\!\left(-\frac{(x - c_{j,x})^2 + (y - c_{j,y})^2}{2\sigma_j^2}\right)$ mit $C = \sum_j A_j$ - eine Summe von $K$ Gauß-Mulden unterschiedlicher Tiefe $A_j$ und Breite $\sigma_j$.

**Turnierselektion.** Für jeden benötigten Elternteil werden $k$ Individuen gleichverteilt gezogen, das mit dem niedrigsten $f$ gewinnt.

**Order Crossover (OX).** Aus zwei Positionen $i < j$ wird das Stück $x_1[i{:}j]$ wörtlich übernommen; die restlichen Positionen werden in der Reihenfolge von $x_2$ mit den noch fehlenden Knoten aufgefüllt - das Ergebnis ist stets eine gültige Permutation.

**BLX-α-Crossover.** Je Koordinate: $c \sim \mathcal{U}(\min(x_1, x_2) - \alpha\,|x_1 - x_2|,\; \max(x_1, x_2) + \alpha\,|x_1 - x_2|)$, auf die Gebietsgrenzen gekappt.

**Mutation.** Tausch (Route): mit Wahrscheinlichkeit $p_m$ werden zwei zufällige Positionen vertauscht. Gauß (Standort): je Koordinate mit Wahrscheinlichkeit $p_m$ wird $\mathcal{N}(0, \sigma)$ addiert, $\sigma = 5$ km fest.

**Elitismus.** Die $e$ besten Individuen aus $P_t$ übernehmen unverändert in $P_{t+1}$; die restlichen $\mu - e$ entstehen aus Selektion, Crossover (mit Wahrscheinlichkeit $p_c$) und Mutation.

**Diversität.** Route: $1 - \overline{\text{Kantenanteil}}$ über Paare der Population (0 = identische Touren). Standort: mittlerer euklidischer Abstand vom Populationsschwerpunkt.

**Referenz.** Route: Brute-Force über alle $(N-1)!$ Touren, nur bis $N \le 9$ (8! = 40 320 Touren) berechenbar. Standort: feinstes Gitter (Schrittweite 1 km) über das ganze Gebiet.

Implementiert in `ga_algorithm.py` (Operatoren, GA-Schleife), `ga_scenario.py` (Vehikel), `ga_evaluation.py` (Kennzahlen, Sweeps, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
