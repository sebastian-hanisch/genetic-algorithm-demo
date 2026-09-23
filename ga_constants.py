"""Konstanten der Genetic-Algorithm-Demo: zwei Vehikel (diskrete Lieferroute, kontinuierliche Standortwahl), GA-Regler, Presets (Presets folgen nach den Messungen)."""

# --- Diskretes Vehikel: Lieferroute (wie hill-climbing-demo) -----------------------------------------------------------------------------

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0               # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0             # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 100, 30, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25

# CO2-Faktor je Straßenabschnitt (Kante), unabhängig von der Distanz - manche Verbindungen sind stauanfälliger
# als andere. Die CO2-Kosten einer Kante sind Distanz * Faktor dieser Kante; reine Distanzminimierung
# ignoriert das - der echte Zielkonflikt (nicht nur eine Umgewichtung) motiviert das gewichtete-Summe-Experiment.
CO2_FACTOR_LO, CO2_FACTOR_HI = 0.6, 3.4
CO2_HIGHLIGHT_PERCENTILE = 75      # ab diesem Perzentil der Kantenfaktoren gilt ein Abschnitt als "stark belastet" (Karte)

# --- Kontinuierliches Vehikel: Standortwahl -----------------------------------------------------------------------------------------------
# Kosten eines Standorts (x, y) im Gebiet: mehrere Nachfrageschwerpunkte ziehen wie Trichter unterschiedlicher Tiefe
# (Gauß-Mulden); nur der tiefste Trichter ist die global günstigste Lage - die anderen sind lokale Minima.

K_WELLS = 5
WELL_MARGIN = 12.0                # Schwerpunkte liegen mindestens so weit vom Rand entfernt
AMP_MIN, AMP_MAX = 15.0, 40.0     # Tiefe eines Trichters
SIGMA_MIN, SIGMA_MAX = 6.0, 14.0  # Breite eines Trichters
GRID_STEP = 1.0                   # Auflösung des Referenz-Gitters (Grid-Search-Minimum) in km

# --- Genetischer Algorithmus --------------------------------------------------------------------------------------------------------------

ENCODINGS = ("perm", "real")
ENCODING_LABELS = {"perm": "Diskret – Lieferroute", "real": "Kontinuierlich – Standortwahl"}

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 10, 200, 60, 10
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 150, 10
CX_MIN, CX_MAX, DEFAULT_CX, CX_STEP = 0.0, 1.0, 0.9, 0.05
MUT_MIN, MUT_MAX, DEFAULT_MUT, MUT_STEP = 0.0, 1.0, 0.2, 0.05
ELITE_MIN, ELITE_MAX, DEFAULT_ELITE, ELITE_STEP = 0, 10, 2, 1
K_MIN, K_MAX, DEFAULT_K, K_STEP = 2, 8, 3, 1
SEED_MAX = 999999
DEFAULT_SEED = 35                 # Vehikel-Seed (wie hill-climbing-demo)
DEFAULT_GA_SEED = 7               # Seed des GA-Laufs selbst (Population, Selektion, Crossover, Mutation)
BLX_ALPHA = 0.5                   # Ausdehnung der BLX-α-Crossover über die Elternwerte hinaus

CO2_WEIGHT_MIN, CO2_WEIGHT_MAX, DEFAULT_CO2_WEIGHT, CO2_WEIGHT_STEP = 0.0, 1.0, 0.0, 0.1

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

CONVERGENCE_SEEDS = tuple(range(200000, 200020))   # 20 GA-Seeds je (Populationsgröße) im Vorzeitige-Konvergenz-Experiment
CONVERGENCE_POP_SIZES = (10, 20, 40, 100)
GLOBAL_TOL_KM = 3.0                # Standort gilt als "im globalen Trichter" gefunden, wenn er höchstens so weit vom besten Gitterpunkt entfernt liegt

PARETO_N = 8                       # kleine Instanz für die Brute-Force-Pareto-Front (8! Touren, in < 1 s)
PARETO_VEHICLE_SEED = 19           # per Suche gewählt: liefert eine Front mit mehreren Punkten (nicht jeder Seed tut das)
PARETO_WEIGHTS = tuple(round(w, 2) for w in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0))
PARETO_SEEDS = tuple(range(300000, 300005))        # GA-Seeds je Gewicht

OPERATOR_MUT_VALUES = (0.0, 0.02, 0.1, 0.4, 0.9)
OPERATOR_CX_VALUES = (0.0, 0.3, 0.6, 0.9, 1.0)
OPERATOR_SEEDS = tuple(range(400000, 400020))
OPERATOR_POP, OPERATOR_GENS = 20, 25    # eigenes, knappes Budget (unabhängig von der Seitenleiste) - erst hier zeigt sich der Effekt deutlich

SWEEP_SEEDS = tuple(range(100000, 100005))         # Vehikel-Seeds für die Sweeps (wie hill-climbing-demo)
SWEEP_GA_SEEDS = tuple(range(500000, 500003))       # GA-Seeds je Vehikel-Seed


def _preset(encoding="perm", n=DEFAULT_N, ballung=DEFAULT_BALLUNG, co2_weight=DEFAULT_CO2_WEIGHT, pop=DEFAULT_POP, gens=DEFAULT_GEN, cx=DEFAULT_CX, mut=DEFAULT_MUT, elitism=DEFAULT_ELITE, k=DEFAULT_K, ga_seed=DEFAULT_GA_SEED):
    return {"encoding": encoding, "n": n, "ballung": ballung, "co2_weight": co2_weight, "seed": DEFAULT_SEED, "pop": pop, "gens": gens, "cx": cx, "mut": mut, "elitism": elitism, "k": k, "ga_seed": ga_seed}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    # ga_seed=4 per Suche gewählt: landet klar in einem lokalen (nicht dem globalen) Trichter - siehe PRESET_HELP
    "Kleine Population (vorzeitige Konvergenz)": _preset(encoding="real", pop=10, ga_seed=4),
    "Große Population": _preset(pop=150),
    "Hohe Mutationsrate": _preset(mut=0.7),
    "Kaum Mutation": _preset(mut=0.02),
    "Standortwahl (kontinuierlich)": _preset(encoding="real"),
    "Distanz und CO2 gewichtet": _preset(co2_weight=0.4),
    "Große Instanz": _preset(n=80),
}
# Gemessen mit dem jeweiligen Preset-Seed (siehe PRESETS) - ein einzelner Lauf, kein Mittel über mehrere Seeds
# (Ausnahme: die beiden Verweise auf die Experimente weiter unten in der App, die selbst über viele Seeds mitteln).
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "30 Lieferstopps, Populationsgröße 60, 150 Generationen: die beste Tour schrumpft von 1434 km (Startpopulation) auf 733 km (48,9 % kürzer), die Diversität der Population sinkt von 0,93 auf 0,06.",
    "Kleine Population (vorzeitige Konvergenz)": "Nur 10 Individuen auf der Standortwahl: die Suche bleibt 16,6 % über der global günstigsten Lage hängen (nur 1,5 % besser als die Startpopulation), die Diversität kollabiert von 31,9 auf 1,3. Bei Populationsgröße 100 trifft der GA das globale Optimum in 95 % von 20 Läufen (siehe Experiment „Vorzeitige Konvergenz“).",
    "Große Population": "150 statt 60 Individuen, sonst wie im Standardfall: die beste Tour endet bei 614 km statt 733 km (56 % statt 49 % kürzer als die Startpopulation).",
    "Hohe Mutationsrate": "Mutationsrate 0,7 statt 0,2: die Population bleibt bis zum Schluss deutlich vielfältiger (Diversität 0,44 statt 0,06) und die Tour wird hier sogar kürzer (621 km statt 733 km).",
    "Kaum Mutation": "Mutationsrate 0,02 statt 0,2: die Population konvergiert fast vollständig (Diversität am Ende 0,01). Wie stark eine zu niedrige Mutationsrate tatsächlich schadet, zeigt das Experiment „Operator-Sensitivität“ (auf der Standortwahl mit knappem Budget: Mutationsrate 0 ist dort klar am schlechtesten).",
    "Standortwahl (kontinuierlich)": "Dieselben GA-Regler wie im Standardfall, aber auf der Standortwahl statt der Lieferroute: die Suche trifft die global günstigste Lage praktisch exakt (-0,02 % Abstand).",
    "Distanz und CO2 gewichtet": "Gewicht 0,4 auf die CO2-Kosten: gegenüber reiner Distanzminimierung sinken die CO2-Kosten der gefundenen Tour um 24 % (bei dieser Instanz bleibt die Distanz nahezu gleich). Wie viel von der Pareto-Front eine feste Gewichtung insgesamt erreicht, zeigt das Experiment weiter unten.",
    "Große Instanz": "80 statt 30 Lieferstopps: die beste Tour schrumpft von 3817 km auf 1639 km (57 % kürzer) - ähnlich wie im Standardfall (49 %), bei mehr als doppelt so großer Instanz.",
}

SWEEP_VALUES = {"pop": (10, 20, 40, 60, 100, 150), "gens": (20, 50, 100, 150, 250, 400), "mut": OPERATOR_MUT_VALUES, "cx": OPERATOR_CX_VALUES}
SWEEP_LABELS = {"pop": "Populationsgröße", "gens": "Generationen", "mut": "Mutationsrate", "cx": "Crossover-Rate"}
