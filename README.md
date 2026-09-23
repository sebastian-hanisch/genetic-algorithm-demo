# 🧬 Genetischer Algorithmus – eine Population statt einer einzelnen Lösung

Wurzelstück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Anders als die Fall-Demos (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein**
Verfahren – den Genetischen Algorithmus (GA) – an einem wachsenden Beispiel, auf zwei Vehikeln gleichzeitig: eine **Lieferroute** (diskret,
Permutationskodierung, Order Crossover + Tausch-Mutation) und eine **Standortwahl** (kontinuierlich, BLX-α-Crossover + Gauß-Mutation). Beide
teilen sich Turnierselektion und Elitismus.

Die direkten Nachfolger der Linie (NSGA-II, CMA-ES, Differential Evolution, Partikelschwarm-Optimierung, Ameisenalgorithmus → Max-Min Ant System)
setzen an je einer eigenen Schwäche oder einem eigenen Kontrast zu dieser Wurzel an; mit dem Hill Climbing der bereits gebauten
[Trajektorien-Metaheuristiken-Linie](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) ergäbe sich später ein Memetischer Algorithmus
(lokale Suche + Population). Keiner dieser Nachfolger ist bisher gebaut.

## Warum dieses Problem

Ein Genetischer Algorithmus ist die bekannteste, allgemeinste populationsbasierte Metaheuristik: statt einer einzelnen Lösung Schritt für Schritt
zu verbessern, hält er eine ganze Population und verbessert sie über Selektion, Crossover und Mutation als Ganzes – auch wenn einzelne Nachkommen
schlechter sind als ihre Eltern. Die Demo zeigt live, wo genau das hilft (mehrere Trichter gleichzeitig erkunden) und wo es an seine Grenzen stößt
(eine zu kleine Population verliert ihre Vielfalt, bevor sie das beste Ergebnis erreicht hat; eine feste Gewichtung mehrerer Ziele trifft nur einen
Teil der Pareto-Front) – genau die Lücken, an denen die geplanten Nachfolger ansetzen.

## Befunde / Korrekturen

Der erste CO2-Entwurf (ein Lastfaktor je **Stopp**, Kantenkosten = Distanz × Mittel der beiden Knotenfaktoren) erzeugte eine praktisch immer nur
**einpunktige** Pareto-Front: weil der Faktor nur den Knoten anklebt, ist er im Kern eine lokale Umgewichtung der Distanz, keine unabhängige
zweite Zielgröße – die distanzkürzeste Tour war fast immer auch die CO2-günstigste. Gefunden beim ersten Live-Test des Pareto-Experiments, nicht
vorab angenommen. Fix: der CO2-Faktor sitzt jetzt auf der **Kante** (Straßenabschnitt), unabhängig gezogen für jedes Stopp-Paar – manche
Verbindungen sind stauanfälliger als andere, unabhängig vom Zielort. Damit hat die Brute-Force-Pareto-Front bei einer kleinen Instanz (8 Stopps)
im Mittel mehrere Punkte (siehe Tests), und eine feste Gewichtung trifft nachweislich nur einen Teil davon.

## Modell

**Lieferroute** (Vehikel wie [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/)): ein Depot in der Mitte und *n*
Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen, eine Rundtour. Jedes Stopp-Paar hat einen unabhängigen CO2-Faktor (0,6–3,4)
für die Kante zwischen ihnen; die Kosten sind eine gewichtete Summe aus Distanz und CO2 (Gewicht 0 = reine Distanz).

**Standortwahl**: Lage (x, y) eines Verteilzentrums im selben Gebiet. Die Kosten sind eine Summe aus mehreren Gauß-Mulden unterschiedlicher Tiefe
und Breite (Nachfrageschwerpunkte) – nur die tiefste ist die global günstigste Lage, die anderen sind lokale Minima.

**GA**: Turnierselektion (*k* Kandidaten, der beste gewinnt), Elitismus (die besten *e* Individuen überleben unverändert), Crossover mit
Wahrscheinlichkeit *p*_c, Mutation mit Wahrscheinlichkeit *p*_m je Kind/Gen. Details und Formeln im Expander "📐 Mathematische Formulierung" der
App.

## Methodik

Ein numpy-von-Grund-auf-Kern (`ga_algorithm.py`) für beide Kodierungen, geteilte Selektions-/Elitismus-Logik, kodierungsspezifische Operatoren.
Referenz: Brute-Force über alle Touren bis 9 Stopps (8! = 40 320), sonst ein feines Gitter (1-km-Schritte) für die Standortwahl – bei mehr als 9
Lieferstopps gibt es keine schnell berechenbare exakte Referenz (siehe die Exakte-Suche-Linie für Verfahren, die trotzdem eine Schranke liefern).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Verbessert der GA die Startpopulation deutlich? | Standardfall (30 Stopps, Pop. 60, 150 Generationen): beste Tour 1434 km → 733 km (**48,9 %** kürzer) | `test_standardfall_preset_claims` |
| Verliert eine kleine Population zu früh an Vielfalt? | Bei Populationsgröße 10: **55 %** der 20 GA-Läufe landen im globalen Trichter der Standortwahl; bei Populationsgröße 100: **95 %** | `test_convergence_experiment_headline_claims` |
| Trifft eine feste Gewichtung die Pareto-Front? | Bei 8 Stopps hat die Brute-Force-Front **8** Punkte; über 8 verschiedene Gewichte (je 5 GA-Seeds) werden nur **4 von 8** getroffen | `test_pareto_experiment_headline_claims` |
| Wie stark schadet Mutationsrate 0? | Mit knappem Budget (Pop. 20, 25 Generationen): Mutationsrate 0 klar am schlechtesten; jede positive Rate liegt deutlich darunter, die genaue Höhe spielt kaum noch eine Rolle | `test_operator_sensitivity_headline_claims` |
| Hilft eine höhere CO2-Gewichtung? | Gewicht 0,4 senkt die CO2-Kosten der gefundenen Tour um **24 %** gegenüber reiner Distanzminimierung (bei dieser Instanz bleibt die Distanz nahezu gleich) | `test_distanz_und_co2_preset_claims` |

## Ehrliche Grenzen

- **Keine exakte Referenz oberhalb von 9 Lieferstopps** – der Abstand zum Optimum lässt sich bei der Standard-Instanz (30 Stopps) nicht direkt
  messen, nur die Verbesserung gegenüber der Startpopulation.
- **Feste, von Hand eingestellte Operatorraten** – der GA lernt nicht aus dem Verlauf der Suche, welche Rate gerade sinnvoll ist (das behebt
  CMA-ES).
- **Eine gewichtete Summe für mehrere Ziele** – sie kann nur den konvexen Teil der Pareto-Front erreichen, egal wie fein das Gewicht durchgefahren
  wird (das behebt NSGA-II).
- **Kein Gradient, keine gerichtete Bewegung** – jedes Individuum wird unabhängig behandelt, anders als bei Partikelschwarm-Optimierung
  (Geschwindigkeit) oder dem Ameisenalgorithmus (Pheromonspur).

## Tests

141 Tests (`pytest tests/ -v`): Algorithmus-Kern gegen Handrechnung (Order-Crossover-Lehrbuchbeispiel) und Invarianten (OX bleibt immer eine
gültige Permutation, BLX/Gauß bleiben in den Grenzen), der GA findet auf sehr kleinen Instanzen nachweislich das Brute-Force-/Gitter-Optimum,
Vehikel-Reproduzierbarkeit, Presets-Mechanik, AppTest-Rauchtests (jedes Preset, der Kodierungs-Umschalter, der Generation-Slider inkl. Abspielen,
Permalink-Grenzen, jedes Experiment auf Abruf) und `test_claims.py` (jede Zahl aus diesem README).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `ga_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `ga_presets.py` | Permalink/Presets-Mechanik, Kodierungs-Umschalter |
| `ga_scenario.py` | Vehikel-Erzeuger (Lieferroute, Standortwahl) |
| `ga_algorithm.py` | GA-Kern: Selektion, Crossover, Mutation, Hauptschleife |
| `ga_evaluation.py` | Kennzahlen, Referenz, Sweeps, die drei Experimente |
| `ga_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Mehr als zwei Kodierungen (z. B. eine Baum- oder Graphkodierung) – die geplanten Nachfolger (NSGA-II, ACO) bringen jeweils ihre eigene
  Kodierung mit.
- Adaptive Operatorraten (Selbstanpassung wie bei CMA-ES) – bewusst der Wurzel-Demo vorbehalten, die zeigt, was *ohne* Adaption passiert.
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
