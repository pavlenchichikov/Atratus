# Atratus

![Atratus](assets/atratus-banner.svg)

[![CI](https://github.com/pavlenchichikov/Atratus/actions/workflows/ci.yml/badge.svg)](https://github.com/pavlenchichikov/Atratus/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Lint: Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey.svg)](LICENSE)

[English](README.md) | [Русский](README.ru.md) | [Français](README.fr.md) | [**Deutsch**](README.de.md)

**Multi-Asset-Engine für Handelssignale auf Basis maschinellen Lernens.** Ein eigenes Ensemble je Wert (CatBoost + LSTM + Transformer + TCN) über 847 Märkte - Krypto, US-, europäische und russische Aktien, Indizes, Zinsen, Volatilität, Anleihen- und Sektor-ETFs, Devisen und Rohstoffe - mit Walk-Forward-Auswahl, kalibrierten Wahrscheinlichkeiten, Kelly-Positionsgrößen, Kontrolle des Extremrisikos, einem FastAPI-Dashboard und einem autonomen, statistisch abgesicherten Forschungsagenten. Ausschließlich Signale, mit dem Menschen in der Schleife: keine automatische Ausführung.

> **Haftungsausschluss.** Atratus ist ein Forschungs- und Lehrprojekt. Sein Ergebnis besteht aus Modellvorhersagen, **nicht aus Anlageberatung und nicht aus einer Empfehlung, ein Wertpapier zu kaufen oder zu verkaufen**. Märkte bergen Risiken, und Sie können Geld verlieren. Die Software wird "wie besehen" bereitgestellt, ohne jede Gewährleistung. Nutzung auf eigenes Risiko; führen Sie eigene Recherchen durch und ziehen Sie vor jeder finanziellen Entscheidung eine zugelassene Fachperson hinzu. Im Zweifelsfall sind die englische Fassung und die Datei [`LICENSE`](LICENSE) maßgeblich.

[![Android-App herunterladen](https://img.shields.io/badge/Download-Android%20APK-brightgreen?logo=android&logoColor=white&style=for-the-badge)](https://github.com/pavlenchichikov/Atratus/releases/latest/download/Atratus.apk)

> **Umfang dieser Übersetzung.** Dieses Dokument deckt alles ab, was zum Installieren, Ausführen und Verstehen des Systems nötig ist: Funktionsweise, vollständige Menüreferenz, täglicher Betrieb, Training, Konfiguration und Fehlersuche. Die langen Forschungsprotokolle - die Theorie hinter dem A/B-Gate, die Varianzzerlegung, die Versuchshistorie - sind hier zusammengefasst und nur im [englischen README](README.md) vollständig ausgeführt, das die maßgebliche Fassung bleibt.

## Inhaltsverzeichnis

- [Funktionsumfang](#funktionsumfang)
- [Funktionsweise](#funktionsweise)
- [Weboberfläche](#weboberfläche)
- [Autonomer Forschungsagent](#autonomer-forschungsagent)
- [Adoption je Wert](#adoption-je-wert)
- [Analystenagent](#analystenagent)
- [Selbsterhaltende Schleife](#selbsterhaltende-schleife)
- [Voraussetzungen](#voraussetzungen)
- [Umgebung und GPU](#umgebung-und-gpu)
- [Schnellstart](#schnellstart)
- [Das Startmenü](#das-startmenü)
- [Täglicher Betrieb](#täglicher-betrieb)
- [Training](#training)
- [Konfiguration](#konfiguration)
- [Projektstruktur](#projektstruktur)
- [Tests](#tests)
- [Lizenz](#lizenz)

## Funktionsumfang

- **Ein Modell je Wert, nicht ein Modell für den Markt.** Jeder Wert trainiert sein eigenes Ensemble aus vier Mitgliedern (CatBoost, LSTM, Transformer, TCN), und der Champion wird durch einen Walk-Forward-Backtest mit Gebühren, Slippage und einem Embargo gegen Datenlecks bestimmt. Das ist die zentrale Entwurfsentscheidung, und die folgenden Zahlen sind ihr Preis und ihr Ertrag.

  Stand 2026-09-03: **847 Ticker in `FULL_ASSET_MAP`, davon 827 trainiert und aktuell, 20 warten auf ein erstes Training.** Diese Lücke ist kein Verfall. Ein Ticker kommt in die Karte, sobald er beobachtenswert ist, während ihn zu trainieren ein eigener, teurer Vorgang bleibt; die Karte läuft `models/` deshalb immer voraus. Diese zwanzig sind junge Notierungen und Indexaufnahmen (`ARB`, `CRWV`, `NBIS`, `SNOW`, `GEV`, `TAO`, `MBNK`, `RENI`, `KLVZ`, `DIAS`, `PRMD`, `EGX30`, `WIG20` und sieben weitere); geschlossen wird die Lücke über den Menüpunkt `[5F]`.

  Es gibt auch den umgekehrten Fall: **vier Modellsätze auf der Platte, deren Karteneintrag verschwunden ist** (`avb`, `eqr`, `wbs`, `brkb`). Drei sind aus der Karte entfernte Werte. Der vierte ist eine Umbenennung: Berkshire wechselte von `BRKB` zu `BRK-B`, dessen Modelldateiname `brk_b` lautet; das trainierte Modell wurde damit zur Waise, und der Wert zählt nun als untrainiert. Einen Kartenschlüssel umzubenennen, ohne die Modelldateien mitzuziehen, bewirkt genau das - stillschweigend, und sichtbar wird es in `[M] Model Health`.

- **Kalibrierte Wahrscheinlichkeiten statt roher Scores.** Jede Wahrscheinlichkeit durchläuft eine außerhalb der Stichprobe angepasste isotone Kalibrierung, sodass eine 0,70 tatsächlich sieben von zehn Mal bedeutet.

- **Walk-Forward-Auswahl mit Embargo.** Der Champion wird nie auf den Daten gewählt, die ihn trainiert haben, und ein Embargo entfernt die Balken an der Grenze jeder Falte, damit das Label nicht rückwärts durchsickert.

- **Kontrolle des Extremrisikos.** Fraktionale Kelly-Positionsgrößen, Verlust- und Drawdown-Grenzen sowie ein Taleb-Gate, das in Regimen, in denen die Verlustverteilung unbrauchbar wird, gar nicht erst eine Position eröffnet.

- **Ein autonomer Forschungsagent, der Nein sagen darf.** Er sucht neue Konfigurationen, misst sie an einem Haltebestand, den er nie optimiert hat, und übernimmt nur, was eine vorab festgelegte statistische Schwelle überschreitet. Die meisten Kampagnen führen zu nichts, und das ist das erwartete Verhalten.

- **Ein Analystenagent, der die Meinung des Modells nie sieht.** Eine zweite Einschätzung, gebildet aus denselben Rohdaten, aber ohne Zugriff auf die Wahrscheinlichkeit des Ensembles, das ausgegebene Signal, die Timing-Entscheidung oder die Positionsgröße.

- **Alles Gemessene wird protokolliert.** `prediction_log`, `level_log`, `guru_log`, `analyst_log`: jede Einschätzung wird niedergeschrieben, bevor ihr Ergebnis existiert, und ein Abgleich bewertet sie später. Eine Zahl, die das Projekt im Nachhinein nicht prüfen kann, wird auch nicht so dargestellt, als könnte es das.

## Funktionsweise

```
data_engine.py    Tagesbalken von Yahoo Finance und MOEX -> market.db
train_hybrid.py   vier Mitglieder je Wert, Walk-Forward-Auswahl -> models/
predict.py        bewertet jeden Wert, schreibt prediction_log
webapp.py         FastAPI-Dashboard; app.py ist die Streamlit-Fassung
```

Der Kern ist ein Ensemble je Wert. Die vier Mitglieder sehen dieselben Merkmale und stimmen ab; ein Meta-Modell lernt auf den Validierungsfalten, wie stark jedes zählt. Heraus kommt eine kalibrierte Wahrscheinlichkeit, die über wertspezifische Schwellen in ein Signal übersetzt und anschließend durch Regime, Risiko-Gate und die angepassten Politiken (Timing, Positionsgröße, Einstiegsniveaus) gefiltert wird.

Nichts wird automatisch ausgeführt. Das System liefert Signale; die Entscheidung bleibt beim Menschen.

## Weboberfläche

`webapp.py` liefert ein FastAPI-Dashboard: das Radar aller heute bewerteten Werte, je Wert eine Karte mit Niveaus und Historie, die Forschungs- und Erfahrungsseiten, die Analystenseite und den Zustand der Tagesschleife.

```bash
uvicorn webapp:app --port 8000
```

Oder Menüpunkt `[WU]`, der sie direkt auf dem Dashboard öffnet.

## Autonomer Forschungsagent

`auto_research.py` automatisiert die Suche nach besseren Konfigurationen. Ein Vorschlagsmechanismus schlägt ein "Genom" vor - einen Satz Hebel auf Merkmale, Labels und Modell - ein billiger Vorfilter allein auf CatBoost-Basis wirft die offensichtlichen Verlierer heraus, und was übrig bleibt, wird gegen die zwischengespeicherte Referenz gemessen.

Standardmäßig ist der Vorschlagsmechanismus eine evolutionäre Suche, ohne LLM und ohne API-Schlüssel. `GTRADE_AR_PROPOSER=llm` setzt stattdessen ein Modell ein (Anthropic als Vorgabe, OpenAI oder jeder OpenAI-kompatible Endpunkt, auch ein **lokales Ollama** über `GTRADE_AR_LLM=ollama`).

Der wesentliche und zugleich unintuitive Punkt: **die meisten Kampagnen finden nichts, und das ist das erwartete Ergebnis.** Eine Übernahme verlangt statistische Signifikanz, einen Effekt oberhalb einer praktischen Untergrenze und den Nachweis, dass der Gewinn nicht durch Aushungern der neuronalen Mitglieder erkauft wurde. Ein HOLD ist eine Messung, kein Scheitern.

Die vollständige Darstellung - Score-Basis, Quality-Diversity-Illumination, die Teststärke des A/B-Gates und warum sie der begrenzende Faktor ist - steht im [englischen README](README.md#auto-research-agent).

## Adoption je Wert

Ein Genom wirkt nicht auf alle Werte gleich. Am 2026-09-02 gemessen: der Kandidat, der sein Gate mit -0,30 über 40 Werte **nicht** bestand, war **+1,20 auf RTX** und **-3,84 auf ROSN** wert, beides anschließend auf Seeds bestätigt, die die Auswahl nie gesehen hatte. Ihn überall oder nirgends zu übernehmen wirft beide Befunde weg; deshalb darf ein Wert das Genom behalten, das an *ihm* gemessen wurde, während alles Übrige auf der globalen Adoption bleibt.

Drei Schritte in dieser Reihenfolge, jeder beantwortet eine Frage, die der nächste braucht:

1. **Unterschiede je Wert suchen** (`ab_per_asset.py`, kostenlos, trainiert nichts). Zu lesen am Verhältnis, nicht an der Größe: im ersten Durchlauf war der Wert mit dem größten Standardfehler derjenige, der später das Vorzeichen wechselte.
2. **Auf frischen Seeds bestätigen** (`ab_confirm.py`, Stunden, trainiert). Extreme sind allein dadurch überzeichnet, dass sie ausgewählt wurden: die drei am 2026-09-02 ausgewählten behielten nur **30 %** ihres gemessenen Effekts.
3. **Übernehmen, was standgehalten hat** (`adopt_genome.py --asset WERT --evidence TEXT`). Das Werkzeug verweigert ohne Beleg, und der Beleg muss die **Replikation** sein, nie der Durchlauf, der den Wert ausgewählt hat.

Ausbeute über drei Kampagnen: etwa eines von drei ausgewählten Extremen übersteht seine Replikation.

**Einen späteren Durchlauf gegen das Bereits-Übernommene lesen.** Sobald ein Wert ein eigenes Genom hat, lautet die Frage nicht mehr "ist dieser Kandidat gut", sondern "ist er besser als das, was dieser Wert ohnehin fährt". Der Referenzarm beantwortete das längst - er läuft durch dasselbe `config.py` -, aber nichts in der Zeile sagte es. Schritt 1 zeigt jetzt eine Spalte `on` (`own` oder `global`), benennt ausdrücklich jeden übernommenen Wert, den der Kandidat verschlechtern würde, und warnt, wenn der Vergleich nicht der ist, für den man ihn hält: eine Adoption, die nach der Messung des Referenzarms datiert, bedeutet, dass die Grundlinie ihr vorausgeht und derselbe Gewinn doppelt gezählt würde.

Zwei Folgen sollte man erwarten statt entdecken: **jede Adoption hebt die Latte** (ein übernommener Wert hat seinen Gewinn bereits eingefahren, ein späterer Kandidat wirkt dort also schwächer), und **der Backtest kann nicht sagen, ob eine Adoption geholfen hat**. Das kann nur die reale Trefferquote; `core/adoption_ledger.py` hält sie im Moment der Adoption fest, weil sie danach nicht mehr rekonstruierbar ist.

## Analystenagent

Eine zweite Einschätzung je Wert, gebildet **ohne je die Einschätzung des Ensembles zu sehen**. Er sieht weder die Modellwahrscheinlichkeit noch das ausgegebene Signal, die Timing-Entscheidung oder die Positionsgröße; zwei Tests erzwingen das - einer durchsucht das serialisierte Dossier nach verbotenen Schlüsseln, der andere fixiert dessen exakte Schlüsselmenge, sodass kein Feld hinzukommen kann, ohne dass jemand es deklariert.

Er liest, was das Projekt ohnehin berechnet - 80 Felder in zwölf benannten Blöcken: Preis und Bewegung, wo der Kurs in seinen eigenen Volatilitätseinheiten steht, das zurückliegende Jahr gegen seinen Index, Handelsfluss, der Markt, in dem er sich bewegt hat, sein Regime und sein Sektor, Fundamentaldaten, rohe Schlagzeilen, Kalender und Leitzins, das Urteil des Guru Council sowie seine eigenen früheren Einschätzungen zu diesem Wert.

**Ein Feld, nach dem der Prompt nicht fragt, ist ein Feld, das das Modell nicht liest.** Über die ersten 35 Einschätzungen gemessen: von den 21 Feldern, die die Anweisungsliste benannte, wurden 16 als Beleg zitiert; von den übrigen 39 nur neun, meist ein einziges Mal. Keine einzige Schlagzeile wurde gelesen. Die Liste benennt inzwischen 65 der 80 Felder, und jeder Lauf gibt seine eigene Abdeckung aus.

**Er liefert ein diskretes Urteil, nie einen Prozentwert**: eine Richtung, eine Überzeugung von 1 bis 5, ein erwartetes Volatilitätsregime, das Risiko, das ihn am ehesten widerlegt, und seine Begründung. Eine frei formulierte Zahl aus einem Modell ist messbar, aber nicht korrigierbar, weil jede Antwort einmalig ist und keine Zelle je genug Historie sammelt, um nachkalibriert zu werden.

Der Prozentwert auf der Karte ist deshalb **nicht die Zahl des Analysten**: er ist das, was diese Urteilszelle historisch wert war, und darf der Einschätzung darüber widersprechen. Diese Zahl ist um die Marktdrift bereinigt: von Juni bis September 2026 zeigte die russische Klasse einen rohen Kaufertrag von -0,116 ATR, wovon -0,101 schlicht ein fallender Markt war.

**Quellen, die er anfordern darf.** Über das ihm gereichte Dossier hinaus darf der Analyst vor der Entscheidung weitere Belege *anfordern*: `insider_filings` (Geschäfte, die Unternehmensverantwortliche der SEC per Formular 4 **offengelegt** haben) und `news_search` (die Nachrichtenquellen des Projekts zu einer selbst gewählten Anfrage). Drei Regeln sichern die Reproduzierbarkeit: jeder Aufruf und sein Ergebnis werden auf der Urteilszeile **protokolliert**; jedes Werkzeug erklärt, ob es ein **vergangenes Datum respektiert**, andernfalls wird es in einem zurückgespulten Lauf verweigert; und die Registrierung ist eine **Positivliste**, nie ein freier Zugriff auf beliebige URLs.

Ein Werkzeug darf Material liefern. Die Schlussfolgerung anderer darf es nicht liefern: Analystenkonsens, Kursziele und Broker-Ratings sind bewusst ausgeschlossen, und `tools.register()` wirft eine Ausnahme, statt so etwas aufzunehmen. Der Analyst existiert, um sich eine eigene Meinung zu bilden, und einen Konsens kann man selbst nachschlagen.

**Nichts, was er sagt, gilt bisher als belastbar.** `analyst.py score` misst ihn gegen drei Vergleichsmaßstäbe und gibt SHIP oder HOLD aus. SHIP verlangt mindestens 500 bewertete Urteile, eine Intervallabdeckung zwischen 0,75 und 0,85, eine Permutationskontrolle, die zusammenbricht, und einen geringeren Fehler als beide Maßstäbe. Ein HOLD ist ein Ergebnis, und die Kriterien standen fest, bevor die erste Zahl vorlag.

## Selbsterhaltende Schleife

`loop_cycle.py` führt die sichere Tagesroutine aus (Daten, Makrokalender, Vorhersage, Abgleich, Nachtragen der Analystenergebnisse) und prüft jeden Wert auf Drift: gleitende Trefferquote unter einer Untergrenze, Abfall gegenüber der Trainingsgrundlinie, Modellalter oder veraltete Daten. Vorschläge erscheinen unter `/loop`.

**Die Schleife trainiert nie von sich aus nach; ein Nachtraining wartet immer auf Ihre Freigabe.** Tragen Sie `run_loop.bat` in die Aufgabenplanung ein, um sie täglich laufen zu lassen. Die Drift-Schwellen stehen in `core/drift.py`.

## Voraussetzungen

- **Python 3.12** (3.11+ dürfte funktionieren; die CI läuft auf 3.12).
- **Betriebssystem:** Linux, macOS oder Windows. Unter Windows verlangt eine GPU die unten beschriebene festgelegte Umgebung: TensorFlow liefert ab 2.11 nur noch CPU-Wheels für Windows, eine Standardinstallation sieht die Karte also nie.
- **Speicherplatz:** ~8 GB frei - trainierte Modelle (~5,8 GB für 831 trainierte Werte) plus `market.db` (~310 MB). Der reine Betrieb braucht weit weniger.
- **Arbeitsspeicher:** 8 GB genügen für Dashboard und `predict.py` (im Betrieb kein TensorFlow). Das gesamte Universum zu trainieren will ~16 GB, oder in Blöcken von etwa 15 Werten (`GTRADE_ASSETS`) auf kleinerer Hardware.
- **GPU:** optional, aber lohnend. Auf einer RTX 2050 trainiert ein Wert in 158 s gegenüber 2850 bis 10480 s für denselben Wert auf einer CPU mit 12 Threads.
- **Netzwerk:** ausgehender Zugriff auf Yahoo Finance und MOEX (`SOCKS5_PROXY` wird unterstützt).

## Umgebung und GPU

Unter Windows läuft das Projekt in einer eigenen conda-Umgebung. Das ist keine Vorliebe: **TensorFlow hat die native CUDA-Unterstützung unter Windows nach 2.10 eingestellt**, jedes Wheel ab 2.11 ist also CPU-only, unabhängig von der verbauten Karte, und für 2.10 existiert kein Build für Python 3.11+. Die festgelegte Kombination ist deshalb Python 3.10 mit TensorFlow 2.10.

Vorzubereiten ist nichts: jeder Starter legt die Umgebung beim ersten Aufruf an und aktiviert sie danach.

```bat
auto_research.bat        :: Forschungsagent
run_gtrade.bat           :: Hauptmenü
call activate_env.bat    :: nur die Umgebung, für eine manuelle Sitzung
```

Drei Dateien mit je einer Zuständigkeit: `env_config.bat` (die einzige Stelle, die Umgebung und Versionen festschreibt), `activate_env.bat` (findet conda, legt an, aktiviert, prüft) und `setup_gpu.bat` (der Installer, der die Karte erkennt und passendes CUDA/cuDNN wählt).

**Wichtig:** das Menü selbst bleibt auf dem Basis-Python. Alles, was trainiert, läuft über `run_in_env.bat` in einem Kindprozess, damit ein späterer Betrieb nicht stillschweigend seine neuronalen Champions verliert.

## Schnellstart

```bash
pip install -r requirements.txt
cp .env.example .env          # Telegram-Token, Proxy falls nötig

python data_engine.py         # Marktdaten laden
python train_hybrid.py        # Modelle trainieren
python predict.py             # Signale in der Konsole
streamlit run app.py          # Dashboard
```

`run_gtrade.bat` öffnet ein Textmenü über all das. `python db_check.py` prüft `market.db` schreibgeschützt (`--fix` repariert Duplikate und Datumsformate). `python scheduler.py` läuft als Dienst: Daten alle 6 h, Vorhersagen alle 4 h, täglich eine Datenbankprüfung.

## Das Startmenü

`run_gtrade.bat` ist der Eingang. Tasten sind unabhängig von Groß- und Kleinschreibung; Enter allein an einer Unterabfrage übernimmt den in Klammern gezeigten Vorgabewert.

### TÄGLICH

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `1` | `data_engine.py`, dann `train_hybrid.py`, dann `predict.py` | Der vollständige Zyklus. Mehrere Stunden. |
| `2` | `streamlit run app.py` | Das Streamlit-Dashboard. |
| `3` | `predict.py` | Bewertet jeden Wert und schreibt `prediction_log`. |
| `4` | `data_engine.py` | Nur die heutigen Balken. |
| `WU` | `uvicorn webapp:app --port 8000` | Die FastAPI-Oberfläche, geöffnet auf dem Dashboard. |

### TRAINING

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `5` | `train_hybrid.py` | Alle Werte in einem Prozess. |
| `5C` | `train_chunked.py` | Ein frischer Prozess je Block, fortsetzbar, Champion-Challenger. |
| `5R` | `train_hybrid.py` auf einer Liste | Fragt nach den Werten und ob die Beförderung erzwungen wird. |
| `5F` | `model_health.py --list`, dann `train_chunked.py` | Fragt: Werte ohne Champion nachziehen, verschlechterte reparieren, oder beides. |
| `T` | `optuna_tune.py` | Hyperparametersuche je Wert. |

### SIGNALE

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `6` | `backtest.py` | Walk-Forward-Backtest über die Champions auf der Platte. Fragt nach Wertliste und Zeitfenster. |
| `M` | `model_health.py` | Bestand der Champions und ihrer Generationen. |
| `E` | `export_signals.py` | CSV-Export. |
| `L` | `signal_log.py` | Die tatsächlich ausgegebenen Signale, neueste zuerst, samt Folgeergebnis, sofern bekannt. |
| `H` | `performance_report.py` | HTML-Bericht. |
| `Q` | `equity_curve.py` | Die Kapitalkurve, die diese Signale ergeben hätten, als PNG. |
| `SG` | `push_signals.py` | Veröffentlicht den letzten Stand nach Supabase für die Landingpage. |

### AUSWERTUNG

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `N` | `news_analyzer.py` | Schlagzeilen je Wert aus rund 30 gewichteten Feeds plus Google News, mit Stimmungswert. Fragt nach dem Wert. |
| `D` | `news_analyzer.py --digest` | Ein marktweiter Überblick statt einer Auswertung je Wert. |
| `R` | `regime_detector.py` | Trend, Volatilität und Momentum je Wert, dazu die Marktbreite. |
| `C` | `correlation_alert.py` | Korrelation zwischen den Werten und das Stressniveau. |
| `WL` | `watchlist.py` | Zeigt und bearbeitet `watchlist.json`, die kleine feste Menge, die der Analyst täglich beurteilt. |
| `P` | `paper_trading.py` | Führt die realen Signale gegen ein Papierdepot, damit eine Ausführungsidee ohne Geld erprobt werden kann. |
| `W1` bis `W4` | `whatif_simulator.py` mit festen Vorgaben | Top-5 oder Top-10, 90 oder 180 Tage, Gleichgewichtung oder Kelly. |
| `W5` | `whatif_simulator.py` | Fragt nach Werten, Tagen und Kapital. |
| `PF` | `performance.py` | Was ein Wert über einen Zeitraum erbracht hat, gegen den Index seiner Klasse: Gesamt- und annualisierte Rendite, Volatilität, maximaler Drawdown, Mehrertrag und Beta. Fragt nach Wert und Zeitfenstern. Liest ausschließlich `market.db`, funktioniert also auch ohne VPN. |
| `MC` | `macro_calendar.py` | Aktualisiert `macro_calendar.json` aus den veröffentlichten Terminplänen der Bank von Russland und der Fed. Zugleich ein Schritt des Tageszyklus; dieser Punkt dient nur der Aktualisierung von Hand. |

### FORSCHUNG

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `RS` | `auto_research.bat` | Eigenes Menü. |
| `AN` | `analyst.py` | Eigenes Menü. |
| `AL` | `auto_loop.py` | Der unbeaufsichtigte Zyklus aus Suche, A/B und Adoption. |
| `ALS` | `auto_loop.py --status` | Fragt, ob die Schleife auch gestoppt werden soll. |
| `LC` | `loop_cycle.py` | Ein täglicher Wartungsdurchlauf. |

### POLITIKEN

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `TP` | `train_timing.py` | Passt die Timing-Regeln der Stufe A an. |
| `TB` | `train_timing.py --stage b` | Der Fitted-Q-Herausforderer. Fragt nach der Zahl der Iterationen. |
| `TO` | `train_timing_online.py` | Ein Online-Takt. Fragt nach dem Anteil der Selbsterhebung. |
| `TL` | `train_levels.py` | Einstiegszone und Stopp. Fragt nach dem Suchbudget. |
| `SZ` | `train_sizing.py` | Positionsgröße bei angeglichenem Risiko. Fragt nach dem Budget. |
| `DR` | `train_direction.py` | Folgen, aussetzen oder umkehren, an REALEN Ergebnissen angepasst. |
| `RC` | `recalibrate_live.py` | Kalibriert die Wahrscheinlichkeiten im Betrieb nach. |
| `OS` | eine der Politiken | Passt eine Politik auf Werten an, für die sie nie bewertet wurde. |
| `PS` | `policy_status.py` | Wie sich die angepassten Politiken auf REALEN Signalen geschlagen haben. |
| `TR` | `train_timing.py --replay` | Wie oft die Entscheidung jeder Schicht richtig war. |

### GENOM

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `AG` | `adopt_genome.py` | Ein Genom übernehmen. |
| `AS` | `adopt_genome.py --show` | Was gerade übernommen ist. |
| `AR` | `adopt_genome.py --revert` | Die Übernahme zurücknehmen. |
| `PA` | Untermenü | Adoption je Wert, Schritt für Schritt. |
| `ABC` | `ab_build.py` | Konfiguriert einen A/B-Test. |
| `ABR` | `ab_build.py --run` | Führt den konfigurierten aus. |

### DIENSTE

| Taste | Startet | Anmerkungen |
| --- | --- | --- |
| `7` | `alert_bot.py` | Telegram-Bot, läuft bis zum Abbruch. |
| `8` | `scheduler.py` | Dienst: Daten alle 6 h, Vorhersagen alle 4 h, tägliche Datenbankprüfung. |
| `9` | `db_check.py` | Schreibgeschützte Prüfung von `market.db`. |
| `F` | `db_check.py --fix` | Repariert Duplikate und Datumsformate. |
| `B` | `db_backup.py` | Legt eine Kopie von `market.db` mit Zeitstempel daneben. Dauert Sekunden und lohnt sich vor jedem `--fix`. |
| `I` | `pip install ...` | Installiert oder repariert den Abhängigkeitssatz. |
| `0` | nichts | Verlässt den Starter. Was in einem eigenen Fenster gestartet wurde, läuft weiter. |

## Täglicher Betrieb

Der Mindestrhythmus besteht aus zwei Menüpunkten:

1. `[4] Daten aktualisieren`, dann `[3] Vorhersage` - oder direkt `[LC]`, das beides verkettet und Abgleich sowie Drifterkennung hinzufügt.
2. `[WU]`, um das Ergebnis im Browser zu lesen.

Danach je nach Frage:

- **Ein bestimmter Wert:** seine Karte in der Weboberfläche, dann `[PF]` für das, was er im interessierenden Zeitraum tatsächlich erbracht hat.
- **Der Zustand des Ganzen:** `[M]` für den Bestand der Champions, `[PS]` für das, was die angepassten Politiken auf realen Signalen geleistet haben.
- **Eine zweite Meinung:** `[AN] -> [R]`; das kostet einen Modellaufruf je beurteiltem Wert und verlangt ein getipptes YES.
- **Eine Verbesserung suchen:** `[RS]`, danach das Urteil mit `[PA] -> 1` lesen. Rechnen Sie damit, dass die meisten Kampagnen nichts ergeben.

**Zwei Warnungen, deren Missachtung teuer wird.** Führen Sie niemals einen Anpasser (`train_payoff.py`, `train_sizing.py`, `train_timing.py`) in der Hauptarbeitskopie aus, während ein Agent läuft: diese Skripte überschreiben ihre Berichtsdatei bei jedem Lauf, und ein Probelauf über zehn Werte hat bereits die Belege für 207 vernichtet. Und starten Sie die vollständige Testsuite nicht in der Hauptkopie, während trainiert wird.

## Training

```bash
python train_hybrid.py                       # alles in einem Prozess
python train_chunked.py                      # ein Block je Prozess, fortsetzbar
GTRADE_ASSETS=SBER,GAZP python train_hybrid.py   # nur diese
```

TensorFlow sammelt über viele Werte hinweg innerhalb eines Prozesses Speicher an, ein vollständiges Nachtraining über 847 Werte läuft auf speicherbegrenzter Hardware daher blockweise. `train_chunked.py` startet je Block einen frischen Prozess und setzt dort fort, wo es aufgehört hat.

Champion-Challenger ist die Vorgabe: ein frisch trainiertes Modell ersetzt den amtierenden Champion nur, wenn es ihn in der Walk-Forward-Auswahl schlägt. Sichern Sie `models/` vor einem vollständigen Nachtraining - rund 5,8 GB.

**Ändern Sie den Trainingscode nicht, während ein blockweiser Lauf aktiv ist.** Jeder Block startet einen frischen Prozess und liest die geänderte Datei mitten im Lauf neu ein.

## Konfiguration

Konfiguriert wird über Umgebungsvariablen, gelesen zuerst aus `.env` und dann aus der echten Umgebung, wobei letztere gewinnt. Die wichtigsten:

| Variable | Wirkung |
| --- | --- |
| `GTRADE_ASSETS` | beschränkt einen Lauf auf diese Werteliste |
| `GTRADE_SEED` | legt den Trainings-Seed fest |
| `GTRADE_ANALYST=0` | schaltet den Analystenagenten vollständig ab, in Konsole wie im Web |
| `GTRADE_ANALYST_TOOL_CALLS` | wie viele zusätzliche Quellen ein Urteil anfordern darf (Vorgabe 2, `0` verbietet es) |
| `GTRADE_SEC_CONTACT` | eine E-Mail-Adresse für den von der SEC verlangten Header; ohne sie liefert `insider_filings` den Hinweis statt eines 403 |
| `SOCKS5_PROXY` | ausgehender Proxy für die Datenquellen |
| `GTRADE_MODEL_DIR` | wo Modelle gelesen und geschrieben werden |

Die vollständige Liste steht im [englischen README](README.md#configuration).

## Projektstruktur

```text
data_engine.py        holt Tages- und Wochenkurse (Yahoo + MOEX) nach market.db
train_hybrid.py       trainiert das Ensemble je Wert + Walk-Forward-Auswahl
train_chunked.py      RAM-schonendes vollständiges Nachtraining
predict.py            Signalradar in der Konsole
backtest.py           Auswertung außerhalb der Stichprobe
webapp.py             FastAPI-Dashboard (app.py = Streamlit)
analyst.py            CLI des Analystenagenten: run / score / backfill
core/analyst/         sein Dossier, der Urteilsparser, die Werkzeug-
                      registrierung, das Protokoll, Kalibrierung und Bewertung
train_payoff.py       passt payoff_stats.json an: was eine Position wert war
performance.py        was ein Wert über einen Zeitraum erbracht hat
core/performance.py   die zugehörige Arithmetik und ihre drei bewussten Verzichte
macro_calendar.py     aktualisiert macro_calendar.json aus BdR- und Fed-Terminen
core/macro.py         diese Parser sowie der Leitzins und seine Richtung
auto_research.py      autonomer Forschungsagent
auto_loop.py          unbeaufsichtigter Zyklus aus Suche / A-B / Adoption
ab_per_asset.py       Schritt 1: welchen Werten ein Genom wirklich geholfen hat
ab_confirm.py         Schritt 2: diese auf ungesehenen Seeds nachmessen
run_gtrade.bat        Windows-Textmenü über die gesamte Pipeline
core/                 gemeinsame Bibliothek
tests/                pytest-Suite (1986 Tests, ~2 min)
```

## Tests

```bash
python -m pytest
ruff check .
```

## Lizenz

PolyForm Noncommercial License 1.0.0. Nichtkommerzielle Nutzung: Forschung, Lehre und private Projekte. Maßgeblich ist der englische Text in [`LICENSE`](LICENSE).

## Haftungsausschluss

Atratus ist ein Forschungs- und Lehrprojekt. Sein Ergebnis besteht aus Modellvorhersagen, **nicht aus Anlageberatung und nicht aus einer Empfehlung, ein Wertpapier zu kaufen oder zu verkaufen**. Keine vergangene Wertentwicklung, ob real oder simuliert, sichert ein künftiges Ergebnis; Backtests tragen Verzerrungen, die dieses Repository ausdrücklich dokumentiert, statt sie zu verbergen.

Märkte bergen Risiken, und Sie können Ihr Kapital ganz oder teilweise verlieren. Die Software wird "wie besehen" bereitgestellt, ohne ausdrückliche oder stillschweigende Gewährleistung. Die Autoren übernehmen keine Haftung für Verluste aus ihrer Nutzung.

Das System führt keine Order aus. Es liefert Signale; jede Anlageentscheidung bleibt Ihre. Führen Sie eigene Recherchen durch und ziehen Sie vor jeder finanziellen Entscheidung eine zugelassene Fachperson hinzu.
