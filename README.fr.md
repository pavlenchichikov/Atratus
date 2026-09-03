# Atratus

![Atratus](assets/atratus-banner.svg)

[![CI](https://github.com/pavlenchichikov/Atratus/actions/workflows/ci.yml/badge.svg)](https://github.com/pavlenchichikov/Atratus/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Lint: Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey.svg)](LICENSE)

[English](README.md) | [Русский](README.ru.md) | [**Français**](README.fr.md) | [Deutsch](README.de.md)

**Moteur multi-actifs de signaux de trading par apprentissage automatique.** Un ensemble propre à chaque actif (CatBoost + LSTM + Transformer + TCN) sur 847 marchés - crypto, actions américaines / européennes / russes, indices, taux, volatilité, ETF obligataires et sectoriels, forex et matières premières - avec sélection walk-forward, probabilités calibrées, dimensionnement de Kelly, contrôle du risque extrême, tableau de bord FastAPI et un agent de recherche autonome soumis à un contrôle statistique. Des signaux uniquement, avec l'humain dans la boucle : aucune exécution automatique.

> **Avertissement.** Atratus est un projet de recherche et d'enseignement. Ce qu'il produit est un ensemble de prédictions de modèles, **et non un conseil financier ni une recommandation d'acheter ou de vendre un quelconque titre**. Les marchés comportent des risques et vous pouvez perdre de l'argent. Le logiciel est fourni « en l'état », sans garantie d'aucune sorte. Utilisez-le à vos propres risques ; faites vos propres recherches et consultez un professionnel agréé avant toute décision financière. En cas de divergence, la version anglaise et le fichier [`LICENSE`](LICENSE) font foi.

[![Télécharger l'application Android](https://img.shields.io/badge/T%C3%A9l%C3%A9charger-Android%20APK-brightgreen?logo=android&logoColor=white&style=for-the-badge)](https://github.com/pavlenchichikov/Atratus/releases/latest/download/Atratus.apk)

> **Portée de cette traduction.** Ce document couvre l'intégralité de ce qui est nécessaire pour installer, lancer et comprendre le système : fonctionnement, référence complète du menu, usage quotidien, entraînement, configuration et dépannage. Les longs passages de journal de recherche - la théorie derrière le contrôle A/B, la décomposition de la variance, l'historique des expériences - ne sont résumés ici et détaillés que dans le [README anglais](README.md), qui reste le document canonique.

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Comment ça marche](#comment-ça-marche)
- [Interface web](#interface-web)
- [Agent de recherche automatique](#agent-de-recherche-automatique)
- [Adoption par actif](#adoption-par-actif)
- [Agent analyste](#agent-analyste)
- [Boucle auto-entretenue](#boucle-auto-entretenue)
- [Prérequis](#prérequis)
- [Environnement et GPU](#environnement-et-gpu)
- [Démarrage rapide](#démarrage-rapide)
- [Le menu du lanceur](#le-menu-du-lanceur)
- [Usage quotidien](#usage-quotidien)
- [Entraînement](#entraînement)
- [Configuration](#configuration)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Licence](#licence)

## Fonctionnalités

- **Un modèle par actif, et non un modèle pour le marché.** Chaque actif entraîne son propre ensemble de quatre membres (CatBoost, LSTM, Transformer, TCN), et le champion est choisi par un backtest walk-forward tenant compte des commissions, du slippage et d'un embargo contre les fuites de données. C'est la décision d'architecture centrale, et les chiffres ci-dessous en sont le coût et la contrepartie.

  Au 2026-09-03 : **847 tickers dans `FULL_ASSET_MAP`, dont 827 entraînés et à jour, et 20 en attente d'un premier entraînement.** Cet écart n'est pas une dégradation. Un ticker entre dans la carte dès qu'il mérite d'être suivi, alors que l'entraîner est un acte distinct et coûteux ; la carte devance donc toujours `models/`. Ces vingt-là sont des cotations récentes et des ajouts d'indices (`ARB`, `CRWV`, `NBIS`, `SNOW`, `GEV`, `TAO`, `MBNK`, `RENI`, `KLVZ`, `DIAS`, `PRMD`, `EGX30`, `WIG20` et sept autres) ; l'entrée `[5F]` du menu comble l'écart.

  Il existe aussi le cas inverse : **quatre jeux de modèles sur le disque dont l'entrée dans la carte a disparu** (`avb`, `eqr`, `wbs`, `brkb`). Trois sont des actifs retirés de la carte. Le quatrième est un renommage : Berkshire est passé de `BRKB` à `BRK-B`, dont le nom de fichier de modèle est `brk_b` ; le modèle entraîné s'est donc retrouvé orphelin et l'actif compte désormais comme non entraîné. Renommer une clé sans déplacer les fichiers de modèle produit cela en silence, et c'est `[M] Model Health` qui le révèle.

- **Probabilités calibrées, pas des scores bruts.** Chaque probabilité passe par une calibration isotonique ajustée hors échantillon, de sorte qu'un 0,70 signifie réellement sept fois sur dix.

- **Sélection walk-forward avec embargo.** Le champion n'est jamais choisi sur les données qui l'ont entraîné, et un embargo retire les barres situées à la frontière de chaque pli pour empêcher l'étiquette de fuir en arrière.

- **Contrôle du risque de queue.** Dimensionnement de Kelly fractionné, limites de perte et de drawdown, et une porte de type Taleb qui refuse de prendre position dans les régimes où la distribution des pertes cesse d'être exploitable.

- **Un agent de recherche autonome qui a le droit de dire non.** Il cherche de nouvelles configurations, les mesure sur un jeu de validation qu'il n'a jamais optimisé, et n'adopte que ce qui franchit un seuil statistique fixé à l'avance. La plupart des campagnes ne débouchent sur rien, et c'est le comportement attendu.

- **Un agent analyste qui ne voit jamais l'avis du modèle.** Un second regard, formé à partir des mêmes données brutes mais sans accès à la probabilité de l'ensemble, au signal émis, à la décision de timing ni au dimensionnement.

- **Tout ce qui est mesuré est consigné.** `prediction_log`, `level_log`, `guru_log`, `analyst_log` : chaque avis est écrit avant que son issue n'existe, et un processus de rapprochement les note ensuite. Un chiffre que le projet ne peut pas vérifier après coup n'est pas affiché comme s'il l'était.

## Comment ça marche

```
data_engine.py    barres quotidiennes depuis Yahoo Finance et MOEX -> market.db
train_hybrid.py   quatre membres par actif, sélection walk-forward -> models/
predict.py        note chaque actif, écrit prediction_log
webapp.py         tableau de bord FastAPI ; app.py est la version Streamlit
```

Le cœur du système est un ensemble par actif. Les quatre membres voient les mêmes caractéristiques et votent ; un méta-modèle apprend le poids de chacun sur les plis de validation. Le résultat est une probabilité calibrée, convertie en signal par des seuils propres à chaque actif, puis filtrée par le régime, la porte de risque et les politiques ajustées (timing, dimensionnement, niveaux d'entrée).

Rien ne s'exécute automatiquement. Le système produit des signaux ; la décision reste humaine.

## Interface web

`webapp.py` sert un tableau de bord FastAPI : le radar de tous les actifs notés aujourd'hui, une fiche par actif avec ses niveaux et son historique, les pages de recherche et d'expérience, la page de l'analyste et l'état de la boucle quotidienne.

```bash
uvicorn webapp:app --port 8000
```

Ou l'entrée `[WU]` du menu, qui l'ouvre directement sur le tableau de bord.

## Agent de recherche automatique

`auto_research.py` automatise la recherche de meilleures configurations. Un proposeur suggère un « génome » - un jeu de leviers sur les caractéristiques, les étiquettes et le modèle - un pré-filtre bon marché fondé sur CatBoost seul écarte les perdants évidents, et ce qui survit est mesuré contre la référence mise en cache.

Le proposeur par défaut est une recherche évolutionnaire, sans LLM ni clé d'API. `GTRADE_AR_PROPOSER=llm` fait appel à un modèle à la place (Anthropic par défaut, OpenAI, ou tout point d'accès compatible OpenAI, y compris un **Ollama local** via `GTRADE_AR_LLM=ollama`).

Le point important, et il est contre-intuitif : **la plupart des campagnes ne trouvent rien, et c'est le résultat attendu.** L'adoption exige une significativité statistique, un effet au-dessus d'un plancher pratique, et la preuve que le gain n'a pas été payé en affamant les membres neuronaux. Un verdict HOLD est une mesure, pas un échec.

Le détail complet - la base de score, l'illumination quality-diversity, la puissance du contrôle A/B et pourquoi elle est le facteur limitant - se trouve dans le [README anglais](README.md#auto-research-agent).

## Adoption par actif

L'effet d'un génome n'est pas le même sur tous les actifs. Mesuré le 2026-09-02 : le candidat qui a **échoué** son contrôle à -0,30 sur 40 actifs valait **+1,20 sur RTX** et **-3,84 sur ROSN**, les deux confirmés ensuite sur des graines que la sélection n'avait jamais vues. L'adopter partout ou nulle part jette ces deux faits, donc un actif peut conserver le génome mesuré sur *lui* pendant que le reste demeure sur l'adoption globale.

Trois étapes, dans cet ordre, chacune répondant à une question dont la suivante a besoin :

1. **Chercher les différences par actif** (`ab_per_asset.py`, gratuit, n'entraîne rien). Se lit au rapport, pas à la taille : lors du premier passage, l'actif dont l'erreur type était la plus grande est celui qui a ensuite changé de signe.
2. **Confirmer sur des graines fraîches** (`ab_confirm.py`, plusieurs heures, entraîne). Les extrêmes sont surestimés du seul fait d'avoir été choisis : les trois sélectionnés le 2026-09-02 n'ont conservé que **30 %** de leur effet mesuré.
3. **Adopter ce qui a survécu** (`adopt_genome.py --asset ACTIF --evidence TEXTE`). L'outil refuse sans preuve, et la preuve doit être la **réplication**, jamais le passage qui a sélectionné l'actif.

Rendement observé sur trois campagnes : environ un extrême sélectionné sur trois survit à sa réplication.

**Lire un passage ultérieur face à ce qui est déjà adopté.** Dès qu'un actif possède son propre génome, la question n'est plus « ce candidat est-il bon » mais « est-il meilleur que ce que cet actif fait déjà tourner ». Le bras de référence y répondait déjà - il passe par le même `config.py` - mais rien sur la ligne ne le disait. L'étape 1 affiche désormais une colonne `on` (`own` ou `global`), nomme explicitement tout actif adopté que le candidat dégraderait, et avertit lorsque la comparaison n'est pas celle que l'on croit : une adoption datée d'après la mesure du bras de référence signifie que la ligne de base la précède, et le gain serait compté deux fois.

Deux conséquences valent mieux d'être attendues que découvertes : **chaque adoption relève la barre** (un actif adopté a déjà pris son gain, donc un candidat ultérieur y paraîtra moins bon), et **le backtest ne peut pas dire qu'une adoption a aidé**. Seul le chiffre réel le peut ; `core/adoption_ledger.py` enregistre la précision réelle de chaque actif au moment de l'adoption, parce qu'ensuite elle est irrécupérable.

## Agent analyste

Un second avis sur chaque actif, formé **sans jamais voir l'avis de l'ensemble**. Il ne voit ni la probabilité du modèle, ni le signal émis, ni l'action de timing, ni le dimensionnement ; deux tests l'imposent, l'un balayant le dossier sérialisé à la recherche de clés interdites, l'autre figeant l'ensemble exact des clés pour qu'aucun champ ne puisse être ajouté sans être déclaré.

Il lit ce que le projet calcule déjà - 80 champs répartis en douze blocs nommés : prix et mouvement, position du prix dans ses propres unités de volatilité, l'année écoulée face à son indice, les flux, le marché dans lequel il a bougé, son régime et son secteur, les fondamentaux, les titres de presse bruts, le calendrier et le taux directeur, le verdict du Guru Council, et ses propres appels passés sur cet actif.

**Un champ que l'invite ne demande pas est un champ que le modèle ne lit pas.** Mesuré sur les 35 premiers jugements : sur les 21 champs que la liste d'instructions nommait, 16 ont été cités comme preuve ; sur les 39 autres, neuf, le plus souvent une seule fois. Aucun titre de presse n'a été lu. La liste nomme désormais 65 des 80 champs, et chaque exécution imprime sa propre couverture.

**Il renvoie un jugement discret, jamais un pourcentage** : une direction, une conviction de 1 à 5, un régime de volatilité attendu, le risque le plus susceptible de le démentir, et son raisonnement. Un nombre libre produit par un modèle est mesurable mais incorrigible, car chaque réponse est unique et aucune case n'accumule jamais l'historique nécessaire pour la recalibrer.

Le pourcentage affiché sur la fiche n'est donc **pas le chiffre de l'analyste** : c'est ce que cette case de jugement a historiquement rapporté, et il peut contredire l'appel qui le surmonte. Ce chiffre est net de la dérive du marché : sur juin-septembre 2026, la classe russe affichait un rendement brut à l'achat de -0,116 ATR, dont -0,101 n'était que la baisse générale du marché.

**Sources qu'il peut demander.** Au-delà du dossier qu'on lui remet, l'analyste peut *réclamer* des éléments avant de trancher : `insider_filings` (les opérations que les dirigeants ont **déclarées** à la SEC via le formulaire 4) et `news_search` (les flux du projet sur une requête qu'il choisit). Trois règles préservent la reproductibilité : chaque appel et son résultat sont **enregistrés** sur la ligne du jugement ; chaque outil déclare s'il **respecte une date passée**, faute de quoi il est refusé lors d'une exécution rembobinée ; et le registre est une **liste blanche**, jamais un accès libre à une URL quelconque.

Un outil peut renvoyer de la matière. Il ne peut pas renvoyer la conclusion d'autrui : consensus des analystes, objectifs de cours et notations de courtiers sont exclus par décision, et `tools.register()` lève une exception plutôt que d'en accepter un. L'analyste existe pour se forger son propre avis, et un consensus se consulte très bien soi-même.

**Rien de ce qu'il dit n'est encore tenu pour acquis.** `analyst.py score` le mesure face à trois références et affiche SHIP ou HOLD. SHIP exige au moins 500 jugements notés, une couverture d'intervalle entre 0,75 et 0,85, un contrôle par permutation qui s'effondre, et une erreur inférieure aux deux références. Un HOLD est un résultat, et les critères ont été fixés avant l'arrivée du moindre chiffre.

## Boucle auto-entretenue

`loop_cycle.py` exécute le pipeline quotidien sûr (données, calendrier macro, prédiction, rapprochement, notation des jugements de l'analyste) et examine chaque actif à la recherche d'une dérive : précision glissante sous un plancher, écart par rapport à la référence d'entraînement, âge du modèle, ou données périmées. Les propositions apparaissent sur `/loop`.

**La boucle ne réentraîne jamais d'elle-même ; un réentraînement attend toujours votre approbation.** Enregistrez `run_loop.bat` dans le Planificateur de tâches pour une exécution quotidienne. Les seuils de dérive vivent dans `core/drift.py`.

## Prérequis

- **Python 3.12** (3.11+ fonctionne probablement ; c'est 3.12 que la CI utilise).
- **Système :** Linux, macOS ou Windows. Sous Windows, un GPU exige l'environnement épinglé décrit ci-dessous : TensorFlow ne publie plus que des wheels Windows sans CUDA depuis la 2.11, donc une installation par défaut ne voit jamais la carte.
- **Disque :** ~8 Go libres - modèles entraînés (~5,8 Go pour 831 actifs entraînés) plus `market.db` (~310 Mo). Le simple service en demande bien moins.
- **RAM :** 8 Go suffisent pour le tableau de bord et `predict.py` (aucun TensorFlow au moment du service). Entraîner l'univers complet demande ~16 Go, ou bien par tranches d'une quinzaine d'actifs (`GTRADE_ASSETS`) sur une machine plus modeste.
- **GPU :** facultatif mais utile. Sur une RTX 2050, un actif s'entraîne en 158 s contre 2850 à 10480 s pour le même actif sur un processeur à 12 fils.
- **Réseau :** accès sortant vers Yahoo Finance et MOEX (`SOCKS5_PROXY` pris en charge).

## Environnement et GPU

Sous Windows, le projet tourne dans un environnement conda dédié. Ce n'est pas une préférence : **TensorFlow a abandonné le support CUDA natif sous Windows après la 2.10**, donc toute wheel à partir de la 2.11 est sans GPU quelle que soit la carte, et la 2.10 n'existe pas pour Python 3.11+. La combinaison épinglée est donc Python 3.10 avec TensorFlow 2.10.

Rien à préparer : n'importe quel lanceur crée l'environnement au premier usage puis l'active.

```bat
auto_research.bat        :: agent de recherche
run_gtrade.bat           :: menu principal
call activate_env.bat    :: juste l'environnement, pour une session manuelle
```

Trois fichiers, une responsabilité chacun : `env_config.bat` (le seul endroit qui nomme l'environnement et fige les versions), `activate_env.bat` (trouve conda, crée, active, vérifie) et `setup_gpu.bat` (l'installeur, qui détecte la carte et choisit CUDA/cuDNN en conséquence).

**Important :** le menu lui-même reste sur le Python de base. Tout ce qui entraîne passe par `run_in_env.bat` dans un processus enfant, afin qu'un service ultérieur ne perde pas silencieusement ses champions neuronaux.

## Démarrage rapide

```bash
pip install -r requirements.txt
cp .env.example .env          # jeton Telegram, proxy si nécessaire

python data_engine.py         # télécharger les données de marché
python train_hybrid.py        # entraîner les modèles
python predict.py             # signaux en console
streamlit run app.py          # tableau de bord
```

`run_gtrade.bat` ouvre un menu texte au-dessus de tout cela. `python db_check.py` réalise un audit en lecture seule de `market.db` (`--fix` répare les doublons et les formats de date). `python scheduler.py` tourne en démon : données toutes les 6 h, prédictions toutes les 4 h, un contrôle de base quotidien.

## Le menu du lanceur

`run_gtrade.bat` est la porte d'entrée. Les touches sont insensibles à la casse ; Entrée seule à une sous-invite reprend la valeur par défaut affichée entre crochets.

### QUOTIDIEN

| Touche | Lance | Notes |
| --- | --- | --- |
| `1` | `data_engine.py`, puis `train_hybrid.py`, puis `predict.py` | Le cycle complet. Plusieurs heures. |
| `2` | `streamlit run app.py` | Le tableau de bord Streamlit. |
| `3` | `predict.py` | Note chaque actif et écrit `prediction_log`. |
| `4` | `data_engine.py` | Les barres du jour uniquement. |
| `WU` | `uvicorn webapp:app --port 8000` | L'interface FastAPI, ouverte sur le tableau de bord. |

### ENTRAÎNEMENT

| Touche | Lance | Notes |
| --- | --- | --- |
| `5` | `train_hybrid.py` | Tous les actifs dans un seul processus. |
| `5C` | `train_chunked.py` | Un processus neuf par tranche, reprenable, champion-challenger. |
| `5R` | `train_hybrid.py` sur une liste | Demande quels actifs, et s'il faut forcer la promotion. |
| `5F` | `model_health.py --list`, puis `train_chunked.py` | Demande : compléter les actifs sans champion, réparer les dégradés, ou les deux. |
| `T` | `optuna_tune.py` | Recherche d'hyperparamètres par actif. |

### SIGNAUX

| Touche | Lance | Notes |
| --- | --- | --- |
| `6` | `backtest.py` | Backtest walk-forward sur les champions présents sur le disque. Demande la liste des actifs et la fenêtre. |
| `M` | `model_health.py` | Inventaire des champions et de leurs générations. |
| `E` | `export_signals.py` | Export CSV. |
| `L` | `signal_log.py` | Les signaux réellement émis, les plus récents en tête, avec la suite lorsqu'elle est connue. |
| `H` | `performance_report.py` | Rapport HTML. |
| `Q` | `equity_curve.py` | La courbe de capital qu'auraient produite ces signaux, en PNG. |
| `SG` | `push_signals.py` | Publie le dernier instantané sur Supabase pour le site vitrine. |

### ANALYSE

| Touche | Lance | Notes |
| --- | --- | --- |
| `N` | `news_analyzer.py` | Titres par actif issus d'une trentaine de flux pondérés plus Google News, avec une lecture de tonalité. Demande l'actif. |
| `D` | `news_analyzer.py --digest` | Un digest de marché unique au lieu d'une lecture par actif. |
| `R` | `regime_detector.py` | Tendance, volatilité et momentum par actif, plus l'ampleur du marché. |
| `C` | `correlation_alert.py` | Corrélation inter-actifs et lecture du stress. |
| `WL` | `watchlist.py` | Affiche et modifie `watchlist.json`, le petit ensemble fixe que l'analyste juge chaque jour. |
| `P` | `paper_trading.py` | Confronte les signaux réels à un portefeuille papier, pour tester une idée d'exécution sans argent. |
| `W1` à `W4` | `whatif_simulator.py` avec des préréglages | Top-5 ou top-10, 90 ou 180 jours, pondération égale ou Kelly. |
| `W5` | `whatif_simulator.py` | Demande les actifs, les jours et le capital. |
| `PF` | `performance.py` | Ce qu'un actif a rapporté sur une période, face à l'indice de sa classe : rendement total et annualisé, volatilité, drawdown maximal, excès et bêta. Demande l'actif et les fenêtres. Ne lit que `market.db`, donc fonctionne même sans VPN. |
| `MC` | `macro_calendar.py` | Rafraîchit `macro_calendar.json` depuis les calendriers publiés de la Banque de Russie et de la Fed. C'est aussi une étape du cycle quotidien ; cette entrée ne sert qu'à le faire à la main. |

### RECHERCHE

| Touche | Lance | Notes |
| --- | --- | --- |
| `RS` | `auto_research.bat` | Son propre menu. |
| `AN` | `analyst.py` | Son propre menu. |
| `AL` | `auto_loop.py` | Le cycle non surveillé recherche / A-B / adoption. |
| `ALS` | `auto_loop.py --status` | Demande s'il faut aussi arrêter la boucle. |
| `LC` | `loop_cycle.py` | Un passage de maintenance quotidien. |

### POLITIQUES

| Touche | Lance | Notes |
| --- | --- | --- |
| `TP` | `train_timing.py` | Ajuste les règles de timing de l'étape A. |
| `TB` | `train_timing.py --stage b` | Le challenger fitted-Q. Demande le nombre d'itérations. |
| `TO` | `train_timing_online.py` | Un tic en ligne. Demande la part d'auto-collecte. |
| `TL` | `train_levels.py` | Zone d'entrée et stop. Demande le budget de recherche. |
| `SZ` | `train_sizing.py` | Dimensionnement à exposition égalisée. Demande le budget. |
| `DR` | `train_direction.py` | Suivre, s'abstenir ou inverser, ajusté sur les résultats RÉELS. |
| `RC` | `recalibrate_live.py` | Recalibre les probabilités en production. |
| `OS` | une des politiques | Réajuste une politique sur des actifs jamais notés pour elle. |
| `PS` | `policy_status.py` | Ce que les politiques ajustées ont donné sur les signaux RÉELS. |
| `TR` | `train_timing.py --replay` | À quelle fréquence la décision de chaque couche était juste. |

### GÉNOME

| Touche | Lance | Notes |
| --- | --- | --- |
| `AG` | `adopt_genome.py` | Adopter un génome. |
| `AS` | `adopt_genome.py --show` | Ce qui est adopté en ce moment. |
| `AR` | `adopt_genome.py --revert` | Annuler l'adoption. |
| `PA` | sous-menu | Adoption par actif, étape par étape. |
| `ABC` | `ab_build.py` | Configure un contrôle A/B. |
| `ABR` | `ab_build.py --run` | Lance celui qui est configuré. |

### SERVICES

| Touche | Lance | Notes |
| --- | --- | --- |
| `7` | `alert_bot.py` | Bot Telegram, tourne jusqu'à l'arrêt. |
| `8` | `scheduler.py` | Démon : données toutes les 6 h, prédictions toutes les 4 h, contrôle quotidien de la base. |
| `9` | `db_check.py` | Audit en lecture seule de `market.db`. |
| `F` | `db_check.py --fix` | Répare les doublons et les formats de date. |
| `B` | `db_backup.py` | Place une copie de `market.db` à côté, horodatée. Prend quelques secondes et vaut d'être fait avant tout `--fix`. |
| `I` | `pip install ...` | Installe ou répare le jeu de dépendances. |
| `0` | rien | Quitte le lanceur. Ce qui a été lancé dans sa propre fenêtre continue de tourner. |

## Usage quotidien

Le rythme minimal tient en deux entrées :

1. `[4] Mise à jour des données` puis `[3] Prédire`, ou directement `[LC]` qui enchaîne les deux et y ajoute le rapprochement et la détection de dérive.
2. `[WU]` pour lire le résultat dans le navigateur.

Ensuite, selon ce que vous cherchez :

- **Un actif en particulier :** sa fiche dans l'interface web, puis `[PF]` pour ce qu'il a réellement rapporté sur la période qui vous intéresse.
- **La santé de l'ensemble :** `[M]` pour l'inventaire des champions, `[PS]` pour ce que les politiques ajustées ont donné sur les signaux réels.
- **Un second avis :** `[AN] -> [R]`, qui coûte un appel de modèle par actif jugé et demande un YES tapé à la main.
- **Chercher une amélioration :** `[RS]`, puis lire le verdict avec `[PA] -> 1`. Attendez-vous à ce que la plupart des campagnes ne donnent rien.

**Deux avertissements qui coûtent cher lorsqu'on les ignore.** N'exécutez jamais un ajusteur (`train_payoff.py`, `train_sizing.py`, `train_timing.py`) dans la copie de travail principale pendant qu'un agent tourne : ces scripts réécrivent leur fichier de rapport à chaque exécution, et un passage d'essai sur dix actifs a déjà détruit les données de 207. Et ne lancez pas la suite de tests complète dans la copie principale pendant un entraînement.

## Entraînement

```bash
python train_hybrid.py                       # tout, dans un seul processus
python train_chunked.py                      # une tranche par processus, reprenable
GTRADE_ASSETS=SBER,GAZP python train_hybrid.py   # seulement ceux-là
```

TensorFlow accumule de la mémoire d'un actif à l'autre au sein d'un même processus, donc un réentraînement complet sur 847 actifs, sur une machine à mémoire limitée, se fait par tranches. `train_chunked.py` démarre un processus neuf par tranche et reprend là où il s'était arrêté.

Le mode champion-challenger est le défaut : un modèle fraîchement entraîné ne remplace le champion en place que s'il le bat sur la sélection walk-forward. Sauvegardez `models/` avant un réentraînement complet - environ 5,8 Go.

**Ne modifiez pas le code d'entraînement pendant qu'une exécution par tranches est en cours.** Chaque tranche démarre un processus neuf, qui relit donc le fichier modifié à mi-parcours.

## Configuration

La configuration se fait par variables d'environnement, lues depuis `.env` puis depuis l'environnement réel, ce dernier l'emportant. Les principales :

| Variable | Effet |
| --- | --- |
| `GTRADE_ASSETS` | limite une exécution à cette liste d'actifs |
| `GTRADE_SEED` | fixe la graine d'entraînement |
| `GTRADE_ANALYST=0` | coupe entièrement l'agent analyste, en console comme sur le web |
| `GTRADE_ANALYST_TOOL_CALLS` | nombre de sources supplémentaires qu'un jugement peut demander (2 par défaut, `0` interdit) |
| `GTRADE_SEC_CONTACT` | une adresse e-mail pour l'en-tête que la SEC exige ; sans elle, `insider_filings` renvoie la consigne au lieu d'un 403 |
| `SOCKS5_PROXY` | proxy sortant pour les sources de données |
| `GTRADE_MODEL_DIR` | où lire et écrire les modèles |

La liste complète figure dans le [README anglais](README.md#configuration).

## Structure du projet

```text
data_engine.py        récupère les cours quotidiens (Yahoo + MOEX) vers market.db
train_hybrid.py       entraîne l'ensemble par actif + sélection walk-forward
train_chunked.py      réentraînement complet économe en RAM
predict.py            radar de signaux en console
backtest.py           évaluation hors échantillon
webapp.py             tableau de bord FastAPI (app.py = Streamlit)
analyst.py            CLI de l'agent analyste : run / score / backfill
core/analyst/         son dossier, l'analyseur de jugements, le registre
                      d'outils, le journal, la calibration et le scoreur
train_payoff.py       ajuste payoff_stats.json : ce qu'une position a rapporté
performance.py        ce qu'un actif a rapporté sur une période, face à son indice
core/performance.py   l'arithmétique correspondante, et ses trois refus assumés
macro_calendar.py     rafraîchit macro_calendar.json depuis la BdR et la Fed
core/macro.py         ces analyseurs, plus le taux directeur et sa direction
auto_research.py      agent de recherche autonome
auto_loop.py          cycle non surveillé recherche / A-B / adoption
ab_per_asset.py       étape 1 : quels actifs un génome a réellement aidés
ab_confirm.py         étape 2 : les remesurer sur des graines inédites
run_gtrade.bat        menu texte Windows sur tout le pipeline
core/                 bibliothèque partagée
tests/                suite pytest (1986 tests, ~2 min)
```

## Tests

```bash
python -m pytest
ruff check .
```

## Licence

PolyForm Noncommercial License 1.0.0. Usage non commercial : recherche, enseignement et projets personnels. Le texte anglais du fichier [`LICENSE`](LICENSE) fait foi.

## Avertissement

Atratus est un projet de recherche et d'enseignement. Ce qu'il produit est un ensemble de prédictions de modèles, **et non un conseil financier ni une recommandation d'acheter ou de vendre un quelconque titre**. Aucune performance passée, réelle ou simulée, ne garantit un résultat futur ; les backtests comportent des biais que ce dépôt documente explicitement plutôt que de les masquer.

Les marchés comportent des risques et vous pouvez perdre tout ou partie de votre capital. Le logiciel est fourni « en l'état », sans garantie d'aucune sorte, expresse ou implicite. Les auteurs déclinent toute responsabilité pour les pertes découlant de son utilisation.

Le système n'exécute aucun ordre. Il produit des signaux ; chaque décision d'investissement reste la vôtre. Faites vos propres recherches et consultez un professionnel agréé avant toute décision financière.
