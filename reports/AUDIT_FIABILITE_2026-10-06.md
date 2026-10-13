**Audit de fiabilité — prédiction cacao / café — 6 octobre 2026**

Le projet possède une infrastructure utile : collecte, modèles par horizon, API, tableau de bord, validation chronologique et intervalles. En revanche, la fiabilité de la chaîne actuellement servie n'est pas démontrée par les métriques disponibles. Les priorités sont de corriger la mesure de performance, l'alignement temporel et l'affichage de l'incertitude avant d'ajouter des modèles.

Périmètre : code et artefacts locaux dans `E:/prediction`, y compris les modifications déjà présentes. Aucun réentraînement complet, appel à la production distante ou modification des données en base. Aucun code applicatif modifié par cet audit. Les constats sur le comportement en production décrivent le code local ; la version effectivement déployée n'a pas été vérifiée.

**1. P1 — Le rapport récent teste une période ancienne**

La commande quotidienne impose `--max-origins 48`, puis le validateur conserve les premières origines. Le rapport généré le 6 octobre 2026 couvre des origines du **22 décembre 2020 au 22 novembre 2021**, avec des cibles jusqu'au **3 janvier 2022**. Sa date de création peut donc donner une impression trompeuse de validation récente.

Preuves : [commande quotidienne](E:/prediction/update_system.bat:132), [sélection des origines](E:/prediction/src/validation/walk_forward_validator.py:67), [observations évaluées](E:/prediction/reports/walk_forward/20261006_024613_walk_forward_predictions.csv:1).

Correction : sélectionner explicitement une période récente ayant toutes ses cibles réalisées, conserver aussi des tests sur plusieurs régimes historiques, et publier les dates de début/fin effectivement évaluées. Une limite de calcul ne doit pas sélectionner silencieusement le début de l'historique.

**2. P1 — Le modèle évalué ne bat pas le dernier cours connu**

Recalcul sur exactement les mêmes observations du rapport du 6 octobre. La référence simple prévoit que le prix futur sera égal au prix à l'origine. MAPE = moyenne des erreurs absolues relatives, en pourcentage ; plus petit est meilleur.

| Horizon | Nombre de prévisions | Référence : cours inchangé | XGBoost évalué |
|---|---:|---:|---:|
| J+1 | 46 | **1,18 %** | 1,80 % |
| J+7 | 48 | **2,63 %** | 3,07 % |
| J+14 | 48 | **3,34 %** | 3,84 % |
| J+30 | 48 | **4,75 %** | 4,87 % |

Cela concerne le XGBoost évalué dans ce fichier, pas une mesure du modèle direct actuellement chargé. Les écarts sont descriptifs ; cet échantillon limité et dépendant ne suffit pas pour conclure à une différence statistiquement établie.

Correction : inclure systématiquement cette référence, une référence avec dérive et les erreurs par horizon. Exiger une amélioration hors échantillon documentée avant de promouvoir un modèle plus complexe. La comparaison aux méthodes simples est une pratique standard en prévision : [Hyndman et Athanasopoulos, méthodes de référence](https://otexts.com/fpp3/simple-methods.html).

**3. P1 — Le backtest ne reproduit pas le modèle réellement servi**

Le service privilégie les modèles directs M3, dont la cible actuelle est la variation `close_delta`, avec 34 variables. La validation quotidienne désactive les directs par défaut et évalue le XGBoost à variables figées. Même activer `--direct-hstep` ne suffit pas : le validateur utilise les variables de base, et les calibrateurs continuent de consommer `xgb_pred`. Le sentiment, les garde-fous et les ajustements finaux ne sont pas reproduits ensemble dans les résultats calibrés.

Preuves : [choix du modèle servi](E:/prediction/src/models/improved_price_predictor.py:270), [métadonnées du modèle direct](E:/prediction/models/model_info_direct_horizon_20261006_022402.json), [configuration du validateur](E:/prediction/src/validation/walk_forward_validator.py:35), [entraînement direct du validateur](E:/prediction/src/validation/walk_forward_validator.py:141), [calibration](E:/prediction/scripts/run_walk_forward_validation.py:137).

Correction : partager une seule chaîne de prévision entre validation et service. Enregistrer marché, variables, cible, version du code, version des données et toutes les transformations. Calibrer les erreurs de la prévision finale réellement publiée. Pour le sentiment, reconstruire les nouvelles disponibles à chaque date historique ou le désactiver dans la version dont on revendique les performances.

**4. P1 — Une fuite de cible rend le score de l'hybride trop rassurant**

L'hybride apprend `price[t]` avec des variables qui contiennent déjà ce prix, notamment `price_change_1d[t]` et les moyennes mobiles contemporaines. La formule `price_lag_1 * (1 + price_change_1d)` reconstruit exactement la cible. Le holdout réutilise les clôtures réelles de validation pour calculer ces variables ; son « MAPE holdout 1-step » ne mesure donc pas une véritable prévision à une séance.

Preuves : [cible hybride](E:/prediction/src/models/hybrid_trainer.py:65), [variables contemporaines](E:/prediction/src/models/hybrid_features.py:259), [construction et mesure du holdout](E:/prediction/train_hybrid_improved.py:108). Une vérification synthétique locale reconstruit la cible à moins de `5e-13` près.

Nuance : les modèles directs ont une cible future ; ce constat ne leur est pas imputé. Il concerne le modèle hybride, ses métriques et son utilisation comme repli.

Correction : définir clairement l'instant où les informations sont disponibles. Après la clôture de t, les variables peuvent inclure cette clôture pour prévoir t+h. Pour prévoir la clôture de t avant sa disponibilité, décaler les variables. Utiliser la même convention à l'entraînement, à l'évaluation et dans l'API.

**5. P1 — La couverture des intervalles n'est pas vérifiée indépendamment**

Le même CSV sert à optimiser les poids, construire les marges, puis calculer `empirical_coverage`. Les couvertures proches de 95,8 % sont donc mesurées sur les résidus de calibration, pas sur un test réservé. Elles ne prouvent pas que les bandes annoncées à 90 % couvriront les futurs prix. Il n'y a que 46/48 observations par horizon, et les horizons longs se chevauchent.

Preuves : [réutilisation du CSV](E:/prediction/scripts/run_walk_forward_validation.py:166), [quantile et couverture sur les mêmes résidus](E:/prediction/src/validation/conformal_interval_calibrator.py:82), [marges actuellement configurées](E:/prediction/config/conformal_intervals.json).

Correction : séparer chronologiquement entraînement, choix du modèle, calibration et test final ; exclure les exemples dont les cibles dépassent la frontière autorisée. Mesurer couverture et largeur sur les prévisions arrivées à échéance, par marché/horizon et sur une fenêtre récente. Étudier ensuite une calibration adaptative à la volatilité ; une couverture marginale ne signifie pas une probabilité individuelle de hausse. Références : [découpages temporels scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [Gibbs et Candès, calibration adaptative](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

**6. P1 — L'évaluation en exploitation compare parfois les mauvaises dates et les mauvais marchés**

Les prévisions partent du dernier jour de données et ajoutent des jours ouvrés. Le script d'évaluation calcule `created_at + horizon` en jours calendaires. Même si calcul et dernier cours datent du vendredi 2 octobre 2026, J+1 vise le lundi 5 dans le modèle mais le samedi 3 dans l'évaluation ; J+7 vise le 13 contre le 9. Une collecte en retard augmente encore le décalage.

Le journal de prévisions n'enregistre ni marché, ni date d'origine, ni date cible. L'évaluateur compare toutes les lignes à un index de prix cacao, avec un repli possible sur des prix legacy, et agrège horizons/versions. Le calcul directionnel compare les lignes successives au lieu de comparer chaque réalisation à son prix d'origine. Le tableau de bord café reçoit aussi les métriques de validation cacao et un historique non filtré.

Preuves : [journal API](E:/prediction/src/api/app.py:1321), [évaluation calendaire](E:/prediction/evaluate_predictions.py:93), [agrégation](E:/prediction/evaluate_predictions.py:113), [direction entre lignes](E:/prediction/src/monitoring/performance_monitor.py:154), [métriques globales cacao](E:/prediction/src/api/app.py:1448), [appels du dashboard](E:/prediction/frontend/src/components/SimplifiedMarketDashboard.tsx:201).

Correction : stocker `market`, instrument/contrat, devise/unité, date et prix d'origine, date cible, heure de calcul, version exacte, statut et intervalles. Évaluer sur la date cible persistée et le bon instrument, puis grouper par marché/horizon/version. Calculer la direction à partir du prix d'origine. Utiliser un calendrier de cotation commun ; `BusinessDay` seul ne gère pas les jours de fermeture de la place.

**7. P1 — L'interface présente des scores heuristiques comme des probabilités**

La confiance est calculée à partir de la seule largeur de l'intervalle et bornée entre 35 et 92 %. La probabilité directionnelle part de 62 % par défaut, puis augmente selon l'amplitude prévue. Ce ne sont pas des probabilités calibrées. De plus, un garde-fou peut remplacer la prévision par le dernier cours et renseigner `feature_failure`, sans que la carte affiche cet échec : le résultat ressemble alors à un signal neutre normal.

Preuves : [formules des scores](E:/prediction/frontend/src/lib/marketAnalytics.ts:43), [affichage](E:/prediction/frontend/src/components/dashboard/PredictionHorizonCard.tsx:77), [repli sur le dernier cours](E:/prediction/src/models/hybrid_features.py:452), [diagnostic dans les composants](E:/prediction/src/models/improved_price_predictor.py:467).

Correction : afficher l'erreur historique, la période évaluée et la largeur de l'intervalle séparément. Retirer la probabilité de hausse/baisse tant qu'elle n'a pas été apprise et vérifiée hors échantillon. Lorsqu'un garde-fou se déclenche, afficher « prévision indisponible / modèle dégradé ; dernier cours comme référence », sans transformer l'échec en recommandation HOLD.

**8. P1/P2 — Le nettoyage historique peut effacer les crises et dépend du futur**

Les cours hors moyenne globale ±3 écarts-types sont supprimés avant la validation chronologique. Les données futures influencent donc la sélection des observations passées. Le même filtre s'applique dans le service et peut éliminer un dernier cours valide exceptionnel.

Preuve : [filtre global](E:/prediction/src/models/hybrid_features.py:226), [usage dans le service](E:/prediction/src/models/improved_price_predictor.py:220). Reproduction synthétique : une ancienne observation est supprimée, puis conservée après ajout de données futures, sans modification de cette observation.

Correction : conserver les vrais mouvements extrêmes, vérifier les anomalies par source et cohérence OHLC, mettre en quarantaine les erreurs suspectes et appliquer uniquement des contrôles utilisant l'information disponible à la date considérée.

**Autres ajustements à prévoir**

- **Promotion des modèles :** le service charge les derniers artefacts par nom, et le déploiement sélectionne les derniers fichiers séparément. La validation quotidienne peut échouer sans arrêter le processus. Introduire un manifeste versionné regroupant modèles, variables, données et calibration, puis une promotion explicite après validation, avec retour à la version précédente. Voir [chargement](E:/prediction/src/api/app.py:442), [déploiement](E:/prediction/deploy_models.ps1:195), [validation non bloquante](E:/prediction/update_system.bat:132).
- **Fraîcheur :** distinguer date de cotation, heure de calcul et heure de consultation. Le dashboard met « Dernière MAJ » à l'heure de rechargement, même avec une réponse en cache. Vérifier les séances attendues, signaler une donnée trop ancienne et utiliser un seul snapshot pour le cours affiché et la prévision. Voir [horodatage de l'interface](E:/prediction/frontend/src/components/SimplifiedMarketDashboard.tsx:240).
- **N-HiTS :** sa validation réutilise une instance déjà entraînée, sans remise à zéro explicite. La dépendance installée conserve les poids dans ce chemin. Refaire la validation sur une instance neuve avant de lui attribuer un poids. Les poids cacao actuels de N-HiTS sont nuls, ce qui limite l'impact sur la configuration actuelle. Voir [entraînement puis validation](E:/prediction/train_nhits.py:164), [validateur](E:/prediction/src/validation/nhits_validator.py:91), [poids](E:/prediction/config/ensemble_weights.json).
- **Open interest :** conserver contrat, date de publication, disponibilité et révisions. Le collecteur peut substituer l'OI d'un autre contrat et ne conserve pas l'instant de disponibilité nécessaire à un replay historique. C'est un risque identifié dans le code ; une contamination effective de la base n'a pas été mesurée. Voir [rattachement des statistiques](E:/prediction/src/data_collection/databento_london_collector.py:220), [substitution](E:/prediction/src/data_collection/databento_london_collector.py:408).
- **RSI :** une série strictement croissante produit `NaN` lorsque la perte moyenne vaut zéro, puis peut être imputée à zéro/une ancienne valeur. Traiter ce cas explicitement. Voir [calcul RSI](E:/prediction/src/models/hybrid_features.py:245).
- **Reproductibilité :** `pip check` détecte quatre conflits dans l'environnement local : gotrue/httpx, mlflow/packaging, supafunc/httpx et torch/sympy. Verrouiller un ensemble compatible et vérifier une installation propre. Cela décrit cet environnement local, pas une panne démontrée du serveur.

**Ordre de travail recommandé**

1. **Rendre les mesures fiables.** Corriger les origines du backtest, partager la chaîne de production, ajouter la référence au dernier cours et réparer les dates/marchés du journal. Supprimer les probabilités non calibrées de l'interface.
2. **Refaire l'évaluation.** Tester les modèles directs actuels sur plusieurs périodes, dont une période récente réservée, sans fuite de cible ni nettoyage global. Publier erreurs, direction, couverture, largeur, nombre d'observations, dates et performances du repli.
3. **Sécuriser l'exploitation.** Contrôles de fraîcheur, disponibilité des variables, état dégradé visible, suivi des erreurs réalisées et promotion conditionnelle d'un ensemble d'artefacts cohérent.
4. **Améliorer ensuite les variables.** Tester séparément stocks, récoltes, météo, change, courbe des contrats et sentiment, avec leurs dates réelles de publication. Ne conserver que les ajouts apportant un gain stable hors échantillon. Le [test local M3 contre M3 + USD/GBP](E:/prediction/reports/macro_comparison/m3_vs_usdgbp_latest.md) montre déjà qu'ajouter une variable n'améliore pas automatiquement la prévision.

**Critères de réussite proposés**

Une prévision doit être traçable jusqu'au marché, à la date cible et à la version qui l'a produite. Les performances annoncées doivent provenir de cette même chaîne sur une période non utilisée pour choisir ou calibrer le modèle. L'amélioration face au dernier cours doit être mesurée par horizon, avec son incertitude ; si elle n'est pas démontrée, afficher ce constat et garder la référence simple disponible. Les bandes à 90 % doivent faire l'objet d'un suivi de couverture sur des prévisions réalisées et d'un contrôle de leur largeur. Les données périmées et les replis doivent être visibles.

Si le produit doit servir à des décisions de trading, compléter ensuite par une simulation tenant compte des coûts, du spread, des changements de contrat et des pertes maximales. Un MAPE faible ne démontre pas une stratégie rentable.

**Vérifications réalisées**

- 28 tests existants réussis en 34,42 secondes : validation walk-forward, modèles directs, simulation multi-pas, poids d'ensemble et intervalles conformes. Ces tests attestent leur comportement couvert ; ils ne prouvent pas la qualité prévisionnelle.
- Recalcul indépendant de la référence « dernier cours » et des bornes de dates du CSV du 6 octobre.
- Reproduction de la reconstruction de cible, du filtre non causal et du décalage jours ouvrés/jours calendaires.
- Lecture des métadonnées récentes, calibrations, rapports de comparaison, API, interface, collecte et scripts de déploiement.
- Vérification locale des dépendances avec `pip check`.

Les taux de réussite futurs du modèle direct actuel, la qualité complète de la base distante et les résultats financiers en exploitation restent à mesurer après correction du protocole.
