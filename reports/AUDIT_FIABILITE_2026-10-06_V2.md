**Nouvel audit de fiabilité — cacao et café — 6 octobre 2026**

**Avis : les corrections améliorent le projet, mais la fiabilité des prévisions publiées reste à démontrer.** Le travail a corrigé plusieurs défauts du premier audit. Les points bloquants sont maintenant surtout la validité du contrôle qualité, la correspondance entre le backtest et le service, et la mise en exploitation d'un ensemble cohérent de modèles et de calibrations.

Audit de l'état local, avec les changements non commités. Les fichiers ont évolué entre l'interruption et la reprise : cette version tient notamment compte de l'ajout de `served_forecast.py` au déploiement et du nouvel entraînement cacao `20261006_213603`. Aucun accès à la base distante, aucun déploiement et aucune correction applicative réalisés dans cet audit. La migration en base et la version réellement exécutée sur le serveur ne sont pas confirmées.

**Ce qui a réellement progressé**

| Sujet du premier audit | État du code actuel | Limite restante |
|---|---|---|
| Les premières origines étaient retenues | Le mode récent retient maintenant les dernières ; un rapport historique distinct est prévu | Aucun nouveau rapport issu de cette chaîne trouvé au moment de la vérification |
| Absence de référence simple | Cours inchangé et dérive sont ajoutés | La comparaison accepte des échantillons différents |
| Cible contemporaine de l'hybride | L'entraînement vise maintenant la clôture suivante | Les anciens artefacts doivent être distingués des nouveaux |
| Suppression des prix extrêmes | Le filtre statistique global est retiré de `clean_price_dataframe` | Il existe toujours dans `train_nhits.py` |
| Marchés et dates mélangés | Journal enrichi ; évaluation groupée par marché/horizon/version ; direction calculée depuis le prix d'origine | Le journal peut encore utiliser un autre snapshot que le modèle |
| Confiance et probabilité des cartes | Pourcentages artificiels retirés des cartes ; échec affiché « Prévision indisponible » | Les probabilités des scénarios subsistent |
| Statistiques cacao affichées pour le café | Validation et historique reçoivent le marché | L'endpoint historique `/performance` reste global |
| Module partagé manquant au déploiement | `served_forecast.py` et le moniteur sont maintenant inclus | L'acceptation statistique ne commande toujours pas le déploiement |

Preuves : [origines récentes](E:/prediction/src/validation/walk_forward_validator.py:80), [cible future](E:/prediction/src/models/hybrid_trainer.py:67), [nettoyage](E:/prediction/src/models/hybrid_features.py:214), [carte](E:/prediction/frontend/src/components/dashboard/PredictionHorizonCard.tsx:59), [appels par marché](E:/prediction/frontend/src/components/SimplifiedMarketDashboard.tsx:189), [déploiement](E:/prediction/deploy_models.ps1:303).

**1. P1 — Un modèle refusé par la validation peut quand même être chargé et déployé**

La nouvelle variable `promoted` contrôle uniquement l'écriture des poids et des marges de calibration. Les modèles entraînés sont déjà enregistrés ; l'API et le déploiement choisissent les derniers fichiers par leur nom. Le batch poursuit aussi après un échec du walk-forward. Une mauvaise nouvelle version peut donc remplacer l'ancienne tout en conservant ses anciennes marges.

Preuves : [portée de la promotion](E:/prediction/scripts/run_walk_forward_validation.py:165), [échec non bloquant](E:/prediction/update_system.bat:132), [chargement des derniers modèles](E:/prediction/src/api/app.py:446), [sélection du déploiement](E:/prediction/deploy_models.ps1:199), [déploiement final](E:/prediction/update_system.bat:296).

Correction : entraîner dans un répertoire candidat ; valider puis activer un manifeste contenant l'ensemble exact modèles/variables/cibles/calibration. Un refus ou une erreur doit conserver intégralement le manifeste actif. Tester explicitement qu'un candidat refusé n'est jamais sélectionné au redémarrage.

**2. P1 — Le contrôle qualité peut approuver une comparaison faussée et une couverture nulle**

`beats_naive_on_all_horizons` calcule séparément les métriques du modèle et de la référence. Chaque calcul supprime ses propres valeurs manquantes. Les échecs du modèle peuvent ainsi disparaître de son score, tandis que la référence reste évaluée sur ces cas difficiles. Il n'y a pas de nombre minimal significatif d'observations, ni de liste explicite des horizons attendus.

Reproduction synthétique avec le code actuel :

- Deux observations, prix d'origine `[100, 100]`, réalisations `[101, 200]`.
- Modèle `[102, NaN]`, référence `[100, 100]`.
- Score modèle : **0,99 % sur une observation** ; référence : **25,50 % sur deux observations**.
- Le contrôle retourne **True**, alors que sur l'unique observation commune le modèle égale seulement la référence.

Deuxième reproduction : des marges nulles issues de 24 erreurs de calibration nulles, puis 24 erreurs test non nulles mais inférieures à la référence. La couverture test vaut **0 %**, et le contrôle de promotion retourne encore **True**, car cette couverture n'intervient pas dans la décision.

Preuves : [comparaison](E:/prediction/src/validation/served_backtest.py:37), [suppression séparée des valeurs manquantes](E:/prediction/src/validation/metrics.py:18), [couverture calculée puis ignorée dans la décision](E:/prediction/scripts/run_walk_forward_validation.py:120), [exceptions converties en prévisions manquantes](E:/prediction/src/validation/walk_forward_validator.py:228).

Correction : comparer sur les mêmes observations, rendre explicites tous les échecs/abstentions et leur fréquence, exiger les horizons attendus et un effectif minimal défini à l'avance. Vérifier également couverture, largeur et qualité des données avant activation. Les seuils doivent être fixés avant consultation du test, avec une incertitude adaptée aux séries temporelles.

**3. P1 — La séparation calibration/test laisse passer des réalisations futures**

La séparation porte seulement sur la date d'origine. Pour J+30, certaines prévisions émises dans la partie calibration n'arrivent à échéance qu'après le début du test. Leurs erreurs n'auraient donc pas été connues pour calibrer les premières prévisions du test.

Reproduction : appliquer le nouveau découpage au CSV local existant fait démarrer le test le **14 juin 2021**, tandis que la calibration utilise des réalisations allant jusqu'au **19 juillet 2021**. Cinq des 24 résidus J+30 dépassent le début du test, ainsi que deux à J+14 et un à J+7. Ces dates illustrent le défaut du découpage ; ce n'est pas un nouveau backtest du modèle actuel.

Preuve : [split_chronological](E:/prediction/src/validation/served_backtest.py:21).

Correction : ne calibrer qu'avec les cibles déjà réalisées avant l'instant de calcul de chaque prévision test. Prévoir une séparation tenant compte des dates cibles, ou une calibration glissante qui n'absorbe que les erreurs arrivées à échéance. Tester cette propriété pour chaque horizon.

**4. P1 — La colonne « prévision publiée » ne reproduit toujours pas toute la chaîne servie**

Le backtest combine le modèle direct et le garde-fou. L'API ajoute aussi les poids d'ensemble, le sentiment, les bornes de prix récentes et, pour les intervalles, GARCH. Pour le café, les poids actuellement configurés attribuent 45 à 50 % à N-HiTS et 10 à 15 % à Prophet : omettre ces moteurs change substantiellement le système évalué. Pour le cacao, le sentiment reste susceptible de déplacer le prix.

La calibration introduit un autre écart : les prévisions sont calculées avec les anciennes marges, utilisées par le garde-fou ; ensuite de nouvelles marges sont produites et peuvent être activées. Le garde-fou peut donc décider différemment après activation. Les nouveaux poids sont par ailleurs optimisés sur `xgb_pred`/`prophet_pred`, alors que l'acceptation porte sur `published_pred`.

Preuves : [chemin du backtest](E:/prediction/src/validation/walk_forward_validator.py:203), [ensemble, sentiment et intervalles du service](E:/prediction/src/models/improved_price_predictor.py:386), [chargement des anciennes marges](E:/prediction/scripts/run_walk_forward_validation.py:198), [nouveaux poids](E:/prediction/scripts/run_walk_forward_validation.py:149), [poids café](E:/prediction/config/coffee_robusta/ensemble_weights.json).

Correction : extraire la chaîne complète sous une configuration figée, puis l'appeler à l'identique dans le service et le replay historique. Choisir et figer poids, marges et garde-fous avant le test final. Si le sentiment historique ne peut pas être reconstruit, identifier clairement la variante sans sentiment évaluée et ne pas lui attribuer la performance de la variante servie.

**5. P1 — Le nouveau journal peut encore enregistrer une mauvaise origine et une mauvaise cible**

Le prédicteur conserve son historique jusqu'à une heure. Après son calcul, l'API relit séparément le dernier prix en base, puis utilise cette deuxième lecture pour enregistrer `origin_date`, `origin_price` et `target_date`. Lors d'une nouvelle collecte, la prévision et son journal peuvent donc porter sur deux séances différentes.

Reproduction isolée du corps réel de l'endpoint, avec dépendances simulées et aucune base distante : un modèle utilisant le cours du **2 octobre à 4 390** produit J+1 pour le **5 octobre**. Une nouvelle lecture de base fournit le **5 octobre à 4 500**. Le journal enregistre alors une origine au **5 octobre** et une cible au **6 octobre**. La correction du calendrier ne suffit donc pas à garantir une évaluation correcte.

Preuves : [cache du modèle](E:/prediction/src/models/improved_price_predictor.py:182), [lecture après calcul](E:/prediction/src/api/app.py:1276), [dates enregistrées](E:/prediction/src/api/app.py:1323).

Correction : faire retourner au prédicteur son origine, son prix d'origine, sa date cible et l'identifiant du snapshot utilisé ; les persister sans les recalculer. Utiliser ce même snapshot pour l'affichage de la variation et invalider les caches à l'arrivée d'une nouvelle séance.

**6. P1 — Les artefacts disponibles ne prouvent pas encore la qualité de la nouvelle version**

Un nouvel entraînement cacao `20261006_213603` est présent, avec un MAPE holdout hybride d'environ **8,32 %**. Ce chiffre concerne l'hybride sur son protocole de holdout ; il ne constitue pas une mesure de la prévision finale multi-horizon. Ses métadonnées reprennent encore le rapport `20261006_024613`, qui évaluait les anciennes origines 2020–2021.

Au contrôle, le dernier rapport cacao reste `20261006_024613`, celui du café `20260703_141553`. Aucun ne contient `published_pred` ou une décision de promotion issue de la nouvelle chaîne. Les marges cacao restent celles du rapport ancien et celles du café datent du 3 juillet. L'API accepte ces rapports et ces marges sans vérifier leur compatibilité avec le modèle chargé.

Preuves : [nouveau modèle cacao](E:/prediction/models/model_info_improved_20261006_213603.json), [rapport cacao](E:/prediction/reports/walk_forward/20261006_024613_summary.json), [rapport café](E:/prediction/reports/walk_forward/coffee_robusta/20260703_141553_summary.json), [marges cacao](E:/prediction/config/conformal_intervals.json), [chargement des marges sans contrôle de version](E:/prediction/src/models/conformal_intervals.py:12), [endpoint de validation](E:/prediction/src/api/app.py:1481).

Correction : relancer le backtest après correction de ses défauts, relier explicitement chaque résultat à son code, ses données et sa configuration, puis afficher « non validé pour cette version » quand aucun rapport compatible n'existe. Le passage de la cible hybride à t+1 doit être inscrit dans les métadonnées ; les anciens fichiers ne doivent pas être réutilisés silencieusement avec la nouvelle convention d'inférence.

**7. P2 — Des probabilités non calibrées subsistent dans les scénarios**

Les cartes d'horizon sont corrigées, mais `buildScenarios` calcule toujours des pourcentages à partir de la position du prix entre les bornes. Avec des bornes symétriques, cette formule produit mécaniquement **28 % / 44 % / 28 %**, sans apprentissage de probabilité. Le panneau les affiche sous « Probabilité ». La mention « indicatif » ne rend pas ces nombres calibrés.

Preuves : [formule](E:/prediction/frontend/src/lib/marketAnalytics.ts:55), [affichage](E:/prediction/frontend/src/components/dashboard/AnalysisPanel.tsx:126).

Correction : retirer les pourcentages et présenter des bornes basse/centrale/haute. Ne réintroduire des probabilités qu'après définition précise des événements et validation hors échantillon. Ajouter un test portant sur l'ensemble des panneaux, pas seulement sur le texte des cartes d'horizon.

**8. P2 — Le suivi opérationnel reste incomplet**

- **Migration nécessaire :** le nouveau journal requiert [la migration SQL](E:/prediction/sql/20261006_prediction_journal.sql). Son application distante n'a pas été vérifiée. Sans elle, l'insertion échoue mais l'API renvoie toujours la prévision après un simple log. Ajouter un contrôle de schéma au démarrage et une alerte visible en cas de perte du journal ; ne pas annoncer une migration effectuée sans preuve.
- **Évaluation tronquée :** [evaluate_predictions.py](E:/prediction/evaluate_predictions.py:35) lit les 2 000 premières lignes des 90 derniers jours, sans pagination. À volume suffisant, les observations les plus récentes disparaissent du suivi. Plusieurs appels pour la même séance sont aussi comptés comme plusieurs prévisions sans règle explicite. Paginer, définir l'unité d'évaluation et enregistrer effectif/période/taux de repli.
- **Endpoint global :** [/performance](E:/prediction/src/api/app.py:1370) ignore toujours marché et horizon, même si l'évaluateur les écrit maintenant séparément. Le dashboard simplifié ne l'utilise plus dans son chargement principal, ce qui limite l'impact actuel. Filtrer et conserver ces dimensions dans la réponse pour les autres clients.
- **Fraîcheur affichée :** [l'heure de rechargement](E:/prediction/frontend/src/components/SimplifiedMarketDashboard.tsx:224) reste présentée comme « Dernière MAJ données ». Distinguer date de cotation, heure de calcul et heure de consultation ; contrôler les séances attendues.

**Autres défauts du premier audit toujours présents**

- N-HiTS conserve [le filtre moyenne globale ±3 écarts-types](E:/prediction/train_nhits.py:92), ainsi que l'enchaînement entraînement puis validation sur la même instance ([script](E:/prediction/train_nhits.py:164), [validateur](E:/prediction/src/validation/nhits_validator.py:91)). Refaire sa validation avec remise à zéro explicite des poids et nettoyage causal. Ce point est particulièrement pertinent pour le café, qui lui attribue un poids élevé.
- Le calendrier reste `BusinessDay`, sans calendrier des fermetures de la place : [fonction](E:/prediction/src/models/hybrid_features.py:521).
- Le RSI en hausse continue remplace encore une perte moyenne nulle par `NaN` : [calcul](E:/prediction/src/models/hybrid_features.py:250).
- Les garanties de disponibilité historique et de correspondance du contrat pour l'open interest restent à renforcer. L'audit n'a pas mesuré une contamination effective de la base distante.
- `compute_holdout_baseline` cherche toujours la validation dans une trame construite uniquement sur l'entraînement et renvoie donc une tranche vide : [fonction](E:/prediction/src/validation/metrics.py:67).
- Quatre incompatibilités de dépendances locales persistent : gotrue/httpx, mlflow/packaging, supafunc/httpx et torch/sympy. Cela ne démontre pas une panne du serveur, mais empêche de considérer l'environnement comme proprement reproductible.

**Vérifications effectuées lors de ce nouvel audit**

- **85 tests Python réussis**, 36 avertissements, en 36,00 secondes : fiabilité de mesure, walk-forward, modèles directs, multi-pas, calibrateurs, moniteur, schémas API et registre des marchés.
- **TypeScript : vérification sans émission réussie** (`tsc --noEmit --incremental false`). Cela ne remplace pas un test d'interface ni un build de production.
- Reproductions des cibles de calibration dépassant la frontière test, de l'acceptation avec prévision manquante, de l'acceptation avec couverture nulle et du mauvais snapshot enregistré par l'API.
- Lecture des artefacts actuels et vérification des dépendances avec `pip check`.
- Pas de nouveau backtest long, ni de test du service distant. Les bons scores des tests logiciels ne démontrent pas l'avantage prédictif du modèle actuel.

**Ordre des ajustements recommandés**

1. Bloquer l'activation des candidats refusés et figer un ensemble complet de modèles/calibrations dans un manifeste.
2. Corriger la frontière calibration/test, les comparaisons sur échantillons différents et les critères d'acceptation des intervalles.
3. Faire passer le backtest et le service par la même chaîne complète ; persister le snapshot exact de chaque prévision.
4. Exécuter une validation récente réservée et des tests sur plusieurs régimes pour chacun des deux marchés. Publier les résultats de la version réellement retenue, avec effectifs et taux de repli.
5. Terminer l'affichage de l'incertitude et les contrôles d'exploitation, puis seulement tester de nouvelles variables ou de nouveaux modèles.

Critère décisif : le modèle activé doit apporter un gain mesuré face au dernier cours sur des observations comparables, avec des intervalles vérifiés et des échecs visibles. Cette preuve n'est pas encore disponible dans les artefacts examinés.
