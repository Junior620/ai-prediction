# Audit de fiabilité — 8 octobre 2026

Périmètre : dépôt local `E:\prediction`, commit `f19e4c9`, avec les modifications non commitées présentes pendant l'audit, notamment météo et événements d'actualité. Audit du code, des configurations, des rapports sauvegardés et de contre-exemples locaux. Le serveur déployé et la base distante n'ont pas été vérifiés.

**Avis : le projet progresse sur la transparence et la maîtrise des versions, mais sa supériorité prédictive reste non démontrée. Aucun horizon des deux manifestes actifs n'est validé. Maintenir cet état est justifié.**

Les défauts de cet audit ont été revérifiés sur l'état du 8 octobre. Plusieurs constats du 6 octobre sont maintenant corrigés ; ils ne doivent pas être présentés comme encore ouverts.

## Résultats effectivement disponibles

Les deux derniers rapports de la famille `acceptance` contiennent 96 origines annoncées. Voici leurs MAPE sur l'ensemble des lignes évaluées, avant la séparation calibration/test. Une MAPE plus faible est meilleure.

| Horizon | Cacao : modèle | Cacao : clôture inchangée | Robusta : modèle | Robusta : clôture inchangée |
|---|---:|---:|---:|---:|
| J+1 | 2,55 % | 2,45 % | 2,27 % | 1,97 % |
| J+7 | 7,76 % | 7,21 % | 5,96 % | 4,75 % |
| J+14 | 13,12 % | 11,09 % | 8,91 % | 6,92 % |
| J+30 | 18,36 % | 17,85 % | 13,16 % | 10,64 % |

Sources : [cacao, 6 octobre à 23:32](E:/prediction/reports/walk_forward/acceptance/20261006_233213_summary.json), [robusta, 7 octobre à 00:06](E:/prediction/reports/walk_forward/coffee_robusta/acceptance/20261007_000645_summary.json).

**Ces chiffres sont des diagnostics historiques, pas une certification de la version actuelle.** Ces rapports précèdent les dernières corrections de composition et de couverture du 7 octobre au soir. Les décisions de couverture qu'ils contiennent ont notamment été produites par l'ancien code. Leurs cibles se terminent au 2 octobre pour le cacao et au 14 août pour le robusta. Les rapports quotidiens du 7 octobre utilisent 24 origines, ne contiennent pas de promotion et ne constituent pas non plus une nouvelle preuve d'acceptation.

Les manifestes actifs restent `improved_20261006_213603` pour le cacao et `improved_20261006_220101` pour le robusta. Tous leurs horizons sont `validated: false`. Tous les chemins directement nommés par ces manifestes existent localement.

## Corrections confirmées

- La sélection des origines récentes utilise désormais la fin de l'historique.
- La séparation calibration/test exclut les cibles de calibration qui ne sont pas encore réalisées avant la première origine de test.
- `evaluate_horizon` reçoit maintenant des marges figées sur la calibration : il ne les réestime plus sur les résultats qu'il juge.
- Le prix central et la construction des bandes passent par `compose_served_price` dans le service et le backtest direct.
- Les modèles hybrides actifs déclarent une cible `next_session`, contrôlée au chargement.
- Les modèles nouvellement entraînés Prophet/XGBoost sont rangés dans des dossiers candidats. Le chargement principal utilise un manifeste.
- Le prix candidat est conservé séparément ; un horizon non validé publie la clôture. Les cartes et les scénarios refusent d'afficher un modèle comme validé. Les pourcentages artificiels des scénarios ont été retirés.
- Le journal conserve le marché, l'origine et la cible issus du calcul. L'évaluateur pagine et déduplique les prévisions.
- Le filtre global des prix extrêmes a été remplacé par un contrôle des barres impossibles. Le RSI d'une série strictement croissante vaut maintenant 100.
- Les dépendances installées passent `pip check`.

## Défauts à corriger en priorité

### 1. [P1] Une seule observation peut suffire à valider un horizon

**Preuve.** Avec une seule observation, une clôture de 100, un réalisé de 110, une prévision de 109 et des marges préalablement figées, `evaluate_horizon` renvoie `validated=True`, `n=1`, `n_effective=1`. Le bootstrap retourne un intervalle dégénéré et aucun minimum d'effectif ne bloque la décision.

Emplacements : [acceptance.py:48](E:/prediction/src/validation/acceptance.py:48), [acceptance.py:103](E:/prediction/src/validation/acceptance.py:103), [acceptance.json](E:/prediction/config/acceptance.json).

**Conséquence.** Le gain minimal de 10 % et l'intervalle bootstrap ne protègent pas d'une acceptation sur trop peu de résultats. La borne supérieure de couverture supérieure à 90 % signifie seulement que la cible n'est pas exclue ; elle ne démontre pas qu'elle est atteinte.

**Ajustement.** Exiger un effectif brut minimal et suffisamment de blocs temporels, fixés avant le test. Refuser les résultats à faible puissance. Utiliser l'espacement réel des origines : le script de calibration ne transmet actuellement pas le `--step-size` du run à la décision. Un nombre effectif calculé par une formule n'est pas, à lui seul, une preuve d'indépendance.

La période réservée doit également être attachée à une date de gel du protocole et aux **origines** de prévision. Le contrôle actuel ne vérifie que `target_date > 2026-10-02` ([script:141](E:/prediction/scripts/run_walk_forward_validation.py:141)). Cela ne garantit pas que les prévisions auraient pu être décidées sans connaissance d'une partie de leur trajectoire future.

### 2. [P1] Les prévisions manquantes ne sont pas toujours comptées comme replis

**Preuve.** Sur 100 observations synthétiques dont 80 prévisions sont `NaN`, avec la colonne `feature_failure=False`, les prix manquants deviennent bien la clôture pour la MAPE. Pourtant le taux de repli renvoyé est **0 %**, et l'horizon est accepté.

Emplacements : [acceptance.py:75](E:/prediction/src/validation/acceptance.py:75), [acceptance.py:141](E:/prediction/src/validation/acceptance.py:141), [walk_forward_validator.py](E:/prediction/src/validation/walk_forward_validator.py).

**Cause.** Dès que `feature_failure` existe, le masque des échecs utilise cette seule colonne. Or une exception de prédiction dans le backtest peut écrire un prix manquant sans positionner cet indicateur.

**Ajustement.** Construire un masque unique : échec explicite **ou** prix absent/non fini **ou** moteur requis indisponible. L'utiliser dans la MAPE, le taux de repli, la calibration et le sous-score des prévisions disponibles. Produire une ligne explicite pour chaque origine attendue ; ne pas laisser disparaître un échec de préparation.

### 3. [P1] Le modèle activé n'est pas lié de façon vérifiable au modèle évalué

**Preuve.** Une promotion portant le rapport du candidat A active le candidat B s'il est devenu le dossier complet le plus récent. Le contre-exemple local conserve également l'ancien N-HiTS et les anciens modèles directs lorsque le dossier sélectionné ne les contient pas.

Emplacements : [release_manifest.py:58](E:/prediction/src/models/release_manifest.py:58), [release_manifest.py:102](E:/prediction/src/models/release_manifest.py:102), [run_walk_forward_validation.py:200](E:/prediction/scripts/run_walk_forward_validation.py:200).

Le backtest reconstruit une procédure d'entraînement, puis la promotion choisit `latest_candidate` au moment d'écrire le manifeste. Le rapport ne verrouille pas les empreintes des modèles, des données, des poids, du calendrier et du protocole. Si aucun candidat n'est trouvé, la fonction peut même marquer l'ancienne version comme validée.

Autre ouverture : [improved_price_predictor.py:112](E:/prediction/src/models/improved_price_predictor.py:112) relit les métadonnées directes les plus récentes du dossier, même lorsque les modèles ont été fournis par le manifeste. Un modèle dont la sortie `5` signifie « +5 depuis 100 » peut ainsi être interprété comme un prix de `5`, au lieu de `105`, en présence de métadonnées plus récentes déclarant une autre cible. Risque particulièrement pertinent lors d'un retour à une version antérieure.

**Ajustement.** Désigner un candidat explicite avant l'évaluation ; enregistrer une empreinte de la procédure et du lot complet. Promouvoir ce lot précis, avec contrôle de présence et de compatibilité de tous ses moteurs. Transmettre les métadonnées directes du manifeste au prédicteur. Écrire les marges et le manifeste comme un ensemble cohérent, sans sélection par date de fichier et sans compléter silencieusement avec d'anciens moteurs.

### 4. [P1] Le déploiement ne transporte pas encore un lot autonome

Deux défauts vérifiables dans le script courant :

- Les modèles promus peuvent être nommés `models/candidates/<marché>/<version>/...` par le manifeste. [deploy_models.ps1:242](E:/prediction/deploy_models.ps1:242) et [deploy_models.ps1:268](E:/prediction/deploy_models.ps1:268) les copient dans `models/` et `models/coffee_robusta/`, puis transfèrent le manifeste sans réécrire ses chemins. Après une première promotion dans un dossier candidat, le serveur ne trouvera donc pas ces artefacts à l'emplacement annoncé.
- Le script copie maintenant `hybrid_features.py` et `app.py`, qui importent respectivement `src.models.weather_features` et `src.data_collection.cocoa_weather`. Ces nouveaux modules ne figurent pas dans la liste transférée ([deploy_models.ps1:319](E:/prediction/deploy_models.ps1:319)). Un serveur qui ne les possède pas déjà peut échouer dès l'import de l'API.

Le calendrier `exchange_holidays.json` n'est pas transféré non plus. Les rapports transférés sont les deux plus récents des dossiers racines, pas nécessairement le rapport nommé par le manifeste, notamment s'il est dans `acceptance/`.

**Ajustement.** Préparer un lot dans un dossier temporaire, avec tous les chemins du manifeste conservés, les modules requis, le calendrier et le rapport nommé. Tester ses imports et sa résolution des artefacts dans un environnement propre, puis basculer vers ce lot. La présence des fichiers sur le poste de développement ne valide pas le déploiement.

### 5. [P1] La couverture jugée n'est pas celle des bandes réellement servies

**Preuve.** Pour un prix central de 110, une marge de ±20 et une borne haute autorisée de 120, le service fournit `[90,120]`. Le validateur juge implicitement `[90,130]`. Avec des réalisés à 125, le contre-exemple obtient **100 % de couverture et une acceptation**, alors que la bande du service couvre **0 %** des réalisés.

Emplacements : [served_forecast.py:121](E:/prediction/src/models/served_forecast.py:121), [acceptance.py:164](E:/prediction/src/validation/acceptance.py:164), [walk_forward_validator.py](E:/prediction/src/validation/walk_forward_validator.py).

La composition est partagée, mais le backtest ne sauvegarde pas les bornes retournées et l'acceptation reconstruit une bande simple `prix ± marge`. Elle ignore ainsi le plafonnement aux bornes de prix et l'élargissement GARCH.

**Ajustement.** Après calibration, rejouer la composition des bandes sur le test, avec les seules informations disponibles à chaque origine. Sauvegarder les bornes finales et calculer leur couverture et leur largeur. Appliquer un protocole comparable à la référence naïve.

### 6. [P1] Le cache peut servir une ancienne version sous l'état d'une nouvelle

**Preuve.** L'exécution isolée du vrai début de `predict_price`, avec un cache A et un prédicteur B, renvoie la réponse A dès que la date de séance est identique. Elle lui applique l'état du manifeste B et peut marquer `served_as_close=False`.

Emplacement : [app.py:1242](E:/prediction/src/api/app.py:1242). Le contrôle vérifie la date, pas `model_version`, les poids, les marges ni l'identité complète des données. La purge Redis du déploiement réduit ce risque, mais son échec est explicitement non bloquant.

**Ajustement.** Mettre l'identité du lot actif et du snapshot dans la clé de cache, et la vérifier à la lecture. Faire correspondre également la version de la réponse de prévision à celle des métriques dans le frontend. Une correction de prix ou d'OI sur une même date doit pouvoir invalider le cache.

### 7. [P1] La validation du robusta peut omettre N-HiTS sans bloquer l'acceptation

Les poids actifs du robusta allouent **45 à 50 %** à N-HiTS ([configuration](E:/prediction/config/coffee_robusta/ensemble_weights.json)). Pourtant `replay_nhits` vaut `False` par défaut et une erreur de replay est seulement journalisée. `compose_served_price` renormalise les autres poids quand N-HiTS manque. Le validateur n'interdit pas ensuite l'acceptation de cette composition différente.

Emplacements : [walk_forward_validator.py:42](E:/prediction/src/validation/walk_forward_validator.py:42), [walk_forward_validator.py:147](E:/prediction/src/validation/walk_forward_validator.py:147), [served_forecast.py:99](E:/prediction/src/models/served_forecast.py:99).

**Ajustement.** Conserver le choix demandé : un N-HiTS neuf, entraîné sur le passé, à chaque origine du test d'acceptation. Si un moteur requis manque, déclarer le run incomplet, ou évaluer explicitement une variante dégradée distincte. Enregistrer par origine sa présence, son entraînement, ses sorties et les poids effectivement utilisés. Les rapports actuels n'enregistrent pas suffisamment ces éléments pour certifier a posteriori le replay complet.

## Autres défauts concrets

### 8. [P2] Le suivi candidat utilise les intervalles de la clôture publiée

`publish_or_close` conserve `candidate_price`, mais remplace la bande par `[clôture, clôture]`. Le journal enregistre cette bande. `evaluate_predictions.py` choisit ensuite le prix candidat pour l'erreur, tout en prenant les bornes publiées pour la couverture.

Contre-exemple : candidat 110 avec bande `[90,120]`, clôture 100 ; le suivi évalue le candidat 110 avec `[100,100]`.

Emplacements : [served_forecast.py:30](E:/prediction/src/models/served_forecast.py:30), [app.py:1398](E:/prediction/src/api/app.py:1398), [evaluate_predictions.py:145](E:/prediction/evaluate_predictions.py:145).

**Ajustement.** Journaliser séparément prix/bornes du candidat et prix/bornes publiés, avec état de validation, raison du repli et identité du snapshot. Produire deux groupes de métriques distincts. Enregistrer aussi l'effectif et la période de réalisation : actuellement ils sont affichés dans le terminal, mais pas conservés dans l'insertion `model_metrics`. Le pas 5 utilisé pour l'effectif effectif ne convient pas automatiquement au journal quotidien.

Pour qualifier un suivi de prospectif, vérifier aussi que la prévision a été enregistrée avant la cible, avec des données disponibles à l'origine. Le test actuel sur la date extraite du nom de modèle ne suffit pas à établir cela.

### 9. [P2] Le calendrier des séances reste incomplet

Le fichier ne contient que trois fermetures de 2026. Le calcul effectif donne `2026-04-02 + J+1 = 2026-04-03`. Or les produits agricoles ICE Europe sont fermés le 3 et le 6 avril ; la séance suivante est le 7 avril. Voir le [calendrier officiel ICE 2026](https://www.ice.com/publicdocs/Trading_Schedule.pdf).

Emplacements : [exchange_holidays.json](E:/prediction/config/exchange_holidays.json), [hybrid_features.py:532](E:/prediction/src/models/hybrid_features.py:532).

**Conséquence.** Certaines cibles sont inexistantes et sont supprimées du backtest, ou les horizons ne correspondent pas au nombre de séances annoncé. Les années historiques ne disposent pas non plus de leurs fermetures. N-HiTS utilise parallèlement `freq='B'` et une sélection par position.

**Ajustement.** Un calendrier versionné couvrant entraînement, test et prévision, commun à tous les moteurs. Tester de vrais jours fermés de la place, et non uniquement une fermeture fictive injectée au test.

### 10. [P2] Le candidat météo ne compare pas encore le bon produit avec des données disponibles au bon moment

**Mauvais rattachement au marché.** [run_weather_candidate.py:71](E:/prediction/scripts/run_weather_candidate.py:71) charge `cocoa_prices`, alors que le cacao actif utilise `cocoa_london_prices` ([config.yaml:47](E:/prediction/config/config.yaml:47)). Le script réentraîne aussi son propre témoin avant 2025, au lieu de comparer la procédure servie complète. Son libellé « modèle actuel » ne suffit donc pas à établir un gain sur le service actif.

**Disponibilité future.** [weather_features.py:45](E:/prediction/src/models/weather_features.py:45) inclut les totaux du jour courant, puis les joint à la même date de prix. Modifier la pluie de ce jour fait passer la feature `rain_sum_7` de 7 à 107 pour la même origine. NASA annonce une latence de 2–3 jours pour ses paramètres météorologiques ; ses historiques sont aussi révisés ultérieurement. Voir la [FAQ NASA POWER](https://power.larc.nasa.gov/docs/faqs/data/) et la [documentation des produits révisés](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/).

**Ajustement.** Utiliser la configuration du marché, sa devise, ses mêmes origines et sa même procédure de publication. Joindre la météo selon `available_at`, avec un décalage documenté lorsque les publications historiques ne sont pas archivées. Distinguer une étude sur historique révisé d'une preuve prospective.

Le [rapport météo actuel](E:/prediction/reports/weather_candidate/20261008_163256.json) refuse la promotion : gains de MAE de 2,8 %, 5,8 % et 3,6 % à J+30/J+60/J+90, sans cible après le 2 octobre. Il ne prouve pas encore un apport exploitable au modèle de Londres. De plus, `_acceptance_j30` fournit volontairement des marges absentes : cette commande ne peut actuellement pas accepter ce candidat, même si les autres résultats s'amélioraient.

### 11. [P2] Deux chemins N-HiTS restent incohérents avec le nouveau cycle de versions

- `train_nhits.py` entraîne un réseau, appelle ensuite `cross_validation` sur ce même objet, puis sauvegarde l'objet après cette étape ([train_nhits.py:159](E:/prediction/train_nhits.py:159), [train_nhits.py:174](E:/prediction/train_nhits.py:174)). Dans la bibliothèque installée, `use_init_models=False` par défaut. Le nouveau validateur dédié crée bien un réseau neuf ; le script d'entraînement principal n'a pas reçu cette même séparation. Sa MAPE imprimée ne doit pas servir de preuve indépendante.
- Les nouveaux N-HiTS sont enregistrés dans les dossiers historiques de marché, tandis que le sélectionneur de candidat recherche N-HiTS dans le dossier du candidat hybride. La promotion peut ainsi conserver l'ancien N-HiTS. Le nettoyage quotidien garde seulement cinq répertoires par date et ne protège pas les chemins des manifestes ([cleanup_old_models.py:36](E:/prediction/scripts/cleanup_old_models.py:36), [update_system.bat:237](E:/prediction/update_system.bat:237)). Après assez de nouveaux entraînements, un N-HiTS toujours actif peut être supprimé.

**Ajustement.** Réseau d'évaluation neuf, entraînement final distinct, puis rangement de tous les moteurs du candidat dans un lot explicite. Protéger du nettoyage les lots actifs et ceux conservés pour retour arrière.

### 12. [P2] La disponibilité historique de l'OI n'est toujours pas démontrée

Le collecteur conserve la dernière révision par date de référence, puis peut affecter le plus grand OI du jour au contrat continu quand le symbole ne correspond pas. Les horodatages de disponibilité ne sont pas transmis jusqu'aux features ([databento_london_collector.py:220](E:/prediction/src/data_collection/databento_london_collector.py:220), [databento_london_collector.py:392](E:/prediction/src/data_collection/databento_london_collector.py:392)).

Le test « modifier une valeur de date future ne change pas une ligne passée » passe, mais ne teste pas une révision publiée plus tard concernant une date passée.

**Ajustement.** Conserver contrat réel, date de référence, date de publication/réception et version. Faire une jointure selon la disponibilité à l'origine. Déclarer l'OI inconnu si le contrat exact ne peut pas être identifié. Sans cette provenance, ne pas présenter le gain des features OI comme une preuve causale.

## Ordre de travail recommandé

1. **Réparer les portes d'acceptation** : effectifs, comptage des échecs, espacement réel, période véritablement réservée et bandes finales. Ajouter les contre-exemples ci-dessous aux tests de non-régression.
2. **Fixer une identité de lot de bout en bout** : candidat évalué, moteurs, métadonnées, poids, marges, données, calendrier, cache, rapport, déploiement et retour arrière.
3. **Fiabiliser les preuves prospectives** : journal quotidien automatique, prix et bandes du candidat distincts du résultat publié, provenance des données et contrôle de fraîcheur. Vérifier sur le serveur réel les migrations et l'enregistrement effectif ; ce point n'a pas été vérifié ici.
4. **Refaire une comparaison complète** : clôture inchangée, XGBoost direct seul, ensemble complet. Pour le robusta, rejouer N-HiTS à chaque origine comme demandé. Conserver les replis dans le score principal. Montrer MAPE, MAE, couverture, largeur, effectifs et incertitude par horizon.
5. **Évaluer ensuite les nouvelles sources une à une** : météo correctement retardée, OI disponible à l'époque, puis événements. Le rapport local d'événements annonce actuellement zéro événement rejouable et refuse l'entraînement ; cela ne signifie pas qu'il n'existe aucune actualité, mais qu'aucun historique exploitable n'a été démontré par ce rapport.

Le critère de progrès est un gain mesuré sur la clôture inchangée, sur des prévisions réalisables au moment annoncé et avec des bandes honnêtes. Ajouter un moteur ou une source ne garantit pas ce gain.

## Vérifications réalisées et limites

- **120 tests Python ciblés réussis**, 38 avertissements, en 143,08 s : activation, mesure, walk-forward, horizons directs, prédiction multiétape, calibration, monitoring, modèles API, marchés, météo et événements.
- **TypeScript** : `tsc --project frontend/tsconfig.json --noEmit --incremental false`, sortie 0.
- **Dépendances** : `python -m pip check`, aucun conflit déclaré.
- **9 contre-exemples locaux réussis**, dans [audit_repros_2026_10_08.py](E:/prediction/reports/audit_repros_2026_10_08.py). Ils utilisent des données synthétiques ; l'exemple API exécute la fonction réelle isolée de son démarrage, avec cache et services simulés. Aucun serveur ni base n'est modifié.

```powershell
.\venv_py311\Scripts\python.exe reports\audit_repros_2026_10_08.py
```

Ces contre-exemples prouvent les défauts de logique décrits, pas leur fréquence sur le serveur. Les tests existants réussis ne les couvraient pas.

Aucun entraînement long ni déploiement n'a été lancé, aucune migration n'a été appliquée, aucun fichier applicatif n'a été modifié pour cet audit. Les seuls fichiers ajoutés sont ce rapport et le script de reproduction. Il reste à mesurer le système après correction ; les performances futures ne peuvent pas être déduites de la seule réussite des tests logiciels.
