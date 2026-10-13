# Audit de fiabilité — 13 octobre 2026

Priorité : **cacao**, puis robusta. Revue du code actuellement présent dans `E:\prediction`, des dernières études corrigées, des instantanés, des modèles nommés, du journal et de l'API locale. Les modifications non commitées préexistantes font partie du périmètre examiné.

**Avis : la traçabilité progresse, mais la fiabilité prédictive du cacao n'est pas démontrée. Une divergence réelle entre le calcul local et l'API peut encore changer un prix sans nouveau modèle ni nouvelle observation de prix.** Il faut corriger les données et cette divergence avant d'attribuer les retournements au réentraînement.

Cet audit n'a modifié ni code métier, ni modèles, ni manifestes, ni données de marché. Aucun entraînement, backtest long, déploiement ou appel POST à l'API n'a été lancé. Les seuls fichiers ajoutés sont ce rapport, le script de contre-exemples et ses résultats.

**Preuves et contrôles**

- Contre-exemples exécutables : `reports/audit_repros_2026_10_13.py` ; résultats conservés dans `reports/audit_evidence_2026_10_13.jsonl`. Le script lit les artefacts locaux, intercepte complètement l'appel HTTP du vérificateur et ne travaille en écriture que dans un répertoire temporaire automatiquement supprimé.
- Tests ciblés : **79 réussis, 1 ignoré**, 31 avertissements. Fichiers : `test_forecast_tracking.py`, `test_measurement_consistency.py`, `test_measurement_reliability.py`, `test_direct_horizon_trainer.py`, `test_activation_gate.py`, `test_walk_forward_validator.py`. Le test Windows/Docker conditionnel n'a pas été activé.
- Frontend : `tsc --noEmit --incremental false` réussi. Cela vérifie le typage, pas la qualité des prix.
- GET locaux à 05:55 UTC le 13 octobre : `/health` répond `healthy` ; `/api/v1/markets` annonce cacao `improved_20261013_045358` et robusta `improved_20261010_093542`.
- Manifestes actifs : cacao `improved_20261006_213603`, robusta `improved_20261006_220101`, tous deux non validés. Les lots expérimentaux inspectés sont également non validés.
- Le répertoire de travail reste actif pendant l'audit : des lignes concernant un nouveau lot robusta apparaissent dans le journal. Les versions ci-dessus décrivent l'API observée, pas une garantie sur son état après l'audit.

**Progrès depuis le 10 octobre**

Les défauts suivants ont été corrigés ou sensiblement réduits dans le code actuel :

- La politique de cinq séances conserve les paramètres et reconstruit les variables d'entrée sur l'historique de chaque séance. N-HiTS peut recevoir cet historique actualisé.
- L'étude annote maintenant les prix réalisés des révisions et publie leur erreur. La référence « cours inchangé » est incluse pour les initiales.
- La reprise reconstitue le lot précédent et l'ancrage de cinq séances ; le test de reprise séquentielle passe.
- Les empreintes couvrent désormais les colonnes OHLCV/OI et les fichiers des modèles directs référencés. Le contrôle du cache tient compte des corrections à date identique, sous réserve de la divergence de préparation décrite ci-dessous.
- Les anciennes prévisions initiales sont recherchables avec filtre et pagination ; le suivi conserve la cible initiale et la référence de prix. Le comptage des inversions utilise la règle commune.
- Les dates d'entraînement sont lues dans les métadonnées. `update_system.bat` traite le cacao en premier, fournit un instantané à N-HiTS, attend l'API et ne déploie pas automatiquement sur le VPS.

Ces progrès ne suffisent pas à démontrer une supériorité prédictive.

**Résultats historiques actuels du cacao**

Source : `reports/revision_stability/cocoa/study_corrected/20261012_183344_summary.json`. Les valeurs du rapport `20261012_182756` sont identiques pour ces métriques. Les 60 origines vont du **2 juin au 24 août 2026**, avec 90 dates de suivi jusqu'au **6 octobre**. Ce sont des dates historiques, pas 60 jours à attendre à partir d'aujourd'hui.

MAPE = moyenne des erreurs absolues en pourcentage du prix réalisé ; **plus faible = meilleur**.

| Horizon | Réentraînement quotidien | Tous les cinq points de séance | Cours inchangé | Initiales évaluées par politique |
| --- | ---: | ---: | ---: | ---: |
| J+1 | 2,71 % | 2,62 % | 2,50 % | 60 |
| J+7 | 8,33 % | 8,29 % | 8,21 % | 60 |
| J+14 | 12,26 % | 12,28 % | 10,68 % | 60 |
| J+30 | 18,83 % | 19,02 % | 11,67 % | 60 |

Le quotidien reste moins bon que le cours inchangé sur les quatre horizons de cette fenêtre. À J+30, son erreur est environ **61 % plus élevée**, et non 61 points de pourcentage. La cadence de cinq séances ne règle pas le problème : 96 inversions au lieu de 100 pour les trajectoires initialement J+30, avec une erreur légèrement supérieure. Les 1 740 révisions de ces trajectoires ne sont pas 1 740 observations indépendantes.

Ces résultats sont descriptifs : ils concernent la procédure de l'étude et ne valident pas le lot chargé le 13 octobre. Le snapshot cacao actuel tronqué au 6 octobre reproduit exactement l'empreinte de données de l'étude. En revanche, l'empreinte de procédure recalculée aujourd'hui diffère de celle sauvegardée pour les deux marchés. La reprise doit donc rester refusée tant que cette différence n'est pas expliquée ; ce refus serait le comportement correct.

**Défauts confirmés et corrections proposées**

**1. [P1] Le journal local et l'API donnent des prix différents avec le même modèle.**

Localisation : `src/models/improved_price_predictor.py:243`, `:246`, `:322` ; `scripts/journal_forecast_revisions.py:110`, `:117`.

Le chemin API remplit les valeurs OHLCV brutes manquantes par propagation de la dernière observation avant de calculer les variables. Le chemin de l'instantané lié avec `bind_frame` ne le fait pas : il calcule les variables puis remplit certaines variables manquantes. Les deux opérations ne sont pas équivalentes. Le fichier cacao actuel comporte **29 lignes avec OHLCV manquants**.

Reproduction avec le lot `improved_20261013_045358`, origine 13 octobre et cours 4 277 £ :

| Horizon | Commande sur instantané | Préparation API | Écart API − commande |
| --- | ---: | ---: | ---: |
| J+1 | 4 268,87 £ | 4 278,70 £ | +9,83 £ |
| J+7 | 4 259,12 £ | 4 328,63 £ | +69,52 £ |
| J+14 | 4 085,55 £ | 4 084,28 £ | −1,27 £ |
| J+30 | 3 597,79 £ | 3 579,32 £ | −18,47 £ |

Les poids cacao sont actuellement 100 % XGBoost aux quatre horizons : les composantes reproduites correspondent aux prix publiés. **Le journal contient effectivement ces deux séries**, enregistrées vers 03:56:12 UTC puis 03:57:13 UTC, pour le même modèle et les mêmes cibles. J+7 passe de sous le cours à au-dessus, sans réentraînement entre ces deux séries. Ces deux faibles variations restent toutefois dans la bande neutre de 1,5 % du suivi : ce cas ne doit pas être présenté comme une inversion directionnelle au sens de ce compteur.

Effet supplémentaire : le hash distant est calculé avant remplissage OHLCV, tandis que le hash du cache interne est calculé après ; sur cet instantané ils diffèrent. Une simple transformation locale peut donc être prise pour une différence d'instantané et provoquer des recalculs inutiles.

Correction : une seule préparation des données, explicitement définie et versionnée, avant calcul des variables, utilisée par entraînement, commande de journal, API et backtest. Distinguer identité du snapshot brut et identité des entrées transformées. Test attendu : mêmes artefacts + même snapshot → mêmes prix, mêmes cibles, mêmes variables et identité cohérente via les deux chemins, y compris avec OHLCV absents. Ne pas écraser les anciennes prévisions lors de cette correction.

**2. [P1] La collecte peut transformer une date de récupération en date de cotation.**

Localisation : `src/data_collection/ice_london_collector.py:116`, `:249`, `:273` ; `src/data_collection/investing_price_collector.py:139` ; `collect_ice_london_cocoa.py:296`, `:350`.

Le prix direct Playwright reçoit systématiquement `datetime.now()` comme date. Le parseur de tableau fait de même si sa première cellule n'est pas une date. Le recours à Investing utilise aussi le jour de collecte. Le collecteur principal accepte ensuite cette date artificiellement récente.

Contre-exemples hors réseau : un HTML portant le 9 octobre produit une observation datée du 13 octobre lorsque le prix direct est trouvé ; une ligne `DEC26 / 4277` devient une ligne de prix journalier datée du jour. Le snapshot réel comporte notamment **2026-10-10**, un samedi, source `ice_london`. L'audit ne prétend pas identifier la vraie séance de chaque prix sans consulter sa source.

Impact : fraîcheur trompeuse, doublons de clôtures à des dates différentes, décalage des variables retardées et origine de prévision incorrecte. Ce défaut touche la collecte cacao et le chemin Investing partagé avec le robusta.

Correction : conserver séparément `session_date`, heure de cotation si connue, `collected_at`, contrat, source et nature du prix (intraday/settlement). Une date de cotation absente ou ambiguë doit être signalée/quarantainée, jamais remplacée par aujourd'hui. Conserver le dernier cours connu avec sa vraie date. Revoir les anciennes lignes suspectes à partir des sources, sans inventer de prix ou de dates de remplacement.

**3. [P1] L'étude de stabilité n'utilise pas les mêmes séances que la chaîne quotidienne.**

Localisation : `scripts/run_revision_stability.py:193`, `:195`, `:246` ; `src/models/hybrid_features.py:583`, `:594` ; `train_hybrid_improved.py:67` ; `src/models/improved_price_predictor.py:243`.

L'étude applique `trading_sessions`, alors que l'entraînement quotidien et l'API se limitent à `prepare_market_frame`. Cette dernière conserve les dates que le calendrier du projet considère fermées. L'instantané cacao actuel compte **1 740 lignes**, contre **1 712 après le filtre de l'étude**. Ces 28 dates comprennent des samedis et des jours classés fermés par le calendrier du projet ; la justesse de toutes les fermetures de ce calendrier n'est pas certifiée par cet audit.

Avec le même modèle actuel, supprimer ces lignes change déjà `price_lag_1` de 4 241 à 4 384 et le prix direct J+7 de 4 259,12 à 4 370,22 £. Ce calcul illustre la sensibilité aux entrées ; il ne démontre pas que le second prix serait plus juste.

Correction : vérifier le calendrier propre à chaque marché, appliquer la même règle partout, distinguer séances absentes et jours fermés, puis réentraîner sur le snapshot corrigé. Ne pas simplement filtrer l'API tout en conservant des modèles entraînés avec une autre convention. Pour le robusta, définir aussi ce que « cinq séances » signifie en présence de données manquantes : aujourd'hui le compteur avance sur les lignes présentes.

**4. [P1] Le checkpoint cacao contient chaque séance deux fois.**

Localisation : `scripts/run_revision_stability.py:236` à `:243`, `:267` ; `reports/revision_stability/cocoa/study_corrected/sessions.jsonl`.

Le fichier contient **180 lots pour 90 dates uniques** ; chaque paire est identique. L'écriture est un ajout sans verrou propre à l'étude. La lecture déduplique seulement les dates déjà faites, puis ajoute tous les enregistrements à `prior`. L'audit n'a pas déterminé le processus responsable des doublons.

Recalcul hors entraînement sur le contenu sauvegardé : J+30 quotidien passe artificiellement de **60 à 120 initiales**, de **1 740 à 3 480 révisions**, et de **100 à 200 inversions**. La MAPE reste 18,82558 %, puisque toutes les lignes sont doublées. Les synthèses déjà écrites annoncent bien 60 initiales : elles ne présentent pas elles-mêmes ces effectifs doublés. Le défaut concerne la relecture/reconstruction du checkpoint, une fois son identité acceptée.

Correction : verrou ou propriétaire unique par étude, clé `(identité d'étude, séance)`, écriture atomique, rejet des doublons contradictoires et déduplication explicite des doublons identiques lors d'une récupération contrôlée. Préserver le fichier original. Recalculer la synthèse depuis les séances uniques ne nécessite pas de réentraîner les modèles. Tester deux écritures concurrentes et une reprise complète comparée au résultat continu.

**5. [P1] La vérification finale de `update_system.bat` peut déclarer quatre prix absents consultables.**

Localisation : `scripts/verify_served_forecast.py:53` à `:65` ; `update_system.bat:173`.

Le script contrôle la version et l'existence d'une chaîne de statut, mais n'exige ni prix ni statut disponible. Une réponse simulée avec la bonne version, les quatre horizons `unavailable` et tous les prix `null` produit **code de sortie 0** et `[OK] … consultable`. L'appel HTTP a été entièrement intercepté pendant la reproduction.

Correction : contrôler les quatre horizons exacts, des prix numériques finis positifs, des statuts autorisés, les dates d'origine/cible et l'identité du lot/snapshot attendu. Si un horizon échoue, annoncer un succès partiel avec le détail et un code exploitable par le batch. La vérification ne doit pas exiger `validated` : les quatre cartes expérimentales restent permises après entraînement. Tester le scénario de quatre absences, une absence, un ancien lot et une ancienne origine.

**6. [P2] Le score de stabilité ne rend pas compte des prévisions indisponibles.**

Localisation : `src/validation/revision_stability.py:163`, `:175`, `:183`, `:192`.

Les initiales évaluées sont filtrées sur `price is not None`. Une cible arrivée à échéance sans prévision disparaît du dénominateur ; `absent_targets` compte les réalisations absentes, pas les prévisions absentes. Contre-exemple : deux cibles réalisées et une seule prévision donnent un rapport avec une initiale mature et zéro cible absente, sans annoncer le taux de disponibilité de 50 %. Cela peut favoriser une politique qui échoue davantage. L'audit ne prétend pas que les résultats cacao actuels contiennent effectivement de telles abstentions.

En outre, les erreurs des révisions sont ventilées par horizon restant, mais la référence du cours du jour de révision est agrégée sur tous les horizons restants. Leur comparaison directe n'est pas homogène.

Correction : publier effectifs attendus, prix disponibles, abstentions, cibles réalisées/manquantes et taux de disponibilité ; comparer les politiques sur des couples identiques, avec résultat complémentaire sur l'ensemble attendu. Calculer la référence pour chaque horizon restant, sur exactement les mêmes lignes que le modèle. Ajouter une incertitude tenant compte des cibles communes et des horizons qui se chevauchent avant toute sélection de cadence.

**7. [P2] L'empreinte de procédure reste incomplète.**

Localisation : `src/models/release_manifest.py:275` ; `scripts/run_revision_stability.py:138`.

La liste des fichiers couverts omet notamment `hybrid_trainer.py`, `horizon_schedule.py`, `ensemble_weights.py` et le prédicteur utilisé par le service. Les paramètres de l'étude incluent les poids aux ancrages, les plafonds et la cadence, mais pas tous les paramètres et dépendances exécutés. Dans un répertoire temporaire, changer `hybrid_trainer.py` laisse l'empreinte inchangée : cette dépendance n'est pas lue.

Correction : un manifeste explicite de procédure couvrant l'ensemble des dépendances réellement utilisées, leurs paramètres, les versions des bibliothèques et les seeds. Conserver une copie immuable des entrées de chaque étude, distincte du chemin `data/freeze/*` réécrit par les mises à jour quotidiennes. La nouvelle empreinte doit invalider une reprise lorsqu'une dépendance change ; elle doit aussi rester identifiable indépendamment du modèle courant.

**8. [P2] L'historique visible ne donne toujours pas accès aux dernières études.**

Localisation : `src/api/app.py:1808`, `:1835` ; `frontend/src/lib/api.ts:142` ; `frontend/src/components/SimplifiedMarketDashboard.tsx:483`.

L'API d'historique utilise un `glob` non récursif à la racine des rapports walk-forward. Elle ignore les sous-dossiers d'acceptation/mesure et les études dans `reports/revision_stability`. Le GET cacao renvoie 34 rapports dont le dernier est `20261006_214926`, alors que les rapports des 9, 12 et 13 octobre existent. Côté frontend, `getValidationHistory` est défini mais aucun composant ne l'appelle. La section actuelle affiche surtout le rapport rattaché au lot courant ou un message d'absence.

Correction : index explicite des rapports par marché et type d'évaluation, route de lecture des métriques historiques et branchement de la liste dans le tableau de bord. Afficher période, protocole, effectifs et compatibilité avec le lot actuel. Un rapport incompatible doit rester consultable comme historique sans être utilisé comme validation du lot courant.

**Robusta : résultats et limites distincts**

Source : `reports/revision_stability/coffee_robusta/study_corrected/20261013_022008_summary.json`. Les 60 origines vont du 1er mai au 21 août, avec 75 dates présentes de suivi jusqu'au 6 octobre. Son checkpoint ne présente pas les doublons constatés pour le cacao.

| Horizon | MAPE quotidienne | Cours inchangé | Initiales effectivement évaluées / prévues |
| --- | ---: | ---: | ---: |
| J+1 | 1,84 % | 1,68 % | 50 / 60 |
| J+7 | 4,00 % | 4,07 % | 48 / 60 |
| J+14 | 6,14 % | 5,51 % | 44 / 60 |
| J+30 | 6,77 % | 7,06 % | 33 / 60 |

Les petits avantages observés à J+7 et J+30 ne suffisent pas à valider le robusta : J+30 n'est évalué que sur 33 cibles sur 60, et ce rapport ne donne pas d'incertitude statistique pour cet avantage. Les compteurs de cibles absentes incluant les révisions ne doivent pas être interprétés comme autant de dates indépendantes. Ces résultats ne sont pas ceux de l'ancienne comparaison sur 96 origines.

**Ordre d'action recommandé, cacao en premier**

1. Corriger la datation de collecte et construire un snapshot cacao contrôlé : dates de séance, source, contrat, trous, OHLCV absents. Garder les anomalies en quarantaine avec leur provenance.
2. Unifier la préparation et le calendrier dans les quatre chemins. Faire passer le contre-exemple réel du 13 octobre : une seule série de prix à artefacts et données identiques. Étendre ensuite le contrôle au robusta et à N-HiTS. Une cause informatique d'écart doit être distinguée d'une révision liée à de nouvelles données ou à de nouveaux paramètres.
3. Sécuriser l'étude : identité complète, écritures exclusives, reprise sans doublons, dénominateurs et références appariés. Préserver puis reconstruire le rapport cacao depuis les enregistrements uniques lorsque c'est pertinent ; ne pas reprendre une procédure différente sous l'ancien identifiant.
4. Renforcer le contrôle de fin de `update_system.bat`, y compris pour les statuts partiels. L'entraînement local continue à produire J+1/J+7/J+14/J+30 avec le libellé expérimental ; une défaillance reste visible au lieu d'un faux succès.
5. Essai court de cohérence avant tout calcul long, puis nouvelle mesure cacao sur un protocole figé. Conserver l'initiale, montrer les révisions pour la même cible et comparer le modèle précédent et le nouveau sur le même snapshot. Ne pas imposer artificiellement une direction constante.
6. Une fois la mesure fiable, comparer des variantes simples : cours inchangé, XGBoost direct actuel, versions plus régularisées/moins riches et ensembles utiles. Tester séparément la fenêtre d'historique : le nettoyage courant commence en janvier 2020 ; avoir dix ans de données disponibles ne signifie pas qu'ils sont tous consommés ni qu'ils améliorent automatiquement le modèle. Réserver une période finale non utilisée pour choisir les réglages.
7. Choisir une cadence ou un modèle seulement si l'erreur, la disponibilité et la stabilité mesurées justifient ce choix. Une réduction du nombre de changements de signe n'est pas suffisante. L'affichage expérimental reste distinct de la promotion, qui n'est pas autorisée par ce rapport.

Le premier résultat attendu est concret : **le même lot et le même snapshot doivent produire le même prix dans le journal et sur le tableau de bord**. Ensuite seulement, on peut mesurer ce que le réentraînement apporte ou dégrade, particulièrement à J+14 et J+30 pour le cacao.
