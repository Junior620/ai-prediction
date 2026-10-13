# Audit de fiabilité — 10 octobre 2026

Priorité : cacao. Revue du code présent dans l'espace de travail, des études terminées le 10 octobre, du journal local et de l'API locale. Les fichiers métier, données, modèles et manifestes n'ont pas été modifiés par cet audit. Aucun entraînement, déploiement ou calcul historique long n'a été lancé.

Le projet a progressé dans la traçabilité, mais la fiabilité prédictive du cacao reste à démontrer. La nouvelle étude ne permet pas encore de choisir une cadence de réentraînement : elle compare des entrées différentes et ne mesure pas l'erreur des révisions. Les points ci-dessous distinguent défauts constatés, reproductions synthétiques et limites des résultats existants.

## État constaté

- Les deux lots expérimentaux contiennent les horizons directs 1 à 30 : cacao `improved_20261010_053100`, robusta `improved_20261010_093542`.
- Les manifestes actifs restent ceux du 6 octobre, sans différence avec Git, et aucun horizon actif ou expérimental n'est validé.
- Le calcul commun conserve les prix avec avertissement en cas de grand mouvement, retire les bandes incohérentes et rend indisponible un moteur requis manquant.
- Le journal distingue les diagnostics des prévisions publiées, conserve les initiales et rattache les révisions à la même cible.
- `update_system.bat` traite maintenant le cacao avant le robusta et ne déploie plus automatiquement sur le VPS.
- L'API locale répond et annonce les deux lots ci-dessus. Le journal examiné contient 219 lignes. Le suivi cacao renvoie huit références initiales.

## Résultat cacao : comparaison absente du rapport, recalculée pendant l'audit

Recalcul à partir de `reports/revision_stability/cocoa/study/sessions.jsonl`, avec exactement les mêmes 60 initiales de la politique quotidienne par horizon. La référence est le cours de l'origine conservé jusqu'à la cible. MAPE = moyenne de `100 * abs(prévision - réalisé) / réalisé` ; une valeur plus faible est meilleure.

| Horizon | Erreur modèle quotidien | Erreur cours inchangé | Effectif |
| --- | ---: | ---: | ---: |
| J+1 | 2,74 % | 2,50 % | 60 |
| J+7 | 8,95 % | 8,21 % | 60 |
| J+14 | 12,57 % | 10,68 % | 60 |
| J+30 | 24,09 % | 11,67 % | 60 |

À J+30, l'erreur observée du modèle est environ 2,07 fois celle de la référence. Ce sont des résultats historiques descriptifs sur les lignes sauvegardées, pas une estimation certifiée des performances futures. Les anomalies de calendrier ci-dessous imposent une nouvelle mesure après correction. Ils ne constituent pas davantage un test final indépendant pour choisir de nouveaux réglages.

Les 60 origines cacao vont du **2 juin au 24 août 2026** ; leurs cibles vont jusqu'au 6 octobre. Le texte de synthèse confond cette fenêtre avec les 92 dates de suivi, qui vont jusqu'au 6 octobre. Pour le robusta, les 60 origines vont du 23 mars au 14 août, avec 105 dates de suivi. La politique quotidienne robusta a une MAPE modèle/référence de 1,84/1,72 à J+1, 3,69/3,78 à J+7, 4,90/4,19 à J+14 et 6,11/5,53 à J+30. Ces écarts ne prouvent pas une supériorité statistique.

## Défauts prioritaires — P1

### 1. L'étude fige les variables d'entrée avec le modèle

**Localisation :** `scripts/run_revision_stability.py:64`, `:83`, `:315`, `:331`.

`_train()` conserve la dernière ligne de variables dans `bundle['last']`, puis `_price()` la réutilise. La politique « cinq séances » ne recalcule donc pas ses variables avec les nouvelles données. La comparaison ancien/nouveau conserve aussi les variables du jour d'entraînement de l'ancien modèle. Seuls le cours de référence et la date passés à `_price()` changent.

Pour le robusta, `bundle['nhits']` est une ancienne série de prévisions, pas un réseau conservé auquel fournir le nouvel historique. Relire son élément J+h plusieurs jours plus tard ne vise plus la bonne date. Les résultats de stabilité et les diagnostics de changement de modèle sont donc affectés sur les deux marchés.

**Reproduction :** avec le même modèle synthétique et le même cours courant de 200, l'ancienne variable donne 210, la variable à jour donne 390. Aucun réentraînement n'est nécessaire pour produire cet écart.

**Correction :** conserver les paramètres des anciens modèles, reconstruire les variables sur le même historique courant pour chaque modèle, et refaire l'inférence N-HiTS avec le réseau gelé. Passer par la même préparation que le service. Ajouter un test dans lequel les données changent mais pas le modèle ; vérifier aussi les dates des sorties N-HiTS. Refaire ensuite les études de cadence.

### 2. Aucune révision historique n'est évaluée à maturité

**Localisation :** `scripts/run_revision_stability.py:162`, `:395`, `:432`.

Le script ne conserve `realized` sur une révision que si sa cible est déjà atteinte ; or il n'émet des révisions qu'avant cette cible. Il ne rattache pas le réalisé après coup. Résultat vérifié : **0 réalisé sur 5 864 révisions cacao**, et **0 sur 5 128 révisions robusta**. Tous les `revision_mape_by_remaining` sont vides.

La synthèse ne calcule pas non plus l'erreur du cours inchangé ni les effectifs appariés par horizon restant. On peut observer des mouvements de prévision, mais pas vérifier qu'ils la rapprochent du prix final.

**Correction :** faire une étape de notation après génération, par marché et date cible, sans exposer le réalisé à l'entraînement. Noter initiales et révisions, référence à l'origine et référence au jour de révision, effectifs comparables, absences et erreurs par horizon restant. Les prévisions existantes restent immuables.

### 3. Une reprise de l'étude change les résultats

**Localisation :** `scripts/run_revision_stability.py:285`, `:303`, `:311`.

Les pistes sont restaurées à leur prix initial, sans rejouer leurs dernières révisions. Pendant le saut des séances déjà terminées, le lot de cinq séances est reconstruit une fois au premier bloc, puis peut être conservé à tort lorsqu'on reprend au milieu d'un bloc ultérieur. `previous_daily` reste absent pour le premier diagnostic repris.

**Reproduction avec huit séances synthétiques et interruption avant la huitième :** prix J+7 de la politique cinq séances = **181,7 sans interruption, 181,2 après reprise** ; variation d'une révision quotidienne = **0,1 contre 0,7** ; diagnostics de la dernière séance = **20 contre 0**.

Cela démontre le défaut de reprise ; l'audit n'établit pas qu'une interruption a eu lieu pendant les études terminées.

**Correction :** restaurer le dernier état de chaque piste, l'index exact du bloc et les lots précédents, ou rejouer ces états à partir des événements enregistrés. Exiger l'égalité des sorties d'un calcul continu et d'un calcul interrompu à plusieurs positions, y compris après un changement de bloc.

### 4. Une correction de cours à date inchangée peut rester invisible pendant une heure

**Localisation :** `src/models/improved_price_predictor.py:184`, `:201` ; `src/api/app.py:1382`.

Le cache Redis dispose d'une empreinte des prix, mais le cache interne du prédicteur ne vérifie que la date de dernière séance et une durée de 3 600 secondes. Après une correction de cours sans nouvelle date, Redis peut manquer son cache, puis le prédicteur recalculer sur son ancien historique. Cette réponse peut ensuite être stockée sous l'empreinte récente.

**Reproduction :** la source simulée annonce 999 pour la même date ; le prédicteur conserve 105.

**Correction :** utiliser le même instantané et sa même empreinte pour charger les données, prédire, mettre en cache et journaliser. Une correction de valeur doit invalider les deux caches. Tester une correction ancienne et une correction de la dernière séance, sans changer leurs dates.

### 5. Les empreintes ne couvrent pas toutes les entrées ni les modèles directs

**Localisation :** `src/models/revision_command.py:29` ; `scripts/run_revision_stability.py:131` ; `src/models/release_manifest.py:242` ; `src/api/app.py:497`.

- Le journal et le cache API ne couvrent pas OHLC et volume, pourtant utilisés par le cacao.
- L'étude ne couvre que date/prix, pas OHLC, volume ou open interest.
- L'empreinte commune omet notamment le script de stabilité et le code du formateur direct.
- Hacher le JSON `direct_info` ne hache pas les 30 fichiers de modèles qu'il référence.

**Reproductions :** changer le volume ne change pas l'empreinte du journal ; changer l'OI ne change pas celle de l'étude ; modifier le code du formateur direct ou un fichier de modèle référencé laisse l'empreinte de procédure inchangée.

**Correction :** définir une identité commune comprenant toutes les colonnes réellement consommées, leur normalisation, le calendrier, les paramètres, les versions pertinentes et les octets de chaque artefact effectivement chargé. Enregistrer la liste exacte des origines et tous les arguments de reprise. Un changement pertinent doit produire une nouvelle identité ou interdire la reprise.

### 6. Les lignes de données et le calendrier de prévision ne décrivent pas les mêmes séances

**Localisation :** `src/models/hybrid_features.py:264` ; `src/validation/revision_stability.py:26` ; `scripts/run_revision_stability.py:300`.

Le nettoyage garde les dates hors calendrier et les études incrémentent leur cadence sur les lignes présentes. L'instantané cacao contient les samedis **29 août et 12 septembre 2026**, présents dans les 92 dates de suivi. Le robusta contient aussi le **3 octobre**. Le cacao contient en outre 25 dates que le calendrier du projet déclare fermées, le robusta une. Cela établit une contradiction interne ; il faut vérifier les sources avant de décider si une date ou le calendrier doit être corrigé.

Les retards de variables et les blocs de cinq lignes peuvent ainsi différer de cinq séances de cotation. À l'inverse, une séance manquante est simplement sautée. `study_window()` filtre en outre sur la présence future de la cible J+30 avant de choisir les 60 origines : une cible manquante retire son origine et en fait entrer une plus ancienne.

**Correction :** distinguer date de cotation et date de collecte ; réconcilier les anomalies avec la source, sans déplacer ni interpoler les cours arbitrairement. Fixer la liste de séances attendues et les origines avant de compter les données disponibles. Signaler les cibles absentes et les exclure explicitement du score, sans remplacer leur origine silencieusement.

## Autres corrections nécessaires — P2

### 7. L'entraînement et le service n'appliquent pas la même règle à l'open interest cacao

**Localisation :** `src/models/hybrid_features.py:180`, `:226` ; `src/models/improved_price_predictor.py:241`.

Le chargement d'entraînement applique les reçus datés d'OI ; le chargement du service lit l'OI brut et propage la dernière valeur disponible. Sur la même source simulée, l'entraînement conserve zéro valeur OI renseignée, le service deux. L'instantané cacao étudié a 1 735 valeurs OI manquantes ; ajouter ces variables ne fournit donc pas actuellement l'information attendue dans cette étude. L'audit ne chiffre pas un impact sur le modèle courant, qui peut ignorer des variables toujours nulles.

**Correction :** unifier le chargement et la disponibilité temporelle des variables. Tester l'identité des variables produites par l'entraînement, la commande et l'API pour un même instantané.

### 8. L'historique long devient difficile à retrouver dans le tableau de bord

**Localisation :** `src/models/prediction_journal.py:318`, `:333` ; `src/api/app.py:1631` ; `frontend/src/components/dashboard/ForecastTrackingPanel.tsx:38`.

L'API ne propose que les 12 dernières initiales, sans pagination ni recherche. À quatre initiales par jour, une ancienne prévision J+30 quitte les boutons après environ trois jours. Elle reste dans le fichier et accessible par son identifiant connu, mais le tableau de bord ne permet plus de la retrouver après rechargement. Le panneau ne recharge également ses données qu'au montage ou au changement de sélection, pas après un nouvel entraînement tant qu'il reste ouvert.

**Correction :** recherche par date cible/horizon, pagination, échéances suivies persistantes et actualisation après publication d'un lot. Tester la consultation d'un J+30 initial après dix jours de nouvelles prévisions.

### 9. Le journal donne une fausse date de fin d'entraînement

**Localisation :** `src/models/revision_command.py:166` ; `src/api/app.py:559` ; `train_hybrid_improved.py:211`.

`train_end_date` est rempli avec la date d'origine de la prévision. Un modèle utilisé plusieurs jours après son entraînement paraît ainsi entraîné jusqu'à la nouvelle origine. Les métadonnées du lot ne fournissent pas actuellement une borne précise permettant de corriger cette attribution ; `data_period` reste générique.

**Correction :** enregistrer les dates min/max des données d'entraînement, leur empreinte et le détail par moteur dans les métadonnées du lot. La date d'émission, la dernière donnée observée et la fin d'entraînement sont trois informations distinctes. Les anciens champs inconnus doivent rester inconnus.

### 10. `update_system.bat` n'établit pas que le nouveau lot est effectivement consultable

**Localisation :** `update_system.bat:163`, `:178`, `:181`, `:222`, `:256` ; `train_nhits.py:49`.

Après le redémarrage cacao, l'épuisement de l'attente API n'interrompt pas la commande. Le contrôle de prévision affiche version et prix sans vérifier la version attendue ni les quatre statuts ; son bloc d'erreur affiche un avertissement. Le redémarrage robusta n'attend pas la disponibilité de l'API avant son contrôle. Une page `/docs` joignable ne prouve pas que les artefacts attendus sont chargés.

Par ailleurs, N-HiTS robusta se réentraîne depuis Supabase et n'accepte pas l'instantané utilisé par XGBoost et le journal. La cohérence des données entre les deux moteurs n'est donc pas garantie si la source change pendant le traitement.

**Correction :** contrôler les codes de sortie, attendre le service, vérifier lot nommé, empreinte, date, présence des quatre résultats et états techniques ; distinguer réussite, échec et réussite partielle. Un horizon expérimental reste autorisé à être affiché : ces contrôles ne doivent pas réintroduire une condition de performance historique. Donner le même instantané à N-HiTS. Garder le cacao accessible en cas d'échec robusta. La collecte des news, sans effet sur le prix central actuel, ne devrait pas bloquer tout le calcul cacao.

### 11. La nouvelle publication expérimentale fait disparaître l'accès aux évaluations existantes

**Localisation :** `scripts/write_experimental_release.py:102`, `:119` ; `src/api/app.py:1822`.

Le manifeste est réécrit sans `evaluation_report`. Les anciens rapports restent nommés dans les manifestes précédents, mais la vue d'évaluation ne les consulte pas. Vérification locale : `/api/v1/validation/metrics` renvoie **404 sur les deux marchés** avec leur version actuelle, car aucun rapport n'est nommé. Le tableau de bord indique alors qu'aucun rapport n'est rattaché. Les diagnostics quotidiens du batch ne rétablissent pas ce lien.

**Correction :** exposer une liste persistante d'évaluations historiques avec leur lot, procédure, dates et statut de compatibilité. Rattacher un rapport au nouveau lot uniquement après contrôle de compatibilité ; ne pas présenter un ancien résultat comme une validation du nouveau modèle.

### 12. Le tableau de bord et l'étude ne comptent pas les retournements de la même manière

**Localisation :** `src/models/prediction_journal.py:271` ; `scripts/run_revision_stability.py:413` ; `frontend/src/components/dashboard/ForecastTrackingPanel.tsx:113`.

Pour la séquence hausse → neutre → baisse, le suivi compte **un retournement**, l'étude **zéro**. Le passage par une indisponibilité appelle aussi une règle explicite. Comparer leurs compteurs peut donner une impression de stabilité contradictoire.

Le message de l'interface « ce n'est pas un retournement de l'objectif » est déduit uniquement d'une différence entre direction au cours initial et direction au cours récent. Cette différence n'exclut pas qu'une véritable inversion de l'objectif ait aussi eu lieu.

**Correction :** partager une fonction de comptage et une politique explicite pour les neutres et les trous. Déduire le commentaire de la succession réelle des objectifs à référence constante, pas uniquement de deux directions calculées sur la dernière valeur.

## Vérifications et limites

- Commande exécutée : `venv_py311\Scripts\python.exe -m pytest tests/test_forecast_tracking.py tests/test_measurement_consistency.py tests/test_measurement_reliability.py tests/test_direct_horizon_trainer.py tests/test_activation_gate.py -q --disable-warnings`.
- Résultat : **61 réussites, 2 échecs, 1 test ignoré**, 31 avertissements, environ 49 secondes.
- Échec `test_recent_origins_are_the_last_ones` : le test passe un entier à `_origin_indices`, qui attend désormais des dates. Mettre le test à jour en conservant sa vérification de la sélection récente.
- Échec `test_unavailable_card_does_not_present_hold` : le texte attendu « Prévision indisponible » est devenu « Indisponible ». Il s'agit d'une attente textuelle obsolète, pas de la preuve d'un prix inventé.
- Le test Windows/Docker est conditionnel et a été ignoré ; aucune validation réelle de concurrence entre ces deux environnements n'est revendiquée.
- `node node_modules/typescript/bin/tsc --noEmit --incremental false` : réussi dans `frontend`.
- Contre-exemples isolés : `reports/audit_repros_2026_10_10.py`, exécuté avec succès. Les modèles factices servent uniquement à vérifier le calcul de la procédure. Ses résultats de précision proviennent des véritables fichiers de l'étude.
- API : lectures authentifiées uniquement, sans afficher de secret. Aucun appel POST de génération ou écriture Supabase pendant cet audit.
- Pas de nouvelle étude longue, pas d'essai visuel du navigateur ni de contrôle du VPS. Les conclusions portent sur les chemins inspectés, pas sur une certification exhaustive du projet.

## Ordre d'action proposé

1. **Données cacao et procédure commune :** réconcilier les séances, unifier OI/variables, charger un instantané unique et corriger les identités/caches.
2. **Mesure cacao :** réparer entrées à jour pour modèles gelés, reprise, notation après maturité, référence inchangée et compteurs. Vérifier l'équivalence avec le service sur quelques séances, puis l'équivalence calcul continu/repris.
3. **Usage quotidien :** sécuriser les contrôles du batch, garder les anciennes échéances consultables, conserver les diagnostics historiques et les vraies dates d'entraînement.
4. **Nouvelle étude cacao :** figer le protocole, exécuter l'étude corrigée et comparer stabilité ET erreur sur les mêmes cibles ; rapporter les absences et l'incertitude. Refaire le robusta ensuite.
5. **Amélioration des modèles seulement après une mesure cohérente :** petit nombre de variantes prédéfinies par horizon, avec par exemple modèle direct moins complexe, cible normalisée plutôt que variation absolue, fenêtres d'apprentissage différentes. Régler sur une période de développement, conserver une période chronologique distincte pour la décision et tenir compte du chevauchement des horizons. Les 60 origines déjà analysées restent des données de diagnostic, pas une nouvelle preuve indépendante après réglage.

Les quatre prix expérimentaux peuvent rester visibles pendant ces travaux. Aucun lissage forcé ni interdiction de changer de direction ne doit être présenté comme une amélioration de précision. Aucune activation automatique n'est nécessaire pour mesurer les progrès.
