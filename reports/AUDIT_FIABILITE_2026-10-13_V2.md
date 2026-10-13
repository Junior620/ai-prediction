# Audit de fiabilité — 13 octobre 2026, deuxième passe

Audit effectué sur le code présent dans `E:\prediction` entre environ 17 h 55 et 18 h 10, heure de Douala, après les corrections et l'ajout des cartes conservées et de l'évaluation des révisions. Cacao prioritaire ; robusta contrôlé sur les composants partagés et la collecte. Ce rapport complète celui du matin et distingue les corrections vérifiées des défauts encore présents.

**Avis : le projet est mieux traçable, mais le nouveau verdict « avantage historique des révisions établi » ne répond pas encore à la question « la révision du lendemain est-elle meilleure que la prévision conservée ? ».** Les corrections doivent maintenant porter sur cette comparaison, la cohérence des objets affichés et l'intégration entre composants. Ajouter un modèle ou figer davantage les flèches ne résout pas ces défauts.

Le code métier, les modèles, les manifestes et les données de marché n'ont pas été modifiés par cet audit. Aucun entraînement, backtest long, déploiement ou POST de prédiction n'a été exécuté. Les contre-exemples utilisent des fichiers temporaires et des services simulés. Les appels réels à l'API sont des GET de lecture.

**Éléments vérifiés**

- 99 tests ciblés réussis, 1 ignoré, 31 avertissements, en 23,65 s. Le seul test ignoré est le contrôle conditionnel du journal partagé Windows/Docker. Les fichiers couvrent suivi, mesure, préparation identique, valeur des révisions, collecte ICE, activation et modèles directs.
- Frontend : `tsc --noEmit --incremental false` réussi.
- Reproductions : `reports/audit_repros_2026_10_13_v2.py` ; résultats : `reports/audit_evidence_2026_10_13_v2.jsonl`.
- API locale : santé annoncée `healthy`, cacao `improved_20261013_045358`, robusta `improved_20261013_045834`. Les deux manifestes expérimentaux restent non validés.
- L'API `/api/v1/revision-value` renvoie `etabli` pour J+7, J+14 et J+30 sur les deux marchés. Le premier GET cacao a dépassé 25 s ; une seconde lecture a réussi en 10,52 s. Ce constat ponctuel ne suffit pas à identifier la cause de la lenteur.
- L'historique cacao retourne désormais 52 rapports et inclut les dernières études. Leur rattachement au bon marché reste imparfait, voir point 8.

**Corrections précédentes effectivement présentes**

Le remplissage OHLCV est maintenant commun au chemin de l'instantané et à celui de l'API ; le test utilisant le véritable lot cacao du 13 octobre passe. L'ancien écart local/API de 4 259,12 contre 4 328,63 £ ne doit donc pas être présenté comme encore reproduit sur le nouveau chemin.

Les identités brute et transformée sont séparées. Les dépendances principales ajoutées à l'empreinte comprennent désormais le trainer, le prédicteur, les poids et la règle des horizons. Les écritures de l'étude disposent d'un verrou, d'une déduplication et d'un rejet des contradictions. Une synthèse cacao sur 90 séances uniques a été produite sans écraser l'archive dupliquée. Le score de stabilité comporte maintenant des effectifs attendus, des abstentions et des références appariées.

La collecte ICE distingue date de séance et date de récupération, et met les observations sans date en quarantaine. Le vérificateur de fin de commande rejette maintenant les prix absents. Les cartes conservent l'initiale, indiquent l'échéance et le nombre de séances restantes et peuvent afficher une révision visant la même cible. Ces progrès sont réels ; les contre-exemples ci-dessous portent sur leurs limites ou leur intégration.

**1. [P1] Le verdict favorable aux révisions compare surtout J+14 à J+1, pas J+14 à J+13.**

Localisation : `src/validation/revision_value.py:58`, `:101`, `:176` ; `frontend/src/components/dashboard/PredictionHorizonCard.tsx:174`.

`_latest_revision` sélectionne la dernière révision disponible avant la fin de chaque trajectoire archivée. Pour les 60 cibles cacao de chacun des trois horizons, cette révision est systématiquement à **une séance de l'échéance**. La comparaison est bien faite sur la même date cible ; le problème est son interprétation et son utilisation sur une carte dont il reste, par exemple, treize séances.

Recalcul des erreurs absolues moyennes sur les mêmes cibles cacao, en £/tonne :

| Horizon initial | Erreur initiale | Erreur de la première révision | Gain de cette première révision | Révisions améliorantes |
| --- | ---: | ---: | ---: | ---: |
| J+7 → J+6 | 334,23 | 325,41 | +8,82 | 58,3 % |
| J+14 → J+13 | 507,08 | 511,16 | −4,07 | 43,3 % |
| J+30 → J+29 | 805,73 | 946,57 | −140,83 | 35,0 % |

Gain = erreur initiale − erreur révisée. Un gain négatif signifie davantage d'erreur. Il y a 60 paires par ligne et la première révision correspond ici exactement à la séance suivante.

Le gain actuellement affiché par l'API est pourtant de **+386,85 £ pour les initiales J+14** et **+699,69 £ pour les initiales J+30** : il s'agit du gain obtenu en attendant la révision à J+1. Ce résultat répond à « que vaut la dernière estimation juste avant l'échéance ? », pas à « dois-je préférer l'estimation recalculée demain ? ».

Même la dernière estimation reste descriptivement moins bonne que le cours disponible lors de cette dernière révision, sur ces mêmes cibles : erreurs de 117,81 contre 108,60 £ pour les initiales J+7 ; 120,23 contre 111,33 £ pour J+14 ; 106,04 contre 104,38 £ pour J+30. Battre l'ancienne prévision ne démontre donc pas à lui seul une valeur ajoutée du modèle.

Correction : produire une mesure par transition ou par horizon restant, par exemple initiale J+14 contre révision J+13, puis J+12, etc. Distinguer aussi cette comparaison de celle entre deux révisions successives. Afficher la mesure correspondant à l'âge actuel de la carte, avec la référence « cours connu au moment de la révision » et les effectifs disponibles. Si l'on conserve l'indicateur actuel, le nommer explicitement « initiale contre dernière estimation avant échéance », et ne pas l'utiliser comme justification automatique d'une révision du lendemain.

Ces chiffres proviennent du rejeu historique existant, pas du nouveau lot servi aujourd'hui. Ils ne constituent pas une promesse de précision future. Les intervalles du module actuel ne sont pas traités ici comme une certification, notamment à cause du point suivant.

**2. [P1] L'intervalle statistique peut se réduire artificiellement à un point et déclarer un avantage établi.**

Localisation : `src/validation/revision_value.py:124` à `:167`.

La largeur du bloc est `min(horizon, nombre_de_paires)`. Quand le nombre de paires est inférieur ou égal à l'horizon, chaque tirage circulaire reprend exactement toutes les valeurs, dans un ordre différent. Toutes les moyennes tirées sont donc identiques. Le seuil de huit échéances ne protège pas contre cette situation.

Contre-exemple exécuté : huit gains `[10, 10, 10, 10, 10, 10, 10, -60]`, horizon 14. Le programme renvoie moyenne **+1,25**, intervalle **[+1,25 ; +1,25]** et verdict **`etabli`**, malgré une dispersion importante. L'absence d'incertitude provient du tirage, pas de huit observations parfaitement concordantes.

Les jeux actuels comportent davantage de lignes que l'horizon ; ce contre-exemple ne signifie pas que leurs intervalles sont tous exactement nuls. Cependant, le robusta J+30 ne comporte que 33 paires pour une largeur de 30 : un seuil portant seulement sur le nombre brut d'échéances ne suffit pas pour juger la quantité d'information indépendante.

Correction : imposer un critère portant sur les blocs effectivement exploitables, refuser le verdict établi lorsque le rééchantillonnage dégénère, expliciter le niveau d'intervalle et vérifier la sensibilité au choix des blocs. Ajouter des essais de couverture sur séries dépendantes simulées et réserver une période de confirmation indépendante avant toute règle de sélection. Ne pas corriger ce défaut en réduisant arbitrairement le bloc jusqu'à obtenir un verdict favorable.

**3. [P1] Une révision valide d'une ancienne cible peut être transformée en simple diagnostic et disparaître du suivi.**

Localisation : `src/models/prediction_journal.py:335` à `:340`.

`append_forecast` transforme une prévision en diagnostic dès que son horizon est 7, 14 ou 30 et qu'une carte de cet horizon est déjà ouverte. Cette décision ne vérifie pas si le calcul vise aussi la cible d'une autre initiale conservée.

Contre-exemple exécuté : une initiale J+30 ouverte le 1er octobre vise le 12 novembre. Une nouvelle carte J+14 est ouverte le 21 octobre. Le 23 octobre, une estimation J+14 vise précisément le 12 novembre et devrait donc réviser l'ancienne J+30. Elle est enregistrée avec `published=false`, `diagnostic=true`, `revises=[]`. La dernière estimation de cette cible reste alors obsolète dans le suivi, alors qu'un calcul plus récent existe.

Correction : séparer « ouvrir une nouvelle carte de cet horizon » et « réviser une initiale de cette date cible ». Empêcher l'ouverture d'une carte supplémentaire ne doit pas empêcher une révision visant une cible existante. Une ligne peut ne pas ouvrir de nouvelle initiale tout en révisant une ou plusieurs initiales identifiées. Tester les passages d'une trajectoire J+30 par les horizons restants 14 et 7, ainsi qu'un nouveau calcul à la même séance.

**4. [P1] La carte conservée mélange l'ancien prix avec des informations du nouveau calcul.**

Localisation : `src/models/prediction_journal.py:246` à `:282` ; `src/api/app.py:1422`, `:1559`, `:1586`.

L'overlay copie d'abord le calcul frais, remplace le prix par l'initiale, puis ne remplace la bande que si l'initiale en possède une. Les composantes et leur date cible restent celles du calcul frais.

Deux contre-exemples exécutés :

- Initiale à **4 500 £ sans intervalle**, calcul frais à **4 000 £ avec intervalle [3 900 ; 4 100]** : la carte conservée devient **4 500 £ avec [3 900 ; 4 100]**, et sa cible affichée diffère encore de celle des composantes. L'intervalle ne doit pas être emprunté à une autre prévision.
- Si le calcul frais échoue, l'overlay rétablit l'ancien prix à 4 500 £ et le statut expérimental tout en conservant `calculation_failed=true`. Au passage dans la même fonction de publication utilisée lors d'une lecture du cache, cette carte redevient `unavailable` avec prix absent. Les deux chemins peuvent ainsi donner des états différents pour la même carte.

Le niveau de réponse porte également la version courante, alors que le prix conservé peut provenir du lot du 6 octobre. La version de l'initiale et celle de la révision doivent rester distinguables dans les données de la carte.

Correction : construire un objet explicite pour l'initiale, un autre pour la dernière estimation, avec prix, bande nullable, cible, date d'émission, version, statut et avertissements propres à chacun. L'absence de bande initiale doit produire `null`, jamais la bande fraîche. Appliquer une politique de publication cohérente sur le résultat final, identique avec ou sans cache. Si la dernière révision échoue, garder l'initiale consultable et indiquer « dernière estimation indisponible », plutôt que « pas encore de révision ».

**5. [P1] Le contrôle de fin de mise à jour et la réponse J+1 ne respectent pas le même contrat.**

Localisation : `src/api/app.py:1480`, `:1586` ; `src/api/models.py:76` ; `scripts/verify_served_forecast.py:53`.

Le vérificateur exige `target_date` au niveau de chacun des quatre objets de prédiction. Lors de la construction de la réponse, l'API garde cette date dans `components`, sans remplir le champ principal. L'overlay le remplit ensuite pour les cartes longues seulement. J+1 reste donc sans `target_date` au niveau attendu.

Reproduction avec les mêmes modèles de réponse : J+1 possède un prix positif, un statut expérimental et `components.target_date = 2026-10-14`, mais le champ principal est `null`. Le vérificateur renvoie **code 3**, `[PARTIEL] … J+1 échéance absente`. Ce contrôle n'est donc pas encore une preuve exploitable que la commande a correctement fini.

Autre limite du contrôle : la version vérifiée est celle de la réponse globale. Les anciens prix conservés peuvent satisfaire le contrôle de présence même si le nouveau lot ne fournit pas ses nouvelles estimations longues. Il faut définir séparément la disponibilité des initiales et celle du nouveau calcul.

Correction : renseigner les dates cibles dès la conversion des prédictions, pour tous les horizons ; contrôler les identités et dates de chaque objet réellement évalué. Tester la réponse complète telle que l'API la construit, puis la passer au vrai vérificateur. Les tests actuels lui fournissent directement des réponses contenant toutes les dates et ne couvrent pas ce raccordement.

**6. [P1] La correction du collecteur Investing casse le programme de collecte robusta.**

Localisation : `src/data_collection/investing_price_collector.py:139` ; `collect_coffee_robusta_price.py:63` ; `update_system.bat:195`.

Le collecteur partagé retourne maintenant `session_date`, `collected_at` et un motif de quarantaine, et ne retourne plus `date`. Le programme robusta accède toujours à `data["date"]`.

L'exécution du programme avec Supabase et le collecteur entièrement simulés reproduit **`KeyError: 'date'`**, avant toute tentative d'insertion. Réessayer la collecte ne corrige pas cette incompatibilité ; le batch finit par ignorer la mise à jour robusta. Le cacao peut continuer indépendamment, conformément à l'ordre du batch.

Correction : migrer aussi le consommateur robusta vers le nouveau schéma, traiter explicitement une séance absente par quarantaine et statut explicatif, puis intégrer une source de date vérifiable. Ne pas rétablir `date=aujourd'hui` pour faire disparaître l'erreur. Ajouter un test d'intégration collecteur → commande robusta, avec date valide et sans date.

**7. [P1] Les calendriers de l'étude et du service restent différents, même si cette différence est maintenant déclarée.**

Localisation : `src/models/hybrid_features.py:565`, `:613` ; `scripts/run_revision_stability.py:150`, `:251`.

Le service conserve explicitement les lignes fermées (`keep_closed_rows`), tandis que l'étude applique `trading_sessions`. Le snapshot cacao comporte encore **1 740 lignes contre 1 712 après ce filtre**, dont le samedi 10 octobre. Déclarer ces conventions dans les empreintes est un progrès de traçabilité, mais ne rend pas les deux procédures équivalentes.

Il est raisonnable de ne pas changer silencieusement les entrées d'un ancien lot déjà entraîné. En revanche, ses résultats ne peuvent pas être assimilés à une étude utilisant un autre calendrier. L'archive actuelle précède également la dernière harmonisation OHLCV ; ses performances restent historiques.

Correction : faire une migration explicite, avec vérification du calendrier propre au marché et des observations suspectes, snapshot conservé, nouveau lot et nouvelle mesure utilisant la même préparation de bout en bout. Le choix de conserver l'ancien lot doit rester visible tant que cette migration n'est pas réalisée. Ne pas simplement supprimer les lignes en service sans réentraîner. Les dates classées fermées par le calendrier du projet ne sont pas toutes certifiées comme telles par cet audit.

**8. [P2] L'index attribue certains rapports robusta au cacao.**

Localisation : `src/validation/report_index.py:86` à `:98`.

La recherche cacao exclut seulement le dossier robusta directement sous `walk_forward`. Elle descend ensuite dans tous les autres sous-dossiers et ramasse notamment :

- `walk_forward/measure/coffee_robusta/20261009_150625_summary.json` ;
- `walk_forward/trial_measure/coffee_robusta/20261009_124610_summary.json`.

Ces deux rapports apparaissent dans l'index cacao et sont absents de l'index robusta, qui ne parcourt que `walk_forward/coffee_robusta`. Les derniers rapports sont donc désormais listés, mais leur classification n'est pas entièrement fiable.

Correction : déterminer le marché à partir des métadonnées vérifiées du rapport et d'une convention de chemin explicite, plutôt que par la seule exclusion d'un dossier de premier niveau. Rejeter ou signaler les contradictions. Ajouter un test avec les véritables arborescences `measure/<marché>` et `trial_measure/<marché>`. Prévoir aussi une consultation des métriques du rapport : la liste actuelle du tableau de bord ne montre que les six premières notices.

**Limites à conserver visibles dans l'interprétation**

L'évaluation des révisions lit les prix réalisés dans `data/freeze`, chemin qui est réécrit par les mises à jour quotidiennes, et ne vérifie pas l'identité archivée des données. Les futurs rapports doivent enregistrer la version des observations utilisées pour noter les cibles, la procédure, la fenêtre et les absences. Une correction de prix réalisé peut légitimement changer une erreur, mais doit être traçable. Les révisions sans prix sont actuellement écartées lors de la sélection de la dernière estimation : le taux de disponibilité doit accompagner le gain annoncé.

Les 60 cibles historiques cacao ne constituent pas 60 expériences indépendantes à J+30. Le diagnostic historique ne choisit pas automatiquement le nouveau lot. Pour le robusta, il n'y a que 48, 44 et 33 paires réalisées respectivement pour J+7, J+14 et J+30, et la première révision disponible ne tombe pas toujours à la séance suivante en raison des trous de données. Il faut filtrer sur l'horizon restant réellement comparé plutôt que qualifier toutes ces lignes de « lendemain ».

La mesure initiale du cacao reste celle de l'étude corrigée : MAPE quotidienne 2,71 %, 8,33 %, 12,26 %, 18,83 % pour J+1/7/14/30, contre 2,50 %, 8,21 %, 10,68 %, 11,67 % pour le cours inchangé. La synthèse unique du 13 octobre ne correspond pas à un nouvel entraînement ou à un nouveau backtest démontrant une amélioration.

**Ordre de correction recommandé**

1. Corriger les objets des cartes et les liens de révision : même cible, initiale intacte, dernière estimation complète, bande et statut propres, comportement identique via le cache. Cela concerne les deux marchés.
2. Corriger la réponse J+1 et la vérification de `update_system.bat`, puis le consommateur Investing robusta. Garder le cacao prioritaire et les statuts partiels explicites.
3. Remplacer le verdict global sur les révisions par une comparaison correspondant au délai restant. Ajouter la référence du cours disponible à la révision, les absences et une règle d'incertitude qui ne dégénère pas. Les sorties des replays existants permettent déjà cette analyse descriptive sans réentraîner.
4. Préparer la migration calendrier/données du cacao et vérifier la parité entraînement → commande → API → étude sur un essai court. Une nouvelle mesure longue n'a de sens qu'après cette parité.
5. Sur la procédure figée, comparer modèles et cadences sur une période de développement, puis confirmer sur une période restée à l'écart des choix. Garder les horizons J+1, J+7, J+14 et J+30 visibles comme expérimentaux ; aucune promotion automatique ni déploiement n'est demandé par ce rapport.

Le critère de réussite est double : **les calculs doivent être cohérents et la nouvelle estimation doit apporter un gain mesuré pour le délai qui t'intéresse réellement**. Une carte stable ou un résultat favorable à la veille de l'échéance ne suffit pas à démontrer que la révision du lendemain est meilleure.
