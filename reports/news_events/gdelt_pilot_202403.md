# Pilote GDELT mars 2024

Deux passages. Le second a été refusé par le quota HTTP 429 et n'a pas pu remplacer les articles du premier.

## API DOC 2.0

Le premier passage a reçu 225 articles sur trois semaines (8, 15 et 22 mars 2024), 75 par semaine. La semaine du 1er mars a répondu 429. Le champ `seendate` de GDELT a la forme `20240315T153000Z`. Le premier lecteur exigeait 14 chiffres d'affilée et a donc classé ces 225 lignes comme non rejouables. Ce lecteur accepte maintenant le `T`.

Le passage suivant, après correction, n'a obtenu aucun article : les quatre semaines ont répondu 429. On ne peut pas encore publier la part de `seendate` réellement parseable sur ce mois.

`seendate` reste une heure d'observation GDELT. Elle ne prouve pas que le texte lu aujourd'hui est celui qui existait à cette heure. Sans cette heure, la ligne reste hors des backtests.

## GKG 2.0 et Mentions

Un seul créneau de 15 minutes, `20240315120000`, pas le mois entier.

- GKG : fichier d'environ 5,8 Mo, 1 482 lignes lues, 1 ligne contenant cocoa ou cacao.
- Mentions : fichier d'environ 131 Ko. Aucune URL du tirage DOC n'y a été reconnue, parce que ces URL n'avaient pas encore d'heure d'observation retenue.

Cette densité ne justifie pas de télécharger le mois complet de GKG ou de Mentions, soit des dizaines de gigaoctets.

## Conséquence

L'historique 2020-2026 n'est pas rejouable à partir de ce pilote. L'entraînement des trois candidats reste bloqué. La collecte quotidienne peut commencer à dater les événements avec `first_seen_at`.
