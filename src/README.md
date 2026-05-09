# Organisation canonique du code source

Cette arborescence suit la logique méthodologique du mémoire.

- `database_construction/`
  Construction, nettoyage, fusion et préparation de la base de données.
- `csds_implementation/`
  Implémentation, calibration, validation et reconstruction du modèle CSDS.
- `descriptive_stats/`
  Statistiques descriptives et tableaux de synthèse.
- `scatter_plots/`
  Nuages de points et heatmaps exploratoires.
- `histograms/`
  Histogrammes et boîtes à moustaches.
- `methodologie_1/`
  Estimation indirecte via `d`, puis reconstruction de `e` et `b`.
- `methodologie_2/`
  Estimation de `z = log(e - c)`, puis reconstruction de `e`, `d` et `b`.
- `comparison/`
  Comparaison des méthodologies 1 et 2 et figures de synthèse.
- `final_model/`
  Variables récurrentes, ACP, recherche de lambda, modèle final et comparaisons.
- `residual_analysis/`
  Analyse des résidus et diagnostics statistiques finaux.
- `utils/`
  Fonctions communes, métriques, helpers CSDS et gestion centralisée des chemins.

Cette structure est maintenant la seule structure active a utiliser pour les nouveaux imports
et pour les futures executions.
