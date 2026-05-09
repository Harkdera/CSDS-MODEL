# TEXTME

Pense-bete rapide du depot.

## Regle generale

- `src/` contient le code
- `results/` contient les sorties
- `src/` suit la structure canonique du memoire et du pipeline
- `results/` suit le miroir des sorties consultables

## Sections de `src/`

- `database_construction/` : construction, nettoyage, preparation et sous-jeux
- `csds_implementation/` : calibration iterative, courbes et validation CSDS
- `descriptive_stats/` : tableaux descriptifs
- `scatter_plots/` : nuages de points
- `histograms/` : histogrammes et boites a moustaches
- `methodologie_1/` : branche indirecte `d -> e`
- `methodologie_2/` : branche directe `log(e-c) -> e`
- `comparison/` : comparaison, classement, interpretation
- `final_model/` : variables recurrentes, ACP, lambda et evaluations finales
- `residual_analysis/` : diagnostics des residus
- `utils/` : helpers communs et chemins

## Sections de `results/`

- `database-construction/`
- `csds-implementation/`
- `descriptive-stats/`
- `scatter-plots/`
- `histograms/`
- `methodologie_1/`
- `methodologie_2/`
- `comparison/`
- `final-results/`
- `residual-analysis/`

## Sous-sections typiques

- base de donnees : `merge/`, `preparation/`, `splits/`
- methodologie 1 : `heuristic/`, `cross_validation/`, `reconstruction/`
- methodologie 2 : `heuristic/`, `cross_validation/`, `reconstruction/`
- comparaison : `model-ranking/`, `inter-methodologies/`, `interpretation/`
- modele final : `recurrent_variables/`, `pca/`, `lambda_search/`, `regularized_regression/`

## Quand ajouter un script

- si le script produit un tableau descriptif, il va dans `src/descriptive_stats/`
- si le script produit surtout des scatter plots, il va dans `src/scatter_plots/`
- si le script produit surtout des histogrammes, il va dans `src/histograms/`
- si le script sert a la calibration CSDS, il va dans `src/csds_implementation/`
- si le script compare les deux methodologies, il va dans `src/comparison/`
