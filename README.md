# CSDS Project

Ce depot sert a construire, calibrer et comparer des modeles empiriques relies au modele CSDS.

Deux branches principales sont etudiees :

- `methodologie_1` : predire `d`, puis reconstruire `e`
- `methodologie_2` : predire `z = log(e-c)`, puis reconstruire `e`, `d` et `b`

## Structure actuelle

Le depot est maintenant organise directement selon la logique du memoire.

### Code

Le dossier `src/` suit maintenant une structure canonique par etape :

- `database_construction/`
- `csds_implementation/`
- `descriptive_stats/`
- `scatter_plots/`
- `histograms/`
- `methodologie_1/`
- `methodologie_2/`
- `comparison/`
- `final_model/`
- `residual_analysis/`
- `utils/`

Chaque section est ensuite decoupee en sous-sections quand c'est utile, par exemple :

- `merge`, `preparation`, `splits`
- `calibration`, `curves`, `validation`
- `heuristic`, `cross_validation`, `reconstruction`
- `recurrent_variables`, `pca`, `lambda_search`, `regularized_regression`

### Resultats

Le dossier `results/` est le miroir des sorties consultables :

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

## Lecture rapide

Si vous voulez suivre la logique du travail :

1. construire la base de donnees
2. calibrer et valider le modele CSDS
4. produire les statistiques descriptives
5. produire les scatter plots
6. produire les histogrammes
7. analyser la `methodologie_1`
8. analyser la `methodologie_2`
9. comparer les methodologies
10. construire le modele final
11. analyser les residus

## Arborescence resumee

```text
python/
|-- data/
|-- results/
|   |-- database-construction/
|   |-- csds-implementation/
|   |-- descriptive-stats/
|   |-- scatter-plots/
|   |-- histograms/
|   |-- methodologie_1/
|   |-- methodologie_2/
|   |-- comparison/
|   |-- final-results/
|   `-- residual-analysis/
|-- src/
|   |-- database_construction/
|   |-- csds_implementation/
|   |-- descriptive_stats/
|   |-- scatter_plots/
|   |-- histograms/
|   |-- methodologie_1/
|   |-- methodologie_2/
|   |-- comparison/
|   |-- final_model/
|   |-- residual_analysis/
|   `-- utils/
|-- PARCOURS_METHODOLOGIQUE.md
|-- TEXTME.md
`-- README.md
```

## Fichiers utiles

- [PARCOURS_METHODOLOGIQUE.md](/Users/hariderarako/Desktop/python/PARCOURS_METHODOLOGIQUE.md) : correspondance entre la demarche du memoire, le code et les resultats
- [TEXTME.md](/Users/hariderarako/Desktop/python/TEXTME.md) : pense-bete rapide de rangement
- [results/README.md](/Users/hariderarako/Desktop/python/results/README.md) : lecture rapide des sections de `results/`
