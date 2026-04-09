# CSDS Project

## Description en français

### Objectif

Ce projet vise a mieux comprendre les parametres du modele CSDS afin de proposer une nouvelle approche de calibration.

Actuellement, la calibration du modele repose sur une procedure iterative. L'idee developpee ici est d'estimer d'abord le parametre `d`, puis de calculer `e` a l'aide de la formule proposee par Simon lorsque `d` est connu. Cette approche est ensuite comparee a la methode iterative classique afin d'en evaluer la pertinence.

### Methodologie

Le workflow du projet combine :
- l'extraction des donnees
- le nettoyage des donnees
- la construction des parametres
- la calibration du modele CSDS
- le filtrage des cas converges
- la visualisation
- la regression et la comparaison de modeles

### Resultats attendus

Les principaux resultats que l'on peut obtenir dans ce projet sont :
- une analyse exploratoire des variables (EDA)
- l'etude des relations entre les parametres du modele CSDS
- la recherche genetique de combinaisons pertinentes de variables
- la comparaison de plusieurs regressions pour approximer `d`
- la comparaison entre la methode proposee et la methode iterative classique

## English Description

### Objective

This project aims to better understand the parameters of the CSDS model in order to propose a new calibration approach.

At the moment, the model is calibrated through an iterative procedure. The idea developed here is to first estimate the parameter `d`, then compute `e` using the formula proposed by Simon when `d` is known. This approach is then compared with the classical iterative method in order to evaluate its relevance.

### Methodology

The project workflow combines:
- data extraction
- data cleaning
- parameter construction
- CSDS model calibration
- filtering of converged cases
- visualization
- regression and model comparison

### Expected Results

The main types of results that can be obtained in this project are:
- exploratory data analysis (EDA) of the variables
- analysis of the relationships between CSDS model parameters
- genetic search for relevant combinations of variables
- comparison of several regression approaches to approximate `d` or `e`
- comparison between the proposed method and the classical iterative method

---

## Project structure

```text
CSDS-MODEL-copy/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── regressions/
├── docs/
├── figures/
│   ├── curves/
│   ├── eda/
│   └── splits/
├── src/
│   ├── 01_extract_raw_table.py
│   ├── 02_build_csds_parameters.py
│   ├── 03_select_csds_columns.py
│   ├── ...
│   └── 16_evaluate_polynomial_models_e.py
├── test/
├── README.md
├── requirements.txt
└── .venv/
