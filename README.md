# CSDS Project

## Overview

This repository studies the calibration of the CSDS model and compares two alternative strategies to the classical iterative fit:

- `methodologie_1`: estimate `d`, then recover `e` with Simon's relation
- `methodologie_2`: estimate `z = log(e-c)`, then reconstruct `e = c + exp(z)` so that `e > c` is guaranteed by construction

The goal of the project is to better understand the CSDS parameters and evaluate whether a regression-based calibration strategy can compete with, or improve on, the iterative approach.

## Apercu en francais

Ce depot etudie la calibration du modele CSDS et compare deux strategies alternatives a la calibration iterative classique :

- `methodologie_1` : estimer `d`, puis reconstruire `e` avec la relation de Simon
- `methodologie_2` : estimer `z = log(e-c)`, puis reconstruire `e = c + exp(z)` afin de garantir automatiquement `e > c`

L'objectif du projet est de mieux comprendre les parametres du modele CSDS et d'evaluer si une calibration basee sur la regression peut concurrencer, ou ameliorer, l'approche iterative.

## Quick Start

### Environment

```bash
python3 -m venv mdst_env
source mdst_env/bin/activate
pip install -r requirements.txt
```

### Core pipeline

Run the shared preprocessing first:

```bash
python src/01_extract_raw_table.py
python src/02_build_csds_parameters.py
python src/03_select_csds_columns.py
python src/04_fit_csds_model.py
python src/05_export_converged_cases.py
python src/06_plot_csds_curves.py
python src/07_explore_converged_csds.py
python src/08_split_tau_peak_low_high.py
python src/09_split_tau_peak_low_subgroups.py
```

Then choose one branch:

Indirect branch:

```bash
python src/methodologie_1/10_search_polynomial_models_d.py
python src/methodologie_1/11_search_polynomial_models_log_d.py
python src/methodologie_1/12_search_exponential_models_d.py
python src/methodologie_1/13_summarize_best_models.py
python src/methodologie_1/14_evaluate_e_from_retained_d_models.py
python src/methodologie_1/15_evaluate_b_from_retained_d_models.py
python src/methodologie_1/16_compare_d_b_e_and_plot_tau_u.py
```

Direct constrained branch:

```bash
python src/methodologie_2/11_explore_methodologie_2_targets.py
python src/methodologie_2/12_prepare_log_e_minus_c_targets.py
python src/methodologie_2/13_search_exponential_models_log_e_gap.py
python src/methodologie_2/14_search_polynomial_models_log_e_gap.py
python src/methodologie_2/15_summarize_best_e_models.py
python src/methodologie_2/16_evaluate_b_d_from_retained_e_models.py
python src/methodologie_2/17_compare_methodologie_2_with_methodologie_1.py
python src/methodologie_2/18_compare_b_d_e_and_plot_tau_u.py
```

## Current Research Question

The project compares two non-iterative calibration routes:

1. predict `d`, then compute `e`
2. predict constrained `e` directly through `log(e-c)`, then reconstruct `d` and `b`

The final comparison is based on:

- parameter accuracy on `d`, `b`, and `e`
- validation and held-out test performance
- cross-validation stability
- reconstructed `tau(u)` curve accuracy
- physical admissibility of the predicted parameters

## Genetic Heuristic for Variable Selection

The regression stages in both `methodologie_1` and `methodologie_2` use a genetic heuristic to explore combinations of candidate variables instead of testing every possible subset exhaustively.

In this project, the candidate predictors include not only the original measured variables, but also engineered variables such as products, ratios, and transformed terms. Because the number of possible combinations becomes very large, a full combinatorial search would be impractical.

In the genetic heuristic, each individual represents one candidate subset of predictors. The algorithm then:

- generates an initial population of feature combinations
- fits a regularized regression model for each combination
- evaluates each candidate using validation performance and cross-validation stability
- keeps the strongest candidates through selection
- generates new candidates through crossover and mutation
- repeats the process over several generations

This procedure is used to identify strong and diverse subsets of variables for:

- search for predictor combinations for `d` in the indirect branch
- search for predictor combinations for `log(e-c)` in the direct constrained branch

The retained models are then ranked, summarized, and re-evaluated in later stages of the workflow. Final comparisons are not based only on regression performance, but also on reconstructed parameter accuracy and on the quality of the reconstructed `tau(u)` curves.

## Data Source and Reference Documents

The data used in this project come primarily from the paper by Deiminiat, Aubertin, and Ethier (2024), which presents an updated calibration procedure for the CSDS model, and from Richard Simon's doctoral thesis (1999), *Analysis of fault-slip mechanisms in hard rock mining*, where the original CSDS formulation was introduced.

These documents are used as the main references for:

- the CSDS model formulation
- the interpretation of parameters `a`, `b`, `c`, `d`, and `e`
- the indirect calibration route based on estimating `d` and reconstructing `e`
- the direct constrained route based on estimating `log(e-c)`

### Main references

- Deiminiat, A., Aubertin, J. D., & Ethier, Y. (2024). *On the calibration of a shear stress criterion for rock joints to represent the full stress-strain profile*. *Journal of Rock Mechanics and Geotechnical Engineering, 16*, 379-392. [https://doi.org/10.1016/j.jrmge.2023.07.019](https://doi.org/10.1016/j.jrmge.2023.07.019)
- Simon, R. (1999). *Analysis of fault-slip mechanisms in hard rock mining* [Doctoral dissertation, McGill University].
- Asadollahi, P., & Tonon, F. (2010). *Constitutive model for rock fractures: Revisiting Barton's empirical model*. *Engineering Geology, 113*(1-4), 11-32. [https://doi.org/10.1016/j.enggeo.2010.01.007](https://doi.org/10.1016/j.enggeo.2010.01.007)

Local working copies are kept in `docs/` under descriptive filenames:

- `docs/Deiminiat_Aubertin_Ethier_2024_On_the_calibration_of_a_shear_stress_criterion_for_rock_joints_to_represent_the_full_stress_strain_profile.pdf`
- `docs/Simon_1999_Analysis_of_fault_slip_mechanisms_in_hard_rock_mining_McGill_University.pdf`
- `docs/Asadollahi_Tonon_2010_Constitutive_model_for_rock_fractures_Revisiting_Bartons_empirical_model.pdf`

## Project Structure

```text
python/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── csds_parameters_converged_only.csv
├── docs/
├── results/
│   ├── datasets/
│   ├── figures/
│   │   └── curves/
│   ├── eda/
│   │   └── descriptive_statistics/
│   ├── split/
│   ├── methodologie_1/
│   │   ├── figures/
│   │   │   └── models/
│   │   ├── regressions/
│   │   │   └── top5/
│   │   ├── e_from_all_retained_d_models/
│   │   ├── b_from_all_retained_d_models/
│   │   └── compare_d_b_e_tau_u/
│   └── methodologie_2/
│       ├── datasets/
│       ├── eda/
│       ├── regressions/
│       │   └── top5/
│       ├── evaluations/
│       ├── compare_b_d_e_tau_u/
│       └── figures/
│           └── models/
│   ├── comparaison/
│       └── methodologie_2_vs_methodologie_1/
├── src/
│   ├── 01_extract_raw_table.py
│   ├── 02_build_csds_parameters.py
│   ├── 03_select_csds_columns.py
│   ├── 04_fit_csds_model.py
│   ├── 05_export_converged_cases.py
│   ├── 06_plot_csds_curves.py
│   ├── 07_explore_converged_csds.py
│   ├── 08_split_tau_peak_low_high.py
│   ├── 09_split_tau_peak_low_subgroups.py
│   ├── methodologie_1/
│   └── methodologie_2/
├── README.md
└── requirements.txt
```

## Inputs and Outputs

### Main inputs

- main converged dataset:
  [data/processed/csds_parameters_converged_only.csv](/Users/hariderarako/Desktop/python/data/processed/csds_parameters_converged_only.csv)
- split datasets:
  [data/interim/csds_tau_peak_low.csv](/Users/hariderarako/Desktop/python/data/interim/csds_tau_peak_low.csv)
  [data/interim/csds_tau_peak_low_1.csv](/Users/hariderarako/Desktop/python/data/interim/csds_tau_peak_low_1.csv)
  [data/interim/csds_tau_peak_low_2.csv](/Users/hariderarako/Desktop/python/data/interim/csds_tau_peak_low_2.csv)
  [data/interim/csds_tau_peak_high.csv](/Users/hariderarako/Desktop/python/data/interim/csds_tau_peak_high.csv)

### Shared outputs

- dataset outputs from scripts `01` to `05`, `08`, and `09`:
  [results/datasets](/Users/hariderarako/Desktop/python/results/datasets)

- shared preprocessing figures:
  [results/figures](/Users/hariderarako/Desktop/python/results/figures)
- shared EDA figures:
  [results/eda](/Users/hariderarako/Desktop/python/results/eda)
- shared descriptive statistics for `full`, `low`, `low_1`, `low_2`, and `high`:
  [results/eda/descriptive_statistics](/Users/hariderarako/Desktop/python/results/eda/descriptive_statistics)
- shared split figures:
  [results/split](/Users/hariderarako/Desktop/python/results/split)

### Indirect branch outputs

- regression search outputs:
  [results/methodologie_1/regressions](/Users/hariderarako/Desktop/python/results/methodologie_1/regressions)
- top retained models:
  [results/methodologie_1/regressions/top5](/Users/hariderarako/Desktop/python/results/methodologie_1/regressions/top5)
- recovered `e` from estimated `d`:
  [results/methodologie_1/e_from_all_retained_d_models](/Users/hariderarako/Desktop/python/results/methodologie_1/e_from_all_retained_d_models)
- recovered `b` from estimated `d`:
  [results/methodologie_1/b_from_all_retained_d_models](/Users/hariderarako/Desktop/python/results/methodologie_1/b_from_all_retained_d_models)
- curve and parameter comparison tables:
  [results/methodologie_1/compare_d_b_e_tau_u](/Users/hariderarako/Desktop/python/results/methodologie_1/compare_d_b_e_tau_u)
- indirect branch figures:
  [results/methodologie_1/figures](/Users/hariderarako/Desktop/python/results/methodologie_1/figures)

### Direct constrained branch outputs

- prepared datasets for `log(e-c)`:
  [results/methodologie_2/datasets](/Users/hariderarako/Desktop/python/results/methodologie_2/datasets)
- EDA outputs:
  [results/methodologie_2/eda](/Users/hariderarako/Desktop/python/results/methodologie_2/eda)
- regression search outputs:
  [results/methodologie_2/regressions](/Users/hariderarako/Desktop/python/results/methodologie_2/regressions)
- top retained models:
  [results/methodologie_2/regressions/top5](/Users/hariderarako/Desktop/python/results/methodologie_2/regressions/top5)
- evaluation of reconstructed `b`, `d`, `e`, and `tau(u)`:
  [results/methodologie_2/evaluations](/Users/hariderarako/Desktop/python/results/methodologie_2/evaluations)
- direct-e comparison tables for `b`, `d`, `e`, and `tau(u)`:
  [results/methodologie_2/compare_b_d_e_tau_u](/Users/hariderarako/Desktop/python/results/methodologie_2/compare_b_d_e_tau_u)
- direct branch figures:
  [results/methodologie_2/figures](/Users/hariderarako/Desktop/python/results/methodologie_2/figures)

### Method comparison outputs

- direct-vs-indirect comparison tables:
  [results/comparaison/methodologie_2_vs_methodologie_1](/Users/hariderarako/Desktop/python/results/comparaison/methodologie_2_vs_methodologie_1)

## Main Workflows

### `methodologie_1`

This branch studies the indirect route:

- estimate `d`
- derive `e` with Simon's equation
- reconstruct `b`
- compare parameter errors and reconstructed `tau(u)` curves

Main scripts:

- [src/methodologie_1/10_search_polynomial_models_d.py](/Users/hariderarako/Desktop/python/src/methodologie_1/10_search_polynomial_models_d.py)
- [src/methodologie_1/11_search_polynomial_models_log_d.py](/Users/hariderarako/Desktop/python/src/methodologie_1/11_search_polynomial_models_log_d.py)
- [src/methodologie_1/12_search_exponential_models_d.py](/Users/hariderarako/Desktop/python/src/methodologie_1/12_search_exponential_models_d.py)
- [src/methodologie_1/13_summarize_best_models.py](/Users/hariderarako/Desktop/python/src/methodologie_1/13_summarize_best_models.py)
- [src/methodologie_1/14_evaluate_e_from_retained_d_models.py](/Users/hariderarako/Desktop/python/src/methodologie_1/14_evaluate_e_from_retained_d_models.py)
- [src/methodologie_1/15_evaluate_b_from_retained_d_models.py](/Users/hariderarako/Desktop/python/src/methodologie_1/15_evaluate_b_from_retained_d_models.py)
- [src/methodologie_1/16_compare_d_b_e_and_plot_tau_u.py](/Users/hariderarako/Desktop/python/src/methodologie_1/16_compare_d_b_e_and_plot_tau_u.py)

### `methodologie_2`

This branch studies the direct constrained route:

- build the target `z = log(e-c)`
- estimate constrained `e`
- reconstruct `d` and `b`
- compare parameter errors and reconstructed `tau(u)` curves

Main scripts:

- [src/methodologie_2/11_explore_methodologie_2_targets.py](/Users/hariderarako/Desktop/python/src/methodologie_2/11_explore_methodologie_2_targets.py)
- [src/methodologie_2/12_prepare_log_e_minus_c_targets.py](/Users/hariderarako/Desktop/python/src/methodologie_2/12_prepare_log_e_minus_c_targets.py)
- [src/methodologie_2/13_search_exponential_models_log_e_gap.py](/Users/hariderarako/Desktop/python/src/methodologie_2/13_search_exponential_models_log_e_gap.py)
- [src/methodologie_2/14_search_polynomial_models_log_e_gap.py](/Users/hariderarako/Desktop/python/src/methodologie_2/14_search_polynomial_models_log_e_gap.py)
- [src/methodologie_2/15_summarize_best_e_models.py](/Users/hariderarako/Desktop/python/src/methodologie_2/15_summarize_best_e_models.py)
- [src/methodologie_2/16_evaluate_b_d_from_retained_e_models.py](/Users/hariderarako/Desktop/python/src/methodologie_2/16_evaluate_b_d_from_retained_e_models.py)
- [src/methodologie_2/17_compare_methodologie_2_with_methodologie_1.py](/Users/hariderarako/Desktop/python/src/methodologie_2/17_compare_methodologie_2_with_methodologie_1.py)
- [src/methodologie_2/18_compare_b_d_e_and_plot_tau_u.py](/Users/hariderarako/Desktop/python/src/methodologie_2/18_compare_b_d_e_and_plot_tau_u.py)

## Where to Look First

If you want the main story of the project quickly:

1. open the converged dataset:
   [data/processed/csds_parameters_converged_only.csv](/Users/hariderarako/Desktop/python/data/processed/csds_parameters_converged_only.csv)
2. look at the shared datasets, EDA, and split figures:
   [results/datasets](/Users/hariderarako/Desktop/python/results/datasets)
   [results/eda](/Users/hariderarako/Desktop/python/results/eda)
   [results/eda/descriptive_statistics](/Users/hariderarako/Desktop/python/results/eda/descriptive_statistics)
   [results/split](/Users/hariderarako/Desktop/python/results/split)
3. look at the indirect branch summaries:
   [results/methodologie_1/regressions](/Users/hariderarako/Desktop/python/results/methodologie_1/regressions)
4. look at the direct constrained branch summaries:
   [results/methodologie_2/regressions](/Users/hariderarako/Desktop/python/results/methodologie_2/regressions)
5. look at the direct-vs-indirect comparison tables:
   [results/comparaison/methodologie_2_vs_methodologie_1](/Users/hariderarako/Desktop/python/results/comparaison/methodologie_2_vs_methodologie_1)
6. look at the sample-by-sample model plots:
   [results/methodologie_1/figures/models](/Users/hariderarako/Desktop/python/results/methodologie_1/figures/models)
   [results/methodologie_2/figures/models](/Users/hariderarako/Desktop/python/results/methodologie_2/figures/models)

## Current Status

At this stage, the repository is organized as a research workflow rather than as a minimal library package. The strongest current comparison is:

- indirect method: predict `d`, then recover `e`
- direct constrained method: predict `log(e-c)`, then reconstruct `e`

The direct constrained approach is especially promising because it respects `e > c` by construction and can be compared directly against the indirect route on both parameter errors and reconstructed `tau(u)` curves.
