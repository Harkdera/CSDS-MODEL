# CSDS Project

## Overview

This repository studies the calibration of the CSDS model and compares two alternative strategies to the classical iterative fit:

- `direct_d`: estimate `d`, then recover `e` with Simon's relation
- `direct_e`: estimate `z = log(e-c)`, then reconstruct `e = c + exp(z)` so that `e > c` is guaranteed by construction

The goal of the project is to better understand the CSDS parameters and evaluate whether a regression-based calibration strategy can compete with, or improve on, the iterative approach.

## Apercu en francais

Ce depot etudie la calibration du modele CSDS et compare deux strategies alternatives a la calibration iterative classique :

- `direct_d` : estimer `d`, puis reconstruire `e` avec la relation de Simon
- `direct_e` : estimer `z = log(e-c)`, puis reconstruire `e = c + exp(z)` afin de garantir automatiquement `e > c`

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
python src/direct_d/10_search_polynomial_models_d.py
python src/direct_d/11_search_polynomial_models_log_d.py
python src/direct_d/12_search_exponential_models_d.py
python src/direct_d/13_summarize_best_models.py
python src/direct_d/14_evaluate_e_from_retained_d_models.py
python src/direct_d/15_evaluate_b_from_retained_d_models.py
python src/direct_d/16_compare_d_b_e_and_plot_tau_u.py
```

Direct constrained branch:

```bash
python src/direct_e/11_explore_direct_e_targets.py
python src/direct_e/12_prepare_log_e_minus_c_targets.py
python src/direct_e/13_search_exponential_models_log_e_gap.py
python src/direct_e/14_search_polynomial_models_log_e_gap.py
python src/direct_e/15_summarize_best_e_models.py
python src/direct_e/16_evaluate_b_d_from_retained_e_models.py
python src/direct_e/17_compare_direct_e_with_indirect_method.py
python src/direct_e/18_compare_b_d_e_and_plot_tau_u.py
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

## Data Source and Reference Paper

The database used in this project comes from the reference work associated with Simon's CSDS formulation. Local copies of the reference material are stored in:

- [docs/NQ50263.pdf](/Users/hariderarako/Desktop/python/docs/NQ50263.pdf)
- [docs/reference_paper.pdf](/Users/hariderarako/Desktop/python/docs/reference_paper.pdf)

These documents are used as references for:

- the CSDS model formulation
- the interpretation of parameters `a`, `b`, `c`, `d`, and `e`
- the Simon relation used in the indirect branch to reconstruct `e`
- the physical constraint `e > c` used in the direct constrained branch

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
│   ├── figures/
│   │   ├── curves/
│   │   ├── eda/
│   │   └── splits/
│   ├── direct_d/
│   │   ├── figures/
│   │   │   ├── curves/
│   │   │   └── models/
│   │   ├── regressions/
│   │   │   └── top5/
│   │   ├── e_from_all_retained_d_models/
│   │   ├── b_from_all_retained_d_models/
│   │   └── compare_d_b_e_tau_u/
│   └── direct_e/
│       ├── datasets/
│       ├── eda/
│       ├── regressions/
│       │   └── top5/
│       ├── evaluations/
│       ├── comparisons/
│       │   └── direct_e_models/
│       └── figures/
│           ├── curves/
│           └── models/
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
│   ├── direct_d/
│   └── direct_e/
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

- shared preprocessing figures:
  [results/figures](/Users/hariderarako/Desktop/python/results/figures)

### Indirect branch outputs

- regression search outputs:
  [results/direct_d/regressions](/Users/hariderarako/Desktop/python/results/direct_d/regressions)
- top retained models:
  [results/direct_d/regressions/top5](/Users/hariderarako/Desktop/python/results/direct_d/regressions/top5)
- recovered `e` from estimated `d`:
  [results/direct_d/e_from_all_retained_d_models](/Users/hariderarako/Desktop/python/results/direct_d/e_from_all_retained_d_models)
- recovered `b` from estimated `d`:
  [results/direct_d/b_from_all_retained_d_models](/Users/hariderarako/Desktop/python/results/direct_d/b_from_all_retained_d_models)
- curve and parameter comparison tables:
  [results/direct_d/compare_d_b_e_tau_u](/Users/hariderarako/Desktop/python/results/direct_d/compare_d_b_e_tau_u)
- indirect branch figures:
  [results/direct_d/figures](/Users/hariderarako/Desktop/python/results/direct_d/figures)

### Direct constrained branch outputs

- prepared datasets for `log(e-c)`:
  [results/direct_e/datasets](/Users/hariderarako/Desktop/python/results/direct_e/datasets)
- EDA outputs:
  [results/direct_e/eda](/Users/hariderarako/Desktop/python/results/direct_e/eda)
- regression search outputs:
  [results/direct_e/regressions](/Users/hariderarako/Desktop/python/results/direct_e/regressions)
- top retained models:
  [results/direct_e/regressions/top5](/Users/hariderarako/Desktop/python/results/direct_e/regressions/top5)
- evaluation of reconstructed `b`, `d`, `e`, and `tau(u)`:
  [results/direct_e/evaluations](/Users/hariderarako/Desktop/python/results/direct_e/evaluations)
- direct-vs-indirect comparison tables:
  [results/direct_e/comparisons](/Users/hariderarako/Desktop/python/results/direct_e/comparisons)
- direct branch figures:
  [results/direct_e/figures](/Users/hariderarako/Desktop/python/results/direct_e/figures)

## Main Workflows

### `direct_d`

This branch studies the indirect route:

- estimate `d`
- derive `e` with Simon's equation
- reconstruct `b`
- compare parameter errors and reconstructed `tau(u)` curves

Main scripts:

- [src/direct_d/10_search_polynomial_models_d.py](/Users/hariderarako/Desktop/python/src/direct_d/10_search_polynomial_models_d.py)
- [src/direct_d/11_search_polynomial_models_log_d.py](/Users/hariderarako/Desktop/python/src/direct_d/11_search_polynomial_models_log_d.py)
- [src/direct_d/12_search_exponential_models_d.py](/Users/hariderarako/Desktop/python/src/direct_d/12_search_exponential_models_d.py)
- [src/direct_d/13_summarize_best_models.py](/Users/hariderarako/Desktop/python/src/direct_d/13_summarize_best_models.py)
- [src/direct_d/14_evaluate_e_from_retained_d_models.py](/Users/hariderarako/Desktop/python/src/direct_d/14_evaluate_e_from_retained_d_models.py)
- [src/direct_d/15_evaluate_b_from_retained_d_models.py](/Users/hariderarako/Desktop/python/src/direct_d/15_evaluate_b_from_retained_d_models.py)
- [src/direct_d/16_compare_d_b_e_and_plot_tau_u.py](/Users/hariderarako/Desktop/python/src/direct_d/16_compare_d_b_e_and_plot_tau_u.py)

### `direct_e`

This branch studies the direct constrained route:

- build the target `z = log(e-c)`
- estimate constrained `e`
- reconstruct `d` and `b`
- compare parameter errors and reconstructed `tau(u)` curves

Main scripts:

- [src/direct_e/11_explore_direct_e_targets.py](/Users/hariderarako/Desktop/python/src/direct_e/11_explore_direct_e_targets.py)
- [src/direct_e/12_prepare_log_e_minus_c_targets.py](/Users/hariderarako/Desktop/python/src/direct_e/12_prepare_log_e_minus_c_targets.py)
- [src/direct_e/13_search_exponential_models_log_e_gap.py](/Users/hariderarako/Desktop/python/src/direct_e/13_search_exponential_models_log_e_gap.py)
- [src/direct_e/14_search_polynomial_models_log_e_gap.py](/Users/hariderarako/Desktop/python/src/direct_e/14_search_polynomial_models_log_e_gap.py)
- [src/direct_e/15_summarize_best_e_models.py](/Users/hariderarako/Desktop/python/src/direct_e/15_summarize_best_e_models.py)
- [src/direct_e/16_evaluate_b_d_from_retained_e_models.py](/Users/hariderarako/Desktop/python/src/direct_e/16_evaluate_b_d_from_retained_e_models.py)
- [src/direct_e/17_compare_direct_e_with_indirect_method.py](/Users/hariderarako/Desktop/python/src/direct_e/17_compare_direct_e_with_indirect_method.py)
- [src/direct_e/18_compare_b_d_e_and_plot_tau_u.py](/Users/hariderarako/Desktop/python/src/direct_e/18_compare_b_d_e_and_plot_tau_u.py)

## Where to Look First

If you want the main story of the project quickly:

1. open the converged dataset:
   [data/processed/csds_parameters_converged_only.csv](/Users/hariderarako/Desktop/python/data/processed/csds_parameters_converged_only.csv)
2. look at the shared EDA and split figures:
   [results/figures](/Users/hariderarako/Desktop/python/results/figures)
3. look at the indirect branch summaries:
   [results/direct_d/regressions](/Users/hariderarako/Desktop/python/results/direct_d/regressions)
4. look at the direct constrained branch summaries:
   [results/direct_e/regressions](/Users/hariderarako/Desktop/python/results/direct_e/regressions)
5. look at the direct-vs-indirect comparison tables:
   [results/direct_e/comparisons](/Users/hariderarako/Desktop/python/results/direct_e/comparisons)
6. look at the sample-by-sample model plots:
   [results/direct_d/figures/models](/Users/hariderarako/Desktop/python/results/direct_d/figures/models)
   [results/direct_e/figures/models](/Users/hariderarako/Desktop/python/results/direct_e/figures/models)

## Current Status

At this stage, the repository is organized as a research workflow rather than as a minimal library package. The strongest current comparison is:

- indirect method: predict `d`, then recover `e`
- direct constrained method: predict `log(e-c)`, then reconstruct `e`

The direct constrained approach is especially promising because it respects `e > c` by construction and can be compared directly against the indirect route on both parameter errors and reconstructed `tau(u)` curves.
