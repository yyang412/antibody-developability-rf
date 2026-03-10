# Antibody Developability Prediction with Random Forest
This repository contains my individual implementation of a Random Forest–based model for predicting antibody hydrophobicity, measured by Hydrophobic Interaction Chromatography (HIC), from antibody sequence data.

The project investigates whether engineered physicochemical features derived from antibody sequences can capture developability-related behavior, with an emphasis on relative ranking performance rather than precise prediction of absolute HIC values.


## Project Overview
This work originates from an antibody developability competition dataset provided by Ginkgo Bioworks.
Within a broader team project that explored multiple modeling approaches, this repository focuses exclusively on the Random Forest approach that I designed, implemented, and evaluated.

Other approaches explored by the team (not included here) included:
- CNNs on raw amino-acid sequences
- LSTM-based sequence models
- Graph Neural Networks using residue-level graphs


## Modeling Approach: Random Forest Regression
The Random Forest model predicts HIC values using features derived from:
- Global Fv sequence (VH + VL combined)
- Individual VH and VL chains
- CDR regions (HCDR3 and LCDR1, extracted using AHo numbering)

Feature design is strongly focused on hydrophobicity-related physicochemical properties, including GRAVY, charge, aromaticity, amino-acid composition, and local hydrophobic clustering.

The primary modeling goal is to rank antibodies by relative hydrophobicity, rather than to minimize absolute prediction error.


## Feature Structure
Features are organized hierarchically to support interpretability and systematic analysis:
1. Level 1 — Global Fv features
- Length, GRAVY, hydrophobic residue counts
- pI, net charge, amino-acid class fractions

2. Level 2 — Chain-level features (VH / VL)
- Physicochemical descriptors (GRAVY, pI, charge, instability, secondary structure)
- Amino-acid composition and derived counts
- Cross-chain hydrophobicity relationships

3. Level 3 — CDR-level features
- HCDR3 and LCDR1 length and hydrophobicity
- Aromatic clustering
- Hydrophobic cluster size, density, and hydrophobic moment

This structured feature representation was introduced after exploratory analysis showed that individual descriptors have weak linear correlation with HIC, motivating the use of non-linear models and hierarchical feature organization.


## Training and Evaluation Strategy
### 1. Cross-validation
- Uses predefined folds from hierarchical_cluster_IgG_isotype_stratified_fold
- Each fold is held out once as an external test set

### 2. Model

RandomForestRegressor with:

- `n_estimators = 600`
- `max_depth = 5`
- `min_samples_leaf = 10`

### 3. Evaluation Metrics

- R² (train / test)
- RMSE (train / test)
- Spearman rank correlation on test folds

Rank-based evaluation is emphasized, reflecting the practical importance of relative developability ordering.


## Feature Importance Analysis
- Feature importances are extracted from full models (Levels 1 + 2 + 3)
- Importances are computed per fold
- Average rank and average importance are aggregated across folds
- The top 11 most consistently important features are visualized and saved as:

```
rf_top11_avg_feature_importance.png
```

This analysis highlights stable, interpretable physicochemical drivers of hydrophobicity.

 ## Repository Structure
```
antibody-developability-rf/
│
├── configs/
│   └── rf_baseline.yaml           # experiment configuration
│
├── features/
│   └── rf_features.py             # sequence feature engineering
│
├── model/
│   ├── rf_feature_sets.py         # Level1 / Level2 / Level3 feature groups
│   ├── rf_pipeline.py             # RF training + fold split utilities
│   ├── rf_evaluation.py           # evaluation, feature importance analysis
│   └── run_rf.py                  # main experiment runner
│
├── results/                       # saved experiment outputs
│
└── README.md
```

# Configuration 
Experiment settings are controlled through:
```
configs/rf_baseline.yaml
```
Example configuration:

```yaml
data:
  sequences_csv: data/GDPa1_v1.2_sequences.csv
  properties_csv: data/GDPa1_v1.2_20250814.csv
  target_col: HIC
  fold_col: hierarchical_cluster_IgG_isotype_stratified_fold
  id_col: antibody_id

model:
  n_estimators: 600
  max_depth: 5
  min_samples_leaf: 10
  random_state: 42

analysis:
  top_n: 11
  save_per_fold_analysis: false

output:
  results_dir: results
```


# Output Files
The pipeline saves experiment outputs to:

```
results/
```


Primary results used for reporting:

| File                                     | Description                                     |
| ---------------------------------------- | ----------------------------------------------- |
| `rf_cv_summary.csv`                      | Cross-validated performance of baseline models  |
| `rf_top11_cv_summary.csv`                | Performance of the top-N feature models         |
| `rf_feature_importance_rank_summary.csv` | Average feature importance ranking across folds |
| `rf_top11_avg_feature_importance.png`    | Visualization of the most important features    |

Optional debugging outputs (saved only if save_per_fold_analysis=true):

- rf_full_model_importances.csv
- rf_top11_metrics_per_fold.csv
- rf_top11_spearman_per_fold.csv



## How to Run
1. Environment setup
From the project root:
```
pip install -r requirements.txt
```

2. Run the Random Forest model
```
python -m model.run_rf
```

The pipeline performs the following steps:
  1. Load configuration from configs/rf_baseline.yaml
  2. Build sequence-derived features
  3. Train Random Forest models across predefined folds
  4. Evaluate performance using R² and Spearman correlation
  5. Perform feature importance analysis
  6. Retrain models using fold-specific top-N features

## Notes
Model evaluation prioritizes robustness and ranking consistency over absolute error.
The codebase is modular and structured to support future extensions or alternative models.
Raw data files are excluded from version control by default; paths and formats are documented for reproducibility.
