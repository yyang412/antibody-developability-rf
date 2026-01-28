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
1. Cross-validation
- Uses predefined folds from hierarchical_cluster_IgG_isotype_stratified_fold
- Each fold is held out once as an external test set

2. Model
- RandomForestRegressor
- n_estimators = 600
- max_depth = 5
- min_samples_leaf = 10

2. Evaluation Metrics
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
RF_top11_avg_feature_importance.png
```

This analysis highlights stable, interpretable physicochemical drivers of hydrophobicity.

 ## Repository Structure
```
├── feature_engineering/     # Feature construction (RF_utils.py)
├── model/                   # Random Forest pipeline (RF_model.py)
├── data/                    # Input CSV files (ignored by default)
├── sandbox/                 # Exploratory notebooks and scratch work
├── requirements.txt         # Python dependencies
└── README.md
```

## Key Files
1. feature_engineering/RF_utils.py

Feature engineering utilities converting raw antibody sequences into numerical descriptors.

2. model/RF_model.py

End-to-end Random Forest pipeline:
- feature generation
- fold-based training and evaluation
- performance metrics
- feature importance analysis

## How to Run
1. Environment setup
From the project root
```
pip install -r requirements.txt
```

2. Run the Random Forest model
```
python -m model.RF_model
```

## Notes
Model evaluation prioritizes robustness and ranking consistency over absolute error.
The codebase is modular and structured to support future extensions or alternative models.
Raw data files are excluded from version control by default; paths and formats are documented for reproducibility.
