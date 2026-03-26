# Antibody Developability Prediction: RF vs GNN
This repository contains two complementary modeling approaches for predicting antibody hydrophobicity (HIC):

- Random Forest (RF) based on engineered physicochemical features
- Graph Neural Network (GNN) based on 3D structural graphs derived from antibody models

The project investigates whether engineered physicochemical features derived from antibody sequences can capture developability-related behavior, and further asks whether structure-aware graph representations provide a stronger signal than sequence-derived features, with a focus on ranking performance (Spearman correlation).




## Project Overview

This work originates from an antibody developability competition dataset provided by Ginkgo Bioworks.

Within a broader team project that explored multiple modeling approaches, this repository focuses on two approaches that I designed, implemented, and evaluated: Random Forest (RF) and Graph Neural Networks (GNN).

Other approaches explored by the team (not included here) included:
- CNNs on raw amino-acid sequences
- LSTM-based sequence models

By consolidating both RF and GNN within a single codebase, this repository enables a direct comparison between feature-based and structure-aware modeling approaches:

| Approach | Input | Inductive Bias |
|----------|------|---------------|
| RF | Sequence-derived features | Human-designed physicochemical descriptors |
| GNN | 3D structure graphs | Learned spatial + biochemical interactions |


## Random Forest (Feature-Based Approach)
### Modeling Approach
The Random Forest model predicts HIC values using features derived from:
- Global Fv sequence (VH + VL combined)
- Individual VH and VL chains
- CDR regions (HCDR3 and LCDR1, extracted using AHo numbering)

Feature design is strongly focused on hydrophobicity-related physicochemical properties, including GRAVY, charge, aromaticity, amino-acid composition, and local hydrophobic clustering.


### Feature Structure
For the Random Forest model, features are organized hierarchically to support interpretability and systematic analysis:
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


### Training and Evaluation Strategy
1. Cross-validation
- Uses predefined folds from hierarchical_cluster_IgG_isotype_stratified_fold
- Each fold is held out once as an external test set

2. Model

RandomForestRegressor with:

- `n_estimators = 600`
- `max_depth = 5`
- `min_samples_leaf = 10`

3. Evaluation Metrics

- R² (train / test)
- RMSE (train / test)
- Spearman rank correlation on test folds

Rank-based evaluation is emphasized, reflecting the practical importance of relative developability ordering.


### Feature Importance Analysis
- Feature importances are extracted from full models (Levels 1 + 2 + 3)
- Importances are computed per fold
- Average rank and average importance are aggregated across folds
- The top 11 most consistently important features are visualized and saved as:

```
rf_top11_avg_feature_importance.png
```

This analysis highlights stable, interpretable physicochemical drivers of hydrophobicity.

## Graph Neural Network (Structure-Based Approach)
### Modeling Approach
To complement the feature-based Random Forest model, a Graph Neural Network (GNN) is introduced to model residue-level structural interactions directly from 3D antibody structures. Unlike the Random Forest model, the GNN learns representations directly from structural graphs without relying on manually engineered features.

### Graph Construction
- Nodes: residues (Cα atoms)
- Edges: distance-based connections (cutoff = 5Å)

### Node Features
- 21-dimensional amino acid one-hot encoding
- 1-dimensional hydrophobicity (Kyte-Doolittle scale)

### Edge Features
- Normalized distance
- Inverse distance

### Model Architecture
- GENConv-based message passing layers
- 3 graph convolution layers
- global mean pooling
- MLP regression head

### Training and Evaluation Strategy

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 50
- Batch size: 16

Evaluation metric:
- Spearman rank correlation (primary)


## Model Comparison

| Model | Input | Mean Spearman |
|------|------|--------------|
| RF | Sequence features | 0.401 |
| GNN | Structure graphs | 0.576 |




 ## Repository Structure
```
antibody-developability-rf/
│
├── data/
│   ├── GDPa1_v1.2_sequences.csv
│   └── pdb_files/                 # generated structures for GNN
│
├── features/
│   └── rf_features.py
│
├── model_gnn/
│   ├── generate_pdbs.py          # IgFold structure generation
│   ├── gnn_dataset.py            # graph construction
│   ├── gnn_model.py              # GNN architecture
│   ├── gnn_pipeline.py           # training + evaluation
│   └── run_gnn.py                # main entry point
│
├── model_rf/
│   ├── rf_feature_sets.py        # hierarchical feature grouping (Level 1/2/3)
│   ├── rf_pipeline.py            # training loop, cross-validation, and data handling
│   ├── rf_evaluation.py          # evaluation metrics and feature importance analysis
│   └── run_rf.py                 # main entry point for RF experiments
│
├── results/
│   ├── gnn/
│   │   ├── gnn_cv_summary.csv
│   │   └── gnn_fold_results.csv
│   └── rf/
│       ├── rf_cv_summary.csv
│       └── feature importance outputs
│
├── utils/
│   └── utils.py
│
├── configs/
│   ├── rf_baseline.yaml
│   └── gnn_baseline.yaml
│
└── README.md
```

# Configuration

Experiment settings are controlled through separate configuration files:

- **RF**: `configs/rf_baseline.yaml`
- **GNN**: `configs/gnn_baseline.yaml`



# Output Files
The pipeline saves experiment outputs to:

```
results/
```


## RF Outputs
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

## GNN Outputs
| File                         | Description                              |
|------------------------------|------------------------------------------|
| `gnn_cv_summary.csv`         | Cross-validated performance summary      |
| `gnn_fold_results.csv`       | Per-fold best Spearman correlation       |


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
 - 1. Load configuration from configs/rf_baseline.yaml
 - 2. Build sequence-derived features
 - 3. Train Random Forest models across predefined folds
 - 4. Evaluate performance using R² and Spearman correlation
 - 5. Perform feature importance analysis
 - 6. Retrain models using fold-specific top-N features


3. Generate structures (required for GNN)
```
-m model_gnn.generate_pdbs
```

4. Run GNN
```
-m model_gnn.run_gnn
```

The GNN pipeline performs:
 - 1. Load configuration from configs/gnn_baseline.yaml
 - 2. Construct residue-level graphs from PDB structures
 - 3. Train GNN models across folds  
 - 4. Evaluate using Spearman correlation  


## Notes
Model evaluation prioritizes robustness and ranking consistency over absolute error.

The Random Forest model emphasizes interpretability through engineered features, while the GNN captures structural interactions directly from 3D representations.

The codebase is modular and structured to support future extensions or alternative models.

Raw data files are excluded from version control by default; paths and formats are documented for reproducibility.
