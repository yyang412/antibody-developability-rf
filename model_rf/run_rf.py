import pandas as pd
import yaml
from pathlib import Path

from features.rf_features import create_features_from_raw_df
from model_rf.rf_feature_sets import get_feature_sets
from model_rf.rf_pipeline import (
    train_rf_model_with_fold,
    compute_spearman_with_fold,
)
from model_rf.rf_evaluation import (
    summarize_cv_results,
    collect_full_model_importances,
    run_fold_specific_top_n_analysis,
    summarize_feature_importance_ranks,
    plot_top_feature_importances,
)
from utils.utils import split_by_fold


def main():
    """
    Run the full Random Forest experiment pipeline:
    1. Load config and data
    2. Build features
    3. Train RF models across predefined folds
    4. Summarize CV results
    5. Run fold-specific top-N feature analysis
    6. Save metrics, summaries, and plots
    """
    # ============================================================
    # 0. Configuration
    # ============================================================
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "rf_baseline.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sequences_csv = project_root / config["data"]["sequences_csv"]
    properties_csv = project_root / config["data"]["properties_csv"]

    target_col = config["data"]["target_col"]
    fold_col = config["data"]["fold_col"]
    id_col = config["data"]["id_col"]

    n_estimators = config["model"]["n_estimators"]
    max_depth = config["model"]["max_depth"]
    min_samples_leaf = config["model"]["min_samples_leaf"]
    random_state = config["model"]["random_state"]

    top_n = config["analysis"]["top_n"]
    save_per_fold_analysis = config["analysis"].get("save_per_fold_analysis", False)

    results_dir = project_root / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # I. Data Preparation
    # ============================================================
    sequences = pd.read_csv(sequences_csv)
    properties = pd.read_csv(properties_csv)

    sequence_features = create_features_from_raw_df(sequences)

    df = (
        sequence_features
        .merge(sequences[[id_col, fold_col]], on=id_col, how="left")
        .merge(properties[[id_col, target_col]], on=id_col, how="left")
    )

    df = df.dropna(subset=[target_col, fold_col]).reset_index(drop=True)

    # ============================================================
    # II. Feature Sets
    # ============================================================
    feature_sets = get_feature_sets(df)

    level1_cols = feature_sets["level1"]
    level2_cols = feature_sets["level2"]
    level3_cols = feature_sets["level3"]
    full_cols = feature_sets["full"]

    # ============================================================
    # III. Cross-Fold Training & Evaluation
    # ============================================================
    unique_folds = sorted(df[fold_col].dropna().unique())

    all_metrics = []
    all_spearman = []
    models_per_fold = {}

    for fold_id in unique_folds:
        print(f"\n===== Fold {fold_id} =====")

        train_df_fold, test_df_fold = split_by_fold(
            df,
            fold_col=fold_col,
            fold=fold_id,
        )

        # Level 1: Fv-only features
        rf_lvl1, res_lvl1 = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            level1_cols,
            target_col=target_col,
            model_name="Level 1 (Fv only)",
            fold_id=fold_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        sp_lvl1 = compute_spearman_with_fold(
            rf_lvl1,
            test_df_fold,
            level1_cols,
            target_col=target_col,
            model_name="Level 1 (Fv only)",
            fold_id=fold_id,
        )

        # Level 2: chain-level features
        rf_lvl2, res_lvl2 = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            level2_cols,
            target_col=target_col,
            model_name="Level 2 (Chain-level)",
            fold_id=fold_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        sp_lvl2 = compute_spearman_with_fold(
            rf_lvl2,
            test_df_fold,
            level2_cols,
            target_col=target_col,
            model_name="Level 2 (Chain-level)",
            fold_id=fold_id,
        )

        # Level 3: CDR-level features
        rf_lvl3, res_lvl3 = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            level3_cols,
            target_col=target_col,
            model_name="Level 3 (CDR-level)",
            fold_id=fold_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        sp_lvl3 = compute_spearman_with_fold(
            rf_lvl3,
            test_df_fold,
            level3_cols,
            target_col=target_col,
            model_name="Level 3 (CDR-level)",
            fold_id=fold_id,
        )

        # Full model: Level 1 + 2 + 3
        rf_full, res_full = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            full_cols,
            target_col=target_col,
            model_name="Full (Level1+2+3)",
            fold_id=fold_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        sp_full = compute_spearman_with_fold(
            rf_full,
            test_df_fold,
            full_cols,
            target_col=target_col,
            model_name="Full (Level1+2+3)",
            fold_id=fold_id,
        )

        models_per_fold[fold_id] = {
            "lvl1": rf_lvl1,
            "lvl2": rf_lvl2,
            "lvl3": rf_lvl3,
            "full": rf_full,
        }

        all_metrics.extend([res_lvl1, res_lvl2, res_lvl3, res_full])
        all_spearman.extend([sp_lvl1, sp_lvl2, sp_lvl3, sp_full])

    # ============================================================
    # IV. Aggregate Results
    # ============================================================
    metrics_df = pd.DataFrame(all_metrics)
    spearman_df = pd.DataFrame(all_spearman)

    summary_df = summarize_cv_results(metrics_df, spearman_df)

    print("\n=== Per-fold RF Metrics ===")
    print(metrics_df)

    print("\n=== Per-fold Spearman Correlations ===")
    print(spearman_df)

    print("\n=== Cross-validated Summary Across Folds ===")
    print(summary_df)

    # ============================================================
    # V. Feature Importance Analysis
    # ============================================================
    full_importances = collect_full_model_importances(
        models_per_fold=models_per_fold,
        full_cols=full_cols,
        df=df,
    )

    metrics_top_df, spearman_top_df, models_per_fold_top = run_fold_specific_top_n_analysis(
        df=df,
        unique_folds=unique_folds,
        full_importances=full_importances,
        top_n=top_n,
        target_col=target_col,
        fold_col=fold_col,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    top_summary_df = summarize_cv_results(metrics_top_df, spearman_top_df)
    avg_summary_df = summarize_feature_importance_ranks(full_importances)

    print(f"\n=== Cross-validated Summary (Top {top_n}, fold-specific) ===")
    print(top_summary_df)

    plot_top_feature_importances(
        avg_summary_df=avg_summary_df,
        top_n=top_n,
        output_path=results_dir / f"rf_top{top_n}_avg_feature_importance.png",
    )

    # ============================================================
    # VI. Save Final Report-Ready Artifacts
    # ============================================================
    
    summary_df.to_csv(results_dir / "rf_cv_summary.csv", index=False)
    top_summary_df.to_csv(results_dir / f"rf_top{top_n}_cv_summary.csv", index=False)
    avg_summary_df.to_csv(results_dir / "rf_feature_importance_rank_summary.csv", index=False)
    
    # Optional: save fold-level / debug analysis artifacts
    if save_per_fold_analysis:
        full_importances.to_csv(results_dir / "rf_full_model_importances.csv", index=False)
        metrics_df.to_csv(results_dir / "rf_metrics_per_fold.csv", index=False)
        spearman_df.to_csv(results_dir / "rf_spearman_per_fold.csv", index=False)
        metrics_top_df.to_csv(results_dir / f"rf_top{top_n}_metrics_per_fold.csv", index=False)
        spearman_top_df.to_csv(results_dir / f"rf_top{top_n}_spearman_per_fold.csv", index=False)


if __name__ == "__main__":
    main()