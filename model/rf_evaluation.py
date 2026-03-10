import pandas as pd
import matplotlib.pyplot as plt

from model.rf_pipeline import (
    split_by_fold,
    train_rf_model_with_fold,
    compute_spearman_with_fold,
)


def summarize_cv_results(metrics_df: pd.DataFrame, spearman_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validated regression and ranking performance across folds.
    """
    r2_summary = (
        metrics_df
        .groupby("model_name")["R2_test"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "CV_R2_mean", "std": "CV_R2_std"})
        .reset_index()
    )

    rho_summary = (
        spearman_df
        .groupby("model_name")["Spearman_rho_test"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "rho_mean", "std": "rho_std"})
        .reset_index()
    )

    summary_df = r2_summary.merge(rho_summary, on="model_name")
    return summary_df


def collect_full_model_importances(
    models_per_fold: dict,
    full_cols: list,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collect feature importances from each fold's full RF model.
    """
    full_importances_list = []

    for fold_id, models in models_per_fold.items():
        rf_full = models["full"]

        if rf_full is None:
            continue

        feature_cols = [c for c in full_cols if c in df.columns]
        importances = rf_full.feature_importances_

        fold_imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
            "fold": fold_id,
        })
        full_importances_list.append(fold_imp_df)

    if not full_importances_list:
        return pd.DataFrame(columns=["feature", "importance", "fold"])

    full_importances = pd.concat(full_importances_list, axis=0)
    return full_importances


def run_fold_specific_top_n_analysis(
    df,
    unique_folds,
    full_importances,
    top_n,
    target_col,
    fold_col,
    n_estimators,
    max_depth,
    min_samples_leaf,
    random_state,
):
    """
    Retrain fold-specific RF models using the top-N features selected from the
    corresponding full model importance scores.
    """
    all_metrics_top = []
    all_spearman_top = []
    models_per_fold_top = {}

    for fold_id in unique_folds:
        print(f"\n===== Fold {fold_id} (Top {top_n} features) =====")

        fold_imp = (
            full_importances[full_importances["fold"] == fold_id]
            .sort_values("importance", ascending=False)
        )

        fold_top_features = fold_imp.head(top_n)["feature"].tolist()

        train_df_fold, test_df_fold = split_by_fold(
            df,
            fold_col=fold_col,
            fold=fold_id,
        )

        rf_top, res_top = train_rf_model_with_fold(
            train_df_fold,
            test_df_fold,
            fold_top_features,
            target_col=target_col,
            model_name=f"Full_top{top_n}_fold_specific",
            fold_id=fold_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        sp_top = compute_spearman_with_fold(
            rf_top,
            test_df_fold,
            fold_top_features,
            target_col=target_col,
            model_name=f"Full_top{top_n}_fold_specific",
            fold_id=fold_id,
        )

        models_per_fold_top[fold_id] = {
            "model": rf_top,
            "features": fold_top_features,
        }

        all_metrics_top.append(res_top)
        all_spearman_top.append(sp_top)

    metrics_top_df = pd.DataFrame(all_metrics_top)
    spearman_top_df = pd.DataFrame(all_spearman_top)

    return metrics_top_df, spearman_top_df, models_per_fold_top


def summarize_feature_importance_ranks(full_importances: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average feature rank and average importance across folds.
    """
    full_importances_ranked = (
        full_importances
        .sort_values(["fold", "importance"], ascending=[True, False])
    )

    full_importances_ranked["rank"] = (
        full_importances_ranked
        .groupby("fold")["importance"]
        .rank(method="average", ascending=False)
    )

    avg_summary_df = (
        full_importances_ranked
        .groupby("feature")
        .agg(
            avg_rank=("rank", "mean"),
            avg_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .reset_index()
        .sort_values("avg_rank")
    )

    return avg_summary_df


def plot_top_feature_importances(
    avg_summary_df: pd.DataFrame,
    top_n: int,
    output_path,
):
    """
    Plot and save the top-N average feature importances across folds.
    """
    top_features = (
        avg_summary_df
        .sort_values("avg_importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features["avg_importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Random Forest Feature Importances (Top {top_n})")
    plt.xlabel("Mean Feature Importance (across folds)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()