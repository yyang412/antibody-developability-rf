import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from model_rf.rf_feature_sets import existing_cols

def train_rf_model_with_fold(
    train_df,
    test_df,
    feature_cols,
    target_col: str = "HIC",
    model_name: str = "",
    fold_id=None,
    n_estimators: int = 600,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    random_state: int = 42,
):
    """
    Train a Random Forest model on a predefined train/test split and return
    both the fitted model and fold-level regression metrics.
    """
    feature_cols = existing_cols(feature_cols, train_df)

    if len(feature_cols) == 0:
        return None, {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "R2_train": np.nan,
            "R2_test": np.nan,
            "RMSE_train": np.nan,
            "RMSE_test": np.nan,
        }

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    train_data = pd.concat([X_train, y_train], axis=1).dropna()
    test_data = pd.concat([X_test, y_test], axis=1).dropna()

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    if len(y_train) == 0 or len(y_test) == 0:
        return None, {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": len(feature_cols),
            "R2_train": np.nan,
            "R2_test": np.nan,
            "RMSE_train": np.nan,
            "RMSE_test": np.nan,
        }

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results = {
        "model_name": model_name,
        "fold": fold_id,
        "n_samples_train": len(y_train),
        "n_samples_test": len(y_test),
        "n_features": len(feature_cols),
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test,
    }

    return rf, results


def compute_spearman_with_fold(
    rf,
    test_df,
    feature_cols,
    target_col: str = "HIC",
    model_name: str = "",
    fold_id=None,
):
    """
    Compute test-set Spearman rank correlation for a fitted fold-specific RF model.
    """
    if rf is None:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    feature_cols = existing_cols(feature_cols, test_df)

    if len(feature_cols) == 0:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": 0,
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    data = pd.concat([X_test, y_test], axis=1).dropna()
    X_test = data[feature_cols]
    y_test = data[target_col]

    if len(y_test) == 0:
        return {
            "model_name": model_name,
            "fold": fold_id,
            "n_features": len(feature_cols),
            "Spearman_rho_test": np.nan,
            "p_value": np.nan,
        }

    y_pred_test = rf.predict(X_test)
    rho, pval = spearmanr(y_test, y_pred_test)

    return {
        "model_name": model_name,
        "fold": fold_id,
        "n_features": len(feature_cols),
        "Spearman_rho_test": rho,
        "p_value": pval,
    }