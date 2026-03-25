import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_fold(
    df,
    fold_col: str = "hierarchical_cluster_IgG_isotype_stratified_fold",
    fold=0,
):
    """
    Split the dataframe into train and test partitions using a predefined fold column.
    """
    if fold_col not in df.columns:
        raise ValueError(
            f"{fold_col} not found in dataframe. "
            "Make sure the fold assignment column is included."
        )

    test_mask = df[fold_col] == fold
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    return train_df, test_df