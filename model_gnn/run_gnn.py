import pandas as pd
import torch
import yaml

from pathlib import Path
from torch_geometric.loader import DataLoader

from utils.utils import split_by_fold, set_seed

from model_gnn.gnn_dataset import (
    load_merged_table,
    get_pdb_paths_and_targets,
    AntibodyGraphDataset,
)
from model_gnn.gnn_model import AntibodyGNN
from model_gnn.gnn_pipeline import train_one_epoch, evaluate


def main():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "gnn_baseline.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sequences_csv = project_root / config["data"]["sequences_csv"]
    properties_csv = project_root / config["data"]["properties_csv"]
    pdb_dir = project_root / config["data"]["pdb_dir"]

    target_col = config["data"]["target_col"]
    fold_col = config["data"]["fold_col"]
    id_col = config["data"]["id_col"]

    cutoff = config["graph"]["cutoff"]

    hidden_dim = config["model"]["hidden_dim"]
    num_conv_layers = config["model"]["num_conv_layers"]
    dropout = config["model"]["dropout"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]

    random_state = config["training"]["random_state"]
    set_seed(random_state)
    
    results_dir = project_root / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    

    df = load_merged_table(
        sequences_csv=sequences_csv,
        properties_csv=properties_csv,
        id_col=id_col,
        target_col=target_col,
        fold_col=fold_col,
    )

    unique_folds = sorted(df[fold_col].dropna().unique())
    fold_results = []

    for fold_id in unique_folds:
        print(f"\n===== Fold {fold_id} =====")

        train_df, test_df = split_by_fold(df, fold_col=fold_col, fold=fold_id)

        train_pdbs, train_targets = get_pdb_paths_and_targets(
            train_df, pdb_dir, id_col, target_col
        )
        test_pdbs, test_targets = get_pdb_paths_and_targets(
            test_df, pdb_dir, id_col, target_col
        )

        train_dataset = AntibodyGraphDataset(train_pdbs, train_targets, cutoff=cutoff)
        test_dataset = AntibodyGraphDataset(test_pdbs, test_targets, cutoff=cutoff)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = AntibodyGNN(
            input_dim=22,
            edge_dim=2,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_test_rho = -999.0

        for epoch in range(1, epochs + 1):
            train_loss, train_rho = train_one_epoch(model, train_loader, optimizer, device)
            test_loss, test_rho = evaluate(model, test_loader, device)

            print(
                f"Fold {fold_id} | Epoch {epoch} | "
                f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f} | "
                f"train_rho={train_rho:.3f}, test_rho={test_rho:.3f}"
            )

            if test_rho > best_test_rho:
                best_test_rho = test_rho

        fold_results.append({
            "fold": fold_id,
            "best_test_spearman": best_test_rho,
        })

    results_df = pd.DataFrame(fold_results)
    summary_df = pd.DataFrame([{
        "mean_best_test_spearman": results_df["best_test_spearman"].mean(),
        "std_best_test_spearman": results_df["best_test_spearman"].std(),
    }])

    print("\n=== Fold Results ===")
    print(results_df)

    print("\n=== Cross-Validation Summary ===")
    print(summary_df)

    results_df.to_csv(results_dir / "gnn_fold_results.csv", index=False)
    summary_df.to_csv(results_dir / "gnn_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()