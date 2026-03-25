import numpy as np
import pandas as pd
import torch

from pathlib import Path
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from torch_geometric.data import Data, Dataset


AA_TO_IDX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
    "X": 20,
}

HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    "X": 0.0,
}

THREE_TO_ONE = {
    k.upper(): v.upper() for k, v in IUPACData.protein_letters_3to1.items()
}


def load_merged_table(sequences_csv, properties_csv, id_col, target_col, fold_col):
    sequences = pd.read_csv(sequences_csv)
    properties = pd.read_csv(properties_csv)

    df = (
        sequences[[id_col, fold_col]]
        .merge(properties[[id_col, target_col]], on=id_col, how="inner")
        .dropna(subset=[target_col, fold_col])
        .reset_index(drop=True)
    )
    return df


def get_pdb_paths_and_targets(df, pdb_dir, id_col, target_col):
    pdb_dir = Path(pdb_dir)
    pdb_paths = [(pdb_dir / f"{ab_id}.pdb").as_posix() for ab_id in df[id_col]]
    targets = df[target_col].tolist()
    return pdb_paths, targets


def load_pdb_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antibody", pdb_path)

    residues = []
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues.append(residue.get_resname())
                    coords.append(residue["CA"].get_coord())

    coords = np.array(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"[Data Shape Error] File: {pdb_path}: \n"
                         f"Expected a 2D matrix with 3 columns (x, y, z), but got shape {coords.shape}.\n"
                        f"This usually means the PDB file is corrupted or atomic data is missing."
        )

    return residues, coords


def build_graph(pdb_path, target, cutoff=5.0):
    residues, coords = load_pdb_residues(pdb_path)
    n = len(residues)

    # Node features: 21 one-hot + 1 hydrophobicity = 22 dims
    x = np.zeros((n, 22), dtype=np.float32)
    residue_one_letter = []

    for i, aa3 in enumerate(residues):
        aa1 = THREE_TO_ONE.get(aa3, "X")
        residue_one_letter.append(aa1)
        idx = AA_TO_IDX.get(aa1, 20)
        x[i, idx] = 1.0
        x[i, 21] = HYDRO.get(aa1, 0.0)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    row, col = np.meshgrid(np.arange(n), np.arange(n))
    row = row.flatten()
    col = col.flatten()

    mask = dist.flatten() <= cutoff
    edge_index = np.vstack([row[mask], col[mask]])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Edge features: normalized distance, inverse distance
    dist_flat = dist.flatten()[mask]
    dist_norm = dist_flat / cutoff
    inv_dist = 1.0 / (dist_norm + 1e-6)

    edge_attr = np.stack([dist_norm, inv_dist], axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor([[target]], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class AntibodyGraphDataset(Dataset):
    def __init__(self, pdb_paths, targets, cutoff=5.0):
        super().__init__()
        self.graphs = []

        for pdb_path, target in zip(pdb_paths, targets):
            graph = build_graph(pdb_path, target, cutoff=cutoff)
            self.graphs.append(graph)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]