import math
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


AA20 = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBIC_AA = ["A", "V", "I", "L", "M", "F", "W", "Y"]
AROMATIC_AA = ["F", "Y", "W"]

POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
POLAR_AA = set("STNQ")
SPECIAL_AA = set("PGC")

HYDROPHOBIC_SET = set(HYDROPHOBIC_AA)

HYDRO_VALUES = {
    "A": 1.8, "I": 4.5, "L": 3.8, "M": 1.9, "V": 4.2,
    "F": 2.8, "W": -0.9, "Y": -1.3, "C": 2.5,
    "H": -3.2, "K": -3.9, "R": -4.5,
    "D": -3.5, "E": -3.5,
    "N": -3.5, "Q": -3.5,
    "G": -0.4, "P": -1.6, "S": -0.8, "T": -0.7,
}

HCDR3_RANGE_AHO = (108, 138)
LCDR1_RANGE_AHO = (24, 42)


def aa20_counts(seq_series: pd.Series, suffix: str) -> pd.DataFrame:
    """
    Return amino-acid counts and sequence length for each sequence in a Series.

    Output columns:
    - {AA}_{suffix} for each amino acid in AA20
    - {suffix}_length
    """
    s = seq_series.fillna("").astype(str).str.upper()
    out = pd.DataFrame(index=s.index)

    for aa in AA20:
        out[f"{aa}_{suffix}"] = s.str.count(aa).astype(float)

    out[f"{suffix}_length"] = s.str.len().astype(float)
    return out


def extract_cdr_from_aho(aligned_seq: str, start_pos: int, end_pos: int) -> str:
    """
    Extract a contiguous CDR sequence from an AHo-aligned sequence.

    Parameters
    ----------
    aligned_seq : str
        AHo-aligned amino-acid sequence.
    start_pos : int
        Inclusive AHo start position.
    end_pos : int
        Inclusive AHo end position.
    """
    if not isinstance(aligned_seq, str):
        return ""

    start_idx = start_pos - 1
    end_idx = end_pos
    window = aligned_seq[start_idx:end_idx]
    return "".join(ch for ch in window if ch.isalpha())


def cdr_basic_features(seq: str, prefix: str) -> dict:
    """
    Compute basic CDR features for a given sequence.
    """
    feats = {}

    if not seq:
        feats[f"{prefix}_length"] = 0
        feats[f"{prefix}_gravy"] = 0.0
        feats[f"{prefix}_hydrophobic_count"] = 0
        feats[f"{prefix}_aromaticity"] = 0.0
        return feats

    pa = ProteinAnalysis(seq)
    feats[f"{prefix}_length"] = len(seq)
    feats[f"{prefix}_gravy"] = pa.gravy()
    feats[f"{prefix}_hydrophobic_count"] = sum(seq.count(a) for a in HYDROPHOBIC_AA)

    aromatic_count = sum(seq.count(a) for a in AROMATIC_AA)
    feats[f"{prefix}_aromaticity"] = aromatic_count / len(seq)

    return feats


def has_aromatic_cluster(seq: str, min_run: int = 2) -> int:
    """
    Return 1 if the sequence contains at least min_run consecutive aromatic residues.
    Otherwise return 0.
    """
    if not seq:
        return 0

    run = 0
    for aa in seq:
        if aa in AROMATIC_AA:
            run += 1
            if run >= min_run:
                return 1
        else:
            run = 0
    return 0


def aa_class_fractions(seq: str) -> dict:
    """
    Compute amino-acid class fractions for a sequence.
    """
    length = len(seq)
    if length == 0:
        return {
            "frac_positive": 0.0,
            "frac_negative": 0.0,
            "frac_polar": 0.0,
            "frac_special": 0.0,
        }

    return {
        "frac_positive": sum(seq.count(a) for a in POSITIVE_AA) / length,
        "frac_negative": sum(seq.count(a) for a in NEGATIVE_AA) / length,
        "frac_polar": sum(seq.count(a) for a in POLAR_AA) / length,
        "frac_special": sum(seq.count(a) for a in SPECIAL_AA) / length,
    }


def _chain_features(seq: str, ph: float = 7.0) -> dict:
    """
    Compute VH/VL chain-level physicochemical features.
    """
    if not seq:
        return {
            "gravy": 0.0,
            "hydrophobic_count": 0,
            "aromaticity": 0.0,
            "instability": 0.0,
            "pI": 0.0,
            "charge_pH7": 0.0,
            "helix": 0.0,
            "turn": 0.0,
            "sheet": 0.0,
            "molecular_weight": 0.0,
            "charge_pH7_35": 0.0,
            "charge_pH7_45": 0.0,
            "molar_extinction_reduced": 0.0,
            "molar_extinction_oxidized": 0.0,
            "frac_positive": 0.0,
            "frac_negative": 0.0,
            "frac_polar": 0.0,
            "frac_special": 0.0,
        }

    pa = ProteinAnalysis(seq)
    hydrophobic_count = sum(seq.count(a) for a in HYDROPHOBIC_AA)
    aromatic_count = sum(seq.count(a) for a in AROMATIC_AA)

    helix, turn, sheet = pa.secondary_structure_fraction()
    ext_red, ext_ox = pa.molar_extinction_coefficient()

    feats = {
        "gravy": pa.gravy(),
        "hydrophobic_count": hydrophobic_count,
        "aromaticity": aromatic_count / len(seq),
        "instability": pa.instability_index(),
        "pI": pa.isoelectric_point(),
        "charge_pH7": pa.charge_at_pH(ph),
        "helix": helix,
        "turn": turn,
        "sheet": sheet,
        "molecular_weight": pa.molecular_weight(),
        "charge_pH7_35": pa.charge_at_pH(7.35),
        "charge_pH7_45": pa.charge_at_pH(7.45),
        "molar_extinction_reduced": ext_red,
        "molar_extinction_oxidized": ext_ox,
    }
    feats.update(aa_class_fractions(seq))
    return feats


def max_hydrophobic_cluster_len(seq: str) -> int:
    """Return the maximum length of consecutive hydrophobic residues."""
    max_len, cur = 0, 0
    for aa in seq:
        if aa in HYDROPHOBIC_SET:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len


def count_hydrophobic_clusters(seq: str) -> int:
    """Return the number of hydrophobic residue clusters."""
    count, cur = 0, 0
    for aa in seq:
        if aa in HYDROPHOBIC_SET:
            if cur == 0:
                count += 1
            cur += 1
        else:
            cur = 0
    return count


def helix_hydrophobic_moment(seq: str) -> float:
    """Compute hydrophobic moment assuming an alpha-helix geometry."""
    moment_x, moment_y = 0.0, 0.0
    theta = math.radians(100)

    for i, aa in enumerate(seq):
        h = HYDRO_VALUES.get(aa, 0.0)
        angle = theta * i
        moment_x += h * math.cos(angle)
        moment_y += h * math.sin(angle)

    return math.sqrt(moment_x**2 + moment_y**2)


def terminal_hydrophobicity(seq: str, k: int = 3) -> float:
    """Compute mean hydrophobicity over the N- and C-terminal segments."""
    if not seq or len(seq) < 2:
        return 0.0

    segment = seq[:k] + seq[-k:]
    vals = [HYDRO_VALUES.get(aa, 0.0) for aa in segment]
    return sum(vals) / len(vals)


def create_features_from_raw_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RF-ready antibody developability features from raw sequence data.

    Expected input columns include:
    - antibody_id
    - vh_protein_sequence
    - vl_protein_sequence
    - hc_subtype
    - lc_subtype

    Optional columns:
    - heavy_aligned_aho
    - light_aligned_aho

    Output feature groups:
    - Level 1: global Fv features
    - Level 2: VH/VL chain-level and composition features
    - Level 3: CDR-level features derived from AHo-aligned sequences
    """
    X = pd.DataFrame(index=input_df.index)
    X["antibody_id"] = input_df["antibody_id"]

    # Level 1: global Fv features
    vh_seqs = input_df["vh_protein_sequence"].fillna("").astype(str)
    vl_seqs = input_df["vl_protein_sequence"].fillna("").astype(str)
    fv_seqs = vh_seqs + vl_seqs

    X["fv_length"] = fv_seqs.str.len()
    X["fv_gravy"] = fv_seqs.map(lambda s: ProteinAnalysis(s).gravy() if s else 0.0)
    X["fv_hydrophobic_count"] = fv_seqs.map(
        lambda s: sum(s.count(aa) for aa in HYDROPHOBIC_AA)
    )
    X["fv_pI"] = fv_seqs.map(
        lambda s: ProteinAnalysis(s).isoelectric_point() if s else 0.0
    )
    X["fv_charge_pH7"] = fv_seqs.map(
        lambda s: ProteinAnalysis(s).charge_at_pH(7.0) if s else 0.0
    )

    fv_aa_fraction_dicts = fv_seqs.map(aa_class_fractions)
    fv_frac_df = pd.DataFrame(list(fv_aa_fraction_dicts), index=input_df.index)
    for col in fv_frac_df.columns:
        X[f"fv_{col}"] = fv_frac_df[col]

    # Level 2: VH/VL chain-level features
    for chain in ["vh", "vl"]:
        seq_col = f"{chain}_protein_sequence"
        seqs = input_df[seq_col].fillna("").astype(str)

        X[f"{chain}_length"] = seqs.str.len()

        chain_feature_dicts = seqs.map(lambda s: _chain_features(s, ph=7.0))
        chain_df = pd.DataFrame(list(chain_feature_dicts), index=input_df.index)

        X[f"{chain}_gravy"] = chain_df["gravy"]
        X[f"{chain}_hydrophobic_count"] = chain_df["hydrophobic_count"]
        X[f"{chain}_aromaticity"] = chain_df["aromaticity"]
        X[f"{chain}_instability"] = chain_df["instability"]
        X[f"{chain}_pI"] = chain_df["pI"]
        X[f"{chain}_charge_pH7"] = chain_df["charge_pH7"]

        X[f"{chain}_helix"] = chain_df["helix"]
        X[f"{chain}_turn"] = chain_df["turn"]
        X[f"{chain}_sheet"] = chain_df["sheet"]
        X[f"{chain}_molecular_weight"] = chain_df["molecular_weight"]
        X[f"{chain}_ph_7_35_charge"] = chain_df["charge_pH7_35"]
        X[f"{chain}_ph_7_45_charge"] = chain_df["charge_pH7_45"]
        X[f"{chain}_molar_extinction_reduced"] = chain_df["molar_extinction_reduced"]
        X[f"{chain}_molar_extinction_oxidized"] = chain_df["molar_extinction_oxidized"]

        X[f"{chain}_frac_positive"] = chain_df["frac_positive"]
        X[f"{chain}_frac_negative"] = chain_df["frac_negative"]
        X[f"{chain}_frac_polar"] = chain_df["frac_polar"]
        X[f"{chain}_frac_special"] = chain_df["frac_special"]

    X["vh_vl_hydrophobicity_gap"] = X["vh_gravy"] - X["vl_gravy"]
    X["vh_vl_hydrophobicity_ratio"] = X["vh_hydrophobic_count"] / (
        X["vl_hydrophobic_count"] + 1e-6
    )

    aa_vh = aa20_counts(input_df["vh_protein_sequence"], "vh_protein_sequence")
    aa_vl = aa20_counts(input_df["vl_protein_sequence"], "vl_protein_sequence")
    X = pd.concat([X, aa_vh, aa_vl], axis=1)

    for chain in ["vh", "vl"]:
        base_col = f"{chain}_protein_sequence"

        aromatic_cols = [
            f"{aa}_{base_col}" for aa in ["F", "Y", "W"]
            if f"{aa}_{base_col}" in X.columns
        ]
        X[f"{chain}_aromatic_count"] = X[aromatic_cols].sum(axis=1) if aromatic_cols else 0

        aliphatic_cols = [
            f"{aa}_{base_col}" for aa in ["A", "V", "I", "L"]
            if f"{aa}_{base_col}" in X.columns
        ]
        X[f"{chain}_aliphatic_count"] = X[aliphatic_cols].sum(axis=1) if aliphatic_cols else 0

    hc_dummies = pd.get_dummies(input_df["hc_subtype"].astype(str)).add_suffix("_hc_subtype")
    lc_dummies = pd.get_dummies(input_df["lc_subtype"].astype(str)).add_suffix("_lc_subtype")
    X = pd.concat([X, hc_dummies, lc_dummies], axis=1)

    # Level 3: CDR features
    has_heavy_aho = "heavy_aligned_aho" in input_df.columns
    has_light_aho = "light_aligned_aho" in input_df.columns

    cdr_cols = [
        "HCDR3_length",
        "HCDR3_gravy",
        "HCDR3_hydrophobic_count",
        "HCDR3_aromaticity",
        "HCDR3_aromatic_cluster",
        "HCDR3_hydrophobic_cluster_max_len",
        "HCDR3_hydrophobic_cluster_count",
        "HCDR3_hydrophobic_moment",
        "HCDR3_hydrophobic_density",
        "HCDR3_terminal_hydrophobicity",
        "LCDR1_length",
        "LCDR1_gravy",
        "LCDR1_hydrophobic_count",
        "LCDR1_aromaticity",
        "LCDR1_hydrophobic_cluster_max_len",
        "LCDR1_hydrophobic_cluster_count",
        "LCDR1_hydrophobic_moment",
        "LCDR1_hydrophobic_density",
        "LCDR1_terminal_hydrophobicity",
    ]
    X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=cdr_cols)], axis=1)

    if has_heavy_aho:
        h_start, h_end = HCDR3_RANGE_AHO
        hcdr3_seqs = input_df["heavy_aligned_aho"].map(
            lambda s: extract_cdr_from_aho(s, h_start, h_end)
        )
        hcdr3_feature_dicts = hcdr3_seqs.map(
            lambda seq: cdr_basic_features(seq, prefix="HCDR3")
        )
        hcdr3_df = pd.DataFrame(list(hcdr3_feature_dicts), index=input_df.index)

        for col in ["HCDR3_length", "HCDR3_gravy", "HCDR3_hydrophobic_count", "HCDR3_aromaticity"]:
            if col in hcdr3_df.columns:
                X[col] = hcdr3_df[col]

        X["HCDR3_aromatic_cluster"] = hcdr3_seqs.map(has_aromatic_cluster)
        X["HCDR3_hydrophobic_cluster_max_len"] = hcdr3_seqs.map(max_hydrophobic_cluster_len)
        X["HCDR3_hydrophobic_cluster_count"] = hcdr3_seqs.map(count_hydrophobic_clusters)
        X["HCDR3_hydrophobic_moment"] = hcdr3_seqs.map(helix_hydrophobic_moment)
        X["HCDR3_hydrophobic_density"] = (
            X["HCDR3_hydrophobic_count"] / X["HCDR3_length"].replace(0, np.nan)
        )
        X["HCDR3_terminal_hydrophobicity"] = hcdr3_seqs.map(terminal_hydrophobicity)

    if has_light_aho:
        l_start, l_end = LCDR1_RANGE_AHO
        lcdr1_seqs = input_df["light_aligned_aho"].map(
            lambda s: extract_cdr_from_aho(s, l_start, l_end)
        )
        lcdr1_feature_dicts = lcdr1_seqs.map(
            lambda seq: cdr_basic_features(seq, prefix="LCDR1")
        )
        lcdr1_df = pd.DataFrame(list(lcdr1_feature_dicts), index=input_df.index)

        for col in ["LCDR1_length", "LCDR1_gravy", "LCDR1_hydrophobic_count", "LCDR1_aromaticity"]:
            if col in lcdr1_df.columns:
                X[col] = lcdr1_df[col]

        X["LCDR1_hydrophobic_cluster_max_len"] = lcdr1_seqs.map(max_hydrophobic_cluster_len)
        X["LCDR1_hydrophobic_cluster_count"] = lcdr1_seqs.map(count_hydrophobic_clusters)
        X["LCDR1_hydrophobic_moment"] = lcdr1_seqs.map(helix_hydrophobic_moment)
        X["LCDR1_hydrophobic_density"] = (
            X["LCDR1_hydrophobic_count"] / X["LCDR1_length"].replace(0, np.nan)
        )
        X["LCDR1_terminal_hydrophobicity"] = lcdr1_seqs.map(terminal_hydrophobicity)

    X = X.fillna(0)
    return X