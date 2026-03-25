"""
Feature group definitions for Random Forest experiments.
"""


def existing_cols(cols, df):
    """Return only feature columns that exist in the given DataFrame."""
    return [c for c in cols if c in df.columns]


def get_feature_sets(df):
    """
    Return feature groups used for Level 1, Level 2, Level 3, and Full RF models.
    """
    # Level 1: Global Fv
    level1_cols = [
        "fv_length",
        "fv_gravy",
        "fv_hydrophobic_count",
        "fv_pI",
        "fv_charge_pH7",
        "fv_frac_positive",
        "fv_frac_negative",
        "fv_frac_polar",
        "fv_frac_special",
    ]

    # Level 2: VH/VL chain-level
    #   - Physicochemical (sequence-derived)
    #   - Composition & counts (sequence-derived)
    #   - Categorical / annotation-based (subtype)
    level2_physchem_cols = [
        # length
        "vh_length", "vl_length", 
        
        # GRAVY / hydrophobic
        "vh_gravy", "vl_gravy", 
        "vh_hydrophobic_count", "vl_hydrophobic_count",
        
        # aromaticity / instability
        "vh_aromaticity", "vl_aromaticity",
        "vh_instability", "vl_instability",
        
        # pI / charge pH7
        "vh_pI", "vl_pI", 
        "vh_charge_pH7", "vl_charge_pH7",
        
        # secondary structure fractions
        "vh_helix", "vh_turn", "vh_sheet",
        "vl_helix", "vl_turn", "vl_sheet",

        # MW / charge at other pH / extinction
        "vh_molecular_weight", "vl_molecular_weight",
        "vh_ph_7_35_charge", "vl_ph_7_35_charge",
        "vh_ph_7_45_charge", "vl_ph_7_45_charge",
        "vh_molar_extinction_reduced", "vh_molar_extinction_oxidized",
        "vl_molar_extinction_reduced", "vl_molar_extinction_oxidized",

        # AA class fractions
        "vh_frac_positive", "vh_frac_negative",
        "vh_frac_polar", "vh_frac_special",
        "vl_frac_positive", "vl_frac_negative",
        "vl_frac_polar", "vl_frac_special",

        # VH–VL relation (hydrophobicity)
        "vh_vl_hydrophobicity_gap",
        "vh_vl_hydrophobicity_ratio",
    ]

    # 2-b. Composition & per-AA counts per chain (sequence-derived)
    aa_count_cols = [
        c for c in df.columns
        if c.endswith("_vh_protein_sequence") or c.endswith("_vl_protein_sequence")
    ]

    derived_count_cols = [
        "vh_aromatic_count", "vl_aromatic_count",
        "vh_aliphatic_count", "vl_aliphatic_count",
    ]

    # 2-c. Chain-level categorical / annotation-based (subtype one-hot)
    subtype_cols = [
        c for c in df.columns
        if c.endswith("_hc_subtype") or c.endswith("_lc_subtype")
    ]

    # extended Level 2 = Physicochemical + Composition/Counts + Categorical
    level2_cols = (
        level2_physchem_cols
        + aa_count_cols
        + derived_count_cols
        + subtype_cols
    )

    # Level 3: CDR-level (HCDR3 & LCDR1)
    level3_cols = [
        # HCDR3 basic
        "HCDR3_length",
        "HCDR3_gravy",
        "HCDR3_hydrophobic_count",
        "HCDR3_aromaticity",
        "HCDR3_aromatic_cluster",

        # HCDR3 hydrophobicity-focused
        "HCDR3_hydrophobic_cluster_max_len",
        "HCDR3_hydrophobic_cluster_count",
        "HCDR3_hydrophobic_moment",
        "HCDR3_hydrophobic_density",
        "HCDR3_terminal_hydrophobicity",

        # LCDR1 basic
        "LCDR1_length",
        "LCDR1_gravy",
        "LCDR1_hydrophobic_count",
        "LCDR1_aromaticity",

        # LCDR1 hydrophobicity-focused
        "LCDR1_hydrophobic_cluster_max_len",
        "LCDR1_hydrophobic_cluster_count",
        "LCDR1_hydrophobic_moment",
        "LCDR1_hydrophobic_density",
        "LCDR1_terminal_hydrophobicity",
    ]

    # Full model: Level 1 + 2 + 3
    full_cols = level1_cols + level2_cols + level3_cols

    level1_cols = existing_cols(level1_cols, df)
    level2_cols = existing_cols(level2_cols, df)
    level3_cols = existing_cols(level3_cols, df)
    full_cols = existing_cols(full_cols, df)

    return {
        "level1": level1_cols,
        "level2": level2_cols,
        "level3": level3_cols,
        "full": full_cols,
    }