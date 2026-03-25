import pandas as pd
from pathlib import Path
from igfold import IgFoldRunner


def main():
    project_root = Path(__file__).resolve().parents[1]

    sequences_csv = project_root / "data" / "GDPa1_v1.2_sequences.csv"
    output_dir = project_root / "data" / "pdb_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sequences_csv)

    required_cols = ["antibody_id", "vh_protein_sequence", "vl_protein_sequence"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    runner = IgFoldRunner()

    for _, row in df.iterrows():
        antibody_id = str(row["antibody_id"]).strip()
        heavy = str(row["vh_protein_sequence"]).strip()
        light = str(row["vl_protein_sequence"]).strip()

        if not antibody_id:
            print("Skipping row: missing antibody_id")
            continue
        
        if not heavy or not light or heavy == "nan" or light == "nan":
            print(f"Skipping {antibody_id}: missing heavy/light sequence")
            continue

        out_pdb = output_dir / f"{antibody_id}.pdb"

        if out_pdb.exists():
            print(f"Skipping {antibody_id}: already exists")
            continue

        sequences = {
            "H": heavy,
            "L": light,
        }

        print(f"Generating structure for {antibody_id} ...")

        try:
            runner.fold(
                out_pdb.as_posix(),
                sequences=sequences,
                do_refine=False,
                do_renum=False,
            )
            print(f"Saved: {out_pdb}")
        except Exception as e:
            print(f"Failed for {antibody_id}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()