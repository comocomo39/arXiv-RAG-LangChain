from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RAW_DIR / "arxiv_metadata.csv"

def main():
    # carica il dataset HF
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train")

    # converti a pandas (può richiedere tempo/memoria)
    df = ds.to_pandas()

    # ispeziona colonne
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    # Salva CSV compatibile col Punto 1
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

if __name__ == "__main__":
    main()