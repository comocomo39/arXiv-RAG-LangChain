from pathlib import Path
import pandas as pd

from config import RAW_DIR, INTERIM_DIR, PROCESSED_DIR, TARGET_CATEGORIES
from data_loading import load_arxiv_csv
from preprocess import preprocess_arxiv_df
from build_documents import build_langchain_documents


def main():
    # Cambia il nome file in base a quello che scarichi (Kaggle/HF export)
    input_path = RAW_DIR / "arxiv_metadata.csv"

    print(f"[1/5] Loading dataset from: {input_path}")
    df = load_arxiv_csv(input_path)
    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print(f"[2/5] Preprocessing + filtering categories: {sorted(TARGET_CATEGORIES)}")
    df_clean = preprocess_arxiv_df(df, target_categories=TARGET_CATEGORIES)
    print(f"Filtered rows: {len(df_clean):,}")

    print("[3/5] Saving interim processed dataframe")
    interim_csv = INTERIM_DIR / "arxiv_filtered_mvp.csv"
    df_clean.to_csv(interim_csv, index=False)
    print(f"Saved: {interim_csv}")

    print("[4/5] Building LangChain documents")
    docs = build_langchain_documents(df_clean)
    print(f"Built LangChain documents: {len(docs):,}")

    # Optional lightweight export for debugging (JSONL-ish via pandas)
    print("[5/5] Saving processed parquet")
    parquet_path = PROCESSED_DIR / "arxiv_filtered_mvp.parquet"
    df_clean.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path}")

    # Small sanity preview
    if docs:
        d0 = docs[0]
        print("\nSample Document metadata:")
        print(d0.metadata)
        print("\nSample page_content (first 400 chars):")
        print(d0.page_content[:400])

    print("\nDone. Punto 1 completato ✅")


if __name__ == "__main__":
    main()