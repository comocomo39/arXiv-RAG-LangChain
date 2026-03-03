import pandas as pd
from pathlib import Path


COLUMN_CANDIDATES = {
    "id": ["id", "arxiv_id"],
    "title": ["title"],
    "abstract": ["abstract", "summary"],
    "categories": ["categories", "category", "tags"],
    "authors": ["authors"],
    "published": ["published", "update_date", "submitted", "created"],
    "url": ["url", "link"],
}


def _resolve_column(df: pd.DataFrame, logical_name: str):
    for c in COLUMN_CANDIDATES[logical_name]:
        if c in df.columns:
            return c
    return None


def load_arxiv_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # low_memory=False avoids mixed dtype warnings in Kaggle-like CSVs
    df = pd.read_csv(path, low_memory=False)

    # Map to canonical columns
    rename_map = {}
    for logical in COLUMN_CANDIDATES:
        col = _resolve_column(df, logical)
        if col is not None:
            rename_map[col] = logical

    df = df.rename(columns=rename_map)

    required = ["id", "title", "abstract", "categories"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Optional columns: create if absent
    for opt in ["authors", "published", "url"]:
        if opt not in df.columns:
            df[opt] = None

    return df