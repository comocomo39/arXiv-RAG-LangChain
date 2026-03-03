from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import INTERIM_DIR, PROCESSED_DIR


# -----------------------------
# Text preprocessing (light)
# -----------------------------
def normalize_text_for_bm25(text: str) -> str:
    """
    Light normalization for BM25 tokenization.
    Keep it simple in v1 to avoid harming technical terms too much.
    """
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Keep letters/numbers and common separators useful in technical text
    text = re.sub(r"[^a-z0-9\-\._/\+\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bm25_preprocess_func(text: str) -> list[str]:
    """
    Tokenizer function passed to LangChain BM25Retriever.
    """
    text = normalize_text_for_bm25(text)
    return text.split()


# -----------------------------
# Corpus loading
# -----------------------------
def load_processed_corpus_df(
    parquet_path: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Load the processed corpus created in Punto 1.
    Prefers parquet, falls back to CSV.
    """
    if parquet_path is None:
        parquet_path = PROCESSED_DIR / "arxiv_filtered_mvp.parquet"
    else:
        parquet_path = Path(parquet_path)

    if csv_path is None:
        csv_path = INTERIM_DIR / "arxiv_filtered_mvp.csv"
    else:
        csv_path = Path(csv_path)

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df

    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        # reparse date if needed
        if "published" in df.columns:
            df["published"] = pd.to_datetime(df["published"], errors="coerce")
        return df

    raise FileNotFoundError(
        f"No processed corpus found. Checked:\n- {parquet_path}\n- {csv_path}\n"
        "Run Punto 1 first (run_prepare_corpus.py)."
    )


def dataframe_to_langchain_documents(df: pd.DataFrame) -> list[Document]:
    """
    Rebuild LangChain Documents from the processed dataframe.
    This mirrors Punto 1 to avoid serialization complexity.
    """
    docs: list[Document] = []

    for _, row in df.iterrows():
        page_content = str(row.get("doc_text", "")).strip()
        if not page_content:
            continue

        year_value = row.get("year")
        try:
            year_value = int(year_value) if pd.notna(year_value) else None
        except Exception:
            year_value = None

        doc = Document(
            page_content=page_content,
            metadata={
                "arxiv_id": row.get("id"),
                "title": row.get("title"),
                "authors": row.get("authors", ""),
                "categories": row.get("categories", ""),
                "primary_category": row.get("primary_category"),
                "year": year_value,
                "url": row.get("url"),
            },
        )
        docs.append(doc)

    return docs


# -----------------------------
# BM25 retriever builder
# -----------------------------
def build_bm25_retriever(
    docs: list[Document],
    k: int = 8,
    preprocess_func: Optional[Callable[[str], list[str]]] = bm25_preprocess_func,
) -> BM25Retriever:
    """
    Build LangChain BM25Retriever from documents.
    """
    if not docs:
        raise ValueError("Cannot build BM25 retriever: docs list is empty.")

    retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=preprocess_func,
    )
    retriever.k = k
    return retriever


def load_and_build_bm25(
    k: int = 8,
    parquet_path: Optional[str | Path] = None,
    csv_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, list[Document], BM25Retriever]:
    """
    Convenience function for demos/services.
    """
    df = load_processed_corpus_df(parquet_path=parquet_path, csv_path=csv_path)
    docs = dataframe_to_langchain_documents(df)
    retriever = build_bm25_retriever(docs, k=k)
    return df, docs, retriever