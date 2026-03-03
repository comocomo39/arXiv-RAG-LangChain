from langchain_core.documents import Document
import pandas as pd


def row_to_document(row: pd.Series) -> Document:
    return Document(
        page_content=row["doc_text"],
        metadata={
            "arxiv_id": row["id"],
            "title": row["title"],
            "authors": row.get("authors", ""),
            "categories": " ".join(row.get("category_list", [])) if isinstance(row.get("category_list", []), list) else str(row.get("categories", "")),
            "primary_category": row.get("primary_category"),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "url": row.get("url"),
        },
    )


def build_langchain_documents(df: pd.DataFrame) -> list[Document]:
    docs = [row_to_document(row) for _, row in df.iterrows()]
    return docs
