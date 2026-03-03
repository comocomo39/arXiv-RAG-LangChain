from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.documents import Document

from bm25_retriever import load_and_build_bm25


@dataclass
class SearchFilters:
    categories_any: Optional[set[str]] = None   # e.g. {"cs.IR", "cs.LG"}
    year_min: Optional[int] = None
    year_max: Optional[int] = None


def _doc_matches_filters(doc: Document, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True

    meta = doc.metadata or {}

    # Category filtering
    if filters.categories_any:
        raw_categories = str(meta.get("categories", "") or "")
        doc_categories = set(raw_categories.split())
        if doc_categories.isdisjoint(filters.categories_any):
            return False

    # Year filtering
    year = meta.get("year")
    if year is not None:
        try:
            year = int(year)
        except Exception:
            year = None

    if filters.year_min is not None:
        if year is None or year < filters.year_min:
            return False

    if filters.year_max is not None:
        if year is None or year > filters.year_max:
            return False

    return True


def _make_snippet(text: str, max_chars: int = 320) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_result(doc: Document, rank: int) -> dict:
    meta = doc.metadata or {}
    content = doc.page_content or ""

    # doc.page_content = title + "\n\n" + abstract (from Punto 1)
    parts = content.split("\n\n", 1)
    title_from_text = parts[0].strip() if parts else ""
    abstract_from_text = parts[1].strip() if len(parts) > 1 else content

    title = meta.get("title") or title_from_text or "Untitled"
    snippet = _make_snippet(abstract_from_text)

    return {
        "rank": rank,
        "title": title,
        "arxiv_id": meta.get("arxiv_id"),
        "year": meta.get("year"),
        "categories": meta.get("categories"),
        "primary_category": meta.get("primary_category"),
        "authors": meta.get("authors"),
        "url": meta.get("url"),
        "snippet": snippet,
    }


class ArxivBM25SearchService:
    """
    Thin service layer around LangChain BM25Retriever.
    Keeps retrieval logic separated from CLI/UI/LLM orchestration.
    """

    def __init__(self, k: int = 8):
        self.df, self.docs, self.retriever = load_and_build_bm25(k=k)

    def search(
        self,
        query: str,
        k_override: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        overfetch_factor: int = 4,
    ) -> list[dict]:
        """
        Search arXiv corpus with BM25.
        Because filtering is post-retrieval, we overfetch to avoid empty results.
        """
        if not query or not query.strip():
            return []

        original_k = self.retriever.k
        target_k = k_override or original_k
        self.retriever.k = max(target_k * overfetch_factor, target_k)

        try:
            retrieved_docs = self.retriever.invoke(query)
        finally:
            self.retriever.k = original_k

        filtered_docs = [d for d in retrieved_docs if _doc_matches_filters(d, filters)]
        top_docs = filtered_docs[:target_k]

        return [format_result(doc, rank=i + 1) for i, doc in enumerate(top_docs)]

    def retrieve_documents(
        self,
        query: str,
        k_override: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        overfetch_factor: int = 4,
    ) -> list[Document]:
        """
        Returns raw LangChain Documents (useful for LLM context in Punto 3).
        """
        if not query or not query.strip():
            return []

        original_k = self.retriever.k
        target_k = k_override or original_k
        self.retriever.k = max(target_k * overfetch_factor, target_k)

        try:
            retrieved_docs = self.retriever.invoke(query)
        finally:
            self.retriever.k = original_k

        filtered_docs = [d for d in retrieved_docs if _doc_matches_filters(d, filters)]
        return filtered_docs[:target_k]

    @staticmethod
    def format_docs_for_llm(docs: list[Document]) -> str:
        """
        Prepares a grounded context string for the LLM (Punto 3).
        """
        chunks = []
        for i, d in enumerate(docs, start=1):
            m = d.metadata or {}
            chunks.append(
                f"[Paper {i}]\n"
                f"Title: {m.get('title')}\n"
                f"arXiv ID: {m.get('arxiv_id')}\n"
                f"Authors: {m.get('authors')}\n"
                f"Categories: {m.get('categories')}\n"
                f"Year: {m.get('year')}\n"
                f"URL: {m.get('url')}\n"
                f"Content:\n{d.page_content}\n"
            )
        return "\n\n---\n\n".join(chunks)