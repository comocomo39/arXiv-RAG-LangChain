from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from llm_provider import get_llm
from llm_prompts import QUERY_REWRITE_PROMPT, ANSWER_SYNTHESIS_PROMPT
from search_service import ArxivBM25SearchService, SearchFilters


@dataclass
class LLMSearchConfig:
    provider: str = "openai"        # "openai" or "ollama"
    model: Optional[str] = None
    temperature: float = 0.0
    top_k: int = 3
    use_query_rewrite: bool = False


class ArxivLLMSearchChain:
    def __init__(self, config: LLMSearchConfig):
        self.config = config
        self.search_service = ArxivBM25SearchService(k=config.top_k)
        self.llm = get_llm(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
        )

    # -------------------------
    # Query rewriting
    # -------------------------
    def rewrite_query(self, query: str) -> list[str]:
        if not self.config.use_query_rewrite:
            return [query]

        messages = QUERY_REWRITE_PROMPT.invoke({"query": query})
        response = self.llm.invoke(messages)
        text = getattr(response, "content", str(response))

        lines = []
        for line in str(text).splitlines():
            line = line.strip()
            # remove "1) ", "2) ", etc.
            line = re.sub(r"^\d+\)\s*", "", line).strip()
            if line:
                lines.append(line)

        # Fallback safety
        lines = [q for q in lines if len(q) > 2]
        if not lines:
            return [query]

        # Keep max 3 + include original if not present
        unique = []
        for q in [query] + lines:
            if q not in unique:
                unique.append(q)
        return unique[:4]

    # -------------------------
    # Multi-query retrieval
    # -------------------------
    def retrieve_with_multiquery(
        self,
        user_query: str,
        filters: Optional[SearchFilters] = None,
        final_k: Optional[int] = None,
    ) -> tuple[list[str], list[Document]]:
        final_k = final_k or self.config.top_k
        queries = self.rewrite_query(user_query)

        all_docs: list[Document] = []
        seen = set()

        # Collect docs from each query, deduplicate by arxiv_id
        for q in queries:
            docs = self.search_service.retrieve_documents(
                q,
                k_override=final_k,
                filters=filters,
                overfetch_factor=4,
            )
            for d in docs:
                arxiv_id = (d.metadata or {}).get("arxiv_id")
                key = arxiv_id or id(d)
                if key in seen:
                    continue
                seen.add(key)
                all_docs.append(d)

        # Simple heuristic rerank (lexical coverage on title+abstract)
        scored = []
        uq_tokens = set(user_query.lower().split())
        for d in all_docs:
            text = (d.page_content or "").lower()
            overlap = sum(1 for t in uq_tokens if t in text)
            scored.append((overlap, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in scored[:final_k]]

        return queries, top_docs

    # -------------------------
    # Answer synthesis
    # -------------------------
    def answer(
        self,
        user_query: str,
        filters: Optional[SearchFilters] = None,
        final_k: Optional[int] = None,
    ) -> dict:
        queries_used, docs = self.retrieve_with_multiquery(
            user_query=user_query,
            filters=filters,
            final_k=final_k,
        )

        if not docs:
            return {
                "query": user_query,
                "queries_used": queries_used,
                "retrieved_docs": [],
                "answer": "No relevant papers found for the current query/filters.",
            }

        context = self.search_service.format_docs_for_llm(docs)
        messages = ANSWER_SYNTHESIS_PROMPT.invoke(
            {"query": user_query, "context": context}
        )
        response = self.llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        # Also return structured retrieval preview (for UI/debugging)
        retrieved_preview = []
        for i, d in enumerate(docs, start=1):
            m = d.metadata or {}
            retrieved_preview.append({
                "rank": i,
                "title": m.get("title"),
                "arxiv_id": m.get("arxiv_id"),
                "year": m.get("year"),
                "categories": m.get("categories"),
                "url": m.get("url"),
            })

        return {
            "query": user_query,
            "queries_used": queries_used,
            "retrieved_docs": retrieved_preview,
            "answer": answer_text,
        }