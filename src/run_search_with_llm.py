from __future__ import annotations

import os

from llm_search_chain import ArxivLLMSearchChain, LLMSearchConfig
from search_service import SearchFilters


def print_result(result: dict):
    print("\n" + "=" * 110)
    print(f"USER QUERY: {result['query']}")
    print("-" * 110)
    print("QUERIES USED:")
    for q in result.get("queries_used", []):
        print(f"  - {q}")

    print("\nRETRIEVED PAPERS (preview):")
    for d in result.get("retrieved_docs", []):
        print(f"  [{d['rank']}] {d['title']} ({d['arxiv_id']}) | {d['year']} | {d['categories']}")
        print(f"      {d['url']}")

    print("\nLLM ANSWER:")
    print(result.get("answer", ""))
    print("=" * 110 + "\n")


def main():
    # Scegli provider:
    # - "openai" (richiede OPENAI_API_KEY)
    # - "ollama" (richiede Ollama attivo in locale)
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", None)

    chain = ArxivLLMSearchChain(
        LLMSearchConfig(
            provider=provider,
            model=model,
            temperature=0.0,
            top_k=3,
            use_query_rewrite=False,
        )
    )

    # Demo query iniziali
    demo_queries = [
        "papers about dense retrieval training with hard negatives",
        #"query expansion and pseudo relevance feedback in information retrieval",
        #"recommendation systems with contrastive learning",
    ]

    for q in demo_queries:
        result = chain.answer(q)
        print_result(result)

    # Esempio con filtro
    filters = SearchFilters(categories_any={"cs.IR"})
    result = chain.answer("retrieval ranking learning to rank", filters=filters, final_k=6)
    print_result(result)

    # Interactive mode
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        q = input("Search query> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = chain.answer(q)
        print_result(result)


if __name__ == "__main__":
    main()