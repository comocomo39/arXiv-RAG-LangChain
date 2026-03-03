from __future__ import annotations

from search_service import ArxivBM25SearchService, SearchFilters


def pretty_print_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return

    for r in results:
        print("=" * 100)
        print(f"[{r['rank']}] {r['title']}")
        print(f"arXiv ID: {r['arxiv_id']} | Year: {r['year']} | Primary: {r['primary_category']}")
        print(f"Categories: {r['categories']}")
        print(f"URL: {r['url']}")
        print(f"Snippet: {r['snippet']}")
    print("=" * 100)


def main():
    print("Loading BM25 search service...")
    service = ArxivBM25SearchService(k=8)
    print(f"Corpus loaded: {len(service.docs):,} documents\n")

    demo_queries = [
        "dense retrieval for information retrieval",
        "query expansion pseudo relevance feedback",
        "transformer language model fine tuning",
        "recommendation systems contrastive learning",
    ]

    for q in demo_queries:
        print(f"\n\nQUERY: {q}")
        results = service.search(q)
        pretty_print_results(results)

    # Example with filters
    print("\n\nQUERY WITH FILTERS: retrieval and ranking")
    filters = SearchFilters(categories_any={"cs.IR"}, year_min=2018)
    results = service.search("retrieval and ranking", filters=filters, k_override=5)
    pretty_print_results(results)

    # Interactive mode (optional)
    print("\nEnter interactive mode (type 'exit' to quit).")
    while True:
        q = input("\nSearch query> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        results = service.search(q)
        pretty_print_results(results)


if __name__ == "__main__":
    main()