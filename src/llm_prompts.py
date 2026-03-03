from langchain_core.prompts import ChatPromptTemplate


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are helping improve search queries for an academic paper search engine.

Task:
Rewrite the user query into 3 keyword-focused search queries suitable for BM25 lexical retrieval.
Keep them concise and technical.
Prefer domain terms, synonyms, and closely related phrases.
Do NOT add explanations.

User query:
{query}

Output format (exactly 3 lines):
1) ...
2) ...
3) ...
""")


ANSWER_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are an academic search assistant.

You MUST answer ONLY using the retrieved papers below.
If the retrieved papers are not sufficient, say so clearly.
Do not invent papers, authors, or claims.

User query:
{query}

Retrieved papers:
{context}

Return in this exact structure:

1. Short answer
- A concise answer (3-6 lines) grounded in the retrieved papers.

2. Most relevant papers
- For each paper, include:
  - Title
  - arXiv ID
  - Why it is relevant (1-2 lines)
  - URL

3. Suggested refined queries
- Provide 3 improved search queries the user can try next.

Important:
- Prefer precision over broad claims.
- If the query is broad or ambiguous, mention it.
""")