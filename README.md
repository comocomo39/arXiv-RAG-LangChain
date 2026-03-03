# arXiv-RAG with LangChain

A lightweight **arXiv paper search assistant** that combines:

* **Classical IR (BM25)** for transparent lexical retrieval
* **LangChain** for orchestration (LLM provider abstraction, prompts, grounded answer synthesis)
* **Local or API-based LLMs** (Ollama / OpenAI) for query rewriting and answer generation

The project is built around **arXiv metadata + abstracts** (not PDFs), using a **topic-focused subset** (e.g. `cs.IR`, `cs.LG`, `cs.CL`) to create a practical and reusable research search tool.

---

## Why this project

This project is designed to keep **retrieval and generation separated**:

* **BM25** remains the main ranking mechanism (interpretable and controllable)
* **LangChain + LLM** improve usability (query rewriting, synthesis, suggested follow-up queries)

This makes the system easier to debug and more faithful to classical IR principles, while still providing a modern user experience.

---

## Features

* ✅ Download and prepare arXiv abstracts dataset (Hugging Face)
* ✅ Filter by target categories (e.g. `cs.IR`, `cs.LG`, `cs.CL`)
* ✅ Build a **BM25 retriever** with LangChain
* ✅ Search papers with title, snippet, and metadata
* ✅ Add an LLM layer for:

  * query rewriting
  * grounded answer synthesis
  * suggested refined queries
* ✅ Support both:

  * **OpenAI** (API)
  * **Ollama** (local models)

---

## Project Structure

```text
arXiv-RAG/
├── data/
│   ├── raw/         # downloaded dataset (not tracked)
│   ├── interim/     # filtered CSV (not tracked)
│   └── processed/   # processed parquet (not tracked)
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── download_hf_arxiv.py
│   ├── preprocess.py
│   ├── build_documents.py
│   ├── run_prepare_corpus.py
│   ├── bm25_retriever.py
│   ├── search_service.py
│   ├── run_bm25_demo.py
│   ├── llm_provider.py
│   ├── llm_prompts.py
│   ├── llm_search_chain.py
│   └── run_search_with_llm.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/comocomo39/arXiv-RAG.git
cd arXiv-RAG
```

### 2) Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset (arXiv abstracts)

This project uses the Hugging Face dataset:

* `gfissore/arxiv-abstracts-2021`

### Download / convert dataset

```bash
python src/download_hf_arxiv.py
```

This should generate a local file like:

* `data/raw/arxiv_metadata.csv`

> **Note:** The dataset files are intentionally **not tracked** in Git because they are large.

---

## Step 1 — Prepare the corpus

This step:

* loads the raw arXiv dataset
* cleans and normalizes text
* parses categories
* filters target categories (e.g. `cs.IR`, `cs.LG`, `cs.CL`)
* builds a processed corpus for retrieval

Run:

```bash
python src/run_prepare_corpus.py
```

Expected outputs:

* `data/interim/arxiv_filtered_mvp.csv`
* `data/processed/arxiv_filtered_mvp.parquet`

---

## Step 2 — Run BM25 retrieval (classical IR)

This step builds a **BM25 retriever** and runs demo queries.

```bash
python src/run_bm25_demo.py
```

You should see:

* ranked paper titles
* arXiv IDs
* categories
* snippets
* URLs

---

## Step 3 — Run BM25 + LLM (LangChain orchestration)

This step adds:

* optional query rewriting
* grounded answer synthesis
* suggested refined queries

### Option A: Ollama (local LLM)

Install Ollama and pull a model (example: `qwen3:4b`).

Then run:

#### Windows (PowerShell)

```powershell
$env:LLM_PROVIDER="ollama"
$env:LLM_MODEL="qwen3:4b"
python src/run_search_with_llm.py
```

#### macOS / Linux

```bash
export LLM_PROVIDER="ollama"
export LLM_MODEL="qwen3:4b"
python src/run_search_with_llm.py
```

### Option B: OpenAI API

#### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:LLM_PROVIDER="openai"
# Optional:
# $env:LLM_MODEL="gpt-4.1-mini"
python src/run_search_with_llm.py
```

#### macOS / Linux

```bash
export OPENAI_API_KEY="YOUR_KEY"
export LLM_PROVIDER="openai"
python src/run_search_with_llm.py
```

---

## Why LangChain here?

LangChain is used as an **orchestration layer**, not as a replacement for retrieval.

It helps by providing:

* a unified interface across LLM providers (OpenAI / Ollama)
* prompt templating for query rewriting and answer synthesis
* document abstractions (`Document`) and retriever integration
* modular composition between retrieval and generation

BM25 remains the primary retrieval/ranking mechanism.



```
```
