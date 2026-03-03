from __future__ import annotations

import os


def get_llm(provider: str = "openai", model: str | None = None, temperature: float = 0.0):
    provider = provider.lower().strip()

    if provider == "openai":
        # pip install langchain-openai openai
        from langchain_openai import ChatOpenAI

        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        # Nota: usa un modello disponibile nel tuo account/API
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    elif provider == "ollama":
        # pip install langchain-ollama
        from langchain_ollama import ChatOllama

        model_name = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'.")