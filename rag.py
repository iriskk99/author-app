# rag.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def make_docs(df_llm: pd.DataFrame) -> List[str]:
    """
    Extract list of documents (strings) from df_llm['doc'].
    """
    return df_llm["doc"].fillna("").astype(str).tolist()


def build_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def compute_embeddings(
    model: SentenceTransformer,
    docs: List[str],
) -> np.ndarray:
    """
    Compute normalized embeddings for a list of documents.
    Returns a float32 array of shape (n_docs, dim).
    """
    embs = model.encode(docs, normalize_embeddings=True)
    return embs.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> np.ndarray:
    """
    Compatibility wrapper: previously returned a FAISS index.
    Now we simply return the embeddings matrix itself, which will act as our index.

    So `index` is a numpy array of shape (n_docs, dim).
    """
    return embeddings


def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    index: np.ndarray,
    df_llm: pd.DataFrame,
    docs: List[str],
    k: int = 5,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Retrieve top-k documents using pure NumPy cosine similarity.

    - `index` is the embeddings matrix of shape (n_docs, dim).
    - Returns subset of df_llm and list of row indices.
    """
    if index is None or len(index) == 0:
        return df_llm.iloc[0:0].copy(), []

    # Compute query embedding (normalized)
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")  # (1, dim)
    q_vec = q_emb[0]  # (dim,)

    # Cosine similarity with normalized embeddings is just dot product
    sims = np.dot(index, q_vec)  # shape (n_docs,)

    # Get top-k indices
    k = min(k, len(sims))
    top_ids = np.argsort(-sims)[:k].tolist()

    subset = df_llm.iloc[top_ids].copy()
    return subset, top_ids


def build_qa_prompt(question: str, context_df: pd.DataFrame) -> str:
    """
    Construct an LLM prompt from retrieved papers + user question.
    """
    parts = []
    for _, row in context_df.iterrows():
        title = row.get("title") or ""
        year = row.get("year")
        venue = row.get("venue") or ""
        abstract = row.get("abstract") or ""

        header = f"Title: {title}"
        if pd.notna(year):
            header += f" ({int(year)})"
        if venue:
            header += f", {venue}"

        body = f"{header}\nAbstract: {abstract}"
        parts.append(body)

    context_text = "\n\n---\n\n".join(parts)

    prompt = f"""
You are a helpful assistant answering questions about a single researcher's publications.

Use ONLY the information in the following papers to answer the question.
If something is not supported by the text, say you are not sure.

Papers:
{context_text}

Question: {question}

Answer concisely.
"""

    return prompt.strip()

