# src/processing.py

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def works_to_dataframe(works: List[Dict], target_author_id: str) -> pd.DataFrame:
    """
    Convert raw OpenAlex works for a given author into a clean DataFrame.
    Adds first/last-author flags and extracts key fields.
    """
    rows = []

    for w in works:
        authorships = w.get("authorships", [])
        is_first_author = False
        is_last_author = False

        for i, a in enumerate(authorships):
            auth_id = (a.get("author") or {}).get("id")
            if auth_id == target_author_id:
                if i == 0:
                    is_first_author = True
                if i == len(authorships) - 1:
                    is_last_author = True

        concepts = w.get("concepts") or []
        concept_names = [c.get("display_name") for c in concepts if c.get("display_name")]

        host_venue = w.get("host_venue") or {}

        row = {
            "id": w.get("id"),
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "type": w.get("type"),
            "venue": host_venue.get("display_name"),
            "doi": w.get("doi"),
            "cited_by_count": w.get("cited_by_count"),
            "is_paratext": w.get("is_paratext"),
            "is_retracted": w.get("is_retracted"),
            "is_first_author": is_first_author,
            "is_last_author": is_last_author,
            "concepts": concept_names,
            "abstract": w.get("abstract"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Basic cleaning
    if not df.empty:
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        if "cited_by_count" in df.columns:
            df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")
            df["cited_by_count"] = df["cited_by_count"].fillna(0).astype(int)

    return df


def save_clean_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_clean_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def publications_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Count of publications per year."""
    tmp = df.dropna(subset=["year"])
    return (
        tmp.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "n_publications"})
    )


def citations_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Total citations per year (based on current cited_by_count)."""
    tmp = df.dropna(subset=["year"])
    return (
        tmp.groupby("year", as_index=False)["cited_by_count"]
        .sum()
        .rename(columns={"cited_by_count": "total_citations"})
    )


def first_author_counts_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Number of first-author papers per year for this author."""
    tmp = df.dropna(subset=["year"])
    tmp = tmp[tmp["is_first_author"] == True]  # noqa: E712
    return (
        tmp.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "n_first_author"})
    )


def venue_counts(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top venues by number of publications."""
    tmp = df.dropna(subset=["venue"])
    vc = (
        tmp.groupby("venue", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return vc


def concept_counts(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top concepts/topics across all works."""
    all_rows = []
    for _, row in df.iterrows():
        concepts = row.get("concepts") or []
        for c in concepts:
            all_rows.append({"concept": c})

    if not all_rows:
        return pd.DataFrame(columns=["concept", "count"])

    tmp = pd.DataFrame(all_rows)
    counts = (
        tmp.groupby("concept", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return counts


def build_llm_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 'doc' column combining title, year, venue, and abstract
    for LLM summarization / RAG.
    """
    df_llm = df.copy()
    def _make_doc(row):
        title = row.get("title") or ""
        year = row.get("year")
        venue = row.get("venue") or ""
        abstract = row.get("abstract") or ""
        header = f"{title}"
        if pd.notna(year):
            header += f" ({int(year)})"
        if venue:
            header += f", {venue}"
        if abstract:
            return f"{header}\n\nAbstract: {abstract}"
        return header

    df_llm["doc"] = df_llm.apply(_make_doc, axis=1)
    return df_llm
