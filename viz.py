# viz.py

from __future__ import annotations

import altair as alt
import pandas as pd


def pubs_per_year_chart(df_pubs_per_year: pd.DataFrame) -> alt.Chart:
    """
    Bar chart: number of publications per year.
    Expects columns: 'year', 'n_publications'
    """
    if df_pubs_per_year.empty:
        return alt.Chart(pd.DataFrame({"year": [], "n_publications": []})).mark_bar()

    return (
        alt.Chart(df_pubs_per_year)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("n_publications:Q", title="Publications"),
            tooltip=["year", "n_publications"],
        )
        .properties(title="Publications per Year")
    )


def citations_per_year_chart(df_cites_per_year: pd.DataFrame) -> alt.Chart:
    """
    Line chart: total citations per year.
    Expects columns: 'year', 'total_citations'
    """
    if df_cites_per_year.empty:
        return alt.Chart(
            pd.DataFrame({"year": [], "total_citations": []})
        ).mark_line()

    return (
        alt.Chart(df_cites_per_year)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("total_citations:Q", title="Total citations (current)"),
            tooltip=["year", "total_citations"],
        )
        .properties(title="Citations per Year (current count)")
    )


def first_author_chart(df_first_author: pd.DataFrame) -> alt.Chart:
    """
    Bar chart: first-author publications per year.
    Expects columns: 'year', 'n_first_author'
    """
    if df_first_author.empty:
        return alt.Chart(
            pd.DataFrame({"year": [], "n_first_author": []})
        ).mark_bar()

    return (
        alt.Chart(df_first_author)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("n_first_author:Q", title="First-author publications"),
            tooltip=["year", "n_first_author"],
        )
        .properties(title="First-author Publications per Year")
    )


def top_venues_chart(df_venues: pd.DataFrame) -> alt.Chart:
    """
    Horizontal bar chart: top venues by publication count.
    Expects columns: 'venue', 'count'
    """
    if df_venues.empty:
        return alt.Chart(pd.DataFrame({"venue": [], "count": []})).mark_bar()

    return (
        alt.Chart(df_venues)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Publications"),
            y=alt.Y("venue:N", sort="-x", title="Venue"),
            tooltip=["venue", "count"],
        )
        .properties(title="Top Venues")
    )


def concept_bar_chart(df_concepts: pd.DataFrame) -> alt.Chart:
    """
    Horizontal bar chart: top concepts/topics by frequency.
    Expects columns: 'concept', 'count'
    """
    if df_concepts.empty:
        return alt.Chart(pd.DataFrame({"concept": [], "count": []})).mark_bar()

    return (
        alt.Chart(df_concepts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("concept:N", sort="-x", title="Concept"),
            tooltip=["concept", "count"],
        )
        .properties(title="Top Concepts / Topics")
    )
