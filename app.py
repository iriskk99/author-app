# app.py

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import data_loader
import processing
import rag
import viz



# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()  # load .env if present

st.set_page_config(
    page_title="Research Trends Dashboard",
    page_icon="📚",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers & caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_author_and_df(
    author_name: str,
    institution_hint: str,
    mailto: str | None,
    data_dir: str = "data",
) -> tuple[dict, pd.DataFrame]:
    """
    Cached pipeline:
      1. Get author + works from OpenAlex (or local JSON)
      2. Convert to clean DataFrame
      3. Save CSV for inspection
    """
    author, works = data_loader.get_author_and_works(
        author_name=author_name,
        institution_hint=institution_hint or None,
        data_dir=data_dir,
        mailto=mailto,
        refresh=False,
    )
    author_id = author["id"]

    df = processing.works_to_dataframe(works, target_author_id=author_id)

    clean_path = os.path.join(data_dir, "author_works_clean.csv")
    processing.save_clean_dataframe(df, clean_path)

    return author, df


def get_openai_client() -> OpenAI | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def call_openai_chat(prompt: str) -> str:
    client = get_openai_client()
    if client is None:
        return "OPENAI_API_KEY not set. Please configure your environment to enable LLM features."

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


# -----------------------------------------------------------------------------
# Sidebar: configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

default_author_name = "Xinyue Ye"
default_institution_hint = "Alabama"  # Department of Geography & the Environment, UA

author_name = st.sidebar.text_input(
    "Author name",
    value=default_author_name,
    help="Name of the author as in OpenAlex (display_name).",
)

institution_hint = st.sidebar.text_input(
    "Institution hint (optional)",
    value=default_institution_hint,
    help="Part of the institution name, e.g., 'Alabama', to disambiguate.",
)

mailto = st.sidebar.text_input(
    "Contact email for OpenAlex (optional)",
    value="youremail@example.com",
    help="Used in OpenAlex requests. Not required, but recommended.",
)

llm_enabled = st.sidebar.checkbox(
    "Enable LLM features (summary & chatbot)",
    value=True,
)

st.sidebar.markdown("---")
load_button = st.sidebar.button("Load / Refresh Publications", type="primary")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
st.title("📚 Research Trends Dashboard")
st.caption(
    "Interactive toolbox for exploring publication trends and integrating a simple LLM over the corpus."
)

if not author_name.strip():
    st.warning("Please enter an author name in the sidebar.")
    st.stop()

# Lazy load: only fetch when user clicks button at least once
if "author_data_loaded" not in st.session_state and not load_button:
    st.info("Set options in the sidebar, then click **Load / Refresh Publications**.")
    st.stop()

# If button pressed or state exists, load data
if load_button or "author_data_loaded" in st.session_state:
    with st.spinner("Loading publications from OpenAlex and preparing data..."):
        try:
            author_meta, df = load_author_and_df(
                author_name=author_name,
                institution_hint=institution_hint,
                mailto=mailto,
                data_dir="data",
            )
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    st.session_state["author_data_loaded"] = True

    if df.empty:
        st.warning("No publications found for this author.")
        st.stop()

    # -------------------------------------------------------------------------
    # Author header
    # -------------------------------------------------------------------------
    author_display_name = author_meta.get("display_name", author_name)
    last_inst = (author_meta.get("last_known_institution") or {}).get("display_name", "")
    st.subheader(f"Author: {author_display_name}")
    if last_inst:
        st.write(f"Last known institution: **{last_inst}**")
    st.markdown(
        f"OpenAlex ID: `{author_meta.get('id', 'unknown')}`  "
        f"| Works in dataset: **{len(df)}**"
    )

    # -------------------------------------------------------------------------
    # Filters
    # -------------------------------------------------------------------------
    with st.expander("Filters", expanded=True):
        years = sorted(
            [int(y) for y in df["year"].dropna().unique().tolist()]
        ) if df["year"].notna().any() else []
        if years:
            year_min, year_max = st.slider(
                "Publication year range",
                min_value=min(years),
                max_value=max(years),
                value=(min(years), max(years)),
                step=1,
            )
            df_filtered = df[
                df["year"].notna()
                & (df["year"] >= year_min)
                & (df["year"] <= year_max)
            ].copy()
        else:
            df_filtered = df.copy()

        types = sorted(df_filtered["type"].dropna().unique().tolist())
        if types:
            selected_types = st.multiselect(
                "Publication types",
                options=types,
                default=types,
            )
            df_filtered = df_filtered[df_filtered["type"].isin(selected_types)]

    # -------------------------------------------------------------------------
    # KPIs
    # -------------------------------------------------------------------------
    total_pubs = len(df_filtered)
    total_cites = int(df_filtered["cited_by_count"].sum())
    avg_cites = float(df_filtered["cited_by_count"].mean()) if total_pubs > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Publications (filtered)", total_pubs)
    col2.metric("Total citations (filtered)", total_cites)
    col3.metric("Avg citations per paper", round(avg_cites, 1))

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    st.markdown("### 📈 Trends and Distributions")

    col_left, col_right = st.columns(2)

    # Publications per year
    df_pubs_year = processing.publications_per_year(df_filtered)
    chart_pubs = viz.pubs_per_year_chart(df_pubs_year)
    col_left.altair_chart(chart_pubs, use_container_width=True)

    # Citations per year
    df_cites_year = processing.citations_per_year(df_filtered)
    chart_cites = viz.citations_per_year_chart(df_cites_year)
    col_right.altair_chart(chart_cites, use_container_width=True)

    # Second row: first-author + venues
    col_left2, col_right2 = st.columns(2)

    df_first_author = processing.first_author_counts_per_year(df_filtered)
    chart_first = viz.first_author_chart(df_first_author)
    col_left2.altair_chart(chart_first, use_container_width=True)

    df_venues = processing.venue_counts(df_filtered, top_n=10)
    chart_venues = viz.top_venues_chart(df_venues)
    col_right2.altair_chart(chart_venues, use_container_width=True)

    # Concepts / topics
    st.markdown("### 🧠 Top Concepts / Topics")
    df_concepts = processing.concept_counts(df_filtered, top_n=15)
    if not df_concepts.empty:
        chart_concepts = viz.concept_bar_chart(df_concepts)
        st.altair_chart(chart_concepts, use_container_width=True)
    else:
        st.info("No concepts available from OpenAlex for these works.")

    # -------------------------------------------------------------------------
    # Table of publications
    # -------------------------------------------------------------------------
    st.markdown("### 📄 Publications (filtered)")
    st.dataframe(
        df_filtered[
            ["year", "title", "venue", "type", "cited_by_count", "is_first_author"]
        ].sort_values(["year", "title"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # LLM: Summary of trends
    # -------------------------------------------------------------------------
    if llm_enabled:
        st.markdown("### 🧾 LLM Summary of Research Trends")

        df_llm = processing.build_llm_corpus(df_filtered)
        # Limit the number of docs in the prompt to keep it lightweight
        max_docs_for_summary = 40
        docs_sample = df_llm["doc"].head(max_docs_for_summary).tolist()

        if st.button("Generate Trend Summary"):
            if not docs_sample:
                st.info("No documents available for summarization.")
            else:
                joined = "\n\n".join(docs_sample)
                prompt = f"""
You are analyzing the publication record of a single researcher.

Below is a list of up to {max_docs_for_summary} publications. Each publication contains title, year, venue, and abstract.

Publications:
{joined}

1. Identify 3–5 key research themes.
2. Describe how these themes have evolved over time.
3. Suggest 2 potential future research directions consistent with these trends.

Respond in structured paragraphs.
"""
                summary = call_openai_chat(prompt)
                st.markdown(summary)

    # -------------------------------------------------------------------------
    # LLM: Mini RAG-based chatbot
    # -------------------------------------------------------------------------
    if llm_enabled:
        st.markdown("### 💬 Ask a Question about this Research")

        question = st.text_input(
            "Ask any question based on this author's publications (filtered):",
            value="What are the main GeoAI contributions in this publication set?",
        )

        if "rag_initialized" not in st.session_state:
            st.session_state["rag_initialized"] = False

        if question:
            if not st.session_state["rag_initialized"]:
                with st.spinner("Preparing embeddings for RAG..."):
                    df_llm_full = processing.build_llm_corpus(df_filtered)
                    docs = rag.make_docs(df_llm_full)
                    model = rag.build_embedding_model("all-MiniLM-L6-v2")
                    embs = rag.compute_embeddings(model, docs)
                    index = rag.build_faiss_index(embs)

                    st.session_state["rag_df_llm"] = df_llm_full
                    st.session_state["rag_docs"] = docs
                    st.session_state["rag_model"] = model
                    st.session_state["rag_index"] = index
                    st.session_state["rag_initialized"] = True

            df_llm_full = st.session_state.get("rag_df_llm")
            docs = st.session_state.get("rag_docs")
            model = st.session_state.get("rag_model")
            index = st.session_state.get("rag_index")

            if df_llm_full is None or docs is None or model is None or index is None:
                st.error("RAG resources not initialized correctly.")
            else:
                with st.spinner("Retrieving relevant papers and generating answer..."):
                    context_df, indices = rag.retrieve_top_k(
                        query=question,
                        model=model,
                        index=index,
                        df_llm=df_llm_full,
                        docs=docs,
                        k=5,
                    )
                    prompt = rag.build_qa_prompt(question, context_df)
                    answer = call_openai_chat(prompt)

                st.markdown("**Answer:**")
                st.markdown(answer)

                with st.expander("Show retrieved publications used as context"):
                    st.dataframe(
                        context_df[
                            ["year", "title", "venue", "cited_by_count"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
