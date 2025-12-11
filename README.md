# Research Trends Interactive Dashboard (OpenAlex + Streamlit + LLM)

Author: Ke Wang

## Overview
This project is an interactive research-trends visualization toolbox built for evaluating Python tool development, data processing, and visualization skills. It ingests the publication record of Prof. Xinyue Ye from OpenAlex, processes the metadata, and presents:

- Publication trends over time

- Citation patterns

- First-author statistics

- Venue distributions

- Concept/topic distributions

- Filters for year and publication type

- LLM-powered trend summary

- RAG question-answering over the professor’s work


## 1. Features

## Data Ingestion

- Automatically queries OpenAlex API for author ID + works

- Caches raw and processed data locally

- Robust to missing fields and multiple institutions

## Data Processing

- Cleaning and normalization of publication metadata

- Derivation of:

- publication year

- first-author boolean

- venue

- citation counts

- concept frequencies

## Interactive Visualizations (Altair + Streamlit)

- Publications per year

- Citations per year

- First-author publications per year

- Top venues

- Top concepts

## LLM Features 

- Generate a natural-language summary of research trends

- RAG-based Q&A grounded in selected publications

- Uses SentenceTransformers + retrieval + OpenAI API


