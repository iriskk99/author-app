# Research Trends Dashboard (OpenAlex + Streamlit + LLM)

This project is a small Python-based interactive toolbox that:

- Loads a researcher's publication data from the **OpenAlex API**
- Cleans and summarizes the data using **pandas**
- Produces **interactive visualizations** in Streamlit:
  - Publications per year
  - Citations per year
  - First-author publications over time
  - Top venues
  - Top concepts / topics
- Integrates a **lightweight LLM layer**:
  - Trend summarization
  - A mini RAG-based Q&A chatbot over the publication corpus

It is configured by default for **Prof. Xinyue Ye** (Department of Geography and the Environment, Alabama Center for the Advancement of AI, The University of Alabama) but can be reused for any author in OpenAlex.

## 1. Installation

Create and activate a virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt
