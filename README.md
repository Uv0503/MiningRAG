# Mining LLM – RAG for Indian Mining Operations

Retrieval-Augmented Generation over DGMS Acts/Rules/Regulations and mining books.

**Stack:** Streamlit UI · fastembed (BAAI/bge-small-en-v1.5) embeddings · NumPy cosine search · Groq LLM (openai/gpt-oss-120b)

## Run
    source .venv/bin/activate
    streamlit run app.py
(Index is prebuilt in `index/`; rebuild with `python rag.py` or the sidebar button.)

## Pipeline
PDFs (`knowledge_base/`) -> pypdf text -> 1000-char chunks (200 overlap) -> embeddings -> top-k cosine retrieval -> prompt with sources -> Groq LLM -> answer with [Source N] citations.

## Knowledge base
Mines Act 1952, Coal Mines Regulations 2017 (+gazette notifications), Metalliferous Mines Regulations 1961, Oil Mines Regulations 2017, Mines Rules 1955, Rescue/Creche/Vocational Training Rules, Explosives Rules 2008, Central Electricity Regulations 2023, OSH Code, DGMS circulars 2018-24, plus 3 open-access IntechOpen mining books. Source: dgms.gov.in, intechopen.com.
# MiningRAG
