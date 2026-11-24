"""Streamlit frontend for Mining LLM (RAG over DGMS rules & mining books)."""
import os

import streamlit as st

import rag

st.set_page_config(page_title="Mining LLM – Indian Mining Assistant", page_icon="⛏️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(90deg,#3a2c1a,#7a5a2b); padding: 1.2rem 1.5rem;
       border-radius: 12px; color: #fff; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.9rem; color:#fff;}
.hero p {margin: .3rem 0 0; opacity: .85;}
.src {font-size: .85rem; border-left: 3px solid #c8963e; padding-left: .6rem; margin-bottom: .6rem;}
</style>
<div class="hero">
  <h1>⛏️ Mining LLM</h1>
  <p>AI assistant for Indian mining operations: DGMS Acts, Rules, Regulations, Circulars and mining textbooks</p>
</div>
""", unsafe_allow_html=True)

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    default_model = os.getenv("GROQ_MODEL", MODELS[0])
    model = st.selectbox("LLM (Groq)", MODELS,
                         index=MODELS.index(default_model) if default_model in MODELS else 0)
    top_k = st.slider("Passages to retrieve (top-k)", 2, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.header("📚 Knowledge Base")
    pdfs = sorted(rag.KB_DIR.glob("*.pdf"))
    with st.expander(f"{len(pdfs)} documents", expanded=False):
        for p in pdfs:
            st.caption(f"📄 {p.stem.replace('_', ' ')}")

    uploaded = st.file_uploader("Add your own PDF", type="pdf")
    if uploaded and st.button("Add & re-index"):
        (rag.KB_DIR / uploaded.name).write_bytes(uploaded.getbuffer())
        st.session_state.pop("index", None)
        st.session_state.rebuild = True
        st.rerun()

    if st.button("🔄 Rebuild index"):
        st.session_state.pop("index", None)
        st.session_state.rebuild = True
        st.rerun()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY missing – add it to the .env file.")

# ---------------------------------------------------------------- index
if st.session_state.pop("rebuild", False) or not rag.index_exists():
    bar = st.progress(0.0, "Building knowledge-base index...")
    n_pdfs, n_chunks = rag.build_index(lambda p, m: bar.progress(min(p, 1.0), m))
    bar.empty()
    st.success(f"Indexed {n_pdfs} PDFs into {n_chunks} chunks.")

if "index" not in st.session_state:
    with st.spinner("Loading index..."):
        st.session_state.index = rag.load_index()
        rag.get_embedder()
vectors, chunks = st.session_state.index
st.sidebar.caption(f"Index: {len(chunks):,} chunks · embeddings: {rag.EMBED_MODEL}")

# ---------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(hits):
    with st.expander("📑 Sources used"):
        for n, h in enumerate(hits, start=1):
            st.markdown(f"<div class='src'><b>[Source {n}] {h['source']}</b>, page {h['page']} "
                        f"· similarity {h['score']:.2f}<br>{h['text'][:400]}…</div>",
                        unsafe_allow_html=True)


EXAMPLES = [
    "What are the duties of a mine manager under CMR 2017?",
    "What is the minimum age for employment in a mine under the Mines Act 1952?",
    "What are the ventilation requirements for underground coal mines?",
    "Explain the requirements for shot firing and blasting in a coal mine.",
]

if not st.session_state.messages:
    st.markdown("##### Try asking:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⛏️" if msg["role"] == "assistant" else "👷"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

query = st.chat_input("Ask about DGMS regulations, mine safety, mining methods...")
query = query or st.session_state.pop("pending", None)

if query:
    with st.chat_message("user", avatar="👷"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="⛏️"):
        hits = rag.retrieve(query, vectors, chunks, k=top_k)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        try:
            answer = st.write_stream(
                rag.stream_answer(rag.build_messages(query, hits, history), model, temperature))
        except Exception as e:
            answer = f"⚠️ Error calling the LLM: {e}"
            st.error(answer)
        show_sources(hits)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})
