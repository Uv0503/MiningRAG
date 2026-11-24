"""Core RAG logic: PDF ingestion, embedding index, retrieval and answer generation."""
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "index"
EMB_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between consecutive chunks

SYSTEM_PROMPT = """You are "Mining LLM", an expert assistant for Indian mining operations.
You answer questions on DGMS (Directorate General of Mines Safety) rules, the Mines Act 1952,
Coal Mines Regulations 2017, Metalliferous Mines Regulations 1961, Mines Rules 1955,
explosives, mine safety, ventilation, ground control and general mining engineering.

Rules:
- Base your answer primarily on the CONTEXT passages provided. Cite sources inline as [Source N].
- Quote regulation / section numbers exactly when they appear in the context.
- If the context does not contain the answer, say so clearly, then give a brief general answer
  from your own mining knowledge, marked as "(general knowledge, not from knowledge base)".
- For numerical problems, show step-by-step calculations with units.
- Be concise and structured (use bullet points / headings where helpful)."""

_embedder = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


# ---------------------------------------------------------------- ingestion
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_useful(chunk: str) -> bool:
    """Skip chunks that are mostly garbage (bad OCR, Hindi font glyphs, tables of dots)."""
    if len(chunk) < 100:
        return False
    letters = sum(c.isascii() and c.isalpha() for c in chunk)
    return letters / len(chunk) > 0.5


def split_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # try to break at a sentence / line boundary
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_pdf_chunks(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    out = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            continue
        for chunk in split_text(text):
            if is_useful(chunk):
                out.append({"source": pdf_path.name, "page": page_no, "text": chunk})
    return out


def build_index(progress_cb=None):
    """Read every PDF in knowledge_base/, chunk, embed and save to index/."""
    INDEX_DIR.mkdir(exist_ok=True)
    pdfs = sorted(KB_DIR.glob("*.pdf"))
    chunks = []
    for i, pdf in enumerate(pdfs):
        if progress_cb:
            progress_cb(i / len(pdfs), f"Reading {pdf.name}")
        try:
            chunks.extend(load_pdf_chunks(pdf))
        except Exception as e:
            print(f"Skipping {pdf.name}: {e}")
    if progress_cb:
        progress_cb(0.95, f"Embedding {len(chunks)} chunks (first run downloads the model)...")
    vectors = np.array(list(get_embedder().embed([c["text"] for c in chunks], batch_size=64)),
                       dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalise for cosine sim
    np.save(EMB_FILE, vectors)
    CHUNKS_FILE.write_text(json.dumps(chunks))
    if progress_cb:
        progress_cb(1.0, "Done")
    return len(pdfs), len(chunks)


def index_exists() -> bool:
    return EMB_FILE.exists() and CHUNKS_FILE.exists()


def load_index():
    return np.load(EMB_FILE), json.loads(CHUNKS_FILE.read_text())


# ---------------------------------------------------------------- retrieval
def retrieve(query: str, vectors, chunks, k=5):
    q = np.array(list(get_embedder().query_embed(query)), dtype=np.float32)[0]
    q /= np.linalg.norm(q)
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def build_messages(query: str, hits, history):
    context = "\n\n".join(
        f"[Source {n}] ({h['source']}, page {h['page']})\n{h['text']}"
        for n, h in enumerate(hits, start=1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # short conversational memory
    messages.append({"role": "user",
                     "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"})
    return messages


def stream_answer(messages, model=None, temperature=0.2):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
        stream=True,
    )
    for part in stream:
        delta = part.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    n_pdfs, n_chunks = build_index(lambda p, m: print(f"[{p:4.0%}] {m}"))
    print(f"Indexed {n_pdfs} PDFs into {n_chunks} chunks.")
