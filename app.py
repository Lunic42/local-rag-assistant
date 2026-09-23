import math
import sqlite3
import ollama
import streamlit as st
from pypdf import PdfReader

DB_PATH = "research.db"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            page INTEGER,
            text TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            source TEXT,
            page INTEGER,
            chunk_index INTEGER,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def extract_documents(uploaded_files):
    pages = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        source = uploaded_file.name
        if source.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"source": source, "page": page_number, "text": text})
        else:
            text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
            if text:
                pages.append({"source": source, "page": 1, "text": text})
    return pages

def chunk_pages(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append({
                    "id": f"{page['source']} | {page['page']} | {chunk_index}",
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "content": content,
                })
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
            chunk_index += 1
    return chunks

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def semantic_search(query, chunks_with_embeddings, selected_sources, top_k=2):
    # Filter chunks to include only selected sources
    if selected_sources:
        filtered_chunks = [c for c in chunks_with_embeddings if c["source"] in selected_sources]
    else:
        filtered_chunks = chunks_with_embeddings

    if not filtered_chunks:
        return []

    query_emb = get_embedding(query)
    scored_chunks = []
    for chunk in filtered_chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        scored_chunks.append((sim, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

def generate_rag_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant context found in the selected source document(s)."

    context_blocks = []
    for score, chunk in retrieved_chunks:
        context_blocks.append(f"Source [{chunk['id']}]:\n{chunk['content']}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful research assistant. Answer the question using ONLY the provided context below.
Include explicit source citations in brackets (e.g. [source_file | page | chunk]) where appropriate.
If the information is not present in the context, state that you cannot answer based on the provided documents.

Context:
{context_str}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# App Initialization
init_db()

st.title("Local RAG Research Assistant")
st.write("Upload PDF or TXT files, filter sources, and generate cited answers using local Ollama models.")

uploaded_files = st.file_uploader("Upload research documents", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    pages = extract_documents(uploaded_files)
    chunks = chunk_pages(pages)

    # Sidebar Source Selector (Secret Mission)
    all_sources = list(set([p["source"] for p in pages]))
    st.sidebar.header("Secret Mission: Filter Sources")
    selected_sources = st.sidebar.multiselect(
        "Select source documents to search:",
        options=all_sources,
        default=all_sources
    )

    with st.spinner("Generating embeddings for document chunks..."):
        chunks_with_embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk["content"])
            chunk_copy = dict(chunk)
            chunk_copy["embedding"] = emb
            chunks_with_embeddings.append(chunk_copy)

    query = st.text_input("Ask a question about your uploaded documents:")

    if query:
        top_chunks = semantic_search(query, chunks_with_embeddings, selected_sources, top_k=2)

        st.subheader("Retrieved Context Chunks")
        if top_chunks:
            for score, chunk in top_chunks:
                with st.expander(f"Chunk ID: {chunk['id']} (Similarity: {score:.4f})"):
                    st.write(chunk["content"])
        else:
            st.warning("No chunks retrieved for the selected source document filter.")

        st.subheader("Generated Answer with Citations")
        with st.spinner("Generating answer with local LLM..."):
            answer = generate_rag_answer(query, top_chunks)
            st.markdown(answer)