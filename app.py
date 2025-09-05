# app.py  (no typing hints, cleaner structure)
import os
from dotenv import load_dotenv
import PyPDF2
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai  # Gemini

load_dotenv()

# === 1) Extract text from PDF ===
def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# === 2) Chunk by character limit (with overlap) ===
def chunk_text(text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

# === 3) Store chunks in ChromaDB (always clears collection first) ===
def store_chunks_in_chroma(
    chunks,
    client,
    collection_name,
    embedding_function,
    batch_size=128,
    base_metadata=None,
    id_prefix=None,
    verbose=True,
):
    try:
        client.delete_collection(collection_name)
        if verbose:
            print(f"[Chroma] Deleted existing collection: {collection_name}")
    except Exception as e:
        if verbose:
            print(f"[Chroma] Could not delete collection (may not exist): {e}")

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    n = len(chunks)
    if n == 0:
        return vector_store, {"added": 0, "batches": 0}

    def make_id(i):
        return f"{id_prefix}-{i}" if id_prefix else f"{collection_name}-{i}"

    added, batches = 0, 0
    base_metadata = base_metadata or {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = chunks[start:end]
        batch_ids = [make_id(i) for i in range(start, end)]
        batch_metadatas = [{**base_metadata, "chunk_index": i} for i in range(start, end)]

        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
        added += len(batch_texts)
        batches += 1
        if verbose:
            print(f"[Chroma] Added batch {batches}: {start}..{end-1} ({len(batch_texts)} chunks)")

    return vector_store, {"added": added, "batches": batches}

# === 4) Query most relevant chunks (semantic similarity) ===
def query_chroma(vector_store, query_text, k=3, truncate=300):
    hits = vector_store.similarity_search(query_text, k=k)
    print(f'\n=== Top {min(k, len(hits))} results for: "{query_text}" ===')
    for idx, doc in enumerate(hits, 1):
        content = doc.page_content or ""
        preview = (content[:truncate] + "…") if len(content) > truncate else content
        print(f"\nResult {idx}:")
        print(f"- metadata: {doc.metadata}")
        print(f"- content (truncated): {preview}")
    return hits

# === 5) Ask Gemini with retrieved context ===
def answer_with_gemini(vector_store, query_text, k=3, model_name=None, max_context_chars=8000):
    # Retrieve top-k chunks
    docs = vector_store.similarity_search(query_text, k=k)

    # Build a compact context block (respect a soft char budget)
    context_parts = []
    used = 0
    for i, d in enumerate(docs, 1):
        snippet = d.page_content or ""
        remain = max_context_chars - used
        if remain <= 0:
            break
        snippet = snippet[:remain]
        md = d.metadata if isinstance(d.metadata, dict) else {}
        src = md.get("source", "unknown_source")
        idx = md.get("chunk_index", i-1)
        context_parts.append(f"[{i}] source={src} chunk_index={idx}\n{snippet}")
        used += len(snippet)

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "N/A"

    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    # System-style instruction + user query + retrieved context
    prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say "I couldn't find that in the provided documents."

# QUESTION
{query_text}

# CONTEXT
{context_block}

# INSTRUCTIONS
- Prefer exact numbers, names, and facts from context.
- If multiple snippets disagree, mention that briefly and pick the most consistent.
- Keep the answer concise and directly address the question.
""".strip()

    resp = model.generate_content(prompt)
    answer = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp and resp.candidates else "")
    print("\n=== Gemini Answer ===\n" + (answer or "[No text returned]"))
    return answer, docs


if __name__ == "__main__":
    # === Config ===
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(base_dir, "files", "test_pdf.pdf")  # adjust as needed
    collection_name = "pdf_chunks"
    chunk_size, overlap = 500, 100

    # Env vars
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "RAG")
    if not CHROMA_API_KEY or not CHROMA_TENANT:
        raise RuntimeError("Missing CHROMA_API_KEY or CHROMA_TENANT. Set them as environment variables.")

    print(f"Loading: {os.path.relpath(pdf_path, base_dir)}")
    text = extract_text_from_pdf(pdf_path)

    # Chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Total chunks (by_length): {len(chunks)} (size={chunk_size}, overlap={overlap})")

    # Embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Chroma Cloud Client
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

    try:
        hb = client.heartbeat()
        print(f"[Chroma] Heartbeat: {hb}")
    except Exception as e:
        print(f"[Chroma] Heartbeat not available or failed: {e}")

    # Store chunks
    vector_store, stats = store_chunks_in_chroma(
        chunks=chunks,
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
        batch_size=128,
        base_metadata={
            "source": os.path.relpath(pdf_path, base_dir),
            "chunk_method": "by_length",
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        id_prefix=os.path.basename(pdf_path).replace(".", "_"),
        verbose=True,
    )
    print(f"[Chroma] Stored {stats['added']} chunks in {stats['batches']} batch(es).")

    # Quick peek at first chunk
    try:
        sample = vector_store._collection.get(include=["documents", "metadatas", "embeddings"], limit=1)
        ids, docs, metas, embs = sample.get("ids", []), sample.get("documents", []), sample.get("metadatas", []), sample.get("embeddings", [])
        if len(ids) > 0 and len(docs) > 0 and len(metas) > 0 and len(embs) > 0:
            print("\n=== Chunk 1 from Chroma ===")
            print(f"- id: {ids[0]}")
            print(f"- metadata: {metas[0]}")
            print(f"- document (truncated): {docs[0][:300]}{'…' if len(docs[0]) > 300 else ''}")
            print(f"- embedding length: {len(embs[0])}")
            print(f"- embedding head: {embs[0][:10]}")
    except Exception as e:
        print(f"[Chroma] Sample fetch failed: {e}")

    # Optional semantic query
    query_text = os.getenv("RAG_QUERY")
    if query_text:
        query_chroma(vector_store, query_text, k=3, truncate=300)

    # Optional: ask Gemini with retrieved context
    gemini_query = os.getenv("GEMINI_QUERY") or query_text  # reuse RAG_QUERY if set
    if gemini_query:
        answer_with_gemini(vector_store, gemini_query, k=3, model_name=None, max_context_chars=8000)    
