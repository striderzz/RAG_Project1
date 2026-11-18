# RAG Document QA System: interview prep

## 30-second pitch

I built a retrieval-augmented question-answering pipeline for PDFs in Python. It extracts the text with PyPDF2, splits
it into 500-character chunks with a 100-character overlap, embeds them with the MiniLM sentence-transformer and stores
them in Chroma Cloud. For a question, it retrieves the top 3 chunks and has Gemini answer using only that context, with
each passage tagged by source and chunk index. If the answer is not in the document, the model is told to say so
instead of guessing.

## 2-minute walkthrough

1. **Problem.** I wanted answers about a specific document that are grounded in that document, not in the model's
   general knowledge. The test input is a synthetic 100-page PDF with one fictional employee profile per page.
2. **Ingestion.** `extract_text_from_pdf` reads each page with PyPDF2 and joins the text. `chunk_text` cuts it into
   500-character windows that overlap by 100 characters, so a fact sitting on a boundary appears whole in at least one
   chunk. The function validates the size and overlap.
3. **Indexing.** `store_chunks_in_chroma` deletes and recreates the `pdf_chunks` collection, then writes chunks in
   batches of 128 through LangChain's Chroma wrapper. LangChain computes `all-MiniLM-L6-v2` embeddings locally. Each
   chunk gets a stable ID built from the file name and index, plus metadata: source path, chunk method, size, overlap
   and chunk index.
4. **Retrieval.** `similarity_search` with `k=3` returns the nearest chunks. `query_chroma` prints them with their
   metadata as a preview.
5. **Generation.** `answer_with_gemini` builds a context block from the hits, labels each with `source` and
   `chunk_index`, caps it at 8000 characters and sends a prompt with explicit grounding rules to Gemini
   (`gemini-1.5-flash` by default, configurable).
6. **Config and safety.** All keys come from the environment through python-dotenv; `.env` is git-ignored and
   `.env.example` documents the variables.
7. **What I would do next.** Separate ingestion from querying, add page-aware chunking with page citations, add a
   retrieval evaluation set and put a small web UI on top.

## Questions and answers

### Overview

**1. What does the project do, in one sentence?**
It answers questions about a PDF by retrieving the 3 most relevant text chunks from a Chroma vector store and having
Gemini answer only from those chunks.

**2. What are the five stages?**
Extract (PyPDF2), chunk (500/100 characters), embed and store (MiniLM into Chroma Cloud), retrieve (top-3 similarity
search) and answer (Gemini with a grounded prompt).

**3. Why RAG instead of pasting the whole PDF into the prompt?**
A 100-page PDF is a lot of tokens for every question, most of which are irrelevant. Retrieval sends only the relevant
passages, which keeps the prompt small and makes it easy to see which text the answer came from.

**4. What is in the sample data?**
`files/test_pdf.pdf` is synthetic: 100 pages, one fictional employee per page with name, age, title, salary, hobbies,
height and weight. All values are dummy data made for testing retrieval.

**5. How do you run it?**
Set `CHROMA_API_KEY`, `CHROMA_TENANT` and `GOOGLE_API_KEY` in `.env`, set `RAG_QUERY` to a question and run
`python app.py`. It prints the chunk count, the Chroma batches, a sample stored chunk, the top 3 hits and Gemini's
answer.

### Design decisions

**6. Why 500-character chunks with 100 characters of overlap?**
The sample PDF has short, self-contained profiles, so 500 characters keeps one chunk close to one topic. The overlap
protects facts that would otherwise be split across two chunks. Both values are variables at the bottom of `app.py`
so they can be tuned. I have not measured retrieval quality across settings yet; that would be the next step.

**7. Why characters and not tokens or sentences?**
Character windows are deterministic, need no tokenizer and are easy to reason about. The cost is that a chunk can end
mid-word or straddle two profiles. Sentence- or page-aware chunking is on the roadmap.

**8. Why delete the collection on every run?**
So results never mix with stale data from an older version of the PDF or different chunk settings. It is simple and
safe for a single-document demo, but it means every run re-indexes. At scale I would upsert by stable ID instead.

**9. Why stable IDs like `test_pdf_pdf-0`?**
Deterministic IDs make writes idempotent and make a chunk easy to find again when debugging a bad answer.

**10. What metadata do you store, and why?**
`source`, `chunk_method`, `chunk_size`, `overlap` and `chunk_index`. Source and index let the prompt and the reader
trace an answer back to a chunk. The chunking settings record how the index was built.

**11. Why `k=3`?**
For the sample data an answer usually lives in one profile, so a few chunks are enough context while keeping the
prompt short. It is a parameter on both retrieval functions.

**12. How do you stop Gemini from making things up?**
The prompt says to use only the provided context, to reply "I couldn't find that in the provided documents" when the
answer is missing, to prefer exact numbers and names, and to flag disagreements between snippets. That reduces
hallucination but does not guarantee it; there is no automated check of the answer against the context yet.

**13. What is the 8000-character context budget for?**
It is a soft cap on how much retrieved text goes into the prompt, so a larger `k` or larger chunks cannot blow up the
request size. Snippets are truncated once the budget is used.

### PyPDF2

**14. What are PyPDF2's limits here?**
It only reads embedded text. Scanned or image-only pages return nothing and are skipped, and complex layouts such as
tables can come out in an odd order. pdfplumber, PyMuPDF or an OCR step would handle those cases.

**15. Where would you add page numbers?**
In `extract_text_from_pdf`: keep a list of (page number, text) instead of one joined string, chunk per page, and add
`page` to each chunk's metadata.

### Embeddings and Chroma

**16. Why `all-MiniLM-L6-v2`?**
It is small and fast, runs locally on CPU and costs nothing per call. The trade-off is that larger or hosted embedding
models may retrieve better on harder text.

**17. Where are the embeddings computed?**
Locally, in the Python process, by `HuggingFaceEmbeddings`. Chroma Cloud only stores the vectors and runs the search.
The query is embedded with the same model, which matters: query and documents must share one embedding space.

**18. Why Chroma Cloud rather than FAISS or a local Chroma?**
Chroma Cloud stores documents, vectors and metadata together and needs no server to run. FAISS is an in-process index
without a metadata store. A local `chromadb.PersistentClient` would be the offline option and is listed as a next step.

**19. What does LangChain add on top of `chromadb`?**
The `Chroma` wrapper takes the embedding function, so `add_texts` and `similarity_search` embed and query in one call.
The raw client is still used for `delete_collection`, `heartbeat` and the sample `get`.

**20. Why write in batches of 128?**
To keep each request to Chroma Cloud a reasonable size and to show progress per batch, instead of one large request
that fails all at once.

### Gemini

**21. How is Gemini configured?**
`genai.configure(api_key=GOOGLE_API_KEY)` and `GenerativeModel(GEMINI_MODEL)`, with `gemini-1.5-flash` as the default.
The function reads `resp.text` and falls back to the first candidate's first part if `text` is missing.

**22. What happens if the key is missing?**
`answer_with_gemini` raises a `RuntimeError` with a clear message. Missing Chroma credentials also raise before any
work starts.

### Scaling and failure modes

**23. What breaks first with many documents?**
The full re-index on every run. Ingestion should be its own step that only processes new or changed files, with a
document ID in the metadata so queries can filter by document.

**24. What if the answer spans more than 3 chunks?**
Gemini only sees 3 chunks, so it may give a partial answer or say it could not find it. Options are a larger `k`,
bigger chunks, a re-ranker over a larger candidate set, or query rewriting.

**25. How would you measure quality?**
Build a labelled set of questions with the expected chunk, then report hit rate and MRR for retrieval and a groundedness
check for answers. Current numbers: hit rate —, MRR —, answer accuracy — (not yet measured).

**26. What are the main failure modes in production?**
Chroma Cloud or Gemini being unreachable or rate-limited, scanned PDFs producing no text, and weak matches reaching the
prompt because there is no similarity threshold. The Chroma heartbeat is logged but not enforced, and there are no
retries around the Gemini call.

### Debugging scenarios

**27. Gemini keeps answering "I couldn't find that". How do you debug it?**
First look at the `query_chroma` output: are the right chunks in the top 3? If not, it is a retrieval problem (chunk
size, overlap, `k`, embedding model). If the right chunk is there, check whether the fact was cut at a chunk boundary
or truncated by the context budget, then look at the prompt.

**28. The run says 0 chunks. What happened?**
`extract_text_from_pdf` returned empty text, most likely because the PDF is scanned images. Check
`page.extract_text()` on one page; if it is empty, the document needs OCR.

**29. The answer mixes facts from two different employees. Why?**
A 500-character window can straddle the end of one profile and the start of the next, so one chunk contains two
people. Page-aware chunking fixes this for one-profile-per-page documents.

### Behavioral (STAR outlines)

**30. Tell me about a design decision you would revisit.**
- Situation: the pipeline indexes and answers in one script.
- Task: keep the demo simple and avoid stale data.
- Action: rebuild the collection on every run, with stable IDs and metadata.
- Result: results are always consistent with the current PDF and settings; the cost is a full re-index per question,
  which is why separating ingestion is first on my list.

**31. Tell me about making an LLM output trustworthy.**
- Situation: a plain LLM answers confidently even when it does not know.
- Task: make answers traceable to the document.
- Action: retrieval first, a strict "context only" prompt with a fixed fallback sentence, and source plus chunk index
  on every snippet.
- Result: answers can be checked against the printed chunks; groundedness rate — (not yet measured).

**32. Tell me about handling secrets in a project.**
- Situation: the pipeline needs Chroma and Gemini keys.
- Task: keep them out of the repository.
- Action: read them from the environment with python-dotenv, git-ignore `.env`, ship a `.env.example` and fail fast
  with a clear error when a key is missing.
- Result: the repository contains no keys and setup is documented in the README.

## Resume bullet templates

- Built a 5-stage RAG pipeline in Python: PDF ingestion with PyPDF2, 500/100-character overlapping chunks, MiniLM
  embeddings and ChromaDB (Chroma Cloud) storage with source and chunk metadata.
- Used Gemini for context-grounded question answering over top-3 semantic retrieval, with a prompt that restricts
  answers to retrieved passages and returns a fixed fallback when the answer is missing.
- Measured retrieval on a labelled question set: hit rate — and MRR — across — chunking settings (fill in after an
  evaluation run).
