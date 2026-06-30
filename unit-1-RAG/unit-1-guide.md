# Unit 1: Embeddings, Vector DBs & RAG with Hybrid Search, Advanced Parsing, and Eval Harnesses

## Progress Summary
- **Current step:** Build the ingestion pipeline
- **Last updated:** 2026-06-30
- **Notes/blockers:** Python package is named `ragu` (not `unit_1`). Langfuse smoke test at `tests/langfuse_smoke_test.py`. Corpus: 8 research PDFs on AI, cognitive offloading, and student learning (in `data/`). First parse validated: `parsed/Knowledge about neuroscience doesnt protect teachers from myths.md`.

## Main Topics
* Embeddings & Cosine Similarity
* Top-k Retrieval & Citation Grounding
* Vector Databases (Qdrant)
* Chunking Strategies & Metadata Filtering
* Hybrid Search (BM25 + Dense Vector + RRF)
* Document Parsing (LlamaParse)
* Reranking (Cohere Rerank / Cross-encoders)
* Evaluation Frameworks (Ragas)
* LLM Observability & Tracing (Langfuse)

## Short Description of Goals
Build a robust semantic retrieval system over a folder of technical PDFs or markdown notes. Ingest documents using a structural parser like LlamaParse to preserve tables and layout. Experiment with chunk size and overlap — see how it changes retrieval quality. Combine dense vector embeddings with keyword search (BM25) using Reciprocal Rank Fusion (RRF) for hybrid search. Build a CLI that answers questions and cites its sources. Add a reranker (Cohere Rerank or a cross-encoder) and measure the difference. Instrument with Langfuse from day one — log every query, retrieved chunks, and final answer as a trace. Write 20–30 question/answer pairs by hand over your ingested documents to act as your ground truth test set. Run them through Ragas to get faithfulness, answer relevancy, and context recall scores automatically. Use these scores to compare chunking strategies and reranker variants quantitatively, keeping this test set alive to score every change made. Understand embeddings, cosine similarity, top-k retrieval, citation grounding, metadata filtering, hybrid search, eval-driven development, and numerical retrieval quality metrics.

---

## Concrete Project (2 Hours/Day Practice)

### Project: Technical Documentation RAG System & Eval Harness
Build a Python pipeline that ingests a folder of technical PDFs or engineering notes, indexes them into a vector database, and answers questions with cited sources — evaluated rigorously with Ragas and Langfuse.

* **Why it fits the timeframe:** It relies on pre-built SDKs and CLI outputs. It forces you to deal with noisy source documents and mathematically evaluate your architecture without writing a frontend.

* **What you will do:**

  **Week 1 — Ingestion, retrieval, and eval foundations**

  - [x] **Scaffold the project.** Create a `ragu/` Python package with a virtual environment, `requirements.txt`, and a `.env` for API keys. Add a `data/` folder for source documents and an `eval/` folder for your test set.
     > **Theory:** Treat this like any backend service: separate config (`.env`) from code, pin dependencies, and keep raw data out of git. You'll be iterating on retrieval logic constantly — a clean layout saves hours later.
     > **Trade-offs:** Monorepo (`unit-1-RAG/ragu/`) vs separate repos — use a monorepo here so Unit 2 can import your RAG code directly (e.g. `from ragu.search import ...`). `uv` or `poetry` vs plain `pip` — either is fine; pick one and move on.

  - [x] **Set up Langfuse.** Create a Langfuse account, install the SDK, and write a 10-line script that logs a dummy trace. Confirm it appears in the Langfuse UI before building anything else.
     > **Theory:** Observability for AI systems is closer to distributed tracing (Jaeger, OpenTelemetry) than to traditional logging. A *trace* is one end-to-end request; *spans* are sub-steps within it. Logging retrieval inputs/outputs from day one means you can debug bad answers without re-running experiments blind.
     > **Trade-offs:** Langfuse vs LangSmith vs Phoenix — all solve a similar problem. Langfuse is open-source and self-hostable; LangSmith is tightly integrated with LangChain. Pick one and use it consistently across all 4 units.

  - [x] **Curate your document corpus.** Collect 5–10 technical PDFs or markdown files (course notes, API docs, engineering blog posts). Pick material you can actually answer questions about — you'll write eval questions from this.
     > **Theory:** RAG quality has a ceiling set by your source data — "garbage in, garbage out" applies literally. Your eval set can only be as good as the documents you choose. Heterogeneous formats (PDFs + markdown) stress-test your parser pipeline early.
     > **Trade-offs:** Small curated corpus (5–10 docs you know well) vs large messy dump — start small. You need to hand-write eval questions, which requires actually reading the material.

  - [x] **Parse your first document.** Use `LlamaParse` to convert one PDF to structured Markdown. Inspect the output manually: are tables preserved? Are headings intact? Note what breaks before scaling up.
     > **Theory:** PDFs are a presentation format, not a storage format — text extraction is genuinely hard. Layout-aware parsers use vision models or heuristics to reconstruct structure (headings, tables, columns). Bad parsing silently corrupts chunks downstream; no amount of embedding tuning fixes a table that became a word salad.
     > **Trade-offs:** LlamaParse (cloud API, high quality, costs money) vs Marker (open-source, self-hosted, GPU-hungry) vs PyMuPDF/pdfplumber (free, fast, layout-naive). For technical docs with tables, layout-aware parsing is worth the cost. For clean markdown, skip parsing entirely.

  - [ ] **Build the ingestion pipeline.** Write a script that walks your `data/` folder, parses every file, and saves clean Markdown to a `parsed/` directory. Attach metadata to each document: `source_file`, `page_number`, `section_title`.
     > **Theory:** Ingestion is an ETL pipeline — Extract (parse), Transform (clean/chunk), Load (embed + index). Metadata isn't optional decoration; it powers citation grounding ("answer came from `api-docs.pdf`, page 12") and filtered retrieval ("only search `section_title: Authentication`").
     > **Trade-offs:** Re-parse on every run vs cache parsed output — cache to `parsed/` so you don't re-bill LlamaParse during retrieval experiments. Store metadata in the vector DB payload, not just in a sidecar file, so filters work at query time.

  - [ ] **Implement chunking.** Split parsed text into chunks with configurable `chunk_size` and `chunk_overlap` (start with 512 / 64). Store chunk text plus metadata. Print a few chunks to verify boundaries look sensible.
     > **Theory:** Embeddings encode *a fixed amount of text* into a single vector. Too large a chunk → the vector averages over unrelated topics (diluted semantics). Too small → you lose context ("it" no longer refers to anything). Overlap ensures sentences at chunk boundaries aren't orphaned. Cosine similarity then finds chunks whose *meaning* is closest to the query vector — but meaning is only as good as the chunk boundary.
     > **Trade-offs:** Fixed-size chunking (simple, predictable) vs semantic/recursive splitting (respects paragraph/heading boundaries, harder to implement) vs document-structure chunking (split on `##` headings — often the best free option for markdown). Start fixed at 512 tokens; later try heading-based and compare on your eval set. Rule of thumb: one chunk ≈ one idea.

  - [ ] **Embed and index into Qdrant.** Run chunks through an embedding model (OpenAI `text-embedding-3-small` or similar), upsert vectors + metadata into a local Qdrant instance (Docker is fine). Write a `search(query, top_k=5)` function that returns ranked chunks.
     > **Theory:** Embedding models are *bi-encoders*: query and document are embedded independently, then compared via cosine similarity. This is fast (pre-compute document vectors) but approximate — the model never "sees" query and document together. Vector DBs store millions of these pre-computed vectors and return the nearest neighbors in milliseconds via approximate nearest neighbor (ANN) indexes (HNSW in Qdrant).
     > **Trade-offs:** Qdrant vs Chroma vs pgvector — Qdrant has strong hybrid search and metadata filtering built in; Chroma is simpler to start; pgvector makes sense if you already run Postgres. For embedding models: OpenAI `text-embedding-3-small` (cheap, good general quality) vs Cohere Embed v3 (better multilingual) vs open-source `bge-small` (free, self-hosted, slightly lower quality). API embeddings are fine for learning; switch to local only if cost or privacy demands it.

  - [ ] **Write your eval set (start small).** Hand-write 5 question/answer pairs where the answer is explicitly grounded in your documents. Save as JSON: `{ "question", "expected_answer", "relevant_source" }`. Expand to 25 by end of week 1.
     > **Theory:** This is your *golden test set* — the single most valuable artifact in the project. Hand-written Q&A pairs anchor quantitative evaluation: without them, you're guessing whether changes help. Include *hard* questions: multi-hop (answer spans two sections), exact-match (specific numbers, API names), and paraphrased (question uses different words than the source).
     > **Trade-offs:** Hand-written (high quality, labor-intensive) vs LLM-generated eval sets (fast, but the LLM may ask questions that are too easy or hallucinate expected answers). Always hand-write at least the first 10. Synthetic expansion is fine after you have a quality seed set.

  **Week 2 — Hybrid search, reranking, generation, and measurement**

  - [ ] **Build a retrieval-only CLI.** A script that takes a question, runs `search()`, and prints the top-5 chunks with source citations. No LLM yet — just verify retrieval quality by eye against your eval questions.
     > **Theory:** Separating retrieval evaluation from generation is a core SWE discipline. If the right chunks aren't in the top-5, no LLM prompt engineering will save you. This CLI is your retrieval unit test — run it every time you change chunking, embeddings, or search strategy.
     > **Trade-offs:** Manual inspection (fast feedback, subjective) vs automated recall@k on your eval set (objective, requires labeling which chunks are relevant per question). Do both: eyeball for the first 5 questions, automate as you scale to 25.

  - [ ] **Add BM25 + RRF hybrid search.** Build a BM25 inverted index over the same chunks (e.g. `rank_bm25`). Merge BM25 and dense results with Reciprocal Rank Fusion. Compare retrieval on 5 eval questions: dense-only vs hybrid.
      > **Theory:** Dense vectors excel at semantic similarity ("error handling" matches "exception management") but miss exact terms (product names, error codes, function signatures). BM25 is a sparse keyword method — the classic TF-IDF successor — that rewards exact token overlap. RRF merges ranked lists without needing score normalization: `score = Σ 1/(k + rank)` across methods. Hybrid search matters most for technical docs with jargon, acronyms, and proper nouns.
      > **Trade-offs:** RRF (simple, parameter-free, works well) vs weighted linear combination (tunable but needs a dev set to tune weights) vs cross-encoder-only (accurate but too slow for first-stage retrieval). Add BM25 when dense-only misses questions containing specific identifiers; skip it if your docs are prose-heavy and semantic.

  - [ ] **Add a reranker.** Pipe hybrid top-10 results through `Cohere Rerank` (or a local cross-encoder) and keep the best 3. Log before/after rankings in Langfuse. Re-run your 5-question comparison.
      > **Theory:** Bi-encoders (step 7) embed query and document *separately* — fast but each side is computed in isolation. Cross-encoders (rerankers) feed query + document *together* through the model, producing a much more accurate relevance score. The standard pattern: bi-encoder retrieves top-50 cheaply, cross-encoder reranks to top-3 accurately. This is a latency/cost trade-off you'll see everywhere in production RAG.
      > **Trade-offs:** Cohere Rerank API (excellent quality, per-call cost) vs local cross-encoder like `ms-marco-MiniLM` (free, GPU needed, slightly lower quality) vs no reranker (cheapest, fine for simple doc sets). Reranking is usually worth it when top-10 from hybrid still contains irrelevant chunks.

  - [ ] **Wire up the full RAG loop.** Add an LLM generation step: retrieve → rerank → prompt with context → answer with inline citations. Log the full trace (query, chunks, prompt, answer) to Langfuse. Run all 25 eval questions through `Ragas` (faithfulness, context recall, answer relevancy). Save scores to a `results/` file. This is your baseline — every future change gets compared against it.
      > **Theory:** RAG generation is "stuff retrieved context into a prompt, ask the LLM to answer using only that context." Citations are what make RAG trustworthy — they let users verify claims. Ragas automates what you'd otherwise eyeball: *faithfulness* (is the answer supported by the retrieved chunks?), *context recall* (did retrieval find the right chunks?), *answer relevancy* (does the answer address the question?). This baseline JSON file is your regression test suite for the rest of the curriculum.
      > **Trade-offs:** Stuff-all-chunks-in-prompt (simple, hits context limits) vs map-reduce / iterative refinement (handles more context, complex). Single LLM call is correct for now. For the generator: GPT-4o-mini (cheap, good enough for eval) vs Claude/GPT-4o (better citation adherence, higher cost). Ragas vs custom eval scripts — Ragas gives standard metrics comparable across experiments; custom scripts are fine if Ragas setup fights you, but standardization pays off in Unit 4 when you compare fine-tuned models.
