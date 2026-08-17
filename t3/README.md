# RAG Pipeline Notebooks

A progression of Retrieval-Augmented Generation (RAG) implementations, from a dependency-free toy example to a full from-scratch pipeline over a Wikipedia-scale dataset, plus a configurable, reusable pipeline template.

## Contents

| Notebook | Complexity | Data | Embeddings | Vector Store | LLM |
|---|---|---|---|---|---|
| `simple_rag.ipynb` | Minimal / educational | 4 inline toy documents | Sentence-Transformers → TF-IDF fallback | NumPy cosine similarity | OpenAI → extractive fallback |
| `t3_rag.ipynb` | Full build-from-scratch | Wikipedia-style `a.parquet` (442,726 docs) | `all-MiniLM-L6-v2` | FAISS (`IndexFlatL2`) | TinyLlama-1.1B-Chat |
| `rag_pipeline.ipynb` / `rag_pipeline (1).ipynb` | Reusable pipeline / template | `a.parquet`, auto-detected text column | `all-MiniLM-L6-v2` (configurable) | FAISS `IndexFlatIP` (cosine via inner product) or plain cosine similarity | TinyLlama (local) or Claude (API) |
| `sample_doc.zip` | — | `sample_doc.txt` — a small sample text document for testing | — | — | — |

## `simple_rag.ipynb` — Minimal RAG, No Dependencies Required

A self-contained teaching notebook that runs end-to-end with **no API key and no heavy dependencies**, using graceful fallbacks at every stage.

**Flow:** `corpus → chunk → embed → store → retrieve (cosine similarity) → generate answer`

- **Corpus:** 4 short inline documents covering RAG concepts themselves (what RAG is, embeddings, chunking, vector search) — a nice self-referential demo
- **Chunking:** fixed word-window splitter (`chunk_size=60`, `overlap=15`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` if installed, otherwise falls back to TF-IDF (`scikit-learn`)
- **Retrieval:** plain NumPy cosine similarity, no vector DB
- **Generation:** OpenAI (`gpt-4o-mini`) if `OPENAI_API_KEY` is set, otherwise an **extractive fallback** that stitches together the top retrieved snippets — guarantees the notebook always produces output
- Includes a `Config` dataclass, an `ask()` convenience function, and worked example queries with printed sources and similarity scores

**Good for:** understanding the RAG concept end-to-end before adding infrastructure.

## `t3_rag.ipynb` — RAG From Scratch on Wikipedia Data

Builds every stage manually, then evaluates retrieval quality quantitatively.

**Pipeline:**
1. Load `a.parquet` (442,726 Wikipedia-style rows: `id`, `title`, `text`, `categories`), inspect text-length distribution
2. Combine `title` + `text` + `categories` into a single `document` field per row
3. **Chunking experiments:**
   - `CharacterTextSplitter` (naive, fails silently on documents with no matching separator — noted directly in the notebook)
   - `RecursiveCharacterTextSplitter` (750 chars, 150 overlap) — the one actually used going forward
4. **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` over chunks from the first 1,000 documents (~6,233 chunks)
5. **Indexing:** FAISS `IndexFlatL2` for exact nearest-neighbor search
6. **Retrieval:** a `search()` function tested against hand-picked queries (e.g. *"football clubs in Greece"*, *"Greek financial crisis"*), plus an interactive `input()`-driven query cell
7. **Generation:** local `TinyLlama/TinyLlama-1.1B-Chat-v1.0` via `transformers.pipeline`, prompted with retrieved context
8. **Evaluation:** a hand-built `eval_set` of (query, expected title) pairs, scored with:
   - `recall@k` for k = 1, 3, 5
   - `MRR` (Mean Reciprocal Rank)
9. **Chunk-size sweep:** re-runs the full embed → index → evaluate loop for chunk sizes `[200, 500, 750, 1000, 1500, 2000, 3000]`, comparing recall@k, MRR, chunk count, and embedding time — visualized in a 3-panel plot

**Good for:** understanding what each RAG component actually does, and how retrieval quality changes with chunking strategy.

## `rag_pipeline.ipynb` (and duplicate `rag_pipeline (1).ipynb`) — Reusable Pipeline

Two variants of a productionized, reusable RAG pipeline packaged as a documented notebook with a numbered table-of-contents structure.

**`rag_pipeline.ipynb`** (local-LLM variant):
- `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64 — tuned to fit typical embedding-model context)
- `all-MiniLM-L6-v2` embeddings, L2-normalized
- FAISS `IndexFlatIP` (inner product ≡ cosine similarity on normalized vectors)
- `search()` / `pretty_search()` helpers, a `build_prompt()` formatter, and a `rag(query, k, max_new_tokens)` function tying retrieval + generation together
- Local generation via `TinyLlama-1.1B-Chat` (swappable for any HuggingFace model)
- Interactive Q&A cell plus **FAISS index persistence** (saves/reloads the index and chunk list via `pickle` so re-embedding isn't needed on restart)
- Ends with a summary table of every component choice

**`rag_pipeline (1).ipynb`** (API-driven variant):
- Class-based `RAGPipeline` wrapping load → embed → retrieve → generate as reusable methods (`.query()`, `.semantic_search()`)
- Auto-detects the text column in any parquet file (looks for `text`/`content`/`body`/`desc`/`summary`/`doc` in column names, falls back to the first string column)
- Plain NumPy + `cosine_similarity` (scikit-learn) retrieval — no FAISS
- Generation via the **Anthropic API** (`claude-sonnet-4-20250514`), grounded with an explicit "answer only from context, cite chunks" system prompt
- Includes similarity-score visualizations (histogram of all scores, bar chart of top-k)

Both are designed to be edited at the top `CONFIG` cell and rerun end-to-end on a new dataset.

## Requirements

```
pandas
numpy
scikit-learn
sentence-transformers
faiss-cpu
langchain-text-splitters
transformers
torch
anthropic       # for the Claude-generation variant
openai          # optional, for simple_rag.ipynb
pyarrow         # for reading parquet
matplotlib      # for score/eval visualizations
```

## Data

- `a.parquet` — Wikipedia-style dataset (`id`, `title`, `text`, `categories`) used by `t3_rag.ipynb` and both `rag_pipeline` notebooks. Not included; point `PARQUET_PATH` / `DATA_PATH` at your own copy.
- `sample_doc.txt` (in `sample_doc.zip`) — a small standalone text file for quick smoke-testing a pipeline without loading the full parquet dataset.

## Suggested Reading Order

1. **`simple_rag.ipynb`** — grasp the concept with zero setup
2. **`t3_rag.ipynb`** — see every stage built by hand, plus how to measure retrieval quality
3. **`rag_pipeline.ipynb`** / **`rag_pipeline (1).ipynb`** — reusable, config-driven versions ready to point at a new dataset or swap in a different LLM (local vs. API)

## Notes

- Chunking strategy has a measurable effect on retrieval quality — `t3_rag.ipynb`'s chunk-size sweep is worth checking before picking a default.
- `IndexFlatIP` requires L2-normalized embeddings for the inner product to equal cosine similarity; `all-MiniLM-L6-v2` embeddings are normalized by default via `sentence-transformers`.
- The two `rag_pipeline` notebooks diverge mainly in the generation backend (local TinyLlama vs. Anthropic API) and vector store (FAISS vs. plain NumPy) — pick based on whether you need an API key—free / offline demo or production-quality answers.
