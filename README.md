# RAG Explorer

An interactive Streamlit application that makes every stage of a retrieval-augmented generation pipeline visible.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first embedding run downloads the `all-MiniLM-L6-v2` sentence-transformer model. Uploaded documents, embeddings, and the Chroma collection are held in memory and are not persisted.

## Smart Retrieval

Smart Retrieval is the application's only search flow. It detects common question intents with deterministic Python rules, expands the retrieval query without an LLM, retrieves 15 semantic candidates from the selected vector engine, and locally reranks the best 5 for answerability. Gemini receives only those final five results.

For multi-part questions, it deterministically splits interrogative clauses, retrieves eight candidates per sub-query, deduplicates chunks, and selects a coverage-first evidence set so each question part is represented where possible.

During generation, each sub-question is handled independently. Coverage is determined by deterministic inspection of chunk text—not candidate count or similarity alone. Supported parts use direct evidence, partial parts answer only the verifiable portion, and unsupported parts receive a precise per-part insufficiency response without a Gemini call or outside-knowledge inference.

The reranking formula is:

```text
final_score = 0.80 × normalized_semantic_score + 0.20 × answerability_score
```

## Gemini answer generation

Generation uses `gemini-3.5-flash` through the official `google-genai` SDK. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder `GEMINI_API_KEY`. Never commit the real secrets file.

### Local test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Replace the placeholder GEMINI_API_KEY in the copied secrets file.
streamlit run app.py
```

### Streamlit Community Cloud

Push the repository to GitHub, create an app at `share.streamlit.io`, select `app.py`, and paste the values from `.streamlit/secrets.toml.example`—with the real API key—in Advanced settings → Secrets. Do not commit `.streamlit/secrets.toml`.

## Architecture

- `parser.py`: PDF extraction with PyMuPDF
- `chunker.py`: fixed-size chunks with overlap and source metadata
- `embedder.py`: normalized sentence-transformer embeddings
- `retriever.py`: observable brute-force cosine retrieval
- `smart_retriever.py`: deterministic intent detection, query transformation, and local answerability reranking
- `vector_store.py`: brute-force and ChromaDB adapters
- `generator.py`: replaceable grounded answer-generation seam
- `components.py`: all Streamlit presentation functions
- `utils.py`: shared immutable data models
