# RAG Retriever Pipeline

A simple RAG system using **LangChain + Ollama + PostgreSQL/pgvector + BM25**.

## Architecture

```text
                INDEXING
                   
Documents
   ↓
Load & Split
   ↓
Chunks
   ├──────────────→ chunks.pkl → BM25
   │
   ↓
Ollama Embeddings
   ↓
PostgreSQL + pgvector
```

```text
                 QUERYING

User Question
      ↓
 ┌────┴─────┐
 ↓          ↓
Vector     BM25
Retriever  Retriever
 └────┬─────┘
      ↓
Ensemble Retriever
      ↓
RAG Prompt
      ↓
Ollama LLM
      ↓
Answer
```

## Project Structure

```text
RAG_Retrievers/
├── app/
│   ├── ingestion.py
│   ├── vectorstore.py
│   ├── retrievers.py
│   ├── prompts.py
│   ├── pipeline.py
│   ├── index.py
│   └── main.py
│
├── data/
│   ├── documents/
│   └── chunks.pkl
│
├── .env
├── requirements.txt
└── README.md
```

## Responsibilities

| File             | Responsibility                      |
| ---------------- | ----------------------------------- |
| `ingestion.py`   | Load and chunk documents            |
| `vectorstore.py` | Create/load PostgreSQL vector store |
| `retrievers.py`  | Vector + BM25 + Ensemble            |
| `prompts.py`     | RAG prompt                          |
| `pipeline.py`    | RAG/LLM chain                       |
| `index.py`       | Build the index                     |
| `main.py`        | Ask questions                       |

## Commands

Build/rebuild the index:

```bash
python -m app.index
```

Run the RAG application:

```bash
python -m app.main
```

### Key idea

```text
index.py → expensive indexing
main.py  → retrieval + generation
```

Run `index.py` only when your documents change.
