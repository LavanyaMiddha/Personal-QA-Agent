# 🤖 Personal QA Agent

A Retrieval-Augmented Generation (RAG) system that lets you query your own PDF documents using natural language. Built with LangChain, Pinecone, and HuggingFace embeddings, this project serves as a research platform for exploring and comparing different RAG strategies, chunking techniques, and retrieval methods.

---

## Overview

The Personal QA Agent ingests PDF documents, chunks and embeds them into a Pinecone vector store, and retrieves relevant context to answer user queries. The project is designed not just as a working QA tool, but as an experimental framework for understanding what makes RAG pipelines performant and robust.

---

## Tech Stack

| Layer            | Technology                                           |
| ---------------- | ---------------------------------------------------- |
| Document Loading | PyMuPDF                                              |
| Text Splitting   | LangChain `RecursiveCharacterTextSplitter`           |
| Embeddings       | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store     | Pinecone (Serverless)                                |
| Retrieval        | LangChain `PineconeVectorStore`                      |
| Orchestration    | LangChain                                            |
| API              | FastAPI _(planned)_                                  |
| Frontend         | React.js _(planned)_                                 |
| Environment      | Python 3.11+, dotenv                                 |

---

## Project Structure

```
personal-qa-agent/
├── rag/
│   ├── data_loader.py       # PDF ingestion and text splitting
│   ├── embeddings.py        # Embedding model wrapper
│   ├── vector_store.py      # Pinecone index management and document upsert
│   └── retriever.py         # Retrieval logic (similarity, MMR)
├── data/
│   └── pdf_files/           # Place your PDF documents here
├── .env                     # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/personal-qa-agent.git
cd personal-qa-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

### 4. Add your PDFs

Place your PDF files in the `data/pdf_files/` directory.

### 5. Ingest documents

```bash
python rag/vector_store.py
```

### 6. Run a query

```bash
python rag/retriever.py
```

---

## Roadmap

### Retrieval Strategies

Comparing how different retrieval approaches affect answer quality:

- Similarity search vs MMR (Maximal Marginal Relevance)
- Top-k sensitivity analysis
- Re-ranking retrieved chunks before passing to LLM

### Chunking Strategies

Exploring how document splitting affects retrieval performance:

- Fixed size vs recursive vs semantic chunking
- Chunk size and overlap sensitivity
- Sentence-level vs paragraph-level splitting

### Memory and Context Position

Investigating how context placement affects LLM answer quality:

- Comparing prepended vs appended context
- Lost-in-the-middle analysis for long context windows
- Conversational memory with multi-turn QA

### Answer Robustness Evaluation

Building an evaluation framework to measure answer quality:

- Faithfulness, relevance, and groundedness metrics
- RAGAS integration for automated evaluation
- Hallucination detection

### LangChain Agent Integration

Moving from a simple retrieval chain to an agentic architecture:

- Tool-calling agent with retrieval as a tool
- Multi-step reasoning over documents
- Self-correction and query rewriting

### Hybrid RAG

Combining vector search with keyword search for better recall:

- BM25 + semantic search fusion
- Metadata filtering (by source, date, topic)
- Pinecone sparse-dense hybrid index

### Confidence Scores

Adding transparency to retrieved results:

- Cosine similarity scores alongside retrieved chunks
- Confidence thresholding to avoid low-quality answers
- Score normalization and display in the frontend

### Frontend (FastAPI + React.js)

Building a full-stack interface:

- FastAPI backend serving the RAG pipeline
- React.js chat interface with source citations
- Document upload and management UI

---

## Contributing

This is a personal research project, but feedback and suggestions are welcome. Feel free to open an issue or submit a pull request.

---

## License

MIT License
