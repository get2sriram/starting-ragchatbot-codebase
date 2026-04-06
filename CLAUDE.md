# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system for querying course materials. Users upload/ingest course documents and ask questions; the system retrieves relevant content via semantic search (ChromaDB) and generates AI-powered responses.

## Key Commands

```bash
# Install dependencies
uv sync

# Start the app (manual)
cd backend && uv run uvicorn app:app --reload --port 8000

# Quick start (if run.sh exists and is executable)
chmod +x run.sh && ./run.sh
```

- **Web UI**: http://localhost:8000
- **API docs**: http://localhost:8000/docs

## Architecture

### Backend (`backend/`)

FastAPI application with the following modules:

| File | Purpose |
|------|---------|
| `app.py` | FastAPI entry point, API endpoints (`POST /api/query`, `GET /api/courses`), static file serving |
| `config.py` | Configuration via dataclass, loads `.env` for API keys (Anthropic, OpenRouter). Uses `sentence-transformers` for local embeddings, ChromaDB for vector storage |
| `rag_system.py` | Main orchestrator: `DocumentProcessor` → `VectorStore` → `AIGenerator`. Manages tool-based search via `ToolManager` |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (text chunks). Uses `SentenceTransformerEmbeddingFunction` |
| `ai_generator.py` | Dual provider: Anthropic SDK (native) or OpenAI SDK (OpenRouter, detected via `base_url` param). Supports tool calling in both formats |
| `openrouter_embeddings.py` | Custom embedding function for OpenRouter. Currently unused — `VectorStore` uses local embeddings by default |
| `search_tools.py` | Tool abstractions — `CourseSearchTool` wraps `VectorStore.search()` |
| `document_processor.py` | Parses course documents (PDF, DOCX, TXT), extracts course/lesson metadata, creates `CourseChunk` objects |
| `session_manager.py` | In-memory conversation history management |
| `models.py` | Pydantic data models: `Course`, `Lesson`, `CourseChunk` |

### Data Flow

1. Courses are indexed from `../docs` on app startup
2. Query arrives → `RAGSystem.query()` → `AIGenerator.generate_response()`
3. AIGenerator may invoke `CourseSearchTool` to retrieve relevant chunks from ChromaDB
4. Results synthesized into a response

### Key Design Notes

- **Embeddings**: Uses local `SentenceTransformer` (`all-MiniLM-L6-v2`) by default, not remote APIs
- **AI Provider**: OpenRouter (`base_url` detected) vs direct Anthropic — tool format is auto-converted between Anthropic and OpenAI schemas
- **Sessions**: In-memory only, no persistent conversation storage
