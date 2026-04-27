"""Settings for RAG NXP. Single LLM backend: local Ollama."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Filesystem layout
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    pdf_filename: str = "document.pdf"
    chroma_dir: Path = project_root / "chroma_db"

    # Embedding + retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking strategy. One of:
    #   "fixed"    - legacy character-window with overlap (per page).
    #   "logical"  - structure-aware (Article/Section/Amendment/headings) +
    #                sentence-aware packing. Default.
    #   "semantic" - "logical" + extra cuts at embedding-detected topic shifts.
    chunking_strategy: str = "logical"
    # Used by "fixed". Char-based.
    chunk_size: int = 900
    chunk_overlap: int = 150
    # Used by "logical" and "semantic".
    chunk_size_target: int = 900
    chunk_size_max: int = 1400
    chunk_overlap_sentences: int = 1
    # Used by "semantic" only. Higher percentile = fewer extra cuts.
    semantic_breakpoint_percentile: int = 90

    retrieval_k: int = 5

    # Ollama (local LLM). Default URL works on the host machine. Inside Docker,
    # set OLLAMA_BASE_URL=http://ollama:11434 (docker-compose) or
    # http://host.docker.internal:11434 (Ollama running on the host).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_s: float = 180.0
    ollama_num_predict: int = 512
    # How long Ollama should keep the model loaded in memory between requests.
    # "-1" or a duration string ("30m", "24h"). Keeping the model warm avoids
    # paying the cold-load cost on every query — the single biggest latency win
    # on CPU-only machines.
    ollama_keep_alive: str = "30m"


settings = Settings()
