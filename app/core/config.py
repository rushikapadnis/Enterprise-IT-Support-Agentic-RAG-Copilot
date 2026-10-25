import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env", override=False)


class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # Groq LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"

    # External search
    tavily_api_key: str = ""

    # Pinecone vector database
    pinecone_api_key: str = ""
    pinecone_index_name: str = "enterprise-it-support-minilm"
    pinecone_namespace: str = "company-it-kb"
    pinecone_region: str = "us-east-1"

    # Local Hugging Face embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval and workflow
    top_k: int = 4
    max_retries: int = 1

    # Application security and storage
    admin_api_key: str = "change-me"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()