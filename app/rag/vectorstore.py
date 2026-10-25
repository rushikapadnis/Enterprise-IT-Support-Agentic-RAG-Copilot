import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings

load_dotenv()

settings = get_settings()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = 384

_embeddings = None
_vectorstore = None


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Return the known vector dimension for the configured embedding model."""
    name = (
        model_name or EMBEDDING_MODEL
    ).strip().lower()

    if "all-minilm-l6-v2" in name or name == "all-minilm-l6-v2":
        return EMBEDDING_DIMENSION

    raise ValueError(
        f"Unsupported embedding model: {name}. "
        "This configuration expects "
        "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."
    )


def get_embeddings():
    """Load the embedding model locally; no OpenAI API is used."""
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def ensure_index():
    """Create the Pinecone index if needed and validate its dimensions."""
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    index_name = settings.pinecone_index_name
    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_names = {
        item["name"] for item in pc.list_indexes()
    }

    if index_name in existing_names:
        index_info = pc.describe_index(index_name)
        current_dimension = getattr(index_info, "dimension", None)

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone index '{index_name}' has dimension "
                f"{current_dimension}, but '{EMBEDDING_MODEL}' requires "
                f"{desired_dimension}. Set PINECONE_INDEX_NAME to a new "
                "index name. The existing index has not been deleted."
            )
    else:
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )

        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:
            index_info = pc.describe_index(index_name)
            status = getattr(index_info, "status", None)

            if status is None and isinstance(index_info, dict):
                status = index_info.get("status", {})

            if isinstance(status, dict) and status.get("ready"):
                break

            time.sleep(2)
        else:
            raise TimeoutError(
                f"Pinecone index '{index_name}' did not become ready "
                "within 180 seconds."
            )

    return pc.Index(index_name)


def get_vectorstore():
    """Return the shared Pinecone vector store."""
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    """Embed and add document chunks to the configured Pinecone namespace."""
    if not chunks:
        return []

    return get_vectorstore().add_documents(chunks)