"""Central configuration for the RAG travel & visa chatbot.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
INDEX_DIR = DATA_DIR / "index"
EVAL_FILE = DATA_DIR / "eval" / "queries.json"

INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"

SUPPORTED_SUFFIXES = (".pdf", ".html", ".htm")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0.2
CONDENSE_TEMPERATURE = 0.0

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 3072
EMBEDDING_BATCH_SIZE = 100

DOCUMENT_TITLE_FALLBACK = "none"

TOKENIZER_ENCODING = "cl100k_base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.65

MEMORY_TOKEN_BUDGET = 2000


QUERY_TEMPLATE = "task: question answering | query: {text}"
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"


def query_embedding_text(text: str) -> str:
    return QUERY_TEMPLATE.format(text=text)


def document_embedding_text(title: str | None, text: str) -> str:
    return DOCUMENT_TEMPLATE.format(title=title or DOCUMENT_TITLE_FALLBACK, text=text)
