"""
config.py
----------
Configuration module for the Handbook RAG project.
Loads environment variables and defines constants for
database connections, OpenAI models, and retrieval parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# === PostgreSQL / pgvector ===
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDATABASE = os.getenv("PGDATABASE", "handbook_rag")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")

# === RAG parameters ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))

# === Sanity checks ===
def validate_config():
    """Raise errors if critical configuration values are missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("❌ OPENAI_API_KEY is not set. Add it to your .env file.")
    if not PGHOST or not PGDATABASE or not PGUSER:
        raise RuntimeError("❌ Database connection details are incomplete in .env.")

# Run validation when imported
validate_config()

# Optional: construct Postgres connection string
def get_db_url() -> str:
    """Return SQLAlchemy-compatible Postgres connection URL."""
    return f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"