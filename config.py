import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

OPENAI_GRADER_MODEL = os.getenv("OPENAI_GRADER_MODEL", OPENAI_CHAT_MODEL)

PGHOST = os.getenv("PGHOST", "localhost")

PGPORT = int(os.getenv("PGPORT", "5432"))

PGDATABASE = os.getenv("PGDATABASE", "handbook_rag")

PGUSER = os.getenv("PGUSER", "postgres")

PGPASSWORD = os.getenv("PGPASSWORD", "postgres")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "6"))

def validate_config():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    if not PGHOST or not PGDATABASE or not PGUSER:
        raise RuntimeError("Database connection not made. Error in PGHOST.")

validate_config()

def get_db_url() -> str:
    return f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
