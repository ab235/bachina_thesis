from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String
from config import PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

DB_URL = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

engine = create_engine(DB_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

# Minimal model here for table creation; full ORM lives in models.py but this ensures extensions/indexes.
def ensure_db():
    with engine.begin() as conn:
        # Enable pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Create tables and vector index if missing
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(3072) -- matches text-embedding-3-large dimensionality
        );
        """))
        # HNSW or IVFFLAT (requires Postgres 16+ for HNSW extension; IVFFLAT is common).
        # We'll use ivfflat with cosine distance:
        conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_chunks_embedding'
            ) THEN
                CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            END IF;
        END
        $$;
        """))
