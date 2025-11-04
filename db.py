from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String
from config import PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, get_db_url

DB_URL = get_db_url()

engine = create_engine(DB_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

def ensure_db():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
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
            embedding vector(1536)
        );
        """))
        
        conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_chunks_embedding'
            ) THEN
                CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            END IF;
        END
        $$;
        """))
