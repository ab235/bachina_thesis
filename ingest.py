import argparse
from sqlalchemy.orm import Session
from tqdm import tqdm  # Progress bar library for visual feedback

from config import CHUNK_SIZE, CHUNK_OVERLAP
from db import SessionLocal, ensure_db
from models import create_all, Document, Chunk
from chunking import simple_chunk, semantic_chunking
from embeddings import embed_texts

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a text document into the RAG system.",
        epilog="Example: python ingest.py handbook.txt --doc-title 'Student Handbook 2024'"
    )
    parser.add_argument(
        "txt_path", 
        type=str, 
        help="Path to the .txt file to ingest (e.g., handbook.txt)"
    )
    parser.add_argument(
        "--doc-title", 
        type=str, 
        default="Student Handbook", 
        help="Descriptive title for this document (default: 'Student Handbook')"
    )
    args = parser.parse_args()

    print("📦 Setting up database...")
    ensure_db()
    create_all()

    print(f"Reading file: {args.txt_path}")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    print(f"Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = simple_chunk(raw, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    if not chunks:
        raise RuntimeError("No chunky.")
    
    print(f"Created {len(chunks)} chunks")
    print("Generating embeddings")
    embeddings = []
    BATCH = 64
    
    for i in tqdm(range(0, len(chunks), BATCH), desc="Embedding"):
        batch = chunks[i:i+BATCH]
        embeddings.extend(embed_texts(batch))

    print("Saving to database")
    with SessionLocal() as session:
        doc = Document(title=args.doc_title)
        session.add(doc)
        
        session.flush()

        to_add = []
        for idx, (content, emb) in enumerate(zip(chunks, embeddings), start=1):
            to_add.append(Chunk(
                document_id=doc.id,
                ordinal=idx,
                content=content,
                embedding=emb
            ))
        
        session.add_all(to_add)
        
        session.commit()

    print(f"Successfully ingested {len(chunks)} chunks into document '{args.doc_title}'")

if __name__ == "__main__":
    main()
