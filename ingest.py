import argparse
from sqlalchemy.orm import Session
from tqdm import tqdm

from config import CHUNK_SIZE, CHUNK_OVERLAP
from db import SessionLocal, ensure_db
from models import create_all, Document, Chunk
from chunking import simple_chunk
from embeddings import embed_texts

def main():
    parser = argparse.ArgumentParser(description="Ingest a handbook .txt into pgvector.")
    parser.add_argument("txt_path", type=str, help="Path to the .txt handbook")
    parser.add_argument("--doc-title", type=str, default="Student Handbook", help="Title to tag this document")
    args = parser.parse_args()

    ensure_db()
    create_all()

    with open(args.txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = simple_chunk(raw, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No text found to ingest.")

    embeddings = []
    # Embed in batches to handle very large handbooks
    BATCH = 64
    for i in tqdm(range(0, len(chunks), BATCH), desc="Embedding"):
        batch = chunks[i:i+BATCH]
        embeddings.extend(embed_texts(batch))

    with SessionLocal() as session:
        doc = Document(title=args.doc_title)
        session.add(doc)
        session.flush()  # get doc.id

        to_add = []
        for idx, (content, emb) in enumerate(zip(chunks, embeddings), start=1):
            to_add.append(Chunk(document_id=doc.id, ordinal=idx, content=content, embedding=emb))
        session.add_all(to_add)
        session.commit()

    print(f"Ingested {len(chunks)} chunks into document '{args.doc_title}'.")

if __name__ == "__main__":
    main()
