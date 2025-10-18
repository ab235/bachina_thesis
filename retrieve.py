from typing import List, Tuple
from sqlalchemy import text
from db import SessionLocal
from embeddings import embed_texts
from config import TOP_K

def retrieve_similar(question: str, top_k: int = TOP_K) -> List[Tuple[int, str, int, float]]:
    """
    Returns list of (document_id, content, ordinal, score) where lower score = closer (cosine distance).
    """
    [q_vec] = embed_texts([question])

    sql = text("""
        SELECT c.document_id, c.content, c.ordinal, (c.embedding <-> :qvec) AS score
        FROM chunks c
        ORDER BY c.embedding <-> :qvec
        LIMIT :k
    """)

    with SessionLocal().begin() as conn:
        rows = conn.execute(sql, {"qvec": q_vec, "k": top_k}).fetchall()

    # rows are Row objects; convert to python types
    return [(r[0], r[1], r[2], float(r[3])) for r in rows]
