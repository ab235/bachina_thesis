from typing import List, Tuple
from sqlalchemy import text
from db import SessionLocal
from embeddings import embed_texts
from config import TOP_K

def retrieve_similar(question: str, top_k: int = TOP_K) -> List[Tuple[int, str, int, float]]:
    [q_vec] = embed_texts([question])
    sql = text("""
        SELECT c.document_id, c.content, c.ordinal, (c.embedding <-> CAST(:qvec AS vector)) AS score
        FROM chunks c
        ORDER BY c.embedding <-> CAST(:qvec AS vector)
        LIMIT :k
    """)
    with SessionLocal() as session:
        rows = session.execute(sql, {"qvec": q_vec, "k": top_k}).fetchall()
    return [(r[0], r[1], r[2], float(r[3])) for r in rows]
