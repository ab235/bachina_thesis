from typing import List
import nltk
from tqdm import tqdm
import numpy as np

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from embeddings import embed_texts


def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    i = 0
    n = len(text)
    
    while i < n:
        j = min(i + max_chars, n)
        
        chunk = text[i:j]
        chunks.append(chunk)
        
        if j == n:
            break
        
        # Move start position forward for next chunk
        # Subtract overlap so chunks share some content
        # Ensure we always move forward (prevent infinite loop)
        i = j
        if j - overlap >= i:
            i = j - overlap
    return chunks



def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _carryover_overlap(sentences: List[str], overlap: int) -> List[str]:
    if overlap <= 0:
        return []

    selected = []
    total = 0

    for sentence in reversed(sentences):
        size = len(sentence) + (1 if selected else 0)
        selected.append(sentence)
        total += size
        if total >= overlap:
            break

    return list(reversed(selected))


def semantic_chunking(text: str, max_chars: int = 1200, overlap: int = 200):
    text = text.strip()
    if not text:
        return []

    chunks = []
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    embeddings = embed_texts(sentences)

    current_sentences: List[str] = []
    current_len = 0
    prev_embedding = None
    SIM_THRESHOLD = 0.8

    for sentence, emb in tqdm(
        zip(sentences, embeddings),
        total=len(sentences),
        desc="Chunking...",
    ):
        added_length = len(sentence) if not current_sentences else len(sentence) + 1
        similarity = (
            cosine_similarity(prev_embedding, emb) if prev_embedding is not None else None
        )

        should_split = False
        if (similarity is not None and similarity < SIM_THRESHOLD):
            should_split = True
        if current_len + added_length > max_chars:
            should_split = True

        if should_split and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

            carryover = _carryover_overlap(current_sentences, overlap)
            current_sentences = carryover
            current_len = len(" ".join(current_sentences)) if current_sentences else 0

        current_sentences.append(sentence)
        current_len += added_length if current_len else len(sentence)
        prev_embedding = emb

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
