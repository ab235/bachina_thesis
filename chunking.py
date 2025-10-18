from typing import List

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple, robust character-based chunker with overlap.
    Works fine for .txt handbooks; you can swap in a token-aware chunker later.
    """
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
        i = j - overlap if j - overlap > i else j
    return chunks
