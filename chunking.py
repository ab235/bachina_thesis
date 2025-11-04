from typing import List

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
