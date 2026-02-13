from typing import Callable, Iterable, List, Optional, Sequence, Any
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm
import numpy as np

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from embeddings import embed_texts
from langchain_text_splitters import RecursiveCharacterTextSplitter


def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    
    # Sliding window over raw characters with optional overlap.
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


def token_chunk(
    text: str,
    target_size: int = 200,
    overlap: int = 0,
    tokenize_fn: Optional[Callable[[str], Sequence[Any]]] = None,
    detokenize_fn: Optional[Callable[[Sequence[Any]], str]] = None,
) -> List[str]:
    """
    Fixed-size token chunking.
    Defaults to NLTK word tokens, but can use model-aware tokenizers
    via tokenize_fn/detokenize_fn.
    """
    text = text.strip()
    if not text:
        return []
    # Tokenize once, then slice the token list with optional overlap.
    tokens = list(tokenize_fn(text)) if tokenize_fn is not None else word_tokenize(text)
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if overlap >= target_size:
        raise ValueError("overlap must be < target_size")
    chunks: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + target_size, n)
        chunk_tokens = tokens[i:j]
        if detokenize_fn is not None:
            chunk_text = detokenize_fn(chunk_tokens).strip()
        else:
            chunk_text = " ".join(str(tok) for tok in chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if j == n:
            break
        i = j - overlap
    return chunks


def sentence_chunk(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Each sentence is its own chunk.
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _merge_segments(segments: Iterable[str], min_chars: int, max_chars: int,) -> List[str]:
    '''
    For remerging segments that are below a minimum length
    '''
    # Greedily merge segments into chunks up to max_chars.
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_len = len(seg) + (1 if current else 0)#no chunk is above max_chars
        if current and current_len + seg_len > max_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [seg]
            current_len = len(seg)
            continue
        current.append(seg)
        current_len += seg_len if current_len else len(seg)

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    if min_chars <= 0:
        return chunks

    # Merge tiny tail chunks when possible to satisfy min_chars.
    merged: List[str] = []
    buffer = ""
    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue
        if len(buffer) < min_chars and len(buffer) + 1 + len(chunk) <= max_chars:
            buffer = f"{buffer} {chunk}"
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    return merged


def recursive_chunk(
    text: str,
    min_chars: int = 200,
    max_chars: int = 1200,
    overlap: int = 0,
) -> List[str]:
    """
    Recursive splitting via LangChain's RecursiveCharacterTextSplitter.
    """
    text = text.strip()
    if not text:
        return []

    # Prefer larger boundaries first (paragraphs -> lines -> sentences).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return _merge_segments(chunks, min_chars=min_chars, max_chars=max_chars)#ensure each chunk is between min_chars and max_chars



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


def semantic_chunking(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
    similarity_threshold: float = 0.6,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    show_progress: bool = False,
):
    text = text.strip()
    if not text:
        return []

    chunks = []
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Embed each sentence so we can merge by semantic similarity.
    embedding_function = embed_fn or embed_texts
    embeddings = embedding_function(sentences)

    current_sentences: List[str] = []
    current_len = 0
    prev_embedding = None
    iterator = zip(sentences, embeddings)
    if show_progress:
        iterator = tqdm(iterator, total=len(sentences), desc="Chunking...")
    for sentence, emb in iterator:
        added_length = len(sentence) + 1
        if prev_embedding is None:
            prev_embedding = emb
            current_sentences.append(sentence)
            current_len += added_length
            continue
        # Similarity vs. previous sentence controls semantic boundaries.
        similarity = (
            cosine_similarity(prev_embedding, emb) if prev_embedding is not None else None
        )

        should_split = False
        if (similarity is not None and similarity < similarity_threshold):
            should_split = True
        if current_len + added_length > max_chars:
            should_split = True

        if should_split and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

            carryover = _carryover_overlap(current_sentences, overlap)
            current_sentences = carryover
            if current_sentences:
                current_len = len(" ".join(current_sentences))
            else:
                current_len = 0

        current_sentences.append(sentence)
        current_len += added_length
        prev_embedding = emb

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks
