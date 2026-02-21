from typing import Callable, Dict, Iterable, List, Optional, Sequence, Any, Tuple
import nltk
nltk.download('punkt_tab')
from tqdm import tqdm
import numpy as np

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from embeddings import embed_texts
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datatypes import LateChunkData


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


def token_chunk_with_spans(
    text: str,
    target_size: int = 200,
    overlap: int = 0,
) -> List[Tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []
    tokens = word_tokenize(text)
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if overlap >= target_size:
        raise ValueError("overlap must be < target_size")
    if not tokens:
        return []

    token_starts: List[int] = []
    token_ends: List[int] = []
    cursor = 0
    for tok in tokens:
        pos = text.find(tok, cursor)
        if pos < 0:
            pos = text.find(tok)
        if pos < 0:
            pos = cursor
        token_starts.append(pos)
        end = pos + len(tok)
        token_ends.append(end)
        cursor = end

    out: List[Tuple[str, int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + target_size, n)
        chunk_text = " ".join(tokens[i:j]).strip()
        if chunk_text:
            out.append((chunk_text, token_starts[i], token_ends[j - 1]))
        if j == n:
            break
        i = j - overlap
    return out


def _enforce_min_chunks_by_chars(chunks: List[str], min_chars: int) -> List[str]:
    if min_chars <= 0:
        return [c for c in chunks if c and c.strip()]
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    if not cleaned:
        return []
    out: List[str] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        i += 1
        while len(cur) < min_chars and i < len(cleaned):
            cur = f"{cur} {cleaned[i]}".strip()
            i += 1
        out.append(cur)
    if len(out) >= 2 and len(out[-1]) < min_chars:
        out[-2] = f"{out[-2]} {out[-1]}".strip()
        out.pop()
    return out


def _enforce_min_chunks_by_tokens(chunks: List[str], min_tokens: int) -> List[str]:
    if min_tokens <= 0:
        return [c for c in chunks if c and c.strip()]
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    if not cleaned:
        return []
    out: List[str] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        i += 1
        while len(word_tokenize(cur)) < min_tokens and i < len(cleaned):
            cur = f"{cur} {cleaned[i]}".strip()
            i += 1
        out.append(cur)
    if len(out) >= 2 and len(word_tokenize(out[-1])) < min_tokens:
        out[-2] = f"{out[-2]} {out[-1]}".strip()
        out.pop()
    return out


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


def join_doc(doc: Dict[str, str]) -> str:
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return "\n\n".join([x for x in (title, text) if x]).strip()


ChunkerFn = Callable[[str, object, Optional[object]], List[str]]


def _chunk_token_eval(text: str, args: object, _sentence_embed_fn: Optional[object]) -> List[str]:
    chunks = token_chunk(text, target_size=args.token_size, overlap=args.overlap)
    min_tokens = max(1, int(args.token_size) // 2)
    return _enforce_min_chunks_by_tokens(chunks, min_tokens=min_tokens)


def _chunk_sentence_eval(text: str, args: object, _sentence_embed_fn: Optional[object]) -> List[str]:
    sentences = sentence_chunk(text)
    min_chars = max(1, int(args.min_chars))
    max_chars = max(min_chars, int(args.max_chars))
    overlap = max(0, int(getattr(args, "char_overlap", args.overlap)))

    if not sentences:
        return []

    # Greedily pack adjacent sentences up to max_chars, then carry a tail
    # overlap into the next chunk.
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current_sentences else 0)
        if current_sentences and current_len + add_len > max_chars:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_sentences = _carryover_overlap(current_sentences, overlap)
            current_len = len(" ".join(current_sentences)) if current_sentences else 0

        add_len = len(sentence) + (1 if current_sentences else 0)
        if not current_sentences or current_len + add_len <= max_chars:
            current_sentences.append(sentence)
            current_len += add_len
        else:
            # Very long single sentence fallback.
            chunks.append(sentence)
            current_sentences = []
            current_len = 0

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return _enforce_min_chunks_by_chars(chunks, min_chars=min_chars)


def _chunk_recursive_eval(text: str, args: object, _sentence_embed_fn: Optional[object]) -> List[str]:
    min_chars = max(1, int(args.min_chars))
    return recursive_chunk(
        text,
        min_chars=min_chars,
        max_chars=args.max_chars,
        overlap=max(0, int(getattr(args, "char_overlap", args.overlap))),
    )


def _chunk_semantic_eval(text: str, args: object, sentence_embed_fn: Optional[object]) -> List[str]:
    chunks = semantic_chunking(
        text,
        max_chars=args.max_chars,
        overlap=max(0, int(getattr(args, "char_overlap", args.overlap))),
        similarity_threshold=args.similarity_threshold,
        embed_fn=sentence_embed_fn,
        show_progress=False,
    )
    min_chars = max(1, int(args.min_chars))
    return _enforce_min_chunks_by_chars(chunks, min_chars=min_chars)


def get_chunker_factory() -> Dict[str, ChunkerFn]:
    return {
        "token": _chunk_token_eval,
        "sentence": _chunk_sentence_eval,
        "recursive": _chunk_recursive_eval,
        "semantic": _chunk_semantic_eval,
    }


def chunk_text_for_eval(
    text: str,
    chunker: str,
    args: object,
    sentence_embed_fn: Optional[object] = None,
) -> List[str]:
    chunker_impl = get_chunker_factory().get(chunker)
    if chunker_impl is None:
        raise ValueError(f"Unsupported chunker: {chunker}")
    return chunker_impl(text, args, sentence_embed_fn)


def build_late_token_pool_chunks(
    doc_texts: Dict[str, str],
    encoder: object,
    args: object,
    desc: str,
) -> LateChunkData:
    chunk_texts: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    chunk_spans: Dict[str, Tuple[int, int]] = {}
    chunk_vectors: Dict[str, np.ndarray] = {}
    truncated_docs = 0
    for doc_id, joined in tqdm(doc_texts.items(), desc=desc, leave=False):
        if not joined:
            continue
        chunks, vecs, spans, truncated = encoder.build_doc_chunks(
            joined,
            target_size=args.token_size,
            overlap=args.overlap,
            min_size=max(1, int(args.token_size) // 2),
        )
        if truncated:
            truncated_docs += 1
        for idx, (chunk_text_value, vec, span) in enumerate(zip(chunks, vecs, spans)):
            cid = f"{doc_id}#chunk{idx}"
            chunk_texts[cid] = chunk_text_value
            chunk_to_doc[cid] = doc_id
            chunk_spans[cid] = span
            chunk_vectors[cid] = vec
    return LateChunkData(
        chunk_texts=chunk_texts,
        chunk_to_doc=chunk_to_doc,
        chunk_spans=chunk_spans,
        chunk_vectors=chunk_vectors,
        truncated_docs=truncated_docs,
    )


def chunk_corpus_for_eval_with_spans(
    corpus: Dict[str, Dict[str, str]],
    chunker: str,
    args: object,
    sentence_embed_fn: Optional[object] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Tuple[int, int]]]:
    chunk_texts: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    chunk_spans: Dict[str, Tuple[int, int]] = {}
    for doc_id, doc in tqdm(corpus.items(), desc=f"Chunking ({chunker})", leave=False):
        joined = join_doc(doc)
        if not joined:
            continue
        if chunker == "token":
            chunks_with_spans = token_chunk_with_spans(
                joined,
                target_size=args.token_size,
                overlap=args.overlap,
            )
            min_tokens = max(1, int(args.token_size) // 2)
            merged: List[Tuple[str, int, int]] = []
            i = 0
            while i < len(chunks_with_spans):
                cur_text, cur_start, cur_end = chunks_with_spans[i]
                i += 1
                while len(word_tokenize(cur_text)) < min_tokens and i < len(chunks_with_spans):
                    nxt_text, _nxt_start, nxt_end = chunks_with_spans[i]
                    cur_text = f"{cur_text} {nxt_text}".strip()
                    cur_end = nxt_end
                    i += 1
                merged.append((cur_text, cur_start, cur_end))
            if len(merged) >= 2 and len(word_tokenize(merged[-1][0])) < min_tokens:
                prev_text, prev_start, _prev_end = merged[-2]
                last_text, _last_start, last_end = merged[-1]
                merged[-2] = (f"{prev_text} {last_text}".strip(), prev_start, last_end)
                merged.pop()
            chunks = [c for c, _s, _e in merged]
            spans = [(s, e) for _c, s, e in merged]
        else:
            chunks = chunk_text_for_eval(joined, chunker=chunker, args=args, sentence_embed_fn=sentence_embed_fn)
            spans = []
        cursor = 0
        for i, c in enumerate(chunks):
            cid = f"{doc_id}#chunk{i}"
            chunk_texts[cid] = c
            chunk_to_doc[cid] = doc_id
            if i < len(spans):
                chunk_spans[cid] = spans[i]
                continue
            start = joined.find(c, max(0, cursor - 256))
            if start < 0:
                start = joined.find(c)
            if start < 0:
                continue
            end = start + len(c)
            cursor = start
            chunk_spans[cid] = (start, end)
    return chunk_texts, chunk_to_doc, chunk_spans


def chunk_corpus_for_eval(
    corpus: Dict[str, Dict[str, str]],
    chunker: str,
    args: object,
    sentence_embed_fn: Optional[object] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    chunk_texts: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    for doc_id, doc in tqdm(corpus.items(), desc=f"Chunking ({chunker})", leave=False):
        joined = join_doc(doc)
        if not joined:
            continue
        chunks = chunk_text_for_eval(joined, chunker=chunker, args=args, sentence_embed_fn=sentence_embed_fn)
        for i, c in enumerate(chunks):
            cid = f"{doc_id}#chunk{i}"
            chunk_texts[cid] = c
            chunk_to_doc[cid] = doc_id
    return chunk_texts, chunk_to_doc
