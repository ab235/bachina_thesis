from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RetrieverContext:
    shared_embedder: Optional[object]
    dense_model_name: str
    late_pool_encoder: Optional[object]
    do_hotpot_support_coverage: bool
    do_hotpot_official_emf1: bool
    need_raw_chunk_scores: bool


@dataclass
class RetrievalArtifacts:
    results: Dict[str, Dict[str, float]]
    coverage_raw_chunks: Dict[str, Dict[str, float]]
    coverage_chunk_texts: Dict[str, str]
    coverage_chunk_to_doc: Dict[str, str]
    num_chunks: int


@dataclass
class EvaluationInputs:
    dataset_name: str
    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    chunker: str
    retriever: str
    args: object
    hotpot_gold_facts: Optional[Dict[str, set]]
    hotpot_doc_sentences: Optional[Dict[str, List[str]]]
    hotpot_answers: Optional[Dict[str, str]]


@dataclass
class LateChunkData:
    chunk_texts: Dict[str, str]
    chunk_to_doc: Dict[str, str]
    chunk_vectors: Dict[str, np.ndarray]
    truncated_docs: int
