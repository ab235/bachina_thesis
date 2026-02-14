import pathlib
import random
from typing import Dict, Tuple

from beir.datasets.data_loader import GenericDataLoader

from beir_eval import download_beir_dataset
from .sampling import sample_qids


def load_beir_dataset(
    dataset: str,
    data_dir: pathlib.Path,
    split: str,
    max_queries: int,
    max_corpus_docs: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    local_dataset = data_dir / dataset
    if local_dataset.exists() and local_dataset.is_dir():
        data_path = str(local_dataset)
    else:
        try:
            data_path = download_beir_dataset(dataset, data_dir)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {dataset} via BEIR downloader. "
                f"Your cached datasets/{dataset}.zip may be invalid, "
                f"or the current BEIR mirror does not host {dataset} at that URL. "
                f"Either provide an extracted local BEIR-format folder at datasets/{dataset} "
                "(corpus.jsonl, queries.jsonl, qrels/*.tsv), or run without that dataset."
            ) from exc
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    qids = sample_qids(list(queries.keys()), max_queries=max_queries, seed=seed)
    queries = {qid: queries[qid] for qid in qids}
    qrels = {qid: qrels[qid] for qid in qids if qid in qrels}

    rel_doc_ids = set()
    for qid in qrels:
        for did, rel in qrels[qid].items():
            if rel > 0:
                rel_doc_ids.add(did)

    if max_corpus_docs > 0:
        all_doc_ids = list(corpus.keys())
        rng = random.Random(seed)
        neg_pool = [d for d in all_doc_ids if d not in rel_doc_ids]
        target_neg = max(0, max_corpus_docs - len(rel_doc_ids))
        if len(neg_pool) > target_neg:
            neg_pool = rng.sample(neg_pool, target_neg)
        keep_docs = rel_doc_ids.union(neg_pool)
        corpus = {did: corpus[did] for did in keep_docs if did in corpus}
    return corpus, queries, qrels

