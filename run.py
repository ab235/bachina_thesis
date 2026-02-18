from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from chunking import build_late_token_pool_chunks, chunk_corpus_for_eval, chunk_text_for_eval, join_doc
from embeddings import (
    E5Embedder,
    LateTokenPoolEncoder,
    SBERTEmbedder,
    build_sentence_embed_fn,
    retrieve_bm25,
    retrieve_bm25_chunks,
)
from evaluation import (
    build_predicted_supporting_facts,
    compute_hotpot_support_fact_coverage,
    recall_at_k_from_top_chunks,
)
from generation import generate_answers_from_top_chunks
from metrics import remap_supporting_facts_to_titles, score_hotpot_predictions, score_squad_predictions
from preprocessing.sampling import sample_qids
from retrieve import (
    retrieve_dense,
    retrieve_dense_chunks,
    retrieve_dense_pooled,
    retrieve_dense_pooled_chunks,
)
from datatypes import RetrievalArtifacts, RetrieverContext


def topk_rank(scores: Dict[str, float], k: int) -> List[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def init_retriever_context(
    dataset_name: str,
    chunker: str,
    retriever: str,
    args: object,
) -> RetrieverContext:
    is_hotpot_dataset = dataset_name in {"hotpotqa_distractor", "hotpotqa_fullwiki"}
    shared_embedder = None
    dense_model_name = args.sbert_model
    if retriever == "sbert":
        shared_embedder = SBERTEmbedder(
            model_name=args.sbert_model,
            batch_size=args.batch_size,
            normalize=True,
        )
    elif retriever == "e5":
        dense_model_name = args.e5_model
        shared_embedder = E5Embedder(
            model_name=args.e5_model,
            batch_size=args.batch_size,
            normalize=True,
        )

    late_pool_encoder: Optional[LateTokenPoolEncoder] = None
    if chunker == "late_token_pool":
        late_pool_encoder = LateTokenPoolEncoder(
            model_name=dense_model_name,
            use_e5_format=(retriever == "e5"),
        )

    return RetrieverContext(
        shared_embedder=shared_embedder,
        dense_model_name=dense_model_name,
        late_pool_encoder=late_pool_encoder,
        do_hotpot_support_coverage=bool(
            args.hotpot_support_fact_coverage and is_hotpot_dataset
        ),
        do_hotpot_official_emf1=bool(
            args.hotpot_official_emf1 and is_hotpot_dataset
        ),
        need_raw_chunk_scores=True,
    )


def run_early_mode(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    chunker: str,
    retriever: str,
    args: object,
    ctx: RetrieverContext,
) -> RetrievalArtifacts:
    coverage_top_n = max(
        1,
        int(
            max(
                list(getattr(args, "k_values", [1]))
                + [getattr(args, "answer_recall_k", 1), getattr(args, "hotpot_answer_top_k", 1)]
            )
        ),
    )
    coverage_raw_chunks: Dict[str, Dict[str, float]] = {}
    coverage_chunk_texts: Dict[str, str] = {}
    coverage_chunk_to_doc: Dict[str, str] = {}

    if retriever in {"sbert", "e5"}:
        if chunker == "late_token_pool":
            if ctx.late_pool_encoder is None:
                raise RuntimeError("late_token_pool encoder was not initialized.")
            doc_texts = {doc_id: join_doc(doc) for doc_id, doc in corpus.items()}
            late_data = build_late_token_pool_chunks(
                doc_texts=doc_texts,
                encoder=ctx.late_pool_encoder,
                args=args,
                desc="Late token pool build (early)",
            )
            if late_data.truncated_docs:
                import logging

                logging.info(
                    "late_token_pool docs above model max length (windowed): %d",
                    late_data.truncated_docs,
                )
            results = retrieve_dense_pooled(
                queries=queries,
                chunk_vectors=late_data.chunk_vectors,
                chunk_to_doc=late_data.chunk_to_doc,
                encoder=ctx.late_pool_encoder,
            )
            num_chunks = len(late_data.chunk_texts)
            if ctx.need_raw_chunk_scores:
                coverage_raw_chunks = retrieve_dense_pooled_chunks(
                    queries=queries,
                    chunk_vectors=late_data.chunk_vectors,
                    chunk_to_doc=late_data.chunk_to_doc,
                    encoder=ctx.late_pool_encoder,
                    top_n=coverage_top_n,
                )
                coverage_chunk_texts = late_data.chunk_texts
                coverage_chunk_to_doc = late_data.chunk_to_doc
        else:
            sentence_embed_fn = build_sentence_embed_fn(args) if chunker == "semantic" else None
            chunk_texts, chunk_to_doc = chunk_corpus_for_eval(
                corpus,
                chunker=chunker,
                args=args,
                sentence_embed_fn=sentence_embed_fn,
            )
            results = retrieve_dense(
                queries=queries,
                chunk_texts=chunk_texts,
                chunk_to_doc=chunk_to_doc,
                sbert_model=ctx.dense_model_name,
                batch_size=args.batch_size,
                embedder=ctx.shared_embedder,
            )
            num_chunks = len(chunk_texts)
            if ctx.need_raw_chunk_scores:
                coverage_raw_chunks = retrieve_dense_chunks(
                    queries=queries,
                    chunk_texts=chunk_texts,
                    chunk_to_doc=chunk_to_doc,
                    sbert_model=ctx.dense_model_name,
                    batch_size=args.batch_size,
                    embedder=ctx.shared_embedder,
                    top_n=coverage_top_n,
                )
                coverage_chunk_texts = chunk_texts
                coverage_chunk_to_doc = chunk_to_doc
    elif retriever in {"bm25", "bm25s"}:
        if chunker == "late_token_pool":
            if ctx.late_pool_encoder is None:
                raise RuntimeError("late_token_pool encoder was not initialized.")
            doc_texts = {doc_id: join_doc(doc) for doc_id, doc in corpus.items()}
            late_data = build_late_token_pool_chunks(
                doc_texts=doc_texts,
                encoder=ctx.late_pool_encoder,
                args=args,
                desc="Late token pool build (early)",
            )
            if late_data.truncated_docs:
                import logging

                logging.info(
                    "late_token_pool docs above model max length (windowed): %d",
                    late_data.truncated_docs,
                )
            results = retrieve_bm25(
                queries=queries,
                chunk_texts=late_data.chunk_texts,
                chunk_to_doc=late_data.chunk_to_doc,
                k_values=args.k_values,
                size=max(args.k_values),
            )
            num_chunks = len(late_data.chunk_texts)
            if ctx.need_raw_chunk_scores:
                coverage_raw_chunks = retrieve_bm25_chunks(
                    queries=queries,
                    chunk_texts=late_data.chunk_texts,
                    chunk_to_doc=late_data.chunk_to_doc,
                    top_n=coverage_top_n,
                )
                coverage_chunk_texts = late_data.chunk_texts
                coverage_chunk_to_doc = late_data.chunk_to_doc
        else:
            sentence_embed_fn = build_sentence_embed_fn(args) if chunker == "semantic" else None
            chunk_texts, chunk_to_doc = chunk_corpus_for_eval(
                corpus,
                chunker=chunker,
                args=args,
                sentence_embed_fn=sentence_embed_fn,
            )
            results = retrieve_bm25(
                queries=queries,
                chunk_texts=chunk_texts,
                chunk_to_doc=chunk_to_doc,
                k_values=args.k_values,
                size=max(args.k_values),
            )
            num_chunks = len(chunk_texts)
            if ctx.need_raw_chunk_scores:
                coverage_raw_chunks = retrieve_bm25_chunks(
                    queries=queries,
                    chunk_texts=chunk_texts,
                    chunk_to_doc=chunk_to_doc,
                    top_n=coverage_top_n,
                )
                coverage_chunk_texts = chunk_texts
                coverage_chunk_to_doc = chunk_to_doc
    else:
        raise ValueError(f"Unsupported retriever: {retriever}")

    return RetrievalArtifacts(
        results=results,
        coverage_raw_chunks=coverage_raw_chunks,
        coverage_chunk_texts=coverage_chunk_texts,
        coverage_chunk_to_doc=coverage_chunk_to_doc,
        num_chunks=num_chunks,
    )


def run_hierarchical_mode(
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    chunker: str,
    retriever: str,
    args: object,
    ctx: RetrieverContext,
) -> RetrievalArtifacts:
    coverage_top_n = max(
        1,
        int(
            max(
                list(getattr(args, "k_values", [1]))
                + [getattr(args, "answer_recall_k", 1), getattr(args, "hotpot_answer_top_k", 1)]
            )
        ),
    )
    doc_texts = {doc_id: join_doc(doc) for doc_id, doc in corpus.items()}
    doc_id_map = {doc_id: doc_id for doc_id in doc_texts}

    if retriever in {"sbert", "e5"}:
        doc_results = retrieve_dense(
            queries=queries,
            chunk_texts=doc_texts,
            chunk_to_doc=doc_id_map,
            sbert_model=ctx.dense_model_name,
            batch_size=args.batch_size,
            embedder=ctx.shared_embedder,
        )
    elif retriever in {"bm25", "bm25s"}:
        doc_results = retrieve_bm25(
            queries=queries,
            chunk_texts=doc_texts,
            chunk_to_doc=doc_id_map,
            k_values=args.k_values,
            size=args.hierarchical_top_docs,
        )
    else:
        raise ValueError(f"Unsupported retriever: {retriever}")

    top_docs_by_qid: Dict[str, List[str]] = {
        qid: topk_rank(doc_results.get(qid, {}), args.hierarchical_top_docs)
        for qid in queries
    }
    needed_docs = {doc_id for docs in top_docs_by_qid.values() for doc_id in docs}

    late_chunk_texts: Dict[str, str] = {}
    late_chunk_to_doc: Dict[str, str] = {}
    late_chunk_vectors: Dict[str, np.ndarray] = {}
    if chunker == "late_token_pool":
        if ctx.late_pool_encoder is None:
            raise RuntimeError("late_token_pool encoder was not initialized.")
        sub_doc_texts = {doc_id: doc_texts.get(doc_id, "") for doc_id in needed_docs}
        late_data = build_late_token_pool_chunks(
            doc_texts=sub_doc_texts,
            encoder=ctx.late_pool_encoder,
            args=args,
            desc="Late token pool build (hierarchical)",
        )
        if late_data.truncated_docs:
            import logging

            logging.info(
                "late_token_pool docs above model max length (windowed): %d",
                late_data.truncated_docs,
            )
        late_chunk_texts = late_data.chunk_texts
        late_chunk_to_doc = late_data.chunk_to_doc
        late_chunk_vectors = late_data.chunk_vectors
    else:
        sentence_embed_fn = build_sentence_embed_fn(args) if chunker == "semantic" else None
        for doc_id in tqdm(
            needed_docs,
            total=len(needed_docs),
            desc=f"Hierarchical chunk build ({chunker})",
            leave=False,
        ):
            joined = doc_texts.get(doc_id, "")
            if not joined:
                continue
            chunks = chunk_text_for_eval(
                joined,
                chunker=chunker,
                args=args,
                sentence_embed_fn=sentence_embed_fn,
            )
            for idx, chunk in enumerate(chunks):
                cid = f"{doc_id}#chunk{idx}"
                late_chunk_texts[cid] = chunk
                late_chunk_to_doc[cid] = doc_id

    if retriever in {"sbert", "e5"}:
        if chunker == "late_token_pool":
            if ctx.late_pool_encoder is None:
                raise RuntimeError("late_token_pool encoder was not initialized.")
            results = retrieve_dense_pooled(
                queries=queries,
                chunk_vectors=late_chunk_vectors,
                chunk_to_doc=late_chunk_to_doc,
                encoder=ctx.late_pool_encoder,
                allowed_docs_by_qid=top_docs_by_qid,
            )
            coverage_raw_chunks = retrieve_dense_pooled_chunks(
                queries=queries,
                chunk_vectors=late_chunk_vectors,
                chunk_to_doc=late_chunk_to_doc,
                encoder=ctx.late_pool_encoder,
                allowed_docs_by_qid=top_docs_by_qid,
                top_n=coverage_top_n,
            )
        else:
            all_results = retrieve_dense(
                queries=queries,
                chunk_texts=late_chunk_texts,
                chunk_to_doc=late_chunk_to_doc,
                sbert_model=ctx.dense_model_name,
                batch_size=args.batch_size,
                embedder=ctx.shared_embedder,
            )
            results = {}
            for qid in queries:
                allowed = set(top_docs_by_qid.get(qid, []))
                if not allowed:
                    results[qid] = {}
                    continue
                results[qid] = {
                    did: score
                    for did, score in all_results.get(qid, {}).items()
                    if did in allowed
                }
            coverage_raw_chunks = retrieve_dense_chunks(
                queries=queries,
                chunk_texts=late_chunk_texts,
                chunk_to_doc=late_chunk_to_doc,
                sbert_model=ctx.dense_model_name,
                batch_size=args.batch_size,
                embedder=ctx.shared_embedder,
                allowed_docs_by_qid=top_docs_by_qid,
                top_n=coverage_top_n,
            )
    else:
        results = retrieve_bm25(
            queries=queries,
            chunk_texts=late_chunk_texts,
            chunk_to_doc=late_chunk_to_doc,
            k_values=args.k_values,
            allowed_docs_by_qid=top_docs_by_qid,
            size=max(args.k_values) * max(2, args.hierarchical_top_docs),
        )
        coverage_raw_chunks = retrieve_bm25_chunks(
            queries=queries,
            chunk_texts=late_chunk_texts,
            chunk_to_doc=late_chunk_to_doc,
            allowed_docs_by_qid=top_docs_by_qid,
            top_n=coverage_top_n,
        )

    return RetrievalArtifacts(
        results=results,
        coverage_raw_chunks=coverage_raw_chunks,
        coverage_chunk_texts=late_chunk_texts,
        coverage_chunk_to_doc=late_chunk_to_doc,
        num_chunks=len(late_chunk_texts),
    )


def compute_metrics_bundle(
    dataset_name: str,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    artifacts: RetrievalArtifacts,
    args: object,
    ctx: RetrieverContext,
    hotpot_gold_facts: Optional[Dict[str, Set[Tuple[str, int]]]],
    hotpot_doc_sentences: Optional[Dict[str, List[str]]],
    hotpot_answers: Optional[Dict[str, List[str]]],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "chunk_recall": recall_at_k_from_top_chunks(
            raw_chunk_results=artifacts.coverage_raw_chunks,
            chunk_texts=artifacts.coverage_chunk_texts,
            answers_by_qid=hotpot_answers or {},
            k=max(1, int(args.answer_recall_k)),
            min_answer_tokens=max(1, int(args.answer_match_min_tokens)),
        )
    }

    if ctx.do_hotpot_support_coverage and artifacts.coverage_raw_chunks and artifacts.coverage_chunk_to_doc:
        metrics.update(
            compute_hotpot_support_fact_coverage(
                raw_chunk_results=artifacts.coverage_raw_chunks,
                chunk_texts=artifacts.coverage_chunk_texts,
                chunk_to_doc=artifacts.coverage_chunk_to_doc,
                k_values=sorted(set(args.k_values)),
                hotpot_gold_facts=hotpot_gold_facts or {},
                hotpot_doc_sentences=hotpot_doc_sentences or {},
            )
        )
    if ctx.do_hotpot_official_emf1 and artifacts.coverage_raw_chunks and artifacts.coverage_chunk_to_doc:
        target_qids = [
            qid
            for qid in queries
            if qid in (hotpot_answers or {})
            and qid in artifacts.coverage_raw_chunks
            and (hotpot_gold_facts or {}).get(qid)
        ]
        if args.hotpot_answer_max_queries > 0 and len(target_qids) > args.hotpot_answer_max_queries:
            target_qids = sample_qids(
                target_qids,
                max_queries=args.hotpot_answer_max_queries,
                seed=args.seed,
            )

        if target_qids:
            predicted_answers = generate_answers_from_top_chunks(
                queries=queries,
                raw_chunk_results=artifacts.coverage_raw_chunks,
                chunk_texts=artifacts.coverage_chunk_texts,
                provider=args.answer_provider,
                model_family=args.hotpot_answer_model,
                top_k=max(1, args.hotpot_answer_top_k),
                seed=args.seed,
                qids=target_qids,
                bedrock_model_id=args.bedrock_model_id,
                bedrock_region=args.bedrock_region,
            )
            predicted_sp_doc = build_predicted_supporting_facts(
                raw_chunk_results=artifacts.coverage_raw_chunks,
                chunk_texts=artifacts.coverage_chunk_texts,
                chunk_to_doc=artifacts.coverage_chunk_to_doc,
                hotpot_doc_sentences=hotpot_doc_sentences or {},
                qids=target_qids,
                top_k=max(1, args.hotpot_answer_top_k),
                max_facts=max(1, args.hotpot_sp_max_facts),
            )
            doc_id_to_title = {doc_id: (doc.get("title", "") or doc_id) for doc_id, doc in corpus.items()}
            predicted_sp_title = remap_supporting_facts_to_titles(predicted_sp_doc, doc_id_to_title)
            gold_sp_title = remap_supporting_facts_to_titles(hotpot_gold_facts or {}, doc_id_to_title)
            gold_answers_for_scoring = {
                qid: (aliases[0] if aliases else "")
                for qid, aliases in (hotpot_answers or {}).items()
            }
            metrics["hotpot_official_emf1"] = score_hotpot_predictions(
                pred_answers=predicted_answers,
                pred_supporting_facts=predicted_sp_title,
                gold_answers=gold_answers_for_scoring,
                gold_supporting_facts=gold_sp_title,
                qids=target_qids,
            )
        else:
            metrics["hotpot_official_emf1"] = {"num_scored": 0}

    if dataset_name == "squad_v11" and artifacts.coverage_raw_chunks and artifacts.coverage_chunk_texts:
        target_qids = [
            qid
            for qid in queries
            if qid in (hotpot_answers or {})
            and qid in artifacts.coverage_raw_chunks
        ]
        if args.hotpot_answer_max_queries > 0 and len(target_qids) > args.hotpot_answer_max_queries:
            target_qids = sample_qids(
                target_qids,
                max_queries=args.hotpot_answer_max_queries,
                seed=args.seed,
            )
        if target_qids:
            predicted_answers = generate_answers_from_top_chunks(
                queries=queries,
                raw_chunk_results=artifacts.coverage_raw_chunks,
                chunk_texts=artifacts.coverage_chunk_texts,
                provider=args.answer_provider,
                model_family=args.hotpot_answer_model,
                top_k=max(1, args.hotpot_answer_top_k),
                seed=args.seed,
                qids=target_qids,
                bedrock_model_id=args.bedrock_model_id,
                bedrock_region=args.bedrock_region,
                prompt_style="squad",
            )
            metrics["squad_rag_generated_emf1"] = score_squad_predictions(
                pred_answers=predicted_answers,
                gold_answers=hotpot_answers or {},
                qids=target_qids,
            )
        else:
            metrics["squad_rag_generated_emf1"] = {
                "label": "SQuAD RAG-generated EM/F1 (official normalization)",
                "num_scored": 0,
            }
    return metrics


def build_result_row(
    dataset_name: str,
    chunking_mode: str,
    chunker: str,
    retriever: str,
    num_queries: int,
    num_docs: int,
    num_chunks: int,
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
    metrics: Dict[str, object],
    answer_provider: str,
    generator_model_family: str,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "chunking_mode": chunking_mode,
        "chunker": chunker,
        "retriever": retriever,
        "answer_provider": answer_provider,
        "generator_model_family": generator_model_family,
        "num_queries": num_queries,
        "num_docs": num_docs,
        "num_chunks": num_chunks,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(duration_seconds, 3),
    }
    row.update(metrics)
    return row


def evaluate_one(
    dataset_name: str,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    chunker: str,
    retriever: str,
    args: object,
    hotpot_gold_facts: Optional[Dict[str, Set[Tuple[str, int]]]] = None,
    hotpot_doc_sentences: Optional[Dict[str, List[str]]] = None,
    hotpot_answers: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    started_at = datetime.now(timezone.utc)
    t0 = perf_counter()
    ctx = init_retriever_context(
        dataset_name=dataset_name,
        chunker=chunker,
        retriever=retriever,
        args=args,
    )
    if args.chunking_mode == "early":
        artifacts = run_early_mode(
            corpus=corpus,
            queries=queries,
            chunker=chunker,
            retriever=retriever,
            args=args,
            ctx=ctx,
        )
    else:
        artifacts = run_hierarchical_mode(
            corpus=corpus,
            queries=queries,
            chunker=chunker,
            retriever=retriever,
            args=args,
            ctx=ctx,
        )

    metrics = compute_metrics_bundle(
        dataset_name=dataset_name,
        corpus=corpus,
        queries=queries,
        artifacts=artifacts,
        args=args,
        ctx=ctx,
        hotpot_gold_facts=hotpot_gold_facts,
        hotpot_doc_sentences=hotpot_doc_sentences,
        hotpot_answers=hotpot_answers,
    )
    finished_at = datetime.now(timezone.utc)
    return build_result_row(
        dataset_name=dataset_name,
        chunking_mode=args.chunking_mode,
        chunker=chunker,
        retriever=retriever,
        num_queries=len(queries),
        num_docs=len(corpus),
        num_chunks=artifacts.num_chunks,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=perf_counter() - t0,
        metrics=metrics,
        answer_provider=str(getattr(args, "answer_provider", "")),
        generator_model_family=str(getattr(args, "hotpot_answer_model", "")),
    )
