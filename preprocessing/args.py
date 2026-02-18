import argparse
import os
import pathlib
from config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    EVAL_ANSWER_PROVIDER,
    EVAL_ANSWER_MATCH_MIN_TOKENS,
    EVAL_BACKEND,
    EVAL_BATCH_SIZE,
    EVAL_CHUNKERS,
    EVAL_CHUNKING_MODE,
    EVAL_DATASET_PATH_MODE1,
    EVAL_DATASET_PATH_MODE2,
    EVAL_DATASET_PATH_MODE3,
    EVAL_E5_MODEL,
    EVAL_HIERARCHICAL_TOP_DOCS,
    EVAL_HOTPOT_ANSWER_MODEL,
    EVAL_HOTPOT_OFFICIAL_EMF1,
    EVAL_HOTPOT_SP_MAX_FACTS,
    EVAL_HOTPOT_SUPPORT_FACT_COVERAGE,
    EVAL_JOB_COUNT,
    EVAL_JOB_INDEX,
    EVAL_K,
    EVAL_MAX_CHARS,
    EVAL_MAX_QUERIES,
    EVAL_MIN_CHARS,
    EVAL_MODE,
    EVAL_MODEL,
    EVAL_NORMALIZE,
    EVAL_OUTPUT,
    EVAL_OVERLAP,
    EVAL_RETRIEVERS,
    EVAL_SBERT_MODEL,
    EVAL_SEED,
    EVAL_SIMILARITY_THRESHOLD,
    EVAL_TOKEN_SIZE,
    EVAL_GPU_ID,
    EVAL_WIKI_CORPUS_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunking evaluation runner."
    )
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=EVAL_MODE)
    parser.add_argument(
        "--dataset-path-mode1",
        type=pathlib.Path,
        default=pathlib.Path(EVAL_DATASET_PATH_MODE1),
        help="Dataset path for mode 1 (official SQuAD v1.1 JSON).",
    )
    parser.add_argument(
        "--dataset-path-mode2",
        type=pathlib.Path,
        default=pathlib.Path(EVAL_DATASET_PATH_MODE2),
        help="Dataset path for mode 2 (HotpotQA distractor).",
    )
    parser.add_argument(
        "--dataset-path-mode3",
        type=pathlib.Path,
        default=pathlib.Path(EVAL_DATASET_PATH_MODE3),
        help="Dataset path for mode 3 (HotpotQA fullwiki questions).",
    )
    parser.add_argument(
        "--wiki-corpus-path",
        type=pathlib.Path,
        default=pathlib.Path(EVAL_WIKI_CORPUS_PATH),
        help="Global Wikipedia corpus path for fullwiki mode (JSON/JSONL file or shard directory).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=EVAL_K,
        help="Top-k for retrieval/evaluation.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=EVAL_MAX_QUERIES,
        help="Max number of queries to evaluate. Set <=0 to use all queries.",
    )
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--token-size", type=int, default=EVAL_TOKEN_SIZE)
    parser.add_argument("--overlap", type=int, default=EVAL_OVERLAP)
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=EVAL_CHUNKERS,
        choices=["token", "sentence", "recursive", "semantic", "late_token_pool"],
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=EVAL_RETRIEVERS,
        choices=["sbert", "e5", "bm25s"],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(EVAL_OUTPUT),
    )
    parser.add_argument(
        "--answer-provider",
        choices=["ollama", "bedrock"],
        default=EVAL_ANSWER_PROVIDER,
        help="Answer generation provider for Hotpot EM/F1 scoring.",
    )
    parser.add_argument(
        "--hotpot-answer-model",
        choices=["llama", "mistral", "qwen"],
        default=EVAL_HOTPOT_ANSWER_MODEL,
        help="Model family for answer generation.",
    )
    parser.add_argument(
        "--bedrock-model-id",
        type=str,
        default=BEDROCK_MODEL_ID,
        help="Optional explicit Bedrock model ID. Overrides --hotpot-answer-model.",
    )
    parser.add_argument(
        "--bedrock-region",
        type=str,
        default=BEDROCK_REGION,
        help="Optional AWS region override for Bedrock runtime calls.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", str(EVAL_JOB_INDEX))),
        help="Zero-based shard index for combo parallelization (array-task index).",
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=int(os.getenv("AWS_BATCH_JOB_ARRAY_SIZE", str(EVAL_JOB_COUNT))),
        help="Total shard count for combo parallelization (array size).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=EVAL_GPU_ID,
        help="If >=0, sets CUDA_VISIBLE_DEVICES to this GPU id for the current process.",
    )
    args = parser.parse_args()

    # Defaults are config-backed; CLI only overrides when provided.
    args.seed = EVAL_SEED
    args.sbert_model = EVAL_SBERT_MODEL
    args.e5_model = EVAL_E5_MODEL
    args.chunking_mode = EVAL_CHUNKING_MODE
    args.hierarchical_top_docs = EVAL_HIERARCHICAL_TOP_DOCS
    args.min_chars = EVAL_MIN_CHARS
    args.max_chars = EVAL_MAX_CHARS
    args.similarity_threshold = EVAL_SIMILARITY_THRESHOLD
    args.k_values = list(range(1, max(1, int(args.k)) + 1))
    args.answer_recall_k = max(1, int(args.k))
    args.hotpot_support_fact_coverage = EVAL_HOTPOT_SUPPORT_FACT_COVERAGE
    args.hotpot_official_emf1 = EVAL_HOTPOT_OFFICIAL_EMF1
    args.hotpot_answer_top_k = max(1, int(args.k))
    args.hotpot_answer_max_queries = int(args.max_queries)
    args.hotpot_sp_max_facts = EVAL_HOTPOT_SP_MAX_FACTS
    args.answer_match_min_tokens = EVAL_ANSWER_MATCH_MIN_TOKENS
    args.backend = EVAL_BACKEND
    args.model = EVAL_MODEL
    args.normalize = EVAL_NORMALIZE
    return args
