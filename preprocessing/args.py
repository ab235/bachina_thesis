import argparse
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunking evaluation runner."
    )
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument(
        "--dataset-path-mode1",
        type=pathlib.Path,
        default=pathlib.Path("datasets/squad_or_nq.json"),
        help="Dataset path for mode 1 (placeholder until implemented).",
    )
    parser.add_argument(
        "--dataset-path-mode2",
        type=pathlib.Path,
        default=pathlib.Path("datasets/hotpot_dev_distractor_v1.json"),
        help="Dataset path for mode 2 (HotpotQA distractor).",
    )
    parser.add_argument(
        "--dataset-path-mode3",
        type=pathlib.Path,
        default=pathlib.Path("datasets/hotpot_dev_distractor_v1.json"),
        help="Dataset path for mode 3 (placeholder until implemented).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for retrieval/evaluation.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max number of queries to evaluate. Set <=0 to use all queries.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--token-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=["token", "sentence", "recursive", "semantic", "late_token_pool"],
        choices=["token", "sentence", "recursive", "semantic", "late_token_pool"],
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["sbert", "e5", "bm25s"],
        choices=["sbert", "e5", "bm25s"],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results.json"),
    )
    args = parser.parse_args()

    # Fixed internal defaults to keep downstream pipeline behavior stable.
    args.seed = 42
    args.sbert_model = "sentence-transformers/msmarco-MiniLM-L6-v3"
    args.e5_model = "intfloat/e5-base-v2"
    args.chunking_mode = "early"
    args.hierarchical_top_docs = 20
    args.min_chars = 200
    args.max_chars = 1200
    args.similarity_threshold = 0.8
    args.k_values = list(range(1, max(1, int(args.k)) + 1))
    args.answer_recall_k = max(1, int(args.k))
    args.hotpot_support_fact_coverage = True
    args.hotpot_official_emf1 = True
    args.hotpot_answer_model = "gpt-5.2-chat-latest"
    args.hotpot_answer_top_k = max(1, int(args.k))
    args.hotpot_answer_max_queries = int(args.max_queries)
    args.hotpot_sp_max_facts = 100
    args.backend = "sbert"
    args.model = ""
    args.normalize = True
    return args
