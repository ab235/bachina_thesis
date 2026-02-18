import os
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_csv(name: str, default_csv: str) -> list[str]:
    raw = os.getenv(name, default_csv)
    return [part.strip() for part in raw.split(",") if part.strip()]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

OPENAI_GRADER_MODEL = os.getenv("OPENAI_GRADER_MODEL", OPENAI_CHAT_MODEL)

BEDROCK_REGION = os.getenv(
    "BEDROCK_REGION",
    os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "")),
)
BEDROCK_LLAMA_MODEL_ID = os.getenv("BEDROCK_LLAMA_MODEL_ID", "")
BEDROCK_MISTRAL_MODEL_ID = os.getenv("BEDROCK_MISTRAL_MODEL_ID", "")
BEDROCK_QWEN_MODEL_ID = os.getenv("BEDROCK_QWEN_MODEL_ID", "")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")

PGHOST = os.getenv("PGHOST", "localhost")

PGPORT = int(os.getenv("PGPORT", "5432"))

PGDATABASE = os.getenv("PGDATABASE", "handbook_rag")

PGUSER = os.getenv("PGUSER", "postgres")

PGPASSWORD = os.getenv("PGPASSWORD", "postgres")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

TOP_K = int(os.getenv("TOP_K", "6"))

# Evaluation defaults (CLI overrides still supported)
EVAL_MODE = _env_int("EVAL_MODE", 2)
EVAL_DATASET_PATH_MODE1 = os.getenv("EVAL_DATASET_PATH_MODE1", "datasets/train-v1.1.json")
EVAL_DATASET_PATH_MODE2 = os.getenv("EVAL_DATASET_PATH_MODE2", "datasets/hotpot_dev_distractor_v1.json")
EVAL_DATASET_PATH_MODE3 = os.getenv("EVAL_DATASET_PATH_MODE3", "datasets/hotpot_dev_distractor_v1.json")
EVAL_WIKI_CORPUS_PATH = os.getenv("EVAL_WIKI_CORPUS_PATH", "datasets/wiki_corpus.json")
EVAL_K = _env_int("EVAL_K", 5)
EVAL_MAX_QUERIES = _env_int("EVAL_MAX_QUERIES", 0)
EVAL_BATCH_SIZE = _env_int("EVAL_BATCH_SIZE", 32)
EVAL_TOKEN_SIZE = _env_int("EVAL_TOKEN_SIZE", 256)
EVAL_OVERLAP = _env_int("EVAL_OVERLAP", 64)
EVAL_CHUNKERS = _env_csv(
    "EVAL_CHUNKERS",
    "token,sentence,recursive,semantic,late_token_pool",
)
EVAL_RETRIEVERS = _env_csv("EVAL_RETRIEVERS", "sbert,e5,bm25s")
EVAL_OUTPUT = os.getenv("EVAL_OUTPUT", "results.json")
EVAL_ANSWER_PROVIDER = os.getenv("EVAL_ANSWER_PROVIDER", "ollama")
EVAL_HOTPOT_ANSWER_MODEL = os.getenv("EVAL_HOTPOT_ANSWER_MODEL", "llama")
OLLAMA_TIMEOUT_SECONDS = _env_int("OLLAMA_TIMEOUT_SECONDS", 300)
OLLAMA_MAX_RETRIES = _env_int("OLLAMA_MAX_RETRIES", 5)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_float("OLLAMA_RETRY_BACKOFF_SECONDS", 3.0)

# Internal pipeline defaults
EVAL_SEED = _env_int("EVAL_SEED", 42)
EVAL_SBERT_MODEL = os.getenv("EVAL_SBERT_MODEL", "sentence-transformers/msmarco-MiniLM-L6-v3")
EVAL_E5_MODEL = os.getenv("EVAL_E5_MODEL", "intfloat/e5-base-v2")
EVAL_CHUNKING_MODE = os.getenv("EVAL_CHUNKING_MODE", "early")
EVAL_HIERARCHICAL_TOP_DOCS = _env_int("EVAL_HIERARCHICAL_TOP_DOCS", 20)
EVAL_MIN_CHARS = _env_int("EVAL_MIN_CHARS", 200)
EVAL_MAX_CHARS = _env_int("EVAL_MAX_CHARS", 1200)
EVAL_SIMILARITY_THRESHOLD = _env_float("EVAL_SIMILARITY_THRESHOLD", 0.8)
EVAL_HOTPOT_SUPPORT_FACT_COVERAGE = os.getenv("EVAL_HOTPOT_SUPPORT_FACT_COVERAGE", "true").lower() == "true"
EVAL_HOTPOT_OFFICIAL_EMF1 = os.getenv("EVAL_HOTPOT_OFFICIAL_EMF1", "true").lower() == "true"
EVAL_HOTPOT_SP_MAX_FACTS = _env_int("EVAL_HOTPOT_SP_MAX_FACTS", 60)
EVAL_ANSWER_MATCH_MIN_TOKENS = _env_int("EVAL_ANSWER_MATCH_MIN_TOKENS", 2)
EVAL_JOB_INDEX = _env_int("EVAL_JOB_INDEX", 0)
EVAL_JOB_COUNT = _env_int("EVAL_JOB_COUNT", 1)
EVAL_GPU_ID = _env_int("EVAL_GPU_ID", -1)
EVAL_BACKEND = os.getenv("EVAL_BACKEND", "sbert")
EVAL_MODEL = os.getenv("EVAL_MODEL", "")
EVAL_NORMALIZE = os.getenv("EVAL_NORMALIZE", "true").lower() == "true"

def validate_config():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    if not PGHOST or not PGDATABASE or not PGUSER:
        raise RuntimeError("Database connection not made. Error in PGHOST.")

validate_config()

def get_db_url() -> str:
    return f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
