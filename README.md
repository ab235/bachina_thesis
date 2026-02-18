# Chunking Evaluation Pipeline

This repo currently runs chunking/retrieval evaluation on:
- **SQuAD v1.1** (mode 1)
- **HotpotQA distractor/fullwiki** (modes 2/3)

Metrics include:
- chunk-level recall
- Hotpot supporting-fact coverage
- generated-answer Hotpot EM/F1-style metrics

The main entrypoint is:

```bash
python cpu_hotpot_qasper_grid_eval.py --mode 2 --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json --k 5
```

## Current Status

- `mode=2`: implemented (HotpotQA distractor)
- `mode=1`: implemented (SQuAD v1.1)
- `mode=3`: implemented (HotpotQA fullwiki, requires global wiki corpus file)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Set OpenAI key for answer generation:

```bash
export OPENAI_API_KEY=...
```

## CLI Arguments

Minimal CLI surface (see `preprocessing/args.py`):

- `--mode` (required): `1 | 2 | 3`
- `--dataset-path-mode1`
- `--dataset-path-mode2`
- `--dataset-path-mode3`
- `--wiki-corpus-path` (required for `mode=3`; accepts a JSON/JSONL file or a directory of shard files)
- `--k` (top-k for retrieval/eval)
- `--max-queries` (`<=0` means all)
- `--batch-size`
- `--token-size` (default `256`)
- `--overlap` (default `64`)
- `--chunkers` (default: all available)
- `--retrievers` (default: all available)
- `--output` (default: `results.json`)
- `--answer-provider` (`ollama | bedrock`, default: `ollama`)
- `--hotpot-answer-model` (`llama | mistral | qwen`, default: `llama`)
- `--bedrock-model-id` (optional explicit Bedrock model id)
- `--bedrock-region` (optional AWS region override for Bedrock)
- `--job-index` (array/shard index for parallel config execution)
- `--job-count` (array/shard size for parallel config execution)
- `--gpu-id` (if `>=0`, sets `CUDA_VISIBLE_DEVICES` for this process)

Internal fixed settings include:
- `seed = 42`
- answer provider/model defaults = `ollama` + `llama`

## Example Runs

Run all default chunkers/retrievers on 20 queries:

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json \
  --k 5 \
  --max-queries 20
```

Run SQuAD v1.1 (mode 1):

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 1 \
  --dataset-path-mode1 datasets/train-v1.1.json \
  --k 5 \
  --max-queries 20
```

Run a single configuration:

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json \
  --chunkers recursive \
  --retrievers e5 \
  --k 5 \
  --max-queries 20 \
  --output results_recursive_e5.json
```

Run with AWS Bedrock answer generation (example with explicit model id):

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json \
  --chunkers recursive \
  --retrievers e5 \
  --answer-provider bedrock \
  --hotpot-answer-model llama \
  --bedrock-model-id meta.llama3-1-70b-instruct-v1:0 \
  --bedrock-region us-east-1 \
  --k 5 \
  --max-queries 20
```

Run fullwiki (mode 3) with a global wiki corpus:

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 3 \
  --dataset-path-mode3 datasets/hotpot_dev_fullwiki_v1.json \
  --wiki-corpus-path datasets/wiki_corpus.json \
  --k 5 \
  --max-queries 20
```

Run parallel config shards (AWS Batch/ECS array-style):

```bash
# Example: 12 total (chunker,retriever) configs split across 4 array tasks.
# Each task runs only its assigned shard and writes a shard-specific output file.
AWS_BATCH_JOB_ARRAY_INDEX=0 AWS_BATCH_JOB_ARRAY_SIZE=4 EVAL_GPU_ID=0 \
  ./scripts/run_eval_array_worker.sh \
  --mode 2 \
  --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json \
  --k 5 \
  --max-queries 50 \
  --output results_array.json
```

Notes:
- The runner maps configs by index and selects those where `config_index % job_count == job_index`.
- With `--job-count > 1`, output is automatically suffixed as:
  `...job{index}-of-{count}.json`

## Output

Results are saved as JSON:

- top-level `config`
- list of `results` rows per `(chunker, retriever)` run

Each row includes:
- `chunk_recall`
- `squad_rag_generated_emf1` (SQuAD mode only; label: `SQuAD RAG-generated EM/F1 (official normalization)`)
- `support_fact_all` (evidence coverage exactness by k)
- `support_fact_recall` (evidence recall by k)
- `hotpot_official_emf1`:
  - answer EM/F1/precision/recall
  - supporting-fact EM/F1/precision/recall
  - joint EM/F1/precision/recall

## Notes

- LLM generation can be the slowest/costliest step.
- If you hit token/rate limits, lower `--max-queries`, reduce grid size (`--chunkers`/`--retrievers`), or lower `--k`.
