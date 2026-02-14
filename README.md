# Chunking Evaluation Pipeline

This repo currently runs chunking/retrieval evaluation on **HotpotQA distractor** with:
- chunk-level recall
- Hotpot supporting-fact coverage
- generated-answer Hotpot EM/F1-style metrics

The main entrypoint is:

```bash
python cpu_hotpot_qasper_grid_eval.py --mode 2 --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json --k 5
```

## Current Status

- `mode=2`: implemented (HotpotQA distractor)
- `mode=1`: placeholder (not implemented yet)
- `mode=3`: placeholder (not implemented yet)

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
- `--k` (top-k for retrieval/eval)
- `--max-queries` (`<=0` means all)
- `--batch-size`
- `--token-size` (default `256`)
- `--overlap` (default `64`)
- `--chunkers` (default: all available)
- `--retrievers` (default: all available)
- `--output` (default: `results.json`)

Internal fixed settings include:
- `seed = 42`
- answer model = `gpt-5.2-chat-latest`

## Example Runs

Run all default chunkers/retrievers on 20 queries:

```bash
python cpu_hotpot_qasper_grid_eval.py \
  --mode 2 \
  --dataset-path-mode2 datasets/hotpot_dev_distractor_v1.json \
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

## Output

Results are saved as JSON:

- top-level `config`
- list of `results` rows per `(chunker, retriever)` run

Each row includes:
- `chunk_recall`
- `support_fact_all` (evidence coverage exactness by k)
- `support_fact_recall` (evidence recall by k)
- `hotpot_official_emf1`:
  - answer EM/F1/precision/recall
  - supporting-fact EM/F1/precision/recall
  - joint EM/F1/precision/recall

## Notes

- LLM generation can be the slowest/costliest step.
- If you hit token/rate limits, lower `--max-queries`, reduce grid size (`--chunkers`/`--retrievers`), or lower `--k`.
