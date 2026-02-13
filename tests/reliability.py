import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from textwrap import dedent

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import OPENAI_API_KEY, OPENAI_GRADER_MODEL
from ask import answer as rag_answer
from ground_truth import GROUND_TRUTH, GroundTruthCase

ALLOWED_SCORES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
SCORE_GUIDE = "\n".join(
    [
        "1.0 = perfect match",
        "0.9 = very close",
        "0.8 = wording is off but meaning is close",
        "0.7 = mostly wrong",
        "0.6 = completely incorrect",
        "0.5 = no answer",
    ]
)
JUDGE_SYSTEM_PROMPT = dedent(
    """\
    You are an impartial grader. Score the model's answer against the gold answer for the question.
    Return only JSON like {"score": 0.9, "rationale": "..."}.
    Use the provided scoring scale without inventing new values.
    """
)
_judge_client: Any = None


@dataclass
class EvaluationResult:
    case: GroundTruthCase
    response: str
    score: float
    passed: bool
    rationale: str


def get_judge_client():
    global _judge_client
    if not _judge_client:
        try:
            from openai import OpenAI  # local import so --dry-run works without the dependency
        except ModuleNotFoundError as exc:
            raise RuntimeError("The 'openai' package is required to judge answers.") from exc
        _judge_client = OpenAI(api_key=OPENAI_API_KEY)
    return _judge_client


def normalize_score(raw_score: float) -> float:
    return min(ALLOWED_SCORES, key=lambda allowed: abs(allowed - raw_score))


def judge_response(question: str, gold_answer: str, model_answer: str) -> Tuple[float, str]:
    trimmed_answer = model_answer.strip()
    if not trimmed_answer:
        return 0.5, "No answer produced by the model."

    client = get_judge_client()
    user_prompt = dedent(
        f"""\
        Score the model response using this scale:
        {SCORE_GUIDE}

        Question:
        {question}

        Gold answer:
        {gold_answer}

        Model answer:
        {trimmed_answer}

        Respond with JSON only.
        """
    )
    completion = client.chat.completions.create(
        model=OPENAI_GRADER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return 0.6, f"Judge returned invalid JSON: {content!r}"

    raw_score = payload.get("score")
    try:
        score_value = float(raw_score)
    except (TypeError, ValueError):
        score_value = 0.6
    rationale = str(payload.get("rationale", "")).strip() or "No rationale provided."
    return normalize_score(score_value), rationale


def load_cached_responses(path: Optional[Path]) -> Dict[str, Dict[str, Optional[str]]]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Cached responses file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    normalized: Dict[str, Dict[str, Optional[str]]] = {}
    for case_id, value in data.items():
        if isinstance(value, str):
            normalized[case_id] = {"response": value, "asked_at": None}
        elif isinstance(value, dict) and "response" in value:
            normalized[case_id] = {
                "response": value["response"],
                "asked_at": value.get("asked_at"),
            }
        else:
            raise ValueError(f"Unrecognized cache entry for {case_id!r}: {value}")
    return normalized


def save_cached_responses(path: Path, cache: Dict[str, Dict[str, Optional[str]]]) -> None:
    serializable: Dict[str, Dict[str, Optional[str]]] = {}
    for case_id, payload in cache.items():
        serializable[case_id] = {
            "response": payload.get("response"),
            "asked_at": payload.get("asked_at"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def evaluate_response(case: GroundTruthCase, response: str) -> EvaluationResult:
    score, rationale = judge_response(case.question, case.answer, response)
    passed = score >= case.pass_threshold
    return EvaluationResult(case=case, response=response, score=score, passed=passed, rationale=rationale)


def select_cases(args: argparse.Namespace) -> List[GroundTruthCase]:
    cases = list(GROUND_TRUTH)
    if args.questions:
        wanted = {case_id.strip() for case_id in args.questions}
        cases = [case for case in cases if case.id in wanted]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected. Check your filters.")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the handbook RAG system against curated ground-truth questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, help="Maximum number of questions to ask.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle questions before running.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when shuffling.")
    parser.add_argument("--questions", nargs="+", help="Specific question IDs to run (e.g., sbr-001 sbr-010).")
    parser.add_argument(
        "--responses",
        type=Path,
        help="Path to a JSON file containing cached model responses to reuse instead of calling the API.",
    )
    parser.add_argument(
        "--save-responses",
        type=Path,
        help="Where to store the collected responses (including newly generated ones).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected questions without calling the model or scoring.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cases = select_cases(args)
    cache = load_cached_responses(args.responses)
    updated_cache = dict(cache)

    if args.dry_run:
        print("Dry run selected. The following cases would be evaluated:")
        for case in cases:
            print(f"- {case.id}: {case.question}")
        return

    results: List[EvaluationResult] = []
    for case in cases:
        cached = cache.get(case.id)
        if cached:
            response = cached["response"] or ""
        else:
            response = rag_answer(case.question)
            timestamp = datetime.now(tz=timezone.utc).isoformat()
            updated_cache[case.id] = {"response": response, "asked_at": timestamp}

        evaluation = evaluate_response(case, response)
        results.append(evaluation)
        status = "PASS" if evaluation.passed else "FAIL"
        print(f"[{case.id}] {status} score={evaluation.score:.1f}")
        print(f"Question : {case.question}")
        print(f"Expected : {case.answer}")
        print(f"Reference: {case.reference}")
        print(f"Judge rationale: {evaluation.rationale}")
        print(f"Model answer:\n{response.strip()}\n{'-' * 80}")

    passed = sum(1 for result in results if result.passed)
    average_score = sum(result.score for result in results) / len(results)
    print(f"\nCompleted {len(results)} questions. Passed {passed}/{len(results)} with avg keyword score {average_score:.2f}.")

    if args.save_responses:
        save_cached_responses(args.save_responses, updated_cache)
        print(f"Saved response cache to {args.save_responses}")


if __name__ == "__main__":
    run()
