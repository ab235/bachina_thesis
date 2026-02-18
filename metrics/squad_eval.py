import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional


SQUAD_RAG_LABEL = "SQuAD RAG-generated EM/F1 (official normalization)"


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def _exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / float(len(pred_tokens))
    recall = num_same / float(len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def score_squad_predictions(
    pred_answers: Dict[str, str],
    gold_answers: Dict[str, List[str]],
    qids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    eval_qids = list(qids) if qids is not None else list(pred_answers.keys())
    eval_qids = [
        qid
        for qid in eval_qids
        if qid in gold_answers and any(str(a or "").strip() for a in gold_answers[qid])
    ]
    if not eval_qids:
        return {"label": SQUAD_RAG_LABEL, "num_scored": 0}

    em_sum = 0.0
    f1_sum = 0.0
    for qid in eval_qids:
        pred = str(pred_answers.get(qid, "") or "")
        golds = [str(a) for a in gold_answers.get(qid, []) if str(a or "").strip()]
        if not golds:
            continue
        em_sum += max(_exact_match_score(pred, gold) for gold in golds)
        f1_sum += max(_f1_score(pred, gold) for gold in golds)

    n = float(len(eval_qids))
    return {
        "label": SQUAD_RAG_LABEL,
        "num_scored": int(n),
        "exact_match": em_sum / n,
        "f1": f1_sum / n,
    }
