import re
import string
from collections import Counter
from typing import Dict, Iterable, Optional, Set, Tuple


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


def _f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    pred = _normalize_answer(prediction)
    gold = _normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)
    special = {"yes", "no", "noanswer"}
    if pred in special or gold in special:
        return (1.0, 1.0, 1.0) if pred == gold else zero

    pred_toks = pred.split()
    gold_toks = gold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = num_same / float(len(pred_toks))
    recall = num_same / float(len(gold_toks))
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


def _exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _supporting_fact_metrics(
    pred: Set[Tuple[str, int]],
    gold: Set[Tuple[str, int]],
) -> Tuple[float, float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    em = float(fp + fn == 0)
    return (em, f1, precision, recall)


def remap_supporting_facts_to_titles(
    sp_by_qid: Dict[str, Iterable[Tuple[str, int]]],
    doc_id_to_title: Dict[str, str],
) -> Dict[str, Set[Tuple[str, int]]]:
    out: Dict[str, Set[Tuple[str, int]]] = {}
    for qid, facts in sp_by_qid.items():
        mapped: Set[Tuple[str, int]] = set()
        for doc_id, sent_idx in facts:
            title = doc_id_to_title.get(doc_id, doc_id)
            mapped.add((title, int(sent_idx)))
        out[qid] = mapped
    return out


def score_hotpot_predictions(
    pred_answers: Dict[str, str],
    pred_supporting_facts: Dict[str, Set[Tuple[str, int]]],
    gold_answers: Dict[str, str],
    gold_supporting_facts: Dict[str, Set[Tuple[str, int]]],
    qids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    eval_qids = list(qids) if qids is not None else list(pred_answers.keys())
    eval_qids = [qid for qid in eval_qids if qid in gold_answers and qid in gold_supporting_facts]
    if not eval_qids:
        return {"num_scored": 0}

    ans_em_sum = ans_f1_sum = ans_prec_sum = ans_rec_sum = 0.0
    sp_em_sum = sp_f1_sum = sp_prec_sum = sp_rec_sum = 0.0
    joint_em_sum = joint_f1_sum = joint_prec_sum = joint_rec_sum = 0.0

    for qid in eval_qids:
        pred_ans = pred_answers.get(qid, "")
        gold_ans = gold_answers.get(qid, "")
        ans_em = _exact_match_score(pred_ans, gold_ans)
        ans_f1, ans_prec, ans_rec = _f1_score(pred_ans, gold_ans)

        pred_sp = set(pred_supporting_facts.get(qid, set()))
        gold_sp = set(gold_supporting_facts.get(qid, set()))
        sp_em, sp_f1, sp_prec, sp_rec = _supporting_fact_metrics(pred_sp, gold_sp)

        joint_prec = ans_prec * sp_prec
        joint_rec = ans_rec * sp_rec
        joint_f1 = (2 * joint_prec * joint_rec / (joint_prec + joint_rec)) if joint_prec + joint_rec > 0 else 0.0
        joint_em = ans_em * sp_em

        ans_em_sum += ans_em
        ans_f1_sum += ans_f1
        ans_prec_sum += ans_prec
        ans_rec_sum += ans_rec
        sp_em_sum += sp_em
        sp_f1_sum += sp_f1
        sp_prec_sum += sp_prec
        sp_rec_sum += sp_rec
        joint_em_sum += joint_em
        joint_f1_sum += joint_f1
        joint_prec_sum += joint_prec
        joint_rec_sum += joint_rec

    n = float(len(eval_qids))
    return {
        "num_scored": int(n),
        "answer_em": ans_em_sum / n,
        "answer_f1": ans_f1_sum / n,
        "answer_precision": ans_prec_sum / n,
        "answer_recall": ans_rec_sum / n,
        "supporting_fact_em": sp_em_sum / n,
        "supporting_fact_f1": sp_f1_sum / n,
        "supporting_fact_precision": sp_prec_sum / n,
        "supporting_fact_recall": sp_rec_sum / n,
        "joint_em": joint_em_sum / n,
        "joint_f1": joint_f1_sum / n,
        "joint_precision": joint_prec_sum / n,
        "joint_recall": joint_rec_sum / n,
    }

