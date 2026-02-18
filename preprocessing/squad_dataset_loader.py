import json
import pathlib
from typing import Dict, List, Tuple

from .sampling import sample_qids


def _expect_dict(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid SQuAD v1.1 format at {where}: expected object.")
    return value


def _expect_list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"Invalid SQuAD v1.1 format at {where}: expected array.")
    return value


def load_squad_v11(
    squad_path: pathlib.Path,
    max_queries: int,
    seed: int,
    keep_all_docs: bool = False,
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, List[str]],
]:
    with squad_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    root = _expect_dict(raw, "$")
    version = str(root.get("version", "")).strip()
    if version != "1.1":
        raise ValueError(
            f"Expected SQuAD v1.1 file (version='1.1'), but found version='{version or '<missing>'}'."
        )

    data = _expect_list(root.get("data"), "$.data")

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    answers: Dict[str, List[str]] = {}
    qid_to_doc: Dict[str, str] = {}

    for ai, article_obj in enumerate(data):
        article = _expect_dict(article_obj, f"$.data[{ai}]")
        title = str(article.get("title", "")).strip()
        paragraphs = _expect_list(article.get("paragraphs"), f"$.data[{ai}].paragraphs")

        for pi, para_obj in enumerate(paragraphs):
            para = _expect_dict(para_obj, f"$.data[{ai}].paragraphs[{pi}]")
            context = str(para.get("context", "")).strip()
            if not context:
                raise ValueError(f"Invalid SQuAD v1.1 format at $.data[{ai}].paragraphs[{pi}].context: empty.")

            doc_id = f"squad::{ai}::p{pi}"
            corpus[doc_id] = {"title": title, "text": context}

            qas = _expect_list(para.get("qas"), f"$.data[{ai}].paragraphs[{pi}].qas")
            for qi, qa_obj in enumerate(qas):
                qa = _expect_dict(qa_obj, f"$.data[{ai}].paragraphs[{pi}].qas[{qi}]")
                if bool(qa.get("is_impossible", False)):
                    raise ValueError(
                        "Found `is_impossible=true`, which indicates SQuAD v2.0-like rows. "
                        "Mode 1 supports SQuAD v1.1 only."
                    )

                qid = str(qa.get("id", "")).strip()
                question = str(qa.get("question", "")).strip()
                if not qid:
                    raise ValueError(f"Invalid SQuAD v1.1 format at $.data[{ai}].paragraphs[{pi}].qas[{qi}].id: empty.")
                if not question:
                    raise ValueError(
                        f"Invalid SQuAD v1.1 format at $.data[{ai}].paragraphs[{pi}].qas[{qi}].question: empty."
                    )

                answer_objs = _expect_list(qa.get("answers"), f"$.data[{ai}].paragraphs[{pi}].qas[{qi}].answers")
                alias_list: List[str] = []
                seen = set()
                for aj, answer_obj in enumerate(answer_objs):
                    answer = _expect_dict(
                        answer_obj,
                        f"$.data[{ai}].paragraphs[{pi}].qas[{qi}].answers[{aj}]",
                    )
                    answer_text = str(answer.get("text", "")).strip()
                    if not answer_text:
                        continue
                    lowered = answer_text.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    alias_list.append(answer_text)

                if not alias_list:
                    raise ValueError(
                        f"Invalid SQuAD v1.1 format at $.data[{ai}].paragraphs[{pi}].qas[{qi}].answers: "
                        "no non-empty answer text."
                    )

                queries[qid] = question
                answers[qid] = alias_list
                qid_to_doc[qid] = doc_id

    keep = set(sample_qids(list(queries.keys()), max_queries=max_queries, seed=seed))
    queries = {qid: q for qid, q in queries.items() if qid in keep}
    answers = {qid: a for qid, a in answers.items() if qid in keep}
    if not keep_all_docs:
        kept_docs = {qid_to_doc[qid] for qid in keep if qid in qid_to_doc}
        corpus = {did: doc for did, doc in corpus.items() if did in kept_docs}

    return corpus, queries, answers
