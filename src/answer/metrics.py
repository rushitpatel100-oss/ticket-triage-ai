"""Scoring the answer step: the judge's ranking, and the drafted answers (abstention, citations, overlap)."""

import re
import string
from collections import Counter

import numpy as np

from src import config

_CITE = re.compile(r"\[(\d)\]")


def cited_positions(answer: str) -> list[int]:
    """Article numbers cited as [1], [2], [3] (1-based, in order of first mention)."""
    return list(dict.fromkeys(int(n) for n in _CITE.findall(answer or "")))


def is_not_found(answer: str) -> bool:
    return config.NOT_FOUND in (answer or "")


def _normalise(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """Word-overlap F1 (the SQuAD measure). Crude for free-form answers, but cheap and reproducible."""
    p, r = _normalise(_CITE.sub(" ", prediction or "")), _normalise(reference or "")  # citations are not answer words
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def answer_report(answers: list[str], evidence_ids: list[list[str]], gold: list[list[str]],
                  resolvable: np.ndarray, gold_answers: list[str]) -> dict:
    """How the drafted answers behave on resolvable and unresolvable tickets."""
    resolvable = np.asarray(resolvable, dtype=bool)
    not_found = np.array([is_not_found(a) for a in answers])
    cites = [cited_positions(a) for a in answers]
    cited_ids = [[ids[i - 1] for i in c if 1 <= i <= len(ids)] for c, ids in zip(cites, evidence_ids)]
    cites_gold = np.array([bool(set(c) & set(g)) for c, g in zip(cited_ids, gold)])
    answered = ~not_found
    f1 = np.array([token_f1(a, g) for a, g in zip(answers, gold_answers)])

    def rate(mask_values, mask):
        return round(float(mask_values[mask].mean()), 4) if mask.any() else None

    return {
        "n": int(len(answers)),
        "n_resolvable": int(resolvable.sum()),
        "said_not_found": {"resolvable": rate(not_found, resolvable), "unresolvable": rate(not_found, ~resolvable)},
        "answered_with_a_citation": rate(np.array([bool(c) for c in cites]), answered),
        "cites_correct_article_when_resolvable_and_answered": rate(cites_gold, resolvable & answered),
        "token_f1_vs_reference_when_resolvable_and_answered": rate(f1, resolvable & answered),
    }
