"""Classic machine-learning baselines, so we can tell whether the transformer is worth it.

1. Majority class: always predicts the most common label (the "do nothing" floor).
2. TF-IDF + logistic regression: a strong, fast, explainable text classifier.

Run:  python -m src.baseline            (all tasks)
      python -m src.baseline --task queue
"""

import argparse
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src import config
from src.data import label_names, load_splits
from src.evaluate import compute_metrics, log_result, plot_confusion_matrix, save_errors


def make_tfidf_model(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


def run_task(task: str, train, val, test) -> dict:
    col = config.TASKS[task]
    labels = label_names(pd.concat([train, val, test]), task)  # every class, even rare ones
    print(f"\n=== {task} ({len(labels)} classes) ===")

    # 1. Majority class
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train[col])
    dummy_metrics = compute_metrics(test[col], dummy.predict(test["text"]), labels)
    log_result(task, "Majority class", dummy_metrics)
    print(f"Majority class      acc={dummy_metrics['accuracy']:.3f}  macro-F1={dummy_metrics['macro_f1']:.3f}")

    # 2. TF-IDF + logistic regression; pick regularisation strength C on the validation set
    best_c, best_f1 = None, -1.0
    for c in (0.5, 2.0, 8.0):
        model = make_tfidf_model(c).fit(train["text"], train[col])
        f1 = f1_score(val[col], model.predict(val["text"]), average="macro")
        print(f"  C={c:<4} validation macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    model = make_tfidf_model(best_c).fit(train["text"], train[col])
    preds = model.predict(test["text"])
    confidence = model.predict_proba(test["text"]).max(axis=1)
    metrics = compute_metrics(test[col], preds, labels)
    metrics["params"] = {"C": best_c}
    log_result(task, "TF-IDF + LogReg", metrics)
    print(f"TF-IDF + LogReg     acc={metrics['accuracy']:.3f}  macro-F1={metrics['macro_f1']:.3f}  (C={best_c})")

    plot_confusion_matrix(test[col], preds, labels, f"TF-IDF + LogReg: {task}",
                          config.RESULTS_DIR / f"confusion_{task}_tfidf.png")
    save_errors(test["text"], test[col], preds, confidence, config.RESULTS_DIR / f"errors_{task}_tfidf.csv")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": labels}, config.MODELS_DIR / f"baseline_{task}.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(config.TASKS), help="run one task only (default: all)")
    args = parser.parse_args()

    train, val, test = load_splits()
    tasks = [args.task] if args.task else [t for t, col in config.TASKS.items() if col in train]
    summary = {task: run_task(task, train, val, test)["macro_f1"] for task in tasks}
    print("\nTF-IDF macro-F1 by task:", json.dumps(summary))


if __name__ == "__main__":
    main()
