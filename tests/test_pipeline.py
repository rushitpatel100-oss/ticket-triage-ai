"""End-to-end tests that run offline on a small generated dataset (no downloads, no GPU).

Run:  pytest
"""

import json
import random
from argparse import Namespace

import pandas as pd
import pytest

from src import config


def make_fake_tickets(n_per_queue: int = 40, seed: int = 0) -> pd.DataFrame:
    """Tiny made-up dataset with the same columns as the real one. For testing only."""
    rng = random.Random(seed)
    topics = {
        "IT Support": ["VPN disconnects", "laptop will not boot", "password reset failed", "printer offline"],
        "Billing and Payments": ["invoice charged twice", "refund not received", "card payment declined", "wrong VAT"],
        "Human Resources": ["holiday request", "payroll question", "contract update", "parental leave form"],
    }
    rows = []
    for queue, issues in topics.items():
        for i in range(n_per_queue):
            issue = rng.choice(issues)
            urgent = rng.random() < 0.4
            rows.append({
                "subject": issue.capitalize(),
                "body": f"Hello, ticket {queue[:3]}-{i}: {issue}. "
                        + ("This is urgent, it blocks the whole team." if urgent else "No rush, whenever possible."),
                "queue": queue,
                "priority": "high" if urgent else rng.choice(["low", "medium"]),
                "type": "Incident" if "failed" in issue or "offline" in issue or "not" in issue else "Request",
                "language": "en",
            })
    rows += [  # rows that cleaning should remove
        {**rows[0]},  # exact duplicate
        {"subject": "Drucker", "body": "Der Drucker geht nicht.", "queue": "IT Support",
         "priority": "low", "type": "Incident", "language": "de"},
        {"subject": None, "body": "Missing label", "queue": None, "priority": "low", "type": "Request", "language": "en"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every output folder at a temporary directory and create the data splits."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")

    from src.data import clean, save_splits, split

    train, val, test = split(clean(make_fake_tickets()))
    save_splits(train, val, test)
    return tmp_path


def test_clean_removes_bad_rows():
    from src.data import clean

    raw = make_fake_tickets()
    df = clean(raw)
    assert len(df) == len(raw) - 3
    assert df["text"].is_unique
    assert set(df.columns) == {"text", "queue", "priority", "type"}
    assert not df["text"].str.contains("Drucker").any()


def test_split_keeps_every_queue_in_every_part():
    from src.data import clean, split

    df = clean(make_fake_tickets())
    parts = split(df)
    assert sum(len(p) for p in parts) == len(df)
    for part in parts:
        assert set(part["queue"]) == set(df["queue"])


def test_baseline_end_to_end(workspace):
    from src.baseline import run_task
    from src.data import load_splits
    from src.predict import TicketClassifier

    train, val, test = load_splits()
    for task in config.TASKS:
        metrics = run_task(task, train, val, test)
        assert 0 <= metrics["macro_f1"] <= 1

    results = json.loads((workspace / "results" / "metrics.json").read_text())
    assert set(results) == set(config.TASKS)
    assert (workspace / "results" / "confusion_queue_tfidf.png").exists()

    clf = TicketClassifier()
    scores = clf.predict("Invoice charged twice", "Please refund the duplicate payment")
    assert set(scores) == set(config.TASKS)
    assert scores["queue"] and abs(sum(scores["queue"].values()) - 1) < 1e-6
    assert next(iter(scores["queue"])) == "Billing and Payments"


def make_tiny_transformer(path, texts) -> str:
    """Build a tiny randomly initialised DistilBERT + tokenizer so training can be tested offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                        cls_token="[CLS]", sep_token="[SEP]",
                                        model_input_names=["input_ids", "attention_mask"])
    tiny = DistilBertConfig(vocab_size=tok.get_vocab_size(), dim=32, n_layers=1, n_heads=2,
                            hidden_dim=64, max_position_embeddings=128)
    DistilBertModel(tiny).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def test_transformer_training_end_to_end(workspace):
    from src.data import load_splits
    from src.predict import TicketClassifier
    from src.report import END, START
    from src.report import main as report_main
    from src.train import train_task

    train, val, test = load_splits()
    model_dir = make_tiny_transformer(workspace / "tiny-model", train["text"].tolist())
    args = Namespace(model_name=model_dir, epochs=1, batch_size=8, lr=1e-3, max_length=64, push_to_hub=None)

    metrics = train_task("queue", train, val, test, args)
    assert 0 <= metrics["macro_f1"] <= 1
    assert (workspace / "models" / "queue" / "config.json").exists()
    assert (workspace / "results" / "confusion_queue_transformer.png").exists()

    clf = TicketClassifier(tasks=["queue"])
    assert clf.models["queue"]["kind"] == "transformer"
    scores = clf.predict("VPN disconnects", "It keeps dropping")
    assert set(scores["queue"]) == set(train["queue"])

    (workspace / "README.md").write_text(f"# Demo\n{START}\nPLACEHOLDER\n{END}\nfooter\n")
    report_main()
    readme = (workspace / "README.md").read_text()
    assert "| queue |" in readme and "PLACEHOLDER" not in readme and readme.endswith("footer\n")
    assert (workspace / "results" / "model_comparison.png").exists()
