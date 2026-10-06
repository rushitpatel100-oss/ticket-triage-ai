"""Offline tests for intake clean-up, redaction and the E5 robustness harness."""

import json

import joblib
import pandas as pd

from src import config
from src.intake import clean_ticket, redact
from src.robustness import VARIANTS, add_typos, make_short, make_variants, retrieval_robustness, routing_robustness
from tests.test_pipeline import workspace  # noqa: F401  (pytest fixture)
from tests.test_retrieval import FakeEncoder

TICKET = "VPN disconnects every 10 minutes\n\nSince the update the VPN client drops and I have to log in again."


def test_clean_ticket_removes_email_noise():
    messy = (f"{TICKET}\n\nKind regards,\nAlex Morgan\nFinance Analyst\nT: +44 20 7946 0000\n\n"
             "This email and any attachments are confidential and intended solely for the addressee.\n\n"
             "-----Original Message-----\nFrom: Service Desk <servicedesk@example.com>\nSent: Monday\n"
             "To: Alex\nSubject: RE: printer\n\n> Your printer ticket was closed.")
    assert clean_ticket(messy) == TICKET
    on_wrote = f"{TICKET}\n\nOn Mon, 5 Oct 2026 at 09:12, Service Desk <sd@example.com> wrote:\n> old reply"
    assert clean_ticket(on_wrote) == TICKET
    assert clean_ticket(f"{TICKET}\n\nThanks,\nSam\n\nSent from my iPhone") == TICKET


def test_clean_ticket_leaves_clean_text_alone():
    assert clean_ticket(TICKET) == TICKET
    # "Thanks" early in a long ticket is content, not a sign-off
    long = "Thanks\n" + "\n".join(f"step {i}: the installer stops with error {i}" for i in range(12))
    assert clean_ticket(long) == long
    assert clean_ticket("") == "" and clean_ticket("Thanks,") == "Thanks,"


def test_redact_masks_personal_data_only():
    text = "Call me on +44 20 7946 0000 or mail alex.morgan@example.com, card 4111 1111 1111 1111. Error 0x80070005 in 8.5.5.0"
    out = redact(text)
    assert "[PHONE]" in out and "[EMAIL]" in out and "[NUMBER]" in out
    assert "alex.morgan" not in out and "4111" not in out
    assert "0x80070005" in out and "8.5.5.0" in out


def test_variants_are_deterministic_and_noisy():
    texts = [TICKET, "Printer offline\n\nThe office printer shows offline for everyone.", "Outlook crashes on start"]
    a, b = make_variants(texts), make_variants(texts)
    assert a == b and set(a) == set(VARIANTS)
    assert all(len(v) == 3 for v in a.values())
    assert all(o != t for o, t in zip(a["forwarded"], texts)) and "Original Message" in a["forwarded"][0]
    assert a["short"][0] == "VPN disconnects every 10 minutes" and make_short("one two", 1) == "one"
    assert a["signature+cleanup"] == texts  # clean-up removes exactly what was added
    assert a["forwarded+cleanup"] == texts
    import random
    typo = add_typos(" ".join(["configuration"] * 200), random.Random(0))
    assert 0.05 < sum(w != "configuration" for w in typo.split()) / 200 < 0.25


def write_kb(tmp_path):
    from src.sources import common

    topics = ["vpn certificate expired", "printer driver crash", "mailbox quota exceeded", "laptop battery drain",
              "excel macro blocked", "wifi authentication loop"]
    docs = pd.DataFrame({"doc_id": [f"kb:{i}" for i in range(6)], "source": "kb", "title": topics,
                         "text": [f"{t}: follow these steps to fix the {t} problem." for t in topics], "url": ""})
    q = pd.DataFrame([{"query_id": f"q{i}", "source": "kb", "text": f"{topics[i % 6]} again\n\nhelp with {topics[i % 6]}",
                       "answerable": True, "gold_doc_ids": [f"kb:{i % 6}"], "gold_answer": "", "split": "test"}
                      for i in range(12)])
    common.save_tables("techqa", docs, q)


def test_retrieval_robustness_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    write_kb(tmp_path)
    out = retrieval_robustness("techqa", FakeEncoder())
    assert out["questions"] == 12 and set(out["variants"]) == set(VARIANTS)
    clean = out["variants"]["clean"]["dense"]["mrr@10"]
    assert clean > 0.9 and out["variants"]["signature+cleanup"]["dense"]["mrr@10"] == clean


def test_routing_robustness_with_saved_baseline(workspace):  # noqa: F811
    from src.baseline import run_task
    from src.data import load_splits

    train, val, test = load_splits()
    run_task("queue", train, val, test)
    assert (workspace / "models" / "baseline_queue.joblib").exists()
    out = routing_robustness(tasks=("queue",))
    row = out["tasks"]["queue"]["TF-IDF + LogReg"]
    assert set(row) == set(VARIANTS) and all(0 <= v <= 1 for v in row.values())
    assert json.dumps(out)  # serialisable
    assert joblib.load(workspace / "models" / "baseline_queue.joblib")["labels"]
