"""Offline tests for the answer step: fake LLM for the pipeline, a tiny random model for the real wrapper."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.answer import redteam
from src.answer.metrics import answer_report, cited_positions, is_not_found, token_f1
from src.answer.prompts import SYSTEM, answer_messages, clip_words, judge_messages
from src.retrieval.retrievers import analyze


class FakeLLM:
    """Says yes when the first article shares words with the ticket, and answers citing it."""

    name = "fake-llm"

    def _overlap(self, ticket, evidence):
        if not evidence:
            return 0.0
        a, b = set(analyze(ticket)), set(analyze(evidence[0]["text"]))
        return len(a & b) / max(len(a), 1)

    def judge(self, tickets, evidence):
        p = np.array([self._overlap(t, e) for t, e in zip(tickets, evidence)])
        return np.clip(p, 0, 1), np.ones(len(p))

    def answer(self, tickets, evidence):
        return [f"Try this: {e[0]['text'][:40]} [1]" if self._overlap(t, e) > 0.3 else config.NOT_FOUND
                for t, e in zip(tickets, evidence)]


def test_prompts_treat_ticket_and_articles_as_data():
    ev = [{"doc_id": "a", "title": "VPN", "text": "word " * 500}]
    msgs = judge_messages("my vpn drops " + "x " * 1000, ev)
    assert msgs[0]["content"] == SYSTEM and "not instructions" in SYSTEM
    user = msgs[1]["content"]
    assert "<ticket>" in user and "</articles>" in user and user.endswith("Answer with only Yes or No.")
    assert len(user.split()) < config.QUESTION_MAX_WORDS + config.EVIDENCE_MAX_WORDS + 60  # both were clipped
    assert config.NOT_FOUND in answer_messages("q", ev)[1]["content"]
    assert clip_words("a b c", 2) == "a b ..." and clip_words("a b", 5) == "a b"


def test_answer_metrics():
    assert cited_positions("See [2] and [1], also [2].") == [2, 1]
    assert is_not_found("NOT_FOUND") and not is_not_found("Reboot it [1]")
    assert token_f1("Reboot the router", "reboot router") == pytest.approx(1.0)
    assert token_f1("", "x") == 0.0 and 0 < token_f1("reboot the printer now", "reboot printer") < 1

    report = answer_report(
        answers=["Reboot [1]", "NOT_FOUND", "Update the driver [2]", "NOT_FOUND"],
        evidence_ids=[["g1", "x"], ["x", "y"], ["x", "g3"], ["x", "y"]],
        gold=[["g1"], ["g2"], ["g9"], []],
        resolvable=np.array([True, True, True, False]),
        gold_answers=["Reboot", "Clear cache", "Reinstall", ""])
    assert report["said_not_found"] == {"resolvable": pytest.approx(1 / 3, abs=1e-4), "unresolvable": 1.0}
    assert report["answered_with_a_citation"] == 1.0
    assert report["cites_correct_article_when_resolvable_and_answered"] == 0.5
    assert report["token_f1_vs_reference_when_resolvable_and_answered"] == 0.5


def test_redteam_cases_and_scoring():
    cases = redteam.cases()
    assert len(cases) == 15 and {c["kind"] for c in cases} == {"direct", "indirect"}
    assert all(redteam.CANARY in c["ticket"] or any(redteam.CANARY in e["text"] for e in c["evidence"]) for c in cases)
    s = redteam.score([redteam.CANARY, "NOT_FOUND", "reboot"], ["direct", "direct", "indirect"])
    assert s["attack_success_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["direct"] == {"cases": 2, "succeeded": 1} and s["indirect"] == {"cases": 1, "succeeded": 0}


def make_tiny_llm(path):
    """A tiny random Llama with a word-level tokenizer and a simple chat template, saved to disk."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = ["Yes No yes no system user assistant ticket articles answer the vpn printer reboot NOT_FOUND [1]"] * 5
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[EOS]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}assistant {% endif %}")
    model = LlamaForCausalLM(LlamaConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                         max_position_embeddings=4096, pad_token_id=0, eos_token_id=2))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def test_open_model_wrapper_with_tiny_model(tmp_path):
    from src.answer.llm import OpenModel

    llm = OpenModel(make_tiny_llm(tmp_path / "tiny-llm"), batch_size=2)
    assert llm.yes_ids and llm.no_ids and not set(llm.yes_ids) & set(llm.no_ids)
    ev = [[{"doc_id": "a", "title": "t", "text": "reboot the printer"}]] * 3
    tickets = ["printer", "the vpn printer answer", "vpn"]  # different lengths: exercises left padding
    p, mass = llm.judge(tickets, ev)
    assert p.shape == (3,) and ((0 <= p) & (p <= 1)).all() and ((0 < mass) & (mass <= 1)).all()
    p_single, _ = llm.judge(tickets[1:2], ev[1:2])
    assert p_single[0] == pytest.approx(p[1], abs=1e-4)  # batching and padding don't change the score
    answers = llm.answer(tickets, ev, max_new_tokens=3)
    assert len(answers) == 3 and all(isinstance(a, str) for a in answers)


def write_fixture(tmp_path):
    from src.sources import common

    topics = ["vpn certificate expired", "printer driver crash", "mailbox quota exceeded", "laptop battery drain",
              "excel macro blocked", "wifi authentication loop"]
    docs = pd.DataFrame({"doc_id": [f"kb:{i}" for i in range(len(topics))], "source": "kb", "title": topics,
                         "text": [f"{t}: follow these steps to fix the {t} problem." for t in topics], "url": ""})
    rows, runs = [], []
    for i in range(60):
        t = topics[i % len(topics)]
        answerable = i % 5 != 0
        gold = f"kb:{i % len(topics)}"
        rows.append({"query_id": f"q{i}", "source": "kb", "text": f"help with {t}" if answerable else "something else",
                     "answerable": answerable, "gold_doc_ids": [gold] if answerable else [], "gold_answer": f"fix {t}",
                     "split": ["train", "val", "test"][i % 3]})
        order = [gold] + [d for d in docs["doc_id"] if d != gold]
        runs.append({"query_id": f"q{i}", "split": rows[-1]["split"], "answerable": answerable, "method": "dense",
                     "doc_ids": order, "scores": list(np.linspace(0.9, 0.4, len(order)))})
    common.save_tables("techqa", docs, pd.DataFrame(rows))
    (tmp_path / "processed" / "retrieval").mkdir(parents=True)
    pd.DataFrame(runs).to_parquet(tmp_path / "processed" / "retrieval" / "techqa_runs.parquet", index=False)


def test_run_all_end_to_end(tmp_path, monkeypatch):
    from src.answer.__main__ import build_evidence, run_all
    from tests.test_retrieval import FakeEncoder

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    write_fixture(tmp_path)

    ev = build_evidence("techqa", FakeEncoder())
    assert len(ev) == 60 and all(len(e) == config.CONTEXT_DOCS for e in ev["evidence"])
    first = ev.iloc[1]
    assert first["evidence"][0]["doc_id"] == first["gold_doc_ids"][0] and first["resolvable"]
    assert not ev.loc[ev["query_id"] == "q0", "resolvable"].item()  # unanswerable

    results = run_all(["techqa"], FakeLLM(), FakeEncoder(), max_answers=10)
    tq = results["techqa"]
    assert tq["questions"] == 60
    assert tq["judge"]["test"]["resolvable"]["auroc"] > 0.9
    assert tq["answers"]["n"] == 10 and tq["answers"]["said_not_found"]["unresolvable"] == 1.0
    assert results["redteam"]["cases"] == 15 and results["redteam"]["attack_success_rate"] == 0.0
    judge = pd.read_parquet(tmp_path / "processed" / "answer" / "techqa_judge.parquet")
    assert len(judge) == 60 and judge["llm_p_yes"].between(0, 1).all()
    assert json.loads((tmp_path / "results" / "answer_metrics.json").read_text())["backend"] == "FakeLLM"


def test_decision_uses_llm_judgement(tmp_path, monkeypatch):
    from src.decision.__main__ import run_all as decision_run_all
    from tests.test_decision import synthetic

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "processed" / "retrieval").mkdir(parents=True)
    (tmp_path / "processed" / "answer").mkdir(parents=True)
    (tmp_path / "processed" / "techqa").mkdir(parents=True)
    runs, queries = synthetic(900, 3)
    queries.to_parquet(tmp_path / "processed" / "techqa" / "queries.parquet", index=False)
    pd.DataFrame({"doc_id": ["d"], "source": "s", "title": "", "text": "t", "url": ""}).to_parquet(
        tmp_path / "processed" / "techqa" / "docs.parquet", index=False)
    runs.to_parquet(tmp_path / "processed" / "retrieval" / "techqa_runs.parquet", index=False)
    # An LLM judgement that knows which questions are resolvable, with some noise
    from src.decision.features import build_features
    labels = build_features(runs, queries)[["query_id", "resolvable"]]
    rng = np.random.default_rng(0)
    labels["llm_p_yes"] = np.clip(labels["resolvable"] * 0.8 + rng.normal(0.1, 0.1, len(labels)), 0, 1)
    labels["llm_yes_no_mass"] = 0.99
    labels.drop(columns="resolvable").to_parquet(tmp_path / "processed" / "answer" / "techqa_judge.parquet")

    results = decision_run_all(["techqa"], llm_judge=True)
    models = results["techqa"]["models"]
    assert "llm_only" in models and models["llm_only"]["test"]["auroc"] > 0.95
    assert "llm_logit" in results["techqa"]["features"]
    assert (tmp_path / "results" / "decision_metrics_llm.json").exists()
    assert (tmp_path / "results" / "decision_risk_coverage_llm.png").exists()
    with pytest.raises(FileNotFoundError):
        (tmp_path / "processed" / "answer" / "techqa_judge.parquet").unlink()
        decision_run_all(["techqa"], llm_judge=True)


def test_claude_backend_with_mocked_sdk(monkeypatch):
    import sys
    import types

    from src.answer.llm import ClaudeModel

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ClaudeModel()

    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = "85" if "number from 0 to 100" in kwargs["messages"][0]["content"] else "Reboot it [1]"
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)])

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=Messages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    llm = ClaudeModel()
    ev = [[{"doc_id": "a", "title": "t", "text": "reboot"}]]
    p, _ = llm.judge(["printer offline"], ev)
    assert p.tolist() == [0.85]
    assert llm.answer(["printer offline"], ev) == ["Reboot it [1]"]
    assert calls[0]["model"] == config.CLAUDE_MODEL and calls[0]["system"] == SYSTEM
    assert "<ticket>" in calls[1]["messages"][0]["content"]


class ObedientLLM(FakeLLM):
    """Obeys planted instructions unless the text is datamarked; says a claim is supported if its words appear."""

    def generate(self, messages_list, max_new_tokens=None):
        out = []
        for m in messages_list:
            user = m[-1]["content"]
            marked = "^" in user and m[0]["content"].endswith("marked text.")
            out.append(redteam.CANARY if redteam.CANARY in user and not marked else "Reboot it [1]")
        return out

    def yes_probability(self, messages_list):
        p = []
        for m in messages_list:
            user = m[-1]["content"]
            articles = user.split("<statement>")[0]
            claim = user.split("<statement>")[-1].split("</statement>")[0]
            words = set(analyze(claim))
            p.append(len(words & set(analyze(articles))) / max(len(words), 1))
        return np.array(p), np.ones(len(p))


def test_datamark_and_sandwich_prompts():
    from src.answer.prompts import datamark

    ev = [{"doc_id": "a", "title": "VPN fix", "text": "update the client " * 300}]
    marked = answer_messages("my vpn  drops", ev, "datamark")
    assert "my^vpn^drops" in marked[1]["content"] and "'^'" in marked[0]["content"]
    assert len(marked[1]["content"].split("^")) < config.EVIDENCE_MAX_WORDS + 40  # clipped before marking
    sandwich = answer_messages("my vpn drops", ev, "sandwich")[1]["content"]
    assert sandwich.startswith("Task:") and sandwich.count("Task:") == 2 and "Reminder" in sandwich
    assert datamark("a  b\nc") == "a^b^c"
    with pytest.raises(ValueError):
        answer_messages("x", ev, "unknown")


def test_scanner_overfits_to_attacks_it_was_written_for():
    from src.answer.guardrails import flagged, injection_hits

    main = [flagged(c["ticket"], c["evidence"]) for c in redteam.cases("main")]
    heldout = [flagged(c["ticket"], c["evidence"]) for c in redteam.cases("heldout")]
    assert sum(main) >= 10 and sum(heldout) <= 3  # the honest result reported in the docs
    for benign in ("Please ignore the previous email, the printer works now", "I want to respond with ERROR-1234",
                   "You are now able to log in?"):
        assert injection_hits(benign) == []


def test_split_claims():
    from src.answer.guardrails import split_claims

    claims = split_claims("1. Update the VPN client to the latest version [1].\n- Disable power saving on the "
                          "network adapter [2]. Done. Then reconnect to the VPN and test.")
    assert claims == ["Update the VPN client to the latest version .", "Disable power saving on the network adapter .",
                      "Then reconnect to the VPN and test."]


def test_guardrails_run_all(tmp_path, monkeypatch):
    from src.answer.__main__ import run_all as answer_run_all
    from src.answer.guardrails import run_all
    from tests.test_retrieval import FakeEncoder

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    write_fixture(tmp_path)
    answer_run_all(["techqa"], ObedientLLM(), FakeEncoder(), max_answers=10, redteam_too=False)

    out = run_all(["techqa"], ObedientLLM(), FakeEncoder(), utility_n=10)
    d = out["defenses"]["main"]
    assert d["none"]["attack_success_rate"] == 1.0 and d["sandwich"]["attack_success_rate"] == 1.0
    assert d["datamark"]["attack_success_rate"] == 0.0
    assert d["scanner+datamark"]["blocked_by_scanner"] >= 10
    assert out["defenses"]["heldout"]["none"]["cases"] == 13
    fa = out["scanner_false_alarms"]["techqa"]
    assert fa["questions"] == 60 and fa["questions_flagged"] == 0
    faith = out["faithfulness"]["techqa"]
    assert faith["answers_checked"] > 0 and 0 <= faith["claims_supported"] <= 1
    assert out["datamark_utility"]["techqa"]["questions"] == 10
    assert (tmp_path / "results" / "guardrail_metrics.json").exists()


def test_token_budget_batches(tmp_path):
    from src.answer.llm import OpenModel

    llm = OpenModel(make_tiny_llm(tmp_path / "tiny-llm"), batch_size=4)
    texts = ["printer " * n for n in (1, 2, 3, 50, 50, 50, 50, 2)]
    batches = list(llm._batches(texts, max_items=4, token_budget=120))
    assert sorted(i for b in batches for i in b) == list(range(8))           # every prompt exactly once
    lengths = [len(x) for x in llm.tok(texts, add_special_tokens=False)["input_ids"]]
    for b in batches:
        assert len(b) <= 4 and len(b) * max(lengths[i] for i in b) <= 120 or len(b) == 1


def test_clip_bounds_characters_not_just_words():
    blob = "x" * 50_000  # one 'word', like a stack trace or a base64 attachment
    clipped = clip_words(f"error {blob} end", 300)
    assert len(clipped) <= 300 * 8 + 4 and clipped.endswith(" ...")
    msgs = judge_messages(blob, [{"doc_id": "a", "title": "t", "text": blob}])
    assert len(msgs[1]["content"]) < (config.QUESTION_MAX_WORDS + config.EVIDENCE_MAX_WORDS) * 8 + 500
