"""LLM backends with the same two calls:

    judge(tickets, evidence)  -> probability that the articles contain the answer, per ticket
    answer(tickets, evidence) -> a drafted reply with [n] citations, or NOT_FOUND

OpenModel runs a Hugging Face model locally (free; its judgement is read from the probability it gives to
"Yes" versus "No" as the first word, so it is a smooth score rather than a yes/no).
ClaudeModel calls the Claude API (better answers; needs ANTHROPIC_API_KEY; the judgement is a 0-100 number).
"""

import os
import re

import numpy as np

from src import config
from src.answer.prompts import SYSTEM, answer_messages, judge_messages, user_message


def yes_no_scores(logits, yes_ids, no_ids) -> tuple[np.ndarray, np.ndarray]:
    """P(Yes | Yes or No) and the Yes+No probability mass from next-token logits (rows = prompts).

    Worked out in log space: a confident model gives "No" a probability near 1e-9, so a float32 ratio of
    probabilities rounds to exactly 1.0 and every confident Yes ties. The log-odds keep their order.
    """
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    yes = torch.logsumexp(logp[:, yes_ids], dim=-1)
    no = torch.logsumexp(logp[:, no_ids], dim=-1)
    log_odds = (yes - no).double().cpu().numpy()
    mass = torch.logsumexp(torch.stack([yes, no], dim=-1), dim=-1).exp().double().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-log_odds)), mass


class OpenModel:
    def __init__(self, name: str = config.OPEN_LLM, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.batch_size, self.torch = name, batch_size, torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"  # so the last position of every row is the end of its prompt
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        yes = self._first_token_ids(["Yes", "yes", "YES"])
        no = self._first_token_ids(["No", "no", "NO"])
        shared = yes & no  # e.g. a variant the tokenizer maps to the same piece: it says nothing either way
        self.yes_ids, self.no_ids = sorted(yes - shared), sorted(no - shared)
        if not self.yes_ids or not self.no_ids:
            raise ValueError(f"cannot find distinct Yes/No tokens for {name}")

    def _first_token_ids(self, words) -> set[int]:
        ids = set()
        for w in words:
            for variant in (w, " " + w):
                tokens = self.tok.encode(variant, add_special_tokens=False)
                if tokens and tokens[0] != self.tok.unk_token_id:
                    ids.add(tokens[0])
        return ids

    def _texts(self, messages_list) -> list[str]:
        return [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]

    def _batches(self, texts, max_items: int, token_budget: int):
        """Index batches of similar length, each with at most max_items prompts and token_budget padded tokens."""
        lengths = [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]
        batch, longest = [], 0
        for i in np.argsort(lengths, kind="stable"):  # similar lengths together = less padding
            longest_if_added = max(longest, lengths[i])
            if batch and (len(batch) >= max_items or (len(batch) + 1) * longest_if_added > token_budget):
                yield batch
                batch, longest_if_added = [], lengths[i]
            batch.append(int(i))
            longest = longest_if_added
        if batch:
            yield batch

    def _run(self, idx: list[int], texts: list[str], step) -> None:
        """Run `step(idx, encoded)`; if the GPU runs out of memory, split the batch in half and retry."""
        torch = self.torch
        enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
        try:
            with torch.no_grad():
                step(idx, enc)
        except torch.cuda.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            half = len(idx) // 2
            self._run(idx[:half], texts, step)
            self._run(idx[half:], texts, step)

    def judge(self, tickets, evidence) -> tuple[np.ndarray, np.ndarray]:
        return self.yes_probability([judge_messages(t, e) for t, e in zip(tickets, evidence)])

    def answer(self, tickets, evidence, max_new_tokens: int = config.ANSWER_MAX_TOKENS) -> list[str]:
        return self.generate([answer_messages(t, e) for t, e in zip(tickets, evidence)], max_new_tokens)

    def yes_probability(self, messages_list, token_budget: int = 8000) -> tuple[np.ndarray, np.ndarray]:
        """P(first word is Yes | first word is Yes or No) for each conversation, plus the Yes+No probability mass."""
        texts = self._texts(messages_list)
        p_yes, mass = np.zeros(len(texts)), np.zeros(len(texts))

        def step(idx, enc):
            try:
                logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = self.model(**enc, use_cache=False).logits[:, -1, :]
            # mass: how much of the model's first word was Yes/No at all
            p_yes[idx], mass[idx] = yes_no_scores(logits, self.yes_ids, self.no_ids)

        for idx in self._batches(texts, self.batch_size, token_budget):
            self._run(idx, texts, step)
        return p_yes, mass

    def generate(self, messages_list, max_new_tokens: int = config.ANSWER_MAX_TOKENS,
                 token_budget: int = 6000) -> list[str]:
        texts = self._texts(messages_list)
        out = [""] * len(texts)

        def step(idx, enc):
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, text in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                out[i] = text.strip()

        # generation keeps a cache that grows with every new token, so batches are smaller than for judging
        for idx in self._batches(texts, max(self.batch_size // 2, 1), token_budget):
            self._run(idx, texts, step)
        return out


NUMERIC_JUDGE = ("How likely is it that these articles contain the information needed to resolve this ticket? "
                 "Reply with only a whole number from 0 to 100.")


class ClaudeModel:
    """Claude through the official SDK. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, name: str = config.CLAUDE_MODEL):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first (never put the key in code).")
        from anthropic import Anthropic

        self.name, self.client = name, Anthropic()

    def _ask(self, content: str, max_tokens: int, system: str = SYSTEM) -> str:
        reply = self.client.messages.create(model=self.name, max_tokens=max_tokens, system=system,
                                            messages=[{"role": "user", "content": content}])
        return "".join(block.text for block in reply.content if getattr(block, "type", "") == "text").strip()

    def judge(self, tickets, evidence) -> tuple[np.ndarray, np.ndarray]:
        p = []
        for t, e in zip(tickets, evidence):
            text = self._ask(user_message(t, e, NUMERIC_JUDGE), max_tokens=5)
            m = re.search(r"\d+", text)
            p.append(min(int(m.group(0)), 100) / 100 if m else 0.5)
        return np.array(p), np.ones(len(p))

    def answer(self, tickets, evidence, max_new_tokens: int = config.ANSWER_MAX_TOKENS) -> list[str]:
        return self.generate([answer_messages(t, e) for t, e in zip(tickets, evidence)], max_new_tokens)

    def generate(self, messages_list, max_new_tokens: int = config.ANSWER_MAX_TOKENS) -> list[str]:
        return [self._ask(m[-1]["content"], max_new_tokens, system=m[0]["content"]) for m in messages_list]

    def yes_probability(self, messages_list) -> tuple[np.ndarray, np.ndarray]:
        """A Yes/No question asked as a 0-100 likelihood (the API does not expose token probabilities)."""
        p = []
        for m in messages_list:
            text = self._ask(m[-1]["content"] + "\nInstead of Yes or No, reply with only a whole number from 0 to 100 "
                             "for how likely the answer is Yes.", max_tokens=5, system=m[0]["content"])
            n = re.search(r"\d+", text)
            p.append(min(int(n.group(0)), 100) / 100 if n else 0.5)
        return np.array(p), np.ones(len(p))
