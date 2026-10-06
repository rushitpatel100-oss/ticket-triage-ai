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

    def _batches(self, texts, batch_size: int | None = None):
        batch_size = batch_size or self.batch_size
        order = np.argsort([len(t) for t in texts], kind="stable")  # similar lengths together = less padding
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            yield idx, enc

    def judge(self, tickets, evidence) -> tuple[np.ndarray, np.ndarray]:
        return self.yes_probability([judge_messages(t, e) for t, e in zip(tickets, evidence)])

    def answer(self, tickets, evidence, max_new_tokens: int = config.ANSWER_MAX_TOKENS) -> list[str]:
        return self.generate([answer_messages(t, e) for t, e in zip(tickets, evidence)], max_new_tokens)

    def yes_probability(self, messages_list) -> tuple[np.ndarray, np.ndarray]:
        """P(first word is Yes | first word is Yes or No) for each conversation, plus the Yes+No probability mass."""
        torch = self.torch
        texts = self._texts(messages_list)
        p_yes, mass = np.zeros(len(texts)), np.zeros(len(texts))
        for idx, enc in self._batches(texts):
            with torch.no_grad():
                try:
                    logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**enc, use_cache=False).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            yes = probs[:, self.yes_ids].sum(-1)
            no = probs[:, self.no_ids].sum(-1)
            p_yes[idx] = (yes / (yes + no + 1e-12)).cpu().numpy()
            mass[idx] = (yes + no).cpu().numpy()  # how much of the model's first word was Yes/No at all
        return p_yes, mass

    def generate(self, messages_list, max_new_tokens: int = config.ANSWER_MAX_TOKENS) -> list[str]:
        torch = self.torch
        texts = self._texts(messages_list)
        out = [""] * len(texts)
        for idx, enc in self._batches(texts, max(self.batch_size // 2, 1)):  # generation keeps a cache: smaller batches
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, text in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                out[i] = text.strip()
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
