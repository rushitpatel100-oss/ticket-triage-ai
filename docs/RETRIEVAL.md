# Retrieval: finding the right knowledge article

Before the system can decide whether it can resolve a ticket by itself, it has to find the knowledge that
might answer it. This step compares four ways of doing that on two real knowledge bases. All figures come
from `results/retrieval_metrics.json`, `results/retrieval_paired.json` and
`results/retrieval_diagnostics.json` (rebuild with `notebooks/retrieval_colab.ipynb`).

## Set-up

| | TechQA | Stack Exchange |
|---|---|---|
| Knowledge base | 28,481 IBM Technotes (89,614 passages) | 32,025 accepted answers (38,350 passages) |
| Questions with an answer | 450 dev, 160 test | 3,154 dev, 1,615 test (2,000 questions sampled per split) |
| Median question length | 44 words | 72 words |

Methods:
- **BM25**: keyword search. Error codes, versions and file names are kept whole as well as split, and
  plain words are stemmed.
- **Dense**: `BAAI/bge-small-en-v1.5` embeddings. Documents are split into 200-word passages (50-word
  overlap) and a document scores as its best passage.
- **Hybrid**: reciprocal rank fusion of BM25 and dense (k = 60).
- **Hybrid + reranker**: the top 30 hybrid results re-scored by a cross-encoder that reads question and
  passage together: `cross-encoder/ms-marco-MiniLM-L-6-v2` (trained on web search) and `BAAI/bge-reranker-base`.

Methods are compared on **dev** (train + validation questions); **test** is reported but was not used to
choose anything. Differences are checked with a **paired bootstrap**: every method answers the same
questions, so we resample questions and look at the difference, which is much more sensitive than
comparing two separate intervals.

## Results

| | TechQA dev MRR@10 | TechQA test MRR@10 | Stack Exchange dev MRR@10 | Stack Exchange test MRR@10 | ms per question (T4) |
|---|---|---|---|---|---|
| BM25 | 0.49 | 0.51 | 0.27 | 0.28 | 1 |
| Dense | 0.54 | 0.56 | **0.42** | **0.43** | 3–4 |
| Hybrid | 0.54 | **0.58** | 0.37 | 0.38 | 4–6 |
| Hybrid + MiniLM reranker | 0.52 | 0.53 | 0.24 | 0.25 | 71–142 |
| Hybrid + BGE reranker | 0.54 | 0.55 | 0.27 | 0.29 | 213–438 |

Paired difference in dev MRR@10 against hybrid (95% interval):

| | TechQA | Stack Exchange |
|---|---|---|
| BM25 | −0.047 (−0.069 to −0.026), worse | −0.094 (−0.103 to −0.085), worse |
| Dense | +0.005 (−0.023 to +0.033), no clear difference | **+0.046 (+0.035 to +0.058), better** |
| MiniLM reranker | −0.024 (−0.055 to +0.005), no clear difference | −0.125 (−0.137 to −0.112), worse |
| BGE reranker | −0.002 (−0.031 to +0.029), no clear difference | −0.095 (−0.109 to −0.082), worse |

![retrieval comparison](../results/retrieval_comparison.png)

## What we learned

**1. Embeddings beat keywords on both knowledge bases.** On dev, dense search is ahead of BM25 by
+0.052 MRR on TechQA (paired 95% interval +0.017 to +0.087) and +0.140 on Stack Exchange (+0.127 to
+0.155), for about 1.5 to 3 ms more per question. On TechQA's small test set the gap is similar (+0.048)
but its interval crosses zero (−0.009 to +0.104).

**2. Adding keywords (hybrid) did not help.** On TechQA hybrid and dense are tied; on Stack Exchange BM25
pulls the fusion down. Reciprocal rank fusion gives both lists equal say, which hurts when one list is much
weaker.

**3. Off-the-shelf rerankers did not help, and were about 19 to 75 times slower than hybrid.** On TechQA
there is no clear difference either way (every paired interval includes zero). On Stack Exchange they are
clearly worse: the BGE reranker moved the correct answer **down for 1,021** of the 1,964 dev questions
where hybrid had found it, and up for only 469. Of the 926 questions hybrid ranked first, 280 dropped below
5th place.

**4. Long questions explain part of it.** A cross-encoder reads question and passage through one 512-token
window, so a long ticket (with logs or stack traces) leaves little room for the answer. Embedding models
read each separately.

![accuracy by question length](../results/retrieval_by_question_length.png)

- Stack Exchange: the BGE reranker falls from 0.36 MRR on the shortest quarter of questions to **0.14** on
  the longest, while hybrid stays between 0.35 and 0.39.
- TechQA: the BGE reranker is the **best** method on the shortest quarter (0.60 against 0.48 for hybrid)
  and the worst on the longest (0.32 against 0.44).

A controlled test reranked the same 20 hybrid candidates twice, once with the full question and once with
only its first 64 words:

| | Full question | First 64 words | Hybrid order (same candidates) |
|---|---|---|---|
| TechQA, BGE reranker | 0.526 | 0.539 | 0.540 |
| TechQA, MiniLM reranker | 0.522 | 0.527 | 0.540 |
| Stack Exchange, BGE reranker | 0.294 | 0.327 | 0.376 |
| Stack Exchange, MiniLM reranker | 0.262 | 0.259 | 0.376 |

On TechQA, shortening the question lets the BGE reranker almost exactly match hybrid (0.539 against 0.540).
On Stack Exchange it recovers about 40% of the gap for BGE and nothing for MiniLM, so there length is part
of the story, not all of it. This test differs from the main run in three ways, so its numbers are not
directly comparable with the main table: it scores whole documents (truncated by the model) rather than
the best three passages, reranks 20 candidates rather than 30, and on Stack Exchange uses a random sample of
1,000 dev questions (hence hybrid's 0.376 here against 0.369 on all dev questions).

**5. The rest looks like genuine topical mistakes.** In the examples the reranker chose answers that share
the question's words but solve a different problem: asked how to install a LaTeX package through
Synaptic, it preferred an answer about pinning a package version in Synaptic.

## Caveats

- **The embedding model has seen Stack Exchange.** BGE's training data includes Stack Exchange text in
  pre-training and Stack Exchange duplicate questions in fine-tuning
  ([C-Pack paper](https://arxiv.org/abs/2309.07597)), so its lead on Stack Exchange is probably optimistic.
  TechQA's IBM Technotes are the fairer test, and there dense and hybrid are tied.
- Stack Exchange has exactly one "correct" answer per question, but other answers in the base may solve it
  too, which lowers every method's score.
- TechQA's test set is small (160 answerable questions), so its intervals are wide (about ±0.07 hit@5).
- Rankings were computed in half precision on a T4 GPU; timings include encoding the question but not
  building the index.

## Decision for the next step

- **Use dense retrieval** to find candidate articles: best or tied on both bases, at 3 to 4 ms per question.
- **Keep BM25 and reranker scores as signals, not as the ranking.** A reranker that orders results poorly
  may still help tell "nothing relevant here" from "this is the answer", which is what the
  resolve-or-escalate model needs. Phase 3 tests that.
- The length problem suggests a fix that fits the LLM step later: have the LLM rewrite a long ticket into a
  short search query before retrieval and reranking.
