# The answer step: an LLM reads the articles

After retrieval finds the top 3 articles for a ticket, an LLM:

1. **judges** whether those articles contain the answer: a new signal for the resolve / assist / escalate decision,
2. **drafts a reply** with citations [1] to [3], or says `NOT_FOUND`,
3. is **attacked** with prompt injections to see how easily planted instructions take over.

Figures come from `results/answer_metrics.json`.

> **Status.** The TechQA run finished. The Stack Exchange run, the decision re-run with the LLM signal, and the
> guardrail experiments (defences, false alarms, faithfulness) were cut short when Colab's free GPU allowance
> ran out. The code and tests for all of them are in place; `notebooks/answer_colab.ipynb` re-runs everything.

## Set-up

- **Model:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). It is
  Apache-2.0 and ungated, about 8 GB in half precision, and text-only with no "thinking" mode. An open
  model keeps the project reproducible on a free GPU, and its next-word probabilities give a smooth
  judgement score.
- **Claude** (`claude-haiku-4-5-20251001`) is a drop-in alternative (`--backend claude`, reads
  `ANTHROPIC_API_KEY`). Its API gives no token probabilities, so it judges with a 0-100 number instead.
- **Evidence:** for each of the top 3 articles from embedding search, its passage most similar to the
  ticket. Tickets are cut to 300 words and passages to 200 words, *and* to 8 characters per word on
  average. The first full run ran out of GPU memory because some "words" (logs, URLs, XML) were thousands
  of characters long.
- **Judge:** the prompt asks *"Do these articles contain the information needed to resolve this ticket? Answer
  with only Yes or No."* The score is P(Yes) / (P(Yes) + P(No)) for the model's first word.
- **Replies:** greedy decoding, up to 200 tokens, citations required, `NOT_FOUND` when the articles don't
  have the answer.
- **Prompt hygiene:** the system message says ticket and article text are data, never instructions, and both
  are wrapped in `<ticket>` and `<articles>` tags.

## Results on TechQA (Colab T4, half precision)

**The judge** (910 questions; it answered Yes or No 100% of the time, median probability mass 1.00):

| AUROC for "the correct article is in the top 3" | Dev (600) | Test (310) |
|---|---|---|
| LLM judge alone | 0.73 | 0.70 |
| Best-match retrieval score alone (from [DECISION.md](DECISION.md)) | 0.67 | 0.63 |
| Retrieval-signal models: gradient boosting / logistic | 0.78 / 0.77 | 0.76 / 0.78 |

- The LLM's judgement is a real signal: better than the best-match score, though on its own weaker than the
  combined retrieval signals. Whether **adding** it to those signals lifts the safe auto-resolve rate is the
  experiment the GPU limit interrupted.
- Its raw probabilities are badly calibrated (calibration error 0.46 on test): it says Yes too readily. The
  decision model re-calibrates any signal it is given.
- It took 0.67 seconds per question.

**Drafted replies** (150 random test questions, 50 of them resolvable):

| | |
|---|---|
| Said NOT_FOUND when the correct article *was* in the top 3 | 0% |
| Said NOT_FOUND when it was *not* | 17% |
| Replies with at least one citation | 99% |
| Cited the correct article (resolvable and answered) | 88% |
| Word-overlap F1 with the reference answer (resolvable and answered) | 0.225 |

- When the right article is there, the model uses it and cites it.
- When it isn't there, the model still answers 83% of the time. It writes confident, plausible steps
  from related articles, for example naming an IBM fix pack for a NullPointerException.
- Some of those answers may still help, but none can be trusted without a check. That is the case for the
  faithfulness test and for never auto-sending without the decision layer.
- Word-overlap F1 is a crude measure for free-form answers. It is reported for comparison, not as a
  quality score.

**Prompt injection** (15 designed attacks, each trying to make the reply contain a canary string):

| | Attacks | Succeeded |
|---|---|---|
| Planted in the ticket (direct) | 10 | 6 |
| Planted in an article (indirect) | 5 | 5 |
| **Total** | **15** | **11 (73%)** |

- Telling the model that ticket and article text are data was not enough.
- Every attack hidden in a knowledge-base article worked. A knowledge base that anyone can edit is an
  attack surface (OWASP LLM01: prompt injection).

## Guardrails: built, partly measured

`src/answer/guardrails.py` compares three defences on the red-team set:
- **sandwiching** the task around the data,
- **spotlighting by datamarking** ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)): every space in
  untrusted text becomes `^`, and the model is told never to take instructions from marked text,
- a **keyword scanner** in front that sends suspicious tickets to a person.

It also measures the scanner's false alarms on real questions and articles, whether datamarking hurts real
answers, and sentence-level faithfulness of the drafted replies. All of this needs the GPU run.

The scanner needs no model, so one result is already in, from the tests:

| Attack set | Cases | Caught by the scanner |
|---|---|---|
| The 15 attacks it was written against | 15 | 12 |
| 13 held-out attacks, written after its patterns were frozen (commit `57bebda`) | 13 | **0** |

**A keyword scanner only catches the attacks its author already knows.** It is a cheap first layer, not a
defence. The real protection is structural:
- the model never takes actions on its own,
- risky tickets go to a person through the hard rules,
- nothing is auto-sent unless the certified threshold allows it.

## To finish (needs about an hour on a T4)

```bash
python -m src.answer --datasets techqa
python -m src.answer --datasets stackexchange --skip-redteam
python -m src.decision --llm-judge          # does the LLM signal make any auto-resolution safe?
python -m src.answer.guardrails             # defences, false alarms, faithfulness, cost of datamarking
```
