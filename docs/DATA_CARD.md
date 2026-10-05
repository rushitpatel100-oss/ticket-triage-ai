# Data card

v2 combines four datasets, because no public dataset has real tickets, routing labels and resolutions
together. Each one is used only for what it is genuinely real for. Figures below come from
`results/real_data_summary.json` (rebuild with `python -m src.sources`, or the notebook
`notebooks/prepare_real_data.ipynb`).

| Dataset | Real? | Used for | Licence |
|---|---|---|---|
| [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | Synthetic | Routing: team, priority, ITIL type | CC BY-NC 4.0 |
| [nvidia/TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) (from IBM's [TechQA](https://github.com/IBM/techqa)) | Real questions and documents | Resolve-or-escalate decision; retrieval | Apache-2.0 |
| [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) (Super User, Ask Ubuntu) | Real | Large knowledge base; robustness | CC BY-SA 4.0 |
| [UCI incident management event log](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log) | Real (ServiceNow) | Operating figures for the business simulation | CC BY 4.0 |

All text tables share one layout: `docs` (doc_id, source, title, text, url) and `queries` (query_id, source,
text, answerable, gold_doc_ids, gold_answer, split).

## TechQA

- **910 questions** from IBM's developer support forums. **300 (33%) have no answer** in the documents
  (`is_impossible`), the realistic "AI can't solve this, escalate" case.
- **Knowledge base: 28,481 IBM Technotes**, median 385 words. All 496 Technotes referenced as answers are in it.
- **Split:** TechQA's official DEV questions are our test set (310); its TRAIN questions are split 85/15
  into train (510) and validation (90).
- **The test set is harder than training:** 48% of test questions are unanswerable against 25% in
  training. A threshold tuned on the training distribution would auto-resolve too often on the test set,
  so v2 reports results on both and treats this shift as part of the problem, the way a service desk
  sees its mix of tickets change.

## Stack Exchange (Super User, Ask Ubuntu)

- **40,000 questions** with an accepted answer: 20,000 sampled per site from the first two data files of
  each site (seed 42). Median question length 72 words.
- **Knowledge base: 32,025 accepted answers.** A document is the accepted answer only. The question is
  the query, so putting it in the document too would let retrieval find the query itself.
- **Knowledge gaps on purpose:** for 20% of questions the accepted answer is removed from the knowledge base
  and the question is labelled `answerable = False`, to simulate new issues with no article yet. Another
  answer in the base may still cover such a question, so these labels are somewhat noisy.
- Queries are the question body only (this release has no separate title). HTML is stripped, so images
  and screenshots in questions are lost, which real tickets also suffer from.
- Licence requires attribution: every document keeps a link to its original answer (`url`).

## ServiceNow incident event log (UCI)

Real process data from an IT company's ServiceNow instance; **no ticket text**. 141,712 events collapsed
to **24,918 incidents** (one row per incident, final state from its latest event).

| Figure | Value |
|---|---|
| Reported by phone | 99% (24,688) |
| Priority "3 - Moderate" | 94% (23,466) |
| Reassigned at least once (group or analyst) | 45.6% |
| Handled by more than one group | 38.4% |
| Reopened at least once | 1.1% |
| `made_sla` = True | 63.4% |
| Knowledge article used | 14.3% |
| Median time to resolution | 22.1 hours |
| Median time to resolution, not reassigned / reassigned | 0.7 h / 79.9 h |
| Median time to resolution, no knowledge / knowledge used | 12.2 h / 77.5 h |
| Median time to resolution by priority (1 Critical to 4 Low) | 80.2 / 38.2 / 21.6 / 5.0 h |

How to read these:
- **They are associations, not effects.** Reassigned tickets taking far longer does not prove the
  reassignment caused the delay; harder tickets are both more likely to be reassigned and slower to fix.
  The same goes for knowledge use, and for critical incidents resolving more slowly than low-priority ones.
- The UCI description of `made_sla` is ambiguous, so we report the raw share of True values.
- 1,556 incidents (6.2%) have no usable resolution time (not resolved, or resolved before opened).
- The organisation is anonymised and the log covers one company, so the rates are one real example,
  not an industry benchmark.

## Known gaps

- Routing labels are synthetic; results on real text come from TechQA (small) and Stack Exchange.
- TechQA and Stack Exchange are forum questions, not internal help desk tickets: they are more
  technical and longer than typical "I can't log in" tickets.
- Licences differ (non-commercial, share-alike), so data stays out of the repository and is downloaded
  by the scripts. Code is MIT.
