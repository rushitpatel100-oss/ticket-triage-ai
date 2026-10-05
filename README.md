# Support Ticket Triage with DistilBERT

[![tests](https://github.com/rushitpatel100-oss/ticket-triage-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/rushitpatel100-oss/ticket-triage-ai/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushitpatel100-oss/ticket-triage-ai/blob/main/notebooks/train_on_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

An NLP model that reads a support ticket and decides **which team should handle it**, **how urgent it is**
and **what ITIL ticket type it is** (Incident, Request, Problem or Change). Tickets the model is unsure about
are flagged for a human instead of being routed automatically.

Built with Hugging Face Transformers (fine-tuned DistilBERT), compared against classic machine-learning
baselines, and served through a Gradio web demo.

## Why I built this

I work in first-line IT support, where a big part of the job is reading incoming requests (login problems,
access issues, platform errors) and deciding where they should go. When a ticket lands in the wrong queue,
it bounces between teams and the user waits longer. This project asks: **how much of that first triage step
can a small language model do reliably, and how do we keep a human in the loop when it can't?**

## What it does

```mermaid
flowchart LR
    T["New ticket<br/>subject + description"] --> M["Fine-tuned DistilBERT<br/>(one model per task)"]
    M --> Q["Team<br/>10 queues"]
    M --> P["Priority<br/>urgency level"]
    M --> Y["ITIL type<br/>Incident / Request / Problem / Change"]
    Q & P & Y --> C{"Confidence<br/>≥ 60%?"}
    C -- yes --> A["Auto-route"]
    C -- no --> H["Human review"]
```

## Results

<!-- RESULTS:START -->
_Results appear here after training. See [Quick start](#quick-start)._
<!-- RESULTS:END -->

**Why macro F1?** Some teams receive far fewer tickets than others. Accuracy can look good while a model
ignores the small teams completely; macro F1 gives every class equal weight, so it exposes that.

## How it works

1. **Data** (`src/data.py`): about 60,000 synthetic support tickets from the
   [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
   dataset. I keep the English tickets, join subject and body into one text, remove exact duplicates (so the
   same ticket can't appear in both training and test data), and make a stratified 80/10/10 train/validation/test split.
2. **Baselines** (`src/baseline.py`): a majority-class model (the "do nothing" floor) and TF-IDF with
   logistic regression, with its regularisation strength tuned on the validation set.
3. **Fine-tuning** (`src/train.py`): `distilbert-base-uncased` fine-tuned separately for each task with the
   Hugging Face `Trainer`. The best epoch is chosen by validation macro F1.
4. **Evaluation** (`src/evaluate.py`): accuracy, macro and weighted F1, a confusion matrix per task, and a CSV of
   the model's most confident mistakes for error analysis. Everything is measured on the untouched test set.
5. **Demo** (`app.py`): a Gradio app that shows the top predictions with confidence scores, the ITIL next
   step for the predicted ticket type, and a "needs human review" flag below 60% confidence.

## Quick start

### Option A: train on Google Colab (free GPU, recommended)

Click the **Open in Colab** badge above, switch the runtime to a T4 GPU and run the cells top to bottom.
The notebook trains everything, writes the results into this README and lets you download them.

### Option B: run on your own computer

```bash
git clone https://github.com/rushitpatel100-oss/ticket-triage-ai.git
cd ticket-triage-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data        # download and split the data
python -m src.baseline    # train the baselines (a few minutes on a laptop)
python -m src.train       # fine-tune DistilBERT (needs a GPU to be practical)
python -m src.report      # update the results table in this README
python app.py             # open http://127.0.0.1:7860
```

The demo works without a GPU: if no fine-tuned model is found, it falls back to the TF-IDF baseline.

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests build a tiny dataset and a tiny model, so they run offline in seconds. They also run automatically
on every push through GitHub Actions.

### Live demo on Hugging Face Spaces (optional)

1. Upload the trained models to the Hub (last section of the Colab notebook).
2. Create a new Space with the **Gradio** SDK and upload `app.py`, `requirements.txt` and the `src/` folder.
3. In the Space settings, add the variables `MODEL_QUEUE`, `MODEL_PRIORITY` and `MODEL_TYPE`,
   e.g. `MODEL_QUEUE = your-hf-username/ticket-triage-queue`.

## Project structure

```
ticket-triage-ai/
├── app.py                      # Gradio demo
├── notebooks/
│   └── train_on_colab.ipynb    # end-to-end training on a free GPU
├── src/
│   ├── config.py               # all settings in one place
│   ├── data.py                 # download, clean, split
│   ├── baseline.py             # majority class + TF-IDF/logistic regression
│   ├── train.py                # DistilBERT fine-tuning
│   ├── evaluate.py             # metrics, confusion matrices, error analysis
│   ├── predict.py              # TicketClassifier used by the demo
│   └── report.py               # writes the results table into this README
├── tests/test_pipeline.py      # offline end-to-end tests
├── results/                    # metrics, charts and error samples (committed)
├── requirements.txt
└── .github/workflows/tests.yml # CI
```

## Design decisions

- **DistilBERT** is 40% smaller and about 60% faster than BERT while keeping most of its accuracy, so it
  trains in minutes on a free GPU and can run on a CPU in the demo.
- **One model per task** keeps the code simple and lets each task pick its own best epoch.
  A shared multi-task model is a natural next step.
- **A baseline first.** If TF-IDF gets close to DistilBERT, the simpler model may be the better choice in
  production (cheaper, faster, easier to explain). The results table answers this with numbers.
- **Human in the loop.** In a real service desk a wrong auto-route costs more than a manual check, so
  low-confidence predictions are sent to a person.

## Limitations

- The tickets are **synthetic**. Real tickets are messier (signatures, forwarded threads, screenshots), so
  scores on real data would likely be lower.
- English only. The dataset also has German tickets, which a multilingual model could handle.
- The 60% review threshold is a starting point, not tuned against the cost of a wrong route.

## Next steps

- Train a single multi-task model with three output heads
- Calibrate the confidence scores and pick the review threshold from the validation data
- Add German with a multilingual model such as `distilbert-base-multilingual-cased`
- Connect to a ticketing tool's API (e.g. Jira) to suggest a queue when a ticket is created

<!--
## What I found
After training, read results/errors_*_transformer.csv and the confusion matrices, then replace this
comment with 3-4 bullet points, for example:
- Which teams get confused with each other, and why that makes sense (or doesn't)
- Whether DistilBERT beat the TF-IDF baseline, and by how much on each task
- Whether some "mistakes" are actually wrong or ambiguous labels in the data
-->

## Dataset and licence

Data: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
on Hugging Face, a synthetic dataset licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The data is downloaded by the scripts; only small
samples of misclassified tickets are kept in `results/` for error analysis.

Code: MIT licence, see [LICENSE](LICENSE).
