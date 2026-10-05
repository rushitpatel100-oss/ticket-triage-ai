"""Load the trained models and classify new tickets.

Model lookup for each task, in order:
  1. Environment variable MODEL_<TASK> (e.g. MODEL_QUEUE=your-user/ticket-triage-queue on the HF Hub)
  2. A fine-tuned transformer saved in models/<task>/
  3. The TF-IDF baseline saved in models/baseline_<task>.joblib

Example:
    from src.predict import TicketClassifier
    clf = TicketClassifier()
    clf.predict("VPN keeps disconnecting", "Since this morning the VPN drops every 10 minutes...")
"""

import os

import joblib

from src import config


class TicketClassifier:
    def __init__(self, tasks=None):
        self.models = {}
        for task in tasks or config.TASKS:
            loaded = self._load(task)
            if loaded:
                self.models[task] = loaded

    @staticmethod
    def _load(task: str):
        source = os.getenv(f"MODEL_{task.upper()}")
        local_dir = config.MODELS_DIR / task
        if source or (local_dir / "config.json").exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            source = source or str(local_dir)
            model = AutoModelForSequenceClassification.from_pretrained(source).eval()
            return {"kind": "transformer", "name": "DistilBERT", "model": model,
                    "tokenizer": AutoTokenizer.from_pretrained(source)}

        baseline = config.MODELS_DIR / f"baseline_{task}.joblib"
        if baseline.exists():
            return {"kind": "baseline", "name": "TF-IDF baseline", "model": joblib.load(baseline)["model"]}
        return None

    @property
    def tasks(self) -> list[str]:
        return list(self.models)

    def backend(self, task: str) -> str:
        return self.models[task]["name"]

    def predict(self, subject: str, body: str = "") -> dict[str, dict[str, float]]:
        """Return {task: {label: probability}} with labels sorted from most to least likely."""
        text = f"{subject or ''}\n\n{body or ''}".strip()
        return {task: self._scores(entry, text) for task, entry in self.models.items()}

    @staticmethod
    def _scores(entry: dict, text: str) -> dict[str, float]:
        if entry["kind"] == "transformer":
            import torch

            inputs = entry["tokenizer"](text, truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(entry["model"](**inputs).logits, dim=-1)[0].tolist()
            id2label = entry["model"].config.id2label
            scores = {id2label[i]: p for i, p in enumerate(probs)}
        else:
            model = entry["model"]
            scores = dict(zip(model.classes_, model.predict_proba([text])[0].tolist()))
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
