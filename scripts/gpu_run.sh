#!/usr/bin/env bash
# Re-runs every GPU experiment for TechQA on a free GPU (Kaggle or Colab T4), in order:
#   real data -> retrieval with both rerankers -> decision (retrieval signals only)
#   -> LLM answer step -> decision with the LLM signal -> guardrails.
# Stack Exchange tables are built too: the guardrail run measures scanner false alarms on them.
# Takes about 80 minutes on one T4. Usage, from the repository root:  bash scripts/gpu_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

git log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q "sentence-transformers>=3.0" "nltk>=3.8" "transformers>=4.51" "pyarrow>=14"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
# Hide progress bars and the machine-readable JSON (it is in results/); never fail just because nothing is left
quiet() { grep -v -e Batches -e _START -e _END -e '^{' || true; }

step "1/6 real data"
python -m src.sources --only techqa stackexchange
step "2/6 retrieval"
python -m src.retrieval --datasets techqa --fp16 \
    --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base 2>&1 | quiet | tail -12
step "3/6 decision, retrieval signals only"
python -m src.decision --datasets techqa | quiet
step "4/6 answer step"
python -m src.answer --datasets techqa 2>&1 | quiet | tail -8
step "5/6 decision with the LLM signal"
python -m src.decision --datasets techqa --llm-judge | quiet
step "6/6 guardrails"
python -m src.answer.guardrails --datasets techqa stackexchange 2>&1 | quiet | tail -8

step "done"
sha256sum results/decision_metrics.json results/decision_metrics_llm.json results/answer_metrics.json \
    results/guardrail_metrics.json results/retrieval_metrics.json
