#!/usr/bin/env bash
# Second GPU batch: the Stack Exchange answer step, routing calibration (E1) and robustness (E5).
#
# On Kaggle's two T4s the two halves run side by side (about 3 hours for the longer one):
#   bash scripts/gpu_run_more.sh setup
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/gpu_run_more.sh stackexchange > se.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/gpu_run_more.sh routing > routing.log 2>&1 &
# With one GPU:  bash scripts/gpu_run_more.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() { echo; echo "=== $1 ($(date -u +%H:%M:%S) UTC) ==="; }
# Hide progress bars and the machine-readable JSON (it is in results/); never fail just because nothing is left
quiet() { grep -v -e Batches -e 'it/s' -e 's/it' -e _START -e _END -e '^{' || true; }
RERANKERS="cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base"

setup() {
    git log --oneline -1
    nvidia-smi --query-gpu=name,memory.total --format=csv
    pip install -q "sentence-transformers>=3.0" "nltk>=3.8" "transformers>=4.51" "pyarrow>=14"
    step "real data and the synthetic routing tickets"
    python -m src.sources --only techqa stackexchange | tail -3
    python -m src.data | tail -3
}

stackexchange() {
    step "SE 1/5 retrieval (2,000 questions per split, as before)"
    python -m src.retrieval --datasets stackexchange --fp16 --rerankers $RERANKERS --max-queries-per-split 2000 \
        2>&1 | quiet | tail -8
    step "SE 2/5 decision, retrieval signals only"
    python -m src.decision --datasets stackexchange | quiet
    step "SE 3/5 answer step"
    python -m src.answer --datasets stackexchange --skip-redteam 2>&1 | quiet | tail -6
    step "SE 4/5 decision with the LLM signal"
    python -m src.decision --datasets stackexchange --llm-judge | quiet
    step "SE 5/5 guardrails: faithfulness and datamark utility"
    python -m src.answer.guardrails --datasets stackexchange 2>&1 | quiet | tail -6
}

routing() {
    step "E1 1/3 TF-IDF baselines"
    python -m src.baseline | tail -2
    step "E1 2/3 calibration: DistilBERT x 3 seeds x 3 tasks"
    python -m src.calibration 2>&1 | quiet | { grep -e seed -e TF-IDF -e Error -e error || true; } | tail -20
    step "E5 3/3 robustness"
    python -m src.robustness 2>&1 | quiet | { grep -v Using || true; } | tail -40
}

case "${1:-all}" in
    setup) setup ;;
    stackexchange) stackexchange ;;
    routing) routing ;;
    all) setup; routing; stackexchange ;;
    *) echo "usage: $0 setup|stackexchange|routing|all"; exit 2 ;;
esac
step "done: ${1:-all}"
touch "done_${1:-all}"
