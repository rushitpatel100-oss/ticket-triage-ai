"""Project-wide settings. Change values here rather than inside the scripts."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# --- Data ------------------------------------------------------------------
# Public synthetic dataset (CC BY-NC 4.0): https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
DATASET_ID = "Tobi-Bueck/customer-support-tickets"
LANGUAGE = "en"  # the dataset also has German tickets; we keep English only

# What we predict. Key = name used in this project, value = column in the dataset.
TASKS = {
    "queue": "queue",        # which team the ticket should be routed to
    "priority": "priority",  # how urgent it is
    "type": "type",          # ITIL ticket type: Incident / Request / Problem / Change
}

VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 42

# --- Transformer fine-tuning -----------------------------------------------
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256
EPOCHS = 6  # the first run used 3 and was still improving
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

# --- Real-world data (v2, see docs/DATA_CARD.md) -----------------------------
# Real IBM support-forum questions with IBM Technotes as the knowledge base (Apache-2.0).
TECHQA_REPO = "nvidia/TechQA-RAG-Eval"
# Real Stack Exchange questions with accepted answers (CC BY-SA 4.0).
STACKEXCHANGE_REPO = "HuggingFaceH4/stack-exchange-preferences"
STACKEXCHANGE_SITES = ("superuser.com", "askubuntu.com")
STACKEXCHANGE_MAX_FILES = 2          # parquet shards read per site (each is roughly 40 MB)
STACKEXCHANGE_MAX_QUESTIONS = 20_000  # per site, sampled from questions with an accepted answer
# Share of Stack Exchange questions whose accepted answer is removed from the knowledge base,
# to simulate new issues the knowledge base cannot answer yet.
KB_HOLDOUT_SHARE = 0.2
# Real ServiceNow incident event log from an IT company (CC BY 4.0). No ticket text, only process data.
UCI_INCIDENT_LOG_URL = (
    "https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip"
)

# --- Retrieval (v2) ----------------------------------------------------------
# Small, widely used open models so everything runs on a free Colab GPU.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended by BGE for queries
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS = 200        # long documents are split into overlapping passages (models read ~512 tokens)
CHUNK_OVERLAP = 50
CANDIDATES = 100         # documents each retriever returns before fusion
RRF_K = 60               # reciprocal rank fusion constant (the value from the original RRF paper)
RERANK_DEPTH = 30        # hybrid candidates the reranker re-scores
RERANK_CHUNKS_PER_DOC = 3  # best passages per document (by embedding similarity) shown to the reranker
RUN_DEPTH = 20           # results kept per query for the resolve-or-escalate model

# --- Resolve, assist or escalate (v2) -----------------------------------------
PRIMARY_RETRIEVER = "dense"   # chosen in docs/RETRIEVAL.md
CONTEXT_DOCS = 3              # articles the answer step will read; "resolvable" = correct one among them
TARGET_RISK = 0.10            # at most 10% of auto-resolved tickets may be wrong...
RISK_CONFIDENCE = 0.95        # ...with 95% confidence (Clopper-Pearson bounds, Bonferroni over a binary search)
ASSIST_RECALL = 0.90          # escalate without a draft only below the score that keeps 90% of resolvable tickets
CV_FOLDS = 5

# --- Answer step (v2) ----------------------------------------------------------
# Open model (Apache-2.0, ungated, ~8 GB in fp16) so anyone can reproduce the results on a free Colab GPU.
OPEN_LLM = "Qwen/Qwen3-4B-Instruct-2507"
# Optional: Claude through the API (set ANTHROPIC_API_KEY yourself; never put it in the code).
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
QUESTION_MAX_WORDS = 300      # long tickets are cut so the articles still fit in the prompt
EVIDENCE_MAX_WORDS = 200      # per article: its passage most similar to the ticket
ANSWER_MAX_TOKENS = 200     # a short reply; also keeps generation time down
NOT_FOUND = "NOT_FOUND"       # what the model must reply when the articles do not contain the answer

# --- Demo ------------------------------------------------------------------
# Below this confidence the demo flags the ticket for a human to check.
REVIEW_THRESHOLD = 0.60
