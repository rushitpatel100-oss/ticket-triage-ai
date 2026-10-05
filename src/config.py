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

# --- Demo ------------------------------------------------------------------
# Below this confidence the demo flags the ticket for a human to check.
REVIEW_THRESHOLD = 0.60
