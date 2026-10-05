"""Project-wide settings. Change values here rather than inside the scripts."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
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
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 5e-5

# --- Demo ------------------------------------------------------------------
# Below this confidence the demo flags the ticket for a human to check.
REVIEW_THRESHOLD = 0.60
