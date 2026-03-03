from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if they do not exist
for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Target categories for MVP
TARGET_CATEGORIES = {"cs.IR", "cs.LG", "cs.CL"}

# Text quality thresholds
MIN_TITLE_CHARS = 5
MIN_ABSTRACT_CHARS = 40
MAX_ABSTRACT_CHARS = 10000  # sanity check only