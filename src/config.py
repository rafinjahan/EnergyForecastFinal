from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"

# Defaults
DEFAULT_FREQ = "H"  # hourly data
DEFAULT_HORIZON = 24

# Create directories if missing
for p in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)
