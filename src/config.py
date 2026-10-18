"""
Central config so switching from the MIMIC-IV Demo (100 patients) to full
credentialed MIMIC-IV (once PhysioNet access is approved) is a one-line change.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Point this at whichever MIMIC-IV export you're using:
#   demo:  physionet.org/content/mimic-iv-demo  -> data/mimic-iv-demo/
#   full:  physionet.org/content/mimicidv        -> data/mimic-iv/
DATA_DIR = PROJECT_ROOT / "data" / "mimic-iv-demo"

HOSP_DIR = DATA_DIR / "hosp"
ICU_DIR = DATA_DIR / "icu"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- cohort criteria ---
MIN_AGE = 18
MIN_ICU_STAY_HOURS = 4
FEATURE_WINDOW_HOURS = 24  # only use data from the first N hours of the ICU stay

RANDOM_STATE = 42
