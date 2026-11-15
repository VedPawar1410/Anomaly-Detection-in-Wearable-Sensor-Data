"""Single source of truth for paths, column names and the subject split."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROTOCOL_DIR = DATA_DIR / "raw" / "PAMAP2_Dataset" / "Protocol"
CACHE_DIR = DATA_DIR / "interim"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SAMPLE_RATE_HZ = 100

# --- Raw file layout (54 columns, see data/README.md) -------------------------
LOCATIONS = ["hand", "chest", "ankle"]
IMU_FIELDS = (
    ["temp"]
    + [f"acc16_{a}" for a in "xyz"]  # +-16g accelerometer (recommended by the authors)
    + [f"acc6_{a}" for a in "xyz"]  # +-6g accelerometer (saturates; duplicates acc16)
    + [f"gyro_{a}" for a in "xyz"]
    + [f"mag_{a}" for a in "xyz"]
    + [f"orient_{i}" for i in range(1, 5)]  # marked invalid by the dataset authors
)
RAW_COLUMNS = ["timestamp", "activity_id", "heart_rate"] + [
    f"{loc}_{field}" for loc in LOCATIONS for field in IMU_FIELDS
]

# --- What the detectors use ---------------------------------------------------
# Dropped: orientation (invalid) and acc6 (a near-copy of acc16, so predicting one
# from the other is trivial and hides how hard the real problem is).
KEPT_IMU_FIELDS = ["temp"] + [f"{s}_{a}" for s in ("acc16", "gyro", "mag") for a in "xyz"]
SENSOR_COLUMNS = ["heart_rate"] + [f"{loc}_{f}" for loc in LOCATIONS for f in KEPT_IMU_FIELDS]
META_COLUMNS = ["subject", "timestamp", "activity_id"]

# Rows per protocol file, checked after download. Loading asserts these so a
# truncated file (what happened to the 2024 copy) fails loudly instead of silently.
EXPECTED_ROWS = {
    101: 376_417,
    102: 447_000,
    103: 252_833,
    104: 329_576,
    105: 374_783,
    106: 361_817,
    107: 313_599,
    108: 408_031,
    109: 8_477,
}

ACTIVITIES = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}
TRANSIENT_ACTIVITY = 0

# --- Subject split ------------------------------------------------------------
# Split by person, never by row: neighbouring rows 0.01 s apart are near-copies,
# so a random row split leaks the test set into training.
#   train: fit the models
#   calib: a clean, unseen subject used only to pick anomaly thresholds
#   test:  unseen subjects for the final numbers
# 109 is excluded: 8,477 rows (~1.5 min) of mostly one activity.
TRAIN_SUBJECTS = [101, 102, 103, 104, 106]
CALIB_SUBJECTS = [107]
TEST_SUBJECTS = [105, 108]

# Sensor dropouts shorter than this (in samples) are interpolated; longer ones are dropped.
MAX_INTERP_GAP = 10

# --- Training -----------------------------------------------------------------
RANDOM_SEED = 42

# Compute constraint: trained on an 8-core, 16 GB laptop. Fitting 31 forests on all
# ~1.2M training rows at 100 Hz doesn't fit that budget, so training uses every
# 10th row (10 Hz, ~120k rows). Calibration and testing still use every row.
# Set TRAIN_STRIDE = 1 to train on everything when more compute is available.
TRAIN_STRIDE = 10

# Per-sensor Random Forest. Depth and leaf size are capped to keep each model
# small (the 2024 pickles were up to 12.6 MB each, with 5 unbounded trees).
RF_PARAMS = {
    "n_estimators": 40,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# Per-sensor histogram gradient boosting (early stopping on a 10% validation split).
HGB_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "random_state": RANDOM_SEED,
}

# One Isolation Forest per sensor on trailing-window features.
IFOREST_PARAMS = {
    "n_estimators": 100,
    "max_samples": 512,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# Thresholds are set so that this share of readings on the clean calibration
# subject would be flagged: the expected false-alarm rate, chosen up front.
TARGET_FALSE_ALARM_RATE = 0.005
