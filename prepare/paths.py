# paths.py
import os

# ---- Base folders ----
DATASET1_DIR = os.path.join("dataset1_SenseCobot")
DATASET2_DIR = os.path.join("dataset2_RobotBehaviour", "Measurements_fixed")
DATASET3_DIR = os.path.join("dataset3_MAUS", "Data", "Raw_data")
DATASET8_DIR = os.path.join("dataset8_POPANE")

OUTPUT1_DIR = os.path.join("output", "dataset1")
OUTPUT2_DIR = os.path.join("output", "dataset2")
OUTPUT3_DIR = os.path.join("output", "dataset3")
OUTPUT8_DIR = os.path.join("output", "dataset8")

# ---- Labels ----
LABELS1_CSV = os.path.join("agg_data", "dataset1", "labels.csv")
LABELS1_PAPER_CSV = os.path.join("agg_data", "dataset1", "labels_paper.csv")  # used by main.py
LABELS2_CSV = os.path.join("agg_data", "dataset2", "labels.csv")
LABELS3_CSV = os.path.join("agg_data", "dataset3", "labels.csv")
LABELS8_CSV = os.path.join("agg_data", "dataset8", "labels.csv")

# ---- Signal layout ----
D1_SIGNALS = {
    "EDA": dict(kind="eda", folder="GSR_Shimmer3_Signals",
                pattern="GSR_Task {task}_P_{p2}.csv",
                col_candidates=["GSR Conductance CAL", "GSR"]),
    "ECG": dict(kind="ecg", folder="ECG_Shimmer3_Signals",
                pattern="ECG_Task {task}_P_{p2}.csv",
                col_candidates=["ECG LL-RA CAL"]),
}

D2_SIGNALS = {
    "EDA": dict(kind="eda", ext="EDA"),
    "ECG": dict(kind="ecg", ext="ECG"),
}

D3_SIGNALS = {
    "EDA": {"kind": "eda"},
    "ECG": {"kind": "ecg"},
}

D8_SIGNALS = {
    "EDA": dict(kind="eda", col="EDA"),
    "ECG": dict(kind="ecg", col="ECG"),
}
