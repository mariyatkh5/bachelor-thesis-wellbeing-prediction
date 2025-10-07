import pandas as pd
import os
import sys

# Falls __file__ nicht vorhanden ist (z.B. Notebook), diesen Block optional absichern:
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except NameError:
    pass

import constants as const

input_file = os.path.join(const.BASE_DIR, "dataset2_RobotBehaviour", "Questionnaire_data1.xlsx")
output_dir = os.path.join(const.OUTPUT_DIR, "dataset2")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "labels.csv")

df = pd.read_excel(input_file)
df = df.rename(columns={"Index": "Participant"})

if df["Participant"].dtype != "Int64":
    df["Participant"] = pd.to_numeric(df["Participant"], errors="coerce").astype("Int64")

label_rows = []

# Spalten erfassen
arousal_cols = [c for c in df.columns if str(c).count('.') == 2 and str(c).endswith(".4")]
valence_cols = [c for c in df.columns if str(c).count('.') == 2 and str(c).endswith(".3")]

# Basen bilden: "<speed>.<robots>"
def base_of(col): 
    parts = str(col).split(".")
    return ".".join(parts[:2]) if len(parts) == 3 else None

valence_map = {base_of(c): c for c in valence_cols}
arousal_map = {base_of(c): c for c in arousal_cols}

common_bases = sorted(set(valence_map).intersection(arousal_map))

for base in common_bases:
    val_col = valence_map[base]
    aro_col = arousal_map[base]
    try:
        speed_str, robots_str = base.split(".")
        speed = int(speed_str)
        robots = int(robots_str)
    except Exception:
        continue

    task = (speed - 1) * 3 + robots

    for _, row in df.iterrows():
        participant = row["Participant"]
        if pd.isna(participant):
            continue

        val = row[val_col]
        aro = row[aro_col]

        if pd.isna(val) and pd.isna(aro):
            continue


        wellbeing = None
        if pd.notna(aro):
            wellbeing = 1 if aro < 5  else 0
        label_rows.append({
            "Participant": int(participant),
            "Speed": speed,
            "Robots": robots,
            "Task": task,
            "Valence": None if pd.isna(val) else float(val),
            "Arousal": None if pd.isna(aro) else float(aro),
            "Well-being": wellbeing
        })

labels_df = pd.DataFrame(label_rows).sort_values(by=["Participant", "Task"])
labels_df.to_csv(output_file, index=False)
