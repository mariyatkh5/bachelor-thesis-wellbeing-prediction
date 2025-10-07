# build_labels_by_stimuli_name.py
import os
import re
import pandas as pd
import sys

# Projekt-Constants laden (wie in deinem bestehenden Skript)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Pfade
input_dir = os.path.join(const.BASE_DIR, "dataset8_POPANE")
output_file = os.path.join(const.OUTPUT_DIR, "dataset8/labels.csv")

# ---- Nur diese Studies zulassen (wie gehabt)
ALLOWED_STUDIES = {3, 5, 6}
STUDY_RE = re.compile(r'^\s*#?Study_name\s*,\s*Study_(\d+)\s*$')

# ---- Well-being aus Stimuli-Name (Ordner) ableiten
POSITIVE_PREFIXES = {"Amusement", "Tenderness"}
NEGATIVE_PREFIXES = {"Anger", "Disgust", "Fear", "Sadness"}

# Optional: exakte Overrides pro Stimulus-Ordnername (falls mal etwas abweicht)
EXPLICIT_OVERRIDES = {
    # "Amusement1": 1,
    # "Anger1": 0,
}

def base_prefix(name: str) -> str | None:
    """
    Extrahiert den alphabetischen Präfix (z. B. 'Amusement' aus 'Amusement1').
    Nutzt den Ordnernamen (Stimuli).
    """
    if not isinstance(name, str):
        return None
    m = re.match(r"[A-Za-z]+", name.strip())
    return m.group(0) if m else None

def wellbeing_from_stimuli_name(stimuli_folder: str) -> int | None:
    """
    Entscheidet Well-being (1/0) nur anhand des Stimuli-Ordnernamens.
    """
    # exakter Override (falls du ganze Ordner exakt mappen willst)
    if stimuli_folder in EXPLICIT_OVERRIDES:
        return EXPLICIT_OVERRIDES[stimuli_folder]

    # über Präfix entscheiden
    prefix = base_prefix(stimuli_folder)
    if prefix in POSITIVE_PREFIXES:
        return 1
    if prefix in NEGATIVE_PREFIXES:
        return 0
    return None  # unbekannt → wir überspringen diese Zeilen später

data_list = []

for stimuli_folder in os.listdir(input_dir):
    # Skip Ordner mit "Neutral" oder "Baselines"
    if "Neutral" in stimuli_folder or stimuli_folder == "Baselines":
        continue

    stimuli_path = os.path.join(input_dir, stimuli_folder)
    if not os.path.isdir(stimuli_path):
        continue

    # Well-being nur aus Ordnernamen bestimmen
    wb_from_name = wellbeing_from_stimuli_name(stimuli_folder)
    if wb_from_name is None:
        # Unbekannter Stimulus-Typ → überspringen
        continue

    for file_name in os.listdir(stimuli_path):
        # Skip Dateien mit "Neutral" und Nicht-CSV
        if "Neutral" in file_name or not file_name.endswith(".csv"):
            continue
        # Zusätzliche Regel: "Gratitude" auslassen
        if "Gratitude" in file_name:
            continue

        parts = file_name.split("_")
        if len(parts) < 3:
            continue

        task = parts[0][1:]         # 'Txx' → xx
        participant = parts[1][1:]  # 'Pxx' → xx

        file_path = os.path.join(stimuli_path, file_name)

        # Datei lesen, Headerzeile finden/prüfen
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Study-Nummer filtern
        m = STUDY_RE.match(lines[0].strip()) if lines else None
        if not m:
            continue
        study_num = int(m.group(1))
        if study_num not in ALLOWED_STUDIES:
            continue

        # Datenteil ab "timestamp" suchen
        try:
            data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
        except StopIteration:
            continue

        df = pd.read_csv(file_path, skiprows=data_start_idx)
        df.columns = df.columns.str.strip()

        # optionale Infos (Affect/Marker), beeinflussen NICHT das Label
        affect_mean = None
        if "affect" in df.columns:
            try:
                affect_mean = float(df["affect"].mean())
            except Exception:
                affect_mean = None

        # Bereich 4 bis 6 (inklusive) ausschließen → nur behalten, wenn <4 oder >6
        if affect_mean is None or (4 <= affect_mean <= 6):
            continue

        first_marker = (
            df["marker"].dropna().iloc[0]
            if "marker" in df.columns and not df["marker"].dropna().empty
            else None
        )

        data_list.append({
            "Participant": int(participant),
            "Task": int(task),
            "Marker": first_marker,
            "Stimuli": stimuli_folder,
            "Well-being": wb_from_name,                # aus Stimuli-Namen
            "Affect": round(affect_mean, 2),
        })

# DataFrame bauen & speichern
output_df = pd.DataFrame(data_list)

if not output_df.empty:
    # sortieren wie gehabt
    output_df = output_df.sort_values(by=["Participant", "Task", "Marker"]).reset_index(drop=True)
    output_df = output_df[["Participant", "Task", "Marker", "Stimuli", "Well-being", "Affect"]]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)

    # kurze Übersicht
    df_out = pd.read_csv(output_file)
    print(df_out['Well-being'].value_counts(dropna=False))
    print(df_out['Well-being'].value_counts(normalize=True, dropna=False) * 100)
    print(f"\nGespeichert: {output_file}")
else:
    print("Keine Daten gefunden, die den Kriterien entsprechen.")
