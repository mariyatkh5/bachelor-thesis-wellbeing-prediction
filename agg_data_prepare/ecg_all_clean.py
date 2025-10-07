import os
import pandas as pd

def remove_nested_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """Unwrap values like [[123]] -> 123; leave others unchanged."""
    def unwrap(v):
        # unwrap nested lists: [[x]] -> x
        while isinstance(v, list) and len(v) > 0:
            v = v[0]
        # unwrap strings like "[[123.45]]"
        if isinstance(v, str):
            s = v.strip()
            if s.startswith('[[') and s.endswith(']]'):
                inner = s.strip('[]')
                try:
                    num = float(inner)
                    return int(num) if num.is_integer() else num
                except ValueError:
                    return inner
        return v
    return df.applymap(unwrap)

def clean_ecg_features_in_all_subfolders(base_folder: str, overwrite: bool = True, suffix: str = "_clean") -> None:
    """Find 'ecg_features*.csv', clean, and save (in-place by default)."""
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.startswith("ecg_features") and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    print(f"Cleaning file: {file_path}")
                    df = pd.read_csv(file_path)

                    df = remove_nested_brackets(df)
                    df = df.dropna(axis=1, how='all').fillna(-1)

                    if overwrite:
                        out_path = file_path
                    else:
                        name, ext = os.path.splitext(file)
                        out_path = os.path.join(root, f"{name}{suffix}{ext}")

                    df.to_csv(out_path, index=False)
                    print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    base_folder = "agg_data"
    clean_ecg_features_in_all_subfolders(base_folder, overwrite=True)
    print("Done! Empty columns removed and missing values filled with -1.")
