import pandas as pd
from typing import List

def _norm_for_merge(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Lower/strip key columns for stable joins."""
    for k in keys:
        if k in df.columns:
            df[k] = df[k].astype(str).str.strip().str.lower()
    return df

def join_labels(meta: pd.DataFrame, labels: pd.DataFrame, keys: List[str], label_col: str) -> pd.DataFrame:
    """
    Join meta with labels on keys (many-to-one).
    If labels duplicate on keys, aggregate deterministically.
    Returns meta+label with 'orig_idx' for array slicing.
    """
    if meta.empty:
        return meta

    keys_use = [k for k in keys if (k in meta.columns and k in labels.columns)]
    if not keys_use:
        print(f"[WARN] Label join: no common keys in meta/labels (asked: {keys})")
        return pd.DataFrame()

    m = meta.reset_index().rename(columns={"index": "orig_idx"})
    m = _norm_for_merge(m, keys_use)
    l = _norm_for_merge(labels.copy(), keys_use)

    if label_col not in labels.columns:
        print(f"[WARN] Label join: label column '{label_col}' not found.")
        return pd.DataFrame()

    l = l[keys_use + [label_col]].dropna(subset=[label_col])

    if l.duplicated(subset=keys_use).any():
        if pd.api.types.is_numeric_dtype(l[label_col]):
            l = l.groupby(keys_use, as_index=False)[label_col].mean()
        else:
            def _mode_or_first(s: pd.Series):
                try:
                    m0 = s.mode(dropna=True)
                    return m0.iloc[0] if len(m0) else s.iloc[0]
                except Exception:
                    return s.iloc[0]
            l = l.groupby(keys_use, as_index=False)[label_col].agg(_mode_or_first)

    df = m.merge(l, on=keys_use, how="left", validate="m:1")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    return df
