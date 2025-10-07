import pandas as pd

# Helper to extract scalar features
def extract_scalar_features(features):
    """Flatten and extract scalar values."""
    return {key: (val.iloc[0].item() if isinstance(val, pd.Series) else float(val)) for key, val in features.items()}
