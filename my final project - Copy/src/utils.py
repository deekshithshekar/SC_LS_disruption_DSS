# src/utils.py

import pandas as pd
import numpy as np

def to_numeric_clean(series: pd.Series):
    """
    Convert object column to numeric, coercing errors, stripping text phrases.
    """
    s = series.astype(str).replace(
        [
            "Freight Included in Commodity Cost",
            "Weight Captured Separately",
            "Invoiced Separately",
            "See ASN",
            "N/A",
            "",
        ],
        np.nan,
        regex=True,
    )
    s = s.str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")
