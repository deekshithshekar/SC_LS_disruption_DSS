# src/eda.py

import pandas as pd


def delay_summary(df: pd.DataFrame) -> dict:
    group_cols = ["Country", "Shipment Mode", "Vendor", "Product Group"]
    summaries = {}
    for col in group_cols:
        if col in df.columns:
            tmp = (
                df.groupby(col)["Delivery_Delayed"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "delay_rate", "count": "n_shipments"})
                .sort_values("delay_rate", ascending=False)
            )
            summaries[col] = tmp
    return summaries


def numeric_distributions(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return df[cols].describe().T
