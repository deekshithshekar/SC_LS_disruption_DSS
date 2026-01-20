# src/features.py

import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CATEGORICAL_COLS, NUMERIC_COLS_RAW
from .utils import to_numeric_clean


def build_feature_matrix(df: pd.DataFrame):
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS_RAW:
        if col in df.columns and df[col].dtype == "object":
            df[col] = to_numeric_clean(df[col])

    # IMPORTANT: do NOT include delivery_delay_days as a feature (leakage)
    numeric_features = NUMERIC_COLS_RAW + [
        "po_to_schedule_days",
        "po_to_delivery_days",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]

    cat_features = [c for c in CATEGORICAL_COLS if c in df.columns]

    df = df.dropna(subset=["Delivery_Delayed"])

    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    df[cat_features] = df[cat_features].fillna("Missing")

    X_numeric = df[numeric_features]
    X_cat = df[cat_features]
    y = df["Delivery_Delayed"]

    return X_numeric, X_cat, y, numeric_features, cat_features


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X_numeric, X_cat, y, num_cols, cat_cols = build_feature_matrix(df)

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_numeric, X_cat, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (
        X_num_train,
        X_num_test,
        X_cat_train,
        X_cat_test,
        y_train,
        y_test,
        num_cols,
        cat_cols,
    )
