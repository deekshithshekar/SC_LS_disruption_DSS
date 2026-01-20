# src/data_cleaning.py

import pandas as pd
from .config import DATA_PATH, DATE_COLS, NUMERIC_COLS_RAW
from .utils import to_numeric_clean


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS_RAW:
        if col in df.columns and df[col].dtype == "object":
            df[col] = to_numeric_clean(df[col])
    return df


def create_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = convert_dates(df)

    # Delivery delay
    df["delivery_delay_days"] = (
        df["Delivered to Client Date"] - df["Scheduled Delivery Date"]
    ).dt.days

    df["Delivery_Delayed"] = (df["delivery_delay_days"] > 0).astype(int)

    # Lead time features
    df["po_to_schedule_days"] = (
        df["Scheduled Delivery Date"] - df["PO Sent to Vendor Date"]
    ).dt.days

    df["po_to_delivery_days"] = (
        df["Delivered to Client Date"] - df["PO Sent to Vendor Date"]
    ).dt.days

    return df


def basic_clean_pipeline(path: str = DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    df = clean_numeric_columns(df)
    df = create_target_and_features(df)

    df = df.dropna(
        subset=[
            "Scheduled Delivery Date",
            "Delivered to Client Date",
            "delivery_delay_days",
            "Delivery_Delayed",
        ]
    )

    df = df[df["delivery_delay_days"].between(-365, 365)]
    return df


if __name__ == "__main__":
    df_clean = basic_clean_pipeline()
    print(df_clean.head())
    print(df_clean["Delivery_Delayed"].value_counts(dropna=False))
