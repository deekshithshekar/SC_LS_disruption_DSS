# src/modeling.py

import os
import joblib
import pandas as pd
from typing import Tuple

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from .features import train_test_split_data
from .data_cleaning import basic_clean_pipeline

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")


def build_pipeline(num_cols, cat_cols) -> Pipeline:
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    (
        X_num_train,
        X_num_test,
        X_cat_train,
        X_cat_test,
        y_train,
        y_test,
        num_cols,
        cat_cols,
    ) = train_test_split_data(df)

    X_train = pd.concat([X_num_train, X_cat_train], axis=1)
    X_test = pd.concat([X_num_test, X_cat_test], axis=1)

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return pipeline, metrics


def save_model(model: Pipeline, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    df_clean = basic_clean_pipeline()
    model, metrics = train_and_evaluate(df_clean)
    save_model(model)
    print("Model trained and saved to", MODEL_PATH)
    print("ROC-AUC:", metrics["roc_auc"])
