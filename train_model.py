"""
Trains the SMF batch-job failure-risk model and saves a single
self-contained pickle (job_failure_model.pkl) for the Streamlit app.

This is a direct, deployable port of the logic in `smf_data.ipynb`.
The only change is *how the data is loaded*: the notebook pulled
df_smf.csv from IBM Cloud Object Storage; this script just reads a
local CSV, so it works anywhere.

Usage
-----
    python train_model.py                     # looks for ./df_smf.csv
    python train_model.py --csv /path/to/df_smf.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42

FEATURES_NUMERIC = [
    "EXCP", "IO_CONN_SEC", "PAGE_IN", "PAGE_OUT", "PAGE_SWAP",
    "SSCH", "TCB_CPU_SEC", "SRB_CPU_SEC", "SERV_UNIT", "MSO_UNIT",
    "TOTAL_QUEUE_SEC", "CPU_SEC", "START_HOUR", "START_DOW", "CLASS",
]
FEATURES_CATEGORICAL = ["SERV_CLASS"]


def build_label(data: pd.DataFrame, top_pct: float = 0.20):
    """
    Composite, rank-based risk score computed WITHIN each service class
    (so a BATCH_A job is only compared against other BATCH_A jobs, etc.),
    then the worst `top_pct` runs overall are labelled FAILURE_RISK = 1.
    Using ranks (0-1 percentile) instead of raw thresholds keeps the score
    comparable across service classes with very different workloads.
    """
    excp_cpu_ratio = data["EXCP"] / (data["CPU_SEC"] + 1)

    def pct_rank(s):
        return s.groupby(data["SERV_CLASS"]).rank(pct=True)

    score = (
        0.40 * pct_rank(data["ELAPSED_SEC"])
        + 0.40 * pct_rank(data["TOTAL_QUEUE_SEC"])
        + 0.20 * pct_rank(excp_cpu_ratio)
    )
    threshold = score.quantile(1 - top_pct)
    return (score >= threshold).astype(int), score


def main(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    # ---- 1. feature engineering -------------------------------------
    df["START_DTSTR"] = pd.to_datetime(
        df["START_DTSTR"], format="%Y-%m-%d-%H.%M.%S.%f"
    )
    df["START_HOUR"] = df["START_DTSTR"].dt.hour
    df["START_DOW"] = df["START_DTSTR"].dt.dayofweek
    df["TOTAL_QUEUE_SEC"] = df["JQ_SEC"] + df["HQ_SEC"]

    # ---- 2. proxy failure-risk label ---------------------------------
    df["FAILURE_RISK"], df["RISK_SCORE"] = build_label(df)
    print("Label distribution:\n", df["FAILURE_RISK"].value_counts(normalize=True))

    # ---- 3. encode + assemble feature matrix -------------------------
    le_servclass = LabelEncoder()
    df["SERV_CLASS_ENC"] = le_servclass.fit_transform(df["SERV_CLASS"])
    FEATURE_COLS = FEATURES_NUMERIC + ["SERV_CLASS_ENC"]

    X = df[FEATURE_COLS]
    y = df["FAILURE_RISK"]

    # ---- 4. train / test split + model --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred))
    try:
        auc = roc_auc_score(y_test, proba)
        print("ROC AUC:", auc)
    except Exception as e:
        auc = None
        print("ROC AUC could not be computed:", e)

    # ---- 5. per-job historical profile (lets the app pre-fill inputs) -
    job_profile = (
        df.groupby("JOB_NAME")[FEATURES_NUMERIC + ["ELAPSED_SEC", "FAILURE_RISK"]]
        .mean()
        .reset_index()
    )
    job_profile["SERV_CLASS"] = df.groupby("JOB_NAME")["SERV_CLASS"].agg(
        lambda s: s.mode().iloc[0]
    ).values
    job_profile["RUN_COUNT"] = df.groupby("JOB_NAME").size().values

    global_defaults = df[FEATURES_NUMERIC].median().to_dict()

    # ---- 6. persist everything the Streamlit app needs into ONE pickle
    artifact = {
        "model": model,
        "label_encoder_servclass": le_servclass,
        "feature_cols": FEATURE_COLS,
        "numeric_features": FEATURES_NUMERIC,
        "job_profile": job_profile,
        "global_defaults": global_defaults,
        "serv_class_options": sorted(df["SERV_CLASS"].unique().tolist()),
        "class_options": sorted(df["CLASS"].unique().tolist()),
        "job_name_options": sorted(df["JOB_NAME"].unique().tolist()),
        "training_metrics": {
            "roc_auc": float(auc) if auc is not None else None,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "label_positive_rate": float(y.mean()),
        },
        "raw_df": df,  # kept for the dashboard's charts
    }

    joblib.dump(artifact, out_path)
    print(f"\nSaved {out_path}")
    print("Feature importances:")
    for f, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f}: {imp:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="df_smf.csv", help="Path to df_smf.csv")
    parser.add_argument("--out", default="job_failure_model.pkl", help="Output pickle path")
    args = parser.parse_args()
    main(args.csv, args.out)
