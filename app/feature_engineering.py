"""
Feature engineering module for survival analysis.

This module handles preprocessing, cohort definition, and feature transformations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_admissions, load_patients, load_labevents, load_d_labitems


def define_cohort(mimic_root_dir: Optional[str] = None) -> pd.DataFrame:
    """Construct the 30-day readmission survival frame.

    - Index: each discharge is a candidate
    - event = 1 if 0 < days_to_next <= 30 else 0
    - time_to_event = min(days_to_next, 30)
    - Exclusions: in-hospital deaths; death before next admission within 30 days is censored

    Args:
        mimic_root_dir: Optional MIMIC root directory override.

    Returns:
        pd.DataFrame: Cohort with minimally required columns.
    """
    admissions = load_admissions(mimic_root_dir)
    patients = load_patients(mimic_root_dir)

    admissions = admissions[admissions["dischtime"].notna()].copy()

    cohort = admissions.merge(patients, on="subject_id", how="left")
    cohort["age_at_discharge"] = cohort["anchor_age"] + (
        cohort["dischtime"].dt.year - cohort["anchor_year"]
    )

    # Basic LOS
    if "admittime" in cohort.columns and "dischtime" in cohort.columns:
        cohort["los_days"] = (
            cohort["dischtime"] - cohort["admittime"]
        ).dt.total_seconds() / (24 * 3600)

    cohort = cohort.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    cohort["next_admittime"] = cohort.groupby("subject_id")["admittime"].shift(-1)
    cohort["days_to_next"] = (
        (cohort["next_admittime"] - cohort["dischtime"]).dt.total_seconds() / (24 * 3600)
    )

    # Filter out likely transfers: <= 6 hours after discharge counts as same encounter
    cohort.loc[cohort["days_to_next"] <= (6 / 24), "days_to_next"] = np.nan

    cohort["event"] = ((cohort["days_to_next"] > 0) & (cohort["days_to_next"] <= 30)).astype(int)
    cohort["time_to_event"] = np.where(
        cohort["days_to_next"].notna(), np.minimum(cohort["days_to_next"], 30), 30
    )

    # Exclude in-hospital deaths from candidacy
    if "hospital_expire_flag" in cohort.columns:
        cohort = cohort[cohort["hospital_expire_flag"] == 0].copy()

    # Censor at death if earlier than time_to_event
    if "deathtime" in cohort.columns:
        time_to_death = (cohort["deathtime"] - cohort["dischtime"]).dt.total_seconds() / (24 * 3600)
        mask = (time_to_death.notna()) & (time_to_death >= 0) & (time_to_death < cohort["time_to_event"])
        cohort.loc[mask, "event"] = 0
        cohort.loc[mask, "time_to_event"] = time_to_death[mask]

    cohort["time_to_event"] = cohort["time_to_event"].clip(lower=0, upper=30)

    cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "age_at_discharge",
        "next_admittime",
        "days_to_next",
        "event",
        "time_to_event",
        "admission_type",
        "discharge_location",
        "insurance",
        "ethnicity",
        "gender",
    ]
    existing = [c for c in cols if c in cohort.columns]
    return cohort[existing].copy()


def engineer_features(cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a basic feature matrix and target arrays from the cohort.

    Produces simple, explainable features that are available at discharge.

    Args:
        cohort: Cohort dataframe from ``define_cohort``.

    Returns:
        features (X), targets (y) where ``y`` contains ``event`` and ``time_to_event``.
    """
    df = cohort.copy()

    # Merge pre-discharge lab summaries (last value per hadm_id, itemid)
    try:
        labs = load_labevents(None)
        if {"hadm_id"}.issubset(labs.columns) and {"hadm_id"}.issubset(df.columns):
            # Choose top itemids globally to increase chance of overlap in demo
            top_items_global = labs["itemid"].value_counts().head(20).index.tolist() if "itemid" in labs.columns else []

            X_labs = None
            if {"charttime"}.issubset(labs.columns) and {"admittime", "dischtime"}.issubset(df.columns):
                labs_win = labs.merge(
                    df[["hadm_id", "admittime", "dischtime"]].drop_duplicates(),
                    on="hadm_id",
                    how="inner",
                )
                labs_win = labs_win[(labs_win["charttime"] >= labs_win["admittime"]) & (labs_win["charttime"] <= labs_win["dischtime"])].copy()
                if "itemid" in labs_win.columns:
                    # Prefer frequent items; fallback to all if empty
                    sel = labs_win["itemid"].isin(top_items_global) if top_items_global else pd.Series([True] * len(labs_win))
                    labs_sel = labs_win[sel]
                else:
                    labs_sel = labs_win
                if not labs_sel.empty:
                    labs_last = labs_sel.sort_values(["hadm_id", "itemid", "charttime"]).groupby(["hadm_id", "itemid"], as_index=False).tail(1)
                    keep_cols = [c for c in ["hadm_id", "itemid", "valuenum"] if c in labs_last.columns]
                    if {"hadm_id", "itemid", "valuenum"}.issubset(set(keep_cols)):
                        X_labs = labs_last[keep_cols].pivot_table(index="hadm_id", columns="itemid", values="valuenum")
            # Fallback: no windowed labs found; pivot last values per hadm_id from global top items
            if X_labs is None or X_labs.empty:
                if {"itemid", "valuenum"}.issubset(labs.columns):
                    labs_any = labs[labs["itemid"].isin(top_items_global)] if top_items_global else labs
                    if "charttime" in labs_any.columns:
                        labs_any = labs_any.sort_values(["hadm_id", "itemid", "charttime"])\
                                           .groupby(["hadm_id", "itemid"], as_index=False).tail(1)
                    keep_cols = [c for c in ["hadm_id", "itemid", "valuenum"] if c in labs_any.columns]
                    if {"hadm_id", "itemid", "valuenum"}.issubset(set(keep_cols)):
                        X_labs = labs_any[keep_cols].pivot_table(index="hadm_id", columns="itemid", values="valuenum")

            if X_labs is not None and not X_labs.empty:
                X_labs.columns = [f"lab_{c}" for c in X_labs.columns]
                df = df.merge(X_labs, on="hadm_id", how="left")
                # Fill NaN values in lab features with median values
                lab_cols = [c for c in df.columns if c.startswith('lab_')]
                for col in lab_cols:
                    if df[col].isna().any():
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
    except Exception:
        # If labs missing in demo or not downloaded, continue with core features only
        pass

    # Categorical encodings (simple, low-cardinality)
    for col in ["gender", "admission_type", "discharge_location", "insurance"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # One-hot encode selected categoricals
    cat_cols = [c for c in ["gender", "admission_type", "discharge_location", "insurance"] if c in df.columns]
    
    # Include lab features in the final feature matrix
    lab_cols = [c for c in df.columns if c.startswith('lab_')]
    numeric_cols = [c for c in ["age_at_discharge", "los_days"] if c in df.columns]
    
    # Combine all features
    all_features = numeric_cols + cat_cols + lab_cols
    X = pd.get_dummies(
        df[all_features],
        drop_first=True,
    )

    # Targets
    y = df[["event", "time_to_event"]].copy()
    return X, y


