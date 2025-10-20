"""
Interpretability helpers for survival models.

Includes utilities for SHAP explanations on tree models and hazard ratio tables
for Cox PH models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_values(tree_model: Any, X: pd.DataFrame):
    """Compute SHAP values for a tree-based model (e.g., XGBoost).

    Requires ``shap`` to be installed. Returns SHAP values array and expected
    value for downstream plots.
    """
    import shap  # lazy import to avoid mandatory dependency

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    return shap_values, expected_value


def compute_hazard_ratios(cox_model: Any) -> pd.DataFrame:
    """Return hazard ratios with 95% confidence intervals for a lifelines CoxPHFitter.

    Args:
        cox_model: Fitted lifelines.CoxPHFitter instance.
    """
    if not hasattr(cox_model, "summary"):
        raise ValueError("cox_model must be a lifelines.CoxPHFitter with a summary attribute")

    summary = cox_model.summary.copy()
    # lifelines provides exp(coef) and confidence intervals already
    out = summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={
            "exp(coef)": "HR",
            "exp(coef) lower 95%": "HR_lower_95",
            "exp(coef) upper 95%": "HR_upper_95",
            "p": "p_value",
        }
    )
    return out


def plot_survival_curves(kmf, groups: pd.Series):
    """Plot Kaplan-Meier survival curves stratified by groups.

    Args:
        kmf: lifelines.KaplanMeierFitter instance (unfitted).
        groups: Group labels per sample (e.g., risk tertiles).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for g in pd.unique(groups):
        mask = groups == g
        kmf.fit(durations=groups.index[mask].get_level_values("time"))
        kmf.plot_survival_function(ax=ax, label=str(g))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan–Meier Survival Curves")
    return fig


