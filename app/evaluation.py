"""
Evaluation metrics module for survival models.

This module implements metrics including C-index, time-dependent AUC,
Brier score, and calibration metrics.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


def compute_concordance_index(event: Iterable[int], time: Iterable[float], risk_scores: Iterable[float]) -> float:
    """Compute Harrell's C-index.

    Args:
        event: Iterable of 0/1 event indicators.
        time: Iterable of event or censoring times.
        risk_scores: Higher implies higher risk.

    Returns:
        C-index in [0, 1].
    """
    event_bool = np.asarray(event, dtype=bool)
    time_arr = np.asarray(time, dtype=float)
    risk_arr = np.asarray(risk_scores, dtype=float)
    c_index, *_ = concordance_index_censored(event_bool, time_arr, risk_arr)
    return float(c_index)


def compute_td_auc(
    y_train: Tuple[np.ndarray, np.ndarray],
    y_test: Tuple[np.ndarray, np.ndarray],
    risk_scores_test: Iterable[float],
    times: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-dependent AUC at given times.

    Args:
        y_train: (event_train_bool, time_train)
        y_test: (event_test_bool, time_test)
        risk_scores_test: Scores for test set.
        times: Time horizons for AUC.

    Returns:
        (auc_values, mean_auc_per_time) where shapes match ``times``.
    """
    (e_tr, t_tr) = y_train
    (e_te, t_te) = y_test
    # Create structured arrays for scikit-survival
    y_train_struct = np.array(list(zip(e_tr.astype(bool), t_tr)), dtype=[("event", bool), ("time", float)])
    y_test_struct = np.array(list(zip(e_te.astype(bool), t_te)), dtype=[("event", bool), ("time", float)])
    
    auc, mean_auc = cumulative_dynamic_auc(
        y_train_struct,
        y_test_struct,
        np.asarray(risk_scores_test, dtype=float),
        np.asarray(times, dtype=float),
    )
    return auc, mean_auc


def compute_brier_score(
    y_train: Tuple[np.ndarray, np.ndarray],
    y_test: Tuple[np.ndarray, np.ndarray],
    survival_prob_matrix: np.ndarray,
    times: Iterable[float],
) -> float:
    """Compute Integrated Brier Score (IBS).

    Args:
        y_train: Structured (event, time) arrays' raw components for training.
        y_test: Structured raw components for test.
        survival_prob_matrix: Matrix of survival probabilities S(t | x) for test
            samples, shape (n_test, len(times)).
        times: Evaluation time grid.

    Returns:
        Integrated Brier Score.
    """
    (e_tr, t_tr) = y_train
    (e_te, t_te) = y_test
    surv_train = np.array(list(zip(e_tr.astype(bool), t_tr)), dtype=[("event", bool), ("time", float)])
    surv_test = np.array(list(zip(e_te.astype(bool), t_te)), dtype=[("event", bool), ("time", float)])
    times_arr = np.asarray(list(times), dtype=float)
    # survival_prob_matrix expected shape (n_test, len(times))
    ibs = integrated_brier_score(surv_train, surv_test, survival_prob_matrix, times_arr)
    return float(ibs)


def compute_calibration(pred_probs: np.ndarray, observed_event: np.ndarray, n_bins: int = 10) -> dict:
    """Compute simple calibration by risk deciles.

    Args:
        pred_probs: Predicted event probabilities at horizon (e.g., 30d).
        observed_event: Binary observed events at the same horizon.
        n_bins: Number of bins (deciles default).

    Returns:
        Dict with per-bin predicted vs observed rates.
    """
    df = np.vstack([pred_probs, observed_event]).T
    # Rank by predicted probs and split into bins
    order = np.argsort(df[:, 0])
    df_sorted = df[order]
    bins = np.array_split(df_sorted, n_bins)
    calib = []
    for b in bins:
        if len(b) == 0:
            continue
        calib.append({
            "n": int(len(b)),
            "pred": float(b[:, 0].mean()),
            "obs": float(b[:, 1].mean()),
        })
    return {"bins": calib}


