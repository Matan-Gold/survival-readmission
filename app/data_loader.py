"""
Data loader module for MIMIC-IV data.

This module handles loading admissions and patient data from MIMIC-IV Demo or
full MIMIC-IV, depending on the configured data directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import load_config


def _resolve_hosp_dir(mimic_root_dir: Optional[str] = None) -> Path:
    cfg = load_config()
    root = Path(mimic_root_dir or cfg.mimic_demo_dir)
    hosp_dir = root / "hosp"
    return hosp_dir


def load_admissions(mimic_root_dir: Optional[str] = None) -> pd.DataFrame:
    """Load MIMIC-IV admissions table.

    Args:
        mimic_root_dir: Optional override for the MIMIC root directory. If not
            provided, uses ``MIMIC_DEMO_DIR`` or repo-relative fallback via
            ``load_config``.

    Returns:
        pd.DataFrame: Admissions data.
    """
    hosp_dir = _resolve_hosp_dir(mimic_root_dir)
    admissions_path = hosp_dir / "admissions.csv.gz"
    df = pd.read_csv(admissions_path, compression="gzip")
    # Parse common datetime columns if present
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_patients(mimic_root_dir: Optional[str] = None) -> pd.DataFrame:
    """Load MIMIC-IV patients table.

    Args:
        mimic_root_dir: Optional override for the MIMIC root directory.

    Returns:
        pd.DataFrame: Patient demographics information (anchor age/year, gender, etc.).
    """
    hosp_dir = _resolve_hosp_dir(mimic_root_dir)
    patients_path = hosp_dir / "patients.csv.gz"
    df = pd.read_csv(patients_path, compression="gzip")
    return df


def load_labevents(mimic_root_dir: Optional[str] = None) -> pd.DataFrame:
    """Load MIMIC-IV hospital laboratory events (hosp/labevents.csv.gz).

    Parses ``charttime`` to datetime and keeps numeric ``valuenum`` as float.
    """
    hosp_dir = _resolve_hosp_dir(mimic_root_dir)
    path = hosp_dir / "labevents.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    if "valuenum" in df.columns:
        df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


def load_d_labitems(mimic_root_dir: Optional[str] = None) -> pd.DataFrame:
    """Load lab item dictionary (hosp/d_labitems.csv.gz)."""
    hosp_dir = _resolve_hosp_dir(mimic_root_dir)
    path = hosp_dir / "d_labitems.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    return df


