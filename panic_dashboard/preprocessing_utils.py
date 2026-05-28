"""
preprocessing_utils.py
Complete preprocessing pipeline for PanicGuard AI.
Handles CSV loading, validation, normalization, and window creation.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load configuration from config.json."""
    if config_path is None:
        # Try to resolve relative to this file's location
        base = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base, "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config.json not found at: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed config.json: {e}")


# ─────────────────────────────────────────────
# CSV Loading
# ─────────────────────────────────────────────

def load_patient_csv(file_path_or_buffer) -> pd.DataFrame:
    """
    Load and basic-validate a patient CSV file.

    Parameters
    ----------
    file_path_or_buffer : str | path | file-like object
        Accepts a file path string or a Streamlit UploadedFile buffer.

    Returns
    -------
    pd.DataFrame
        Raw data frame with at least one row and one column.

    Raises
    ------
    ValueError
        If the file is empty or cannot be parsed.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Standardise column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Parse 'date' column if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass  # Non-critical; continue without sorting

    return df


# ─────────────────────────────────────────────
# Feature Validation
# ─────────────────────────────────────────────

def validate_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """
    Ensure all required feature columns exist in the DataFrame.
    Missing columns are added with NaN values and a warning is recorded.

    Parameters
    ----------
    df : pd.DataFrame
    required_features : list[str]

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to contain all required_features columns.
    """
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    return df


# ─────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using:
      1. Forward fill  (carries last known value forward)
      2. Backward fill (fills remaining NaNs at the start)
      3. Column median (fallback for columns that are entirely NaN)

    Returns a copy of the DataFrame.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Forward then backward fill for temporal continuity
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Fallback: fill still-missing with column median (or 0 if all-NaN)
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    return df


# ─────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────

def normalize_features(df: pd.DataFrame, features: list) -> tuple:
    """
    Apply StandardScaler normalisation to selected feature columns.

    Returns
    -------
    (df_normalised, scaler)
        df_normalised : DataFrame with the selected features scaled to z-scores.
        scaler        : Fitted StandardScaler instance (useful for inverse transforms).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler


# ─────────────────────────────────────────────
# Temporal Window Creation
# ─────────────────────────────────────────────

def create_temporal_windows(
    df: pd.DataFrame,
    features: list,
    window_size: int = 7,
) -> np.ndarray:
    """
    Slide a window of `window_size` days over the DataFrame to produce
    overlapping sequences suitable for the BiLSTM model.

    Parameters
    ----------
    df          : DataFrame (rows = days, columns include the feature columns)
    features    : ordered list of 24 feature column names
    window_size : number of consecutive days per window (default 7)

    Returns
    -------
    np.ndarray of shape (N, window_size, n_features)
        where N = max(0, len(df) - window_size + 1)
    """
    data = df[features].values.astype(np.float32)
    n_days, n_features = data.shape

    if n_days < window_size:
        # Pad with repeated first row so we always produce at least one window
        pad_rows = window_size - n_days
        pad = np.tile(data[0:1, :], (pad_rows, 1))
        data = np.vstack([pad, data])
        n_days = window_size

    windows = []
    for i in range(n_days - window_size + 1):
        windows.append(data[i : i + window_size, :])

    return np.array(windows, dtype=np.float32)


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────

def preprocess_pipeline(file_path_or_buffer) -> dict:
    """
    End-to-end preprocessing pipeline.

    Steps
    -----
    1. Load config
    2. Load CSV
    3. Validate features (add missing as NaN)
    4. Handle missing values
    5. Normalise features
    6. Create (N, 7, 24) temporal windows

    Returns
    -------
    dict with keys:
        'windows'    : np.ndarray (N, 7, 24)
        'df_raw'     : original DataFrame before normalisation
        'df_clean'   : DataFrame after missing-value handling (pre-scale)
        'df_scaled'  : DataFrame after normalisation
        'scaler'     : fitted StandardScaler
        'features'   : list of 24 feature names
        'n_days'     : number of usable days
        'metadata'   : dict with file summary info
    """
    config = load_config()
    features = config["features"]
    window_size = config["model"]["input_days"]  # 7

    # Load
    df_raw = load_patient_csv(file_path_or_buffer)

    # Validate
    df_validated = validate_features(df_raw, features)

    # Handle missing
    df_clean = handle_missing_values(df_validated)

    # Normalise
    df_scaled, scaler = normalize_features(df_clean, features)

    # Windows
    windows = create_temporal_windows(df_scaled, features, window_size)

    metadata = {
        "n_rows": len(df_raw),
        "n_days": len(df_scaled),
        "n_windows": len(windows),
        "features_found": [f for f in features if f in df_raw.columns],
        "features_imputed": [f for f in features if f not in df_raw.columns],
        "date_range": (
            str(df_clean["date"].min().date()) + " → " + str(df_clean["date"].max().date())
            if "date" in df_clean.columns and not df_clean["date"].isna().all()
            else "N/A"
        ),
    }

    return {
        "windows": windows,
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_scaled": df_scaled,
        "scaler": scaler,
        "features": features,
        "n_days": len(df_scaled),
        "metadata": metadata,
    }


# ─────────────────────────────────────────────
# Synthetic Patient Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_patient(scenario: str = "normal") -> pd.DataFrame:
    """
    Generate 30 days of realistic synthetic wearable data.

    Parameters
    ----------
    scenario : 'normal' | 'elevated' | 'critical'

    Returns
    -------
    pd.DataFrame  (30 rows × 25 columns: date + 24 features)
    """
    rng = np.random.default_rng(seed=42 if scenario == "normal" else (43 if scenario == "elevated" else 44))
    n = 30
    dates = [datetime.today().date() - timedelta(days=(n - 1 - i)) for i in range(n)]
    t = np.linspace(0, 1, n)  # trend variable [0 → 1]

    def add_noise(base, sigma, lo=None, hi=None):
        arr = base + rng.normal(0, sigma, n)
        if lo is not None:
            arr = np.clip(arr, lo, hi)
        return arr

    if scenario == "normal":
        hr             = add_noise(70 + 3 * np.sin(2 * np.pi * t), 3, 55, 90)
        rhr            = add_noise(60 + 2 * np.sin(2 * np.pi * t), 2, 50, 75)
        hrv            = add_noise(55 - 5 * np.sin(2 * np.pi * t), 6, 30, 90)
        spo2           = add_noise(98, 0.4, 96, 100)
        sleep_dur      = add_noise(7.5, 0.4, 6.5, 9)
        sleep_qual     = add_noise(78, 6, 60, 100)
        deep_sleep     = add_noise(1.8, 0.2, 1.0, 2.8)
        rem_sleep      = add_noise(2.0, 0.25, 1.2, 3.0)
        steps          = add_noise(9500, 1200, 5000, 16000)
        activity       = add_noise(62, 8, 35, 90)
        calories       = add_noise(2200, 180, 1600, 3000)
        distance       = add_noise(7.5, 1.2, 4, 14)
        stress         = add_noise(25, 5, 10, 45)
        cortisol       = add_noise(30, 6, 12, 55)
        movement       = add_noise(55, 8, 30, 80)
        sedentary      = add_noise(7.5, 0.8, 5, 11)
        skin_temp      = add_noise(34.5, 0.4, 33, 36.5)
        gsr            = add_noise(4.0, 0.8, 1.0, 9)
        breathing      = add_noise(15, 1.2, 11, 20)
        sys_bp         = add_noise(118, 5, 100, 135)
        dia_bp         = add_noise(75, 4, 62, 88)
        mood           = add_noise(76, 7, 55, 100)
        fatigue        = add_noise(22, 6, 5, 42)
        anxiety        = add_noise(18, 5, 5, 38)

    elif scenario == "elevated":
        hr             = add_noise(83 + 8 * t, 4, 68, 100)
        rhr            = add_noise(72 + 6 * t, 3, 60, 90)
        hrv            = add_noise(42 - 12 * t, 5, 18, 62)
        spo2           = add_noise(97 - 0.5 * t, 0.5, 94, 99)
        sleep_dur      = add_noise(5.8 - 0.8 * t, 0.4, 4.0, 7.0)
        sleep_qual     = add_noise(58 - 12 * t, 6, 35, 72)
        deep_sleep     = add_noise(1.2 - 0.3 * t, 0.2, 0.6, 1.8)
        rem_sleep      = add_noise(1.5 - 0.4 * t, 0.2, 0.8, 2.2)
        steps          = add_noise(6500 - 1000 * t, 900, 3000, 10000)
        activity       = add_noise(48 - 10 * t, 7, 22, 68)
        calories       = add_noise(1900 - 200 * t, 160, 1300, 2400)
        distance       = add_noise(5.2 - 0.8 * t, 0.9, 2.5, 8)
        stress         = add_noise(55 + 12 * t, 6, 35, 80)
        cortisol       = add_noise(58 + 10 * t, 7, 38, 88)
        movement       = add_noise(42 - 8 * t, 7, 20, 62)
        sedentary      = add_noise(9.5 + 1.2 * t, 0.8, 7, 13)
        skin_temp      = add_noise(35.2 + 0.5 * t, 0.4, 34, 37)
        gsr            = add_noise(7.5 + 2.0 * t, 1.0, 3.5, 14)
        breathing      = add_noise(17 + 2 * t, 1.5, 13, 22)
        sys_bp         = add_noise(128 + 8 * t, 5, 115, 148)
        dia_bp         = add_noise(82 + 5 * t, 4, 72, 95)
        mood           = add_noise(52 - 14 * t, 7, 28, 68)
        fatigue        = add_noise(50 + 14 * t, 7, 30, 78)
        anxiety        = add_noise(48 + 16 * t, 7, 28, 78)

    else:  # critical
        hr             = add_noise(95 + 10 * t, 5, 80, 120)
        rhr            = add_noise(82 + 8 * t, 4, 70, 100)
        hrv            = add_noise(22 - 8 * t, 4, 6, 36)
        spo2           = add_noise(95 - 1.5 * t, 0.6, 91, 98)
        sleep_dur      = add_noise(4.0 - 0.8 * t, 0.5, 2.5, 5.5)
        sleep_qual     = add_noise(38 - 10 * t, 6, 18, 55)
        deep_sleep     = add_noise(0.75 - 0.2 * t, 0.15, 0.3, 1.2)
        rem_sleep      = add_noise(0.9 - 0.3 * t, 0.2, 0.3, 1.5)
        steps          = add_noise(4200 - 1000 * t, 700, 1500, 7000)
        activity       = add_noise(30 - 8 * t, 6, 10, 48)
        calories       = add_noise(1650 - 250 * t, 150, 1000, 2200)
        distance       = add_noise(3.2 - 0.8 * t, 0.6, 1.2, 5.5)
        stress         = add_noise(78 + 12 * t, 6, 58, 100)
        cortisol       = add_noise(82 + 10 * t, 7, 62, 100)
        movement       = add_noise(28 - 8 * t, 5, 10, 45)
        sedentary      = add_noise(12 + 1.5 * t, 0.8, 9.5, 15)
        skin_temp      = add_noise(36.0 + 0.8 * t, 0.5, 35, 38.5)
        gsr            = add_noise(14 + 4 * t, 1.5, 8, 25)
        breathing      = add_noise(20 + 3 * t, 2, 16, 28)
        sys_bp         = add_noise(142 + 10 * t, 6, 128, 165)
        dia_bp         = add_noise(92 + 6 * t, 4, 82, 108)
        mood           = add_noise(32 - 12 * t, 6, 12, 50)
        fatigue        = add_noise(72 + 16 * t, 7, 55, 100)
        anxiety        = add_noise(75 + 15 * t, 7, 58, 100)

    df = pd.DataFrame({
        "date":                 dates,
        "heart_rate":           np.round(hr, 1),
        "resting_heart_rate":   np.round(rhr, 1),
        "hrv":                  np.round(hrv, 1),
        "spo2":                 np.round(spo2, 1),
        "sleep_duration":       np.round(sleep_dur, 2),
        "sleep_quality":        np.round(sleep_qual, 1),
        "deep_sleep":           np.round(deep_sleep, 2),
        "rem_sleep":            np.round(rem_sleep, 2),
        "steps":                np.round(steps).astype(int),
        "activity_level":       np.round(activity, 1),
        "calories":             np.round(calories).astype(int),
        "distance":             np.round(distance, 2),
        "stress_score":         np.round(stress, 1),
        "cortisol_proxy":       np.round(cortisol, 1),
        "movement_intensity":   np.round(movement, 1),
        "sedentary_time":       np.round(sedentary, 2),
        "skin_temperature":     np.round(skin_temp, 2),
        "galvanic_skin_response": np.round(gsr, 2),
        "breathing_rate":       np.round(breathing, 1),
        "systolic_bp":          np.round(sys_bp, 1),
        "diastolic_bp":         np.round(dia_bp, 1),
        "mood_score":           np.round(mood, 1),
        "fatigue_level":        np.round(fatigue, 1),
        "anxiety_proxy":        np.round(anxiety, 1),
    })

    return df
