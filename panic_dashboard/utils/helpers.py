"""
utils/helpers.py
Shared helper / formatting functions for PanicGuard AI.
"""

import random
import string
from datetime import datetime, date
from typing import Union

import numpy as np


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────

def format_percentage(value: float) -> str:
    """Format a 0–1 float as '65.3%'."""
    return f"{safe_float(value) * 100:.1f}%"


def format_score(value: float, decimals: int = 4) -> str:
    """Format a float to a fixed number of decimal places, e.g. '0.4902'."""
    return f"{safe_float(value):.{decimals}f}"


# ─────────────────────────────────────────────
# Risk Helpers
# ─────────────────────────────────────────────

def get_risk_color(score: float) -> str:
    """Return hex color string for a given risk probability."""
    s = safe_float(score)
    if s < 0.30:
        return "#2ecc71"   # green
    elif s < 0.60:
        return "#f39c12"   # orange
    else:
        return "#e74c3c"   # red


def get_risk_label(score: float) -> str:
    """Return human-readable risk label: 'Normal' / 'Elevated Risk' / 'Critical Risk'."""
    s = safe_float(score)
    if s < 0.30:
        return "Normal"
    elif s < 0.60:
        return "Elevated Risk"
    else:
        return "Critical Risk"


def get_risk_emoji(score: float) -> str:
    """Return a coloured circle emoji for the given risk level."""
    s = safe_float(score)
    if s < 0.30:
        return "🟢"
    elif s < 0.60:
        return "🟡"
    else:
        return "🔴"


# ─────────────────────────────────────────────
# Date / Time
# ─────────────────────────────────────────────

def days_ago_label(n: int) -> str:
    """
    Return a human-readable relative date label.

    Examples
    --------
    days_ago_label(0)  → 'Today'
    days_ago_label(1)  → 'Yesterday'
    days_ago_label(3)  → '3 days ago'
    """
    if n == 0:
        return "Today"
    elif n == 1:
        return "Yesterday"
    elif n < 7:
        return f"{n} days ago"
    elif n < 14:
        return "1 week ago"
    elif n < 30:
        weeks = n // 7
        return f"{weeks} weeks ago"
    else:
        months = n // 30
        return f"{months} month{'s' if months > 1 else ''} ago"


def timestamp_now() -> str:
    """Return current datetime as a formatted string: '2024-05-15 14:32:07'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────
# Patient ID Generation
# ─────────────────────────────────────────────

def generate_patient_id() -> str:
    """
    Generate a realistic 4-digit patient ID string.

    Example output: 'PT-2024-0847'
    """
    year = datetime.now().year
    number = random.randint(1000, 9999)
    return f"PT-{year}-{number:04d}"


# ─────────────────────────────────────────────
# Alert Record
# ─────────────────────────────────────────────

def create_alert_record(
    patient_id: str,
    risk_score: float,
    alert_level: str,
) -> dict:
    """
    Build a standardised alert record dictionary.

    Returns
    -------
    dict with keys: timestamp, patient_id, risk_score, risk_pct, alert_level
    """
    return {
        "timestamp":   timestamp_now(),
        "patient_id":  patient_id,
        "risk_score":  round(safe_float(risk_score), 4),
        "risk_pct":    format_percentage(risk_score),
        "alert_level": alert_level,
    }


# ─────────────────────────────────────────────
# Trend Analysis
# ─────────────────────────────────────────────

def calculate_trend(values: Union[list, np.ndarray]) -> str:
    """
    Determine whether a time series is trending upward, downward, or stable.

    Uses the slope of a linear regression over the last min(7, len) points.

    Returns
    -------
    'improving' | 'worsening' | 'stable'
    """
    if values is None or len(values) < 2:
        return "stable"

    arr = np.array([safe_float(v) for v in values])
    recent = arr[-min(7, len(arr)):]

    if len(recent) < 2:
        return "stable"

    x = np.arange(len(recent))
    # Linear regression slope
    slope = np.polyfit(x, recent, deg=1)[0]
    std   = float(np.std(recent)) or 1.0
    normalised_slope = slope / std

    if normalised_slope > 0.10:
        return "worsening"
    elif normalised_slope < -0.10:
        return "improving"
    else:
        return "stable"


# ─────────────────────────────────────────────
# Safe Type Conversion
# ─────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning `default` on failure.

    Handles None, NaN, strings, and other non-numeric types.
    """
    if value is None:
        return default
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default
