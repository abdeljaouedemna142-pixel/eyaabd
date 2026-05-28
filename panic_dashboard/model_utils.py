"""
model_utils.py
Model loading, inference, and demo-mode simulation for PanicGuard AI.
"""

import os
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# Feature & scenario constants
# ─────────────────────────────────────────────

FEATURES_24 = [
    "heart_rate", "resting_heart_rate", "hrv", "spo2",
    "sleep_duration", "sleep_quality", "deep_sleep", "rem_sleep",
    "steps", "activity_level", "calories", "distance",
    "stress_score", "cortisol_proxy", "movement_intensity", "sedentary_time",
    "skin_temperature", "galvanic_skin_response", "breathing_rate",
    "systolic_bp", "diastolic_bp", "mood_score", "fatigue_level", "anxiety_proxy",
]

# Which features signal risk (used for demo attention weighting)
RISK_FEATURES = {
    "critical": ["stress_score", "anxiety_proxy", "hrv", "heart_rate",
                 "fatigue_level", "sleep_duration", "cortisol_proxy", "galvanic_skin_response"],
    "elevated": ["stress_score", "sleep_duration", "hrv", "fatigue_level",
                 "heart_rate", "activity_level"],
    "normal":   ["activity_level", "sleep_quality", "steps", "hrv", "mood_score"],
}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_model_cached(model_path: str):
    """
    Load a Keras model with caching.

    Returns the model object, or None if the file is not found or
    TensorFlow/Keras is not available.
    """
    if not os.path.isfile(model_path):
        return None

    try:
        # Lazy import so the app still works without TensorFlow installed
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        return model
    except ImportError:
        st.warning("TensorFlow is not installed – running in demo mode.")
        return None
    except Exception as e:
        st.error(f"Could not load model from '{model_path}': {e}")
        return None


# ─────────────────────────────────────────────
# Real Model Prediction
# ─────────────────────────────────────────────

def predict_risk(model, windows: np.ndarray) -> np.ndarray:
    """
    Run inference with the real Keras model.

    Parameters
    ----------
    model   : Keras model object (loaded via load_model_cached)
    windows : np.ndarray of shape (N, 7, 24)

    Returns
    -------
    np.ndarray of shape (N,) with probabilities in [0, 1]
    """
    if model is None:
        raise ValueError("Model is None – use predict_demo() instead.")

    try:
        preds = model.predict(windows, verbose=0)
        # Support both (N,1) and (N,) output shapes
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds[:, 0]
        return preds.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")


# ─────────────────────────────────────────────
# Demo Mode Prediction
# ─────────────────────────────────────────────

def predict_demo(windows: np.ndarray, scenario: str = "normal") -> np.ndarray:
    """
    Simulate model predictions without a real model.

    Parameters
    ----------
    windows  : np.ndarray of shape (N, 7, 24) – used for shape only
    scenario : 'normal' | 'elevated' | 'critical'

    Returns
    -------
    np.ndarray of shape (N,) with simulated probabilities
    """
    rng = np.random.default_rng(seed=7)
    n = max(windows.shape[0], 1)

    if scenario == "normal":
        base = rng.uniform(0.05, 0.25, n)
        # Slight downward trend  (healthy recovery)
        trend = -0.02 * np.linspace(0, 1, n)
    elif scenario == "elevated":
        base = rng.uniform(0.35, 0.58, n)
        trend = 0.04 * np.linspace(0, 1, n)   # gradual worsening
    else:  # critical
        base = rng.uniform(0.65, 0.90, n)
        trend = 0.06 * np.linspace(0, 1, n)

    noise = rng.normal(0, 0.015, n)
    probs = np.clip(base + trend + noise, 0.0, 1.0).astype(np.float32)
    return probs


# ─────────────────────────────────────────────
# Risk Categorisation
# ─────────────────────────────────────────────

def get_risk_category(score: float) -> tuple:
    """
    Return (category, hex_color, emoji) for a given risk probability.

    Thresholds: 0–0.30 normal | 0.30–0.60 elevated | 0.60–1.0 critical
    """
    if score < 0.30:
        return ("normal",   "#2ecc71", "🟢")
    elif score < 0.60:
        return ("elevated", "#f39c12", "🟡")
    else:
        return ("critical", "#e74c3c", "🔴")


# ─────────────────────────────────────────────
# Attention Weights (Demo)
# ─────────────────────────────────────────────

def get_attention_weights_demo(
    window_size: int = 7,
    n_features: int = 24,
    scenario: str = "normal",
) -> dict:
    """
    Generate simulated attention weights for explainability visualisations.

    Returns
    -------
    dict with:
        'temporal_weights' : np.ndarray (window_size,)   – sum = 1
        'feature_weights'  : np.ndarray (n_features,)    – sum = 1
        'attention_matrix' : np.ndarray (window_size, n_features) – row-normalised
    """
    rng = np.random.default_rng(seed=42)

    # --- Temporal weights ---
    if scenario == "critical":
        # Recent days matter most
        raw_t = np.array([0.04, 0.06, 0.08, 0.12, 0.18, 0.24, 0.28])
    elif scenario == "elevated":
        raw_t = np.array([0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.22])
    else:
        raw_t = np.ones(window_size)

    raw_t += rng.uniform(0, 0.02, window_size)
    temporal_weights = raw_t / raw_t.sum()

    # --- Feature weights ---
    base_fw = np.ones(n_features) * 0.5
    feature_names = FEATURES_24[:n_features]

    priority_map = {
        "critical": {"stress_score": 2.8, "anxiety_proxy": 2.6, "hrv": 2.5,
                     "heart_rate": 2.3, "fatigue_level": 2.1, "sleep_duration": 2.0,
                     "cortisol_proxy": 1.9, "galvanic_skin_response": 1.8,
                     "resting_heart_rate": 1.6, "sleep_quality": 1.5},
        "elevated": {"stress_score": 2.2, "sleep_duration": 2.0, "hrv": 1.9,
                     "fatigue_level": 1.8, "heart_rate": 1.7, "activity_level": 1.5,
                     "anxiety_proxy": 1.6, "sleep_quality": 1.4},
        "normal":   {"activity_level": 2.0, "sleep_quality": 1.9, "steps": 1.7,
                     "hrv": 1.6, "mood_score": 1.5},
    }

    pmap = priority_map.get(scenario, {})
    for i, fn in enumerate(feature_names):
        if fn in pmap:
            base_fw[i] = pmap[fn]

    base_fw += rng.uniform(0, 0.15, n_features)
    feature_weights = base_fw / base_fw.sum()

    # --- 2-D attention matrix ---
    attn_matrix = np.outer(temporal_weights, feature_weights)
    # Row normalise
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    attn_matrix = attn_matrix / row_sums

    return {
        "temporal_weights": temporal_weights.astype(np.float32),
        "feature_weights":  feature_weights.astype(np.float32),
        "attention_matrix": attn_matrix.astype(np.float32),
    }


# ─────────────────────────────────────────────
# Feature Importance (Demo)
# ─────────────────────────────────────────────

def compute_feature_importance_demo(scenario: str = "normal") -> dict:
    """
    Return a dictionary of {feature_name: importance_score (0–1)}
    for the given scenario.

    Scores reflect domain knowledge about panic-relevant physiological signals.
    """
    base_scores = {
        "heart_rate":             0.55,
        "resting_heart_rate":     0.48,
        "hrv":                    0.72,
        "spo2":                   0.30,
        "sleep_duration":         0.68,
        "sleep_quality":          0.62,
        "deep_sleep":             0.50,
        "rem_sleep":              0.45,
        "steps":                  0.38,
        "activity_level":         0.42,
        "calories":               0.28,
        "distance":               0.22,
        "stress_score":           0.85,
        "cortisol_proxy":         0.76,
        "movement_intensity":     0.35,
        "sedentary_time":         0.44,
        "skin_temperature":       0.33,
        "galvanic_skin_response": 0.60,
        "breathing_rate":         0.54,
        "systolic_bp":            0.40,
        "diastolic_bp":           0.36,
        "mood_score":             0.58,
        "fatigue_level":          0.70,
        "anxiety_proxy":          0.88,
    }

    multipliers = {
        "critical": {
            "stress_score": 1.18, "anxiety_proxy": 1.15, "hrv": 1.20,
            "heart_rate": 1.12, "fatigue_level": 1.10, "sleep_duration": 1.08,
            "cortisol_proxy": 1.12, "galvanic_skin_response": 1.10,
            "mood_score": 0.85, "steps": 0.70, "calories": 0.65,
        },
        "elevated": {
            "stress_score": 1.10, "sleep_duration": 1.08, "hrv": 1.10,
            "fatigue_level": 1.05, "heart_rate": 1.06, "anxiety_proxy": 1.08,
            "activity_level": 0.90, "distance": 0.80,
        },
        "normal": {
            "activity_level": 1.15, "sleep_quality": 1.12, "steps": 1.10,
            "mood_score": 1.08, "hrv": 1.05,
            "stress_score": 0.75, "anxiety_proxy": 0.70,
        },
    }

    mmap = multipliers.get(scenario, {})
    result = {}
    for feat, score in base_scores.items():
        m = mmap.get(feat, 1.0)
        result[feat] = float(np.clip(score * m, 0.0, 1.0))

    return result


# ─────────────────────────────────────────────
# Risk Explanation
# ─────────────────────────────────────────────

def format_risk_explanation(
    score: float,
    attention_weights: dict,
    feature_importance: dict,
    feature_names: list,
) -> list:
    """
    Produce a list of human-readable clinical interpretation strings.

    Parameters
    ----------
    score             : risk probability (0–1)
    attention_weights : dict from get_attention_weights_demo()
    feature_importance: dict from compute_feature_importance_demo()
    feature_names     : list of 24 feature name strings

    Returns
    -------
    list[str]  – ordered explanation bullets
    """
    category, _, _ = get_risk_category(score)
    tw = attention_weights.get("temporal_weights", np.ones(7) / 7)
    fw = attention_weights.get("feature_weights",  np.ones(24) / 24)

    # Most important temporal window
    peak_day_idx = int(np.argmax(tw))
    days_back = 7 - peak_day_idx - 1
    if days_back == 0:
        day_label = "today"
    elif days_back == 1:
        day_label = "yesterday"
    else:
        day_label = f"{days_back} days ago"

    # Top 3 features
    sorted_feats = sorted(zip(feature_names, fw), key=lambda x: x[1], reverse=True)[:3]
    top_feat_labels = [f.replace("_", " ").title() for f, _ in sorted_feats]

    # Top 3 importance features
    sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_imp_labels = [f.replace("_", " ").title() for f, _ in sorted_imp]

    explanations = [
        f"**Overall Risk Score:** {score * 100:.1f}% — classified as **{category.upper()}**.",
        f"**Peak Temporal Relevance:** The model assigned the highest attention weight "
        f"to data recorded **{day_label}** (window position {peak_day_idx + 1}/7). "
        f"Recent physiological changes are driving this assessment.",
        f"**Most Influential Signals (Attention):** "
        f"{', '.join(top_feat_labels)} received the highest attention scores, "
        f"suggesting these signals carried the most discriminative information "
        f"in the temporal window.",
        f"**Feature Importance (SHAP-style Demo):** "
        f"{', '.join(top_imp_labels)} are estimated to be the top contributing "
        f"features to the risk classification.",
    ]

    if category == "critical":
        explanations += [
            "**Clinical Note:** The model detects a pattern consistent with elevated "
            "autonomic dysregulation — characterised by reduced HRV, elevated stress "
            "markers, and disrupted sleep architecture over the past week.",
            "**Recommendation:** This alert should trigger a clinical review. "
            "Physiological data suggests sustained sympathetic nervous system activation "
            "potentially indicative of pre-panic phenotype.",
        ]
    elif category == "elevated":
        explanations += [
            "**Clinical Note:** Early warning indicators are present. Stress and sleep "
            "metrics show a deteriorating trend. Close monitoring is advised over the "
            "next 3–5 days.",
            "**Recommendation:** Consider lifestyle interventions (sleep hygiene, stress "
            "reduction) and schedule a follow-up assessment.",
        ]
    else:
        explanations += [
            "**Clinical Note:** Physiological markers are within expected normal ranges. "
            "The model detected no significant risk patterns in the current window.",
            "**Recommendation:** Continue routine monitoring. Maintain current health "
            "behaviours.",
        ]

    return explanations
