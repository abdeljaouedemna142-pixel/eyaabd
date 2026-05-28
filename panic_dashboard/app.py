"""
app.py — PanicGuard AI · Home Page
Streamlit-based medical AI dashboard for panic attack risk detection.

Run with:
    streamlit run app.py
"""

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Page Configuration (MUST be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PanicGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "PanicGuard AI — Master Thesis Research Prototype",
    },
)

# ─────────────────────────────────────────────
# Imports (after set_page_config)
# ─────────────────────────────────────────────

from model_utils import load_model_cached
from utils.helpers import generate_patient_id, timestamp_now


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_config() -> dict:
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "config.json"), "r") as f:
        return json.load(f)


def load_css() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base, "assets", "style.css")
    if os.path.isfile(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state(config: dict) -> None:
    """Initialise all session-state variables with sensible defaults."""
    defaults = {
        "patient_data":       None,          # pd.DataFrame of raw patient CSV
        "alerts":             [],             # list of alert record dicts
        "current_risk_score": None,           # float 0–1 or None
        "prediction_history": [],             # list of {date, score, level}
        "patient_id":         generate_patient_id(),
        "selected_scenario":  "normal",       # 'normal' | 'elevated' | 'critical'
        "notification_config": config.get("notifications", {}),
        "model_loaded":       False,
        "demo_mode":          True,
        "preprocessing_result": None,         # dict from preprocess_pipeline()
        "attention_weights":  None,           # dict from get_attention_weights_demo()
        "feature_importance": None,           # dict from compute_feature_importance_demo()
        "last_prediction_time": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    load_css()
    config = load_config()
    init_session_state(config)

    # Try to load the real model (cached)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              config["model"]["path"])
    model = load_model_cached(model_path)
    st.session_state.model_loaded = model is not None
    st.session_state.demo_mode    = model is None

    perf  = config["performance"]
    feat_list = config["features"]

    # ─── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center;padding:12px 0 20px'>
              <span style='font-size:3rem'>🫀</span><br>
              <span style='font-size:1.3rem;font-weight:700;color:#e8eaf0'>PanicGuard AI</span><br>
              <span style='font-size:0.75rem;color:#6b7280'>v1.0.0 · Research Prototype</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # System status
        if st.session_state.model_loaded:
            st.markdown(
                "<div style='padding:8px 12px;background:rgba(46,204,113,0.12);border:1px solid #2ecc71;"
                "border-radius:8px;margin-bottom:12px'>"
                "<span style='color:#2ecc71;font-weight:600'>🟢 Model Loaded</span><br>"
                "<span style='color:#6b7280;font-size:0.8rem'>BiLSTM + Attention active</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:8px 12px;background:rgba(243,156,18,0.12);border:1px solid #f39c12;"
                "border-radius:8px;margin-bottom:12px'>"
                "<span style='color:#f39c12;font-weight:600'>🟡 Demo Mode</span><br>"
                "<span style='color:#6b7280;font-size:0.8rem'>Model file not found — simulated predictions</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Current risk badge
        score = st.session_state.current_risk_score
        if score is not None:
            from utils.helpers import get_risk_color, get_risk_label, get_risk_emoji
            color = get_risk_color(score)
            label = get_risk_label(score)
            emoji = get_risk_emoji(score)
            st.markdown(
                f"<div style='text-align:center;margin-bottom:12px'>"
                f"<div style='font-size:0.75rem;color:#6b7280;margin-bottom:4px'>CURRENT RISK</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{color}'>"
                f"{emoji} {score*100:.1f}%</div>"
                f"<div style='font-size:0.85rem;color:{color}'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='color:#6b7280;font-size:0.78rem;text-align:center;margin-bottom:8px'>"
            f"📅 {datetime.now().strftime('%B %d, %Y')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='padding:8px 12px;background:rgba(231,76,60,0.07);border:1px solid rgba(231,76,60,0.2);"
            "border-radius:6px;font-size:0.72rem;color:#9ca3af;line-height:1.5'>"
            "⚠ <strong style='color:#e74c3c'>Research Only</strong><br>"
            "Not a medical device. Not for clinical diagnosis."
            "</div>",
            unsafe_allow_html=True,
        )

    # ─── Hero Section ──────────────────────────────────────
    st.markdown(
        """
        <div class='hero-section'>
          <div class='gradient-title'>🫀 PanicGuard AI</div>
          <p class='hero-subtitle'>
            Bidirectional LSTM + Attention for Panic Attack Risk Detection from Wearable Biosignals<br>
            <em>Master Thesis Research Prototype · 2024</em>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Top Metric Cards ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    cards = [
        ("🧠", "BiLSTM + Attention", "Architecture", "#4F8BF9"),
        ("🎯", f"{perf['recall_mean']:.2%}", f"Recall ± {perf['recall_std']:.2%}", "#2ecc71"),
        ("📈", f"{perf['roc_auc_mean']:.4f}", f"ROC-AUC ± {perf['roc_auc_std']:.4f}", "#f39c12"),
        ("📅", "7 Days × 24 Signals", "Input Window", "#7C5CBF"),
    ]

    for col, (icon, value, label, color) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class='metric-card'>
                  <span class='card-icon'>{icon}</span>
                  <div class='card-value' style='color:{color}'>{value}</div>
                  <div class='card-label'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── About + Performance ────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("## About PanicGuard AI")
        st.markdown(
            """
            **PanicGuard AI** is a research prototype developed as part of a master thesis
            investigating the automatic detection of panic attack risk from wearable biosignals
            using deep learning.

            The system monitors **24 physiological and behavioural features** collected over
            a rolling 7-day window from consumer-grade wearable devices (e.g., smartwatches,
            fitness trackers) and uses a **Bidirectional LSTM with an Attention mechanism**
            to generate a continuous risk probability.

            ### Core Capabilities
            """
        )

        capabilities = [
            ("🔬", "Real-time Risk Scoring", "Continuous 0–100% risk probability updated with each new day of data"),
            ("🧠", "Attention Explainability", "Identifies which of the 7 days and 24 signals drove the model's decision"),
            ("🚨", "Multi-channel Alerts", "Configurable email (SMTP) and Telegram notifications for clinical thresholds"),
            ("📊", "Trend Analysis", "30-day physiological history with recovery and deterioration tracking"),
            ("📁", "CSV Upload", "Accepts patient data in standard CSV format matching wearable exports"),
            ("🔒", "Demo Mode", "Operates fully without the real model file using realistic simulations"),
        ]

        for icon, title, desc in capabilities:
            st.markdown(
                f"<div class='info-card blue-border'>"
                f"<strong style='color:#e8eaf0'>{icon} {title}</strong><br>"
                f"<span style='font-size:0.85rem;color:#9ca3af'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("## Model Performance")
        st.markdown(
            "<p style='font-size:0.85rem;color:#6b7280;margin-bottom:12px'>"
            "LOSO cross-validation results (Leave-One-Subject-Out)</p>",
            unsafe_allow_html=True,
        )

        metrics = [
            ("ROC-AUC",  perf["roc_auc_mean"], perf["roc_auc_std"],  0.49, "#f39c12",
             "Near-chance — high inter-subject variability"),
            ("PR-AUC",   perf["pr_auc_mean"],  perf["pr_auc_std"],   0.22, "#4F8BF9",
             "Imbalanced classes challenge precision"),
            ("F1 Score", perf["f1_mean"],       perf["f1_std"],       0.29, "#7C5CBF",
             "Reflects precision–recall trade-off"),
            ("Recall",   perf["recall_mean"],   perf["recall_std"],   0.81, "#2ecc71",
             "Optimised — clinical sensitivity priority"),
        ]

        for metric_name, mean_val, std_val, _, color, note in metrics:
            pct = int(mean_val * 100)
            st.markdown(
                f"""
                <div style='margin-bottom:14px'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                    <span style='color:#e8eaf0;font-weight:600;font-size:0.9rem'>{metric_name}</span>
                    <span style='color:{color};font-weight:700;font-size:0.9rem'>
                      {mean_val:.4f} <span style='color:#6b7280;font-size:0.75rem'>±{std_val:.4f}</span>
                    </span>
                  </div>
                  <div class='risk-bar-container'>
                    <div class='risk-bar-fill' style='width:{pct}%;background:{color}'></div>
                  </div>
                  <div style='font-size:0.75rem;color:#6b7280;margin-top:2px'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='info-card orange-border' style='margin-top:16px'>"
            "<strong style='color:#f39c12'>📋 Validation Strategy</strong><br>"
            "<span style='font-size:0.82rem;color:#9ca3af'>"
            "Leave-One-Subject-Out (LOSO) cross-validation. Each subject's data is held out "
            "as a test set while the model is trained on remaining subjects — reflecting "
            "realistic generalisation to unseen individuals."
            "</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ─── Navigation Cards ──────────────────────────────────
    st.markdown("## Dashboard Navigation")

    pages = [
        ("🏥", "Patient Monitoring", "pages/1_🏥_Patient_Monitoring.py",
         "Upload patient CSV or select a demo scenario to run the risk prediction pipeline"),
        ("🧠", "AI Explainability", "pages/2_🧠_AI_Explainability.py",
         "Inspect temporal attention weights and feature importance for the latest prediction"),
        ("🚨", "Alert Center", "pages/3_🚨_Alert_Center.py",
         "Configure email/Telegram notifications and send clinical alerts"),
        ("📊", "Patient History", "pages/4_📊_Patient_History.py",
         "View 30-day risk trend, alert log, and physiological evolution"),
        ("📖", "About Thesis", "pages/5_📖_About_Thesis.py",
         "Read the thesis background, model details, results, and ethical considerations"),
        ("⚙️", "Configuration", None,
         "Adjust thresholds, notification settings, and system preferences via config.json"),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)
    grid = row1 + row2

    for col, (icon, title, _, desc) in zip(grid, pages):
        with col:
            st.markdown(
                f"""
                <div class='nav-card'>
                  <span class='nav-icon'>{icon}</span>
                  <div class='nav-title'>{title}</div>
                  <div class='nav-desc'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ─── Monitored Signals Table ────────────────────────────
    st.markdown("## 24 Monitored Signals")

    feat_desc = config.get("feature_descriptions", {})
    normal_ranges = config.get("normal_ranges", {})
    categories = {
        "Cardiovascular": ["heart_rate", "resting_heart_rate", "hrv", "systolic_bp", "diastolic_bp"],
        "Sleep & Recovery": ["sleep_duration", "sleep_quality", "deep_sleep", "rem_sleep"],
        "Activity": ["steps", "activity_level", "calories", "distance", "movement_intensity", "sedentary_time"],
        "Stress & Autonomic": ["stress_score", "cortisol_proxy", "galvanic_skin_response",
                               "skin_temperature", "breathing_rate"],
        "Psychophysiological": ["spo2", "mood_score", "fatigue_level", "anxiety_proxy"],
    }

    cat_cols = st.columns(len(categories))
    for col, (cat_name, feats) in zip(cat_cols, categories.items()):
        with col:
            st.markdown(f"**{cat_name}**")
            for feat in feats:
                nr = normal_ranges.get(feat, ["–", "–"])
                desc_short = feat_desc.get(feat, feat).split(" (")[0]
                st.markdown(
                    f"<div style='padding:4px 0;border-bottom:1px solid #1e2130'>"
                    f"<span style='color:#e8eaf0;font-size:0.82rem'>{desc_short}</span><br>"
                    f"<span style='color:#4F8BF9;font-size:0.72rem'>Normal: {nr[0]}–{nr[1]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ─── Ethical Disclaimer ────────────────────────────────
    st.markdown(
        """
        <div class='disclaimer-box'>
          <div class='disclaimer-title'>⚠ Research Prototype — Ethical Disclaimer</div>
          <div class='disclaimer-text'>
            <strong>PanicGuard AI is NOT a medical device and has NOT been clinically validated.</strong>
            It is an academic research prototype developed for a master thesis and must not be
            used for clinical diagnosis, treatment decisions, or emergency response.
            <br><br>
            The AI model was trained and evaluated on a small research dataset using LOSO cross-validation.
            The ROC-AUC of 0.4902 ± 0.2082 indicates near-chance discriminative performance with
            high inter-subject variability. These results reflect the difficulty of generalising
            panic risk detection across individuals without personalised calibration.
            <br><br>
            If you or someone you know is experiencing a mental health crisis, please contact a
            qualified healthcare professional or emergency services immediately.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Footer ────────────────────────────────────────────
    st.markdown(
        f"""
        <div class='footer'>
          🫀 <strong>PanicGuard AI</strong> — Master Thesis Research Prototype · 2024<br>
          BiLSTM + Attention · Leave-One-Subject-Out Validation · 24 Biosignals<br>
          <span style='color:#4d5568'>Built with Streamlit · TensorFlow/Keras · Plotly</span><br>
          <span style='color:#374151;font-size:0.72rem'>
            {datetime.now().strftime('%Y')} · Research use only · Not for clinical deployment
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
