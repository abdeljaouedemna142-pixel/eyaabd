"""
pages/2_🧠_AI_Explainability.py
AI Explainability page — attention weights, feature importance, and clinical interpretation.
"""

import os
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="AI Explainability — PanicGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model_utils import (
    get_risk_category,
    get_attention_weights_demo,
    compute_feature_importance_demo,
    format_risk_explanation,
    FEATURES_24,
)
from visualization_utils import (
    plot_attention_heatmap,
    plot_feature_importance,
)
from utils.helpers import generate_patient_id, get_risk_color, get_risk_label, get_risk_emoji


# ── Helpers ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.json"), "r") as f:
        return json.load(f)


def _load_css() -> None:
    css_path = os.path.join(_ROOT, "assets", "style.css")
    if os.path.isfile(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _init_session(config: dict) -> None:
    defaults = {
        "patient_data":         None,
        "alerts":               [],
        "current_risk_score":   None,
        "prediction_history":   [],
        "patient_id":           generate_patient_id(),
        "selected_scenario":    "normal",
        "notification_config":  config.get("notifications", {}),
        "model_loaded":         False,
        "demo_mode":            True,
        "preprocessing_result": None,
        "attention_weights":    None,
        "feature_importance":   None,
        "last_prediction_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _attention_matrix_fig(attention_weights: dict, feature_names: list) -> go.Figure:
    """2-D attention matrix heatmap: days × features."""
    matrix  = attention_weights.get("attention_matrix")
    if matrix is None:
        tw = attention_weights.get("temporal_weights", np.ones(7) / 7)
        fw = attention_weights.get("feature_weights",  np.ones(24) / 24)
        matrix = np.outer(tw, fw)

    day_labels  = [f"Day -{6-i}" if i < 6 else "Today" for i in range(matrix.shape[0])]
    feat_labels = [f.replace("_", " ").title() for f in feature_names[:matrix.shape[1]]]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=feat_labels,
        y=day_labels,
        colorscale=[
            [0.0,  "#0d1117"],
            [0.25, "#16213e"],
            [0.5,  "#4F8BF9"],
            [0.75, "#7C5CBF"],
            [1.0,  "#e74c3c"],
        ],
        hovertemplate="<b>%{y}</b> / <b>%{x}</b><br>Attention: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Attention", tickfont=dict(color="#e8eaf0"),
                      titlefont=dict(color="#e8eaf0")),
    ))
    fig.update_layout(
        title="2-D Attention Matrix (Days × Features)",
        height=350,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#1e2130",
        font=dict(color="#e8eaf0"),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="#2d3748"),
        yaxis=dict(tickfont=dict(size=10), gridcolor="#2d3748"),
        margin=dict(l=80, r=60, t=60, b=130),
    )
    return fig


def _temporal_bar_fig(temporal_weights: np.ndarray) -> go.Figure:
    day_labels = [f"Day -{6-i}" if i < 6 else "Today" for i in range(len(temporal_weights))]
    max_w = float(np.max(temporal_weights)) + 1e-8
    colors = [
        f"rgba(79,139,249,{0.3 + 0.7 * float(w) / max_w})"
        for w in temporal_weights
    ]
    fig = go.Figure(go.Bar(
        x=day_labels,
        y=temporal_weights.tolist(),
        marker_color=colors,
        marker_line=dict(color="#2d3748", width=0.5),
        hovertemplate="<b>%{x}</b><br>Attention weight: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Temporal Attention — Which Days Mattered Most",
        height=320,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#1e2130",
        font=dict(color="#e8eaf0"),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(title="Normalised attention weight", gridcolor="#2d3748"),
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def _comparison_fig(feature_names: list) -> go.Figure:
    """Side-by-side feature importance comparison across all three scenarios."""
    scenarios = ["normal", "elevated", "critical"]
    colors    = {"normal": "#2ecc71", "elevated": "#f39c12", "critical": "#e74c3c"}
    feat_labels = [f.replace("_", " ").title() for f in feature_names]

    fig = go.Figure()
    for sc in scenarios:
        imp   = compute_feature_importance_demo(sc)
        vals  = [imp.get(f, 0.0) for f in feature_names]
        fig.add_trace(go.Bar(
            name=sc.title(),
            x=feat_labels,
            y=vals,
            marker_color=colors[sc],
            opacity=0.85,
            hovertemplate=f"<b>{sc.title()}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Feature Importance Comparison Across Scenarios",
        height=420,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#1e2130",
        font=dict(color="#e8eaf0"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#2d3748", borderwidth=1),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="#2d3748"),
        yaxis=dict(title="Importance Score", gridcolor="#2d3748", range=[0, 1.05]),
        margin=dict(l=60, r=20, t=60, b=130),
    )
    return fig


# ── Main ─────────────────────────────────────────────────────

def main():
    _load_css()
    config = _load_config()
    _init_session(config)

    feature_names = config.get("features", FEATURES_24)

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 16px'>"
            "<span style='font-size:2.5rem'>🧠</span><br>"
            "<span style='font-weight:700;color:#e8eaf0;font-size:1.1rem'>AI Explainability</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        score = st.session_state.current_risk_score
        if score is not None:
            color = get_risk_color(score)
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:rgba(79,139,249,0.08);"
                f"border-radius:8px;border:1px solid {color}33'>"
                f"<div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Current Risk</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{color}'>{score*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:{color}'>{get_risk_emoji(score)} {get_risk_label(score)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No prediction yet.\nGo to Patient Monitoring first.")

        st.divider()
        st.markdown(
            "<div class='info-card blue-border' style='font-size:0.82rem;color:#9ca3af'>"
            "<strong style='color:#4F8BF9'>About Explainability</strong><br>"
            "Attention weights are produced by the BiLSTM attention layer. "
            "Higher weight = greater influence on the risk prediction. "
            "Feature importance scores are derived from attention-weighted activations."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Guard: need a prediction ─────────────────────────────
    if st.session_state.current_risk_score is None:
        st.markdown("## 🧠 AI Explainability")
        st.info(
            "**No prediction found.** Please go to **🏥 Patient Monitoring**, "
            "load patient data and run the prediction first."
        )
        st.markdown(
            "<div style='text-align:center;padding:3rem;background:#1e2130;"
            "border-radius:16px;border:2px dashed #2d3748;color:#6b7280;margin-top:1rem'>"
            "<span style='font-size:3rem'>🧠</span><br><br>"
            "<strong style='color:#9ca3af'>Awaiting prediction</strong><br>"
            "<span style='font-size:0.85rem'>Run a prediction in Patient Monitoring to unlock explainability.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    score   = st.session_state.current_risk_score
    scenario = st.session_state.selected_scenario
    category, color, emoji = get_risk_category(score)

    # ── Ensure weights are available ─────────────────────────
    if st.session_state.attention_weights is None:
        st.session_state.attention_weights = get_attention_weights_demo(scenario=scenario)
    if st.session_state.feature_importance is None:
        st.session_state.feature_importance = compute_feature_importance_demo(scenario=scenario)

    aw = st.session_state.attention_weights
    fi = st.session_state.feature_importance
    tw = aw["temporal_weights"]
    fw = aw["feature_weights"]

    # ── Page Header ─────────────────────────────────────────
    st.markdown("## 🧠 AI Explainability")
    st.markdown(
        f"<span style='color:#6b7280;font-size:0.9rem'>"
        f"Patient: <strong style='color:#4F8BF9'>{st.session_state.patient_id}</strong> &nbsp;·&nbsp; "
        f"Risk score: <strong style='color:{color}'>{score*100:.1f}% — {category.upper()}</strong>"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Explainability overview card ─────────────────────────
    st.markdown(
        f"<div class='info-card {category}-border' style='margin-bottom:1.5rem'>"
        f"<strong style='color:{color}'>{emoji} Explainability Overview — {category.upper()} Risk ({score*100:.1f}%)</strong><br>"
        f"<span style='font-size:0.85rem;color:#9ca3af'>"
        "The BiLSTM encoder processes the 7-day physiological window across all 24 features. "
        "The attention layer assigns a learned weight to each time-step, indicating its relevance "
        "to the final risk classification. Feature importance is derived from the magnitude of "
        "attention-weighted hidden states — analogous to a first-order sensitivity analysis."
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Temporal Attention ────────────────────────
    st.markdown("### 📅 Temporal Attention — 7-Day Window")

    t_col, t_info = st.columns([2, 1], gap="large")

    with t_col:
        st.plotly_chart(_temporal_bar_fig(tw), use_container_width=True)

    with t_info:
        peak_idx   = int(np.argmax(tw))
        days_back  = 6 - peak_idx
        day_label  = "Today" if days_back == 0 else ("Yesterday" if days_back == 1 else f"{days_back} days ago")
        st.markdown(
            f"<div class='info-card blue-border' style='margin-top:1rem'>"
            f"<strong style='color:#4F8BF9'>Peak Temporal Relevance</strong><br>"
            f"<span style='font-size:0.82rem;color:#9ca3af'>"
            f"The model placed highest attention on data recorded <strong style='color:#e8eaf0'>{day_label}</strong> "
            f"(window position {peak_idx + 1}/7). "
            f"Attention weight: <strong style='color:#4F8BF9'>{tw[peak_idx]:.4f}</strong>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        for i, (d_lbl, w) in enumerate(
            zip([f"Day -{6-j}" if j < 6 else "Today" for j in range(7)], tw.tolist())
        ):
            bar_pct = int(w / (float(np.max(tw)) + 1e-8) * 100)
            bar_col = "#e74c3c" if i == peak_idx else "#4F8BF9"
            st.markdown(
                f"<div class='feature-bar'>"
                f"<div class='feature-bar-label'>{d_lbl}</div>"
                f"<div class='feature-bar-track'>"
                f"<div class='feature-bar-fill' style='width:{bar_pct}%;background:{bar_col}'></div>"
                f"</div>"
                f"<div class='feature-bar-value'>{w:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Section 2: Feature Attention ────────────────────────
    st.markdown("### 📊 Feature Attention — 24 Biosignals")
    st.plotly_chart(
        plot_attention_heatmap(tw, fw, feature_names),
        use_container_width=True,
    )

    st.divider()

    # ── Section 3: Feature Importance ───────────────────────
    st.markdown("### 🔥 Feature Importance (BiLSTM Attention Proxy)")
    st.caption(
        "Importance is estimated from attention-weighted hidden state activations — "
        "analogous to a first-order sensitivity measure. Red = top-10 features. Blue = remaining."
    )
    st.plotly_chart(
        plot_feature_importance(fi, feature_names),
        use_container_width=True,
    )

    st.divider()

    # ── Section 4: 2D Attention Matrix ──────────────────────
    st.markdown("### 🗺 2-D Attention Matrix (Days × Features)")
    st.caption("Each cell represents the combined temporal × feature attention weight for that day–signal pair.")
    st.plotly_chart(_attention_matrix_fig(aw, feature_names), use_container_width=True)

    st.divider()

    # ── Section 5: Clinical Interpretation ──────────────────
    st.markdown("### 🩺 Clinical Interpretation")

    explanations = format_risk_explanation(score, aw, fi, feature_names)
    for i, exp in enumerate(explanations):
        border = "#4F8BF9" if i < 2 else (color if i >= 4 else "#7C5CBF")
        st.markdown(
            f"<div class='info-card' style='border-left:4px solid {border};margin-bottom:0.6rem'>"
            f"<span style='font-size:0.87rem;color:#9ca3af'>{exp}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Section 6: Scenario Comparison ──────────────────────
    st.markdown("### 🔄 Feature Importance Across Scenarios")
    st.caption(
        "This comparison shows how the model's attention profile changes across different "
        "physiological states. Critical scenarios emphasise stress, HRV, and sleep signals."
    )
    st.plotly_chart(_comparison_fig(feature_names), use_container_width=True)

    st.divider()

    # ── Section 7: Model Decision Card ──────────────────────
    st.markdown("### 📋 Model Decision Summary")

    # Top 3 important features
    sorted_imp  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_text   = ", ".join(
        [f"**{f.replace('_', ' ').title()}** ({v:.3f})" for f, v in sorted_imp]
    )
    day_labels  = [f"Day -{6-i}" if i < 6 else "Today" for i in range(7)]
    peak_d_lbl  = day_labels[int(np.argmax(tw))]

    st.markdown(
        f"<div class='info-card orange-border' style='padding:1.5rem'>"
        f"<strong style='color:#f39c12;font-size:1rem'>Decision Summary — {category.upper()} Risk</strong>"
        f"<hr style='border-color:#2d3748;margin:0.75rem 0'>"
        f"<table style='width:100%;font-size:0.85rem'>"
        f"<tr><td style='color:#6b7280;padding:4px 0'>Risk Score</td>"
        f"<td style='color:{color};font-weight:700'>{score*100:.1f}% — {category.upper()}</td></tr>"
        f"<tr><td style='color:#6b7280;padding:4px 0'>Peak Day</td>"
        f"<td style='color:#e8eaf0'>{peak_d_lbl} (attention: {float(np.max(tw)):.4f})</td></tr>"
        f"<tr><td style='color:#6b7280;padding:4px 0'>Top Features</td>"
        f"<td style='color:#e8eaf0'>{', '.join([f.replace('_', ' ').title() for f, _ in sorted_imp])}</td></tr>"
        f"<tr><td style='color:#6b7280;padding:4px 0'>Model</td>"
        f"<td style='color:#e8eaf0'>BiLSTM + Attention · 7 days × 24 features</td></tr>"
        f"<tr><td style='color:#6b7280;padding:4px 0'>Validation</td>"
        f"<td style='color:#e8eaf0'>LOSO (Leave-One-Subject-Out)</td></tr>"
        f"</table>"
        f"<hr style='border-color:#2d3748;margin:0.75rem 0'>"
        f"<span style='font-size:0.78rem;color:#6b7280'>"
        "⚠ Explainability values shown are attention-based estimates for research purposes. "
        "They are NOT clinically validated causal explanations."
        "</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footer'>🫀 PanicGuard AI · Research Prototype · Not for clinical diagnosis</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
