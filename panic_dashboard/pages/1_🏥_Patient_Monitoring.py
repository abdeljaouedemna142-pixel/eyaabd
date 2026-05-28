"""
pages/1_🏥_Patient_Monitoring.py
Main patient monitoring interface for PanicGuard AI.
"""

import io
import os
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Path setup (makes parent-dir imports work)
# ─────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="Patient Monitoring — PanicGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model_utils import (
    load_model_cached,
    predict_risk,
    predict_demo,
    get_risk_category,
    get_attention_weights_demo,
    compute_feature_importance_demo,
)
from preprocessing_utils import preprocess_pipeline, generate_synthetic_patient
from visualization_utils import (
    plot_risk_gauge,
    plot_physiological_trends,
    create_radar_chart,
)
from notification_utils import log_alert
from utils.helpers import (
    generate_patient_id,
    get_risk_color,
    get_risk_label,
    get_risk_emoji,
    timestamp_now,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

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
        "patient_data":          None,
        "alerts":                [],
        "current_risk_score":    None,
        "prediction_history":    [],
        "patient_id":            generate_patient_id(),
        "selected_scenario":     "normal",
        "notification_config":   config.get("notifications", {}),
        "model_loaded":          False,
        "demo_mode":             True,
        "preprocessing_result":  None,
        "attention_weights":     None,
        "feature_importance":    None,
        "last_prediction_time":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _run_prediction(windows: np.ndarray, scenario: str, config: dict):
    model_path = os.path.join(_ROOT, config["model"]["path"])
    model = load_model_cached(model_path)
    if model is not None:
        probs = predict_risk(model, windows)
    else:
        probs = predict_demo(windows, scenario)
    return float(probs[-1]), probs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    _load_css()
    config    = _load_config()
    _init_session(config)

    features      = config["features"]
    normal_ranges = config.get("normal_ranges", {})
    threshold     = config["notifications"].get("alert_threshold", 0.60)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 16px'>"
            "<span style='font-size:2.5rem'>🏥</span><br>"
            "<span style='font-weight:700;color:#e8eaf0;font-size:1.1rem'>Patient Monitoring</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Patient ID**")
        pid_val = st.text_input(
            "Patient ID", value=st.session_state.patient_id,
            label_visibility="collapsed",
        )
        if pid_val != st.session_state.patient_id:
            st.session_state.patient_id = pid_val

        if st.button("🔄 Generate New ID", use_container_width=True):
            st.session_state.patient_id = generate_patient_id()
            st.rerun()

        st.divider()
        st.markdown("**Scenario (Demo Mode)**")
        scenario = st.selectbox(
            "Scenario", ["normal", "elevated", "critical"],
            index=["normal", "elevated", "critical"].index(
                st.session_state.selected_scenario
            ),
            format_func=lambda s: {
                "normal":   "🟢 Normal",
                "elevated": "🟡 Elevated Risk",
                "critical": "🔴 Critical Risk",
            }[s],
            label_visibility="collapsed",
        )
        st.session_state.selected_scenario = scenario

        st.divider()
        model_exists = os.path.isfile(os.path.join(_ROOT, config["model"]["path"]))
        if model_exists:
            st.success("✅ AI Model: Loaded")
        else:
            st.warning("⚠️ Demo Mode Active")
            st.caption("Add best_model_B.keras to enable real predictions.")

        if st.session_state.last_prediction_time:
            st.caption(f"Last run: {st.session_state.last_prediction_time}")

        if st.session_state.current_risk_score is not None:
            sc = st.session_state.current_risk_score
            col = get_risk_color(sc)
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:rgba(79,139,249,0.08);"
                f"border-radius:8px;border:1px solid {col}33;margin-top:8px'>"
                f"<div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Current Risk</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{col}'>{sc*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:{col}'>{get_risk_emoji(sc)} {get_risk_label(sc)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Page Header ──────────────────────────────────────────
    st.markdown("## 🏥 Patient Monitoring")
    st.markdown(
        f"<span style='color:#6b7280;font-size:0.9rem'>"
        f"Patient: <strong style='color:#4F8BF9'>{st.session_state.patient_id}</strong> &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%B %d, %Y — %H:%M')}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data Input ────────────────────────────────────────────
    tab_up, tab_demo = st.tabs(["📁  Upload CSV", "🔬  Generate Demo Patient"])

    with tab_up:
        st.markdown(
            "Upload a patient CSV file with a `date` column and the 24 physiological features. "
            "Minimum 7 rows required."
        )
        uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                with st.spinner("Processing CSV…"):
                    result = preprocess_pipeline(uploaded)
                st.session_state.preprocessing_result = result
                st.session_state.patient_data = result["df_raw"]
                meta = result["metadata"]
                st.success(
                    f"✅ Loaded {meta['n_days']} days · {meta['n_windows']} prediction windows · "
                    f"Date range: {meta['date_range']}"
                )
                if meta["features_imputed"]:
                    imp_list = ", ".join(meta["features_imputed"][:6])
                    st.warning(f"Missing features imputed with column median: {imp_list}"
                               + ("…" if len(meta["features_imputed"]) > 6 else ""))
            except Exception as exc:
                st.error(f"Could not process CSV: {exc}")

    with tab_demo:
        st.markdown(
            "Generate a synthetic 30-day wearable record. "
            "Each scenario simulates a realistic physiological trajectory."
        )
        c1, c2, c3 = st.columns(3)

        def _gen_btn(col, label, sc, border_color, desc):
            with col:
                st.markdown(
                    f"<div class='info-card' style='border-left:4px solid {border_color};"
                    f"text-align:center;padding:1rem;min-height:110px'>"
                    f"<strong style='color:{border_color};font-size:1rem'>{label}</strong><br>"
                    f"<span style='font-size:0.78rem;color:#6b7280;line-height:1.4'>{desc}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                return st.button(f"Generate {label.split()[1]}", use_container_width=True, key=f"gen_{sc}")

        triggered_scenario = None
        if _gen_btn(c1, "🟢 Normal",   "normal",   "#2ecc71", "Stable vitals, good sleep, low stress, healthy HRV"):
            triggered_scenario = "normal"
        if _gen_btn(c2, "🟡 Elevated", "elevated", "#f39c12", "Increasing stress, declining sleep, rising HR trend"):
            triggered_scenario = "elevated"
        if _gen_btn(c3, "🔴 Critical", "critical", "#e74c3c", "Severe anxiety, poor sleep, high HR, very low HRV"):
            triggered_scenario = "critical"

        if triggered_scenario:
            st.session_state.selected_scenario = triggered_scenario
            with st.spinner(f"Generating {triggered_scenario.upper()} patient…"):
                df_syn = generate_synthetic_patient(triggered_scenario)
                st.session_state.patient_data = df_syn
                buf = io.StringIO()
                df_syn.to_csv(buf, index=False)
                buf.seek(0)
                result = preprocess_pipeline(buf)
                st.session_state.preprocessing_result = result
            st.success(f"✅ Synthetic {triggered_scenario.upper()} patient generated — 30 days, 24 features.")
            st.rerun()

    st.divider()

    # Raw data preview
    if st.session_state.patient_data is not None:
        with st.expander("📋 Patient Data Preview (last 10 rows)", expanded=False):
            st.dataframe(
                st.session_state.patient_data.tail(10),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ── Prediction ────────────────────────────────────────────
    st.markdown("### 🧠 BiLSTM + Attention Risk Prediction")

    no_data = st.session_state.preprocessing_result is None
    if no_data:
        st.info("↑ Upload a CSV file or generate a demo patient to enable prediction.")

    if st.button(
        "▶  Run Panic Risk Prediction",
        disabled=no_data,
        type="primary",
    ):
        with st.spinner("Running BiLSTM + Attention inference…"):
            try:
                windows = st.session_state.preprocessing_result["windows"]
                score, _ = _run_prediction(windows, st.session_state.selected_scenario, config)

                st.session_state.current_risk_score  = score
                st.session_state.last_prediction_time = timestamp_now()
                st.session_state.attention_weights   = get_attention_weights_demo(
                    scenario=st.session_state.selected_scenario
                )
                st.session_state.feature_importance  = compute_feature_importance_demo(
                    scenario=st.session_state.selected_scenario
                )
                st.session_state.prediction_history.append({
                    "date":       datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "score":      score,
                    "level":      get_risk_category(score)[0],
                    "patient_id": st.session_state.patient_id,
                })

                if score >= threshold:
                    log_alert(
                        patient_id=st.session_state.patient_id,
                        risk_score=score,
                        alert_level=get_risk_category(score)[0],
                        notifications_sent={},
                    )
                st.rerun()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    # ── Results ──────────────────────────────────────────────
    if st.session_state.current_risk_score is not None:
        score    = st.session_state.current_risk_score
        category, color, emoji = get_risk_category(score)
        label    = get_risk_label(score)

        st.markdown("<br>", unsafe_allow_html=True)

        # Alert banner
        if score >= threshold:
            st.markdown(
                f"<div class='alert-banner critical'>"
                f"<span class='alert-icon'>🚨</span>"
                f"<div class='alert-text'>"
                f"<div class='alert-title'>CRITICAL RISK ALERT — {st.session_state.patient_id}</div>"
                f"<div class='alert-message'>Risk score <strong>{score*100:.1f}%</strong> exceeds the "
                f"{threshold*100:.0f}% critical threshold. Navigate to <strong>Alert Center</strong> "
                f"to send email / Telegram notifications.</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        elif score >= 0.30:
            st.markdown(
                f"<div class='alert-banner elevated'>"
                f"<span class='alert-icon'>⚠️</span>"
                f"<div class='alert-text'>"
                f"<div class='alert-title'>ELEVATED RISK — {st.session_state.patient_id}</div>"
                f"<div class='alert-message'>Risk score <strong>{score*100:.1f}%</strong>. "
                f"Enhanced monitoring recommended over next 24–48 h.</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='alert-banner normal'>"
                f"<span class='alert-icon'>✅</span>"
                f"<div class='alert-text'>"
                f"<div class='alert-title'>NORMAL — {st.session_state.patient_id}</div>"
                f"<div class='alert-message'>Risk score <strong>{score*100:.1f}%</strong>. "
                f"No significant physiological anomaly detected.</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + summary
        g_col, s_col = st.columns([1, 1], gap="large")

        with g_col:
            st.plotly_chart(plot_risk_gauge(score), use_container_width=True)

        with s_col:
            st.markdown(
                f"<div class='score-display'>"
                f"<div class='score-number {category}'>{score*100:.1f}%</div>"
                f"<div class='score-label' style='color:{color}'>{emoji} {label}</div>"
                f"<div style='margin-top:8px;color:#6b7280;font-size:0.78rem'>"
                f"BiLSTM + Attention · 7-day rolling window · 24 biosignals</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.session_state.patient_data is not None:
                df  = st.session_state.patient_data
                row = df.iloc[-1]
                qa, qb = st.columns(2)
                key_signals = [
                    ("❤️ Heart Rate",   "heart_rate",    "bpm"),
                    ("😓 Stress",        "stress_score",  ""),
                    ("💤 Sleep",         "sleep_duration","h"),
                    ("⚡ HRV",           "hrv",           "ms"),
                ]
                for i, (lbl, feat, unit) in enumerate(key_signals):
                    val = row.get(feat, float("nan"))
                    nr  = normal_ranges.get(feat, [None, None])
                    ok  = (not pd.isna(val) and nr[0] is not None and nr[0] <= val <= nr[1])
                    with (qa if i % 2 == 0 else qb):
                        st.metric(
                            label=lbl,
                            value=f"{val:.1f} {unit}".strip() if not pd.isna(val) else "N/A",
                            delta="✓ Normal" if ok else "⚠ Abnormal",
                            delta_color="normal" if ok else "inverse",
                        )

        st.divider()

        # ── Physiological Trends ──────────────────────────────
        st.markdown("### 📈 Physiological Signal Trends")

        if st.session_state.patient_data is not None:
            df_plot = st.session_state.patient_data.copy()
            all_sig = ["heart_rate", "stress_score", "sleep_duration",
                       "activity_level", "hrv", "anxiety_proxy",
                       "fatigue_level", "breathing_rate", "galvanic_skin_response"]
            avail   = [s for s in all_sig if s in df_plot.columns]

            selected_sig = st.multiselect(
                "Select signals",
                options=avail,
                default=avail[:5],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            if selected_sig:
                st.plotly_chart(
                    plot_physiological_trends(df_plot, selected_sig),
                    use_container_width=True,
                )

        st.divider()

        # ── Feature Summary ───────────────────────────────────
        st.markdown("### 📊 Feature Summary vs. Normal Ranges")

        if st.session_state.patient_data is not None:
            df  = st.session_state.patient_data
            row = df.iloc[-1]
            rows = []
            for feat in features:
                val = row.get(feat, float("nan"))
                nr  = normal_ranges.get(feat, [None, None])
                if not pd.isna(val) and nr[0] is not None:
                    if val < nr[0]:
                        status = "⬇ Below Normal"
                    elif val > nr[1]:
                        status = "⬆ Above Normal"
                    else:
                        status = "✓ Normal"
                else:
                    status = "— N/A"
                rows.append({
                    "Feature":       feat.replace("_", " ").title(),
                    "Current":       f"{val:.2f}" if not pd.isna(val) else "N/A",
                    "Normal Range":  f"{nr[0]} – {nr[1]}" if nr[0] is not None else "—",
                    "Status":        status,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Radar Chart ───────────────────────────────────────
        st.markdown("### 🕸 Multi-Signal Radar Overview")

        if st.session_state.patient_data is not None:
            df  = st.session_state.patient_data
            row = df.iloc[-1]
            radar_feat  = ["heart_rate","stress_score","sleep_quality",
                           "activity_level","hrv","mood_score","anxiety_proxy","fatigue_level"]
            radar_ranges = {
                "heart_rate":    (50, 120),  "stress_score":  (0, 100),
                "sleep_quality": (0, 100),   "activity_level":(0, 100),
                "hrv":           (0, 100),   "mood_score":    (0, 100),
                "anxiety_proxy": (0, 100),   "fatigue_level": (0, 100),
            }
            radar_dict = {}
            for feat in radar_feat:
                if feat in df.columns:
                    val       = float(row.get(feat, 50))
                    lo, hi    = radar_ranges.get(feat, (0, 100))
                    radar_dict[feat] = float(np.clip((val - lo) / max(hi - lo, 1) * 100, 0, 100))
            if radar_dict:
                st.plotly_chart(
                    create_radar_chart(radar_dict, "Current Day — Normalised Signal Profile"),
                    use_container_width=True,
                )

    else:
        st.markdown(
            "<div style='text-align:center;padding:3rem;background:#1e2130;"
            "border-radius:16px;border:2px dashed #2d3748;color:#6b7280'>"
            "<span style='font-size:3rem'>🔍</span><br><br>"
            "<strong style='color:#9ca3af;font-size:1.1rem'>No prediction yet</strong><br><br>"
            "<span style='font-size:0.85rem'>Upload patient data or generate a demo patient, "
            "then click <em>Run Panic Risk Prediction</em>.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='footer'>🫀 PanicGuard AI · Research Prototype · Not for clinical diagnosis</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
