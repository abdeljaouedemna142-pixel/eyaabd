"""
pages/4_📊_Patient_History.py
Patient history — 30-day risk trend, alert log, and physiological evolution.
"""

import io
import os
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="Patient History — PanicGuard AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model_utils import get_risk_category, predict_demo, FEATURES_24
from preprocessing_utils import generate_synthetic_patient, preprocess_pipeline
from visualization_utils import (
    plot_risk_history,
    plot_physiological_trends,
    plot_timeline_events,
    plot_signal_correlation,
)
from utils.helpers import (
    generate_patient_id,
    get_risk_color,
    get_risk_label,
    get_risk_emoji,
    calculate_trend,
)


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


@st.cache_data(show_spinner=False)
def _build_demo_history(scenario: str, n_days: int = 30) -> pd.DataFrame:
    """Build a 30-day risk score history for the selected scenario."""
    rng   = np.random.default_rng(seed=99 if scenario == "normal" else (88 if scenario == "elevated" else 77))
    dates = [datetime.today().date() - timedelta(days=(n_days - 1 - i)) for i in range(n_days)]
    t     = np.linspace(0, 1, n_days)

    if scenario == "normal":
        base  = rng.uniform(0.05, 0.25, n_days)
        trend = -0.02 * t
    elif scenario == "elevated":
        base  = rng.uniform(0.28, 0.55, n_days)
        trend =  0.08 * t
    else:
        base  = rng.uniform(0.55, 0.85, n_days)
        trend =  0.10 * t

    scores = np.clip(base + trend + rng.normal(0, 0.02, n_days), 0, 1)

    threshold = 0.60
    df = pd.DataFrame({
        "date":       dates,
        "risk_score": np.round(scores, 4),
        "alert":      scores >= threshold,
    })
    return df


def _build_event_df(history_df: pd.DataFrame, alerts: list) -> pd.DataFrame:
    """Combine history risk levels with logged alerts into event timeline."""
    events = []
    for _, row in history_df.iterrows():
        cat, _, _ = get_risk_category(float(row["risk_score"]))
        events.append({
            "date":        pd.Timestamp(row["date"]),
            "event_type":  cat,
            "risk_score":  float(row["risk_score"]),
            "description": f"Daily assessment: {float(row['risk_score'])*100:.1f}%",
        })
    for a in alerts:
        try:
            dt = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        events.append({
            "date":        pd.Timestamp(dt),
            "event_type":  "alert",
            "risk_score":  a.get("risk_score", 0.0),
            "description": f"Alert logged — patient {a.get('patient_id', '?')}",
        })
    return pd.DataFrame(events)


# ── Main ─────────────────────────────────────────────────────

def main():
    _load_css()
    config   = _load_config()
    _init_session(config)

    scenario    = st.session_state.selected_scenario
    patient_id  = st.session_state.patient_id
    alerts      = st.session_state.alerts
    features    = config.get("features", FEATURES_24)

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 16px'>"
            "<span style='font-size:2.5rem'>📊</span><br>"
            "<span style='font-weight:700;color:#e8eaf0;font-size:1.1rem'>Patient History</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        score = st.session_state.current_risk_score
        if score is not None:
            col = get_risk_color(score)
            st.markdown(
                f"<div style='text-align:center;padding:12px;border-radius:8px;border:1px solid {col}33'>"
                f"<div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Current Risk</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{col}'>{score*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:{col}'>{get_risk_emoji(score)} {get_risk_label(score)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No prediction yet.\nGo to Patient Monitoring first.")

        st.divider()

        st.markdown("**Display Scenario**")
        view_scenario = st.selectbox(
            "History scenario",
            ["normal", "elevated", "critical"],
            index=["normal", "elevated", "critical"].index(scenario),
            format_func=lambda s: {"normal": "🟢 Normal", "elevated": "🟡 Elevated", "critical": "🔴 Critical"}[s],
            label_visibility="collapsed",
        )

        st.divider()
        st.metric("Logged Alerts", len(alerts))
        st.metric("Prediction Runs", len(st.session_state.prediction_history))

    # ── Page Header ─────────────────────────────────────────
    st.markdown("## 📊 Patient History")
    st.markdown(
        f"<span style='color:#6b7280;font-size:0.9rem'>"
        f"Patient: <strong style='color:#4F8BF9'>{patient_id}</strong> &nbsp;·&nbsp; "
        f"Monitoring scenario: <strong style='color:#9ca3af'>{view_scenario.upper()}</strong>"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generate / Load History ──────────────────────────────
    history_df = _build_demo_history(view_scenario, 30)

    # Merge with any real prediction history from session
    if st.session_state.prediction_history:
        real_rows = []
        for p in st.session_state.prediction_history:
            try:
                dt = datetime.strptime(p["date"], "%Y-%m-%d %H:%M").date()
            except Exception:
                dt = datetime.today().date()
            real_rows.append({
                "date":       dt,
                "risk_score": p["score"],
                "alert":      p["score"] >= 0.60,
            })
        real_df   = pd.DataFrame(real_rows)
        history_df = pd.concat([history_df, real_df], ignore_index=True)
        history_df = history_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ── Summary Metrics ──────────────────────────────────────
    scores_arr    = history_df["risk_score"].values
    mean_risk     = float(np.mean(scores_arr))
    max_risk      = float(np.max(scores_arr))
    min_risk      = float(np.min(scores_arr))
    n_critical    = int((scores_arr >= 0.60).sum())
    n_elevated    = int(((scores_arr >= 0.30) & (scores_arr < 0.60)).sum())
    n_normal      = int((scores_arr < 0.30).sum())
    trend_label   = calculate_trend(scores_arr.tolist())
    trend_emoji   = {"improving": "📉", "worsening": "📈", "stable": "➡️"}.get(trend_label, "➡️")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        mc = get_risk_color(mean_risk)
        st.metric("Average Risk",   f"{mean_risk*100:.1f}%",
                  delta=f"{get_risk_label(mean_risk)}", delta_color="off")
    with m2:
        st.metric("Peak Risk",      f"{max_risk*100:.1f}%",
                  delta=f"{get_risk_label(max_risk)}", delta_color="inverse" if max_risk >= 0.60 else "off")
    with m3:
        st.metric("Critical Days",  n_critical,
                  delta=f"of {len(scores_arr)} days", delta_color="inverse" if n_critical > 0 else "off")
    with m4:
        st.metric("Trend",          f"{trend_emoji} {trend_label.title()}",
                  delta=None)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Risk Score Timeline ──────────────────────────────────
    st.markdown("### 📈 30-Day Risk Score Timeline")
    st.plotly_chart(plot_risk_history(history_df), use_container_width=True)

    st.divider()

    # ── Risk Distribution ────────────────────────────────────
    dist_col1, dist_col2 = st.columns([2, 1], gap="large")

    with dist_col1:
        st.markdown("### 📊 Risk Level Distribution")
        import plotly.graph_objects as go
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Elevated", "Critical"],
            values=[n_normal, n_elevated, n_critical],
            hole=0.55,
            marker=dict(
                colors=["#2ecc71", "#f39c12", "#e74c3c"],
                line=dict(color="#0d1117", width=2),
            ),
            textfont=dict(size=13, color="#e8eaf0"),
            hovertemplate="<b>%{label}</b><br>Days: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            height=320,
            paper_bgcolor="#0d1117",
            font=dict(color="#e8eaf0"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#2d3748", borderwidth=1),
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f"<b>{len(scores_arr)}</b><br>days",
                x=0.5, y=0.5, font_size=16, font_color="#e8eaf0",
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with dist_col2:
        st.markdown("### 📋 Period Summary")
        date_start = history_df["date"].iloc[0]
        date_end   = history_df["date"].iloc[-1]

        rows = [
            ("Period",          f"{date_start} → {date_end}"),
            ("Days Monitored",  str(len(history_df))),
            ("Normal Days",     f"{n_normal} ({n_normal/len(scores_arr)*100:.0f}%)"),
            ("Elevated Days",   f"{n_elevated} ({n_elevated/len(scores_arr)*100:.0f}%)"),
            ("Critical Days",   f"{n_critical} ({n_critical/len(scores_arr)*100:.0f}%)"),
            ("Mean Risk",       f"{mean_risk*100:.1f}%"),
            ("Max Risk",        f"{max_risk*100:.1f}%"),
            ("Min Risk",        f"{min_risk*100:.1f}%"),
            ("Logged Alerts",   str(len(alerts))),
            ("Trend",           f"{trend_emoji} {trend_label.title()}"),
        ]

        for label_r, val_r in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #2d3748'>"
                f"<span style='color:#6b7280;font-size:0.82rem'>{label_r}</span>"
                f"<span style='color:#e8eaf0;font-size:0.82rem;font-weight:600'>{val_r}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Physiological Evolution ──────────────────────────────
    st.markdown("### 🩺 30-Day Physiological Evolution")

    if st.session_state.patient_data is not None:
        df_phys = st.session_state.patient_data.copy()
        st.caption("Showing data from the loaded patient record.")
    else:
        df_phys = generate_synthetic_patient(view_scenario)
        st.caption(f"Showing synthetic {view_scenario} patient data for demonstration.")

    evo_signals = ["heart_rate", "stress_score", "sleep_duration", "hrv", "anxiety_proxy"]
    avail_evo   = [s for s in evo_signals if s in df_phys.columns]
    st.plotly_chart(plot_physiological_trends(df_phys, avail_evo), use_container_width=True)

    st.divider()

    # ── Signal Correlation ───────────────────────────────────
    st.markdown("### 🔗 Physiological Signal Correlations")
    st.caption("Pearson correlation matrix of key physiological signals over the monitoring period.")

    corr_signals = ["heart_rate", "stress_score", "sleep_duration", "hrv",
                    "anxiety_proxy", "fatigue_level", "activity_level", "mood_score"]
    avail_corr = [s for s in corr_signals if s in df_phys.columns]
    if len(avail_corr) >= 2:
        st.plotly_chart(plot_signal_correlation(df_phys, avail_corr), use_container_width=True)

    st.divider()

    # ── Alert History Table ──────────────────────────────────
    st.markdown("### 🚨 Logged Alert History")

    if alerts:
        alert_rows = []
        for a in reversed(alerts):
            nc_sent = a.get("notifications_sent", {})
            ok_channels = [ch for ch, info in nc_sent.items() if info.get("success")]
            alert_rows.append({
                "Timestamp":   a["timestamp"],
                "Patient ID":  a["patient_id"],
                "Risk Score":  a["risk_pct"],
                "Level":       a["alert_level"].upper(),
                "Sent To":     ", ".join(ok_channels) or "—",
            })
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            "<div style='text-align:center;padding:1.5rem;background:#1e2130;"
            "border-radius:12px;border:1px dashed #2d3748;color:#6b7280'>"
            "No alerts logged. Alerts are recorded automatically when risk exceeds the critical threshold.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Event Timeline ───────────────────────────────────────
    st.markdown("### 📅 Clinical Event Timeline")
    event_df = _build_event_df(history_df, alerts)
    st.plotly_chart(plot_timeline_events(event_df), use_container_width=True)

    st.divider()

    # ── Recovery Analysis ────────────────────────────────────
    st.markdown("### 🔄 Recovery & Deterioration Analysis")

    window_size = 7
    if len(scores_arr) >= window_size:
        rolling_mean = pd.Series(scores_arr).rolling(window_size, min_periods=1).mean().values

        import plotly.graph_objects as go
        fig_rec = go.Figure()
        fig_rec.add_hrect(y0=0, y1=0.30, fillcolor="rgba(46,204,113,0.05)", line_width=0,
                          annotation_text="Normal", annotation_position="right",
                          annotation_font=dict(size=9, color="#6b7280"))
        fig_rec.add_hrect(y0=0.30, y1=0.60, fillcolor="rgba(243,156,18,0.05)", line_width=0,
                          annotation_text="Elevated", annotation_position="right",
                          annotation_font=dict(size=9, color="#6b7280"))
        fig_rec.add_hrect(y0=0.60, y1=1.0, fillcolor="rgba(231,76,60,0.05)", line_width=0,
                          annotation_text="Critical", annotation_position="right",
                          annotation_font=dict(size=9, color="#6b7280"))
        x_vals = history_df["date"].tolist()
        fig_rec.add_trace(go.Scatter(
            x=x_vals, y=(scores_arr * 100).tolist(),
            mode="lines+markers", name="Daily Risk",
            line=dict(color="#4F8BF9", width=1.5, dash="dot"),
            marker=dict(size=4, color="#4F8BF9"),
        ))
        fig_rec.add_trace(go.Scatter(
            x=x_vals, y=(rolling_mean * 100).tolist(),
            mode="lines", name=f"{window_size}-Day Trend",
            line=dict(color="#e74c3c" if view_scenario == "critical" else "#f39c12", width=3),
            fill="tozeroy", fillcolor="rgba(79,139,249,0.07)",
        ))
        fig_rec.add_hline(y=60, line_dash="dash", line_color="#e74c3c", opacity=0.6,
                          annotation_text="Critical Threshold", annotation_font_color="#e74c3c")
        fig_rec.update_layout(
            title=f"Risk Trajectory with {window_size}-Day Rolling Average",
            height=360,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#1e2130",
            font=dict(color="#e8eaf0"),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(title="Risk Score (%)", gridcolor="#2d3748", range=[0, 105]),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#2d3748", borderwidth=1),
            margin=dict(l=55, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    st.divider()

    # ── Export ───────────────────────────────────────────────
    st.markdown("### 💾 Export History")
    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        csv_hist = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Risk History CSV",
            data=csv_hist,
            file_name=f"panicguard_risk_history_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with ex_col2:
        if st.session_state.patient_data is not None:
            csv_phys = st.session_state.patient_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Patient Data CSV",
                data=csv_phys,
                file_name=f"panicguard_patient_data_{patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇ Download Patient Data CSV", disabled=True, use_container_width=True)

    st.markdown(
        "<div class='footer'>🫀 PanicGuard AI · Research Prototype · Not for clinical diagnosis</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
