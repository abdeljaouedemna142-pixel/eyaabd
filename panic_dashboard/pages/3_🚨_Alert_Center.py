"""
pages/3_🚨_Alert_Center.py
Alert center — notification configuration and dispatch for PanicGuard AI.
"""

import os
import json
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="Alert Center — PanicGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model_utils import get_risk_category
from notification_utils import (
    send_email_alert,
    send_telegram_alert,
    log_alert,
    format_telegram_message,
    format_alert_email_html,
)
from visualization_utils import plot_risk_history, plot_timeline_events
from utils.helpers import (
    generate_patient_id,
    get_risk_color,
    get_risk_label,
    get_risk_emoji,
    timestamp_now,
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


def _demo_send(channel: str, patient_id: str, score: float, level: str) -> tuple:
    """Simulate a notification send in demo mode."""
    ts = timestamp_now()
    if channel == "email":
        msg = (
            f"[DEMO] Email alert would be sent to configured recipient.\n"
            f"Subject: [PanicGuard AI] {level.upper()} Risk Alert — Patient {patient_id} — {score*100:.1f}%\n"
            f"Timestamp: {ts}"
        )
    elif channel == "telegram":
        msg = format_telegram_message(patient_id, score, level, ts)
        msg = f"[DEMO] Telegram message preview:\n{msg}"
    else:
        msg = f"[DEMO] Notification sent via {channel}."
    return True, msg


def _features_summary(patient_data: pd.DataFrame) -> dict:
    """Extract last row key values as a summary dict."""
    if patient_data is None or patient_data.empty:
        return {}
    row = patient_data.iloc[-1]
    keys = ["heart_rate", "stress_score", "sleep_duration", "hrv",
            "anxiety_proxy", "fatigue_level", "spo2", "breathing_rate"]
    summary = {}
    for k in keys:
        if k in row.index:
            val = row[k]
            if not pd.isna(val):
                unit_map = {
                    "heart_rate": "bpm", "stress_score": "",
                    "sleep_duration": "h", "hrv": "ms",
                    "anxiety_proxy": "", "fatigue_level": "",
                    "spo2": "%", "breathing_rate": "br/min",
                }
                summary[k] = f"{val:.1f} {unit_map.get(k, '')}".strip()
    return summary


# ── Main ─────────────────────────────────────────────────────

def main():
    _load_css()
    config = _load_config()
    _init_session(config)

    threshold = config["notifications"].get("alert_threshold", 0.60)
    score     = st.session_state.current_risk_score
    patient_id = st.session_state.patient_id

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 16px'>"
            "<span style='font-size:2.5rem'>🚨</span><br>"
            "<span style='font-weight:700;color:#e8eaf0;font-size:1.1rem'>Alert Center</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        if score is not None:
            col = get_risk_color(score)
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:rgba(79,139,249,0.08);"
                f"border-radius:8px;border:1px solid {col}33'>"
                f"<div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px'>Current Risk</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:{col}'>{score*100:.1f}%</div>"
                f"<div style='font-size:0.8rem;color:{col}'>{get_risk_emoji(score)} {get_risk_label(score)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No prediction. Go to Patient Monitoring first.")

        st.divider()
        total_alerts = len(st.session_state.alerts)
        critical_alerts = sum(1 for a in st.session_state.alerts if a.get("alert_level") == "critical")
        st.metric("Total Alerts Logged", total_alerts)
        st.metric("Critical Alerts", critical_alerts)

        if st.button("🗑 Clear Alert Log", use_container_width=True):
            st.session_state.alerts = []
            st.rerun()

    # ── Page Header ─────────────────────────────────────────
    st.markdown("## 🚨 Alert Center")
    st.markdown(
        f"<span style='color:#6b7280;font-size:0.9rem'>"
        f"Patient: <strong style='color:#4F8BF9'>{patient_id}</strong> &nbsp;·&nbsp; "
        f"{datetime.now().strftime('%B %d, %Y — %H:%M')}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active Alert Banner ──────────────────────────────────
    if score is None:
        st.markdown(
            "<div class='alert-banner normal'>"
            "<span class='alert-icon'>ℹ️</span>"
            "<div class='alert-text'>"
            "<div class='alert-title'>No Active Prediction</div>"
            "<div class='alert-message'>Run a prediction in Patient Monitoring to activate the alert system.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    elif score >= threshold:
        category, color, emoji = get_risk_category(score)
        st.markdown(
            f"<div class='alert-banner critical'>"
            f"<span class='alert-icon'>🚨</span>"
            f"<div class='alert-text'>"
            f"<div class='alert-title'>🔴 CRITICAL RISK — ALERT TRIGGERED</div>"
            f"<div class='alert-message'>"
            f"Patient <strong>{patient_id}</strong> · Risk score <strong>{score*100:.1f}%</strong> "
            f"exceeds the {threshold*100:.0f}% critical threshold. "
            f"Send notifications to the clinical team immediately.</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    elif score >= 0.30:
        st.markdown(
            f"<div class='alert-banner elevated'>"
            f"<span class='alert-icon'>⚠️</span>"
            f"<div class='alert-text'>"
            f"<div class='alert-title'>🟡 ELEVATED RISK — Monitoring Recommended</div>"
            f"<div class='alert-message'>"
            f"Risk score <strong>{score*100:.1f}%</strong>. Below critical threshold ({threshold*100:.0f}%). "
            f"Consider sending a precautionary notification.</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='alert-banner normal'>"
            f"<span class='alert-icon'>✅</span>"
            f"<div class='alert-text'>"
            f"<div class='alert-title'>🟢 NO ALERT — Normal Readings</div>"
            f"<div class='alert-message'>Risk score <strong>{score*100:.1f}%</strong>. "
            f"No alert conditions triggered.</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Notification Configuration ───────────────────────────
    st.markdown("### ⚙️ Notification Configuration")

    nc = st.session_state.notification_config
    smtp_cfg  = nc.get("smtp", {})
    tg_cfg    = nc.get("telegram", {})

    cfg_col1, cfg_col2 = st.columns(2, gap="large")

    with cfg_col1:
        with st.expander("📧 Email (SMTP) Configuration", expanded=True):
            smtp_server   = st.text_input("SMTP Server",    value=smtp_cfg.get("server", "smtp.gmail.com"))
            smtp_port     = st.number_input("SMTP Port",    value=int(smtp_cfg.get("port", 587)), min_value=1, max_value=65535)
            sender_email  = st.text_input("Sender Email",   value=smtp_cfg.get("sender_email", ""), placeholder="your.app@gmail.com")
            sender_pass   = st.text_input("App Password",   value=smtp_cfg.get("sender_password", ""), type="password",
                                          help="Use a Gmail App Password, not your account password.")
            doctor_email  = st.text_input("Doctor Email",   value=nc.get("doctor_email", ""), placeholder="doctor@clinic.com")
            family_email  = st.text_input("Family Email",   value=nc.get("family_email", ""), placeholder="family@example.com")

            if st.button("💾 Save Email Settings", use_container_width=True):
                st.session_state.notification_config.update({
                    "smtp": {
                        "server":          smtp_server,
                        "port":            smtp_port,
                        "sender_email":    sender_email,
                        "sender_password": sender_pass,
                        "use_tls":         True,
                    },
                    "doctor_email": doctor_email,
                    "family_email": family_email,
                })
                st.success("Email settings saved.")

    with cfg_col2:
        with st.expander("💬 Telegram Bot Configuration", expanded=True):
            bot_token = st.text_input(
                "Bot Token",
                value=tg_cfg.get("bot_token", ""),
                type="password",
                placeholder="123456789:ABCdef...",
                help="Create a bot with @BotFather on Telegram to get your token.",
            )
            chat_id   = st.text_input(
                "Chat ID",
                value=tg_cfg.get("chat_id", ""),
                placeholder="-100123456789",
                help="Your chat or channel ID. Use @userinfobot to find it.",
            )

            if st.button("💾 Save Telegram Settings", use_container_width=True):
                st.session_state.notification_config.update({
                    "telegram": {"bot_token": bot_token, "chat_id": chat_id}
                })
                st.success("Telegram settings saved.")

            st.markdown(
                "<div class='info-card blue-border' style='margin-top:1rem;font-size:0.8rem;color:#9ca3af'>"
                "<strong style='color:#4F8BF9'>How to set up a Telegram bot:</strong><br>"
                "1. Open Telegram → search @BotFather<br>"
                "2. Send /newbot and follow instructions<br>"
                "3. Copy the token provided<br>"
                "4. Start a chat with your bot or add it to a group<br>"
                "5. Use @userinfobot to get your Chat ID"
                "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Send Notifications ───────────────────────────────────
    st.markdown("### 📤 Send Notifications")

    if score is None:
        st.warning("Run a prediction first to enable notifications.")
    else:
        category, color, emoji_r = get_risk_category(score)
        feat_summary = _features_summary(st.session_state.patient_data)
        nc_current   = st.session_state.notification_config
        smtp_current = nc_current.get("smtp", {})
        tg_current   = nc_current.get("telegram", {})
        is_demo      = not smtp_current.get("sender_email", "")

        if is_demo:
            st.markdown(
                "<div class='info-card orange-border' style='margin-bottom:1rem'>"
                "<strong style='color:#f39c12'>⚠ Demo Mode</strong><br>"
                "<span style='font-size:0.82rem;color:#9ca3af'>"
                "Email credentials not configured — buttons will show a preview of what would be sent. "
                "Enter SMTP / Telegram settings above to send real notifications."
                "</span></div>",
                unsafe_allow_html=True,
            )

        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        results_log = {}

        with btn_col1:
            if st.button("📧 Doctor Alert", use_container_width=True, type="primary"):
                to = nc_current.get("doctor_email", "")
                if is_demo or not to:
                    ok, msg = _demo_send("email", patient_id, score, category)
                else:
                    ok, msg = send_email_alert(
                        to, patient_id, score, category, smtp_current, feat_summary
                    )
                results_log["email_doctor"] = (ok, to, msg)
                log_alert(patient_id, score, category, {"email_doctor": (ok, msg)})
                if ok:
                    st.success(f"{'[DEMO] ' if is_demo else ''}Doctor email sent")
                else:
                    st.error(msg)

        with btn_col2:
            if st.button("👨‍👩‍👧 Family Alert", use_container_width=True):
                to = nc_current.get("family_email", "")
                if is_demo or not to:
                    ok, msg = _demo_send("email", patient_id, score, category)
                else:
                    ok, msg = send_email_alert(
                        to, patient_id, score, category, smtp_current, feat_summary
                    )
                log_alert(patient_id, score, category, {"email_family": (ok, msg)})
                if ok:
                    st.success(f"{'[DEMO] ' if is_demo else ''}Family email sent")
                else:
                    st.error(msg)

        with btn_col3:
            if st.button("💬 Telegram", use_container_width=True):
                bt = tg_current.get("bot_token", "")
                ci = tg_current.get("chat_id", "")
                if not bt or not ci:
                    ok, msg = _demo_send("telegram", patient_id, score, category)
                else:
                    ok, msg = send_telegram_alert(bt, ci, patient_id, score, category)
                log_alert(patient_id, score, category, {"telegram": (ok, msg)})
                if ok:
                    st.success(f"{'[DEMO] ' if not (bt and ci) else ''}Telegram sent")
                else:
                    st.error(msg)

        with btn_col4:
            if st.button("🔔 Send All", use_container_width=True, type="primary"):
                sent = {}
                # Doctor email
                to_d = nc_current.get("doctor_email", "")
                if is_demo or not to_d:
                    ok_d, msg_d = _demo_send("email", patient_id, score, category)
                else:
                    ok_d, msg_d = send_email_alert(to_d, patient_id, score, category, smtp_current, feat_summary)
                sent["email_doctor"] = (ok_d, msg_d)
                # Family email
                to_f = nc_current.get("family_email", "")
                if is_demo or not to_f:
                    ok_f, msg_f = _demo_send("email", patient_id, score, category)
                else:
                    ok_f, msg_f = send_email_alert(to_f, patient_id, score, category, smtp_current, feat_summary)
                sent["email_family"] = (ok_f, msg_f)
                # Telegram
                bt = tg_current.get("bot_token", "")
                ci = tg_current.get("chat_id", "")
                if not bt or not ci:
                    ok_t, msg_t = _demo_send("telegram", patient_id, score, category)
                else:
                    ok_t, msg_t = send_telegram_alert(bt, ci, patient_id, score, category)
                sent["telegram"] = (ok_t, msg_t)

                log_alert(patient_id, score, category, sent)
                ok_all = ok_d and ok_f and ok_t
                if ok_all:
                    st.success("All notifications dispatched successfully.")
                else:
                    st.warning("Some notifications failed — check settings above.")

        # ── Preview Panel ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📝 Preview Telegram Message", expanded=False):
            ts_preview = timestamp_now()
            preview_msg = format_telegram_message(patient_id, score, category, ts_preview)
            st.code(preview_msg, language=None)

        with st.expander("📝 Preview Email HTML", expanded=False):
            ts_preview = timestamp_now()
            st.markdown(
                format_alert_email_html(patient_id, score, category, ts_preview, feat_summary),
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Alert Log ────────────────────────────────────────────
    st.markdown("### 📋 Alert Log")

    alerts = st.session_state.alerts
    if not alerts:
        st.markdown(
            "<div style='text-align:center;padding:2rem;background:#1e2130;"
            "border-radius:12px;border:1px dashed #2d3748;color:#6b7280'>"
            "<span style='font-size:2rem'>📋</span><br>No alerts logged yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        log_rows = []
        for a in reversed(alerts):
            nc_sent = a.get("notifications_sent", {})
            channels_ok = ", ".join(
                [ch for ch, info in nc_sent.items() if info.get("success")]
            ) or "—"
            log_rows.append({
                "Timestamp":     a["timestamp"],
                "Patient ID":    a["patient_id"],
                "Risk Score":    a["risk_pct"],
                "Alert Level":   a["alert_level"].upper(),
                "Notifications": channels_ok,
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        if len(alerts) >= 2:
            st.divider()
            st.markdown("### 📈 Alert History Timeline")
            import numpy as np
            from datetime import timedelta

            evt_rows = []
            for a in alerts:
                try:
                    dt = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.now()
                evt_rows.append({
                    "date":        dt,
                    "event_type":  a["alert_level"],
                    "risk_score":  a["risk_score"],
                    "description": f"Patient {a['patient_id']} — {a['risk_pct']}",
                })
            evt_df = pd.DataFrame(evt_rows)
            st.plotly_chart(plot_timeline_events(evt_df), use_container_width=True)

    st.markdown(
        "<div class='footer'>🫀 PanicGuard AI · Research Prototype · Not for clinical diagnosis</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
