"""
notification_utils.py
Email and Telegram notification system for PanicGuard AI.
"""

import json
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

import requests
import streamlit as st


# ─────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────

def load_notification_config() -> dict:
    """
    Load notification config from config.json or session state override.

    Returns a dict with 'smtp' and 'telegram' sub-dicts.
    """
    # Session state overrides take priority
    if "notification_config" in st.session_state and st.session_state.notification_config:
        return st.session_state.notification_config

    base = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base, "config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return cfg.get("notifications", {})
    except Exception:
        return {
            "smtp": {
                "server": "smtp.gmail.com",
                "port": 587,
                "sender_email": "",
                "sender_password": "",
                "use_tls": True,
            },
            "telegram": {
                "bot_token": "",
                "chat_id": "",
            },
            "alert_threshold": 0.60,
            "cooldown_minutes": 30,
        }


# ─────────────────────────────────────────────
# HTML Email Template
# ─────────────────────────────────────────────

def format_alert_email_html(
    patient_id: str,
    risk_score: float,
    alert_level: str,
    timestamp: str,
    features_summary: Optional[dict] = None,
) -> str:
    """
    Render a professional HTML clinical alert email.

    Parameters
    ----------
    patient_id      : patient identifier string
    risk_score      : float 0–1
    alert_level     : 'normal' | 'elevated' | 'critical'
    timestamp       : formatted datetime string
    features_summary: optional dict of {feature_name: value} for key signals

    Returns
    -------
    str – complete HTML document
    """
    level_colors = {
        "critical": "#e74c3c",
        "elevated": "#f39c12",
        "normal":   "#2ecc71",
    }
    level_emojis = {
        "critical": "🔴",
        "elevated": "🟡",
        "normal":   "🟢",
    }
    color = level_colors.get(alert_level.lower(), "#4F8BF9")
    emoji = level_emojis.get(alert_level.lower(), "⚪")
    pct   = f"{risk_score * 100:.1f}%"

    # Build feature rows
    feature_rows = ""
    if features_summary:
        for feat, val in list(features_summary.items())[:8]:
            label = feat.replace("_", " ").title()
            feature_rows += f"""
            <tr>
                <td style="padding:6px 12px;border-bottom:1px solid #2d3748;color:#9ca3af">{label}</td>
                <td style="padding:6px 12px;border-bottom:1px solid #2d3748;color:#e8eaf0;font-weight:600">{val}</td>
            </tr>"""

    features_block = f"""
    <h3 style="color:#9ca3af;font-size:14px;margin:20px 0 8px">Physiological Summary</h3>
    <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden">
        <thead>
            <tr>
                <th style="padding:8px 12px;background:#16213e;color:#e8eaf0;text-align:left;font-size:12px">Signal</th>
                <th style="padding:8px 12px;background:#16213e;color:#e8eaf0;text-align:left;font-size:12px">Current Value</th>
            </tr>
        </thead>
        <tbody>{feature_rows}</tbody>
    </table>""" if features_summary else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PanicGuard AI Alert</title>
</head>
<body style="margin:0;padding:0;background:#0d1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif">
<div style="max-width:600px;margin:0 auto;padding:20px">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px 12px 0 0;padding:28px 32px;border:1px solid #2d3748;border-bottom:none">
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:32px">🫀</span>
      <div>
        <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700">PanicGuard AI</h1>
        <p style="margin:4px 0 0;color:#6b7280;font-size:13px">Clinical Risk Alert — Research Prototype</p>
      </div>
    </div>
  </div>

  <!-- Alert Banner -->
  <div style="background:{color}1a;border-left:5px solid {color};padding:20px 28px;border-right:1px solid #2d3748">
    <h2 style="margin:0 0 6px;color:{color};font-size:18px;font-weight:700">
      {emoji} {alert_level.upper()} RISK ALERT
    </h2>
    <p style="margin:0;color:#9ca3af;font-size:13px">Automated alert triggered by risk threshold detection</p>
  </div>

  <!-- Main Content -->
  <div style="background:#1e2130;padding:24px 28px;border:1px solid #2d3748;border-top:none">
    <!-- Key Metrics -->
    <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap">
      <div style="flex:1;min-width:140px;background:#0d1117;border-radius:8px;padding:16px;text-align:center;border:1px solid #2d3748">
        <div style="font-size:28px;font-weight:800;color:{color}">{pct}</div>
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Risk Score</div>
      </div>
      <div style="flex:1;min-width:140px;background:#0d1117;border-radius:8px;padding:16px;text-align:center;border:1px solid #2d3748">
        <div style="font-size:18px;font-weight:700;color:#e8eaf0">{patient_id}</div>
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Patient ID</div>
      </div>
      <div style="flex:1;min-width:140px;background:#0d1117;border-radius:8px;padding:16px;text-align:center;border:1px solid #2d3748">
        <div style="font-size:13px;font-weight:600;color:#e8eaf0">{timestamp}</div>
        <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Timestamp</div>
      </div>
    </div>

    {features_block}

    <!-- Disclaimer -->
    <div style="margin-top:20px;padding:12px 16px;background:rgba(231,76,60,0.08);border:1px solid rgba(231,76,60,0.2);border-radius:8px">
      <p style="margin:0;color:#9ca3af;font-size:12px;line-height:1.6">
        <strong style="color:#e74c3c">⚠ Research Disclaimer:</strong> This alert is generated by an experimental AI system
        and is NOT a medical diagnosis. It is intended for research monitoring purposes only.
        Clinical judgment must be applied before any medical decision.
      </p>
    </div>
  </div>

  <!-- Footer -->
  <div style="background:#16213e;border-radius:0 0 12px 12px;padding:16px 28px;border:1px solid #2d3748;border-top:none;text-align:center">
    <p style="margin:0;color:#6b7280;font-size:11px">
      PanicGuard AI — Master Thesis Research Prototype | Generated at {timestamp}
    </p>
  </div>

</div>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────
# Telegram Message Template
# ─────────────────────────────────────────────

def format_telegram_message(
    patient_id: str,
    risk_score: float,
    alert_level: str,
    timestamp: str,
) -> str:
    """
    Return a formatted Telegram message string for clinical alerts.
    """
    level_emojis = {
        "critical": "🔴",
        "elevated": "🟡",
        "normal":   "🟢",
    }
    emoji = level_emojis.get(alert_level.lower(), "⚪")
    pct   = f"{risk_score * 100:.1f}%"

    msg = (
        f"{emoji} *PanicGuard AI — {alert_level.upper()} RISK ALERT*\n"
        f"{'━' * 32}\n"
        f"🏥 *Patient:* `{patient_id}`\n"
        f"📊 *Risk Score:* `{pct}`\n"
        f"⏰ *Timestamp:* `{timestamp}`\n"
        f"{'━' * 32}\n"
    )

    if alert_level.lower() == "critical":
        msg += (
            "🚨 *Action Required:* Critical risk threshold exceeded.\n"
            "Please review patient physiological data immediately.\n"
        )
    elif alert_level.lower() == "elevated":
        msg += (
            "⚠️ *Note:* Elevated risk detected.\n"
            "Monitor closely over the next 24–48 hours.\n"
        )
    else:
        msg += "✅ Risk is within normal parameters.\n"

    msg += (
        f"\n_{'—' * 20}_\n"
        "⚠️ _Research prototype only — not a medical diagnosis._\n"
        "_PanicGuard AI Master Thesis System_"
    )
    return msg


# ─────────────────────────────────────────────
# Email Sending
# ─────────────────────────────────────────────

def send_email_alert(
    to_email: str,
    patient_id: str,
    risk_score: float,
    alert_level: str,
    smtp_config: dict,
    features_summary: Optional[dict] = None,
) -> tuple:
    """
    Send an HTML clinical alert email via SMTP.

    Parameters
    ----------
    to_email        : recipient email address
    patient_id      : patient identifier
    risk_score      : float 0–1
    alert_level     : 'normal' | 'elevated' | 'critical'
    smtp_config     : dict with keys: server, port, sender_email, sender_password, use_tls
    features_summary: optional key→value dict of physiological signals

    Returns
    -------
    (success: bool, message: str)
    """
    if not to_email:
        return False, "Recipient email address is empty."
    if not smtp_config.get("sender_email"):
        return False, "Sender email is not configured."
    if not smtp_config.get("sender_password"):
        return False, "Sender password is not configured."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    html_body = format_alert_email_html(
        patient_id, risk_score, alert_level, timestamp, features_summary
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = (
        f"[PanicGuard AI] {alert_level.upper()} Risk Alert — Patient {patient_id} — "
        f"{risk_score * 100:.1f}%"
    )
    msg["From"] = smtp_config["sender_email"]
    msg["To"]   = to_email
    msg.attach(MIMEText(html_body, "html"))

    server   = smtp_config.get("server", "smtp.gmail.com")
    port     = int(smtp_config.get("port", 587))
    use_tls  = smtp_config.get("use_tls", True)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(server, port, timeout=15) as smtp:
            if use_tls:
                smtp.starttls(context=context)
            smtp.login(smtp_config["sender_email"], smtp_config["sender_password"])
            smtp.sendmail(smtp_config["sender_email"], to_email, msg.as_string())
        return True, f"Email alert successfully sent to {to_email}."
    except smtplib.SMTPAuthenticationError:
        return False, ("Authentication failed. Check email/password. "
                       "For Gmail, use an App Password (not your account password).")
    except smtplib.SMTPConnectError:
        return False, f"Could not connect to SMTP server '{server}:{port}'. Check network and server settings."
    except smtplib.SMTPRecipientsRefused:
        return False, f"Recipient address '{to_email}' was refused by the mail server."
    except TimeoutError:
        return False, "SMTP connection timed out. The server may be unreachable."
    except Exception as e:
        return False, f"Failed to send email: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────
# Telegram Sending
# ─────────────────────────────────────────────

def send_telegram_alert(
    bot_token: str,
    chat_id: str,
    patient_id: str,
    risk_score: float,
    alert_level: str,
) -> tuple:
    """
    Send a clinical alert via a Telegram bot using the sendMessage API.

    Parameters
    ----------
    bot_token  : Telegram bot token (from @BotFather)
    chat_id    : Target chat / channel ID
    patient_id : patient identifier
    risk_score : float 0–1
    alert_level: 'normal' | 'elevated' | 'critical'

    Returns
    -------
    (success: bool, message: str)
    """
    if not bot_token:
        return False, "Telegram bot token is not configured."
    if not chat_id:
        return False, "Telegram chat ID is not configured."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    text = format_telegram_message(patient_id, risk_score, alert_level, timestamp)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200 and response.json().get("ok"):
            return True, "Telegram alert sent successfully."
        else:
            err = response.json().get("description", "Unknown Telegram API error.")
            return False, f"Telegram API error: {err}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to Telegram API. Check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Telegram API request timed out."
    except Exception as e:
        return False, f"Failed to send Telegram alert: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────
# Alert Logging
# ─────────────────────────────────────────────

def log_alert(
    patient_id: str,
    risk_score: float,
    alert_level: str,
    notifications_sent: dict,
) -> None:
    """
    Append an alert record to st.session_state.alerts.

    Parameters
    ----------
    patient_id         : patient identifier string
    risk_score         : float 0–1
    alert_level        : 'normal' | 'elevated' | 'critical'
    notifications_sent : dict mapping channel→(success, message)
                         e.g. {'email': (True, 'Sent'), 'telegram': (False, 'No token')}
    """
    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    record = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id":         patient_id,
        "risk_score":         round(float(risk_score), 4),
        "risk_pct":           f"{risk_score * 100:.1f}%",
        "alert_level":        alert_level,
        "notifications_sent": {
            ch: {"success": ok, "message": msg}
            for ch, (ok, msg) in notifications_sent.items()
        },
    }
    st.session_state.alerts.append(record)
