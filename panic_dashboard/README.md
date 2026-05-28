# 🫀 PanicGuard AI

**Real-Time Panic Risk Monitoring Dashboard**  
Master Thesis Research Prototype — AI & Biomedical Signal Processing

---

## Overview

PanicGuard AI is a proof-of-concept clinical monitoring dashboard built with Streamlit.
It uses a **Bidirectional LSTM + Attention** deep learning model to estimate panic-risk
probability from 24 physiological wearable signals measured over a rolling 7-day window.

> ⚠️ **Research Prototype Only.** This system is NOT a certified medical device and must
> NOT be used for clinical diagnosis, treatment decisions, or emergency response.

---

## Features

- **Real-time risk scoring** — continuous 0–100% panic-risk probability
- **Bidirectional LSTM + Attention** — temporal deep learning architecture
- **24-signal monitoring** — cardiac, sleep, activity, stress, autonomic signals
- **AI explainability** — temporal attention weights + feature importance visualization
- **Multi-channel alerts** — email (SMTP) and Telegram bot notifications
- **Patient history** — 30-day risk trend, alert log, physiological evolution
- **Demo mode** — fully functional without the real model file using synthetic data
- **CSV upload** — accepts standard wearable export format

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd panic_dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## File Structure

```
panic_dashboard/
├── app.py                      # Home page (entry point)
├── config.json                 # Application configuration
├── requirements.txt            # Python dependencies
├── sample_patient.csv          # Demo patient CSV (critical scenario)
├── README.md                   # This file
│
├── pages/
│   ├── 1_🏥_Patient_Monitoring.py    # Patient monitoring interface
│   ├── 2_🧠_AI_Explainability.py     # Attention & feature importance
│   ├── 3_🚨_Alert_Center.py          # Notification configuration & dispatch
│   ├── 4_📊_Patient_History.py       # 30-day history & trends
│   └── 5_📖_About_Thesis.py          # Thesis documentation
│
├── models/
│   └── best_model_B.keras            # ← Place your trained model here
│
├── assets/
│   └── style.css                     # Dark medical theme CSS
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                    # Formatting & utility functions
│
├── model_utils.py              # Model loading, inference, demo simulation
├── preprocessing_utils.py      # Data pipeline: load → clean → normalize → window
├── visualization_utils.py      # Plotly chart functions
└── notification_utils.py       # Email (SMTP) & Telegram notifications
```

---

## Model Integration

The dashboard supports loading your trained Keras model:

1. Copy `best_model_B.keras` to the `models/` directory:
   ```
   panic_dashboard/models/best_model_B.keras
   ```

2. The app will automatically detect and load it on startup.

3. **Expected model interface:**
   - **Input shape:** `(batch_size, 7, 24)` — 7 days × 24 features
   - **Output shape:** `(batch_size, 1)` or `(batch_size,)` — sigmoid probability
   - **Architecture:** Bidirectional LSTM + Attention + Dense classifier
   - **Loss:** `binary_crossentropy`

If the model file is not found, the app runs in **Demo Mode** with simulated predictions.

---

## Configuration

Edit `config.json` to customise the application:

```json
{
  "model": {
    "path": "models/best_model_B.keras",
    "threshold": 0.5
  },
  "notifications": {
    "smtp": {
      "server": "smtp.gmail.com",
      "port": 587,
      "sender_email": "your.app@gmail.com",
      "sender_password": "your-app-password"
    },
    "telegram": {
      "bot_token": "123456:ABCdef...",
      "chat_id": "-100123456789"
    },
    "alert_threshold": 0.60
  }
}
```

---

## Notification Setup

### Gmail SMTP

1. Enable **2-Step Verification** on your Google account
2. Generate an **App Password**: Google Account → Security → App Passwords
3. Use the App Password (NOT your account password) in `config.json`

### Telegram Bot

1. Open Telegram → search for **@BotFather**
2. Send `/newbot` and follow the instructions
3. Copy the **bot token** provided
4. Start a conversation with your bot (or add to a group)
5. Use **@userinfobot** to find your **chat_id**
6. Enter both in the Alert Center settings panel

---

## Demo Mode

Without a real model, the dashboard generates realistic synthetic patients:

- **Normal scenario** — stable vitals, good sleep, low stress
- **Elevated Risk** — rising stress, declining sleep, increasing HR trend
- **Critical Risk** — severe anxiety markers, very poor sleep, high HR, low HRV

Navigate to **Patient Monitoring** → **Generate Demo Patient** to explore.

---

## CSV Upload Format

Your CSV must contain a `date` column (YYYY-MM-DD) and the 24 feature columns:

```
date, heart_rate, resting_heart_rate, hrv, spo2, sleep_duration, sleep_quality,
deep_sleep, rem_sleep, steps, activity_level, calories, distance, stress_score,
cortisol_proxy, movement_intensity, sedentary_time, skin_temperature,
galvanic_skin_response, breathing_rate, systolic_bp, diastolic_bp,
mood_score, fatigue_level, anxiety_proxy
```

Missing columns are imputed with column medians. Minimum 7 rows required.
See `sample_patient.csv` for a complete example.

---

## Model Performance (LOSO Cross-Validation)

| Metric   | Mean   | Std Dev | Note                              |
|----------|--------|---------|-----------------------------------|
| ROC-AUC  | 0.4902 | ±0.2082 | Near-chance — high variability    |
| PR-AUC   | 0.2238 | ±0.2941 | Class imbalance challenge         |
| F1 Score | 0.2943 | ±0.2874 | Precision–recall trade-off        |
| Recall   | 0.8092 | ±0.3087 | ★ Optimised — clinical priority   |

*Evaluated using Leave-One-Subject-Out (LOSO) cross-validation.*

---

## Ethical Disclaimer

This system:
- Is **NOT** a medical device
- Has **NOT** been clinically validated
- Must **NOT** be used for diagnosis or treatment
- Is intended for **research purposes only**

If you or someone you know is experiencing a mental health crisis, please contact
a qualified healthcare professional or emergency services immediately.

---

## Technology Stack

| Component         | Technology                    |
|-------------------|-------------------------------|
| Dashboard         | Streamlit ≥ 1.28              |
| Deep Learning     | TensorFlow / Keras ≥ 2.13     |
| Visualisation     | Plotly ≥ 5.15                 |
| Data Processing   | Pandas ≥ 2.0, NumPy ≥ 1.24   |
| ML Utilities      | scikit-learn ≥ 1.3            |
| Email             | Python smtplib (built-in)     |
| Telegram          | requests (HTTP API)           |

---

## License

This project is released for academic research purposes only.
Commercial use or clinical deployment is strictly prohibited without explicit authorisation.

---

*PanicGuard AI · Master Thesis Research Prototype · 2024*
