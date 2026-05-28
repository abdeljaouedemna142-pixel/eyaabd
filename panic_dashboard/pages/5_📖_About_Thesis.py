"""
pages/5_📖_About_Thesis.py
Full thesis documentation page for PanicGuard AI.
"""

import os
import json
import sys

import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="About / Thesis — PanicGuard AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import generate_patient_id


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
        "patient_data":         None, "alerts":               [],
        "current_risk_score":   None, "prediction_history":   [],
        "patient_id":           generate_patient_id(),
        "selected_scenario":    "normal",
        "notification_config":  config.get("notifications", {}),
        "model_loaded":         False, "demo_mode": True,
        "preprocessing_result": None, "attention_weights":    None,
        "feature_importance":   None, "last_prediction_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _section(title: str, content: str, border_color: str = "#4F8BF9") -> None:
    st.markdown(
        f"<div class='thesis-section' style='border-left:4px solid {border_color}'>"
        f"<h3 style='color:{border_color};margin-top:0'>{title}</h3>"
        f"<div style='color:#9ca3af;font-size:0.88rem;line-height:1.75'>{content}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _radar_results_fig(config: dict) -> go.Figure:
    """Spider chart of model metrics vs ideal (1.0)."""
    perf   = config.get("performance", {})
    labels = ["ROC-AUC", "PR-AUC", "F1 Score", "Recall"]
    actual = [
        perf.get("roc_auc_mean", 0.49),
        perf.get("pr_auc_mean",  0.22),
        perf.get("f1_mean",      0.29),
        perf.get("recall_mean",  0.81),
    ]
    labels_c = labels + [labels[0]]
    actual_c = actual + [actual[0]]
    ideal_c  = [1.0] * len(labels_c)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ideal_c, theta=labels_c, fill="toself",
        fillcolor="rgba(46,204,113,0.06)", line=dict(color="#2ecc71", width=1, dash="dash"),
        name="Ideal (1.0)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=actual_c, theta=labels_c, fill="toself",
        fillcolor="rgba(79,139,249,0.15)", line=dict(color="#4F8BF9", width=2.5),
        marker=dict(size=8, color="#4F8BF9"),
        name="Model B (BiLSTM + Attention)",
        hovertemplate="<b>%{theta}</b>: %{r:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=400,
        paper_bgcolor="#0d1117",
        font=dict(color="#e8eaf0"),
        polar=dict(
            bgcolor="#1e2130",
            angularaxis=dict(color="#6b7280", gridcolor="#2d3748", tickfont=dict(size=12)),
            radialaxis=dict(visible=True, range=[0, 1], color="#6b7280",
                            gridcolor="#2d3748", tickfont=dict(size=9)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#2d3748", borderwidth=1),
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def _loso_diagram_fig() -> go.Figure:
    """Simple Sankey-style diagram explaining LOSO cross-validation."""
    labels = ["Subject 1", "Subject 2", "Subject 3", "Subject N",
              "Train Set", "Test Set", "Final Score"]
    source = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
    target = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6]
    values = [1, 1, 1, 1, 1, 1, 1, 1, 4, 4]
    colors = ["rgba(79,139,249,0.5)"] * 8 + ["rgba(46,204,113,0.4)", "rgba(231,76,60,0.4)"]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=labels,
            color=["#4F8BF9", "#4F8BF9", "#4F8BF9", "#4F8BF9",
                   "#2ecc71", "#e74c3c", "#f39c12"],
            line=dict(color="#2d3748", width=0.5),
        ),
        link=dict(source=source, target=target, value=values, color=colors),
    ))
    fig.update_layout(
        title="Leave-One-Subject-Out (LOSO) Cross-Validation",
        height=350,
        paper_bgcolor="#0d1117",
        font=dict(color="#e8eaf0"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ── Main ─────────────────────────────────────────────────────

def main():
    _load_css()
    config = _load_config()
    _init_session(config)

    perf = config.get("performance", {})

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:10px 0 16px'>"
            "<span style='font-size:2.5rem'>📖</span><br>"
            "<span style='font-weight:700;color:#e8eaf0;font-size:1.1rem'>About / Thesis</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<div style='font-size:0.82rem;color:#9ca3af'>"
            "<strong style='color:#4F8BF9'>Quick Navigation</strong><br>"
            "• Abstract<br>"
            "• Datasets<br>"
            "• Model Architecture<br>"
            "• Evaluation Results<br>"
            "• Limitations<br>"
            "• Ethical Considerations<br>"
            "• Future Work"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class='hero-section' style='margin-bottom:2rem'>
          <div class='gradient-title'>📖 Thesis Documentation</div>
          <p class='hero-subtitle'>
            <strong>PanicGuard AI</strong> — Automatic Panic Attack Risk Detection<br>
            from Wearable Physiological Signals Using Bidirectional LSTM with Attention<br>
            <em>Master Thesis · AI &amp; Biomedical Signal Processing · 2024</em>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Thesis title card ────────────────────────────────────
    st.markdown(
        "<div class='info-card blue-border' style='padding:1.5rem;margin-bottom:2rem'>"
        "<table style='width:100%;font-size:0.87rem'>"
        "<tr><td style='color:#6b7280;padding:5px 0;width:180px'>Thesis Title</td>"
        "<td style='color:#e8eaf0;font-weight:600'>"
        "Physiological Panic Risk Detection Using Bidirectional LSTM with Attention Mechanism "
        "on Multivariate Wearable Time-Series Data</td></tr>"
        "<tr><td style='color:#6b7280;padding:5px 0'>Field</td>"
        "<td style='color:#e8eaf0'>AI &amp; Biomedical Signal Processing</td></tr>"
        "<tr><td style='color:#6b7280;padding:5px 0'>Model</td>"
        "<td style='color:#4F8BF9;font-weight:600'>Model B — Bidirectional LSTM + Attention</td></tr>"
        "<tr><td style='color:#6b7280;padding:5px 0'>Validation</td>"
        "<td style='color:#e8eaf0'>LOSO (Leave-One-Subject-Out) Cross-Validation</td></tr>"
        "<tr><td style='color:#6b7280;padding:5px 0'>Task</td>"
        "<td style='color:#e8eaf0'>Binary classification: panic-risk vs. normal (0/1)</td></tr>"
        "<tr><td style='color:#6b7280;padding:5px 0'>Input</td>"
        "<td style='color:#e8eaf0'>7 days × 24 physiological features</td></tr>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Abstract ─────────────────────────────────────────────
    _section(
        "📝 Abstract",
        """
        Panic disorder is a debilitating condition characterised by recurrent episodes of intense fear
        accompanied by physiological symptoms including tachycardia, hyperventilation, and autonomic
        dysregulation. Early detection of pre-panic physiological states may enable timely clinical
        intervention and improve patient outcomes.
        <br><br>
        This thesis investigates the feasibility of predicting panic-risk probability from
        multivariate physiological time-series data captured by commercial wearable devices.
        A <strong>Bidirectional LSTM (BiLSTM) architecture augmented with an Attention mechanism</strong>
        is trained to classify 7-day physiological windows as either at-risk or normal.
        The model is evaluated using <strong>Leave-One-Subject-Out (LOSO) cross-validation</strong>
        to assess its generalisation capability across individuals. Training data combines
        the <strong>WESAD stress dataset</strong> with digital phenotyping wearable records,
        yielding 24 physiological and behavioural features per daily observation.
        <br><br>
        Results indicate a mean Recall of <strong>80.9% ± 30.9%</strong>, reflecting the model's
        sensitivity to positive risk states — prioritised over precision given the clinical cost
        asymmetry between false negatives and false positives in mental health monitoring.
        The ROC-AUC of <strong>0.49 ± 0.21</strong> suggests near-chance discriminative performance,
        attributed to high inter-subject variability and the limited size of the available dataset.
        """,
        "#4F8BF9",
    )

    # ── Motivation ───────────────────────────────────────────
    _section(
        "💡 Motivation & Clinical Context",
        """
        Panic disorder affects approximately <strong>2–3% of the global population</strong> and is
        characterised by unpredictable panic attacks — acute episodes of overwhelming fear accompanied
        by physical symptoms that often mimic cardiac events. The disorder imposes a significant burden
        on patients, caregivers, and healthcare systems.
        <br><br>
        Consumer-grade wearable devices now offer continuous, passive collection of physiological data
        including heart rate, heart rate variability (HRV), sleep architecture, skin conductance,
        movement patterns, and estimated stress levels. This <strong>digital phenotyping</strong>
        approach allows longitudinal monitoring of physiological state without requiring active
        patient participation.
        <br><br>
        The thesis proposes that patterns in these wearable biosignals preceding panic episodes
        may be detectable through deep learning, enabling a <strong>passive early-warning system</strong>
        that alerts patients and clinicians to elevated risk states before a full panic episode occurs.
        <br><br>
        <strong>Key clinical questions addressed:</strong>
        <ul>
          <li>Can wearable physiological signals distinguish pre-panic from baseline states?</li>
          <li>Do temporal patterns over a 7-day window carry predictive information?</li>
          <li>Which physiological signals are most informative for panic risk classification?</li>
          <li>How well does a BiLSTM + Attention model generalise across individuals (LOSO)?</li>
        </ul>
        """,
        "#7C5CBF",
    )

    st.divider()

    # ── Datasets ─────────────────────────────────────────────
    st.markdown("## 📁 Datasets Used")

    ds_col1, ds_col2 = st.columns(2, gap="large")

    with ds_col1:
        st.markdown(
            "<div class='thesis-section'>"
            "<h3 style='color:#4F8BF9'>WESAD Dataset</h3>"
            "<p style='color:#9ca3af;font-size:0.85rem'>"
            "<strong>Full name:</strong> Wearable Stress and Affect Detection<br>"
            "<strong>Type:</strong> Laboratory physiological dataset<br>"
            "<strong>Subjects:</strong> 15 participants<br>"
            "<strong>Signals:</strong> ECG, EDA, EMG, respiration, skin temperature, BVP, accelerometer<br>"
            "<strong>Protocol:</strong> Controlled stress induction (Trier Social Stress Test)<br>"
            "<strong>Labels:</strong> Baseline, stress, amusement, meditation<br><br>"
            "WESAD provides high-quality ground-truth physiological data collected under "
            "controlled laboratory conditions. It was used to learn physiological stress signatures "
            "as a proxy for autonomic dysregulation associated with panic states."
            "</p></div>",
            unsafe_allow_html=True,
        )

    with ds_col2:
        st.markdown(
            "<div class='thesis-section'>"
            "<h3 style='color:#7C5CBF'>Digital Phenotyping Data</h3>"
            "<p style='color:#9ca3af;font-size:0.85rem'>"
            "<strong>Type:</strong> Longitudinal wearable monitoring<br>"
            "<strong>Collection:</strong> Consumer wearable devices<br>"
            "<strong>Signals:</strong> Heart rate, HRV, sleep metrics, activity, stress score<br>"
            "<strong>Duration:</strong> Daily observations over multiple weeks<br>"
            "<strong>Features:</strong> 24 engineered physiological indicators<br><br>"
            "Digital phenotyping data provides the ecological validity needed to simulate "
            "real-world wearable monitoring. Combined with WESAD, it enables learning of "
            "temporally structured patterns spanning multiple days of physiological evolution."
            "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='info-card orange-border' style='margin:1rem 0'>"
        "<strong style='color:#f39c12'>⚠ Dataset Limitations</strong><br>"
        "<span style='font-size:0.82rem;color:#9ca3af'>"
        "The combined dataset has a limited sample size, high class imbalance (panic-risk events are rare), "
        "and significant inter-subject physiological variability. These factors directly contribute "
        "to the high variance observed in LOSO cross-validation results and near-chance ROC-AUC scores."
        "</span></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model Architecture ───────────────────────────────────
    st.markdown("## 🏗 Model Architecture — Model B (BiLSTM + Attention)")

    arch_col, detail_col = st.columns([1, 1], gap="large")

    with arch_col:
        st.markdown(
            """
            ```
            ┌─────────────────────────────────────────┐
            │           INPUT LAYER                   │
            │     Shape: (batch, 7, 24)                │
            │     7 days × 24 features                 │
            └──────────────────┬──────────────────────┘
                               │
            ┌──────────────────▼──────────────────────┐
            │       BIDIRECTIONAL LSTM                 │
            │  → Forward LSTM  ←  Backward LSTM        │
            │  Captures forward & backward             │
            │  temporal dependencies                   │
            └──────────────────┬──────────────────────┘
                               │
            ┌──────────────────▼──────────────────────┐
            │          ATTENTION LAYER                 │
            │  Learnable per-timestep weights          │
            │  Weighted sum of hidden states           │
            │  Produces context vector                 │
            └──────────────────┬──────────────────────┘
                               │
            ┌──────────────────▼──────────────────────┐
            │         DENSE CLASSIFIER                 │
            │  Fully connected + Dropout               │
            │  Batch Normalisation                     │
            └──────────────────┬──────────────────────┘
                               │
            ┌──────────────────▼──────────────────────┐
            │          OUTPUT LAYER                    │
            │  Dense(1) + Sigmoid activation           │
            │  Output: panic-risk probability [0,1]    │
            └─────────────────────────────────────────┘
            ```
            """,
        )

    with detail_col:
        st.markdown("#### Training Configuration")
        config_rows = [
            ("Loss Function",    "Binary Cross-Entropy"),
            ("Optimizer",        "Adam"),
            ("Learning Rate",    "3 × 10⁻⁴ (0.0003)"),
            ("Input Shape",      "(batch, 7, 24)"),
            ("Output",           "Sigmoid — panic-risk probability"),
            ("Classification",   "Binary (0 = normal, 1 = panic-risk)"),
            ("Validation",       "LOSO (Leave-One-Subject-Out)"),
        ]
        for lbl, val in config_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #2d3748'>"
                f"<span style='color:#6b7280;font-size:0.82rem'>{lbl}</span>"
                f"<span style='color:#e8eaf0;font-size:0.82rem;font-weight:600'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Primary Metrics")
        metric_rows = [
            ("ROC-AUC",   "Area under ROC curve — overall discrimination"),
            ("PR-AUC",    "Area under PR curve — imbalanced class performance"),
            ("F1-Score",  "Harmonic mean of precision and recall"),
            ("Recall",    "Sensitivity — fraction of true positives detected"),
        ]
        for m, d in metric_rows:
            st.markdown(
                f"<div class='info-card blue-border' style='padding:0.5rem 0.75rem;margin-bottom:0.4rem'>"
                f"<strong style='color:#4F8BF9;font-size:0.82rem'>{m}</strong> "
                f"<span style='color:#6b7280;font-size:0.78rem'>— {d}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='info-card green-border' style='margin-top:0.75rem'>"
            "<strong style='color:#2ecc71'>Why Recall is Prioritised</strong><br>"
            "<span style='font-size:0.8rem;color:#9ca3af'>"
            "In clinical early-warning systems, <em>missing a true positive</em> (false negative) "
            "carries a higher cost than a false alarm (false positive). "
            "Maximising Recall ensures the system captures the majority of at-risk states, "
            "even at the expense of increased false alerts."
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Evaluation Results ───────────────────────────────────
    st.markdown("## 📊 Evaluation Results")

    res_col, radar_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown("#### LOSO Cross-Validation — Model B (BiLSTM + Attention)")

        results = [
            ("ROC-AUC",  perf.get("roc_auc_mean", 0.49), perf.get("roc_auc_std", 0.21), "#f39c12",
             "Near-chance — high inter-subject variability"),
            ("PR-AUC",   perf.get("pr_auc_mean",  0.22), perf.get("pr_auc_std",  0.29), "#4F8BF9",
             "Low precision due to severe class imbalance"),
            ("F1 Score", perf.get("f1_mean",       0.29), perf.get("f1_std",       0.29), "#7C5CBF",
             "Reflects precision–recall trade-off"),
            ("Recall",   perf.get("recall_mean",   0.81), perf.get("recall_std",   0.31), "#2ecc71",
             "★ Optimised metric — clinical sensitivity priority"),
        ]

        st.markdown(
            "<table class='results-table'>"
            "<thead><tr>"
            "<th>Metric</th><th>Mean</th><th>Std Dev</th><th>Interpretation</th>"
            "</tr></thead><tbody>",
            unsafe_allow_html=True,
        )
        for name, mean_v, std_v, col, note in results:
            pct = int(mean_v * 100)
            st.markdown(
                f"<tr>"
                f"<td style='font-weight:600;color:{col}'>{name}</td>"
                f"<td style='color:#e8eaf0;font-weight:700'>{mean_v:.4f}</td>"
                f"<td style='color:#6b7280'>± {std_v:.4f}</td>"
                f"<td style='color:#9ca3af;font-size:0.82rem'>{note}</td>"
                f"</tr>",
                unsafe_allow_html=True,
            )
        st.markdown("</tbody></table>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-card orange-border'>"
            "<strong style='color:#f39c12'>LOSO Strategy</strong><br>"
            "<span style='font-size:0.82rem;color:#9ca3af'>"
            "In each LOSO fold, all data from one subject is held out as the test set. "
            "The model is trained on the remaining subjects and evaluated on the unseen individual. "
            "This reflects real-world generalisation: the model must perform on a <em>new person</em> "
            "it has never seen during training — the hardest and most realistic evaluation paradigm."
            "</span></div>",
            unsafe_allow_html=True,
        )

    with radar_col:
        st.plotly_chart(_radar_results_fig(config), use_container_width=True)

    st.divider()

    # ── Transfer Learning ────────────────────────────────────
    _section(
        "🔄 Transfer Learning Experiments",
        """
        In addition to Model B (BiLSTM + Attention trained from scratch), the thesis explored
        <strong>transfer learning</strong> strategies to leverage pre-trained temporal representations:
        <br><br>
        <strong>Approach A — Pre-training on WESAD:</strong> The model was initially trained on the
        full WESAD dataset (stress vs. baseline) and then fine-tuned on the combined digital phenotyping
        dataset with panic-risk labels. This warm-start approach aimed to encode physiological stress
        representations before adapting to the specific panic-risk classification task.
        <br><br>
        <strong>Approach B — Cross-dataset adaptation:</strong> Features learned from WESAD's
        high-resolution multimodal recordings were projected into the 24-feature wearable space
        using feature alignment, enabling indirect knowledge transfer despite modality mismatch.
        <br><br>
        <strong>Findings:</strong> Transfer learning provided marginal improvements in PR-AUC but
        did not significantly boost ROC-AUC or F1, suggesting that domain shift between laboratory
        stress data and naturalistic wearable panic risk remains a key challenge. Personalisation
        through subject-specific fine-tuning remains a promising direction for future work.
        """,
        "#1abc9c",
    )

    st.divider()

    # ── Limitations ──────────────────────────────────────────
    st.markdown("## ⚠ Limitations")

    lims = [
        ("📉 Small Dataset",
         "The combined dataset contains a limited number of subjects with confirmed panic disorder diagnoses. "
         "This restricts the model's ability to learn generalisable representations and leads to high variance "
         "in LOSO cross-validation results.",
         "#e74c3c"),
        ("⚖️ Severe Class Imbalance",
         "Panic-risk events are rare relative to normal physiological states. The dataset exhibits strong "
         "class imbalance, making precision-oriented metrics (PR-AUC, F1) difficult to optimise without "
         "aggressive oversampling or class weighting strategies.",
         "#f39c12"),
        ("👤 Inter-Subject Variability",
         "Physiological baselines differ substantially across individuals. A heart rate of 90 bpm may be "
         "normal for one subject and indicative of stress in another. Without personalisation, a "
         "population-level model will produce highly variable per-subject performance — as reflected "
         "in the large standard deviations across LOSO folds.",
         "#f39c12"),
        ("🎯 Near-Chance ROC-AUC",
         "The ROC-AUC of 0.49 ± 0.21 is statistically indistinguishable from random classification (0.50). "
         "This suggests the current model formulation may not capture sufficient discriminative signal "
         "from the available features at the population level.",
         "#e74c3c"),
        ("🔬 Proxy Labels",
         "True panic attack ground truth is difficult to capture outside clinical settings. The project "
         "relies on stress and anxiety proxy labels derived from WESAD and digital phenotyping data, "
         "which may not perfectly correspond to clinical panic disorder episodes.",
         "#7C5CBF"),
        ("🏥 No Clinical Validation",
         "The system has not been evaluated in a real clinical setting. No prospective study has been "
         "conducted to assess the model's performance on actual patients with diagnosed panic disorder "
         "under naturalistic monitoring conditions.",
         "#e74c3c"),
    ]

    lim_cols = st.columns(2)
    for i, (title, text, col) in enumerate(lims):
        with lim_cols[i % 2]:
            st.markdown(
                f"<div class='info-card' style='border-left:4px solid {col};margin-bottom:0.75rem'>"
                f"<strong style='color:{col}'>{title}</strong><br>"
                f"<span style='font-size:0.82rem;color:#9ca3af'>{text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Ethics ───────────────────────────────────────────────
    st.markdown("## 🔒 Ethical Considerations")

    ethics = [
        ("🚫 Not a Medical Device",
         "PanicGuard AI is an academic research prototype and has not been registered or certified "
         "as a medical device. It must not be used for clinical diagnosis, psychiatric evaluation, "
         "or treatment decisions.",
         "#e74c3c"),
        ("🔐 Data Privacy",
         "Wearable physiological data is sensitive personal health information. Any real-world "
         "deployment must comply with applicable data protection regulations (GDPR, HIPAA) and "
         "ensure data minimisation, encryption at rest and in transit, and strict access control.",
         "#4F8BF9"),
        ("✅ Informed Consent",
         "Participants whose data is used to train or evaluate the model must provide explicit "
         "informed consent. Patients using a monitoring system must be clearly informed about "
         "how their data is processed and what the system can and cannot detect.",
         "#2ecc71"),
        ("⚖️ Algorithmic Fairness",
         "The model may exhibit differential performance across demographic groups (age, sex, "
         "ethnicity) if these are under-represented in training data. Bias evaluation across "
         "subgroups is essential before any clinical deployment.",
         "#f39c12"),
        ("🧠 Psychological Impact",
         "False positive alerts (predicting high risk when the patient is healthy) may cause "
         "anxiety or unnecessary clinical escalation. The system should be designed with "
         "appropriate alert fatigue management and clear communication of uncertainty.",
         "#7C5CBF"),
        ("🌐 Explainability Obligation",
         "Clinical AI systems should be interpretable to clinicians and patients. "
         "The attention-based explainability in this system provides a first step toward "
         "transparency, but further validation of the explanations is required.",
         "#1abc9c"),
    ]

    eth_cols = st.columns(2)
    for i, (t, txt, col) in enumerate(ethics):
        with eth_cols[i % 2]:
            st.markdown(
                f"<div class='info-card' style='border-left:4px solid {col};margin-bottom:0.75rem'>"
                f"<strong style='color:{col}'>{t}</strong><br>"
                f"<span style='font-size:0.82rem;color:#9ca3af'>{txt}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Future Work ──────────────────────────────────────────
    st.markdown("## 🚀 Future Work")

    future = [
        ("📈", "Larger Clinical Dataset",
         "Prospective data collection from patients with confirmed panic disorder diagnoses "
         "would significantly improve label quality and class balance."),
        ("👤", "Personalised / Federated Learning",
         "Subject-specific fine-tuning or federated learning across individuals would address "
         "inter-subject variability without centralising sensitive health data."),
        ("🔗", "Multi-Modal Fusion",
         "Integrating smartphone behavioural data (typing patterns, GPS mobility, screen time) "
         "with physiological wearable signals may improve discriminative power."),
        ("🏥", "Clinical Validation Study",
         "A prospective clinical study with real-time monitoring, expert psychiatric evaluation, "
         "and ecological momentary assessment is required before any deployment."),
        ("🧠", "Foundation Model Pre-training",
         "Pre-training a large temporal model on population-level wearable datasets "
         "(e.g. UK Biobank) and fine-tuning on panic risk labels may dramatically "
         "improve generalisation."),
        ("📱", "Mobile Deployment",
         "Edge-optimised model variants (TensorFlow Lite, ONNX) enabling real-time "
         "on-device inference without cloud dependency would enhance privacy and latency."),
    ]

    fut_cols = st.columns(3)
    for i, (ico, title, txt) in enumerate(future):
        with fut_cols[i % 3]:
            st.markdown(
                f"<div class='nav-card' style='text-align:left;margin-bottom:0.75rem'>"
                f"<span style='font-size:1.8rem'>{ico}</span>"
                f"<div class='nav-title' style='text-align:left;margin-top:0.4rem'>{title}</div>"
                f"<div class='nav-desc'>{txt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── References ───────────────────────────────────────────
    _section(
        "📚 Key References",
        """
        <ol style='margin-left:1.2rem'>
          <li>Schmidt, P. et al. (2018). <em>WESAD: A multimodal dataset for wearable stress and affect detection.</em>
          ICMI '18. ACM. <a href='#' style='color:#4F8BF9'>doi:10.1145/3242969.3242985</a></li>
          <li>Hochreiter, S. &amp; Schmidhuber, J. (1997). <em>Long Short-Term Memory.</em>
          Neural Computation, 9(8), 1735–1780.</li>
          <li>Bahdanau, D., Cho, K., &amp; Bengio, Y. (2015). <em>Neural Machine Translation by Jointly
          Learning to Align and Translate.</em> ICLR 2015.</li>
          <li>Cornet, V.P. &amp; Holden, R.J. (2018). <em>Systematic review of smartphone-based passive
          sensing for health and wellbeing.</em> Journal of Biomedical Informatics, 77, 120–132.</li>
          <li>Torous, J. et al. (2021). <em>Digital phenotyping and mobile sensing.</em>
          World Psychiatry, 20(2), 189–196.</li>
          <li>Cho, K. et al. (2014). <em>Learning Phrase Representations using RNN Encoder-Decoder
          for Statistical Machine Translation.</em> EMNLP 2014.</li>
        </ol>
        """,
        "#6b7280",
    )

    # ── Final disclaimer ─────────────────────────────────────
    st.markdown(
        """
        <div class='disclaimer-box' style='margin-top:2rem'>
          <div class='disclaimer-title'>⚠ Research Prototype Disclaimer</div>
          <div class='disclaimer-text'>
            PanicGuard AI is an academic research prototype developed exclusively for a master's thesis.
            It has <strong>NOT</strong> been clinically validated, certified as a medical device,
            or approved for use in patient care. The ROC-AUC of 0.49 indicates near-chance
            discriminative performance under LOSO cross-validation.
            <br><br>
            This system is intended to demonstrate the technical feasibility of wearable-based
            panic risk monitoring as a proof-of-concept platform. Any real-world application
            would require substantial further research, ethical review, regulatory approval,
            and clinical validation.
            <br><br>
            <strong>If you or someone you know is experiencing a mental health crisis, please contact
            a qualified healthcare professional or emergency services immediately.</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footer'>🫀 PanicGuard AI · Master Thesis Research Prototype · 2024 · "
        "AI &amp; Biomedical Signal Processing · Not for clinical use</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
