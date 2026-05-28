"""
visualization_utils.py
Complete Plotly visualization functions for PanicGuard AI.
All charts use dark medical theme (template='plotly_dark').
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────

_DARK_BG   = "#0d1117"
_CARD_BG   = "#1e2130"
_BORDER    = "#2d3748"
_TEXT      = "#e8eaf0"
_MUTED     = "#6b7280"
_BLUE      = "#4F8BF9"
_PURPLE    = "#7C5CBF"
_GREEN     = "#2ecc71"
_ORANGE    = "#f39c12"
_RED       = "#e74c3c"
_CYAN      = "#1abc9c"

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_CARD_BG,
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial", color=_TEXT, size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
    yaxis=dict(gridcolor=_BORDER, zerolinecolor=_BORDER),
)


def _apply_base(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, **_BASE_LAYOUT)
    return fig


# ─────────────────────────────────────────────
# 1. Risk Gauge
# ─────────────────────────────────────────────

def plot_risk_gauge(score: float, title: str = "Panic Risk Score") -> go.Figure:
    """
    Plotly gauge chart showing risk probability 0–100%.

    Parameters
    ----------
    score : float in [0, 1]
    title : chart title string

    Returns
    -------
    go.Figure
    """
    pct = score * 100
    if score < 0.30:
        bar_color = _GREEN
        status = "Normal"
    elif score < 0.60:
        bar_color = _ORANGE
        status = "Elevated"
    else:
        bar_color = _RED
        status = "Critical"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color=bar_color)),
        title=dict(text=f"<b>{title}</b><br><span style='font-size:14px;color:{bar_color}'>{status} Risk</span>",
                   font=dict(size=16)),
        delta=dict(reference=30, increasing=dict(color=_RED), decreasing=dict(color=_GREEN)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=_MUTED,
                tickfont=dict(size=11),
            ),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor=_CARD_BG,
            borderwidth=1,
            bordercolor=_BORDER,
            steps=[
                dict(range=[0,  30], color="rgba(46,204,113,0.12)"),
                dict(range=[30, 60], color="rgba(243,156,18,0.12)"),
                dict(range=[60,100], color="rgba(231,76,60,0.12)"),
            ],
            threshold=dict(
                line=dict(color=_RED, width=3),
                thickness=0.8,
                value=60,
            ),
        ),
    ))

    fig.update_layout(
        height=340,
        paper_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─────────────────────────────────────────────
# 2. Physiological Trends
# ─────────────────────────────────────────────

def plot_physiological_trends(df: pd.DataFrame, features_to_plot: list = None) -> go.Figure:
    """
    Multi-panel time-series chart for physiological signals.

    Default panels: heart_rate, stress_score, sleep_duration, activity_level, hrv
    """
    if features_to_plot is None:
        features_to_plot = ["heart_rate", "stress_score", "sleep_duration", "activity_level", "hrv"]

    # Filter to only features that exist in df
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    if not features_to_plot:
        fig = go.Figure()
        fig.add_annotation(text="No matching features found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_MUTED, size=14))
        return _apply_base(fig)

    n_panels = len(features_to_plot)
    colours = [_BLUE, _RED, _CYAN, _GREEN, _PURPLE, _ORANGE, "#ff6b6b", "#a29bfe"]
    labels = {
        "heart_rate":     "Heart Rate (bpm)",
        "resting_heart_rate": "Resting HR (bpm)",
        "hrv":            "HRV (ms)",
        "spo2":           "SpO₂ (%)",
        "sleep_duration": "Sleep Duration (h)",
        "sleep_quality":  "Sleep Quality",
        "stress_score":   "Stress Score",
        "activity_level": "Activity Level",
        "steps":          "Steps",
        "fatigue_level":  "Fatigue Level",
        "anxiety_proxy":  "Anxiety Proxy",
        "cortisol_proxy": "Cortisol Proxy",
    }

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[labels.get(f, f.replace("_", " ").title()) for f in features_to_plot],
    )

    x_vals = df["date"].tolist() if "date" in df.columns else list(range(len(df)))

    for i, feat in enumerate(features_to_plot):
        col = colours[i % len(colours)]
        y = df[feat].values
        row = i + 1

        # Shaded area
        fig.add_trace(
            go.Scatter(x=x_vals + x_vals[::-1],
                       y=np.concatenate([y, np.full(len(y), float(np.min(y)))]),
                       fill="toself",
                       fillcolor=col.replace("#", "rgba(") + ",0.08)" if col.startswith("#") else col,
                       line=dict(width=0),
                       showlegend=False,
                       hoverinfo="skip"),
            row=row, col=1,
        )

        # Main line
        fig.add_trace(
            go.Scatter(x=x_vals, y=y,
                       mode="lines+markers",
                       name=labels.get(feat, feat),
                       line=dict(color=col, width=2),
                       marker=dict(size=5, color=col),
                       hovertemplate=f"<b>{labels.get(feat, feat)}</b>: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>"),
            row=row, col=1,
        )

        # Detect and annotate anomalies (values > mean + 1.5*std)
        mean_v = float(np.mean(y))
        std_v  = float(np.std(y))
        anomaly_thresh = mean_v + 1.5 * std_v
        anomaly_idx = np.where(y > anomaly_thresh)[0]
        for idx in anomaly_idx[:3]:  # max 3 annotations per panel
            fig.add_annotation(
                x=x_vals[idx], y=float(y[idx]),
                text="⚠", showarrow=True,
                arrowhead=2, arrowcolor=_ORANGE,
                font=dict(color=_ORANGE, size=12),
                row=row, col=1,
            )

    fig.update_layout(
        height=max(250 * n_panels, 400),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=30),
    )
    for i in range(1, n_panels + 1):
        fig.update_xaxes(gridcolor=_BORDER, row=i, col=1)
        fig.update_yaxes(gridcolor=_BORDER, row=i, col=1)

    return fig


# ─────────────────────────────────────────────
# 3. Attention Heatmap
# ─────────────────────────────────────────────

def plot_attention_heatmap(
    temporal_weights: np.ndarray,
    feature_weights: np.ndarray,
    feature_names: list,
) -> go.Figure:
    """
    Two-panel figure: temporal attention bar chart (top) +
    feature attention bar chart (bottom).
    """
    day_labels = [f"Day -{6-i}" if i < 6 else "Today" for i in range(len(temporal_weights))]
    feat_labels = [f.replace("_", " ").title() for f in feature_names]

    t_colors = [
        f"rgba(79,139,249,{0.3 + 0.7 * float(w) / float(np.max(temporal_weights) + 1e-8)})"
        for w in temporal_weights
    ]
    f_colors = [
        f"rgba(124,92,191,{0.3 + 0.7 * float(w) / float(np.max(feature_weights) + 1e-8)})"
        for w in feature_weights
    ]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Temporal Attention (7-Day Window)", "Feature Attention (24 Signals)"],
        vertical_spacing=0.18,
        row_heights=[0.35, 0.65],
    )

    fig.add_trace(
        go.Bar(
            x=day_labels, y=temporal_weights.tolist(),
            marker_color=t_colors,
            name="Temporal",
            hovertemplate="<b>%{x}</b><br>Attention: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=feat_labels, y=feature_weights.tolist(),
            marker_color=f_colors,
            name="Feature",
            hovertemplate="<b>%{x}</b><br>Attention: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=560,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        margin=dict(l=50, r=20, t=50, b=100),
    )
    fig.update_xaxes(tickangle=-45, row=2, col=1, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=_BORDER)

    return fig


# ─────────────────────────────────────────────
# 4. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(importance_dict: dict, feature_names: list = None) -> go.Figure:
    """
    Horizontal bar chart of feature importance scores, sorted descending.
    Top 10 features are highlighted.
    """
    if feature_names:
        items = [(f, importance_dict.get(f, 0.0)) for f in feature_names if f in importance_dict]
    else:
        items = list(importance_dict.items())

    items.sort(key=lambda x: x[1])  # ascending for horizontal bar (plotly bottom→top)
    features = [i[0].replace("_", " ").title() for i in items]
    values   = [i[1] for i in items]
    n        = len(values)
    top10_cutoff = sorted(values)[-10] if n >= 10 else 0.0

    colors = [
        _RED if v >= top10_cutoff else _BLUE
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=_BORDER, width=0.5),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=top10_cutoff, line_dash="dash",
                  line_color=_ORANGE, opacity=0.6,
                  annotation_text="Top-10 threshold",
                  annotation_font_color=_ORANGE)

    fig.update_layout(
        title="Feature Importance (Demo — BiLSTM Attention Proxy)",
        height=max(400, 22 * n),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        xaxis=dict(title="Importance Score", gridcolor=_BORDER, range=[0, 1.05]),
        yaxis=dict(gridcolor=_BORDER, tickfont=dict(size=11)),
        margin=dict(l=180, r=30, t=50, b=40),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# 5. Risk History
# ─────────────────────────────────────────────

def plot_risk_history(history_df: pd.DataFrame) -> go.Figure:
    """
    Time-series of daily risk scores with color-coded background zones
    and alert markers.

    Expected columns: date, risk_score, [alert] (bool optional)
    """
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No history available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_MUTED))
        return _apply_base(fig, 300)

    x = history_df["date"].tolist()
    y = (history_df["risk_score"] * 100).tolist()

    fig = go.Figure()

    # Background zones
    x_range = [x[0], x[-1]]
    for lo, hi, col, name in [
        (0,  30, "rgba(46,204,113,0.07)",  "Normal"),
        (30, 60, "rgba(243,156,18,0.07)",  "Elevated"),
        (60, 100,"rgba(231,76,60,0.07)",   "Critical"),
    ]:
        fig.add_hrect(y0=lo, y1=hi,
                      fillcolor=col, line_width=0,
                      annotation_text=name,
                      annotation_position="right",
                      annotation_font=dict(size=9, color=_MUTED))

    # Area fill under curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill="tozeroy",
        fillcolor="rgba(79,139,249,0.1)",
        line=dict(color=_BLUE, width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color=[
            _RED if v >= 60 else (_ORANGE if v >= 30 else _GREEN) for v in y
        ]),
        name="Risk Score",
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
    ))

    # Alert markers
    if "alert" in history_df.columns:
        alert_x = [x[i] for i, a in enumerate(history_df["alert"].tolist()) if a]
        alert_y = [y[i] for i, a in enumerate(history_df["alert"].tolist()) if a]
        if alert_x:
            fig.add_trace(go.Scatter(
                x=alert_x, y=alert_y,
                mode="markers",
                marker=dict(symbol="star", size=14, color=_RED,
                            line=dict(color="white", width=1)),
                name="Alert",
                hovertemplate="<b>Alert</b><br>%{x}<br>Risk: %{y:.1f}%<extra></extra>",
            ))

    # Threshold line
    fig.add_hline(y=60, line_dash="dash", line_color=_RED, opacity=0.7,
                  annotation_text="Critical Threshold (60%)",
                  annotation_font_color=_RED, annotation_font_size=10)

    fig.update_layout(
        title="Risk Score Timeline",
        height=380,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        xaxis=dict(title="Date", gridcolor=_BORDER),
        yaxis=dict(title="Risk Score (%)", gridcolor=_BORDER, range=[0, 105]),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=_BORDER, borderwidth=1),
        margin=dict(l=55, r=50, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# 6. Signal Correlation
# ─────────────────────────────────────────────

def plot_signal_correlation(df: pd.DataFrame, features: list = None) -> go.Figure:
    """
    Correlation matrix heatmap with clinical colour scale.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    features = [f for f in features if f in df.columns]
    if len(features) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric features", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_MUTED))
        return _apply_base(fig)

    corr = df[features].corr()
    labels = [f.replace("_", " ").title() for f in features]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[
            [0.0,  "#2166ac"],
            [0.25, "#74add1"],
            [0.5,  _CARD_BG],
            [0.75, "#f4a582"],
            [1.0,  "#d6604d"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(title="r", tickfont=dict(color=_TEXT), titlefont=dict(color=_TEXT)),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        title="Physiological Signal Correlations",
        height=max(500, 22 * len(features)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        margin=dict(l=150, r=50, t=50, b=150),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────
# 7. Timeline Events
# ─────────────────────────────────────────────

def plot_timeline_events(events_df: pd.DataFrame) -> go.Figure:
    """
    Scatter-based timeline of clinical events (alerts, risk levels, notes).

    Expected columns: date, event_type, risk_score, description (optional)
    """
    if events_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No events to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_MUTED))
        return _apply_base(fig, 280)

    color_map = {"critical": _RED, "elevated": _ORANGE, "normal": _GREEN, "alert": _RED}

    fig = go.Figure()

    for event_type in events_df["event_type"].unique():
        subset = events_df[events_df["event_type"] == event_type]
        col = color_map.get(event_type.lower(), _BLUE)
        hover_text = subset.get("description", subset["date"].astype(str)) if "description" in subset.columns else subset["date"].astype(str)

        fig.add_trace(go.Scatter(
            x=subset["date"].tolist(),
            y=[event_type] * len(subset),
            mode="markers+text",
            marker=dict(size=14, color=col, symbol="diamond",
                        line=dict(color="white", width=1)),
            name=event_type.title(),
            text=[f"{r:.0%}" for r in subset["risk_score"].tolist()] if "risk_score" in subset.columns else None,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Type: " + event_type + "<extra></extra>",
        ))

    fig.update_layout(
        title="Clinical Event Timeline",
        height=300,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT),
        xaxis=dict(title="Date", gridcolor=_BORDER),
        yaxis=dict(title="Event Type", gridcolor=_BORDER),
        margin=dict(l=80, r=30, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# 8. Radar Chart
# ─────────────────────────────────────────────

def create_radar_chart(feature_dict: dict, title: str = "Multi-Signal Overview") -> go.Figure:
    """
    Radar / spider chart of normalised signal values (0–100 scale).

    Parameters
    ----------
    feature_dict : {feature_name: normalised_value (0–100)}
    """
    if not feature_dict:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=_MUTED))
        return _apply_base(fig, 400)

    labels  = [k.replace("_", " ").title() for k in feature_dict]
    values  = list(feature_dict.values())
    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(79,139,249,0.15)",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=6, color=_BLUE),
        name="Current",
        hovertemplate="<b>%{theta}</b>: %{r:.1f}<extra></extra>",
    ))

    # Reference (healthy) polygon at 50
    ref = [50] * (len(labels) + 1)
    fig.add_trace(go.Scatterpolar(
        r=ref,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(46,204,113,0.06)",
        line=dict(color=_GREEN, width=1, dash="dash"),
        name="Healthy Baseline",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=title,
        height=460,
        paper_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        polar=dict(
            bgcolor=_CARD_BG,
            angularaxis=dict(color=_MUTED, gridcolor=_BORDER, tickfont=dict(size=10)),
            radialaxis=dict(visible=True, range=[0, 100], color=_MUTED,
                            gridcolor=_BORDER, tickfont=dict(size=9)),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=_BORDER, borderwidth=1,
                    font=dict(color=_TEXT)),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
