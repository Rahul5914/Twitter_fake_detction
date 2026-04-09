"""
utils.py
--------
Shared helper functions: model loading, label mapping, plotting helpers.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── label maps ──────────────────────────────────────────────────────────────
LABEL_MAP   = {0: "FAKE", 1: "TRUE", 2: "NEUTRAL"}
LABEL_COLOR = {"FAKE": "#EF4444", "TRUE": "#22C55E", "NEUTRAL": "#F59E0B"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


# ── persistence ─────────────────────────────────────────────────────────────

def save_model(obj, filename: str):
    """Pickle an object into the model/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def load_model(filename: str):
    """Load a pickled object from the model/ directory."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `python src/train_fake_model.py` and "
            "`python src/train_viral_model.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── plotting helpers ─────────────────────────────────────────────────────────

def plot_fake_confidence(proba: np.ndarray) -> plt.Figure:
    """
    Horizontal bar chart showing confidence for each class.
    proba: 1-D array of length 3  [fake_prob, true_prob, neutral_prob]
    """
    labels = ["FAKE", "TRUE", "NEUTRAL"]
    colors = [LABEL_COLOR[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    bars = ax.barh(labels, proba * 100, color=colors, height=0.5, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#94A3B8", fontsize=9)
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color("#94A3B8")
    ax.tick_params(axis="x", colors="#94A3B8")
    ax.tick_params(axis="y", colors="#CBD5E1")

    for bar, val in zip(bars, proba * 100):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="#F1F5F9", fontsize=9)

    fig.tight_layout(pad=0.6)
    return fig


def plot_viral_gauge(probability: float) -> plt.Figure:
    """
    Semi-circle gauge showing viral probability.
    probability: float in [0, 1]
    """
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color="#1E293B", solid_capstyle="round")

    # Filled arc
    fill_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    color = "#22C55E" if probability < 0.5 else "#F59E0B" if probability < 0.75 else "#EF4444"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=color, solid_capstyle="round")

    # Needle
    angle = np.pi - probability * np.pi
    ax.annotate("", xy=(0.68 * np.cos(angle), 0.68 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#F1F5F9", lw=1.5))

    ax.text(0, -0.22, f"{probability * 100:.1f}%",
            ha="center", va="center", fontsize=22,
            color=color, fontweight="bold")
    ax.text(0, -0.48, "Viral Probability",
            ha="center", va="center", fontsize=9, color="#94A3B8")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.6, 1.05)
    fig.tight_layout(pad=0.3)
    return fig


def plot_feature_importance(names: list, importances: np.ndarray,
                            title: str = "Feature Importance") -> plt.Figure:
    """Horizontal bar chart for feature importance."""
    idx    = np.argsort(importances)
    names  = [names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(names) * 0.45)))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names, values, color=colors, edgecolor="none", height=0.55)

    ax.set_title(title, color="#F1F5F9", fontsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#CBD5E1", labelsize=8)
    ax.xaxis.set_tick_params(color="#475569")
    ax.set_xlabel("Importance", color="#94A3B8", fontsize=8)

    fig.tight_layout(pad=0.8)
    return fig


def plot_impact_radar(scores: dict) -> plt.Figure:
    """
    Pentagon radar chart for the 5 impact sub-scores.
    """
    keys   = ["social_reputation", "engagement", "topic_engagement",
              "likability", "credibility"]
    labels = ["Social\nRep.", "Engagement", "Topic\nEng.", "Likability", "Credibility"]
    vals   = [scores[k] for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals  += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    ax.plot(angles, vals, color="#38BDF8", lw=2)
    ax.fill(angles, vals, color="#38BDF8", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#CBD5E1", fontsize=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_tick_params(labelcolor="#475569", labelsize=7)
    ax.grid(color="#1E293B", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#1E293B")

    fig.tight_layout(pad=0.5)
    return fig
