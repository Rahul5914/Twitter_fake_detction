"""
train_viral_model.py
--------------------
Trains the viral-tweet prediction model.

Two models:
  1. Logistic Regression  → viral_model.pkl  (baseline)
  2. XGBoost              → viral_model_xgb.pkl (improved)

Features (matching paper):
  - number_of_followers
  - retweets
  - likes
  - account_verified  (0/1)
  - tweet_age         (hours since posted)

Run:  python src/train_viral_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import classification_report, roc_auc_score
from src.utils               import save_model

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ── Synthetic dataset ─────────────────────────────────────────────────────────

def generate_viral_dataset(n: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Realistic synthetic dataset.  Viral = retweets > 500 OR likes > 2000.

    Correlations baked in:
      • verified accounts → more followers, more RTs
      • more followers    → more likes
      • recent tweets     → fewer engagements so far
    """
    rng = np.random.default_rng(random_state)

    verified        = rng.binomial(1, 0.08, n)                          # 8% verified
    followers_base  = rng.lognormal(7.0, 1.8, n).astype(int)            # median ~1k
    followers       = (followers_base * (1 + verified * 4)).astype(int)

    # Engagement driven by follower count + verified status
    rt_base  = rng.exponential(50, n) * (followers / 5000 + 0.1) * (1 + verified * 5)
    lk_base  = rng.exponential(80, n) * (followers / 3000 + 0.15) * (1 + verified * 3)

    retweets = np.clip(rng.poisson(rt_base), 0, 50_000).astype(int)
    likes    = np.clip(rng.poisson(lk_base), 0, 200_000).astype(int)
    tweet_age = rng.uniform(0, 168, n)                                   # up to 7 days

    # Viral threshold (paper-inspired)
    viral = ((retweets > 500) | (likes > 2000)).astype(int)

    df = pd.DataFrame({
        "number_of_followers": followers,
        "retweets":            retweets,
        "likes":               likes,
        "account_verified":    verified,
        "tweet_age":           tweet_age.round(2),
        "viral":               viral,
    })

    viral_pct = viral.mean() * 100
    print(f"  Viral dataset: {n} samples  |  "
          f"viral={viral.sum()} ({viral_pct:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed and interaction features."""
    df = df.copy()
    df["log_followers"] = np.log1p(df["number_of_followers"])
    df["log_retweets"]  = np.log1p(df["retweets"])
    df["log_likes"]     = np.log1p(df["likes"])
    df["engagement_rt_lk"] = df["retweets"] + df["likes"]
    df["log_engagement"]   = np.log1p(df["engagement_rt_lk"])
    return df


FEATURE_COLS = [
    "log_followers", "log_retweets", "log_likes",
    "account_verified", "tweet_age", "log_engagement",
]

FEATURE_DISPLAY_NAMES = [
    "Log(Followers)", "Log(Retweets)", "Log(Likes)",
    "Account Verified", "Tweet Age (hrs)", "Log(Engagement)",
]


def train():
    print("\n=== VIRAL PREDICTION MODEL TRAINING ===\n")

    df  = generate_viral_dataset()
    df  = engineer_features(df)
    X   = df[FEATURE_COLS]
    y   = df["viral"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── 1. Logistic Regression baseline ─────────────────────────────────────
    print("Training Logistic Regression (baseline)…")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred   = lr_pipe.predict(X_test)
    y_prob   = lr_pipe.predict_proba(X_test)[:, 1]
    print("\n[Logistic Regression] Test Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Viral","Viral"]))
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

    cv = cross_val_score(lr_pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  5-fold CV AUC: {cv.mean():.3f} ± {cv.std():.3f}")

    coef = lr_pipe.named_steps["clf"].coef_[0]
    artifact = {
        "pipeline":              lr_pipe,
        "feature_cols":          FEATURE_COLS,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "importances":           np.abs(coef).tolist(),
        "model_type":            "LogisticRegression",
    }
    save_model(artifact, "viral_model.pkl")

    # ── 2. XGBoost improved ─────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost (improved)…")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
        print("\n[XGBoost] Test Report:")
        print(classification_report(y_test, y_pred_xgb,
                                     target_names=["Not Viral","Viral"]))
        print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.3f}")

        artifact_xgb = {
            "pipeline":              xgb,
            "feature_cols":          FEATURE_COLS,
            "feature_display_names": FEATURE_DISPLAY_NAMES,
            "importances":           xgb.feature_importances_.tolist(),
            "model_type":            "XGBoost",
            "scaler":                lr_pipe.named_steps["scaler"],  # reuse scaler
        }
        save_model(artifact_xgb, "viral_model_xgb.pkl")
    else:
        print("\nSkipped XGBoost (not installed).")

    print("\n✅ Viral-prediction models saved to model/")


if __name__ == "__main__":
    train()
