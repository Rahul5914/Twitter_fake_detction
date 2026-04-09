"""
train_fake_model.py
-------------------
Trains the fake-tweet detection model.

Two models are trained and saved:
  1. TF-IDF + Logistic Regression  → fake_model.pkl  (baseline, fast)
  2. TF-IDF + XGBoost              → fake_model_xgb.pkl (advanced)

Dataset: Synthetic dataset inspired by the Boston Marathon paper
         (also works with any CSV that has 'text' and 'label' columns).

Run:  python src/train_fake_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import classification_report, confusion_matrix
from src.preprocess          import preprocess_tweet
from src.utils               import save_model

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed – skipping advanced model.")

# ── Synthetic dataset ────────────────────────────────────────────────────────
FAKE_TWEETS = [
    "Breaking: explosion at marathon finish line kills hundreds!!",
    "FAKE NEWS: government hiding real death toll from bombing",
    "Secret source confirms Boston attack was inside job",
    "ALERT: second bomb found near Fenway, media covering up",
    "Shocking truth about marathon bomber revealed SHARE NOW",
    "Deep state orchestrated Boston attack to grab guns",
    "Witness says FBI knew about bombing 3 days in advance",
    "Crisis actors used at Boston Marathon false flag PROOF",
    "Unnamed official confirms explosives were military grade",
    "Social media censoring real photos from bombing scene",
    "Marathon death toll being hidden by mainstream media",
    "Anonymous hacker leaks proof Boston was staged event",
    "MUST WATCH video proves bombing was planned weeks ahead",
    "Bomber was on FBI watchlist but they let him go on purpose",
    "Government testing new crowd control weapon at marathon",
    "Breaking news authorities covering up second suspect",
    "Doctor claims real injuries far worse than reported",
    "Exclusive photos prove media lying about explosion size",
    "Sources say Tsarnaev brothers were CIA informants",
    "Marathon attack was distraction from secret bill passing",
    "URGENT police scanner audio reveals hidden second attack",
    "Eyewitness account suppressed by mainstream news outlets",
    "Bomb squad drills ran same day as attack coincidence",
    "Runners told to expect controlled explosion beforehand",
    "Financial records show suspicious trades before bombing",
    "Unexplained military presence spotted days before attack",
    "Official story falling apart new evidence emerges today",
    "Family of victims silenced by government gag order",
    "Pressure cooker bombs bought with government money proof",
    "Real perpetrators never caught FBI protecting someone",
]

TRUE_TWEETS = [
    "Boston Marathon bombings: two explosions near finish line",
    "Authorities confirm two bombs detonated at marathon",
    "President Obama addresses nation after Boston attack",
    "FBI releases photos of two suspects in marathon bombing",
    "Boston strong community rallies after tragic bombing",
    "Suspect Dzhokhar Tsarnaev apprehended after manhunt",
    "Marathon bombing suspect Tamerlan Tsarnaev killed in shootout",
    "Victims of Boston Marathon bombing remembered one year on",
    "Trial of Boston bomber Tsarnaev begins in federal court",
    "Security increased at marathons nationwide after Boston",
    "Survivors of marathon bombing speak about their recovery",
    "Watertown neighborhood lockdown lifted after suspect caught",
    "Boston Police confirm suspect in custody after manhunt",
    "Memorial held for victims of marathon bombing attack",
    "Governor declares state of emergency following bombing",
    "Hospitals report treating over 100 victims from bombing",
    "Investigators examine pressure cooker fragments as evidence",
    "Surveillance footage used to identify marathon suspects",
    "Boston community mourns loss of bombing victims",
    "Red Cross sets up relief fund for bombing survivors",
    "Three killed and hundreds injured in marathon bombings",
    "Sentence handed to Boston marathon bomber Tsarnaev",
    "Boston Marathon resumes year after deadly bombing attack",
    "Witnesses describe chaos immediately after explosions",
    "First responders praised for quick action at bombing scene",
    "Investigators work to determine motive behind bombings",
    "National Guard deployed to support Boston Police response",
    "Bomber motivated by extremist ideology court hears",
    "Victim of bombing completes marathon on prosthetic leg",
    "Annual memorial run held in honour of bombing victims",
]

NEUTRAL_TWEETS = [
    "Running the Boston Marathon this year fingers crossed",
    "Beautiful day for a marathon good luck to all runners",
    "Training for Boston Marathon anyone have tips",
    "Boston is such an amazing city love visiting every year",
    "Just registered for my first marathon so excited",
    "Marathon season is here stay safe out there everyone",
    "Watching the Boston Marathon from home on TV",
    "Boston weather looks great for marathon day this year",
    "Cheering on friends running in the marathon today",
    "The marathon route goes past my office every year",
    "Good luck to everyone running in Boston this weekend",
    "Boston Marathon has such incredible history and tradition",
    "Getting ready to watch the marathon broadcast later",
    "My friend just finished her first marathon congrats",
    "What time does the Boston Marathon start today",
    "Huge crowds gathering on Boylston Street for the race",
    "Elite runners expected to break records this year",
    "Marathon volunteers doing amazing work today respect",
    "Boston in April means marathon and I love it",
    "Watching elite athletes push limits at Boston today",
    "Historic race with thousands of participants each year",
    "Weather conditions perfect for fast times today",
    "Boston Marathon is one of the most prestigious races",
    "Runners from over 100 countries competing today",
    "Incredible atmosphere at Copley Square during marathon",
    "How long is a full marathon someone help me out",
    "Inspirational stories from Boston Marathon runners today",
    "First-time marathoner in Boston this year wish me luck",
    "Boston strong always loved this city and this race",
    "Best running event in the world hands down Boston",
]

def build_synthetic_dataset(n_augment: int = 6) -> pd.DataFrame:
    """
    Create a balanced synthetic dataset with light augmentation
    to reach ~500 samples per class.
    """
    rng = np.random.default_rng(42)
    records = []

    def augment(tweet: str, label: int) -> list:
        rows = [{"text": tweet, "label": label}]
        words = tweet.split()
        for _ in range(n_augment):
            # random word drop
            drop = rng.choice([True, False], size=len(words), p=[0.15, 0.85])
            aug  = " ".join(w for w, d in zip(words, drop) if not d) or tweet
            rows.append({"text": aug, "label": label})
        return rows

    for t in FAKE_TWEETS:
        records.extend(augment(t, 0))
    for t in TRUE_TWEETS:
        records.extend(augment(t, 1))
    for t in NEUTRAL_TWEETS:
        records.extend(augment(t, 2))

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Dataset: {len(df)} samples  |  "
          f"FAKE={sum(df.label==0)}  TRUE={sum(df.label==1)}  NEUTRAL={sum(df.label==2)}")
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    print("\n=== FAKE TWEET DETECTION MODEL TRAINING ===\n")

    df = build_synthetic_dataset()

    # Preprocess
    df["clean"] = df["text"].apply(preprocess_tweet)
    X, y = df["clean"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ── 1. Logistic Regression baseline ─────────────────────────────────────
    print("Training Logistic Regression (baseline)…")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred = lr_pipe.predict(X_test)
    print("\n[Logistic Regression] Test Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["FAKE","TRUE","NEUTRAL"]))

    cv = cross_val_score(lr_pipe, X, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    # Save top-level feature importances for visualisation
    vectorizer   = lr_pipe.named_steps["tfidf"]
    classifier   = lr_pipe.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    # Average absolute coefficient across classes
    avg_coef = np.mean(np.abs(classifier.coef_), axis=0)
    top_idx  = np.argsort(avg_coef)[-15:]

    artifact = {
        "pipeline":       lr_pipe,
        "feature_names":  feature_names[top_idx].tolist(),
        "importances":    avg_coef[top_idx].tolist(),
        "label_map":      {0: "FAKE", 1: "TRUE", 2: "NEUTRAL"},
        "model_type":     "LogisticRegression",
    }
    save_model(artifact, "fake_model.pkl")

    # ── 2. XGBoost advanced ─────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost (advanced)…")
        xgb_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
            )),
        ])
        xgb_pipe.fit(X_train, y_train)
        y_pred_xgb = xgb_pipe.predict(X_test)
        print("\n[XGBoost] Test Report:")
        print(classification_report(y_test, y_pred_xgb,
                                     target_names=["FAKE","TRUE","NEUTRAL"]))
        save_model({"pipeline": xgb_pipe, "model_type": "XGBoost",
                    "label_map": {0: "FAKE", 1: "TRUE", 2: "NEUTRAL"}},
                   "fake_model_xgb.pkl")
    else:
        print("\nSkipped XGBoost (not installed).")

    print("\n✅ Fake-detection models saved to model/")


if __name__ == "__main__":
    train()
