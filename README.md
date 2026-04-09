# 🔬 TweetScope — Fake Tweet Detection & Viral Prediction

> **Research-backed ML system** inspired by *"Analyzing Fake Content on Twitter"*
> (Boston Marathon Dataset)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

TweetScope is a **production-ready machine learning application** that:

1. **Detects whether a tweet is FAKE / TRUE / NEUTRAL** using NLP + ML
2. **Predicts whether a tweet will go viral** in the next time window
3. **Calculates an Impact Score** using a multi-factor formula from the paper
4. **Visualises results** with confidence charts, gauges, and radar plots

---

## 🏗️ Architecture

```
User Input (Tweet + Metrics)
        │
        ▼
┌────────────────────────────────────────────────┐
│              Streamlit App  (app.py)           │
└────────┬──────────────────────┬───────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  Fake Detection │   │  Viral Prediction    │
│  TF-IDF + LR   │   │  Feature Eng. + LR  │
│  (+ XGBoost)   │   │  (+ XGBoost)        │
└────────┬────────┘   └──────────┬───────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌─────────────────────┐
         │   Impact Score      │
         │  (5-factor formula) │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Visualisations    │
         │ Confidence / Gauge  │
         │ Radar / Feature Imp │
         └─────────────────────┘
```

---

## 📁 Project Structure

```
project/
├── app.py                      ← Streamlit UI (main entry point)
├── requirements.txt            ← Python dependencies
├── README.md
│
├── model/                      ← Saved model files (auto-generated)
│   ├── fake_model.pkl          ← TF-IDF + Logistic Regression
│   ├── fake_model_xgb.pkl      ← TF-IDF + XGBoost (optional)
│   ├── viral_model.pkl         ← LR viral predictor
│   └── viral_model_xgb.pkl     ← XGBoost viral predictor (optional)
│
├── src/
│   ├── preprocess.py           ← Text cleaning, tokenisation, impact score
│   ├── train_fake_model.py     ← Train fake-detection model
│   ├── train_viral_model.py    ← Train viral prediction model
│   └── utils.py                ← Model I/O, plotting helpers
│
└── data/                       ← Place real datasets here (CSV)
    └── .gitkeep
```

---

## 🧠 Models

### Fake Tweet Detection
| Model | Description |
|-------|-------------|
| TF-IDF + Logistic Regression | Baseline; fast, interpretable |
| TF-IDF + XGBoost | Advanced; higher accuracy |

- **Input**: raw tweet text
- **Output**: FAKE / TRUE / NEUTRAL + confidence scores
- **Preprocessing**: lowercase → remove URLs/mentions → remove stopwords → TF-IDF (1–2 grams)

### Viral Prediction
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline (as in paper) |
| XGBoost | Improved; handles non-linearity |

- **Features**: log(followers), log(retweets), log(likes), verified, tweet_age, log(engagement)
- **Output**: viral probability [0–1]

### Impact Score Formula
```
Impact = 0.20 × SocialReputation
       + 0.25 × Engagement
       + 0.20 × TopicEngagement
       + 0.15 × Likability
       + 0.20 × Credibility
```
All sub-scores are normalised to [0, 100].

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/tweetscope.git
cd tweetscope
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train models (optional — app auto-trains on first run)
```bash
python src/train_fake_model.py
python src/train_viral_model.py
```

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

> ✅ Models are trained automatically on first deploy — no manual step needed.

---

## 📦 Upload to GitHub

```bash
# Inside the project folder
git init
git add .
git commit -m "Initial commit: TweetScope ML system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tweetscope.git
git push -u origin main
```

---

## 🖼️ Screenshots

> *(Replace with actual screenshots after running the app)*

| Section | Description |
|---------|-------------|
| **Main Input** | Tweet text area + sidebar metrics |
| **Verdict Banner** | FAKE/TRUE/NEUTRAL with colour coding |
| **Confidence Chart** | Horizontal bars for each class |
| **Viral Gauge** | Semi-circle needle gauge |
| **Impact Radar** | Pentagon radar of 5 sub-scores |
| **Feature Importance** | Top TF-IDF terms / XGBoost features |

---

## 🤖 Example Output

**Input tweet**: `"Breaking: explosion at marathon finish line kills hundreds!!"`

| Metric | Value |
|--------|-------|
| Verdict | 🚨 FAKE |
| Confidence | 87.3% |
| Viral Probability | 62.1% |
| Impact Score | 41.2 / 100 |
| Social Reputation | 28.4 |
| Credibility | 0.0 |

---

## 📚 References

- *Analyzing Fake Content on Twitter* — Boston Marathon Dataset paper
- Zubiaga, A. et al. (2016). *Analysing How People Orient to and Spread Rumours in Social Media*
- scikit-learn documentation: https://scikit-learn.org

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Streamlit, scikit-learn, XGBoost, and Matplotlib*
