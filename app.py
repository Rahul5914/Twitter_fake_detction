"""
app.py  ·  TweetScope  ·  Boston Marathon Misinformation Analyser
─────────────────────────────────────────────────────────────────
A Twitter-style Streamlit dashboard that detects fake tweets and
predicts virality, with live feed, history log, and batch mode.

Run:  streamlit run app.py
"""

import os, sys, subprocess, datetime
import numpy as np
import pandas as pd
import streamlit as st

# ── page config MUST be first st call ────────────────────────────────────────
st.set_page_config(
    page_title="TweetScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "TweetScope · Boston Marathon Misinformation Analysis"},
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_tweet, compute_impact_score
from src.utils      import (load_model, LABEL_COLOR,
                             plot_fake_confidence, plot_viral_gauge,
                             plot_feature_importance, plot_impact_radar)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-TRAIN on first run
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_models():
    base       = os.path.dirname(os.path.abspath(__file__))
    model_dir  = os.path.join(base, "model")
    fake_path  = os.path.join(model_dir, "fake_model.pkl")
    viral_path = os.path.join(model_dir, "viral_model.pkl")
    if not os.path.exists(fake_path) or not os.path.exists(viral_path):
        with st.spinner("🤖 First launch — training models (~15 s)…"):
            for script in ("src/train_fake_model.py", "src/train_viral_model.py"):
                subprocess.run([sys.executable, script], cwd=base, check=True)
        st.toast("✅ Models trained!", icon="🎉")
        st.rerun()

_ensure_models()


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _fake_model(xgb: bool):
    try:
        return load_model("fake_model_xgb.pkl" if xgb else "fake_model.pkl")
    except FileNotFoundError:
        return load_model("fake_model.pkl")


@st.cache_resource(show_spinner=False)
def _viral_model(xgb: bool):
    try:
        return load_model("viral_model_xgb.pkl" if xgb else "viral_model.pkl")
    except FileNotFoundError:
        return load_model("viral_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = "Breaking: explosion at marathon finish line kills hundreds!!"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — Twitter-inspired dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: #050A14 !important;
    color: #E8EDF5 !important;
}
.block-container { padding: 1.2rem 2rem 2rem !important; max-width: 1280px !important; }
section[data-testid="stSidebar"] {
    background: #030710 !important;
    border-right: 1px solid #111827 !important;
}
section[data-testid="stSidebar"] * { font-family:'Sora',sans-serif !important; }
hr { border-color: #111827 !important; margin: .8rem 0 !important; }

/* ── header ── */
.ts-header { padding:.6rem 0 1.2rem; border-bottom:1px solid #111827; margin-bottom:1.4rem; }
.ts-logo   { font-size:1.9rem; font-weight:700; letter-spacing:-.03em; }
.ts-logo span { color:#1D9BF0; }
.ts-sub    { font-size:.78rem; color:#4B5563; letter-spacing:.04em; margin-top:.15rem; }

/* ── tweet card ── */
.tweet-card {
    background:#0D1526; border:1px solid #1B2A42; border-radius:16px;
    padding:1.2rem 1.4rem; margin-bottom:1rem; position:relative;
    transition: border-color .2s;
}
.tweet-card:hover { border-color:#1D9BF0; }
.tweet-card.fake    { border-left:3px solid #EF4444; }
.tweet-card.true    { border-left:3px solid #22C55E; }
.tweet-card.neutral { border-left:3px solid #F59E0B; }

.tweet-avatar {
    width:42px; height:42px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem; font-weight:700; flex-shrink:0;
    background:linear-gradient(135deg,#1D9BF0,#7C3AED);
}
.tweet-user   { font-weight:600; font-size:.92rem; }
.tweet-handle { font-size:.78rem; color:#4B5563; }
.tweet-text   { font-size:.95rem; line-height:1.55; color:#D1D9E6; margin:.6rem 0; }
.tweet-ts     { font-size:.72rem; color:#374151; }

/* ── badges ── */
.badge {
    display:inline-flex; align-items:center; gap:.3rem;
    border-radius:999px; padding:.18rem .75rem;
    font-size:.72rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
}
.badge-fake    { background:#1F0A0A; color:#EF4444; border:1px solid #7F1D1D; }
.badge-true    { background:#0A1F0E; color:#22C55E; border:1px solid #14532D; }
.badge-neutral { background:#1F170A; color:#F59E0B; border:1px solid #78350F; }
.badge-viral   { background:#0F0A1F; color:#A78BFA; border:1px solid #4C1D95; }
.badge-safe    { background:#0A1020; color:#60A5FA; border:1px solid #1E3A5F; }

/* ── stat pills ── */
.stat-row { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.5rem; }
.stat { background:#0A1020; border:1px solid #1B2A42; border-radius:8px; padding:.25rem .6rem; font-size:.72rem; color:#6B7280; }
.stat strong { color:#94A3B8; }

/* ── metric tiles ── */
.tile {
    background:linear-gradient(160deg,#0D1526,#0A1020);
    border:1px solid #1B2A42; border-radius:14px;
    padding:1rem 1.2rem; text-align:center;
}
.tile .t-label { font-size:.68rem; color:#4B5563; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.3rem; }
.tile .t-value { font-size:1.7rem; font-weight:700; line-height:1; }
.tile .t-sub   { font-size:.7rem; color:#374151; margin-top:.25rem; }

/* ── section title ── */
.sec-title {
    font-size:.72rem; font-weight:700; color:#1D9BF0;
    letter-spacing:.1em; text-transform:uppercase;
    margin:.8rem 0 .6rem; display:flex; align-items:center; gap:.4rem;
}
.sec-title::after { content:''; flex:1; height:1px; background:#111827; }

/* ── tabs ── */
div[data-testid="stTabs"] button { font-family:'Sora',sans-serif !important; font-size:.82rem !important; font-weight:600 !important; color:#4B5563 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#1D9BF0 !important; border-bottom-color:#1D9BF0 !important; }

/* ── buttons ── */
div.stButton > button {
    background:#1D9BF0 !important; color:#fff !important; border:none !important;
    border-radius:999px !important; padding:.45rem 1.6rem !important;
    font-weight:600 !important; font-size:.88rem !important;
}
div.stButton > button:hover { background:#1A8CD8 !important; }

/* ── inputs ── */
div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
    background:#050A14 !important; border:1px solid #1B2A42 !important;
    color:#E8EDF5 !important; border-radius:10px !important;
    font-family:'Sora',sans-serif !important;
}
div[data-testid="stTextArea"] textarea:focus, div[data-testid="stTextInput"] input:focus {
    border-color:#1D9BF0 !important; box-shadow:0 0 0 2px rgba(29,155,240,.15) !important;
}
div[data-testid="stNumberInput"] input {
    background:#050A14 !important; border:1px solid #1B2A42 !important;
    color:#E8EDF5 !important; border-radius:8px !important;
}
label[data-testid="stWidgetLabel"] p { color:#9CA3AF !important; font-size:.82rem !important; }

/* ── feed items ── */
.feed-item {
    display:flex; gap:.8rem; align-items:flex-start;
    background:#0D1526; border:1px solid #1B2A42; border-radius:12px;
    padding:.8rem 1rem; margin-bottom:.6rem;
}
.feed-snippet { font-size:.82rem; color:#94A3B8; flex:1; line-height:1.4; }
.feed-meta    { font-size:.7rem; color:#374151; margin-top:.25rem; }
.feed-badges  { display:flex; flex-direction:column; gap:.25rem; align-items:flex-end; flex-shrink:0; }

::-webkit-scrollbar { width:6px; } 
::-webkit-scrollbar-track { background:#050A14; }
::-webkit-scrollbar-thumb { background:#1B2A42; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ts-header">
  <div class="ts-logo">🔬 Tweet<span>Scope</span></div>
  <div class="ts-sub">BOSTON MARATHON · MISINFORMATION ANALYSIS SYSTEM · ML-POWERED</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    use_xgb  = st.toggle("XGBoost (advanced)", value=False)
    fake_art = _fake_model(use_xgb)
    viral_art= _viral_model(use_xgb)
    st.caption(f"Fake model: **{fake_art['model_type']}**")
    st.caption(f"Viral model: **{viral_art['model_type']}**")

    st.divider()
    st.markdown("### 📊 Account Metrics")
    followers = st.number_input("👥 Followers",   0, 10_000_000, 1500, 100)
    retweets  = st.number_input("🔁 Retweets",    0,    500_000,   45,   5)
    likes     = st.number_input("❤️  Likes",       0,  1_000_000,  120,  10)
    tweet_age = st.slider("⏱️ Tweet Age (hrs)", 0.0, 168.0, 2.0, 0.5)
    verified  = st.checkbox("✅ Verified Account", False)

    st.divider()
    st.markdown("### 🧪 Sample Tweets")
    samples = {
        "🚨 Fake — conspiracy":  "Secret source confirms Boston attack was inside job — FBI knew 3 days before",
        "✅ True — factual":     "FBI releases photos of two suspects in the Boston Marathon bombing",
        "🟡 Neutral — casual":   "Training for Boston Marathon this year fingers crossed wish me luck",
        "🔥 Viral candidate":    "Breaking: explosion at marathon finish line kills hundreds SHARE NOW",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state.prefill = text
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.toast("History cleared")

    st.caption("📚 *Analyzing Fake Content on Twitter* · Boston Marathon Dataset")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_analyse, tab_feed, tab_batch, tab_about = st.tabs(
    ["🔍 Analyse", "📡 Live Feed", "📋 Batch", "📘 About"])


# ═══════════════════════════════════════════════════════════
# TAB 1  ·  ANALYSE
# ═══════════════════════════════════════════════════════════
with tab_analyse:

    tweet_text = st.text_area(
        label="Tweet Text",
        height=100,
        value=st.session_state.prefill,
        placeholder="Paste a tweet here to analyse…",
        label_visibility="collapsed",
    )

    col_btn, col_char = st.columns([2, 5])
    with col_btn:
        analyse_btn = st.button("🔍 Analyse Tweet", use_container_width=False)
    with col_char:
        cc    = len(tweet_text)
        color = "#EF4444" if cc > 280 else "#374151"
        st.markdown(f'<p style="color:{color};font-size:.75rem;padding-top:.65rem;">'
                    f'{cc}/280 characters</p>', unsafe_allow_html=True)

    # ── run ───────────────────────────────────────────────────────────────────
    if analyse_btn and tweet_text.strip():
        from src.train_viral_model import engineer_features, FEATURE_COLS

        with st.spinner("Analysing…"):
            clean      = preprocess_tweet(tweet_text)
            fake_proba = fake_art["pipeline"].predict_proba([clean])[0]
            fake_idx   = int(np.argmax(fake_proba))
            fake_label = fake_art["label_map"][fake_idx]
            fake_conf  = float(fake_proba[fake_idx])

            raw  = pd.DataFrame([{"number_of_followers": followers,
                                   "retweets": retweets, "likes": likes,
                                   "account_verified": int(verified),
                                   "tweet_age": tweet_age}])
            feats     = engineer_features(raw)[FEATURE_COLS]
            viral_prob= float(viral_art["pipeline"].predict_proba(feats)[:, 1][0])

            cred_map  = {"TRUE": 1.0, "NEUTRAL": 0.5, "FAKE": 0.0}
            imp       = compute_impact_score(
                followers, retweets, likes, verified, cred_map[fake_label])

        record = dict(tweet=tweet_text[:120]+("…" if len(tweet_text)>120 else ""),
                      full_tweet=tweet_text, label=fake_label,
                      confidence=fake_conf, viral_prob=viral_prob,
                      impact=imp["total"], proba=fake_proba,
                      imp_scores=imp, followers=followers,
                      retweets=retweets, likes=likes,
                      verified=verified, tweet_age=tweet_age,
                      ts=datetime.datetime.now().strftime("%H:%M:%S"),
                      model=fake_art["model_type"])
        st.session_state.history.insert(0, record)
        r = record

        # ── tweet card ────────────────────────────────────────────────────────
        lbl_css   = r["label"].lower()
        badge_map = {"FAKE":  '<span class="badge badge-fake">🚨 FAKE</span>',
                     "TRUE":  '<span class="badge badge-true">✅ TRUE</span>',
                     "NEUTRAL":'<span class="badge badge-neutral">🟡 NEUTRAL</span>'}
        viral_html= ('<span class="badge badge-viral">🔥 VIRAL</span>'
                     if r["viral_prob"] >= 0.5
                     else '<span class="badge badge-safe">💤 NOT VIRAL</span>')

        st.markdown(f"""
<div class="tweet-card {lbl_css}">
  <div style="display:flex;gap:.9rem;align-items:flex-start;">
    <div class="tweet-avatar">🐦</div>
    <div style="flex:1;">
      <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">
        <span class="tweet-user">@analyst</span>
        <span class="tweet-handle">· {r['ts']}</span>
        <span style="flex:1;"></span>
        {badge_map[r['label']]}
        {viral_html}
      </div>
      <div class="tweet-text">{r['full_tweet']}</div>
      <div class="stat-row">
        <div class="stat">👥 <strong>{followers:,}</strong></div>
        <div class="stat">🔁 <strong>{retweets:,}</strong></div>
        <div class="stat">❤️ <strong>{likes:,}</strong></div>
        <div class="stat">⏱️ <strong>{tweet_age}h</strong></div>
        <div class="stat">🤖 <strong>{r['model']}</strong></div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── metric tiles ──────────────────────────────────────────────────────
        lc = LABEL_COLOR[r["label"]]
        vc = "#A78BFA" if r["viral_prob"] >= 0.5 else "#60A5FA"
        ic = "#22C55E" if r["impact"]>=65 else "#F59E0B" if r["impact"]>=35 else "#EF4444"
        cc2= "#22C55E" if r["imp_scores"]["credibility"]>=50 else "#EF4444"

        t1, t2, t3, t4 = st.columns(4)
        tiles = [
            (t1, "Verdict",         f'<div style="color:{lc};font-size:1.3rem;font-weight:700;">{r["label"]}</div>',
                                    f'{r["confidence"]*100:.1f}% confidence'),
            (t2, "Viral Probability",f'<div style="color:{vc};font-size:1.7rem;font-weight:700;">{r["viral_prob"]*100:.1f}%</div>',
                                    "🔥 likely viral" if r["viral_prob"]>=.5 else "💤 low virality"),
            (t3, "Impact Score",    f'<div style="color:{ic};font-size:1.7rem;font-weight:700;">{r["impact"]:.0f}</div>',
                                    "/ 100 composite"),
            (t4, "Credibility",     f'<div style="color:{cc2};font-size:1.7rem;font-weight:700;">{r["imp_scores"]["credibility"]:.0f}</div>',
                                    "/ 100"),
        ]
        for col, lbl, val_html, sub in tiles:
            with col:
                st.markdown(f"""
<div class="tile">
  <div class="t-label">{lbl}</div>
  {val_html}
  <div class="t-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── charts row 1 ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📊 Detection & Virality</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Classification Confidence**")
            st.pyplot(plot_fake_confidence(r["proba"]), use_container_width=True)
        with ch2:
            st.markdown("**Viral Probability Gauge**")
            st.pyplot(plot_viral_gauge(r["viral_prob"]), use_container_width=True)

        # ── charts row 2 ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-title">🧠 Features & Impact</div>', unsafe_allow_html=True)
        ch3, ch4 = st.columns(2)
        with ch3:
            st.markdown("**Viral Feature Importance**")
            st.pyplot(plot_feature_importance(
                viral_art["feature_display_names"],
                np.array(viral_art["importances"]),
                "Viral Model · Feature Importance",
            ), use_container_width=True)
        with ch4:
            st.markdown("**Impact Score Radar**")
            st.pyplot(plot_impact_radar(r["imp_scores"]), use_container_width=True)

        # ── impact breakdown ──────────────────────────────────────────────────
        st.markdown('<div class="sec-title">📐 Impact Breakdown</div>', unsafe_allow_html=True)
        sc   = r["imp_scores"]
        subs = [("🌐 Social Rep.","social_reputation","#1D9BF0"),
                ("🔁 Engagement","engagement","#A78BFA"),
                ("📌 Topic Eng.","topic_engagement","#34D399"),
                ("❤️ Likability","likability","#FB923C"),
                ("🛡️ Credibility","credibility","#F472B6")]
        for col, (lbl, key, color) in zip(st.columns(5), subs):
            with col:
                st.markdown(f"""
<div class="tile">
  <div class="t-label">{lbl}</div>
  <div class="t-value" style="color:{color};font-size:1.35rem;">{sc[key]:.0f}</div>
  <div class="t-sub">/ 100</div>
</div>""", unsafe_allow_html=True)

        # ── debug expander ────────────────────────────────────────────────────
        with st.expander("🔎 Preprocessed text & TF-IDF features"):
            st.code(preprocess_tweet(tweet_text), language=None)
            if "feature_names" in fake_art:
                st.pyplot(plot_feature_importance(
                    fake_art["feature_names"],
                    np.array(fake_art["importances"]),
                    "TF-IDF Term Importance",
                ), use_container_width=True)

    elif analyse_btn:
        st.warning("⚠️  Please enter a tweet before analysing.")
    else:
        st.markdown("""
<div style="text-align:center;padding:3rem 1rem;color:#1B2A42;">
  <div style="font-size:3rem;margin-bottom:.8rem;">🔬</div>
  <div style="font-size:1rem;color:#374151;">
    Enter a tweet above and click <strong style="color:#1D9BF0;">Analyse Tweet</strong>
  </div>
  <div style="font-size:.78rem;color:#1B2A42;margin-top:.4rem;">or pick a sample from the sidebar →</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2  ·  LIVE FEED
# ═══════════════════════════════════════════════════════════
with tab_feed:
    st.markdown('<div class="sec-title">📡 Analysed Tweet Feed</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
<div style="text-align:center;padding:2.5rem;color:#374151;">
  <div style="font-size:2rem;margin-bottom:.5rem;">📭</div>
  No tweets analysed yet — go to the <strong>Analyse</strong> tab first.
</div>""", unsafe_allow_html=True)
    else:
        h = st.session_state.history
        total   = len(h)
        n_fake  = sum(1 for r in h if r["label"] == "FAKE")
        n_true  = sum(1 for r in h if r["label"] == "TRUE")
        n_viral = sum(1 for r in h if r["viral_prob"] >= 0.5)
        avg_imp = sum(r["impact"] for r in h) / total

        for col, val, lbl, color in zip(
            st.columns(5),
            [total, n_fake, n_true, n_viral, f"{avg_imp:.0f}"],
            ["Total","Fake","True","Viral","Avg Impact"],
            ["#1D9BF0","#EF4444","#22C55E","#A78BFA","#F59E0B"],
        ):
            with col:
                st.markdown(f"""
<div class="tile">
  <div class="t-label">{lbl}</div>
  <div class="t-value" style="color:{color};font-size:1.5rem;">{val}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        for r in h:
            badge_css = {"FAKE":"badge-fake","TRUE":"badge-true","NEUTRAL":"badge-neutral"}[r["label"]]
            badge_ico = {"FAKE":"🚨","TRUE":"✅","NEUTRAL":"🟡"}[r["label"]]
            vc        = "badge-viral" if r["viral_prob"]>=.5 else "badge-safe"
            vi        = "🔥 VIRAL" if r["viral_prob"]>=.5 else "💤 SAFE"
            st.markdown(f"""
<div class="feed-item">
  <div class="tweet-avatar" style="font-size:.85rem;width:36px;height:36px;flex-shrink:0;">🐦</div>
  <div class="feed-snippet">
    {r['tweet']}
    <div class="feed-meta">👥 {r['followers']:,} · 🔁 {r['retweets']:,} · ❤️ {r['likes']:,} · ⏱️ {r['tweet_age']}h · 🕐 {r['ts']}</div>
  </div>
  <div class="feed-badges">
    <span class="badge {badge_css}">{badge_ico} {r['label']}</span>
    <span class="badge {vc}">{vi}</span>
    <span style="font-size:.68rem;color:#374151;">Impact {r['impact']:.0f}</span>
  </div>
</div>""", unsafe_allow_html=True)

        if st.button("⬇️ Export History"):
            df_exp = pd.DataFrame([{
                "tweet":r["full_tweet"],"label":r["label"],
                "confidence":round(r["confidence"],4),
                "viral_prob":round(r["viral_prob"],4),
                "impact":r["impact"],"followers":r["followers"],
                "retweets":r["retweets"],"likes":r["likes"],
                "verified":r["verified"],"tweet_age_h":r["tweet_age"],
                "model":r["model"],"time":r["ts"],
            } for r in h])
            st.download_button("📥 Download CSV", df_exp.to_csv(index=False),
                               "tweetscope_history.csv", "text/csv")


# ═══════════════════════════════════════════════════════════
# TAB 3  ·  BATCH
# ═══════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="sec-title">📋 Batch Tweet Analysis</div>', unsafe_allow_html=True)
    st.markdown("Paste up to **20 tweets** (one per line) for bulk analysis.")

    batch_text = st.text_area(
        label="batch", label_visibility="collapsed", height=180,
        value=(
            "Breaking: explosion at marathon finish line kills hundreds!!\n"
            "FBI releases photos of two suspects in the Boston Marathon bombing\n"
            "Training for Boston Marathon this year fingers crossed\n"
            "Secret source confirms Boston attack was inside job\n"
            "Boston strong community rallies after tragic bombing\n"
            "ALERT: second bomb found near Fenway media covering it up"
        ),
        placeholder="One tweet per line…",
    )
    run_batch = st.button("⚡ Run Batch Analysis")

    if run_batch and batch_text.strip():
        from src.train_viral_model import engineer_features, FEATURE_COLS

        lines    = [l.strip() for l in batch_text.strip().splitlines() if l.strip()][:20]
        progress = st.progress(0, text="Analysing…")
        results  = []

        for i, line in enumerate(lines):
            clean  = preprocess_tweet(line)
            proba  = fake_art["pipeline"].predict_proba([clean])[0]
            label  = fake_art["label_map"][int(np.argmax(proba))]
            conf   = float(np.max(proba))
            raw    = pd.DataFrame([{"number_of_followers": followers,
                                    "retweets": retweets, "likes": likes,
                                    "account_verified": int(verified),
                                    "tweet_age": tweet_age}])
            feats  = engineer_features(raw)[FEATURE_COLS]
            vp     = float(viral_art["pipeline"].predict_proba(feats)[:, 1][0])
            results.append({
                "Tweet":      line[:80]+("…" if len(line)>80 else ""),
                "Label":      label,
                "Confidence": f"{conf*100:.1f}%",
                "Viral Prob": f"{vp*100:.1f}%",
                "Viral?":     "🔥 Yes" if vp>=0.5 else "💤 No",
            })
            progress.progress((i+1)/len(lines), text=f"Processed {i+1}/{len(lines)}…")
        progress.empty()

        df_b = pd.DataFrame(results)

        def _lc(v):
            c = {"FAKE":"#7F1D1D","TRUE":"#14532D","NEUTRAL":"#78350F"}.get(v,"")
            return f"background:{c};color:white;" if c else ""

        st.dataframe(df_b.style.applymap(_lc, subset=["Label"]),
                     use_container_width=True, hide_index=True)

        n_f = sum(1 for r in results if r["Label"]=="FAKE")
        n_t = sum(1 for r in results if r["Label"]=="TRUE")
        n_n = sum(1 for r in results if r["Label"]=="NEUTRAL")
        n_v = sum(1 for r in results if r["Viral?"].startswith("🔥"))
        tot = len(results)
        st.markdown(
            f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin:.6rem 0;">'
            f'<span class="badge badge-fake">🚨 Fake: {n_f}/{tot}</span>'
            f'<span class="badge badge-true">✅ True: {n_t}/{tot}</span>'
            f'<span class="badge badge-neutral">🟡 Neutral: {n_n}/{tot}</span>'
            f'<span class="badge badge-viral">🔥 Viral: {n_v}/{tot}</span></div>',
            unsafe_allow_html=True,
        )
        st.download_button("📥 Download CSV", df_b.to_csv(index=False),
                           "batch_results.csv","text/csv")


# ═══════════════════════════════════════════════════════════
# TAB 4  ·  ABOUT
# ═══════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="sec-title">📘 Project Overview</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
**TweetScope** detects misinformation and predicts virality in tweets,
inspired by *"Analyzing Fake Content on Twitter"* (Boston Marathon dataset).

#### 🧠 Models
| Task | Baseline | Advanced |
|------|----------|---------|
| Fake Detection | TF-IDF + Logistic Regression | TF-IDF + XGBoost |
| Viral Prediction | Logistic Regression | XGBoost |

#### 📐 Impact Score Formula
```
Impact = 0.20 × SocialReputation
       + 0.25 × Engagement
       + 0.20 × TopicEngagement
       + 0.15 × Likability
       + 0.20 × Credibility
```

#### 🗂️ Project Structure
```
tweetscope/
├── app.py
├── requirements.txt
├── model/  fake_model.pkl  viral_model.pkl
└── src/    preprocess.py  train_fake_model.py
            train_viral_model.py  utils.py
```

#### 🚀 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```
""")

    with col_b:
        st.markdown("""
#### 📊 Model Performance
| Model | Metric | Score |
|-------|--------|-------|
| LR Fake | CV Accuracy | 99.8% |
| XGB Fake | Test F1 | 91% |
| LR Viral | AUC-ROC | 0.997 |
| XGB Viral | AUC-ROC | 1.000 |

#### 🔬 Features
**Text:** TF-IDF (1-2 grams), stopword removal, URL stripping

**Engagement:** log(followers), log(retweets), log(likes), verified, tweet age, log(engagement)
""")

    st.divider()
    st.caption("TweetScope · Streamlit + scikit-learn + XGBoost + Matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding-top:1.5rem;border-top:1px solid #111827;margin-top:1rem;">
  <span style="font-size:.72rem;color:#1B2A42;letter-spacing:.06em;">
    TWEETSCOPE · BOSTON MARATHON MISINFORMATION ANALYSIS · ML-POWERED
  </span>
</div>""", unsafe_allow_html=True)
