"""
preprocess.py
-------------
Text cleaning and preprocessing utilities for tweet analysis.
Handles: cleaning, tokenization, stopword removal, feature extraction.
"""

import re
import string
import numpy as np

# ---------------------------------------------------------------------------
# Lightweight stopword list (no NLTK download required)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "dare","ought","used","i","me","my","we","our","you","your","he","his",
    "she","her","it","its","they","their","this","that","these","those",
    "what","which","who","how","when","where","why","all","each","every",
    "both","few","more","most","other","some","such","no","not","only","own",
    "same","so","than","too","very","just","as","up","about","after","before",
    "between","into","through","during","above","below","from","out","off",
    "over","under","again","then","once","here","there","while","although",
    "because","since","unless","until","though",
}


def clean_text(text: str) -> str:
    """
    Full tweet cleaning pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove mentions (@user)
      4. Remove hashtag symbol (keep word)
      5. Remove numbers
      6. Remove punctuation
      7. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # hashtags → word
    text = re.sub(r"\d+", "", text)                      # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Remove common stopwords from token list."""
    return [t for t in tokens if t not in STOPWORDS]


def preprocess_tweet(text: str) -> str:
    """
    Full pipeline: clean → tokenize → remove stopwords → rejoin.
    Returns a single cleaned string ready for TF-IDF vectorisation.
    """
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tokens  = remove_stopwords(tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Impact Score (from paper)
# ---------------------------------------------------------------------------

def compute_impact_score(
    followers: int,
    retweets: int,
    likes: int,
    verified: bool,
    credibility_score: float,      # 0-1, output of fake-detection model
) -> dict:
    """
    Impact = SocialReputation + Engagement + TopicEngagement + Likability + Credibility
    All sub-scores normalised to [0, 100].
    """
    # Social Reputation: log-scaled follower count
    social_reputation = min(np.log1p(followers) / np.log1p(1_000_000) * 100, 100)
    if verified:
        social_reputation = min(social_reputation * 1.2, 100)

    # Engagement: retweet-driven
    engagement = min(np.log1p(retweets) / np.log1p(10_000) * 100, 100)

    # Topic Engagement: like-driven
    topic_engagement = min(np.log1p(likes) / np.log1p(50_000) * 100, 100)

    # Likability: combined social signal
    likability = (engagement * 0.5 + topic_engagement * 0.5)

    # Credibility: 0→fake, 100→true  (model passes probability of TRUE class)
    credibility = credibility_score * 100

    total = (
        social_reputation * 0.20 +
        engagement        * 0.25 +
        topic_engagement  * 0.20 +
        likability        * 0.15 +
        credibility       * 0.20
    )

    return {
        "social_reputation": round(social_reputation, 1),
        "engagement":        round(engagement, 1),
        "topic_engagement":  round(topic_engagement, 1),
        "likability":        round(likability, 1),
        "credibility":       round(credibility, 1),
        "total":             round(total, 1),
    }
