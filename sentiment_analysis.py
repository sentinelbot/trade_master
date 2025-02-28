import re
import random
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize sentiment analyzer
try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
except:
    # If NLTK is not available, create a placeholder
    sentiment_analyzer = None

# Try to load the transformers model if available
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    transformer_available = True
except:
    transformer_available = False

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    Returns sentiment (positive, negative, neutral) and confidence score.
    """
    # Clean the text of URLs, hashtags, mentions, etc.
    text = clean_text(text)

    if not text:
        return {"sentiment": "neutral", "score": 50, "text": "No analyzable text found"}

    # Try to use the transformers pipeline if available
    if transformer_available:
        try:
            result = sentiment_pipeline(text[:512])[0]  # Limit input size
            sentiment = result['label'].lower()
            confidence = round(result['score'] * 100, 1)
            return {"sentiment": sentiment, "score": confidence, "text": text}
        except Exception as e:
            print(f"Transformer sentiment analysis error: {e}")

    # Fall back to NLTK's VADER if transformers not available or fails
    if sentiment_analyzer:
        try:
            scores = sentiment_analyzer.polarity_scores(text)
            compound = scores['compound']

            if compound >= 0.05:
                sentiment = "positive"
                score = round((compound + 1) * 50, 1)  # Scale from [-1,1] to [0,100]
            elif compound <= -0.05:
                sentiment = "negative"
                score = round((1 - abs(compound)) * 50, 1)
            else:
                sentiment = "neutral"
                score = 50

            return {"sentiment": sentiment, "score": score, "text": text}
        except Exception as e:
            print(f"NLTK sentiment analysis error: {e}")

    # If all else fails, return a mock result
    sentiment = random.choice(["positive", "neutral", "negative"])
    score = random.randint(50, 95) if sentiment == "positive" else \
            (random.randint(5, 50) if sentiment == "negative" else random.randint(40, 60))

    return {"sentiment": sentiment, "score": score, "text": text}

def clean_text(text):
    """Clean text by removing URLs, hashtags, mentions, etc."""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_related_assets(sentiment):
    """
    Get assets that might perform well based on the sentiment.
    This is for demonstration purposes - in a real system, this would
    be based on more sophisticated analysis.
    """
    if sentiment == "positive":
        return [
            {"symbol": "BTC/USDT", "reason": "Bitcoin often rises with positive market sentiment"},
            {"symbol": "ETH/USDT", "reason": "Ethereum typically follows Bitcoin in bullish markets"},
            {"symbol": "SOL/USDT", "reason": "Solana performs well in risk-on environments"}
        ]
    elif sentiment == "negative":
        return [
            {"symbol": "USDC/USDT", "reason": "Stablecoins provide safety during market uncertainty"},
            {"symbol": "PAXG/USDT", "reason": "Gold-backed tokens can hedge against crypto volatility"},
            {"symbol": "AAVE/USDT", "reason": "Lending protocols may benefit from volatility"}
        ]
    else:  # neutral
        return [
            {"symbol": "BNB/USDT", "reason": "Exchange tokens often have more stable price action"},
            {"symbol": "LINK/USDT", "reason": "Oracle services continue regardless of market sentiment"},
            {"symbol": "XMR/USDT", "reason": "Privacy coins have unique market dynamics"}
        ]

def analyze_news_sentiment(news_items):
    """
    Analyze sentiment from a collection of news headlines.
    Returns an overall sentiment and confidence.
    """
    if not news_items:
        return {"sentiment": "neutral", "score": 50}

    # Combine all headlines into one text
    combined_text = " ".join([item.get("title", "") for item in news_items])

    # Analyze the combined sentiment
    result = analyze_sentiment(combined_text)

    # Add additional context
    result["source"] = "news_analysis"
    result["items_analyzed"] = len(news_items)
    result["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return result

def get_market_sentiment():
    """
    Get overall market sentiment from various sources.
    This would normally connect to external APIs and services.
    For demo purposes, returning mock data.
    """
    return {
        "overall": "cautiously optimistic",
        "bitcoin": {
            "sentiment": "positive",
            "score": 65.5,
            "fear_greed_index": 72,
            "social_sentiment": "bullish"
        },
        "ethereum": {
            "sentiment": "neutral",
            "score": 52.3,
            "social_sentiment": "mixed"
        },
        "altcoins": {
            "sentiment": "positive",
            "score": 58.2,
            "trend": "following bitcoin"
        },
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
