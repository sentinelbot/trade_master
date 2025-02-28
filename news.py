import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application
# Cache for storing news to reduce API calls
news_cache = {}
cache_expiry = 3600  # Cache valid for 1 hour

async def get_news_for_symbol(update: Update, context: Application, symbol=None):
    """
    Fetches news articles related to a specific cryptocurrency symbol.
    
    Args:
        update (Update): The Telegram update object
        context (ContextTypes.DEFAULT_TYPE): The Telegram context
        symbol (str, optional): The cryptocurrency symbol to get news for
    """
    # Use passed symbol or get from args
    if not symbol and context.args:
        symbol = context.args[0].upper()
    elif not symbol:
        await update.message.reply_text("Please provide a symbol to get news for. Example: /news BTC")
        return
    
    # Check if valid symbol (could be expanded with a proper list)
    valid_symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "DOT", "AVAX", "MATIC"]
    search_term = symbol
    
    # Map symbols to full names for better search results
    symbol_to_name = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "DOGE": "Dogecoin",
        "XRP": "Ripple",
        "ADA": "Cardano",
        "DOT": "Polkadot",
        "AVAX": "Avalanche",
        "MATIC": "Polygon"
    }
    
    if symbol in symbol_to_name:
        search_term = symbol_to_name[symbol]
    
    # Check cache first
    current_time = datetime.now()
    if symbol in news_cache and (current_time - news_cache[symbol]["timestamp"]).seconds < cache_expiry:
        await format_and_send_news(update, context, symbol, news_cache[symbol]["articles"])
        return
    
    # Send loading message
    loading_message = await update.message.reply_text(f"Fetching latest news for {symbol}...")
    
    try:
        # In a real implementation, you would call a news API
        # Example: NewsAPI, CryptoCompare News API, or CoinGecko News
        
        # For demo, simulate API response with sample data
        await asyncio.sleep(1)  # Simulate network delay
        
        # Generate sample news based on the symbol
        articles = generate_sample_news(search_term, symbol)
        
        # Update cache
        news_cache[symbol] = {
            "articles": articles,
            "timestamp": current_time
        }
        
        # Send formatted news
        await context.bot.delete_message(chat_id=update.message.chat_id, message_id=loading_message.message_id)
        await format_and_send_news(update, context, symbol, articles)
        
    except Exception as e:
        await context.bot.edit_message_text(
            chat_id=update.message.chat_id,
            message_id=loading_message.message_id,
            text=f"Error fetching news: {str(e)}"
        )

def generate_sample_news(search_term, symbol):
    """
    Generates sample news data for demonstration purposes.
    
    Args:
        search_term (str): The search term (coin name)
        symbol (str): The cryptocurrency symbol
        
    Returns:
        list: List of news article dictionaries
    """
    current_date = datetime.now()
    
    # Generate different news based on symbol
    if search_term == "Bitcoin":
        articles = [
            {
                "title": f"{search_term} Price Analysis: Bulls Push Higher After Key Resistance Break",
                "description": f"{search_term} has broken through the $51,000 resistance level, suggesting further upward momentum.",
                "url": "https://example.com/btc-analysis",
                "publishedAt": (current_date - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CryptoAnalytics"
            },
            {
                "title": f"Institutional Investors Increase {search_term} Holdings",
                "description": f"Major financial institutions have increased their {search_term} allocations by 15% in the past month.",
                "url": "https://example.com/btc-institutional",
                "publishedAt": (current_date - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "InvestmentDaily"
            },
            {
                "title": f"New {search_term} ETF Shows Strong Performance",
                "description": f"The recently launched {search_term} ETF has attracted over $2 billion in assets under management.",
                "url": "https://example.com/btc-etf",
                "publishedAt": (current_date - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "FinanceToday"
            }
        ]
    elif search_term == "Ethereum":
        articles = [
            {
                "title": f"{search_term} Upgrade Scheduled for Next Month",
                "description": f"The upcoming {search_term} network upgrade promises significant improvements in transaction throughput.",
                "url": "https://example.com/eth-upgrade",
                "publishedAt": (current_date - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "BlockchainNews"
            },
            {
                "title": f"DeFi Projects on {search_term} See Record Growth",
                "description": f"Total value locked in {search_term}-based DeFi protocols has reached an all-time high of $45 billion.",
                "url": "https://example.com/eth-defi",
                "publishedAt": (current_date - timedelta(hours=18)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "DeFiInsider"
            },
            {
                "title": f"{search_term} Layer 2 Solutions Gaining Traction",
                "description": f"Adoption of Layer 2 scaling solutions for {search_term} has increased by 300% since last quarter.",
                "url": "https://example.com/eth-layer2",
                "publishedAt": (current_date - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CryptoTech"
            }
        ]
    else:
        # Generic news for other symbols
        articles = [
            {
                "title": f"{search_term} ({symbol}) Shows Strong Recovery",
                "description": f"{search_term} has gained 12% in the past week, outperforming the broader crypto market.",
                "url": "https://example.com/crypto-recovery",
                "publishedAt": (current_date - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CryptoMarketWatch"
            },
            {
                "title": f"Development Update: What's New with {search_term}",
                "description": f"The {search_term} team has released their Q1 development roadmap with several major features planned.",
                "url": "https://example.com/dev-update",
                "publishedAt": (current_date - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CoinDevelopers"
            },
            {
                "title": f"Market Analysis: Is {search_term} Undervalued?",
                "description": f"Technical indicators suggest {search_term} ({symbol}) may be positioned for a significant move upward.",
                "url": "https://example.com/market-analysis",
                "publishedAt": (current_date - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "TradingView"
            }
        ]
    
    return articles

async def format_and_send_news(update, context, symbol, articles):
    """
    Formats and sends news articles as a message with inline buttons.
    
    Args:
        update (Update): The Telegram update object
        context (ContextTypes.DEFAULT_TYPE): The Telegram context
        symbol (str): The cryptocurrency symbol
        articles (list): List of news article dictionaries
    """
    if not articles:
        await update.message.reply_text(f"No recent news found for {symbol}.")
        return
    
    # Format message
    message = f"📰 *Latest News for {symbol}*\n\n"
    
    for i, article in enumerate(articles, 1):
        pub_date = article.get("publishedAt", "N/A")
        source = article.get("source", "Unknown")
        
        message += f"{i}. *{article['title']}*\n"
        message += f"   _Published: {pub_date} by {source}_\n\n"
    
    # Add buttons for related actions
    keyboard = [
        [
            InlineKeyboardButton(f"{symbol} Price", callback_data=f"market_binance_{symbol}/USDT"),
            InlineKeyboardButton(f"{symbol} Chart", callback_data=f"chart_{symbol}/USDT_1d")
        ],
        [
            InlineKeyboardButton("More News", callback_data=f"more_news_{symbol}"),
            InlineKeyboardButton("Add to Watchlist", callback_data=f"addwatch_{symbol}/USDT")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def get_related_news_by_sentiment(sentiment, keyword=None):
    """
    Gets news articles related to a specific market sentiment.
    
    Args:
        sentiment (str): The sentiment (positive, negative, neutral)
        keyword (str, optional): Additional keyword filter
        
    Returns:
        list: List of news article dictionaries
    """
    current_date = datetime.now()
    
    # Generate different news based on sentiment
    if sentiment.lower() == "positive":
        articles = [
            {
                "title": "Crypto Market Sees Bullish Momentum After Regulatory Clarity",
                "description": "New regulatory frameworks provide certainty for institutional investors, leading to increased adoption.",
                "url": "https://example.com/bullish-momentum",
                "publishedAt": (current_date - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CryptoOptimist"
            },
            {
                "title": "Bitcoin Hashrate Reaches All-Time High, Network Security Strengthens",
                "description": "The Bitcoin network is more secure than ever as mining difficulty and hashrate continue to climb.",
                "url": "https://example.com/hashrate-ath",
                "publishedAt": (current_date - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "MiningInsider"
            }
        ]
    elif sentiment.lower() == "negative":
        articles = [
            {
                "title": "Market Downturn: What's Behind the Recent Crypto Correction",
                "description": "Analysts point to macroeconomic factors and profit taking as reasons for the latest market pullback.",
                "url": "https://example.com/market-correction",
                "publishedAt": (current_date - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "BearMarket"
            },
            {
                "title": "Regulatory Concerns Hit DeFi Projects Hard",
                "description": "Uncertainty around upcoming regulations has led to increased selling pressure across DeFi tokens.",
                "url": "https://example.com/defi-concerns",
                "publishedAt": (current_date - timedelta(hours=15)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "RegWatch"
            }
        ]
    else:  # neutral
        articles = [
            {
                "title": "Crypto Market Analysis: Consolidation Period Continues",
                "description": "The market appears to be in an accumulation phase as trading volumes decrease and volatility normalizes.",
                "url": "https://example.com/consolidation",
                "publishedAt": (current_date - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "MarketAnalysis"
            },
            {
                "title": "Bitcoin Holds Steady Despite Stock Market Fluctuations",
                "description": "Crypto assets show increasing independence from traditional market movements, suggesting maturing market dynamics.",
                "url": "https://example.com/btc-steady",
                "publishedAt": (current_date - timedelta(hours=22)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "CryptoCorrelation"
            }
        ]
    
    # Filter by keyword if provided
    if keyword:
        keyword = keyword.lower()
        filtered_articles = [
            article for article in articles 
            if keyword in article['title'].lower() or keyword in article['description'].lower()
        ]
        return filtered_articles if filtered_articles else articles
    
    return articles