import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import mplfinance as mpf

# Set style for all plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

async def get_candlestick_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send a candlestick chart for a specific symbol and timeframe"""
    # Check if the update comes from a callback query
    is_callback = hasattr(update, 'callback_query')
    
    if is_callback:
        if not context.args or len(context.args) < 2:
            await update.edit_message_text("Please provide a symbol and timeframe. Example: /chart BTC/USDT 1h")
            return
        
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
        
    else:  # Command from message
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Please provide a symbol and timeframe. Example: /chart BTC/USDT 1h")
            return
        
        symbol = context.args[0].upper()
        timeframe = context.args[1].lower()
    
    await (update.edit_message_text if is_callback else update.message.reply_text)(
        f"Generating {timeframe} candlestick chart for {symbol}..."
    )
    
    # Get historical data (this would normally call an API)
    data = get_historical_data(symbol, timeframe)
    
    if not data:
        message = f"Failed to retrieve {timeframe} data for {symbol}."
        await (update.edit_message_text if is_callback else update.message.reply_text)(message)
        return
    
    # Generate the candlestick chart
    buffer = io.BytesIO()
    
    # Convert data to DataFrame for mplfinance
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Set up style for the chart
    mc = mpf.make_marketcolors(
        up='#00ff00', down='#ff0000',
        wick={'up':'#00ff00', 'down':'#ff0000'},
        volume='#0000ff',
    )
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    # Create the plot
    mpf.plot(
        df,
        type='candle',
        volume=True,
        style=style,
        title=f'\n{symbol} {timeframe.upper()} Chart',
        figsize=(10, 7),
        tight_layout=True,
        savefig=dict(fname=buffer, format='png', dpi=150)
    )
    
    buffer.seek(0)
    
    # Create a keyboard for additional options
    keyboard = [
        [
            InlineKeyboardButton("15m", callback_data=f"chart_{symbol}_15m"),
            InlineKeyboardButton("1h", callback_data=f"chart_{symbol}_1h"),
            InlineKeyboardButton("4h", callback_data=f"chart_{symbol}_4h"),
            InlineKeyboardButton("1d", callback_data=f"chart_{symbol}_1d")
        ],
        [
            InlineKeyboardButton("Market Data", callback_data=f"market_binance_{symbol}"),
            InlineKeyboardButton("Add to Watchlist", callback_data=f"addwatch_{symbol}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the chart
    if is_callback:
        await update.message.reply_photo(
            photo=buffer,
            caption=f"{symbol} {timeframe.upper()} Chart",
            reply_markup=reply_markup
        )
        # Delete the "generating..." message
        await update.delete_message()
    else:
        await update.message.reply_photo(
            photo=buffer,
            caption=f"{symbol} {timeframe.upper()} Chart",
            reply_markup=reply_markup
        )

async def get_profit_loss_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send a profit/loss chart for the user's portfolio"""
    await update.message.reply_text("Generating profit/loss chart for your portfolio...")
    
    # Get historical P&L data (this would normally be pulled from a database)
    data = get_pnl_history()
    
    if not data:
        await update.message.reply_text("No profit/loss data available for your portfolio.")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Plot the data
    plt.plot(df['date'], df['pnl_usd'], marker='o', linestyle='-', color='cyan')
    plt.fill_between(df['date'], 0, df['pnl_usd'], where=(df['pnl_usd'] > 0), alpha=0.3, color='green')
    plt.fill_between(df['date'], 0, df['pnl_usd'], where=(df['pnl_usd'] < 0), alpha=0.3, color='red')
    
    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every 7 days
    
    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Profit/Loss (USD)")
    plt.title("Portfolio Profit/Loss Over Time")
    plt.grid(True, alpha=0.3)
    
    # Add annotations for best and worst days
    best_day = df.loc[df['pnl_usd'].idxmax()]
    worst_day = df.loc[df['pnl_usd'].idxmin()]
    
    plt.annotate(f"+${best_day['pnl_usd']:.2f}",
                 xy=(best_day['date'], best_day['pnl_usd']),
                 xytext=(10, 20),
                 textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="white"),
                 color="green")
    
    plt.annotate(f"${worst_day['pnl_usd']:.2f}",
                 xy=(worst_day['date'], worst_day['pnl_usd']),
                 xytext=(10, -20),
                 textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="white"),
                 color="red")
    
    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    # Create a keyboard with options
    keyboard = [
        [
            InlineKeyboardButton("Daily View", callback_data="pnl_daily"),
            InlineKeyboardButton("Weekly View", callback_data="pnl_weekly"),
            InlineKeyboardButton("Monthly View", callback_data="pnl_monthly")
        ],
        [
            InlineKeyboardButton("Portfolio Overview", callback_data="portfolio"),
            InlineKeyboardButton("Full Report", callback_data="report")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the chart
    await update.message.reply_photo(
        photo=buffer,
        caption="Your Portfolio Profit/Loss Chart",
        reply_markup=reply_markup
    )

async def get_asset_comparison_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol1, symbol2):
    """Generate and send a chart comparing the performance of two assets"""
    await update.message.reply_text(f"Generating comparison chart for {symbol1} vs {symbol2}...")
    
    # Get historical data for both symbols (normally from API)
    data1 = get_historical_data(symbol1, '1d')
    data2 = get_historical_data(symbol2, '1d')
    
    if not data1 or not data2:
        await update.message.reply_text(f"Failed to retrieve data for comparison.")
        return
    
    # Convert to DataFrames
    df1 = pd.DataFrame(data1)
    df1['date'] = pd.to_datetime(df1['date'])
    df1.set_index('date', inplace=True)
    
    df2 = pd.DataFrame(data2)
    df2['date'] = pd.to_datetime(df2['date'])
    df2.set_index('date', inplace=True)
    
    # Normalize the data for comparison (starting at 100%)
    df1['normalized'] = df1['close'] / df1['close'].iloc[0] * 100
    df2['normalized'] = df2['close'] / df2['close'].iloc[0] * 100
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(df1.index, df1['normalized'], label=symbol1, linewidth=2)
    plt.plot(df2.index, df2['normalized'], label=symbol2, linewidth=2)
    
    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (%)")
    plt.title(f"Performance Comparison: {symbol1} vs {symbol2}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate performance metrics
    symbol1_change = ((df1['normalized'].iloc[-1] - 100) / 100) * 100
    symbol2_change = ((df2['normalized'].iloc[-1] - 100) / 100) * 100
    
    # Add annotations
    plt.annotate(f"{symbol1}: {symbol1_change:.2f}%",
                 xy=(df1.index[-1], df1['normalized'].iloc[-1]),
                 xytext=(10, 0),
                 textcoords="offset points",
                 color="white")
    
    plt.annotate(f"{symbol2}: {symbol2_change:.2f}%",
                 xy=(df2.index[-1], df2['normalized'].iloc[-1]),
                 xytext=(10, 0),
                 textcoords="offset points",
                 color="white")
    
    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    # Create keyboard with options
    keyboard = [
        [
            InlineKeyboardButton("7 Days", callback_data=f"compare_{symbol1}_{symbol2}_7d"),
            InlineKeyboardButton("30 Days", callback_data=f"compare_{symbol1}_{symbol2}_30d"),
            InlineKeyboardButton("90 Days", callback_data=f"compare_{symbol1}_{symbol2}_90d")
        ],
        [
            InlineKeyboardButton(f"{symbol1} Chart", callback_data=f"chart_{symbol1}_1d"),
            InlineKeyboardButton(f"{symbol2} Chart", callback_data=f"chart_{symbol2}_1d")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the chart
    await update.message.reply_photo(
        photo=buffer,
        caption=f"Performance Comparison: {symbol1} vs {symbol2}\n{symbol1}: {symbol1_change:.2f}%, {symbol2}: {symbol2_change:.2f}%",
        reply_markup=reply_markup
    )

# Helper function to get historical data
def get_historical_data(symbol, timeframe):
    """
    Get historical price data for a given symbol and timeframe.
    This is a mock function - in a real application, this would call an exchange API.
    """
    # Replace '/' in symbol for API compatibility
    symbol_formatted = symbol.replace('/', '')
    
    try:
        # For demo purposes, generate random data
        # In production, you would call an API like:
        # url = f"https://api.binance.com/api/v3/klines?symbol={symbol_formatted}&interval={timeframe}&limit=100"
        # response = requests.get(url)
        # data = response.json()
        
        end_date = datetime.now()
        
        # Set the time interval based on the timeframe
        if timeframe == '1d':
            interval = timedelta(days=1)
            num_periods = 30
        elif timeframe == '4h':
            interval = timedelta(hours=4)
            num_periods = 30
        elif timeframe == '1h':
            interval = timedelta(hours=1)
            num_periods = 24
        elif timeframe == '15m':
            interval = timedelta(minutes=15)
            num_periods = 32
        else:
            interval = timedelta(days=1)
            num_periods = 30
        
        # Generate dates
        dates = [end_date - i * interval for i in range(num_periods)]
        dates.reverse()  # Earliest first
        
        # Generate random price data
        base_price = 50000 if 'BTC' in symbol else 2500 if 'ETH' in symbol else 100
        volatility = 0.02  # 2% volatility
        
        data = []
        prev_close = base_price
        
        for date in dates:
            # Generate realistic OHLCV data
            change = np.random.normal(0, volatility)
            close = prev_close * (1 + change)
            high = close * (1 + abs(np.random.normal(0, volatility/2)))
            low = close * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = prev_close  # Open at previous close
            volume = np.random.uniform(100, 1000) * base_price
            
            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
            prev_close = close
        
        return data
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

# Helper function to get P&L history
def get_pnl_history():
    """
    Get historical profit/loss data for the user's portfolio.
    This is a mock function - in a real application, this would query a database.
    """
    # For demo purposes, generate random data
    end_date = datetime.now()
    
    # Generate 30 days of data
    dates = [end_date - timedelta(days=i) for i in range(30)]
    dates.reverse()  # Earliest first
    
    # Start with $0 P&L and add daily changes
    data = []
    pnl = 0
    
    for date in dates:
        # Generate a random daily P&L change
        # More volatility on weekdays, less on weekends
        is_weekend = date.weekday() >= 5
        volatility = 200 if is_weekend else 500
        daily_change = np.random.normal(0, volatility)
        
        pnl += daily_change
        
        data.append({
            'date': date,
            'pnl_usd': pnl,
            'daily_change': daily_change
        })
    
    return data