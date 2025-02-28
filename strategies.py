import os
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ContextTypes
import krakenex

# Load environment variables
load_dotenv()
# Initialize Binance client
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Initialize Kraken client
kraken_client = krakenex.API()
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

async def mean_reversion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Execute a mean reversion trading strategy.
    This strategy looks for assets that have deviated significantly from their historical average
    and takes positions based on the expectation they will revert to the mean.
    """
    await update.message.reply_text("Initializing mean reversion strategy analysis...")
    
    # Default to BTC/USDT if no symbol is provided
    symbol = "BTCUSDT"
    if context.args and len(context.args) > 0:
        symbol = context.args[0]
    
    try:
        # Fetch historical data from Binance
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
        closes = [float(k[4]) for k in klines]
        df = pd.DataFrame({'close': closes})
        
        # Calculate mean and standard deviation
        mean_price = df['close'].mean()
        std_price = df['close'].std()
        
        # Get current market price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Calculate z-score (deviation from mean in units of standard deviation)
        z_score = (current_price - mean_price) / std_price
        
        # Determine trading signal based on z-score
        if z_score < -2:
            signal = "BUY"
            reason = f"Price is significantly below the mean (z-score: {z_score:.2f})"
        elif z_score > 2:
            signal = "SELL"
            reason = f"Price is significantly above the mean (z-score: {z_score:.2f})"
        else:
            signal = "HOLD"
            reason = f"Price is within normal range of the mean (z-score: {z_score:.2f})"
        
        # Prepare response
        response = f"""
Mean Reversion Analysis for {symbol}:
- Current Price: ${current_price}
- 24-hour Mean: ${mean_price:.2f}
- Standard Deviation: ${std_price:.2f}
- Z-Score: {z_score:.2f}

Signal: {signal}
Reason: {reason}
        """
        
        # If signal is to buy or sell, suggest stop loss and take profit levels
        if signal in ["BUY", "SELL"]:
            stop_pct = 0.05  # 5% stop loss
            profit_pct = 0.10  # 10% take profit
            
            if signal == "BUY":
                stop_loss = current_price * (1 - stop_pct)
                take_profit = current_price * (1 + profit_pct)
            else:  # SELL
                stop_loss = current_price * (1 + stop_pct)
                take_profit = current_price * (1 - profit_pct)
            
            response += f"""
Recommended Risk Management:
- Stop Loss: ${stop_loss:.2f}
- Take Profit: ${take_profit:.2f}

Use /stoploss {symbol} {stop_loss:.2f} {take_profit:.2f} to set these levels.
            """
        
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error executing mean reversion strategy: {str(e)}")


async def momentum_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Execute a momentum trading strategy.
    This strategy identifies assets with strong price trends and takes positions
    in the direction of the trend.
    """
    await update.message.reply_text("Initializing momentum strategy analysis...")
    
    # Default to BTC/USDT if no symbol is provided
    symbol = "BTCUSDT"
    if context.args and len(context.args) > 0:
        symbol = context.args[0]
    
    try:
        # Fetch historical data from Binance
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 day ago UTC")
        closes = [float(k[4]) for k in klines]
        df = pd.DataFrame({'close': closes})
        
        # Calculate short and long-term moving averages
        df['short_ma'] = df['close'].rolling(window=3).mean()
        df['long_ma'] = df['close'].rolling(window=7).mean()
        
        # Calculate momentum (rate of change)
        df['momentum'] = df['close'].pct_change(periods=5) * 100
        
        # Get current price and indicators
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        current_momentum = df['momentum'].iloc[-1]
        short_ma = df['short_ma'].iloc[-1]
        long_ma = df['long_ma'].iloc[-1]
        
        # Determine trading signal based on moving averages and momentum
        if short_ma > long_ma and current_momentum > 5:
            signal = "STRONG BUY"
            reason = f"Strong upward momentum ({current_momentum:.2f}%) with short-term MA above long-term MA"
        elif short_ma > long_ma:
            signal = "BUY"
            reason = f"Short-term MA above long-term MA with moderate momentum ({current_momentum:.2f}%)"
        elif short_ma < long_ma and current_momentum < -5:
            signal = "STRONG SELL"
            reason = f"Strong downward momentum ({current_momentum:.2f}%) with short-term MA below long-term MA"
        elif short_ma < long_ma:
            signal = "SELL"
            reason = f"Short-term MA below long-term MA with moderate momentum ({current_momentum:.2f}%)"
        else:
            signal = "NEUTRAL"
            reason = f"Mixed signals with momentum at {current_momentum:.2f}%"
        
        # Prepare response
        response = f"""
Momentum Strategy Analysis for {symbol}:
- Current Price: ${current_price}
- 3-period MA: ${short_ma:.2f}
- 7-period MA: ${long_ma:.2f}
- 5-period Momentum: {current_momentum:.2f}%

Signal: {signal}
Reason: {reason}
        """
        
        # If signal is to buy or sell, suggest stop loss and take profit levels
        if "BUY" in signal or "SELL" in signal:
            stop_pct = 0.03  # 3% stop loss for momentum strategy (typically tighter)
            profit_pct = 0.09  # 9% take profit (3:1 risk-reward ratio)
            
            if "BUY" in signal:
                stop_loss = current_price * (1 - stop_pct)
                take_profit = current_price * (1 + profit_pct)
            else:  # SELL
                stop_loss = current_price * (1 + stop_pct)
                take_profit = current_price * (1 - profit_pct)
            
            response += f"""
Recommended Risk Management:
- Stop Loss: ${stop_loss:.2f}
- Take Profit: ${take_profit:.2f}

Use /stoploss {symbol} {stop_loss:.2f} {take_profit:.2f} to set these levels.
            """
        
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error executing momentum strategy: {str(e)}")


async def grid_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Execute a grid trading strategy.
    This strategy places buy and sell orders at predefined intervals to profit from price fluctuations.
    """
    await update.message.reply_text("Initializing grid trading strategy...")
    
    # Default to BTC/USDT if no symbol is provided
    symbol = "BTCUSDT"
    if context.args and len(context.args) > 0:
        symbol = context.args[0]
    
    # Default grid parameters
    lower_price = 40000
    upper_price = 50000
    grid_levels = 10
    
    if context.args and len(context.args) >= 3:
        lower_price = float(context.args[1])
        upper_price = float(context.args[2])
        if len(context.args) >= 4:
            grid_levels = int(context.args[3])
    
    try:
        price_range = upper_price - lower_price
        grid_step = price_range / grid_levels
        
        # Place buy and sell orders at each grid level
        for i in range(grid_levels):
            buy_price = lower_price + i * grid_step
            sell_price = buy_price + grid_step
            
            # Place buy order
            client.create_order(
                symbol=symbol,
                side='BUY',
                type='LIMIT',
                timeInForce='GTC',
                quantity=0.001,  # Example quantity
                price=str(buy_price)
            )
            
            # Place sell order
            client.create_order(
                symbol=symbol,
                side='SELL',
                type='LIMIT',
                timeInForce='GTC',
                quantity=0.001,  # Example quantity
                price=str(sell_price)
            )
        
        response = f"""
Grid Trading Strategy Executed for {symbol}:
- Lower Price: ${lower_price}
- Upper Price: ${upper_price}
- Grid Levels: {grid_levels}
- Grid Step: ${grid_step:.2f}

✅ Buy and sell orders placed at each grid level.
        """
        
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error executing grid trading strategy: {str(e)}")


async def find_arbitrage_opportunities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Find arbitrage opportunities across exchanges.
    This strategy compares prices between Binance and Kraken to identify arbitrage opportunities.
    """
    await update.message.reply_text("Searching for arbitrage opportunities...")
    
    try:
        # Fetch prices from Binance and Kraken
        binance_price = float(client.get_symbol_ticker(symbol='BTCUSDT')['price'])
        kraken_price = float(kraken_client.query_public('Ticker', {'pair': 'BTCUSDT'})['result']['BTCUSDT']['c'][0])
        
        if binance_price < kraken_price:
            response = f"""
Arbitrage Opportunity:
- Buy on Binance: ${binance_price}
- Sell on Kraken: ${kraken_price}
- Profit Margin: ${kraken_price - binance_price:.2f}
            """
        elif binance_price > kraken_price:
            response = f"""
Arbitrage Opportunity:
- Buy on Kraken: ${kraken_price}
- Sell on Binance: ${binance_price}
- Profit Margin: ${binance_price - kraken_price:.2f}
            """
        else:
            response = "No arbitrage opportunities found."
        
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error finding arbitrage opportunities: {str(e)}")