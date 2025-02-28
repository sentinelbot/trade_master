import pandas as pd
import numpy as np
import ccxt
from scipy import stats
from sklearn.ensemble import IsolationForest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import matplotlib.pyplot as plt
import io

# Setup exchanges
exchange = ccxt.binance()

async def detect_anomalies(update, context):
    """Detect anomalies in price data"""
    if not context.args:
        await update.message.reply_text("Please provide a symbol to analyze. Example: /anomaly BTC/USDT")
        return
    
    symbol = context.args[0]
    await update.message.reply_text(f"Detecting anomalies for {symbol}. This may take a moment...")
    
    try:
        # Fetch price data
        since = exchange.parse8601('1 week ago')
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Simple Z-score method
        data['returns'] = data['close'].pct_change()
        data['z_score'] = stats.zscore(data['returns'].fillna(0))
        
        # Isolation Forest method
        model = IsolationForest(contamination=0.05)
        data['isolation_forest'] = model.fit_predict(data[['returns', 'volume']].fillna(0))
        
        # Find anomalies
        z_score_anomalies = data[abs(data['z_score']) > 3].copy()
        isolation_forest_anomalies = data[data['isolation_forest'] == -1].copy()
        
        # Combine anomalies
        combined_anomalies = pd.concat([z_score_anomalies, isolation_forest_anomalies]).drop_duplicates()
        
        # Create response
        if len(combined_anomalies) > 0:
            response = f"Detected {len(combined_anomalies)} anomalies for {symbol}:\n\n"
            for idx, row in combined_anomalies.iterrows():
                response += f"• {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: "
                response += f"Price: ${row['close']:.2f}, "
                response += f"Return: {row['returns']*100:.2f}%, "
                response += f"Z-score: {row['z_score']:.2f}\n"
            
            # Create anomaly visualization
            plt.figure(figsize=(10, 6))
            plt.plot(data['timestamp'], data['close'], 'b-', label='Close Price')
            plt.scatter(combined_anomalies['timestamp'], combined_anomalies['close'], 
                        color='red', s=50, label='Anomalies')
            plt.title(f'Price Anomalies for {symbol}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Convert plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Send response and image
            await update.message.reply_text(response)
            await update.message.reply_photo(buf)
            
            # Offer analysis options
            keyboard = [
                [InlineKeyboardButton("Get Alert for Next Anomaly", callback_data=f"anomaly_alert_{symbol}")],
                [InlineKeyboardButton("Detailed Report", callback_data=f"anomaly_report_{symbol}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Would you like to:", reply_markup=reply_markup)
        else:
            await update.message.reply_text(f"No significant anomalies detected for {symbol} in the past week.")
    
    except Exception as e:
        await update.message.reply_text(f"Error detecting anomalies: {str(e)}")

async def get_anomaly_report(update, context, symbol):
    """Generate a detailed anomaly report"""
    await update.message.reply_text(f"Generating detailed anomaly report for {symbol}...")
    
    # This would connect to your analytics module for a detailed report
    # For demonstration, returning a placeholder
    report = (
        f"Anomaly Analysis Report for {symbol}\n\n"
        f"1. Found 3 price volatility anomalies\n"
        f"2. Found 2 volume spike anomalies\n"
        f"3. Risk assessment: MEDIUM\n\n"
        f"Recommended actions:\n"
        f"- Monitor {symbol} closely over the next 24 hours\n"
        f"- Consider reducing position size if volatility continues"
    )
    
    await update.message.reply_text(report)

async def set_anomaly_alert(update, context, symbol):
    """Set an alert for the next detected anomaly"""
    # This would connect to your notifications module
    # For demonstration, returning a placeholder
    await update.message.reply_text(f"Alert set! You will be notified when a new anomaly is detected for {symbol}.")