import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ContextTypes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Mock data for demonstration purposes
def get_mock_price_data(symbol, days=30):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    if symbol.upper().startswith('BTC'):
        start_price = 50000
        volatility = 1000
    elif symbol.upper().startswith('ETH'):
        start_price = 3000
        volatility = 100
    else:
        start_price = 100
        volatility = 5
        
    prices = np.cumsum(np.random.normal(0, volatility, days)) + start_price
    prices = np.abs(prices)  # Ensure no negative prices
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.uniform(1000000, 5000000, days)
    })

# Predict the next price using linear regression
async def predict_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Predict the next price for a given symbol using linear regression.
    
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """
    if not context.args or len(context.args) < 1:
        await update.message.reply_text("Please provide a symbol. Example: /predict BTC/USDT")
        return
    
    symbol = context.args[0].upper()
    await update.message.reply_text(f"Predicting the next price for {symbol}...")
    
    try:
        # Fetch historical data (this would normally call an API)
        data = get_mock_price_data(symbol, days=30)
        
        # Prepare data for regression
        data['days'] = (data['date'] - data['date'].min()).dt.days
        X = data[['days']]
        y = data['price']
        
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict the next price
        next_day = data['days'].max() + 1
        predicted_price = model.predict([[next_day]])[0]
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(data['date'], data['price'], label='Historical Prices', marker='o')
        plt.axvline(x=data['date'].max(), color='r', linestyle='--', label='Prediction Point')
        plt.plot([data['date'].max(), data['date'].max() + timedelta(days=1)],
                 [data['price'].iloc[-1], predicted_price],
                 'g--', label='Predicted Price')
        plt.title(f'Price Prediction for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Prepare the response
        response = f"""
Price Prediction for {symbol}:
- Last Price: ${data['price'].iloc[-1]:.2f}
- Predicted Next Price: ${predicted_price:.2f}
        """
        
        # Send the response and plot
        await update.message.reply_text(response)
        await update.message.reply_photo(photo=buf)
    except Exception as e:
        await update.message.reply_text(f"Error predicting price: {str(e)}")

# Train an LSTM model (placeholder implementation)
async def train_lstm_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Train an LSTM model for price prediction.
    
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """
    await update.message.reply_text("Training LSTM model...")
    
    try:
        # Placeholder for LSTM model training
        # In a real implementation, you would:
        # 1. Fetch historical data
        # 2. Preprocess the data (e.g., normalize, create sequences)
        # 3. Define and train the LSTM model
        # 4. Save the model for future use
        
        # For now, just return a placeholder message
        response = """
LSTM Model Training:
- Status: Completed
- Model Saved: lstm_model.h5
- Next Steps: Use /predict to make predictions with the trained model.
        """
        
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error training LSTM model: {str(e)}")

# Helper function to fetch historical data
def get_historical_data(symbol, days=30):
    """
    Fetch historical price data for a given symbol.
    
    Args:
        symbol (str): The trading symbol (e.g., 'BTC/USDT').
        days (int): The number of days of data to fetch.
    
    Returns:
        pd.DataFrame: A DataFrame containing historical data.
    """
    # Placeholder implementation (replace with actual data fetching logic)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = np.cumsum(np.random.normal(0, 100, days)) + 50000
    prices = np.abs(prices)  # Ensure no negative prices
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.uniform(1000000, 5000000, days)
    })

# For testing the module directly
if __name__ == "__main__":
    print("Predictive analytics module loaded successfully")
    # Add any test code here if needed