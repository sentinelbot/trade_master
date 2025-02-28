import asyncio
import json
import os
import sys
import pandas as pd  # For ATR calculation
from datetime import datetime
from typing import Dict, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update
from telegram.ext import ContextTypes

# Import notification module
try:
    from modules.notifications import send_notification
except ImportError:
    from modules.notifications import send_notification

# File to store stop-loss and take-profit levels
STOP_LOSS_FILE = "data/stop_loss_take_profit.json"

# Global dictionary to store stop-loss and take-profit levels
STOP_LOSS_TAKE_PROFIT: Dict[str, Dict[str, Dict[str, float]]] = {}

# Initialize storage
def init_storage():
    """
    Initialize the storage for stop-loss and take-profit levels.
    Creates the directory and file if they don't exist.
    """
    os.makedirs(os.path.dirname(STOP_LOSS_FILE), exist_ok=True)
    if not os.path.exists(STOP_LOSS_FILE):
        with open(STOP_LOSS_FILE, 'w') as f:
            json.dump({}, f)
    else:
        global STOP_LOSS_TAKE_PROFIT
        with open(STOP_LOSS_FILE, 'r') as f:
            try:
                STOP_LOSS_TAKE_PROFIT = json.load(f)
            except json.JSONDecodeError:
                STOP_LOSS_TAKE_PROFIT = {}
                with open(STOP_LOSS_FILE, 'w') as f:
                    json.dump({}, f)

# Save stop-loss and take-profit levels to file
def save_stop_loss_take_profit():
    """
    Save the current stop-loss and take-profit levels to the file.
    """
    with open(STOP_LOSS_FILE, 'w') as f:
        json.dump(STOP_LOSS_TAKE_PROFIT, f)

# Command: Set stop-loss and take-profit levels
async def set_stop_loss_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Set stop-loss and take-profit levels for a given symbol.
    
    Args:
        update (Update): The Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Please provide a symbol, stop-loss, and take-profit levels. "
            "Example: /stoploss BTC/USDT 45000 52000"
        )
        return
    
    try:
        symbol = context.args[0].upper()
        stop_loss = float(context.args[1])
        take_profit = float(context.args[2])
        
        # Store in memory and in file
        user_id = str(update.message.from_user.id)
        if user_id not in STOP_LOSS_TAKE_PROFIT:
            STOP_LOSS_TAKE_PROFIT[user_id] = {}
        
        STOP_LOSS_TAKE_PROFIT[user_id][symbol] = {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.now().isoformat()
        }
        
        save_stop_loss_take_profit()
        
        await update.message.reply_text(
            f"Stop-loss and take-profit set for {symbol}:\n"
            f"- Stop-loss: ${stop_loss}\n"
            f"- Take-profit: ${take_profit}\n\n"
            f"You will be notified when price reaches these levels."
        )
    except ValueError:
        await update.message.reply_text(
            "Invalid number format. Please provide valid numbers for stop-loss and take-profit."
        )
    except Exception as e:
        await update.message.reply_text(f"Error setting stop-loss and take-profit: {str(e)}")

# Check stop-loss and take-profit levels against current prices
async def check_stop_loss_take_profit(context: ContextTypes.DEFAULT_TYPE):
    """
    Check if current prices have triggered any stop-loss or take-profit levels.
    
    Args:
        context (ContextTypes.DEFAULT_TYPE): The context object.
    """
    try:
        # Load latest settings
        init_storage()
        
        if not STOP_LOSS_TAKE_PROFIT:
            return  # No stop-loss or take-profit levels set
        
        # Get current prices (this would connect to exchange API)
        # This is a placeholder - in production you would fetch real prices
        current_prices = {
            "BTC/USDT": 48000,
            "ETH/USDT": 3400,
            "SOL/USDT": 120
        }
        
        for user_id, symbols in STOP_LOSS_TAKE_PROFIT.items():
            for symbol, levels in symbols.items():
                if symbol not in current_prices:
                    continue  # Skip if we don't have price data
                
                current_price = current_prices[symbol]
                stop_loss = levels["stop_loss"]
                take_profit = levels["take_profit"]
                
                # Check if price has reached stop-loss or take-profit
                if current_price <= stop_loss:
                    message = f"⚠️ STOP LOSS TRIGGERED for {symbol} ⚠️\n" \
                              f"Current price: ${current_price}\n" \
                              f"Stop-loss level: ${stop_loss}"
                    await send_notification(context, user_id, message)
                    
                    # Remove this stop-loss/take-profit pair
                    del STOP_LOSS_TAKE_PROFIT[user_id][symbol]
                    if not STOP_LOSS_TAKE_PROFIT[user_id]:
                        del STOP_LOSS_TAKE_PROFIT[user_id]
                    save_stop_loss_take_profit()
                
                elif current_price >= take_profit:
                    message = f"🎯 TAKE PROFIT REACHED for {symbol} 🎯\n" \
                              f"Current price: ${current_price}\n" \
                              f"Take-profit level: ${take_profit}"
                    await send_notification(context, user_id, message)
                    
                    # Remove this stop-loss/take-profit pair
                    del STOP_LOSS_TAKE_PROFIT[user_id][symbol]
                    if not STOP_LOSS_TAKE_PROFIT[user_id]:
                        del STOP_LOSS_TAKE_PROFIT[user_id]
                    save_stop_loss_take_profit()
    
    except Exception as e:
        print(f"Error checking stop-loss and take-profit levels: {str(e)}")

# Calculate position size based on risk percentage
def calculate_position_size(account_balance: float, risk_percentage: float, entry_price: float, stop_loss_price: float) -> float:
    """
    Calculate the appropriate position size based on risk management principles.
    
    Args:
        account_balance (float): Total account balance.
        risk_percentage (float): Percentage of account willing to risk (e.g., 1 for 1%).
        entry_price (float): Price at which the trade will be entered.
        stop_loss_price (float): Price at which the trade will be exited if it goes against the trader.
    
    Returns:
        float: Position size in units of the base currency.
    """
    risk_amount = account_balance * (risk_percentage / 100)
    price_difference = abs(entry_price - stop_loss_price)
    risk_per_unit = price_difference
    
    # Calculate position size
    position_size = risk_amount / risk_per_unit
    
    return position_size

# Calculate Average True Range (ATR)
def calculate_atr(symbol: str, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) for a given symbol.
    
    Args:
        symbol (str): The trading symbol (e.g., 'BTC/USDT').
        period (int): The period for ATR calculation (default is 14).
    
    Returns:
        float: The ATR value.
    """
    try:
        # Fetch historical data (this is a placeholder; replace with actual data fetching logic)
        historical_data = get_historical_data(symbol, period)
        
        # Calculate True Range (TR)
        high = historical_data['high']
        low = historical_data['low']
        close = historical_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1]  # Return the latest ATR value
    
    except Exception as e:
        print(f"Error calculating ATR for {symbol}: {str(e)}")
        return 0.0  # Return 0 if there's an error

# Calculate Risk-Reward Ratio
def calculate_risk_reward(entry_price: float, stop_loss_price: float, take_profit_price: float) -> float:
    """
    Calculate the risk-reward ratio for a trade.
    
    Args:
        entry_price (float): The price at which the trade is entered.
        stop_loss_price (float): The price at which the trade will be exited if it goes against the trader.
        take_profit_price (float): The price at which the trade will be exited if it goes in favor of the trader.
    
    Returns:
        float: The risk-reward ratio.
    """
    try:
        # Calculate risk (distance from entry to stop-loss)
        risk = abs(entry_price - stop_loss_price)
        
        # Calculate reward (distance from entry to take-profit)
        reward = abs(entry_price - take_profit_price)
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward / risk
        
        return risk_reward_ratio
    
    except ZeroDivisionError:
        print("Error: Stop-loss price cannot be equal to entry price.")
        return 0.0
    except Exception as e:
        print(f"Error calculating risk-reward ratio: {str(e)}")
        return 0.0

# Fetch historical data (placeholder implementation)
def get_historical_data(symbol: str, period: int) -> pd.DataFrame:
    """
    Fetch historical price data for a given symbol.
    
    Args:
        symbol (str): The trading symbol (e.g., 'BTC/USDT').
        period (int): The number of periods to fetch.
    
    Returns:
        pd.DataFrame: A DataFrame containing historical data (high, low, close).
    """
    # Placeholder implementation (replace with actual data fetching logic)
    # Example: Fetch data from an exchange API or a local database
    data = {
        'high': [50000, 51000, 49000, 50500, 51500],
        'low': [48000, 49000, 47000, 48500, 49500],
        'close': [49000, 50000, 48000, 49500, 50500]
    }
    return pd.DataFrame(data)

# Calculate portfolio risk metrics
def calculate_portfolio_risk(positions: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate risk metrics for a portfolio.
    
    Args:
        positions (List[Dict[str, float]]): List of dictionaries containing position information.
    
    Returns:
        Dict[str, float]: Dictionary with risk metrics.
    """
    total_value = sum(position["value"] for position in positions)
    max_drawdown = 0
    var_95 = 0  # Value at Risk (95% confidence)
    
    # Calculate diversification score based on concentration
    symbols = {}
    for position in positions:
        symbol = position["symbol"]
        if symbol not in symbols:
            symbols[symbol] = 0
        symbols[symbol] += position["value"]
    
    # Calculate Herfindahl-Hirschman Index (HHI) as a measure of concentration
    hhi = sum((value / total_value) ** 2 for value in symbols.values())
    diversification_score = 1 - hhi  # Higher is better (more diversified)
    
    return {
        "total_value": total_value,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "diversification_score": diversification_score
    }

# For testing the module directly
if __name__ == "__main__":
    print("Risk management module loaded successfully")
    # Add any test code here if needed