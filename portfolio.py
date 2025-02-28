import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Mock data for demonstration purposes
def get_mock_portfolio_data():
    return {
        'total_value': 58642.75,
        'assets': [
            {'symbol': 'BTC/USDT', 'amount': 0.73, 'value': 36500.00, 'allocation': 62.24},
            {'symbol': 'ETH/USDT', 'amount': 4.21, 'value': 12630.00, 'allocation': 21.54},
            {'symbol': 'SOL/USDT', 'amount': 85.2, 'value': 5964.00, 'allocation': 10.17},
            {'symbol': 'ADA/USDT', 'amount': 12000, 'value': 3548.75, 'allocation': 6.05}
        ]
    }

# Mock watchlist data and functions
_watchlist = {
    "default": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
}

def get_watchlist(user_id="default"):
    return _watchlist.get(user_id, [])

def add_to_watchlist(symbol, user_id="default"):
    if user_id not in _watchlist:
        _watchlist[user_id] = []
    if symbol not in _watchlist[user_id]:
        _watchlist[user_id].append(symbol)
        return True
    return False

def remove_from_watchlist(symbol, user_id="default"):
    if user_id in _watchlist and symbol in _watchlist[user_id]:
        _watchlist[user_id].remove(symbol)
        return True
    return False

# Function to get portfolio overview
async def get_portfolio_overview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # In a real implementation, you would fetch data from your database or exchange API
        portfolio_data = get_mock_portfolio_data()
        
        # Create the portfolio overview message
        message = "📊 Portfolio Overview 📊\n\n"
        message += f"Total Value: ${portfolio_data['total_value']:.2f}\n\n"
        message += "Holdings:\n"
        
        for asset in portfolio_data['assets']:
            message += f"• {asset['symbol']}: {asset['amount']} (${asset['value']:.2f}, {asset['allocation']}%)\n"
        
        # Create inline keyboard for portfolio actions
        keyboard = [
            [InlineKeyboardButton("Asset Allocation", callback_data="allocation"),
             InlineKeyboardButton("Rebalance", callback_data="rebalance")],
            [InlineKeyboardButton("PnL Chart", callback_data="pnl"),
             InlineKeyboardButton("Trade History", callback_data="trades")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Determine if we're responding to a message or callback query
        if hasattr(update, 'message'):
            await update.message.reply_text(message, reply_markup=reply_markup)
        else:
            await update.callback_query.edit_message_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        error_message = f"Error retrieving portfolio overview: {str(e)}"
        if hasattr(update, 'message'):
            await update.message.reply_text(error_message)
        else:
            await update.callback_query.edit_message_text(error_message)

# Function to get asset allocation
async def get_asset_allocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # In a real implementation, you would fetch data from your database or exchange API
        portfolio_data = get_mock_portfolio_data()
        
        # Create a pie chart of asset allocation
        symbols = [asset['symbol'] for asset in portfolio_data['assets']]
        allocations = [asset['allocation'] for asset in portfolio_data['assets']]
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.pie(allocations, labels=symbols, autopct='%1.1f%%', startangle=90)
        plt.title('Asset Allocation')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Determine if we're responding to a message or callback query
        if hasattr(update, 'message'):
            await update.message.reply_photo(photo=buf, caption="Current Asset Allocation")
        else:
            # We can't send a photo with edit_message, so we'll send it as a new message
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=buf, caption="Current Asset Allocation")
            
    except Exception as e:
        error_message = f"Error retrieving asset allocation: {str(e)}"
        if hasattr(update, 'message'):
            await update.message.reply_text(error_message)
        else:
            await update.callback_query.edit_message_text(error_message)

# Function to suggest portfolio rebalancing
async def suggest_rebalancing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # In a real implementation, you would fetch data from your database or exchange API
        portfolio_data = get_mock_portfolio_data()
        
        # Create mock target allocations
        target_allocations = {
            'BTC/USDT': 60.0,
            'ETH/USDT': 20.0,
            'SOL/USDT': 15.0,
            'ADA/USDT': 5.0
        }
        
        # Calculate rebalancing suggestions
        message = "📋 Rebalancing Suggestions 📋\n\n"
        
        for asset in portfolio_data['assets']:
            symbol = asset['symbol']
            current_allocation = asset['allocation']
            target_allocation = target_allocations.get(symbol, current_allocation)
            
            difference = target_allocation - current_allocation
            action = "Buy" if difference > 0 else "Sell"
            
            # Calculate amount to buy or sell
            value_to_change = abs(difference / 100 * portfolio_data['total_value'])
            price = asset['value'] / asset['amount']  # Calculate current price
            amount_to_change = value_to_change / price
            
            # Only suggest rebalancing if difference is significant (>1%)
            if abs(difference) > 1.0:
                message += f"{symbol}: {action} {amount_to_change:.4f} (${value_to_change:.2f})\n"
                message += f"  Current: {current_allocation:.2f}% → Target: {target_allocation:.2f}%\n\n"
        
        if message == "📋 Rebalancing Suggestions 📋\n\n":
            message += "No significant rebalancing needed. Your portfolio is well-balanced!"
        
        # Determine if we're responding to a message or callback query
        if hasattr(update, 'message'):
            await update.message.reply_text(message)
        else:
            await update.callback_query.edit_message_text(message)
            
    except Exception as e:
        error_message = f"Error generating rebalancing suggestions: {str(e)}"
        if hasattr(update, 'message'):
            await update.message.reply_text(error_message)
        else:
            await update.callback_query.edit_message_text(error_message)

# New function to view watchlist
async def view_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = str(update.effective_user.id)
        watchlist = get_watchlist(user_id)
        
        if not watchlist:
            message = "Your watchlist is empty. Add symbols using /addwatch <symbol>"
            keyboard = [[InlineKeyboardButton("Add Popular Symbols", callback_data="add_popular_symbols")]]
        else:
            message = "📋 Your Watchlist 📋\n\n"
            for i, symbol in enumerate(watchlist, 1):
                message += f"{i}. {symbol}\n"
            
            # Create inline keyboard for watchlist actions
            keyboard = []
            for symbol in watchlist:
                keyboard.append([
                    InlineKeyboardButton(f"Chart {symbol}", callback_data=f"chart_{symbol}_1h"),
                    InlineKeyboardButton(f"Remove {symbol}", callback_data=f"removewatch_{symbol}")
                ])
        
        keyboard.append([InlineKeyboardButton("Add New Symbol", callback_data="addwatch_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Determine if we're responding to a message or callback query
        if hasattr(update, 'message'):
            await update.message.reply_text(message, reply_markup=reply_markup)
        else:
            await update.callback_query.edit_message_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        error_message = f"Error retrieving watchlist: {str(e)}"
        if hasattr(update, 'message'):
            await update.message.reply_text(error_message)
        else:
            await update.callback_query.edit_message_text(error_message)

# New function to add to watchlist
async def add_to_watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("Please provide a symbol to add to your watchlist. Example: /addwatch BTC/USDT")
            return
        
        symbol = context.args[0].upper()
        user_id = str(update.effective_user.id)
        
        success = add_to_watchlist(symbol, user_id)
        
        if success:
            message = f"{symbol} has been added to your watchlist."
        else:
            message = f"{symbol} is already in your watchlist."
        
        # Add quick actions for the newly added symbol
        keyboard = [
            [InlineKeyboardButton(f"View Chart for {symbol}", callback_data=f"chart_{symbol}_1h")],
            [InlineKeyboardButton("View Watchlist", callback_data="watchlist")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        await update.message.reply_text(f"Error adding to watchlist: {str(e)}")

# New function to remove from watchlist
async def remove_from_watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("Please provide a symbol to remove from your watchlist. Example: /removewatch BTC/USDT")
            return
        
        symbol = context.args[0].upper()
        user_id = str(update.effective_user.id)
        
        success = remove_from_watchlist(symbol, user_id)
        
        if success:
            message = f"{symbol} has been removed from your watchlist."
        else:
            message = f"{symbol} is not in your watchlist."
        
        keyboard = [[InlineKeyboardButton("View Watchlist", callback_data="watchlist")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        await update.message.reply_text(f"Error removing from watchlist: {str(e)}")

# Handler for watchlist callback queries
async def watchlist_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = str(query.from_user.id)
    
    if data == "addwatch_menu":
        popular_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
        keyboard = []
        
        for symbol in popular_symbols:
            keyboard.append([InlineKeyboardButton(f"Add {symbol}", callback_data=f"add_to_watch_{symbol}")])
        
        keyboard.append([InlineKeyboardButton("Back to Watchlist", callback_data="watchlist")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text("Select a symbol to add to your watchlist:", reply_markup=reply_markup)
    
    elif data.startswith("add_to_watch_"):
        symbol = data.replace("add_to_watch_", "")
        success = add_to_watchlist(symbol, user_id)
        
        if success:
            message = f"{symbol} has been added to your watchlist."
        else:
            message = f"{symbol} is already in your watchlist."
        
        await query.edit_message_text(message)
        
        # After a short delay, show the updated watchlist
        await view_watchlist(update, context)
    
    elif data.startswith("removewatch_"):
        symbol = data.replace("removewatch_", "")
        success = remove_from_watchlist(symbol, user_id)
        
        if success:
            message = f"{symbol} has been removed from your watchlist."
        else:
            message = f"{symbol} is not in your watchlist."
        
        await query.edit_message_text(message)
        
        # After a short delay, show the updated watchlist
        await view_watchlist(update, context)
    
    elif data == "add_popular_symbols":
        popular_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for symbol in popular_symbols:
            add_to_watchlist(symbol, user_id)
        
        await query.edit_message_text("Popular symbols have been added to your watchlist.")
        
        # After a short delay, show the updated watchlist
        await view_watchlist(update, context)