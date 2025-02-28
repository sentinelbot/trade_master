import asyncio
import json
from datetime import datetime
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Set price alert for a specific symbol
async def set_price_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Please provide a symbol and price. Example: /alert BTC/USDT 50000")
        return
    
    symbol = context.args[0]
    try:
        price = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Price must be a number. Example: /alert BTC/USDT 50000")
        return
    
    # In a real scenario, you would store this alert in a database
    # For simplicity, we'll store it in the user_data dictionary
    if 'price_alerts' not in context.user_data:
        context.user_data['price_alerts'] = []
    
    alert_id = len(context.user_data['price_alerts'])
    context.user_data['price_alerts'].append({
        'id': alert_id,
        'symbol': symbol,
        'price': price,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    keyboard = [
        [InlineKeyboardButton("Cancel Alert", callback_data=f"cancel_alert_{alert_id}")],
        [InlineKeyboardButton("View All Alerts", callback_data="view_alerts")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🔔 Alert set for {symbol} at ${price:,.2f}",
        reply_markup=reply_markup
    )

# Check margin levels
async def check_margin_levels(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This would connect to your exchange API to check margin levels
    # For now, just returning a placeholder
    
    margin_data = {
        'account_equity': 15000.00,
        'margin_used': 5000.00,
        'available_margin': 10000.00,
        'margin_level': 300.00,  # 300%
        'liquidation_price': 42000.00,
        'positions': [
            {'symbol': 'BTC/USDT', 'size': 0.5, 'entry_price': 45000.00, 'liquidation_price': 42000.00},
            {'symbol': 'ETH/USDT', 'size': 2.0, 'entry_price': 2800.00, 'liquidation_price': 2500.00}
        ]
    }
    
    # Check if margin level is critical
    danger_level = margin_data['margin_level'] < 150
    warning_level = margin_data['margin_level'] < 250
    
    # Format the response
    message = f"📊 **Margin Account Overview**\n\n"
    message += f"Account Equity: ${margin_data['account_equity']:,.2f}\n"
    message += f"Margin Used: ${margin_data['margin_used']:,.2f}\n"
    message += f"Available Margin: ${margin_data['available_margin']:,.2f}\n"
    
    # Format margin level with warning emoji if needed
    if danger_level:
        message += f"⚠️ **CRITICAL** Margin Level: {margin_data['margin_level']:.2f}%\n"
    elif warning_level:
        message += f"⚠️ **WARNING** Margin Level: {margin_data['margin_level']:.2f}%\n"
    else:
        message += f"Margin Level: {margin_data['margin_level']:.2f}%\n"
    
    message += "\n**Open Positions:**\n"
    for pos in margin_data['positions']:
        message += f"• {pos['symbol']}: {pos['size']} @ ${pos['entry_price']:,.2f} (Liq: ${pos['liquidation_price']:,.2f})\n"
    
    # Add buttons for actions
    keyboard = [
        [InlineKeyboardButton("Close All Positions", callback_data="close_all_positions")],
        [InlineKeyboardButton("Add Margin", callback_data="add_margin")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(message, reply_markup=reply_markup)

# Check price alerts
async def check_price_alerts(context: ContextTypes.DEFAULT_TYPE):
    # This would be called periodically by the job queue
    # It would check current prices against set alerts
    # For simplicity, we'll just print a log message
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Checking price alerts...")
    
    # Example of what this would do:
    # 1. Get all price alerts from the database or user_data
    # 2. Get current prices from exchange APIs
    # 3. Compare and send notifications for triggered alerts
    
    # Example notification (uncomment when ready to implement)
    # for user_id, user_data in context.dispatcher.user_data.items():
    #     if 'price_alerts' in user_data:
    #         for alert in user_data['price_alerts']:
    #             # Check if alert is triggered
    #             # If triggered, send notification and remove alert
    #             current_price = get_current_price(alert['symbol'])
    #             if (current_price >= alert['price']):
    #                 await context.bot.send_message(
    #                     chat_id=user_id,
    #                     text=f"🔔 Alert triggered: {alert['symbol']} has reached ${alert['price']:,.2f}"
    #                 )

# Send scheduled report
async def send_scheduled_report(context: ContextTypes.DEFAULT_TYPE, chat_id, report_type="daily"):
    """Send a scheduled report to a specific chat."""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if report_type == "daily":
        report_title = f"📊 Daily Trading Summary - {datetime.now().strftime('%b %d, %Y')}"
    elif report_type == "weekly":
        report_title = f"📊 Weekly Trading Summary - Week {datetime.now().strftime('%U, %Y')}"
    else:
        report_title = f"📊 Trading Report - {current_time}"
    
    # This would generate a report based on user's portfolio and preferences
    # For now, we'll just send a placeholder message
    
    message = f"{report_title}\n\n"
    message += "Performance Summary:\n"
    message += "• Total P&L: +$1,245.67 (+3.2%)\n"
    message += "• Best Performing: ETH/USDT (+8.5%)\n"
    message += "• Worst Performing: DOT/USDT (-2.3%)\n\n"
    message += "Trading Activity:\n"
    message += "• Trades Executed: 12\n"
    message += "• Volume: $25,432.10\n\n"
    message += "Market Overview:\n"
    message += "• BTC Dominance: 42.3%\n"
    message += "• Market Trend: Bullish\n"
    
    keyboard = [
        [InlineKeyboardButton("Detailed Report", callback_data="detailed_report")],
        [InlineKeyboardButton("Portfolio Overview", callback_data="portfolio")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=message,
        reply_markup=reply_markup
    )

# Notify about system errors
async def notify_system_error(context: ContextTypes.DEFAULT_TYPE, error_message):
    """Send notification about system errors to admin."""
    
    admin_chat_id = int(os.environ.get("ADMIN_CHAT_ID", "0"))
    if admin_chat_id != 0:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"⚠️ **SYSTEM ERROR** - {current_time}\n\n"
        message += f"Error: {error_message}\n\n"
        
        try:
            await context.bot.send_message(
                chat_id=admin_chat_id,
                text=message
            )
        except Exception as e:
            print(f"Failed to send error notification: {e}")

# Send news alerts
async def send_news_alert(context: ContextTypes.DEFAULT_TYPE, chat_id, news_item):
    """Send a news alert to a user based on their watchlist."""
    
    message = f"📰 **Breaking News: {news_item['title']}**\n\n"
    message += f"{news_item['summary']}\n\n"
    message += f"Published: {news_item['published_at']}"
    
    keyboard = [
        [InlineKeyboardButton("Read Full Article", url=news_item['url'])],
        [InlineKeyboardButton("Related Assets", callback_data=f"news_assets_{news_item['id']}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=message,
        reply_markup=reply_markup
    )

# Function to monitor and notify about news related to watchlist assets
async def check_news_for_watchlist(context: ContextTypes.DEFAULT_TYPE):
    """Periodically check for news relevant to users' watchlists."""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Checking news for watchlists...")
    
    # This would connect to a news API and check for relevant news
    # It would then send notifications to users based on their watchlists
    # For simplicity, this is just a placeholder

# Send notification to user
async def send_notification(context: ContextTypes.DEFAULT_TYPE, user_id, message):
    """
    Send a notification to a user via Telegram.
    
    Parameters:
    - context: The Telegram context
    - user_id: The user's Telegram ID
    - message: The message to send
    
    Returns:
    - Boolean indicating success or failure
    """
    try:
        await context.bot.send_message(chat_id=user_id, text=message)
        return True
    except Exception as e:
        print(f"Error sending notification: {str(e)}")
        return False