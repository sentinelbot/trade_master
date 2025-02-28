import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Set user preferences
async def set_preferences(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide preferences in key=value format. Example: /preferences theme=dark")
        return
    
    # Initialize preferences if not already present
    if 'preferences' not in context.user_data:
        context.user_data['preferences'] = {
            'theme': 'light',
            'notification_frequency': 'high',
            'default_chart_timeframe': '1h',
            'default_exchange': 'binance',
            'show_btc_trades': 'true',
            'show_portfolio_percentage': 'true',
            'default_report_type': 'daily',
            'news_alerts': 'true',
            'price_alerts': 'true',
            'locale': 'en'
        }
    
    # Process each preference
    success_count = 0
    for arg in context.args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in context.user_data['preferences']:
                context.user_data['preferences'][key] = value
                success_count += 1
            else:
                await update.message.reply_text(f"Invalid preference key: {key}")
        else:
            await update.message.reply_text(f"Invalid format for preference: {arg}. Use key=value format.")
    
    if success_count > 0:
        # Create confirmation message
        message = f"Updated {success_count} preference{'s' if success_count > 1 else ''}:\n\n"
        for key, value in context.user_data['preferences'].items():
            message += f"• {key}: {value}\n"
        
        # Add buttons for common actions
        keyboard = [
            [InlineKeyboardButton("View All Preferences", callback_data="getpreferences")],
            [InlineKeyboardButton("Reset to Defaults", callback_data="reset_preferences")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
    else:
        await update.message.reply_text("No preferences were updated.")

# Get user preferences
async def get_preferences(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'preferences' not in context.user_data:
        # Set default preferences
        context.user_data['preferences'] = {
            'theme': 'light',
            'notification_frequency': 'high',
            'default_chart_timeframe': '1h',
            'default_exchange': 'binance',
            'show_btc_trades': 'true',
            'show_portfolio_percentage': 'true',
            'default_report_type': 'daily',
            'news_alerts': 'true',
            'price_alerts': 'true',
            'locale': 'en'
        }
    
    # Format the preferences for display
    message = "🔧 **Your Current Preferences**\n\n"
    for key, value in context.user_data['preferences'].items():
        message += f"• {key}: {value}\n"
    
    # Add explanation for key preferences
    message += "\n**Key Settings:**\n"
    message += "• theme: Appearance of the bot (light/dark)\n"
    message += "• notification_frequency: How often to notify (low/medium/high)\n"
    message += "• default_chart_timeframe: Default chart period (5m/15m/1h/4h/1d)\n"
    message += "• news_alerts: Receive news for watchlist assets (true/false)\n"
    
    # Add buttons for common actions
    keyboard = [
        [InlineKeyboardButton("Change Theme", callback_data="set_theme")],
        [InlineKeyboardButton("Change Notification Frequency", callback_data="set_notification")],
        [InlineKeyboardButton("Reset to Defaults", callback_data="reset_preferences")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the formatted message with buttons
    await update.message.reply_text(message, reply_markup=reply_markup)

# Reset preferences to defaults
async def reset_preferences(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Set default preferences
    context.user_data['preferences'] = {
        'theme': 'light',
        'notification_frequency': 'high',
        'default_chart_timeframe': '1h',
        'default_exchange': 'binance',
        'show_btc_trades': 'true',
        'show_portfolio_percentage': 'true',
        'default_report_type': 'daily',
        'news_alerts': 'true',
        'price_alerts': 'true',
        'locale': 'en'
    }
    
    await update.callback_query.answer("Preferences reset to defaults")
    await update.callback_query.edit_message_text("All preferences have been reset to default values.")

# Set theme preference
async def set_theme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Light Theme", callback_data="set_preference_theme=light")],
        [InlineKeyboardButton("Dark Theme", callback_data="set_preference_theme=dark")],
        [InlineKeyboardButton("Auto (System)", callback_data="set_preference_theme=auto")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "Select your preferred theme:",
        reply_markup=reply_markup
    )

# Set notification frequency
async def set_notification_frequency(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("High (All alerts)", callback_data="set_preference_notification_frequency=high")],
        [InlineKeyboardButton("Medium (Important only)", callback_data="set_preference_notification_frequency=medium")],
        [InlineKeyboardButton("Low (Critical only)", callback_data="set_preference_notification_frequency=low")],
        [InlineKeyboardButton("None", callback_data="set_preference_notification_frequency=none")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.callback_query.edit_message_text(
        "Select your notification frequency:",
        reply_markup=reply_markup
    )

# Process preference setting from callback
async def process_preference_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data):
    # Extract the preference from the callback data
    # Format is "set_preference_key=value"
    parts = callback_data.split('_', 2)
    if len(parts) == 3:
        preference_part = parts[2]
        if '=' in preference_part:
            key, value = preference_part.split('=', 1)
            
            # Initialize preferences if not already present
            if 'preferences' not in context.user_data:
                context.user_data['preferences'] = {}
            
            # Update the preference
            context.user_data['preferences'][key] = value
            
            # Confirm the change
            await update.callback_query.answer(f"Updated {key} to {value}")
            await update.callback_query.edit_message_text(f"Your {key} preference has been updated to {value}.")
        else:
            await update.callback_query.answer("Invalid preference format")
    else:
        await update.callback_query.answer("Invalid callback data")

# Get user locale/language
def get_user_locale(context: ContextTypes.DEFAULT_TYPE):
    """Get the user's preferred locale from preferences, or default to English."""
    if 'preferences' in context.user_data and 'locale' in context.user_data['preferences']:
        return context.user_data['preferences']['locale']
    return 'en'  # Default to English